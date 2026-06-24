//! Old-generation allocator.
//!
//! Objects that survive nursery collection are promoted here. Small objects
//! (`<= SMALL_REQUEST_THRESHOLD`, and not card-marked) come from the
//! [`ArenaCollection`] page allocator (`minimarkpage.py`); larger and
//! card-marked objects are raw-malloced and tracked in `large`, mirroring
//! incminimark's `old_rawmalloced_objects`.
//!
//! Membership (`contains`) is the arena page-bounds check for small objects
//! plus a payload set for the few large objects — replacing the former
//! all-objects `HashSet` side-table (see git history / AGENTS.md "Data
//! structure parity").
use std::alloc::{self, Layout};
use std::collections::HashSet;
use std::ptr;

use crate::arena::ArenaCollection;
use crate::flags;
use crate::header::GcHeader;

/// `incminimark.TRANSLATION_PARAMS`: 512 KiB arenas of 8 KiB pages; the
/// `ArenaCollection` serves requests up to 280 bytes (35 words).
const WORD: usize = std::mem::size_of::<usize>();
const ARENA_SIZE: usize = 65536 * WORD;
const PAGE_SIZE: usize = 1024 * WORD;
const SMALL_REQUEST_THRESHOLD: usize = 35 * WORD;

/// A single raw-malloced (large or card-marked) old-gen allocation.
struct OldObject {
    /// Start of the raw allocation (for dealloc). Before the card header bytes.
    alloc_start: usize,
    /// Address of the GcHeader = `alloc_start + card_header_bytes`.
    header_addr: usize,
    /// Layout used for deallocation (covers card header + object).
    layout: Layout,
}

/// Two-tier old generation: an [`ArenaCollection`] for small objects and a
/// raw-malloced list for large / card-marked objects.
pub struct OldGen {
    /// Page allocator for small objects (`minimarkpage.ArenaCollection`).
    ac: ArenaCollection,
    /// Live small-object count (arena blocks in use).
    small_count: usize,
    /// Raw-malloced large / card-marked objects.
    large: Vec<OldObject>,
    /// Payload addresses (`header_addr + GcHeader::SIZE`) of the `large`
    /// objects, for `contains`. Small objects answer membership from arena
    /// bounds and are not in this set.
    large_payloads: HashSet<usize>,
    /// Bytes held by `large` (counting card header + object).
    large_total: usize,
}

// Safety: OldGen owns all its allocations, single-threaded access.
unsafe impl Send for OldGen {}

impl OldGen {
    pub fn new() -> Self {
        OldGen {
            ac: ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, SMALL_REQUEST_THRESHOLD),
            small_count: 0,
            large: Vec::new(),
            large_payloads: HashSet::new(),
            large_total: 0,
        }
    }

    /// Allocate space for an object of `total_size` bytes (header + payload).
    /// Returns a pointer to the header location. Memory is zero-filled.
    pub fn alloc(&mut self, total_size: usize) -> *mut u8 {
        self.alloc_with_card_header(total_size, 0)
    }

    /// Allocate with `card_header_bytes` card bytes prepended.
    ///
    /// Layout: `[card_bytes][GcHeader][payload]`. Returns a pointer to the
    /// GcHeader (after the card bytes). Card-marked objects are always
    /// raw-malloced (incminimark only cards rawmalloced objects), so a non-zero
    /// `card_header_bytes` routes to the large path regardless of size.
    pub fn alloc_with_card_header(
        &mut self,
        total_size: usize,
        card_header_bytes: usize,
    ) -> *mut u8 {
        let obj_size = total_size.max(GcHeader::MIN_NURSERY_OBJ_SIZE);
        let alloc_size = card_header_bytes + obj_size;

        if card_header_bytes == 0 && alloc_size <= SMALL_REQUEST_THRESHOLD {
            // Small path: a block from the arena allocator. The arena does not
            // zero reused blocks, so clear it to preserve the zero-fill contract
            // callers rely on (untyped payload fields default to NULL).
            let block = self.ac.malloc(alloc_size);
            unsafe {
                ptr::write_bytes(block, 0, alloc_size);
            }
            self.small_count += 1;
            block
        } else {
            // Large / card-marked path: raw, individually tracked.
            let layout = Layout::from_size_align(alloc_size, 8).expect("invalid layout");
            let ptr = unsafe { alloc::alloc_zeroed(layout) };
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            let header_ptr = unsafe { ptr.add(card_header_bytes) };
            self.large.push(OldObject {
                alloc_start: ptr as usize,
                header_addr: header_ptr as usize,
                layout,
            });
            self.large_payloads.insert(header_ptr as usize + GcHeader::SIZE);
            self.large_total += alloc_size;
            header_ptr
        }
    }

    /// Allocate and copy `total_size` bytes from `src`.
    /// Returns a pointer to the header of the new copy.
    ///
    /// # Safety
    /// `src` must point to at least `total_size` bytes of readable memory.
    pub unsafe fn alloc_and_copy(&mut self, src: *const u8, total_size: usize) -> *mut u8 {
        let dst = self.alloc(total_size);
        unsafe { ptr::copy_nonoverlapping(src, dst, total_size) };
        dst
    }

    /// Total bytes currently allocated in old gen (small arena blocks + large).
    pub fn total_bytes(&self) -> usize {
        self.ac.total_memory_used() + self.large_total
    }

    /// Number of objects in old gen.
    pub fn object_count(&self) -> usize {
        self.small_count + self.large.len()
    }

    /// Perform mark-sweep collection.
    /// Before calling this, the caller must have set VISITED on all reachable
    /// objects. Frees every object that does NOT have VISITED, and clears
    /// VISITED on survivors.
    pub fn sweep(&mut self) {
        // Small objects: ask the arena to free every unmarked block. The block
        // pointer is the GcHeader address (small objects carry no card bytes).
        let mut small_survivors = 0usize;
        self.ac.mass_free(|block| {
            let hdr = unsafe { &mut *(block as *mut GcHeader) };
            if hdr.has_flag(flags::VISITED) {
                hdr.clear_flag(flags::VISITED);
                small_survivors += 1;
                false
            } else {
                true
            }
        });
        self.small_count = small_survivors;

        // Large objects: free unmarked, keep marked.
        let mut surviving = Vec::new();
        for obj_record in self.large.drain(..) {
            let hdr = unsafe { &mut *(obj_record.header_addr as *mut GcHeader) };
            if hdr.has_flag(flags::VISITED) {
                hdr.clear_flag(flags::VISITED);
                surviving.push(obj_record);
            } else {
                self.large_payloads
                    .remove(&(obj_record.header_addr + GcHeader::SIZE));
                self.large_total -= obj_record.layout.size();
                unsafe {
                    alloc::dealloc(obj_record.alloc_start as *mut u8, obj_record.layout);
                }
            }
        }
        self.large = surviving;
    }

    /// Check whether `obj_addr` is within old-gen memory: an arena page-bounds
    /// check for small objects (the `Nursery::contains` model, answered without
    /// dereferencing `obj_addr`) plus the large-object payload set. Called per
    /// root in minor collection (`walk_jf_roots`) and per traced field in
    /// `is_managed_heap_object`; arena membership is O(number of arenas), not
    /// O(number of objects).
    pub fn contains(&self, obj_addr: usize) -> bool {
        self.ac.contains(obj_addr) || self.large_payloads.contains(&obj_addr)
    }
}

impl Default for OldGen {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for OldGen {
    fn drop(&mut self) {
        // The ArenaCollection frees its arenas in its own Drop; free the large
        // raw allocations here.
        for obj_record in self.large.drain(..) {
            unsafe {
                alloc::dealloc(obj_record.alloc_start as *mut u8, obj_record.layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A size that routes to the large (raw) path.
    const BIG: usize = SMALL_REQUEST_THRESHOLD + 64;

    #[test]
    fn test_oldgen_alloc() {
        let mut oldgen = OldGen::new();
        let ptr = oldgen.alloc(32);
        assert!(!ptr.is_null());
        assert_eq!(oldgen.object_count(), 1);
        assert!(oldgen.total_bytes() >= 32);
        assert!(oldgen.contains(ptr as usize)); // arena bounds
    }

    #[test]
    fn test_oldgen_alloc_zeroes() {
        let mut oldgen = OldGen::new();
        let p = oldgen.alloc(64);
        for i in 0..64 {
            assert_eq!(unsafe { *p.add(i) }, 0, "fresh small alloc must be zeroed");
        }
    }

    #[test]
    fn test_oldgen_alloc_and_copy() {
        let mut oldgen = OldGen::new();
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let dst = unsafe { oldgen.alloc_and_copy(src.as_ptr(), 16) };
        assert!(!dst.is_null());
        for i in 0..16 {
            assert_eq!(unsafe { *dst.add(i) }, src[i]);
        }
    }

    #[test]
    fn test_oldgen_sweep_small() {
        let mut oldgen = OldGen::new();
        let p1 = oldgen.alloc(GcHeader::SIZE + 16);
        let p2 = oldgen.alloc(GcHeader::SIZE + 16);
        let p3 = oldgen.alloc(GcHeader::SIZE + 16);
        assert_eq!(oldgen.object_count(), 3);

        for (p, visited) in [(p1, true), (p2, false), (p3, true)] {
            let hdr = unsafe { &mut *(p as *mut GcHeader) };
            *hdr = GcHeader::new(0);
            if visited {
                hdr.set_flag(flags::VISITED);
            }
        }

        oldgen.sweep();

        assert_eq!(oldgen.object_count(), 2);
        assert!(!unsafe { &*(p1 as *mut GcHeader) }.has_flag(flags::VISITED));
        assert!(!unsafe { &*(p3 as *mut GcHeader) }.has_flag(flags::VISITED));
        // Survivors stay contained.
        assert!(oldgen.contains(p1 as usize));
        assert!(oldgen.contains(p3 as usize));
    }

    #[test]
    fn test_oldgen_large_path() {
        let mut oldgen = OldGen::new();
        let big = oldgen.alloc(BIG);
        let small = oldgen.alloc(32);
        assert_eq!(oldgen.object_count(), 2);
        let big_payload = big as usize + GcHeader::SIZE;
        assert!(oldgen.contains(big_payload)); // large payload set
        assert!(oldgen.contains(small as usize)); // arena bounds
        assert!(!oldgen.contains(0xdead_beef_usize));

        // Mark only the small one; sweep frees the large one.
        unsafe {
            *(small as *mut GcHeader) = GcHeader::new(0);
            (*(small as *mut GcHeader)).set_flag(flags::VISITED);
            *(big as *mut GcHeader) = GcHeader::new(0); // unmarked -> dead
        }
        oldgen.sweep();
        assert_eq!(oldgen.object_count(), 1);
        assert!(!oldgen.contains(big_payload)); // large freed -> removed
        assert!(oldgen.contains(small as usize));
    }

    #[test]
    fn test_oldgen_survivors_persist() {
        let mut oldgen = OldGen::new();
        let keep_small = oldgen.alloc(48);
        let keep_big = oldgen.alloc(BIG);
        let keep_big_payload = keep_big as usize + GcHeader::SIZE;
        for _ in 0..20 {
            oldgen.alloc(48);
        }
        for p in [keep_small, keep_big] {
            unsafe {
                *(p as *mut GcHeader) = GcHeader::new(0);
                (*(p as *mut GcHeader)).set_flag(flags::VISITED);
            }
        }
        oldgen.sweep();
        assert!(oldgen.contains(keep_small as usize));
        assert!(oldgen.contains(keep_big_payload));
    }
}
