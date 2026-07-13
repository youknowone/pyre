//! Old-generation allocation on `minimarkpage.py:ArenaCollection`.

use std::alloc::{self, Layout};
use std::collections::HashSet;
use std::ptr;

use crate::flags;
use crate::header::{GcHeader, header_of};
use crate::minimarkpage::ArenaCollection;

const WORD: usize = std::mem::size_of::<usize>();
const DEFAULT_PAGE_SIZE: usize = 1024 * WORD;
const DEFAULT_ARENA_SIZE: usize = 65536 * WORD;
const SMALL_REQUEST_THRESHOLD: usize = 35 * WORD;

/// incminimark.py `old_rawmalloced_objects` entry.  The list shape is kept
/// because upstream individually frees objects above the small threshold.
struct RawMallocedObject {
    alloc_start: usize,
    header_addr: usize,
    layout: Layout,
}

pub struct OldGen {
    ac: ArenaCollection,
    old_rawmalloced_objects: Vec<RawMallocedObject>,
    /// Exact payload membership for objects routed to individual rawmalloc
    /// (oversized or card-header allocations).  This is the same address-dict
    /// shape upstream uses when it needs exact rawmalloc membership:
    /// incminimark.py:1219-1221 and 2153-2158.  Unlike the removed F2-era
    /// payload side table, it has no entry for ordinary arena survivors.
    rawmalloced_payloads: HashSet<usize>,
    rawmalloced_total_size: usize,
    /// llarena debug-fill parity for old-generation allocations.  The same
    /// opt-in detector covers both recycled arena blocks and rawmalloc blocks.
    poison_on_alloc: bool,
}

unsafe impl Send for OldGen {}

impl OldGen {
    pub fn new() -> Self {
        Self {
            ac: ArenaCollection::new(
                DEFAULT_ARENA_SIZE,
                DEFAULT_PAGE_SIZE,
                SMALL_REQUEST_THRESHOLD,
            ),
            old_rawmalloced_objects: Vec::new(),
            rawmalloced_payloads: HashSet::new(),
            rawmalloced_total_size: 0,
            poison_on_alloc: std::env::var_os("MAJIT_GC_NURSERY_POISON").is_some(),
        }
    }

    /// Allocate header + payload.  Like incminimark.py:999-1009, the arena
    /// path is not cleared; callers initialize the object explicitly.
    pub fn alloc(&mut self, total_size: usize) -> *mut u8 {
        self.alloc_with_card_header(total_size, 0)
    }

    /// incminimark.py:1012-1080: card headers occur in the rawmalloc branch,
    /// are prepended to the allocation, and only the card bytes are cleared.
    pub fn alloc_with_card_header(
        &mut self,
        total_size: usize,
        card_header_bytes: usize,
    ) -> *mut u8 {
        let obj_size = round_up(total_size.max(GcHeader::MIN_NURSERY_OBJ_SIZE));
        let alloc_size = round_up(card_header_bytes + obj_size);
        let header_ptr = if card_header_bytes == 0 && alloc_size <= SMALL_REQUEST_THRESHOLD {
            self.ac.malloc(alloc_size)
        } else {
            let layout = Layout::from_size_align(alloc_size, WORD).expect("invalid layout");
            let raw = unsafe { alloc::alloc(layout) };
            if raw.is_null() {
                alloc::handle_alloc_error(layout);
            }
            if card_header_bytes > 0 {
                unsafe { ptr::write_bytes(raw, 0, card_header_bytes) };
            }
            let header_ptr = unsafe { raw.add(card_header_bytes) };
            self.old_rawmalloced_objects.push(RawMallocedObject {
                alloc_start: raw as usize,
                header_addr: header_ptr as usize,
                layout,
            });
            self.rawmalloced_payloads
                .insert(header_ptr as usize + GcHeader::SIZE);
            self.rawmalloced_total_size += alloc_size;
            header_ptr
        };
        if self.poison_on_alloc {
            // ArenaCollection.malloc intentionally returns uninitialized
            // memory.  In detector mode make that contract observable for
            // both fresh/recycled arena blocks and rawmalloced objects.
            unsafe { ptr::write_bytes(header_ptr, 0xAA, obj_size) };
        }
        header_ptr
    }

    /// Allocate uninitialized space and overwrite the complete object from
    /// `src`, as nursery promotion does.
    pub unsafe fn alloc_and_copy(&mut self, src: *const u8, total_size: usize) -> *mut u8 {
        let dst = self.alloc(total_size);
        unsafe { ptr::copy_nonoverlapping(src, dst, total_size) };
        dst
    }

    /// incminimark.py:1268: arena live bytes plus rawmalloced live bytes.
    pub fn total_bytes(&self) -> usize {
        self.ac.total_memory_used + self.rawmalloced_total_size
    }

    #[cfg(test)]
    pub(crate) fn object_count(&self) -> usize {
        self.ac.object_count() + self.old_rawmalloced_objects.len()
    }

    /// Non-incremental major sweep: `ArenaCollection.mass_free` for small
    /// objects and individual freeing of unvisited rawmalloced objects.
    pub fn sweep(&mut self) {
        self.ac.mass_free(|header_ptr| unsafe {
            let hdr = &mut *(header_ptr as *mut GcHeader);
            if hdr.has_flag(flags::VISITED) {
                hdr.clear_flag(flags::VISITED);
                false
            } else {
                true
            }
        });

        let mut surviving = Vec::new();
        let mut freed_bytes = 0;
        for object in self.old_rawmalloced_objects.drain(..) {
            let hdr = unsafe { &mut *(object.header_addr as *mut GcHeader) };
            if hdr.has_flag(flags::VISITED) {
                hdr.clear_flag(flags::VISITED);
                surviving.push(object);
            } else {
                freed_bytes += object.layout.size();
                let removed = self
                    .rawmalloced_payloads
                    .remove(&(object.header_addr + GcHeader::SIZE));
                debug_assert!(removed);
                unsafe { alloc::dealloc(object.alloc_start as *mut u8, object.layout) };
            }
        }
        self.rawmalloced_total_size -= freed_bytes;
        self.old_rawmalloced_objects = surviving;
    }

    /// Arena membership is intentionally an address-range answer, not a live
    /// block answer.  Rawmalloced objects retain exact payload membership.
    pub fn contains(&self, obj_addr: usize) -> bool {
        self.ac.contains(obj_addr) || self.rawmalloced_payloads.contains(&obj_addr)
    }

    pub fn mark_visited(obj_addr: usize) {
        unsafe { (*header_of(obj_addr)).set_flag(flags::VISITED) };
    }
}

fn round_up(size: usize) -> usize {
    size.checked_add(WORD - 1)
        .expect("allocation size overflow")
        & !(WORD - 1)
}

impl Default for OldGen {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for OldGen {
    fn drop(&mut self) {
        for object in self.old_rawmalloced_objects.drain(..) {
            unsafe { alloc::dealloc(object.alloc_start as *mut u8, object.layout) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_alloc_and_copy() {
        let mut oldgen = OldGen::new();
        let src = [1u8; 32];
        let dst = unsafe { oldgen.alloc_and_copy(src.as_ptr(), src.len()) };
        assert_eq!(unsafe { *dst.add(17) }, 1);
        assert!(oldgen.contains(dst as usize + GcHeader::SIZE));
    }

    #[test]
    fn sweep_reuses_small_blocks_and_clears_visited() {
        let mut oldgen = OldGen::new();
        let p1 = oldgen.alloc(GcHeader::SIZE + 16);
        let p2 = oldgen.alloc(GcHeader::SIZE + 16);
        unsafe {
            *p1.cast::<GcHeader>() = GcHeader::new(0);
            *p2.cast::<GcHeader>() = GcHeader::new(0);
            (*p1.cast::<GcHeader>()).set_flag(flags::VISITED);
        }
        oldgen.sweep();
        assert!(!unsafe { (*p1.cast::<GcHeader>()).has_flag(flags::VISITED) });
        let p3 = oldgen.alloc(GcHeader::SIZE + 16);
        assert_eq!(p3, p2);
    }

    #[test]
    fn large_and_card_allocations_use_rawmalloc_accounting() {
        let mut oldgen = OldGen::new();
        let size = SMALL_REQUEST_THRESHOLD + WORD;
        let ptr = oldgen.alloc_with_card_header(size, 2 * WORD);
        assert_eq!(unsafe { *ptr.sub(1) }, 0);
        unsafe {
            *ptr.cast::<GcHeader>() = GcHeader::new(0);
            (*ptr.cast::<GcHeader>()).set_flag(flags::VISITED);
        }
        assert!(oldgen.contains(ptr as usize + GcHeader::SIZE));
        assert_eq!(oldgen.total_bytes(), round_up(size + 2 * WORD));
        oldgen.sweep();
        assert!(oldgen.contains(ptr as usize + GcHeader::SIZE));

        let dead = oldgen.alloc(size);
        let dead_payload = dead as usize + GcHeader::SIZE;
        unsafe { *dead.cast::<GcHeader>() = GcHeader::new(0) };
        assert!(oldgen.contains(dead_payload));
        oldgen.sweep();
        assert!(!oldgen.contains(dead_payload));
    }
}
