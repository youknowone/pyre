//! Old-generation allocation on `minimarkpage.py:ArenaCollection`.

use std::alloc::{self, Layout};
use std::ptr;

use crate::address_dict::AddressSet;
use crate::flags;
use crate::header::{GcHeader, header_of};
use crate::minimarkpage::ArenaCollection;

const WORD: usize = std::mem::size_of::<usize>();
const OBJECT_ALIGN: usize = if GcHeader::ALIGN > WORD {
    GcHeader::ALIGN
} else {
    WORD
};
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
    /// incminimark.py:2688-2694 `raw_malloc_might_sweep`.  At sweep
    /// preparation the old rawmalloc stack is swapped into this one, isolating
    /// it from rawmalloc allocations made by minors between sweep steps.
    raw_malloc_might_sweep: Vec<RawMallocedObject>,
    /// Exact payload membership for objects routed to individual rawmalloc
    /// (oversized or card-header allocations).  This is the same address-dict
    /// shape upstream uses when it needs exact rawmalloc membership:
    /// incminimark.py:1219-1221 and 2153-2158.  Unlike the removed F2-era
    /// payload side table, it has no entry for ordinary arena survivors.
    rawmalloced_payloads: AddressSet,
    rawmalloced_total_size: usize,
    /// incminimark.py:386,1073-1074 `rawmalloced_peak_size`.
    rawmalloced_peak_size: usize,
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
            raw_malloc_might_sweep: Vec::new(),
            rawmalloced_payloads: AddressSet::default(),
            rawmalloced_total_size: 0,
            rawmalloced_peak_size: 0,
            poison_on_alloc: std::env::var_os("MAJIT_GC_NURSERY_POISON").is_some(),
        }
    }

    /// Allocate header + payload.  Like incminimark.py:999-1009, the arena
    /// path is not cleared; callers initialize the object explicitly.
    pub fn alloc(&mut self, total_size: usize) -> *mut u8 {
        self.alloc_with_card_header(total_size, 0)
    }

    /// Fallible rawmalloc allocation for host-side helpers whose RPython
    /// callers translate a null result into `MemoryError`.
    pub fn try_alloc(&mut self, total_size: usize) -> Option<*mut u8> {
        self.try_alloc_with_card_header(total_size, 0)
    }

    /// incminimark.py:1012-1080: card headers occur in the rawmalloc branch,
    /// are prepended to the allocation, and only the card bytes are cleared.
    pub fn alloc_with_card_header(
        &mut self,
        total_size: usize,
        card_header_bytes: usize,
    ) -> *mut u8 {
        self.try_alloc_with_card_header(total_size, card_header_bytes)
            .unwrap_or_else(|| {
                // The fallible path also returns None for a request no
                // allocation could ever satisfy, and `handle_alloc_error`
                // reports only the byte count. Name the request first, or an
                // undecodable object size reaches the operator as a bare
                // `LayoutError` with nothing to attribute it to.
                let alloc_size = Self::allocation_size(total_size)
                    .checked_add(card_header_bytes)
                    .and_then(try_round_up);
                let layout = alloc_size
                    .and_then(|alloc_size| Layout::from_size_align(alloc_size, WORD).ok());
                let Some(layout) = layout else {
                    panic!(
                        "GC BUG: oldgen request describes no allocation: \
                         total_size={total_size} card_header_bytes={card_header_bytes}"
                    );
                };
                alloc::handle_alloc_error(layout)
            })
    }

    /// Fallible counterpart of `alloc_with_card_header`. RPython's
    /// `raw_malloc` can return NULL and the translated allocation helper then
    /// raises `MemoryError`; do not turn that recoverable result into a Rust
    /// process abort.
    pub fn try_alloc_with_card_header(
        &mut self,
        total_size: usize,
        card_header_bytes: usize,
    ) -> Option<*mut u8> {
        let obj_size = try_round_up(total_size.max(GcHeader::MIN_NURSERY_OBJ_SIZE))?;
        let alloc_size = try_round_up(card_header_bytes.checked_add(obj_size)?)?;
        debug_assert_eq!(card_header_bytes % OBJECT_ALIGN, 0);
        let header_ptr = if card_header_bytes == 0 && alloc_size <= SMALL_REQUEST_THRESHOLD {
            self.ac.malloc(alloc_size)
        } else {
            let layout = Layout::from_size_align(alloc_size, OBJECT_ALIGN).ok()?;
            let raw = unsafe { alloc::alloc(layout) };
            if raw.is_null() {
                return None;
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
            self.rawmalloced_peak_size =
                self.rawmalloced_peak_size.max(self.rawmalloced_total_size);
            header_ptr
        };
        if self.poison_on_alloc {
            // ArenaCollection.malloc intentionally returns uninitialized
            // memory.  In detector mode make that contract observable for
            // both fresh/recycled arena blocks and rawmalloced objects.
            unsafe { ptr::write_bytes(header_ptr, 0xAA, obj_size) };
        }
        Some(header_ptr)
    }

    /// `raw_malloc_usage(totalsize)` for an object allocation.
    ///
    /// The collector's finalizer accounting needs the same rounded byte count
    /// that `ArenaCollection.total_memory_used` / `rawmalloced_total_size`
    /// record.  Keep that geometry owned by the allocator rather than
    /// duplicating its minimum-size and alignment rules in `collector.rs`.
    pub(crate) fn allocation_size(total_size: usize) -> usize {
        round_up(total_size.max(GcHeader::MIN_NURSERY_OBJ_SIZE))
    }

    /// Allocate uninitialized space and overwrite the complete object from
    /// `src`, as nursery promotion does.
    ///
    /// # Safety
    ///
    /// `src` must be readable for `total_size` bytes and must not overlap the
    /// newly allocated destination.
    pub unsafe fn alloc_and_copy(&mut self, src: *const u8, total_size: usize) -> *mut u8 {
        let dst = self.alloc(total_size);
        unsafe { ptr::copy_nonoverlapping(src, dst, total_size) };
        dst
    }

    /// incminimark.py:1268: arena live bytes plus rawmalloced live bytes.
    pub fn total_bytes(&self) -> usize {
        self.ac.total_memory_used + self.rawmalloced_total_size
    }

    pub(crate) fn arenas_count(&self) -> usize {
        self.ac.arenas_count
    }

    pub(crate) fn arenas_bytes(&self) -> usize {
        self.ac.total_memory_used
    }

    pub(crate) fn rawmalloced_bytes(&self) -> usize {
        self.rawmalloced_total_size
    }

    pub(crate) fn peak_rawmalloced_bytes(&self) -> usize {
        self.rawmalloced_peak_size
    }

    /// incminimark.py:1270-1286 memory-stat helpers.
    pub(crate) fn total_allocated_bytes(&self) -> usize {
        self.ac.total_memory_alloced + self.rawmalloced_total_size
    }

    pub(crate) fn peak_allocated_bytes(&self) -> usize {
        self.ac.peak_memory_alloced + self.rawmalloced_peak_size
    }

    pub(crate) fn peak_used_bytes(&self) -> usize {
        self.ac.peak_memory_used.max(self.ac.total_memory_used) + self.rawmalloced_peak_size
    }

    pub(crate) fn peak_arena_bytes(&self) -> usize {
        self.ac.peak_memory_used.max(self.ac.total_memory_used)
    }

    #[cfg(test)]
    pub(crate) fn object_count(&self) -> usize {
        self.ac.object_count()
            + self.old_rawmalloced_objects.len()
            + self.raw_malloc_might_sweep.len()
    }

    pub(crate) fn debug_validate_freeblocks(&self, site: &str) {
        self.ac.debug_validate_freeblocks(site);
    }

    /// incminimark.py:2512-2514 and :2688-2694: freeze the arena pages and
    /// rawmalloc stack belonging to this major cycle.  Allocations made while
    /// sweeping go to the fresh active page lists and
    /// `old_rawmalloced_objects`, so this cycle never visits them.
    pub fn sweep_prepare(&mut self) {
        self.ac.mass_free_prepare();
        debug_assert!(
            self.raw_malloc_might_sweep.is_empty(),
            "raw_malloc_might_sweep must be empty"
        );
        std::mem::swap(
            &mut self.raw_malloc_might_sweep,
            &mut self.old_rawmalloced_objects,
        );
    }

    /// incminimark.py:2695-2702 `free_unvisited_rawmalloc_objects_step`.
    /// Process at most `nobjects` candidates and return the unused part of the
    /// budget, exactly like the upstream routine.
    pub fn sweep_rawmalloc_step(&mut self, mut nobjects: usize) -> usize {
        while !self.raw_malloc_might_sweep.is_empty() && nobjects > 0 {
            let object = self.raw_malloc_might_sweep.pop().unwrap();
            let hdr = unsafe { &mut *(object.header_addr as *mut GcHeader) };
            if hdr.has_flag(flags::VISITED) {
                hdr.clear_flag(flags::VISITED);
                self.old_rawmalloced_objects.push(object);
            } else {
                if crate::gc_lifetime_log_enabled() {
                    eprintln!(
                        "[gc][free] addr={:#x} type_id={} kind=raw",
                        object.header_addr + GcHeader::SIZE,
                        hdr.type_id()
                    );
                }
                self.rawmalloced_total_size -= object.layout.size();
                let removed = self
                    .rawmalloced_payloads
                    .remove(&(object.header_addr + GcHeader::SIZE));
                debug_assert!(removed);
                unsafe { alloc::dealloc(object.alloc_start as *mut u8, object.layout) };
            }
            nobjects -= 1;
        }
        nobjects
    }

    /// Whether the rawmalloc half of the current incremental sweep remains.
    pub fn rawmalloc_sweep_pending(&self) -> bool {
        !self.raw_malloc_might_sweep.is_empty()
    }

    /// incminimark.py:2549-2555: sweep at most `max_pages` frozen arena pages.
    pub fn sweep_arenas_step(&mut self, max_pages: usize) -> bool {
        self.ac.mass_free_incremental(
            &mut |header_ptr| unsafe {
                let hdr = &mut *(header_ptr as *mut GcHeader);
                if hdr.has_flag(flags::VISITED) {
                    hdr.clear_flag(flags::VISITED);
                    false
                } else {
                    if crate::gc_lifetime_log_enabled() {
                        eprintln!(
                            "[gc][free] addr={:#x} type_id={} kind=arena",
                            header_ptr as usize + GcHeader::SIZE,
                            hdr.type_id()
                        );
                    }
                    true
                }
            },
            max_pages,
        )
    }

    pub fn page_size(&self) -> usize {
        DEFAULT_PAGE_SIZE
    }

    pub fn small_request_threshold(&self) -> usize {
        SMALL_REQUEST_THRESHOLD
    }

    /// Non-incremental compatibility entry point, expressed as prepare plus
    /// draining steps just like minimarkpage.py:376-383 `mass_free`.
    #[allow(dead_code)]
    pub fn sweep(&mut self) {
        self.sweep_prepare();
        while self.rawmalloc_sweep_pending() {
            self.sweep_rawmalloc_step(usize::MAX);
        }
        let complete = self.sweep_arenas_step(usize::MAX);
        assert!(complete, "non-incremental oldgen sweep returned false");
    }

    #[cfg(test)]
    pub(crate) fn rawmalloc_sweep_candidate_count(&self) -> usize {
        self.raw_malloc_might_sweep.len()
    }

    #[cfg(test)]
    pub(crate) fn active_rawmalloc_count(&self) -> usize {
        self.old_rawmalloced_objects.len()
    }

    /// Arena membership is intentionally an address-range answer, not a live
    /// block answer.  Rawmalloced objects retain exact payload membership.
    ///
    /// Split so the arena range test can inline into the caller and the payload
    /// set cannot. Major marking asks this question once per root and once per
    /// traced edge, and the arena answer settles almost all of them; keeping the
    /// hashed lookup in the same body made the whole thing too large to inline,
    /// so every edge paid a call.
    #[inline]
    pub fn contains(&self, obj_addr: usize) -> bool {
        self.ac.contains(obj_addr) || self.rawmalloced_contains(obj_addr)
    }

    #[inline(never)]
    fn rawmalloced_contains(&self, obj_addr: usize) -> bool {
        !self.rawmalloced_payloads.is_empty() && self.rawmalloced_payloads.contains(&obj_addr)
    }

    pub fn mark_visited(obj_addr: usize) {
        unsafe { (*header_of(obj_addr)).set_flag(flags::VISITED) };
    }
}

fn round_up(size: usize) -> usize {
    try_round_up(size).expect("allocation size overflow")
}

fn try_round_up(size: usize) -> Option<usize> {
    Some(size.checked_add(OBJECT_ALIGN - 1)? & !(OBJECT_ALIGN - 1))
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
        for object in self.raw_malloc_might_sweep.drain(..) {
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
    fn fallible_allocation_reports_size_overflow() {
        let mut oldgen = OldGen::new();
        assert!(oldgen.try_alloc(usize::MAX).is_none());
    }

    #[test]
    fn allocations_preserve_gc_header_alignment() {
        let mut oldgen = OldGen::new();
        for payload_size in 1..=4 * OBJECT_ALIGN {
            let ptr = oldgen.alloc(GcHeader::SIZE + payload_size);
            assert_eq!(
                ptr as usize % GcHeader::ALIGN,
                0,
                "payload size {payload_size} produced a misaligned GC header"
            );
        }
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
        assert!(oldgen.peak_rawmalloced_bytes() >= round_up(size + 2 * WORD) + round_up(size));
        assert!(oldgen.peak_allocated_bytes() >= oldgen.total_allocated_bytes());
    }

    #[test]
    fn rawmalloc_sweep_step_is_bounded_and_isolates_new_allocations() {
        let mut oldgen = OldGen::new();
        let size = SMALL_REQUEST_THRESHOLD + WORD;
        for _ in 0..3 {
            let ptr = oldgen.alloc(size);
            unsafe { *ptr.cast::<GcHeader>() = GcHeader::new(0) };
        }

        oldgen.sweep_prepare();
        assert_eq!(oldgen.rawmalloc_sweep_candidate_count(), 3);
        assert_eq!(oldgen.active_rawmalloc_count(), 0);

        // incminimark.py:2688-2694: allocations made after the swap land in
        // the fresh active stack and are not swept in this cycle.
        let fresh = oldgen.alloc(size);
        let fresh_payload = fresh as usize + GcHeader::SIZE;
        unsafe { *fresh.cast::<GcHeader>() = GcHeader::new(0) };
        assert_eq!(oldgen.active_rawmalloc_count(), 1);

        let bytes_before = oldgen.total_bytes();
        assert_eq!(oldgen.sweep_rawmalloc_step(1), 0);
        assert_eq!(oldgen.rawmalloc_sweep_candidate_count(), 2);
        assert_eq!(oldgen.total_bytes(), bytes_before - round_up(size));

        while oldgen.rawmalloc_sweep_pending() {
            oldgen.sweep_rawmalloc_step(1);
        }
        assert_eq!(oldgen.active_rawmalloc_count(), 1);
        assert_eq!(oldgen.total_bytes(), round_up(size));
        assert!(oldgen.contains(fresh_payload));
        assert!(oldgen.sweep_arenas_step(1));
    }
}
