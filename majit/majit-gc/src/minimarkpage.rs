//! Size-class arena allocator for old-generation objects.
//!
//! Structural port of `rpython/memory/gc/minimarkpage.py`.  Memory is split
//! into arenas, pages, and same-sized blocks.  Free blocks and free pages are
//! chained through their first machine word, exactly like upstream's
//! `llarena` representation; allocations are deliberately not zero-filled.

use std::alloc::{self, Layout};
use std::ptr;

const WORD: usize = std::mem::size_of::<usize>();

#[repr(C)]
struct ArenaReference {
    base: *mut u8,
    layout: Layout,
    nfreepages: usize,
    totalpages: usize,
    freepages: *mut u8,
    nextarena: *mut ArenaReference,
}

#[repr(C)]
struct PageHeader {
    nextpage: *mut PageHeader,
    arena: *mut ArenaReference,
    nfree: usize,
    freeblock: *mut u8,
}

/// `minimarkpage.py:ArenaCollection`.
pub struct ArenaCollection {
    pub arena_size: usize,
    pub page_size: usize,
    pub small_request_threshold: usize,
    pub arenas_count: usize,
    page_for_size: Vec<*mut PageHeader>,
    full_page_for_size: Vec<*mut PageHeader>,
    old_page_for_size: Vec<*mut PageHeader>,
    old_full_page_for_size: Vec<*mut PageHeader>,
    nblocks_for_size: Vec<usize>,
    hdrsize: usize,
    max_pages_per_arena: usize,
    arenas_lists: Vec<*mut ArenaReference>,
    old_arenas_lists: Vec<*mut ArenaReference>,
    current_arena: *mut ArenaReference,
    /// pyre-local old-generation membership support.  The arena lists above
    /// retain minimarkpage.py's allocator shape; this flat range index avoids
    /// pointer-chasing every bucket for arbitrary-word validity checks.
    /// It changes only when an arena is allocated or freed.
    arena_ranges: Vec<(usize, usize)>,
    min_empty_nfreepages: usize,
    pub num_uninitialized_pages: usize,
    pub total_memory_used: usize,
    pub peak_memory_used: usize,
    pub total_memory_alloced: usize,
    pub peak_memory_alloced: usize,
    live_objects: usize,
    size_class_with_old_pages: isize,
}

// ArenaCollection owns its arena allocations and is only mutated through
// `&mut self`; the collector itself provides single-threaded access.
unsafe impl Send for ArenaCollection {}

impl ArenaCollection {
    pub fn new(arena_size: usize, page_size: usize, small_request_threshold: usize) -> Self {
        assert_eq!(arena_size % WORD, 0);
        assert_eq!(page_size % WORD, 0);
        assert_eq!(small_request_threshold % WORD, 0);
        let length = small_request_threshold / WORD + 1;
        let hdrsize = std::mem::size_of::<PageHeader>();
        assert!(page_size > hdrsize);
        let mut nblocks_for_size = vec![0; length];
        for (i, nblocks) in nblocks_for_size.iter_mut().enumerate().skip(1) {
            *nblocks = (page_size - hdrsize) / (WORD * i);
            assert!(*nblocks > 0);
        }
        let max_pages_per_arena = arena_size / page_size;
        assert!(max_pages_per_arena > 0);
        Self {
            arena_size,
            page_size,
            small_request_threshold,
            arenas_count: 0,
            page_for_size: vec![ptr::null_mut(); length],
            full_page_for_size: vec![ptr::null_mut(); length],
            old_page_for_size: vec![ptr::null_mut(); length],
            old_full_page_for_size: vec![ptr::null_mut(); length],
            nblocks_for_size,
            hdrsize,
            max_pages_per_arena,
            arenas_lists: vec![ptr::null_mut(); max_pages_per_arena],
            old_arenas_lists: vec![ptr::null_mut(); max_pages_per_arena],
            current_arena: ptr::null_mut(),
            arena_ranges: Vec::new(),
            min_empty_nfreepages: max_pages_per_arena,
            num_uninitialized_pages: 0,
            total_memory_used: 0,
            peak_memory_used: 0,
            total_memory_alloced: 0,
            peak_memory_alloced: 0,
            live_objects: 0,
            size_class_with_old_pages: -1,
        }
    }

    /// `ArenaCollection.malloc`: allocate an uninitialized arena block.
    pub fn malloc(&mut self, size: usize) -> *mut u8 {
        let nsize = size;
        assert!(nsize > 0, "malloc: size is null");
        assert!(
            nsize <= self.small_request_threshold,
            "malloc: size too big"
        );
        assert_eq!(nsize & (WORD - 1), 0, "malloc: size is not aligned");
        self.total_memory_used += nsize;
        self.live_objects += 1;
        let size_class = nsize / WORD;
        let mut page = self.page_for_size[size_class];
        if page.is_null() {
            page = self.allocate_new_page(size_class);
        }
        unsafe {
            let result = (*page).freeblock;
            let freeblock = if (*page).nfree > 0 {
                (*page).nfree -= 1;
                *(result as *mut *mut u8)
            } else {
                result.add(nsize)
            };
            (*page).freeblock = freeblock;
            let pageaddr = page as *mut u8;
            if freeblock.offset_from(pageaddr) as usize > self.page_size - nsize {
                self.page_for_size[size_class] = (*page).nextpage;
                (*page).nextpage = self.full_page_for_size[size_class];
                self.full_page_for_size[size_class] = page;
            }
            result
        }
    }

    /// `ArenaCollection.allocate_new_page`.
    fn allocate_new_page(&mut self, size_class: usize) -> *mut PageHeader {
        if self.current_arena.is_null() {
            self.allocate_new_arena();
        }
        unsafe {
            let arena = self.current_arena;
            let result = (*arena).freepages;
            let freepages = if (*arena).nfreepages > 0 {
                (*arena).nfreepages -= 1;
                *(result as *mut *mut u8)
            } else {
                assert!(self.num_uninitialized_pages > 0);
                self.num_uninitialized_pages -= 1;
                if self.num_uninitialized_pages > 0 {
                    result.add(self.page_size)
                } else {
                    ptr::null_mut()
                }
            };
            (*arena).freepages = freepages;
            if freepages.is_null() {
                assert_eq!((*arena).nfreepages, 0);
                (*arena).nextarena = self.arenas_lists[0];
                self.arenas_lists[0] = arena;
                self.current_arena = ptr::null_mut();
            }
            let page = result as *mut PageHeader;
            ptr::write(
                page,
                PageHeader {
                    nextpage: ptr::null_mut(),
                    arena,
                    nfree: 0,
                    freeblock: result.add(self.hdrsize),
                },
            );
            assert!(self.page_for_size[size_class].is_null());
            self.page_for_size[size_class] = page;
            page
        }
    }

    /// `ArenaCollection._pick_next_arena`.
    fn _pick_next_arena(&mut self) -> bool {
        let mut i = self.min_empty_nfreepages;
        while i < self.max_pages_per_arena {
            if !self.arenas_lists[i].is_null() {
                self.current_arena = self.arenas_lists[i];
                unsafe {
                    self.arenas_lists[i] = (*self.current_arena).nextarena;
                }
                return true;
            }
            i += 1;
            self.min_empty_nfreepages = i;
        }
        false
    }

    /// `ArenaCollection.allocate_new_arena`.
    fn allocate_new_arena(&mut self) {
        if self._pick_next_arena() {
            return;
        }
        self._rehash_arenas_lists();
        if self._pick_next_arena() {
            return;
        }
        let layout = Layout::from_size_align(self.arena_size, WORD).expect("invalid arena layout");
        let arena_base = unsafe { alloc::alloc(layout) };
        if arena_base.is_null() {
            alloc::handle_alloc_error(layout);
        }
        self.total_memory_alloced += self.arena_size;
        self.peak_memory_alloced = self.peak_memory_alloced.max(self.total_memory_alloced);
        let arena_end = unsafe { arena_base.add(self.arena_size) } as usize;
        let firstpage = start_of_page(arena_base as usize + self.page_size - 1, self.page_size);
        let npages = (arena_end - firstpage) / self.page_size;
        assert!(npages > 0);
        let arena = Box::into_raw(Box::new(ArenaReference {
            base: arena_base,
            layout,
            nfreepages: 0,
            totalpages: npages,
            freepages: firstpage as *mut u8,
            nextarena: ptr::null_mut(),
        }));
        self.num_uninitialized_pages = npages;
        self.current_arena = arena;
        self.arenas_count += 1;
        let range = (arena_base as usize, arena_end);
        let index = self
            .arena_ranges
            .binary_search_by_key(&range.0, |&(start, _)| start)
            .unwrap_err();
        self.arena_ranges.insert(index, range);
    }

    /// `ArenaCollection.mass_free_prepare`.
    pub(crate) fn mass_free_prepare(&mut self) {
        self.peak_memory_used = self.peak_memory_used.max(self.total_memory_used);
        let mut size_class = self.small_request_threshold / WORD;
        self.size_class_with_old_pages = size_class as isize;
        while size_class >= 1 {
            self.old_page_for_size[size_class] = self.page_for_size[size_class];
            self.old_full_page_for_size[size_class] = self.full_page_for_size[size_class];
            self.page_for_size[size_class] = ptr::null_mut();
            self.full_page_for_size[size_class] = ptr::null_mut();
            size_class -= 1;
        }
    }

    /// `ArenaCollection.mass_free_incremental`.
    pub(crate) fn mass_free_incremental(
        &mut self,
        ok_to_free_func: &mut impl FnMut(*mut u8) -> bool,
        mut max_pages: usize,
    ) -> bool {
        let mut size_class = self.size_class_with_old_pages;
        while size_class >= 1 {
            max_pages = self.mass_free_in_pages(size_class as usize, ok_to_free_func, max_pages);
            if max_pages == 0 {
                self.size_class_with_old_pages = size_class;
                return false;
            }
            size_class -= 1;
        }
        if size_class >= 0 {
            self._rehash_arenas_lists();
            self.size_class_with_old_pages = -1;
        }
        true
    }

    /// `ArenaCollection.mass_free`, the non-incremental STW entry point.
    pub fn mass_free(&mut self, mut ok_to_free_func: impl FnMut(*mut u8) -> bool) {
        self.mass_free_prepare();
        let complete = self.mass_free_incremental(&mut ok_to_free_func, usize::MAX);
        assert!(
            complete,
            "non-incremental mass_free_in_pages returned false"
        );
    }

    /// `ArenaCollection._rehash_arenas_lists`.
    fn _rehash_arenas_lists(&mut self) {
        std::mem::swap(&mut self.old_arenas_lists, &mut self.arenas_lists);
        self.arenas_lists.fill(ptr::null_mut());
        for i in 0..self.max_pages_per_arena {
            let mut arena = self.old_arenas_lists[i];
            self.old_arenas_lists[i] = ptr::null_mut();
            while !arena.is_null() {
                unsafe {
                    let nextarena = (*arena).nextarena;
                    if (*arena).nfreepages == (*arena).totalpages {
                        self.free_arena(arena);
                    } else {
                        let n = (*arena).nfreepages;
                        assert!(n < self.max_pages_per_arena);
                        (*arena).nextarena = self.arenas_lists[n];
                        self.arenas_lists[n] = arena;
                    }
                    arena = nextarena;
                }
            }
        }
        self.min_empty_nfreepages = 1;
    }

    fn mass_free_in_pages(
        &mut self,
        size_class: usize,
        ok_to_free_func: &mut impl FnMut(*mut u8) -> bool,
        mut max_pages: usize,
    ) -> usize {
        let nblocks = self.nblocks_for_size[size_class];
        let block_size = size_class * WORD;
        let mut remaining_partial_pages = self.page_for_size[size_class];
        let mut remaining_full_pages = self.full_page_for_size[size_class];
        for step in 0..2 {
            let mut page = if step == 0 {
                std::mem::replace(
                    &mut self.old_full_page_for_size[size_class],
                    ptr::null_mut(),
                )
            } else {
                std::mem::replace(&mut self.old_page_for_size[size_class], ptr::null_mut())
            };
            while !page.is_null() {
                let surviving = self.walk_page(page, block_size, ok_to_free_func);
                let nextpage = unsafe { (*page).nextpage };
                if surviving == nblocks {
                    assert_eq!(step, 0, "a non-full page became full while freeing");
                    unsafe { (*page).nextpage = remaining_full_pages };
                    remaining_full_pages = page;
                } else if surviving > 0 {
                    unsafe { (*page).nextpage = remaining_partial_pages };
                    remaining_partial_pages = page;
                } else {
                    self.free_page(page);
                }
                max_pages -= 1;
                if max_pages == 0 {
                    if step == 0 {
                        self.old_full_page_for_size[size_class] = nextpage;
                    } else {
                        self.old_page_for_size[size_class] = nextpage;
                    }
                    self.page_for_size[size_class] = remaining_partial_pages;
                    self.full_page_for_size[size_class] = remaining_full_pages;
                    return 0;
                }
                page = nextpage;
            }
        }
        self.page_for_size[size_class] = remaining_partial_pages;
        self.full_page_for_size[size_class] = remaining_full_pages;
        max_pages
    }

    /// `ArenaCollection.free_page`.
    fn free_page(&mut self, page: *mut PageHeader) {
        unsafe {
            let arena = (*page).arena;
            (*arena).nfreepages += 1;
            let pageaddr = page as *mut u8;
            *(pageaddr as *mut *mut u8) = (*arena).freepages;
            (*arena).freepages = pageaddr;
        }
    }

    /// `ArenaCollection.walk_page`.
    fn walk_page(
        &mut self,
        page: *mut PageHeader,
        block_size: usize,
        ok_to_free_func: &mut impl FnMut(*mut u8) -> bool,
    ) -> usize {
        unsafe {
            let mut freeblock = (*page).freeblock;
            let mut prevfreeblockat: *mut *mut u8 = ptr::addr_of_mut!((*page).freeblock);
            let mut obj = (page as *mut u8).add(self.hdrsize);
            let mut surviving = 0;
            let mut freed = 0;
            let mut skip_free_blocks = (*page).nfree;
            loop {
                if obj == freeblock {
                    if skip_free_blocks == 0 {
                        break;
                    }
                    skip_free_blocks -= 1;
                    prevfreeblockat = obj as *mut *mut u8;
                    freeblock = *prevfreeblockat;
                } else {
                    assert!(
                        (freeblock as usize) > obj as usize,
                        "freeblocks are not ordered"
                    );
                    if ok_to_free_func(obj) {
                        *prevfreeblockat = obj;
                        prevfreeblockat = obj as *mut *mut u8;
                        *prevfreeblockat = freeblock;
                        (*page).nfree += 1;
                        freed += 1;
                    } else {
                        surviving += 1;
                    }
                }
                obj = obj.add(block_size);
            }
            self.total_memory_used -= freed * block_size;
            self.live_objects -= freed;
            surviving
        }
    }

    #[cfg(test)]
    pub(crate) fn object_count(&self) -> usize {
        self.live_objects
    }

    /// Arena-owned address-range membership, matching the answer available to
    /// incminimark from its arena pages.  It intentionally does not distinguish
    /// live blocks from free/uninitialized bytes inside a live arena.  The
    /// allocator's bucketed arena lists remain the source of allocation state;
    /// this sorted flat index serves pyre's arbitrary-word membership query.
    pub fn contains(&self, addr: usize) -> bool {
        let index = self
            .arena_ranges
            .partition_point(|&(start, _)| start <= addr);
        index > 0 && addr < self.arena_ranges[index - 1].1
    }

    unsafe fn free_arena(&mut self, arena: *mut ArenaReference) {
        unsafe {
            let base = (*arena).base as usize;
            let index = self
                .arena_ranges
                .binary_search_by_key(&base, |&(start, _)| start)
                .expect("freed arena missing from range index");
            self.arena_ranges.remove(index);
            alloc::dealloc((*arena).base, (*arena).layout);
            self.total_memory_alloced -= self.arena_size;
            self.arenas_count -= 1;
            drop(Box::from_raw(arena));
        }
    }
}

impl Drop for ArenaCollection {
    fn drop(&mut self) {
        unsafe {
            if !self.current_arena.is_null() {
                self.free_arena(self.current_arena);
                self.current_arena = ptr::null_mut();
            }
            for i in 0..self.arenas_lists.len() {
                let mut arena = self.arenas_lists[i];
                self.arenas_lists[i] = ptr::null_mut();
                while !arena.is_null() {
                    let next = (*arena).nextarena;
                    self.free_arena(arena);
                    arena = next;
                }
            }
        }
    }
}

/// `minimarkpage.py:start_of_page` (translated path).
pub fn start_of_page(addr: usize, page_size: usize) -> usize {
    addr - addr % page_size
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE_SIZE: usize = 256;
    const ARENA_SIZE: usize = PAGE_SIZE * 8;
    const THRESHOLD: usize = 9 * WORD;

    #[test]
    fn allocate_and_malloc_mixed_sizes() {
        let mut ac = ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, THRESHOLD);
        let a = ac.malloc(2 * WORD);
        let b = ac.malloc(3 * WORD);
        let c = ac.malloc(2 * WORD);
        assert_ne!(a, c);
        assert!(ac.contains(a as usize));
        assert!(ac.contains(b as usize));
        assert_eq!(ac.total_memory_used, 7 * WORD);
        assert_eq!((a as usize) % WORD, 0);
    }

    #[test]
    fn malloc_reuses_ordered_free_blocks() {
        let mut ac = ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, THRESHOLD);
        let objects: Vec<_> = (0..5).map(|_| ac.malloc(2 * WORD)).collect();
        ac.mass_free(|obj| obj == objects[1] || obj == objects[3]);
        assert_eq!(ac.total_memory_used, 3 * 2 * WORD);
        assert_eq!(ac.malloc(2 * WORD), objects[1]);
        assert_eq!(ac.malloc(2 * WORD), objects[3]);
    }

    #[test]
    fn mass_free_partial_and_emptied_pages() {
        let mut ac = ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, THRESHOLD);
        let block_size = 2 * WORD;
        let per_page = (PAGE_SIZE - std::mem::size_of::<PageHeader>()) / block_size;
        let objects: Vec<_> = (0..per_page + 2).map(|_| ac.malloc(block_size)).collect();
        let survivor = objects[0];
        ac.mass_free(|obj| obj != survivor);
        assert_eq!(ac.total_memory_used, block_size);
        assert!(ac.contains(survivor as usize));
        let reused = ac.malloc(block_size);
        assert_ne!(reused, survivor);
    }

    #[test]
    fn whole_arena_is_returned_after_mass_free() {
        let mut ac = ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, THRESHOLD);
        let first = ac.malloc(WORD) as usize;
        while !ac.current_arena.is_null() {
            ac.malloc(WORD);
        }
        assert_eq!(ac.arenas_count, 1);
        ac.mass_free(|_| true);
        assert_eq!(ac.total_memory_used, 0);
        assert_eq!(ac.arenas_count, 0);
        assert_eq!(ac.total_memory_alloced, 0);
        assert!(!ac.contains(first));
    }

    #[test]
    fn randomized_mass_free_matches_live_set() {
        let mut ac = ArenaCollection::new(ARENA_SIZE, PAGE_SIZE, THRESHOLD);
        let mut live = Vec::new();
        let mut state = 0x1234_5678_u64;
        for _round in 0..50 {
            for _ in 0..37 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let size = ((state as usize % 6) + 1) * WORD;
                live.push((ac.malloc(size), size));
            }
            let before = ac.total_memory_used;
            let mut decisions: Vec<_> = live
                .iter()
                .map(|&(ptr, size)| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (ptr, size, (state >> 63) != 0, false)
                })
                .collect();
            ac.mass_free(|obj| {
                let entry = decisions.iter_mut().find(|entry| entry.0 == obj).unwrap();
                assert!(!entry.3);
                entry.3 = true;
                entry.2
            });
            assert!(decisions.iter().all(|entry| entry.3));
            let freed_bytes: usize = decisions
                .iter()
                .filter(|entry| entry.2)
                .map(|entry| entry.1)
                .sum();
            assert_eq!(ac.total_memory_used, before - freed_bytes);
            live = decisions
                .into_iter()
                .filter(|entry| !entry.2)
                .map(|entry| (entry.0, entry.1))
                .collect();
        }
    }

    #[test]
    fn start_of_page_rounds_down() {
        assert_eq!(start_of_page(0x12345, 0x1000), 0x12000);
    }
}
