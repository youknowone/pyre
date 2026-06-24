//! Small-object page allocator — port of `minimarkpage.py` `ArenaCollection`.
//!
//! Memory is subdivided into "arenas" containing "pages"; a page contains
//! fixed-size "blocks" of a single size class. The structure mirrors
//! `rpython/memory/gc/minimarkpage.py` method-for-method (`malloc`,
//! `allocate_new_page`, `allocate_new_arena`, `mass_free`, `mass_free_in_pages`,
//! `free_page`, `walk_page`, `_pick_next_arena`, `_rehash_arenas_lists`).
//!
//! RPython runs this over `llarena`, a simulated arena for untranslated tests
//! that collapses to plain pointer arithmetic over `arena_malloc`'d memory when
//! translated. This port uses real memory directly (the translated semantics):
//! `arena_malloc` is a page-aligned `std::alloc::alloc_zeroed`, and
//! `arena_reserve`/`arena_reset` collapse to nothing (we write the real bytes
//! that thread the free lists). `PAGE_HEADER` is embedded at the page start
//! exactly as upstream; `ARENA` is a raw `Box` (mirroring upstream's
//! `lltype.malloc(ARENA, flavor='raw')`) threaded by `nextarena`.

use std::alloc::{self, Layout};

/// Bytes per machine word. `WORD_POWER_2` is `log2(WORD)`.
const WORD: usize = std::mem::size_of::<usize>();
const WORD_POWER_2: u32 = WORD.trailing_zeros();

/// Read/write a pointer-sized address stored at `p` (free-list threading,
/// `result.address[0]` upstream).
#[inline]
unsafe fn read_addr(p: *mut u8) -> *mut u8 {
    *(p as *mut *mut u8)
}
#[inline]
unsafe fn write_addr(p: *mut u8, v: *mut u8) {
    *(p as *mut *mut u8) = v;
}

/// `minimarkpage.ARENA`: bookkeeping for one `arena_malloc`'d region.
struct Arena {
    /// Start of the malloced memory (`arena_malloc` result). Page-aligned here.
    base: *mut u8,
    /// Number of free pages and total pages in this arena.
    nfreepages: usize,
    totalpages: usize,
    /// Chained list of free pages (each free page's first word points to the
    /// next; ends NULL). Initially the first uninitialized page address.
    freepages: *mut u8,
    /// Linked list of arenas (within an `arenas_lists[i]` bucket).
    nextarena: *mut Arena,
}

/// `minimarkpage.PAGE_HEADER`, embedded at the start of every allocated page.
#[repr(C)]
struct PageHeader {
    /// Chained list of pages in the same size class (partial or full lists),
    /// or the arena free-page list for free pages.
    nextpage: *mut PageHeader,
    /// The arena this page belongs to.
    arena: *mut Arena,
    /// Number of free blocks threaded onto `freeblock` (excludes the trailing
    /// uninitialized blocks).
    nfree: usize,
    /// Next free block; ends as a pointer to the first uninitialized block (or
    /// the end of the page).
    freeblock: *mut u8,
}

const PAGE_NULL: *mut PageHeader = std::ptr::null_mut();
const ARENA_NULL: *mut Arena = std::ptr::null_mut();

/// Port of `minimarkpage.ArenaCollection`.
pub struct ArenaCollection {
    arena_size: usize,
    page_size: usize,
    /// Largest size accepted by `malloc` (inclusive).
    small_request_threshold: usize,
    hdrsize: usize,

    /// Per-size-class lists (index = `nsize >> WORD_POWER_2`); index 0 unused.
    page_for_size: Vec<*mut PageHeader>,
    full_page_for_size: Vec<*mut PageHeader>,
    /// Snapshots moved aside by `mass_free_prepare`.
    old_page_for_size: Vec<*mut PageHeader>,
    old_full_page_for_size: Vec<*mut PageHeader>,
    /// Blocks per page for each size class.
    nblocks_for_size: Vec<usize>,

    max_pages_per_arena: usize,
    /// `arenas_lists[i]` = head of arenas whose `nfreepages == i`.
    arenas_lists: Vec<*mut Arena>,
    old_arenas_lists: Vec<*mut Arena>,
    /// Arena currently consumed; not in any `arenas_lists`.
    current_arena: *mut Arena,
    min_empty_nfreepages: usize,
    num_uninitialized_pages: usize,

    /// Bytes in use, counting every block (no bookkeeping overhead).
    total_memory_used: usize,

    arenas_count: usize,
}

// Single-threaded GC ownership, like the rest of majit-gc.
unsafe impl Send for ArenaCollection {}

impl ArenaCollection {
    pub fn new(arena_size: usize, page_size: usize, small_request_threshold: usize) -> Self {
        let length = small_request_threshold / WORD + 1;
        let hdrsize = std::mem::size_of::<PageHeader>();
        assert!(page_size > hdrsize, "page_size must exceed the page header");
        assert!(
            arena_size >= page_size && arena_size % page_size == 0,
            "arena_size must be a positive multiple of page_size"
        );

        let mut nblocks_for_size = vec![0usize; length];
        for (i, slot) in nblocks_for_size.iter_mut().enumerate().skip(1) {
            *slot = (page_size - hdrsize) / (WORD * i);
        }

        let max_pages_per_arena = arena_size / page_size;

        ArenaCollection {
            arena_size,
            page_size,
            small_request_threshold,
            hdrsize,
            page_for_size: vec![PAGE_NULL; length],
            full_page_for_size: vec![PAGE_NULL; length],
            old_page_for_size: vec![PAGE_NULL; length],
            old_full_page_for_size: vec![PAGE_NULL; length],
            nblocks_for_size,
            max_pages_per_arena,
            arenas_lists: vec![ARENA_NULL; max_pages_per_arena],
            old_arenas_lists: vec![ARENA_NULL; max_pages_per_arena],
            current_arena: ARENA_NULL,
            min_empty_nfreepages: max_pages_per_arena,
            num_uninitialized_pages: 0,
            total_memory_used: 0,
            arenas_count: 0,
        }
    }

    /// Round a request up to a whole number of words.
    #[inline]
    fn round_up(size: usize) -> usize {
        (size + WORD - 1) & !(WORD - 1)
    }

    pub fn total_memory_used(&self) -> usize {
        self.total_memory_used
    }

    pub fn arenas_count(&self) -> usize {
        self.arenas_count
    }

    /// `minimarkpage.malloc`: allocate one block large enough for `size`.
    /// Returns a pointer to the block (the allocation start). `size` must be
    /// `> 0` and `<= small_request_threshold`.
    pub fn malloc(&mut self, size: usize) -> *mut u8 {
        let nsize = Self::round_up(size);
        debug_assert!(nsize > 0, "malloc: size is null");
        debug_assert!(
            nsize <= self.small_request_threshold,
            "malloc: size too big for the arena allocator"
        );
        self.total_memory_used += nsize;

        let size_class = nsize >> WORD_POWER_2;
        let mut page = self.page_for_size[size_class];
        if page == PAGE_NULL {
            page = self.allocate_new_page(size_class);
        }

        // The result is `page.freeblock`.
        let result = unsafe { (*page).freeblock };
        let freeblock = if unsafe { (*page).nfree } > 0 {
            // `result` was a threaded free block; read the next one.
            unsafe {
                (*page).nfree -= 1;
                read_addr(result)
            }
        } else {
            // `result` is the first uninitialized block.
            unsafe { result.add(nsize) }
        };
        unsafe {
            (*page).freeblock = freeblock;
        }

        let pageaddr = unsafe { page as *mut u8 };
        if (freeblock as usize) - (pageaddr as usize) > self.page_size - nsize {
            // No room for another block: move the page to the full list.
            unsafe {
                self.page_for_size[size_class] = (*page).nextpage;
                (*page).nextpage = self.full_page_for_size[size_class];
            }
            self.full_page_for_size[size_class] = page;
        }

        result
    }

    /// `minimarkpage.allocate_new_page`: take a free page from `current_arena`,
    /// initialize its header for `size_class`, and root it in `page_for_size`.
    fn allocate_new_page(&mut self, size_class: usize) -> *mut PageHeader {
        if self.current_arena == ARENA_NULL {
            self.allocate_new_arena();
        }
        let arena = self.current_arena;
        let result = unsafe { (*arena).freepages };

        let freepages = if unsafe { (*arena).nfreepages } > 0 {
            unsafe {
                (*arena).nfreepages -= 1;
                read_addr(result)
            }
        } else {
            debug_assert!(
                self.num_uninitialized_pages > 0,
                "fully allocated arena found in current_arena"
            );
            self.num_uninitialized_pages -= 1;
            if self.num_uninitialized_pages > 0 {
                unsafe { result.add(self.page_size) }
            } else {
                std::ptr::null_mut()
            }
        };

        unsafe {
            (*arena).freepages = freepages;
        }
        if freepages.is_null() {
            // Last page consumed: park the arena in arenas_lists[0].
            debug_assert!(unsafe { (*arena).nfreepages } == 0);
            unsafe {
                (*arena).nextarena = self.arenas_lists[0];
            }
            self.arenas_lists[0] = arena;
            self.current_arena = ARENA_NULL;
        }

        // Initialize the page header in place.
        let page = result as *mut PageHeader;
        unsafe {
            (*page).arena = arena;
            (*page).nfree = 0;
            (*page).freeblock = result.add(self.hdrsize);
            (*page).nextpage = PAGE_NULL;
        }
        debug_assert!(
            self.page_for_size[size_class] == PAGE_NULL,
            "allocate_new_page() called but a page is already waiting"
        );
        self.page_for_size[size_class] = page;
        page
    }

    /// `minimarkpage._pick_next_arena`: load `current_arena` from the
    /// non-empty `arenas_lists[i]` with the smallest `i > 0`.
    fn pick_next_arena(&mut self) -> bool {
        let mut i = self.min_empty_nfreepages;
        while i < self.max_pages_per_arena {
            if self.arenas_lists[i] != ARENA_NULL {
                self.current_arena = self.arenas_lists[i];
                self.arenas_lists[i] = unsafe { (*self.current_arena).nextarena };
                return true;
            }
            i += 1;
            self.min_empty_nfreepages = i;
        }
        false
    }

    /// `minimarkpage.allocate_new_arena`.
    fn allocate_new_arena(&mut self) {
        if self.pick_next_arena() {
            return;
        }
        // An incremental collection may have freed pages into arenas beyond
        // what arenas_lists[] accounts for; rehash and retry.
        self.rehash_arenas_lists();
        if self.pick_next_arena() {
            return;
        }

        // Allocate a fresh, page-aligned arena.
        let layout = Layout::from_size_align(self.arena_size, self.page_size)
            .expect("invalid arena layout");
        let arena_base = unsafe { alloc::alloc_zeroed(layout) };
        if arena_base.is_null() {
            alloc::handle_alloc_error(layout);
        }
        // Page-aligned base ⇒ the first page is the base, all pages usable.
        let npages = self.arena_size / self.page_size;

        let arena = Box::into_raw(Box::new(Arena {
            base: arena_base,
            nfreepages: 0, // all pages start uninitialized
            totalpages: npages,
            freepages: arena_base,
            nextarena: ARENA_NULL,
        }));
        self.num_uninitialized_pages = npages;
        self.current_arena = arena;
        self.arenas_count += 1;
    }

    /// `minimarkpage._rehash_arenas_lists`: re-bin arenas into
    /// `arenas_lists[nfreepages]`, freeing wholly-empty arenas to the OS.
    fn rehash_arenas_lists(&mut self) {
        std::mem::swap(&mut self.old_arenas_lists, &mut self.arenas_lists);
        for slot in self.arenas_lists.iter_mut() {
            *slot = ARENA_NULL;
        }

        for i in 0..self.max_pages_per_arena {
            let mut arena = self.old_arenas_lists[i];
            while arena != ARENA_NULL {
                let nextarena = unsafe { (*arena).nextarena };
                if unsafe { (*arena).nfreepages == (*arena).totalpages } {
                    // Whole arena empty: return it to the OS.
                    let base = unsafe { (*arena).base };
                    let layout = Layout::from_size_align(self.arena_size, self.page_size)
                        .expect("invalid arena layout");
                    unsafe {
                        alloc::dealloc(base, layout);
                        drop(Box::from_raw(arena));
                    }
                    self.arenas_count -= 1;
                } else {
                    let n = unsafe { (*arena).nfreepages };
                    debug_assert!(n < self.max_pages_per_arena);
                    unsafe {
                        (*arena).nextarena = self.arenas_lists[n];
                    }
                    self.arenas_lists[n] = arena;
                }
                arena = nextarena;
            }
            self.old_arenas_lists[i] = ARENA_NULL;
        }
        self.min_empty_nfreepages = 1;
    }

    /// True if `addr` lies within any arena's memory (the membership test that
    /// replaces a payload side-table; `Nursery::contains` is the model). Field
    /// targets are always object starts inside an arena; foreign (non-GC)
    /// addresses fall outside every arena and answer false without any deref.
    pub fn contains(&self, addr: usize) -> bool {
        if self.arena_holds(self.current_arena, addr) {
            return true;
        }
        for &head in &self.arenas_lists {
            let mut arena = head;
            while arena != ARENA_NULL {
                if self.arena_holds(arena, addr) {
                    return true;
                }
                arena = unsafe { (*arena).nextarena };
            }
        }
        false
    }

    #[inline]
    fn arena_holds(&self, arena: *mut Arena, addr: usize) -> bool {
        if arena == ARENA_NULL {
            return false;
        }
        let base = unsafe { (*arena).base } as usize;
        addr >= base && addr < base + self.arena_size
    }

    /// `minimarkpage.mass_free`: free every block for which `ok_to_free(block)`
    /// returns true; pages emptied entirely are returned to their arena's free
    /// list and wholly-empty arenas back to the OS.
    pub fn mass_free<F: FnMut(*mut u8) -> bool>(&mut self, mut ok_to_free: F) {
        self.mass_free_prepare();
        let mut size_class = self.small_request_threshold >> WORD_POWER_2;
        while size_class >= 1 {
            self.mass_free_in_pages(size_class, &mut ok_to_free);
            size_class -= 1;
        }
        self.rehash_arenas_lists();
    }

    /// `minimarkpage.mass_free_prepare`: move the live page lists aside so
    /// `mass_free_in_pages` can rebuild them.
    fn mass_free_prepare(&mut self) {
        let mut size_class = self.small_request_threshold >> WORD_POWER_2;
        while size_class >= 1 {
            self.old_page_for_size[size_class] = self.page_for_size[size_class];
            self.old_full_page_for_size[size_class] = self.full_page_for_size[size_class];
            self.page_for_size[size_class] = PAGE_NULL;
            self.full_page_for_size[size_class] = PAGE_NULL;
            size_class -= 1;
        }
    }

    /// `minimarkpage.mass_free_in_pages` (non-incremental): sweep every page of
    /// one size class, re-chaining survivors into the partial/full lists.
    fn mass_free_in_pages<F: FnMut(*mut u8) -> bool>(
        &mut self,
        size_class: usize,
        ok_to_free: &mut F,
    ) {
        let nblocks = self.nblocks_for_size[size_class];
        let block_size = size_class * WORD;
        let mut remaining_partial_pages = self.page_for_size[size_class];
        let mut remaining_full_pages = self.full_page_for_size[size_class];

        for step in 0..2 {
            let mut page = if step == 0 {
                let p = self.old_full_page_for_size[size_class];
                self.old_full_page_for_size[size_class] = PAGE_NULL;
                p
            } else {
                let p = self.old_page_for_size[size_class];
                self.old_page_for_size[size_class] = PAGE_NULL;
                p
            };
            while page != PAGE_NULL {
                let surviving = self.walk_page(page, block_size, ok_to_free);
                let nextpage = unsafe { (*page).nextpage };
                if surviving == nblocks {
                    debug_assert!(step == 0, "a non-full page became full while freeing");
                    unsafe {
                        (*page).nextpage = remaining_full_pages;
                    }
                    remaining_full_pages = page;
                } else if surviving > 0 {
                    unsafe {
                        (*page).nextpage = remaining_partial_pages;
                    }
                    remaining_partial_pages = page;
                } else {
                    self.free_page(page);
                }
                page = nextpage;
            }
        }
        self.page_for_size[size_class] = remaining_partial_pages;
        self.full_page_for_size[size_class] = remaining_full_pages;
    }

    /// `minimarkpage.free_page`: return a fully-empty page to its arena's free
    /// list (the page header is overwritten by the free-page link).
    fn free_page(&mut self, page: *mut PageHeader) {
        let arena = unsafe { (*page).arena };
        unsafe {
            (*arena).nfreepages += 1;
        }
        let pageaddr = page as *mut u8;
        unsafe {
            write_addr(pageaddr, (*arena).freepages);
            (*arena).freepages = pageaddr;
        }
    }

    /// `minimarkpage.walk_page`: visit every allocated block in `page`, asking
    /// `ok_to_free`; freed blocks are threaded back onto the page free list.
    /// Returns the number of surviving (not-freed) blocks.
    fn walk_page<F: FnMut(*mut u8) -> bool>(
        &mut self,
        page: *mut PageHeader,
        block_size: usize,
        ok_to_free: &mut F,
    ) -> usize {
        let mut freeblock = unsafe { (*page).freeblock };
        // Address of the slot from which `freeblock` was read (initially the
        // page's `freeblock` field, so freeing the head updates the page).
        let mut prevfreeblockat = unsafe { std::ptr::addr_of_mut!((*page).freeblock) as *mut u8 };
        let mut obj = unsafe { (page as *mut u8).add(self.hdrsize) };
        let mut surviving = 0usize;
        let mut freed = 0usize;
        let mut skip_free_blocks = unsafe { (*page).nfree };

        loop {
            if obj == freeblock {
                if skip_free_blocks == 0 {
                    // `obj` is the first uninitialized block, or the page end.
                    break;
                }
                skip_free_blocks -= 1;
                prevfreeblockat = obj;
                freeblock = unsafe { read_addr(obj) };
            } else {
                debug_assert!(
                    (freeblock as usize) > (obj as usize),
                    "freeblocks are linked out of order"
                );
                if ok_to_free(obj) {
                    // The block dies: insert it into the free list.
                    unsafe {
                        write_addr(prevfreeblockat, obj);
                        prevfreeblockat = obj;
                        write_addr(obj, freeblock);
                        (*page).nfree += 1;
                    }
                    freed += 1;
                } else {
                    surviving += 1;
                }
            }
            obj = unsafe { obj.add(block_size) };
        }
        self.total_memory_used -= freed * block_size;
        surviving
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // arena_size = 2 pages so a third page forces a second arena.
    fn ac() -> ArenaCollection {
        ArenaCollection::new(8192, 4096, 512)
    }

    #[test]
    fn malloc_distinct_aligned_contained() {
        let mut ac = ac();
        let a = ac.malloc(64);
        let b = ac.malloc(64);
        assert!(!a.is_null() && !b.is_null());
        assert_ne!(a, b);
        assert_eq!(a as usize % WORD, 0);
        assert!(ac.contains(a as usize));
        assert!(ac.contains(b as usize));
        assert!(!ac.contains(0xdead_beef_usize));
        assert!(!ac.contains(a as usize - 1_000_000));
        // The block is usable, full width.
        unsafe {
            std::ptr::write_bytes(a, 0xAB, 64);
            assert_eq!(*a, 0xAB);
            assert_eq!(*a.add(63), 0xAB);
        }
        assert_eq!(ac.total_memory_used(), 128);
    }

    #[test]
    fn rounds_up_to_word_and_size_classes_do_not_overlap() {
        let mut ac = ac();
        let small = ac.malloc(1); // -> 1 word
        let mid = ac.malloc(64);
        assert_eq!(small as usize % WORD, 0);
        assert_ne!(small, mid);
        assert_eq!(ac.total_memory_used(), WORD + 64);
        // Distinct size classes occupy distinct pages; both contained.
        assert!(ac.contains(small as usize));
        assert!(ac.contains(mid as usize));
    }

    #[test]
    fn spills_to_new_pages_and_arenas() {
        let mut ac = ac();
        // block_size 256 -> nblocks/page = (4096-32)/256 = 15; 2 pages/arena.
        let mut ptrs = Vec::new();
        for _ in 0..40 {
            ptrs.push(ac.malloc(256) as usize);
        }
        let uniq: HashSet<usize> = ptrs.iter().copied().collect();
        assert_eq!(uniq.len(), 40, "all blocks distinct");
        for &p in &ptrs {
            assert!(ac.contains(p));
        }
        assert!(ac.arenas_count() >= 2, "40 blocks > 30 per arena -> 2nd arena");
        assert_eq!(ac.total_memory_used(), 40 * 256);
    }

    #[test]
    fn mass_free_reclaims_and_reuses() {
        let mut ac = ac();
        // block_size 64 -> nblocks/page = (4096-32)/64 = 63; 20 fit in one page.
        let ptrs: Vec<usize> = (0..20).map(|_| ac.malloc(64) as usize).collect();
        let live: HashSet<usize> = ptrs.iter().copied().step_by(2).collect(); // 10 live
        let before = ac.total_memory_used();
        ac.mass_free(|b| !live.contains(&(b as usize)));
        assert_eq!(ac.total_memory_used(), before - 10 * 64);
        for &p in &live {
            assert!(ac.contains(p));
        }
        // A freed block is handed back out before any new page is taken.
        let reused = ac.malloc(64) as usize;
        assert!(ptrs.contains(&reused) && !live.contains(&reused));
    }

    #[test]
    fn mass_free_all_empties_and_releases() {
        let mut ac = ac();
        for _ in 0..40 {
            ac.malloc(256);
        }
        let peak = ac.arenas_count();
        assert!(peak >= 2);
        ac.mass_free(|_| true);
        assert_eq!(ac.total_memory_used(), 0);
        // Fully-consumed arenas are returned; at most the current arena remains.
        assert!(ac.arenas_count() < peak);
        // The collection is still usable afterwards.
        let p = ac.malloc(256);
        assert!(ac.contains(p as usize));
    }

    #[test]
    fn survivors_persist_across_two_collections() {
        let mut ac = ac();
        let keep = ac.malloc(128) as usize;
        for _ in 0..30 {
            ac.malloc(128);
        }
        let live: HashSet<usize> = [keep].into_iter().collect();
        ac.mass_free(|b| !live.contains(&(b as usize)));
        assert!(ac.contains(keep));
        ac.mass_free(|b| !live.contains(&(b as usize)));
        assert!(ac.contains(keep));
        assert_eq!(ac.total_memory_used(), 128);
    }
}

impl Drop for ArenaCollection {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.arena_size, self.page_size)
            .expect("invalid arena layout");
        let mut free_arena = |arena: *mut Arena| unsafe {
            if arena != ARENA_NULL {
                alloc::dealloc((*arena).base, layout);
                drop(Box::from_raw(arena));
            }
        };
        free_arena(self.current_arena);
        self.current_arena = ARENA_NULL;
        for i in 0..self.arenas_lists.len() {
            let mut arena = self.arenas_lists[i];
            while arena != ARENA_NULL {
                let next = unsafe { (*arena).nextarena };
                free_arena(arena);
                arena = next;
            }
            self.arenas_lists[i] = ARENA_NULL;
        }
    }
}
