/// Bump-pointer nursery allocator.
///
/// A fixed-size memory region where young objects are allocated by
/// advancing a free pointer. When the nursery is full, a minor
/// collection copies live objects out.
///
/// Layout: [header0|payload0|header1|payload1|...|free...top]
///          ^nursery_start                       ^free  ^top

use std::alloc::{self, Layout};
use std::ptr;

use crate::header::GcHeader;

/// Default nursery size: 896KB, matching incminimark's TRANSLATION_PARAMS.
pub const DEFAULT_NURSERY_SIZE: usize = 896 * 1024;

/// Nursery memory region with bump-pointer allocation.
pub struct Nursery {
    /// Start of the nursery memory region.
    start: *mut u8,
    /// Current allocation pointer (next free byte).
    free: *mut u8,
    /// End of the nursery region (one past the last byte).
    top: *const u8,
    /// Total size of the nursery.
    size: usize,
}

// Safety: The nursery owns its memory exclusively and only one thread accesses it.
unsafe impl Send for Nursery {}

impl Nursery {
    /// Create a new nursery of the given size (in bytes).
    pub fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, 16).expect("invalid nursery layout");
        let start = unsafe { alloc::alloc_zeroed(layout) };
        if start.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Nursery {
            start,
            free: start,
            top: unsafe { start.add(size) },
            size,
        }
    }

    /// Try to allocate `total_size` bytes (header + payload) from the nursery.
    ///
    /// Returns a pointer to the start of the allocated region (i.e., where
    /// the GcHeader will be written), or null if there's not enough space.
    ///
    /// The allocated memory is already zero-filled (from initialization or
    /// post-collection reset).
    #[inline]
    pub fn alloc(&mut self, total_size: usize) -> *mut u8 {
        // Ensure minimum size for forwarding during collection.
        let total_size = total_size.max(GcHeader::MIN_NURSERY_OBJ_SIZE);
        // Align to 8 bytes.
        let total_size = (total_size + 7) & !7;

        let new_free = unsafe { self.free.add(total_size) };
        if new_free as *const u8 > self.top {
            return ptr::null_mut();
        }
        let result = self.free;
        self.free = new_free;
        result
    }

    /// Reset the nursery: set the free pointer back to start and zero-fill.
    pub fn reset(&mut self) {
        unsafe {
            ptr::write_bytes(self.start, 0, self.size);
        }
        self.free = self.start;
    }

    /// Current free pointer.
    #[inline]
    pub fn free_ptr(&self) -> *mut u8 {
        self.free
    }

    /// Set the free pointer (used after collection with pinned objects).
    ///
    /// # Safety
    /// `ptr` must be within the nursery bounds.
    pub unsafe fn set_free_ptr(&mut self, ptr: *mut u8) {
        debug_assert!(ptr as usize >= self.start as usize);
        debug_assert!(ptr as usize <= self.top as usize);
        self.free = ptr;
    }

    /// End of nursery (top pointer).
    #[inline]
    pub fn top_ptr(&self) -> *const u8 {
        self.top
    }

    /// Start of nursery.
    #[inline]
    pub fn start_ptr(&self) -> *const u8 {
        self.start
    }

    /// Total nursery size.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Bytes currently used.
    #[inline]
    pub fn used(&self) -> usize {
        self.free as usize - self.start as usize
    }

    /// Bytes remaining.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.top as usize - self.free as usize
    }

    /// Check if an address is within the nursery.
    #[inline]
    pub fn contains(&self, addr: usize) -> bool {
        addr >= self.start as usize && addr < self.top as usize
    }
}

impl Drop for Nursery {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, 16).unwrap();
        unsafe {
            alloc::dealloc(self.start, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nursery_create() {
        let nursery = Nursery::new(4096);
        assert_eq!(nursery.size(), 4096);
        assert_eq!(nursery.used(), 0);
        assert_eq!(nursery.remaining(), 4096);
    }

    #[test]
    fn test_nursery_alloc() {
        let mut nursery = Nursery::new(4096);

        let p1 = nursery.alloc(32);
        assert!(!p1.is_null());
        // At least MIN_NURSERY_OBJ_SIZE, aligned to 8
        assert!(nursery.used() >= 32);

        let p2 = nursery.alloc(64);
        assert!(!p2.is_null());
        assert!(p2 > p1);
    }

    #[test]
    fn test_nursery_full() {
        let mut nursery = Nursery::new(64);

        let p1 = nursery.alloc(32);
        assert!(!p1.is_null());

        // The nursery is 64 bytes; after allocating 32 (aligned up to at least 16),
        // we may or may not have room for another 32.
        // But asking for the full size should eventually fail.
        let p_big = nursery.alloc(64);
        // Should fail: not enough space
        assert!(p_big.is_null());
    }

    #[test]
    fn test_nursery_reset() {
        let mut nursery = Nursery::new(4096);

        nursery.alloc(100);
        nursery.alloc(100);
        assert!(nursery.used() > 0);

        nursery.reset();
        assert_eq!(nursery.used(), 0);
        assert_eq!(nursery.remaining(), 4096);
    }

    #[test]
    fn test_nursery_contains() {
        let nursery = Nursery::new(4096);
        let start = nursery.start_ptr() as usize;
        let top = nursery.top_ptr() as usize;

        assert!(nursery.contains(start));
        assert!(nursery.contains(start + 100));
        assert!(!nursery.contains(top));
        assert!(!nursery.contains(start.wrapping_sub(1)));
    }

    #[test]
    fn test_nursery_zero_filled() {
        let mut nursery = Nursery::new(4096);
        let p = nursery.alloc(64);
        assert!(!p.is_null());
        // Check memory is zero-filled
        for i in 0..64 {
            assert_eq!(unsafe { *p.add(i) }, 0);
        }
    }

    #[test]
    fn test_nursery_alignment() {
        let mut nursery = Nursery::new(4096);
        for _ in 0..10 {
            let p = nursery.alloc(17); // unaligned request
            assert!(!p.is_null());
            assert_eq!(p as usize % 8, 0); // result is 8-byte aligned
        }
    }
}
