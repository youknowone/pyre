//! llsupport/gcmap.py: GC bitmap allocation helpers.

use crate::arch::WORD;

/// llsupport/gcmap.py:7 allocate_gcmap.
pub fn allocate_gcmap(frame_depth: usize, fixed_size: usize) -> *mut usize {
    let size = frame_depth + fixed_size;
    let malloc_size = (size / WORD / 8 + 1) + 1;
    let mut gcmap = vec![0usize; malloc_size].into_boxed_slice();
    gcmap[0] = malloc_size - 1;
    Box::into_raw(gcmap) as *mut usize
}

pub fn gcmap_set_bit(gcmap: *mut usize, index: usize) {
    if gcmap.is_null() {
        return;
    }
    let word_index = index / (WORD * 8);
    let bit_index = index % (WORD * 8);
    unsafe {
        *gcmap.add(1 + word_index) |= 1usize << bit_index;
    }
}
