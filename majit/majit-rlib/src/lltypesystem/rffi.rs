//! Raw-memory leaves: `lltype.malloc(..., flavor='raw')`, `lltype.free`,
//! `rffi.ptradd` and one-word loads and stores through a raw address.
//!
//! Each leaf carries the oopspec jtransform rewrites: the two allocation
//! leaves stay residual calls tagged `OS_RAW_MALLOC_VARSIZE_CHAR` /
//! `OS_RAW_FREE`, which `virtualize.py` turns into a virtual raw buffer, and
//! the access leaves become `int_add`, `raw_load` and `raw_store`.
//!
//! Addresses are spelled `usize`, not `*mut u8`: a raw block is neither traced
//! nor moved, so it belongs in the integer register bank. `*mut u8` is
//! `rffi.CCHARP` / `llmemory.Address`; [`super::llmemory::GCREF`] is the
//! erased managed object.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};

/// Alignment of every block [`raw_malloc_varsize_char`] returns, and the
/// header word [`raw_free`] reads the size back from.
const RAW_BLOCK_ALIGN: usize = std::mem::align_of::<u128>();

/// `lltype.malloc(rffi.CCHARP.TO, size, flavor='raw')` —
/// `support.py ll_raw_malloc_varsize_char`. A zero-size request still returns
/// a distinct address.
#[majit_macros::oopspec("raw_malloc_varsize_char(size)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_malloc_varsize_char(size: usize) -> usize {
    // The size is kept in front of the block so `raw_free(ptr)` can rebuild
    // the layout `dealloc` needs.
    let total = size.saturating_add(RAW_BLOCK_ALIGN);
    let layout = Layout::from_size_align(total, RAW_BLOCK_ALIGN)
        .unwrap_or_else(|_| handle_alloc_error(Layout::new::<u128>()));
    unsafe {
        let base = alloc(layout);
        if base.is_null() {
            handle_alloc_error(layout);
        }
        (base as *mut usize).write(total);
        base as usize + RAW_BLOCK_ALIGN
    }
}

/// `lltype.free(ptr, flavor='raw')` — `support.py ll_raw_free`. Frees a block
/// [`raw_malloc_varsize_char`] returned.
#[majit_macros::oopspec("raw_free(ptr)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_free(ptr: usize) {
    unsafe {
        let base = (ptr - RAW_BLOCK_ALIGN) as *mut u8;
        let total = (base as *const usize).read();
        dealloc(
            base,
            Layout::from_size_align_unchecked(total, RAW_BLOCK_ALIGN),
        );
    }
}

/// `rffi.ptradd` — advance a raw byte address without a memory access.
#[majit_macros::oopspec("raw_ptradd(ptr, offset)")]
#[majit_macros::elidable_cannot_raise]
pub fn raw_ptradd(ptr: usize, offset: usize) -> usize {
    ptr.wrapping_add(offset)
}

/// One word loaded from a raw address.
#[majit_macros::oopspec("raw_read_ptr(data)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_ptr(data: usize) -> usize {
    unsafe { (data as *const usize).read() }
}

/// One word stored to a raw address.
#[majit_macros::oopspec("raw_write_ptr(data, value)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_ptr(data: usize, value: usize) {
    unsafe { (data as *mut usize).write(value) }
}

/// One `f64` loaded from a raw address.
#[majit_macros::oopspec("raw_read_f64(data)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_f64(data: usize) -> f64 {
    unsafe { (data as *const f64).read() }
}

/// One `f64` stored to a raw address.
#[majit_macros::oopspec("raw_write_f64(data, value)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_f64(data: usize, value: f64) {
    unsafe { (data as *mut f64).write(value) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_block_round_trips_words() {
        let block = raw_malloc_varsize_char(3 * std::mem::size_of::<usize>());
        let word = std::mem::size_of::<usize>();
        for index in 0..3 {
            raw_write_ptr(raw_ptradd(block, index * word), index + 10);
        }
        for index in 0..3 {
            assert_eq!(raw_read_ptr(raw_ptradd(block, index * word)), index + 10);
        }
        raw_write_f64(block, 1.5);
        assert_eq!(raw_read_f64(block), 1.5);
        raw_free(block);
        let empty = raw_malloc_varsize_char(0);
        assert_ne!(empty, 0);
        raw_free(empty);
    }
}
