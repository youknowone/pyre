//! W_BytesObject — Python `bytes` type (immutable).
//!
//! PyPy equivalent: pypy/objspace/std/bytesobject.py W_BytesObject
//!
//! Immutable byte sequence. Shares the same internal layout as
//! W_BytearrayObject but provides no mutation functions.

use crate::pyobject::*;

pub static BYTES_TYPE: PyType = crate::pyobject::new_pytype("bytes");

/// Python bytes object — immutable byte sequence.
///
/// PyPy: W_BytesObject stores `_value` (RPython string).
/// pyre: stores a heap-allocated `Vec<u8>` behind a raw pointer,
/// same layout as W_BytearrayObject but without setitem/extend.
#[repr(C)]
pub struct W_BytesObject {
    pub ob_header: PyObject,
    pub data: *const Vec<u8>,
    pub len: usize,
}

/// GC type id assigned to `W_BytesObject` at JitDriver init time.
pub const W_BYTES_GC_TYPE_ID: u32 = 27;

/// Fixed payload size (`framework.py:811`).
pub const W_BYTES_OBJECT_SIZE: usize = std::mem::size_of::<W_BytesObject>();

impl crate::lltype::GcType for W_BytesObject {
    const TYPE_ID: u32 = W_BYTES_GC_TYPE_ID;
    const SIZE: usize = W_BYTES_OBJECT_SIZE;
}

/// Allocate a new bytes object from a byte slice.
pub fn w_bytes_from_bytes(bytes: &[u8]) -> PyObjectRef {
    let len = bytes.len();
    // The `data` Vec lives on the raw heap (manually freed elsewhere),
    // so it is allocated through `malloc_raw`. The W_BytesObject itself
    // is GC-managed via `malloc_typed`.
    let data = crate::lltype::malloc_raw(bytes.to_vec());
    crate::lltype::malloc_typed(W_BytesObject {
        ob_header: PyObject {
            ob_type: &BYTES_TYPE as *const PyType,
            w_class: get_instantiate(&BYTES_TYPE),
        },
        data,
        len,
    }) as PyObjectRef
}

/// Allocate an empty bytes object.
pub fn w_bytes_empty() -> PyObjectRef {
    w_bytes_from_bytes(&[])
}

#[inline]
pub unsafe fn is_bytes(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BYTES_TYPE) }
}

#[inline]
pub unsafe fn w_bytes_len(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_BytesObject)).len }
}

#[inline]
pub unsafe fn w_bytes_getitem(obj: PyObjectRef, index: usize) -> u8 {
    unsafe { w_bytes_data(obj)[index] }
}

/// Get a reference to the internal data.
pub unsafe fn w_bytes_data(obj: PyObjectRef) -> &'static [u8] {
    unsafe {
        let b = obj as *const W_BytesObject;
        let data_ref: &Vec<u8> = &*(*b).data;
        data_ref.as_slice()
    }
}

/// bytes.find(sub, start) — find first occurrence of byte value.
pub unsafe fn w_bytes_find(obj: PyObjectRef, value: u8, start: usize) -> i64 {
    unsafe {
        let data = w_bytes_data(obj);
        for i in start..data.len() {
            if data[i] == value {
                return i as i64;
            }
        }
        -1
    }
}

// ── bytes-like helpers ────────────────────────────────────────────────
//
// Many Python operations accept both bytes and bytearray ("bytes-like").
// These helpers abstract over both types for read-only operations.

/// Check if obj is bytes or bytearray (bytes-like object).
#[inline]
pub unsafe fn is_bytes_like(obj: PyObjectRef) -> bool {
    unsafe { is_bytes(obj) || crate::bytearrayobject::is_bytearray(obj) }
}

/// Get length of a bytes-like object.
#[inline]
pub unsafe fn bytes_like_len(obj: PyObjectRef) -> usize {
    unsafe {
        if is_bytes(obj) {
            w_bytes_len(obj)
        } else {
            crate::bytearrayobject::w_bytearray_len(obj)
        }
    }
}

/// Get byte at index from a bytes-like object.
#[inline]
pub unsafe fn bytes_like_getitem(obj: PyObjectRef, index: usize) -> u8 {
    unsafe {
        if is_bytes(obj) {
            w_bytes_getitem(obj, index)
        } else {
            crate::bytearrayobject::w_bytearray_getitem(obj, index)
        }
    }
}

/// Get data slice from a bytes-like object.
#[inline]
pub unsafe fn bytes_like_data(obj: PyObjectRef) -> &'static [u8] {
    unsafe {
        if is_bytes(obj) {
            w_bytes_data(obj)
        } else {
            crate::bytearrayobject::w_bytearray_data(obj)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_basic() {
        let b = w_bytes_from_bytes(b"hello");
        unsafe {
            assert!(is_bytes(b));
            assert_eq!(w_bytes_len(b), 5);
            assert_eq!(w_bytes_getitem(b, 0), b'h');
            assert_eq!(w_bytes_getitem(b, 4), b'o');
            assert_eq!(w_bytes_data(b), b"hello");
            assert_eq!(w_bytes_find(b, b'l', 0), 2);
            assert_eq!(w_bytes_find(b, b'x', 0), -1);
        }
    }

    #[test]
    fn test_bytes_empty() {
        let b = w_bytes_empty();
        unsafe {
            assert!(is_bytes(b));
            assert_eq!(w_bytes_len(b), 0);
        }
    }

    #[test]
    fn w_bytes_gc_type_id_matches_descr() {
        assert_eq!(W_BYTES_GC_TYPE_ID, 27);
        assert_eq!(
            <W_BytesObject as crate::lltype::GcType>::TYPE_ID,
            W_BYTES_GC_TYPE_ID
        );
        assert_eq!(
            <W_BytesObject as crate::lltype::GcType>::SIZE,
            W_BYTES_OBJECT_SIZE
        );
    }
}
