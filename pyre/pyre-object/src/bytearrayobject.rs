//! W_BytearrayObject — Python `bytearray` type.
//!
//! PyPy equivalent: pypy/objspace/std/bytearrayobject.py

use crate::pyobject::*;

pub static BYTEARRAY_TYPE: PyType = crate::pyobject::new_pytype("bytearray");

/// Python bytearray object.
///
/// Layout: `[ob_type | data]`
#[repr(C)]
pub struct W_BytearrayObject {
    pub ob_header: PyObject,
    pub data: *mut Vec<u8>,
}

/// GC type id assigned to `W_BytearrayObject` at JitDriver init time.
pub const W_BYTEARRAY_GC_TYPE_ID: u32 = 28;

/// Fixed payload size (`framework.py:811`).
pub const W_BYTEARRAY_OBJECT_SIZE: usize = std::mem::size_of::<W_BytearrayObject>();

impl crate::lltype::GcType for W_BytearrayObject {
    const TYPE_ID: u32 = W_BYTEARRAY_GC_TYPE_ID;
    const SIZE: usize = W_BYTEARRAY_OBJECT_SIZE;
}

/// Allocate a new bytearray filled with zeros.
pub fn w_bytearray_new(size: usize) -> PyObjectRef {
    let data = crate::lltype::malloc_raw(vec![0u8; size]);
    crate::lltype::malloc_typed(W_BytearrayObject {
        ob_header: PyObject {
            ob_type: &BYTEARRAY_TYPE as *const PyType,
            w_class: get_instantiate(&BYTEARRAY_TYPE),
        },
        data,
    }) as PyObjectRef
}

/// Allocate a new bytearray from a byte slice.
pub fn w_bytearray_from_bytes(bytes: &[u8]) -> PyObjectRef {
    let data = crate::lltype::malloc_raw(bytes.to_vec());
    crate::lltype::malloc_typed(W_BytearrayObject {
        ob_header: PyObject {
            ob_type: &BYTEARRAY_TYPE as *const PyType,
            w_class: get_instantiate(&BYTEARRAY_TYPE),
        },
        data,
    }) as PyObjectRef
}

pub unsafe fn is_bytearray(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BYTEARRAY_TYPE) }
}

pub unsafe fn w_bytearray_len(obj: PyObjectRef) -> usize {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        (*ba.data).len()
    }
}

pub unsafe fn w_bytearray_getitem(obj: PyObjectRef, index: usize) -> u8 {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        (&*ba.data)[index]
    }
}

pub unsafe fn w_bytearray_setitem(obj: PyObjectRef, index: usize, value: u8) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        (&mut *ba.data)[index] = value;
    }
}

/// bytearray.find(sub, start) — find first occurrence of byte value.
pub unsafe fn w_bytearray_find(obj: PyObjectRef, value: u8, start: usize) -> i64 {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        let data = &*ba.data;
        for i in start..data.len() {
            if data[i] == value {
                return i as i64;
            }
        }
        -1
    }
}

/// Concatenate bytearray + bytes (b'\0' * N pattern).
pub unsafe fn w_bytearray_extend(obj: PyObjectRef, other: &[u8]) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        (*ba.data).extend_from_slice(other);
    }
}

/// Get a reference to the internal data.
pub unsafe fn w_bytearray_data(obj: PyObjectRef) -> &'static [u8] {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        &*ba.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytearray_basic() {
        let ba = w_bytearray_new(10);
        unsafe {
            assert!(is_bytearray(ba));
            assert_eq!(w_bytearray_len(ba), 10);
            assert_eq!(w_bytearray_getitem(ba, 0), 0);
            w_bytearray_setitem(ba, 3, 1);
            assert_eq!(w_bytearray_getitem(ba, 3), 1);
            assert_eq!(w_bytearray_find(ba, 1, 0), 3);
            assert_eq!(w_bytearray_find(ba, 1, 4), -1);
        }
    }

    #[test]
    fn w_bytearray_gc_type_id_matches_descr() {
        assert_eq!(W_BYTEARRAY_GC_TYPE_ID, 28);
        assert_eq!(
            <W_BytearrayObject as crate::lltype::GcType>::TYPE_ID,
            W_BYTEARRAY_GC_TYPE_ID
        );
        assert_eq!(
            <W_BytearrayObject as crate::lltype::GcType>::SIZE,
            W_BYTEARRAY_OBJECT_SIZE
        );
    }
}
