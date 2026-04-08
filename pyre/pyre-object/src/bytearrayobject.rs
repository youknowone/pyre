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

/// Allocate a new bytearray filled with zeros.
pub fn w_bytearray_new(size: usize) -> PyObjectRef {
    let obj = Box::new(W_BytearrayObject {
        ob_header: PyObject {
            ob_type: &BYTEARRAY_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        data: Box::into_raw(Box::new(vec![0u8; size])),
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Allocate a new bytearray from a byte slice.
pub fn w_bytearray_from_bytes(bytes: &[u8]) -> PyObjectRef {
    let obj = Box::new(W_BytearrayObject {
        ob_header: PyObject {
            ob_type: &BYTEARRAY_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        data: Box::into_raw(Box::new(bytes.to_vec())),
    });
    Box::into_raw(obj) as PyObjectRef
}

pub unsafe fn is_bytearray(obj: PyObjectRef) -> bool {
    py_type_check(obj, &BYTEARRAY_TYPE)
}

pub unsafe fn w_bytearray_len(obj: PyObjectRef) -> usize {
    let ba = &*(obj as *const W_BytearrayObject);
    (*ba.data).len()
}

pub unsafe fn w_bytearray_getitem(obj: PyObjectRef, index: usize) -> u8 {
    let ba = &*(obj as *const W_BytearrayObject);
    (&*ba.data)[index]
}

pub unsafe fn w_bytearray_setitem(obj: PyObjectRef, index: usize, value: u8) {
    let ba = &mut *(obj as *mut W_BytearrayObject);
    (&mut *ba.data)[index] = value;
}

/// bytearray.find(sub, start) — find first occurrence of byte value.
pub unsafe fn w_bytearray_find(obj: PyObjectRef, value: u8, start: usize) -> i64 {
    let ba = &*(obj as *const W_BytearrayObject);
    let data = &*ba.data;
    for i in start..data.len() {
        if data[i] == value {
            return i as i64;
        }
    }
    -1
}

/// Concatenate bytearray + bytes (b'\0' * N pattern).
pub unsafe fn w_bytearray_extend(obj: PyObjectRef, other: &[u8]) {
    let ba = &mut *(obj as *mut W_BytearrayObject);
    (*ba.data).extend_from_slice(other);
}

/// Get a reference to the internal data.
pub unsafe fn w_bytearray_data(obj: PyObjectRef) -> &'static [u8] {
    let ba = &*(obj as *const W_BytearrayObject);
    &*ba.data
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
}
