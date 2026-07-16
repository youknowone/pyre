//! W_BytearrayObject — Python `bytearray` type.
//!
//! PyPy equivalent: pypy/objspace/std/bytearrayobject.py

use crate::pyobject::*;

pub static BYTEARRAY_TYPE: PyType = crate::pyobject::new_pytype("bytearray");

/// Python bytearray object.
///
/// Layout: `[ob_type | data | exports]`
#[repr(C)]
pub struct W_BytearrayObject {
    pub ob_header: PyObject,
    pub data: *mut Vec<u8>,
    /// `_exports` — count of active buffer exports.  Size-changing mutators
    /// are refused while this is positive (`_check_exports`).
    pub exports: i64,
}

/// GC type id assigned to `W_BytearrayObject` at JitDriver init time.
pub const W_BYTEARRAY_GC_TYPE_ID: u32 = 28;

/// Fixed payload size (`framework.py:811`).
pub const W_BYTEARRAY_OBJECT_SIZE: usize = std::mem::size_of::<W_BytearrayObject>();

impl crate::lltype::GcType for W_BytearrayObject {
    fn type_id() -> u32 {
        W_BYTEARRAY_GC_TYPE_ID
    }
    const SIZE: usize = W_BYTEARRAY_OBJECT_SIZE;
}

/// Free the off-GC byte buffer owned by a `W_BytearrayObject`.
///
/// # Safety
/// `obj` must point at a valid `W_BytearrayObject` whose `data` Box is not
/// aliased by another owner.
pub unsafe fn w_bytearray_dealloc(obj: PyObjectRef) {
    let raw = unsafe { &mut *(obj as *mut W_BytearrayObject) };
    if !raw.data.is_null() {
        unsafe { drop(Box::from_raw(raw.data)) };
        raw.data = std::ptr::null_mut();
    }
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
        exports: 0,
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
        exports: 0,
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

/// Get a mutable reference to the internal data. Caller must ensure
/// the bytearray is not aliased while the returned slice is live.
pub unsafe fn w_bytearray_data_mut(obj: PyObjectRef) -> &'static mut [u8] {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        &mut *ba.data
    }
}

/// Get a mutable reference to the backing `Vec`, for length-changing
/// mutators (append / insert / remove / pop / clear).  Caller must
/// ensure the bytearray is not aliased while the reference is live.
pub unsafe fn w_bytearray_vec_mut(obj: PyObjectRef) -> &'static mut Vec<u8> {
    unsafe {
        let ba = &*(obj as *const W_BytearrayObject);
        &mut *ba.data
    }
}

/// `_exports` — number of live buffer exports over this bytearray.
pub unsafe fn w_bytearray_exports(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_BytearrayObject)).exports }
}

/// `buffer_w` — record a new live buffer export.
pub unsafe fn w_bytearray_exports_incref(obj: PyObjectRef) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        ba.exports += 1;
    }
}

/// `bf_releasebuffer` — a consumer released its buffer export.  A release
/// without a matching acquisition is a fatal accounting bug
/// (`_exports_underflow`).
pub unsafe fn w_bytearray_exports_decref(obj: PyObjectRef) {
    unsafe {
        let ba = &mut *(obj as *mut W_BytearrayObject);
        if ba.exports <= 0 {
            panic!(
                "bytearray bf_releasebuffer: _exports underflow: id={obj:?} exports={}",
                ba.exports
            );
        }
        ba.exports -= 1;
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
            <W_BytearrayObject as crate::lltype::GcType>::type_id(),
            W_BYTEARRAY_GC_TYPE_ID
        );
        assert_eq!(
            <W_BytearrayObject as crate::lltype::GcType>::SIZE,
            W_BYTEARRAY_OBJECT_SIZE
        );
    }
}
