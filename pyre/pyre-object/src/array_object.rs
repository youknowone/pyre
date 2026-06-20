//! W_ArrayObject — Python `array.array` type.
//!
//! PyPy: pypy/module/array/interp_array.py
//!
//! A fixed `#[pyre_class]` header carrying the typecode and item size plus
//! an off-GC `*mut Vec<u8>` element buffer (the `bytearray` storage model).
//! Elements are unboxed scalars stored in native machine byte order, so the
//! collector traces no inner pointers (zero GC ptr offsets).  Boxing an
//! element back into a Python object (`w_array_unpack_item`) lives here;
//! the reverse direction (range-checked packing of a Python object into
//! bytes) needs `int_w`/`float_w` and so lives in the interpreter.

use crate::pyobject::*;
use malachite_bigint::BigInt;
use pyre_macros::pyre_class;
use rustpython_wtf8::{CodePoint, Wtf8Buf};

/// Python `array.array` object.
///
/// `data` points to a heap `Vec<u8>` holding `len * itemsize` bytes in
/// native byte order; the live element count is `data.len() / itemsize`.
#[pyre_class("array.array", static_name = "ARRAY")]
pub struct W_ArrayObject {
    pub typecode: u8,
    pub itemsize: u8,
    pub data: *mut Vec<u8>,
}

/// The supported typecodes, in `array.typecodes` order
/// (`interp_array.py:904`).  No `w` code — this mirrors the PyPy set.
pub const TYPECODES: &str = "bBuhHiIlLqQfd";

/// `itemsize` (bytes per element) for a typecode, or `None` if the code is
/// not one of the supported `bBuhHiIlLqQfd` (`interp_array.py:885-899`,
/// 64-bit `l`/`L` = 8).
pub fn typecode_itemsize(tc: u8) -> Option<u8> {
    Some(match tc {
        b'b' | b'B' => 1,
        b'u' => 4,
        b'h' | b'H' => 2,
        b'i' | b'I' => 4,
        b'l' | b'L' => 8,
        b'q' | b'Q' => 8,
        b'f' => 4,
        b'd' => 8,
        _ => return None,
    })
}

/// Allocate an empty array of the given typecode.  `itemsize` must match
/// `typecode_itemsize(typecode)` (the caller validates the code).
pub fn w_array_new(typecode: u8, itemsize: u8) -> PyObjectRef {
    let data = crate::lltype::malloc_raw(Vec::<u8>::new());
    W_ArrayObject::allocate(W_ArrayObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        typecode,
        itemsize,
        data,
    })
}

/// Allocate an array from raw native-order element bytes.  `bytes.len()`
/// must be a multiple of `itemsize`.
pub fn w_array_from_bytes(typecode: u8, itemsize: u8, bytes: Vec<u8>) -> PyObjectRef {
    let data = crate::lltype::malloc_raw(bytes);
    W_ArrayObject::allocate(W_ArrayObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        typecode,
        itemsize,
        data,
    })
}

/// # Safety
/// `obj` must be a valid, non-null `PyObject` pointer.
#[inline]
pub unsafe fn is_array(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ARRAY_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `W_ArrayObject`.
pub unsafe fn w_array_typecode(obj: PyObjectRef) -> u8 {
    unsafe {
        let a = &*(obj as *const W_ArrayObject);
        a.typecode
    }
}

/// # Safety
/// `obj` must point to a valid `W_ArrayObject`.
pub unsafe fn w_array_itemsize(obj: PyObjectRef) -> usize {
    unsafe {
        let a = &*(obj as *const W_ArrayObject);
        a.itemsize as usize
    }
}

/// Live element count (`self.len`).
///
/// # Safety
/// `obj` must point to a valid `W_ArrayObject`.
pub unsafe fn w_array_len(obj: PyObjectRef) -> usize {
    unsafe {
        let a = &*(obj as *const W_ArrayObject);
        let data = &*a.data;
        data.len() / a.itemsize as usize
    }
}

/// Borrow the raw native-order element bytes (`len * itemsize`).
///
/// # Safety
/// `obj` must point to a valid `W_ArrayObject`; the array must not be
/// mutated while the slice is live.
pub unsafe fn w_array_bytes(obj: PyObjectRef) -> &'static [u8] {
    unsafe {
        let a = &*(obj as *const W_ArrayObject);
        &*a.data
    }
}

/// Borrow the backing byte `Vec` mutably (for length-changing mutators).
///
/// # Safety
/// `obj` must point to a valid `W_ArrayObject`; the array must not be
/// aliased while the reference is live.
pub unsafe fn w_array_vec_mut(obj: PyObjectRef) -> &'static mut Vec<u8> {
    unsafe {
        let a = &*(obj as *const W_ArrayObject);
        &mut *a.data
    }
}

/// Box element `index` as a Python object per the array's typecode
/// (`interp_array.py W_Array.w_getitem`).  `index` must be `< len`.
///
/// # Safety
/// `obj` must point to a valid `W_ArrayObject` and `index < w_array_len`.
pub unsafe fn w_array_unpack_item(obj: PyObjectRef, index: usize) -> PyObjectRef {
    unsafe {
        let a = &*(obj as *const W_ArrayObject);
        let isz = a.itemsize as usize;
        let off = index * isz;
        let data = &*a.data;
        unpack_value(a.typecode, &data[off..off + isz])
    }
}

/// Box a single element from `buf` (exactly `itemsize` native-order bytes).
pub fn unpack_value(typecode: u8, buf: &[u8]) -> PyObjectRef {
    match typecode {
        b'b' => crate::intobject::w_int_new(buf[0] as i8 as i64),
        b'B' => crate::intobject::w_int_new(buf[0] as i64),
        b'h' => crate::intobject::w_int_new(i16::from_ne_bytes([buf[0], buf[1]]) as i64),
        b'H' => crate::intobject::w_int_new(u16::from_ne_bytes([buf[0], buf[1]]) as i64),
        b'i' => crate::intobject::w_int_new(i32::from_ne_bytes(buf.try_into().unwrap()) as i64),
        b'I' => crate::intobject::w_int_new(u32::from_ne_bytes(buf.try_into().unwrap()) as i64),
        b'l' | b'q' => crate::intobject::w_int_new(i64::from_ne_bytes(buf.try_into().unwrap())),
        b'L' | b'Q' => {
            let v = u64::from_ne_bytes(buf.try_into().unwrap());
            if v <= i64::MAX as u64 {
                crate::intobject::w_int_new(v as i64)
            } else {
                crate::longobject::w_long_new(BigInt::from(v))
            }
        }
        b'f' => crate::floatobject::w_float_new(f32::from_ne_bytes(buf.try_into().unwrap()) as f64),
        b'd' => crate::floatobject::w_float_new(f64::from_ne_bytes(buf.try_into().unwrap())),
        b'u' => {
            let cp = u32::from_ne_bytes(buf.try_into().unwrap());
            match char::from_u32(cp) {
                Some(c) => crate::strobject::w_str_new(&c.to_string()),
                None => {
                    // Lone surrogate / out-of-range Py_UCS4 — represent via
                    // WTF-8 (an out-of-range value yields the empty string).
                    let mut wb = Wtf8Buf::new();
                    if let Some(point) = CodePoint::from_u32(cp) {
                        wb.push(point);
                    }
                    crate::strobject::w_str_from_wtf8(wb)
                }
            }
        }
        _ => PY_NULL,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_array_gc_descriptor_has_no_traced_pointers() {
        // Elements are unboxed scalars in an off-GC byte buffer, so the
        // header carries no traced edges.
        assert_eq!(W_ARRAY_GC_PTR_OFFSETS.len(), 0);
        assert_eq!(
            <W_ArrayObject as crate::lltype::GcType>::SIZE,
            W_ARRAY_OBJECT_SIZE
        );
    }

    #[test]
    fn typecode_itemsizes() {
        for (tc, sz) in [
            (b'b', 1),
            (b'B', 1),
            (b'u', 4),
            (b'h', 2),
            (b'H', 2),
            (b'i', 4),
            (b'I', 4),
            (b'l', 8),
            (b'L', 8),
            (b'q', 8),
            (b'Q', 8),
            (b'f', 4),
            (b'd', 8),
        ] {
            assert_eq!(typecode_itemsize(tc), Some(sz));
        }
        assert_eq!(typecode_itemsize(b'x'), None);
        assert_eq!(typecode_itemsize(b'c'), None);
    }

    #[test]
    fn roundtrip_unsigned_long_into_bigint() {
        // Q value above i64::MAX must box into a W_LongObject.
        let v: u64 = u64::MAX;
        let w = unpack_value(b'Q', &v.to_ne_bytes());
        assert!(unsafe { crate::pyobject::is_long(w) });
    }
}
