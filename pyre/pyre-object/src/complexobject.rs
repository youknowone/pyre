//! W_ComplexObject — Python `complex` type backed by two f64s.

use crate::pyobject::*;

/// Python complex object.
///
/// Layout: `[ob_header: PyObject { ob_type, w_class } | real: f64 | imag: f64]`
/// Mirrors `Objects/complexobject.c`'s `Py_complex cval { double real; double imag }`.
#[repr(C)]
pub struct W_ComplexObject {
    pub ob_header: PyObject,
    pub real: f64,
    pub imag: f64,
    /// Native-subclass mapdict owner. Exact complex values keep this null.
    pub w_dict: PyObjectRef,
    /// Native-subclass `__slots__` storage indexed by `Member.index`.
    pub w_slots: PyObjectRef,
}

/// Field offset of `real` within `W_ComplexObject`.
pub const COMPLEX_REAL_OFFSET: usize = std::mem::offset_of!(W_ComplexObject, real);

/// Field offset of `imag` within `W_ComplexObject`.
pub const COMPLEX_IMAG_OFFSET: usize = std::mem::offset_of!(W_ComplexObject, imag);
pub const COMPLEX_W_DICT_OFFSET: usize = std::mem::offset_of!(W_ComplexObject, w_dict);
pub const COMPLEX_W_SLOTS_OFFSET: usize = std::mem::offset_of!(W_ComplexObject, w_slots);

/// GC type id assigned to `W_ComplexObject` at JitDriver init time.
/// Like `W_FLOAT_GC_TYPE_ID`, held as a constant so the allocation hook
/// can reach it without a back-channel.
pub const W_COMPLEX_GC_TYPE_ID: u32 = 54;

/// Fixed payload size for `W_ComplexObject`.
pub const W_COMPLEX_OBJECT_SIZE: usize = std::mem::size_of::<W_ComplexObject>();

impl crate::lltype::GcType for W_ComplexObject {
    fn type_id() -> u32 {
        W_COMPLEX_GC_TYPE_ID
    }
    const SIZE: usize = W_COMPLEX_OBJECT_SIZE;
}

/// Allocate a new W_ComplexObject on the heap.
///
/// Routes through [`crate::lltype::malloc_typed`], the typed unified
/// allocation lowering, mirroring `complexobject.c complex_subtype_from_doubles`
/// / `PyComplex_FromCComplex`.
pub fn w_complex_new(real: f64, imag: f64) -> PyObjectRef {
    crate::lltype::malloc_typed(W_ComplexObject {
        ob_header: PyObject {
            ob_type: &COMPLEX_TYPE as *const PyType,
            w_class: get_instantiate(&COMPLEX_TYPE),
        },
        real,
        imag,
        w_dict: PY_NULL,
        w_slots: PY_NULL,
    }) as PyObjectRef
}

/// Allocate a `W_ComplexObject` for a `complex` subclass instance, on the
/// managed heap so it can be reclaimed. See [`crate::intobject::w_int_subclass_new`]
/// for why the shared constructor cannot be used.
pub fn w_complex_subclass_new(real: f64, imag: f64) -> PyObjectRef {
    let obj = W_ComplexObject {
        ob_header: PyObject {
            ob_type: &COMPLEX_TYPE as *const PyType,
            w_class: get_instantiate(&COMPLEX_TYPE),
        },
        real,
        imag,
    };
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(W_COMPLEX_GC_TYPE_ID, W_COMPLEX_OBJECT_SIZE);
    if raw.is_null() {
        crate::lltype::malloc_typed(obj) as PyObjectRef
    } else {
        unsafe {
            std::ptr::write(raw as *mut W_ComplexObject, obj);
            raw as PyObjectRef
        }
    }
}

/// Extract the real component from a known W_ComplexObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_ComplexObject`.
#[inline]
pub unsafe fn w_complex_get_real(obj: PyObjectRef) -> f64 {
    unsafe { (*(obj as *const W_ComplexObject)).real }
}

/// Extract the imaginary component from a known W_ComplexObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_ComplexObject`.
#[inline]
pub unsafe fn w_complex_get_imag(obj: PyObjectRef) -> f64 {
    unsafe { (*(obj as *const W_ComplexObject)).imag }
}

#[inline]
pub unsafe fn w_complex_getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ComplexObject)).w_dict }
}

#[inline]
pub unsafe fn w_complex_setdict(obj: PyObjectRef, w_dict: PyObjectRef) {
    unsafe { (*(obj as *mut W_ComplexObject)).w_dict = w_dict };
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

pub unsafe fn w_complex_slot_get(obj: PyObjectRef, index: usize) -> Option<PyObjectRef> {
    let slots = unsafe { (*(obj as *const W_ComplexObject)).w_slots };
    if slots.is_null() {
        return None;
    }
    unsafe { crate::listobject::w_list_getitem(slots, index as i64) }
        .filter(|value| !value.is_null())
}

pub unsafe fn w_complex_slot_set(obj: PyObjectRef, index: usize, value: PyObjectRef) {
    let mut slots = unsafe { (*(obj as *const W_ComplexObject)).w_slots };
    if slots.is_null() {
        slots = crate::listobject::w_list_new(vec![PY_NULL; index + 1]);
        unsafe { (*(obj as *mut W_ComplexObject)).w_slots = slots };
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    } else {
        while unsafe { crate::listobject::w_list_len(slots) } <= index {
            unsafe { crate::listobject::w_list_append(slots, PY_NULL) };
        }
    }
    unsafe { crate::listobject::w_list_setitem(slots, index as i64, value) };
}

pub unsafe fn w_complex_slot_del(obj: PyObjectRef, index: usize) -> bool {
    let slots = unsafe { (*(obj as *const W_ComplexObject)).w_slots };
    if slots.is_null() || unsafe { crate::listobject::w_list_len(slots) } <= index {
        return false;
    }
    let present = unsafe { crate::listobject::w_list_getitem(slots, index as i64) }
        .is_some_and(|value| !value.is_null());
    if present {
        unsafe { crate::listobject::w_list_setitem(slots, index as i64, PY_NULL) };
    }
    present
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_create_and_read() {
        let obj = w_complex_new(3.0, 4.0);
        unsafe {
            assert!(is_complex(obj));
            assert!(!is_float(obj));
            assert_eq!(w_complex_get_real(obj), 3.0);
            assert_eq!(w_complex_get_imag(obj), 4.0);
        }
    }

    #[test]
    fn test_complex_field_offsets() {
        // after PyObject { ob_type(8) + w_class(8) }
        assert_eq!(COMPLEX_REAL_OFFSET, 16);
        assert_eq!(COMPLEX_IMAG_OFFSET, 24);
    }
}
