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
}

/// Field offset of `real` within `W_ComplexObject`.
pub const COMPLEX_REAL_OFFSET: usize = std::mem::offset_of!(W_ComplexObject, real);

/// Field offset of `imag` within `W_ComplexObject`.
pub const COMPLEX_IMAG_OFFSET: usize = std::mem::offset_of!(W_ComplexObject, imag);

/// GC type id assigned to `W_ComplexObject` at JitDriver init time.
/// Like `W_FLOAT_GC_TYPE_ID`, held as a constant so the allocation hook
/// can reach it without a back-channel.  Complex carries no managed
/// pointers, so its trace is a leaf (same shape as float).
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
    }) as PyObjectRef
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
