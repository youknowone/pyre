//! W_NoneObject — Python `None` singleton.

use crate::pyobject::*;

/// Python None object (singleton).
#[repr(C)]
pub struct W_NoneObject {
    pub ob_header: PyObject,
}

/// GC type id assigned by the foreign-PyType registration sequence.
pub const W_NONE_GC_TYPE_ID: u32 = 45;

impl crate::lltype::GcType for W_NoneObject {
    fn type_id() -> u32 {
        W_NONE_GC_TYPE_ID
    }
    const SIZE: usize = std::mem::size_of::<W_NoneObject>();
}

/// Process-global translated-prebuilt None object. PyPy's `space.w_None` is
/// one prebuilt GC object, including its GC header; storing a bare Rust static
/// in `__DATA_CONST` loses that header and makes direct meta-traced GC marking
/// impossible. `OnceLock<usize>` is the established process-global immortal
/// owner used by the other pyre prebuilt objects.
static NONE_SINGLETON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Get the None singleton as a PyObjectRef.
pub fn w_none() -> PyObjectRef {
    *NONE_SINGLETON.get_or_init(|| {
        crate::lltype::malloc_typed_immortal(W_NoneObject {
            ob_header: PyObject {
                ob_type: &NONE_TYPE as *const PyType,
                w_class: std::ptr::null_mut(),
            },
        }) as usize
    }) as PyObjectRef
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_none_is_singleton() {
        let a = w_none();
        let b = w_none();
        assert_eq!(a, b);
        unsafe {
            assert!(is_none(a));
            assert!(!is_int(a));
            let hdr = majit_gc::header::header_of(a as usize);
            assert_eq!((*hdr).type_id(), W_NONE_GC_TYPE_ID);
            assert!((*hdr).has_flag(majit_gc::flags::NO_HEAP_PTRS));
        }
    }
}
