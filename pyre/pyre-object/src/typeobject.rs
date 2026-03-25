//! W_TypeObject — Python `type` object for user-defined classes.
//!
//! PyPy equivalent: pypy/objspace/std/typeobject.py → W_TypeObject
//!
//! A type object holds the class name, tuple of base types, and a namespace
//! dict containing class-level attributes and methods.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python type object (user-defined class).
///
/// Layout: `[ob_type | name | bases | dict]`
///
/// - `name`: heap-allocated class name string
/// - `bases`: tuple of base type objects (PyObjectRef to tuple)
/// - `dict`: raw pointer to PyNamespace (class methods/attrs)
///
/// PyPy equivalent fields: W_TypeObject.name, bases_w, dict_w
#[repr(C)]
pub struct W_TypeObject {
    pub ob_header: PyObject,
    /// Class name (heap-allocated, leaked).
    pub name: *mut String,
    /// Tuple of base type objects (PyObjectRef → W_TupleObject or PY_NULL).
    pub bases: PyObjectRef,
    /// Raw pointer to the class namespace (PyNamespace from pyre-runtime).
    pub dict: *mut u8,
    /// Cached C3 MRO — PyPy: W_TypeObject.mro_w.
    /// Computed once at type creation and cached.
    pub mro_w: *mut Vec<PyObjectRef>,
}

/// Allocate a new W_TypeObject.
///
/// PyPy equivalent: W_TypeObject.__init__(space, name, bases_w, dict_w)
pub fn w_type_new(name: &str, bases: PyObjectRef, dict_ptr: *mut u8) -> PyObjectRef {
    let obj = Box::new(W_TypeObject {
        ob_header: PyObject {
            ob_type: &TYPE_TYPE as *const PyType,
        },
        mro_w: std::ptr::null_mut(), // set after construction via set_mro
        name: Box::into_raw(Box::new(name.to_string())),
        bases,
        dict: dict_ptr,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Get the class name.
pub unsafe fn w_type_get_name(obj: PyObjectRef) -> &'static str {
    &*(*(obj as *const W_TypeObject)).name
}

/// Get the bases tuple.
pub unsafe fn w_type_get_bases(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_TypeObject)).bases
}

/// Get the class namespace pointer (as *mut u8).
pub unsafe fn w_type_get_dict_ptr(obj: PyObjectRef) -> *mut u8 {
    (*(obj as *const W_TypeObject)).dict
}

/// Get the cached MRO, or null if not yet set.
pub unsafe fn w_type_get_mro(obj: PyObjectRef) -> *mut Vec<PyObjectRef> {
    (*(obj as *const W_TypeObject)).mro_w
}

/// Set the cached MRO.
pub unsafe fn w_type_set_mro(obj: PyObjectRef, mro: Vec<PyObjectRef>) {
    (*(obj as *mut W_TypeObject)).mro_w = Box::into_raw(Box::new(mro));
}

/// Check if an object is a type (user-defined class).
#[inline]
pub unsafe fn is_type(obj: PyObjectRef) -> bool {
    py_type_check(obj, &TYPE_TYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_create_and_check() {
        let obj = w_type_new("Foo", PY_NULL, std::ptr::null_mut());
        unsafe {
            assert!(is_type(obj));
            assert!(!is_int(obj));
            assert_eq!(w_type_get_name(obj), "Foo");
            assert!(w_type_get_dict_ptr(obj).is_null());
        }
    }
}
