//! W_ModuleObject — Python `module` type.
//!
//! PyPy equivalent: pypy/interpreter/module.py → Module
//!
//! A module holds a name (str) and a namespace dict (PyNamespace pointer).
//! The namespace stores all names defined in the module after execution.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python module object.
///
/// Layout: `[ob_type | name: *mut String | dict: *mut u8]`
///
/// `dict` is a raw pointer to a `PyNamespace` (from pyre-interpreter).
/// We store it as `*mut u8` to avoid a circular dependency on pyre-interpreter.
#[repr(C)]
pub struct W_ModuleObject {
    pub ob_header: PyObject,
    /// Heap-allocated module name string.
    pub name: *mut String,
    /// Raw pointer to the module's PyNamespace (globals after execution).
    pub dict: *mut u8,
}

/// Allocate a new W_ModuleObject.
///
/// `name` — the module name (e.g. "math", "os.path")
/// `dict_ptr` — raw pointer to the module's PyNamespace
pub fn w_module_new(name: &str, dict_ptr: *mut u8) -> PyObjectRef {
    let obj = Box::new(W_ModuleObject {
        ob_header: PyObject {
            ob_type: &MODULE_TYPE as *const PyType,
            w_class: get_instantiate(&MODULE_TYPE),
        },
        name: Box::into_raw(Box::new(name.to_string())),
        dict: dict_ptr,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Get the module name.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleObject`.
pub unsafe fn w_module_get_name(obj: PyObjectRef) -> &'static str {
    let module = &*(obj as *const W_ModuleObject);
    &*module.name
}

/// Get the module's namespace pointer (as *mut u8).
///
/// # Safety
/// `obj` must point to a valid `W_ModuleObject`.
pub unsafe fn w_module_get_dict_ptr(obj: PyObjectRef) -> *mut u8 {
    let module = &*(obj as *const W_ModuleObject);
    module.dict
}

/// Check if an object is a module.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_module(obj: PyObjectRef) -> bool {
    py_type_check(obj, &MODULE_TYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_create_and_check() {
        let obj = w_module_new("test_mod", std::ptr::null_mut());
        unsafe {
            assert!(is_module(obj));
            assert!(!is_int(obj));
            assert_eq!(w_module_get_name(obj), "test_mod");
            assert!(w_module_get_dict_ptr(obj).is_null());
        }
    }
}
