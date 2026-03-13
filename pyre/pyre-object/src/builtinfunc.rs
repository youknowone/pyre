//! Built-in function objects.
//!
//! A `W_BuiltinFunction` wraps a Rust function pointer that implements
//! a Python builtin like `print`, `len`, etc.

use crate::pyobject::*;

/// Type descriptor for built-in functions.
pub static BUILTIN_FUNC_TYPE: PyType = PyType {
    tp_name: "builtin_function_or_method",
};

/// Signature of a built-in function: takes a slice of positional arguments
/// and returns a PyObjectRef (or could panic for now on errors).
pub type BuiltinFn = fn(&[PyObjectRef]) -> PyObjectRef;

/// A built-in function object.
#[repr(C)]
pub struct W_BuiltinFunction {
    pub ob: PyObject,
    pub name: &'static str,
    pub func: BuiltinFn,
}

/// Allocate a new `W_BuiltinFunction`.
pub fn w_builtin_func_new(name: &'static str, func: BuiltinFn) -> PyObjectRef {
    let obj = Box::new(W_BuiltinFunction {
        ob: PyObject {
            ob_type: &BUILTIN_FUNC_TYPE,
        },
        name,
        func,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Check if an object is a built-in function.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_builtin_func(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BUILTIN_FUNC_TYPE) }
}

/// Get the function pointer from a built-in function object.
///
/// # Safety
/// `obj` must point to a valid `W_BuiltinFunction`.
#[inline]
pub unsafe fn w_builtin_func_get(obj: PyObjectRef) -> BuiltinFn {
    let func_obj = obj as *const W_BuiltinFunction;
    unsafe { (*func_obj).func }
}

/// Get the name of a built-in function.
///
/// # Safety
/// `obj` must point to a valid `W_BuiltinFunction`.
#[inline]
pub unsafe fn w_builtin_func_name(obj: PyObjectRef) -> &'static str {
    let func_obj = obj as *const W_BuiltinFunction;
    unsafe { (*func_obj).name }
}
