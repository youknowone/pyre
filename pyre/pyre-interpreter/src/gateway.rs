//! Built-in function objects.
//!
//! A `BuiltinCode` wraps a Rust function pointer that implements
//! a Python builtin like `print`, `len`, etc.

use pyre_object::pyobject::*;

/// Type descriptor for built-in functions.
pub static BUILTIN_CODE_TYPE: PyType = PyType {
    tp_name: "builtin_function_or_method",
};

/// Signature of a built-in function.
///
/// PyPy: all interp-level functions can raise OperationError.
/// pyre equivalent: returns Result so errors propagate through the call stack.
pub type BuiltinCodeFn = fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>;

/// A built-in function object.
#[repr(C)]
pub struct BuiltinCode {
    pub ob: PyObject,
    pub name: &'static str,
    pub func: BuiltinCodeFn,
}

/// Allocate a new `BuiltinCode`.
pub fn builtin_code_new(name: &'static str, func: BuiltinCodeFn) -> PyObjectRef {
    let obj = Box::new(BuiltinCode {
        ob: PyObject {
            ob_type: &BUILTIN_CODE_TYPE,
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
pub unsafe fn is_builtin_code(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BUILTIN_CODE_TYPE) }
}

/// Get the function pointer from a built-in function object.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
#[inline]
pub unsafe fn builtin_code_get(obj: PyObjectRef) -> BuiltinCodeFn {
    let func_obj = obj as *const BuiltinCode;
    unsafe { (*func_obj).func }
}

/// Get the name of a built-in function.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
#[inline]
pub unsafe fn builtin_code_name(obj: PyObjectRef) -> &'static str {
    let func_obj = obj as *const BuiltinCode;
    unsafe { (*func_obj).name }
}
