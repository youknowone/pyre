//! W_FunctionObject — Python user-defined function object.
//!
//! Wraps a code object pointer and a function name. When called, the
//! interpreter creates a new PyFrame from the code object and executes it.

use pyre_object::pyobject::*;

/// Type descriptor for user-defined functions.
pub static FUNCTION_TYPE: PyType = PyType {
    tp_name: "function",
};

/// User-defined function object.
///
/// Layout: `[ob_type | code_ptr | name_ptr]`
/// - `code_ptr`: opaque pointer to the CodeObject (owned by W_CodeObject,
///    the function borrows it)
/// - `name_ptr`: leaked `Box<String>` containing the function name
#[repr(C)]
pub struct W_FunctionObject {
    pub ob: PyObject,
    /// Opaque pointer to the CodeObject (borrowed from constants).
    pub code_ptr: *const (),
    /// Function name (leaked Box<String>).
    pub name: *const String,
}

/// Field offset of `code_ptr` within `W_FunctionObject`, for JIT field access.
pub const FUNC_CODE_OFFSET: usize = std::mem::offset_of!(W_FunctionObject, code_ptr);
/// Field offset of `name` within `W_FunctionObject`.
pub const FUNC_NAME_OFFSET: usize = std::mem::offset_of!(W_FunctionObject, name);

/// Allocate a new `W_FunctionObject`.
///
/// `code_ptr` is an opaque pointer to the CodeObject.
/// `name` is the function name string (leaked for Phase 1).
pub fn w_func_new(code_ptr: *const (), name: String) -> PyObjectRef {
    let name_ptr = Box::into_raw(Box::new(name)) as *const String;
    let obj = Box::new(W_FunctionObject {
        ob: PyObject {
            ob_type: &FUNCTION_TYPE as *const PyType,
        },
        code_ptr,
        name: name_ptr,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Check if an object is a user-defined function.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_func(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FUNCTION_TYPE) }
}

/// Get the opaque code pointer from a function object.
///
/// # Safety
/// `obj` must point to a valid `W_FunctionObject`.
#[inline]
pub unsafe fn w_func_get_code_ptr(obj: PyObjectRef) -> *const () {
    unsafe { (*(obj as *const W_FunctionObject)).code_ptr }
}

/// Get the function name.
///
/// # Safety
/// `obj` must point to a valid `W_FunctionObject`.
#[inline]
pub unsafe fn w_func_get_name(obj: PyObjectRef) -> &'static str {
    unsafe { &*(*(obj as *const W_FunctionObject)).name }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_func_create() {
        let code_ptr = 0xDEAD_BEEF as *const ();
        let obj = w_func_new(code_ptr, "myfunc".to_string());
        unsafe {
            assert!(is_func(obj));
            assert!(!is_int(obj));
            assert_eq!(w_func_get_code_ptr(obj), code_ptr);
            assert_eq!(w_func_get_name(obj), "myfunc");
        }
    }

    #[test]
    fn test_func_field_offsets() {
        assert_eq!(FUNC_CODE_OFFSET, 8); // after ob_type pointer
        assert_eq!(FUNC_NAME_OFFSET, 16); // after code_ptr
    }
}
