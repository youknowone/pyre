//! Function object.
//!
//! Wraps a code object pointer, a function name, a pointer to the
//! defining module's globals namespace, and an optional closure tuple.
//! When called, the interpreter creates a new PyFrame that *shares*
//! the globals pointer (no clone).

use crate::executioncontext::PyNamespace;
use pyre_object::pyobject::*;

/// Type descriptor for user-defined functions.
pub static FUNCTION_TYPE: PyType = PyType {
    tp_name: "function",
};

/// User-defined function object.
///
/// Layout: `[ob_type | code | name_ptr | w_func_globals | closure]`
/// - `code`: opaque pointer to the CodeObject
/// - `name_ptr`: leaked `Box<String>` containing the function name
/// - `w_func_globals`: raw pointer to the module-level namespace (shared)
/// - `closure`:  tuple of cell objects, or PY_NULL if no closure
#[repr(C)]
pub struct Function {
    pub ob: PyObject,
    /// Opaque pointer to the CodeObject (borrowed from constants).
    pub code: *const (),
    /// Function name (leaked Box<String>).
    pub name: *const String,
    /// PyPy: W_Function.w_func_globals
    pub w_func_globals: *mut PyNamespace,
    /// Closure: tuple of cell objects from the enclosing scope,
    /// or PY_NULL if this function has no free variables.
    pub closure: PyObjectRef,
    /// Default argument values.
    /// PyPy: W_Function.defs_w
    pub defs_w: PyObjectRef,
    /// Keyword-only default values.
    /// PyPy: W_Function.w_kw_defs
    pub w_kw_defs: PyObjectRef,
}

/// Field offset of `code` within `Function`, for JIT field access.
pub const FUNCTION_CODE_OFFSET: usize = std::mem::offset_of!(Function, code);
/// Field offset of `name` within `Function`.
pub const FUNCTION_NAME_OFFSET: usize = std::mem::offset_of!(Function, name);
/// Field offset of `w_func_globals` within `Function`.
pub const FUNCTION_GLOBALS_OFFSET: usize = std::mem::offset_of!(Function, w_func_globals);
/// Field offset of `closure` within `Function`.
pub const FUNCTION_CLOSURE_OFFSET: usize = std::mem::offset_of!(Function, closure);

/// Allocate a new `Function`.
///
/// `code` is an opaque pointer to the CodeObject.
/// `name` is the function name string (leaked).
/// `w_func_globals` is the defining module's namespace pointer (shared).
pub fn function_new(
    code: *const (),
    name: String,
    w_func_globals: *mut PyNamespace,
) -> PyObjectRef {
    function_new_with_closure(code, name, w_func_globals, PY_NULL)
}

/// Allocate a new `Function` with a closure.
///
/// `closure` is a tuple of cell objects, or PY_NULL if no closure.
pub fn function_new_with_closure(
    code: *const (),
    name: String,
    w_func_globals: *mut PyNamespace,
    closure: PyObjectRef,
) -> PyObjectRef {
    let name_ptr = Box::into_raw(Box::new(name)) as *const String;
    let obj = Box::new(Function {
        ob: PyObject {
            ob_type: &FUNCTION_TYPE as *const PyType,
        },
        code,
        name: name_ptr,
        w_func_globals,
        closure,
        defs_w: PY_NULL,
        w_kw_defs: PY_NULL,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// PyPy `function._get_immutable_code`.
#[inline]
pub unsafe fn _get_immutable_code(func: PyObjectRef) -> *const () {
    unsafe { function_get_code(func) }
}

/// Check if an object is a user-defined function.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_function(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FUNCTION_TYPE) }
}

/// Get the opaque code pointer from a function object.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_code(obj: PyObjectRef) -> *const () {
    unsafe { (*(obj as *const Function)).code }
}

/// Get the function name.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_name(obj: PyObjectRef) -> &'static str {
    unsafe { &*(*(obj as *const Function)).name }
}

/// Get the globals namespace pointer from a function object.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_globals(obj: PyObjectRef) -> *mut PyNamespace {
    unsafe { (*(obj as *const Function)).w_func_globals }
}

/// Get the closure tuple from a function object.
/// Returns PY_NULL if the function has no closure.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_closure(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).closure }
}

/// Set the closure on a function object.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_set_closure(obj: PyObjectRef, closure: PyObjectRef) {
    unsafe { (*(obj as *mut Function)).closure = closure }
}

/// Get defaults tuple.
#[inline]
pub unsafe fn function_get_defaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).defs_w }
}

/// Set defaults tuple.
#[inline]
pub unsafe fn function_set_defaults(obj: PyObjectRef, defaults: PyObjectRef) {
    unsafe { (*(obj as *mut Function)).defs_w = defaults }
}

/// Get kwdefaults dict.
#[inline]
pub unsafe fn function_get_kwdefaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).w_kw_defs }
}

/// Set kwdefaults dict.
#[inline]
pub unsafe fn function_set_kwdefaults(obj: PyObjectRef, kwdefaults: PyObjectRef) {
    unsafe { (*(obj as *mut Function)).w_kw_defs = kwdefaults }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_create() {
        let code = 0xDEAD_BEEF as *const ();
        let mut ns = PyNamespace::new();
        let obj = function_new(code, "myfunc".to_string(), &mut ns);
        unsafe {
            assert!(is_function(obj));
            assert!(!is_int(obj));
            assert_eq!(function_get_code(obj), code);
            assert_eq!(function_get_name(obj), "myfunc");
            assert_eq!(function_get_globals(obj), &mut ns as *mut PyNamespace);
            assert!(function_get_closure(obj).is_null());
        }
    }

    #[test]
    fn test_function_field_offsets() {
        assert_eq!(FUNCTION_CODE_OFFSET, 8); // after ob_type pointer
        assert_eq!(FUNCTION_NAME_OFFSET, 16); // after code
        assert_eq!(FUNCTION_GLOBALS_OFFSET, 24); // after name
        assert_eq!(FUNCTION_CLOSURE_OFFSET, 32); // after w_func_globals
    }
}
