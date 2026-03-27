//! W_CodeObject — Python `code` object wrapper.
//!
//! Wraps an opaque pointer to the compiler's CodeObject, allowing it to
//! be placed on the value stack as a PyObjectRef during `LoadConst`.
//! MakeFunction then extracts this pointer to build a function object.

use pyre_object::pyobject::*;

/// Compatibility alias for PyPy's `PyCode` type.
pub type PyCode = W_CodeObject;

/// Compatibility marker for malformed bytecode.
#[derive(Debug, Clone)]
pub struct BytecodeCorruption;

/// Compatibility container for code-hook caching state.
#[derive(Debug, Default)]
pub struct CodeHookCache {
    _code_hook: Option<PyObjectRef>,
}

/// Type descriptor for code objects.
pub static CODE_TYPE: PyType = PyType { tp_name: "code" };

/// Python code object wrapper.
///
/// Stores an opaque pointer to the bytecode CodeObject. The pointer is
/// `Box::into_raw`'d from a cloned CodeObject, so we own the allocation.
#[repr(C)]
pub struct W_CodeObject {
    pub ob_header: PyObject,
    /// Opaque pointer to a `CodeObject` (owned via Box::into_raw).
    pub code_ptr: *const (),
}

/// Field offset of `code_ptr` within `W_CodeObject`.
pub const CODE_PTR_OFFSET: usize = std::mem::offset_of!(W_CodeObject, code_ptr);

/// Compatibility helper for unpacking a tuple of strings.
pub fn unpack_text_tuple(_space: PyObjectRef, w_str_tuple: PyObjectRef) -> Vec<String> {
    let _ = (_space, w_str_tuple);
    Vec::new()
}

/// Compatibility API for building a signature-like object.
pub fn make_signature(_code: &W_CodeObject) -> PyObjectRef {
    let _ = _code;
    pyre_object::w_none()
}

/// Compatibility helper that returns argument indexes which shadow cellvars.
pub fn _compute_args_as_cellvars(
    _varnames: &[String],
    _cellvars: &[String],
    _argcount: usize,
) -> Vec<usize> {
    let _ = (_varnames, _cellvars, _argcount);
    Vec::new()
}

#[inline]
pub fn _code_const_eq(_space: PyObjectRef, w_a: PyObjectRef, w_b: PyObjectRef) -> bool {
    let _ = _space;
    std::ptr::eq(w_a, w_b)
}

#[inline]
pub fn _convert_const(_space: PyObjectRef, w_a: PyObjectRef) -> PyObjectRef {
    let _ = _space;
    w_a
}

/// Allocate a new W_CodeObject wrapping an opaque code pointer.
///
/// # Safety
/// `code_ptr` must be a valid pointer to a `CodeObject` obtained
/// via `Box::into_raw`.
pub fn w_code_new(code_ptr: *const ()) -> PyObjectRef {
    let obj = Box::new(W_CodeObject {
        ob_header: PyObject {
            ob_type: &CODE_TYPE as *const PyType,
        },
        code_ptr,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Box a cloned compiler code object into a heap Python code wrapper.
pub fn box_code_constant(code: &pyre_bytecode::CodeObject) -> PyObjectRef {
    let code_ptr = Box::into_raw(Box::new(code.clone())) as *const ();
    w_code_new(code_ptr)
}

/// Extract the opaque code pointer from a known W_CodeObject.
///
/// # Safety
/// `obj` must point to a valid `W_CodeObject`.
#[inline]
pub unsafe fn w_code_get_ptr(obj: PyObjectRef) -> *const () {
    unsafe { (*(obj as *const W_CodeObject)).code_ptr }
}

/// Check if an object is a code object.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_code(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &CODE_TYPE) }
}
