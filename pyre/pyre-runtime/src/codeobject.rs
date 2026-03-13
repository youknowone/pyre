//! W_CodeObject — Python `code` object wrapper.
//!
//! Wraps an opaque pointer to the compiler's CodeObject, allowing it to
//! be placed on the value stack as a PyObjectRef during `LoadConst`.
//! MakeFunction then extracts this pointer to build a function object.

use pyre_object::pyobject::*;

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
