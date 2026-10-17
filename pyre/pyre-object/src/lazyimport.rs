//! PEP 810 `lazy_import` payload (`lazyimportobject.c` `PyLazyImportObject`).

use crate::pyobject::*;
use pyre_macros::pyre_class;

#[pyre_class("lazy_import", static_name = "LAZY_IMPORT")]
pub struct W_LazyImportObject {
    /// Builtins mapping that owns `__import__` for a root; null on a projection.
    pub lz_builtins: PyObjectRef,
    /// Root: absolute module name. Projection: source placeholder.
    pub lz_from: PyObjectRef,
    /// Root: original fromlist. Projection: attribute name (str).
    pub lz_attr: PyObjectRef,
    pub lz_code: PyObjectRef,
    pub lz_instr_offset: i32,
    pub lz_active: i32,
}

pub fn w_lazy_import_new(
    lz_builtins: PyObjectRef,
    lz_from: PyObjectRef,
    lz_attr: PyObjectRef,
    lz_code: PyObjectRef,
    lz_instr_offset: i32,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::pin_roots(&[lz_builtins, lz_from, lz_attr, lz_code]);
    W_LazyImportObject::allocate(W_LazyImportObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        lz_builtins: crate::gc_roots::shadow_stack_get(save_point),
        lz_from: crate::gc_roots::shadow_stack_get(save_point + 1),
        lz_attr: crate::gc_roots::shadow_stack_get(save_point + 2),
        lz_code: crate::gc_roots::shadow_stack_get(save_point + 3),
        lz_instr_offset,
        lz_active: 0,
    })
}

/// # Safety
/// `obj` must be a live Python object pointer.
pub unsafe fn is_lazy_import(obj: PyObjectRef) -> bool {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return false;
    }
    unsafe { !obj.is_null() && std::ptr::eq((*obj).ob_type, &LAZY_IMPORT_TYPE) }
}

/// # Safety
/// `obj` must be a `lazy_import` instance.
pub unsafe fn w_lazy_import_cast(obj: PyObjectRef) -> *mut W_LazyImportObject {
    obj as *mut W_LazyImportObject
}
