//! W_Super — Python `super` proxy object.
//!
//! PyPy equivalent: pypy/objspace/descroperation.py + superobject.py
//!
//! Stores (super_type, obj) and resolves attribute lookups
//! starting from the next class after super_type in obj's MRO.

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// super proxy: [ob_type | super_type (cls) | obj (self)]
#[pyre_class("super", type_id = 18, static_name = "SUPER")]
pub struct W_SuperObject {
    /// The class passed to super() — lookup starts after this in MRO.
    pub super_type: PyObjectRef,
    /// The instance (self) or class for classmethod.
    pub obj: PyObjectRef,
}

/// Create a new super proxy.
pub fn w_super_new(super_type: PyObjectRef, obj: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(super_type);
    crate::gc_roots::pin_root(obj);
    W_SuperObject::allocate(W_SuperObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        super_type,
        obj,
    })
}

#[inline]
pub unsafe fn is_super(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SUPER_TYPE) }
}

/// Get the super_type (cls) from a super proxy.
#[inline]
pub unsafe fn w_super_get_type(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SuperObject)).super_type }
}

/// Get the bound object (self) from a super proxy.
#[inline]
pub unsafe fn w_super_get_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_SuperObject)).obj }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_super_gc_type_id_matches_descr() {
        assert_eq!(W_SUPER_GC_TYPE_ID, 18);
        assert_eq!(
            <W_SuperObject as crate::lltype::GcType>::type_id(),
            W_SUPER_GC_TYPE_ID
        );
        assert_eq!(
            <W_SuperObject as crate::lltype::GcType>::SIZE,
            W_SUPER_OBJECT_SIZE
        );
    }
}
