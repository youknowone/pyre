//! W_Super — Python `super` proxy object.
//!
//! PyPy equivalent: pypy/module/__builtin__/descriptor.py W_Super
//!
//! Stores (super_type, obj) and resolves attribute lookups
//! starting from the next class after super_type in obj's MRO.

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// super proxy: [ob_type | super_type (cls) | obj (self)]
#[pyre_class("super", type_id = 18, static_name = "SUPER")]
pub struct W_Super {
    /// The class passed to super() — lookup starts after this in MRO.
    pub super_type: PyObjectRef,
    /// The instance (self) or class for classmethod.
    pub obj: PyObjectRef,
}

/// Create a new super proxy.
pub fn w_super_new(super_type: PyObjectRef, obj: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`): pin the
    // `super_type`/`obj` pair across the GC malloc and re-read their
    // relocated addresses afterwards (a minor collection inside the malloc
    // may move them). A super proxy whose members are reachable only
    // through it must be GC-traced; a `malloc_typed` proxy is invisible to
    // mark-sweep, whereas `register_pyre_class` registers this layout's
    // `ptr_offsets`, so mark-sweep follows the members. The write barrier
    // below keeps the old-gen proxy in the remembered set so young members
    // survive a later minor collection.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(super_type);
    crate::gc_roots::pin_root(obj);

    let header = PyObject {
        ob_type: &SUPER_TYPE as *const PyType,
        w_class: get_instantiate(&SUPER_TYPE),
    };
    let raw = crate::gc_hook::try_gc_alloc_stable(W_SUPER_GC_TYPE_ID, W_SUPER_OBJECT_SIZE)
        .filter(|p| !p.is_null());
    let super_type = crate::gc_roots::shadow_stack_get(save_point);
    let obj = crate::gc_roots::shadow_stack_get(save_point + 1);
    if let Some(raw) = raw {
        unsafe {
            std::ptr::write(
                raw as *mut W_Super,
                W_Super {
                    ob: header,
                    super_type,
                    obj,
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    W_Super::allocate(W_Super {
        ob: header,
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
    unsafe { (*(obj as *const W_Super)).super_type }
}

/// Get the bound object (self) from a super proxy.
#[inline]
pub unsafe fn w_super_get_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Super)).obj }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_super_gc_type_id_matches_descr() {
        assert_eq!(W_SUPER_GC_TYPE_ID, 18);
        assert_eq!(
            <W_Super as crate::lltype::GcType>::type_id(),
            W_SUPER_GC_TYPE_ID
        );
        assert_eq!(
            <W_Super as crate::lltype::GcType>::SIZE,
            W_SUPER_OBJECT_SIZE
        );
    }
}
