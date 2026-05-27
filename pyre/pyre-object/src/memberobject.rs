//! typedef.py:443-500 Member — slot descriptor for __slots__.
//!
//! A Member descriptor provides attribute access to a specific __slots__
//! entry. In PyPy, slots are stored at fixed offsets in the object struct;
//! in pyre, instance attributes are stored in a dict, so the Member acts
//! as a marker and accessor by name.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// typedef.py:443-456 Member(index, name, w_cls).  The macro skips the
/// non-PyObjectRef `index` (u32) and `name` (`*const String`) fields
/// when emitting GC pointer offsets — only `w_cls` is traced.
#[pyre_class("member_descriptor", type_id = 26, static_name = "MEMBER")]
pub struct W_MemberDescr {
    /// Slot index (base_nslots + position in newslotnames).
    pub index: u32,
    /// Slot name (owned, leaked).
    pub name: *const String,
    /// Owning type object (for typecheck).
    pub w_cls: PyObjectRef,
}

/// Create a new Member descriptor.
pub fn w_member_new(index: u32, name: String, w_cls: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_cls);
    let name = crate::lltype::malloc_raw(name);
    W_MemberDescr::allocate(W_MemberDescr {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        index,
        name,
        w_cls,
    })
}

/// Check if an object is a Member descriptor.
#[inline]
pub unsafe fn is_member(obj: PyObjectRef) -> bool {
    py_type_check(obj, &MEMBER_TYPE)
}

/// Get the Member's slot name.
pub unsafe fn w_member_get_name(obj: PyObjectRef) -> &'static str {
    &*(*(obj as *const W_MemberDescr)).name
}

/// Get the Member's owning class.
pub unsafe fn w_member_get_cls(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_MemberDescr)).w_cls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_member_gc_type_id_matches_descr() {
        assert_eq!(W_MEMBER_GC_TYPE_ID, 26);
        assert_eq!(
            <W_MemberDescr as crate::lltype::GcType>::type_id(),
            W_MEMBER_GC_TYPE_ID
        );
        assert_eq!(
            <W_MemberDescr as crate::lltype::GcType>::SIZE,
            W_MEMBER_OBJECT_SIZE
        );
    }
}
