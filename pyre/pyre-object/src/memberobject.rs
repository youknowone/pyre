//! typedef.py:443-500 Member — slot descriptor for __slots__.
//!
//! A Member descriptor provides attribute access to a specific __slots__
//! entry. In PyPy, slots are stored at fixed offsets in the object struct;
//! in pyre, instance attributes are stored in a dict, so the Member acts
//! as a marker and accessor by name.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// typedef.py:443-456 Member(index, name, w_cls).
#[repr(C)]
pub struct W_MemberDescr {
    pub ob_header: PyObject,
    /// Slot index (base_nslots + position in newslotnames).
    pub index: u32,
    /// Slot name (owned, leaked).
    pub name: *const String,
    /// Owning type object (for typecheck).
    pub w_cls: PyObjectRef,
}

pub static MEMBER_TYPE: PyType = crate::pyobject::new_pytype("member_descriptor");

/// Field offset of `w_cls` within `W_MemberDescr`.
pub const MEMBER_W_CLS_OFFSET: usize = std::mem::offset_of!(W_MemberDescr, w_cls);

/// GC type id assigned to `W_MemberDescr` at JitDriver init time.
pub const W_MEMBER_GC_TYPE_ID: u32 = 26;

/// Fixed payload size (`framework.py:811`).
pub const W_MEMBER_OBJECT_SIZE: usize = std::mem::size_of::<W_MemberDescr>();

/// Byte offsets of the inline `PyObjectRef` fields the GC must trace.
/// Only `w_cls` is a PyObjectRef — `name` is a `*const String` allocated
/// by `lltype::malloc_raw` and not a managed PyObject reference.
pub const W_MEMBER_GC_PTR_OFFSETS: [usize; 1] = [MEMBER_W_CLS_OFFSET];

impl crate::lltype::GcType for W_MemberDescr {
    const TYPE_ID: u32 = W_MEMBER_GC_TYPE_ID;
    const SIZE: usize = W_MEMBER_OBJECT_SIZE;
}

/// Create a new Member descriptor.
pub fn w_member_new(index: u32, name: String, w_cls: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_cls);

    let name = crate::lltype::malloc_raw(name);
    crate::lltype::malloc_typed(W_MemberDescr {
        ob_header: PyObject {
            ob_type: &MEMBER_TYPE as *const PyType,
            w_class: get_instantiate(&MEMBER_TYPE),
        },
        index,
        name,
        w_cls,
    }) as PyObjectRef
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
            <W_MemberDescr as crate::lltype::GcType>::TYPE_ID,
            W_MEMBER_GC_TYPE_ID
        );
        assert_eq!(
            <W_MemberDescr as crate::lltype::GcType>::SIZE,
            W_MEMBER_OBJECT_SIZE
        );
    }
}
