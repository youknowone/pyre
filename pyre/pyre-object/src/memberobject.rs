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

/// Create a new Member descriptor.
pub fn w_member_new(index: u32, name: String, w_cls: PyObjectRef) -> PyObjectRef {
    let obj = Box::new(W_MemberDescr {
        ob_header: PyObject {
            ob_type: &MEMBER_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        index,
        name: Box::into_raw(Box::new(name)),
        w_cls,
    });
    Box::into_raw(obj) as PyObjectRef
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
