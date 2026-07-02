//! `pypy/interpreter/typedef.py` descriptor payload parity port.
//!
//! PyPy stores `fget` / `fset` / `fdel` / `doc` / `reqcls` /
//! `use_closure` / `name` as instance fields on the GetSetProperty
//! object itself — `class GetSetProperty(W_Root): _immutable_fields_
//! = [...]` (typedef.py:312-326).  Pyre previously emulated this with
//! a process-global `RwLock<HashMap<usize, GetSetFields>>` keyed by
//! descriptor pointer; that side table was a pure adaptation with no
//! RPython justification (and quietly leaked entries when descriptors
//! were collected).
//!
//! This module replaces the side table with a real W_Root struct
//! whose layout mirrors PyPy's instance shape line-for-line — readers
//! reach the slots via `&*(obj as *const GetSetProperty)`, the GC
//! traces every `PyObjectRef`-shaped field, and there is no global
//! state to fall out of sync with the descriptor's actual lifetime.

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// `pypy/interpreter/typedef.py:312-346 class GetSetProperty(W_Root)`.
///
/// All `PyObjectRef`-shaped slots default to `PY_NULL` to mark
/// "absent" (PyPy uses `None`); `use_closure` is a `bool` mirroring
/// the eponymous PyPy field.
///
/// `pytype_static = "GETSET_DESCRIPTOR_TYPE"` keeps the PyType under
/// its existing public name (`typedef.py:444 GetSetProperty.typedef =
/// TypeDef("getset_descriptor", ...)`) while the GC consts stay on
/// the `W_GETSET_PROPERTY_*` convention.
#[pyre_class(
    "getset_descriptor",
    type_id = 40,
    static_name = "GETSET_PROPERTY",
    pytype_static = "GETSET_DESCRIPTOR_TYPE"
)]
pub struct GetSetProperty {
    /// `typedef.py:339 self.fget` — getter callable.
    pub fget: PyObjectRef,
    /// `typedef.py:340 self.fset` — setter callable.
    pub fset: PyObjectRef,
    /// `typedef.py:341 self.fdel` — deleter callable.
    pub fdel: PyObjectRef,
    /// `typedef.py:342 self.doc` — wrapped docstring.
    pub doc: PyObjectRef,
    /// `typedef.py:343 self.reqcls` — required receiver class for
    /// `descr_self_interp_w` mismatch checking.
    pub reqcls: PyObjectRef,
    /// `typedef.py:346 self.name` — descriptor name (defaults to
    /// `'<generic property>'` when the caller passes None).
    pub name: PyObjectRef,
    /// `typedef.py:320 w_objclass = None` class default + per-instance
    /// override stamped by `copy_for_type` (typedef.py:353).  Read by
    /// `descr_get_objclass` (typedef.py:414-418) before falling back
    /// to `space.gettypeobject(self.reqcls.typedef)`.
    pub w_objclass: PyObjectRef,
    /// `typedef.py:344 self.w_qualname = None` — lazy cache for
    /// `descr_get_qualname` (typedef.py:420-433); first reader stamps
    /// `"<class>.<name>"` (or `"?.<name>"` when `reqcls is None`).
    pub w_qualname: PyObjectRef,
    /// `typedef.py:345 self.use_closure` — passes `(self, space, obj)`
    /// vs `(space, obj)` to the wrapped callbacks.
    pub use_closure: bool,
}

/// Allocate a `GetSetProperty` bound to `GETSET_DESCRIPTOR_TYPE`.
/// Mirrors `typedef.py:327-336 _init` — every slot is set in one shot
/// so the descriptor is fully initialised before the first reader.
///
/// `name` may be `PY_NULL`, in which case the caller is responsible
/// for substituting `'<generic property>'` (matching `typedef.py:336
/// self.name = name if name is not None else '<generic property>'`);
/// pyre's call sites pass an already-resolved name to keep the
/// allocation hot path branchless.
pub fn w_getset_property_new(
    fget: PyObjectRef,
    fset: PyObjectRef,
    fdel: PyObjectRef,
    doc: PyObjectRef,
    reqcls: PyObjectRef,
    use_closure: bool,
    name: PyObjectRef,
) -> PyObjectRef {
    GetSetProperty::allocate(GetSetProperty {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        fget,
        fset,
        fdel,
        doc,
        reqcls,
        name,
        w_objclass: PY_NULL,
        w_qualname: PY_NULL,
        use_closure,
    })
}

/// Test whether `obj` is a `GetSetProperty`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_getset_property(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &GETSET_DESCRIPTOR_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_fget(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).fget }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_fset(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).fset }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_fdel(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).fdel }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_reqcls(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).reqcls }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_name(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).name }
}

/// `typedef.py:58 add_entries` parity — overwrite the descriptor's
/// `name` slot with the dict-key it was registered under.  Used by
/// the post-init namespace walker so descriptors built without an
/// explicit name (most `make_getset_descriptor` callers) carry the
/// matching `__name__` instead of the `<generic property>` sentinel.
///
/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_set_name(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GetSetProperty)).name = value }
}

/// `typedef.py:343 self.reqcls = cls` — write the required-receiver
/// class slot.  Used by `patch_builtin_function_descriptors` to
/// install the BuiltinFunction class onto the shared
/// `__self__`/`__doc__` GetSetProperty descriptors after the
/// W_TypeObject for BuiltinFunction is materialised.
#[inline]
pub unsafe fn w_getset_set_reqcls(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GetSetProperty)).reqcls = value }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_doc(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).doc }
}

/// `typedef.py:320 / 348-356 copy_for_type` writes `new.w_objclass`.
/// Pyre keeps the slot directly on the struct so the descriptor's
/// `descr_get_objclass` reads it without any side-table.
///
/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_objclass(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).w_objclass }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_set_objclass(obj: PyObjectRef, value: PyObjectRef) {
    // Immortal-descriptor slot reached only by `walk_raw_getset_roots`,
    // skipped on clean minor collections; record the store.
    crate::gc_roots::mark_prebuilt_roots_dirty();
    unsafe { (*(obj as *mut GetSetProperty)).w_objclass = value }
}

/// `typedef.py:344 self.w_qualname = None` — lazy cache slot for
/// `descr_get_qualname` (typedef.py:420-433).
///
/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_qualname(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).w_qualname }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_set_qualname(obj: PyObjectRef, value: PyObjectRef) {
    // Immortal-descriptor slot (see `w_getset_set_objclass`); the lazy
    // qualname cache stores a freshly allocated string.
    crate::gc_roots::mark_prebuilt_roots_dirty();
    unsafe { (*(obj as *mut GetSetProperty)).w_qualname = value }
}

/// `typedef.py:345 self.use_closure` — read-only accessor.
///
/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_use_closure(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const GetSetProperty)).use_closure }
}

/// `pypy/interpreter/typedef.py:443-500 Member` — slot descriptor
/// for `__slots__`.
///
/// A Member descriptor provides attribute access to a specific
/// `__slots__` entry. In PyPy, slots are stored at fixed offsets in
/// the object struct; in pyre, instance attributes are stored in a
/// dict, so the Member acts as a marker and accessor by name.
///
/// The macro skips the non-PyObjectRef `index` (u32) and `name`
/// (`*const String`) fields when emitting GC pointer offsets — only
/// `w_cls` is traced.
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
    unsafe { py_type_check(obj, &MEMBER_TYPE) }
}

/// Get the Member's slot name.
pub unsafe fn w_member_get_name(obj: PyObjectRef) -> &'static str {
    unsafe { &*(*(obj as *const W_MemberDescr)).name }
}

/// Get the Member's owning class.
pub unsafe fn w_member_get_cls(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_MemberDescr)).w_cls }
}

/// `typedef.py:446 Member.index` — the slot index (`base_nslots + position`),
/// used by the LOAD_ATTR/STORE_ATTR cache to form the `SLOTS_STARTING_FROM +
/// index` attrkind (mapdict.py:1520).
pub unsafe fn w_member_get_index(obj: PyObjectRef) -> u32 {
    unsafe { (*(obj as *const W_MemberDescr)).index }
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
