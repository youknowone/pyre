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
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
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
    /// `PyMemberDef.doc` for native member descriptors.  Like CPython's
    /// static C string, this is non-GC metadata; Python `__slots__` members
    /// leave it null.
    pub doc: *const String,
}

/// Python 3.14's function type exposes five direct `PyMemberDef` entries.
/// PyPy represents the same values with GetSetProperty, while ordinary PyPy
/// `Member` objects use `index` for `__slots__`.  Reserve the high bit so the
/// existing slot-index shape stays intact and the interpreter can distinguish
/// the 3.14 direct members without a side table.
pub const MEMBER_DIRECT_FLAG: u32 = 1 << 31;
pub const MEMBER_FUNCTION_CLOSURE: u32 = MEMBER_DIRECT_FLAG;
pub const MEMBER_FUNCTION_DOC: u32 = MEMBER_DIRECT_FLAG | 1;
pub const MEMBER_FUNCTION_GLOBALS: u32 = MEMBER_DIRECT_FLAG | 2;
pub const MEMBER_FUNCTION_MODULE: u32 = MEMBER_DIRECT_FLAG | 3;
pub const MEMBER_FUNCTION_BUILTINS: u32 = MEMBER_DIRECT_FLAG | 4;
/// CPython 3.14 `module_members`: the authoritative Module.w_dict field.
pub const MEMBER_MODULE_DICT: u32 = MEMBER_DIRECT_FLAG | 5;
/// CPython 3.14 `complex_members`: `Py_T_DOUBLE`, `Py_READONLY`.
pub const MEMBER_COMPLEX_REAL: u32 = MEMBER_DIRECT_FLAG | 6;
pub const MEMBER_COMPLEX_IMAG: u32 = MEMBER_DIRECT_FLAG | 7;
/// `descrobject.c descr_members`, shared by every descriptor type: the owning
/// class (`PyDescrObject.d_type`) and the attribute name (`d_name`), both
/// read-only.  PyPy publishes the same two values as GetSetProperty
/// (`typedef.py:470-472`, `:538-539`); the descriptor kind is the 3.14
/// difference.  The descriptor payloads here — GetSetProperty, Member and the
/// Function carrier — do not share a header, so the reader dispatches on the
/// receiver instead of reading one fixed offset.
pub const MEMBER_DESCR_OBJCLASS: u32 = MEMBER_DIRECT_FLAG | 8;
pub const MEMBER_DESCR_NAME: u32 = MEMBER_DIRECT_FLAG | 9;
/// CPython 3.14 `BaseExceptionGroup_members`: the immutable constructor
/// message and tuple of nested exceptions.
pub const MEMBER_EXCEPTION_GROUP_MESSAGE: u32 = MEMBER_DIRECT_FLAG | 10;
pub const MEMBER_EXCEPTION_GROUP_EXCEPTIONS: u32 = MEMBER_DIRECT_FLAG | 11;
/// CPython 3.14 `SyntaxError_members._metadata`, a writable private slot.
pub const MEMBER_SYNTAX_ERROR_METADATA: u32 = MEMBER_DIRECT_FLAG | 12;
/// CPython 3.14 `StopIteration.value`, backed by PyPy's `w_value` field.
pub const MEMBER_STOP_ITERATION_VALUE: u32 = MEMBER_DIRECT_FLAG | 13;
/// CPython 3.14 `BaseException.__suppress_context__` boolean member.
pub const MEMBER_EXCEPTION_SUPPRESS_CONTEXT: u32 = MEMBER_DIRECT_FLAG | 14;
/// CPython 3.14 `AttributeError_members.name`, backed by PyPy's `w_name`.
pub const MEMBER_ATTRIBUTE_ERROR_NAME: u32 = MEMBER_DIRECT_FLAG | 15;
/// CPython 3.14 `AttributeError_members.obj`, backed by PyPy's `w_obj`.
pub const MEMBER_ATTRIBUTE_ERROR_OBJ: u32 = MEMBER_DIRECT_FLAG | 16;
/// CPython 3.14 `NameError_members.name`, backed by PyPy's `w_name`.
pub const MEMBER_NAME_ERROR_NAME: u32 = MEMBER_DIRECT_FLAG | 17;
/// CPython 3.14 `ImportError_members.msg`, backed by PyPy's `w_msg`.
pub const MEMBER_IMPORT_ERROR_MSG: u32 = MEMBER_DIRECT_FLAG | 18;
/// CPython 3.14 `ImportError_members.name`, backed by PyPy's `w_name`.
pub const MEMBER_IMPORT_ERROR_NAME: u32 = MEMBER_DIRECT_FLAG | 19;
/// CPython 3.14 `ImportError_members.name_from`, a pyre-flattened PyPy field.
pub const MEMBER_IMPORT_ERROR_NAME_FROM: u32 = MEMBER_DIRECT_FLAG | 20;
/// CPython 3.14 `ImportError_members.path`, backed by PyPy's `w_path`.
pub const MEMBER_IMPORT_ERROR_PATH: u32 = MEMBER_DIRECT_FLAG | 21;
/// CPython 3.14 `OSError_members.errno`, backed by PyPy's `w_errno`.
pub const MEMBER_OS_ERROR_ERRNO: u32 = MEMBER_DIRECT_FLAG | 22;
/// CPython 3.14 `OSError_members.strerror`, backed by PyPy's `w_strerror`.
pub const MEMBER_OS_ERROR_STRERROR: u32 = MEMBER_DIRECT_FLAG | 23;
/// CPython 3.14 `OSError_members.filename`, backed by PyPy's `w_filename`.
pub const MEMBER_OS_ERROR_FILENAME: u32 = MEMBER_DIRECT_FLAG | 24;
/// CPython 3.14 `OSError_members.filename2`, backed by PyPy's `w_filename2`.
pub const MEMBER_OS_ERROR_FILENAME2: u32 = MEMBER_DIRECT_FLAG | 25;
/// CPython 3.14 `SystemExit_members.code`, backed by PyPy's `w_code`.
pub const MEMBER_SYSTEM_EXIT_CODE: u32 = MEMBER_DIRECT_FLAG | 26;
/// CPython 3.14 `SyntaxError_members.msg`, backed by PyPy's `w_msg`.
pub const MEMBER_SYNTAX_ERROR_MSG: u32 = MEMBER_DIRECT_FLAG | 27;
/// CPython 3.14 `SyntaxError_members.filename`, backed by PyPy's `w_filename`.
pub const MEMBER_SYNTAX_ERROR_FILENAME: u32 = MEMBER_DIRECT_FLAG | 28;
/// CPython 3.14 `SyntaxError_members.lineno`, backed by PyPy's `w_lineno`.
pub const MEMBER_SYNTAX_ERROR_LINENO: u32 = MEMBER_DIRECT_FLAG | 29;
/// CPython 3.14 `SyntaxError_members.offset`, backed by PyPy's `w_offset`.
pub const MEMBER_SYNTAX_ERROR_OFFSET: u32 = MEMBER_DIRECT_FLAG | 30;
/// CPython 3.14 `SyntaxError_members.text`, backed by PyPy's `w_text`.
pub const MEMBER_SYNTAX_ERROR_TEXT: u32 = MEMBER_DIRECT_FLAG | 31;
/// CPython 3.14 `SyntaxError_members.end_lineno`, backed by PyPy's field.
pub const MEMBER_SYNTAX_ERROR_END_LINENO: u32 = MEMBER_DIRECT_FLAG | 32;
/// CPython 3.14 `SyntaxError_members.end_offset`, backed by PyPy's field.
pub const MEMBER_SYNTAX_ERROR_END_OFFSET: u32 = MEMBER_DIRECT_FLAG | 33;
/// CPython 3.14 `SyntaxError_members.print_file_and_line`, backed by PyPy's field.
pub const MEMBER_SYNTAX_ERROR_PRINT_FILE_AND_LINE: u32 = MEMBER_DIRECT_FLAG | 34;
/// CPython 3.14 Unicode*Error `encoding`, backed by PyPy's `w_encoding`.
pub const MEMBER_UNICODE_ERROR_ENCODING: u32 = MEMBER_DIRECT_FLAG | 35;
/// CPython 3.14 Unicode*Error `object`, backed by PyPy's `w_object`.
pub const MEMBER_UNICODE_ERROR_OBJECT: u32 = MEMBER_DIRECT_FLAG | 36;
/// CPython 3.14 Unicode*Error `start`, backed by PyPy's `w_start`.
pub const MEMBER_UNICODE_ERROR_START: u32 = MEMBER_DIRECT_FLAG | 37;
/// CPython 3.14 Unicode*Error `end`, backed by PyPy's `w_end`.
pub const MEMBER_UNICODE_ERROR_END: u32 = MEMBER_DIRECT_FLAG | 38;
/// CPython 3.14 Unicode*Error `reason`, backed by PyPy's `w_reason`.
pub const MEMBER_UNICODE_ERROR_REASON: u32 = MEMBER_DIRECT_FLAG | 39;
/// CPython 3.14 `staticmethod_members`: both `__func__` and `__wrapped__`
/// expose PyPy's `StaticMethod.w_function` field read-only.
pub const MEMBER_STATICMETHOD_FUNCTION: u32 = MEMBER_DIRECT_FLAG | 40;
/// CPython 3.14 `classmethod_members`: both `__func__` and `__wrapped__`
/// expose PyPy's `ClassMethod.w_function` field read-only.
pub const MEMBER_CLASSMETHOD_FUNCTION: u32 = MEMBER_DIRECT_FLAG | 41;
/// CPython 3.14 `property_members.fget`, backed by PyPy's `w_fget` field.
pub const MEMBER_PROPERTY_FGET: u32 = MEMBER_DIRECT_FLAG | 42;
/// CPython 3.14 `property_members.fset`, backed by PyPy's `w_fset` field.
pub const MEMBER_PROPERTY_FSET: u32 = MEMBER_DIRECT_FLAG | 43;
/// CPython 3.14 `property_members.fdel`, backed by PyPy's `w_fdel` field.
pub const MEMBER_PROPERTY_FDEL: u32 = MEMBER_DIRECT_FLAG | 44;
/// CPython 3.14 `property_members.__doc__`, the writable `w_doc` field.
pub const MEMBER_PROPERTY_DOC: u32 = MEMBER_DIRECT_FLAG | 45;
/// CPython 3.14 `range_members.start`, backed by PyPy's `w_start` field.
pub const MEMBER_RANGE_START: u32 = MEMBER_DIRECT_FLAG | 46;
/// CPython 3.14 `range_members.stop`, backed by PyPy's `w_stop` field.
pub const MEMBER_RANGE_STOP: u32 = MEMBER_DIRECT_FLAG | 47;
/// CPython 3.14 `range_members.step`, backed by PyPy's `w_step` field.
pub const MEMBER_RANGE_STEP: u32 = MEMBER_DIRECT_FLAG | 48;
/// CPython 3.14 `slice_members.start`, backed by PyPy's `w_start` field.
pub const MEMBER_SLICE_START: u32 = MEMBER_DIRECT_FLAG | 49;
/// CPython 3.14 `slice_members.stop`, backed by PyPy's `w_stop` field.
pub const MEMBER_SLICE_STOP: u32 = MEMBER_DIRECT_FLAG | 50;
/// CPython 3.14 `slice_members.step`, backed by PyPy's `w_step` field.
pub const MEMBER_SLICE_STEP: u32 = MEMBER_DIRECT_FLAG | 51;
/// CPython 3.14 `super_members.__thisclass__`, backed by PyPy's `w_starttype`.
pub const MEMBER_SUPER_THISCLASS: u32 = MEMBER_DIRECT_FLAG | 52;
/// CPython 3.14 `super_members.__self__`, backed by PyPy's `w_self`.
pub const MEMBER_SUPER_SELF: u32 = MEMBER_DIRECT_FLAG | 53;
/// CPython 3.14 `super_members.__self_class__`, backed by PyPy's `w_objtype`.
pub const MEMBER_SUPER_SELF_CLASS: u32 = MEMBER_DIRECT_FLAG | 54;
/// CPython 3.14 `OSError_members.winerror`, backed by PyPy's `w_winerror`.
/// Declared on every platform, registered on `OSError` only where the
/// platform has Windows error codes (`interp_exceptions.py:723-728`).
pub const MEMBER_OS_ERROR_WINERROR: u32 = MEMBER_DIRECT_FLAG | 61;

/// Create a new Member descriptor.
pub fn w_member_new(index: u32, name: String, w_cls: PyObjectRef) -> PyObjectRef {
    w_member_new_with_doc(index, name, None, w_cls)
}

/// Create a Member descriptor with CPython `PyMemberDef.doc` metadata.
pub fn w_member_new_with_doc(
    index: u32,
    name: String,
    doc: Option<String>,
    w_cls: PyObjectRef,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(w_cls);
    let name = crate::lltype::malloc_raw(name);
    let doc = doc.map_or(std::ptr::null(), |doc| {
        crate::lltype::malloc_raw(doc) as *const String
    });
    let w_cls = crate::gc_roots::shadow_stack_get(root_base);
    // Managed (`allocate_stable`), not the movable `malloc_typed` immortal a
    // bare `allocate` would give: a `__slots__` member outlives the statement
    // that created it (`d = C.x` keeps the descriptor after `C` is dropped),
    // and `w_cls` is then its only reference to the owning type.  An immortal
    // is outside the collector's sweep set, so the marker takes GCFLAG_VISITED
    // on it during the first major and never clears it again; from the second
    // major on it is skipped, `w_cls` is never re-marked, and the type is swept
    // while the descriptor still points at it.  A stable managed allocation
    // keeps the address fixed for the raw `*mut W_MemberDescr` accessors below
    // while putting the object under the ordinary mark-and-clear cycle.
    W_MemberDescr::allocate_stable(W_MemberDescr {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        index,
        name,
        doc,
        w_cls,
    })
}

/// Create one of Python 3.14's direct function member descriptors.
pub fn w_member_new_direct(kind: u32, name: String, w_cls: PyObjectRef) -> PyObjectRef {
    debug_assert_ne!(kind & MEMBER_DIRECT_FLAG, 0);
    w_member_new(kind, name, w_cls)
}

/// Create a Python 3.14 direct member descriptor with `PyMemberDef.doc`.
pub fn w_member_new_direct_with_doc(
    kind: u32,
    name: String,
    doc: String,
    w_cls: PyObjectRef,
) -> PyObjectRef {
    debug_assert_ne!(kind & MEMBER_DIRECT_FLAG, 0);
    w_member_new_with_doc(kind, name, Some(doc), w_cls)
}

/// Check if an object is a Member descriptor.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_member(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &MEMBER_TYPE) }
}

/// Get the Member's slot name.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_get_name(obj: PyObjectRef) -> &'static str {
    unsafe { &*(*(obj as *const W_MemberDescr)).name }
}

/// Get the optional CPython `PyMemberDef.doc`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_get_doc(obj: PyObjectRef) -> Option<&'static str> {
    let doc = unsafe { (*(obj as *const W_MemberDescr)).doc };
    if doc.is_null() {
        None
    } else {
        Some(unsafe { &*doc })
    }
}

/// Get the Member's owning class.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_get_cls(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_MemberDescr)).w_cls }
}

/// Fill a descriptor owner after the built-in type registry is published.
///
/// Two remembered-set notifications, because the descriptor can be either
/// shape: the prebuilt-family bit covers a descriptor that predates the
/// managed-allocation hooks and so fell back to `malloc_typed`, and the write
/// barrier covers the ordinary `allocate_stable` case, where an old descriptor
/// gaining a young or unmarked `w_cls` must re-enter the collector's worklist.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_set_cls(obj: PyObjectRef, w_cls: PyObjectRef) {
    crate::gc_roots::mark_prebuilt_roots_dirty();
    unsafe { (*(obj as *mut W_MemberDescr)).w_cls = w_cls };
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// `typedef.py:446 Member.index` — the slot index (`base_nslots + position`),
/// used by the LOAD_ATTR/STORE_ATTR cache to form the `SLOTS_STARTING_FROM +
/// index` attrkind (mapdict.py:1520).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_get_index(obj: PyObjectRef) -> u32 {
    unsafe { (*(obj as *const W_MemberDescr)).index }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_is_direct(obj: PyObjectRef) -> bool {
    unsafe { w_member_get_index(obj) & MEMBER_DIRECT_FLAG != 0 }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_get_direct_kind(obj: PyObjectRef) -> u32 {
    let kind = unsafe { w_member_get_index(obj) };
    debug_assert_ne!(kind & MEMBER_DIRECT_FLAG, 0);
    kind
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
