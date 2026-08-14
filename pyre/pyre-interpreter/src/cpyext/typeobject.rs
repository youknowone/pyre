//! C-defined types -- PyPy `cpyext/typeobject.py`.
//!
//! A `PyTypeObject` is the extension's own static storage, so it *is* the type
//! mirror: `PyType_Ready` links it to the interpreter type it builds rather
//! than allocating a block of its own.  A type pyre defines gets a synthesized
//! block of the same shape, so `Py_TYPE(x)->tp_name` reads something either
//! way.

use super::pyobject::{self, CPyObject, REFCNT_IMMORTAL};
use pyre_object::PyObjectRef;
use std::ffi::{CString, c_char, c_int, c_uint, c_void};

/// `PyVarObject` — a `PyObject` with the variable-part length.
#[repr(C)]
pub struct CPyVarObject {
    pub ob_base: CPyObject,
    pub ob_size: isize,
}

/// The protocol tables, opaque to this slice: they are pointers on
/// [`CPyTypeObject`] and the slice that reads them fills these in.
#[repr(C)]
pub struct CPyNumberMethods {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CPySequenceMethods {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CPyMappingMethods {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CPyAsyncMethods {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CPyBufferProcs {
    _private: [u8; 0],
}

/// One row of a `PyMemberDef` table.
#[repr(C)]
pub struct CPyMemberDef {
    pub name: *const c_char,
    pub type_code: c_int,
    pub offset: isize,
    pub flags: c_int,
    pub doc: *const c_char,
}

/// One row of a `PyGetSetDef` table.
#[repr(C)]
pub struct CPyGetSetDef {
    pub name: *const c_char,
    pub get: *const c_void,
    pub set: *const c_void,
    pub doc: *const c_char,
    pub closure: *mut c_void,
}

/// `PyTypeObject`, in the CPython 3.14 field order.
///
/// Every slot is a raw pointer here: the ones this slice reads are typed at
/// their use site, and the ones it does not read still have to occupy their
/// exact offsets, because the extension's static initializer writes them all.
#[repr(C)]
pub struct CPyTypeObject {
    pub ob_base: CPyVarObject,
    pub tp_name: *const c_char,
    pub tp_basicsize: isize,
    pub tp_itemsize: isize,
    pub tp_dealloc: *const c_void,
    pub tp_vectorcall_offset: isize,
    pub tp_getattr: *const c_void,
    pub tp_setattr: *const c_void,
    pub tp_as_async: *mut CPyAsyncMethods,
    pub tp_repr: *const c_void,
    pub tp_as_number: *mut CPyNumberMethods,
    pub tp_as_sequence: *mut CPySequenceMethods,
    pub tp_as_mapping: *mut CPyMappingMethods,
    pub tp_hash: *const c_void,
    pub tp_call: *const c_void,
    pub tp_str: *const c_void,
    pub tp_getattro: *const c_void,
    pub tp_setattro: *const c_void,
    pub tp_as_buffer: *mut CPyBufferProcs,
    pub tp_flags: std::ffi::c_ulong,
    pub tp_doc: *const c_char,
    pub tp_traverse: *const c_void,
    pub tp_clear: *const c_void,
    pub tp_richcompare: *const c_void,
    pub tp_weaklistoffset: isize,
    pub tp_iter: *const c_void,
    pub tp_iternext: *const c_void,
    pub tp_methods: *mut super::methodobject::CPyMethodDef,
    pub tp_members: *mut CPyMemberDef,
    pub tp_getset: *mut CPyGetSetDef,
    pub tp_base: *mut CPyTypeObject,
    pub tp_dict: *mut CPyObject,
    pub tp_descr_get: *const c_void,
    pub tp_descr_set: *const c_void,
    pub tp_dictoffset: isize,
    pub tp_init: *const c_void,
    pub tp_alloc: *const c_void,
    pub tp_new: *const c_void,
    pub tp_free: *const c_void,
    pub tp_is_gc: *const c_void,
    pub tp_bases: *mut CPyObject,
    pub tp_mro: *mut CPyObject,
    pub tp_cache: *mut CPyObject,
    pub tp_subclasses: *mut c_void,
    pub tp_weaklist: *mut CPyObject,
    pub tp_del: *const c_void,
    pub tp_version_tag: c_uint,
    pub tp_finalize: *const c_void,
    pub tp_vectorcall: *const c_void,
    pub tp_watched: std::ffi::c_uchar,
    pub tp_versions_used: u16,
}

pub const PY_TPFLAGS_DEFAULT: std::ffi::c_ulong = 0;
pub const PY_TPFLAGS_HEAPTYPE: std::ffi::c_ulong = 1 << 9;
pub const PY_TPFLAGS_BASETYPE: std::ffi::c_ulong = 1 << 10;
pub const PY_TPFLAGS_READY: std::ffi::c_ulong = 1 << 12;
pub const PY_TPFLAGS_READYING: std::ffi::c_ulong = 1 << 13;
pub const PY_TPFLAGS_HAVE_GC: std::ffi::c_ulong = 1 << 14;

const fn immortal_type() -> CPyTypeObject {
    CPyTypeObject {
        ob_base: CPyVarObject {
            ob_base: CPyObject {
                ob_refcnt: REFCNT_IMMORTAL,
                ob_pyre_link: pyre_object::PY_NULL,
                ob_type: std::ptr::null_mut(),
            },
            ob_size: 0,
        },
        tp_name: std::ptr::null(),
        tp_basicsize: 0,
        tp_itemsize: 0,
        tp_dealloc: std::ptr::null(),
        tp_vectorcall_offset: 0,
        tp_getattr: std::ptr::null(),
        tp_setattr: std::ptr::null(),
        tp_as_async: std::ptr::null_mut(),
        tp_repr: std::ptr::null(),
        tp_as_number: std::ptr::null_mut(),
        tp_as_sequence: std::ptr::null_mut(),
        tp_as_mapping: std::ptr::null_mut(),
        tp_hash: std::ptr::null(),
        tp_call: std::ptr::null(),
        tp_str: std::ptr::null(),
        tp_getattro: std::ptr::null(),
        tp_setattro: std::ptr::null(),
        tp_as_buffer: std::ptr::null_mut(),
        tp_flags: PY_TPFLAGS_DEFAULT,
        tp_doc: std::ptr::null(),
        tp_traverse: std::ptr::null(),
        tp_clear: std::ptr::null(),
        tp_richcompare: std::ptr::null(),
        tp_weaklistoffset: 0,
        tp_iter: std::ptr::null(),
        tp_iternext: std::ptr::null(),
        tp_methods: std::ptr::null_mut(),
        tp_members: std::ptr::null_mut(),
        tp_getset: std::ptr::null_mut(),
        tp_base: std::ptr::null_mut(),
        tp_dict: std::ptr::null_mut(),
        tp_descr_get: std::ptr::null(),
        tp_descr_set: std::ptr::null(),
        tp_dictoffset: 0,
        tp_init: std::ptr::null(),
        tp_alloc: std::ptr::null(),
        tp_new: std::ptr::null(),
        tp_free: std::ptr::null(),
        tp_is_gc: std::ptr::null(),
        tp_bases: std::ptr::null_mut(),
        tp_mro: std::ptr::null_mut(),
        tp_cache: std::ptr::null_mut(),
        tp_subclasses: std::ptr::null_mut(),
        tp_weaklist: std::ptr::null_mut(),
        tp_del: std::ptr::null(),
        tp_version_tag: 0,
        tp_finalize: std::ptr::null(),
        tp_vectorcall: std::ptr::null(),
        tp_watched: 0,
        tp_versions_used: 0,
    }
}

/// Sentinel type for a `PyModuleDef`, which is C static storage rather than a
/// mirror of an interpreter object: its `ob_pyre_link` stays null and it is
/// never entered in the census.
pub static mut CPY_MODULE_DEF_TYPE: CPyTypeObject = immortal_type();

/// Fill a synthesized mirror for an interpreter type.
///
/// `tp_basicsize` stays 0 on purpose: an instance of a pyre type is exactly a
/// `PyObject` mirror, and `make_ref` reads this field to size the block.
pub(super) fn describe_interpreter_type(mirror: *mut CPyTypeObject, w_type: PyObjectRef) {
    let name = unsafe { pyre_object::typeobject::w_type_get_name(w_type) };
    // Leaked because `tp_name` is a `const char *` the extension may keep; the
    // type it names is itself immortal, so the string's lifetime matches.
    let name = CString::new(name).unwrap_or_default().into_raw();
    unsafe {
        (*mirror).ob_base.ob_base.ob_refcnt = REFCNT_IMMORTAL;
        (*mirror).tp_name = name;
        (*mirror).tp_flags = PY_TPFLAGS_DEFAULT | PY_TPFLAGS_READY | PY_TPFLAGS_BASETYPE;
    }
}

/// `true` when a mirror is a `PyModuleDef` rather than a linked object.
pub(super) fn is_module_def(raw: *mut CPyObject) -> bool {
    !raw.is_null() && unsafe { std::ptr::eq((*raw).ob_type, &raw mut CPY_MODULE_DEF_TYPE) }
}

/// The interpreter type a `PyTypeObject` stands for, or null before
/// `PyType_Ready`.
pub(super) fn interpreter_type(tp: *mut CPyTypeObject) -> PyObjectRef {
    if tp.is_null() {
        return pyre_object::PY_NULL;
    }
    unsafe { pyobject::from_ref(&raw mut (*tp).ob_base.ob_base) }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(&raw const CPY_MODULE_DEF_TYPE);
}
