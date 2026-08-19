//! C-defined types -- PyPy `cpyext/typeobject.py`.
//!
//! A `PyTypeObject` is the extension's own static storage, so it *is* the type
//! mirror: `PyType_Ready` links it to the interpreter type it builds rather
//! than allocating a block of its own.  A type pyre defines gets a synthesized
//! block of the same shape, so `Py_TYPE(x)->tp_name` reads something either
//! way.

use super::pyobject::{self, CPyObject, REFCNT_FROM_PYRE, REFCNT_IMMORTAL};
use pyre_object::PyObjectRef;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

/// `PyVarObject` — a `PyObject` with the variable-part length.
#[repr(C)]
pub struct CPyVarObject {
    pub ob_base: CPyObject,
    pub ob_size: isize,
}

/// `PyNumberMethods`.  Every slot is a raw pointer for the reason
/// [`CPyTypeObject`] gives.
#[repr(C)]
pub struct CPyNumberMethods {
    pub nb_add: *const c_void,
    pub nb_subtract: *const c_void,
    pub nb_multiply: *const c_void,
    pub nb_remainder: *const c_void,
    pub nb_divmod: *const c_void,
    pub nb_power: *const c_void,
    pub nb_negative: *const c_void,
    pub nb_positive: *const c_void,
    pub nb_absolute: *const c_void,
    pub nb_bool: *const c_void,
    pub nb_invert: *const c_void,
    pub nb_lshift: *const c_void,
    pub nb_rshift: *const c_void,
    pub nb_and: *const c_void,
    pub nb_xor: *const c_void,
    pub nb_or: *const c_void,
    pub nb_int: *const c_void,
    pub nb_reserved: *const c_void,
    pub nb_float: *const c_void,
    pub nb_inplace_add: *const c_void,
    pub nb_inplace_subtract: *const c_void,
    pub nb_inplace_multiply: *const c_void,
    pub nb_inplace_remainder: *const c_void,
    pub nb_inplace_power: *const c_void,
    pub nb_inplace_lshift: *const c_void,
    pub nb_inplace_rshift: *const c_void,
    pub nb_inplace_and: *const c_void,
    pub nb_inplace_xor: *const c_void,
    pub nb_inplace_or: *const c_void,
    pub nb_floor_divide: *const c_void,
    pub nb_true_divide: *const c_void,
    pub nb_inplace_floor_divide: *const c_void,
    pub nb_inplace_true_divide: *const c_void,
    pub nb_index: *const c_void,
    pub nb_matrix_multiply: *const c_void,
    pub nb_inplace_matrix_multiply: *const c_void,
}

/// `PySequenceMethods`.  The two `was_sq_*` holes are the pre-2.x slice slots,
/// which still occupy their offsets.
#[repr(C)]
pub struct CPySequenceMethods {
    pub sq_length: *const c_void,
    pub sq_concat: *const c_void,
    pub sq_repeat: *const c_void,
    pub sq_item: *const c_void,
    pub was_sq_slice: *const c_void,
    pub sq_ass_item: *const c_void,
    pub was_sq_ass_slice: *const c_void,
    pub sq_contains: *const c_void,
    pub sq_inplace_concat: *const c_void,
    pub sq_inplace_repeat: *const c_void,
}

/// `PyMappingMethods`.
#[repr(C)]
pub struct CPyMappingMethods {
    pub mp_length: *const c_void,
    pub mp_subscript: *const c_void,
    pub mp_ass_subscript: *const c_void,
}

/// `PyAsyncMethods`, declared for its offsets: nothing reads these yet.
#[repr(C)]
pub struct CPyAsyncMethods {
    pub am_await: *const c_void,
    pub am_aiter: *const c_void,
    pub am_anext: *const c_void,
    pub am_send: *const c_void,
}

/// `PyBufferProcs`, declared for its offsets: nothing reads these yet.
#[repr(C)]
pub struct CPyBufferProcs {
    pub bf_getbuffer: *const c_void,
    pub bf_releasebuffer: *const c_void,
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
pub const PY_TPFLAGS_IMMUTABLETYPE: std::ffi::c_ulong = 1 << 8;
pub const PY_TPFLAGS_ITEMS_AT_END: std::ffi::c_ulong = 1 << 23;

/// The fast-subclass flags, in the order `inherit_special` tests them
/// (`typeobject.py:492-509`): the first base that matches wins, so a type is
/// only ever marked as one of these.
const FAST_SUBCLASS_FLAGS: [(&pyre_object::pyobject::PyType, std::ffi::c_ulong); 8] = [
    (&pyre_object::interp_exceptions::EXCEPTION_TYPE, 1 << 30),
    (&pyre_object::pyobject::TYPE_TYPE, 1 << 31),
    (&pyre_object::pyobject::INT_TYPE, 1 << 24),
    (&pyre_object::bytesobject::BYTES_TYPE, 1 << 27),
    (&pyre_object::pyobject::STR_TYPE, 1 << 28),
    (&pyre_object::pyobject::TUPLE_TYPE, 1 << 26),
    (&pyre_object::pyobject::LIST_TYPE, 1 << 25),
    (&pyre_object::pyobject::DICT_TYPE, 1 << 29),
];

/// Mark `tp` with the one fast-subclass flag its base chain earns it.
fn set_fast_subclass_flags(tp: *mut CPyTypeObject, w_type: PyObjectRef) {
    for (builtin, flag) in FAST_SUBCLASS_FLAGS {
        let w_builtin = crate::typedef::gettypeobject(builtin);
        if w_builtin.is_null() {
            continue;
        }
        if crate::baseobjspace::issubclass(w_type, w_builtin).unwrap_or(false) {
            unsafe { (*tp).tp_flags |= flag };
            return;
        }
    }
}

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

// ── the static type mirrors ─────────────────────────────────────────────

/// The `PyTypeObject` statics an extension names by address.
///
/// `api.py:746-790 build_exported_objects` registers the same family: C spells
/// `&PyList_Type`, so each one has to be storage whose address is fixed at link
/// time and whose body the runtime fills in place, which is what a
/// [`PyAPI_DATA`] object is and what a mirror allocated on demand can never be.
/// A name whose type this build does not have stays unbound -- its `tp_name` is
/// then NULL, which is the state `PyType_Ready` would have left it in.
macro_rules! type_mirrors {
    ($($symbol:ident => $resolve:expr,)*) => {
        $(
            /// The mirror of the like-named builtin type.
            #[unsafe(no_mangle)]
            pub static mut $symbol: CPyTypeObject = immortal_type();
        )*

        /// Bind every static type mirror to the interpreter type it stands for.
        ///
        /// Called before the first `PyInit_*`, so a type reached through
        /// `make_ref` finds the static already entered and hands out its
        /// address rather than synthesizing a second block for the same type.
        pub fn init_type_mirrors() {
            let bound: &[(*mut CPyTypeObject, PyObjectRef)] = &[
                $( (&raw mut $symbol, $resolve), )*
            ];
            for &(mirror, w_type) in bound {
                let header = unsafe { &raw mut (*mirror).ob_base.ob_base };
                if w_type.is_null() || unsafe { !(*header).ob_pyre_link.is_null() } {
                    continue;
                }
                pyobject::attach_foreign(w_type, header);
                describe_interpreter_type(mirror, w_type);
            }
            // Deferred until every link above is in the table: a metatype is
            // itself one of these statics, and resolving it earlier would
            // synthesize a block for a type that is about to be bound here.
            for &(mirror, _) in bound {
                let header = unsafe { &raw mut (*mirror).ob_base.ob_base };
                let w_type = unsafe { (*header).ob_pyre_link };
                if w_type.is_null() || unsafe { !(*header).ob_type.is_null() } {
                    continue;
                }
                let of_type = pyobject::type_mirror(w_type);
                unsafe { pyobject::set_ob_type(header, of_type) };
            }
        }

        fn ensure_type_mirrors_linked() {
            $( std::hint::black_box(&raw const $symbol); )*
        }
    };
}

type_mirrors! {
    // The object model.
    PyType_Type => crate::typedef::w_type(),
    PyBaseObject_Type => crate::typedef::w_object(),
    PySuper_Type => builtin_type(&pyre_object::descriptor::SUPER_TYPE),
    // The built-in data types.
    PyBool_Type => builtin_type(&pyre_object::pyobject::BOOL_TYPE),
    PyByteArray_Type => builtin_type(&pyre_object::bytearrayobject::BYTEARRAY_TYPE),
    PyBytes_Type => builtin_type(&pyre_object::bytesobject::BYTES_TYPE),
    PyComplex_Type => builtin_type(&pyre_object::pyobject::COMPLEX_TYPE),
    PyDict_Type => builtin_type(&pyre_object::pyobject::DICT_TYPE),
    PyEllipsis_Type => builtin_type(&pyre_object::pyobject::ELLIPSIS_TYPE),
    PyFloat_Type => builtin_type(&pyre_object::pyobject::FLOAT_TYPE),
    PyFrozenSet_Type => builtin_type(&pyre_object::setobject::FROZENSET_TYPE),
    PyList_Type => builtin_type(&pyre_object::pyobject::LIST_TYPE),
    PyLong_Type => builtin_type(&pyre_object::pyobject::INT_TYPE),
    PyMemoryView_Type => builtin_type(&pyre_object::memoryview::MEMORYVIEW_TYPE),
    PyModule_Type => builtin_type(&pyre_object::pyobject::MODULE_TYPE),
    PySet_Type => builtin_type(&pyre_object::setobject::SET_TYPE),
    PySlice_Type => builtin_type(&pyre_object::sliceobject::SLICE_TYPE),
    PyTuple_Type => builtin_type(&pyre_object::pyobject::TUPLE_TYPE),
    PyUnicode_Type => builtin_type(&pyre_object::pyobject::STR_TYPE),
    Py_GenericAliasType => builtin_type(&pyre_object::_pypy_generic_alias::GENERIC_ALIAS_TYPE),
    // The dict views.
    PyDictProxy_Type => builtin_type(&pyre_object::pyobject::MAPPING_PROXY_TYPE),
    PyDictItems_Type => builtin_type(&pyre_object::dictmultiobject::DICT_ITEMS_TYPE),
    PyDictKeys_Type => builtin_type(&pyre_object::dictmultiobject::DICT_KEYS_TYPE),
    PyDictValues_Type => builtin_type(&pyre_object::dictmultiobject::DICT_VALUES_TYPE),
    // Functions, methods and descriptors.  `PyCFunction_Type` names
    // `methodobject`'s own `builtin_function_or_method` -- the type a method
    // an extension defines carries, and so the one a type derived from it is
    // derived from.  The interpreter's `len` is the other
    // `builtin_function_or_method`, which no symbol here names, and
    // `PyCFunction_Check` answers no for it; `methodobject`'s
    // `pycfunction_type` says why that is the safe half of the gap.
    PyCFunction_Type => super::methodobject::pycfunction_type(),
    PyClassMethodDescr_Type => builtin_type(&crate::function::CLASSMETHOD_DESCRIPTOR_TYPE),
    PyClassMethod_Type => builtin_type(&pyre_object::function::CLASSMETHOD_TYPE),
    PyFunction_Type => builtin_type(&crate::function::FUNCTION_TYPE),
    PyGetSetDescr_Type => builtin_type(&pyre_object::typedef::GETSET_DESCRIPTOR_TYPE),
    PyMemberDescr_Type => builtin_type(&pyre_object::typedef::MEMBER_TYPE),
    PyMethodDescr_Type => builtin_type(&crate::function::METHOD_DESCRIPTOR_TYPE),
    PyMethod_Type => builtin_type(&pyre_object::function::METHOD_TYPE),
    PyProperty_Type => builtin_type(&pyre_object::descriptor::PROPERTY_TYPE),
    PyStaticMethod_Type => builtin_type(&pyre_object::function::STATICMETHOD_TYPE),
    PyWrapperDescr_Type => builtin_type(&crate::function::SLOT_WRAPPER_TYPE),
    // The built-ins that are types.
    PyEnum_Type => builtin_type(&pyre_object::functional::ENUMERATE_TYPE),
    PyFilter_Type => builtin_type(&pyre_object::functional::FILTER_TYPE),
    PyMap_Type => builtin_type(&pyre_object::functional::MAP_TYPE),
    PyRange_Type => builtin_type(&pyre_object::functional::RANGE_TYPE),
    PyReversed_Type => builtin_type(&pyre_object::functional::REVERSED_TYPE),
    PyZip_Type => builtin_type(&pyre_object::functional::ZIP_TYPE),
    // Frames, code and the objects a call leaves behind.
    PyAsyncGen_Type => builtin_type(&pyre_object::generator::ASYNC_GENERATOR_TYPE),
    PyCell_Type => builtin_type(&pyre_object::nestedscope::CELL_TYPE),
    PyCode_Type => builtin_type(&crate::pycode::CODE_TYPE),
    PyCoro_Type => builtin_type(&pyre_object::generator::COROUTINE_TYPE),
    PyFrame_Type => builtin_type(&crate::pyframe::FRAME_TYPE),
    PyGen_Type => builtin_type(&pyre_object::generator::GENERATOR_TYPE),
    PyTraceBack_Type => builtin_type(&crate::pytraceback::PYTRACEBACK_TYPE),
    _PyAsyncGenASend_Type => builtin_type(&pyre_object::generator::ASYNC_GEN_ASEND_TYPE),
    _PyWeakref_RefType => builtin_type(&pyre_object::weakref::WEAKREF_LAYOUT_TYPE),
}

/// The interpreter type object a layout static describes, or null when this
/// build has not built it yet.
fn builtin_type(layout: &'static pyre_object::pyobject::PyType) -> PyObjectRef {
    crate::typedef::gettypeobject(layout)
}

/// The name a synthesized mirror hands out as `tp_name`.
///
/// `tp_name` is a `const char *` an extension may keep for as long as it holds
/// the type, so the string has to outlive every read of the field and die with
/// the mirror rather than with this call.  Keyed by the mirror's address, which
/// is fixed for its life; [`forget_type_name`] is what releases it.
type NameTable = std::collections::HashMap<
    usize,
    CString,
    std::hash::BuildHasherDefault<std::hash::DefaultHasher>,
>;
static TYPE_NAMES: super::ForkMutex<NameTable> =
    super::ForkMutex::new(NameTable::with_hasher(std::hash::BuildHasherDefault::new()));

pub(super) fn forget_type_name(mirror: usize) {
    TYPE_NAMES.lock().remove(&mirror);
}

/// Fill a synthesized mirror for an interpreter type.
///
/// `tp_basicsize` stays 0 on purpose: an instance of a pyre type is exactly a
/// `PyObject` mirror, and `make_ref` reads this field to size the block.
///
/// The refcount is left as [`pyobject::attach`] set it: a synthesized mirror
/// carries the ordinary link share and is released with the type it stands for,
/// which is what keeps a class the extension merely observed collectable.
pub(super) fn describe_interpreter_type(mirror: *mut CPyTypeObject, w_type: PyObjectRef) {
    let name = unsafe { pyre_object::typeobject::w_type_get_name(w_type) };
    let name = CString::new(name).unwrap_or_default();
    // The bytes are boxed, so moving the `CString` into the table below leaves
    // this pointer valid.
    let pointer = name.as_ptr();
    let heaptype = match unsafe { pyre_object::typeobject::w_type_is_heaptype(w_type) } {
        true => PY_TPFLAGS_HEAPTYPE,
        false => 0,
    };
    unsafe {
        (*mirror).tp_name = pointer;
        (*mirror).tp_flags = PY_TPFLAGS_DEFAULT | PY_TPFLAGS_READY | PY_TPFLAGS_BASETYPE | heaptype;
    }
    TYPE_NAMES.lock().insert(mirror as usize, name);
}

/// `true` when `tp` is a type whose storage is the mirror layer's to release —
/// the predicate `type_dealloc` and `_dealloc` both branch on
/// (`typeobject.py:716`, `object.py:72`).
pub(super) fn is_heap_type(tp: *mut CPyTypeObject) -> bool {
    !tp.is_null() && unsafe { (*tp).tp_flags } & PY_TPFLAGS_HEAPTYPE != 0
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

// ── slot lookup ─────────────────────────────────────────────────────────

/// The `PyTypeObject` of `w_type` and of each of its bases, nearest first.
///
/// A pyre type contributes a synthesized mirror whose slots are all null, so
/// walking the whole MRO costs nothing and lets a Python subclass of a
/// C-defined type reach its base's slots.
fn c_bases(w_type: PyObjectRef) -> Vec<*mut CPyTypeObject> {
    let mut out = Vec::new();
    let mro = unsafe { pyre_object::w_type_get_mro(w_type) };
    if mro.is_null() {
        let raw = pyobject::as_pyobj(w_type) as *mut CPyTypeObject;
        if !raw.is_null() {
            out.push(raw);
        }
        return out;
    }
    for &w_base in unsafe { (*mro).as_slice() } {
        let raw = pyobject::as_pyobj(w_base) as *mut CPyTypeObject;
        if !raw.is_null() {
            out.push(raw);
        }
    }
    out
}

/// A dying mirror's `tp_dealloc`, read off its own `ob_type`.
///
/// `cpyext/src/object.c:75-78 _Py_Dealloc` reads `obj->ob_type` directly and
/// performs no MRO walk: by the time a mirror is being deallocated its link is
/// already the deallocating marker, so there is no interpreter type to walk.
/// `PyType_Ready`'s `inherit_slots` is what put a base's `tp_dealloc` on the
/// subtype, so the single read finds it.
///
/// # Safety
/// `raw` must be a live mirror.
pub(super) unsafe fn tp_dealloc_of(raw: *mut CPyObject) -> Option<*const c_void> {
    let tp = unsafe { (*raw).ob_type };
    if tp.is_null() {
        return None;
    }
    let slot = unsafe { (*tp).tp_dealloc };
    if slot.is_null() { None } else { Some(slot) }
}

/// The nearest non-null `pick`ed slot along `w_type`'s MRO.
fn find_slot(w_type: PyObjectRef, pick: fn(*mut CPyTypeObject) -> *const c_void) -> *const c_void {
    for raw in c_bases(w_type) {
        let slot = pick(raw);
        if !slot.is_null() {
            return slot;
        }
    }
    std::ptr::null()
}

pub(super) fn slot_of(
    w_obj: PyObjectRef,
    pick: fn(*mut CPyTypeObject) -> *const c_void,
) -> *const c_void {
    match crate::typedef::r#type(w_obj) {
        Some(w_type) => find_slot(w_type.as_ptr(), pick),
        None => std::ptr::null(),
    }
}

/// The C block that carries a C-defined instance's fields.
///
/// `tp_alloc` built it, so it is already in the identity table; the fallback
/// covers an instance some other path produced and gives it the zero-filled
/// block its type declares.
pub(super) fn instance_block(w_self: PyObjectRef) -> *mut CPyObject {
    // Borrowed, not owned: the block's owner is the link, and the caller
    // reached this through `w_self`, which keeps the instance — and therefore
    // the block — alive for as long as it holds it.
    pyobject::borrow_mirror(w_self)
}

// ── the descriptors a method / member / getset table becomes ────────────

/// Reserved carrier keys, namespaced as `methodobject`'s are.
const DEF_KEY: &str = "__pyre_def__";
const NAME_KEY: &str = "__name__";
const QUALNAME_KEY: &str = "__qualname__";
const DOC_KEY: &str = "__doc__";
const OBJCLASS_KEY: &str = "__objclass__";

fn carrier_get(carrier: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    let dict = crate::baseobjspace::getdict_native(carrier);
    if dict.is_null() {
        return None;
    }
    unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, key) }
}

fn carrier_set(carrier: PyObjectRef, key: &str, value: PyObjectRef) {
    let dict = crate::baseobjspace::getdict_native(carrier);
    if !dict.is_null() {
        unsafe { pyre_object::dictmultiobject::w_dict_setitem_str(dict, key, value) };
    }
}

fn carrier_def(carrier: PyObjectRef) -> usize {
    carrier_get(carrier, DEF_KEY)
        .filter(|&value| unsafe { pyre_object::is_int(value) })
        .map(|value| unsafe { pyre_object::w_int_get_value(value) } as usize)
        .unwrap_or(0)
}

fn text_or_none(pointer: *const c_char) -> PyObjectRef {
    if pointer.is_null() {
        return pyre_object::w_none();
    }
    pyre_object::w_str_new(&unsafe { std::ffi::CStr::from_ptr(pointer) }.to_string_lossy())
}

/// Build one descriptor carrier: the definition address plus the names every
/// descriptor answers with.
fn new_carrier(
    carrier_type: PyObjectRef,
    definition: usize,
    name: *const c_char,
    doc: *const c_char,
    w_class: PyObjectRef,
) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let class_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_class);
    let carrier = pyre_object::w_instance_new(carrier_type);
    let carrier_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(carrier);
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);

    let w_name = text_or_none(name);
    carrier_set(reload(carrier_slot), NAME_KEY, w_name);
    let w_name = carrier_get(reload(carrier_slot), NAME_KEY).unwrap_or_else(pyre_object::w_none);
    carrier_set(reload(carrier_slot), QUALNAME_KEY, w_name);
    let w_doc = text_or_none(doc);
    carrier_set(reload(carrier_slot), DOC_KEY, w_doc);
    carrier_set(reload(carrier_slot), OBJCLASS_KEY, reload(class_slot));
    let w_def = pyre_object::w_int_new(definition as i64);
    carrier_set(reload(carrier_slot), DEF_KEY, w_def);
    reload(carrier_slot)
}

fn descriptor_type(
    cell: &OnceLock<usize>,
    name: &'static str,
    init: fn(PyObjectRef),
) -> PyObjectRef {
    *cell.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type(name, init);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
}

static METHOD_DESCRIPTOR_TYPE: OnceLock<usize> = OnceLock::new();
static MEMBER_DESCRIPTOR_TYPE: OnceLock<usize> = OnceLock::new();
static GETSET_DESCRIPTOR_TYPE: OnceLock<usize> = OnceLock::new();

/// `methodobject.py:W_PyCMethodObject` — a `tp_methods` row.
///
/// Unlike the module-level carrier it is a descriptor: the receiver is the
/// instance the attribute was read through, which `__get__` binds.
fn method_descriptor_type() -> PyObjectRef {
    descriptor_type(&METHOD_DESCRIPTOR_TYPE, "method_descriptor", |ns| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__get__",
            crate::make_builtin_function("__get__", method_descr_get),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__call__",
            crate::make_builtin_function("__call__", method_descr_call),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            crate::make_builtin_function_with_arity("__repr__", method_descr_repr, 1),
        );
    })
}

fn member_descriptor_type() -> PyObjectRef {
    descriptor_type(&MEMBER_DESCRIPTOR_TYPE, "member_descriptor", |ns| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__get__",
            crate::make_builtin_function("__get__", member_descr_get),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__set__",
            crate::make_builtin_function_with_arity("__set__", member_descr_set, 3),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            crate::make_builtin_function_with_arity("__repr__", member_descr_repr, 1),
        );
    })
}

fn getset_descriptor_type() -> PyObjectRef {
    descriptor_type(&GETSET_DESCRIPTOR_TYPE, "getset_descriptor", |ns| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__get__",
            crate::make_builtin_function("__get__", getset_descr_get),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__set__",
            crate::make_builtin_function_with_arity("__set__", getset_descr_set, 3),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            crate::make_builtin_function_with_arity("__repr__", getset_descr_repr, 1),
        );
    })
}

fn descriptor_name(carrier: PyObjectRef) -> String {
    carrier_get(carrier, NAME_KEY)
        .filter(|&name| unsafe { pyre_object::unicodeobject::is_str(name) })
        .map(|name| unsafe { pyre_object::w_str_get_wtf8(name) }.to_string())
        .unwrap_or_else(|| "?".to_string())
}

/// `descrobject.c` spells each kind differently: `method`, `member` and
/// `attribute` for the getset.
fn descr_repr(args: &[PyObjectRef], kind: &str) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let owner = carrier_get(carrier, OBJCLASS_KEY)
        .filter(|&owner| !owner.is_null())
        .map(|owner| unsafe { pyre_object::typeobject::w_type_get_name(owner) }.to_string())
        .unwrap_or_else(|| "?".to_string());
    Ok(pyre_object::w_str_new(&format!(
        "<{kind} '{}' of '{owner}' objects>",
        descriptor_name(carrier)
    )))
}

fn method_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    descr_repr(args, "method")
}

fn member_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    descr_repr(args, "member")
}

fn getset_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    descr_repr(args, "attribute")
}

/// The receiver a `__get__(self, instance, owner)` call names, or `None` when
/// the attribute was read off the class.
fn bound_instance(args: &[PyObjectRef]) -> Option<PyObjectRef> {
    let instance = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    if instance.is_null() || unsafe { pyre_object::is_none(instance) } {
        return None;
    }
    Some(instance)
}

fn method_descr_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let Some(instance) = bound_instance(args) else {
        return Ok(carrier);
    };
    let method = carrier_def(carrier) as *mut super::methodobject::CPyMethodDef;
    if method.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext method descriptor lost its definition",
        ));
    }
    super::methodobject::new_pycfunction(
        method,
        instance,
        carrier_get(carrier, OBJCLASS_KEY).unwrap_or_else(pyre_object::w_none),
    )
}

/// `descr.__call__(instance, *args)`, the unbound spelling.
fn method_descr_call(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    let Some(&instance) = positional.first() else {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{}' of a cpyext type needs an argument",
            descriptor_name(carrier)
        )));
    };
    let method = carrier_def(carrier) as *mut super::methodobject::CPyMethodDef;
    if method.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext method descriptor lost its definition",
        ));
    }
    super::methodobject::call_method_def(method, instance, &positional[1..], kwargs)
}

// ── `tp_members` ────────────────────────────────────────────────────────

/// `structmember.h` type codes.
pub const T_SHORT: c_int = 0;
pub const T_INT: c_int = 1;
pub const T_LONG: c_int = 2;
pub const T_FLOAT: c_int = 3;
pub const T_DOUBLE: c_int = 4;
pub const T_STRING: c_int = 5;
pub const T_OBJECT: c_int = 6;
pub const T_CHAR: c_int = 7;
pub const T_BYTE: c_int = 8;
pub const T_UBYTE: c_int = 9;
pub const T_USHORT: c_int = 10;
pub const T_UINT: c_int = 11;
pub const T_ULONG: c_int = 12;
pub const T_BOOL: c_int = 14;
pub const T_OBJECT_EX: c_int = 16;
pub const T_LONGLONG: c_int = 17;
pub const T_ULONGLONG: c_int = 18;
pub const T_PYSSIZET: c_int = 19;
pub const T_NONE: c_int = 20;
/// `READONLY`.
pub const MEMBER_READONLY: c_int = 1;

fn member_address(w_self: PyObjectRef, member: *mut CPyMemberDef) -> *mut u8 {
    let block = instance_block(w_self);
    if block.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (block as *mut u8).offset((*member).offset) }
}

fn read_member(
    w_self: PyObjectRef,
    member: *mut CPyMemberDef,
) -> Result<PyObjectRef, crate::PyError> {
    let address = member_address(w_self, member);
    if address.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext member read on an object with no C storage",
        ));
    }
    let name = || {
        unsafe { std::ffi::CStr::from_ptr((*member).name) }
            .to_string_lossy()
            .into_owned()
    };
    let value = unsafe {
        match (*member).type_code {
            T_BOOL => pyre_object::boolobject::w_bool_from(*(address as *const i8) != 0),
            T_BYTE => pyre_object::w_int_new(*(address as *const i8) as i64),
            T_UBYTE => pyre_object::w_int_new(*(address as *const u8) as i64),
            T_SHORT => pyre_object::w_int_new(*(address as *const i16) as i64),
            T_USHORT => pyre_object::w_int_new(*(address as *const u16) as i64),
            T_INT => pyre_object::w_int_new(*(address as *const i32) as i64),
            T_UINT => pyre_object::w_int_new(*(address as *const u32) as i64),
            T_LONG | T_LONGLONG => pyre_object::w_int_new(*(address as *const i64)),
            T_ULONG | T_ULONGLONG => {
                let raw = *(address as *const u64);
                match i64::try_from(raw) {
                    Ok(value) => pyre_object::w_int_new(value),
                    Err(_) => {
                        return Err(crate::PyError::new(
                            crate::PyErrorKind::OverflowError,
                            "member value does not fit in a pyre int yet",
                        ));
                    }
                }
            }
            T_PYSSIZET => pyre_object::w_int_new(*(address as *const isize) as i64),
            T_FLOAT => pyre_object::w_float_new(*(address as *const f32) as f64),
            T_DOUBLE => pyre_object::w_float_new(*(address as *const f64)),
            T_CHAR => pyre_object::w_str_new(&(*(address as *const u8) as char).to_string()),
            T_STRING => {
                let text = *(address as *const *const c_char);
                text_or_none(text)
            }
            T_NONE => pyre_object::w_none(),
            T_OBJECT | T_OBJECT_EX => {
                let stored = *(address as *const *mut CPyObject);
                if stored.is_null() {
                    if (*member).type_code == T_OBJECT_EX {
                        return Err(crate::PyError::attribute_error(name()));
                    }
                    pyre_object::w_none()
                } else {
                    pyobject::from_ref(stored)
                }
            }
            other => {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::SystemError,
                    format!("cpyext member '{}' uses unsupported type {other}", name()),
                ));
            }
        }
    };
    Ok(value)
}

fn write_member(
    w_self: PyObjectRef,
    member: *mut CPyMemberDef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    let name = || {
        unsafe { std::ffi::CStr::from_ptr((*member).name) }
            .to_string_lossy()
            .into_owned()
    };
    if unsafe { (*member).flags } & MEMBER_READONLY != 0
        || matches!(unsafe { (*member).type_code }, T_STRING | T_NONE)
    {
        return Err(crate::PyError::attribute_error(format!(
            "attribute '{}' of a cpyext object is not writable",
            name()
        )));
    }
    let type_code = unsafe { (*member).type_code };
    // Every branch below stores into the block, so the integer or float is
    // unwrapped first — the conversion can call back into Python and collect.
    let integer = matches!(
        type_code,
        T_BOOL
            | T_BYTE
            | T_UBYTE
            | T_SHORT
            | T_USHORT
            | T_INT
            | T_UINT
            | T_LONG
            | T_ULONG
            | T_LONGLONG
            | T_ULONGLONG
            | T_PYSSIZET
    );
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_self);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(value);
    let number = if integer {
        crate::baseobjspace::gateway_int_w(pyre_object::gc_roots::shadow_stack_get(value_slot))?
    } else {
        0
    };
    let real = if matches!(type_code, T_FLOAT | T_DOUBLE) {
        crate::baseobjspace::float_w(pyre_object::gc_roots::shadow_stack_get(value_slot))?
    } else {
        0.0
    };
    let w_self = pyre_object::gc_roots::shadow_stack_get(self_slot);
    let value = pyre_object::gc_roots::shadow_stack_get(value_slot);
    let address = member_address(w_self, member);
    if address.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext member write on an object with no C storage",
        ));
    }
    unsafe {
        match type_code {
            T_BOOL => *(address as *mut i8) = (number != 0) as i8,
            T_BYTE => *(address as *mut i8) = number as i8,
            T_UBYTE => *(address as *mut u8) = number as u8,
            T_SHORT => *(address as *mut i16) = number as i16,
            T_USHORT => *(address as *mut u16) = number as u16,
            T_INT => *(address as *mut i32) = number as i32,
            T_UINT => *(address as *mut u32) = number as u32,
            T_LONG | T_LONGLONG => *(address as *mut i64) = number,
            T_ULONG | T_ULONGLONG => *(address as *mut u64) = number as u64,
            T_PYSSIZET => *(address as *mut isize) = number as isize,
            T_FLOAT => *(address as *mut f32) = real as f32,
            T_DOUBLE => *(address as *mut f64) = real,
            T_CHAR => {
                let text = crate::baseobjspace::text0_wtf8_w(value)?;
                let bytes = text.as_bytes();
                if bytes.len() != 1 {
                    return Err(crate::PyError::type_error(
                        "a cpyext char member takes a string of length 1",
                    ));
                }
                *(address as *mut u8) = bytes[0];
            }
            T_OBJECT | T_OBJECT_EX => {
                let slot = address as *mut *mut CPyObject;
                let previous = *slot;
                *slot = pyobject::make_ref(value);
                pyobject::decref(previous);
            }
            other => {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::SystemError,
                    format!("cpyext member '{}' uses unsupported type {other}", name()),
                ));
            }
        }
    }
    Ok(())
}

fn member_descr_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let Some(instance) = bound_instance(args) else {
        return Ok(carrier);
    };
    let member = carrier_def(carrier) as *mut CPyMemberDef;
    if member.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext member descriptor lost its definition",
        ));
    }
    read_member(instance, member)
}

fn member_descr_set(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let member = carrier_def(carrier) as *mut CPyMemberDef;
    if member.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext member descriptor lost its definition",
        ));
    }
    write_member(args[1], member, args[2])?;
    Ok(pyre_object::w_none())
}

// ── `tp_getset` ─────────────────────────────────────────────────────────

type Getter = unsafe extern "C" fn(*mut CPyObject, *mut c_void) -> *mut CPyObject;
type Setter = unsafe extern "C" fn(*mut CPyObject, *mut CPyObject, *mut c_void) -> c_int;

fn getset_descr_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let Some(instance) = bound_instance(args) else {
        return Ok(carrier);
    };
    let getset = carrier_def(carrier) as *mut CPyGetSetDef;
    if getset.is_null() || unsafe { (*getset).get.is_null() } {
        return Err(crate::PyError::attribute_error(format!(
            "attribute '{}' of a cpyext object is not readable",
            descriptor_name(carrier)
        )));
    }
    let receiver = pyobject::make_ref(instance);
    let result = unsafe {
        let get: Getter = std::mem::transmute((*getset).get);
        get(receiver, (*getset).closure)
    };
    unsafe { pyobject::decref(receiver) };
    super::from_c_result(result)
}

fn getset_descr_set(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let getset = carrier_def(carrier) as *mut CPyGetSetDef;
    if getset.is_null() || unsafe { (*getset).set.is_null() } {
        return Err(crate::PyError::attribute_error(format!(
            "attribute '{}' of a cpyext object is not writable",
            descriptor_name(carrier)
        )));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(args[1]);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(args[2]);
    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let value = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(value_slot));
    let result = unsafe {
        let set: Setter = std::mem::transmute((*getset).set);
        set(receiver, value, (*getset).closure)
    };
    unsafe {
        pyobject::decref(receiver);
        pyobject::decref(value);
    }
    if result != 0 {
        return Err(super::pyerrors::take_pending_error().unwrap_or_else(|| {
            crate::PyError::new(
                crate::PyErrorKind::SystemError,
                "a cpyext setter failed without setting an exception",
            )
        }));
    }
    Ok(pyre_object::w_none())
}

// ── the slot wrappers ───────────────────────────────────────────────────

/// The `(args, kwds)` pair a ternary slot takes, as owned references.
///
/// `call_cfunction` builds the same pair for `METH_VARARGS | METH_KEYWORDS`,
/// but a slot's result is not always a `PyObject *`, so the construction is
/// separated from the call here.
struct TernaryArgs {
    arguments: *mut CPyObject,
    keywords: *mut CPyObject,
}

impl Drop for TernaryArgs {
    fn drop(&mut self) {
        unsafe {
            pyobject::decref(self.arguments);
            pyobject::decref(self.keywords);
        }
    }
}

fn ternary_args(positional: &[PyObjectRef], keywords: &[(String, PyObjectRef)]) -> TernaryArgs {
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    for &argument in positional {
        roots.pin_root(argument);
    }
    for (_, value) in keywords {
        roots.pin_root(*value);
    }
    let value_slot = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    let items: Vec<PyObjectRef> = (0..positional.len()).map(value_slot).collect();
    let tuple = pyre_object::tupleobject::w_tuple_new(items);
    let tuple_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(tuple);
    let mut keywords_arg = std::ptr::null_mut();
    if !keywords.is_empty() {
        let dict = pyre_object::dictmultiobject::w_dict_new();
        let dict_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(dict);
        for (index, (name, _)) in keywords.iter().enumerate() {
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str(
                    pyre_object::gc_roots::shadow_stack_get(dict_slot),
                    name,
                    value_slot(positional.len() + index),
                )
            };
        }
        keywords_arg = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(dict_slot));
    }
    TernaryArgs {
        arguments: pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(tuple_slot)),
        keywords: keywords_arg,
    }
}

fn split_call(args: &[PyObjectRef]) -> (Vec<PyObjectRef>, Vec<(String, PyObjectRef)>) {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let keywords: Vec<(String, PyObjectRef)> = match kwargs {
        Some(dict) if crate::builtins::has_real_kwargs(kwargs) => unsafe {
            pyre_object::w_dict_str_entries(dict)
                .into_iter()
                .filter(|(key, _)| key != "__pyre_kw__")
                .collect()
        },
        _ => Vec::new(),
    };
    (positional.to_vec(), keywords)
}

/// `slot_tp_new` — `cls.__new__(cls, *args, **kwds)`.
fn slot_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args[0];
    let (positional, keywords) = split_call(&args[1..]);
    let slot = find_slot(cls, |tp| unsafe { (*tp).tp_new });
    if slot.is_null() {
        return Err(crate::PyError::type_error(
            "cannot create instances of this cpyext type",
        ));
    }
    super::call_cfunction(
        slot,
        super::methodobject::METH_VARARGS | super::methodobject::METH_KEYWORDS,
        cls,
        &positional,
        &keywords,
    )
}

/// `slot_tp_init` — a slot whose result is an `int`, not an object.
fn slot_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    let (positional, keywords) = split_call(&args[1..]);
    let slot = slot_of(w_self, |tp| unsafe { (*tp).tp_init });
    if slot.is_null() {
        return Ok(pyre_object::w_none());
    }
    let built = ternary_args(&positional, &keywords);
    let receiver = pyobject::make_ref(w_self);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject, *mut CPyObject) -> c_int =
            std::mem::transmute(slot);
        call(receiver, built.arguments, built.keywords)
    };
    unsafe { pyobject::decref(receiver) };
    if result != 0 {
        return Err(super::pyerrors::take_pending_error().unwrap_or_else(|| {
            crate::PyError::new(
                crate::PyErrorKind::SystemError,
                "a cpyext initializer failed without setting an exception",
            )
        }));
    }
    Ok(pyre_object::w_none())
}

fn slot_call(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    let (positional, keywords) = split_call(&args[1..]);
    let slot = slot_of(w_self, |tp| unsafe { (*tp).tp_call });
    if slot.is_null() {
        return Err(crate::PyError::type_error("cpyext object is not callable"));
    }
    super::call_cfunction(
        slot,
        super::methodobject::METH_VARARGS | super::methodobject::METH_KEYWORDS,
        w_self,
        &positional,
        &keywords,
    )
}

/// Run a `(self) -> PyObject *` slot.
fn unary_slot(
    w_self: PyObjectRef,
    pick: fn(*mut CPyTypeObject) -> *const c_void,
    missing: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let slot = slot_of(w_self, pick);
    if slot.is_null() {
        return Err(crate::PyError::type_error(format!(
            "cpyext object has no {missing}"
        )));
    }
    let receiver = pyobject::make_ref(w_self);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(receiver)
    };
    unsafe { pyobject::decref(receiver) };
    super::from_c_result(result)
}

fn slot_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unary_slot(args[0], |tp| unsafe { (*tp).tp_repr }, "tp_repr")
}

fn slot_str(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unary_slot(args[0], |tp| unsafe { (*tp).tp_str }, "tp_str")
}

fn slot_iter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unary_slot(args[0], |tp| unsafe { (*tp).tp_iter }, "tp_iter")
}

/// `tp_iternext` reports exhaustion with NULL and no exception set, which is
/// the one place a NULL result is not an error.
fn slot_iternext(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    let slot = slot_of(w_self, |tp| unsafe { (*tp).tp_iternext });
    if slot.is_null() {
        return Err(crate::PyError::type_error(
            "cpyext object is not an iterator",
        ));
    }
    let receiver = pyobject::make_ref(w_self);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(receiver)
    };
    unsafe { pyobject::decref(receiver) };
    if result.is_null() && !super::pyerrors::has_pending_error() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::StopIteration,
            String::new(),
        ));
    }
    super::from_c_result(result)
}

// ── the `tp_as_async` table ─────────────────────────────────────────────

/// One `PyAsyncMethods` slot, picked off a type that may declare no table.
macro_rules! async_slot {
    ($field:ident) => {
        (|tp: *mut CPyTypeObject| unsafe {
            let table = (*tp).tp_as_async;
            if table.is_null() {
                std::ptr::null()
            } else {
                (*table).$field
            }
        }) as fn(*mut CPyTypeObject) -> *const c_void
    };
}

fn slot_await(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unary_slot(args[0], async_slot!(am_await), "am_await")
}

fn slot_aiter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unary_slot(args[0], async_slot!(am_aiter), "am_aiter")
}

/// `am_anext` ends an async iteration the way `tp_iternext` ends a synchronous
/// one, with `StopAsyncIteration` in place of `StopIteration`.
fn slot_anext(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    let slot = slot_of(w_self, async_slot!(am_anext));
    if slot.is_null() {
        return Err(crate::PyError::type_error(
            "cpyext object is not an async iterator",
        ));
    }
    let receiver = pyobject::make_ref(w_self);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(receiver)
    };
    unsafe { pyobject::decref(receiver) };
    if result.is_null() && !super::pyerrors::has_pending_error() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::StopAsyncIteration,
            String::new(),
        ));
    }
    super::from_c_result(result)
}

fn slot_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    let slot = slot_of(w_self, |tp| unsafe { (*tp).tp_hash });
    if slot.is_null() {
        return Err(crate::PyError::type_error("cpyext object is not hashable"));
    }
    let receiver = pyobject::make_ref(w_self);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject) -> isize = std::mem::transmute(slot);
        call(receiver)
    };
    unsafe { pyobject::decref(receiver) };
    if result == -1 && super::pyerrors::has_pending_error() {
        return Err(super::pyerrors::take_pending_error().expect("just observed"));
    }
    Ok(pyre_object::w_int_new(result as i64))
}

/// `Py_LT` .. `Py_GE`.
fn rich_compare(args: &[PyObjectRef], operation: c_int) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    let slot = slot_of(w_self, |tp| unsafe { (*tp).tp_richcompare });
    if slot.is_null() {
        return Ok(pyre_object::w_not_implemented());
    }
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_self);
    let other_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(args[1]);
    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let other = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(other_slot));
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject, c_int) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(receiver, other, operation)
    };
    unsafe {
        pyobject::decref(receiver);
        pyobject::decref(other);
    }
    super::from_c_result(result)
}

macro_rules! comparison_slots {
    ($($wrapper:ident => $dunder:literal, $operation:expr;)*) => {
        $(
            fn $wrapper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                rich_compare(args, $operation)
            }
        )*

        fn install_comparisons(ns: PyObjectRef) {
            $(
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        $dunder,
                        crate::make_builtin_function_with_arity($dunder, $wrapper, 2),
                    )
                };
            )*
        }
    };
}

comparison_slots! {
    slot_lt => "__lt__", 0;
    slot_le => "__le__", 1;
    slot_eq => "__eq__", 2;
    slot_ne => "__ne__", 3;
    slot_gt => "__gt__", 4;
    slot_ge => "__ge__", 5;
}

// ── the protocol tables ─────────────────────────────────────────────────
//
// Each table slot becomes the dunder `slotdefs.py` names for it.  A wrapper
// re-derives the slot from the receiver's MRO, so a Python subclass of a
// C-defined type reaches the base's table through the namespace it inherits.

macro_rules! number_slot {
    ($field:ident) => {
        (|tp| unsafe {
            let table = (*tp).tp_as_number;
            if table.is_null() {
                std::ptr::null()
            } else {
                (*table).$field
            }
        }) as fn(*mut CPyTypeObject) -> *const c_void
    };
}

macro_rules! sequence_slot {
    ($field:ident) => {
        (|tp| unsafe {
            let table = (*tp).tp_as_sequence;
            if table.is_null() {
                std::ptr::null()
            } else {
                (*table).$field
            }
        }) as fn(*mut CPyTypeObject) -> *const c_void
    };
}

macro_rules! mapping_slot {
    ($field:ident) => {
        (|tp| unsafe {
            let table = (*tp).tp_as_mapping;
            if table.is_null() {
                std::ptr::null()
            } else {
                (*table).$field
            }
        }) as fn(*mut CPyTypeObject) -> *const c_void
    };
}

/// Run a `(PyObject *, PyObject *) -> PyObject *` slot.
fn call_binary(
    slot: *const c_void,
    first: PyObjectRef,
    second: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let first_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(first);
    let second_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(second);
    let left = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(first_slot));
    let right = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(second_slot));
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(left, right)
    };
    unsafe {
        pyobject::decref(left);
        pyobject::decref(right);
    }
    super::from_c_result(result)
}

/// `left op right` through `pick`, in the order the operands were written.
///
/// `nb_add` serves `__add__` and `__radd__` both, as `SLOT1BIN` does: the
/// reflected form hands the C function the pair the other way round, and a
/// table that does not handle it answers `NotImplemented` itself.
fn binary(
    args: &[PyObjectRef],
    pick: fn(*mut CPyTypeObject) -> *const c_void,
    reflected: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let slot = slot_of(args[0], pick);
    if slot.is_null() {
        return Ok(pyre_object::w_not_implemented());
    }
    let (first, second) = if reflected {
        (args[1], args[0])
    } else {
        (args[0], args[1])
    };
    call_binary(slot, first, second)
}

macro_rules! number_binaries {
    ($($field:ident, $direct_name:literal, $direct:ident, $reflected_name:literal, $reflected:ident;)*) => {
        $(
            fn $direct(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                binary(args, number_slot!($field), false)
            }

            fn $reflected(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                binary(args, number_slot!($field), true)
            }
        )*

        fn install_number_binaries(ns: PyObjectRef, tp: *mut CPyTypeObject) {
            $(
                if !(number_slot!($field))(tp).is_null() {
                    store(
                        ns,
                        $direct_name,
                        crate::make_builtin_function_with_arity($direct_name, $direct, 2),
                    );
                    store(
                        ns,
                        $reflected_name,
                        crate::make_builtin_function_with_arity($reflected_name, $reflected, 2),
                    );
                }
            )*
        }
    };
}

number_binaries! {
    nb_add, "__add__", nb_add_direct, "__radd__", nb_add_reflected;
    nb_subtract, "__sub__", nb_sub_direct, "__rsub__", nb_sub_reflected;
    nb_multiply, "__mul__", nb_mul_direct, "__rmul__", nb_mul_reflected;
    nb_remainder, "__mod__", nb_mod_direct, "__rmod__", nb_mod_reflected;
    nb_divmod, "__divmod__", nb_divmod_direct, "__rdivmod__", nb_divmod_reflected;
    nb_lshift, "__lshift__", nb_lshift_direct, "__rlshift__", nb_lshift_reflected;
    nb_rshift, "__rshift__", nb_rshift_direct, "__rrshift__", nb_rshift_reflected;
    nb_and, "__and__", nb_and_direct, "__rand__", nb_and_reflected;
    nb_xor, "__xor__", nb_xor_direct, "__rxor__", nb_xor_reflected;
    nb_or, "__or__", nb_or_direct, "__ror__", nb_or_reflected;
    nb_floor_divide, "__floordiv__", nb_floordiv_direct, "__rfloordiv__", nb_floordiv_reflected;
    nb_true_divide, "__truediv__", nb_truediv_direct, "__rtruediv__", nb_truediv_reflected;
    nb_matrix_multiply, "__matmul__", nb_matmul_direct, "__rmatmul__", nb_matmul_reflected;
}

macro_rules! number_inplace {
    ($($field:ident, $name:literal, $wrapper:ident;)*) => {
        $(
            fn $wrapper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                binary(args, number_slot!($field), false)
            }
        )*

        fn install_number_inplace(ns: PyObjectRef, tp: *mut CPyTypeObject) {
            $(
                if !(number_slot!($field))(tp).is_null() {
                    store(
                        ns,
                        $name,
                        crate::make_builtin_function_with_arity($name, $wrapper, 2),
                    );
                }
            )*
        }
    };
}

number_inplace! {
    nb_inplace_add, "__iadd__", nb_iadd;
    nb_inplace_subtract, "__isub__", nb_isub;
    nb_inplace_multiply, "__imul__", nb_imul;
    nb_inplace_remainder, "__imod__", nb_imod;
    nb_inplace_lshift, "__ilshift__", nb_ilshift;
    nb_inplace_rshift, "__irshift__", nb_irshift;
    nb_inplace_and, "__iand__", nb_iand;
    nb_inplace_xor, "__ixor__", nb_ixor;
    nb_inplace_or, "__ior__", nb_ior;
    nb_inplace_floor_divide, "__ifloordiv__", nb_ifloordiv;
    nb_inplace_true_divide, "__itruediv__", nb_itruediv;
    nb_inplace_matrix_multiply, "__imatmul__", nb_imatmul;
}

macro_rules! number_unaries {
    ($($field:ident, $name:literal, $wrapper:ident;)*) => {
        $(
            fn $wrapper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
                unary_slot(args[0], number_slot!($field), stringify!($field))
            }
        )*

        fn install_number_unaries(ns: PyObjectRef, tp: *mut CPyTypeObject) {
            $(
                if !(number_slot!($field))(tp).is_null() {
                    store(
                        ns,
                        $name,
                        crate::make_builtin_function_with_arity($name, $wrapper, 1),
                    );
                }
            )*
        }
    };
}

number_unaries! {
    nb_negative, "__neg__", nb_neg;
    nb_positive, "__pos__", nb_pos;
    nb_absolute, "__abs__", nb_abs;
    nb_invert, "__invert__", nb_invert_wrapper;
    nb_int, "__int__", nb_int_wrapper;
    nb_float, "__float__", nb_float_wrapper;
    nb_index, "__index__", nb_index_wrapper;
}

/// `nb_bool` is an `inquiry`: a negative result is the error indicator.
fn nb_bool_wrapper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let slot = slot_of(args[0], number_slot!(nb_bool));
    if slot.is_null() {
        return Ok(pyre_object::boolobject::w_bool_from(true));
    }
    let receiver = pyobject::make_ref(args[0]);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject) -> c_int = std::mem::transmute(slot);
        call(receiver)
    };
    unsafe { pyobject::decref(receiver) };
    if result < 0 {
        return Err(pending_or(
            "a cpyext truth test failed without setting an exception",
        ));
    }
    Ok(pyre_object::boolobject::w_bool_from(result != 0))
}

/// `nb_power` is a `ternaryfunc`, the third operand being the modulus `pow`
/// takes and every other route leaves as `None`.
fn power(args: &[PyObjectRef], reflected: bool) -> Result<PyObjectRef, crate::PyError> {
    let slot = slot_of(args[0], number_slot!(nb_power));
    if slot.is_null() {
        return Ok(pyre_object::w_not_implemented());
    }
    let (first, second) = if reflected {
        (args[1], args[0])
    } else {
        (args[0], args[1])
    };
    let modulus = args.get(2).copied().unwrap_or_else(pyre_object::w_none);
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(first);
    roots.pin_root(second);
    roots.pin_root(modulus);
    let owned =
        |index: usize| pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(base + index));
    let (left, right, third) = (owned(0), owned(1), owned(2));
    let result = unsafe {
        let call: unsafe extern "C" fn(
            *mut CPyObject,
            *mut CPyObject,
            *mut CPyObject,
        ) -> *mut CPyObject = std::mem::transmute(slot);
        call(left, right, third)
    };
    unsafe {
        pyobject::decref(left);
        pyobject::decref(right);
        pyobject::decref(third);
    }
    super::from_c_result(result)
}

fn nb_pow_direct(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    power(args, false)
}

fn nb_pow_reflected(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    power(args, true)
}

fn nb_ipow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let slot = slot_of(args[0], number_slot!(nb_inplace_power));
    if slot.is_null() {
        return Ok(pyre_object::w_not_implemented());
    }
    // The in-place slot has the same shape, so the direct path serves it once
    // the lookup has been pointed at it.
    call_ternary(slot, args[0], args[1], args.get(2).copied())
}

fn call_ternary(
    slot: *const c_void,
    first: PyObjectRef,
    second: PyObjectRef,
    third: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(first);
    roots.pin_root(second);
    roots.pin_root(third.unwrap_or_else(pyre_object::w_none));
    let owned =
        |index: usize| pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(base + index));
    let (left, right, modulus) = (owned(0), owned(1), owned(2));
    let result = unsafe {
        let call: unsafe extern "C" fn(
            *mut CPyObject,
            *mut CPyObject,
            *mut CPyObject,
        ) -> *mut CPyObject = std::mem::transmute(slot);
        call(left, right, modulus)
    };
    unsafe {
        pyobject::decref(left);
        pyobject::decref(right);
        pyobject::decref(modulus);
    }
    super::from_c_result(result)
}

/// The pending C exception, or a `SystemError` naming the slot that returned
/// its failure indicator without setting one.
pub(super) fn pending_or(message: &str) -> crate::PyError {
    super::pyerrors::take_pending_error()
        .unwrap_or_else(|| crate::PyError::new(crate::PyErrorKind::SystemError, message))
}

// ── the sequence and mapping tables ─────────────────────────────────────

/// Run a `lenfunc`.
fn call_length(slot: *const c_void, w_self: PyObjectRef) -> Result<isize, crate::PyError> {
    let receiver = pyobject::make_ref(w_self);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject) -> isize = std::mem::transmute(slot);
        call(receiver)
    };
    unsafe { pyobject::decref(receiver) };
    if result < 0 {
        return Err(pending_or(
            "a cpyext length slot failed without setting an exception",
        ));
    }
    Ok(result)
}

/// `mp_length` first, `sq_length` second -- the order `PyObject_Size` uses.
fn length_slot(w_self: PyObjectRef) -> *const c_void {
    let slot = slot_of(w_self, mapping_slot!(mp_length));
    if slot.is_null() {
        slot_of(w_self, sequence_slot!(sq_length))
    } else {
        slot
    }
}

fn slot_len(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let slot = length_slot(args[0]);
    if slot.is_null() {
        return Err(crate::PyError::type_error("cpyext object has no length"));
    }
    Ok(pyre_object::w_int_new(call_length(slot, args[0])? as i64))
}

/// The index `sq_item` and `sq_ass_item` take: `__index__` of the key, with a
/// negative value folded through the length as `wrap_sq_item` does.
fn sequence_index(w_self: PyObjectRef, key: PyObjectRef) -> Result<isize, crate::PyError> {
    let mut index = crate::baseobjspace::getindex_w(key)? as isize;
    if index < 0 {
        let slot = length_slot(w_self);
        if !slot.is_null() {
            index += call_length(slot, w_self)?;
        }
    }
    Ok(index)
}

fn slot_getitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    let subscript = slot_of(w_self, mapping_slot!(mp_subscript));
    if !subscript.is_null() {
        return call_binary(subscript, w_self, args[1]);
    }
    let item = slot_of(w_self, sequence_slot!(sq_item));
    if item.is_null() {
        return Err(crate::PyError::type_error(
            "cpyext object is not subscriptable",
        ));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_self);
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(args[1]);
    let index = sequence_index(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        pyre_object::gc_roots::shadow_stack_get(key_slot),
    )?;
    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, isize) -> *mut CPyObject =
            std::mem::transmute(item);
        call(receiver, index)
    };
    unsafe { pyobject::decref(receiver) };
    super::from_c_result(result)
}

/// `__setitem__` and `__delitem__` both, the deletion passing NULL.
fn assign_item(
    w_self: PyObjectRef,
    key: PyObjectRef,
    value: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_self);
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(key);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(value.unwrap_or_else(pyre_object::w_none));
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);

    let subscript = slot_of(reload(self_slot), mapping_slot!(mp_ass_subscript));
    let failed = if !subscript.is_null() {
        let receiver = pyobject::make_ref(reload(self_slot));
        let index = pyobject::make_ref(reload(key_slot));
        let item = match value {
            Some(_) => pyobject::make_ref(reload(value_slot)),
            None => std::ptr::null_mut(),
        };
        let result = unsafe {
            let call: unsafe extern "C" fn(
                *mut CPyObject,
                *mut CPyObject,
                *mut CPyObject,
            ) -> c_int = std::mem::transmute(subscript);
            call(receiver, index, item)
        };
        unsafe {
            pyobject::decref(receiver);
            pyobject::decref(index);
            pyobject::decref(item);
        }
        result != 0
    } else {
        let assign = slot_of(reload(self_slot), sequence_slot!(sq_ass_item));
        if assign.is_null() {
            return Err(crate::PyError::type_error(
                "cpyext object does not support item assignment",
            ));
        }
        let index = sequence_index(reload(self_slot), reload(key_slot))?;
        let receiver = pyobject::make_ref(reload(self_slot));
        let item = match value {
            Some(_) => pyobject::make_ref(reload(value_slot)),
            None => std::ptr::null_mut(),
        };
        let result = unsafe {
            let call: unsafe extern "C" fn(*mut CPyObject, isize, *mut CPyObject) -> c_int =
                std::mem::transmute(assign);
            call(receiver, index, item)
        };
        unsafe {
            pyobject::decref(receiver);
            pyobject::decref(item);
        }
        result != 0
    };
    if failed {
        return Err(pending_or(
            "a cpyext item assignment failed without setting an exception",
        ));
    }
    Ok(pyre_object::w_none())
}

fn slot_setitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assign_item(args[0], args[1], Some(args[2]))
}

fn slot_delitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assign_item(args[0], args[1], None)
}

fn slot_contains(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let slot = slot_of(args[0], sequence_slot!(sq_contains));
    if slot.is_null() {
        return Err(crate::PyError::type_error(
            "cpyext object does not support membership tests",
        ));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(args[0]);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(args[1]);
    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let value = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(value_slot));
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject) -> c_int =
            std::mem::transmute(slot);
        call(receiver, value)
    };
    unsafe {
        pyobject::decref(receiver);
        pyobject::decref(value);
    }
    if result < 0 {
        return Err(pending_or(
            "a cpyext membership test failed without setting an exception",
        ));
    }
    Ok(pyre_object::boolobject::w_bool_from(result != 0))
}

/// `sq_concat` and `sq_inplace_concat` are `binaryfunc`s like `nb_add`.
fn sq_concat_wrapper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    binary(args, sequence_slot!(sq_concat), false)
}

fn sq_inplace_concat_wrapper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    binary(args, sequence_slot!(sq_inplace_concat), false)
}

/// `sq_repeat` takes the count as a `Py_ssize_t`, so `n * seq` and `seq * n`
/// reach it the same way.
fn repeat(
    w_self: PyObjectRef,
    count: PyObjectRef,
    pick: fn(*mut CPyTypeObject) -> *const c_void,
) -> Result<PyObjectRef, crate::PyError> {
    let slot = slot_of(w_self, pick);
    if slot.is_null() {
        return Ok(pyre_object::w_not_implemented());
    }
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_self);
    let count_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(count);
    let times =
        crate::baseobjspace::getindex_w(pyre_object::gc_roots::shadow_stack_get(count_slot))?;
    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, isize) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(receiver, times as isize)
    };
    unsafe { pyobject::decref(receiver) };
    super::from_c_result(result)
}

/// `seq * n` and `n * seq` both repeat `seq`, so one wrapper serves both.
fn sq_repeat_wrapper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    repeat(args[0], args[1], sequence_slot!(sq_repeat))
}

fn sq_inplace_repeat_wrapper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    repeat(args[0], args[1], sequence_slot!(sq_inplace_repeat))
}

/// Install everything the three tables contribute.
///
/// `__len__`, `__getitem__` and the item assignments are shared between the
/// mapping and sequence tables, so each is installed once and the wrapper
/// picks the slot; `__add__` and `__mul__` go to the number table when it has
/// them and to the sequence table otherwise, as `slotdefs.py` orders them.
fn install_protocols(ns: PyObjectRef, tp: *mut CPyTypeObject) {
    install_number_binaries(ns, tp);
    install_number_inplace(ns, tp);
    install_number_unaries(ns, tp);
    if !(number_slot!(nb_bool))(tp).is_null() {
        store(
            ns,
            "__bool__",
            crate::make_builtin_function_with_arity("__bool__", nb_bool_wrapper, 1),
        );
    }
    if !(number_slot!(nb_power))(tp).is_null() {
        store(
            ns,
            "__pow__",
            crate::make_builtin_function("__pow__", nb_pow_direct),
        );
        store(
            ns,
            "__rpow__",
            crate::make_builtin_function("__rpow__", nb_pow_reflected),
        );
    }
    if !(number_slot!(nb_inplace_power))(tp).is_null() {
        store(
            ns,
            "__ipow__",
            crate::make_builtin_function("__ipow__", nb_ipow),
        );
    }

    if !(mapping_slot!(mp_length))(tp).is_null() || !(sequence_slot!(sq_length))(tp).is_null() {
        store(
            ns,
            "__len__",
            crate::make_builtin_function_with_arity("__len__", slot_len, 1),
        );
    }
    if !(mapping_slot!(mp_subscript))(tp).is_null() || !(sequence_slot!(sq_item))(tp).is_null() {
        store(
            ns,
            "__getitem__",
            crate::make_builtin_function_with_arity("__getitem__", slot_getitem, 2),
        );
    }
    if !(mapping_slot!(mp_ass_subscript))(tp).is_null()
        || !(sequence_slot!(sq_ass_item))(tp).is_null()
    {
        store(
            ns,
            "__setitem__",
            crate::make_builtin_function_with_arity("__setitem__", slot_setitem, 3),
        );
        store(
            ns,
            "__delitem__",
            crate::make_builtin_function_with_arity("__delitem__", slot_delitem, 2),
        );
    }
    if !(sequence_slot!(sq_contains))(tp).is_null() {
        store(
            ns,
            "__contains__",
            crate::make_builtin_function_with_arity("__contains__", slot_contains, 2),
        );
    }
    if !(sequence_slot!(sq_concat))(tp).is_null() && (number_slot!(nb_add))(tp).is_null() {
        store(
            ns,
            "__add__",
            crate::make_builtin_function_with_arity("__add__", sq_concat_wrapper, 2),
        );
    }
    if !(sequence_slot!(sq_inplace_concat))(tp).is_null()
        && (number_slot!(nb_inplace_add))(tp).is_null()
    {
        store(
            ns,
            "__iadd__",
            crate::make_builtin_function_with_arity("__iadd__", sq_inplace_concat_wrapper, 2),
        );
    }
    if !(sequence_slot!(sq_repeat))(tp).is_null() && (number_slot!(nb_multiply))(tp).is_null() {
        store(
            ns,
            "__mul__",
            crate::make_builtin_function_with_arity("__mul__", sq_repeat_wrapper, 2),
        );
        store(
            ns,
            "__rmul__",
            crate::make_builtin_function_with_arity("__rmul__", sq_repeat_wrapper, 2),
        );
    }
    if !(sequence_slot!(sq_inplace_repeat))(tp).is_null()
        && (number_slot!(nb_inplace_multiply))(tp).is_null()
    {
        store(
            ns,
            "__imul__",
            crate::make_builtin_function_with_arity("__imul__", sq_inplace_repeat_wrapper, 2),
        );
    }
}

// ── `PyType_Ready` ──────────────────────────────────────────────────────

fn store(ns: PyObjectRef, name: &str, value: PyObjectRef) {
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, name, value) };
}

/// Copy every slot `tp` leaves null from its base -- CPython `inherit_slots`.
fn inherit_slots(tp: *mut CPyTypeObject, base: *mut CPyTypeObject) {
    if base.is_null() {
        return;
    }
    macro_rules! inherit {
        ($($field:ident),* $(,)?) => {
            $(
                unsafe {
                    if (*tp).$field.is_null() {
                        (*tp).$field = (*base).$field;
                    }
                }
            )*
        };
    }
    inherit!(
        tp_dealloc,
        tp_repr,
        tp_hash,
        tp_call,
        tp_str,
        tp_getattro,
        tp_setattro,
        tp_traverse,
        tp_clear,
        tp_richcompare,
        tp_iter,
        tp_iternext,
        tp_descr_get,
        tp_descr_set,
        tp_init,
        tp_alloc,
        tp_new,
        tp_free,
        tp_finalize,
    );
    unsafe {
        if (*tp).tp_basicsize == 0 {
            (*tp).tp_basicsize = (*base).tp_basicsize;
        }
        if (*tp).tp_itemsize == 0 {
            (*tp).tp_itemsize = (*base).tp_itemsize;
        }
    }
}

/// Fill the namespace the interpreter type is built from.
fn install_namespace(ns: PyObjectRef, tp: *mut CPyTypeObject) {
    let roots = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(ns);
    let reload = || pyre_object::gc_roots::shadow_stack_get(ns_slot);

    if unsafe { !(*tp).tp_doc.is_null() } {
        let doc = text_or_none(unsafe { (*tp).tp_doc });
        store(reload(), DOC_KEY, doc);
    }

    let mut index = 0isize;
    while unsafe {
        !(*tp).tp_methods.is_null() && !(*(*tp).tp_methods.offset(index)).ml_name.is_null()
    } {
        let method = unsafe { (*tp).tp_methods.offset(index) };
        index += 1;
        let name = unsafe { std::ffi::CStr::from_ptr((*method).ml_name) }
            .to_string_lossy()
            .into_owned();
        let descriptor = new_carrier(
            method_descriptor_type(),
            method as usize,
            unsafe { (*method).ml_name },
            unsafe { (*method).ml_doc },
            pyre_object::PY_NULL,
        );
        let descriptor_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(descriptor);
        store(
            reload(),
            &name,
            pyre_object::gc_roots::shadow_stack_get(descriptor_slot),
        );
    }

    let mut index = 0isize;
    while unsafe {
        !(*tp).tp_members.is_null() && !(*(*tp).tp_members.offset(index)).name.is_null()
    } {
        let member = unsafe { (*tp).tp_members.offset(index) };
        index += 1;
        let name = unsafe { std::ffi::CStr::from_ptr((*member).name) }
            .to_string_lossy()
            .into_owned();
        let descriptor = new_carrier(
            member_descriptor_type(),
            member as usize,
            unsafe { (*member).name },
            unsafe { (*member).doc },
            pyre_object::PY_NULL,
        );
        let descriptor_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(descriptor);
        store(
            reload(),
            &name,
            pyre_object::gc_roots::shadow_stack_get(descriptor_slot),
        );
    }

    let mut index = 0isize;
    while unsafe { !(*tp).tp_getset.is_null() && !(*(*tp).tp_getset.offset(index)).name.is_null() }
    {
        let getset = unsafe { (*tp).tp_getset.offset(index) };
        index += 1;
        let name = unsafe { std::ffi::CStr::from_ptr((*getset).name) }
            .to_string_lossy()
            .into_owned();
        let descriptor = new_carrier(
            getset_descriptor_type(),
            getset as usize,
            unsafe { (*getset).name },
            unsafe { (*getset).doc },
            pyre_object::PY_NULL,
        );
        let descriptor_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(descriptor);
        store(
            reload(),
            &name,
            pyre_object::gc_roots::shadow_stack_get(descriptor_slot),
        );
    }

    let ns = reload();
    store(
        ns,
        "__new__",
        crate::make_builtin_function("__new__", slot_new),
    );
    store(
        ns,
        "__init__",
        crate::make_builtin_function("__init__", slot_init),
    );
    let unary: [(
        &'static str,
        fn(*mut CPyTypeObject) -> *const c_void,
        crate::gateway::BuiltinCodeFn,
    ); 4] = [
        ("__repr__", |tp| unsafe { (*tp).tp_repr }, slot_repr),
        ("__str__", |tp| unsafe { (*tp).tp_str }, slot_str),
        ("__iter__", |tp| unsafe { (*tp).tp_iter }, slot_iter),
        ("__next__", |tp| unsafe { (*tp).tp_iternext }, slot_iternext),
    ];
    for (dunder, pick, wrapper) in unary {
        if !pick(tp).is_null() {
            store(
                ns,
                dunder,
                crate::make_builtin_function_with_arity(dunder, wrapper, 1),
            );
        }
    }
    if unsafe { !(*tp).tp_hash.is_null() } {
        store(
            ns,
            "__hash__",
            crate::make_builtin_function_with_arity("__hash__", slot_hash, 1),
        );
    }
    if unsafe { !(*tp).tp_call.is_null() } {
        store(
            ns,
            "__call__",
            crate::make_builtin_function("__call__", slot_call),
        );
    }
    if unsafe { !(*tp).tp_richcompare.is_null() } {
        install_comparisons(ns);
    }
    if unsafe { !(*tp).tp_getattro.is_null() } {
        store(
            ns,
            "__getattribute__",
            crate::make_builtin_function_with_arity("__getattribute__", slot_getattro, 2),
        );
    }
    if unsafe { !(*tp).tp_setattro.is_null() } {
        store(
            ns,
            "__setattr__",
            crate::make_builtin_function_with_arity("__setattr__", slot_setattro, 3),
        );
        store(
            ns,
            "__delattr__",
            crate::make_builtin_function_with_arity("__delattr__", slot_delattro, 2),
        );
    }
    if unsafe { !(*tp).tp_descr_get.is_null() } {
        store(
            ns,
            "__get__",
            crate::make_builtin_function_with_arity("__get__", slot_descr_get, 3),
        );
    }
    if unsafe { !(*tp).tp_descr_set.is_null() } {
        store(
            ns,
            "__set__",
            crate::make_builtin_function_with_arity("__set__", slot_descr_set, 3),
        );
        store(
            ns,
            "__delete__",
            crate::make_builtin_function_with_arity("__delete__", slot_descr_delete, 2),
        );
    }
    let asynchronous: [(
        &'static str,
        fn(*mut CPyTypeObject) -> *const c_void,
        crate::gateway::BuiltinCodeFn,
    ); 3] = [
        ("__await__", async_slot!(am_await), slot_await),
        ("__aiter__", async_slot!(am_aiter), slot_aiter),
        ("__anext__", async_slot!(am_anext), slot_anext),
    ];
    for (dunder, pick, wrapper) in asynchronous {
        if !pick(tp).is_null() {
            store(
                ns,
                dunder,
                crate::make_builtin_function_with_arity(dunder, wrapper, 1),
            );
        }
    }
    install_protocols(ns, tp);
}

// ── `tp_getattro`, `tp_setattro` and the descriptor slots ───────────────

fn slot_getattro(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let slot = slot_of(args[0], |tp| unsafe { (*tp).tp_getattro });
    if slot.is_null() {
        return Err(crate::PyError::type_error(
            "cpyext object has no attribute access",
        ));
    }
    call_binary(slot, args[0], args[1])
}

/// `tp_setattro` with a NULL value is the deletion, as it is for item
/// assignment.
fn set_attribute(
    w_self: PyObjectRef,
    name: PyObjectRef,
    value: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let slot = slot_of(w_self, |tp| unsafe { (*tp).tp_setattro });
    if slot.is_null() {
        return Err(crate::PyError::type_error(
            "cpyext object does not support attribute assignment",
        ));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_self);
    roots.pin_root(name);
    roots.pin_root(value.unwrap_or_else(pyre_object::w_none));
    let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    let receiver = pyobject::make_ref(reload(0));
    let key = pyobject::make_ref(reload(1));
    let item = match value {
        Some(_) => pyobject::make_ref(reload(2)),
        None => std::ptr::null_mut(),
    };
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject, *mut CPyObject) -> c_int =
            std::mem::transmute(slot);
        call(receiver, key, item)
    };
    unsafe {
        pyobject::decref(receiver);
        pyobject::decref(key);
        pyobject::decref(item);
    }
    if result != 0 {
        return Err(pending_or(
            "a cpyext attribute assignment failed without setting an exception",
        ));
    }
    Ok(pyre_object::w_none())
}

fn slot_setattro(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    set_attribute(args[0], args[1], Some(args[2]))
}

fn slot_delattro(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    set_attribute(args[0], args[1], None)
}

/// `tp_descr_get(self, obj, type)` — `obj` is NULL for a class access.
fn slot_descr_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let slot = slot_of(args[0], |tp| unsafe { (*tp).tp_descr_get });
    if slot.is_null() {
        return Ok(args[0]);
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(args[0]);
    roots.pin_root(args[1]);
    roots.pin_root(args[2]);
    let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    let none = |value: PyObjectRef| unsafe { pyre_object::is_none(value) };
    let descriptor = pyobject::make_ref(reload(0));
    let instance = if none(reload(1)) {
        std::ptr::null_mut()
    } else {
        pyobject::make_ref(reload(1))
    };
    let owner = if none(reload(2)) {
        std::ptr::null_mut()
    } else {
        pyobject::make_ref(reload(2))
    };
    let result = unsafe {
        let call: unsafe extern "C" fn(
            *mut CPyObject,
            *mut CPyObject,
            *mut CPyObject,
        ) -> *mut CPyObject = std::mem::transmute(slot);
        call(descriptor, instance, owner)
    };
    unsafe {
        pyobject::decref(descriptor);
        pyobject::decref(instance);
        pyobject::decref(owner);
    }
    super::from_c_result(result)
}

fn descr_assign(
    descriptor: PyObjectRef,
    instance: PyObjectRef,
    value: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let slot = slot_of(descriptor, |tp| unsafe { (*tp).tp_descr_set });
    if slot.is_null() {
        return Err(crate::PyError::attribute_error(
            "cpyext descriptor does not support assignment",
        ));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(descriptor);
    roots.pin_root(instance);
    roots.pin_root(value.unwrap_or_else(pyre_object::w_none));
    let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    let owner = pyobject::make_ref(reload(0));
    let target = pyobject::make_ref(reload(1));
    let item = match value {
        Some(_) => pyobject::make_ref(reload(2)),
        None => std::ptr::null_mut(),
    };
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject, *mut CPyObject) -> c_int =
            std::mem::transmute(slot);
        call(owner, target, item)
    };
    unsafe {
        pyobject::decref(owner);
        pyobject::decref(target);
        pyobject::decref(item);
    }
    if result != 0 {
        return Err(pending_or(
            "a cpyext descriptor assignment failed without setting an exception",
        ));
    }
    Ok(pyre_object::w_none())
}

fn slot_descr_set(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    descr_assign(args[0], args[1], Some(args[2]))
}

fn slot_descr_delete(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    descr_assign(args[0], args[1], None)
}

/// Give every descriptor the class it was defined on.
///
/// The carriers are built before the type exists, so `__objclass__` is stamped
/// once it does — the same shape `stamp_new_descr_self` uses for `__new__`.
fn stamp_objclass(w_type: PyObjectRef, tp: *mut CPyTypeObject) {
    let roots = pyre_object::gc_roots::push_roots();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_type);
    let mut names: Vec<String> = Vec::new();
    let mut collect = |mut next: Box<dyn FnMut(isize) -> *const c_char>| {
        let mut index = 0isize;
        loop {
            let name = next(index);
            if name.is_null() {
                break;
            }
            names.push(
                unsafe { std::ffi::CStr::from_ptr(name) }
                    .to_string_lossy()
                    .into_owned(),
            );
            index += 1;
        }
    };
    let methods = unsafe { (*tp).tp_methods };
    if !methods.is_null() {
        collect(Box::new(move |index| unsafe {
            (*methods.offset(index)).ml_name
        }));
    }
    let members = unsafe { (*tp).tp_members };
    if !members.is_null() {
        collect(Box::new(move |index| unsafe {
            (*members.offset(index)).name
        }));
    }
    let getset = unsafe { (*tp).tp_getset };
    if !getset.is_null() {
        collect(Box::new(move |index| unsafe {
            (*getset.offset(index)).name
        }));
    }
    for name in names {
        let w_type = pyre_object::gc_roots::shadow_stack_get(type_slot);
        if let Some(descriptor) = unsafe { crate::baseobjspace::lookup_in_type(w_type, &name) } {
            carrier_set(
                descriptor,
                OBJCLASS_KEY,
                pyre_object::gc_roots::shadow_stack_get(type_slot),
            );
        }
    }
}

fn ready(tp: *mut CPyTypeObject) -> Result<(), crate::PyError> {
    if tp.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyType_Ready(): NULL type",
        ));
    }
    if unsafe { (*tp).tp_flags } & PY_TPFLAGS_READY != 0 {
        return Ok(());
    }
    if unsafe { (*tp).tp_flags } & PY_TPFLAGS_READYING != 0 {
        return Err(crate::PyError::runtime_error(
            "PyType_Ready(): circular type hierarchy",
        ));
    }
    unsafe { (*tp).tp_flags |= PY_TPFLAGS_READYING };

    let base = unsafe { (*tp).tp_base };
    if !base.is_null() {
        ready(base)?;
    }
    inherit_slots(tp, base);
    if unsafe { (*tp).tp_basicsize } == 0 {
        unsafe { (*tp).tp_basicsize = size_of::<CPyObject>() as isize };
    }
    if unsafe { (*tp).tp_alloc.is_null() } {
        unsafe { (*tp).tp_alloc = PyType_GenericAlloc as *const c_void };
    }
    if unsafe { (*tp).tp_free.is_null() } {
        unsafe { (*tp).tp_free = PyObject_Free as *const c_void };
    }
    if unsafe { (*tp).tp_new.is_null() } {
        unsafe { (*tp).tp_new = PyType_GenericNew as *const c_void };
    }

    let w_base = if base.is_null() {
        crate::typedef::w_object()
    } else {
        interpreter_type(base)
    };
    if w_base.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyType_Ready(): tp_base is not a ready type",
        ));
    }
    let qualified = if unsafe { (*tp).tp_name.is_null() } {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr((*tp).tp_name) }
            .to_string_lossy()
            .into_owned()
    };
    if qualified.is_empty() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyType_Ready(): the type has no tp_name",
        ));
    }
    // The qualified name is handed on whole: `make_builtin_type_with_base`
    // publishes the leading component as the type's `__module__`, which is
    // where `tp_name`'s prefix is meant to end up.

    let roots = pyre_object::gc_roots::push_roots();
    let base_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_base);
    let w_type = crate::typedef::make_builtin_type_with_base(
        &qualified,
        |ns| install_namespace(ns, tp),
        pyre_object::gc_roots::shadow_stack_get(base_slot),
    );
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_type);
    unsafe {
        pyre_object::typeobject::w_type_set_hasdict(
            pyre_object::gc_roots::shadow_stack_get(type_slot),
            (*tp).tp_dictoffset != 0,
        )
    };
    stamp_objclass(pyre_object::gc_roots::shadow_stack_get(type_slot), tp);

    // The extension's own static becomes the type mirror: it is at a fixed
    // address for the life of the library, which is what `attach_foreign` is
    // for.
    unsafe {
        (*tp).ob_base.ob_base.ob_refcnt = REFCNT_IMMORTAL;
        pyobject::attach_foreign(
            pyre_object::gc_roots::shadow_stack_get(type_slot),
            &raw mut (*tp).ob_base.ob_base,
        );
        // Written rather than moved: whatever the static declared as its
        // metatype is discarded here, so there is no reference of this layer's
        // to release off it.
        let metatype = pyobject::type_mirror(pyre_object::gc_roots::shadow_stack_get(type_slot));
        (*tp).ob_base.ob_base.ob_type = metatype;
        pyobject::own_heap_type(metatype);
        (*tp).tp_flags = ((*tp).tp_flags & !PY_TPFLAGS_READYING) | PY_TPFLAGS_READY;
    }
    set_fast_subclass_flags(tp, pyre_object::gc_roots::shadow_stack_get(type_slot));
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_Ready(tp: *mut CPyTypeObject) -> c_int {
    match ready(tp) {
        Ok(()) => 0,
        Err(error) => {
            if !tp.is_null() {
                unsafe { (*tp).tp_flags &= !PY_TPFLAGS_READYING };
            }
            super::pyerrors::set_pending_error(error);
            -1
        }
    }
}

// ── heap types: `PyType_FromSpec` ───────────────────────────────────────

/// One `(slot id, function)` pair of a [`CPyTypeSpec`].
#[repr(C)]
pub struct CPyTypeSlot {
    pub slot: c_int,
    pub pfunc: *mut c_void,
}

/// `PyType_Spec`.
#[repr(C)]
pub struct CPyTypeSpec {
    pub name: *const c_char,
    pub basicsize: c_int,
    pub itemsize: c_int,
    pub flags: c_uint,
    pub slots: *mut CPyTypeSlot,
}

/// The `typeslots.h` identifiers.
mod slot_id {
    use std::ffi::c_int;

    /// Declare each identifier once, so the constant and the name the test
    /// looks up in `Python.h` cannot drift apart.
    macro_rules! slot_ids {
        ($($name:ident = $value:expr,)*) => {
            $(pub const $name: c_int = $value;)*

            #[cfg(test)]
            pub(super) const ALL: &[(&str, c_int)] = &[$((stringify!($name), $value),)*];
        };
    }

    slot_ids! {
        BF_GETBUFFER = 1,
        BF_RELEASEBUFFER = 2,
        MP_ASS_SUBSCRIPT = 3,
        MP_LENGTH = 4,
        MP_SUBSCRIPT = 5,
        NB_ABSOLUTE = 6,
        NB_ADD = 7,
        NB_AND = 8,
        NB_BOOL = 9,
        NB_DIVMOD = 10,
        NB_FLOAT = 11,
        NB_FLOOR_DIVIDE = 12,
        NB_INDEX = 13,
        NB_INPLACE_ADD = 14,
        NB_INPLACE_AND = 15,
        NB_INPLACE_FLOOR_DIVIDE = 16,
        NB_INPLACE_LSHIFT = 17,
        NB_INPLACE_MULTIPLY = 18,
        NB_INPLACE_OR = 19,
        NB_INPLACE_POWER = 20,
        NB_INPLACE_REMAINDER = 21,
        NB_INPLACE_RSHIFT = 22,
        NB_INPLACE_SUBTRACT = 23,
        NB_INPLACE_TRUE_DIVIDE = 24,
        NB_INPLACE_XOR = 25,
        NB_INT = 26,
        NB_INVERT = 27,
        NB_LSHIFT = 28,
        NB_MULTIPLY = 29,
        NB_NEGATIVE = 30,
        NB_OR = 31,
        NB_POSITIVE = 32,
        NB_POWER = 33,
        NB_REMAINDER = 34,
        NB_RSHIFT = 35,
        NB_SUBTRACT = 36,
        NB_TRUE_DIVIDE = 37,
        NB_XOR = 38,
        SQ_ASS_ITEM = 39,
        SQ_CONCAT = 40,
        SQ_CONTAINS = 41,
        SQ_INPLACE_CONCAT = 42,
        SQ_INPLACE_REPEAT = 43,
        SQ_ITEM = 44,
        SQ_LENGTH = 45,
        SQ_REPEAT = 46,
        TP_ALLOC = 47,
        TP_BASE = 48,
        TP_BASES = 49,
        TP_CALL = 50,
        TP_CLEAR = 51,
        TP_DEALLOC = 52,
        TP_DEL = 53,
        TP_DESCR_GET = 54,
        TP_DESCR_SET = 55,
        TP_DOC = 56,
        TP_GETATTR = 57,
        TP_GETATTRO = 58,
        TP_HASH = 59,
        TP_INIT = 60,
        TP_IS_GC = 61,
        TP_ITER = 62,
        TP_ITERNEXT = 63,
        TP_METHODS = 64,
        TP_NEW = 65,
        TP_REPR = 66,
        TP_RICHCOMPARE = 67,
        TP_SETATTR = 68,
        TP_SETATTRO = 69,
        TP_STR = 70,
        TP_TRAVERSE = 71,
        // `tp_members` and `tp_getset` break the alphabetical run: they were added
        // after `tp_traverse` and keep the numbers they were given then.
        TP_MEMBERS = 72,
        TP_GETSET = 73,
        TP_FREE = 74,
        NB_MATRIX_MULTIPLY = 75,
        NB_INPLACE_MATRIX_MULTIPLY = 76,
        AM_AWAIT = 77,
        AM_AITER = 78,
        AM_ANEXT = 79,
        TP_FINALIZE = 80,
        AM_SEND = 81,
        TP_VECTORCALL = 82,
        TP_TOKEN = 83,
    }
}

/// Allocate a zeroed sub-table on first use.  A heap type is immortal for the
/// same reason its instances are, so the leak is the intended lifetime.
macro_rules! table_of {
    ($tp:expr, $field:ident, $shape:ty) => {{
        unsafe {
            if (*$tp).$field.is_null() {
                (*$tp).$field = Box::leak(Box::new(std::mem::zeroed::<$shape>()));
            }
            (*$tp).$field
        }
    }};
}

fn apply_slot(tp: *mut CPyTypeObject, id: c_int, value: *mut c_void) -> Result<(), crate::PyError> {
    use slot_id::*;
    let function = value as *const c_void;
    macro_rules! number {
        ($field:ident) => {{
            let table = table_of!(tp, tp_as_number, CPyNumberMethods);
            unsafe { (*table).$field = function };
        }};
    }
    macro_rules! sequence {
        ($field:ident) => {{
            let table = table_of!(tp, tp_as_sequence, CPySequenceMethods);
            unsafe { (*table).$field = function };
        }};
    }
    macro_rules! mapping {
        ($field:ident) => {{
            let table = table_of!(tp, tp_as_mapping, CPyMappingMethods);
            unsafe { (*table).$field = function };
        }};
    }
    macro_rules! asynchronous {
        ($field:ident) => {{
            let table = table_of!(tp, tp_as_async, CPyAsyncMethods);
            unsafe { (*table).$field = function };
        }};
    }
    macro_rules! buffer {
        ($field:ident) => {{
            let table = table_of!(tp, tp_as_buffer, CPyBufferProcs);
            unsafe { (*table).$field = function };
        }};
    }
    macro_rules! own {
        ($field:ident, $shape:ty) => {
            unsafe { (*tp).$field = value as $shape }
        };
    }

    match id {
        BF_GETBUFFER => buffer!(bf_getbuffer),
        BF_RELEASEBUFFER => buffer!(bf_releasebuffer),
        MP_ASS_SUBSCRIPT => mapping!(mp_ass_subscript),
        MP_LENGTH => mapping!(mp_length),
        MP_SUBSCRIPT => mapping!(mp_subscript),
        NB_ABSOLUTE => number!(nb_absolute),
        NB_ADD => number!(nb_add),
        NB_AND => number!(nb_and),
        NB_BOOL => number!(nb_bool),
        NB_DIVMOD => number!(nb_divmod),
        NB_FLOAT => number!(nb_float),
        NB_FLOOR_DIVIDE => number!(nb_floor_divide),
        NB_INDEX => number!(nb_index),
        NB_INPLACE_ADD => number!(nb_inplace_add),
        NB_INPLACE_AND => number!(nb_inplace_and),
        NB_INPLACE_FLOOR_DIVIDE => number!(nb_inplace_floor_divide),
        NB_INPLACE_LSHIFT => number!(nb_inplace_lshift),
        NB_INPLACE_MULTIPLY => number!(nb_inplace_multiply),
        NB_INPLACE_OR => number!(nb_inplace_or),
        NB_INPLACE_POWER => number!(nb_inplace_power),
        NB_INPLACE_REMAINDER => number!(nb_inplace_remainder),
        NB_INPLACE_RSHIFT => number!(nb_inplace_rshift),
        NB_INPLACE_SUBTRACT => number!(nb_inplace_subtract),
        NB_INPLACE_TRUE_DIVIDE => number!(nb_inplace_true_divide),
        NB_INPLACE_XOR => number!(nb_inplace_xor),
        NB_INT => number!(nb_int),
        NB_INVERT => number!(nb_invert),
        NB_LSHIFT => number!(nb_lshift),
        NB_MULTIPLY => number!(nb_multiply),
        NB_NEGATIVE => number!(nb_negative),
        NB_OR => number!(nb_or),
        NB_POSITIVE => number!(nb_positive),
        NB_POWER => number!(nb_power),
        NB_REMAINDER => number!(nb_remainder),
        NB_RSHIFT => number!(nb_rshift),
        NB_SUBTRACT => number!(nb_subtract),
        NB_TRUE_DIVIDE => number!(nb_true_divide),
        NB_XOR => number!(nb_xor),
        NB_MATRIX_MULTIPLY => number!(nb_matrix_multiply),
        NB_INPLACE_MATRIX_MULTIPLY => number!(nb_inplace_matrix_multiply),
        SQ_ASS_ITEM => sequence!(sq_ass_item),
        SQ_CONCAT => sequence!(sq_concat),
        SQ_CONTAINS => sequence!(sq_contains),
        SQ_INPLACE_CONCAT => sequence!(sq_inplace_concat),
        SQ_INPLACE_REPEAT => sequence!(sq_inplace_repeat),
        SQ_ITEM => sequence!(sq_item),
        SQ_LENGTH => sequence!(sq_length),
        SQ_REPEAT => sequence!(sq_repeat),
        AM_AWAIT => asynchronous!(am_await),
        AM_AITER => asynchronous!(am_aiter),
        AM_ANEXT => asynchronous!(am_anext),
        AM_SEND => asynchronous!(am_send),
        TP_ALLOC => own!(tp_alloc, *const c_void),
        TP_BASE => own!(tp_base, *mut CPyTypeObject),
        TP_CALL => own!(tp_call, *const c_void),
        TP_CLEAR => own!(tp_clear, *const c_void),
        TP_DEALLOC => own!(tp_dealloc, *const c_void),
        TP_DEL => own!(tp_del, *const c_void),
        TP_DESCR_GET => own!(tp_descr_get, *const c_void),
        TP_DESCR_SET => own!(tp_descr_set, *const c_void),
        TP_DOC => own!(tp_doc, *const c_char),
        TP_GETATTR => own!(tp_getattr, *const c_void),
        TP_GETATTRO => own!(tp_getattro, *const c_void),
        TP_GETSET => own!(tp_getset, *mut CPyGetSetDef),
        TP_HASH => own!(tp_hash, *const c_void),
        TP_INIT => own!(tp_init, *const c_void),
        TP_IS_GC => own!(tp_is_gc, *const c_void),
        TP_ITER => own!(tp_iter, *const c_void),
        TP_ITERNEXT => own!(tp_iternext, *const c_void),
        TP_METHODS => own!(tp_methods, *mut super::methodobject::CPyMethodDef),
        TP_MEMBERS => own!(tp_members, *mut CPyMemberDef),
        TP_NEW => own!(tp_new, *const c_void),
        TP_REPR => own!(tp_repr, *const c_void),
        TP_RICHCOMPARE => own!(tp_richcompare, *const c_void),
        TP_SETATTR => own!(tp_setattr, *const c_void),
        TP_SETATTRO => own!(tp_setattro, *const c_void),
        TP_STR => own!(tp_str, *const c_void),
        TP_TRAVERSE => own!(tp_traverse, *const c_void),
        TP_FINALIZE => own!(tp_finalize, *const c_void),
        TP_FREE => own!(tp_free, *const c_void),
        // `Py_tp_bases` is a tuple, which the caller may also pass through
        // `PyType_FromSpecWithBases`; both land in `single_base`.
        TP_BASES => {
            let bases = unsafe { pyobject::from_ref(value as *mut CPyObject) };
            unsafe { (*tp).tp_base = single_base(bases)? };
        }
        other => {
            return Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("PyType_FromSpec() does not support slot {other}"),
            ));
        }
    }
    Ok(())
}

/// The one base pyre can build a type on.
///
/// `make_builtin_type_with_base` takes a single base, so a spec naming more
/// than one is rejected rather than silently losing the rest.
fn single_base(bases: PyObjectRef) -> Result<*mut CPyTypeObject, crate::PyError> {
    if bases.is_null() {
        return Ok(std::ptr::null_mut());
    }
    let items: Vec<PyObjectRef> = if unsafe { pyre_object::is_tuple(bases) } {
        unsafe { pyre_object::tupleobject::w_tuple_items_copy_as_vec(bases) }
    } else {
        vec![bases]
    };
    match items.len() {
        0 => Ok(std::ptr::null_mut()),
        1 => Ok(pyobject::make_ref(items[0]) as *mut CPyTypeObject),
        _ => Err(crate::PyError::type_error(
            "PyType_FromSpec() with more than one base is not supported yet",
        )),
    }
}

/// A type's `ht_module` and `ht_token`, which pyre has no heap-type struct to
/// carry.
///
/// Both are keyed by the type's own address: a type built from a spec is
/// leaked deliberately, so the key is stable for the life of the process.  The
/// module is held as an owned mirror reference, and it is that reference — not
/// the table — that roots it.
type TypeSideTable = std::collections::HashMap<
    usize,
    usize,
    std::hash::BuildHasherDefault<std::hash::DefaultHasher>,
>;
static TYPE_MODULES: super::ForkMutex<TypeSideTable> = super::ForkMutex::new(
    TypeSideTable::with_hasher(std::hash::BuildHasherDefault::new()),
);
static TYPE_TOKENS: super::ForkMutex<TypeSideTable> = super::ForkMutex::new(
    TypeSideTable::with_hasher(std::hash::BuildHasherDefault::new()),
);

pub(super) unsafe fn after_fork_child() {
    unsafe {
        TYPE_MODULES.reinit_after_fork();
        TYPE_TOKENS.reinit_after_fork();
        TYPE_NAMES.reinit_after_fork();
    }
}

/// The alignment a block of type data starts at, `_align_up`'s modulus: the
/// strictest a C compiler gives any object, so a field of any type declared in
/// the extra data is aligned wherever the base's fields end.
fn align_up(size: isize) -> isize {
    const ALIGNMENT: isize = super::pyobject::BLOCK_ALIGN as isize;
    (size + ALIGNMENT - 1) & !(ALIGNMENT - 1)
}

fn from_spec(
    spec: *mut CPyTypeSpec,
    bases: *mut CPyObject,
    module: *mut CPyObject,
) -> Result<PyObjectRef, crate::PyError> {
    if spec.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyType_FromSpec(): NULL spec",
        ));
    }
    // The type object and its tables outlive the call for good: a type is
    // immortal here, so the allocation is deliberately leaked.
    let tp: *mut CPyTypeObject = Box::leak(Box::new(immortal_type()));
    unsafe {
        (*tp).tp_name = (*spec).name;
        (*tp).tp_itemsize = (*spec).itemsize as isize;
        (*tp).tp_flags = (*spec).flags as std::ffi::c_ulong | PY_TPFLAGS_HEAPTYPE;
        (*tp).tp_base = single_base(pyobject::from_ref(bases))?;
    }
    let mut token: *mut c_void = std::ptr::null_mut();
    let mut index = 0isize;
    loop {
        let entry = unsafe { (*spec).slots.offset(index) };
        if unsafe { (*spec).slots.is_null() } || unsafe { (*entry).slot } == 0 {
            break;
        }
        // `Py_tp_token` names no slot: it declares the type's identity, and
        // the null it may carry is `Py_TP_USE_SPEC`, which asks for the spec's
        // own address rather than for no token at all.
        if unsafe { (*entry).slot } == slot_id::TP_TOKEN {
            token = match unsafe { (*entry).pfunc } {
                pfunc if pfunc.is_null() => spec as *mut c_void,
                pfunc => pfunc,
            };
        } else {
            apply_slot(tp, unsafe { (*entry).slot }, unsafe { (*entry).pfunc })?;
        }
        index += 1;
    }

    // The base's own size is what a relative or inherited `basicsize` is
    // measured against, so it has to be final before either is resolved.
    let base = unsafe { (*tp).tp_base };
    if !base.is_null() {
        ready(base)?;
    }
    let base_basicsize = if base.is_null() {
        size_of::<CPyObject>() as isize
    } else {
        unsafe { (*base).tp_basicsize }
    };
    let declared = unsafe { (*spec).basicsize } as isize;
    unsafe {
        (*tp).tp_basicsize = match declared {
            // Inherit: an extension that declares no storage of its own gets
            // the block its base needs, not the bare header.
            0 => base_basicsize,
            // Extend: the magnitude is the extra data appended after the base,
            // which is what `PyType_GetTypeDataSize` reports back.
            negative if negative < 0 => align_up(base_basicsize) + align_up(-negative),
            absolute => absolute,
        };
    }

    ready(tp)?;
    if !token.is_null() {
        TYPE_TOKENS.lock().insert(tp as usize, token as usize);
    }
    if !module.is_null() {
        // An owned reference, which is what roots the module for as long as
        // the type: the type itself is immortal, so nothing ever releases it.
        unsafe { pyobject::incref(module) };
        TYPE_MODULES.lock().insert(tp as usize, module as usize);
    }
    let w_type = interpreter_type(tp);
    Ok(w_type)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_FromSpec(spec: *mut CPyTypeSpec) -> *mut CPyObject {
    super::object::result(from_spec(spec, std::ptr::null_mut(), std::ptr::null_mut()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_FromSpecWithBases(
    spec: *mut CPyTypeSpec,
    bases: *mut CPyObject,
) -> *mut CPyObject {
    super::object::result(from_spec(spec, bases, std::ptr::null_mut()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_FromModuleAndSpec(
    module: *mut CPyObject,
    spec: *mut CPyTypeSpec,
    bases: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([module, bases]);
    super::object::result(from_spec(spec, bases, module))
}

/// `PyType_GetSlot` — read one slot back off a ready type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetSlot(tp: *mut CPyTypeObject, id: c_int) -> *mut c_void {
    use slot_id::*;
    if tp.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    macro_rules! number {
        ($field:ident) => {
            (number_slot!($field))(tp)
        };
    }
    macro_rules! sequence {
        ($field:ident) => {
            (sequence_slot!($field))(tp)
        };
    }
    macro_rules! mapping {
        ($field:ident) => {
            (mapping_slot!($field))(tp)
        };
    }
    let value: *const c_void = match id {
        MP_ASS_SUBSCRIPT => mapping!(mp_ass_subscript),
        MP_LENGTH => mapping!(mp_length),
        MP_SUBSCRIPT => mapping!(mp_subscript),
        NB_ABSOLUTE => number!(nb_absolute),
        NB_ADD => number!(nb_add),
        NB_AND => number!(nb_and),
        NB_BOOL => number!(nb_bool),
        NB_DIVMOD => number!(nb_divmod),
        NB_FLOAT => number!(nb_float),
        NB_FLOOR_DIVIDE => number!(nb_floor_divide),
        NB_INDEX => number!(nb_index),
        NB_INT => number!(nb_int),
        NB_INVERT => number!(nb_invert),
        NB_LSHIFT => number!(nb_lshift),
        NB_MULTIPLY => number!(nb_multiply),
        NB_NEGATIVE => number!(nb_negative),
        NB_OR => number!(nb_or),
        NB_POSITIVE => number!(nb_positive),
        NB_POWER => number!(nb_power),
        NB_REMAINDER => number!(nb_remainder),
        NB_RSHIFT => number!(nb_rshift),
        NB_SUBTRACT => number!(nb_subtract),
        NB_TRUE_DIVIDE => number!(nb_true_divide),
        NB_XOR => number!(nb_xor),
        SQ_ASS_ITEM => sequence!(sq_ass_item),
        SQ_CONCAT => sequence!(sq_concat),
        SQ_CONTAINS => sequence!(sq_contains),
        SQ_ITEM => sequence!(sq_item),
        SQ_LENGTH => sequence!(sq_length),
        SQ_REPEAT => sequence!(sq_repeat),
        TP_ALLOC => unsafe { (*tp).tp_alloc },
        TP_CALL => unsafe { (*tp).tp_call },
        TP_DEALLOC => unsafe { (*tp).tp_dealloc },
        TP_DESCR_GET => unsafe { (*tp).tp_descr_get },
        TP_DESCR_SET => unsafe { (*tp).tp_descr_set },
        TP_DOC => unsafe { (*tp).tp_doc as *const c_void },
        TP_GETATTRO => unsafe { (*tp).tp_getattro },
        TP_GETSET => unsafe { (*tp).tp_getset as *const c_void },
        TP_HASH => unsafe { (*tp).tp_hash },
        TP_INIT => unsafe { (*tp).tp_init },
        TP_ITER => unsafe { (*tp).tp_iter },
        TP_ITERNEXT => unsafe { (*tp).tp_iternext },
        TP_METHODS => unsafe { (*tp).tp_methods as *const c_void },
        TP_MEMBERS => unsafe { (*tp).tp_members as *const c_void },
        TP_NEW => unsafe { (*tp).tp_new },
        TP_REPR => unsafe { (*tp).tp_repr },
        TP_RICHCOMPARE => unsafe { (*tp).tp_richcompare },
        TP_SETATTRO => unsafe { (*tp).tp_setattro },
        TP_STR => unsafe { (*tp).tp_str },
        _ => std::ptr::null(),
    };
    value as *mut c_void
}

/// `_PyType_Name(type)` — the tail of `tp_name` after the last dot, which is
/// the name without the module qualifying it.
///
/// The answer points into `tp_name` itself, so it lives exactly as long as the
/// type object does and the caller frees nothing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyType_Name(tp: *mut CPyTypeObject) -> *const c_char {
    if tp.is_null() {
        return std::ptr::null();
    }
    let name = unsafe { (*tp).tp_name };
    if name.is_null() {
        return name;
    }
    let bytes = unsafe { CStr::from_ptr(name) }.to_bytes();
    match bytes.iter().rposition(|&byte| byte == b'.') {
        Some(dot) => unsafe { name.add(dot + 1) },
        None => name,
    }
}

/// `PyType_GetName` and `PyType_GetQualName` — `__name__`, which is the part
/// of `tp_name` after the last dot for a type whose name arrived qualified.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetName(tp: *mut CPyTypeObject) -> *mut CPyObject {
    let w_type = interpreter_type(tp);
    if w_type.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    pyobject::make_ref(unsafe { pyre_object::typeobject::w_type_get_name_obj(w_type) })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetQualName(tp: *mut CPyTypeObject) -> *mut CPyObject {
    unsafe { PyType_GetName(tp) }
}

/// `PyType_GenericAlloc` — build the interpreter object and its C block.
///
/// The block *is* the instance's storage, so it must live exactly as long as
/// the interpreter object.  Pyre's collector has no dead queue yet, so the
/// mirror is immortal instead: releasing it when the C side drops its last
/// reference would destroy fields the interpreter object still exposes.  That
/// makes an instance of a C-defined type unreclaimable, which is the same gap
/// the module header names for reference cycles through C.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GenericAlloc(
    tp: *mut CPyTypeObject,
    nitems: isize,
) -> *mut CPyObject {
    let w_type = interpreter_type(tp);
    if w_type.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyType_GenericAlloc(): the type is not ready",
        ));
        return std::ptr::null_mut();
    }
    let roots = pyre_object::gc_roots::push_roots();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_type);
    let instance = pyre_object::w_instance_new(pyre_object::gc_roots::shadow_stack_get(type_slot));
    let instance_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(instance);
    let size = unsafe { (*tp).tp_basicsize + nitems.max(0) * (*tp).tp_itemsize } as usize;
    // `tp_alloc` returns a new reference, so the count is the link share plus
    // the one this call hands out.  Not immortal: the link is a rawrefcount
    // P-link, so the instance dies when neither side holds it and the
    // collector queues this block for `tp_dealloc`.
    let raw = pyobject::attach(
        pyre_object::gc_roots::shadow_stack_get(instance_slot),
        REFCNT_FROM_PYRE + 1,
        tp,
        size,
    );
    if unsafe { (*tp).tp_itemsize } != 0 {
        unsafe { (*(raw as *mut CPyVarObject)).ob_size = nitems };
    }
    // An instance of a `Py_TPFLAGS_HAVE_GC` type is collected from the moment
    // `tp_alloc` hands it back; only a type that allocates its own storage has
    // to call `PyObject_GC_Track` by hand.
    super::gc::track(raw);
    raw
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GenericNew(
    tp: *mut CPyTypeObject,
    _args: *mut CPyObject,
    _kwds: *mut CPyObject,
) -> *mut CPyObject {
    unsafe { PyType_GenericAlloc(tp, 0) }
}

/// `cpyext/src/object.c:102-105 PyObject_Free` — the deallocator half of
/// [`PyType_GenericAlloc`], and the `tp_free` every type inherits.
///
/// A `tp_dealloc` written for CPython ends in `Py_TYPE(self)->tp_free(self)`,
/// so this has to exist and has to be reachable through the slot before any
/// `tp_dealloc` is dispatched.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Free(object: *mut std::ffi::c_void) {
    if object.is_null() {
        return;
    }
    unsafe { pyobject::free_block(object as *mut CPyObject) };
}

/// `PyObject_Del` — the older spelling of [`PyObject_Free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Del(object: *mut std::ffi::c_void) {
    unsafe { PyObject_Free(object) };
}

/// `PyObject_Init` — stamp a block's header and make it an object.
///
/// A block from [`PyType_GenericAlloc`] arrives linked, and only its `ob_type`
/// is (re)written.  A block from `PyObject_Malloc` arrives as raw bytes, which
/// is the other supported spelling of the same allocation, so the interpreter
/// object it stands for is created here and linked to it: the count it leaves
/// with is the link share plus the one reference the caller now owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Init(
    object: *mut CPyObject,
    tp: *mut CPyTypeObject,
) -> *mut CPyObject {
    if object.is_null() {
        return unsafe { super::pyerrors::PyErr_NoMemory() };
    }
    unsafe { pyobject::set_ob_type(object, tp) };
    if unsafe { !(*object).ob_pyre_link.is_null() } {
        return object;
    }
    let w_type = interpreter_type(tp);
    if w_type.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyObject_Init(): the type is not ready",
        ));
        return std::ptr::null_mut();
    }
    let roots = pyre_object::gc_roots::push_roots();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_type);
    let instance = pyre_object::w_instance_new(pyre_object::gc_roots::shadow_stack_get(type_slot));
    pyobject::link_allocated(instance, object, REFCNT_FROM_PYRE + 1);
    object
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::is_type(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_IsSubtype(
    subtype: *mut CPyTypeObject,
    supertype: *mut CPyTypeObject,
) -> c_int {
    let (sub, sup) = (interpreter_type(subtype), interpreter_type(supertype));
    if sub.is_null() || sup.is_null() {
        return 0;
    }
    crate::baseobjspace::issubclass(sub, sup).unwrap_or(false) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetFlags(tp: *mut CPyTypeObject) -> std::ffi::c_ulong {
    if tp.is_null() {
        return 0;
    }
    unsafe { (*tp).tp_flags }
}

// ── what a type reports about its module, its token and its data ────────

/// `tp_name` as it reads in a message, without pretending it is always there.
fn type_name_of(tp: *mut CPyTypeObject) -> String {
    if tp.is_null() || unsafe { (*tp).tp_name.is_null() } {
        return "<unnamed>".to_string();
    }
    unsafe { CStr::from_ptr((*tp).tp_name) }
        .to_string_lossy()
        .into_owned()
}

/// The module recorded for `tp` alone — no MRO walk, as `ht_module` is a field
/// of one heap type and is never inherited.
fn own_module(tp: *mut CPyTypeObject) -> *mut CPyObject {
    if tp.is_null() {
        return std::ptr::null_mut();
    }
    TYPE_MODULES
        .lock()
        .get(&(tp as usize))
        .map(|&address| address as *mut CPyObject)
        .unwrap_or(std::ptr::null_mut())
}

/// Borrowed: the type's own reference is what keeps the module alive, and the
/// type outlives every caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetModule(tp: *mut CPyTypeObject) -> *mut CPyObject {
    if tp.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    if unsafe { (*tp).tp_flags } & PY_TPFLAGS_HEAPTYPE == 0 {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "PyType_GetModule: Type '{}' is not a heap type",
            type_name_of(tp)
        )));
        return std::ptr::null_mut();
    }
    let module = own_module(tp);
    if module.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "PyType_GetModule: Type '{}' has no associated module",
            type_name_of(tp)
        )));
    }
    module
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetModuleState(tp: *mut CPyTypeObject) -> *mut c_void {
    let module = unsafe { PyType_GetModule(tp) };
    if module.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { super::modsupport::PyModule_GetState(module) }
}

/// The nearest module along `tp`'s MRO built from `def`.
///
/// Borrowed for the same reason [`PyType_GetModule`] is.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetModuleByDef(
    tp: *mut CPyTypeObject,
    def: *mut super::modsupport::CPyModuleDef,
) -> *mut CPyObject {
    if tp.is_null() || def.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    for base in c_bases(interpreter_type(tp)) {
        let module = own_module(base);
        if !module.is_null() && unsafe { super::modsupport::PyModule_GetDef(module) } == def {
            return module;
        }
    }
    super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
        "PyType_GetModuleByDef: No superclass of '{}' has the given module",
        type_name_of(tp)
    )));
    std::ptr::null_mut()
}

/// `PyType_GetBaseByToken` — 1 with `result` filled, 0 for no match, -1 on a
/// caller error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetBaseByToken(
    tp: *mut CPyTypeObject,
    token: *mut c_void,
    result: *mut *mut CPyTypeObject,
) -> c_int {
    if !result.is_null() {
        unsafe { *result = std::ptr::null_mut() };
    }
    if token.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyType_GetBaseByToken called with token=NULL",
        ));
        return -1;
    }
    if tp.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    // A static type has no heap-type superclass, so the walk cannot find one.
    if unsafe { (*tp).tp_flags } & PY_TPFLAGS_HEAPTYPE == 0 {
        return 0;
    }
    let wanted = token as usize;
    for base in c_bases(interpreter_type(tp)) {
        if TYPE_TOKENS.lock().get(&(base as usize)) != Some(&wanted) {
            continue;
        }
        if !result.is_null() {
            unsafe {
                pyobject::incref(&raw mut (*base).ob_base.ob_base);
                *result = base;
            }
        }
        return 1;
    }
    0
}

/// The extra storage a type declared beyond its base's, which is the
/// magnitude of a negative `PyType_Spec.basicsize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetTypeDataSize(tp: *mut CPyTypeObject) -> isize {
    if tp.is_null() {
        return 0;
    }
    let base = unsafe { (*tp).tp_base };
    let base_basicsize = if base.is_null() {
        size_of::<CPyObject>() as isize
    } else {
        unsafe { (*base).tp_basicsize }
    };
    let extra = unsafe { (*tp).tp_basicsize } - align_up(base_basicsize);
    extra.max(0)
}

/// The address of that storage inside one instance.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetTypeData(
    object: *mut CPyObject,
    tp: *mut CPyTypeObject,
) -> *mut c_void {
    if object.is_null() || tp.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let base = unsafe { (*tp).tp_base };
    let base_basicsize = if base.is_null() {
        size_of::<CPyObject>() as isize
    } else {
        unsafe { (*base).tp_basicsize }
    };
    unsafe { (object as *mut u8).offset(align_up(base_basicsize)) as *mut c_void }
}

/// The variable part of an instance of a type that put its items at the end.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetItemData(object: *mut CPyObject) -> *mut c_void {
    if object.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let tp = unsafe { (*object).ob_type };
    if tp.is_null() || unsafe { (*tp).tp_flags } & PY_TPFLAGS_ITEMS_AT_END == 0 {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "type '{}' does not have Py_TPFLAGS_ITEMS_AT_END",
            type_name_of(tp)
        )));
        return std::ptr::null_mut();
    }
    unsafe { (object as *mut u8).offset((*tp).tp_basicsize) as *mut c_void }
}

/// `typeobject.py:PyType_Modified` — drop what the interpreter cached about a
/// type whose C-side namespace has just been rewritten.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_Modified(tp: *mut CPyTypeObject) {
    let w_type = interpreter_type(tp);
    if w_type.is_null() {
        return;
    }
    // No key: the C caller says only that something changed.
    unsafe { crate::baseobjspace::mutated(w_type, None) };
}

/// `PyType_ClearCache` — empty the method cache and report the version the
/// entries it dropped were stamped with.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_ClearCache() -> c_uint {
    crate::baseobjspace::clear_method_cache();
    // Upstream's tag is a per-interpreter counter; pyre mints one identity per
    // type instead, so the only honest whole-cache answer is that no single
    // version described it.
    0
}

/// `PyType_Freeze` — refuse further changes to a type's namespace.
///
/// Immutability is `flag_heaptype = false` here, the state every builtin type
/// is already in, and it is what `type.__setattr__` consults.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_Freeze(tp: *mut CPyTypeObject) -> c_int {
    let w_type = interpreter_type(tp);
    if w_type.is_null() || !unsafe { pyre_object::is_type(w_type) } {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let mro = unsafe { pyre_object::w_type_get_mro(w_type) };
    if mro.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::type_error(
            "unable to get the type MRO",
        ));
        return -1;
    }
    // The type itself is the head of its own MRO and is the one being frozen,
    // so only what it inherits from has to be immutable already.
    for &w_base in unsafe { (*mro).as_slice() }.iter().skip(1) {
        if unsafe { pyre_object::w_type_is_heaptype(w_base) } {
            super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
                "Creating immutable type {} from mutable base {}",
                type_name_of(tp),
                unsafe { pyre_object::typeobject::w_type_get_name(w_base) }
            )));
            return -1;
        }
    }
    unsafe {
        pyre_object::typeobject::w_type_set_heaptype(w_type, false);
        if !tp.is_null() {
            (*tp).tp_flags |= PY_TPFLAGS_IMMUTABLETYPE;
        }
        crate::baseobjspace::mutated(w_type, None);
    }
    0
}

/// `__module__`, which is what a fully qualified name is built from.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetModuleName(tp: *mut CPyTypeObject) -> *mut CPyObject {
    let w_type = interpreter_type(tp);
    if w_type.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    super::object::result(crate::baseobjspace::getattr_str(w_type, "__module__"))
}

/// `module.qualname`, except for the two module names a name is never
/// qualified by.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetFullyQualifiedName(tp: *mut CPyTypeObject) -> *mut CPyObject {
    let w_type = interpreter_type(tp);
    if w_type.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let w_qualname = unsafe { pyre_object::typeobject::w_type_get_name_obj(w_type) };
    let qualified = unsafe { pyre_object::w_str_get_wtf8(w_qualname) }.to_string();
    // Naming the type object once more through its mirror, which never moves:
    // the name above may have been minted, and an allocation is the one thing
    // that can leave the pointer read before it stale.
    let module = crate::baseobjspace::getattr_str(interpreter_type(tp), "__module__")
        .ok()
        .filter(|&w_module| unsafe { pyre_object::is_str(w_module) })
        .map(|w_module| unsafe { pyre_object::w_str_get_wtf8(w_module) }.to_string());
    // A missing `__module__` is not the caller's problem: the bare name is
    // still the answer, so the lookup's own error goes no further.
    super::pyerrors::take_pending_error();
    let name = match module.as_deref() {
        Some("builtins") | Some("__main__") | None => qualified,
        Some(module) => format!("{module}.{qualified}"),
    };
    pyobject::make_ref(pyre_object::w_str_new(&name))
}

/// `PyType_FromMetaclass` — the general form of [`PyType_FromModuleAndSpec`].
///
/// The metaclass a type is built through is `type` itself here: pyre builds a
/// type through its own constructor, which has no place to put another one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_FromMetaclass(
    metaclass: *mut CPyTypeObject,
    module: *mut CPyObject,
    spec: *mut CPyTypeSpec,
    bases: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([module, bases]);
    if !metaclass.is_null() {
        let w_metaclass = interpreter_type(metaclass);
        if w_metaclass.is_null() || w_metaclass != crate::typedef::w_type() {
            super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
                "Metaclass '{}' is not supported; only 'type' is",
                type_name_of(metaclass)
            )));
            return std::ptr::null_mut();
        }
    }
    super::object::result(from_spec(spec, bases, module))
}

/// `PyErr_NewExceptionWithDoc` — `type(name, bases, {'__doc__': doc})`.
///
/// Built through the interpreter's own `type` rather than [`PyType_Ready`]:
/// an exception class needs the exception layout, which only the normal class
/// statement path installs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_NewExceptionWithDoc(
    name: *const c_char,
    doc: *const c_char,
    base: *mut CPyObject,
    dict: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([base, dict]);
    super::object::result(new_exception(name, doc, base, dict))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_NewException(
    name: *const c_char,
    base: *mut CPyObject,
    dict: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([base, dict]);
    super::object::result(new_exception(name, std::ptr::null(), base, dict))
}

fn new_exception(
    name: *const c_char,
    doc: *const c_char,
    base: *mut CPyObject,
    dict: *mut CPyObject,
) -> Result<PyObjectRef, crate::PyError> {
    if name.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyErr_NewException(): NULL name",
        ));
    }
    let qualified = unsafe { std::ffi::CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    let Some((module, short)) = qualified.rsplit_once('.') else {
        return Err(crate::PyError::system_error(
            "PyErr_NewException(): name must be module.class",
        ));
    };
    let w_base = unsafe { pyobject::from_ref(base) };
    let w_dict = unsafe { pyobject::from_ref(dict) };

    let roots = pyre_object::gc_roots::push_roots();
    let base_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(if w_base.is_null() {
        crate::builtins::lookup_exc_class("Exception").unwrap_or(pyre_object::PY_NULL)
    } else {
        w_base
    });
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(if w_dict.is_null() {
        pyre_object::dictmultiobject::w_dict_new()
    } else {
        w_dict
    });
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);

    let namespace = reload(dict_slot);
    store(namespace, "__module__", pyre_object::w_str_new(module));
    if !doc.is_null() {
        let text = text_or_none(doc);
        store(reload(dict_slot), DOC_KEY, text);
    }
    let bases = pyre_object::tupleobject::w_tuple_new(vec![reload(base_slot)]);
    let bases_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(bases);
    let w_name = pyre_object::w_str_new(short);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_name);
    let w_type = crate::typedef::gettypeobject(&pyre_object::TYPE_TYPE);
    crate::call::call_function_impl_result(
        w_type,
        &[reload(name_slot), reload(bases_slot), reload(dict_slot)],
    )
}

pub(super) fn ensure_linked() {
    std::hint::black_box(&raw const CPY_MODULE_DEF_TYPE);
    ensure_type_mirrors_linked();
    std::hint::black_box(PyObject_Free as *const ());
    std::hint::black_box(PyObject_Del as *const ());
    std::hint::black_box(PyType_Ready as *const ());
    std::hint::black_box(PyType_Check as *const ());
    std::hint::black_box(PyType_IsSubtype as *const ());
    std::hint::black_box(PyType_GetFlags as *const ());
    std::hint::black_box(_PyType_Name as *const ());
    std::hint::black_box(PyType_FromSpec as *const ());
    std::hint::black_box(PyType_FromSpecWithBases as *const ());
    std::hint::black_box(PyType_FromModuleAndSpec as *const ());
    std::hint::black_box(PyType_GetSlot as *const ());
    std::hint::black_box(PyType_GetName as *const ());
    std::hint::black_box(PyType_GetQualName as *const ());
    std::hint::black_box(PyType_GenericAlloc as *const ());
    std::hint::black_box(PyType_GenericNew as *const ());
    std::hint::black_box(PyObject_Init as *const ());
    std::hint::black_box(PyErr_NewException as *const ());
    std::hint::black_box(PyErr_NewExceptionWithDoc as *const ());
    std::hint::black_box(PyType_FromMetaclass as *const ());
    std::hint::black_box(PyType_GetModule as *const ());
    std::hint::black_box(PyType_GetModuleState as *const ());
    std::hint::black_box(PyType_GetModuleByDef as *const ());
    std::hint::black_box(PyType_GetModuleName as *const ());
    std::hint::black_box(PyType_GetFullyQualifiedName as *const ());
    std::hint::black_box(PyType_GetBaseByToken as *const ());
    std::hint::black_box(PyType_GetTypeDataSize as *const ());
    std::hint::black_box(PyObject_GetTypeData as *const ());
    std::hint::black_box(PyObject_GetItemData as *const ());
    std::hint::black_box(PyType_Modified as *const ());
    std::hint::black_box(PyType_ClearCache as *const ());
    std::hint::black_box(PyType_Freeze as *const ());
}

#[cfg(test)]
mod tests {
    /// `PyType_FromSpec` reads the numbers an extension compiled against
    /// `typeslots.h` wrote into its slot array, so the two tables are one ABI
    /// in two places.  This walks the header and rejects any identifier whose
    /// number the Rust side spells differently, or does not spell at all.
    #[test]
    fn every_slot_id_is_the_number_the_header_gives_it() {
        const HEADER: &str = include_str!("../../../../include/pyre3.14t/typeslots.h");

        let mut checked = 0;
        for line in HEADER.lines() {
            let Some(rest) = line.strip_prefix("#define Py_") else {
                continue;
            };
            let Some((name, value)) = rest.split_once(' ') else {
                continue;
            };
            // The identifiers are `Py_<table>_<slot>`; nothing else in the
            // header is named that way with a bare integer for a body.
            let Some((table, _)) = name.split_once('_') else {
                continue;
            };
            if !matches!(table, "tp" | "nb" | "sq" | "mp" | "am" | "bf") {
                continue;
            }
            let Ok(value) = value.trim().parse::<std::ffi::c_int>() else {
                continue;
            };

            let upper = name.to_ascii_uppercase();
            let found = super::slot_id::ALL.iter().find(|(n, _)| *n == upper);
            let Some((_, ours)) = found else {
                panic!("Python.h defines Py_{name} = {value}, slot_id has no {upper}");
            };
            assert_eq!(
                *ours, value,
                "Py_{name} is {value} in Python.h and {ours} in slot_id"
            );
            checked += 1;
        }

        assert_eq!(
            checked,
            super::slot_id::ALL.len(),
            "the header defines {checked} slot identifiers and slot_id has {}",
            super::slot_id::ALL.len()
        );
    }
}
