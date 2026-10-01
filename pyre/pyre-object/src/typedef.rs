//! `pypy/interpreter/typedef.py` descriptor payload parity port.
//!
//! PyPy stores `fget` / `fset` / `fdel` / `doc` / `reqcls` /
//! `use_closure` / `name` as instance fields on the GetSetProperty
//! object itself — `class GetSetProperty(W_Root): _immutable_fields_
//! = [...]` (typedef.py).  Pyre previously emulated this with
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

/// Host values accepted by TypeDef.rawdict. Keep Python host strings/None
/// distinct from W_Root values: StdObjSpace.wrap allocates the former but
/// calls spacebind on the latter. In particular a wrapped string is NOT the
/// host string TypeDef.__init__ accepts as its doc candidate.
#[derive(Clone)]
pub enum TypeDefValue {
    Text(String),
    None,
    Root(&'static std::cell::UnsafeCell<PyObjectRef>),
}

impl TypeDefValue {
    /// Own a prebuilt declaration reference, including while its rawdict is
    /// still being assembled. Slots are process-lifetime like the TypeDefs;
    /// their addresses are stable even when the ordered host dict grows.
    /// # Safety
    /// `value` must be a live W_Root object, rooted across this call.
    pub unsafe fn root(value: PyObjectRef) -> Self {
        assert!(!value.is_null(), "use TypeDefValue::None for host None");
        let slot = Box::leak(Box::new(std::cell::UnsafeCell::new(value)));
        TYPEDEF_VALUE_ROOTS.lock().push(slot.get() as usize);
        ensure_declaration_container();
        crate::gc_roots::mark_prebuilt_roots_dirty();
        Self::Root(slot)
    }
}

/// `pypy/interpreter/typedef.py TypeDef`, shared with `typeobject.py Layout`.
///
/// This is the existing runtime metadata, not a second type-definition
/// registry. Derived layouts retain this object's identity. It lives in the
/// object crate because Layout cannot depend on the interpreter crate;
/// pyre-interpreter's typedef module re-exports this same type.
///
/// The current bootstrap supplies the layout-relevant subset of TypeDef's
/// fields. Full `TypeDef.__init__` (bases/rawdict/gateway metadata) and
/// TypeCache.build must converge on this owner, not allocate a parallel key.
pub struct TypeDef {
    /// Present for declaration-driven definitions; the legacy layout-only
    /// producers have not supplied declaration names yet.
    pub name: Option<String>,
    pub bases: Vec<*const TypeDef>,
    /// Host declarations, not the Function-valued namespace produced by
    /// TypeCache.build. IndexMap preserves Python dict insertion order.
    pub rawdict: indexmap::IndexMap<String, TypeDefValue>,
    pub doc: Option<String>,
    /// TypeDef.__init__: `self.text_signature = _text_signature_`. Applied to
    /// the W_TypeObject by TypeCache.build from the override declaration
    /// (`setup_builtin_type`).
    pub text_signature: Option<String>,
    pub weakrefable: bool,
    /// Existing allocation-vtable representation of the interpreter class;
    /// replace with the canonical class metadata when rpy_cls is connected.
    pub instance_type: *const PyType,
    acceptable_as_base_class: std::sync::atomic::AtomicBool,
    pub hasdict: bool,
    /// TypeDef.__init__: initialization-time declaration, not a flag inferred
    /// from the generated W_TypeObject's name or namespace.
    pub method_descriptor: bool,
    /// typeobject.py `typedef.flag_sequence_bug_compat`. Copied onto the
    /// W_TypeObject by TypeCache.build; not inherited by subclasses.
    pub flag_sequence_bug_compat: bool,
    /// TypeDef.__init__: `self.heaptype = False`. TypeCache.build passes
    /// `is_heaptype=overridetypedef.heaptype` into W_TypeObject.__init__.
    pub heaptype: bool,
    /// TypeDef.__init__: `self.applevel_subclasses_base = None`. TypeCache.build
    /// uses `applevel_subclasses_base.typedef` as `overridetypedef` so the
    /// derived declaration reuses that base's Layout.
    pub applevel_subclasses_base: *const TypeDef,
    /// TypeDef.__init__ `__buffer`.  `None` takes the first base that
    /// declared one.  cpyext `make_bf_getbuffer` reads only presence;
    /// the `'read'` / `'read-write'` vocabulary is never consulted.
    pub buffer: Option<TypeDefBuffer>,
}

/// TypeDef.__init__ `__buffer` — `'read'` or `'read-write'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeDefBuffer {
    Read,
    ReadWrite,
}

impl TypeDef {
    /// Construct the metadata subset currently supplied by type bootstrap.
    /// This is not yet the full initialization-time TypeDef.__init__ API.
    pub fn new(
        instance_type: *const PyType,
        acceptable_as_base_class: bool,
        hasdict: bool,
    ) -> Self {
        Self {
            name: None,
            bases: Vec::new(),
            rawdict: indexmap::IndexMap::new(),
            doc: None,
            text_signature: None,
            weakrefable: false,
            instance_type,
            acceptable_as_base_class: std::sync::atomic::AtomicBool::new(acceptable_as_base_class),
            hasdict,
            method_descriptor: false,
            flag_sequence_bug_compat: false,
            heaptype: false,
            applevel_subclasses_base: std::ptr::null(),
            buffer: None,
        }
    }

    #[inline]
    pub fn acceptable_as_base_class(&self) -> bool {
        self.acceptable_as_base_class
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[inline]
    pub fn set_acceptable_as_base_class(&self, value: bool) {
        self.acceptable_as_base_class
            .store(value, std::sync::atomic::Ordering::Relaxed);
    }

    /// TypeDef._freeze_: track individual prebuilt definitions as PBCs.
    pub fn _freeze_(&self) -> bool {
        true
    }

    /// TypeDef.__init__'s declaration-derived metadata. The host dictionary
    /// must contain unbound gateways/properties, not materialized Functions.
    ///
    /// # Safety
    /// Every Root value must be a live W_Root and every base a live prebuilt
    /// TypeDef. Definitions and their native root registrations are immortal.
    #[majit_macros::not_rpython]
    pub unsafe fn from_rawdict(
        name: &str,
        bases: Vec<*const TypeDef>,
        rawdict: indexmap::IndexMap<String, TypeDefValue>,
        instance_type: *const PyType,
    ) -> *const Self {
        let mut definition = Self::new(
            instance_type,
            rawdict.contains_key("__new__"),
            rawdict.contains_key("__dict__"),
        );
        definition.name = Some(name.to_string());
        assert!(
            !rawdict.contains_key("__del__"),
            "TypeDef requires an RPython finalizer"
        );
        definition.weakrefable = rawdict.contains_key("__weakref__");
        definition.doc = match rawdict.get("__doc__") {
            Some(TypeDefValue::Text(doc)) => Some(doc.clone()),
            _ => None,
        };
        for &base in &bases {
            // SAFETY: the constructor contract requires live prebuilt bases.
            unsafe {
                definition.hasdict |= (*base).hasdict;
                definition.weakrefable |= (*base).weakrefable;
                if definition.buffer.is_none() {
                    definition.buffer = (*base).buffer;
                }
            }
        }
        definition.bases = bases;
        unsafe { definition.add_entries(rawdict) };
        crate::lltype::malloc_raw(definition)
    }

    /// TypeDef.add_entries: name descriptors before updating the rawdict.
    /// # Safety
    /// Root entries must contain valid W_Root objects; mutate declarations
    /// only during host initialization, before publishing them to a cache.
    pub unsafe fn add_entries(&mut self, entries: indexmap::IndexMap<String, TypeDefValue>) {
        for (key, entry) in &entries {
            let TypeDefValue::Root(slot) = entry else {
                continue;
            };
            let value = unsafe { *slot.get() };
            if unsafe { crate::gateway::is_interp2app(value) } {
                let gateway = unsafe { &mut *(value as *mut crate::gateway::interp2app) };
                gateway.name = Box::leak(key.clone().into_boxed_str());
                gateway._is_type_method = true;
            } else if unsafe { is_getset_property(value) } {
                let w_name = crate::w_str_new(key);
                let value = unsafe { *slot.get() };
                unsafe { w_getset_set_name(value, w_name) };
            }
        }
        self.rawdict.extend(entries);
    }
}

// `typedef.py` `TypeDef.rawdict`. The host GC owns one dictionary. The
// census is that object. Each value is a referent of its trace, not its
// own extra root. `u32::MAX` until `init_gc_subsystem` publishes the tid.
static DECLARATION_CONTAINER_TID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(u32::MAX);
static DECLARATION_CONTAINER: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static TYPEDEF_VALUE_ROOTS: parking_lot::Mutex<Vec<usize>> = parking_lot::Mutex::new(Vec::new());
/// `(base, offset)` interior GC fields of declarations that have no GC header.
static DECLARATION_INTERIORS: parking_lot::Mutex<Vec<(usize, usize)>> =
    parking_lot::Mutex::new(Vec::new());

/// Visits interior GC fields of one immortal declaration. Does not receive
/// the declaration pointer itself.
pub type DeclarationInteriorWalk = unsafe fn(PyObjectRef, &mut dyn FnMut(&mut majit_ir::GcRef));

static DECLARATION_INTERIOR_WALK: std::sync::atomic::AtomicPtr<()> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

pub fn set_declaration_container_gc_type_id(tid: u32) {
    DECLARATION_CONTAINER_TID.store(tid, std::sync::atomic::Ordering::Release);
    ensure_declaration_container();
}

pub fn set_declaration_interior_walk(walk: DeclarationInteriorWalk) {
    DECLARATION_INTERIOR_WALK.store(walk as *mut (), std::sync::atomic::Ordering::Release);
}

fn declaration_interior_walk() -> Option<DeclarationInteriorWalk> {
    let ptr = DECLARATION_INTERIOR_WALK.load(std::sync::atomic::Ordering::Acquire);
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { std::mem::transmute::<*mut (), DeclarationInteriorWalk>(ptr) })
    }
}

pub fn publish_declaration_container() {
    ensure_declaration_container();
    // Slots published while the container was still null called
    // `remember_declaration_container` and it no-op'd. The alloc that
    // lands the container has to enter the remembered set or the next
    // minor drags the old object out and never runs its custom trace.
    remember_declaration_container();
}

fn ensure_declaration_container() {
    if DECLARATION_CONTAINER.load(std::sync::atomic::Ordering::Acquire) != 0 {
        return;
    }
    let tid = DECLARATION_CONTAINER_TID.load(std::sync::atomic::Ordering::Acquire);
    if tid == u32::MAX {
        return;
    }
    let obj = crate::gc_hook::try_gc_alloc_stable_raw(tid, std::mem::size_of::<usize>());
    if obj.is_null() {
        return;
    }
    if DECLARATION_CONTAINER
        .compare_exchange(
            0,
            obj as usize,
            std::sync::atomic::Ordering::Release,
            std::sync::atomic::Ordering::Acquire,
        )
        .is_ok()
    {
        remember_declaration_container();
    }
}

/// Remember the declaration dictionary after a prebuilt store so the next
/// minor traces it (`incminimark.py` `remember_young_pointer`).
pub fn remember_declaration_container() {
    let addr = DECLARATION_CONTAINER.load(std::sync::atomic::Ordering::Acquire);
    if addr != 0 {
        crate::gc_hook::try_gc_write_barrier(addr as *mut u8);
    }
}

/// The root census for `TypeDef.rawdict`: the container, once.
/// The visitor must not allocate.
pub fn walk_typedef_roots(forward: &mut dyn FnMut(&mut PyObjectRef)) {
    let addr = DECLARATION_CONTAINER.load(std::sync::atomic::Ordering::Acquire);
    if addr == 0 {
        return;
    }
    let mut ptr = addr as PyObjectRef;
    forward(&mut ptr);
    if ptr as usize != addr {
        DECLARATION_CONTAINER.store(ptr as usize, std::sync::atomic::Ordering::Release);
    }
}

/// Visit one GC pointer stored at `base + offset`. Does not read a header at `base`.
pub fn trace_immortal_interior_offsets(
    base: *mut u8,
    offsets: &[usize],
    visitor: &mut dyn FnMut(&mut PyObjectRef),
) {
    for &offset in offsets {
        let slot = unsafe { &mut *(base.add(offset) as *mut PyObjectRef) };
        if !slot.is_null() {
            visitor(slot);
        }
    }
}

/// Trace one declaration dictionary.
///
/// A GC-owned value is the slot itself. A `malloc_typed` value is not passed
/// to `visitor`; `interiors` names its fields.
pub fn trace_declaration_slots(
    slots: &[*mut PyObjectRef],
    owns: impl Fn(PyObjectRef) -> bool,
    mut interiors: impl FnMut(PyObjectRef, &mut dyn FnMut(&mut PyObjectRef)),
    visitor: &mut dyn FnMut(&mut PyObjectRef),
) {
    for &slot_ptr in slots {
        if slot_ptr.is_null() {
            continue;
        }
        let slot = unsafe { &mut *slot_ptr };
        if slot.is_null() {
            continue;
        }
        if owns(*slot) {
            visitor(slot);
        } else {
            interiors(*slot, visitor);
        }
    }
}

/// Record an interior GC field of a declaration that has no GC header.
pub fn register_declaration_interior(base: *mut u8, offset: usize) {
    DECLARATION_INTERIORS.lock().push((base as usize, offset));
    ensure_declaration_container();
    crate::gc_roots::mark_prebuilt_roots_dirty();
}

/// `gctypelayout` custom trace for the declaration dictionary.
pub unsafe fn declaration_container_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    let _ = obj_addr;
    let slots: Vec<usize> = TYPEDEF_VALUE_ROOTS.lock().clone();
    let interiors: Vec<(usize, usize)> = DECLARATION_INTERIORS.lock().clone();
    let slot_ptrs: Vec<*mut PyObjectRef> = slots
        .into_iter()
        .map(|addr| addr as *mut PyObjectRef)
        .collect();
    let mut visit = |slot: &mut PyObjectRef| {
        f(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
    };
    let walk = declaration_interior_walk();
    trace_declaration_slots(
        &slot_ptrs,
        |value| crate::gc_hook::try_gc_owns_object(value as *mut u8),
        |value, visitor| {
            if let Some(walk) = walk {
                let mut as_gcref = |slot: &mut majit_ir::GcRef| {
                    visitor(unsafe { &mut *(slot as *mut majit_ir::GcRef as *mut PyObjectRef) });
                };
                unsafe { walk(value, &mut as_gcref) };
            }
        },
        &mut visit,
    );
    for (base, offset) in interiors {
        trace_immortal_interior_offsets(base as *mut u8, &[offset], &mut visit);
    }
}

/// Existing process-lifetime allocation for module-level `W_X.typedef`
/// metadata. Keep the allocation identity while its bootstrap owner migrates
/// to TypeDef.__init__ / TypeCache.build.
pub fn leak_typedef(
    instance_type: *const PyType,
    acceptable_as_base_class: bool,
    hasdict: bool,
) -> *const TypeDef {
    crate::lltype::malloc_raw(TypeDef::new(
        instance_type,
        acceptable_as_base_class,
        hasdict,
    ))
}

/// `pypy/interpreter/typedef.py class GetSetProperty(W_Root)`.
///
/// All `PyObjectRef`-shaped slots default to `PY_NULL` to mark
/// "absent" (PyPy uses `None`); `use_closure` is a `bool` mirroring
/// the eponymous PyPy field.
///
/// `pytype_static = "GETSET_DESCRIPTOR_TYPE"` keeps the PyType under
/// its existing public name (`typedef.py GetSetProperty.typedef =
/// TypeDef("getset_descriptor", ...)`) while the GC consts stay on
/// the `W_GETSET_PROPERTY_*` convention.
#[pyre_class(
    "getset_descriptor",
    type_id = 40,
    static_name = "GETSET_PROPERTY",
    pytype_static = "GETSET_DESCRIPTOR_TYPE"
)]
pub struct GetSetProperty {
    /// `typedef.py self.fget` — getter callable.
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
    /// `typedef.py w_objclass = None` class default + per-instance
    /// override stamped by `copy_for_type` (typedef.py).  Read by
    /// `descr_get_objclass` (typedef.py) before falling back
    /// to `space.gettypeobject(self.reqcls.typedef)`.
    pub w_objclass: PyObjectRef,
    /// `typedef.py self.w_qualname = None` — lazy cache for
    /// `descr_get_qualname` (typedef.py); first reader stamps
    /// `"<class>.<name>"` (or `"?.<name>"` when `reqcls is None`).
    pub w_qualname: PyObjectRef,
    /// `typedef.py:345 self.use_closure` — passes `(self, space, obj)`
    /// vs `(space, obj)` to the wrapped callbacks.
    pub use_closure: bool,
}

/// Allocate a `GetSetProperty` bound to `GETSET_DESCRIPTOR_TYPE`.
/// Mirrors `typedef.py _init` — every slot is set in one shot
/// so the descriptor is fully initialised before the first reader.
///
/// `name` may be `PY_NULL`, in which case the caller is responsible
/// for substituting `'<generic property>'` (matching `typedef.py
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

/// `typedef.py add_entries` parity — overwrite the descriptor's
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

/// `typedef.py self.reqcls = cls` — write the required-receiver
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

/// `typedef.py / 348-356 copy_for_type` writes `new.w_objclass`.
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

/// `typedef.py self.w_qualname = None` — lazy cache slot for
/// `descr_get_qualname` (typedef.py).
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

/// `pypy/interpreter/typedef.py Member` — slot descriptor
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
/// (`typedef.py`, `:538-539`); the descriptor kind is the 3.14
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
/// CPython 3.14 `tb_memberlist.tb_frame`, backed by PyPy's
/// `interp_attrproperty_w('frame')`.  Read-only on both sides.
pub const MEMBER_TRACEBACK_FRAME: u32 = MEMBER_DIRECT_FLAG | 62;
/// CPython 3.14 `tb_memberlist.tb_lasti`, backed by PyPy's `lasti` field.
/// `Py_READONLY` there, writable through `descr_set_tb_lasti` in PyPy; the
/// setter is not wired here, for the reason `init_pytraceback_type` records.
pub const MEMBER_TRACEBACK_LASTI: u32 = MEMBER_DIRECT_FLAG | 63;
/// `instancemethod.__func__`, the callable stored by `InstanceMethod`.
pub const MEMBER_INSTANCEMETHOD_FUNCTION: u32 = MEMBER_DIRECT_FLAG | 64;

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
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_cls);
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

/// `typedef.py Member.index` — the slot index (`base_nslots + position`),
/// used by the LOAD_ATTR/STORE_ATTR cache to form the `SLOTS_STARTING_FROM +
/// index` attrkind (mapdict.py).
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
    #[test]
    fn declaration_metadata_distinguishes_host_text_and_wrapped_values() {
        use super::*;
        use indexmap::IndexMap;

        unsafe {
            let base = TypeDef::from_rawdict(
                "base",
                vec![],
                IndexMap::from([
                    ("__dict__".into(), TypeDefValue::None),
                    ("__weakref__".into(), TypeDefValue::None),
                    (
                        "__doc__".into(),
                        TypeDefValue::Text("host documentation".into()),
                    ),
                ]),
                &INSTANCE_TYPE,
            );
            assert_eq!((*base).doc.as_deref(), Some("host documentation"));
            assert!(!(*base).acceptable_as_base_class());
            let child = TypeDef::from_rawdict(
                "child",
                vec![base],
                IndexMap::from([
                    ("__new__".into(), TypeDefValue::None),
                    (
                        "__doc__".into(),
                        TypeDefValue::root(crate::w_str_new("wrapped")),
                    ),
                ]),
                &INSTANCE_TYPE,
            );
            assert_eq!((*child).name.as_deref(), Some("child"));
            assert_eq!((*child).bases, vec![base]);
            assert!((*child).hasdict && (*child).weakrefable);
            assert!((*child).acceptable_as_base_class());
            assert_eq!((*child).doc, None);
            assert_eq!(
                (*child)
                    .rawdict
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                vec!["__new__", "__doc__"]
            );
            // add_entries updates declarations; it does not recompute the
            // metadata derived during TypeDef.__init__.
            let child = &mut *(child as *mut TypeDef);
            child.add_entries(IndexMap::from([(
                "__doc__".into(),
                TypeDefValue::Text("later".into()),
            )]));
            assert_eq!(child.doc, None);
        }
    }

    #[test]
    fn declaration_trace_visits_a_gc_value_and_not_an_immortal_header() {
        use super::trace_declaration_slots;

        let mut gc_slot = 0x1100usize as super::PyObjectRef;
        let mut immortal = 0x2200usize as super::PyObjectRef;
        let mut interior = 0x3300usize as super::PyObjectRef;
        let slots = [
            &mut gc_slot as *mut super::PyObjectRef,
            &mut immortal as *mut super::PyObjectRef,
        ];
        let mut seen = Vec::new();
        trace_declaration_slots(
            &slots,
            |value| value as usize == 0x1100,
            |value, visitor| {
                assert_eq!(value as usize, 0x2200);
                visitor(&mut interior);
            },
            &mut |slot| seen.push(*slot as usize),
        );
        assert_eq!(seen, vec![0x1100, 0x3300]);
    }

    fn decl_test_gc() -> majit_gc::collector::MiniMarkGC {
        use majit_gc::collector::{GcConfig, MiniMarkGC};
        MiniMarkGC::with_config(GcConfig {
            nursery_size: 4096,
            large_object_threshold: 2048,
            ..GcConfig::default()
        })
    }

    struct DeclProbe {
        previous_container: usize,
        previous_tid: u32,
        roots_len: usize,
        gc: *mut majit_gc::collector::MiniMarkGC,
    }

    impl Drop for DeclProbe {
        fn drop(&mut self) {
            super::DECLARATION_CONTAINER.store(
                self.previous_container,
                std::sync::atomic::Ordering::Release,
            );
            super::DECLARATION_CONTAINER_TID
                .store(self.previous_tid, std::sync::atomic::Ordering::Release);
            super::TYPEDEF_VALUE_ROOTS.lock().truncate(self.roots_len);
            DECL_WALK_ACTIVE.store(false, std::sync::atomic::Ordering::Release);
            DECL_TEST_GC.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Release);
            DECL_OWNED.with(|slots| slots.borrow_mut().clear());
            crate::gc_hook::clear_gc_owns_object_hook();
            crate::gc_hook::clear_gc_write_barrier_hook();
            let _ = self.gc;
        }
    }

    static DECL_TEST_GC: std::sync::atomic::AtomicPtr<majit_gc::collector::MiniMarkGC> =
        std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
    static DECL_WALK_ACTIVE: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    thread_local! {
        static DECL_OWNED: std::cell::RefCell<Vec<usize>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    fn decl_test_stable_alloc(type_id: u32, payload_size: usize) -> *mut u8 {
        use majit_gc::GcAllocator;
        let gc = DECL_TEST_GC.load(std::sync::atomic::Ordering::Acquire);
        if gc.is_null() {
            return std::ptr::null_mut();
        }
        unsafe { (*gc).alloc_oldgen_typed(type_id, payload_size).0 as *mut u8 }
    }

    fn decl_test_barrier(obj: *mut u8) {
        use majit_gc::GcAllocator;
        if !crate::gc_hook::hook_test_effects_visible() || obj.is_null() {
            return;
        }
        let gc = DECL_TEST_GC.load(std::sync::atomic::Ordering::Acquire);
        if gc.is_null() {
            return;
        }
        unsafe { (*gc).write_barrier(majit_ir::GcRef(obj as usize)) };
    }

    fn decl_test_owns(addr: usize) -> bool {
        if !crate::gc_hook::hook_test_effects_visible() {
            return false;
        }
        DECL_OWNED.with(|slots| slots.borrow().contains(&addr))
    }

    fn decl_test_extra_root(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        if !DECL_WALK_ACTIVE.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        let mut forward = |slot: &mut super::PyObjectRef| {
            visitor(unsafe { &mut *(slot as *mut super::PyObjectRef as *mut majit_ir::GcRef) });
        };
        super::walk_typedef_roots(&mut forward);
    }

    fn install_decl_probe(
        gc: &mut majit_gc::collector::MiniMarkGC,
        container_tid: u32,
    ) -> DeclProbe {
        use std::sync::atomic::Ordering;
        majit_gc::shadow_stack::register_extra_root_walker(
            decl_test_extra_root,
            "declaration_container",
        );
        let probe = DeclProbe {
            previous_container: super::DECLARATION_CONTAINER.swap(0, Ordering::AcqRel),
            previous_tid: super::DECLARATION_CONTAINER_TID.swap(container_tid, Ordering::AcqRel),
            roots_len: super::TYPEDEF_VALUE_ROOTS.lock().len(),
            gc: gc as *mut _,
        };
        DECL_TEST_GC.store(gc, Ordering::Release);
        crate::gc_hook::register_gc_owns_object_hook(decl_test_owns);
        probe
    }

    /// Root walk names the container once. The young declaration value is not
    /// a root. Without the container's remembered-set trace, a minor leaves
    /// that slot on the nursery address.
    #[test]
    fn declaration_root_walk_names_the_container_once() {
        use majit_gc::GcAllocator;
        use majit_gc::trace::TypeInfo;
        use std::sync::atomic::Ordering;

        let _hook_lock = crate::gc_hook::hook_test_guard();
        let mut gc = decl_test_gc();
        let young_tid = gc.register_type(TypeInfo::simple(16));
        let container_tid = gc.register_type(TypeInfo::with_custom_trace(
            std::mem::size_of::<usize>(),
            super::declaration_container_custom_trace,
        ));
        let _probe = install_decl_probe(&mut gc, container_tid);
        let young = gc.alloc_nursery_typed(young_tid, 16);
        let young_before = young.0;
        DECL_OWNED.with(|slots| slots.borrow_mut().push(young_before));
        let _rooted = unsafe { super::TypeDefValue::root(young_before as super::PyObjectRef) };
        let slot = *super::TYPEDEF_VALUE_ROOTS.lock().last().unwrap() as *mut super::PyObjectRef;
        let container = gc.alloc_oldgen_typed(container_tid, std::mem::size_of::<usize>());
        super::DECLARATION_CONTAINER.store(container.0, Ordering::Release);
        assert_eq!(gc.old_objects_pointing_to_young_len(), 0);

        let mut roots = Vec::new();
        super::walk_typedef_roots(&mut |slot| roots.push(*slot as usize));
        assert_eq!(roots, vec![container.0]);
        assert!(!roots.contains(&young_before));

        let mut traced = Vec::new();
        unsafe {
            super::declaration_container_custom_trace(container.0, &mut |gcref| {
                traced.push((*gcref).0);
            });
        }
        assert!(traced.contains(&young_before));
        assert!(!roots.contains(&young_before));

        DECL_WALK_ACTIVE.store(true, Ordering::Release);
        gc.do_collect_nursery();
        assert_eq!(
            unsafe { *slot as usize },
            young_before,
            "dropping the container trace left the young slot unforwarded"
        );
    }

    /// A young declaration stored before the container exists is forwarded
    /// once `publish_declaration_container` allocates and write-barriers it.
    #[test]
    fn declaration_interior_survives_a_minor_collection() {
        use majit_gc::GcAllocator;
        use majit_gc::trace::TypeInfo;
        use std::sync::atomic::Ordering;

        let _hook_lock = crate::gc_hook::hook_test_guard();
        let mut gc = decl_test_gc();
        let young_tid = gc.register_type(TypeInfo::simple(16));
        let container_tid = gc.register_type(TypeInfo::with_custom_trace(
            std::mem::size_of::<usize>(),
            super::declaration_container_custom_trace,
        ));
        let _probe = install_decl_probe(&mut gc, container_tid);
        let young = gc.alloc_nursery_typed(young_tid, 16);
        let young_before = young.0;
        DECL_OWNED.with(|slots| slots.borrow_mut().push(young_before));
        let _rooted = unsafe { super::TypeDefValue::root(young_before as super::PyObjectRef) };
        let slot = *super::TYPEDEF_VALUE_ROOTS.lock().last().unwrap() as *mut super::PyObjectRef;
        assert_eq!(super::DECLARATION_CONTAINER.load(Ordering::Acquire), 0);

        crate::gc_hook::register_gc_alloc_stable_hook(decl_test_stable_alloc);
        crate::gc_hook::register_gc_write_barrier_hook(decl_test_barrier);
        super::publish_declaration_container();
        let container = super::DECLARATION_CONTAINER.load(Ordering::Acquire);
        assert_ne!(container, 0);
        assert!(gc.old_objects_pointing_to_young_len() >= 1);

        let mut roots = Vec::new();
        super::walk_typedef_roots(&mut |slot| roots.push(*slot as usize));
        assert_eq!(roots, vec![container]);
        assert!(!roots.contains(&young_before));

        DECL_WALK_ACTIVE.store(true, Ordering::Release);
        gc.do_collect_nursery();
        let forwarded = unsafe { *slot as usize };
        assert_ne!(forwarded, young_before);
        assert!(gc.is_managed_heap_object(forwarded));
    }

    #[test]
    fn method_descriptor_is_projected_from_the_selected_layout_owner() {
        use super::*;
        use crate::typeobject::*;

        let mut definition = TypeDef::new(&INSTANCE_TYPE, false, false);
        assert!(!definition.method_descriptor);
        definition.method_descriptor = true;
        let definition = crate::lltype::malloc_raw(definition);
        let layout = leak_layout(Layout {
            typedef: definition,
            nslots: 0,
            newslotnames: vec![],
            base_layout: std::ptr::null(),
            dict_data_slot: DICT_DATA_SLOT_UNRESOLVED,
        });
        let plain = leak_layout(Layout {
            typedef: leak_typedef(&INSTANCE_TYPE, true, false),
            nslots: 0,
            newslotnames: vec![],
            base_layout: std::ptr::null(),
            dict_data_slot: DICT_DATA_SLOT_UNRESOLVED,
        });
        unsafe {
            // Neither spelling nor the initial flag selects the behavior.
            let w_type = w_type_new("not_a_descriptor_name", PY_NULL, std::ptr::null_mut());
            assert!(!w_type_get_flag_method_descriptor(w_type));
            w_type_set_layout(w_type, layout);
            assert!(w_type_get_flag_method_descriptor(w_type));
            w_type_set_layout(w_type, plain);
            assert!(!w_type_get_flag_method_descriptor(w_type));
        }
    }

    #[test]
    fn typedef_identity_is_preserved_by_layout_and_freeze() {
        use super::{TypeDef, leak_typedef};
        use crate::typeobject::{DICT_DATA_SLOT_UNRESOLVED, Layout, leak_layout};

        let definition: *const TypeDef = leak_typedef(&crate::pyobject::INSTANCE_TYPE, true, true);
        let root = leak_layout(Layout {
            typedef: definition,
            nslots: 0,
            newslotnames: vec![],
            base_layout: std::ptr::null(),
            dict_data_slot: DICT_DATA_SLOT_UNRESOLVED,
        });
        let child = leak_layout(Layout {
            typedef: definition,
            nslots: 1,
            newslotnames: vec!["x".into()],
            base_layout: root,
            dict_data_slot: DICT_DATA_SLOT_UNRESOLVED,
        });
        unsafe {
            assert!(std::ptr::eq((*root).typedef, (*child).typedef));
            assert!((*definition)._freeze_());
            (*definition).set_acceptable_as_base_class(false);
            assert!(!(*(*root).typedef).acceptable_as_base_class());
            assert!(!(*(*child).typedef).acceptable_as_base_class());
            assert!((*(*child).typedef).hasdict);
            assert!((*child).issublayout(root));
        }
    }

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
