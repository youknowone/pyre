//! W_TypeObject — Python `type` object for user-defined classes.
//!
//! PyPy equivalent: pypy/objspace/std/typeobject.py → W_TypeObject
//!
//! A type object holds the class name, tuple of base types, and a namespace
//! dict containing class-level attributes and methods.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// typeobject.py Layout object.
///
/// Immutable after creation. Shared between types that have the same
/// instance layout (e.g. a class without __slots__ shares its base's layout).
/// Identity comparison via pointer equality.
pub struct Layout {
    /// typeobject.py:113 — the typedef (PyType) that this layout is for.
    pub typedef: *const PyType,
    /// typeobject.py:114 — total number of extra slots.
    pub nslots: u32,
    /// typeobject.py:115 — sorted list of slot names introduced by this class.
    pub newslotnames: Vec<String>,
    /// typeobject.py:116 — parent layout (identity comparison).
    pub base_layout: *const Layout,
    /// typedef.py — `acceptable_as_base_class = '__new__' in rawdict`.
    /// TODO: in RPython this lives on TypeDef, accessed
    /// via `layout.typedef.acceptable_as_base_class`. Stored on Layout
    /// here because Rust has no TypeDef struct yet — Layout.typedef is
    /// `*const PyType` (≈ CLASSTYPE), and many types share INSTANCE_TYPE
    /// but need different acceptable_as_base_class values.
    /// Convergence: introduce a Rust TypeDef struct, move this field there.
    pub acceptable_as_base_class: bool,
    /// typedef.py — `hasdict = '__dict__' in rawdict`: whether the
    /// low-level typedef already manages its own instance dict (so mapdict
    /// must NOT add a second one, typeobject.py:255-257).
    /// TODO: like `acceptable_as_base_class`, this belongs on a Rust TypeDef
    /// struct (`layout.typedef.hasdict`). It is parked on Layout because
    /// `typedef` is only a `*const PyType` tag. On the current shared-Layout
    /// model every reachable instance layout reuses INSTANCE_TYPE's Layout
    /// (whose typedef declares no `__dict__`), so this is `false` everywhere;
    /// populating it `true` for the dict-managing typedefs (module/function/
    /// staticmethod/classmethod) needs the distinct-TypeDef convergence and is
    /// deferred with it.
    pub typedef_hasdict: bool,
}

impl Layout {
    /// typeobject.py issublayout(parent):
    ///   while self is not parent:
    ///       self = self.base_layout
    ///       if self is None: return False
    ///   return True
    pub fn issublayout(&self, parent: *const Layout) -> bool {
        let mut current = self as *const Layout;
        while current != parent {
            let cur = unsafe { &*current };
            if cur.base_layout.is_null() {
                return false;
            }
            current = cur.base_layout;
        }
        true
    }

    /// typeobject.py expand(hasdict, weakrefable):
    ///   return (self.typedef, self.newslotnames, self.base_layout,
    ///           hasdict, weakrefable)
    ///
    /// Two types have compatible layouts iff their expand() tuples are equal.
    #[expect(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
    )]
    pub fn expands_equal(
        a: *const Layout,
        a_hasdict: bool,
        a_weakrefable: bool,
        b: *const Layout,
        b_hasdict: bool,
        b_weakrefable: bool,
    ) -> bool {
        if a == b {
            // Same Layout object → typedef, newslotnames, base_layout all identical.
            return a_hasdict == b_hasdict && a_weakrefable == b_weakrefable;
        }
        if a.is_null() || b.is_null() {
            return false;
        }
        let la = unsafe { &*a };
        let lb = unsafe { &*b };
        std::ptr::eq(la.typedef, lb.typedef)
            && la.newslotnames == lb.newslotnames
            && la.base_layout == lb.base_layout
            && a_hasdict == b_hasdict
            && a_weakrefable == b_weakrefable
    }
}

/// Python type object (user-defined class).
///
/// PyPy: pypy/objspace/std/typeobject.py W_TypeObject
#[repr(C)]
pub struct W_TypeObject {
    pub ob_header: PyObject,
    /// Class name (heap-allocated, leaked).
    pub name: *mut String,
    /// App-level `__name__` object.  CPython preserves this object's identity
    /// (including a str subclass assigned to a heap type); null means the
    /// initial exact string has not been materialised yet.
    pub w_name: PyObjectRef,
    /// Qualified class name.  PyPy `W_TypeObject.qualname` is populated by
    /// consuming `__qualname__` from the class namespace at construction.
    pub qualname: *mut String,
    /// App-level `__qualname__` object, with the same lazy/identity semantics
    /// as `w_name`.
    pub w_qualname: PyObjectRef,
    /// typeobject.py:213 `text_signature` — an optional interpreter-level
    /// string supplied by the builtin TypeDef.  This is not a Python object
    /// and therefore is not a GC edge.
    pub text_signature: *mut String,
    /// Tuple of base type objects (PyObjectRef → W_TupleObject or PY_NULL).
    pub bases: PyObjectRef,
    /// Raw pointer to the class dict backing storage (`dict_w` analogue).
    pub dict: *mut u8,
    /// Cached C3 MRO — W_TypeObject.mro_w (typeobject.py `mro_w?[*]`,
    /// an immutable `[W_Root]`). Stored as a stable `Ptr(GcArray(OBJECTPTR))`
    /// block (`alloc_mro_block_gc`) so the length-prefixed inline layout
    /// matches upstream and a JIT-prepass read types as
    /// `SomeList(SomeInstance(PyObjectRef))`.
    pub mro_w: *mut crate::object_array::FixedObjectArray,
    /// typeobject.py:184 `flag_heaptype` — immutable after creation.
    pub flag_heaptype: bool,
    /// [3.14-spec] CPython's public `Py_TPFLAGS_HEAPTYPE` ownership axis.
    ///
    /// PyPy deliberately collapses every interpreter `TypeDef` to
    /// `flag_heaptype = False` (`TypeDef.heaptype`, `typeobject.py`).
    /// CPython 3.14 instead builds many extension-module types through
    /// `PyType_FromModuleAndSpec`, so they publish HEAPTYPE while remaining
    /// PyPy-style builtin TypeDefs internally.  Keep that observable owner
    /// bit separate rather than changing the load-bearing PyPy field above.
    pub flag_cpython_heaptype: bool,
    /// CPython-internal `_Py_TPFLAGS_STATIC_BUILTIN` (bit 1).  This is not the
    /// inverse of HEAPTYPE: a legacy extension type readied through the public
    /// `PyType_Ready` API has neither bit, whereas an interpreter-owned core
    /// type initialized by `_PyStaticType_InitBuiltin` carries this one.
    pub flag_cpython_static_builtin: bool,
    /// [3.14-spec] CPython's orthogonal `Py_TPFLAGS_IMMUTABLETYPE` axis.
    ///
    /// A CPython heap type may still be immutable (the common extension-type
    /// shape), and `PyType_Freeze` can make an existing heap type immutable.
    /// PyPy uses `not flag_heaptype` for this question, so the 3.14 projection
    /// needs its own field instead of overloading the PyPy owner bit.
    pub flag_cpython_immutabletype: std::sync::atomic::AtomicBool,
    /// Suppress CPython's public `Py_TPFLAGS_BASETYPE` without changing
    /// PyPy's load-bearing `Layout.acceptable_as_base_class` field.
    ///
    /// PyPy can reject subclassing in a custom metaclass while leaving the
    /// ordinary app-level layout acceptable.  A CPython static counterpart
    /// instead records the rejection directly in `tp_flags`; projecting that
    /// observable difference must not perturb constructor dispatch.
    pub flag_cpython_suppress_basetype: bool,
    /// typeobject.py `layout` — pointer to shared Layout object.
    pub layout: *const Layout,
    /// typeobject.py:179 `hasdict` — True when instances have __dict__.
    pub hasdict: bool,
    /// typeobject.py:181 `weakrefable` — True when instances support weakrefs.
    pub weakrefable: bool,
    /// typeobject.py:210 `hasuserdel` — True when instances have a user
    /// `__del__` (computed at type creation, typeobject.py:1406/1475, and
    /// kept fresh by `mutated`).
    pub hasuserdel: bool,
    /// typeobject.py:169 `flag_map_or_seq` (`'?'`, `'M'`, `'S'`).
    ///
    /// Default `'?'` per typeobject.py:216.  Inherited from base
    /// classes during heap-type construction (typeobject.py:1495):
    /// when self's flag is `'?'` and a base's flag is non-`'?'`, copy.
    /// Used by `descroperation.py is_iterable` and `:330-346
    /// iter` to skip the `__getitem__` fallback for mapping-typed
    /// classes.  Stored on `W_TypeObject` (not the low-level
    /// `PyType`) so user-defined `dict`/`list`/`tuple` subclasses
    /// inherit the marker the same way PyPy does.
    pub flag_map_or_seq: std::sync::atomic::AtomicU8,
    /// typeobject.py `compares_by_identity_status?` —
    /// `UNKNOWN=0`, `COMPARES_BY_IDENTITY=1`,
    /// `OVERRIDES_EQ_CMP_OR_HASH=2`.  Cached result of
    /// `W_TypeObject.compares_by_identity` (`:353-371`); UNKNOWN
    /// until first lookup forces a `__eq__` / `__hash__` MRO walk.
    ///
    /// Invalidated by `baseobjspace::setattr_str` /
    /// `baseobjspace::delattr_str` whenever a type-dict entry changes
    /// (matches `typeobject.py mutated()`), which walks
    /// `weak_subclasses` and recurses, so a base-class mutation
    /// eagerly resets cached subclasses.
    pub compares_by_identity_status: std::sync::atomic::AtomicU8,
    /// typeobject.py:640-689 `weak_subclasses` —
    /// per-type list of subclass references populated by
    /// `add_subclass` at heaptype creation time
    /// (`typeobject.py ready()` and
    /// `:1604-1613 _add_mro_classes_as_subclasses`).
    ///
    /// PyPy stores `weakref.ref(w_subclass)` entries so subclasses
    /// can be garbage-collected.  Pyre now follows the rweakref
    /// path via `pyre_object::weakref::Weakref` — each slot is a
    /// `*mut Weakref` whose `weakptr` is invalidated by the GC
    /// when the target subclass becomes unreachable
    /// (gctypelayout.py:587, incminimark.py:3058-3126).  The outer
    /// `Vec` is heap-allocated (`Box::into_raw`); the GC's
    /// custom-trace hook registered for `W_TYPE_GC_TYPE_ID` keeps
    /// each `Weakref` struct alive across collections (`pyre-jit
    /// ::eval`).  Null when no subclasses have been registered.
    pub weak_subclasses: *mut Vec<*mut crate::weakref::Weakref>,
    /// typeobject.py:179 `terminator` — the root of this type's mapdict
    /// attribute map (a `DictTerminator` when `hasdict`, else
    /// `NoDictTerminator`), created once per type (typeobject.py:251-260).
    /// Erased `*const MapNode` (the map node layer lives in the
    /// `pyre-interpreter` crate, which `pyre-object` must not depend on; the
    /// interpreter side casts it back). Null until installed by the mapdict
    /// layer. Mirrors `W_ObjectObject.map`.
    pub terminator: *const u8,
    /// typeobject.py:162 `_version_tag` — bumped to a fresh identity whenever
    /// the content of `dict_w` of any type in the MRO changes (`mutated()`,
    /// typeobject.py:285-286), so caches keyed on it (method cache, LOAD_ATTR
    /// inline cache) invalidate. PyPy uses an opaque `VersionTag()` object whose
    /// identity is the version; pyre uses a monotonic `u64` (minted by
    /// `new_version_tag`), with `0` meaning `None` (uncacheable). Equality of
    /// the token is the only observable property, so the `u64` surrogate is
    /// faithful and needs no GC edge.
    pub version_tag: std::sync::atomic::AtomicU64,
    /// typeobject.py:183-185 `uses_object_getattribute` — `True` once a
    /// lookup has confirmed this type uses the object-default
    /// `__getattribute__` (so the attribute fast paths can skip the
    /// `__getattribute__` MRO lookup + `is`-compare).  `False` is the
    /// conservative default (typeobject.py, 275); `mutated()` resets
    /// it on every type-dict change.
    pub uses_object_getattribute: std::sync::atomic::AtomicBool,
    /// typeobject.py:186 `uses_object_setattr` — the `__setattr__`
    /// companion of [`uses_object_getattribute`].
    pub uses_object_setattr: std::sync::atomic::AtomicBool,
    /// typeobject.py:197 `flag_method_descriptor` (default `False`), set
    /// from `typedef.method_descriptor` at `__init__`
    /// (typeobject.py:256; typedef.py:22/61) — `True` only for the
    /// `function` typedef (typedef.py:807).  Gates the LOAD_METHOD
    /// unbound `[w_descr, w_obj]` fast path (callmethod.py:66).  pyre
    /// has no TypeDef struct, so the creation site of each builtin
    /// W_TypeObject sets it directly.
    pub flag_method_descriptor: bool,
    /// `Py_TPFLAGS_DISALLOW_INSTANTIATION` (`1 << 7`) — set on types
    /// whose `tp_new` is NULL (generator / coroutine / frame / ...).
    /// `type.__call__` raises `cannot create 'X' instances` and
    /// `reduce_newobj` raises `cannot pickle 'X' object` when set.  Set
    /// once after construction via `w_type_set_disallow_instantiation`;
    /// never inherited by heap subclasses (the default is `false`).
    pub flag_disallow_instantiation: std::sync::atomic::AtomicBool,
    /// typeobject.py/216 `flag_abstract?` — set from the
    /// `__abstractmethods__` setattr hook; gates `object.__new__`.
    pub flag_abstract: std::sync::atomic::AtomicBool,
    /// The hidden `mutate__version_tag` field for `typeobject.py:177
    /// _immutable_fields_ = ['_version_tag?']` — see [`QuasiImmut`].
    ///
    /// Allocated on the first registration (`get_current_qmut_instance`,
    /// quasiimmut.py:17-27), null until then, and unlinked + freed on
    /// invalidation (`_invalidate_now`, quasiimmut.py), so a type nobody
    /// has mutated since its last compile is the only one holding a box. Holds
    /// no GC pointers, so the `W_TYPE_GC_TYPE_ID` custom trace has nothing to
    /// walk here.
    ///
    /// Shares [`crate::quasiimmut::QuasiImmutField`] with
    /// `ModuleDictStrategy.version?`, the tree's other `?` declaration, the way
    /// upstream's one `QuasiImmut` class serves every quasi-immutable field.
    pub quasi_immut_watchers: crate::quasiimmut::QuasiImmutField,
    /// `Py_TPFLAGS_HAVE_GC` (`1 << 14`) — whether instances of this type join
    /// the collector's traversal. A build that keeps the global interpreter
    /// lock gives them a two-word `PyGC_Head` for it, which is what
    /// `_PyType_PreHeaderSize` charges; one without the lock keeps the same
    /// bits in the object header and charges nothing, so the flag records the
    /// type property alone.
    ///
    /// A property of the type, not of the heap an instance landed in: the same
    /// `str` value must report one size whether it was folded into a code
    /// constant or built at run time. It is deliberately not the collector's
    /// "does this object hold pointers" question either — a pyre `str` struct
    /// carries `w_dict` and `w_weakreflifeline` slots the collector really does
    /// follow, and the CPython `str` those slots have no counterpart in is not
    /// a GC type.
    ///
    /// True by default, which is the answer for every heap type
    /// (`type_new` sets the flag unconditionally) and for the container
    /// builtins; the creation site of a scalar builtin clears it.
    pub flag_have_gc: bool,
}

/// Source of fresh `version_tag` identities (`VersionTag()`, typeobject.py).
/// `0` is reserved for `None`, so the counter starts at `1`.
static NEXT_VERSION_TAG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Mint a fresh, never-reused version-tag identity (typeobject.py:73-74
/// `VersionTag()`). Never returns `0` (which means `None`/uncacheable).
pub fn new_version_tag() -> u64 {
    NEXT_VERSION_TAG.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// GC type id assigned to `W_TypeObject` at JitDriver init time.
pub const W_TYPE_GC_TYPE_ID: u32 = 33;

/// Fixed payload size (`framework.py:811`).
pub const W_TYPE_OBJECT_SIZE: usize = std::mem::size_of::<W_TypeObject>();

impl crate::lltype::GcType for W_TypeObject {
    fn type_id() -> u32 {
        W_TYPE_GC_TYPE_ID
    }
    const SIZE: usize = W_TYPE_OBJECT_SIZE;
}

/// Byte offset of the `name: *mut String` slot within `W_TypeObject`.
pub const W_TYPE_NAME_OFFSET: usize = std::mem::offset_of!(W_TypeObject, name);

/// GC-managed name-string box shared by `W_TypeObject` and `Function`, whose
/// `name` fields are both `String` behind a raw pointer.
///
/// A leaf (`String` = heap `Vec<u8>`, no inner `PyObjectRef`); its GC box
/// carries only drop glue that reclaims the buffer on sweep. Only *mortal*
/// holders box their name — a heap type (`w_type_new`) and a user function
/// (`PyCode`-backed `function_new_impl`). Immortal holders (builtin types and
/// builtin functions) keep a `malloc_raw` name, because an immortal holder is
/// never greyed and so could never keep an old-gen box alive. Mirrors
/// [`crate::unicodeobject::UnicodeValueStorage`] and the `longobject` bigint box.
pub type NameStorage = String;

/// Runtime-assigned GC type id for [`NameStorage`]. Published by
/// `pyre-jit::eval` after the fixed-constant type registrations; never embedded
/// in a JIT allocation descriptor.
static NAME_STORAGE_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for [`NameStorage`].
pub fn set_name_storage_gc_type_id(id: u32) {
    NAME_STORAGE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for [`NameStorage`].
#[majit_macros::dont_look_inside]
pub fn name_storage_gc_type_id() -> u32 {
    NAME_STORAGE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Leak a Layout to get a 'static pointer for sharing.
pub fn leak_layout(layout: Layout) -> *const Layout {
    crate::lltype::malloc_raw(layout)
}

/// Builtin types (`w_type_new_builtin`) are process-global, so their root
/// registry must have the same owner.  A TLS registry loses types first made
/// on a worker and makes a collection on another thread miss their children.
static BUILTIN_TYPE_NAMESPACE_ROOTS: std::sync::OnceLock<std::sync::Mutex<Vec<usize>>> =
    std::sync::OnceLock::new();

fn builtin_type_namespace_roots() -> &'static std::sync::Mutex<Vec<usize>> {
    BUILTIN_TYPE_NAMESPACE_ROOTS.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

/// Record an immortal type for the collection-time namespace root walk.
#[majit_macros::dont_look_inside]
fn register_builtin_type_roots(addr: usize) {
    // Record the prebuilt-family store so the next minor collection scans it
    // (gc_roots.rs prebuilt-root write tracking).
    crate::gc_roots::mark_prebuilt_roots_dirty();
    builtin_type_namespace_roots()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(addr);
}

/// Snapshot the registered immortal-type addresses for the root walker
/// (`pyre_interpreter::eval::walk_builtin_type_dicts_gc`).
#[majit_macros::dont_look_inside]
pub fn snapshot_builtin_type_roots() -> Vec<usize> {
    builtin_type_namespace_roots()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Allocate a new W_TypeObject with `flag_heaptype = true`.
///
/// typeobject.py:174 `__init__(..., is_heaptype=True)`.
/// Layout is set to null initially; caller must set it via set_layout
/// after running create_all_slots / setup_builtin_type.
pub fn w_type_new(name: &str, bases: PyObjectRef, dict_ptr: *mut u8) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(bases);
    crate::gc_roots::pin_root(dict_ptr as PyObjectRef);
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(W_TYPE_GC_TYPE_ID, W_TYPE_OBJECT_SIZE);
    // A mortal (GC-managed) heap type boxes its name in a GC-managed storage box
    // reclaimed by the box tid's drop glue (`NameStorage`), greyed through the
    // `name` slot in `type_object_custom_trace`. The immortal fallback (pre-GC /
    // snapshot tools) keeps a `malloc_raw` name that an immortal holder can never
    // grey — the non-collecting old-gen alloc above cannot sweep this box before
    // it is stored into the type below.
    let name_value = name.to_string();
    let (name, qualname) = if raw.is_null() {
        (
            crate::lltype::malloc_raw(name_value.clone()),
            crate::lltype::malloc_raw(name_value),
        )
    } else {
        let name =
            crate::gc_storage::gc_alloc_storage_box(name_value.clone(), name_storage_gc_type_id());
        crate::gc_roots::pin_root(name as PyObjectRef);
        let qualname =
            crate::gc_storage::gc_alloc_storage_box(name_value, name_storage_gc_type_id());
        let name = crate::gc_roots::shadow_stack_get(save_point + 2) as *mut String;
        (name, qualname)
    };
    // Install the forwarded bases and managed namespace addresses rather than the
    // pre-collection arguments (the pins survive any collection the alloc forces).
    let bases = crate::gc_roots::shadow_stack_get(save_point);
    let dict_ptr = crate::gc_roots::shadow_stack_get(save_point + 1) as *mut u8;
    let value = W_TypeObject {
        ob_header: PyObject {
            ob_type: &TYPE_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        mro_w: std::ptr::null_mut(),
        name,
        w_name: PY_NULL,
        qualname,
        w_qualname: PY_NULL,
        text_signature: std::ptr::null_mut(),
        bases,
        dict: dict_ptr,
        flag_heaptype: true,
        flag_cpython_heaptype: true,
        flag_cpython_static_builtin: false,
        flag_cpython_immutabletype: std::sync::atomic::AtomicBool::new(false),
        flag_cpython_suppress_basetype: false,
        layout: std::ptr::null(),
        hasdict: false,
        weakrefable: false,
        hasuserdel: false,
        flag_map_or_seq: std::sync::atomic::AtomicU8::new(b'?'),
        compares_by_identity_status: std::sync::atomic::AtomicU8::new(COMPARES_BY_IDENTITY_UNKNOWN),
        weak_subclasses: std::ptr::null_mut(),
        // typeobject.py:251-260: terminator installed by the interpreter's
        // mapdict layer after construction; null until then.
        terminator: std::ptr::null(),
        // typeobject.py:244-250: a fresh version tag at construction.
        // pyre's construction splits the MRO install into a separate
        // `w_type_set_mro` call, so the `is_mro_purely_of_types` gate
        // that demotes the tag to None lives there.
        version_tag: std::sync::atomic::AtomicU64::new(new_version_tag()),
        // typeobject.py:185-186: conservative `False` default, fixed during
        // real usage by the attribute fast paths.
        uses_object_getattribute: std::sync::atomic::AtomicBool::new(false),
        uses_object_setattr: std::sync::atomic::AtomicBool::new(false),
        // typeobject.py:256 — user-defined typedefs never set
        // `method_descriptor` (typedef.py:22 default `False`).
        flag_method_descriptor: false,
        // Heap subclasses are always instantiable (their `tp_new` is the
        // slot wrapper); only builtin disallow-types flip this.
        flag_disallow_instantiation: std::sync::atomic::AtomicBool::new(false),
        flag_abstract: std::sync::atomic::AtomicBool::new(false),
        // Allocated lazily on the first loop registration.
        quasi_immut_watchers: crate::quasiimmut::QuasiImmutField::new(),
        flag_have_gc: true,
    };
    let (w_type, gc_managed) = if !raw.is_null() {
        unsafe { std::ptr::write(raw as *mut W_TypeObject, value) };
        (raw as PyObjectRef, true)
    } else {
        // No GC hook yet (pre-init / snapshot tools): fall back to an immortal box.
        (crate::lltype::malloc_typed(value) as PyObjectRef, false)
    };
    if gc_managed {
        // Fresh old-gen type stores young `bases`/namespace into an old object;
        // remember it so the next minor collection scans it and its custom trace
        // (`W_TYPE_GC_TYPE_ID`) forwards those young children.
        crate::gc_hook::try_gc_write_barrier(w_type as *mut u8);
    } else {
        // Immortal fallback type (pre-GC): its trace never fires, so root its
        // namespace the same way builtin types are rooted.
        register_builtin_type_roots(w_type as usize);
    }
    w_type
}

/// typeobject.py in setup_user_defined_type — copy
/// `flag_map_or_seq` from the first base whose flag is non-`?`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn inherit_flag_map_or_seq(w_self: PyObjectRef, bases: PyObjectRef) {
    if w_self.is_null() || bases.is_null() || !is_type(w_self) {
        return;
    }
    let self_ref = &*(w_self as *const W_TypeObject);
    if self_ref
        .flag_map_or_seq
        .load(std::sync::atomic::Ordering::Acquire)
        != b'?'
    {
        return;
    }
    let n = crate::w_tuple_len(bases);
    for i in 0..n as i64 {
        let Some(w_base) = crate::w_tuple_getitem(bases, i) else {
            continue;
        };
        if w_base.is_null() || !is_type(w_base) {
            continue;
        }
        let base_ref = &*(w_base as *const W_TypeObject);
        let base_flag = base_ref
            .flag_map_or_seq
            .load(std::sync::atomic::Ordering::Acquire);
        if base_flag != b'?' {
            self_ref
                .flag_map_or_seq
                .store(base_flag, std::sync::atomic::Ordering::Release);
            return;
        }
    }
}

/// Allocate a new W_TypeObject with `flag_heaptype = false`.
///
/// typeobject.py:174 `__init__(..., is_heaptype=False)`.
pub fn w_type_new_builtin(
    name: &str,
    bases: PyObjectRef,
    dict_ptr: *mut u8,
    _layout_pytype: *const PyType,
) -> PyObjectRef {
    let qualname_value = name.rsplit('.').next().unwrap_or(name).to_string();
    let name = crate::lltype::malloc_raw(name.to_string());
    let qualname = crate::lltype::malloc_raw(qualname_value);
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(bases);
    crate::gc_roots::pin_root(dict_ptr as PyObjectRef);

    let bases = crate::gc_roots::shadow_stack_get(save_point);
    let dict_ptr = crate::gc_roots::shadow_stack_get(save_point + 1) as *mut u8;

    let w_type = crate::lltype::malloc_typed(W_TypeObject {
        ob_header: PyObject {
            ob_type: &TYPE_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        mro_w: std::ptr::null_mut(),
        name,
        w_name: PY_NULL,
        qualname,
        w_qualname: PY_NULL,
        text_signature: std::ptr::null_mut(),
        bases,
        dict: dict_ptr,
        flag_heaptype: false,
        flag_cpython_heaptype: false,
        flag_cpython_static_builtin: true,
        flag_cpython_immutabletype: std::sync::atomic::AtomicBool::new(true),
        flag_cpython_suppress_basetype: false,
        layout: std::ptr::null(),
        hasdict: false,
        weakrefable: false,
        hasuserdel: false,
        // typeobject.py:216 default; built-in dict/list/tuple
        // override via `w_type_set_flag_map_or_seq` at typedef
        // registration time (see `typedef.rs`).
        flag_map_or_seq: std::sync::atomic::AtomicU8::new(b'?'),
        compares_by_identity_status: std::sync::atomic::AtomicU8::new(COMPARES_BY_IDENTITY_UNKNOWN),
        weak_subclasses: std::ptr::null_mut(),
        // typeobject.py:251-260: terminator installed by the interpreter's
        // mapdict layer after construction; null until then.
        terminator: std::ptr::null(),
        // typeobject.py:244-250: a fresh version tag at construction.
        version_tag: std::sync::atomic::AtomicU64::new(new_version_tag()),
        // typeobject.py:185-186: conservative `False` default.
        uses_object_getattribute: std::sync::atomic::AtomicBool::new(false),
        uses_object_setattr: std::sync::atomic::AtomicBool::new(false),
        // typeobject.py — `typedef.method_descriptor` (typedef.py:22
        // default `False`); the `function` creation site flips it
        // (typedef.py:807).
        flag_method_descriptor: false,
        // `Py_TPFLAGS_DISALLOW_INSTANTIATION` off by default; the
        // generator / coroutine / frame typedefs flip it via
        // `w_type_set_disallow_instantiation`.
        flag_disallow_instantiation: std::sync::atomic::AtomicBool::new(false),
        flag_abstract: std::sync::atomic::AtomicBool::new(false),
        // Allocated lazily on the first loop registration.
        quasi_immut_watchers: crate::quasiimmut::QuasiImmutField::new(),
        flag_have_gc: true,
    }) as PyObjectRef;
    // A builtin type is Box-immortal, so its namespace values and `bases` are reachable only
    // through `walk_builtin_type_dicts_gc` (`pyre_interpreter::eval`).
    // Register it so that walk forwards them; without this a
    // collection could otherwise reclaim a young namespace value.
    register_builtin_type_roots(w_type as usize);
    w_type
}

/// `dictmultiobject.py` `UNKNOWN` — cache miss; recompute via
/// `compares_by_identity` lookup.
pub const COMPARES_BY_IDENTITY_UNKNOWN: u8 = 0;
/// `dictmultiobject.py:154 COMPARES_BY_IDENTITY` — type uses
/// object-default `__eq__`/`__hash__`; identity comparison is
/// observable-equivalent.
pub const COMPARES_BY_IDENTITY_YES: u8 = 1;
/// `dictmultiobject.py:155 OVERRIDES_EQ_CMP_OR_HASH` — type defines a
/// custom `__eq__` or `__hash__`; identity comparison is not safe.
pub const COMPARES_BY_IDENTITY_NO: u8 = 2;

/// `typeobject.py W_TypeObject.compares_by_identity` —
/// status reader.  Returns the cached value directly without
/// recomputation; callers that need the fresh value invoke the
/// `dict_eq_hook::COMPARES_BY_IDENTITY_HOOK` trampoline which
/// forwards to pyre-interpreter for the MRO walk.
///
/// # Safety
/// `w_type` must be a valid PyObjectRef pointing at a `W_TypeObject`.
pub unsafe fn w_type_compares_by_identity_status(w_type: PyObjectRef) -> u8 {
    if w_type.is_null() || !is_type(w_type) {
        return COMPARES_BY_IDENTITY_NO;
    }
    let t = &*(w_type as *const W_TypeObject);
    t.compares_by_identity_status
        .load(std::sync::atomic::Ordering::Acquire)
}

/// Write-side companion to [`w_type_compares_by_identity_status`].
///
/// # Safety
/// Same as the reader; called by pyre-interpreter's lookup after
/// resolving `__eq__` / `__hash__`.
pub unsafe fn w_type_set_compares_by_identity_status(w_type: PyObjectRef, status: u8) {
    if w_type.is_null() || !is_type(w_type) {
        return;
    }
    let t = &*(w_type as *const W_TypeObject);
    t.compares_by_identity_status
        .store(status, std::sync::atomic::Ordering::Release);
}

/// typeobject.py — `flag_map_or_seq` accessor on a `W_TypeObject`.
/// Returns `'?'` if `w_type` is null, not a type object, or never had
/// the marker assigned.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_flag_map_or_seq(w_type: PyObjectRef) -> u8 {
    if w_type.is_null() || !is_type(w_type) {
        return b'?';
    }
    let t = &*(w_type as *const W_TypeObject);
    t.flag_map_or_seq.load(std::sync::atomic::Ordering::Acquire)
}

/// typeobject.py — `flag_map_or_seq` setter.  Used by
/// `init_typeobjects` to mark dict / list / tuple W_TypeObjects at
/// registration time (objspace.py:104-108).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_flag_map_or_seq(w_type: PyObjectRef, flag: u8) {
    if w_type.is_null() || !is_type(w_type) {
        return;
    }
    let t = &*(w_type as *const W_TypeObject);
    t.flag_map_or_seq
        .store(flag, std::sync::atomic::Ordering::Release);
}

/// `Py_TPFLAGS_DISALLOW_INSTANTIATION` reader — `True` when `w_type`'s
/// `tp_new` is conceptually NULL (the type refuses `Type()`).
///
/// # Safety
/// `w_type` must be a valid PyObjectRef pointing at a `W_TypeObject`.
pub unsafe fn w_type_disallows_instantiation(w_type: PyObjectRef) -> bool {
    if w_type.is_null() || !is_type(w_type) {
        return false;
    }
    let t = &*(w_type as *const W_TypeObject);
    t.flag_disallow_instantiation
        .load(std::sync::atomic::Ordering::Acquire)
}

/// `Py_TPFLAGS_DISALLOW_INSTANTIATION` setter — flips a builtin type to
/// refuse instantiation.  Called once at typedef registration for
/// generator / coroutine / frame-shaped types.
///
/// # Safety
/// `w_type` must be a valid PyObjectRef pointing at a `W_TypeObject`.
pub unsafe fn w_type_set_disallow_instantiation(w_type: PyObjectRef) {
    if w_type.is_null() || !is_type(w_type) {
        return;
    }
    let t = &*(w_type as *const W_TypeObject);
    t.flag_disallow_instantiation
        .store(true, std::sync::atomic::Ordering::Release);
}

/// Clear [`W_TypeObject::flag_have_gc`] — the creation site of a builtin type
/// whose CPython counterpart carries no `Py_TPFLAGS_HAVE_GC` calls this once.
///
/// # Safety
/// `w_type` must be a valid PyObjectRef pointing at a `W_TypeObject`.
pub unsafe fn w_type_clear_have_gc(w_type: PyObjectRef) {
    if w_type.is_null() || !is_type(w_type) {
        return;
    }
    (*(w_type as *mut W_TypeObject)).flag_have_gc = false;
}

/// Whether instances of `w_type` carry the collector pre-header
/// (`_PyType_IS_GC`). False for a null or non-type argument, which is the
/// answer a caller with no type to ask wants.
///
/// # Safety
/// `w_type` must be a valid PyObjectRef.
pub unsafe fn w_type_get_have_gc(w_type: PyObjectRef) -> bool {
    if w_type.is_null() || !is_type(w_type) {
        return false;
    }
    (*(w_type as *const W_TypeObject)).flag_have_gc
}

/// `W_TypeObject.is_abstract` (typeobject.py).
///
/// # Safety
/// `w_type` must point at a `W_TypeObject`.
pub unsafe fn w_type_is_abstract(w_type: PyObjectRef) -> bool {
    if w_type.is_null() || !is_type(w_type) {
        return false;
    }
    let t = &*(w_type as *const W_TypeObject);
    t.flag_abstract.load(std::sync::atomic::Ordering::Acquire)
}

/// `W_TypeObject.set_abstract` (typeobject.py).
///
/// `dont_look_inside`: the `flag_abstract` atomic store is a runtime-mutable
/// side effect on per-type state, not a build-time constant, so the JIT
/// residualises the call via the registered fnaddr rather than tracing the
/// store the tracer cannot model.
///
/// # Safety
/// `w_type` must point at a `W_TypeObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_type_set_abstract(w_type: PyObjectRef, abstract_: bool) {
    if w_type.is_null() || !is_type(w_type) {
        return;
    }
    let t = &*(w_type as *const W_TypeObject);
    t.flag_abstract
        .store(abstract_, std::sync::atomic::Ordering::Release);
}

// ── Layout accessors ─────────────────────────────────────────────────

/// Set the Layout pointer on a type object.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_layout(obj: PyObjectRef, layout: *const Layout) {
    (*(obj as *mut W_TypeObject)).layout = layout;
}

/// Get the Layout pointer from a type object.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_layout_ptr(obj: PyObjectRef) -> *const Layout {
    (*(obj as *const W_TypeObject)).layout
}

/// typeobject.py get_full_instance_layout(self).
/// Returns the Layout.typedef pointer (the PyType describing instance struct).
/// For backward-compat with existing code that compares PyType pointers.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_layout(obj: PyObjectRef) -> *const PyType {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        &INSTANCE_TYPE as *const PyType
    } else {
        (*layout).typedef
    }
}

/// Get nslots from the Layout.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_nslots(obj: PyObjectRef) -> u32 {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        0
    } else {
        (*layout).nslots
    }
}

/// Get newslotnames from the Layout.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_newslotnames(obj: PyObjectRef) -> &'static [String] {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        &[]
    } else {
        &(*layout).newslotnames
    }
}

/// Get base_layout pointer for identity comparison.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_base_layout(obj: PyObjectRef) -> *const Layout {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        std::ptr::null()
    } else {
        (*layout).base_layout
    }
}

/// typeobject.py `flag_method_descriptor` getter/setter
/// (callmethod.py:66 `space.type(w_descr).flag_method_descriptor`).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_flag_method_descriptor(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).flag_method_descriptor
}
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_flag_method_descriptor(obj: PyObjectRef, v: bool) {
    (*(obj as *mut W_TypeObject)).flag_method_descriptor = v;
}

/// typeobject.py `hasdict` getter/setter.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_hasdict(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).hasdict
}
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_hasdict(obj: PyObjectRef, v: bool) {
    (*(obj as *mut W_TypeObject)).hasdict = v;
}

/// typeobject.py:295 `self._version_tag` — the raw cache-version field
/// (`0` = `None`/uncacheable).  This is the direct field read; the
/// `we_are_jitted()` / `_pure_version_tag` (`@elidable_promote`) split of
/// `version_tag()` (typeobject.py) lives in the interpreter layer
/// (`baseobjspace::w_type_version_tag`), which has the JIT intrinsics.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_version_tag(obj: PyObjectRef) -> u64 {
    (*(obj as *const W_TypeObject))
        .version_tag
        .load(std::sync::atomic::Ordering::Acquire)
}
/// Store a new version-tag identity (typeobject.py `mutated`).
///
/// Revokes the loops that baked the old identity as a constant before
/// publishing the new one. This is the only writer of the field, so placing the
/// `_version_tag?` invalidation here covers every way the tag can change —
/// `mutated()`'s fresh identity and the two demotions to `0`
/// (uncacheable) alike — rather than leaving each caller to remember it.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_version_tag(obj: PyObjectRef, v: u64) {
    w_type_notify_quasi_immut_watchers(obj);
    (*(obj as *const W_TypeObject))
        .version_tag
        .store(v, std::sync::atomic::Ordering::Release);
}

/// `quasiimmut.py get_current_qmut_instance` for this type's
/// `_version_tag?`.
///
/// Called while the trace is still being recorded, exactly where
/// `QuasiImmutDescr.__init__` (`pyjitpl.py`) calls it: a write reached
/// later in that same trace then sees a non-null `mutate_*` field and aborts
/// the attempt. The instance is handed back so the recording can carry it to
/// `heap.py is_still_valid_for` and `compile.py
/// register_loop_token`, after which [`w_type_notify_quasi_immut_watchers`]
/// revokes the loop when the tag is bumped.
///
/// # Safety
/// `obj` must be null or point at a valid `W_TypeObject`.
pub unsafe fn w_type_current_qmut_instance(
    obj: PyObjectRef,
) -> Option<std::sync::Arc<crate::quasiimmut::QuasiImmut>> {
    if obj.is_null() || !is_type(obj) {
        return None;
    }
    Some(
        (*(obj as *const W_TypeObject))
            .quasi_immut_watchers
            .get_current_qmut_instance(),
    )
}

/// Revoke every loop that baked this type's `version_tag` as a constant
/// (`quasiimmut.py make_invalidation_function._invalidate_now`).
///
/// ```text
///  def _invalidate_now(p):
///      qmut_ptr = getattr(p, mutatefieldname)
///      setattr(p, mutatefieldname, lltype.nullptr(rclass.OBJECT))
///      qmut = cast_base_ptr_to_instance(QuasiImmut, qmut_ptr)
///      qmut.invalidate(descr_repr)
/// ```
///
/// The field is unlinked *before* the sweep and the instance becomes garbage
/// right after it, so the next registration allocates a fresh one; dropping the
/// [`Box`] here is that collection. Called from [`w_type_set_version_tag`] right
/// before the new tag is published.
///
/// Upstream runs the whole walk outside traced code — `_invalidate_now` is
/// reached from the residual `jit_force_quasi_immutable` path, never from a
/// trace — so the sweep is residualised the same way
/// ([`crate::quasiimmut::sweep_quasi_immut_field`]). The installed check stays
/// traced, so mutating a type no loop depends on still makes no call.
///
/// # Safety
/// `obj` must be null or point at a valid `W_TypeObject`.
pub unsafe fn w_type_notify_quasi_immut_watchers(obj: PyObjectRef) {
    if obj.is_null() || !is_type(obj) {
        return;
    }
    let field = &(*(obj as *const W_TypeObject)).quasi_immut_watchers;
    if !field.is_installed() {
        return;
    }
    crate::quasiimmut::sweep_quasi_immut_field(field);
}

/// typeobject.py:183-185 `uses_object_getattribute` reader.  Returns the
/// conservative `false` for a null / non-type pointer (matches the class
/// default before any lookup confirms the flag).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_uses_object_getattribute(obj: PyObjectRef) -> bool {
    if obj.is_null() || !is_type(obj) {
        return false;
    }
    (*(obj as *const W_TypeObject))
        .uses_object_getattribute
        .load(std::sync::atomic::Ordering::Acquire)
}
/// Write-side companion to [`w_type_get_uses_object_getattribute`]
/// (typeobject.py:275, 315).
///
/// Mutates the per-type `uses_object_getattribute` atomic — a side effect on
/// runtime type state the tracer cannot model, so the JIT residualises
/// the call rather than tracing into it (`@dont_look_inside`,
/// `rlib/jit.py:139`), the [`w_type_set_uses_object_setattr`] twin.
#[majit_macros::dont_look_inside]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_uses_object_getattribute(obj: PyObjectRef, v: bool) {
    if obj.is_null() || !is_type(obj) {
        return;
    }
    (*(obj as *const W_TypeObject))
        .uses_object_getattribute
        .store(v, std::sync::atomic::Ordering::Release);
}

/// typeobject.py:186 `uses_object_setattr` reader (see
/// [`w_type_get_uses_object_getattribute`]).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_uses_object_setattr(obj: PyObjectRef) -> bool {
    if obj.is_null() || !is_type(obj) {
        return false;
    }
    (*(obj as *const W_TypeObject))
        .uses_object_setattr
        .load(std::sync::atomic::Ordering::Acquire)
}
/// Write-side companion to [`w_type_get_uses_object_setattr`]
/// (typeobject.py:276, 340).
///
/// Mutates the per-type `uses_object_setattr` atomic — a side effect on
/// runtime type state the tracer cannot model, so the JIT residualises
/// the call rather than tracing into it (`@dont_look_inside`,
/// `rlib/jit.py:139`).
#[majit_macros::dont_look_inside]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_uses_object_setattr(obj: PyObjectRef, v: bool) {
    if obj.is_null() || !is_type(obj) {
        return;
    }
    (*(obj as *const W_TypeObject))
        .uses_object_setattr
        .store(v, std::sync::atomic::Ordering::Release);
}

/// typeobject.py `terminator` getter/setter. The stored value is an
/// erased `*const MapNode`; the `pyre-interpreter` mapdict layer casts it.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_terminator(obj: PyObjectRef) -> *const u8 {
    (*(obj as *const W_TypeObject)).terminator
}
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_terminator(obj: PyObjectRef, terminator: *const u8) {
    (*(obj as *mut W_TypeObject)).terminator = terminator;
}

/// typeobject.py `weakrefable` getter/setter.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_weakrefable(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).weakrefable
}
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_weakrefable(obj: PyObjectRef, v: bool) {
    (*(obj as *mut W_TypeObject)).weakrefable = v;
}

/// typeobject.py `hasuserdel` getter/setter.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_hasuserdel(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).hasuserdel
}
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_hasuserdel(obj: PyObjectRef, v: bool) {
    (*(obj as *mut W_TypeObject)).hasuserdel = v;
}

// ── Other accessors ──────────────────────────────────────────────────

/// Get the class name.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_name(obj: PyObjectRef) -> &'static str {
    &*(*(obj as *const W_TypeObject)).name
}

/// Return the stable app-level `type.__name__` object.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_name_obj(obj: PyObjectRef) -> PyObjectRef {
    let t = &mut *(obj as *mut W_TypeObject);
    if t.w_name.is_null() {
        let full = &*t.name;
        let bare = if t.flag_heaptype {
            full.as_str()
        } else {
            full.rsplit('.').next().unwrap_or(full)
        };
        t.w_name = crate::w_str_new(bare);
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
    t.w_name
}

/// The `type.__name__` slot as it stands, without the lazy materialisation
/// [`w_type_get_name_obj`] performs — `PY_NULL` until the first read of the
/// name has been served.
///
/// The tracer reads it this way: materialising here would allocate a string
/// while the walker holds raw pointers into the heap, and a class whose name
/// has never been asked for is not one a hot loop is reading it from.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_peek_name_obj(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_TypeObject)).w_name
}

/// Replace the class name (`descr_set__name__`, typeobject.py
/// `w_type.name = name`).  `name` is an owned `String` behind a raw
/// pointer (`malloc_raw` = boxed); assigning through it drops the old
/// name and installs the new one, leaving the slot itself unchanged.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_name(obj: PyObjectRef, w_name: PyObjectRef) {
    let t = &mut *(obj as *mut W_TypeObject);
    *t.name = crate::w_str_get_value(w_name).to_string();
    t.w_name = w_name;
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// `typeobject.py` / `getqualname`: the class qualified name lives
/// on `W_TypeObject`, not in its namespace after type creation.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_qualname(obj: PyObjectRef) -> &'static str {
    &*(*(obj as *const W_TypeObject)).qualname
}

/// Return the stable app-level `type.__qualname__` object.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_qualname_obj(obj: PyObjectRef) -> PyObjectRef {
    let t = &mut *(obj as *mut W_TypeObject);
    if t.w_qualname.is_null() {
        t.w_qualname = crate::w_str_new(&*t.qualname);
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
    t.w_qualname
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_qualname(obj: PyObjectRef, w_qualname: PyObjectRef) {
    let t = &mut *(obj as *mut W_TypeObject);
    // `qualname` is the `&str` view display and error messages read, so a name
    // carrying a lone surrogate is stored there with the replacement character
    // `Wtf8`'s `Display` substitutes.  `w_qualname` keeps the name object
    // itself, and `w_type_get_qualname_obj` hands that back verbatim, so
    // `__qualname__` still reads the code points that were assigned.
    *t.qualname = crate::w_str_get_wtf8(w_qualname).to_string();
    t.w_qualname = w_qualname;
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// typeobject.py `type_get_text_signature` backing field.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_text_signature(obj: PyObjectRef) -> Option<&'static str> {
    let signature = (*(obj as *const W_TypeObject)).text_signature;
    if signature.is_null() {
        None
    } else {
        Some(&*signature)
    }
}

/// Set the initialization-time TypeDef `_text_signature_` value.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_text_signature(obj: PyObjectRef, signature: &str) {
    let type_obj = &mut *(obj as *mut W_TypeObject);
    debug_assert!(type_obj.text_signature.is_null());
    type_obj.text_signature = crate::lltype::malloc_raw(signature.to_owned());
}

/// Get the bases tuple.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_bases(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_TypeObject)).bases
}

/// Replace the bases tuple (`type.__bases__` setter).  The caller is
/// responsible for validating layout compatibility and recomputing the MRO.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_bases(obj: PyObjectRef, bases: PyObjectRef) {
    crate::gc_roots::pin_root(bases);
    crate::gc_roots::mark_prebuilt_roots_dirty();
    (*(obj as *mut W_TypeObject)).bases = bases;
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// Get the class namespace pointer (as *mut u8).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_dict_ptr(obj: PyObjectRef) -> *mut u8 {
    (*(obj as *const W_TypeObject)).dict
}

/// Get the cached MRO block, or null if not yet set.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_mro(obj: PyObjectRef) -> *mut crate::object_array::FixedObjectArray {
    (*(obj as *const W_TypeObject)).mro_w
}

/// True when `cls` occurs in `w_type`'s MRO — the pointer-identity subtype
/// membership scan `W_TypeObject.issubtype` / `_issubtype` performs
/// (typeobject.py:603/1640).  The single home for the MRO subtype check;
/// interpreter-level subtype guards and reflected-binop dispatch delegate
/// here rather than each re-scanning the MRO.
///
/// Under the JIT `issubtype` runs this scan inside `_pure_issubtype`
/// (`@elidable_promote`, typeobject.py:1657), so the MRO membership walk
/// is not traced — its result is promoted.  The `dont_look_inside` marker
/// is the equivalent boundary: the JIT residualises the call instead of
/// tracing the per-type MRO read the tracer cannot model.
#[majit_macros::dont_look_inside]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_issubtype(w_type: PyObjectRef, cls: PyObjectRef) -> bool {
    let mro_ptr = w_type_get_mro(w_type);
    if mro_ptr.is_null() {
        // typeobject.py _issubtype_slow_and_wrong — a partially
        // initialised type (custom metaclass mro()) has no mro yet; walk
        // find_best_base up the base chain (single inheritance, deliberately
        // wrong for multiple inheritance).
        let mut w_cls = w_type;
        while !w_cls.is_null() {
            if std::ptr::eq(w_cls, cls) {
                return true;
            }
            w_cls = find_best_base(w_cls);
        }
        return false;
    }
    (*mro_ptr).as_slice().iter().any(|&t| std::ptr::eq(t, cls))
}

/// Whether a class pattern with one positional sub-pattern receives the
/// subject itself. CPython stores this as `_Py_TPFLAGS_MATCH_SELF`; PyPy's
/// pattern opcode recognizes the same builtin atomic families. Membership in
/// any family's MRO makes the property inherit exactly as CPython's
/// `inherit_special` does.
pub unsafe fn w_type_has_match_self(w_type: PyObjectRef) -> bool {
    if w_type.is_null() || !is_type(w_type) {
        return false;
    }
    for base in [
        get_instantiate(&INT_TYPE),
        get_instantiate(&FLOAT_TYPE),
        get_instantiate(&STR_TYPE),
        get_instantiate(&LIST_TYPE),
        get_instantiate(&TUPLE_TYPE),
        get_instantiate(&DICT_TYPE),
        get_instantiate(&crate::bytesobject::BYTES_TYPE),
        get_instantiate(&crate::bytearrayobject::BYTEARRAY_TYPE),
        get_instantiate(&crate::setobject::SET_TYPE),
        get_instantiate(&crate::setobject::FROZENSET_TYPE),
    ] {
        if !base.is_null() && w_type_issubtype(w_type, base) {
            return true;
        }
    }
    false
}

/// typeobject.py find_best_base — the type base whose instance
/// layout a subtype extends (most-derived layout among the type bases).
/// Non-raising variant for the null-mro subtype fallback.
unsafe fn find_best_base(w_type: PyObjectRef) -> PyObjectRef {
    let bases = w_type_get_bases(w_type);
    if bases.is_null() {
        return PY_NULL;
    }
    let mut w_bestbase = PY_NULL;
    let mut best_layout: *const Layout = std::ptr::null();
    for w_cand in crate::tupleobject::w_tuple_items_copy_as_vec(bases) {
        if !is_type(w_cand) {
            continue;
        }
        let cand_layout = w_type_get_layout_ptr(w_cand);
        if w_bestbase.is_null() {
            w_bestbase = w_cand;
            best_layout = cand_layout;
            continue;
        }
        if cand_layout != best_layout
            && !cand_layout.is_null()
            && (*cand_layout).issublayout(best_layout)
        {
            w_bestbase = w_cand;
            best_layout = cand_layout;
        }
    }
    w_bestbase
}

/// PyPy `typeobject.py descr__base` — return the base whose
/// instance layout this type extends.  This is not necessarily the first
/// entry in `__bases__` for multiple inheritance.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_best_base(w_type: PyObjectRef) -> PyObjectRef {
    find_best_base(w_type)
}

/// Set the cached MRO.
///
/// Construction installs the MRO here (rather than in `__init__`
/// itself, typeobject.py:244), so the version-tag cacheability gate
/// (typeobject.py:244-250) is applied here too: a type whose MRO is
/// not purely made of types keeps `_version_tag = None` (tag `0`,
/// uncacheable) — `mutated()` then never refreshes it.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_mro(obj: PyObjectRef, mro: Vec<PyObjectRef>) {
    let purely_of_types = is_mro_purely_of_types(&mro);
    (*(obj as *mut W_TypeObject)).mro_w = crate::object_array::alloc_mro_block_gc(&mro);
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    if !purely_of_types {
        w_type_set_version_tag(obj, 0);
    }
}

/// Restore the pre-`compute_mro` incomplete state (`mro_w is None`).
///
/// `typeobject.py mro_subclasses` keeps the old MRO value and may
/// restore `None` when a reentrant `__bases__` update fails while a type is
/// still being constructed.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_clear_mro(obj: PyObjectRef) {
    (*(obj as *mut W_TypeObject)).mro_w = std::ptr::null_mut();
    w_type_set_version_tag(obj, 0);
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// typeobject.py `is_mro_purely_of_types(mro_w)`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_mro_purely_of_types(mro_w: &[PyObjectRef]) -> bool {
    for &w_class in mro_w {
        if !is_type(w_class) {
            return false;
        }
    }
    true
}

/// Check if an object is a type (user-defined class).
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_type(obj: PyObjectRef) -> bool {
    py_type_check(obj, &TYPE_TYPE)
}

/// typeobject.py `is_heaptype(self)`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_is_heaptype(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).flag_heaptype
}

/// Set the CPython/PyPy heap-type ownership bit during type construction.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_heaptype(obj: PyObjectRef, value: bool) {
    (*(obj as *mut W_TypeObject)).flag_heaptype = value;
}

/// Publish the CPython 3.14 owner and mutability axes for a type whose PyPy
/// storage owner remains unchanged.  Construction sites call this once after
/// creating an interpreter builtin TypeDef; `PyType_Freeze` only changes the
/// immutable half through [`w_type_set_cpython_immutabletype`].
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_cpython_type_flags(
    obj: PyObjectRef,
    heaptype: bool,
    static_builtin: bool,
    immutabletype: bool,
) {
    let t = &mut *(obj as *mut W_TypeObject);
    t.flag_cpython_heaptype = heaptype;
    t.flag_cpython_static_builtin = static_builtin;
    t.flag_cpython_immutabletype
        .store(immutabletype, std::sync::atomic::Ordering::Release);
}

/// CPython 3.14 public HEAPTYPE ownership, distinct from PyPy's
/// [`w_type_is_heaptype`] implementation classification.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_is_cpython_heaptype(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).flag_cpython_heaptype
}

/// Read CPython's internal STATIC_BUILTIN owner bit.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_is_cpython_static_builtin(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).flag_cpython_static_builtin
}

/// Set only CPython's public IMMUTABLETYPE axis (used by `PyType_Freeze`).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_cpython_immutabletype(obj: PyObjectRef, value: bool) {
    (*(obj as *const W_TypeObject))
        .flag_cpython_immutabletype
        .store(value, std::sync::atomic::Ordering::Release);
}

/// Read CPython's public IMMUTABLETYPE axis.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_is_cpython_immutabletype(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject))
        .flag_cpython_immutabletype
        .load(std::sync::atomic::Ordering::Acquire)
}

/// Suppress CPython's public BASETYPE bit while preserving PyPy's canonical
/// layout-level subclassability field and all of its internal readers.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_suppress_cpython_basetype(obj: PyObjectRef) {
    (*(obj as *mut W_TypeObject)).flag_cpython_suppress_basetype = true;
}

/// typeobject.py `W_TypeObject.get_flags` — compute PyPy's public type flags
/// from their canonical fields on `W_TypeObject`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_flags(obj: PyObjectRef) -> i64 {
    if obj.is_null() || !is_type(obj) {
        return 0;
    }
    const STATIC_BUILTIN: i64 = 1 << 1;
    const HEAPTYPE: i64 = 1 << 9; // copy_reg._HEAPTYPE
    const INLINE_VALUES: i64 = 1 << 2;
    const MANAGED_WEAKREF: i64 = 1 << 3;
    const MANAGED_DICT: i64 = 1 << 4;
    const IMMUTABLETYPE: i64 = 1 << 8;
    const DISALLOW_INSTANTIATION: i64 = 1 << 7;
    const BASETYPE: i64 = 1 << 10;
    const READY: i64 = 1 << 12;
    const READYING: i64 = 1 << 13;
    const ABSTRACT: i64 = 1 << 20;
    const MATCH_SELF: i64 = 1 << 22;
    const HAVE_GC: i64 = 1 << 14;
    const PATMA_SEQUENCE: i64 = 1 << 5;
    const PATMA_MAPPING: i64 = 1 << 6;
    const METHOD_DESCRIPTOR: i64 = 1 << 17;
    const LONG_SUBCLASS: i64 = 1 << 24;
    const LIST_SUBCLASS: i64 = 1 << 25;
    const TUPLE_SUBCLASS: i64 = 1 << 26;
    const BYTES_SUBCLASS: i64 = 1 << 27;
    const UNICODE_SUBCLASS: i64 = 1 << 28;
    const DICT_SUBCLASS: i64 = 1 << 29;
    const BASE_EXC_SUBCLASS: i64 = 1 << 30;
    const TYPE_SUBCLASS: i64 = 1 << 31;

    let t = &*(obj as *const W_TypeObject);
    let mut flags = 0;
    if t.flag_cpython_static_builtin {
        flags |= STATIC_BUILTIN;
    }
    if t.flag_cpython_heaptype {
        flags |= HEAPTYPE;
    }

    if t.flag_cpython_heaptype {
        // [3.14-spec] `type_new_descriptors` represents a dict slot added by
        // a heap type with MANAGED_DICT (`typeobject.c`, read at v3.14.6).
        // PyPy records the same ownership boundary as
        // `self.hasdict and not self.layout.typedef.hasdict` when choosing a
        // DictTerminator (typeobject.py:255-257). Project those canonical
        // fields; do not infer ownership merely from the public capability,
        // since module instances own their dict in the builtin layout.
        let layout = t.layout;
        let typedef_hasdict = !layout.is_null() && (*layout).typedef_hasdict;
        if t.hasdict && !typedef_hasdict {
            flags |= MANAGED_DICT;
            // CPython's `type_ready_managed_dict` adds INLINE_VALUES only to
            // fixed-size managed-dict types.  This projection shares the
            // layout typedef used by `type.__itemsize__`.
            if !w_type_cpython_has_variable_items(obj) {
                flags |= INLINE_VALUES;
            }
        }
        if w_type_has_cpython_managed_weakref(obj) {
            flags |= MANAGED_WEAKREF;
        }
    }
    if t.flag_cpython_immutabletype
        .load(std::sync::atomic::Ordering::Acquire)
    {
        // `Py_TPFLAGS_IMMUTABLETYPE` (`object.h`, read at v3.14.6) is
        // orthogonal to HEAPTYPE: modern extension types commonly carry both.
        // PyPy's `descr__flags__` reports neither axis for builtin TypeDefs;
        // this field publishes CPython's observable split without changing
        // PyPy's internal type ownership.
        flags |= IMMUTABLETYPE;
    }
    if t.flag_abstract.load(std::sync::atomic::Ordering::Acquire) {
        flags |= ABSTRACT;
    }
    // [3.14-spec] CPython `type_ready` exposes READYING while a custom
    // metaclass's `mro()` is running, then replaces it with READY when the MRO
    // is installed (`Objects/typeobject.c:450-466, 8986-8991`). PyPy's
    // partially initialized `W_TypeObject` likewise has `mro_w is None`
    // (`typeobject.py:1080-1084`), which is pyre's canonical readiness state.
    if t.mro_w.is_null() {
        flags |= READYING;
    } else {
        flags |= READY;
    }
    if t.flag_disallow_instantiation
        .load(std::sync::atomic::Ordering::Acquire)
    {
        // [3.14-spec] CPython v3.14.6 `Include/object.h:540` assigns bit 7
        // to `Py_TPFLAGS_DISALLOW_INSTANTIATION`, and
        // `Objects/typeobject.c:1407` exposes the complete `tp_flags` word.
        // PyPy `typeobject.py:990-1004` publishes a smaller computed subset.
        // Preserve that field-by-field shape while exposing the canonical
        // flag that pyre's `type.__call__` already enforces.
        flags |= DISALLOW_INSTANTIATION;
    }
    if w_type_get_acceptable_as_base_class(obj) && !t.flag_cpython_suppress_basetype {
        // [3.14-spec] CPython v3.14.6 `Include/object.h:549` assigns bit 10
        // to `Py_TPFLAGS_BASETYPE`, and `Objects/typeobject.c:3638` uses that
        // same bit to accept or reject a base class. PyPy
        // `typeobject.py:990-1004` omits it from the public subset while
        // `typeobject.py:1116-1118` enforces the canonical
        // `layout.typedef.acceptable_as_base_class` value. Keep PyPy's
        // field-by-field shape and publish that existing value.
        flags |= BASETYPE;
    }
    if t.flag_have_gc {
        // [3.14-spec] CPython v3.14.6 exposes the complete `tp_flags` word
        // through `Objects/typeobject.c:1407`, and
        // `Include/object.h:567` assigns this bit to `Py_TPFLAGS_HAVE_GC`.
        // PyPy `typeobject.py:990-1004` computes a deliberately smaller
        // public subset. Keep its field-by-field `descr__flags__` shape, but
        // publish pyre's canonical per-type GC flag for the 3.14 surface.
        flags |= HAVE_GC;
    }
    if t.flag_method_descriptor {
        flags |= METHOD_DESCRIPTOR;
    }
    match t.flag_map_or_seq.load(std::sync::atomic::Ordering::Acquire) {
        b'M' => flags |= PATMA_MAPPING,
        // CPython deliberately omits the sequence-pattern flag from str,
        // bytes and bytearray (`unicodeobject.c:15805`,
        // `bytesobject.c:3118`, `bytearrayobject.c:2867`). Pyre keeps PyPy's
        // internal S marker on these types for `issequence_w`; MATCH_SEQUENCE
        // applies the same exclusion, so only the public bit is masked here.
        b'S' if !w_type_issubtype(obj, get_instantiate(&STR_TYPE))
            && !w_type_issubtype(obj, get_instantiate(&crate::bytesobject::BYTES_TYPE))
            && !w_type_issubtype(
                obj,
                get_instantiate(&crate::bytearrayobject::BYTEARRAY_TYPE),
            ) =>
        {
            flags |= PATMA_SEQUENCE
        }
        _ => {}
    }
    if w_type_has_match_self(obj) {
        // [3.14-spec] CPython `inherit_special` inherits MATCH_SELF from the
        // dominant base (`typeobject.c:8204-8206`). PyPy implements the same
        // observable class-pattern rule with its builtin atomic-type test;
        // `w_type_has_match_self` centralizes that MRO classification for the
        // opcode and this public flag.
        flags |= MATCH_SELF;
    }
    // [3.14-spec] CPython v3.14.6 `Objects/typeobject.c:8175-8200`
    // computes these mutually exclusive fast-subclass flags from the base
    // MRO in this exact order. PyPy does not publish them from
    // `descr__flags`, but its canonical classification is the MRO membership
    // scan in `typeobject.py:603/1640`, ported as `w_type_issubtype`.
    for (base, bit) in [
        (
            crate::interp_exceptions::lookup_exc_class_for_kind(
                crate::interp_exceptions::ExcKind::BaseException,
            ),
            BASE_EXC_SUBCLASS,
        ),
        (get_instantiate(&TYPE_TYPE), TYPE_SUBCLASS),
        (get_instantiate(&INT_TYPE), LONG_SUBCLASS),
        (
            get_instantiate(&crate::bytesobject::BYTES_TYPE),
            BYTES_SUBCLASS,
        ),
        (get_instantiate(&STR_TYPE), UNICODE_SUBCLASS),
        (get_instantiate(&TUPLE_TYPE), TUPLE_SUBCLASS),
        (get_instantiate(&LIST_TYPE), LIST_SUBCLASS),
        (get_instantiate(&DICT_TYPE), DICT_SUBCLASS),
    ] {
        if !base.is_null() && w_type_issubtype(obj, base) {
            flags |= bit;
            break;
        }
    }
    flags
}

/// typedef.py:43 `acceptable_as_base_class` — read from Layout level.
/// typeobject.py:1116: w_bestbase.layout.typedef.acceptable_as_base_class
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_acceptable_as_base_class(obj: PyObjectRef) -> bool {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        true
    } else {
        (*layout).acceptable_as_base_class
    }
}

/// typedef.py:40 `hasdict` — read from Layout level.
/// typeobject.py:255 `typedef = self.layout.typedef; ... not typedef.hasdict`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_get_typedef_hasdict(obj: PyObjectRef) -> bool {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        false
    } else {
        (*layout).typedef_hasdict
    }
}

/// Whether CPython 3.14 projects a non-zero `tp_itemsize` for this PyPy
/// instance layout.  The exact byte count remains in the interpreter's
/// `cpython_type_layout`; `type.__flags__` only needs the zero/non-zero split
/// used by `type_ready_managed_dict` to decide INLINE_VALUES.
unsafe fn w_type_cpython_has_variable_items(obj: PyObjectRef) -> bool {
    let typedef = w_type_get_layout(obj);
    std::ptr::eq(typedef, &TYPE_TYPE)
        || std::ptr::eq(typedef, &INT_TYPE)
        || std::ptr::eq(typedef, &LONG_TYPE)
        || std::ptr::eq(typedef, &TUPLE_TYPE)
        || std::ptr::eq(typedef, &crate::bytesobject::BYTES_TYPE)
        || std::ptr::eq(typedef, &crate::memoryview::MEMORYVIEW_TYPE)
}

/// CPython 3.14 `Py_TPFLAGS_MANAGED_WEAKREF`, projected through PyPy's
/// `weakrefable` inheritance and `find_best_base` layout owner.
///
/// A heap type whose best base is not weakrefable introduced the managed
/// weakref slot itself.  Otherwise the storage kind follows the best base:
/// another heap type propagates the managed flag, while a builtin owner such
/// as `type`, `set`, or `module` terminates the walk with an intrinsic slot.
/// `copy_flags_from_bases` may also obtain the capability from a compatible
/// secondary base; when the best base itself lacks it, CPython creates the
/// managed slot on the new type, which is the first arm below.
///
/// This deliberately follows the owner chain instead of testing whether the
/// `__weakref__` descriptor is still present: deleting a descriptor cannot
/// rewrite the immutable type-layout flag.
unsafe fn w_type_has_cpython_managed_weakref(obj: PyObjectRef) -> bool {
    if obj.is_null() || !is_type(obj) {
        return false;
    }
    let t = &*(obj as *const W_TypeObject);
    if !t.flag_heaptype || !t.weakrefable {
        return false;
    }
    let bestbase = find_best_base(obj);
    if bestbase.is_null() || !w_type_get_weakrefable(bestbase) {
        return true;
    }
    w_type_has_cpython_managed_weakref(bestbase)
}
/// Override acceptable_as_base_class by cloning the Layout.
/// typedef.py:742,765,664 explicit overrides after initial creation.
/// Layouts may be shared (reused from parent), so we clone to avoid
/// corrupting the parent type's flag.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_type_set_acceptable_as_base_class(obj: PyObjectRef, v: bool) {
    let old_layout = (*(obj as *const W_TypeObject)).layout;
    if old_layout.is_null() {
        return;
    }
    let old = &*old_layout;
    if old.acceptable_as_base_class == v {
        return; // already correct
    }
    // Clone with new value to avoid mutating shared Layout.
    let new_layout = leak_layout(Layout {
        typedef: old.typedef,
        nslots: old.nslots,
        newslotnames: old.newslotnames.clone(),
        base_layout: old.base_layout,
        acceptable_as_base_class: v,
        typedef_hasdict: old.typedef_hasdict,
    });
    (*(obj as *mut W_TypeObject)).layout = new_layout;
}

// ── Subclass tree (typeobject.py:640-689) ────────────────────────────

// `add_subclass` / `remove_subclass` / `get_subclasses` are indivisible under
// PyPy's GIL: each one reallocates or reindexes the same out-of-line
// `weak_subclasses` vector.  Pyre keeps the parent type as the sole semantic
// owner and uses the same narrow address-striped reentrant synchronization
// `w_list_lock` / `w_dict_lock` use around those transitions.  Reentrant
// because `get_subclasses` can be reached from inside a mutation on the same
// thread through the `mutated()` recursion.
struct ForkSubclassesLock(std::cell::UnsafeCell<parking_lot::ReentrantMutex<()>>);
unsafe impl Sync for ForkSubclassesLock {}

impl ForkSubclassesLock {
    fn new() -> Self {
        Self(std::cell::UnsafeCell::new(
            parking_lot::ReentrantMutex::new(()),
        ))
    }

    fn get(&self) -> &parking_lot::ReentrantMutex<()> {
        unsafe { &*self.0.get() }
    }

    unsafe fn reinit_after_fork(&self) {
        unsafe { self.0.get().write(parking_lot::ReentrantMutex::new(())) };
    }
}

static SUBCLASSES_LOCKS: std::sync::LazyLock<Vec<ForkSubclassesLock>> =
    std::sync::LazyLock::new(|| (0..256).map(|_| ForkSubclassesLock::new()).collect());

type SubclassesGuard = parking_lot::lock_api::ReentrantMutexGuard<
    'static,
    parking_lot::RawMutex,
    parking_lot::RawThreadId,
    (),
>;

/// Only the acquire is opaque to the tracer, the same split `w_list_lock` uses:
/// the guard-holding bodies stay look-inside.
#[majit_macros::dont_look_inside]
unsafe fn w_type_subclasses_lock(w_parent: PyObjectRef) -> SubclassesGuard {
    let lock = SUBCLASSES_LOCKS[(w_parent as usize >> 4) & (SUBCLASSES_LOCKS.len() - 1)].get();
    if let Some(guard) = lock.try_lock() {
        return guard;
    }
    let blocked = majit_gc::gc_sync::before_external_block();
    let guard = lock.lock();
    drop(blocked);
    guard
}

pub fn subclasses_locks_after_fork_child() {
    for lock in SUBCLASSES_LOCKS.iter() {
        unsafe { lock.reinit_after_fork() };
    }
}

/// `typeobject.py W_TypeObject.add_subclass`.
///
/// Records `w_subclass` in `w_parent.weak_subclasses` if not
/// already present.  Stores `weakref.ref(w_subclass)` via
/// `w_weakref_new` so subclass GC isn't blocked (`:642-650`); each
/// entry is a `try_gc_alloc` WEAKREF GcStruct, so the off-GC
/// `weak_subclasses` list is the WEAKREF's only strong root and must
/// be walked by the collector (`walk_builtin_type_dicts_gc` for builtin
/// types, the `W_TYPE_GC_TYPE_ID` custom trace for heap types).
///
/// # Safety
/// `w_parent` must point at a valid `W_TypeObject`.  `w_subclass`
/// likewise; the function does not type-check the argument since
/// `ready()` already filters non-type bases (`:374-376`).
pub unsafe fn w_type_add_subclass(w_parent: PyObjectRef, w_subclass: PyObjectRef) {
    if w_parent.is_null() || w_subclass.is_null() {
        return;
    }
    if !is_type(w_parent) || !is_type(w_subclass) {
        return;
    }
    // Serialize against a concurrent `remove_subclass` / `get_subclasses` /
    // `add_subclass` on the same parent: the null-check-then-install below and
    // the `push` reallocation both invalidate what another thread is indexing.
    let _subclasses_guard = w_type_subclasses_lock(w_parent);
    let parent = &mut *(w_parent as *mut W_TypeObject);
    if parent.weak_subclasses.is_null() {
        parent.weak_subclasses = Box::into_raw(Box::new(Vec::new()));
    }
    let subs = &mut *parent.weak_subclasses;
    // typeobject.py:651-660 — `newref = weakref.ref(w_subclass);
    // for i in range(...): if ref() is w_subclass: return; if ref()
    // is None: self.weak_subclasses[i] = newref; return;
    // else: self.weak_subclasses.append(newref)`.
    let newref = crate::weakref::w_weakref_new(w_subclass);
    for slot in subs.iter_mut() {
        let existing = crate::weakref::w_weakref_deref(*slot);
        if existing == w_subclass {
            return;
        }
        if existing.is_null() {
            *slot = newref;
            note_weak_subclass_store(w_parent);
            return;
        }
    }
    subs.push(newref);
    note_weak_subclass_store(w_parent);
}

/// Record the store of a freshly allocated weakref into `weak_subclasses`.
///
/// The mark covers builtin parents, whose list is off-GC and reachable only
/// through `walk_builtin_type_dicts_gc`; the write barrier covers GC-managed
/// heap parents, whose list is forwarded by the `W_TYPE_GC_TYPE_ID` custom
/// trace.
///
/// Order matters, and it is the safepoint that fixes it, not the allocation:
/// host-side allocation cannot collect (`dynasm_alloc_nursery_typed` routes to
/// `try_alloc_nursery_no_collect_typed` and spills to old-gen on nursery full),
/// but `try_gc_write_barrier` reaches `gc_sync::gc_op`, which leaves RUNNING and
/// parks on `gc_mutex` — an entry-style safepoint where another thread's
/// stop-the-world collection runs.  The dirty bit is consumable
/// (`gc_roots::clear_prebuilt_roots_dirty` after each walk), so marking on the
/// far side of that safepoint would let a collection walk the prebuilt family
/// with the slot already updated and the bit still clear, and nothing else roots
/// the young weakref.  Mark first, then take the barrier.
#[inline]
unsafe fn note_weak_subclass_store(w_parent: PyObjectRef) {
    crate::gc_roots::mark_prebuilt_roots_dirty();
    crate::gc_hook::try_gc_write_barrier(w_parent as *mut u8);
}

/// `typeobject.py W_TypeObject.remove_subclass`.
///
/// Removes `w_subclass` from `w_parent.weak_subclasses` if
/// present; no-op otherwise.  Pointer equality matches PyPy's
/// `ref() is w_subclass`.
///
/// # Safety
/// Same as [`w_type_add_subclass`].
pub unsafe fn w_type_remove_subclass(w_parent: PyObjectRef, w_subclass: PyObjectRef) {
    if w_parent.is_null() || w_subclass.is_null() {
        return;
    }
    if !is_type(w_parent) {
        return;
    }
    let _subclasses_guard = w_type_subclasses_lock(w_parent);
    let parent = &mut *(w_parent as *mut W_TypeObject);
    if parent.weak_subclasses.is_null() {
        return;
    }
    let subs = &mut *parent.weak_subclasses;
    // typeobject.py:665-669 — `for i in range(len(self
    // .weak_subclasses)): ref = self.weak_subclasses[i]; if ref()
    // is w_subclass: del self.weak_subclasses[i]; return`.
    for i in 0..subs.len() {
        if crate::weakref::w_weakref_deref(subs[i]) == w_subclass {
            subs.remove(i);
            return;
        }
    }
}

/// `typeobject.py W_TypeObject.get_subclasses`.
///
/// Returns the recorded direct subclasses.  Under PyPy's weakref
/// path, dead refs are filtered; pyre's strong-ref fallback has
/// no dead entries to filter so the result is a copy of the
/// stored vector.  `only_real_subclasses` mirrors PyPy's filter for
/// `descr___subclasses__`: custom-MRO ancestors are registered for cache
/// invalidation but are not real entries in the subclass's `__bases__`.
///
/// # Safety
/// `w_parent` must point at a valid `W_TypeObject`.
pub unsafe fn w_type_get_subclasses(
    w_parent: PyObjectRef,
    only_real_subclasses: bool,
) -> Vec<PyObjectRef> {
    if w_parent.is_null() || !is_type(w_parent) {
        return Vec::new();
    }
    let _subclasses_guard = w_type_subclasses_lock(w_parent);
    let parent = &*(w_parent as *const W_TypeObject);
    if parent.weak_subclasses.is_null() {
        return Vec::new();
    }
    // typeobject.py:683-686 — `for ref in self.weak_subclasses: w_ob
    // = ref(); if w_ob is not None: subclasses_w.append(w_ob)`.
    let subs = &*parent.weak_subclasses;
    let mut alive: Vec<PyObjectRef> = Vec::with_capacity(subs.len());
    for &slot in subs.iter() {
        let target = crate::weakref::w_weakref_deref(slot);
        if !target.is_null() {
            if only_real_subclasses {
                let bases = w_type_get_bases(target);
                if bases.is_null()
                    || !(0..crate::tupleobject::w_tuple_len(bases)).any(|index| {
                        crate::tupleobject::w_tuple_getitem(bases, index as i64)
                            .is_some_and(|base| std::ptr::eq(base, w_parent))
                    })
                {
                    continue;
                }
            }
            alive.push(target);
        }
    }
    alive
}

/// `typeobject.py W_TypeObject.ready` — register `w_self`
/// as a direct subclass on each W_TypeObject base.  Called once
/// per heap type after `bases` is set, so the subclass tree
/// reflects the class declaration before any attribute lookup.
///
/// # Safety
/// `w_self.bases` must be a valid tuple (or `PY_NULL`).
pub unsafe fn w_type_ready(w_self: PyObjectRef) {
    if w_self.is_null() || !is_type(w_self) {
        return;
    }
    let bases = (*(w_self as *const W_TypeObject)).bases;
    if bases.is_null() {
        return;
    }
    let n = crate::w_tuple_len(bases);
    for i in 0..n as i64 {
        let Some(w_base) = crate::w_tuple_getitem(bases, i) else {
            continue;
        };
        if w_base.is_null() || !is_type(w_base) {
            continue;
        }
        w_type_add_subclass(w_base, w_self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_create_and_check() {
        let obj = w_type_new("Foo", PY_NULL, std::ptr::null_mut());
        unsafe {
            assert!(is_type(obj));
            assert!(!is_int(obj));
            assert_eq!(w_type_get_name(obj), "Foo");
            assert!(w_type_get_dict_ptr(obj).is_null());
        }
    }

    #[test]
    fn test_layout_issublayout() {
        let root = leak_layout(Layout {
            typedef: &INSTANCE_TYPE,
            nslots: 0,
            newslotnames: vec![],
            base_layout: std::ptr::null(),
            acceptable_as_base_class: true,
            typedef_hasdict: false,
        });
        let child = leak_layout(Layout {
            typedef: &INSTANCE_TYPE,
            nslots: 1,
            newslotnames: vec!["x".to_string()],
            base_layout: root,
            acceptable_as_base_class: true,
            typedef_hasdict: false,
        });
        unsafe {
            assert!((*child).issublayout(root));
            assert!((*root).issublayout(root));
            assert!(!(*root).issublayout(child));
        }
    }

    #[test]
    fn test_layout_expand_equality() {
        let root = leak_layout(Layout {
            typedef: &INSTANCE_TYPE,
            nslots: 1,
            newslotnames: vec!["x".to_string()],
            base_layout: std::ptr::null(),
            acceptable_as_base_class: true,
            typedef_hasdict: false,
        });
        // Same Layout pointer → equal
        assert!(Layout::expands_equal(root, true, true, root, true, true));
        // Different hasdict → not equal
        assert!(!Layout::expands_equal(root, true, true, root, false, true));
    }

    #[test]
    fn type_flags_project_managed_dict_and_inline_values_from_layout_owner() {
        const INLINE_VALUES: i64 = 1 << 2;
        const MANAGED_DICT: i64 = 1 << 4;
        const BASETYPE: i64 = 1 << 10;
        const MASK: i64 = INLINE_VALUES | MANAGED_DICT;

        unsafe fn heap_type_with_layout(
            typedef: *const PyType,
            typedef_hasdict: bool,
        ) -> PyObjectRef {
            let w_type = w_type_new("C", PY_NULL, std::ptr::null_mut());
            let layout = leak_layout(Layout {
                typedef,
                nslots: 0,
                newslotnames: vec![],
                base_layout: std::ptr::null(),
                acceptable_as_base_class: true,
                typedef_hasdict,
            });
            w_type_set_layout(w_type, layout);
            w_type_set_hasdict(w_type, true);
            w_type
        }

        unsafe {
            // A normal heap instance uses PyPy's DictTerminator and CPython's
            // fixed-size inline-values representation.
            let plain = heap_type_with_layout(&INSTANCE_TYPE, false);
            assert_eq!(w_type_get_flags(plain) & MASK, MASK);

            // An app-level PyPy class may stand in for a CPython static
            // builtin.  Keep its internal heap owner, but do not publish the
            // managed-layout bits CPython only assigns to heap types.
            w_type_set_cpython_type_flags(plain, false, true, true);
            assert!(w_type_is_heaptype(plain));
            assert_eq!(w_type_get_flags(plain) & MASK, 0);
            assert_ne!(w_type_get_flags(plain) & BASETYPE, 0);
            w_type_suppress_cpython_basetype(plain);
            assert_eq!(w_type_get_flags(plain) & BASETYPE, 0);

            // A tuple subtype still owns a managed dict, but a non-zero
            // CPython tp_itemsize excludes INLINE_VALUES.
            let tuple_subclass = heap_type_with_layout(&TUPLE_TYPE, false);
            assert_eq!(w_type_get_flags(tuple_subclass) & MASK, MANAGED_DICT);

            // A builtin typedef such as module owns its dict directly, so a
            // heap subtype sharing that layout has neither managed bit.
            let native_dict = heap_type_with_layout(&MODULE_TYPE, true);
            assert_eq!(w_type_get_flags(native_dict) & MASK, 0);
        }
    }

    #[test]
    fn type_flags_project_managed_weakref_through_the_best_base_owner() {
        const MANAGED_WEAKREF: i64 = 1 << 3;
        let layout = leak_layout(Layout {
            typedef: &INSTANCE_TYPE,
            nslots: 0,
            newslotnames: vec![],
            base_layout: std::ptr::null(),
            acceptable_as_base_class: true,
            typedef_hasdict: false,
        });

        unsafe fn builtin_with_layout(name: &str, layout: *const Layout) -> PyObjectRef {
            let w_type = w_type_new_builtin(name, PY_NULL, std::ptr::null_mut(), &INSTANCE_TYPE);
            w_type_set_layout(w_type, layout);
            w_type
        }

        unsafe fn heap_with_base(
            name: &str,
            base: PyObjectRef,
            layout: *const Layout,
        ) -> PyObjectRef {
            let bases = crate::w_tuple_new(vec![base]);
            let w_type = w_type_new(name, bases, std::ptr::null_mut());
            w_type_set_layout(w_type, layout);
            w_type
        }

        unsafe {
            let object = builtin_with_layout("object", layout);
            let plain = heap_with_base("Plain", object, layout);
            w_type_set_weakrefable(plain, true);
            assert_ne!(w_type_get_flags(plain) & MANAGED_WEAKREF, 0);

            let derived = heap_with_base("Derived", plain, layout);
            w_type_set_weakrefable(derived, true);
            assert_ne!(w_type_get_flags(derived) & MANAGED_WEAKREF, 0);

            let intrinsic = builtin_with_layout("intrinsic", layout);
            w_type_set_weakrefable(intrinsic, true);
            let intrinsic_subclass = heap_with_base("IntrinsicChild", intrinsic, layout);
            w_type_set_weakrefable(intrinsic_subclass, true);
            assert_eq!(w_type_get_flags(intrinsic_subclass) & MANAGED_WEAKREF, 0);
        }
    }

    #[test]
    fn type_flags_keep_cpython_owner_and_immutability_orthogonal() {
        const STATIC_BUILTIN: i64 = 1 << 1;
        const IMMUTABLETYPE: i64 = 1 << 8;
        const HEAPTYPE: i64 = 1 << 9;
        const MASK: i64 = STATIC_BUILTIN | IMMUTABLETYPE | HEAPTYPE;

        unsafe {
            // `_PyStaticType_InitBuiltin`: interpreter-owned static builtin.
            let core = w_type_new_builtin("core", PY_NULL, std::ptr::null_mut(), &INSTANCE_TYPE);
            assert_eq!(
                w_type_get_flags(core) & MASK,
                STATIC_BUILTIN | IMMUTABLETYPE
            );

            // `PyType_FromModuleAndSpec`: a heap-owned but immutable extension
            // type.  Its PyPy implementation owner remains a builtin TypeDef.
            let extension =
                w_type_new_builtin("extension", PY_NULL, std::ptr::null_mut(), &INSTANCE_TYPE);
            w_type_set_cpython_type_flags(extension, true, false, true);
            assert!(!w_type_is_heaptype(extension));
            assert_eq!(w_type_get_flags(extension) & MASK, HEAPTYPE | IMMUTABLETYPE);

            // A legacy static extension readied through public PyType_Ready is
            // neither a core static builtin nor a heap type, but is immutable.
            w_type_set_cpython_type_flags(extension, false, false, true);
            assert_eq!(w_type_get_flags(extension) & MASK, IMMUTABLETYPE);

            // An app-level class remains mutable and heap-owned.
            let user = w_type_new("User", PY_NULL, std::ptr::null_mut());
            assert_eq!(w_type_get_flags(user) & MASK, HEAPTYPE);
        }
    }

    #[test]
    fn w_type_gc_type_id_matches_descr() {
        assert_eq!(W_TYPE_GC_TYPE_ID, 33);
        assert_eq!(
            <W_TypeObject as crate::lltype::GcType>::type_id(),
            W_TYPE_GC_TYPE_ID
        );
        assert_eq!(
            <W_TypeObject as crate::lltype::GcType>::SIZE,
            W_TYPE_OBJECT_SIZE
        );
    }

    /// The `_version_tag?` wiring: publishing a new tag runs the invalidation
    /// function (`quasiimmut.py _invalidate_now`), so the loops that
    /// baked the old tag are revoked and the field is left uninstalled.
    /// `quasiimmut::tests` covers the field itself.
    #[test]
    fn version_tag_write_unlinks_the_instance_and_revokes_its_loops() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let obj = w_type_new("Quasi", PY_NULL, std::ptr::null_mut());
        let w_type = unsafe { &*(obj as *const W_TypeObject) };
        assert!(
            !w_type.quasi_immut_watchers.is_installed(),
            "no instance until the first registration",
        );

        let flag = Arc::new(AtomicBool::new(false));
        unsafe { w_type_current_qmut_instance(obj) }
            .expect("a type resolves an instance")
            .register_loop_token(&flag);
        assert!(w_type.quasi_immut_watchers.is_installed());

        unsafe { w_type_set_version_tag(obj, new_version_tag()) };
        assert!(flag.load(Ordering::Acquire), "the loop must be revoked");
        assert!(
            !w_type.quasi_immut_watchers.is_installed(),
            "the field is nulled before the sweep",
        );

        // A second bump with nothing registered must not double-free.
        unsafe { w_type_set_version_tag(obj, new_version_tag()) };
    }

    /// A non-type pointer must not be walked as one.
    #[test]
    fn quasi_immut_watcher_helpers_ignore_non_types() {
        unsafe {
            assert!(w_type_current_qmut_instance(PY_NULL).is_none());
            w_type_notify_quasi_immut_watchers(PY_NULL);
        }
    }

    #[test]
    fn new_version_tag_is_distinct_and_nonzero() {
        let a = new_version_tag();
        let b = new_version_tag();
        assert_ne!(a, 0);
        assert_ne!(b, 0);
        assert!(b > a);
    }

    #[test]
    fn fresh_type_carries_version_tag_with_round_trip() {
        let obj = w_type_new("Foo", PY_NULL, std::ptr::null_mut());
        unsafe {
            // typeobject.py:250 — a fresh type is minted with a non-None tag.
            assert_ne!(w_type_get_version_tag(obj), 0);
            // typeobject.py — mutated() stores a fresh tag.
            let fresh = new_version_tag();
            w_type_set_version_tag(obj, fresh);
            assert_eq!(w_type_get_version_tag(obj), fresh);
        }
    }

    #[test]
    fn fresh_type_uses_object_flags_default_false_with_round_trip() {
        let obj = w_type_new("Bar", PY_NULL, std::ptr::null_mut());
        unsafe {
            // typeobject.py:185-186 — conservative `False` default.
            assert!(!w_type_get_uses_object_getattribute(obj));
            assert!(!w_type_get_uses_object_setattr(obj));
            // typeobject.py/340 — confirmed-default lookup sets the flag.
            w_type_set_uses_object_getattribute(obj, true);
            w_type_set_uses_object_setattr(obj, true);
            assert!(w_type_get_uses_object_getattribute(obj));
            assert!(w_type_get_uses_object_setattr(obj));
        }
        // null / non-type tolerated, reads the conservative default.
        unsafe {
            assert!(!w_type_get_uses_object_getattribute(PY_NULL));
            assert!(!w_type_get_uses_object_setattr(PY_NULL));
        }
    }

    /// Concurrent class creation against one process-global builtin parent is
    /// the shape the striped lock exists for: `add_subclass`'s `push`
    /// reallocates and frees the buffer `get_subclasses` is indexing, and its
    /// null-check-then-install lets two threads each install a `Box`.
    ///
    /// Without `w_type_subclasses_lock` this aborts inside the allocator —
    /// `double free or corruption (fasttop)` on glibc, `STATUS_HEAP_CORRUPTION`
    /// on Windows, SIGSEGV in `w_weakref_deref` on macOS. Delete the three
    /// guards and re-run to confirm the gate still bites before trusting it.
    #[test]
    fn subclass_registry_survives_concurrent_mutation() {
        let w_parent = w_type_new("SharedParent", PY_NULL, std::ptr::null_mut());
        let parent_addr = w_parent as usize;

        std::thread::scope(|scope| {
            for t in 0..4 {
                scope.spawn(move || {
                    let w_parent = parent_addr as PyObjectRef;
                    // Kept alive for the whole thread so `weak_subclasses`
                    // holds live entries rather than immediately-dead refs.
                    let children: Vec<PyObjectRef> = (0..16)
                        .map(|i| {
                            w_type_new(&format!("Child{t}_{i}"), PY_NULL, std::ptr::null_mut())
                        })
                        .collect();
                    for _ in 0..500 {
                        for &w_child in &children {
                            unsafe { w_type_add_subclass(w_parent, w_child) };
                        }
                        // Reads and dereferences every entry in the vector the
                        // other threads are reallocating — a stale buffer shows
                        // up here as a garbage `*mut Weakref`. The contents are
                        // racy by construction, so only the read is asserted on.
                        let seen = unsafe { w_type_get_subclasses(w_parent, false) };
                        assert!(seen.iter().all(|w| !w.is_null()));
                        for &w_child in &children {
                            unsafe { w_type_remove_subclass(w_parent, w_child) };
                        }
                    }
                });
            }
        });

        // Every thread removed everything it added, and no entry outlived it.
        let leftover = unsafe { w_type_get_subclasses(w_parent, false) };
        assert!(leftover.is_empty(), "{} entries leaked", leftover.len());
    }
}
