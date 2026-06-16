//! W_TypeObject — Python `type` object for user-defined classes.
//!
//! PyPy equivalent: pypy/objspace/std/typeobject.py → W_TypeObject
//!
//! A type object holds the class name, tuple of base types, and a namespace
//! dict containing class-level attributes and methods.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// typeobject.py:103-129 Layout object.
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
    /// typedef.py:43 — `acceptable_as_base_class = '__new__' in rawdict`.
    /// TODO: in RPython this lives on TypeDef, accessed
    /// via `layout.typedef.acceptable_as_base_class`. Stored on Layout
    /// here because Rust has no TypeDef struct yet — Layout.typedef is
    /// `*const PyType` (≈ CLASSTYPE), and many types share INSTANCE_TYPE
    /// but need different acceptable_as_base_class values.
    /// Convergence: introduce a Rust TypeDef struct, move this field there.
    pub acceptable_as_base_class: bool,
    /// typedef.py:40 — `hasdict = '__dict__' in rawdict`: whether the
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
    /// typeobject.py:118-123 issublayout(parent):
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

    /// typeobject.py:125-129 expand(hasdict, weakrefable):
    ///   return (self.typedef, self.newslotnames, self.base_layout,
    ///           hasdict, weakrefable)
    ///
    /// Two types have compatible layouts iff their expand() tuples are equal.
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
    /// Tuple of base type objects (PyObjectRef → W_TupleObject or PY_NULL).
    pub bases: PyObjectRef,
    /// Raw pointer to the class dict backing storage (`dict_w` analogue).
    pub dict: *mut u8,
    /// Cached C3 MRO — W_TypeObject.mro_w.
    pub mro_w: *mut Vec<PyObjectRef>,
    /// typeobject.py:184 `flag_heaptype` — immutable after creation.
    pub flag_heaptype: bool,
    /// typeobject.py:195 `layout` — pointer to shared Layout object.
    pub layout: *const Layout,
    /// typeobject.py:179 `hasdict` — True when instances have __dict__.
    pub hasdict: bool,
    /// typeobject.py:181 `weakrefable` — True when instances support weakrefs.
    pub weakrefable: bool,
    /// typeobject.py:169 `flag_map_or_seq` (`'?'`, `'M'`, `'S'`).
    ///
    /// Default `'?'` per typeobject.py:216.  Inherited from base
    /// classes during heap-type construction (typeobject.py:1495):
    /// when self's flag is `'?'` and a base's flag is non-`'?'`, copy.
    /// Used by `descroperation.py:319-326 is_iterable` and `:330-346
    /// iter` to skip the `__getitem__` fallback for mapping-typed
    /// classes.  Stored on `W_TypeObject` (not the low-level
    /// `PyType`) so user-defined `dict`/`list`/`tuple` subclasses
    /// inherit the marker the same way PyPy does.
    pub flag_map_or_seq: std::sync::atomic::AtomicU8,
    /// typeobject.py:171 `compares_by_identity_status?` —
    /// `UNKNOWN=0`, `COMPARES_BY_IDENTITY=1`,
    /// `OVERRIDES_EQ_CMP_OR_HASH=2`.  Cached result of
    /// `W_TypeObject.compares_by_identity` (`:353-371`); UNKNOWN
    /// until first lookup forces a `__eq__` / `__hash__` MRO walk.
    ///
    /// Invalidated by `baseobjspace::setattr_str` /
    /// `baseobjspace::delattr_str` whenever a type-dict entry changes
    /// (matches `typeobject.py:280 mutated()`), which walks
    /// `weak_subclasses` and recurses, so a base-class mutation
    /// eagerly resets cached subclasses.
    pub compares_by_identity_status: std::sync::atomic::AtomicU8,
    /// typeobject.py:640-689 `weak_subclasses` —
    /// per-type list of subclass references populated by
    /// `add_subclass` at heaptype creation time
    /// (`typeobject.py:373-377 ready()` and
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
    /// layer. Mirrors `W_InstanceObject.map`.
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
    /// conservative default (typeobject.py:185, 275); `mutated()` resets
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
}

/// Source of fresh `version_tag` identities (`VersionTag()`, typeobject.py:73).
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

/// Leak a Layout to get a 'static pointer for sharing.
pub fn leak_layout(layout: Layout) -> *const Layout {
    crate::lltype::malloc_raw(layout)
}

thread_local! {
    /// Heap type objects (`w_type_new` — user `class` statements and
    /// `type(name, bases, dict)`) are `malloc_typed` Box-immortal, so the
    /// collector never fires their `W_TYPE_GC_TYPE_ID` custom trace and
    /// therefore never reaches the movable values bound in a type's
    /// namespace dict (methods, class attributes, the per-type
    /// `__dict__`/`__weakref__` getset copies) nor the `bases` tuple.
    /// This registry lets the interpreter root every heap type's namespace
    /// as a pinned root source on each collection — the same shape
    /// `walk_module_dicts_gc` uses for Box-immortal module dicts.
    ///
    /// Builtin types (`w_type_new_builtin`) are created before the GC is
    /// built and only ever hold Box-immortal values, so they are not
    /// registered.  Append-only interim: heap types are themselves immortal,
    /// so a recorded address stays valid for the process lifetime (unlike a
    /// GC-managed object, an immortal type is never freed, so no stale-address
    /// pruning is needed).  Convergence path: GC-manage `W_TypeObject` so its
    /// custom trace fires and this walk is deleted.
    static HEAP_TYPE_REGISTRY: std::cell::RefCell<Vec<usize>> =
        std::cell::RefCell::new(Vec::new());
}

/// Record a heap type for the collection-time namespace root walk.
fn register_heap_type(addr: usize) {
    HEAP_TYPE_REGISTRY.with(|reg| reg.borrow_mut().push(addr));
}

/// Snapshot the registered heap-type addresses for the root walker
/// (`pyre_interpreter::eval::walk_type_dicts_gc`).
pub fn snapshot_heap_types() -> Vec<usize> {
    HEAP_TYPE_REGISTRY.with(|reg| reg.borrow().clone())
}

/// Allocate a new W_TypeObject with `flag_heaptype = true`.
///
/// typeobject.py:174 `__init__(..., is_heaptype=True)`.
/// Layout is set to null initially; caller must set it via set_layout
/// after running create_all_slots / setup_builtin_type.
pub fn w_type_new(name: &str, bases: PyObjectRef, dict_ptr: *mut u8) -> PyObjectRef {
    let name = crate::lltype::malloc_raw(name.to_string());
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(bases);

    let w_type = crate::lltype::malloc_typed(W_TypeObject {
        ob_header: PyObject {
            ob_type: &TYPE_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        mro_w: std::ptr::null_mut(),
        name,
        bases,
        dict: dict_ptr,
        flag_heaptype: true,
        layout: std::ptr::null(),
        hasdict: false,
        weakrefable: false,
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
    }) as PyObjectRef;
    register_heap_type(w_type as usize);
    w_type
}

/// typeobject.py:1507-1508 in setup_user_defined_type — copy
/// `flag_map_or_seq` from the first base whose flag is non-`?`.
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
    let name = crate::lltype::malloc_raw(name.to_string());
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(bases);

    crate::lltype::malloc_typed(W_TypeObject {
        ob_header: PyObject {
            ob_type: &TYPE_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        mro_w: std::ptr::null_mut(),
        name,
        bases,
        dict: dict_ptr,
        flag_heaptype: false,
        layout: std::ptr::null(),
        hasdict: false,
        weakrefable: false,
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
        // typeobject.py:256 — `typedef.method_descriptor` (typedef.py:22
        // default `False`); the `function` creation site flips it
        // (typedef.py:807).
        flag_method_descriptor: false,
        // `Py_TPFLAGS_DISALLOW_INSTANTIATION` off by default; the
        // generator / coroutine / frame typedefs flip it via
        // `w_type_set_disallow_instantiation`.
        flag_disallow_instantiation: std::sync::atomic::AtomicBool::new(false),
    }) as PyObjectRef
}

/// `dictmultiobject.py:153 UNKNOWN` — cache miss; recompute via
/// `compares_by_identity` lookup.
pub const COMPARES_BY_IDENTITY_UNKNOWN: u8 = 0;
/// `dictmultiobject.py:154 COMPARES_BY_IDENTITY` — type uses
/// object-default `__eq__`/`__hash__`; identity comparison is
/// observable-equivalent.
pub const COMPARES_BY_IDENTITY_YES: u8 = 1;
/// `dictmultiobject.py:155 OVERRIDES_EQ_CMP_OR_HASH` — type defines a
/// custom `__eq__` or `__hash__`; identity comparison is not safe.
pub const COMPARES_BY_IDENTITY_NO: u8 = 2;

/// `typeobject.py:353-371 W_TypeObject.compares_by_identity` —
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

/// typeobject.py:169 — `flag_map_or_seq` accessor on a `W_TypeObject`.
/// Returns `'?'` if `w_type` is null, not a type object, or never had
/// the marker assigned.
pub unsafe fn w_type_get_flag_map_or_seq(w_type: PyObjectRef) -> u8 {
    if w_type.is_null() || !is_type(w_type) {
        return b'?';
    }
    let t = &*(w_type as *const W_TypeObject);
    t.flag_map_or_seq.load(std::sync::atomic::Ordering::Acquire)
}

/// typeobject.py:169 — `flag_map_or_seq` setter.  Used by
/// `init_typeobjects` to mark dict / list / tuple W_TypeObjects at
/// registration time (objspace.py:104-108).
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

// ── Layout accessors ─────────────────────────────────────────────────

/// Set the Layout pointer on a type object.
pub unsafe fn w_type_set_layout(obj: PyObjectRef, layout: *const Layout) {
    (*(obj as *mut W_TypeObject)).layout = layout;
}

/// Get the Layout pointer from a type object.
pub unsafe fn w_type_get_layout_ptr(obj: PyObjectRef) -> *const Layout {
    (*(obj as *const W_TypeObject)).layout
}

/// typeobject.py:336-337 get_full_instance_layout(self).
/// Returns the Layout.typedef pointer (the PyType describing instance struct).
/// For backward-compat with existing code that compares PyType pointers.
#[inline]
pub unsafe fn w_type_get_layout(obj: PyObjectRef) -> *const PyType {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        &INSTANCE_TYPE as *const PyType
    } else {
        (*layout).typedef
    }
}

/// Get nslots from the Layout.
pub unsafe fn w_type_get_nslots(obj: PyObjectRef) -> u32 {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        0
    } else {
        (*layout).nslots
    }
}

/// Get newslotnames from the Layout.
pub unsafe fn w_type_get_newslotnames(obj: PyObjectRef) -> &'static [String] {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        &[]
    } else {
        &(*layout).newslotnames
    }
}

/// Get base_layout pointer for identity comparison.
pub unsafe fn w_type_get_base_layout(obj: PyObjectRef) -> *const Layout {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        std::ptr::null()
    } else {
        (*layout).base_layout
    }
}

/// typeobject.py:197 `flag_method_descriptor` getter/setter
/// (callmethod.py:66 `space.type(w_descr).flag_method_descriptor`).
pub unsafe fn w_type_get_flag_method_descriptor(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).flag_method_descriptor
}
pub unsafe fn w_type_set_flag_method_descriptor(obj: PyObjectRef, v: bool) {
    (*(obj as *mut W_TypeObject)).flag_method_descriptor = v;
}

/// typeobject.py:179 `hasdict` getter/setter.
pub unsafe fn w_type_get_hasdict(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).hasdict
}
pub unsafe fn w_type_set_hasdict(obj: PyObjectRef, v: bool) {
    (*(obj as *mut W_TypeObject)).hasdict = v;
}

/// typeobject.py:295 `self._version_tag` — the raw cache-version field
/// (`0` = `None`/uncacheable).  This is the direct field read; the
/// `we_are_jitted()` / `_pure_version_tag` (`@elidable_promote`) split of
/// `version_tag()` (typeobject.py:293-301) lives in the interpreter layer
/// (`baseobjspace::w_type_version_tag`), which has the JIT intrinsics.
pub unsafe fn w_type_get_version_tag(obj: PyObjectRef) -> u64 {
    (*(obj as *const W_TypeObject))
        .version_tag
        .load(std::sync::atomic::Ordering::Acquire)
}
/// Store a new version-tag identity (typeobject.py:286 `mutated`).
pub unsafe fn w_type_set_version_tag(obj: PyObjectRef, v: u64) {
    (*(obj as *const W_TypeObject))
        .version_tag
        .store(v, std::sync::atomic::Ordering::Release);
}

/// typeobject.py:183-185 `uses_object_getattribute` reader.  Returns the
/// conservative `false` for a null / non-type pointer (matches the class
/// default before any lookup confirms the flag).
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
pub unsafe fn w_type_set_uses_object_setattr(obj: PyObjectRef, v: bool) {
    if obj.is_null() || !is_type(obj) {
        return;
    }
    (*(obj as *const W_TypeObject))
        .uses_object_setattr
        .store(v, std::sync::atomic::Ordering::Release);
}

/// typeobject.py:179 `terminator` getter/setter. The stored value is an
/// erased `*const MapNode`; the `pyre-interpreter` mapdict layer casts it.
pub unsafe fn w_type_get_terminator(obj: PyObjectRef) -> *const u8 {
    (*(obj as *const W_TypeObject)).terminator
}
pub unsafe fn w_type_set_terminator(obj: PyObjectRef, terminator: *const u8) {
    (*(obj as *mut W_TypeObject)).terminator = terminator;
}

/// typeobject.py:181 `weakrefable` getter/setter.
pub unsafe fn w_type_get_weakrefable(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).weakrefable
}
pub unsafe fn w_type_set_weakrefable(obj: PyObjectRef, v: bool) {
    (*(obj as *mut W_TypeObject)).weakrefable = v;
}

// ── Other accessors ──────────────────────────────────────────────────

/// Get the class name.
pub unsafe fn w_type_get_name(obj: PyObjectRef) -> &'static str {
    &*(*(obj as *const W_TypeObject)).name
}

/// Replace the class name (`descr_set__name__`, typeobject.py:1058
/// `w_type.name = name`).  `name` is an owned `String` behind a raw
/// pointer (`malloc_raw` = boxed); assigning through it drops the old
/// name and installs the new one, leaving the slot itself unchanged.
pub unsafe fn w_type_set_name(obj: PyObjectRef, name: &str) {
    *(*(obj as *mut W_TypeObject)).name = name.to_string();
}

/// Get the bases tuple.
pub unsafe fn w_type_get_bases(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_TypeObject)).bases
}

/// Get the class namespace pointer (as *mut u8).
pub unsafe fn w_type_get_dict_ptr(obj: PyObjectRef) -> *mut u8 {
    (*(obj as *const W_TypeObject)).dict
}

/// Get the cached MRO, or null if not yet set.
pub unsafe fn w_type_get_mro(obj: PyObjectRef) -> *mut Vec<PyObjectRef> {
    (*(obj as *const W_TypeObject)).mro_w
}

/// Set the cached MRO.
///
/// Construction installs the MRO here (rather than in `__init__`
/// itself, typeobject.py:244), so the version-tag cacheability gate
/// (typeobject.py:244-250) is applied here too: a type whose MRO is
/// not purely made of types keeps `_version_tag = None` (tag `0`,
/// uncacheable) — `mutated()` then never refreshes it.
pub unsafe fn w_type_set_mro(obj: PyObjectRef, mro: Vec<PyObjectRef>) {
    let purely_of_types = is_mro_purely_of_types(&mro);
    (*(obj as *mut W_TypeObject)).mro_w = crate::lltype::malloc_raw(mro);
    if !purely_of_types {
        w_type_set_version_tag(obj, 0);
    }
}

/// typeobject.py:1615-1619 `is_mro_purely_of_types(mro_w)`.
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
pub unsafe fn is_type(obj: PyObjectRef) -> bool {
    py_type_check(obj, &TYPE_TYPE)
}

/// typeobject.py:543-544 `is_heaptype(self)`.
#[inline]
pub unsafe fn w_type_is_heaptype(obj: PyObjectRef) -> bool {
    (*(obj as *const W_TypeObject)).flag_heaptype
}

/// typeobject.py:866 `get_flags(self)` — the `__flags__` bitmask.
///
/// `_HEAPTYPE = 1<<9`, `_CPYTYPE = 1` (non-heap builtin types),
/// `PATMA_SEQUENCE = 1<<5`, `PATMA_MAPPING = 1<<6`,
/// `DISALLOW_INSTANTIATION = 1<<7`.  pyre tracks `flag_heaptype`,
/// `flag_map_or_seq` and `flag_disallow_instantiation`; the cpytype bit
/// follows `!flag_heaptype` (every non-heap type is a builtin C type).
/// The abstract / method-descriptor bits have no pyre flag yet.
pub unsafe fn w_type_get_flags(obj: PyObjectRef) -> i64 {
    if obj.is_null() || !is_type(obj) {
        return 0;
    }
    const HEAPTYPE: i64 = 1 << 9;
    const CPYTYPE: i64 = 1;
    const DISALLOW_INSTANTIATION: i64 = 1 << 7;
    const PATMA_SEQUENCE: i64 = 1 << 5;
    const PATMA_MAPPING: i64 = 1 << 6;
    let t = &*(obj as *const W_TypeObject);
    let mut flags = 0i64;
    if t.flag_heaptype {
        flags |= HEAPTYPE;
    } else {
        flags |= CPYTYPE;
    }
    if t.flag_disallow_instantiation
        .load(std::sync::atomic::Ordering::Acquire)
    {
        flags |= DISALLOW_INSTANTIATION;
    }
    match t.flag_map_or_seq.load(std::sync::atomic::Ordering::Acquire) {
        b'M' => flags |= PATMA_MAPPING,
        b'S' => flags |= PATMA_SEQUENCE,
        _ => {}
    }
    flags
}

/// typedef.py:43 `acceptable_as_base_class` — read from Layout level.
/// typeobject.py:1116: w_bestbase.layout.typedef.acceptable_as_base_class
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
pub unsafe fn w_type_get_typedef_hasdict(obj: PyObjectRef) -> bool {
    let layout = (*(obj as *const W_TypeObject)).layout;
    if layout.is_null() {
        false
    } else {
        (*layout).typedef_hasdict
    }
}
/// Override acceptable_as_base_class by cloning the Layout.
/// typedef.py:742,765,664 explicit overrides after initial creation.
/// Layouts may be shared (reused from parent), so we clone to avoid
/// corrupting the parent type's flag.
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

/// `typeobject.py:640-662 W_TypeObject.add_subclass`.
///
/// Records `w_subclass` in `w_parent.weak_subclasses` if not
/// already present.  Stores `weakref.ref(w_subclass)` via
/// `w_weakref_new` so subclass GC isn't blocked (`:642-650`); each
/// entry is a `try_gc_alloc` WEAKREF GcStruct, so the off-GC
/// `weak_subclasses` list is the WEAKREF's only strong root and must
/// be walked by the collector (`walk_type_dicts_gc` while heap types
/// stay Box-immortal, the `W_TYPE_GC_TYPE_ID` custom trace once they
/// are GC-managed).
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
    for i in 0..subs.len() {
        let existing = crate::weakref::w_weakref_deref(subs[i]);
        if existing == w_subclass {
            return;
        }
        if existing.is_null() {
            subs[i] = newref;
            return;
        }
    }
    subs.push(newref);
}

/// `typeobject.py:664-670 W_TypeObject.remove_subclass`.
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

/// `typeobject.py:672-689 W_TypeObject.get_subclasses`.
///
/// Returns the recorded direct subclasses.  Under PyPy's weakref
/// path, dead refs are filtered; pyre's strong-ref fallback has
/// no dead entries to filter so the result is a copy of the
/// stored vector.  The `only_real_subclasses` flag from PyPy
/// (`:672-688`) — used by `descr___subclasses__` to filter
/// metaclass-mro override leaks — is omitted because pyre has no
/// `_add_mro_classes_as_subclasses` call site yet; invalidation
/// callers only need the inclusive list.
///
/// # Safety
/// `w_parent` must point at a valid `W_TypeObject`.
pub unsafe fn w_type_get_subclasses(w_parent: PyObjectRef) -> Vec<PyObjectRef> {
    if w_parent.is_null() || !is_type(w_parent) {
        return Vec::new();
    }
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
            alive.push(target);
        }
    }
    alive
}

/// `typeobject.py:373-377 W_TypeObject.ready` — register `w_self`
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
            // typeobject.py:286 — mutated() stores a fresh tag.
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
            // typeobject.py:315/340 — confirmed-default lookup sets the flag.
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
}
