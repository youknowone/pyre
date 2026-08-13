//! pypy/objspace/std/mapdict.py
//!
//! Mapdict provides per-instance dict and weakref slots for hasdict /
//! weakrefable types. PyPy stores these inside the mapdict map's "dict"
//! and "weakref" SPECIAL slots; pyre does the same for user instances.
//!
//! The names below mirror PyPy: `MapdictDictSupport.getdict` →
//! `_obj_getdict`, `MapdictWeakrefSupport.setweakref` →
//! `_mapdict_setweakref`, etc.

use crate::PyError;
use parking_lot::ReentrantMutex;
use pyre_object::PyObjectRef;
use pyre_object::quasiimmut::QuasiImmutField;

use rustpython_wtf8::{Wtf8, Wtf8Buf};
use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, LazyLock, Mutex};

// PyPy serializes mapdict map/storage transitions with the GIL.  Pyre is
// free-threaded, so use narrow address-striped reentrant locks around the same
// transition boundaries.  This is synchronization only; attribute state
// remains on `W_ObjectObject.map/storage`, exactly as upstream.
struct ForkReentrantLock(UnsafeCell<ReentrantMutex<()>>);
unsafe impl Sync for ForkReentrantLock {}

impl ForkReentrantLock {
    fn new() -> Self {
        Self(UnsafeCell::new(ReentrantMutex::new(())))
    }

    fn get(&self) -> &ReentrantMutex<()> {
        unsafe { &*self.0.get() }
    }

    unsafe fn reinit_after_fork(&self) {
        unsafe { self.0.get().write(ReentrantMutex::new(())) };
    }
}

static INSTANCE_LOCKS: LazyLock<Vec<ForkReentrantLock>> =
    LazyLock::new(|| (0..256).map(|_| ForkReentrantLock::new()).collect());
static CODE_CACHE_LOCKS: LazyLock<Vec<ForkReentrantLock>> =
    LazyLock::new(|| (0..256).map(|_| ForkReentrantLock::new()).collect());

type MapDictGuard = parking_lot::lock_api::ReentrantMutexGuard<
    'static,
    parking_lot::RawMutex,
    parking_lot::RawThreadId,
    (),
>;

/// A contended stripe must not be waited on from inside the running-mutator
/// census: the owner can allocate under the stripe and request a collection,
/// which then waits for this thread while this thread waits for the stripe.
/// Leave the census around the blocking acquire, as `w_list_lock` does.
fn lock_stripe(lock: &'static ReentrantMutex<()>) -> MapDictGuard {
    if let Some(guard) = lock.try_lock() {
        return guard;
    }
    let blocked = crate::module::thread::before_external_block();
    let guard = lock.lock();
    drop(blocked);
    guard
}

fn instance_lock(obj: PyObjectRef) -> MapDictGuard {
    lock_stripe(INSTANCE_LOCKS[(obj as usize >> 4) & (INSTANCE_LOCKS.len() - 1)].get())
}

fn code_cache_lock(code: PyObjectRef) -> MapDictGuard {
    lock_stripe(CODE_CACHE_LOCKS[(code as usize >> 4) & (CODE_CACHE_LOCKS.len() - 1)].get())
}

pub fn after_fork_child() {
    for lock in INSTANCE_LOCKS.iter() {
        unsafe { lock.reinit_after_fork() };
    }
    for lock in CODE_CACHE_LOCKS.iter() {
        unsafe { lock.reinit_after_fork() };
    }
}

// ── attribute shapes (mapdict.py:32-42, 720-732) ──────────────────────

/// mapdict.py:35 `NUM_DIGITS = 4`.
pub const NUM_DIGITS: u32 = 4;
/// mapdict.py:36 `NUM_DIGITS_POW2 = 1 << NUM_DIGITS`.
///
/// Note: upstream multiplies by `NUM_DIGITS_POW2` rather than shifting by
/// `NUM_DIGITS` so the result is known non-negative (mapdict.py:37-38).
pub const NUM_DIGITS_POW2: usize = 1 << NUM_DIGITS;

/// mapdict.py:40-42 — the maximum number of attributes stored in mapdict
/// (afterwards just use a dict).
pub const LIMIT_MAP_ATTRIBUTES: usize = 80;

/// mapdict.py:30 `ALLOW_UNBOXING_INTS = LONG_BIT == 64`. pyre targets
/// 64-bit, so int unboxing is permitted.
pub const ALLOW_UNBOXING_INTS: bool = usize::BITS == 64;

/// mapdict.py:720 `DICT = 0` — attrkind for instance `__dict__` entries.
pub const DICT: u16 = 0;
/// mapdict.py:721 `SPECIAL = 1` — attrkind for the `"dict"` / `"weakref"`
/// special slots.
pub const SPECIAL: u16 = 1;
/// mapdict.py:722 `INVALID = 2` — sentinel attrkind for empty
/// `MapAttrCache` slots.
pub const INVALID: u16 = 2;
/// mapdict.py:723 `SLOTS_STARTING_FROM = 3` — attrkind for `__slots__`
/// slot `i` is `SLOTS_STARTING_FROM + i`.
pub const SLOTS_STARTING_FROM: u16 = 3;

/// mapdict.py:725-732 `attrkind_name`.
///
/// ```python
/// def attrkind_name(attrkind):
///     if attrkind == DICT:
///         return "DICT"
///     if attrkind == SPECIAL:
///         return "SPECIAL"
///     if attrkind == INVALID:
///         return "INVALID"
///     return str(attrkind)
/// ```
pub fn attrkind_name(attrkind: u16) -> String {
    match attrkind {
        DICT => "DICT".to_string(),
        SPECIAL => "SPECIAL".to_string(),
        INVALID => "INVALID".to_string(),
        other => other.to_string(),
    }
}

// ── map nodes (mapdict.py:45-529) ─────────────────────────────────────
//
// AbstractAttribute hierarchy. PyPy uses a class hierarchy
// (AbstractAttribute → Terminator{Dict,NoDict,Devolved} / PlainAttribute,
// mapdict.py:45/304/420). The Rust port models all map nodes with a single
// `MapNode` enum (the enum-vs-hierarchy adaptation explicitly permitted by
// the parity rules) so the recurring `isinstance(self, PlainAttribute)`
// chain tests (mapdict.py:118-122,186) become a cheap `match`, and the three
// Terminator subclasses become a `TerminatorKind` field (mapdict.py:357-418).
//
// Map nodes are interned and shared per type (PyPy interns transitions so the
// same attribute added from the same map yields the same child map); they are
// never freed, so a node is referenced by a raw `*const MapNode` (`MapRef`)
// and the few mutable fields (`ever_mutated`, `allow_unboxing`) use Cell.
//
// AbstractAttribute.space (mapdict.py:47) is omitted: pyre's object space is
// ambient (global helpers) rather than an object threaded through nodes,
// matching the rest of pyre-interpreter.

/// `2 ** methodcachesizeexp` is the MapAttrCache size (pypyoption.py:230,
/// default 11).
pub const METHODCACHESIZEEXP: u32 = 11;

/// A shared, interned, immortal map node (mapdict.py AbstractAttribute).
pub type MapRef = *const MapNode;

/// mapdict.py:357/376/382 — the three Terminator subclasses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TerminatorKind {
    /// `DictTerminator` (mapdict.py:357).
    Dict,
    /// `NoDictTerminator` (mapdict.py:376).
    NoDict,
    /// `DevolvedDictTerminator` (mapdict.py:382).
    Devolved,
}

/// mapdict.py:304 `Terminator(AbstractAttribute)` — the root of a map chain.
pub struct Terminator {
    /// mapdict.py:307 `w_cls`.
    pub w_cls: PyObjectRef,
    /// mapdict.py:305 `_immutable_fields_` declares `allow_unboxing?`;
    /// mapdict.py:308 `allow_unboxing` permits unboxed attributes until a
    /// differently-typed reassignment freezes them off. [`Self::set_allow_unboxing`]
    /// invalidates watchers before changing it.
    pub allow_unboxing: Cell<bool>,
    /// The hidden watcher field implementing mapdict.py:305
    /// `_immutable_fields_ = [..., 'allow_unboxing?']`. The owner is a
    /// `Box::into_raw` leak that is never freed, so [`QuasiImmutField`]'s
    /// `Drop` is unreachable and its inner box is reclaimed only through
    /// [`QuasiImmutField::invalidate`].
    pub allow_unboxing_watchers: QuasiImmutField,
    /// Which Terminator subclass this is.
    pub kind: TerminatorKind,
    /// mapdict.py:360 `DictTerminator.devolved_dict_terminator` (null unless
    /// `kind == Dict`).
    pub devolved_dict_terminator: Cell<MapRef>,
    /// mapdict.py:47 `AbstractAttribute.cache_attrs` — the per-node transition
    /// cache `(name, attrkind) -> CachedAttributeHolder`. PyPy lazily inits it
    /// to `{}`; the eager empty map here is equivalent.
    pub cache_attrs: Mutex<HashMap<(Wtf8Buf, u16), *const CachedAttributeHolder>>,
    /// mapdict.py:53 `AbstractAttribute.terminator` — a Terminator points to
    /// itself.
    pub terminator: MapRef,
}

impl Terminator {
    /// Change mapdict.py:308 `allow_unboxing`, invalidating any constant-folded
    /// reads first.
    ///
    /// The value-change return is a required pyre addition: the unguarded
    /// writes in mapdict.py `PlainAttribute._direct_write` and
    /// `CachedAttributeHolder.pick_attr` can repeatedly write `False`, and a
    /// re-armed watcher must not retrace without bound. The `is_installed()`
    /// fast path is quasiimmut.py:38-41 `invalidation`'s null test and remains
    /// outside the sweep so the common case is lock-free. Notification must
    /// precede the store: rclass.py:1010-1012 `hook_setfield` emits
    /// `jit_force_quasi_immutable` before `setfield`, as do celldict.rs
    /// `mutated` and typeobject.rs `w_type_set_version_tag`.
    pub fn set_allow_unboxing(&self, v: bool) {
        if self.allow_unboxing.get() == v {
            return;
        }
        if self.allow_unboxing_watchers.is_installed() {
            unsafe {
                pyre_object::quasiimmut::sweep_quasi_immut_field(&self.allow_unboxing_watchers)
            };
        }
        self.allow_unboxing.set(v);
    }

    pub fn register_allow_unboxing_watcher(&self, flag: &Arc<AtomicBool>) {
        self.allow_unboxing_watchers.register_loop_token(flag);
    }

    pub fn install_allow_unboxing_watcher(&self) {
        self.allow_unboxing_watchers.ensure_installed();
    }

    pub fn allow_unboxing_qmut_installed(&self) -> bool {
        self.allow_unboxing_watchers.is_installed()
    }

    pub fn force_allow_unboxing_qmut(&self) {
        self.allow_unboxing_watchers.invalidate();
    }
}

/// Register a loop against a Terminator's `allow_unboxing?` field.
///
/// # Safety
/// `terminator` must be null or point at a live owner.
pub unsafe fn terminator_register_allow_unboxing_watcher(
    terminator: *const Terminator,
    flag: &Arc<AtomicBool>,
) {
    if terminator.is_null() {
        return;
    }
    unsafe { (*terminator).register_allow_unboxing_watcher(flag) };
}

/// Install a Terminator's `allow_unboxing?` watcher without registering a loop.
///
/// # Safety
/// `terminator` must be null or point at a live owner.
pub unsafe fn terminator_install_allow_unboxing_watcher(terminator: *const Terminator) {
    if terminator.is_null() {
        return;
    }
    unsafe { (*terminator).install_allow_unboxing_watcher() };
}

/// Whether a Terminator's `allow_unboxing?` watcher is installed.
///
/// # Safety
/// `terminator` must be null or point at a live owner.
pub unsafe fn terminator_allow_unboxing_qmut_installed(terminator: *const Terminator) -> bool {
    !terminator.is_null() && unsafe { (*terminator).allow_unboxing_qmut_installed() }
}

/// Force a Terminator's `allow_unboxing?` qmut directly. This is the tracer's
/// own `do_force_quasi_immutable` call (pyjitpl.py:1113-1115), not a runtime
/// store, so it calls [`QuasiImmutField::invalidate`] rather than the sweep.
///
/// # Safety
/// `terminator` must be null or point at a live owner.
pub unsafe fn terminator_force_allow_unboxing_qmut(terminator: *const Terminator) {
    if terminator.is_null() {
        return;
    }
    unsafe { (*terminator).force_allow_unboxing_qmut() };
}

/// The unbox type of an `UnboxedPlainAttribute` (mapdict.py:534/547,
/// `space.IntObjectCls` / `space.FloatObjectCls`). pyre exposes no per-type
/// impl-class object, so the int/float distinction is captured by this enum
/// and resolved through `is_int`/`is_float` (the enum-vs-class-object
/// adaptation, parallel to the `MapNode` enum-vs-hierarchy adaptation above).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnboxType {
    /// `space.IntObjectCls`.
    Int,
    /// `space.FloatObjectCls`.
    Float,
}

/// mapdict.py:532-563 `UnboxedPlainAttribute`'s extra fields. PyPy models the
/// unboxed attribute as a `PlainAttribute` subclass; pyre keeps it an optional
/// field on `PlainAttribute` so all the shared `PlainAttribute` machinery
/// (find_map_attr / add_attr / reorder / cache) applies unchanged
/// (enum-vs-hierarchy adaptation). When present, the value is stored unboxed in
/// a shared longlong list at `PlainAttribute.storageindex`; `listindex` is the
/// position in that list.
pub struct UnboxedExtra {
    /// mapdict.py:547 `typ`.
    pub typ: UnboxType,
    /// mapdict.py:563 `listindex` — position in the shared longlong list.
    pub listindex: usize,
    /// mapdict.py:544/561 `firstunwrapped` — this attribute is the first
    /// unboxed one to occupy its storage slot (so it allocates the list).
    pub firstunwrapped: bool,
}

/// mapdict.py:420 `PlainAttribute(AbstractAttribute)` — one stored attribute.
pub struct PlainAttribute {
    /// mapdict.py:425 `name` (utf8-encoded).
    pub name: Wtf8Buf,
    /// mapdict.py:426 `attrkind`.
    pub attrkind: u16,
    /// mapdict.py:427 `storageindex` (= `back.storage_needed()`).
    pub storageindex: usize,
    /// mapdict.py:428 `_num_attributes` (= `back.num_attributes() + 1`).
    pub num_attributes: usize,
    /// mapdict.py:429 `back`.
    pub back: MapRef,
    /// mapdict.py:421 `_immutable_fields_` declares `ever_mutated?`;
    /// mapdict.py:430 `ever_mutated` records whether this attribute has ever
    /// been overwritten or deleted. [`Self::set_ever_mutated`] invalidates
    /// watchers before changing it.
    pub ever_mutated: Cell<bool>,
    /// The hidden watcher field implementing mapdict.py:421
    /// `_immutable_fields_ = [..., 'ever_mutated?', ...]`. The owner is a
    /// `Box::into_raw` leak that is never freed, so [`QuasiImmutField`]'s
    /// `Drop` is unreachable and its inner box is reclaimed only through
    /// [`QuasiImmutField::invalidate`].
    pub ever_mutated_watchers: QuasiImmutField,
    /// mapdict.py:431 `order`.
    pub order: usize,
    /// mapdict.py:47 `AbstractAttribute.cache_attrs` — the per-node transition
    /// cache `(name, attrkind) -> CachedAttributeHolder`.
    pub cache_attrs: Mutex<HashMap<(Wtf8Buf, u16), *const CachedAttributeHolder>>,
    /// mapdict.py:53 `AbstractAttribute.terminator` (= `back.terminator`).
    pub terminator: MapRef,
    /// `Some` for an `UnboxedPlainAttribute` (mapdict.py:532); `None` for a
    /// plain boxed attribute.
    pub unboxed: Option<UnboxedExtra>,
}

impl PlainAttribute {
    /// Change mapdict.py:430 `ever_mutated` using the ordering, fast path, and
    /// repeated-value guard documented on [`Terminator::set_allow_unboxing`].
    pub fn set_ever_mutated(&self, v: bool) {
        if self.ever_mutated.get() == v {
            return;
        }
        if self.ever_mutated_watchers.is_installed() {
            unsafe {
                pyre_object::quasiimmut::sweep_quasi_immut_field(&self.ever_mutated_watchers)
            };
        }
        self.ever_mutated.set(v);
    }

    pub fn register_ever_mutated_watcher(&self, flag: &Arc<AtomicBool>) {
        self.ever_mutated_watchers.register_loop_token(flag);
    }

    pub fn install_ever_mutated_watcher(&self) {
        self.ever_mutated_watchers.ensure_installed();
    }

    pub fn ever_mutated_qmut_installed(&self) -> bool {
        self.ever_mutated_watchers.is_installed()
    }

    pub fn force_ever_mutated_qmut(&self) {
        self.ever_mutated_watchers.invalidate();
    }
}

/// Register a loop against a PlainAttribute's `ever_mutated?` field.
///
/// # Safety
/// `attribute` must be null or point at a live owner.
pub unsafe fn plain_attribute_register_ever_mutated_watcher(
    attribute: *const PlainAttribute,
    flag: &Arc<AtomicBool>,
) {
    if attribute.is_null() {
        return;
    }
    unsafe { (*attribute).register_ever_mutated_watcher(flag) };
}

/// Install a PlainAttribute's `ever_mutated?` watcher without registering a loop.
///
/// # Safety
/// `attribute` must be null or point at a live owner.
pub unsafe fn plain_attribute_install_ever_mutated_watcher(attribute: *const PlainAttribute) {
    if attribute.is_null() {
        return;
    }
    unsafe { (*attribute).install_ever_mutated_watcher() };
}

/// Whether a PlainAttribute's `ever_mutated?` watcher is installed.
///
/// # Safety
/// `attribute` must be null or point at a live owner.
pub unsafe fn plain_attribute_ever_mutated_qmut_installed(
    attribute: *const PlainAttribute,
) -> bool {
    !attribute.is_null() && unsafe { (*attribute).ever_mutated_qmut_installed() }
}

/// Force a PlainAttribute's `ever_mutated?` qmut directly. This is the tracer's
/// own `do_force_quasi_immutable` call (pyjitpl.py:1113-1115), not a runtime
/// store, so it calls [`QuasiImmutField::invalidate`] rather than the sweep.
///
/// # Safety
/// `attribute` must be null or point at a live owner.
pub unsafe fn plain_attribute_force_ever_mutated_qmut(attribute: *const PlainAttribute) {
    if attribute.is_null() {
        return;
    }
    unsafe { (*attribute).force_ever_mutated_qmut() };
}

/// mapdict.py:45 `AbstractAttribute` plus its two concrete subclasses.
pub enum MapNode {
    Terminator(Terminator),
    Plain(PlainAttribute),
}

fn intern_node(node: MapNode) -> MapRef {
    // Immortal: leak the box so the shared node lives for the process.
    Box::into_raw(Box::new(node)) as MapRef
}

/// mapdict.py:306-310 `Terminator.__init__`. `AbstractAttribute.__init__(space,
/// self)` makes the terminator its own `terminator`.
pub fn new_terminator(w_cls: PyObjectRef, kind: TerminatorKind) -> MapRef {
    let raw = Box::into_raw(Box::new(MapNode::Terminator(Terminator {
        w_cls,
        allow_unboxing: Cell::new(true),
        allow_unboxing_watchers: QuasiImmutField::new(),
        kind,
        devolved_dict_terminator: Cell::new(std::ptr::null()),
        cache_attrs: Mutex::new(HashMap::new()),
        terminator: std::ptr::null(),
    })));
    // Patch the self-referential terminator now that the address is known
    // (still uniquely owned here, before it is shared).
    unsafe {
        if let MapNode::Terminator(t) = &mut *raw {
            t.terminator = raw as MapRef;
        }
    }
    raw as MapRef
}

/// mapdict.py:358-360 `DictTerminator.__init__` — also builds the paired
/// `DevolvedDictTerminator` and links it.
pub fn new_dict_terminator(w_cls: PyObjectRef) -> MapRef {
    let devolved = new_terminator(w_cls, TerminatorKind::Devolved);
    let t = new_terminator(w_cls, TerminatorKind::Dict);
    unsafe {
        (*t).as_terminator().devolved_dict_terminator.set(devolved);
    }
    t
}

/// Build the per-type map terminator (typeobject.py:251-260 +
/// mapdict.py:357-360): a `DictTerminator` when the type has `__dict__`, else
/// a `NoDictTerminator`. `allow_unboxing` keeps its `mapdict.py:308` default of
/// `True`; type instability later freezes it off per-class through the reactive
/// paths (`plain_direct_write` type change / `holder_pick_attr` mismatch), and
/// the affected instances rebuild boxed storage via `convert_to_boxed`.
///
/// typeobject.py:255-257 builds a `DictTerminator` only when
/// `self.hasdict and not typedef.hasdict` — a type whose layout typedef
/// already manages its own dict (e.g. module) gets a `NoDictTerminator`.
/// `typedef_hasdict` is `Layout.typedef_hasdict` (typedef.py:40). On the
/// current shared-Layout model it is `false` for every reachable instance
/// layout (all reuse INSTANCE_TYPE's Layout, whose typedef declares no
/// `__dict__`), so the term is inert today; populating it `true` for the
/// dict-managing typedefs is deferred to the distinct-TypeDef convergence
/// (alongside the parked `Layout.acceptable_as_base_class`).
pub fn new_instance_terminator(w_cls: PyObjectRef, hasdict: bool, typedef_hasdict: bool) -> MapRef {
    // `hasdict and not typedef.hasdict`, expressed without a bare `!` so the
    // annotator can lower it on the JIT-reachable terminator path.
    let wants_dict = match typedef_hasdict {
        true => false,
        false => hasdict,
    };
    if wants_dict {
        new_dict_terminator(w_cls)
    } else {
        new_terminator(w_cls, TerminatorKind::NoDict)
    }
}

/// `_mapdict_init_empty` deferred to first attribute access (mapdict.py:758-761
/// `user_setup` calls it at construction with `w_subtype.terminator`; pyre
/// defers to first access). If the instance's `map` is null, fetch the owning
/// type's terminator — lazily creating and storing it on the type if absent,
/// covering types built before the eager install site — then set it as the
/// instance map. Must run before any `node_read`/`node_write`/`node_delete`.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` (the caller guards with
/// `is_instance`). The instance is an immortal `Box`, so the raw
/// pointer is stable across this call.
pub unsafe fn ensure_mapdict_initialized(obj: PyObjectRef) {
    let mut inst = unsafe { mapdict_carrier(obj) };
    if !inst._get_mapdict_map().is_null() {
        return;
    }
    let w_type = pyre_object::w_instance_get_type(obj);
    let term = type_terminator_or_create(w_type);
    inst._set_mapdict_map(term);
}

/// [`type_terminator_or_create`] for callers outside the mapdict layer — the
/// JIT's instantiation emit bakes the terminator into the trace as the fresh
/// instance's `map`, so it needs the same one the interpreter would install
/// rather than the null a not-yet-touched type still carries.
///
/// # Safety
/// `w_type` must be a live `W_TypeObject`.
pub unsafe fn ensure_type_terminator(w_type: PyObjectRef) -> *const u8 {
    unsafe { type_terminator_or_create(w_type) as *const u8 }
}

/// Whether `obj` belongs to the builtin families whose user subclass layout is
/// generated by appending `MapdictStorageMixin` to the base payload
/// (`typedef.py:174-227`). A specialised tuple has its own fixed layout and no
/// generated user variant, so it is not one of them.
///
/// # Safety
/// `obj` must be a live object reference.
#[inline]
unsafe fn is_generated_user_layout_family(obj: PyObjectRef) -> bool {
    unsafe {
        pyre_object::is_int(obj)
            || pyre_object::is_str(obj)
            || (pyre_object::is_tuple(obj)
                && !pyre_object::specialisedtupleobject::is_specialised_tuple(obj))
    }
}

#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
#[inline]
unsafe fn is_ssl_mapdict_layout(obj: PyObjectRef) -> bool {
    use pyre_object::lltype::PyreClassPyTypeOf;
    unsafe {
        pyre_object::py_type_check(
            obj,
            &*<crate::module::_ssl::W_SSLContext as PyreClassPyTypeOf>::PYTYPE,
        ) || pyre_object::py_type_check(
            obj,
            &*<crate::module::_ssl::W_MemoryBIO as PyreClassPyTypeOf>::PYTYPE,
        )
    }
}

#[cfg(not(all(not(target_arch = "wasm32"), not(feature = "sandbox"))))]
#[inline]
unsafe fn is_ssl_mapdict_layout(_obj: PyObjectRef) -> bool {
    false
}

#[cfg(all(any(unix, windows), not(feature = "sandbox")))]
#[inline]
unsafe fn is_mmap_mapdict_layout(obj: PyObjectRef) -> bool {
    use pyre_object::lltype::PyreClassPyTypeOf;
    unsafe {
        pyre_object::py_type_check(
            obj,
            &*<crate::module::mmap::W_MMap as PyreClassPyTypeOf>::PYTYPE,
        )
    }
}

#[cfg(not(all(any(unix, windows), not(feature = "sandbox"))))]
#[inline]
unsafe fn is_mmap_mapdict_layout(_obj: PyObjectRef) -> bool {
    false
}

#[inline]
unsafe fn is_zlib_mapdict_layout(obj: PyObjectRef) -> bool {
    use pyre_object::lltype::PyreClassPyTypeOf;
    unsafe {
        pyre_object::py_type_check(
            obj,
            &*<crate::module::zlib::W_Compress as PyreClassPyTypeOf>::PYTYPE,
        ) || pyre_object::py_type_check(
            obj,
            &*<crate::module::zlib::W_Decompress as PyreClassPyTypeOf>::PYTYPE,
        ) || pyre_object::py_type_check(
            obj,
            &*<crate::module::zlib::W_ZlibDecompressor as PyreClassPyTypeOf>::PYTYPE,
        )
    }
}

/// Whether `obj`'s physical allocation carries the slots supplied by
/// `MapdictStorageMixin` (`mapdict.py:748-761, 905-910`). Ordinary instances
/// and `_random.Random` keep the historical prefix. The generated tuple/int/str
/// user layouts append the mixin after their unchanged builtin base payload
/// (`typedef.py:174-227`) whether or not the class routes attributes through a
/// dict; a `__slots__`-only native subclass is the case that separates the two.
///
/// # Safety
/// `obj` must be null or a live object reference.
#[inline]
pub unsafe fn has_mapdict_layout(obj: PyObjectRef) -> bool {
    if obj.is_null() || unsafe { pyre_object::is_type(obj) } {
        return false;
    }
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return false;
    }
    if (unsafe { pyre_object::is_instance(obj) })
        || unsafe { pyre_object::py_type_check(obj, &crate::module::_random::RANDOM_TYPE) }
        || unsafe { is_ssl_mapdict_layout(obj) }
        || unsafe { is_mmap_mapdict_layout(obj) }
        || unsafe { is_zlib_mapdict_layout(obj) }
    {
        return true;
    }
    if !unsafe { is_generated_user_layout_family(obj) } {
        return false;
    }
    let type_id = unsafe { (*majit_gc::header::header_of(obj as usize)).type_id() };
    type_id == pyre_object::intobject::W_INT_USER_GC_TYPE_ID.get()
        || type_id == pyre_object::unicodeobject::W_UNICODE_USER_GC_TYPE_ID.get()
        || type_id == pyre_object::tupleobject::W_TUPLE_USER_GC_TYPE_ID.get()
}

/// Whether attribute access for `obj` routes through mapdict storage. This is
/// the physical [`has_mapdict_layout`] test plus the owning class's `hasdict`
/// flag for generated tuple/int/str user layouts.
///
/// # Safety
/// `obj` must be null or a live object reference.
#[inline]
pub unsafe fn has_mapdict_storage(obj: PyObjectRef) -> bool {
    if !unsafe { has_mapdict_layout(obj) } {
        return false;
    }
    if !unsafe { is_generated_user_layout_family(obj) } {
        return true;
    }
    let w_class = unsafe { (*obj).w_class };
    !w_class.is_null()
        && unsafe { pyre_object::is_type(w_class) }
        && unsafe { pyre_object::w_type_get_hasdict(w_class) }
}

/// Fetch `w_type`'s instance terminator, lazily creating and storing it on the
/// type if absent (covering types built before the eager install site).
///
/// # Safety
/// `w_type` must be a live `W_TypeObject`.
unsafe fn type_terminator_or_create(w_type: PyObjectRef) -> MapRef {
    let mut term = pyre_object::w_type_get_terminator(w_type);
    if term.is_null() {
        let hasdict = pyre_object::w_type_get_hasdict(w_type);
        let typedef_hasdict = pyre_object::w_type_get_typedef_hasdict(w_type);
        term = new_instance_terminator(w_type, hasdict, typedef_hasdict) as *const u8;
        unsafe { pyre_object::w_type_set_terminator(w_type, term) };
    }
    term as MapRef
}

/// mapdict.py:754-756 `MapdictDictSupport.setclass` — re-root `obj`'s map chain
/// onto `w_cls`'s terminator and transplant the rebuilt storage+map. Called from
/// `descr_set___class__` for a `W_ObjectObject`. pyre additionally keeps the
/// `w_class` field authoritative for `type()` (the node layer's
/// `terminator.w_cls` is never read for `getclass`), so the caller sets that
/// after this returns.
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_setdictvalue`]: `node_set_terminator` rebuilds through
/// `copy_attr`/`plain_direct_read`, whose unboxed storage path
/// (`unerase_unboxed` raw `*mut Vec<i64>` array reads, mapdict.py:565-646) is
/// not annotator-lowerable.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_setclass(obj: PyObjectRef, w_cls: PyObjectRef) {
    unsafe { ensure_mapdict_initialized(obj) };
    let new_term = unsafe { type_terminator_or_create(w_cls) };
    let mut inst = unsafe { mapdict_carrier(obj) };
    let map = inst._get_mapdict_map();
    let new_obj = unsafe { node_set_terminator(map, &inst, new_term) };
    let new_map = new_obj.map;
    inst._set_mapdict_storage_and_map(new_obj.storage, new_map);
}

/// `setdictvalue` routed to the mapdict node layer (mapdict.py:849-850
/// `MapdictDictSupport.setdictvalue` → `map.write(self, attrname, DICT,
/// w_value)`, dispatch at mapdict.py:68-75).
///
/// Returns `map.write`'s flag: `false` when the map is rooted at a
/// `NoDictTerminator` (`__slots__`, no instance `__dict__`), which is the
/// AttributeError signal `object_setattr` raises on.
///
/// `dont_look_inside` makes this a residual-call boundary for the JIT
/// CodeWriter: `setdictvalue` is JIT-reachable via STORE_ATTR, but the node
/// layer's unboxed storage path (`erase_unboxed`/`unerase_unboxed` raw
/// `*mut Vec<i64>` casts and `Box` allocation for the shared longlong list,
/// mapdict.py:565-646) is not annotator-lowerable, so the boundary stays.
/// Look-inside (the `map.write` JIT specialization, mapdict.py:614-628) is a
/// future convergence, once the unboxed storage shape is JIT-representable.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` (caller guards with `is_instance`).
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_setdictvalue(
    obj: PyObjectRef,
    name: &Wtf8,
    value: PyObjectRef,
) -> bool {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    node_write(map, &mut inst, name, DICT, value)
}

/// Whether `map` is rooted at a `DevolvedDictTerminator` (mapdict.py:382),
/// i.e. the instance has spilled its attributes into a real dictionary.
///
/// A devolved map no longer names the instance's attributes, and it does not
/// change when one is added, so a JIT fold that re-proves "no instance
/// attribute shadows this name" by pinning the map value is unsound here and
/// must decline instead.
///
/// # Safety
/// `map`, when non-null, must be a live map node.
pub unsafe fn map_is_devolved(map: MapRef) -> bool {
    if map.is_null() {
        return false;
    }
    let terminator = unsafe { (*map).terminator() };
    if terminator.is_null() {
        return false;
    }
    unsafe { (*terminator).as_terminator() }.kind == TerminatorKind::Devolved
}

/// `getdictvalue` routed to the mapdict node layer (mapdict.py:846-847
/// `MapdictDictSupport.getdictvalue` → `map.read(self, attrname, DICT)`).
/// Returns the value or `None` when the attribute is absent.
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_setdictvalue`]: the node read path's unboxed storage branch
/// (`unerase_unboxed` raw `*mut Vec<i64>` reads + `convert_to_boxed`) is not
/// annotator-lowerable.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` (caller guards with `is_instance`).
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_getdictvalue(obj: PyObjectRef, name: &Wtf8) -> Option<PyObjectRef> {
    unsafe { instance_node_getdictvalue_checked(obj, name) }.unwrap_or(None)
}

/// Fallible [`instance_node_getdictvalue`], for the callers that have an error
/// channel to propagate a raising `__eq__` on.
///
/// Only the devolved terminator's dict probe can raise; the swallowing
/// spelling above is written in terms of this one and its `unwrap_or` consumes
/// the pending error slot, so a dropped error cannot surface on a later
/// operation.
///
/// `dont_look_inside` for the same reason as [`instance_node_getdictvalue`].
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` (caller guards with `is_instance`).
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_getdictvalue_checked(
    obj: PyObjectRef,
    name: &Wtf8,
) -> Result<Option<PyObjectRef>, PyError> {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    let w = unsafe { node_read_checked(map, &inst, name, DICT) }?;
    // mapdict.py:846-847 getdictvalue → read → _direct_read (592-598): lazily
    // migrate to boxed storage when the read attribute is unboxed and its class
    // has frozen unboxing.  A raise in `read` unwinds before the migration tail
    // (846-847 → 58 → 312-313 returns None on the miss path, never reaching
    // _direct_read), so migration is skipped whenever the read raised;
    // propagating the read error here mirrors that — and `maybe_migrate_to_boxed`
    // re-derives the None on the raising path, so it is a no-op there regardless.
    unsafe { maybe_migrate_to_boxed(map, &mut inst, name, DICT) };
    Ok(w)
}

/// `deldictvalue` routed to the mapdict node layer (mapdict.py:852-857
/// `MapdictDictSupport.deldictvalue` → `map.delete(self, attrname, DICT)` then
/// `_set_mapdict_storage_and_map`). Returns `true` if the attribute existed and
/// was removed, `false` otherwise (the caller raises AttributeError on false).
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_setdictvalue`]; the rebuild path (`node_delete` → `node_copy`)
/// is not JIT-traced while the unboxed branches remain unported.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` (caller guards with `is_instance`).
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_deldictvalue(obj: PyObjectRef, name: &Wtf8) -> bool {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    match node_delete(map, &inst, name, DICT) {
        None => false,
        Some(new_obj) => {
            inst._set_mapdict_storage_and_map(new_obj.storage, new_obj.map);
            true
        }
    }
}

/// Read the instance `__dict__` wrapper from the "dict" SPECIAL slot
/// (mapdict.py:828 `w_dict = self._get_mapdict_map().read(self, "dict",
/// SPECIAL)`). Returns `None` when the wrapper has not been materialised.
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_getdictvalue`]: the node read path
/// (`node_read` → `plain_direct_read` → `convert_to_boxed`) still has the
/// unported unboxed branch the annotator cannot lower.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` (caller guards with `is_instance`).
#[majit_macros::dont_look_inside]
pub unsafe fn instance_get_dict_slot(obj: PyObjectRef) -> Option<PyObjectRef> {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    node_read(map, &inst, Wtf8::new("dict"), SPECIAL)
}

/// Write the instance `__dict__` wrapper into the "dict" SPECIAL slot
/// (mapdict.py:833/859 `flag = self._get_mapdict_map().write(self, "dict",
/// SPECIAL, w_dict)`). `node_write` grows the map+storage by the SPECIAL slot on
/// first write (the same transplant path the DICT setter takes). Returns the
/// `write` flag.
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_setdictvalue`].
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` (caller guards with `is_instance`).
#[majit_macros::dont_look_inside]
pub unsafe fn instance_set_dict_slot(obj: PyObjectRef, w_dict: PyObjectRef) -> bool {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    node_write(map, &mut inst, Wtf8::new("dict"), SPECIAL, w_dict)
}

/// Read the weakref lifeline from the instance's `"weakref"` SPECIAL slot
/// (mapdict.py:787 `self._get_mapdict_map().read(self, "weakref", SPECIAL)`).
///
/// # Safety
/// `obj` must be a live `W_ObjectObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_get_weakref_slot(obj: PyObjectRef) -> Option<PyObjectRef> {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    // A cleared lifeline (mapdict.py:802 writes `None` = null into the retained
    // node) reads back as a null slot; report it as absent rather than
    // `Some(null)`.
    node_read(map, &inst, Wtf8::new("weakref"), SPECIAL).filter(|w| !w.is_null())
}

/// Write the weakref lifeline into the instance's `"weakref"` SPECIAL slot
/// (mapdict.py:798 `self._get_mapdict_map().write(...)`).
///
/// # Safety
/// `obj` must be a live `W_ObjectObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_set_weakref_slot(obj: PyObjectRef, lifeline: PyObjectRef) -> bool {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    node_write(map, &mut inst, Wtf8::new("weakref"), SPECIAL, lifeline)
}

/// Clear the weakref lifeline in place, matching mapdict.py:802
/// `self._get_mapdict_map().write(self, "weakref", SPECIAL, None)`.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_del_weakref_slot(obj: PyObjectRef) {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    let _ = node_write(
        map,
        &mut inst,
        Wtf8::new("weakref"),
        SPECIAL,
        pyre_object::PY_NULL,
    );
}

// ── methods needed for slots (mapdict.py:764-780 MapdictSlotsSupport) ──

/// mapdict.py:766-768 `MapdictSlotsSupport.getslotvalue` —
/// `map.read(self, "slot", SLOTS_STARTING_FROM + slotindex)`.
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_getdictvalue`].
///
/// # Safety
/// `obj` must be a live object reference. A non-`W_ObjectObject`
/// receiver hits the `W_Root.getslotvalue` default — NotImplementedError
/// (baseobjspace.py:119-120) — as a panic.
#[majit_macros::dont_look_inside]
pub unsafe fn getslotvalue(obj: PyObjectRef, slotindex: u32) -> Option<PyObjectRef> {
    assert!(
        unsafe { has_mapdict_storage(obj) },
        "W_Root.getslotvalue: receiver has no mapdict slot storage"
    );
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    let attrkind = SLOTS_STARTING_FROM + slotindex as u16;
    let w_res = unsafe { node_read(map, &inst, Wtf8::new("slot"), attrkind) };
    // read → _direct_read (mapdict.py:592-598) lazily migrates an unboxed
    // attribute to boxed storage, as in `instance_node_getdictvalue`.
    unsafe { maybe_migrate_to_boxed(map, &mut inst, Wtf8::new("slot"), attrkind) };
    w_res
}

/// mapdict.py:770-772 `MapdictSlotsSupport.setslotvalue` —
/// `map.write(self, "slot", SLOTS_STARTING_FROM + slotindex, w_value)`.
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_setdictvalue`].
///
/// # Safety
/// `obj` must be a live object reference. A non-`W_ObjectObject`
/// receiver hits the `W_Root.setslotvalue` default — NotImplementedError
/// (baseobjspace.py:122-123) — as a panic.
#[majit_macros::dont_look_inside]
pub unsafe fn setslotvalue(obj: PyObjectRef, slotindex: u32, w_value: PyObjectRef) {
    assert!(
        unsafe { has_mapdict_storage(obj) },
        "W_Root.setslotvalue: receiver has no mapdict slot storage"
    );
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    let attrkind = SLOTS_STARTING_FROM + slotindex as u16;
    let flag = node_write(map, &mut inst, Wtf8::new("slot"), attrkind, w_value);
    debug_assert!(flag, "node_write returned false for a slot attribute");
}

/// mapdict.py:774-780 `MapdictSlotsSupport.delslotvalue` —
/// `map.delete(self, "slot", SLOTS_STARTING_FROM + slotindex)` then
/// `_set_mapdict_storage_and_map`. Returns `false` when the slot was
/// never written (the caller raises AttributeError).
///
/// `dont_look_inside` — same residual-call rationale as
/// [`instance_node_setdictvalue`].
///
/// # Safety
/// `obj` must be a live object reference. A non-`W_ObjectObject`
/// receiver hits the `W_Root.delslotvalue` default — NotImplementedError
/// (baseobjspace.py:125-126) — as a panic.
#[majit_macros::dont_look_inside]
pub unsafe fn delslotvalue(obj: PyObjectRef, slotindex: u32) -> bool {
    assert!(
        unsafe { has_mapdict_storage(obj) },
        "W_Root.delslotvalue: receiver has no mapdict slot storage"
    );
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    let attrkind = SLOTS_STARTING_FROM + slotindex as u16;
    match node_delete(map, &inst, Wtf8::new("slot"), attrkind) {
        None => false,
        Some(new_obj) => {
            inst._set_mapdict_storage_and_map(new_obj.storage, new_obj.map);
            true
        }
    }
}

/// mapdict.py:423-431 `PlainAttribute.__init__`.
///
/// # Safety
/// `back` must point to a live (immortal) map node.
pub unsafe fn new_plain_attribute(
    name: Wtf8Buf,
    attrkind: u16,
    back: MapRef,
    order: usize,
) -> MapRef {
    let back_node = unsafe { &*back };
    intern_node(MapNode::Plain(PlainAttribute {
        name,
        attrkind,
        storageindex: back_node.storage_needed(),
        num_attributes: back_node.num_attributes() + 1,
        back,
        ever_mutated: Cell::new(false),
        ever_mutated_watchers: QuasiImmutField::new(),
        order,
        cache_attrs: Mutex::new(HashMap::new()),
        terminator: back_node.terminator(),
        unboxed: None,
    }))
}

/// mapdict.py:534-563 `UnboxedPlainAttribute.__init__` +
/// `_compute_storageindex_listindex`.
///
/// Unlike `PlainAttribute.__init__`, the storage index is shared with the
/// nearest `UnboxedPlainAttribute` ancestor (all unboxed attributes pack their
/// longlong values into one shared list); only the first unboxed attribute in a
/// slot (`firstunwrapped`) allocates a fresh slot.
///
/// # Safety
/// `back` must point to a live (immortal) map node.
pub unsafe fn new_unboxed_plain_attribute(
    name: Wtf8Buf,
    attrkind: u16,
    back: MapRef,
    order: usize,
    typ: UnboxType,
) -> MapRef {
    let back_node = unsafe { &*back };
    // _compute_storageindex_listindex (mapdict.py:549-563): walk up looking for
    // an existing UnboxedPlainAttribute to share a storage slot with.
    let mut attr = back;
    let mut shared = None;
    loop {
        match unsafe { &*attr } {
            MapNode::Plain(p) => {
                if let Some(u) = &p.unboxed {
                    shared = Some((p.storageindex, u.listindex + 1));
                    break;
                }
                attr = p.back;
            }
            MapNode::Terminator(_) => break,
        }
    }
    let (storageindex, listindex, firstunwrapped) = match shared {
        Some((storageindex, listindex)) => (storageindex, listindex, false),
        None => (back_node.storage_needed(), 0, true),
    };
    intern_node(MapNode::Plain(PlainAttribute {
        name,
        attrkind,
        storageindex,
        num_attributes: back_node.num_attributes() + 1,
        back,
        ever_mutated: Cell::new(false),
        ever_mutated_watchers: QuasiImmutField::new(),
        order,
        cache_attrs: Mutex::new(HashMap::new()),
        terminator: back_node.terminator(),
        unboxed: Some(UnboxedExtra {
            typ,
            listindex,
            firstunwrapped,
        }),
    }))
}

impl MapNode {
    /// Borrow the inner Terminator (panics on PlainAttribute).
    pub fn as_terminator(&self) -> &Terminator {
        match self {
            MapNode::Terminator(t) => t,
            MapNode::Plain(_) => panic!("as_terminator on PlainAttribute"),
        }
    }

    /// Borrow the inner PlainAttribute (panics on Terminator).
    pub fn as_plain(&self) -> &PlainAttribute {
        match self {
            MapNode::Plain(p) => p,
            MapNode::Terminator(_) => panic!("as_plain on Terminator"),
        }
    }

    /// `isinstance(self, PlainAttribute)`.
    pub fn is_plain(&self) -> bool {
        matches!(self, MapNode::Plain(_))
    }

    /// mapdict.py:53,141 `AbstractAttribute.terminator` / `get_terminator`.
    pub fn terminator(&self) -> MapRef {
        match self {
            MapNode::Terminator(t) => t.terminator,
            MapNode::Plain(p) => p.terminator,
        }
    }

    /// mapdict.py:327 (Terminator) / 478 (PlainAttribute) / 565-568
    /// (UnboxedPlainAttribute) `storage_needed`.
    pub fn storage_needed(&self) -> usize {
        match self {
            MapNode::Terminator(_) => 0,
            // mapdict.py:565-568: an unboxed attribute only adds a slot when it
            // is the first unboxed one in its slot (`firstunwrapped`);
            // otherwise it packs into the slot the prior unboxed attribute
            // already reserved, so its size is `back.storage_needed()`.
            MapNode::Plain(p) => match &p.unboxed {
                Some(u) => {
                    if u.firstunwrapped {
                        p.storageindex + 1
                    } else {
                        unsafe { (*p.back).storage_needed() }
                    }
                }
                None => p.storageindex + 1,
            },
        }
    }

    /// mapdict.py:330 (Terminator) / 481 (PlainAttribute) `num_attributes`.
    pub fn num_attributes(&self) -> usize {
        match self {
            MapNode::Terminator(_) => 0,
            MapNode::Plain(p) => p.num_attributes,
        }
    }

    /// mapdict.py:47 `AbstractAttribute.cache_attrs`.
    pub fn cache_attrs(&self) -> &Mutex<HashMap<(Wtf8Buf, u16), *const CachedAttributeHolder>> {
        match self {
            MapNode::Terminator(t) => &t.cache_attrs,
            MapNode::Plain(p) => &p.cache_attrs,
        }
    }
}

/// mapdict.py:140,487-490 `AbstractAttribute.search`.
///
/// # Safety
/// `node` and its `back` chain must point to live map nodes.
pub unsafe fn node_search(node: MapRef, attrtype: u16) -> Option<MapRef> {
    match unsafe { &*node } {
        MapNode::Terminator(_) => None,
        MapNode::Plain(p) => {
            if p.attrkind == attrtype {
                Some(node)
            } else {
                unsafe { node_search(p.back, attrtype) }
            }
        }
    }
}

/// mapdict.py:118-122 `AbstractAttribute._find_map_attr` — the uncached chain
/// walk.
///
/// # Safety
/// `node` and its `back` chain must point to live map nodes.
pub unsafe fn find_map_attr_chain(mut node: MapRef, name: &Wtf8, attrkind: u16) -> Option<MapRef> {
    while let MapNode::Plain(p) = unsafe { &*node } {
        if attrkind == p.attrkind && name == &*p.name {
            return Some(node);
        }
        node = p.back;
    }
    None
}

/// mapdict.py:694-715 `MapAttrCache` — the per-space attribute lookup cache
/// behind `find_map_attr`. A null `MapRef` slot means "empty"/"not found".
pub struct MapAttrCache {
    attrs: Vec<MapRef>,
    names: Vec<Option<Wtf8Buf>>,
    indexes: Vec<u16>,
    cached_attrs: Vec<MapRef>,
}

// MapRef targets are process-lifetime immutable/interned nodes; the only
// mutable cache slots are protected by MAP_ATTR_CACHE's mutex.
unsafe impl Send for MapAttrCache {}

impl MapAttrCache {
    fn new() -> Self {
        let size = 1usize << METHODCACHESIZEEXP;
        MapAttrCache {
            attrs: vec![std::ptr::null(); size],
            names: vec![None; size],
            indexes: vec![INVALID; size],
            cached_attrs: vec![std::ptr::null(); size],
        }
    }

    /// mapdict.py:705-713 `clear`.
    pub fn clear(&mut self) {
        for slot in self.attrs.iter_mut() {
            *slot = std::ptr::null();
        }
        for slot in self.names.iter_mut() {
            *slot = None;
        }
        for slot in self.indexes.iter_mut() {
            *slot = INVALID;
        }
        for slot in self.cached_attrs.iter_mut() {
            *slot = std::ptr::null();
        }
    }
}

/// `space.fromcache(MapAttrCache)` (mapdict.py:80) — one cache on the shared
/// object space.  Pyre's free-threaded execution contexts share that space, so
/// the cache remains process/interpreter-owned and is synchronized rather than
/// duplicated through TLS.
static MAP_ATTR_CACHE: LazyLock<Mutex<MapAttrCache>> =
    LazyLock::new(|| Mutex::new(MapAttrCache::new()));

/// interp_gc.py:14-17 — clear `space.fromcache(MapAttrCache)` before an
/// explicit full collection so cached map nodes do not retain stale entries.
#[majit_macros::dont_look_inside]
pub fn clear_map_attr_cache() {
    MAP_ATTR_CACHE.lock().unwrap().clear();
}

/// `objectmodel.compute_hash(name)` for a (utf8-encoded) str (mapdict.py:94).
///
/// This ports the `fnv` mode of `compute_hash` — `objectmodel._hash_string`,
/// the modified-FNV string hash (`rpython/rlib/objectmodel.py`). `compute_hash`
/// is specialized through an overridable ll hash function
/// (`get_ll_hash_function`), and the production default selected by
/// `ChoiceOption("hash", ["fnv", "siphash24"], default="siphash24")`
/// (`pypy/config/pypyoption.py:186`) is siphash24, not this — so this is a
/// PRE-EXISTING-ADAPTATION choosing the deterministic `fnv` mode. It only
/// affects `MapAttrCache` bucket distribution — `find_map_attr` always rechecks
/// name+attrkind, so a divergent hash causes at most a cache miss, never a
/// wrong result.
fn compute_name_hash(name: &Wtf8) -> i64 {
    let s = name.as_bytes();
    let length = s.len();
    if length == 0 {
        return -1;
    }
    let mut x: i64 = (s[0] as i64) << 7;
    for &b in s {
        x = 1000003i64.wrapping_mul(x) ^ (b as i64);
    }
    x ^= length as i64;
    x
}

/// mapdict.py:86-117 `AbstractAttribute.find_map_attr` with the method cache
/// always enabled — fuses the dispatcher (mapdict.py:86) and
/// `_find_map_attr_cache` (mapdict.py:100, `@jit.dont_look_inside`). The
/// uncached walk is `find_map_attr_chain` (`_find_map_attr`); the JIT calls
/// that directly rather than tracing this cache path.
///
/// # Safety
/// `self_node` and its `back` chain must point to live map nodes.
pub unsafe fn find_map_attr(self_node: MapRef, name: &Wtf8, attrkind: u16) -> Option<MapRef> {
    const SHIFT2: u32 = u64::BITS - METHODCACHESIZEEXP;
    const SHIFT1: u32 = SHIFT2 - 5;
    // current_object_addr_as_int(self) (mapdict.py:88) — the map node address.
    let attrs_as_int = self_node as usize as i64;
    // unrolled hash computation for the 2-tuple (name, attrkind) (mapdict.py:90-95)
    let c1: i64 = 0x34_5678;
    let c2: i64 = 1_000_003;
    let hash_name = compute_name_hash(name);
    let hash_selector = c2.wrapping_mul(c2.wrapping_mul(c1) ^ hash_name) ^ (attrkind as i64);
    let product = attrs_as_int.wrapping_mul(hash_selector) as u64;
    let attr_hash = ((product ^ (product << SHIFT1)) >> SHIFT2) as usize;

    let mut cache = MAP_ATTR_CACHE.lock().unwrap();
    if cache.attrs[attr_hash] == self_node
        && cache.names[attr_hash].as_deref() == Some(name)
        && cache.indexes[attr_hash] == attrkind
    {
        let cached = cache.cached_attrs[attr_hash];
        return if cached.is_null() { None } else { Some(cached) };
    }
    let attr = unsafe { find_map_attr_chain(self_node, name, attrkind) };
    // Populate the cache, gated on `space._side_effects_ok()`
    // (mapdict.py:110). `_side_effects_ok` returns True except under reverse
    // debugging (not ported); the JIT does not trace this write because the
    // cache path is `@jit.dont_look_inside` and the JIT calls
    // `find_map_attr_chain` directly.
    if crate::baseobjspace::side_effects_ok() {
        cache.attrs[attr_hash] = self_node;
        cache.names[attr_hash] = Some(name.to_owned());
        cache.indexes[attr_hash] = attrkind;
        cache.cached_attrs[attr_hash] = attr.unwrap_or(std::ptr::null());
    }
    attr
}

// ── LOAD_ATTR / STORE_ATTR inline cache (mapdict.py:1416-1653) ─────────
//
// The per-(pycode, nameindex) attribute cache that makes interpreter attribute
// access fast (`pycode._mapdict_caches`). It is distinct from `MapAttrCache`
// above: that one is the space-global `find_map_attr` cache, this one is the
// per-code bytecode-slot cache. Each entry remembers the instance map and the
// owning type's `version_tag` last seen at this name slot, plus the resolved
// attribute node; a read whose object still has that map and whose type still
// has that version_tag re-reads the value straight out of storage, skipping the
// type lookup + map walk.
//
// PyPy holds `map` and `attr` through weakrefs (mapdict.py:1452/1468) because
// its `AbstractAttribute` nodes are GC-managed. pyre interns map nodes as
// immortal leaked `Box`es (`intern_node`/`new_terminator`, see comment at the
// top of this file and lines 190-213), so a raw `MapRef` is the faithful
// equivalent — the weakref could never expire. The map/attr node pointers and
// the u64 version_tag therefore need no GC walking; attribute reads re-read
// the live value through `plain_direct_read` on every hit. The one movable
// reference is the LOAD_METHOD `w_method` slot (mapdict.py:1418), forwarded
// during collection by `pycode::walk_mapdict_method_cache_gc`. (Contingency:
// were map nodes ever made movable, the raw pointers would
// dangle and the whole entry would have to switch to that forwarded design.)

/// mapdict.py:1416-1422 `CacheEntry`. PyPy's shared `INVALID_CACHE_ENTRY`
/// sentinel (a `CacheEntry` carrying a fake map, mapdict.py:1451-1454) is
/// represented here by a `None` slot in `pycode._mapdict_caches`, so this struct
/// only ever describes a *valid* (or stale-but-checked) entry. The debug-only
/// `success_counter`/`failure_counter` (mapdict.py:1419-1420, gated on
/// `withmethodcachecounter`) are omitted.
#[derive(Clone, Copy)]
pub struct MapdictCacheEntry {
    /// mapdict.py:1468 `entry.map_wref` target — the instance map this entry was
    /// filled for. Immortal `MapRef`: pyre interns map nodes as leaked `Box`es,
    /// so a raw pointer stands in for PyPy's weakref.
    pub cached_map: MapRef,
    /// mapdict.py:1470/1472 `entry.attr_wref` target — the resolved attribute
    /// node, re-read live via `plain_direct_read` on every hit. `null` (PyPy
    /// `dead_ref`) when the entry caches no attribute.
    pub cached_attr: MapRef,
    /// mapdict.py:1417/1473 `entry.version_tag` (0 = None).
    pub version_tag: u64,
    /// mapdict.py:1421/1475 `entry.valid_for_store`.
    pub valid_for_store: bool,
    /// mapdict.py:1418/1474 `entry.w_method` — filled only by the LOAD_METHOD
    /// cache (callmethod.py); `null` on the LOAD_ATTR / STORE_ATTR paths.
    pub w_method: PyObjectRef,
}

impl MapdictCacheEntry {
    /// mapdict.py:1431-1434 `is_valid_for_map`.
    ///
    /// # Safety
    /// `map` and `self.cached_map` (when non-null) must point to live map nodes.
    pub unsafe fn is_valid_for_map(&self, map: MapRef, store: bool) -> bool {
        // mapdict.py:1432 `if store and not self.valid_for_store: return False`.
        if store {
            match self.valid_for_store {
                true => {}
                false => return false,
            }
        }
        unsafe { self._is_valid_for_map(map) }
    }

    /// mapdict.py:1436-1447 `_is_valid_for_map` — same instance map (pointer
    /// identity on the immortal node) AND the owning type's current
    /// `version_tag` still equals the cached one.
    ///
    /// # Safety
    /// `map` and `self.cached_map` (when non-null) must point to live map nodes.
    unsafe fn _is_valid_for_map(&self, map: MapRef) -> bool {
        // mapdict.py:1439-1440 `mymap = self.map_wref(); if mymap is not None and
        // mymap is map`.
        if !self.cached_map.is_null() && std::ptr::eq(self.cached_map, map) {
            // mapdict.py:1441 `version_tag = map.terminator.w_cls.version_tag()`.
            let w_cls = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
            let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_cls) };
            // mapdict.py:1442 `if version_tag is self.version_tag`.
            if version_tag == self.version_tag {
                return true;
            }
        }
        false
    }
}

/// mapdict.py:905-906 `W_Root._get_mapdict_map` — the instance's current map
/// (`jit.promote(self.map)`), or null for any object that does not use mapdict
/// (the base `W_Root` implementation returns None). `ensure_mapdict_initialized`
/// is a null-check + early return once the map is set, so this stays cheap on
/// the hot LOAD_ATTR path.
///
/// # Safety
/// `w_obj` must be a live object.
unsafe fn mapdict_map_or_null(w_obj: PyObjectRef) -> MapRef {
    // W_TypeObject is represented with the broad INSTANCE_TYPE family for
    // parts of the object-space dispatch, but PyPy's W_TypeObject does not
    // mix in MapdictStorageMixin.  Let type attribute stores reach the
    // metatype data descriptors (__name__, __qualname__, __module__, ...)
    // instead of treating the type object's header as mapdict storage.
    if !unsafe { has_mapdict_storage(w_obj) } {
        return std::ptr::null();
    }
    unsafe { ensure_mapdict_initialized(w_obj) };
    let inst = unsafe { mapdict_carrier(w_obj) };
    inst._get_mapdict_map()
}

/// mapdict.py:443 `PlainAttribute._direct_read` and mapdict.py:591-598
/// `UnboxedPlainAttribute._direct_read` — the converting attribute read.
/// `_prim_direct_read` (boxed slot, or box the longlong for an unboxed
/// attribute), then for an unboxed attribute whose class has frozen unboxing
/// (`terminator.allow_unboxing == False`) migrate the instance off unboxed
/// storage so the class stops minting unboxed map variants. (Same migration
/// condition as `maybe_migrate_to_boxed`, evaluated here on the already-resolved
/// node rather than re-walking the chain.)
///
/// # Safety
/// `attr` must point to a live `PlainAttribute`; `obj` to its live carrier.
unsafe fn direct_read<O: MapdictObject>(attr: MapRef, obj: &mut O) -> PyObjectRef {
    // mapdict.py:592/600-601 `_prim_direct_read`.
    let w_res = unsafe { plain_direct_read(attr, &*obj) };
    let p = unsafe { (*attr).as_plain() };
    if p.unboxed.is_some()
        && !unsafe { (*p.terminator).as_terminator() }
            .allow_unboxing
            .get()
    {
        // mapdict.py:594-596 `_convert_to_boxed(obj)`.
        unsafe { convert_to_boxed(obj) };
    }
    w_res
}

/// mapdict.py:1461-1477 `_fill_cache`. Store the resolved `(map, attr,
/// version_tag)` into slot `nameindex`. PyPy's `INVALID_CACHE_ENTRY` is a `None`
/// slot, so a fill always writes `Some(entry)`.
///
/// `dont_look_inside` — the thread-local `String`-keyed pycode slot store is not
/// annotator-lowerable, and the caller only reaches here under
/// `not we_are_jitted()`.
///
/// # Safety
/// `pycode` must be a live `PyCode`; `map`/`attr` live map nodes.
#[majit_macros::dont_look_inside]
unsafe fn fill_cache(
    pycode: PyObjectRef,
    nameindex: usize,
    map: MapRef,
    version_tag: u64,
    attr: MapRef,
    w_method: PyObjectRef,
    valid_for_store: bool,
) {
    // mapdict.py:1462 `if not pycode.space._side_effects_ok(): return`.
    if !crate::baseobjspace::side_effects_ok() {
        return;
    }
    let _code_cache_guard = code_cache_lock(pycode);
    let entry = MapdictCacheEntry {
        cached_map: map,
        cached_attr: attr,
        version_tag,
        valid_for_store,
        w_method,
    };
    unsafe { crate::pycode::w_code_mapdict_caches_set(pycode, nameindex, entry) };
}

/// mapdict.py:1507-1524 (LOAD_ATTR) / mapdict.py:1612-1626 (STORE_ATTR) —
/// classify the looked-up class descriptor into an `(attrkind, is_slot)` pair.
/// `INVALID` means "give up": the caller falls to `space.getattr`/`space.setattr`
/// without filling the cache. `is_slot` selects the `"slot"` attrname over the
/// bytecode `name`.
///
/// LOAD additionally caches a non-data descriptor whose type is immutable (the
/// instance dict wins for reads, mapdict.py:1520-1524); STORE has no such branch
/// — a non-data descriptor does not intercept writes, so the cache gives up and
/// the plain `setattr` re-checks each time.
///
/// # Safety
/// `w_type` and `w_descr` (when `Some`) must be live objects.
unsafe fn classify_attr(
    w_type: PyObjectRef,
    w_descr: Option<PyObjectRef>,
    for_store: bool,
) -> (u16, bool) {
    match w_descr {
        // mapdict.py:1509-1510 — no such attr in the class: the common case,
        // read/write the instance dict.
        None => (DICT, false),
        Some(d) => {
            // mapdict.py:1511-1512 — a MutableCell can change without bumping the
            // version_tag, so give up. pyre type dicts store values directly, not
            // wrapped in cells (celldict.rs:17-18 — the cell port has not landed),
            // so this never fires today; kept for structural parity.
            if unsafe { pyre_object::celldict::is_mutable_cell(d) } {
                return (INVALID, false);
            }
            // mapdict.py:1513-1519 — a data descriptor shadows the instance dict;
            // only a `__slots__` Member that belongs to this type is cacheable.
            if unsafe { crate::baseobjspace::is_data_descr(d) } {
                if unsafe { pyre_object::is_member(d) }
                    && !unsafe { pyre_object::w_member_is_direct(d) }
                    && unsafe {
                        crate::baseobjspace::issubtype_w(w_type, pyre_object::w_member_get_cls(d))
                    }
                {
                    // mapdict.py:1518 `("slot", SLOTS_STARTING_FROM + w_descr.index)`.
                    let kind =
                        SLOTS_STARTING_FROM + unsafe { pyre_object::w_member_get_index(d) } as u16;
                    return (kind, true);
                }
                return (INVALID, false);
            }
            // mapdict.py:1520-1524 — LOAD only: a non-data descriptor whose type
            // is immutable (not a heap type) lets the instance dict win and stay
            // cacheable; a heap-type descriptor could gain `__get__`/`__set__`
            // without the cache noticing, so it is not cacheable.
            if !for_store && !unsafe { descr_type_is_heaptype(d) } {
                return (DICT, false);
            }
            (INVALID, false)
        }
    }
}

/// mapdict.py:1520 `space.type(w_descr).is_heaptype()`. Conservatively treats an
/// unresolvable type as a heap type (not cacheable).
///
/// # Safety
/// `w_descr` must be a live object.
unsafe fn descr_type_is_heaptype(w_descr: PyObjectRef) -> bool {
    match crate::typedef::r#type(w_descr) {
        Some(t) => unsafe { pyre_object::typeobject::w_type_is_heaptype(t.as_ptr()) },
        None => true,
    }
}

/// mapdict.py:1479-1490 `LOAD_ATTR_caching`. The interpreter LOAD_ATTR fast
/// path (reached only under `not we_are_jitted()`): a monomorphic cache hit
/// re-reads the value straight out of storage; anything else drops to
/// `load_attr_slowpath`.
///
/// `dont_look_inside` — the cache machinery (thread-local pycode slot,
/// `find_map_attr`'s thread-local) is not annotator-lowerable; the JIT reaches
/// LOAD_ATTR through the `we_are_jitted()` getattr branch in the PyFrame
/// executor, never this function.
///
/// # Safety
/// `pycode` must be a live `PyCode`; `w_obj` a live object.
#[majit_macros::dont_look_inside]
pub unsafe fn load_attr_caching(
    pycode: PyObjectRef,
    w_obj: PyObjectRef,
    nameindex: usize,
    name: &str,
) -> Result<PyObjectRef, PyError> {
    // Copy the immortal-node cache entry under its code-owned lock, then
    // release that lock before touching an instance. Holding both locks
    // across the slow path would extend them across arbitrary descriptor
    // calls, unlike PyPy's single GIL critical section, and permits
    // instance-A -> code -> instance-B / instance-B -> code cycles.
    let entry = {
        let _code_cache_guard = code_cache_lock(pycode);
        unsafe { crate::pycode::w_code_mapdict_caches_get(pycode, nameindex) }
    };
    let map = {
        let _instance_guard = instance_lock(w_obj);
        // mapdict.py:1482 `map = w_obj._get_mapdict_map()`.
        let map = unsafe { mapdict_map_or_null(w_obj) };
        if let Some(e) = entry {
            // mapdict.py:1483-1487 — valid cache entry with a live attr.
            if !map.is_null()
                && unsafe { e.is_valid_for_map(map, false) }
                && e.w_method.is_null()
                && !e.cached_attr.is_null()
            {
                let mut inst = unsafe { mapdict_carrier(w_obj) };
                // mapdict.py:1487 `return attr._direct_read(w_obj)`.
                return Ok(unsafe { direct_read(e.cached_attr, &mut inst) });
            }
        }
        map
    };
    unsafe { load_attr_slowpath(pycode, w_obj, nameindex, name, map) }
}

/// mapdict.py:1492-1537 `LOAD_ATTR_slowpath`.
///
/// `dont_look_inside` — reaches the type-lookup method cache and `find_map_attr`
/// thread-locals; only called from `load_attr_caching`.
///
/// # Safety
/// `pycode` must be a live `PyCode`; `w_obj` a live object; `map` its map
/// (or null).
#[majit_macros::dont_look_inside]
unsafe fn load_attr_slowpath(
    pycode: PyObjectRef,
    w_obj: PyObjectRef,
    nameindex: usize,
    name: &str,
    map: MapRef,
) -> Result<PyObjectRef, PyError> {
    // mapdict.py:1495 `if map is not None:`.
    if !map.is_null() {
        // mapdict.py:1496 `w_type = map.terminator.w_cls`.
        let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
        // mapdict.py:1497-1499 — a custom `__getattribute__` handles the access.
        // pyre has no separate `_handle_getattribute`; `space.getattr`
        // re-dispatches the custom `__getattribute__`, the same result.
        if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
            return crate::baseobjspace::getattr_str(w_obj, name);
        }
        // mapdict.py:1500 `version_tag = w_type.version_tag()`.
        let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
        // mapdict.py:1501 `if version_tag is not None:` (0 = None).
        if version_tag != 0 {
            // mapdict.py:1504-1505 `_, w_descr = _pure_lookup_where_with_method_cache`.
            let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
            // mapdict.py:1507-1524 classify.
            let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, false) };
            // mapdict.py:1526 `if attrkind != INVALID:`.
            if attrkind != INVALID {
                let attrname = if is_slot { "slot" } else { name };
                // The type/descriptor lookup above can execute Python. Re-lock
                // only for the direct map/storage observation and abandon the
                // cache path if another thread changed the instance map.
                let direct = {
                    let _instance_guard = instance_lock(w_obj);
                    let current_map = unsafe { mapdict_map_or_null(w_obj) };
                    if std::ptr::eq(current_map, map) {
                        unsafe { find_map_attr(current_map, Wtf8::new(attrname), attrkind) }.map(
                            |attr| {
                                let mut inst = unsafe { mapdict_carrier(w_obj) };
                                let value = unsafe { direct_read(attr, &mut inst) };
                                (current_map, attr, value)
                            },
                        )
                    } else {
                        None
                    }
                };
                if let Some((current_map, attr, value)) = direct {
                    // mapdict.py:1531-1532 `_fill_cache(...,
                    // valid_for_store=w_type.setattr_if_not_from_object() is None)`.
                    let valid_for_store =
                        unsafe { crate::baseobjspace::setattr_if_not_from_object(w_type) }
                            .is_none();
                    unsafe {
                        fill_cache(
                            pycode,
                            nameindex,
                            current_map,
                            version_tag,
                            attr,
                            std::ptr::null_mut(),
                            valid_for_store,
                        );
                    }
                    // mapdict.py:1533 `return attr._direct_read(w_obj)`.
                    return Ok(value);
                }
            }
        }
    }
    // mapdict.py:1537 `return space.getattr(w_obj, w_name)`.
    crate::baseobjspace::getattr_str(w_obj, name)
}

/// The JIT LOAD_ATTR fast-path resolver: the `load_attr_slowpath`
/// (mapdict.py:1492-1537) resolution steps, but instead of reading the value it
/// returns the ingredients the meta-tracer needs to fold the read to a guarded
/// inline `storage[storageindex]` — `(w_type, version_tag, map, storageindex)`.
///
/// Returns `None` (leave the access on the residual `space.getattr`) for every
/// shape the inline read cannot cover: non-instance receiver, missing map,
/// custom `__getattribute__`, uncacheable `version_tag`, a data-descriptor /
/// `INVALID` classification, an attribute not present on this instance's map,
/// or an unboxed slot (whose read boxes a longlong — not a plain slot fetch).
/// This is the mapdict analog of `baseobjspace::load_method_fast_path`, sharing
/// the same gates so the symbolic trace and the concrete frame agree.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn load_attr_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, MapRef, usize)> {
    // mapdict.py:1495 `if map is not None:` — also filters non-instances.
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    // mapdict.py:1496 `w_type = map.terminator.w_cls`.
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    // mapdict.py:1497-1499 — a custom `__getattribute__` handles the access;
    // not soundly foldable to a storage read.
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1500-1501 `version_tag = w_type.version_tag(); if is not None:`.
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // mapdict.py:1504-1524 `_pure_lookup_where_with_method_cache` + classify.
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
    let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, false) };
    // mapdict.py:1526 `if attrkind != INVALID:`.
    if attrkind == INVALID {
        return None;
    }
    let attrname = if is_slot { "slot" } else { name };
    // mapdict.py:1527 `attr = map.find_map_attr(attrname, attrkind)`.
    let attr = unsafe { find_map_attr(map, Wtf8::new(attrname), attrkind) }?;
    let p = unsafe { (*attr).as_plain() };
    // The inline read is a plain `storage[storageindex]` fetch; an unboxed slot
    // (mapdict.py:591-601) instead boxes a longlong out of a shared list, so
    // leave it on the residual.
    if p.unboxed.is_some() {
        return None;
    }
    Some((w_type, version_tag, map, p.storageindex))
}

/// The [`load_attr_fast_path`] twin for a receiver that keeps its attributes in
/// a `newdict(instance=True)` dictionary rather than in header mapdict storage
/// (`mapdict.py:1299-1303 make_instance_dict`). It applies the same
/// `load_attr_slowpath` gates (mapdict.py:1496-1527) and differs only in where
/// the map comes from — the fake carrier the dictionary erases into rather than
/// the receiver's own header. Returns the type and its version tag alongside
/// the fold ingredients: the carrier, its map, the `storageindex`, and the
/// `(typ, listindex)` pair when the attribute took an unboxed transition.
///
/// The carrier's map holds only dictionary attributes, so a `__slots__` member
/// — which mapdict.py:1518 would resolve through the receiver's own map — is
/// declined here rather than looked up under the `"slot"` name.
///
/// Declines a dictionary that is not `MapDictStrategy`-backed. The caller must
/// still prove that at trace time before dereferencing `dstorage` as a carrier:
/// a devolved dictionary erases a smaller storage box, and reading a carrier
/// field off it is out of bounds.
///
/// # Safety
/// `w_obj` must be live, and `w_dict` must point at its live `W_DictObject`.
pub unsafe fn instance_dict_attr_fast_path(
    w_obj: PyObjectRef,
    w_dict: PyObjectRef,
    name: &str,
) -> Option<(
    PyObjectRef,
    u64,
    PyObjectRef,
    MapRef,
    usize,
    Option<(UnboxType, usize)>,
)> {
    // mapdict.py:1496 `w_type = map.terminator.w_cls` — the carrier's own
    // terminator is the shared fake one, so the type comes off the receiver.
    let w_type = unsafe { (*w_obj).w_class };
    if w_type.is_null() {
        return None;
    }
    // mapdict.py:1497-1499 — a custom `__getattribute__` handles the access.
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1500-1501 `version_tag = w_type.version_tag(); if is not None:`.
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // mapdict.py:1504-1526 `_pure_lookup_where_with_method_cache` + classify.
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
    let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, false) };
    if attrkind != DICT || is_slot {
        return None;
    }
    let dict = unsafe { &*(w_dict as *const pyre_object::dictmultiobject::W_DictObject) };
    if dict.dstrategy.strategy_kind() != pyre_object::dictmultiobject::StrategyKind::Map {
        return None;
    }
    let carrier = dict.dstorage as PyObjectRef;
    if carrier.is_null() {
        return None;
    }
    let map = unsafe { mapdict_carrier(carrier) }._get_mapdict_map();
    if map.is_null() || unsafe { map_is_devolved(map) } {
        return None;
    }
    // mapdict.py:1527 `attr = map.find_map_attr(attrname, attrkind)`.
    let attr = unsafe { find_map_attr(map, Wtf8::new(name), DICT) }?;
    let plain = unsafe { (*attr).as_plain() };
    // `_direct_read` migrates the instance off unboxed storage once its class
    // has frozen unboxing (mapdict.py:594-596); the fold performs
    // `_prim_direct_read` alone, so a frozen class stays on the residual that
    // still runs the migration.
    if plain.unboxed.is_some()
        && !unsafe { (*plain.terminator).as_terminator() }
            .allow_unboxing
            .get()
    {
        return None;
    }
    Some((
        w_type,
        version_tag,
        carrier,
        map,
        plain.storageindex,
        plain.unboxed.as_ref().map(|u| (u.typ, u.listindex)),
    ))
}

/// Shared resolution for [`property_get_fast_path`] / [`property_set_fast_path`]:
/// when `name` on `w_obj`'s type resolves to a `property` data descriptor,
/// return the type, its cacheable version tag, and the property object.  Applies
/// the shape gates common to both directions — a mapdict instance and a non-zero
/// `version_tag`.  The read/write hook gate differs (`__getattribute__` for the
/// getter, `__setattr__` for the setter), so each caller applies its own.  A
/// property is a data descriptor, so it takes precedence over the instance dict
/// and the version-pinned type lookup is authoritative; the instance map is not
/// read.
///
/// # Safety
/// `w_obj` must be a live object.
unsafe fn property_descr_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, PyObjectRef)> {
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type(w_type, name) }?;
    if !unsafe { pyre_object::descriptor::is_property(w_descr) } {
        return None;
    }
    Some((w_type, version_tag, w_descr))
}

/// LOAD_ATTR `property` fast path: return the type, version tag, and Python
/// `fget` when `obj.name` reads a property getter, so the full-body walker can
/// inline `fget(obj)` in place of the opaque `getattr` residual.  Returns `None`
/// (leave the residual) for a write-only property or any shape
/// [`property_descr_fast_path`] declines.  A custom `__getattribute__` owns the
/// read (mapdict.py:1497-1499), so it declines to the residual.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn property_get_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, PyObjectRef)> {
    let (w_type, version_tag, w_descr) = unsafe { property_descr_fast_path(w_obj, name) }?;
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    let fget = unsafe { pyre_object::descriptor::w_property_get_fget(w_descr) };
    if fget.is_null() || unsafe { pyre_object::pyobject::is_none(fget) } {
        return None;
    }
    Some((w_type, version_tag, fget))
}

/// STORE_ATTR `property` fast path: the setter twin of
/// [`property_get_fast_path`], returning the type, version tag, and Python
/// `fset` when `obj.name = value` writes a property setter.  Returns `None`
/// (leave the residual) for a read-only property or any shape
/// [`property_descr_fast_path`] declines.  A custom `__setattr__` owns the write
/// (mapdict.py:1612-1614), so it declines to the residual.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn property_set_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, PyObjectRef)> {
    let (w_type, version_tag, w_descr) = unsafe { property_descr_fast_path(w_obj, name) }?;
    if unsafe { crate::baseobjspace::setattr_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    let fset = unsafe { pyre_object::descriptor::w_property_get_fset(w_descr) };
    if fset.is_null() || unsafe { pyre_object::pyobject::is_none(fset) } {
        return None;
    }
    Some((w_type, version_tag, fset))
}

/// The unboxed counterpart of [`load_attr_fast_path`].  It applies the same
/// LOAD_ATTR resolution gates, but accepts only an `UnboxedPlainAttribute`
/// and returns the shared longlong-list coordinates and type needed to perform
/// `_prim_direct_read` (mapdict.py:600-601).
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn load_attr_unboxed_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, MapRef, usize, usize, UnboxType)> {
    // mapdict.py:1495 `if map is not None:` — also filters non-instances.
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    // mapdict.py:1496 `w_type = map.terminator.w_cls`.
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    // mapdict.py:1497-1499 — a custom `__getattribute__` handles the access;
    // not soundly foldable to a storage read.
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1500-1501 `version_tag = w_type.version_tag(); if is not None:`.
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // mapdict.py:1504-1524 `_pure_lookup_where_with_method_cache` + classify.
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
    let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, false) };
    // mapdict.py:1526 `if attrkind != INVALID:`.
    if attrkind == INVALID {
        return None;
    }
    let attrname = if is_slot { "slot" } else { name };
    // mapdict.py:1527 `attr = map.find_map_attr(attrname, attrkind)`.
    let attr = unsafe { find_map_attr(map, Wtf8::new(attrname), attrkind) }?;
    let p = unsafe { (*attr).as_plain() };
    let u = p.unboxed.as_ref()?;
    // mapdict.py:1539 — `LOAD_ATTR_caching` resolves to `attr._direct_read`,
    // which for an unboxed attribute also migrates `obj` off unboxed storage
    // once its class froze unboxing (mapdict.py:594-596). The folded read
    // performs `_prim_direct_read` alone, so leave a frozen class to the
    // residual, whose `direct_read` still runs that migration.
    if !unsafe { (*p.terminator).as_terminator() }
        .allow_unboxing
        .get()
    {
        return None;
    }
    Some((w_type, version_tag, map, p.storageindex, u.listindex, u.typ))
}

/// The STORE_ATTR counterpart of [`load_attr_unboxed_fast_path`].  An
/// existing plain unboxed slot resolves through the same map lookup; the
/// caller separately proves that the incoming value has the slot's unbox
/// type before performing the in-place write (mapdict.py:615-619).
///
/// The resolved attribute node is returned so the caller can mark it
/// `ever_mutated` once it has committed to folding the write out of the trace.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn store_attr_unboxed_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, MapRef, usize, usize, UnboxType, MapRef)> {
    if name == "__class__" {
        return None;
    }
    // mapdict.py:1591 `if map is not None:` — also filters non-instances.
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    // mapdict.py:1592 `w_type = map.terminator.w_cls`.
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    // mapdict.py:1612-1614 — a custom `__setattr__` owns the write.
    if unsafe { crate::baseobjspace::setattr_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1630-1631 — STORE caching also requires the standard
    // `__getattribute__` so the cached map invariant remains valid.
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1616 `if version_tag is not None:`.
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // mapdict.py:1618-1627 `_pure_lookup_where_with_method_cache` + classify.
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
    let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, true) };
    if attrkind == INVALID {
        return None;
    }
    let attrname = if is_slot { "slot" } else { name };
    // mapdict.py:1628 `attr = map.find_map_attr(attrname, attrkind)`.
    let attr = unsafe { find_map_attr(map, Wtf8::new(attrname), attrkind) }?;
    let p = unsafe { (*attr).as_plain() };
    let u = p.unboxed.as_ref()?;
    Some((
        w_type,
        version_tag,
        map,
        p.storageindex,
        u.listindex,
        u.typ,
        attr,
    ))
}

/// Resolve an existing boxed plain slot through the STORE_ATTR caching gates.
/// Unlike an unboxed slot, the value remains a `PyObjectRef`, so the caller
/// needs only the guarded map and storage index for the direct write
/// (mapdict.py:446-447).
///
/// The resolved attribute node is returned so the caller can mark it
/// `ever_mutated` once it has committed to folding the write out of the trace.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn store_attr_boxed_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, MapRef, usize, MapRef)> {
    if name == "__class__" {
        return None;
    }
    // mapdict.py:1591 `if map is not None:` — also filters non-instances.
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    // mapdict.py:1592 `w_type = map.terminator.w_cls`.
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    // mapdict.py:1612-1614 — a custom `__setattr__` owns the write.
    if unsafe { crate::baseobjspace::setattr_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1630-1631 — STORE caching also requires the standard
    // `__getattribute__` so the cached map invariant remains valid.
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1616 `if version_tag is not None:`.
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // mapdict.py:1618-1627 `_pure_lookup_where_with_method_cache` + classify.
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
    let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, true) };
    if attrkind == INVALID {
        return None;
    }
    let attrname = if is_slot { "slot" } else { name };
    // mapdict.py:1628 `attr = map.find_map_attr(attrname, attrkind)`.
    let attr = unsafe { find_map_attr(map, Wtf8::new(attrname), attrkind) }?;
    let p = unsafe { (*attr).as_plain() };
    if p.unboxed.is_some() {
        return None;
    }
    Some((w_type, version_tag, map, p.storageindex, attr))
}

/// mapdict.py:1635-1636 `if not attr.ever_mutated: attr.ever_mutated = True`
/// — performed by the caller once it has committed to folding the store
/// out of the trace, so a declined resolution mutates nothing.
///
/// # Safety
/// `attr` must point at a live `PlainAttribute` map node.
pub unsafe fn mark_attr_ever_mutated(attr: MapRef) {
    unsafe { (*attr).as_plain() }.set_ever_mutated(true);
}

/// The map transition an attribute-adding STORE_ATTR takes, resolved without
/// performing it.  See [`store_attr_add_fast_path`].
pub struct StoreAttrAdd {
    /// The receiver's type, whose `version_tag` the caller guards.
    pub w_type: PyObjectRef,
    /// That type's live version tag.
    pub version_tag: u64,
    /// The instance's current map, which the caller guards.
    pub map: MapRef,
    /// The cached transition holder whose `typ` and `attr` the resolution read.
    pub holder: *const CachedAttributeHolder,
    /// The `PlainAttribute` the instance's map becomes — a direct child of
    /// `map`, so the transition is `map -> new_map` with no reordering.
    pub new_map: MapRef,
    /// `new_map.storageindex`, which is also `map.storage_needed()`: the
    /// transition appends exactly one slot, and the value lands in it.
    pub storageindex: usize,
    /// `Some` for a fresh-slot unboxed transition: the new slot holds a
    /// one-element longlong list. Only `Int` today; a float pick stays on
    /// the residual.
    pub unbox_type: Option<UnboxType>,
    /// The value returned by `pick_unbox_type`. This is distinct from
    /// `unbox_type`, which is `None` when a cached boxed transition serves an
    /// unboxable value. `picked_unbox.is_some()` means the pick read the
    /// terminator's `allow_unboxing` flag as true.
    pub picked_unbox: Option<UnboxType>,
}

/// Resolve the `map -> PlainAttribute` transition that a STORE_ATTR adding a
/// not-yet-present boxed attribute would take, *without* performing it, so the
/// JIT can emit `_set_mapdict_increase_storage1` (mapdict.py:942-959) as trace
/// operations instead of an opaque residual call.  Every value the caller needs
/// is a trace-time constant once the receiver's map and its type's version tag
/// are guarded.
///
/// Declines anything the plain grow-by-one shape does not cover, so the caller
/// can fall back to the general residual: an attribute already in the map (the
/// [`store_attr_boxed_fast_path`] / [`store_attr_unboxed_fast_path`] cases), a
/// value that would take an unboxed float slot or a shared unboxed slot, a
/// terminator that routes the write elsewhere, the `_reorder_and_add` case
/// (mapdict.py:204-258), the `LIMIT_MAP_ATTRIBUTES` devolve, and a storage
/// block the collector does not own (whose replacement the interpreter would
/// have to free).  A fresh-slot unboxed int add resolves with `unbox_type =
/// Some(Int)` and grows storage by one exactly like the boxed add.
///
/// Resolving interns the transition through `_find_branch_to_move_into`, which
/// `add_attr` would do anyway on this very store; it is idempotent and the
/// resulting node is immortal.
///
/// # Safety
/// `w_obj` must be a live object and `w_value` a live value.
pub unsafe fn store_attr_add_fast_path(
    w_obj: PyObjectRef,
    name: &str,
    w_value: PyObjectRef,
) -> Option<StoreAttrAdd> {
    if name == "__class__" {
        return None;
    }
    // mapdict.py:1591 `if map is not None:` — also filters non-instances.
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    let terminator = unsafe { (*map).terminator() };
    let term = unsafe { (*terminator).as_terminator() };
    let w_type = term.w_cls;
    if w_type.is_null() {
        return None;
    }
    // mapdict.py:1612-1614 / 1630-1631 — a custom `__setattr__` owns the
    // write, and STORE caching also requires the standard `__getattribute__`.
    if unsafe { crate::baseobjspace::setattr_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }.is_some() {
        return None;
    }
    // mapdict.py:1616 `if version_tag is not None:`.
    let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
    if version_tag == 0 {
        return None;
    }
    // mapdict.py:1618-1627 `_pure_lookup_where_with_method_cache` + classify.
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
    let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, true) };
    if attrkind == INVALID {
        return None;
    }
    let attrname = Wtf8::new(if is_slot { "slot" } else { name });
    // This arm is the ADD; an attribute already in the map belongs to the
    // in-place write paths (mapdict.py:1628 `map.find_map_attr`).
    if unsafe { find_map_attr(map, attrname, attrkind) }.is_some() {
        return None;
    }
    // mapdict.py:312-321 `_write_terminator` — a `NoDictTerminator` rejects a
    // DICT write and a `DevolvedDictTerminator` sends it to the materialised
    // instance dict; neither reaches `add_attr`.
    if attrkind == DICT && term.kind != TerminatorKind::Dict {
        return None;
    }
    // The int unboxed shape is emittable — the trace builds the same int
    // GcArray `erase_unboxed` allocates; a float pick stays on the residual.
    let unbox = unsafe { pick_unbox_type(map, w_value) };
    if unbox == Some(UnboxType::Float) {
        return None;
    }
    // The emitted transition copies the live slots into the wider block with
    // reference loads. An unboxed attribute already in the chain is fine: its
    // slot holds the longlong list's GcArray, which is a reference like any
    // other (mapdict.py:600-601 `_prim_direct_read`).
    // mapdict.py:170-193 — `number_to_readd != 0` is `_reorder_and_add`, which
    // pops and re-adds intermediate attributes.
    let (number_to_readd, holder) =
        unsafe { find_branch_to_move_into(map, attrname, attrkind, unbox) };
    if number_to_readd != 0 {
        return None;
    }
    let new_map = unsafe { holder_pick_attr(holder, unbox) };
    if !unsafe { (*new_map).is_plain() } {
        return None;
    }
    let p = unsafe { (*new_map).as_plain() };
    if !std::ptr::eq(p.back, map) {
        return None;
    }
    let unbox_type = match &p.unboxed {
        // A cached boxed transition serves an unboxable value too; the write
        // stays boxed.
        None => None,
        // Only the fresh-slot shape grows storage by one; a shared-slot add
        // rewrites an existing slot's list in place.
        Some(u) => {
            if !u.firstunwrapped || u.listindex != 0 {
                return None;
            }
            debug_assert_eq!(Some(u.typ), unbox);
            Some(u.typ)
        }
    };
    // mapdict.py:317-323 — a DICT add that reaches the attribute limit devolves
    // the instance's `__dict__` into a text-strategy dict.
    if attrkind == DICT && unsafe { (*new_map).num_attributes() } >= LIMIT_MAP_ATTRIBUTES {
        return None;
    }
    // The grow-by-one shape `_set_mapdict_increase_storage1` implements: the
    // new slot is appended at the current length.
    let storage_needed = unsafe { (*map).storage_needed() };
    if p.storageindex != storage_needed
        || unsafe { (*new_map).storage_needed() } != storage_needed + 1
    {
        return None;
    }
    // `grow_instance_items_block` frees the block it replaces, which is a no-op
    // only for a collector-owned block.  A `std::alloc` fallback block (no GC
    // hook) would leak once the emitted allocation replaces it without a free.
    let inst = unsafe { mapdict_carrier(w_obj) };
    let storage = unsafe { inst.storage() };
    if !storage.is_null() && !pyre_object::gc_hook::try_gc_owns_object(storage as *mut u8) {
        return None;
    }
    // The emitted transition allocates a `storage_needed + 1` block, so spare
    // capacity would make it a shrink — which is exactly what
    // `_set_mapdict_storage_and_map` refuses to do. An instance that has had an
    // attribute deleted therefore keeps the interpreter's add path, which fills
    // the spare tail in place.
    if unsafe { pyre_object::object_array::items_block_capacity(inst.storage()) } != storage_needed
    {
        return None;
    }
    Some(StoreAttrAdd {
        w_type,
        version_tag,
        map,
        holder,
        new_map,
        storageindex: p.storageindex,
        unbox_type,
        picked_unbox: unbox,
    })
}

/// Perform the transition [`store_attr_add_fast_path`] resolved.  The JIT's
/// walk is the authoritative execution path, so it applies the store itself
/// after emitting the equivalent trace operations.
///
/// # Safety
/// `w_obj` must be the live instance the resolution came from, still on
/// `resolved.map`, and `w_value` a live value.
pub unsafe fn store_attr_add_commit(
    w_obj: PyObjectRef,
    resolved: &StoreAttrAdd,
    w_value: PyObjectRef,
) {
    let _instance_guard = instance_lock(w_obj);
    let mut inst = unsafe { mapdict_carrier(w_obj) };
    debug_assert!(std::ptr::eq(inst._get_mapdict_map(), resolved.map));
    // mapdict.py:449-459 with `storage_needed() > _mapdict_storage_length()`,
    // which `store_attr_add_fast_path` proved by construction.
    match resolved.unbox_type {
        None => inst._set_mapdict_increase_storage1(resolved.new_map, w_value),
        // `switch_map_and_write_increase_storage1`'s `firstunwrapped` arm: a
        // fresh one-element longlong list occupies the new slot.
        Some(typ) => {
            let val = unsafe { unbox_value(typ, w_value) };
            let unboxed = erase_unboxed(&[val]);
            inst._set_mapdict_increase_storage1(resolved.new_map, unboxed);
        }
    }
}

/// mapdict.py:914-916 `_mapdict_read_storage(storageindex)` for a
/// `W_ObjectObject` — the boxed-slot read the JIT LOAD_ATTR fast path folds
/// to. `load_attr_fast_path` already established that the resolved attribute
/// is a boxed plain slot (not unboxed), so this is a straight
/// `storage[storageindex]` fetch with no conversion.
///
/// # Safety
/// `w_obj` must be a live `W_ObjectObject` whose `storage` holds at least
/// `storageindex + 1` slots — guaranteed when `storageindex` came from this
/// instance's map via `load_attr_fast_path`.
pub unsafe fn read_boxed_storage(w_obj: PyObjectRef, storageindex: usize) -> PyObjectRef {
    let inst = unsafe { mapdict_carrier(w_obj) };
    inst._mapdict_read_storage(storageindex)
}

/// Write a boxed attribute value directly to an existing storage slot,
/// matching `PlainAttribute._direct_write` (mapdict.py:446-447).
///
/// # Safety
/// `w_obj` must be a live `W_ObjectObject` whose guarded map owns the supplied
/// boxed storage index.
pub unsafe fn write_boxed_storage(w_obj: PyObjectRef, storageindex: usize, value: PyObjectRef) {
    let mut inst = unsafe { mapdict_carrier(w_obj) };
    inst._mapdict_write_storage(storageindex, value);
}

/// Read the raw longlong at an unboxed attribute's `(storageindex,
/// listindex)`, matching `_prim_direct_read` before it boxes the value
/// (mapdict.py:600-601).
///
/// # Safety
/// `w_obj` must be a live `W_ObjectObject` whose guarded map owns the supplied
/// unboxed storage coordinates.
pub unsafe fn read_unboxed_storage_raw(
    w_obj: PyObjectRef,
    storageindex: usize,
    listindex: usize,
) -> i64 {
    let slot = unsafe { read_boxed_storage(w_obj, storageindex) };
    unsafe { unboxed_items(slot)[listindex] }
}

/// Write a raw longlong to an unboxed attribute's `(storageindex,
/// listindex)`, matching the same-type `_direct_write` update before any
/// boxing conversion (mapdict.py:615-619).
///
/// # Safety
/// `w_obj` must be a live `W_ObjectObject` whose guarded map owns the supplied
/// unboxed storage coordinates.
pub unsafe fn write_unboxed_storage_raw(
    w_obj: PyObjectRef,
    storageindex: usize,
    listindex: usize,
    raw: i64,
) {
    let slot = unsafe { read_boxed_storage(w_obj, storageindex) };
    unsafe {
        unboxed_items_mut(slot)[listindex] = raw;
    }
}

/// mapdict.py:1574-1586 `STORE_ATTR_caching`. The interpreter STORE_ATTR fast
/// path (reached only under `not we_are_jitted()`): a monomorphic hit writes
/// straight through the cached attribute; anything else drops to
/// `store_attr_slowpath`.
///
/// `dont_look_inside` — same rationale as `load_attr_caching`.
///
/// # Safety
/// `pycode` must be a live `PyCode`; `w_obj` a live object.
#[majit_macros::dont_look_inside]
pub unsafe fn store_attr_caching(
    pycode: PyObjectRef,
    w_obj: PyObjectRef,
    nameindex: usize,
    name: &str,
    w_value: PyObjectRef,
) -> Result<(), PyError> {
    let entry = {
        let _code_cache_guard = code_cache_lock(pycode);
        unsafe { crate::pycode::w_code_mapdict_caches_get(pycode, nameindex) }
    };
    let map = {
        let _instance_guard = instance_lock(w_obj);
        // mapdict.py:1577 `map = w_obj._get_mapdict_map()`.
        let map = unsafe { mapdict_map_or_null(w_obj) };
        if let Some(e) = entry {
            // mapdict.py:1578-1585 — valid cache entry with a live attr.
            if !map.is_null()
                && unsafe { e.is_valid_for_map(map, true) }
                && e.w_method.is_null()
                && !e.cached_attr.is_null()
            {
                let attr = e.cached_attr;
                let p = unsafe { (*attr).as_plain() };
                // mapdict.py:1582-1583 `if not attr.ever_mutated: attr.ever_mutated = True`.
                p.set_ever_mutated(true);
                // mapdict.py:1584 `attr._direct_write(w_obj, w_value)`.
                let mut inst = unsafe { mapdict_carrier(w_obj) };
                unsafe { plain_direct_write(attr, &mut inst, w_value) };
                return Ok(());
            }
        }
        map
    };
    unsafe { store_attr_slowpath(pycode, w_obj, nameindex, name, map, w_value, entry) }
}

/// mapdict.py:1588-1653 `STORE_ATTR_slowpath`.
///
/// `dont_look_inside` — same rationale as `load_attr_slowpath`.
///
/// # Safety
/// `pycode` must be a live `PyCode`; `w_obj` a live object; `map` its map
/// (or null).
#[majit_macros::dont_look_inside]
unsafe fn store_attr_slowpath(
    pycode: PyObjectRef,
    w_obj: PyObjectRef,
    nameindex: usize,
    name: &str,
    map: MapRef,
    w_value: PyObjectRef,
    entry: Option<MapdictCacheEntry>,
) -> Result<(), PyError> {
    // `object.__class__` is a getset data descriptor in PyPy, so `_classify_attr`
    // (classify_attr) marks it INVALID and the store falls through to
    // `space.setattr` (mapdict.py:1653) — the assignment re-roots the instance
    // type via `descr_set___class__`, not the instance dict.  pyre models
    // `__class__` through `object_setattr`'s special-case rather than a getset,
    // so it never surfaces as a data descriptor for classify_attr to reject;
    // exclude it from the store cache here so `obj.__class__ = NewCls` reaches
    // that special-case instead of being mis-stored as an ordinary instance-dict
    // attribute (which leaves the real type unchanged).
    if name == "__class__" {
        return crate::baseobjspace::setattr_str(w_obj, name, w_value).map(|_| ());
    }
    // mapdict.py:1591 `if map is not None:`.
    if !map.is_null() {
        // mapdict.py:1592 `w_type = map.terminator.w_cls`.
        let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
        // mapdict.py:1593 `version_tag = w_type.version_tag()`.
        let version_tag = unsafe { crate::baseobjspace::w_type_version_tag(w_type) };
        // mapdict.py:1596-1611 — fast path for stores that add a new attribute
        // that this slot has already cached the transition for.
        if let Some(e) = entry
            && e.valid_for_store
            && version_tag == e.version_tag
        {
            let entry_map = e.cached_map;
            let attr_to_add = e.cached_attr;
            // mapdict.py:1599-1602 `entry_map is not None and
            // isinstance(entry_map, PlainAttribute) and attr_to_add is entry_map
            // and entry_map.back is map`.
            if !entry_map.is_null()
                && unsafe { (*entry_map).is_plain() }
                && std::ptr::eq(attr_to_add, entry_map)
                && std::ptr::eq(unsafe { (*entry_map).as_plain() }.back, map)
            {
                // mapdict.py:1603-1606 — for an unboxed attr the new value must
                // match the unbox type, else fall through to the general path.
                let p = unsafe { (*attr_to_add).as_plain() };
                let typsafe = match &p.unboxed {
                    Some(u) => unsafe { value_has_unbox_type(u.typ, w_value) },
                    None => true,
                };
                if typsafe {
                    let switched = {
                        let _instance_guard = instance_lock(w_obj);
                        let current_map = unsafe { mapdict_map_or_null(w_obj) };
                        if std::ptr::eq(current_map, map) {
                            // mapdict.py:1610
                            // `_switch_map_and_write_increase_storage1`.
                            let mut inst = unsafe { mapdict_carrier(w_obj) };
                            unsafe {
                                switch_map_and_write_increase_storage1(
                                    attr_to_add,
                                    &mut inst,
                                    w_value,
                                )
                            };
                            true
                        } else {
                            false
                        }
                    };
                    if switched {
                        return Ok(());
                    }
                }
            }
        }
        // mapdict.py:1612-1614 — a custom `__setattr__` handles the store. pyre
        // re-dispatches through `space.setattr` (no separate helper).
        if unsafe { crate::baseobjspace::setattr_if_not_from_object(w_type) }.is_some() {
            return crate::baseobjspace::setattr_str(w_obj, name, w_value).map(|_| ());
        }
        // mapdict.py:1616 `if version_tag is not None:` (0 = None).
        if version_tag != 0 {
            // mapdict.py:1618-1619 `_, w_descr = _pure_lookup_where_with_method_cache`.
            let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
            // mapdict.py:1620-1626 classify (no non-data heaptype branch for STORE).
            let (attrkind, is_slot) = unsafe { classify_attr(w_type, w_descr, true) };
            // mapdict.py:1627 `if attrkind != INVALID:`.
            if attrkind != INVALID {
                let attrname = if is_slot { "slot" } else { name };
                // mapdict.py:1628 `attr = map.find_map_attr(attrname, attrkind)`.
                match unsafe { find_map_attr(map, Wtf8::new(attrname), attrkind) } {
                    Some(attr) => {
                        let written = {
                            let _instance_guard = instance_lock(w_obj);
                            let current_map = unsafe { mapdict_map_or_null(w_obj) };
                            if !std::ptr::eq(current_map, map)
                                || unsafe {
                                    find_map_attr(current_map, Wtf8::new(attrname), attrkind)
                                }
                                .is_none()
                            {
                                false
                            } else {
                                let p = unsafe { (*attr).as_plain() };
                                // mapdict.py:1632-1633
                                // `if not attr.ever_mutated: ...`.
                                p.set_ever_mutated(true);
                                // mapdict.py:1634 `attr._direct_write(...)`.
                                let mut inst = unsafe { mapdict_carrier(w_obj) };
                                unsafe { plain_direct_write(attr, &mut inst, w_value) };
                                true
                            }
                        };
                        if !written {
                            return crate::baseobjspace::setattr_str(w_obj, name, w_value)
                                .map(|_| ());
                        }
                        // mapdict.py:1630-1631 — fill only when there is no custom
                        // `__getattribute__` to upset the cache invariant.
                        if unsafe { crate::baseobjspace::getattribute_if_not_from_object(w_type) }
                            .is_none()
                        {
                            unsafe {
                                fill_cache(
                                    pycode,
                                    nameindex,
                                    map,
                                    version_tag,
                                    attr,
                                    std::ptr::null_mut(),
                                    true,
                                );
                            }
                        }
                        return Ok(());
                    }
                    None => {
                        // mapdict.py:1636-1648 — add a brand-new DICT attribute via
                        // the DictTerminator, then fill the slot with the resulting
                        // transition map.
                        if attrkind == DICT
                            && unsafe { (*(*map).terminator()).as_terminator() }.kind
                                == TerminatorKind::Dict
                        {
                            let term = unsafe { (*map).terminator() };
                            let mapnew = {
                                let _instance_guard = instance_lock(w_obj);
                                let current_map = unsafe { mapdict_map_or_null(w_obj) };
                                if !std::ptr::eq(current_map, map) {
                                    std::ptr::null()
                                } else {
                                    let mut inst = unsafe { mapdict_carrier(w_obj) };
                                    // mapdict.py:1639
                                    // `map.terminator._write_terminator(...)`.
                                    unsafe {
                                        write_terminator(
                                            term,
                                            &mut inst,
                                            Wtf8::new(name),
                                            attrkind,
                                            w_value,
                                        )
                                    };
                                    // mapdict.py:1640
                                    // `mapnew = w_obj._get_mapdict_map()`.
                                    inst._get_mapdict_map()
                                }
                            };
                            if mapnew.is_null() {
                                return crate::baseobjspace::setattr_str(w_obj, name, w_value)
                                    .map(|_| ());
                            }
                            // mapdict.py:1642-1648 — fill only when no attribute
                            // reordering happened (the new attr is the leaf whose
                            // `back` is the pre-write map).
                            if unsafe { (*mapnew).is_plain() }
                                && std::ptr::eq(unsafe { (*mapnew).as_plain() }.back, map)
                                && unsafe {
                                    crate::baseobjspace::getattribute_if_not_from_object(w_type)
                                }
                                .is_none()
                            {
                                unsafe {
                                    fill_cache(
                                        pycode,
                                        nameindex,
                                        mapnew,
                                        version_tag,
                                        mapnew,
                                        std::ptr::null_mut(),
                                        true,
                                    );
                                }
                            }
                            return Ok(());
                        }
                    }
                }
            }
        }
    }
    // mapdict.py:1653 `space.setattr(w_obj, w_name, w_value)`.
    crate::baseobjspace::setattr_str(w_obj, name, w_value).map(|_| ())
}

// ── obj storage protocol (mapdict.py:904-964 MapdictStorageMixin) ──────
//
// The map-node layer reads and writes attribute values through this trait.
// PyPy mixes `MapdictStorageMixin` into the instance class; pyre's instance
// (W_ObjectObject) implements this trait instead. Storage holds
// `PyObjectRef`, so PyPy's `erase_item`/`unerase_item` (rerased boxing of a
// W_Root into the untyped storage list) are the identity here.

pub trait MapdictObject {
    /// mapdict.py:905-906 `_get_mapdict_map` (`jit.promote(self.map)`).
    fn _get_mapdict_map(&self) -> MapRef;
    /// mapdict.py:907-908 `_set_mapdict_map`.
    fn _set_mapdict_map(&mut self, map: MapRef);
    /// mapdict.py:914-916 `_mapdict_read_storage`.
    fn _mapdict_read_storage(&self, storageindex: usize) -> PyObjectRef;
    /// mapdict.py:918-919 `_mapdict_write_storage`.
    fn _mapdict_write_storage(&mut self, storageindex: usize, value: PyObjectRef);
    /// mapdict.py:921-924 `_mapdict_storage_length` (= `self.map.storage_needed()`).
    fn _mapdict_storage_length(&self) -> usize;
    /// mapdict.py:926-939 `_mapdict_pop_attribute`.
    fn _mapdict_pop_attribute(&mut self, map: MapRef);
    /// mapdict.py:942-959 `_set_mapdict_increase_storage1`.
    fn _set_mapdict_increase_storage1(&mut self, map: MapRef, value: PyObjectRef);
    /// mapdict.py:961-964 `_set_mapdict_storage_and_map` — install a complete
    /// replacement storage list and map (used by `delete`/`copy`).
    fn _set_mapdict_storage_and_map(&mut self, storage: Vec<PyObjectRef>, map: MapRef);
    /// mapdict.py:859-860 `MapdictDictSupport.getdict` → `_obj_getdict`
    /// (mapdict.py:869-882) — the instance's `__dict__`. Only the live instance
    /// carrier provides this; the transient `Object` (mapdict.py:978,
    /// copy/materialize result) lacks `MapdictDictSupport`, so its impl is
    /// unreachable. Used by `DevolvedDictTerminator`'s read/write/delete
    /// (mapdict.py:383-409), which only ever run on the live instance.
    fn getdict(&self) -> PyObjectRef;
}

/// `W_ObjectObject` (`pyre-object`) is pyre's `MapdictStorageMixin`
/// carrier (`mapdict.py:904-963`): PyPy mixes the mixin into the instance
/// class, here the instance implements the trait. `map` is the erased
/// `*const MapNode`; `storage` is a heap `Vec<PyObjectRef>` (null =
/// `None`, the `_mapdict_init_empty` empty state, mapdict.py:910).
/// Remember an instance that may now hold a young attribute value, mirroring
/// `dict_write_barrier` (dictmultiobject.rs:421). RPython's GC inserts the
/// barrier implicitly at `self.storage[index] = value` (mapdict.py:918-919);
/// pyre's `storage` is an off-GC `*mut Vec<PyObjectRef>`, so the store bypasses
/// the collector's remembered-set tracking and must call the barrier
/// explicitly. Without it a nursery value stored into an old-gen instance is
/// not forwarded during a minor collection: `object_object_custom_trace`
/// (`W_OBJECT_OBJECT_GC_TYPE_ID`) runs only for remembered-set objects in
/// `do_collect_nursery`, never blanket-scanned.
fn instance_write_barrier(obj: PyObjectRef) {
    pyre_object::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

impl MapdictObject for pyre_object::W_ObjectObject {
    fn _get_mapdict_map(&self) -> MapRef {
        // `jit.promote(self.map)` (mapdict.py:905-906).
        self.map as MapRef
    }
    fn _set_mapdict_map(&mut self, map: MapRef) {
        self.map = map as *const u8;
    }
    fn _mapdict_read_storage(&self, storageindex: usize) -> PyObjectRef {
        // mapdict.py:914-916. A read is always preceded by the
        // `_set_mapdict_increase_storage1` that made `storage` non-null; the
        // block is exact-size (capacity == map.storage_needed()), so
        // `storageindex` (from this instance's map) is always in range.
        unsafe {
            let base = pyre_object::object_array::items_block_items_base(self.storage);
            *base.add(storageindex)
        }
    }
    fn _mapdict_write_storage(&mut self, storageindex: usize, value: PyObjectRef) {
        // mapdict.py:918-919. The instance is the remembered-set root: the
        // collector reaches the (non-moving, stable) storage block only through
        // this instance's `object_object_custom_trace`, which walks the block's
        // boxed slots in place, so remembering the instance keeps a young value
        // stored into an old-gen instance's block forwarded on a minor GC.
        //
        // RPython's GC transform emits the barrier before the pointer store.
        // Publish both operands first because a foreign mutator's collection
        // can make the barrier wait; storing first would leave the new value
        // invisible until after that collection had swept it.
        let _roots = pyre_object::gc_roots::push_roots();
        let livevars = vec![self as *const Self as PyObjectRef, value];
        let self_slot = pyre_object::gc_roots::pin_roots(&livevars);
        let value_slot = self_slot + 1;
        let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
        instance_write_barrier(owner);
        let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
        let value = pyre_object::gc_roots::shadow_stack_get(value_slot);
        unsafe {
            let owner = &mut *(owner as *mut Self);
            let base = pyre_object::object_array::items_block_items_base(owner.storage);
            *base.add(storageindex) = value;
        }
    }
    fn _mapdict_storage_length(&self) -> usize {
        // mapdict.py:921-924 (= self.map.storage_needed()).
        unsafe { (*(self.map as MapRef)).storage_needed() }
    }
    fn _mapdict_pop_attribute(&mut self, map: MapRef) {
        // mapdict.py:926-939; structure mirrors the MockObj test impl.
        // Both arms can collect — the unboxed one through
        // `_mapdict_write_storage`, the other through the storage shrink — so
        // the receiver is published for the whole function and every write
        // goes through the reloaded address, as in
        // `_set_mapdict_increase_storage1` and `_set_mapdict_storage_and_map`.
        // `&mut self` is a bare address that a moving collection invalidates;
        // writing `map` through it would leave the live instance on its old
        // map, claiming a storage slot the shrunk block no longer has.
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self as *const Self as PyObjectRef]);
        let current_map = self.map as MapRef;
        let unboxed_slot: Option<(usize, usize)> = unsafe {
            match &(*current_map).as_plain().unboxed {
                Some(u) => match u.firstunwrapped {
                    true => None,
                    false => Some(((*current_map).as_plain().storageindex, u.listindex)),
                },
                None => None,
            }
        };
        match unboxed_slot {
            // mapdict.py:931-934: drop the last entry of the shared
            // longlong list (the slot itself stays).
            Some((storageindex, listindex)) => {
                let slot = self._mapdict_read_storage(storageindex);
                let new_list: Vec<i64> = unsafe { unboxed_items(slot)[..listindex].to_vec() };
                // `erase_unboxed` allocates the replacement bank, so reload the
                // receiver after it rather than before.
                let unboxed = erase_unboxed(&new_list);
                let owner = unsafe {
                    &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self)
                };
                owner._mapdict_write_storage(storageindex, unboxed);
            }
            // mapdict.py:935-938: truncate storage to the parent map's size.
            // The block itself keeps its capacity (see
            // `_set_mapdict_storage_and_map`); NULL the slots the parent map no
            // longer names so their values are released.
            None => {
                let storage_needed = unsafe { (*map).storage_needed() };
                unsafe {
                    let cap = pyre_object::object_array::items_block_capacity(self.storage);
                    if cap > storage_needed {
                        let base = pyre_object::object_array::items_block_items_base(self.storage);
                        for i in storage_needed..cap {
                            *base.add(i) = pyre_object::PY_NULL;
                        }
                    }
                }
            }
        }
        unsafe {
            let owner = &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self);
            owner.map = map as *const u8;
        }
    }
    fn _set_mapdict_increase_storage1(&mut self, map: MapRef, value: PyObjectRef) {
        // grow storage by one, append value (mapdict.py:942-959). The first
        // grow allocates the storage block (was `None`). The new attribute's
        // index comes from the map, not from the block's capacity: a delete
        // leaves the capacity above `storage_needed()` (see
        // `_set_mapdict_storage_and_map`), and that spare tail is where this
        // write lands, so no block is allocated until the map outgrows it.
        let _roots = pyre_object::gc_roots::push_roots();
        let livevars = vec![self as *const Self as PyObjectRef, value];
        let self_slot = pyre_object::gc_roots::pin_roots(&livevars);
        let value_slot = self_slot + 1;
        let needed = unsafe { (*map).storage_needed() };
        unsafe {
            let owner = &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self);
            let old = owner.storage;
            let old_cap = pyre_object::object_array::items_block_capacity(old);
            let block = if needed <= old_cap {
                old
            } else {
                let grown =
                    pyre_object::object_array::grow_instance_items_block(old, needed, old_cap);
                pyre_object::gc_roots::pin_root(grown as PyObjectRef);
                let block_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                pyre_object::gc_roots::shadow_stack_get(block_slot)
                    as *mut pyre_object::object_array::ItemsBlock
            };
            let base = pyre_object::object_array::items_block_items_base(block);
            *base.add(needed - 1) = pyre_object::gc_roots::shadow_stack_get(value_slot);
            let owner = &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self);
            owner.storage = block;
            owner.map = map as *const u8;
            instance_write_barrier(owner as *mut Self as PyObjectRef);
            if !std::ptr::eq(block, old) {
                pyre_object::object_array::dealloc_instance_items_block(old);
            }
        }
    }
    fn _set_mapdict_storage_and_map(&mut self, storage: Vec<PyObjectRef>, map: MapRef) {
        // mapdict.py:961-964. The incoming `Vec` comes from a lightweight
        // `Object` carrier (delete/copy transplant); convert it to a fresh
        // storage block, freeing any prior block.
        //
        // A delete arrives here with fewer values than the instance already
        // holds, but the block never shrinks. `_mapdict_read_storage` indexes
        // without a bounds check, and the JIT folds an attribute access to that
        // bare `storage[storageindex]` load with its map guard checked *before*
        // it — so with no process GIL a concurrent shrink could leave a
        // just-validated index past the new block's capacity. A monotone
        // capacity keeps every index the guard ever accepted in range; the
        // reader's worst case becomes a stale slot value, which is what an
        // unsynchronised attribute read means anyway. The spare tail is NULL,
        // so the dropped attributes' values are still released.
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self as *const Self as PyObjectRef]);
        unsafe {
            let owner = &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self);
            let old = owner.storage;
            let cap = storage
                .len()
                .max(pyre_object::object_array::items_block_capacity(old));
            let fresh = pyre_object::object_array::alloc_instance_items_block(&storage, cap);
            pyre_object::gc_roots::pin_root(fresh as PyObjectRef);
            let fresh_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let owner = &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut Self);
            owner.storage = pyre_object::gc_roots::shadow_stack_get(fresh_slot) as *mut _;
            owner.map = map as *const u8;
            instance_write_barrier(owner as *mut Self as PyObjectRef);
            pyre_object::object_array::dealloc_instance_items_block(old);
        }
    }
    fn getdict(&self) -> PyObjectRef {
        // mapdict.py:859-860 MapdictDictSupport.getdict → _obj_getdict
        // (mapdict.py:869-882). The instance header is at offset 0
        // (`#[repr(C)]`), so the carrier pointer is the instance PyObjectRef.
        _obj_getdict(self as *const Self as PyObjectRef)
    }
}

/// A live view of whichever translated layout owns an object's mapdict mixin.
/// The native user layouts append the two fields after their base payload, so
/// mapdict code must resolve the slots from the proven concrete layout instead
/// of casting every receiver to `W_ObjectObject`.
#[derive(Clone, Copy)]
struct MapdictCarrier {
    obj: PyObjectRef,
}

unsafe fn mapdict_carrier(obj: PyObjectRef) -> MapdictCarrier {
    debug_assert!(unsafe { has_mapdict_layout(obj) });
    MapdictCarrier { obj }
}

impl MapdictCarrier {
    unsafe fn slots(
        &self,
    ) -> (
        *mut *const u8,
        *mut *mut pyre_object::object_array::ItemsBlock,
    ) {
        let obj = self.obj;
        if unsafe { pyre_object::is_int(obj) } {
            let user = obj as *mut pyre_object::intobject::W_IntObjectUser;
            return unsafe {
                (
                    std::ptr::addr_of_mut!((*user).map),
                    std::ptr::addr_of_mut!((*user).storage),
                )
            };
        }
        if unsafe { pyre_object::is_str(obj) } {
            let user = obj as *mut pyre_object::unicodeobject::W_UnicodeObjectUser;
            return unsafe {
                (
                    std::ptr::addr_of_mut!((*user).map),
                    std::ptr::addr_of_mut!((*user).storage),
                )
            };
        }
        // A specialised tuple has no `W_TupleObjectUser` variant, and the
        // admitting predicate excludes it for that reason.
        if unsafe { pyre_object::is_tuple(obj) && is_generated_user_layout_family(obj) } {
            let user = obj as *mut pyre_object::tupleobject::W_TupleObjectUser;
            return unsafe {
                (
                    std::ptr::addr_of_mut!((*user).map),
                    std::ptr::addr_of_mut!((*user).storage),
                )
            };
        }
        let ordinary = obj as *mut pyre_object::W_ObjectObject;
        unsafe {
            (
                std::ptr::addr_of_mut!((*ordinary).map),
                std::ptr::addr_of_mut!((*ordinary).storage),
            )
        }
    }

    unsafe fn storage(&self) -> *mut pyre_object::object_array::ItemsBlock {
        unsafe { *self.slots().1 }
    }

    unsafe fn set_storage(&mut self, storage: *mut pyre_object::object_array::ItemsBlock) {
        unsafe { *self.slots().1 = storage };
    }
}

impl MapdictObject for MapdictCarrier {
    fn _get_mapdict_map(&self) -> MapRef {
        unsafe { *self.slots().0 as MapRef }
    }

    fn _set_mapdict_map(&mut self, map: MapRef) {
        unsafe { *self.slots().0 = map as *const u8 };
    }

    fn _mapdict_read_storage(&self, storageindex: usize) -> PyObjectRef {
        unsafe {
            let base = pyre_object::object_array::items_block_items_base(self.storage());
            *base.add(storageindex)
        }
    }

    fn _mapdict_write_storage(&mut self, storageindex: usize, value: PyObjectRef) {
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self.obj, value]);
        let value_slot = self_slot + 1;
        instance_write_barrier(pyre_object::gc_roots::shadow_stack_get(self_slot));
        unsafe {
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            let carrier = mapdict_carrier(owner);
            let base = pyre_object::object_array::items_block_items_base(carrier.storage());
            *base.add(storageindex) = pyre_object::gc_roots::shadow_stack_get(value_slot);
            self.obj = owner;
        }
    }

    fn _mapdict_storage_length(&self) -> usize {
        unsafe { (*self._get_mapdict_map()).storage_needed() }
    }

    fn _mapdict_pop_attribute(&mut self, map: MapRef) {
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self.obj]);
        let current_map = self._get_mapdict_map();
        let unboxed_slot: Option<(usize, usize)> = unsafe {
            match &(*current_map).as_plain().unboxed {
                Some(u) if !u.firstunwrapped => {
                    Some(((*current_map).as_plain().storageindex, u.listindex))
                }
                _ => None,
            }
        };
        match unboxed_slot {
            Some((storageindex, listindex)) => {
                let slot = self._mapdict_read_storage(storageindex);
                let new_list = unsafe { unboxed_items(slot)[..listindex].to_vec() };
                let unboxed = erase_unboxed(&new_list);
                let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
                let mut carrier = unsafe { mapdict_carrier(owner) };
                carrier._mapdict_write_storage(storageindex, unboxed);
            }
            None => unsafe {
                let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
                let carrier = mapdict_carrier(owner);
                let storage = carrier.storage();
                let storage_needed = (*map).storage_needed();
                let cap = pyre_object::object_array::items_block_capacity(storage);
                if cap > storage_needed {
                    let base = pyre_object::object_array::items_block_items_base(storage);
                    for i in storage_needed..cap {
                        *base.add(i) = pyre_object::PY_NULL;
                    }
                }
            },
        }
        self.obj = pyre_object::gc_roots::shadow_stack_get(self_slot);
        self._set_mapdict_map(map);
    }

    fn _set_mapdict_increase_storage1(&mut self, map: MapRef, value: PyObjectRef) {
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self.obj, value]);
        let value_slot = self_slot + 1;
        let needed = unsafe { (*map).storage_needed() };
        unsafe {
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            let mut carrier = mapdict_carrier(owner);
            let old = carrier.storage();
            let old_cap = pyre_object::object_array::items_block_capacity(old);
            let block = if needed <= old_cap {
                old
            } else {
                let grown =
                    pyre_object::object_array::grow_instance_items_block(old, needed, old_cap);
                pyre_object::gc_roots::pin_root(grown as PyObjectRef);
                pyre_object::gc_roots::shadow_stack_get(
                    pyre_object::gc_roots::shadow_stack_len() - 1,
                ) as *mut pyre_object::object_array::ItemsBlock
            };
            let base = pyre_object::object_array::items_block_items_base(block);
            *base.add(needed - 1) = pyre_object::gc_roots::shadow_stack_get(value_slot);
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            carrier = mapdict_carrier(owner);
            carrier.set_storage(block);
            carrier._set_mapdict_map(map);
            instance_write_barrier(owner);
            if !std::ptr::eq(block, old) {
                pyre_object::object_array::dealloc_instance_items_block(old);
            }
            self.obj = owner;
        }
    }

    fn _set_mapdict_storage_and_map(&mut self, storage: Vec<PyObjectRef>, map: MapRef) {
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self.obj]);
        unsafe {
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            let mut carrier = mapdict_carrier(owner);
            let old = carrier.storage();
            let cap = storage
                .len()
                .max(pyre_object::object_array::items_block_capacity(old));
            let fresh = pyre_object::object_array::alloc_instance_items_block(&storage, cap);
            pyre_object::gc_roots::pin_root(fresh as PyObjectRef);
            let fresh_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let owner = pyre_object::gc_roots::shadow_stack_get(self_slot);
            carrier = mapdict_carrier(owner);
            carrier.set_storage(pyre_object::gc_roots::shadow_stack_get(fresh_slot) as *mut _);
            carrier._set_mapdict_map(map);
            instance_write_barrier(owner);
            pyre_object::object_array::dealloc_instance_items_block(old);
            self.obj = owner;
        }
    }

    fn getdict(&self) -> PyObjectRef {
        _obj_getdict(self.obj)
    }
}

/// mapdict.py:978-985 `Object` — the transient `MapdictStorageMixin` carrier
/// for the result of `delete`/`copy` (its `storage`/`map` are transplanted
/// into the real instance by `_set_mapdict_storage_and_map`). pyre keeps that
/// lightweight owned-`Vec` role here, while a real `W_ObjectObject` provides
/// the fake-instance-dict-backing role.
pub(crate) struct Object {
    map: MapRef,
    storage: Vec<PyObjectRef>,
}

impl Object {
    /// mapdict.py:910-912 `_mapdict_init_empty` — fresh carrier on `map` with
    /// an empty storage list.
    fn new_empty(map: MapRef) -> Object {
        Object {
            map,
            storage: Vec::new(),
        }
    }
}

impl MapdictObject for Object {
    fn _get_mapdict_map(&self) -> MapRef {
        self.map
    }
    fn _set_mapdict_map(&mut self, map: MapRef) {
        self.map = map;
    }
    fn _mapdict_read_storage(&self, storageindex: usize) -> PyObjectRef {
        self.storage[storageindex]
    }
    fn _mapdict_write_storage(&mut self, storageindex: usize, value: PyObjectRef) {
        self.storage[storageindex] = value;
    }
    fn _mapdict_storage_length(&self) -> usize {
        unsafe { (*self.map).storage_needed() }
    }
    fn _mapdict_pop_attribute(&mut self, map: MapRef) {
        // mapdict.py:926-939. `current_map` is the PlainAttribute being popped;
        // `map` is its parent. The unboxed-non-firstunwrapped slot to shrink is
        // a `match` on `firstunwrapped` (not `!firstunwrapped`) so the source
        // walker can lower it.
        let current_map = self.map;
        let unboxed_slot: Option<(usize, usize)> = unsafe {
            match &(*current_map).as_plain().unboxed {
                Some(u) => match u.firstunwrapped {
                    true => None,
                    false => Some(((*current_map).as_plain().storageindex, u.listindex)),
                },
                None => None,
            }
        };
        match unboxed_slot {
            Some((storageindex, listindex)) => {
                let slot = self._mapdict_read_storage(storageindex);
                let new_list: Vec<i64> = unsafe { unboxed_items(slot)[..listindex].to_vec() };
                self._mapdict_write_storage(storageindex, erase_unboxed(&new_list));
            }
            None => {
                let storage_needed = unsafe { (*map).storage_needed() };
                self.storage.truncate(storage_needed);
            }
        }
        self.map = map;
    }
    fn _set_mapdict_increase_storage1(&mut self, map: MapRef, value: PyObjectRef) {
        // grow storage by one, append value (mapdict.py:942-959)
        self.storage.push(value);
        self.map = map;
    }
    fn _set_mapdict_storage_and_map(&mut self, storage: Vec<PyObjectRef>, map: MapRef) {
        self.storage = storage;
        self.map = map;
    }
    fn getdict(&self) -> PyObjectRef {
        // mapdict.py:979-985 gives `Object` both transient-copy and fake-dict
        // roles. pyre gives the fake-dict role to `W_ObjectObject`, whose
        // getdict is implemented, leaving this carrier transient only. The
        // terminator_read Devolved arm, write_terminator's Devolved and LIMIT
        // arms, and node_delete's Devolved arm must therefore never reach it.
        unimplemented!("transient Object carrier has no __dict__")
    }
}

/// rerased `erase_unboxed` / `unerase_unboxed` (mapdict.py:38-41) — the
/// unboxed longlong list lives in an otherwise-`PyObjectRef` storage slot.
///
/// The list is a varsize leaf GcArray (`GC_INT_ARRAY_GC_TYPE_ID`), so the slot
/// holds an ordinary GC reference like every other storage slot: the collector
/// marks it through the instance and reclaims it when the instance dies, and no
/// walker has to be told to skip the slot. Every mutation allocates a fresh
/// exact-size block, so the block's capacity header is the list's length.
///
/// The block is reachable from nothing when this returns; the caller must root
/// it until the storage store lands (see `alloc_typed_items_block`).
fn erase_unboxed(values: &[i64]) -> PyObjectRef {
    debug_assert!(
        !values.is_empty(),
        "an unboxed list holds at least the attribute that created it"
    );
    unsafe {
        let block = pyre_object::object_array::alloc_typed_items_block(
            values.len(),
            pyre_object::object_array::GC_INT_ARRAY_GC_TYPE_ID,
        );
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            pyre_object::object_array::typed_items_block_items_base(block) as *mut i64,
            values.len(),
        );
        block as PyObjectRef
    }
}

/// # Safety
/// `slot` must have been produced by `erase_unboxed` and still be live.
unsafe fn unerase_unboxed(slot: PyObjectRef) -> *mut pyre_object::object_array::TypedItemsBlock {
    slot as *mut pyre_object::object_array::TypedItemsBlock
}

/// The longlong list an unboxed slot holds. Its length is the block's capacity
/// header, because the block is always allocated exact-size.
///
/// # Safety
/// `slot` must have been produced by `erase_unboxed` and still be live.
unsafe fn unboxed_items<'a>(slot: PyObjectRef) -> &'a [i64] {
    unsafe {
        let block = unerase_unboxed(slot);
        std::slice::from_raw_parts(
            pyre_object::object_array::typed_items_block_items_base(block) as *const i64,
            pyre_object::object_array::typed_items_block_capacity(block),
        )
    }
}

/// Mutable twin of [`unboxed_items`], for the same-type in-place update.
///
/// # Safety
/// `slot` must have been produced by `erase_unboxed` and still be live.
unsafe fn unboxed_items_mut<'a>(slot: PyObjectRef) -> &'a mut [i64] {
    unsafe {
        let block = unerase_unboxed(slot);
        std::slice::from_raw_parts_mut(
            pyre_object::object_array::typed_items_block_items_base(block) as *mut i64,
            pyre_object::object_array::typed_items_block_capacity(block),
        )
    }
}

/// mapdict.py:571-577 `UnboxedPlainAttribute._unbox`.
///
/// # Safety
/// `w_value` must be of the type named by `typ`.
unsafe fn unbox_value(typ: UnboxType, w_value: PyObjectRef) -> i64 {
    match typ {
        UnboxType::Int => unsafe { pyre_object::w_int_get_value(w_value) },
        // float2longlong (mapdict.py:577).
        UnboxType::Float => unsafe { pyre_object::w_float_get_value(w_value) }.to_bits() as i64,
    }
}

/// mapdict.py:579-584 `UnboxedPlainAttribute._box`.
fn box_value(typ: UnboxType, val: i64) -> PyObjectRef {
    match typ {
        UnboxType::Int => pyre_object::w_int_new(val),
        // longlong2float (mapdict.py:584).
        UnboxType::Float => pyre_object::w_float_new(f64::from_bits(val as u64)),
    }
}

/// `type(w_value) is space.IntObjectCls` (mapdict.py:198,574,615).
///
/// Pyre gives an int subclass the same native storage layout as W_IntObject
/// and carries its app-level identity in `w_class`. Testing only the layout
/// (`is_int`) would unbox IntEnum/user-int instances and reconstruct them as
/// builtin ints, losing `w_class`. Compare the exact app-level type just as
/// PyPy compares the exact implementation class.
///
/// # Safety
/// `w_value` must point to a live object.
unsafe fn is_unboxable_int(w_value: PyObjectRef) -> bool {
    if unsafe { pyre_object::is_bool(w_value) } || !unsafe { pyre_object::is_int(w_value) } {
        return false;
    }
    let exact = crate::typedef::gettypeobject(&pyre_object::INT_TYPE);
    if exact.is_null() {
        // Interpreter-only unit tests may exercise mapdict before the app-level
        // type registry is initialized. A native int with no class stamp is
        // the bootstrap representation of that same exact type.
        if pyre_object::tagged_int::CAN_BE_TAGGED
            && unsafe { pyre_object::tagged_int::is_tagged_int(w_value) }
        {
            return true;
        }
        return unsafe { (*w_value).w_class.is_null() };
    }
    let Some(actual) = crate::typedef::r#type(w_value) else {
        return false;
    };
    std::ptr::eq(actual.as_ptr(), exact)
}

/// Float half of `_pick_unbox_type`: PyPy uses
/// `type(w_value) is space.FloatObjectCls`, so a float subclass must retain
/// its boxed object and `w_class` too.
unsafe fn is_unboxable_float(w_value: PyObjectRef) -> bool {
    if !unsafe { pyre_object::is_float(w_value) } {
        return false;
    }
    let exact = crate::typedef::gettypeobject(&pyre_object::FLOAT_TYPE);
    if exact.is_null() {
        return unsafe { (*w_value).w_class.is_null() };
    }
    let Some(actual) = crate::typedef::r#type(w_value) else {
        return false;
    };
    std::ptr::eq(actual.as_ptr(), exact)
}

/// mapdict.py:586-590 `UnboxedPlainAttribute._convert_to_boxed` — rebuild the
/// carrier with boxed storage and transplant it onto `obj`, returning the new
/// (boxed) map. `node_copy` re-adds every attribute through `add_attr`, which
/// picks no unbox type because `allow_unboxing` is already frozen off on the
/// terminator (the caller sets it before converting), so the rebuilt chain is
/// all boxed and its storage is a clean `Vec<PyObjectRef>`.
///
/// # Safety
/// `obj` must implement the mapdict storage protocol.
unsafe fn convert_to_boxed<O: MapdictObject>(obj: &mut O) -> MapRef {
    let map = obj._get_mapdict_map();
    let new_obj = unsafe { node_copy(map, obj) };
    let new_map = new_obj.map;
    obj._set_mapdict_storage_and_map(new_obj.storage, new_map);
    new_map
}

/// mapdict.py:620-627 `UnboxedPlainAttribute._direct_write` type-change tail —
/// convert `obj` to boxed storage, then write `(name, attrkind) = w_value`
/// through the now-boxed map (no `UnboxedPlainAttribute` remains because
/// `allow_unboxing` was just frozen off).
///
/// # Safety
/// `obj` must implement the mapdict storage protocol.
unsafe fn convert_to_boxed_and_write<O: MapdictObject>(
    obj: &mut O,
    name: &Wtf8,
    attrkind: u16,
    w_value: PyObjectRef,
) {
    let map = unsafe { convert_to_boxed(obj) };
    let _ = unsafe { node_write(map, obj, name, attrkind, w_value) };
}

/// `type(w_value) is self.typ` (mapdict.py:574,615).
///
/// # Safety
/// `w_value` must point to a live object.
unsafe fn value_has_unbox_type(typ: UnboxType, w_value: PyObjectRef) -> bool {
    match typ {
        UnboxType::Int => unsafe { is_unboxable_int(w_value) },
        UnboxType::Float => unsafe { is_unboxable_float(w_value) },
    }
}

/// mapdict.py:437-444 `PlainAttribute._direct_read` / `_prim_direct_read` /
/// `_pure_direct_read` have identical bodies; the `@jit.elidable`
/// `_pure_direct_read` variant is applied when the read is JIT-wired.
/// `unerase_item` is identity. For an `UnboxedPlainAttribute`,
/// mapdict.py:592-612 gives `_direct_read` the allow-unboxing conversion tail;
/// pyre places that tail in `maybe_migrate_to_boxed`.
///
/// # Safety
/// `attr` must point to a live `PlainAttribute` map node.
pub unsafe fn plain_direct_read<O: MapdictObject>(attr: MapRef, obj: &O) -> PyObjectRef {
    let p = unsafe { (*attr).as_plain() };
    match &p.unboxed {
        // mapdict.py:443 — boxed value straight out of the slot.
        None => obj._mapdict_read_storage(p.storageindex),
        Some(u) => {
            // _prim_direct_read (mapdict.py:600-601): box the longlong at
            // (storageindex, listindex).
            let slot = obj._mapdict_read_storage(p.storageindex);
            let raw = unsafe { unboxed_items(slot)[u.listindex] };
            let w_res = box_value(u.typ, raw);
            // This is `_prim_direct_read` (mapdict.py:600-601), the non-converting
            // read shared by node_read / copy_attr / reorder_and_add /
            // materialize. `_direct_read`'s lazy migrate-to-boxed side effect
            // (mapdict.py:592-598, when the class has frozen unboxing) lives at
            // the getattr boundary in `maybe_migrate_to_boxed`, which `&mut`
            // access permits there; here `&obj` stays pure.
            w_res
        }
    }
}

/// mapdict.py:446-447 `PlainAttribute._direct_write`. `erase_item` is identity.
/// For an `UnboxedPlainAttribute` this is mapdict.py:614-628.
///
/// # Safety
/// `attr` must point to a live `PlainAttribute` map node.
pub unsafe fn plain_direct_write<O: MapdictObject>(
    attr: MapRef,
    obj: &mut O,
    w_value: PyObjectRef,
) {
    let p = unsafe { (*attr).as_plain() };
    match &p.unboxed {
        None => {
            let storageindex = p.storageindex;
            obj._mapdict_write_storage(storageindex, w_value);
        }
        Some(u) => {
            if unsafe { value_has_unbox_type(u.typ, w_value) } {
                // mapdict.py:615-619 — same type: update the longlong in place.
                let val = unsafe { unbox_value(u.typ, w_value) };
                let slot = obj._mapdict_read_storage(p.storageindex);
                unsafe {
                    unboxed_items_mut(slot)[u.listindex] = val;
                }
            } else {
                // mapdict.py:620-627 — type change. Freeze unboxing for the
                // terminator, then convert the instance to boxed storage and
                // rewrite `(name, attrkind)` through the now-boxed map.
                unsafe { (*p.terminator).as_terminator() }.set_allow_unboxing(false);
                let name = p.name.clone();
                let attrkind = p.attrkind;
                unsafe { convert_to_boxed_and_write(obj, &name, attrkind, w_value) };
            }
        }
    }
}

/// mapdict.py:312-313 `Terminator._read_terminator` and its
/// `DevolvedDictTerminator` override (mapdict.py:387-391). Returns the value or
/// `None` when the attribute is absent.
///
/// # Safety
/// `term` must point to a live Terminator map node.
unsafe fn terminator_read<O: MapdictObject>(
    term: MapRef,
    obj: &O,
    name: &Wtf8,
    attrkind: u16,
) -> Option<PyObjectRef> {
    unsafe { terminator_read_checked(term, obj, name, attrkind) }.unwrap_or(None)
}

/// Fallible [`terminator_read`].  The devolved arm is the only one that can
/// raise, and `space.finditem_str` propagates there upstream; the swallowing
/// spelling above exists for the callers that have no error channel and is
/// written in terms of this one.  Its `unwrap_or` also consumes the pending
/// error slot, so a dropped error cannot surface on a later operation.
///
/// # Safety
/// `term` must point to a live Terminator map node.
unsafe fn terminator_read_checked<O: MapdictObject>(
    term: MapRef,
    obj: &O,
    name: &Wtf8,
    attrkind: u16,
) -> Result<Option<PyObjectRef>, PyError> {
    let t = unsafe { (*term).as_terminator() };
    match t.kind {
        TerminatorKind::Devolved if attrkind == DICT => {
            // mapdict.py:383-388: the devolved terminator reads DICT attributes
            // from the materialised instance dict (`space.finditem_str(
            // obj.getdict(space), name)`).  `finditem_str` is fallible: the
            // probe compares against whatever the bucket holds, so a stored
            // non-string key whose hash collides can reach a user `__eq__`
            // that raises, and that must not read back as a miss.
            let w_dict = obj.getdict();
            let backing = crate::type_methods::resolve_dict_backing(w_dict);
            unsafe { pyre_object::dictmultiobject::w_dict_getitem_wtf8_checked(backing, name) }
                .map_err(|_| {
                    crate::baseobjspace::take_pending_dict_key_error(pyre_object::w_str_from_wtf8(
                        name.to_wtf8_buf(),
                    ))
                })
        }
        // Terminator / DictTerminator / NoDictTerminator read nothing.
        _ => Ok(None),
    }
}

/// mapdict.py:55-66 `AbstractAttribute.read`.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
pub unsafe fn node_read<O: MapdictObject>(
    self_node: MapRef,
    obj: &O,
    name: &Wtf8,
    attrkind: u16,
) -> Option<PyObjectRef> {
    unsafe { node_read_checked(self_node, obj, name, attrkind) }.unwrap_or(None)
}

/// Fallible [`node_read`], for the callers that can propagate the raising
/// `__eq__` a devolved terminator's dict probe may reach.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
pub unsafe fn node_read_checked<O: MapdictObject>(
    self_node: MapRef,
    obj: &O,
    name: &Wtf8,
    attrkind: u16,
) -> Result<Option<PyObjectRef>, PyError> {
    match unsafe { find_map_attr(self_node, name, attrkind) } {
        // The `jit.isconstant(attr) and jit.isconstant(obj) and not
        // attr.ever_mutated` guard selects `_pure_direct_read`
        // (mapdict.py:60-65). The PlainAttribute variants have the same body;
        // UnboxedPlainAttribute._direct_read's conversion tail lives in
        // `maybe_migrate_to_boxed`.
        Some(attr) => Ok(Some(unsafe { plain_direct_read(attr, obj) })),
        None => unsafe { terminator_read_checked((*self_node).terminator(), obj, name, attrkind) },
    }
}

/// mapdict.py:592-598 `UnboxedPlainAttribute._direct_read` migration tail.
/// `node_read` is the shared pure value read (`_prim_direct_read` based, also
/// used by `copy_attr`/`node_materialize_dict`/`node_set_terminator`); the
/// getattr `read` path (`getdictvalue`, mapdict.py:846-847 → 55-66) additionally
/// runs `_direct_read`, which lazily migrates `obj` to boxed storage once the
/// read attribute is unboxed and its class has frozen unboxing
/// (`terminator.allow_unboxing` False). A boxed attribute's `_direct_read` is
/// `_prim_direct_read` (no migration), so this is a no-op for them. The
/// `find_map_attr` re-lookup hits the per-VM transition cache.
///
/// # Safety
/// `self_node`/its chain must point to live map nodes; `obj` to a live carrier.
unsafe fn maybe_migrate_to_boxed<O: MapdictObject>(
    self_node: MapRef,
    obj: &mut O,
    name: &Wtf8,
    attrkind: u16,
) {
    let attr = match unsafe { find_map_attr(self_node, name, attrkind) } {
        Some(a) => a,
        None => return,
    };
    let p = unsafe { (*attr).as_plain() };
    let migrate = match &p.unboxed {
        Some(_) => match unsafe { (*p.terminator).as_terminator() }
            .allow_unboxing
            .get()
        {
            true => false,
            false => true,
        },
        None => false,
    };
    if migrate {
        unsafe { convert_to_boxed(obj) };
    }
}

// ── copy / delete path (mapdict.py:326-330, 433-435, 461-475) ─────────

/// mapdict.py:433-435 `PlainAttribute._copy_attr` — read this attribute from
/// `obj` and re-add it to the freshly built `new_obj`. The read is
/// `_prim_direct_read` (mapdict.py:440/600-601, the non-converting raw read);
/// pyre's `plain_direct_read` performs exactly that — boxing the longlong slot
/// when the attribute is unboxed — and defers only the read-path lazy re-box
/// (`_direct_read`'s allow-unboxing/convert distinction, see
/// `plain_direct_read`), which is invisible to the copied value.
///
/// # Safety
/// `attr_node` must point to a live `PlainAttribute`; `obj` to a live carrier.
unsafe fn copy_attr<O: MapdictObject>(attr_node: MapRef, obj: &O, new_obj: &mut Object) {
    let w_value = unsafe { plain_direct_read(attr_node, obj) };
    let p = unsafe { (*attr_node).as_plain() };
    let map = new_obj._get_mapdict_map();
    unsafe { add_attr(map, new_obj, &p.name, p.attrkind, w_value) };
}

/// mapdict.py:326-330 `Terminator.copy` / 472-475 `PlainAttribute.copy` — build
/// a fresh `Object` carrier holding the same attributes as `obj` (in canonical
/// order: the back-chain is copied bottom-up, then the node re-adds itself).
///
/// # Safety
/// `self_node` and its `back` chain must point to live map nodes.
unsafe fn node_copy<O: MapdictObject>(self_node: MapRef, obj: &O) -> Object {
    if unsafe { (*self_node).is_plain() } {
        let back = unsafe { (*self_node).as_plain() }.back;
        let mut new_obj = unsafe { node_copy(back, obj) };
        unsafe { copy_attr(self_node, obj, &mut new_obj) };
        new_obj
    } else {
        // Terminator.copy (mapdict.py:326-330): empty carrier on this terminator.
        Object::new_empty(self_node)
    }
}

/// mapdict.py:338-342 `Terminator.set_terminator`, 483-486
/// `PlainAttribute.set_terminator`, 414-418 `DevolvedDictTerminator.set_terminator`
/// — rebuild a fresh `Object` carrier holding `obj`'s attributes (canonical
/// order, like `node_copy`) but rooted at `new_terminator`. A devolved root
/// re-roots onto `new_terminator`'s paired devolved terminator so the instance
/// stays devolved.
///
/// # Safety
/// `self_node`/its `back` chain and `new_terminator` must point to live map nodes.
unsafe fn node_set_terminator<O: MapdictObject>(
    self_node: MapRef,
    obj: &O,
    new_terminator: MapRef,
) -> Object {
    if unsafe { (*self_node).is_plain() } {
        // mapdict.py:483-486 — recurse into `back` with the new terminator, then
        // re-add this attribute.
        let back = unsafe { (*self_node).as_plain() }.back;
        let mut new_obj = unsafe { node_set_terminator(back, obj, new_terminator) };
        unsafe { copy_attr(self_node, obj, &mut new_obj) };
        new_obj
    } else {
        // mapdict.py:338-342 — empty carrier on `new_terminator`; the devolved
        // override (mapdict.py:414-418) re-targets a devolved root onto the new
        // terminator's devolved pair.
        let term = match unsafe { (*self_node).as_terminator() }.kind {
            TerminatorKind::Devolved => {
                let target = unsafe { (*new_terminator).as_terminator() };
                match target.kind {
                    TerminatorKind::Devolved => new_terminator,
                    _ => target.devolved_dict_terminator.get(),
                }
            }
            _ => new_terminator,
        };
        Object::new_empty(term)
    }
}

/// mapdict.py:77-78 `AbstractAttribute.delete` (Terminator/DictTerminator,
/// returns `None`) and 461-470 `PlainAttribute.delete`. Returns the rebuilt
/// carrier with `(name, attrkind)` removed, or `None` if the attribute is
/// absent.
///
/// # Safety
/// `self_node` and its `back` chain must point to live map nodes.
unsafe fn node_delete<O: MapdictObject>(
    self_node: MapRef,
    obj: &O,
    name: &Wtf8,
    attrkind: u16,
) -> Option<Object> {
    if unsafe { (*self_node).is_plain() } {
        let p = unsafe { (*self_node).as_plain() };
        if attrkind == p.attrkind && &*p.name == name {
            // mapdict.py:462-466 — attribute found; drop it by rebuilding from
            // `back` (which excludes this node).
            p.set_ever_mutated(true);
            return Some(unsafe { node_copy(p.back, obj) });
        }
        // mapdict.py:467-470 — recurse, then re-add this surviving attribute.
        let back = p.back;
        match unsafe { node_delete(back, obj, name, attrkind) } {
            Some(mut new_obj) => {
                unsafe { copy_attr(self_node, obj, &mut new_obj) };
                Some(new_obj)
            }
            None => None,
        }
    } else {
        // mapdict.py:77-78 Terminator.delete (DictTerminator/NoDictTerminator
        // inherit) returns None.
        let kind = unsafe { (*self_node).as_terminator() }.kind;
        match kind {
            // mapdict.py:398-409 DevolvedDictTerminator.delete: drop the DICT
            // attribute from the materialised instance dict (a miss is tolerated
            // — mapdict.py:403-407 swallows KeyError), then return an empty
            // carrier on this terminator (`Terminator.copy(self, obj)`).
            TerminatorKind::Devolved if attrkind == DICT => {
                let w_dict = obj.getdict();
                let backing = crate::type_methods::resolve_dict_backing(w_dict);
                let w_key = pyre_object::w_str_from_wtf8(name.to_owned());
                unsafe { pyre_object::w_dict_delitem(backing, w_key) };
                Some(unsafe { node_copy(self_node, obj) })
            }
            _ => None,
        }
    }
}

/// mapdict.py:344-345 `Terminator.remove_dict_entries` (= `self.copy(obj)`) and
/// :511-515 `PlainAttribute.remove_dict_entries` — rebuild a fresh carrier that
/// keeps every non-`DICT` attribute and drops the `DICT` ones. Used by
/// `MapDictStrategy.clear` (mapdict.py:1222-1225). Reuses the `node_copy` /
/// `copy_attr` machinery already built for `delete`.
///
/// # Safety
/// `self_node` and its `back` chain must point to live map nodes.
unsafe fn node_remove_dict_entries<O: MapdictObject>(self_node: MapRef, obj: &O) -> Object {
    if unsafe { (*self_node).is_plain() } {
        let p = unsafe { (*self_node).as_plain() };
        let back = p.back;
        // mapdict.py:512 — recurse into `back` first.
        let mut new_obj = unsafe { node_remove_dict_entries(back, obj) };
        // mapdict.py:513-514 — re-add this attribute unless it is a DICT entry.
        if p.attrkind != DICT {
            unsafe { copy_attr(self_node, obj, &mut new_obj) };
        }
        new_obj
    } else {
        // mapdict.py:344-345 Terminator.remove_dict_entries = self.copy(obj).
        unsafe { node_copy(self_node, obj) }
    }
}

/// mapdict.py:362-366 `DictTerminator.materialize_r_dict`/`materialize_str_dict`
/// + 493-509 `PlainAttribute.materialize_r_dict`/`materialize_str_dict`. Drain
/// the DICT attributes into `w_dict` (already switched to its real strategy) and
/// rebuild a fresh carrier keeping only the non-DICT attributes, rooted at the
/// paired `DevolvedDictTerminator`. The walk recurses into `back` first so DICT
/// entries land in insertion (oldest-first) order.
///
/// pyre folds `materialize_r_dict` and `materialize_str_dict` into one helper:
/// PyPy's two methods differ only in the dict they fill (an `r_dict` keyed by
/// `space.eq_w`/`hash_w` vs a `str_dict` keyed by unicode), and both insert via
/// `space.newtext(name)`; here both targets are a `W_DictObject` whose strategy
/// (Object or Unicode) is already installed. Attribute names are interned at
/// the object-space boundary, so materialisation uses `box_str_constant(name)`
/// before the single `w_dict_store` shared by both strategies.
///
/// # Safety
/// `self_node` and its `back` chain must point to live map nodes; `w_dict` must
/// be a live `W_DictObject` on its post-switch strategy.
unsafe fn node_materialize_dict<O: MapdictObject>(
    self_node: MapRef,
    obj: &O,
    w_dict: PyObjectRef,
) -> Object {
    if unsafe { (*self_node).is_plain() } {
        let p = unsafe { (*self_node).as_plain() };
        // mapdict.py:494/503 — recurse into `back` first.
        let mut new_obj = unsafe { node_materialize_dict(p.back, obj, w_dict) };
        if p.attrkind == DICT {
            // mapdict.py:495-497/504-506 — move the DICT attribute into the
            // materialised dict (`dict_w[space.newtext(name)] =
            // self._prim_direct_read(obj)`). `plain_direct_read` performs that
            // prim read, boxing the slot when the attribute is unboxed.
            let w_value = unsafe { plain_direct_read(self_node, obj) };
            let w_attr = pyre_object::unicodeobject::box_str_constant(&p.name);
            unsafe { pyre_object::w_dict_store(w_dict, w_attr, w_value) };
        } else {
            // mapdict.py:499/508 — keep the non-DICT attribute on the carrier.
            unsafe { copy_attr(self_node, obj, &mut new_obj) };
        }
        new_obj
    } else {
        // mapdict.py:362-372 DictTerminator.materialize_* → `_make_devolved`: an
        // empty carrier on the paired DevolvedDictTerminator.
        let t = unsafe { (*self_node).as_terminator() };
        match t.kind {
            TerminatorKind::Dict => {
                let devolved = t.devolved_dict_terminator.get();
                Object::new_empty(devolved)
            }
            // mapdict.py:259-263 — the abstract base raises; materialise only
            // ever runs on a not-yet-devolved DictTerminator-rooted instance
            // dict (NoDict has no __dict__; Devolved is already materialised).
            _ => unimplemented!(
                "materialize on non-DictTerminator (mapdict.py:259-263 abstract base)"
            ),
        }
    }
}

/// mapdict.py:1305-1308 `materialize_r_dict` / 1310-1313 `materialize_str_dict`
/// (module-level) — run the chain over `obj`'s map to fill `w_dict`, then
/// transplant the rebuilt (devolved) storage+map back onto `obj`. The backing
/// instance is always a `W_ObjectObject` (the only `MapdictDictSupport`
/// carrier whose `__dict__` adopts `MapDictStrategy`).
///
/// # Safety
/// `obj` must be a live `W_ObjectObject`; `w_dict` a live `W_DictObject` on
/// its post-switch strategy.
unsafe fn materialize_dict(obj: PyObjectRef, w_dict: PyObjectRef) {
    // Filling `w_dict` allocates — boxed attribute names, the dict stores, and
    // the rebuilt carrier's own transitions — so both operands are published
    // and the transplant writes through the instance's post-collection address.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::pin_roots(&[obj, w_dict]);
    let dict_slot = obj_slot + 1;
    let new_obj = unsafe {
        let inst = mapdict_carrier(pyre_object::gc_roots::shadow_stack_get(obj_slot));
        let map = inst._get_mapdict_map();
        node_materialize_dict(
            map,
            &inst,
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
        )
    };
    let mut inst = unsafe { mapdict_carrier(pyre_object::gc_roots::shadow_stack_get(obj_slot)) };
    inst._set_mapdict_storage_and_map(new_obj.storage, new_obj.map);
}

// ── write path (mapdict.py:68-258, 312-321, 668-691) ──────────────────

/// mapdict.py:668-691 `CachedAttributeHolder` — caches the child map produced
/// by adding `(name, attrkind)` to a parent map, so transitions are shared.
/// Interned/immortal like the map nodes it holds.
pub struct CachedAttributeHolder {
    /// mapdict.py:670 `order` (= number of prior children of `back`).
    pub order: usize,
    /// mapdict.py:669 `_immutable_fields_` declares `attr?`; mapdict.py:675
    /// `attr` is the cached child map. [`Self::set_attr`] invalidates watchers
    /// before changing it.
    pub attr: Cell<MapRef>,
    /// The hidden watcher field implementing mapdict.py:669
    /// `_immutable_fields_ = ['attr?', 'typ?']` for `attr?`. The owner is a
    /// `Box::into_raw` leak that is never freed, so [`QuasiImmutField`]'s
    /// `Drop` is unreachable and its inner box is reclaimed only through
    /// [`QuasiImmutField::invalidate`].
    pub attr_watchers: QuasiImmutField,
    /// mapdict.py:669 `_immutable_fields_` declares `typ?`; mapdict.py:676
    /// `typ` is the unbox type (`None` means boxed). [`Self::set_typ`]
    /// invalidates watchers before changing it.
    pub typ: Cell<Option<UnboxType>>,
    /// The hidden watcher field implementing mapdict.py:669
    /// `_immutable_fields_ = ['attr?', 'typ?']` for `typ?`. The owner is a
    /// `Box::into_raw` leak that is never freed, so [`QuasiImmutField`]'s
    /// `Drop` is unreachable and its inner box is reclaimed only through
    /// [`QuasiImmutField::invalidate`].
    pub typ_watchers: QuasiImmutField,
}

impl CachedAttributeHolder {
    /// Change mapdict.py:675 `attr` using the ordering, fast path, and
    /// repeated-value guard documented on [`Terminator::set_allow_unboxing`].
    pub fn set_attr(&self, a: MapRef) {
        if self.attr.get() == a {
            return;
        }
        if self.attr_watchers.is_installed() {
            unsafe { pyre_object::quasiimmut::sweep_quasi_immut_field(&self.attr_watchers) };
        }
        self.attr.set(a);
    }

    /// Change mapdict.py:676 `typ` using the ordering, fast path, and
    /// repeated-value guard documented on [`Terminator::set_allow_unboxing`].
    pub fn set_typ(&self, t: Option<UnboxType>) {
        if self.typ.get() == t {
            return;
        }
        if self.typ_watchers.is_installed() {
            unsafe { pyre_object::quasiimmut::sweep_quasi_immut_field(&self.typ_watchers) };
        }
        self.typ.set(t);
    }

    pub fn register_attr_watcher(&self, flag: &Arc<AtomicBool>) {
        self.attr_watchers.register_loop_token(flag);
    }

    pub fn install_attr_watcher(&self) {
        self.attr_watchers.ensure_installed();
    }

    pub fn attr_qmut_installed(&self) -> bool {
        self.attr_watchers.is_installed()
    }

    pub fn force_attr_qmut(&self) {
        self.attr_watchers.invalidate();
    }

    pub fn register_typ_watcher(&self, flag: &Arc<AtomicBool>) {
        self.typ_watchers.register_loop_token(flag);
    }

    pub fn install_typ_watcher(&self) {
        self.typ_watchers.ensure_installed();
    }

    pub fn typ_qmut_installed(&self) -> bool {
        self.typ_watchers.is_installed()
    }

    pub fn force_typ_qmut(&self) {
        self.typ_watchers.invalidate();
    }
}

/// Register a loop against a CachedAttributeHolder's `attr?` field.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_register_attr_watcher(
    holder: *const CachedAttributeHolder,
    flag: &Arc<AtomicBool>,
) {
    if holder.is_null() {
        return;
    }
    unsafe { (*holder).register_attr_watcher(flag) };
}

/// Install a CachedAttributeHolder's `attr?` watcher without registering a loop.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_install_attr_watcher(holder: *const CachedAttributeHolder) {
    if holder.is_null() {
        return;
    }
    unsafe { (*holder).install_attr_watcher() };
}

/// Whether a CachedAttributeHolder's `attr?` watcher is installed.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_attr_qmut_installed(holder: *const CachedAttributeHolder) -> bool {
    !holder.is_null() && unsafe { (*holder).attr_qmut_installed() }
}

/// Force a CachedAttributeHolder's `attr?` qmut directly. This is the tracer's
/// own `do_force_quasi_immutable` call (pyjitpl.py:1113-1115), not a runtime
/// store, so it calls [`QuasiImmutField::invalidate`] rather than the sweep.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_force_attr_qmut(holder: *const CachedAttributeHolder) {
    if holder.is_null() {
        return;
    }
    unsafe { (*holder).force_attr_qmut() };
}

/// Register a loop against a CachedAttributeHolder's `typ?` field.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_register_typ_watcher(
    holder: *const CachedAttributeHolder,
    flag: &Arc<AtomicBool>,
) {
    if holder.is_null() {
        return;
    }
    unsafe { (*holder).register_typ_watcher(flag) };
}

/// Install a CachedAttributeHolder's `typ?` watcher without registering a loop.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_install_typ_watcher(holder: *const CachedAttributeHolder) {
    if holder.is_null() {
        return;
    }
    unsafe { (*holder).install_typ_watcher() };
}

/// Whether a CachedAttributeHolder's `typ?` watcher is installed.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_typ_qmut_installed(holder: *const CachedAttributeHolder) -> bool {
    !holder.is_null() && unsafe { (*holder).typ_qmut_installed() }
}

/// Force a CachedAttributeHolder's `typ?` qmut directly. This is the tracer's
/// own `do_force_quasi_immutable` call (pyjitpl.py:1113-1115), not a runtime
/// store, so it calls [`QuasiImmutField::invalidate`] rather than the sweep.
///
/// # Safety
/// `holder` must be null or point at a live owner.
pub unsafe fn holder_force_typ_qmut(holder: *const CachedAttributeHolder) {
    if holder.is_null() {
        return;
    }
    unsafe { (*holder).force_typ_qmut() };
}

/// mapdict.py:670-676 `CachedAttributeHolder.__init__`.
///
/// # Safety
/// `back` must point to a live map node.
unsafe fn new_cached_attribute_holder(
    name: Wtf8Buf,
    attrkind: u16,
    back: MapRef,
    unbox_type: Option<UnboxType>,
    order: usize,
) -> *const CachedAttributeHolder {
    let attr = match unbox_type {
        None => unsafe { new_plain_attribute(name, attrkind, back, order) },
        Some(typ) => unsafe { new_unboxed_plain_attribute(name, attrkind, back, order, typ) },
    };
    Box::into_raw(Box::new(CachedAttributeHolder {
        order,
        attr: Cell::new(attr),
        attr_watchers: QuasiImmutField::new(),
        typ: Cell::new(unbox_type),
        typ_watchers: QuasiImmutField::new(),
    }))
}

/// mapdict.py:679-691 `CachedAttributeHolder.pick_attr`.
///
/// # Safety
/// `holder` must point to a live `CachedAttributeHolder`.
unsafe fn holder_pick_attr(
    holder: *const CachedAttributeHolder,
    unbox_type: Option<UnboxType>,
) -> MapRef {
    let h = unsafe { &*holder };
    let typ = h.typ.get();
    if typ.is_none() || typ == unbox_type {
        return h.attr.get();
    }
    // The cached attribute was unboxed but the new value has a different type;
    // invalidate unboxing for this terminator and re-box (mapdict.py:682-690).
    h.set_typ(None);
    let attr = h.attr.get();
    let p = unsafe { (*attr).as_plain() };
    unsafe { (*p.terminator).as_terminator() }.set_allow_unboxing(false);
    let new_attr = unsafe { new_plain_attribute(p.name.clone(), p.attrkind, p.back, h.order) };
    h.set_attr(new_attr);
    new_attr
}

/// mapdict.py:149-156 `AbstractAttribute._get_new_attr`.
///
/// # Safety
/// `self_node` must point to a live map node.
unsafe fn get_new_attr(
    self_node: MapRef,
    name: &Wtf8,
    attrkind: u16,
    unbox_type: Option<UnboxType>,
) -> *const CachedAttributeHolder {
    let key = (name.to_owned(), attrkind);
    // PyPy mutates this ordinary per-node dict while holding the GIL
    // (mapdict.py:149-156). Map nodes are process-global in pyre, so keep the
    // lookup, holder construction, and insertion under one lock: parallel
    // interpreters must intern exactly one transition for a given key.
    let mut cache = unsafe { (*self_node).cache_attrs() }
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if let Some(&holder) = cache.get(&key) {
        return holder;
    }
    let order = cache.len();
    let holder = unsafe {
        new_cached_attribute_holder(name.to_owned(), attrkind, self_node, unbox_type, order)
    };
    cache.insert(key, holder);
    holder
}

/// mapdict.py:170-193 `AbstractAttribute._find_branch_to_move_into`.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
unsafe fn find_branch_to_move_into(
    self_node: MapRef,
    name: &Wtf8,
    attrkind: u16,
    unbox_type: Option<UnboxType>,
) -> (usize, *const CachedAttributeHolder) {
    let mut current_order = usize::MAX; // sys.maxint
    let mut number_to_readd = 0usize;
    let mut current = self_node;
    let key = (name.to_owned(), attrkind);
    loop {
        let holder = unsafe { (*current).cache_attrs() }
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&key)
            .copied();
        let reached_top = match holder {
            None => true,
            Some(h) => (unsafe { (*h).order }) > current_order,
        };
        if reached_top {
            // didn't find it anywhere yet; if we reached a non-PlainAttribute
            // (the terminator), just add it at the top attribute
            if unsafe { (*current).is_plain() } {
                // keep walking up
            } else {
                return (0, unsafe {
                    get_new_attr(self_node, name, attrkind, unbox_type)
                });
            }
        } else {
            return (number_to_readd, holder.unwrap());
        }
        // not found here, try the parent
        number_to_readd += 1;
        let p = unsafe { (*current).as_plain() };
        current_order = p.order;
        current = p.back;
    }
}

/// Side-effect-free twin of mapdict.py:170-193
/// `AbstractAttribute._find_branch_to_move_into` for the tracer's
/// `opimpl_jit_force_quasi_immutable` predicate.
///
/// The ordinary resolver interns a missing transition through `get_new_attr`.
/// This resolver instead returns `None` at exactly that point, making the
/// caller decline rather than changing later `CachedAttributeHolder.order`
/// assignments merely because a trace was attempted. Each `cache_attrs` lock
/// is released before the chain walk continues; no `QuasiImmutField` lock may
/// be taken while a `cache_attrs` mutex or an `INSTANCE_LOCKS` stripe is held.
/// In particular, moving the force check inside `add_attr` would recursively
/// take the non-reentrant `cache_attrs` mutex and deadlock.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
unsafe fn find_branch_to_move_into_readonly(
    self_node: MapRef,
    name: &Wtf8,
    attrkind: u16,
    _unbox_type: Option<UnboxType>,
) -> Option<(usize, *const CachedAttributeHolder)> {
    let mut current_order = usize::MAX; // sys.maxint
    let mut number_to_readd = 0usize;
    let mut current = self_node;
    loop {
        let holder = {
            let cache = unsafe { (*current).cache_attrs() }
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            cache
                .iter()
                .find(|((cached_name, cached_kind), _)| {
                    cached_kind == &attrkind && &**cached_name == name
                })
                .map(|(_, &holder)| holder)
        };
        let reached_top = match holder {
            None => true,
            Some(h) => (unsafe { (*h).order }) > current_order,
        };
        if reached_top {
            if !unsafe { (*current).is_plain() } {
                // `find_branch_to_move_into` would call `get_new_attr` here.
                return None;
            }
        } else {
            return Some((number_to_readd, holder.unwrap()));
        }
        number_to_readd += 1;
        let p = unsafe { (*current).as_plain() };
        current_order = p.order;
        current = p.back;
    }
}

/// The first mapdict quasi-immutable field a write would change.
#[derive(Clone, Copy)]
pub enum MapdictQmutTarget {
    PlainEverMutated(*const PlainAttribute),
    TerminatorAllowUnboxing(*const Terminator),
    HolderTyp(*const CachedAttributeHolder),
    HolderAttr(*const CachedAttributeHolder),
}

/// Resolve STORE_ATTR's mapdict.py:1618-1627 attribute classification without
/// applying the write. `bool` selects the `"slot"` attrname.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn classify_mapdict_write_attr(w_obj: PyObjectRef, name: &str) -> Option<(u16, bool)> {
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    let w_type = unsafe { (*(*map).terminator()).as_terminator() }.w_cls;
    if w_type.is_null() {
        return None;
    }
    let w_descr = unsafe { crate::baseobjspace::lookup_in_type_where(w_type, name) };
    Some(unsafe { classify_attr(w_type, w_descr, true) })
}

/// Return the first `?` field mapdict.py:68-75 `AbstractAttribute.write` would
/// flip, without performing the write or interning a transition.
///
/// Watcher installation belongs to the tracer. This predicate never reads an
/// `*_qmut_installed` flag and never holds a `cache_attrs` mutex or an
/// `INSTANCE_LOCKS` stripe while a `QuasiImmutField` lock can be taken.
///
/// # Safety
/// `w_obj` and `w_value` must be live objects.
pub unsafe fn setattr_would_force_quasi_immut(
    w_obj: PyObjectRef,
    name: &Wtf8,
    attrkind: u16,
    w_value: PyObjectRef,
) -> Option<MapdictQmutTarget> {
    let map = unsafe { mapdict_map_or_null(w_obj) };
    if map.is_null() {
        return None;
    }
    if let Some(attr) = unsafe { find_map_attr(map, name, attrkind) } {
        let p = unsafe { (*attr).as_plain() };
        if !p.ever_mutated.get() {
            return Some(MapdictQmutTarget::PlainEverMutated(p));
        }
        if let Some(unboxed) = &p.unboxed {
            let term = unsafe { (*p.terminator).as_terminator() };
            if !unsafe { value_has_unbox_type(unboxed.typ, w_value) } && term.allow_unboxing.get() {
                return Some(MapdictQmutTarget::TerminatorAllowUnboxing(term));
            }
        }
        return None;
    }

    let term = unsafe { (*(*map).terminator()).as_terminator() };
    if attrkind == DICT && term.kind != TerminatorKind::Dict {
        return None;
    }
    let unbox_type = unsafe { pick_unbox_type(map, w_value) };
    let (_, holder) =
        unsafe { find_branch_to_move_into_readonly(map, name, attrkind, unbox_type) }?;
    let typ = unsafe { (*holder).typ.get() };
    if typ.is_some() && typ != unbox_type {
        Some(MapdictQmutTarget::HolderTyp(holder))
    } else {
        None
    }
}

/// Return the first `?` field mapdict.py:77-78 / 461-470 delete would flip,
/// without applying the deletion.
///
/// The `PlainAttribute.delete` → `_copy_attr` → `add_attr` → `pick_attr`
/// re-add chain (mapdict.py:461-470) is deliberately not modelled: simulating
/// that rebuild side-effect-free is not tractable against the current
/// `node_copy` / `add_attr` shape. The optimizer's
/// `quasiimmut_field_still_valid` revalidation (heap.py:798-804, ported in
/// majit-metainterp's `optimizeopt/heap.rs`) discards a loop whose recorded `?`
/// value moved during tracing, so a missed force costs a wasted trace, never
/// correctness. Watcher installation belongs to the tracer; this predicate
/// reads no `*_qmut_installed` flag and takes no `QuasiImmutField` lock while a
/// `cache_attrs` mutex or an `INSTANCE_LOCKS` stripe is held.
///
/// # Safety
/// `w_obj` must be a live object.
pub unsafe fn delattr_would_force_quasi_immut(
    w_obj: PyObjectRef,
    name: &Wtf8,
    attrkind: u16,
) -> Option<MapdictQmutTarget> {
    let mut current = unsafe { mapdict_map_or_null(w_obj) };
    while !current.is_null() && unsafe { (*current).is_plain() } {
        let p = unsafe { (*current).as_plain() };
        if p.attrkind == attrkind && &*p.name == name {
            return (!p.ever_mutated.get()).then_some(MapdictQmutTarget::PlainEverMutated(p));
        }
        current = p.back;
    }
    None
}

/// mapdict.py:195-202 `AbstractAttribute._pick_unbox_type`.
///
/// Returns the unbox type when the terminator allows unboxing and the value is
/// an unboxable int (only on 64-bit, `ALLOW_UNBOXING_INTS`) or float.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes; `w_value` to a live
/// object.
unsafe fn pick_unbox_type(self_node: MapRef, w_value: PyObjectRef) -> Option<UnboxType> {
    let term = unsafe { (*(*self_node).terminator()).as_terminator() };
    if term.allow_unboxing.get() {
        if ALLOW_UNBOXING_INTS && unsafe { is_unboxable_int(w_value) } {
            return Some(UnboxType::Int);
        } else if unsafe { is_unboxable_float(w_value) } {
            return Some(UnboxType::Float);
        }
    }
    None
}

/// mapdict.py:449-459 `PlainAttribute._switch_map_and_write_increase_storage1`
/// and the `UnboxedPlainAttribute` override (mapdict.py:629-646).
///
/// # Safety
/// `attr` must point to a live `PlainAttribute` map node.
unsafe fn switch_map_and_write_increase_storage1<O: MapdictObject>(
    attr: MapRef,
    obj: &mut O,
    w_value: PyObjectRef,
) {
    let p = unsafe { (*attr).as_plain() };
    match &p.unboxed {
        None => {
            // mapdict.py:449-459
            if unsafe { (*attr).storage_needed() } > obj._mapdict_storage_length() {
                // erase_item is identity
                obj._set_mapdict_increase_storage1(attr, w_value);
                return;
            }
            // change the map first, then the storage
            obj._set_mapdict_map(attr);
            unsafe { plain_direct_write(attr, obj, w_value) };
        }
        Some(u) => {
            // mapdict.py:629-646
            let val = unsafe { unbox_value(u.typ, w_value) };
            if u.firstunwrapped {
                // a fresh longlong list of one element occupies a new slot
                let unboxed = erase_unboxed(&[val]);
                if unsafe { (*attr).storage_needed() } > obj._mapdict_storage_length() {
                    obj._set_mapdict_increase_storage1(attr, unboxed);
                    return;
                }
                obj._set_mapdict_map(attr);
                obj._mapdict_write_storage(p.storageindex, unboxed);
            } else {
                // append to the existing shared list (a fresh list, matching
                // PyPy's `unboxed + [val]`)
                let slot = obj._mapdict_read_storage(p.storageindex);
                let mut new_list = unsafe { unboxed_items(slot).to_vec() };
                obj._set_mapdict_map(attr);
                debug_assert_eq!(new_list.len(), u.listindex);
                new_list.push(val);
                obj._mapdict_write_storage(p.storageindex, erase_unboxed(&new_list));
            }
        }
    }
}

/// mapdict.py:204-258 `AbstractAttribute._reorder_and_add` — the complicated
/// case where a lower-order ancestor already has the attribute, so the
/// attributes passed on the way up must be saved and re-added in order.
///
/// PyPy stores the to-be-readded `(map, value)` pairs in a flat erased array
/// indexed by `stack_index`; the Rust port uses a `Vec<(MapRef, PyObjectRef)>`
/// with push/pop (same LIFO behaviour). `erase_item`/`unerase_item` and
/// `erase_map`/`unerase_map` are the identity / the typed tuple here.
///
/// # Safety
/// `self_node`/`attr` and their chains must point to live map nodes.
unsafe fn reorder_and_add<O: MapdictObject>(
    mut self_node: MapRef,
    obj: &mut O,
    mut number_to_readd: usize,
    mut attr: MapRef,
    mut w_value: PyObjectRef,
) {
    let mut stack: Vec<(MapRef, PyObjectRef)> =
        Vec::with_capacity(unsafe { (*self_node).num_attributes() } * 2);
    loop {
        // we found the attributes further up, need to save the previous
        // values of the attributes we passed
        if number_to_readd != 0 {
            let mut current = self_node;
            for _ in 0..number_to_readd {
                // current is a PlainAttribute
                let w_self_value = unsafe { plain_direct_read(current, obj) };
                stack.push((current, w_self_value));
                current = unsafe { (*current).as_plain() }.back;
                obj._mapdict_pop_attribute(current);
            }
        }
        unsafe { switch_map_and_write_increase_storage1(attr, obj, w_value) };

        // readd the current top of the stack
        match stack.pop() {
            None => return,
            Some((next_map, next_value)) => {
                w_value = next_value;
                let (name, attrkind) = {
                    let p = unsafe { (*next_map).as_plain() };
                    (p.name.clone(), p.attrkind)
                };
                self_node = obj._get_mapdict_map();
                let unbox_type = unsafe { pick_unbox_type(self_node, w_value) };
                let (n, holder) =
                    unsafe { find_branch_to_move_into(self_node, &name, attrkind, unbox_type) };
                number_to_readd = n;
                attr = unsafe { holder_pick_attr(holder, unbox_type) };
            }
        }
    }
}

/// mapdict.py:157-169 `AbstractAttribute.add_attr`.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
pub unsafe fn add_attr<O: MapdictObject>(
    self_node: MapRef,
    obj: &mut O,
    name: &Wtf8,
    attrkind: u16,
    w_value: PyObjectRef,
) {
    let unbox_type = unsafe { pick_unbox_type(self_node, w_value) };
    let (number_to_readd, holder) =
        unsafe { find_branch_to_move_into(self_node, name, attrkind, unbox_type) };
    let attr = unsafe { holder_pick_attr(holder, unbox_type) };
    if number_to_readd == 0 {
        unsafe { switch_map_and_write_increase_storage1(attr, obj, w_value) };
    } else {
        // the complicated reorder case
        unsafe { reorder_and_add(self_node, obj, number_to_readd, attr, w_value) };
    }
}

/// mapdict.py:312-321 `Terminator._write_terminator` plus the
/// `NoDictTerminator` override (mapdict.py:377-380).
///
/// # Safety
/// `term` must point to a live Terminator map node.
unsafe fn write_terminator<O: MapdictObject>(
    term: MapRef,
    obj: &mut O,
    name: &Wtf8,
    attrkind: u16,
    w_value: PyObjectRef,
) -> bool {
    let kind = unsafe { (*term).as_terminator() }.kind;
    match kind {
        // NoDictTerminator: object without __dict__ rejects DICT writes.
        TerminatorKind::NoDict if attrkind == DICT => return false,
        TerminatorKind::Devolved if attrkind == DICT => {
            // mapdict.py:390-396: the devolved terminator writes DICT attributes
            // into the materialised instance dict (`space.setitem_str(
            // obj.getdict(space), name, w_value)`).
            let w_dict = obj.getdict();
            let backing = crate::type_methods::resolve_dict_backing(w_dict);
            unsafe { pyre_object::w_dict_setitem_wtf8(backing, name, w_value) };
            return true;
        }
        _ => {}
    }
    let map = obj._get_mapdict_map();
    unsafe { add_attr(map, obj, name, attrkind, w_value) };
    if attrkind == DICT
        && unsafe { (*obj._get_mapdict_map()).num_attributes() } >= LIMIT_MAP_ATTRIBUTES
    {
        // mapdict.py:317-323: once a non-devolved instance accumulates
        // >= LIMIT_MAP_ATTRIBUTES DICT attributes, devolve its `__dict__` to a
        // UnicodeDictStrategy r_dict. `obj.getdict()` returns the MapDictStrategy
        // view installed by the `_obj_getdict` flip (asserted MapDictStrategy at
        // mapdict.py:320-322); `switch_to_text_strategy` materialises the DICT
        // attributes into the fresh strategy and rebuilds the map rooted at the
        // DevolvedDictTerminator. Only reachable for a `W_ObjectObject` carrier:
        // a devolved instance returns early through the Devolved arm above, and an
        // `Object` carrier never accrues DICT writes through a Dict terminator.
        let w_dict = obj.getdict();
        debug_assert_eq!(
            unsafe {
                (*(w_dict as *const pyre_object::W_DictObject))
                    .dstrategy
                    .strategy_kind()
            },
            pyre_object::dictmultiobject::StrategyKind::Map,
            "LIMIT-devolve expects a MapDictStrategy __dict__ view",
        );
        unsafe { mapdict_switch_to_text_strategy(w_dict) };
    }
    true
}

/// mapdict.py:68-75 `AbstractAttribute.write`.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
pub unsafe fn node_write<O: MapdictObject>(
    self_node: MapRef,
    obj: &mut O,
    name: &Wtf8,
    attrkind: u16,
    w_value: PyObjectRef,
) -> bool {
    match unsafe { find_map_attr(self_node, name, attrkind) } {
        None => unsafe {
            write_terminator((*self_node).terminator(), obj, name, attrkind, w_value)
        },
        Some(attr) => {
            let p = unsafe { (*attr).as_plain() };
            p.set_ever_mutated(true);
            unsafe { plain_direct_write(attr, obj, w_value) };
            true
        }
    }
}

/// These tables are the temporary carrier for builtin-layout objects that do
/// not yet have mapdict SPECIAL fields.  PyPy's SPECIAL fields are visible to
/// every ExecutionContext, so the compatibility carrier must be
/// interpreter/process-owned too, never TLS.
pub static INSTANCE_DICT: LazyLock<Mutex<HashMap<usize, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
pub static WEAKREF_TABLE: LazyLock<Mutex<HashMap<usize, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Keys whose table value has been stored since the last minor root walk, and
/// so may still be a nursery object.  See [`snapshot_root_entries`].
static INSTANCE_DICT_PENDING: LazyLock<Mutex<HashSet<usize>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));
static WEAKREF_TABLE_PENDING: LazyLock<Mutex<HashSet<usize>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

fn instance_dict_insert(key: PyObjectRef, w_dict: PyObjectRef) {
    INSTANCE_DICT
        .lock()
        .unwrap()
        .insert(key as usize, w_dict as usize);
    INSTANCE_DICT_PENDING.lock().unwrap().insert(key as usize);
}

fn weakref_table_insert(key: PyObjectRef, value: PyObjectRef) {
    WEAKREF_TABLE
        .lock()
        .unwrap()
        .insert(key as usize, value as usize);
    WEAKREF_TABLE_PENDING.lock().unwrap().insert(key as usize);
}

struct MapdictRootArea;
static MAPDICT_ROOT_AREA: MapdictRootArea = MapdictRootArea;

// ── MapdictDictSupport ────────────────────────────────────────────────

/// `MapDictStrategy.length` (mapdict.py:1213-1220) — count the DICT attributes
/// by walking the `search(DICT)` chain. `dont_look_inside`: the map-node layer
/// (incl. `ensure_mapdict_initialized` → `new_instance_terminator`) is a JIT
/// residual boundary while Slice D's unboxed branches stay unported, matching
/// [`instance_node_getdictvalue`].
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` backing a hasdict instance.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_dict_length(obj: PyObjectRef) -> usize {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let inst = mapdict_carrier(obj);
    let mut res: usize = 0;
    let mut curr = node_search(inst._get_mapdict_map(), DICT);
    while let Some(node) = curr {
        // mapdict.py:1216-1219: advance to `back`, re-search, count.
        let back = (*node).as_plain().back;
        curr = node_search(back, DICT);
        res += 1;
    }
    res
}

/// `MapDictStrategy.clear` (mapdict.py:1222-1225) — rebuild the instance's
/// map+storage with every DICT entry dropped. `dont_look_inside` (same rationale
/// as [`instance_node_dict_length`]).
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` backing a hasdict instance.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_dict_clear(obj: PyObjectRef) {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let mut inst = mapdict_carrier(obj);
    let map = inst._get_mapdict_map();
    let new_obj = node_remove_dict_entries(map, &inst);
    inst._set_mapdict_storage_and_map(new_obj.storage, new_obj.map);
}

/// Collect the instance's DICT attribute nodes in insertion order (oldest
/// first): walk `search(DICT)` newest-first (mapdict.py:1240-1247) then reverse
/// (mapdict.py:1250). Shared by the keys/values/items wrappers.
///
/// # Safety
/// `inst` must be a live carrier whose map chain is live.
unsafe fn dict_nodes_in_order<O: MapdictObject>(inst: &O) -> Vec<MapRef> {
    let mut newest_first: Vec<MapRef> = Vec::new();
    let mut curr = node_search(inst._get_mapdict_map(), DICT);
    while let Some(node) = curr {
        newest_first.push(node);
        let back = (*node).as_plain().back;
        curr = node_search(back, DICT);
    }
    let mut ordered: Vec<MapRef> = Vec::new();
    let mut i = newest_first.len();
    while i > 0 {
        i -= 1;
        ordered.push(newest_first[i]);
    }
    ordered
}

/// `MapDictStrategy.iterkeys` materialised (mapdict.py:1269-1272 / w_keys) —
/// the DICT attribute names wrapped as str keys, in insertion order.
/// `dont_look_inside` (same rationale).
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` backing a hasdict instance.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_dict_keys(obj: PyObjectRef) -> Vec<PyObjectRef> {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let inst = mapdict_carrier(obj);
    let nodes = dict_nodes_in_order(&inst);
    let mut keys: Vec<PyObjectRef> = Vec::new();
    let mut i: usize = 0;
    while i < nodes.len() {
        let node = nodes[i];
        let name = &(*node).as_plain().name;
        // baseobjspace.py:new_interned_str / mapdict.py attribute names:
        // materialising an instance dictionary must return the canonical
        // interned name object, not a fresh equal string on every iteration.
        keys.push(pyre_object::unicodeobject::box_str_constant(name));
        i += 1;
    }
    keys
}

/// `MapDictStrategy` values materialised (mapdict.py:1273-1276) — the DICT
/// attribute values in insertion order. `dont_look_inside` (same rationale).
///
/// Reads via `plain_direct_read` (the pure `_prim_direct_read`), intentionally
/// omitting the `_direct_read` convert-on-read tail (mapdict.py:592-598). That
/// tail is value-invisible — it returns the same box and only re-lays-out
/// storage — and upstream performs it safely only because `MapDictIterator*`
/// is name-keyed and lazy, re-resolving each attr against the rebuilt map on
/// every `next_*`. This materialiser instead snapshots raw node pointers up
/// front; converting mid-walk would `_set_mapdict_storage_and_map` in place and
/// leave the already-snapshotted nodes' `storageindex` desynced against the
/// re-laid-out storage Vec. The migrate stays at the name-keyed getattr
/// boundary (`instance_node_getdictvalue`), matching upstream's read site.
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` backing a hasdict instance.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_dict_values(obj: PyObjectRef) -> Vec<PyObjectRef> {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let inst = mapdict_carrier(obj);
    let nodes = dict_nodes_in_order(&inst);
    let mut vals: Vec<PyObjectRef> = Vec::new();
    let mut i: usize = 0;
    while i < nodes.len() {
        let node = nodes[i];
        vals.push(plain_direct_read(node, &inst));
        i += 1;
    }
    vals
}

/// `MapDictStrategy` items materialised (mapdict.py:1275-1276) — (str key,
/// value) pairs in insertion order. `dont_look_inside` (same rationale).
///
/// Uses the pure `plain_direct_read` and omits the `_direct_read` convert-on-
/// read tail for the same reason as [`instance_node_dict_values`].
///
/// # Safety
/// `obj` must be a live `W_ObjectObject` backing a hasdict instance.
#[majit_macros::dont_look_inside]
pub unsafe fn instance_node_dict_items(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    let _instance_guard = instance_lock(obj);
    ensure_mapdict_initialized(obj);
    let inst = mapdict_carrier(obj);
    let nodes = dict_nodes_in_order(&inst);
    let mut out: Vec<(PyObjectRef, PyObjectRef)> = Vec::new();
    let mut i: usize = 0;
    while i < nodes.len() {
        let node = nodes[i];
        let name = &(*node).as_plain().name;
        let w_key = pyre_object::unicodeobject::box_str_constant(name);
        let w_value = plain_direct_read(node, &inst);
        out.push((w_key, w_value));
        i += 1;
    }
    out
}

/// rerased unerase for [`MapDictStrategy`] (mapdict.py:1125-1127): the dict's
/// erased `dstorage` IS the backing instance (mapdict.py:1502
/// `strategy.erase(self)`), so unerasing yields the `W_ObjectObject`
/// PyObjectRef directly.
///
/// # Safety
/// `w_dict` must be a `W_DictObject` whose strategy is [`MapDictStrategy`].
unsafe fn mapdict_strategy_unerase(w_dict: PyObjectRef) -> PyObjectRef {
    let dict = &*(w_dict as *const pyre_object::W_DictObject);
    dict.dstorage as PyObjectRef
}

/// `MapDictStrategy.switch_to_object_strategy` (mapdict.py:1139-1146) —
/// install a fresh ObjectDictStrategy r_dict over the dict, then materialise the
/// instance's DICT attributes into it (the map devolves to its paired
/// DevolvedDictTerminator). `dont_look_inside` keeps the residual boundary like
/// `instance_node_*`: `materialize_dict` reaches `plain_direct_read`, whose
/// unboxed branch (Slice D) the annotator cannot lower; the boundary lets the
/// callers (`getitem`/`setitem`/`delitem` non-str arms) stay lowerable.
///
/// Unlike a typed strategy's switch, the old `dstorage` here is the backing
/// `W_ObjectObject` (mapdict.py:1502 `strategy.erase(self)`), an immortal Box,
/// not an owned r_dict — so it is overwritten, never freed.
///
/// # Safety
/// `w_dict` must be a `W_DictObject` whose strategy is [`MapDictStrategy`].
#[majit_macros::dont_look_inside]
pub unsafe fn mapdict_switch_to_object_strategy(w_dict: PyObjectRef) {
    use pyre_object::dictmultiobject::DictStrategy;
    // w_obj = self.unerase(w_dict.dstorage) — the backing instance.
    let w_obj = unsafe { mapdict_strategy_unerase(w_dict) };
    // dict_w = strategy.unerase(strategy.get_empty_storage()); set_strategy(Object);
    // w_dict.dstorage = strategy.erase(dict_w).
    let dict = unsafe { &mut *(w_dict as *mut pyre_object::W_DictObject) };
    dict.dstorage = pyre_object::dictmultiobject::OBJECT_DICT_STRATEGY.get_empty_storage();
    dict.dstrategy = &pyre_object::dictmultiobject::OBJECT_DICT_STRATEGY_REF;
    // materialize_r_dict(space, w_obj, dict_w).
    unsafe { materialize_dict(w_obj, w_dict) };
}

/// `MapDictStrategy.switch_to_text_strategy` (mapdict.py:1148-1155) — the
/// LIMIT-devolve sibling of [`mapdict_switch_to_object_strategy`]: install a
/// fresh UnicodeDictStrategy r_dict and materialise into it. Same residual
/// boundary and same overwrite-not-free `dstorage` contract.
///
/// # Safety
/// `w_dict` must be a `W_DictObject` whose strategy is [`MapDictStrategy`].
#[majit_macros::dont_look_inside]
pub unsafe fn mapdict_switch_to_text_strategy(w_dict: PyObjectRef) {
    use pyre_object::dictmultiobject::DictStrategy;
    let w_obj = unsafe { mapdict_strategy_unerase(w_dict) };
    let dict = unsafe { &mut *(w_dict as *mut pyre_object::W_DictObject) };
    dict.dstorage = pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY.get_empty_storage();
    dict.dstrategy = &pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY_REF;
    // materialize_str_dict(space, w_obj, str_dict).
    unsafe { materialize_dict(w_obj, w_dict) };
}

/// mapdict.py:1123-1279 `MapDictStrategy` — the dict strategy a user instance's
/// `__dict__` adopts. `dstorage` erases the backing `W_ObjectObject`
/// (mapdict.py:1502), so every routed get/set/del/iter funnels into the
/// instance's mapdict map+storage. `_obj_getdict` installs this view.
pub struct MapDictStrategy;

// mapdict.py:310: allow_unboxing remains true on this shared terminator. A
// type conflict in holder_pick_attr consequently freezes unboxing for every
// instance dictionary sharing this transition tree.
static TERMINATOR_FOR_DICTS: LazyLock<usize> =
    LazyLock::new(|| new_dict_terminator(pyre_object::PY_NULL) as usize);

pub fn get_terminator_for_dicts() -> MapRef {
    *TERMINATOR_FOR_DICTS as MapRef
}

/// mapdict.py:1299-1303 `make_instance_dict`.
#[majit_macros::dont_look_inside]
pub fn make_instance_dict() -> PyObjectRef {
    let w_fake_object = pyre_object::w_instance_new(pyre_object::PY_NULL);
    let _roots = pyre_object::gc_roots::push_roots();
    let fake_slot = pyre_object::gc_roots::pin_roots(&[w_fake_object]);
    unsafe {
        (&mut *(pyre_object::gc_roots::shadow_stack_get(fake_slot)
            as *mut pyre_object::W_ObjectObject))
            ._set_mapdict_map(get_terminator_for_dicts());
    }
    _obj_getdict(pyre_object::gc_roots::shadow_stack_get(fake_slot))
}

/// `space.fromcache(MapDictStrategy)` process-wide singleton — same `&'static`
/// ZST contract as [`pyre_object::dictmultiobject::OBJECT_DICT_STRATEGY`].
pub static MAP_DICT_STRATEGY: MapDictStrategy = MapDictStrategy;

/// The [`pyre_object::dictmultiobject::DictStrategyRef`] holder a dict's
/// `dstrategy` slot points at.
pub static MAP_DICT_STRATEGY_REF: pyre_object::dictmultiobject::DictStrategyRef =
    pyre_object::dictmultiobject::DictStrategyRef {
        imp: &MAP_DICT_STRATEGY,
    };

impl pyre_object::dictmultiobject::DictStrategy for MapDictStrategy {
    fn strategy_kind(&self) -> pyre_object::dictmultiobject::StrategyKind {
        pyre_object::dictmultiobject::StrategyKind::Map
    }

    /// mapdict.py:1132-1137 `get_empty_storage` — "mainly used for tests": a
    /// fresh fake `W_ObjectObject` carrier on the shared dict terminator,
    /// erased. Production dstorage is likewise the backing instance
    /// (mapdict.py:1502).
    fn get_empty_storage(&self) -> *mut u8 {
        let w_result = pyre_object::w_instance_new(pyre_object::PY_NULL);
        unsafe {
            (&mut *(w_result as *mut pyre_object::W_ObjectObject))
                ._set_mapdict_map(get_terminator_for_dicts());
        }
        w_result as *mut u8
    }

    /// mapdict.py:1157-1166 `getitem`.
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> Option<PyObjectRef> {
        if pyre_object::is_exact_type(w_key, &pyre_object::STR_TYPE) {
            // mapdict.py:1161 `space.text_w(w_key)` — an exact str key
            // (including a
            // lone surrogate) is looked up by its full WTF-8 name, so a
            // surrogate-named attribute is a mapdict node like any other.
            return instance_node_getdictvalue(
                mapdict_strategy_unerase(w_dict),
                pyre_object::w_str_get_wtf8(w_key),
            );
        }
        if pyre_object::_never_equal_to_string(w_key) {
            return None;
        }
        self.switch_to_object_strategy(w_dict);
        pyre_object::w_dict_lookup(w_dict, w_key)
    }

    /// mapdict.py:1168-1170 `getitem_str` — `w_obj.getdictvalue(space, key)`.
    /// The trait key is a UTF-8 `&str`; a UTF-8 string is valid WTF-8, so the
    /// node lookup uses its `Wtf8` view directly.
    unsafe fn getitem_str(&self, w_dict: PyObjectRef, key: &str) -> Option<PyObjectRef> {
        instance_node_getdictvalue(mapdict_strategy_unerase(w_dict), Wtf8::new(key))
    }

    /// mapdict.py:1177-1183 `setitem`.
    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        if pyre_object::is_exact_type(w_key, &pyre_object::STR_TYPE) {
            // mapdict.py:1180 — store under the full WTF-8 name so a
            // surrogate-named attribute becomes a mapdict node rather than
            // forcing a strategy switch.
            instance_node_setdictvalue(
                mapdict_strategy_unerase(w_dict),
                pyre_object::w_str_get_wtf8(w_key),
                w_value,
            );
            return;
        }
        // Non-string key: it cannot be a mapdict node, so degrade to the
        // object strategy before storing.
        self.switch_to_object_strategy(w_dict);
        pyre_object::w_dict_store(w_dict, w_key, w_value);
    }

    /// mapdict.py:1172-1175 `setitem_str` — `flag = w_obj.setdictvalue(...);
    /// assert flag`. The receiver is the instance behind a materialised
    /// `__dict__` view, so its terminator has a dict and the write cannot fail.
    unsafe fn setitem_str(&self, w_dict: PyObjectRef, key: &str, w_value: PyObjectRef) {
        let flag =
            instance_node_setdictvalue(mapdict_strategy_unerase(w_dict), Wtf8::new(key), w_value);
        debug_assert!(flag, "mapdict.py:1174 assert flag");
    }

    /// mapdict.py:1185-1196 `setdefault`.
    unsafe fn setdefault(
        &self,
        w_dict: PyObjectRef,
        w_key: PyObjectRef,
        w_default: PyObjectRef,
    ) -> PyObjectRef {
        if pyre_object::is_exact_type(w_key, &pyre_object::STR_TYPE) {
            let key = pyre_object::w_str_get_wtf8(w_key);
            if let Some(w_result) =
                instance_node_getdictvalue(mapdict_strategy_unerase(w_dict), key)
            {
                return w_result;
            }
            instance_node_setdictvalue(mapdict_strategy_unerase(w_dict), key, w_default);
            return w_default;
        }
        self.switch_to_object_strategy(w_dict);
        pyre_object::dictmultiobject::w_dict_setdefault_checked(w_dict, w_key, w_default)
            .unwrap_or(w_default)
    }

    /// mapdict.py:1198-1211 `delitem`. pyre's trait returns `bool` (true =
    /// removed) where PyPy raises KeyError on a miss; the caller raises.
    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        if pyre_object::is_exact_type(w_key, &pyre_object::STR_TYPE) {
            // mapdict.py:1203 — delete by the full WTF-8 name (a surrogate
            // name addresses a real node, no strategy switch).
            return instance_node_deldictvalue(
                mapdict_strategy_unerase(w_dict),
                pyre_object::w_str_get_wtf8(w_key),
            );
        }
        if pyre_object::_never_equal_to_string(w_key) {
            return false;
        }
        self.switch_to_object_strategy(w_dict);
        pyre_object::dictmultiobject::OBJECT_DICT_STRATEGY.delitem(w_dict, w_key)
    }

    /// mapdict.py:1213-1220 `length`.
    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        instance_node_dict_length(mapdict_strategy_unerase(w_dict))
    }

    /// mapdict.py:1269-1272 `iterkeys` materialised.
    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        instance_node_dict_keys(mapdict_strategy_unerase(w_dict))
    }

    /// mapdict.py:1273-1274 `itervalues` materialised.
    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        instance_node_dict_values(mapdict_strategy_unerase(w_dict))
    }

    /// mapdict.py:1275-1276 `iteritems` materialised.
    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        instance_node_dict_items(mapdict_strategy_unerase(w_dict))
    }

    /// mapdict.py:1222-1225 `clear`.
    unsafe fn clear(&self, w_dict: PyObjectRef) {
        instance_node_dict_clear(mapdict_strategy_unerase(w_dict));
    }

    /// mapdict.py:1227-1235 `popitem`.
    unsafe fn popitem(&self, w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        let w_obj = mapdict_strategy_unerase(w_dict);
        let _instance_guard = instance_lock(w_obj);
        ensure_mapdict_initialized(w_obj);
        let inst = mapdict_carrier(w_obj);
        let map = inst._get_mapdict_map();
        let curr = node_search(map, DICT)?;
        let key = &(*curr).as_plain().name;
        // mapdict.py:1231 reads the value with `getitem_str(w_dict, key)`, but
        // the trait's `getitem_str` takes a `&str` and a node name is WTF-8:
        // `setitem` stores the full name (mapdict.py:1180), so a lone surrogate
        // is a node here and has no `&str` form. The `node_search` hit makes
        // the value-present arm explicit: `node_read` would return
        // `Some(plain_direct_read(curr, inst))` (mapdict.py:55-66).
        //
        // Box the key before the read's conversion tail runs: `key` borrows the
        // map node, and the migration below replaces the instance's map.
        //
        // Boxing, the read and the migration all allocate, so each is a point
        // where a minor collection can forward a nursery address out from under
        // a Rust local. Pin the carrier and the two results and read them back
        // from the shadow stack after every such call. `map`, `curr` and `key`
        // need no pin: map nodes are not GC-allocated.
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::pin_roots(&[w_obj]);
        let key_slot =
            pyre_object::gc_roots::pin_roots(&[pyre_object::unicodeobject::box_str_constant(key)]);
        let inst = mapdict_carrier(pyre_object::gc_roots::shadow_stack_get(obj_slot));
        let value_slot = pyre_object::gc_roots::pin_roots(&[plain_direct_read(curr, &inst)]);
        // `plain_direct_read` is only `_prim_direct_read` (mapdict.py:600-601).
        // The read still owes `_direct_read`'s tail (mapdict.py:592-598): an
        // unboxed attribute whose terminator has stopped allowing unboxing
        // converts the whole instance to boxed storage. `getdictvalue` pairs
        // the two the same way.
        let mut inst = mapdict_carrier(pyre_object::gc_roots::shadow_stack_get(obj_slot));
        maybe_migrate_to_boxed(map, &mut inst, key, DICT);
        self.delitem(w_dict, pyre_object::gc_roots::shadow_stack_get(key_slot));
        Some((
            pyre_object::gc_roots::shadow_stack_get(key_slot),
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        ))
    }

    /// mapdict.py:1237-1253 `copy`.
    unsafe fn copy(&self, w_dict: PyObjectRef) -> PyObjectRef {
        use pyre_object::dictmultiobject::DictStrategy;

        let copy = pyre_object::w_dict_new_with(
            &pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY_REF,
            pyre_object::dictmultiobject::UNICODE_DICT_STRATEGY.get_empty_storage(),
        );
        for (w_key, w_value) in instance_node_dict_items(mapdict_strategy_unerase(w_dict)) {
            pyre_object::w_dict_store(copy, w_key, w_value);
        }
        copy
    }

    /// mapdict.py:1268-1276 iterator order.
    unsafe fn nth_item(
        &self,
        w_dict: PyObjectRef,
        index: usize,
    ) -> Option<(PyObjectRef, PyObjectRef)> {
        let w_obj = mapdict_strategy_unerase(w_dict);
        let _instance_guard = instance_lock(w_obj);
        ensure_mapdict_initialized(w_obj);
        let inst = mapdict_carrier(w_obj);
        let nodes = dict_nodes_in_order(&inst);
        let node = *nodes.get(index)?;
        Some((
            pyre_object::unicodeobject::box_str_constant(&(*node).as_plain().name),
            plain_direct_read(node, &inst),
        ))
    }

    /// mapdict.py:1278-1279 `MapDictKeyIteratorReversed`.
    unsafe fn getiterreversed(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        let w_obj = mapdict_strategy_unerase(w_dict);
        let _instance_guard = instance_lock(w_obj);
        ensure_mapdict_initialized(w_obj);
        let inst = mapdict_carrier(w_obj);
        let mut items = Vec::new();
        let mut curr = node_search(inst._get_mapdict_map(), DICT);
        while let Some(node) = curr {
            items.push((
                pyre_object::unicodeobject::box_str_constant(&(*node).as_plain().name),
                plain_direct_read(node, &inst),
            ));
            curr = node_search((*node).as_plain().back, DICT);
        }
        items
    }

    /// mapdict.py:1139-1146 `switch_to_object_strategy` — Slice E (#196). The
    /// default would mis-read the instance `dstorage` as an ObjectDictStrategy
    /// `IndexMap`, so override to the materialise stub.
    unsafe fn switch_to_object_strategy(&self, w_dict: PyObjectRef) {
        mapdict_switch_to_object_strategy(w_dict);
    }

    /// The dict is a view over the backing instance (`dstorage =
    /// erase(self)`, mapdict.py:1502). `MapDictStrategy.erase/unerase =
    /// rerased.new_erasing_pair("map")` (mapdict.py:1125) boxes the
    /// instance as a real GC reference, so `dstorage` is a true GC edge
    /// the translated `W_DictMultiObject` tracer forwards. Visit the
    /// `dstorage` field IN PLACE (`addr_of_mut!`, never a
    /// `mapdict_strategy_unerase` stack temporary — the collector writes
    /// the relocated address back through this pointer, so it must be the
    /// real field) so a moving collector rewrites the back-pointer to the
    /// relocated instance. The instance is itself a primary GC object
    /// (forwarded from frame / shadow roots + its own custom trace), so
    /// this edge is an idempotent redundant forwarder: the cycle
    /// instance → storage[SPECIAL] → wrapper → dstorage → instance
    /// terminates on the collector's `is_forwarded` short-circuit.
    unsafe fn walk_gc_refs(&self, w_dict: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        unsafe {
            let dstorage_field =
                std::ptr::addr_of_mut!((*(w_dict as *mut pyre_object::W_DictObject)).dstorage)
                    as *mut PyObjectRef;
            if (*dstorage_field).is_null() {
                return;
            }
            visitor(dstorage_field);
        }
    }
}

/// objspace/std/mapdict.py:826-840 _obj_getdict.
///
/// ```python
/// @objectmodel.dont_inline
/// def _obj_getdict(self, space):
///     terminator = self._get_mapdict_map().terminator
///     assert isinstance(terminator, DictTerminator) or isinstance(terminator, DevolvedDictTerminator)
///     w_dict = self._get_mapdict_map().read(self, "dict", SPECIAL)
///     if w_dict is not None:
///         assert isinstance(w_dict, W_DictMultiObject)
///         return w_dict
///
///     strategy = space.fromcache(MapDictStrategy)
///     storage = strategy.erase(self)
///     w_dict = W_DictObject(space, strategy, storage)
///     flag = self._get_mapdict_map().write(self, "dict", SPECIAL, w_dict)
///     assert flag
///     return w_dict
/// ```
///
/// `dont_look_inside` — the `_obj_setdict` read twin: the miss path reads
/// the address-keyed `INSTANCE_DICT` thread-local side table (and allocates
/// a fresh dict), state the tracer cannot model; the call residualises via
/// the registered fnaddr (`@objectmodel.dont_inline` upstream,
/// mapdict.py:826).
#[majit_macros::dont_look_inside]
pub fn _obj_getdict(self_ref: PyObjectRef) -> PyObjectRef {
    // mapdict.py:828-838: read the "dict" SPECIAL slot; on a miss build the
    // MapDictStrategy view and write it back into that slot. `strategy.erase(self)`
    // makes the view funnel every get/set/del/iter through the instance map+storage
    // — the single `__dict__` authority.
    //
    // Only an object that carries mapdict storage answers out of the map —
    // the same gate `_obj_setdict` applies. User subclasses of builtin types
    // (`class MyInt(int)`) keep the builtin layout (no map) while their type is
    // hasdict, so their `__dict__` stays in the address-keyed INSTANCE_DICT
    // side table as a plain own-storage dict until subclass instances grow
    // mapdict storage (upstream `user_setup`, mapdict.py:758).
    if unsafe { has_mapdict_storage(self_ref) } {
        // mapdict.py:828-830 `if w_dict is not None`.  RPython's `read` answers
        // None both for an absent slot and for one holding None, and both mean
        // "build the view" — so a null must not reach the caller.  It would be
        // reported as "the receiver has no dict", which is how
        // `descr__setattr__` decides an ordinary instance store is
        // `"'%T' object attribute '%s' is read-only"` (descroperation.py:58-67).
        if let Some(w_dict) = unsafe { instance_get_dict_slot(self_ref) }
            && !w_dict.is_null()
        {
            return w_dict;
        }
        // Allocating the wrapper and claiming the SPECIAL slot both collect, so
        // the instance is published first and every use below reads back the
        // relocated address.
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self_ref]);
        let dict_slot = self_slot + 1;
        pyre_object::gc_roots::pin_root(pyre_object::w_dict_new_with(
            &MAP_DICT_STRATEGY_REF,
            pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut u8,
        ));
        unsafe {
            let w_dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
            // `dstorage` is the view's only link to its backing instance
            // (mapdict.py:1502) and `walk_gc_refs` forwards it — but only from
            // the collection after the wrapper became reachable.  The
            // allocation that produced the wrapper is not covered, so restate
            // the back-pointer from the instance's current address.
            (*(w_dict as *mut pyre_object::W_DictObject)).dstorage =
                pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut u8;
            pyre_object::gc_hook::try_gc_write_barrier(w_dict as *mut u8);
            let flag = instance_set_dict_slot(
                pyre_object::gc_roots::shadow_stack_get(self_slot),
                pyre_object::gc_roots::shadow_stack_get(dict_slot),
            );
            debug_assert!(flag, "write to the \"dict\" SPECIAL slot failed");
        }
        pyre_object::gc_roots::shadow_stack_get(dict_slot)
    } else {
        let existing = INSTANCE_DICT
            .lock()
            .unwrap()
            .get(&(self_ref as usize))
            .copied()
            .map(|dict| dict as PyObjectRef);
        if let Some(w_dict) = existing {
            return w_dict;
        }
        let w_dict = pyre_object::w_dict_new();
        instance_dict_insert(self_ref, w_dict);
        w_dict
    }
}

fn current_owner_key(key: usize) -> usize {
    pyre_object::gc_hook::try_gc_current_object_address(key as *mut u8) as usize
}

/// GC custom trace over a live instance's `storage` value slots.
///
/// `mapdict.py:907-910` — an instance's attribute values live in the
/// `storage` block. Every slot is a GC reference: a boxed attribute's
/// `erase_item` is the identity (`mapdict.py:438/447`) and an
/// `UnboxedPlainAttribute`'s `erase_unboxed` (`:601/612`) yields a varsize
/// leaf GcArray ([`erase_unboxed`]), so the map no longer has to be
/// consulted to tell the two apart. The visitor relocates each slot's
/// contents in place, exactly as `dict_object_custom_trace` relocates dict
/// entry slots.
///
/// Registered as `W_OBJECT_OBJECT_GC_TYPE_ID`'s custom trace
/// (`object_object_custom_trace`) so a moving collector forwards an
/// instance's attributes, and so a mark-sweep old generation keeps the
/// unboxed longlong list alive for exactly as long as the instance.
///
/// # Safety
/// `obj` must point to a live `W_ObjectObject`.
pub unsafe fn instance_walk_boxed_storage(obj: PyObjectRef, f: &mut dyn FnMut(*mut PyObjectRef)) {
    unsafe {
        let inst = mapdict_carrier(obj);
        let storage_slot = inst.slots().1;
        if (*storage_slot).is_null() {
            return;
        }
        let storage = *storage_slot;
        if pyre_object::gc_hook::try_gc_owns_object(storage as *mut u8) {
            f(storage_slot as *mut PyObjectRef);
        }
        // The storage block is exact-size (capacity == map.storage_needed()).
        let len = pyre_object::object_array::items_block_capacity(storage);
        let base = pyre_object::object_array::items_block_items_base(storage);
        // No per-slot test and no `Vec`: this runs inside the collector's
        // marking walk, where both would be a per-object cost and a reentrancy
        // hazard, and every slot is a reference.
        for i in 0..len {
            f(base.add(i));
        }
    }
}

/// Walk roots held by pyre's temporary mapdict side tables.
///
/// PyPy stores the instance dict and weakref lifeline in mapdict SPECIAL slots,
/// so the translated GC sees them as ordinary object fields. A `W_ObjectObject`
/// is GC-managed (`W_OBJECT_OBJECT_GC_TYPE_ID`): its attribute storage and "dict"
/// SPECIAL-slot wrapper are forwarded by `object_object_custom_trace`, so this
/// walk no longer touches instances. The remaining side tables hold the weakref
/// lifeline and the wrappers of non-instance hasdict objects (property/member)
/// which have no map and live in immortal `Box`es the GC never scans. Expose
/// those value slots here so the backend GC can update them when nursery objects
/// move.
pub fn walk_mapdict_roots(mut visitor: impl FnMut(&mut PyObjectRef)) {
    let data = capture_mapdict_root_area();
    unsafe { walk_mapdict_roots_area(data, &mut visitor) };
}

pub fn capture_mapdict_root_area() -> *const () {
    &MAPDICT_ROOT_AREA as *const _ as *const ()
}

/// The `(owner, value)` pairs a root walk of `table` has to visit.
///
/// A major walk visits the whole table.  A minor one visits only the keys
/// stored since the previous minor walk, and retires them: an entry an earlier
/// minor walk already visited had its value dragged out to the old generation
/// and its key rewritten to the owner's post-move address, and an old value's
/// own contents are reached through its write barrier and custom trace
/// (`dict_write_barrier` / `dict_object_custom_trace`), never from here.  A
/// non-moving major (`do_collect_oldgen`) leaves the nursery intact, so only
/// the minor walk may drain the pending set.  A value the GC does not own has
/// neither of those two paths, so [`walk_mapdict_roots_area`] puts its key
/// straight back.
///
/// Without this split each collection cloned and walked the whole table, whose
/// entries are roots and therefore outlive their owners: the per-collection
/// cost grew with the number of hasdict builtin-layout objects the program had
/// ever created.
fn snapshot_root_entries(
    table: &Mutex<HashMap<usize, usize>>,
    pending: &Mutex<HashSet<usize>>,
    minor: bool,
) -> Vec<(usize, PyObjectRef)> {
    if !minor {
        return table
            .lock()
            .unwrap()
            .iter()
            .map(|(&key, &value)| (key, value as PyObjectRef))
            .collect();
    }
    let keys = std::mem::take(&mut *pending.lock().unwrap());
    let table = table.lock().unwrap();
    keys.into_iter()
        .filter_map(|key| table.get(&key).map(|&value| (key, value as PyObjectRef)))
        .collect()
}

/// Drop every entry whose owner did not survive the collection.
///
/// These tables are keyed by owner address and hold their value as a root, so
/// without this an owner's `__dict__` / `__weakref__` outlives it: the tables
/// only grow, and every major collection marks the whole accumulation.
/// Upstream has no equivalent table — `typedef.py`'s generated subclass carries
/// `w_dict` as a field of the object, which dies with it — so the entries have
/// to be given the same ephemeron semantics explicitly.
///
/// Registered with `majit_gc::shadow_stack::register_ephemeron_pruner`, which
/// runs it from a major collection only.  `classify` returns the owner's
/// current address, or `None` if it died; a major is mark-and-sweep and moves
/// nothing, so a surviving owner always answers with the key it was asked
/// about and the surviving entries keep their keys.
pub fn prune_dead_owner_entries(classify: &mut dyn FnMut(usize) -> Option<usize>) {
    for (table, pending) in [
        (&INSTANCE_DICT, &INSTANCE_DICT_PENDING),
        (&WEAKREF_TABLE, &WEAKREF_TABLE_PENDING),
    ] {
        let mut table = table.lock().unwrap();
        let dead: Vec<usize> = table
            .keys()
            .copied()
            .filter(|&key| classify(key) != Some(key))
            .collect();
        if dead.is_empty() {
            continue;
        }
        let mut pending = pending.lock().unwrap();
        for key in dead {
            table.remove(&key);
            pending.remove(&key);
        }
    }
}

/// Mark weakref lifelines only for owners that survived major marking.
///
/// PyPy stores this edge in each concrete object's weakref field, so ordinary
/// tracing reaches the lifeline iff it first reaches the owner. The temporary
/// `WEAKREF_TABLE` carrier must reproduce that conditional edge; treating all
/// values as unconditional roots delays callbacks by one major collection.
pub fn mark_live_weakref_entries(
    classify: &mut dyn FnMut(usize) -> Option<usize>,
    roots: &mut Vec<majit_ir::GcRef>,
) {
    let entries: Vec<(usize, usize)> = WEAKREF_TABLE
        .lock()
        .unwrap()
        .iter()
        .map(|(&owner, &value)| (owner, value))
        .collect();
    for (owner, value) in entries {
        if classify(owner).is_some() && value != 0 {
            roots.push(majit_ir::GcRef(value));
        }
    }
}

/// Re-key and re-point the entries a root walk moved, as `(old, new, value)`.
///
/// Collected during the walk and applied in one pass because the walk must not
/// hold the table lock across a visitor callback, and a major walk that
/// re-locked per entry paid a lock and a hash lookup for every entry the
/// program had ever created — almost none of which move, since only a nursery
/// object has an address to rewrite.
fn apply_root_rekeys(table: &Mutex<HashMap<usize, usize>>, rekeys: Vec<(usize, usize, usize)>) {
    if rekeys.is_empty() {
        return;
    }
    let mut table = table.lock().unwrap();
    for (key, new_key, value) in rekeys {
        if new_key == key {
            if let Some(slot) = table.get_mut(&key) {
                *slot = value;
            }
        } else if table.remove(&key).is_some() {
            table.insert(new_key, value);
        }
    }
}

/// # Safety
/// `data` must come from [`capture_mapdict_root_area`], and the owning thread
/// must be quiesced.
pub unsafe fn walk_mapdict_roots_area(_data: *const (), mut visitor: impl FnMut(&mut PyObjectRef)) {
    // incminimark.py:339-355 prebuilt-object scanning parity: a minor
    // collection reaches an old structure only through the write barrier, so
    // only the entries stored since the previous minor walk are visited here.
    let minor = majit_gc::shadow_stack::extra_root_walk_kind()
        == majit_gc::shadow_stack::ExtraRootWalkKind::Minor;
    let dict_values = snapshot_root_entries(&INSTANCE_DICT, &INSTANCE_DICT_PENDING, minor);
    // SAFETY: do not hold the table lock while invoking callbacks. The visitor
    // and w_dict_walk_entries_mut may re-enter mapdict/dict APIs; every write
    // back into the table is deferred to `apply_root_rekeys`.
    let mut dict_rekeys = Vec::new();
    let mut dict_offgc = Vec::new();
    for (key, mut dict) in dict_values {
        let old_dict = dict;
        visitor(&mut dict);
        let new_key = current_owner_key(key);
        if new_key != key || dict != old_dict {
            dict_rekeys.push((key, new_key, dict as usize));
        }
        // A value the GC does not own — `alloc_dict_object` falls back to
        // `malloc_typed` when no allocation hook is installed — carries no
        // header, so `do_write_barrier` drops it (it admits only a nursery or
        // old-generation address) and no custom trace ever reaches its entries.
        // The entry walk below is the only thing that traces them, so such a
        // key stays pending: a minor has to revisit it after every store into
        // the dict, not once when the table entry was first created.
        if !pyre_object::gc_hook::try_gc_owns_object(dict as *mut u8) {
            dict_offgc.push(new_key);
        }
        // Trace the dict's own r_dict entries. INSTANCE_DICT now holds only
        // non-instance hasdict wrappers (property/member) — never a
        // MapDictStrategy view, since an instance's `__dict__` wrapper lives in
        // its "dict" SPECIAL slot (forwarded by the instance custom trace). The
        // `is_map_view` guard stays defensive: a view's `dstorage` IS the backing
        // instance (mapdict.py:1502), not an `IndexMap`, so
        // `w_dict_walk_entries_mut` must never run on one.
        let is_map_view = unsafe {
            (*(dict as *const pyre_object::W_DictObject))
                .dstrategy
                .strategy_kind()
                == pyre_object::dictmultiobject::StrategyKind::Map
        };
        if !is_map_view {
            unsafe {
                pyre_object::w_dict_walk_entries_mut(dict, |slot| {
                    visitor(slot);
                });
            }
        }
        // An instance's own attribute storage and its "dict" SPECIAL-slot
        // wrapper — including a devolved wrapper's own IndexMap, since that
        // wrapper is a GC-managed `W_DictObject` (`w_dict_new_with` →
        // `try_gc_alloc`) carrying its own `dict_object_custom_trace` and write
        // barrier — are forwarded by `object_object_custom_trace`
        // (`W_OBJECT_OBJECT_GC_TYPE_ID`): in major marking, and in minor collection
        // via the instance/wrapper write barriers that enter the remembered set.
        // So no instance is walked here.
    }
    apply_root_rekeys(&INSTANCE_DICT, dict_rekeys);
    if !dict_offgc.is_empty() {
        INSTANCE_DICT_PENDING.lock().unwrap().extend(dict_offgc);
    }

    // The weakref walk visits the lifeline pointer and stops there, so it needs
    // no such re-arming: an off-GC lifeline never moves, and its own fields are
    // outside what this walk ever traced.
    // A major collection handles this table through the ephemeron marker
    // above, after ordinary owner marking has settled. A minor still forwards
    // freshly stored lifelines because owner liveness is not being decided.
    let weakref_values = if minor {
        snapshot_root_entries(&WEAKREF_TABLE, &WEAKREF_TABLE_PENDING, true)
    } else {
        Vec::new()
    };
    let mut weakref_rekeys = Vec::new();
    for (key, mut value) in weakref_values {
        let old_value = value;
        visitor(&mut value);
        let new_key = current_owner_key(key);
        if new_key != key || value != old_value {
            weakref_rekeys.push((key, new_key, value as usize));
        }
    }
    apply_root_rekeys(&WEAKREF_TABLE, weakref_rekeys);
}

/// objspace/std/mapdict.py:842-860 _obj_setdict.
///
/// ```python
/// @objectmodel.dont_inline
/// def _obj_setdict(self, space, w_dict):
///     from pypy.interpreter.error import oefmt
///     terminator = self._get_mapdict_map().terminator
///     assert isinstance(terminator, DictTerminator) or isinstance(terminator, DevolvedDictTerminator)
///     if not space.isinstance_w(w_dict, space.w_dict):
///         raise oefmt(space.w_TypeError, "setting dictionary to a non-dict")
///     assert isinstance(w_dict, W_DictMultiObject)
///     w_olddict = self.getdict(space)
///     ...
///     flag = self._get_mapdict_map().write(self, "dict", SPECIAL, w_dict)
///     assert flag
/// ```
///
/// Writes the per-instance `INSTANCE_DICT` side table through a closure the
/// tracer cannot model; the JIT residualises the call (`@dont_look_inside`).
#[majit_macros::dont_look_inside]
pub fn _obj_setdict(self_ref: PyObjectRef, w_dict: PyObjectRef) -> Result<(), PyError> {
    // mapdict.py:848 `space.isinstance_w(w_dict, space.w_dict)` accepts
    // dict subclasses. Pyre's composed dict-subclass representation is
    // resolved by the getdict backing helpers at each raw dict operation,
    // while the SPECIAL slot retains the supplied object's identity.
    let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
    if !unsafe { crate::baseobjspace::isinstance_w(w_dict, w_dict_type) } {
        return Err(PyError::type_error(
            "setting dictionary to a non-dict".to_string(),
        ));
    }
    if crate::type_methods::resolve_dict_backing(w_dict).is_null() {
        return Err(PyError::type_error(
            "setting dictionary to a non-dict".to_string(),
        ));
    }
    if unsafe { has_mapdict_storage(self_ref) } {
        // mapdict.py:892-900: the old dict has `self` as its dstorage, so
        // before pointing the "dict" SPECIAL slot at the new dict, force the
        // old view to its own storage if it is still an instance-backed
        // `MapDictStrategy`. `_obj_getdict` returns (or materialises) that
        // view; switching it to an ObjectDictStrategy snapshot stops it
        // delegating to the instance once the slot is overwritten — otherwise
        // `old = obj.__dict__; obj.__dict__ = {}` leaves `old` an empty shell
        // that still mirrors the live instance.
        // Materialising the old view and claiming the slot both allocate, so
        // the receiver and the incoming dict are published across them.
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = pyre_object::gc_roots::pin_roots(&[self_ref, w_dict]);
        let dict_slot = self_slot + 1;
        let w_olddict = _obj_getdict(pyre_object::gc_roots::shadow_stack_get(self_slot));
        let old_backing = crate::type_methods::resolve_dict_backing(w_olddict);
        let is_map_view = unsafe {
            pyre_object::dictmultiobject::w_dict_get_strategy(old_backing).strategy_kind()
                == pyre_object::dictmultiobject::StrategyKind::Map
        };
        if is_map_view {
            unsafe { mapdict_switch_to_object_strategy(old_backing) };
        }
        let flag = unsafe {
            instance_set_dict_slot(
                pyre_object::gc_roots::shadow_stack_get(self_slot),
                pyre_object::gc_roots::shadow_stack_get(dict_slot),
            )
        };
        debug_assert!(flag, "write to the \"dict\" SPECIAL slot failed");
    } else {
        // Non-instance hasdict objects (property/member, baseobjspace
        // 1850/3786) keep a plain own-storage dict in the address-keyed side
        // table; it never delegates to a backing object, so no force step.
        instance_dict_insert(self_ref, w_dict);
    }
    Ok(())
}

// ── MapdictWeakrefSupport ─────────────────────────────────────────────

/// objspace/std/mapdict.py:780-787 MapdictWeakrefSupport.getweakref.
///
/// ```python
/// def getweakref(self):
///     from pypy.module._weakref.interp__weakref import WeakrefLifeline
///     lifeline = self._get_mapdict_map().read(self, "weakref", SPECIAL)
///     if lifeline is None:
///         return None
///     assert isinstance(lifeline, WeakrefLifeline)
///     return lifeline
/// ```
pub fn getweakref(self_ref: PyObjectRef) -> Option<PyObjectRef> {
    if unsafe { has_mapdict_storage(self_ref) } {
        unsafe { instance_get_weakref_slot(self_ref) }
    } else {
        WEAKREF_TABLE
            .lock()
            .unwrap()
            .get(&(self_ref as usize))
            .copied()
            .map(|value| value as PyObjectRef)
    }
}

/// objspace/std/mapdict.py:789-793 MapdictWeakrefSupport.setweakref.
///
/// ```python
/// def setweakref(self, space, weakreflifeline):
///     from pypy.module._weakref.interp__weakref import WeakrefLifeline
///     assert isinstance(weakreflifeline, WeakrefLifeline)
///     self._get_mapdict_map().write(self, "weakref", SPECIAL, weakreflifeline)
/// ```
pub fn setweakref(self_ref: PyObjectRef, weakreflifeline: PyObjectRef) {
    if unsafe { has_mapdict_storage(self_ref) } {
        let flag = unsafe { instance_set_weakref_slot(self_ref, weakreflifeline) };
        debug_assert!(flag, "write to the weakref SPECIAL slot failed");
    } else {
        weakref_table_insert(self_ref, weakreflifeline);
    }
}

/// objspace/std/mapdict.py:795-797 MapdictWeakrefSupport.delweakref.
///
/// ```python
/// def delweakref(self):
///     self._get_mapdict_map().write(self, "weakref", SPECIAL, None)
/// ```
pub fn delweakref(self_ref: PyObjectRef) {
    if unsafe { has_mapdict_storage(self_ref) } {
        unsafe { instance_del_weakref_slot(self_ref) };
    } else {
        WEAKREF_TABLE.lock().unwrap().remove(&(self_ref as usize));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Node names are WTF-8; the test fixtures spell them with plain UTF-8
    // string literals. `wn` borrows a `&Wtf8` view for the lookup/read/write
    // helpers, `wb` produces an owned `Wtf8Buf` for `new_plain_attribute`.
    fn wn(s: &str) -> &Wtf8 {
        Wtf8::new(s)
    }
    fn wb(s: &str) -> Wtf8Buf {
        Wtf8::new(s).to_owned()
    }

    // A map chain is `terminator <- "a"(DICT) <- "b"(DICT)`. The w_cls is a
    // null placeholder: the node layer never dereferences it.
    unsafe fn build_chain() -> (MapRef, MapRef, MapRef) {
        let term = new_dict_terminator(std::ptr::null_mut());
        let a = unsafe { new_plain_attribute(wb("a"), DICT, term, 0) };
        let b = unsafe { new_plain_attribute(wb("b"), DICT, a, 1) };
        (term, a, b)
    }

    #[test]
    fn terminator_is_its_own_terminator_with_zero_storage() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            assert_eq!((*term).terminator(), term);
            assert_eq!((*term).storage_needed(), 0);
            assert_eq!((*term).num_attributes(), 0);
            assert_eq!((*term).as_terminator().kind, TerminatorKind::Dict);
            assert!(
                !(*term)
                    .as_terminator()
                    .devolved_dict_terminator
                    .get()
                    .is_null()
            );
        }
    }

    #[test]
    fn plain_attribute_increments_storage_and_keeps_terminator() {
        unsafe {
            let (term, a, b) = build_chain();
            assert_eq!((*a).as_plain().storageindex, 0);
            assert_eq!((*a).storage_needed(), 1);
            assert_eq!((*a).num_attributes(), 1);
            assert_eq!((*b).as_plain().storageindex, 1);
            assert_eq!((*b).storage_needed(), 2);
            assert_eq!((*b).num_attributes(), 2);
            assert_eq!((*a).terminator(), term);
            assert_eq!((*b).terminator(), term);
            assert_eq!((*b).as_plain().back, a);
        }
    }

    #[test]
    fn cache_entry_validity_keys_on_map_identity_and_version_tag() {
        unsafe {
            // mapdict.py:1431-1447 is_valid_for_map / _is_valid_for_map. Build a
            // real type so the terminator carries a w_cls with a live
            // version_tag.
            crate::typedef::init_typeobjects();
            let w_cls = crate::typedef::make_builtin_type("MapdictCacheEntryT", |_| {});
            let term = new_dict_terminator(w_cls);
            let attr = new_plain_attribute(wb("x"), DICT, term, 0);
            let version_tag = pyre_object::typeobject::w_type_get_version_tag(w_cls);

            let entry = MapdictCacheEntry {
                cached_map: attr,
                cached_attr: attr,
                version_tag,
                valid_for_store: false,
                w_method: std::ptr::null_mut(),
            };
            // matching map + version -> valid for a load
            assert!(entry.is_valid_for_map(attr, false));
            // a different map fails identity (mapdict.py:1440 `mymap is map`)
            let other = new_plain_attribute(wb("y"), DICT, term, 1);
            assert!(!entry.is_valid_for_map(other, false));
            // store gate: valid_for_store == false rejects stores (mapdict.py:1432)
            assert!(!entry.is_valid_for_map(attr, true));

            let store_entry = MapdictCacheEntry {
                valid_for_store: true,
                ..entry
            };
            assert!(store_entry.is_valid_for_map(attr, true));

            // a class mutation bumps version_tag -> the entry goes stale
            // (mapdict.py:1442 `version_tag is self.version_tag`).
            crate::baseobjspace::mutated(w_cls, None);
            assert!(!entry.is_valid_for_map(attr, false));
            assert!(!store_entry.is_valid_for_map(attr, true));
        }
    }

    #[test]
    fn find_map_attr_chain_walks_back_pointers() {
        unsafe {
            let (_term, a, b) = build_chain();
            assert_eq!(find_map_attr_chain(b, wn("a"), DICT), Some(a));
            assert_eq!(find_map_attr_chain(b, wn("b"), DICT), Some(b));
            assert_eq!(find_map_attr_chain(b, wn("c"), DICT), None);
            // attrkind namespaces are distinct: "a" exists only under DICT.
            assert_eq!(find_map_attr_chain(b, wn("a"), SPECIAL), None);
        }
    }

    #[test]
    fn find_map_attr_cached_matches_uncached_on_hit_and_miss() {
        unsafe {
            MAP_ATTR_CACHE.lock().unwrap().clear();
            let (_term, a, b) = build_chain();
            // first call populates the cache, second hits it
            assert_eq!(find_map_attr(b, wn("a"), DICT), Some(a));
            assert_eq!(find_map_attr(b, wn("a"), DICT), Some(a));
            assert_eq!(find_map_attr(b, wn("b"), DICT), Some(b));
            assert_eq!(find_map_attr(b, wn("missing"), DICT), None);
            assert_eq!(find_map_attr(b, wn("missing"), DICT), None);
        }
    }

    #[test]
    fn new_attr_cache_interns_transitions_across_threads() {
        use std::sync::{Arc, Barrier};

        let term_addr = new_dict_terminator(std::ptr::null_mut()) as usize;
        let barrier = Arc::new(Barrier::new(16));
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|_| {
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        let term = term_addr as MapRef;
                        barrier.wait();
                        (0..64)
                            .map(|index| {
                                let name = format!("attr_{index}");
                                unsafe {
                                    get_new_attr(term, Wtf8::new(name.as_str()), DICT, None)
                                        as usize
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        for result in &results[1..] {
            assert_eq!(result, &results[0]);
        }
        assert_eq!(
            unsafe { (*(term_addr as MapRef)).cache_attrs() }
                .lock()
                .unwrap()
                .len(),
            64
        );
    }

    #[test]
    fn search_returns_topmost_match_for_attrkind() {
        unsafe {
            let (term, _a, b) = build_chain();
            // both "a" and "b" are DICT; search from b returns b (topmost)
            assert_eq!(node_search(b, DICT), Some(b));
            assert_eq!(node_search(b, SPECIAL), None);
            assert_eq!(node_search(term, DICT), None);
        }
    }

    // A minimal MapdictObject for read-path tests. Storage holds sentinel
    // pointers that are never dereferenced.
    // mapdict.py:978 `Object` is pyre's production storage carrier (empty map,
    // owned `Vec` storage); the tests drive it under the historical `MockObj`
    // name.
    use super::Object as MockObj;

    fn sentinel(n: usize) -> PyObjectRef {
        n as PyObjectRef
    }

    // A DictTerminator whose `allow_unboxing` is off, so writes take the boxed
    // PlainAttribute path and never type-inspect the (sentinel) value.
    unsafe fn boxed_dict_terminator() -> MapRef {
        let term = new_dict_terminator(std::ptr::null_mut());
        unsafe { (*term).as_terminator() }.set_allow_unboxing(false);
        term
    }

    #[test]
    fn node_read_returns_stored_value_by_storageindex() {
        unsafe {
            let (_term, _a, b) = build_chain();
            // map b: "a"@storageindex 0, "b"@storageindex 1
            let obj = MockObj {
                map: b,
                storage: vec![sentinel(0xa), sentinel(0xb)],
            };
            assert_eq!(node_read(b, &obj, wn("a"), DICT), Some(sentinel(0xa)));
            assert_eq!(node_read(b, &obj, wn("b"), DICT), Some(sentinel(0xb)));
            // absent attribute falls through to the (Dict) terminator → None
            assert_eq!(node_read(b, &obj, wn("missing"), DICT), None);
            assert_eq!(node_read(b, &obj, wn("a"), SPECIAL), None);
            assert_eq!(obj._mapdict_storage_length(), 2);
        }
    }

    #[test]
    fn set_terminator_reroots_chain_preserving_attrs() {
        unsafe {
            // chain "a","b" (DICT, boxed) under terminator t1
            let t1 = boxed_dict_terminator();
            let a = new_plain_attribute(wb("a"), DICT, t1, 0);
            let b = new_plain_attribute(wb("b"), DICT, a, 1);
            let obj = MockObj {
                map: b,
                storage: vec![sentinel(0xa), sentinel(0xb)],
            };
            // re-root onto an unrelated terminator t2
            let t2 = boxed_dict_terminator();
            let new_obj = node_set_terminator(b, &obj, t2);
            // new chain is rooted at t2 with the same attrs/order/values
            assert_eq!((*new_obj.map).terminator(), t2);
            assert_eq!(
                node_read(new_obj.map, &new_obj, wn("a"), DICT),
                Some(sentinel(0xa))
            );
            assert_eq!(
                node_read(new_obj.map, &new_obj, wn("b"), DICT),
                Some(sentinel(0xb))
            );
            assert_eq!(new_obj.storage.len(), 2);
        }
    }

    #[test]
    fn set_terminator_from_devolved_root_targets_devolved_pair() {
        unsafe {
            // obj's map root is t1's devolved terminator (a devolved instance).
            let t1 = new_dict_terminator(std::ptr::null_mut());
            let dev1 = (*t1).as_terminator().devolved_dict_terminator.get();
            let obj = MockObj {
                map: dev1,
                storage: vec![],
            };
            // re-rooting onto t2 lands on t2's devolved pair, not t2 itself.
            let t2 = new_dict_terminator(std::ptr::null_mut());
            let dev2 = (*t2).as_terminator().devolved_dict_terminator.get();
            let new_obj = node_set_terminator(dev1, &obj, t2);
            assert_eq!(new_obj.map, dev2);
        }
    }

    #[test]
    fn plain_direct_write_then_read_roundtrips() {
        unsafe {
            let (_term, a, b) = build_chain();
            let mut obj = MockObj {
                map: b,
                storage: vec![sentinel(0xa), sentinel(0xb)],
            };
            plain_direct_write(a, &mut obj, sentinel(0x111));
            assert_eq!(node_read(b, &obj, wn("a"), DICT), Some(sentinel(0x111)));
            assert_eq!(node_read(b, &obj, wn("b"), DICT), Some(sentinel(0xb)));
        }
    }

    #[test]
    fn add_attr_via_write_grows_map_and_storage() {
        unsafe {
            // start empty: a DictTerminator, no storage
            let term = boxed_dict_terminator();
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            // write two fresh attributes; each takes the common
            // (number_to_readd == 0) append path
            let m = obj._get_mapdict_map();
            assert!(node_write(m, &mut obj, wn("x"), DICT, sentinel(0x1)));
            let m = obj._get_mapdict_map();
            assert!(node_write(m, &mut obj, wn("y"), DICT, sentinel(0x2)));

            assert_eq!(obj.storage.len(), 2);
            assert_eq!(unsafe { (*obj.map).num_attributes() }, 2);
            let m = obj._get_mapdict_map();
            assert_eq!(node_read(m, &obj, wn("x"), DICT), Some(sentinel(0x1)));
            assert_eq!(node_read(m, &obj, wn("y"), DICT), Some(sentinel(0x2)));
            assert_eq!(node_read(m, &obj, wn("z"), DICT), None);

            // overwrite an existing attribute → direct write, no growth
            let m = obj._get_mapdict_map();
            assert!(node_write(m, &mut obj, wn("x"), DICT, sentinel(0x9)));
            assert_eq!(obj.storage.len(), 2);
            let m = obj._get_mapdict_map();
            assert_eq!(node_read(m, &obj, wn("x"), DICT), Some(sentinel(0x9)));
        }
    }

    #[test]
    fn instance_object_write_grows_map_and_storage() {
        unsafe {
            // The real W_ObjectObject (pyre-object) is the
            // MapdictStorageMixin carrier; exercise its trait impl rather
            // than the MockObj double. `map`/`storage` start null
            // (_mapdict_init_empty), so the map terminator is installed
            // here as the mapdict layer would on first attribute access.
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            let m = obj._get_mapdict_map();
            assert!(node_write(m, obj, wn("x"), DICT, sentinel(0x1)));
            let m = obj._get_mapdict_map();
            assert!(node_write(m, obj, wn("y"), DICT, sentinel(0x2)));

            assert_eq!(obj._mapdict_storage_length(), 2);
            assert_eq!((*obj._get_mapdict_map()).num_attributes(), 2);
            let m = obj._get_mapdict_map();
            assert_eq!(node_read(m, obj, wn("x"), DICT), Some(sentinel(0x1)));
            assert_eq!(node_read(m, obj, wn("y"), DICT), Some(sentinel(0x2)));
            assert_eq!(node_read(m, obj, wn("z"), DICT), None);

            // overwrite an existing attribute → direct write, no growth
            let m = obj._get_mapdict_map();
            assert!(node_write(m, obj, wn("x"), DICT, sentinel(0x9)));
            assert_eq!(obj._mapdict_storage_length(), 2);
            let m = obj._get_mapdict_map();
            assert_eq!(node_read(m, obj, wn("x"), DICT), Some(sentinel(0x9)));
        }
    }

    #[test]
    fn instance_node_get_set_roundtrip() {
        unsafe {
            // Pre-install the terminator so `ensure_mapdict_initialized` is a
            // no-op (no real W_TypeObject needed) and exercise the get/set
            // wrappers the attribute read/write paths call.
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            assert!(instance_node_setdictvalue(obj_ref, wn("x"), sentinel(0x11)));
            assert!(instance_node_setdictvalue(obj_ref, wn("y"), sentinel(0x22)));
            assert_eq!(
                instance_node_getdictvalue(obj_ref, wn("x")),
                Some(sentinel(0x11))
            );
            assert_eq!(
                instance_node_getdictvalue(obj_ref, wn("y")),
                Some(sentinel(0x22))
            );
            assert_eq!(instance_node_getdictvalue(obj_ref, wn("z")), None);

            // overwrite an existing attribute
            assert!(instance_node_setdictvalue(obj_ref, wn("x"), sentinel(0x99)));
            assert_eq!(
                instance_node_getdictvalue(obj_ref, wn("x")),
                Some(sentinel(0x99))
            );
        }
    }

    #[test]
    fn instance_node_surrogate_name_stays_a_mapdict_node() {
        unsafe {
            // A lone-surrogate attribute name addresses a real mapdict node,
            // keyed by its full WTF-8 name, rather than degrading the instance
            // dict to the object strategy. mapdict.py:1157-1211 routes any str
            // key — surrogate-bearing included — through getdictvalue.
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            // A '\udc81' lone surrogate, interleaved with a plain-named attr.
            let mut sur = Wtf8Buf::new();
            sur.push(rustpython_wtf8::CodePoint::from_u32(0xDC81).unwrap());

            assert!(instance_node_setdictvalue(obj_ref, &sur, sentinel(0x55)));
            assert!(instance_node_setdictvalue(
                obj_ref,
                wn("ascii"),
                sentinel(0x66)
            ));

            // Both round-trip; the surrogate attribute is a node, so the dict
            // length and the materialised key set include it.
            assert_eq!(
                instance_node_getdictvalue(obj_ref, &sur),
                Some(sentinel(0x55))
            );
            assert_eq!(
                instance_node_getdictvalue(obj_ref, wn("ascii")),
                Some(sentinel(0x66))
            );
            assert_eq!(instance_node_dict_length(obj_ref), 2);

            // Key materialisation rebuilds the surrogate name through
            // w_str_from_wtf8 without panicking on the lone surrogate.
            let keys = instance_node_dict_keys(obj_ref);
            assert_eq!(keys.len(), 2);
            assert_eq!(pyre_object::w_str_get_wtf8(keys[0]), &*sur);

            // Deleting the surrogate node leaves the plain attribute intact.
            assert!(instance_node_deldictvalue(obj_ref, &sur));
            assert_eq!(instance_node_getdictvalue(obj_ref, &sur), None);
            assert_eq!(
                instance_node_getdictvalue(obj_ref, wn("ascii")),
                Some(sentinel(0x66))
            );
            assert_eq!(instance_node_dict_length(obj_ref), 1);
        }
    }

    #[test]
    fn node_delete_rebuilds_without_target() {
        unsafe {
            let term = boxed_dict_terminator();
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            for (name, v) in [("a", 0xa), ("b", 0xb), ("c", 0xc)] {
                let m = obj._get_mapdict_map();
                assert!(node_write(m, &mut obj, wn(name), DICT, sentinel(v)));
            }
            assert_eq!(obj._mapdict_storage_length(), 3);

            // delete the middle attribute and transplant the rebuilt carrier
            // (mapdict.py:852-857 deldictvalue).
            let m = obj._get_mapdict_map();
            let new_obj = node_delete(m, &obj, wn("b"), DICT).expect("b present");
            obj._set_mapdict_storage_and_map(new_obj.storage, new_obj.map);

            assert_eq!(obj._mapdict_storage_length(), 2);
            let m = obj._get_mapdict_map();
            assert_eq!(node_read(m, &obj, wn("a"), DICT), Some(sentinel(0xa)));
            assert_eq!(node_read(m, &obj, wn("b"), DICT), None);
            assert_eq!(node_read(m, &obj, wn("c"), DICT), Some(sentinel(0xc)));

            // deleting an absent attribute returns None
            let m = obj._get_mapdict_map();
            assert!(node_delete(m, &obj, wn("zzz"), DICT).is_none());
        }
    }

    #[test]
    fn instance_node_del_roundtrip() {
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            assert!(instance_node_setdictvalue(obj_ref, wn("x"), sentinel(0x1)));
            assert!(instance_node_setdictvalue(obj_ref, wn("y"), sentinel(0x2)));
            assert!(instance_node_deldictvalue(obj_ref, wn("x")));
            assert_eq!(instance_node_getdictvalue(obj_ref, wn("x")), None);
            assert_eq!(
                instance_node_getdictvalue(obj_ref, wn("y")),
                Some(sentinel(0x2))
            );
            // deleting again reports the attribute is gone
            assert!(!instance_node_deldictvalue(obj_ref, wn("x")));
        }
    }

    #[test]
    fn map_dict_strategy_routes_through_instance() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        unsafe {
            // Back the strategy with a real instance whose terminator is
            // pre-installed (ensure_mapdict_initialized is then a no-op, no
            // W_TypeObject needed). mapdict.py:1502 erases the instance as the
            // dict's dstorage.
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);
            assert!(instance_node_setdictvalue(obj_ref, wn("x"), sentinel(0x11)));
            assert!(instance_node_setdictvalue(obj_ref, wn("y"), sentinel(0x22)));

            let w_dict = pyre_object::w_dict_new_with(&MAP_DICT_STRATEGY_REF, obj_ref as *mut u8);

            assert_eq!(MAP_DICT_STRATEGY.strategy_kind(), StrategyKind::Map);
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 2);

            // getitem_str + getitem(text key) both reach the map; a missing key
            // and the never-equal short-circuit return None.
            assert_eq!(
                MAP_DICT_STRATEGY.getitem_str(w_dict, "x"),
                Some(sentinel(0x11))
            );
            let w_key_y = pyre_object::w_str_new("y");
            assert_eq!(
                MAP_DICT_STRATEGY.getitem(w_dict, w_key_y),
                Some(sentinel(0x22))
            );
            let w_key_z = pyre_object::w_str_new("z");
            assert_eq!(MAP_DICT_STRATEGY.getitem(w_dict, w_key_z), None);

            // keys / items / values in insertion order.
            let keys = MAP_DICT_STRATEGY.w_keys(w_dict);
            assert_eq!(keys.len(), 2);
            assert_eq!(pyre_object::w_str_get_value(keys[0]), "x");
            assert_eq!(pyre_object::w_str_get_value(keys[1]), "y");
            let items = MAP_DICT_STRATEGY.items(w_dict);
            assert_eq!(items.len(), 2);
            assert_eq!(pyre_object::w_str_get_value(items[0].0), "x");
            assert_eq!(items[0].1, sentinel(0x11));
            assert_eq!(pyre_object::w_str_get_value(items[1].0), "y");
            assert_eq!(items[1].1, sentinel(0x22));
            assert_eq!(
                MAP_DICT_STRATEGY.values(w_dict),
                vec![sentinel(0x11), sentinel(0x22)]
            );

            // setitem_str grows; delitem(text) shrinks the backing instance.
            MAP_DICT_STRATEGY.setitem_str(w_dict, "z", sentinel(0x33));
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 3);
            let w_key_x = pyre_object::w_str_new("x");
            assert!(MAP_DICT_STRATEGY.delitem(w_dict, w_key_x));
            assert_eq!(MAP_DICT_STRATEGY.getitem_str(w_dict, "x"), None);
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 2);

            // clear drops every DICT entry.
            MAP_DICT_STRATEGY.clear(w_dict);
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 0);
            assert_eq!(MAP_DICT_STRATEGY.getitem_str(w_dict, "y"), None);
        }
    }

    #[test]
    fn map_dict_strategy_popitem_preserves_surrogate_node_name() {
        use pyre_object::dictmultiobject::DictStrategy;

        unsafe {
            // Install the surrogate-named attribute last so the DICT search
            // reaches it first. The mapdict node name is WTF-8; popitem must
            // preserve that name when materialising the returned key.
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            let mut sur = Wtf8Buf::new();
            sur.push(rustpython_wtf8::CodePoint::from_u32(0xD800).unwrap());

            assert!(instance_node_setdictvalue(
                obj_ref,
                wn("ascii"),
                sentinel(0x44)
            ));
            assert!(instance_node_setdictvalue(obj_ref, &sur, sentinel(0x55)));

            let w_dict = pyre_object::w_dict_new_with(&MAP_DICT_STRATEGY_REF, obj_ref as *mut u8);
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 2);

            let (w_sur_key, w_sur_value) = MAP_DICT_STRATEGY.popitem(w_dict).unwrap();
            assert_eq!(pyre_object::w_str_get_wtf8(w_sur_key), &*sur);
            assert_eq!(w_sur_value, sentinel(0x55));
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 1);
            assert_eq!(MAP_DICT_STRATEGY.getitem(w_dict, w_sur_key), None);
            assert_eq!(instance_node_getdictvalue(obj_ref, &sur), None);

            let (w_ascii_key, w_ascii_value) = MAP_DICT_STRATEGY.popitem(w_dict).unwrap();
            assert_eq!(pyre_object::w_str_get_value(w_ascii_key), "ascii");
            assert_eq!(w_ascii_value, sentinel(0x44));
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 0);
            assert_eq!(MAP_DICT_STRATEGY.getitem_str(w_dict, "ascii"), None);
            assert_eq!(instance_node_getdictvalue(obj_ref, wn("ascii")), None);
        }
    }

    #[test]
    fn make_instance_dict_uses_shared_fake_carriers() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};

        crate::test_hooks::install_hash_hook();
        unsafe {
            let first = make_instance_dict();
            let second = make_instance_dict();
            assert_eq!(
                (*(first as *const pyre_object::W_DictObject))
                    .dstrategy
                    .strategy_kind(),
                StrategyKind::Map
            );
            assert_eq!(MAP_DICT_STRATEGY.length(first), 0);

            // `Object.getdict` stores the wrapper in its SPECIAL slot, so a
            // second lookup must preserve wrapper identity.
            let first_carrier =
                (*(first as *const pyre_object::W_DictObject)).dstorage as PyObjectRef;
            assert_eq!(_obj_getdict(first_carrier), first);
            // The carrier is allocated with `w_class = PY_NULL`, so it reaches
            // `maybe_register_user_finalizer`. Its type resolves through the
            // `INSTANCE_TYPE` fallback, whose `hasuserdel` is false — but the
            // app-level type registry is not initialized in a bare lib test,
            // so `r#type` answers None here and the check only runs once a
            // registry exists.
            if let Some(carrier_type) = crate::typedef::r#type(first_carrier) {
                assert!(
                    !pyre_object::w_type_get_hasuserdel(carrier_type.as_ptr()),
                    "the fake carrier must not enter the user-finalizer queue"
                );
            }

            // The shared terminator keeps mapdict.py:310's `allow_unboxing`
            // default, so every write type-inspects its value. `sentinel` is a
            // bare integer cast to a pointer and would be dereferenced here;
            // this test uses real objects throughout for that reason.
            let first_shared = pyre_object::w_str_new("first");
            let second_shared = pyre_object::w_str_new("second");
            pyre_object::w_dict_setitem_str(first, "shared", first_shared);
            pyre_object::w_dict_setitem_str(second, "shared", second_shared);
            assert_eq!(
                pyre_object::w_dict_getitem_str(first, "shared"),
                Some(first_shared)
            );
            assert_eq!(
                pyre_object::w_dict_getitem_str(second, "shared"),
                Some(second_shared)
            );
            let first_node = node_search(
                (&*(first_carrier as *const pyre_object::W_ObjectObject))._get_mapdict_map(),
                DICT,
            );
            let second_carrier =
                (*(second as *const pyre_object::W_DictObject)).dstorage as PyObjectRef;
            let second_node = node_search(
                (&*(second_carrier as *const pyre_object::W_ObjectObject))._get_mapdict_map(),
                DICT,
            );
            assert_eq!(first_node, second_node);

            // The process-shared terminator preserves mapdict.py:310's true
            // allow_unboxing default, so an int takes the unboxed transition.
            pyre_object::w_dict_setitem_str(first, "number", pyre_object::w_int_new(42));
            let number_node = node_search(
                (&*(first_carrier as *const pyre_object::W_ObjectObject))._get_mapdict_map(),
                DICT,
            )
            .unwrap();
            assert!(
                (*number_node).as_plain().unboxed.is_some(),
                "int attribute did not use the shared terminator's unboxed transition"
            );

            // A non-str write devolves while retaining previous string keys.
            MAP_DICT_STRATEGY.setitem(
                first,
                pyre_object::w_int_new(7),
                pyre_object::w_str_new("v"),
            );
            assert_eq!(
                (*(first as *const pyre_object::W_DictObject))
                    .dstrategy
                    .strategy_kind(),
                StrategyKind::Object
            );
            assert_eq!(
                pyre_object::w_dict_getitem_str(first, "shared"),
                Some(first_shared)
            );

            // A separate carrier drives the LIMIT arm, which reads its wrapper
            // back through getdict before changing to a Unicode dictionary.
            let limit_dict = make_instance_dict();
            for i in 0..LIMIT_MAP_ATTRIBUTES {
                pyre_object::w_dict_setitem_str(
                    limit_dict,
                    &format!("k{i}"),
                    pyre_object::w_str_new(&format!("v{i}")),
                );
            }
            assert_eq!(
                (*(limit_dict as *const pyre_object::W_DictObject))
                    .dstrategy
                    .strategy_kind(),
                StrategyKind::Unicode
            );

            // A new carrier devolves on a non-str key; reads and deletes then
            // use the Devolved read and delete arms' fake-carrier getdict.
            let devolved_dict = make_instance_dict();
            let kept = pyre_object::w_str_new("kept-value");
            pyre_object::w_dict_setitem_str(devolved_dict, "kept", kept);
            MAP_DICT_STRATEGY.setitem(
                devolved_dict,
                pyre_object::w_int_new(8),
                pyre_object::w_str_new("devolve"),
            );
            assert_eq!(
                pyre_object::w_dict_getitem_str(devolved_dict, "kept"),
                Some(kept)
            );
            assert!(
                (*(devolved_dict as *const pyre_object::W_DictObject))
                    .dstrategy
                    .delitem(devolved_dict, pyre_object::w_str_new("kept"))
            );
        }
    }

    #[test]
    fn map_dict_strategy_switch_to_object_materialises() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);
            assert!(instance_node_setdictvalue(obj_ref, wn("x"), sentinel(0x11)));
            assert!(instance_node_setdictvalue(obj_ref, wn("y"), sentinel(0x22)));

            let w_dict = pyre_object::w_dict_new_with(&MAP_DICT_STRATEGY_REF, obj_ref as *mut u8);
            assert_eq!(MAP_DICT_STRATEGY.length(w_dict), 2);

            // A non-str key forces switch_to_object_strategy → materialise.
            MAP_DICT_STRATEGY.switch_to_object_strategy(w_dict);

            // The dict is now ObjectDictStrategy and holds the two str attrs.
            let dict = &*(w_dict as *const pyre_object::W_DictObject);
            assert_eq!(dict.dstrategy.strategy_kind(), StrategyKind::Object);
            assert_eq!(dict.dstrategy.length(w_dict), 2);
            assert_eq!(
                pyre_object::w_dict_getitem_str(w_dict, "x"),
                Some(sentinel(0x11))
            );
            assert_eq!(
                pyre_object::w_dict_getitem_str(w_dict, "y"),
                Some(sentinel(0x22))
            );

            // The backing instance devolved: its map roots at a
            // DevolvedDictTerminator and the DICT attrs left its storage.
            let inst_map = obj._get_mapdict_map();
            assert!(matches!(
                (*inst_map).as_terminator().kind,
                TerminatorKind::Devolved
            ));
            assert_eq!((*inst_map).storage_needed(), 0);
        }
    }

    #[test]
    fn map_dict_strategy_switch_to_text_materialises() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);
            assert!(instance_node_setdictvalue(obj_ref, wn("a"), sentinel(0x55)));

            let w_dict = pyre_object::w_dict_new_with(&MAP_DICT_STRATEGY_REF, obj_ref as *mut u8);

            // The LIMIT-devolve path (mapdict.py:317-323) switches to text.
            mapdict_switch_to_text_strategy(w_dict);

            let dict = &*(w_dict as *const pyre_object::W_DictObject);
            assert_eq!(dict.dstrategy.strategy_kind(), StrategyKind::Unicode);
            assert_eq!(
                pyre_object::w_dict_getitem_str(w_dict, "a"),
                Some(sentinel(0x55))
            );
            let inst_map = obj._get_mapdict_map();
            assert!(matches!(
                (*inst_map).as_terminator().kind,
                TerminatorKind::Devolved
            ));
        }
    }

    #[test]
    fn instance_custom_trace_walks_storage_without_instance_dict() {
        // An instance's attribute values are forwarded by the per-instance
        // custom trace worker (`instance_walk_boxed_storage`), independent of
        // whether its `__dict__` wrapper was ever materialised in INSTANCE_DICT.
        // The low-level `instance_node_setdictvalue` writes the attributes
        // through map+storage WITHOUT calling `getdict`, so no INSTANCE_DICT
        // entry exists — yet the storage walk still visits the value slots.
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            let v1 = sentinel(0xA1);
            let v2 = sentinel(0xB2);
            assert!(instance_node_setdictvalue(obj_ref, wn("x"), v1));
            assert!(instance_node_setdictvalue(obj_ref, wn("y"), v2));

            let addr = obj_ref as usize;
            // Never entered INSTANCE_DICT (no getdict call), proving storage
            // forwarding is decoupled from wrapper materialisation.
            let in_instance_dict = INSTANCE_DICT.lock().unwrap().contains_key(&addr);
            assert!(!in_instance_dict);

            let mut seen: Vec<PyObjectRef> = Vec::new();
            instance_walk_boxed_storage(obj_ref, &mut |slot| seen.push(*slot));
            assert!(seen.contains(&v1), "x value not walked by custom trace");
            assert!(seen.contains(&v2), "y value not walked by custom trace");
        }
    }

    #[test]
    fn slots_only_native_subclass_has_mapdict_layout_without_routing() {
        unsafe {
            let w_class =
                pyre_object::w_type_new("SlotsOnlyStr", pyre_object::PY_NULL, std::ptr::null_mut());
            assert!(!pyre_object::w_type_get_hasdict(w_class));
            let obj = pyre_object::w_str_subclass_from_wtf8(wb("abc"), w_class);

            // Auto-assigned type ids are registered by the JIT driver. Unit
            // tests run without it, so stamp the declared user-layout id while
            // preserving the allocation header's flags.
            let header = &mut *majit_gc::header::header_of(obj as usize);
            header.tid_and_flags = (header.tid_and_flags & !majit_gc::header::TYPE_ID_MASK)
                | u64::from(pyre_object::unicodeobject::W_UNICODE_USER_GC_TYPE_ID.get());

            assert!(has_mapdict_layout(obj));
            assert!(!has_mapdict_storage(obj));
            instance_walk_boxed_storage(obj, &mut |_| {});
        }
    }

    #[test]
    fn instance_dict_wrapper_in_special_slot_not_instance_dict() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        // An instance's `__dict__` wrapper is stored in the
        // mapdict "dict" SPECIAL slot (mapdict.py:826-840 _obj_getdict), not in
        // the INSTANCE_DICT side table. Repeated access returns the same wrapper,
        // and the SPECIAL slot is excluded from the `__dict__` view.
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            let w1 = _obj_getdict(obj_ref);
            // stored in the SPECIAL slot, not INSTANCE_DICT.
            assert_eq!(instance_get_dict_slot(obj_ref), Some(w1));
            let addr = obj_ref as usize;
            let in_instance_dict = INSTANCE_DICT.lock().unwrap().contains_key(&addr);
            assert!(!in_instance_dict);
            // identity stable across repeated access.
            let w2 = _obj_getdict(obj_ref);
            assert_eq!(w1, w2);
            // a fresh wrapper is a MapDictStrategy view.
            let dict = &*(w1 as *const pyre_object::W_DictObject);
            assert_eq!(dict.dstrategy.strategy_kind(), StrategyKind::Map);

            // a DICT attribute is visible in the view; the SPECIAL "dict" slot
            // (storing the wrapper) is excluded from the view.
            assert!(instance_node_setdictvalue(obj_ref, wn("x"), sentinel(0x1)));
            assert_eq!(MAP_DICT_STRATEGY.length(w1), 1);
            assert_eq!(MAP_DICT_STRATEGY.getitem_str(w1, "x"), Some(sentinel(0x1)));
            assert_eq!(MAP_DICT_STRATEGY.getitem_str(w1, "dict"), None);
        }
    }

    #[test]
    fn obj_getdict_rebuilds_the_view_when_the_dict_slot_reads_null() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        // mapdict.py:828-830 `if w_dict is not None` — RPython's `read` answers
        // None both for an absent slot and for one holding None, and both mean
        // "build the view".  Surfacing the null instead makes `getdict` report
        // that the receiver has no dict, which is the single state
        // `descr__setattr__` turns into
        // `"'%T' object attribute '%s' is read-only"` — for any name the type
        // carries, a class-variable default being enough to arm that branch.
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            let w1 = _obj_getdict(obj_ref);
            assert!(!w1.is_null());
            assert_eq!(instance_get_dict_slot(obj_ref), Some(w1));

            // The state a write that never landed leaves behind: the map still
            // claims the SPECIAL attribute, its storage slot holds NULL.
            assert!(instance_set_dict_slot(obj_ref, pyre_object::PY_NULL));
            assert_eq!(instance_get_dict_slot(obj_ref), Some(pyre_object::PY_NULL));

            let w2 = _obj_getdict(obj_ref);
            assert!(!w2.is_null(), "a null SPECIAL slot must rebuild the view");
            assert_eq!(instance_get_dict_slot(obj_ref), Some(w2));
            let dict = &*(w2 as *const pyre_object::W_DictObject);
            assert_eq!(dict.dstrategy.strategy_kind(), StrategyKind::Map);
            // The rebuilt view backs onto the instance it was built for.
            assert_eq!(dict.dstorage as PyObjectRef, obj_ref);
        }
    }

    #[test]
    fn instance_custom_trace_walks_special_wrapper_and_values() {
        // The wrapper stored in the "dict" SPECIAL slot is forwarded by the
        // instance custom trace (it is one of the storage slots), and a
        // non-devolved view's DICT values are forwarded directly from storage.
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            let w_dict = _obj_getdict(obj_ref);
            let v1 = sentinel(0xC1);
            let v2 = sentinel(0xC2);
            assert!(instance_node_setdictvalue(obj_ref, wn("x"), v1));
            assert!(instance_node_setdictvalue(obj_ref, wn("y"), v2));

            let mut seen: Vec<PyObjectRef> = Vec::new();
            instance_walk_boxed_storage(obj_ref, &mut |slot| seen.push(*slot));
            assert!(seen.contains(&w_dict), "SPECIAL-slot wrapper not walked");
            assert!(seen.contains(&v1), "x value not walked");
            assert!(seen.contains(&v2), "y value not walked");
        }
    }

    #[test]
    fn instance_custom_trace_and_wrapper_cover_devolved_values() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        // UAF-prevention case: once an instance devolves (>= LIMIT DICT attrs,
        // mapdict.py:316-323), its materialised DICT values move into the
        // wrapper's own backing storage and leave the instance storage. The
        // wrapper stays in the "dict" SPECIAL slot. Coverage is now two-layer:
        // the instance custom trace forwards the wrapper pointer, and the
        // wrapper's own GC custom trace (`strategy.walk_gc_refs`) forwards its
        // entry values — together covering every devolved value, with no
        // INSTANCE_REGISTRY walk.
        crate::test_hooks::install_hash_hook();
        unsafe {
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            for i in 0..LIMIT_MAP_ATTRIBUTES {
                let name = format!("k{i}");
                assert!(instance_node_setdictvalue(
                    obj_ref,
                    wn(&name),
                    sentinel(0x2000 + i)
                ));
            }
            // The LIMIT write devolved the instance via obj.getdict(); the
            // wrapper now lives in the SPECIAL slot with a non-Map strategy.
            let w_dict = instance_get_dict_slot(obj_ref).expect("wrapper in SPECIAL slot");
            let dict = &*(w_dict as *const pyre_object::W_DictObject);
            assert_ne!(dict.dstrategy.strategy_kind(), StrategyKind::Map);

            // (1) the instance custom trace forwards the wrapper pointer.
            let mut storage_seen: Vec<PyObjectRef> = Vec::new();
            instance_walk_boxed_storage(obj_ref, &mut |slot| storage_seen.push(*slot));
            assert!(
                storage_seen.contains(&w_dict),
                "devolved wrapper not walked by instance custom trace"
            );

            // (2) the wrapper's own custom trace forwards every entry value.
            let mut wrapper_seen: Vec<PyObjectRef> = Vec::new();
            dict.dstrategy
                .walk_gc_refs(w_dict, &mut |slot| wrapper_seen.push(*slot));
            for i in 0..LIMIT_MAP_ATTRIBUTES {
                assert!(
                    wrapper_seen.contains(&sentinel(0x2000 + i)),
                    "devolved value k{i} not walked by wrapper custom trace"
                );
            }
        }
    }

    #[test]
    fn write_terminator_devolves_at_limit_map_attributes() {
        use pyre_object::dictmultiobject::{DictStrategy, StrategyKind};
        crate::test_hooks::install_hash_hook();
        unsafe {
            // mapdict.py:316-323: the (LIMIT_MAP_ATTRIBUTES)th DICT write on a
            // non-devolved instance auto-devolves its `__dict__` to text strategy.
            let term = boxed_dict_terminator();
            let obj_ref = pyre_object::w_instance_new(pyre_object::PY_NULL);
            let obj = &mut *(obj_ref as *mut pyre_object::W_ObjectObject);
            obj._set_mapdict_map(term);

            for i in 0..LIMIT_MAP_ATTRIBUTES {
                let name = format!("k{i}");
                assert!(instance_node_setdictvalue(
                    obj_ref,
                    wn(&name),
                    sentinel(0x1000 + i)
                ));
            }

            // _write_terminator's LIMIT branch fetched obj.getdict() (the
            // MapDictStrategy view installed by _obj_getdict) and devolved it.
            let w_dict = _obj_getdict(obj_ref);
            let dict = &*(w_dict as *const pyre_object::W_DictObject);
            assert_eq!(dict.dstrategy.strategy_kind(), StrategyKind::Unicode);

            // every attribute survived the devolve, read through the dict view.
            for i in 0..LIMIT_MAP_ATTRIBUTES {
                let name = format!("k{i}");
                assert_eq!(
                    pyre_object::w_dict_getitem_str(w_dict, &name),
                    Some(sentinel(0x1000 + i))
                );
            }

            // The backing instance devolved: its map roots at a
            // DevolvedDictTerminator. The "dict" SPECIAL slot (the wrapper,
            // written by obj.getdict() during the LIMIT devolve) is kept on the
            // rebuilt carrier (mapdict.py:362-372 keeps non-DICT attrs), so the
            // outermost node is that PlainAttribute and the terminator is reached
            // via `.terminator()`. Only the SPECIAL slot survives on storage.
            let inst_map = obj._get_mapdict_map();
            let inst_term = (*inst_map).terminator();
            assert!(matches!(
                (*inst_term).as_terminator().kind,
                TerminatorKind::Devolved
            ));
            assert_eq!((*inst_map).storage_needed(), 1);
            assert_eq!(instance_get_dict_slot(obj_ref), Some(w_dict));
        }
    }

    #[test]
    fn add_attr_interns_shared_transition() {
        unsafe {
            // two independent objects adding the same attribute from the same
            // map must converge on the same child map (interned transition)
            let term = boxed_dict_terminator();
            let mut o1 = MockObj {
                map: term,
                storage: vec![],
            };
            let mut o2 = MockObj {
                map: term,
                storage: vec![],
            };
            let m = o1._get_mapdict_map();
            node_write(m, &mut o1, wn("p"), DICT, sentinel(1));
            let m = o2._get_mapdict_map();
            node_write(m, &mut o2, wn("p"), DICT, sentinel(2));
            assert_eq!(o1.map, o2.map);
        }
    }

    #[test]
    fn out_of_order_insert_reorders_to_canonical_map() {
        unsafe {
            let term = boxed_dict_terminator();
            // o1 establishes the canonical insertion order a, b
            let mut o1 = MockObj {
                map: term,
                storage: vec![],
            };
            let m = o1._get_mapdict_map();
            node_write(m, &mut o1, wn("a"), DICT, sentinel(0xa1));
            let m = o1._get_mapdict_map();
            node_write(m, &mut o1, wn("b"), DICT, sentinel(0xb1));

            // o2 inserts b first, then a — adding "a" must trigger
            // _reorder_and_add (a lower-order ancestor already has "a") and
            // converge on the same canonical map as o1
            let mut o2 = MockObj {
                map: term,
                storage: vec![],
            };
            let m = o2._get_mapdict_map();
            node_write(m, &mut o2, wn("b"), DICT, sentinel(0xb2));
            let m = o2._get_mapdict_map();
            node_write(m, &mut o2, wn("a"), DICT, sentinel(0xa2));

            // values preserved through the reorder
            let m = o2._get_mapdict_map();
            assert_eq!(node_read(m, &o2, wn("a"), DICT), Some(sentinel(0xa2)));
            assert_eq!(node_read(m, &o2, wn("b"), DICT), Some(sentinel(0xb2)));
            assert_eq!(o2.storage.len(), 2);
            // reordered to the canonical (insertion-ordered) map shared with o1
            assert_eq!(o2.map, o1.map);
        }
    }

    // ── UnboxedPlainAttribute (mapdict.py:532-665) ────────────────────

    #[test]
    fn unboxed_int_attribute_stores_and_reads() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            // an int written to an unboxing-allowed map becomes an
            // UnboxedPlainAttribute
            assert!(node_write(
                m,
                &mut obj,
                wn("x"),
                DICT,
                pyre_object::w_int_new(42)
            ));
            let p = (*obj.map).as_plain();
            assert!(p.unboxed.is_some());
            assert!(p.unboxed.as_ref().unwrap().firstunwrapped);
            assert_eq!(p.unboxed.as_ref().unwrap().typ, UnboxType::Int);
            // a single storage slot holds the erased longlong list (not a box)
            assert_eq!(obj.storage.len(), 1);
            // reading boxes the longlong back into an int of the same value
            let m = obj._get_mapdict_map();
            let r = node_read(m, &obj, wn("x"), DICT).unwrap();
            assert!(pyre_object::is_int(r));
            assert_eq!(pyre_object::w_int_get_value(r), 42);
        }
    }

    #[test]
    fn two_unboxed_ints_share_one_storage_slot() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("x"), DICT, pyre_object::w_int_new(10));
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("y"), DICT, pyre_object::w_int_new(20));
            // both pack into a single shared longlong list (storageindex 0,
            // listindex 0 and 1) — storage does not grow for the second one
            assert_eq!(obj.storage.len(), 1);
            let p = (*obj.map).as_plain();
            assert_eq!(p.storageindex, 0);
            assert_eq!(p.unboxed.as_ref().unwrap().listindex, 1);
            let m = obj._get_mapdict_map();
            assert_eq!(
                pyre_object::w_int_get_value(node_read(m, &obj, wn("x"), DICT).unwrap()),
                10
            );
            assert_eq!(
                pyre_object::w_int_get_value(node_read(m, &obj, wn("y"), DICT).unwrap()),
                20
            );
        }
    }

    #[test]
    fn unboxed_int_and_float_share_slot_with_correct_boxing() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("i"), DICT, pyre_object::w_int_new(7));
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("f"), DICT, pyre_object::w_float_new(2.5));
            // int and float pack into one longlong list (the float as its bits)
            assert_eq!(obj.storage.len(), 1);
            let m = obj._get_mapdict_map();
            let ri = node_read(m, &obj, wn("i"), DICT).unwrap();
            let rf = node_read(m, &obj, wn("f"), DICT).unwrap();
            // each is re-boxed to its own type
            assert!(pyre_object::is_int(ri));
            assert_eq!(pyre_object::w_int_get_value(ri), 7);
            assert!(pyre_object::is_float(rf));
            assert_eq!(pyre_object::w_float_get_value(rf), 2.5);
        }
    }

    #[test]
    fn unboxed_overwrite_same_type_updates_in_place() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("x"), DICT, pyre_object::w_int_new(1));
            let map_after_first = obj.map;
            // a same-typed overwrite updates the longlong in place: no map
            // transition, no storage growth
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("x"), DICT, pyre_object::w_int_new(2));
            assert_eq!(obj.map, map_after_first);
            assert_eq!(obj.storage.len(), 1);
            let m = obj._get_mapdict_map();
            assert_eq!(
                pyre_object::w_int_get_value(node_read(m, &obj, wn("x"), DICT).unwrap()),
                2
            );
        }
    }

    #[test]
    fn unboxed_float_preserves_bits() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let v = -3.25_f64;
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("f"), DICT, pyre_object::w_float_new(v));
            let m = obj._get_mapdict_map();
            let r = node_read(m, &obj, wn("f"), DICT).unwrap();
            assert!(pyre_object::is_float(r));
            assert_eq!(pyre_object::w_float_get_value(r), v);
        }
    }

    #[test]
    fn read_migrates_to_boxed_when_unboxing_frozen() {
        unsafe {
            // "x" is stored unboxed under a (default unboxing-on) terminator.
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, wn("x"), DICT, pyre_object::w_int_new(10));
            assert!((*obj.map).as_plain().unboxed.is_some());
            // the class becomes type-unstable: freeze unboxing for its terminator.
            (*term).as_terminator().set_allow_unboxing(false);
            // mapdict.py:592-598 — a read now lazily migrates obj to boxed storage.
            let m = obj._get_mapdict_map();
            maybe_migrate_to_boxed(m, &mut obj, wn("x"), DICT);
            // the rebuilt map's attribute is boxed; the value is preserved.
            assert!((*obj.map).as_plain().unboxed.is_none());
            let m = obj._get_mapdict_map();
            assert_eq!(
                pyre_object::w_int_get_value(node_read(m, &obj, wn("x"), DICT).unwrap()),
                10
            );
        }
    }

    // The side-table root bookkeeping below takes its table and pending set as
    // arguments, so these exercise the real functions on local tables — the
    // process-global `INSTANCE_DICT` / `WEAKREF_TABLE` are shared with every
    // other test running concurrently and must not be touched here.
    fn table(entries: &[(usize, usize)]) -> Mutex<HashMap<usize, usize>> {
        Mutex::new(entries.iter().copied().collect())
    }
    fn pending(keys: &[usize]) -> Mutex<HashSet<usize>> {
        Mutex::new(keys.iter().copied().collect())
    }
    fn keys_of(snapshot: &[(usize, PyObjectRef)]) -> Vec<usize> {
        let mut keys: Vec<usize> = snapshot.iter().map(|&(key, _)| key).collect();
        keys.sort_unstable();
        keys
    }

    #[test]
    fn major_snapshot_takes_the_whole_table_and_keeps_the_pending_set() {
        let table = table(&[(0x10, 0xA0), (0x20, 0xB0), (0x30, 0xC0)]);
        let pending = pending(&[0x20]);
        let snapshot = snapshot_root_entries(&table, &pending, false);
        assert_eq!(keys_of(&snapshot), [0x10, 0x20, 0x30]);
        // A major moves nothing, so it must not retire the keys a later minor
        // still owes a visit to.
        assert_eq!(pending.lock().unwrap().len(), 1);
    }

    #[test]
    fn minor_snapshot_drains_the_pending_set() {
        let table = table(&[(0x10, 0xA0), (0x20, 0xB0), (0x30, 0xC0)]);
        let pending = pending(&[0x20, 0x30]);
        let snapshot = snapshot_root_entries(&table, &pending, true);
        assert_eq!(keys_of(&snapshot), [0x20, 0x30]);
        assert!(pending.lock().unwrap().is_empty());
        // Drained: the entries are old by now and reached through their own
        // write barrier, so the next minor visits nothing.
        assert!(snapshot_root_entries(&table, &pending, true).is_empty());
    }

    #[test]
    fn minor_snapshot_drops_a_pending_key_whose_entry_is_gone() {
        let table = table(&[(0x10, 0xA0)]);
        let pending = pending(&[0x10, 0x99]);
        let snapshot = snapshot_root_entries(&table, &pending, true);
        assert_eq!(keys_of(&snapshot), [0x10]);
    }

    #[test]
    fn rekey_moves_a_promoted_owner_and_repoints_a_moved_value() {
        let table = table(&[(0x10, 0xA0), (0x20, 0xB0)]);
        // 0x10's owner moved to 0x11 and its value to 0xA1; 0x20 stayed put but
        // its value moved.
        apply_root_rekeys(&table, vec![(0x10, 0x11, 0xA1), (0x20, 0x20, 0xB1)]);
        let table = table.lock().unwrap();
        assert_eq!(table.get(&0x10), None);
        assert_eq!(table.get(&0x11), Some(&0xA1));
        assert_eq!(table.get(&0x20), Some(&0xB1));
    }

    #[test]
    fn rekey_of_an_entry_deleted_during_the_walk_reinserts_nothing() {
        let table = table(&[]);
        apply_root_rekeys(&table, vec![(0x10, 0x11, 0xA1), (0x20, 0x20, 0xB1)]);
        assert!(table.lock().unwrap().is_empty());
    }
}
