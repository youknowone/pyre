//! W_DictMultiObject family — Python `dict` types.
//!
//! PyPy equivalent: `pypy/objspace/std/dictmultiobject.py`.
//!
//! Hosts the `W_DictMultiObject` Rust trait + concrete subclasses
//! `W_DictObject` (regular dict, `:313-325`) and `W_ModuleDictObject`
//! (module / globals dict backed by `ModuleDictStrategy` per
//! `pypy/objspace/std/celldict.py:28`).  The cutover lifted
//! pyre's `DictStorage`-based module-dict path out of
//! `pyre-interpreter/executioncontext.rs` and into a sibling
//! `celldict.rs` so the upstream strategy-per-W_Root model holds.
//!
//! Supports arbitrary PyObjectRef keys (int, str, etc.) with
//! equality comparison via `dict_keys_equal` (routed through the
//! `dict_eq_hook::EQ_W_HOOK` trampoline → `baseobjspace::eq_w`).
//!
//! ## Phase C-3 strategy dispatch status
//!
//! All eight W_DictMultiObject accessors route through
//! `w_dict_get_strategy(obj).method(obj, ...)` polymorphic dispatch
//! (PyPy's `w_dict.get_strategy().method(w_dict, ...)` shape):
//!
//! - `w_dict_setitem_str` → `dictmultiobject.py:111-112`
//! - `w_dict_getitem_str` → `:103-105`
//! - `w_dict_clear` → `:148-152` (`descr_clear`)
//! - `w_dict_items` → `:117-121`
//! - `w_dict_store` → `:97-99 setitem`
//! - `w_dict_lookup` → `:93-95 getitem`
//! - `w_dict_delitem` → `:101-102`
//! - `w_dict_len` → `:107-109 length`
//!
//! ## Field-deletion status (TODO queue)
//!
//! - `W_DictObject.dict_storage_proxy` + `W_ModuleDictObject.dict_storage_proxy`:
//!   TODO: back-mirror pointers to a legacy `DictStorage`.
//!   Retirement gated on the `PyFrame.w_globals` migration from
//!   `*mut DictStorage` to `PyObjectRef`.  Once frames
//!   use W_DictObject directly, no back-mirror is needed.
//!
//! - `pyre-interpreter::DictStorage` struct: legacy str-keyed
//!   storage carried by `PyFrame.w_globals`.  Final retirement
//!   depends on the above two items.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// `pypy/objspace/std/dictmultiobject.py:1209-1212 ObjectDictStrategy`
/// — `r_dict(dict_keys_equal, hash_w)` key type.  The hash is cached
/// at insertion time so the `Hash` trait impl is infallible; equality
/// dispatches through `dict_keys_equal` so user-defined `__eq__`
/// resolves through the standard comparison protocol via the
/// `dict_eq_hook::EQ_W_HOOK` trampoline.
///
/// GC-move stability: `ObjectKey.hash` is the value of `space.hash_w(obj)`
/// at insertion time, which depends only on the object's *content* and
/// not on its memory address.  Python's data model requires hashable
/// types to have content-derived, identity-stable hashes
/// (`__hash__` must agree with `__eq__` and must not change for the
/// lifetime of the object); built-in immutables (int / str / tuple /
/// frozenset / bytes / bool) all satisfy this by construction, and
/// user-defined classes that override `__hash__` are contractually
/// required to do so.  Pyre's young-gen mark-and-copy collector may
/// move `key.obj` (the `*mut PyObject` payload) but the cached `hash`
/// remains valid because it is not a function of the address.  The
/// `walk_gc_refs` trace in `dictstrategy.rs ObjectDictStrategy` visits
/// `key.obj` only and leaves `key.hash` untouched.  See
/// `identitydict.rs IdentityKey` for the contrasting *pointer-hashed*
/// case where this property does NOT hold.
#[derive(Clone, Copy)]
pub struct ObjectKey {
    pub hash: i64,
    pub obj: PyObjectRef,
}

impl std::hash::Hash for ObjectKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_i64(self.hash);
    }
}

impl PartialEq for ObjectKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            return false;
        }
        unsafe { dict_keys_equal(self.obj, other.obj) }
    }
}

impl Eq for ObjectKey {}

/// View a key as `&str` only when it is a str whose backing is valid UTF-8.
/// A lone-surrogate str returns `None` so the `&str`-keyed proxy fast paths
/// (which cannot represent it) fall through to the generic object-keyed path.
#[inline]
unsafe fn key_as_utf8(key: PyObjectRef) -> Option<&'static str> {
    if crate::is_str(key) {
        crate::w_str_get_value_opt(key)
    } else {
        None
    }
}

/// `dictmultiobject.py:1209-1212 r_dict(_, hash_w, ...)` —
/// `space.hash_w(obj)` precomputed for bucket placement.  There is a
/// single hashing path: the `dict_eq_hook::HASH_W_HOOK` trampoline,
/// installed at boot before any dict is built (production) or per test
/// thread (`#[cfg(test)]`).  A missing hook is a setup bug, not a
/// recoverable state, so [`missing_hash_hook`](crate::dict_eq_hook::missing_hash_hook)
/// fails loud rather than substituting a divergent structural hash.
#[inline]
pub unsafe fn object_key_for(obj: PyObjectRef) -> ObjectKey {
    let hash = crate::dict_eq_hook::try_hash_w(obj)
        .unwrap_or_else(|| crate::dict_eq_hook::missing_hash_hook());
    if crate::dict_eq_hook::take_hash_error() {
        // Infallible path: swallow the error and use structural hash.
        // Checked callers should use `object_key_for_checked` instead.
    }
    // Clear any stale eq flag so the upcoming bucket probe starts clean;
    // an infallible probe's own error is then swallowed by the next key
    // construction, never leaking into a later checked op.
    crate::dict_eq_hook::take_eq_error();
    ObjectKey { hash, obj }
}

/// Fallible variant of [`object_key_for`].  When the `hash_w` hook
/// signals an error (unhashable type, user `__hash__` raised), this
/// returns `Err(DictKeyError)`.  The caller retrieves the concrete
/// error from the interpreter-side error slot.
#[inline]
pub unsafe fn object_key_for_checked(obj: PyObjectRef) -> Result<ObjectKey, DictKeyError> {
    let hash = crate::dict_eq_hook::try_hash_w(obj)
        .unwrap_or_else(|| crate::dict_eq_hook::missing_hash_hook());
    if crate::dict_eq_hook::take_hash_error() {
        return Err(DictKeyError);
    }
    // Clean slate for the bucket probe that follows in the caller; its
    // eq error is read back via `take_dict_key_error` after the access.
    crate::dict_eq_hook::take_eq_error();
    Ok(ObjectKey { hash, obj })
}

#[inline]
pub unsafe fn hash_key_checked(obj: PyObjectRef) -> Result<(), DictKeyError> {
    let _ = crate::dict_eq_hook::try_hash_w(obj);
    if crate::dict_eq_hook::take_hash_error() {
        return Err(DictKeyError);
    }
    Ok(())
}

#[inline]
fn strategy_is(
    current: &'static dyn crate::dictstrategy::DictStrategy,
    expected: &'static dyn crate::dictstrategy::DictStrategy,
) -> bool {
    current.strategy_kind() == expected.strategy_kind()
}

#[inline]
pub unsafe fn key_compares_by_identity(key: PyObjectRef) -> bool {
    let w_type = (*key).w_class as PyObjectRef;
    !w_type.is_null()
        && matches!(
            crate::dict_eq_hook::try_compares_by_identity(w_type),
            Some(true)
        )
}

#[inline]
unsafe fn _never_equal_to_int(key: PyObjectRef) -> bool {
    crate::is_none(key) || crate::is_bytes(key) || crate::is_str(key)
}

/// Marker error returned by checked dict operations when the key's
/// `__hash__` fails.  The concrete `PyError` is stored in a
/// thread-local on the `pyre-interpreter` side; the caller converts
/// this marker to a real exception via `take_pending_hash_error()`.
#[derive(Debug)]
pub struct DictKeyError;

/// `true` when the most recent checked dict op raised through either key
/// callback of the `r_dict(eq_w, hash_w)` pair: `space.hash_w` at key
/// construction or `space.eq_w` during the bucket probe.  Reading clears
/// both flags.  The concrete `PyError` rides the interpreter-side
/// pending slot, retrieved via `take_pending_hash_error`.
#[inline]
unsafe fn take_dict_key_error() -> bool {
    let hash = crate::dict_eq_hook::take_hash_error();
    let eq = crate::dict_eq_hook::take_eq_error();
    hash || eq
}

/// Test-only structural hash hook.  Walks the same built-in type ladder
/// as `dict_keys_equal` (`:1207-1260`) so equal keys land in the same
/// bucket, giving `#[cfg(test)]` dict tests a single deterministic hash
/// path without reaching into `pyre-interpreter`'s `space.hash_w` (which
/// lives in a crate above this one).  Installed via `register_hash_w_hook`
/// by the test harness; never a production code path — production hashes
/// exclusively through `space.hash_w` (`baseobjspace.py:840-845`).
#[cfg(test)]
unsafe fn builtin_structural_hash(obj: PyObjectRef) -> i64 {
    if obj.is_null() {
        return 0;
    }
    if crate::is_bool(obj) {
        return crate::w_bool_get_value(obj) as i64;
    }
    if crate::is_int(obj) {
        return crate::w_int_get_value(obj);
    }
    if crate::is_str(obj) {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        // Hash the WTF-8 bytes so a lone-surrogate key does not panic in
        // `w_str_get_value`; the byte sequence is the hashed identity.
        crate::w_str_get_wtf8(obj).as_bytes().hash(&mut h);
        return h.finish() as i64;
    }
    if crate::bytesobject::is_bytes(obj) {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        crate::bytesobject::w_bytes_data(obj).hash(&mut h);
        return h.finish() as i64;
    }
    if crate::is_tuple(obj) {
        let n = crate::w_tuple_len(obj);
        let mut acc: i64 = 0x345678;
        for i in 0..n {
            let e = crate::w_tuple_getitem(obj, i as i64).unwrap_or(std::ptr::null_mut());
            acc = acc
                .wrapping_mul(1000003)
                .wrapping_add(builtin_structural_hash(e));
        }
        return acc;
    }
    if crate::is_frozenset(obj) {
        // Order-independent (xor) so frozenset({a,b}) and frozenset({b,a})
        // hash equal, matching `dict_keys_equal`'s set-containment arm.
        let mut acc: i64 = 0;
        for &e in crate::w_set_items(obj).iter() {
            acc ^= builtin_structural_hash(e);
        }
        return acc;
    }
    obj as usize as i64
}

/// `pypy/objspace/std/dictmultiobject.py:45-53 W_DictMultiObject(W_Root)`
/// abstract base — both `W_DictObject` (regular dicts) and
/// `W_ModuleDictObject` (module/global dicts) inherit from it.
///
/// ```python
/// class W_DictMultiObject(W_Root):
///     """ Abstract base class that does not store a strategy. """
///     __slots__ = ['space', 'dstorage']
///
///     def get_strategy(self):
///         raise NotImplementedError("abstract method")
///
///     def set_strategy(self, strategy):
///         raise NotImplementedError("abstract method")
/// ```
///
/// Pyre's `space` field is implicit (no `ObjSpace` shim per-dict);
/// `dstorage` lives on each concrete subclass as the appropriate type
/// (`*mut Vec<(PyObjectRef, PyObjectRef)>` for W_DictObject,
/// `*mut ModuleDictStorage` for W_ModuleDictObject).  The trait
/// surfaces the abstract `get_strategy` / `set_strategy` for static
/// trait-dispatch sites; runtime polymorphic dispatch (i.e. when only
/// a `PyObjectRef` is in hand) goes through the free function
/// [`w_dict_get_strategy`].
pub trait W_DictMultiObject {
    /// `dictmultiobject.py:49-50 W_DictMultiObject.get_strategy`
    /// abstract method, overridden by `W_DictObject` (`:321-322`) and
    /// `W_ModuleDictObject` (`:338-339`).  Each concrete subclass
    /// returns its strategy slot.
    fn get_strategy(&self) -> &dyn crate::dictstrategy::DictStrategy;

    /// `dictmultiobject.py:52-53 W_DictMultiObject.set_strategy`
    /// abstract method, overridden by `W_DictObject` (`:324-325`) and
    /// `W_ModuleDictObject` (`:341-342`).  Pyre limits the setter to
    /// `&'static dyn DictStrategy` (the singleton dispatch surface);
    /// W_ModuleDictObject strategy promotion to ObjectDictStrategy
    /// continues to go through `w_module_dict_switch_to_object_strategy`
    /// per `celldict.py:173-186`.
    fn set_strategy(&mut self, strategy: &'static dyn crate::dictstrategy::DictStrategy);
}

/// `pypy/objspace/std/dictmultiobject.py:313-325 W_DictObject(W_DictMultiObject)`
/// — the regular-dict concrete subclass.  PyPy slots are
/// `['dstrategy']` on top of W_DictMultiObject's `['space', 'dstorage']`;
/// pyre carries the same logical slots plus pre-existing-adaptations
/// for the legacy DictStorage bridge.
///
/// Layout: `[ob_header | dstorage | dict_storage_proxy | dstrategy]`
///
/// Slots:
/// - `dstorage`: erased ObjectDictStrategy storage —
///   `Vec<(PyObjectRef, PyObjectRef)>` matches PyPy's
///   `r_dict(space.eq_w, space.hash_w)` storage (`:1209-1212`).  Keys
///   compared by `dict_keys_equal` which routes through the registered
///   `dict_eq_hook::EQ_W_HOOK` trampoline → `baseobjspace::eq_w`.
/// - `dict_storage_proxy`: NEW-DEVIATION back-mirror pointer to a
///   legacy `DictStorage` allocation.  Non-null only when a frame's
///   `w_globals` (or a module's `w_dict`) was constructed from raw
///   `DictStorage` and lifted via `dict_storage_to_dict`; mutations
///   then fan out to the storage as well as the local `dstorage`.
///   Retirement is gated on the PyFrame.w_globals migration (Phase
///   C-1) — once `PyFrame.w_globals` is `PyObjectRef` directly, no
///   storage-side mirror is needed.
/// - `dstrategy`: PyPy `:315 __slots__ = ['dstrategy']`.  Reference
///   to the active strategy singleton; `space.fromcache(StrategyCls)`
///   returns the same instance per space, so pyre stores a `&'static
///   dyn DictStrategy` (fat pointer).  New dicts start in
///   `OBJECT_DICT_STRATEGY`; promotion is `w_dict_set_strategy`
///   per `:324-325 set_strategy`.
#[repr(C)]
pub struct W_DictObject {
    pub ob_header: PyObject,
    /// `dstorage` from `W_DictMultiObject.__slots__` (`dictmultiobject.py:47`).
    /// PyPy's `rerased`-erased storage; pyre uses `*mut u8` for the
    /// same opacity contract.  Each strategy `unerase(dict.dstorage)`
    /// via a typed accessor (`w_dict_object_storage*` for the unified
    /// `Vec<(PyObjectRef, PyObjectRef)>` shape).  Per-strategy native
    /// storage layouts (`Vec<(i64, _)>`, `IndexMap<String, _>`, etc.)
    /// follow in subsequent slices.
    pub dstorage: *mut u8,
    pub dict_storage_proxy: *mut u8,
    pub dstrategy: &'static dyn crate::dictstrategy::DictStrategy,
}

/// Typed accessor — `dictmultiobject.py:1213-1215 ObjectDictStrategy.getitem`
/// (`self.unerase(w_dict.dstorage)`) returns the `r_dict(dict_keys_equal,
/// hash_w)` backing.  Pyre stores this as `IndexMap<ObjectKey,
/// PyObjectRef>`: a hash bucket for O(1) lookup that also preserves
/// insertion order (CPython 3.7+ / PyPy3 dict semantics).  Shared with
/// `UnicodeDictStrategy` per `dictmultiobject.py:1311-1318`'s str
/// fast-path delegation to the Object helpers.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on
/// `ObjectDictStrategy` or `UnicodeDictStrategy`.
#[inline]
pub unsafe fn w_dict_object_storage<'a>(
    obj: PyObjectRef,
) -> &'a indexmap::IndexMap<ObjectKey, PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    &*(dict.dstorage as *const indexmap::IndexMap<ObjectKey, PyObjectRef>)
}

/// Mutable typed accessor — write-side of [`w_dict_object_storage`].
///
/// # Safety
/// Same as [`w_dict_object_storage`].
#[inline]
pub unsafe fn w_dict_object_storage_mut<'a>(
    obj: PyObjectRef,
) -> &'a mut indexmap::IndexMap<ObjectKey, PyObjectRef> {
    let dict = &mut *(obj as *mut W_DictObject);
    &mut *(dict.dstorage as *mut indexmap::IndexMap<ObjectKey, PyObjectRef>)
}

/// Typed accessor for `IntDictStrategy.unerase(w_dict.dstorage)` —
/// `dictmultiobject.py:1349-1352 IntDictStrategy.erase/unerase` pair
/// produced by `rerased.new_erasing_pair("integer")`.  Returns the
/// native `IndexMap<i64, PyObjectRef>` backing — insertion-ordered
/// hash bucket matching PyPy's `Dict[int, W_Root]` (RPython resolves
/// to an order-preserving hash table at translation time).
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` whose strategy is
/// [`crate::dictstrategy::INT_DICT_STRATEGY`].
#[inline]
pub unsafe fn w_dict_int_storage<'a>(obj: PyObjectRef) -> &'a indexmap::IndexMap<i64, PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    &*(dict.dstorage as *const indexmap::IndexMap<i64, PyObjectRef>)
}

/// Mutable counterpart of [`w_dict_int_storage`].
///
/// # Safety
/// Same as [`w_dict_int_storage`].
#[inline]
pub unsafe fn w_dict_int_storage_mut<'a>(
    obj: PyObjectRef,
) -> &'a mut indexmap::IndexMap<i64, PyObjectRef> {
    let dict = &mut *(obj as *mut W_DictObject);
    &mut *(dict.dstorage as *mut indexmap::IndexMap<i64, PyObjectRef>)
}

/// Typed accessor for `BytesDictStrategy.unerase(w_dict.dstorage)` —
/// `dictmultiobject.py:1230-1232 BytesDictStrategy.erase/unerase` pair
/// produced by `rerased.new_erasing_pair("bytes")`.  Returns the
/// native `IndexMap<Vec<u8>, PyObjectRef>` backing — insertion-ordered
/// hash bucket matching PyPy `Dict[str, W_Root]` (RPython resolves to
/// an order-preserving hash table at translation time).
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` whose strategy is
/// [`crate::dictstrategy::BYTES_DICT_STRATEGY`].
#[inline]
pub unsafe fn w_dict_bytes_storage<'a>(
    obj: PyObjectRef,
) -> &'a indexmap::IndexMap<Vec<u8>, PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    &*(dict.dstorage as *const indexmap::IndexMap<Vec<u8>, PyObjectRef>)
}

/// Mutable counterpart of [`w_dict_bytes_storage`].
///
/// # Safety
/// Same as [`w_dict_bytes_storage`].
#[inline]
pub unsafe fn w_dict_bytes_storage_mut<'a>(
    obj: PyObjectRef,
) -> &'a mut indexmap::IndexMap<Vec<u8>, PyObjectRef> {
    let dict = &mut *(obj as *mut W_DictObject);
    &mut *(dict.dstorage as *mut indexmap::IndexMap<Vec<u8>, PyObjectRef>)
}

/// GC type id assigned to `W_DictObject` at JitDriver init time.
pub const W_DICT_GC_TYPE_ID: u32 = 29;

/// Fixed payload size (`framework.py:811`).
pub const W_DICT_OBJECT_SIZE: usize = std::mem::size_of::<W_DictObject>();

impl crate::lltype::GcType for W_DictObject {
    fn type_id() -> u32 {
        W_DICT_GC_TYPE_ID
    }
    const SIZE: usize = W_DICT_OBJECT_SIZE;
}

/// `pypy/objspace/std/dictmultiobject.py:313-325 W_DictObject(W_DictMultiObject)`
/// inheritance — `get_strategy`/`set_strategy` overrides read/write
/// the `dstrategy` slot directly.
impl W_DictMultiObject for W_DictObject {
    #[inline]
    fn get_strategy(&self) -> &dyn crate::dictstrategy::DictStrategy {
        self.dstrategy
    }

    #[inline]
    fn set_strategy(&mut self, strategy: &'static dyn crate::dictstrategy::DictStrategy) {
        self.dstrategy = strategy;
    }
}

#[inline]
fn dict_write_barrier(obj: PyObjectRef) {
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// `pypy/objspace/std/dictmultiobject.py:321-322 W_DictObject.get_strategy`
/// (the regular-dict subclass returns its `dstrategy` slot directly).
///
/// `pypy/objspace/std/dictmultiobject.py:49-50 W_DictMultiObject.get_strategy`
/// abstract method, overridden by `W_DictObject` (`:321-322`) and
/// `W_ModuleDictObject` (`:338-339`).  Polymorphic dispatch: returns
/// the strategy live in the dict's slot — `dstrategy` for regular
/// dicts (currently always `OBJECT_DICT_STRATEGY`), or `mstrategy`
/// (the per-allocation `ModuleDictStrategy`) for module dicts.
///
/// The `&'static` lifetime is sound because pyre's strategy
/// allocations (`malloc_raw` + `Box::leak` pattern for
/// `ModuleDictStrategy`) live for the program duration, matching the
/// strategy-singleton model of PyPy `space.fromcache(StrategyCls)`.
///
/// # Safety
/// `obj` must be a valid PyObjectRef pointing at a `W_DictObject` or
/// `W_ModuleDictObject`.
#[inline]
pub unsafe fn w_dict_get_strategy(
    obj: PyObjectRef,
) -> &'static dyn crate::dictstrategy::DictStrategy {
    if is_module_dict(obj) {
        let strat_ptr = (*(obj as *const W_ModuleDictObject)).mstrategy;
        return &*strat_ptr;
    }
    let dict = &*(obj as *const W_DictObject);
    dict.dstrategy
}

/// `pypy/objspace/std/dictmultiobject.py:52-53 W_DictMultiObject.set_strategy`
/// abstract method, overridden by `W_DictObject` (`:324-325`) and
/// `W_ModuleDictObject` (`:341-342`).  Polymorphic dispatch: writes
/// the strategy slot of whichever subclass `obj` points to.
///
/// W_ModuleDictObject's `mstrategy` slot is typed `*mut
/// ModuleDictStrategy` (concrete) rather than `&'static dyn
/// DictStrategy` (erased); the panic here mirrors
/// `<W_ModuleDictObject as W_DictMultiObject>::set_strategy` —
/// pyre's promotion path is `w_module_dict_switch_to_object_strategy`
/// per `celldict.py:173-186`, not a direct mstrategy swap, so this
/// gateway only ever sees regular W_DictObjects today.
///
/// # Safety
/// `obj` must be a valid PyObjectRef pointing at a `W_DictObject` or
/// `W_ModuleDictObject`.
#[inline]
pub unsafe fn w_dict_set_strategy(
    obj: PyObjectRef,
    strategy: &'static dyn crate::dictstrategy::DictStrategy,
) {
    if is_module_dict(obj) {
        panic!(
            "w_dict_set_strategy: W_ModuleDictObject strategy swap is not the canonical \
             path; use w_module_dict_switch_to_object_strategy (celldict.py:173-186)"
        );
    }
    let dict = &mut *(obj as *mut W_DictObject);
    dict.dstrategy = strategy;
}

/// `celldict.py:42-50 getdictvalue_no_unwrapping` slot lookup for the JIT
/// global cell fast path.  Returns the insertion-order index of `name`
/// inside the module dict's `ModuleDictStorage`, or `None` when `obj` is
/// not a `W_ModuleDictObject` in `ModuleDictStrategy` mode (after
/// `switch_to_object_strategy` the `object_storage` is authoritative and
/// the version-keyed cell path is permanently invalid).  The index is
/// stable across in-place cell mutation and value overwrite (`IndexMap`
/// keeps the slot position), so it serves as the elidable lookup key.
///
/// # Safety
/// `obj` must be null or a valid PyObjectRef.
pub unsafe fn module_dict_cell_slot_of(obj: PyObjectRef, name: &str) -> Option<usize> {
    if obj.is_null() || !is_module_dict(obj) {
        return None;
    }
    let md = &*(obj as *const W_ModuleDictObject);
    if !md.object_storage.is_null() || md.dstorage.is_null() {
        return None;
    }
    (*md.dstorage).entries.get_index_of(name)
}

/// `celldict.py:53-54 _getdictvalue_no_unwrapping_pure` — the raw stored
/// value-or-cell at `slot` (the result of `getdictvalue_no_unwrapping`,
/// _not_ unwrapped).  `None` when `obj` is not a module dict in
/// `ModuleDictStrategy` mode or `slot` is out of range.
///
/// # Safety
/// `obj` must be null or a valid PyObjectRef.
pub unsafe fn module_dict_cell_at(obj: PyObjectRef, slot: usize) -> Option<PyObjectRef> {
    if obj.is_null() || !is_module_dict(obj) {
        return None;
    }
    let md = &*(obj as *const W_ModuleDictObject);
    if !md.object_storage.is_null() || md.dstorage.is_null() {
        return None;
    }
    (*md.dstorage).entries.get_index(slot).map(|(_, v)| *v)
}

/// Direct O(1) entry count of a module dict's `ModuleDictStorage`
/// (`dictmultiobject.py:107-109 length` for the `ModuleDictStrategy`
/// case), bypassing both the strategy vtable and the
/// `dict_storage_proxy` reconciliation that `w_module_dict_length`
/// performs.  `None` when `obj` is not a `W_ModuleDictObject` in
/// `ModuleDictStrategy` mode.  Used by the JIT frame-shape guard, which
/// is on the per-portal-entry hot path and cannot afford the proxy
/// `maybe_items_dict_storage` materialization.
///
/// # Safety
/// `obj` must be null or a valid PyObjectRef.
pub unsafe fn module_dict_storage_len(obj: PyObjectRef) -> Option<usize> {
    if obj.is_null() || !is_module_dict(obj) {
        return None;
    }
    let md = &*(obj as *const W_ModuleDictObject);
    if !md.object_storage.is_null() || md.dstorage.is_null() {
        return None;
    }
    Some((*md.dstorage).len())
}

/// Register a compiled loop's invalidation `flag` against the module
/// dict's `ModuleDictStrategy.version?` quasi-immutable field
/// (`celldict.py:34 _immutable_fields_ = ["version?"]`).  The compile-time
/// glue calls this once per version-keyed module-global dependency so a
/// later `mutated()` (new key, `del`, or `switch_to_object_strategy`)
/// flips the flag and fails the loop's `GUARD_NOT_INVALIDATED`.  No-op
/// when `obj` is not a `W_ModuleDictObject`.
///
/// # Safety
/// `obj` must be null or a valid PyObjectRef.
pub unsafe fn module_dict_register_version_watcher(
    obj: PyObjectRef,
    flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    if obj.is_null() || !is_module_dict(obj) {
        return;
    }
    let md = &*(obj as *const W_ModuleDictObject);
    if md.mstrategy.is_null() {
        return;
    }
    (*md.mstrategy).register_version_watcher(flag);
}

/// Allocate a new empty dict per `dictmultiobject.py:67-69
/// allocate_and_init_instance`:
///
/// ```python
/// strategy = space.fromcache(EmptyDictStrategy)
/// storage = strategy.get_empty_storage()
/// W_DictObject.__init__(w_obj, space, strategy, storage)
/// ```
///
/// The initial strategy is `EMPTY_DICT_STRATEGY`; the first
/// mutating call (setitem / setitem_str / setdefault) promotes the
/// dict to a concrete strategy via
/// `EmptyDictStrategy::setitem`'s `switch_to_correct_strategy` step.
/// Pyre keeps a non-null `dstorage` Vec at construction so legacy
/// helpers reading the Vec directly still see an empty container;
/// when EmptyDictStrategy is active the Vec is observationally
/// empty (the trait readers return empty without touching the slot).
pub fn w_dict_new() -> PyObjectRef {
    let entries: *mut indexmap::IndexMap<ObjectKey, PyObjectRef> =
        crate::lltype::malloc_raw(indexmap::IndexMap::new());
    alloc_dict_object(
        W_DictObject {
            ob_header: PyObject {
                ob_type: &DICT_TYPE as *const PyType,
                w_class: get_instantiate(&DICT_TYPE),
            },
            dstorage: entries as *mut u8,
            dict_storage_proxy: std::ptr::null_mut(),
            dstrategy: &crate::dictstrategy::EMPTY_DICT_STRATEGY,
        },
        false,
    )
}

/// `dictmultiobject.py:77-80 allocate_and_init_instance` kwargs
/// branch — `strategy = space.fromcache(EmptyKwargsDictStrategy)`.
/// Function-call sites that build a `**kwargs` dict route through
/// this allocator so the first unicode setitem promotes the dict
/// directly to `KwargsDictStrategy` (skipping the regular
/// `UnicodeDictStrategy` intermediate).
pub fn w_dict_new_kwargs() -> PyObjectRef {
    alloc_dict_object(
        W_DictObject {
            ob_header: PyObject {
                ob_type: &DICT_TYPE as *const PyType,
                w_class: get_instantiate(&DICT_TYPE),
            },
            dstorage: std::ptr::null_mut(),
            dict_storage_proxy: std::ptr::null_mut(),
            dstrategy: &crate::dictstrategy::EMPTY_KWARGS_DICT_STRATEGY,
        },
        false,
    )
}

/// `dictmultiobject.py:81-89 allocate_and_init_instance` default
/// branch — `strategy = space.fromcache(EmptyDictStrategy)`, return
/// a regular W_DictObject.  Pyre adds a `dict_storage_proxy` slot so
/// the resulting dict shares writes with the supplied legacy
/// `DictStorage` allocation (the same back-mirror contract as
/// `w_module_dict_new_with_storage_proxy` but for the regular dict
/// shape, used by instance / function-locals / type-namespace
/// callers that don't want module-strategy machinery).
///
/// PyPy's `space.newdict(instance=True)` branch (`:70-72`) routes to
/// `mapdict.make_instance_dict`, which is not yet ported in pyre;
/// this helper produces the EmptyDictStrategy fallback instead — a
/// TODO until mapdict lands.
pub fn w_dict_new_with_storage_proxy(ns: *mut u8) -> PyObjectRef {
    let entries: *mut indexmap::IndexMap<ObjectKey, PyObjectRef> =
        crate::lltype::malloc_raw(indexmap::IndexMap::new());
    alloc_dict_object(
        W_DictObject {
            ob_header: PyObject {
                ob_type: &DICT_TYPE as *const PyType,
                w_class: get_instantiate(&DICT_TYPE),
            },
            dstorage: entries as *mut u8,
            dict_storage_proxy: ns,
            dstrategy: &crate::dictstrategy::EMPTY_DICT_STRATEGY,
        },
        false,
    )
}

/// `dictmultiobject.py:81-89 W_DictObject(space, strategy, storage)` —
/// caller-chosen strategy + caller-owned dstorage.  Used by the
/// per-strategy `copy()` overrides (`:1152 AbstractTypedStrategy.copy`)
/// to allocate a fresh W_DictObject that preserves the source's
/// strategy + a freshly cloned typed storage box.  Length is computed
/// on demand by `strategy.length(self)` from the typed storage shape.
/// The new dict starts without a `dict_storage_proxy` since fresh
/// copies are not participants in any back-mirror contract.
pub fn w_dict_new_with(
    strategy: &'static dyn crate::dictstrategy::DictStrategy,
    dstorage: *mut u8,
) -> PyObjectRef {
    alloc_dict_object(
        W_DictObject {
            ob_header: PyObject {
                ob_type: &DICT_TYPE as *const PyType,
                w_class: get_instantiate(&DICT_TYPE),
            },
            dstorage,
            dict_storage_proxy: std::ptr::null_mut(),
            dstrategy: strategy,
        },
        false,
    )
}

/// Allocate a dict for a pyre-side address-keyed side table.
///
/// These tables are not part of the translated object graph yet, so the dict
/// holder itself must keep a stable raw address. The table walker traces the
/// entries through [`w_dict_walk_entries_mut`] instead.
pub fn w_dict_new_unmanaged_side_table_value() -> PyObjectRef {
    let entries: *mut indexmap::IndexMap<ObjectKey, PyObjectRef> =
        crate::lltype::malloc_raw(indexmap::IndexMap::new());
    crate::lltype::malloc_typed(W_DictObject {
        ob_header: PyObject {
            ob_type: &DICT_TYPE as *const PyType,
            w_class: get_instantiate(&DICT_TYPE),
        },
        dstorage: entries as *mut u8,
        dict_storage_proxy: std::ptr::null_mut(),
        dstrategy: &crate::dictstrategy::OBJECT_DICT_STRATEGY,
    }) as PyObjectRef
}

/// Visit the raw `entries` vector's key/value slots with mutable access.
///
/// Used by pyre-side side table walkers whose dict object is not itself
/// GC-managed but whose contained PyObjectRef values still need relocation.
pub unsafe fn w_dict_walk_entries_mut(obj: PyObjectRef, mut visitor: impl FnMut(&mut PyObjectRef)) {
    let entries = w_dict_object_storage_mut(obj);
    for (key, value) in entries.iter_mut() {
        // `ObjectKey.hash` is precomputed at insertion (`object_key_for`)
        // and stays valid across a GC move — Python's `__hash__` contract
        // is identity-stable for hashable (immutable) keys, which is the
        // only class of object permitted in this strategy.  Mutating the
        // embedded `obj` slot in place therefore does not invalidate
        // IndexMap's bucket placement.
        let key_ptr = key as *const ObjectKey as *mut ObjectKey;
        visitor(&mut (*key_ptr).obj);
        visitor(value);
    }
}

fn alloc_dict_object(value: W_DictObject, stable: bool) -> PyObjectRef {
    let raw = if stable {
        crate::gc_hook::try_gc_alloc_stable(W_DICT_GC_TYPE_ID, W_DICT_OBJECT_SIZE)
    } else {
        crate::gc_hook::try_gc_alloc(W_DICT_GC_TYPE_ID, W_DICT_OBJECT_SIZE)
    };
    match raw.filter(|p| !p.is_null()) {
        Some(raw) => unsafe {
            std::ptr::write(raw as *mut W_DictObject, value);
            raw as PyObjectRef
        },
        None => crate::lltype::malloc_typed(value) as PyObjectRef,
    }
}

// ── W_ModuleDictObject ──────────────────────────────────────────────
//
// `pypy/objspace/std/dictmultiobject.py:328-350 W_ModuleDictObject`:
//
//     class W_ModuleDictObject(W_DictMultiObject):
//         """ a dict object for a module, that is not expected to
//         change. It stores the strategy as a quasi-immutable
//         field. """
//         __slots__ = ['mstrategy']
//         _immutable_fields_ = ['mstrategy?']
//
//         def __init__(self, space, strategy, storage):
//             W_DictMultiObject.__init__(self, space, storage)
//             self.mstrategy = strategy
//
//         def get_strategy(self):
//             return self.mstrategy
//
//         def set_strategy(self, strategy):
//             self.mstrategy = strategy
//
// Carries an owned `ModuleDictStorage` and a per-allocation
// `ModuleDictStrategy` (PyPy reuses one strategy instance per space;
// pyre allocates one strategy per W_ModuleDictObject for simplicity
// until the `space` shim grows a strategy cache).

/// Static `PyType` tag for `W_ModuleDictObject`.  Distinct from
/// `DICT_TYPE` so `py_type_check` can tell module dicts apart from
/// regular dicts inside the Rust runtime — but PyPy presents both as
/// `space.w_dict` to user code (`dictmultiobject.py:67
/// allocate_instance(W_ModuleDictObject, space.w_dict)`).  The
/// `TYPEOBJECT_CACHE` registration at
/// `pyre-interpreter/typedef.rs:300-303` maps `MODULE_DICT_TYPE` to
/// the same `dict` W_TypeObject as `DICT_TYPE`, so `type(g) is dict`
/// and `isinstance(g, dict)` hold on `W_ModuleDictObject` instances
/// even though they carry a different Rust layout / GC type id.
pub static MODULE_DICT_TYPE: PyType = new_pytype("dict");

#[repr(C)]
pub struct W_ModuleDictObject {
    pub ob_header: PyObject,
    /// `dstorage` from `W_DictMultiObject.__slots__` (`dictmultiobject.py:47`).
    /// Owned via `malloc_raw` (`Box::into_raw`).  Authoritative while
    /// `object_storage` is null (ModuleDictStrategy mode); after
    /// `switch_to_object_strategy` it is cleared and not consulted.
    pub dstorage: *mut crate::celldict::ModuleDictStorage,
    /// `mstrategy` from `W_ModuleDictObject.__slots__` (`:331`).
    /// Owned via `malloc_raw`.
    pub mstrategy: *mut crate::celldict::ModuleDictStrategy,
    /// `dstorage` after a `switch_to_object_strategy`
    /// (`celldict.py:173-186`).  Null while the dict is in
    /// ModuleDictStrategy mode; non-null once a non-str key forces the
    /// strategy swap.  Holds the unified ObjectKey-keyed entries that
    /// PyPy keeps inside the new `ObjectDictStrategy` storage after the
    /// switch — `dstorage`'s entries are drained into this IndexMap in
    /// their original insertion order so `items()` / `popitem()` LIFO
    /// parity is preserved across mixed-key inserts.  Backing matches
    /// `ObjectDictStrategy` (`dictmultiobject.py:1209-1212 r_dict(space.eq_w,
    /// space.hash_w)`) — same `ObjectKey { hash, obj }` precomputed-hash
    /// + `dict_keys_equal` equality.
    pub object_storage: *mut indexmap::IndexMap<ObjectKey, PyObjectRef>,
    /// Optional sibling-storage mirror.  When non-null, every
    /// `w_module_dict_*` write fans out to this `DictStorage` via
    /// `maybe_sync_dict_storage_*` hooks; reads fall back here on a
    /// local miss.  Used by `dict_storage_to_dict` so the frame-side
    /// `*mut DictStorage`-typed `PyFrame.w_globals` continues to
    /// observe `STORE_GLOBAL` and `module.__dict__[k] = v` writes
    /// uniformly.  TODO until
    /// `PyFrame.w_globals` to `PyObjectRef`; mirrors the
    /// `dict_storage_proxy` pattern on W_DictObject (line 35).
    pub dict_storage_proxy: *mut u8,
}

/// GC type id assigned to `W_ModuleDictObject`.  Lands at slot 48,
/// the first free id after the foreign-pytype loop in
/// `pyre/pyre-jit/src/eval.rs` registers NONE_TYPE (43),
/// NOTIMPLEMENTED_TYPE (44), ELLIPSIS_TYPE (45), CODE_TYPE (46)
/// and PYTRACEBACK_TYPE (47).
pub const W_MODULE_DICT_GC_TYPE_ID: u32 = 48;

/// Fixed payload size used by `gct_fv_gc_malloc`.
pub const W_MODULE_DICT_OBJECT_SIZE: usize = std::mem::size_of::<W_ModuleDictObject>();

/// Byte offset of `dict_storage_proxy` within `W_ModuleDictObject`.
pub const W_MODULE_DICT_STORAGE_PROXY_OFFSET: usize =
    std::mem::offset_of!(W_ModuleDictObject, dict_storage_proxy);

impl crate::lltype::GcType for W_ModuleDictObject {
    fn type_id() -> u32 {
        W_MODULE_DICT_GC_TYPE_ID
    }
    const SIZE: usize = W_MODULE_DICT_OBJECT_SIZE;
}

/// `pypy/objspace/std/dictmultiobject.py:328-342 W_ModuleDictObject(W_DictMultiObject)`
/// inheritance — `get_strategy`/`set_strategy` overrides read/write
/// the `mstrategy` slot (a per-allocation `ModuleDictStrategy`).
/// The `&'static` lifetime on the returned strategy reference is
/// sound because pyre's `ModuleDictStrategy` is malloc_raw'd and
/// lives for the W_ModuleDictObject's full lifetime (which itself
/// outlives any caller currently in scope).
impl W_DictMultiObject for W_ModuleDictObject {
    #[inline]
    fn get_strategy(&self) -> &dyn crate::dictstrategy::DictStrategy {
        unsafe { &*self.mstrategy }
    }

    /// `celldict.py:185 w_dict.set_strategy(strategy)` — the only
    /// strategy transition out of ModuleDictStrategy is to
    /// ObjectDictStrategy via `switch_to_object_strategy` (`:173-186`).
    /// Route through the existing helper so trait-dispatch callers
    /// (e.g. `optimizeopt` rewrites or future `descr_copy` paths) land
    /// correctly without panicking; non-Object target strategies stay
    /// unreachable per the upstream surface.
    ///
    /// TODO: pyre carries `object_storage` as a
    /// side field instead of swapping `dstorage` wholesale (PyPy's
    /// `w_dict.dstorage = strategy.erase(d_new)`).  The trait method
    /// hides that adapter from callers; the side-field layout retires
    /// alongside typed-strategy storage migration.
    fn set_strategy(&mut self, strategy: &'static dyn crate::dictstrategy::DictStrategy) {
        let target = strategy as *const dyn crate::dictstrategy::DictStrategy as *const () as usize;
        let object_singleton = &crate::dictstrategy::OBJECT_DICT_STRATEGY
            as *const crate::dictstrategy::ObjectDictStrategy
            as *const () as usize;
        if target != object_singleton {
            panic!(
                "W_ModuleDictObject::set_strategy: only ObjectDictStrategy transition is \
                 implemented (celldict.py:173-186 is the only documented swap target)"
            );
        }
        let obj = self as *mut Self as PyObjectRef;
        unsafe { w_module_dict_switch_to_object_strategy(obj) };
    }
}

/// Allocate a fresh `W_ModuleDictObject` whose storage is empty and
/// whose strategy carries a fresh `VersionTag`.  Mirrors
/// `dictmultiobject.py:57-69 allocate_and_init_instance(module=True)`
/// path:
///
/// ```python
/// if module:
///     strategy = ModuleDictStrategy(space)
///     storage = strategy.get_empty_storage()
///     w_obj = space.allocate_instance(W_ModuleDictObject, space.w_dict)
///     W_ModuleDictObject.__init__(w_obj, space, strategy, storage)
///     return w_obj
/// ```
pub fn w_module_dict_new() -> PyObjectRef {
    w_module_dict_new_with_storage_proxy(std::ptr::null_mut())
}

/// `w_module_dict_new` variant pairing the W_ModuleDictObject with a
/// legacy `DictStorage` mirror.  Writes through the W_ModuleDictObject
/// fan out to `ns` via the storage proxy hooks; reads fall back to
/// `ns` on a local miss.  Used by `dict_storage_to_dict` so source
/// modules surface as W_ModuleDictObject while the frame-side
/// `PyFrame.w_globals = *mut DictStorage` carrier still works.
pub fn w_module_dict_new_with_storage_proxy(ns: *mut u8) -> PyObjectRef {
    let strategy = crate::lltype::malloc_raw(crate::celldict::ModuleDictStrategy::new());
    let storage = unsafe { crate::lltype::malloc_raw((*strategy).get_empty_storage()) };
    crate::lltype::malloc_typed(W_ModuleDictObject {
        ob_header: PyObject {
            // `dictmultiobject.py:67 space.allocate_instance(...,
            // space.w_dict)` — module dicts present as `dict` to
            // user code (registered via TYPEOBJECT_CACHE on the
            // interpreter side); the Rust static tag distinguishes
            // the layout internally.
            ob_type: &MODULE_DICT_TYPE as *const PyType,
            w_class: get_instantiate(&MODULE_DICT_TYPE),
        },
        dstorage: storage,
        mstrategy: strategy,
        object_storage: std::ptr::null_mut(),
        dict_storage_proxy: ns,
    }) as PyObjectRef
}

/// Predicate: dict is in ObjectDictStrategy mode (post-switch).  When
/// true, `object_storage` is authoritative and `dstorage` is empty +
/// not consulted.  Mirrors `W_DictMultiObject.get_strategy()` returning
/// `ObjectDictStrategy` vs `ModuleDictStrategy` (`dictmultiobject.py:326`).
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
#[inline]
pub unsafe fn w_module_dict_is_object_strategy(obj: PyObjectRef) -> bool {
    !(*(obj as *const W_ModuleDictObject))
        .object_storage
        .is_null()
}

/// Read-only view of the unified object_storage IndexMap; returns `None`
/// when the dict is still in ModuleDictStrategy mode.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
#[inline]
pub unsafe fn w_module_dict_object_storage<'a>(
    obj: PyObjectRef,
) -> Option<&'a indexmap::IndexMap<ObjectKey, PyObjectRef>> {
    let raw = &*(obj as *const W_ModuleDictObject);
    if raw.object_storage.is_null() {
        None
    } else {
        Some(&*raw.object_storage)
    }
}

/// Mutable view of the unified object_storage IndexMap; requires the
/// dict to already be in object-strategy mode.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject` for which
/// `w_module_dict_is_object_strategy(obj)` holds.
#[inline]
pub unsafe fn w_module_dict_object_storage_mut<'a>(
    obj: PyObjectRef,
) -> &'a mut indexmap::IndexMap<ObjectKey, PyObjectRef> {
    let raw = &mut *(obj as *mut W_ModuleDictObject);
    debug_assert!(!raw.object_storage.is_null());
    &mut *raw.object_storage
}

/// Mutable view of the unified object_storage IndexMap when present;
/// returns `None` while the dict is still in ModuleDictStrategy mode.
/// Use this variant when the caller does not control the strategy
/// state at the call site (e.g. trait-dispatch entry points that
/// service both pre- and post-switch dicts).
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
#[inline]
pub unsafe fn w_module_dict_object_storage_mut_opt<'a>(
    obj: PyObjectRef,
) -> Option<&'a mut indexmap::IndexMap<ObjectKey, PyObjectRef>> {
    let raw = &mut *(obj as *mut W_ModuleDictObject);
    if raw.object_storage.is_null() {
        None
    } else {
        Some(&mut *raw.object_storage)
    }
}

/// `pypy/objspace/std/celldict.py:173-186 switch_to_object_strategy`:
///
/// ```python
/// def switch_to_object_strategy(self, w_dict):
///     space = self.space
///     d = self.unerase(w_dict.dstorage)
///     strategy = space.fromcache(ObjectDictStrategy)
///     d_new = strategy.unerase(strategy.get_empty_storage())
///     for key, cell in d.iteritems():
///         d_new[_wrapkey(space, key)] = unwrap_cell(self.space, cell)
///     if self.caches is not None:
///         for cache in self.caches.itervalues():
///             cache.cell = None
///             cache.valid = False
///         self.caches = None
///     w_dict.set_strategy(strategy)
///     w_dict.dstorage = strategy.erase(d_new)
/// ```
///
/// Drains all str entries from `dstorage` into a fresh
/// `object_storage` Vec, preserving insertion order (PyPy's
/// `iteritems` over an RPython dict yields insertion order), clears
/// `dstorage`, and bumps `mstrategy.version` so any quasi-immutable
/// JIT cache keyed on the previous version invalidates.  After this
/// call, all reads / writes route through `object_storage` regardless
/// of key type — matching PyPy's `ObjectDictStrategy` semantics.
///
/// **TODO** vs `celldict.py:185-186`:
/// PyPy actually swaps the strategy (`w_dict.set_strategy(strategy)`)
/// and replaces `w_dict.dstorage` with the new `strategy.erase(d_new)`
/// payload.  Pyre carries TWO storages (`dstorage` + `object_storage`)
/// and flips a flag (`object_storage` non-null) to route reads/writes
/// to the new container.  Functionally equivalent: after the switch,
/// `dstorage` is cleared and never consulted; `object_storage` is the
/// authoritative payload, exactly mirroring the post-`set_strategy`
/// PyPy state.
///
/// **Why it diverges**: full structural parity requires a
/// `DictStrategy` trait + concrete `ObjectDictStrategy` /
/// `UnicodeDictStrategy` ports (see `dictmultiobject.py:236-1369`)
/// so `set_strategy` can replace both the dispatch object and the
/// erased storage type uniformly.  That hierarchy is a large effort
/// (750+ LOC across 4 strategies, 200+ call sites in
/// `dictmultiobject.py` consuming `w_dict.get_strategy()` and
/// `space.fromcache(<Strategy>)`).
///
/// **Convergence path**: when the strategy hierarchy ports land,
/// drop `object_storage` and replace this two-slot carrier with a
/// single `dstorage: *mut dyn DictStorageErased` whose concrete
/// type is dictated by `mstrategy`'s runtime tag.
///
/// No-op when already in object-strategy mode.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_switch_to_object_strategy(obj: PyObjectRef) {
    let raw = &mut *(obj as *mut W_ModuleDictObject);
    if !raw.object_storage.is_null() {
        return;
    }
    let strategy = &mut *raw.mstrategy;
    let storage = &mut *raw.dstorage;
    let mut new_storage: indexmap::IndexMap<ObjectKey, PyObjectRef> =
        indexmap::IndexMap::with_capacity(storage.entries.len());
    for (k, v) in strategy
        .getiterkeys(storage)
        .zip(strategy.getitervalues(storage))
    {
        let key_obj = crate::celldict::_wrapkey(k);
        new_storage.insert(object_key_for(key_obj), v);
    }
    raw.object_storage = crate::lltype::malloc_raw(new_storage);
    storage.clear();
    // `celldict.py:180-184`: every live GlobalCache becomes invalid
    // because the strategy is being swapped out; the JIT must
    // recompile any trace keyed on the prior version.
    strategy.invalidate_caches();
    strategy.mutated();
}

/// `dictmultiobject.py:21-31 _never_equal_to_string`:
///
/// ```python
/// def _never_equal_to_string(space, w_lookup_type):
///     return (space.is_w(w_lookup_type, space.w_NoneType) or
///             space.is_w(w_lookup_type, space.w_int) or
///             space.is_w(w_lookup_type, space.w_bool) or
///             space.is_w(w_lookup_type, space.w_float))
/// ```
///
/// True when `key`'s type guarantees the key can never `==` any Python
/// string, so a still-ModuleDictStrategy dict can skip the strategy
/// switch and report absence directly.  For all other non-str types
/// (e.g. user-defined classes with custom `__eq__`/`__hash__`), the
/// caller must `switch_to_object_strategy` and re-dispatch the lookup
/// on the unified ObjectDictStrategy storage.
///
/// # Safety
/// `key` must be a valid PyObjectRef.
#[inline]
pub unsafe fn _never_equal_to_string(key: PyObjectRef) -> bool {
    crate::is_none(key) || crate::is_int(key) || crate::is_bool(key) || crate::is_float(key)
}

/// `is W_ModuleDictObject` predicate.  Disambiguates `W_ModuleDictObject`
/// from `W_DictObject` even though both surface as Python's `dict`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_module_dict(obj: PyObjectRef) -> bool {
    py_type_check(obj, &MODULE_DICT_TYPE)
}

/// `dictmultiobject.py:326 W_DictObject.get_strategy` / `:339
/// W_ModuleDictObject.get_strategy` — read the owning strategy.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
#[inline]
pub unsafe fn w_module_dict_get_strategy(
    obj: PyObjectRef,
) -> *mut crate::celldict::ModuleDictStrategy {
    (*(obj as *const W_ModuleDictObject)).mstrategy
}

/// Strategy identity stamp used by `_new_next` parity (`dictmultiobject
/// .py:829 self.strategy is self.w_dict.get_strategy()`).
///
/// Returns a `usize` that is stable across the dict's lifetime as long
/// as no strategy transition occurs, and changes whenever the strategy
/// transitions (e.g. `W_ModuleDictObject::switch_to_object_strategy`
/// flipping `object_storage` from null to non-null).  Distinct dicts
/// MAY share the same id when they share the strategy singleton
/// (e.g. two regular W_DictObject instances both on
/// `OBJECT_DICT_STRATEGY`); iterator parity is preserved because
/// the comparison is only performed against the SAME dict the iterator
/// was opened on.
///
/// # Safety
/// `obj` must point to a valid dict (W_DictObject or W_ModuleDictObject).
pub unsafe fn w_dict_strategy_id(obj: PyObjectRef) -> usize {
    if is_module_dict(obj) {
        let m = &*(obj as *const W_ModuleDictObject);
        if !m.object_storage.is_null() {
            // `switch_to_object_strategy` flipped the dict; tag with the
            // object_storage address so the id is distinct from any
            // pre-switch mstrategy stamp.
            return m.object_storage as usize;
        }
        return m.mstrategy as usize;
    }
    let d = &*(obj as *const W_DictObject);
    // `&'static dyn DictStrategy` — the fat pointer carries both a
    // vtable and a data pointer; the data pointer alone uniquely
    // identifies the strategy singleton (`OBJECT_DICT_STRATEGY` etc.).
    let raw: *const dyn crate::dictstrategy::DictStrategy = d.dstrategy;
    raw as *const () as usize
}

/// Read the owning storage pointer.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
#[inline]
pub unsafe fn w_module_dict_get_storage(
    obj: PyObjectRef,
) -> *mut crate::celldict::ModuleDictStorage {
    (*(obj as *const W_ModuleDictObject)).dstorage
}

/// `dictmultiobject.py:111-112 W_DictMultiObject.setitem_str`
/// dispatched through ModuleDictStrategy (`celldict.py:76-78`):
///
/// ```python
/// def setitem_str(self, key, w_value):
///     self.get_strategy().setitem_str(self, key, w_value)
/// ```
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_setitem_str(obj: PyObjectRef, key: &str, w_value: PyObjectRef) {
    w_module_dict_setitem_str_internal(obj, key, w_value, /*fire_proxy=*/ true);
}

/// `w_module_dict_setitem_str` variant that skips the
/// `dict_storage_proxy` forward hook.  Used by `DictStorage::insert`'s
/// `mirror_target` back-write to avoid recursion when the DictStorage
/// IS the mirror's own proxy target.
pub unsafe fn w_module_dict_setitem_str_no_proxy(
    obj: PyObjectRef,
    key: &str,
    w_value: PyObjectRef,
) {
    w_module_dict_setitem_str_internal(obj, key, w_value, /*fire_proxy=*/ false);
}

unsafe fn w_module_dict_setitem_str_internal(
    obj: PyObjectRef,
    key: &str,
    w_value: PyObjectRef,
    fire_proxy: bool,
) {
    let proxy = if fire_proxy {
        (*(obj as *const W_ModuleDictObject)).dict_storage_proxy
    } else {
        std::ptr::null_mut()
    };
    if w_module_dict_is_object_strategy(obj) {
        // Post-switch: ObjectDictStrategy storage = r_dict(space.eq_w,
        // space.hash_w) per `dictmultiobject.py:1210`; pyre's
        // `dict_keys_equal` enforces the same bucket invariant
        // (Item 1.2).  Wrap the str key into a W_StrObject once and
        // dispatch through `dict_keys_equal` so user-defined str
        // subclasses with `__eq__`/`__hash__` overrides honour their
        // own protocol, matching PyPy `setitem_str` which calls
        // `self.setitem(w_dict, self.space.newtext(s), w_value)`
        // (`dictmultiobject.py:1220-1221`).
        let w_key = crate::w_str_new(key);
        let entries = w_module_dict_object_storage_mut(obj);
        entries.insert(object_key_for(w_key), w_value);
        let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
        strategy.mutated();
        maybe_sync_dict_storage_store(proxy, w_key, w_value);
        return;
    }
    {
        let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
        let storage = &mut *(*(obj as *mut W_ModuleDictObject)).dstorage;
        strategy.setitem_str(storage, key, w_value);
    }
    // Only wrap the key into a W_StrObject when there is a proxy to forward
    // to.  The back-mirror from `DictStorage::insert` runs with a null proxy
    // (`fire_proxy=false`), so eagerly allocating the key here would discard it.
    if !proxy.is_null() {
        maybe_sync_dict_storage_store(proxy, crate::w_str_new(key), w_value);
    }
}

/// `celldict.py:143-145 getitem_str`.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_getitem_str(obj: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    let proxy = (*(obj as *const W_ModuleDictObject)).dict_storage_proxy;
    if let Some(entries) = w_module_dict_object_storage(obj) {
        // Post-switch ObjectStrategy: route through `dict_keys_equal`
        // (`dictmultiobject.py:1210` r_dict(eq_w, hash_w)) instead of
        // raw String content equality so str-subclass keys with
        // overridden `__eq__`/`__hash__` are reachable from the
        // str-fast-path lookup.
        let w_key = crate::w_str_new(key);
        if let Some(&v) = entries.get(&object_key_for(w_key)) {
            return Some(v);
        }
        if !proxy.is_null() {
            if let Some(v) = maybe_lookup_dict_storage(proxy, key) {
                return Some(v);
            }
        }
        return None;
    }
    {
        let strategy = &*(*(obj as *const W_ModuleDictObject)).mstrategy;
        let storage = &*(*(obj as *const W_ModuleDictObject)).dstorage;
        if let Some(v) = strategy.getitem_str(storage, key) {
            return Some(v);
        }
    }
    if !proxy.is_null() {
        if let Some(v) = maybe_lookup_dict_storage(proxy, key) {
            return Some(v);
        }
    }
    None
}

/// GC root walk over a `W_ModuleDictObject`'s movable references.
///
/// A module dict's authoritative storage is reached only behind raw
/// pointers — the `dstorage` cell map, the post-`switch_to_object_strategy`
/// `object_storage`, and the `mstrategy.caches` cell registry — none of
/// which are inline `gc_ptr_offsets`.  The W_ModuleDictObject is
/// currently `malloc_typed` (Box-immortal), so its registered
/// `module_dict_object_custom_trace` never fires; without an explicit
/// walk, `LOAD_GLOBAL` (`w_module_dict_getitem_str`, which reads the
/// authoritative `dstorage` cell map ahead of the proxy back-mirror)
/// would observe relocated values through never-forwarded slots.
/// `walk_pyframe_roots` calls this for every live frame's globals and
/// builtins so the real storage stays forwarded across collection.
/// No-op for non-module dicts.
///
/// `visitor` receives each movable value slot; `walk_module_value_slot`
/// unwraps `MutableCell`s (themselves Box-immortal) to forward the inner
/// `w_value` rather than the stable cell pointer.
///
/// # Safety
/// `obj` must be a valid `PyObjectRef` (null tolerated).  `visitor` must
/// tolerate being called on every movable slot reachable here.
pub unsafe fn w_module_dict_walk_gc_cells(
    obj: PyObjectRef,
    visitor: &mut dyn FnMut(&mut PyObjectRef),
) {
    if obj.is_null() || !is_module_dict(obj) {
        return;
    }
    let md = &mut *(obj as *mut W_ModuleDictObject);
    if !md.dstorage.is_null() {
        let storage = &mut *md.dstorage;
        for value in storage.iter_values_mut() {
            crate::celldict::walk_module_value_slot(value, visitor);
        }
    }
    if !md.object_storage.is_null() {
        let object_storage = &mut *md.object_storage;
        for (key, value) in object_storage.iter_mut() {
            // ObjectKey.hash is precomputed and identity-stable across GC
            // moves, so writing through the raw obj slot does not desync
            // the IndexMap bucket index.
            let key_ptr = key as *const ObjectKey as *mut ObjectKey;
            visitor(&mut (*key_ptr).obj);
            crate::celldict::walk_module_value_slot(value, visitor);
        }
    }
    if !md.mstrategy.is_null() {
        (*md.mstrategy).walk_cache_cells(visitor);
    }
}

/// `celldict.py:106-126 delitem` (str path).
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_delitem_str(obj: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    w_module_dict_delitem_str_internal(obj, key, /*fire_proxy=*/ true)
}

/// `w_module_dict_delitem_str` variant that skips the
/// `dict_storage_proxy` delete fan-out — used by
/// `DictStorage::remove`'s `mirror_target` back-write to avoid loop.
pub unsafe fn w_module_dict_delitem_str_no_proxy(
    obj: PyObjectRef,
    key: &str,
) -> Option<PyObjectRef> {
    w_module_dict_delitem_str_internal(obj, key, /*fire_proxy=*/ false)
}

unsafe fn w_module_dict_delitem_str_internal(
    obj: PyObjectRef,
    key: &str,
    fire_proxy: bool,
) -> Option<PyObjectRef> {
    let proxy = if fire_proxy {
        (*(obj as *const W_ModuleDictObject)).dict_storage_proxy
    } else {
        std::ptr::null_mut()
    };
    if w_module_dict_is_object_strategy(obj) {
        // Post-switch ObjectStrategy: route through `dict_keys_equal`
        // for the same r_dict bucket reason as `w_module_dict_setitem_str`
        // / `w_module_dict_getitem_str` (Item 2.2).
        let w_key = crate::w_str_new(key);
        let entries = w_module_dict_object_storage_mut(obj);
        if let Some(removed) = entries.shift_remove(&object_key_for(w_key)) {
            let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
            strategy.mutated();
            maybe_sync_dict_storage_delete(proxy, key);
            return Some(removed);
        }
        if !proxy.is_null() && maybe_lookup_dict_storage(proxy, key).is_some() {
            maybe_sync_dict_storage_delete(proxy, key);
            let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
            strategy.mutated();
            return Some(crate::PY_NULL);
        }
        return None;
    }
    let removed = {
        let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
        let storage = &mut *(*(obj as *mut W_ModuleDictObject)).dstorage;
        strategy.delitem_str(storage, key)
    };
    if removed.is_some() {
        maybe_sync_dict_storage_delete(proxy, key);
        return removed;
    }
    if !proxy.is_null() && maybe_lookup_dict_storage(proxy, key).is_some() {
        maybe_sync_dict_storage_delete(proxy, key);
        let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
        strategy.mutated();
        return Some(crate::PY_NULL);
    }
    None
}

/// `celldict.py:128-129 length`.
///
/// Reconciles with the `dict_storage_proxy` str-keyed view when
/// attached so direct `dict_storage_store` writes (JIT inline
/// LOAD_GLOBAL paths that bypass `w_module_dict_setitem_str`) are
/// reflected in the count, matching `w_module_dict_getitem_str`'s
/// proxy fallback at :730.  Mirrors the W_DictObject pattern at
/// `w_dict_length_object_strategy` (:1640).
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_length(obj: PyObjectRef) -> usize {
    let proxy = (*(obj as *const W_ModuleDictObject)).dict_storage_proxy;
    let local_len = if let Some(entries) = w_module_dict_object_storage(obj) {
        entries.len()
    } else {
        let strategy = &*(*(obj as *const W_ModuleDictObject)).mstrategy;
        let storage = &*(*(obj as *const W_ModuleDictObject)).dstorage;
        strategy.length(storage)
    };
    if proxy.is_null() {
        return local_len;
    }
    let Some(storage_items) = maybe_items_dict_storage(proxy) else {
        return local_len;
    };
    // Proxy is authoritative for str keys (writes fan out via
    // `maybe_sync_dict_storage_store`).  Non-str entries live only in
    // `object_storage` after `switch_to_object_strategy`;
    // ModuleDictStrategy mode itself is str-only.
    let non_str = if let Some(entries) = w_module_dict_object_storage(obj) {
        entries.keys().filter(|k| !crate::is_str(k.obj)).count()
    } else {
        0
    };
    storage_items.len() + non_str
}

/// Compare two dict keys for equality.
///
/// `pypy/objspace/std/dictmultiobject.py:1209 ObjectDictStrategy` —
/// the storage is `r_dict(space.eq_w, space.hash_w)` so every key
/// lookup routes through `space.eq_w` (`baseobjspace.py:823-825`),
/// which honours user-defined `__eq__`.  pyre-object cannot depend on
/// pyre-interpreter, so we go through the `dict_eq_hook::EQ_W_HOOK`
/// trampoline registered at `pyre-jit::eval` init.  When the hook is
/// not installed (pyre-object lib tests, snapshot tools) we fall back
/// to the limited-type builtin equality below — sufficient for the
/// hashable-builtin smoke tests but not for arbitrary user types.
pub(crate) unsafe fn dict_keys_equal(a: PyObjectRef, b: PyObjectRef) -> bool {
    if std::ptr::eq(a, b) {
        return true;
    }
    if a.is_null() || b.is_null() {
        return false;
    }
    // Once `space.eq_w` has raised earlier in this probe, skip every
    // remaining user comparison.  The Rust `Eq` callback cannot abort the
    // `IndexMap` scan, but suppressing further `__eq__` calls means no extra
    // comparison runs and the first exception is the one that propagates —
    // matching `r_dict(space.eq_w, space.hash_w)` raising at the first
    // comparison.  The flag is cleared per op in `object_key_for(_checked)`,
    // so this only fires after a raise within the current probe.
    if crate::dict_eq_hook::eq_error_pending() {
        return false;
    }
    if let Some(eq) = unsafe { crate::dict_eq_hook::try_eq_w(a, b) } {
        // `ObjectKey::eq` already gates on `self.hash != other.hash`
        // before calling `dict_keys_equal`, so the bucket invariant
        // (same hash_w → same bucket) is enforced by the cached hash
        // in `ObjectKey.hash`.  No need to re-hash here.
        return eq;
    }
    // Mixed numeric: bool ↔ int (Python: True == 1 and False == 0).
    let a_is_bool = crate::is_bool(a);
    let b_is_bool = crate::is_bool(b);
    let a_is_int = crate::is_int(a);
    let b_is_int = crate::is_int(b);
    if (a_is_int || a_is_bool) && (b_is_int || b_is_bool) {
        let av = if a_is_bool {
            crate::w_bool_get_value(a) as i64
        } else {
            crate::w_int_get_value(a)
        };
        let bv = if b_is_bool {
            crate::w_bool_get_value(b) as i64
        } else {
            crate::w_int_get_value(b)
        };
        return av == bv;
    }
    // Str keys — compare the WTF-8 bytes so lone-surrogate keys compare by
    // content instead of panicking in `w_str_get_value`.
    if crate::is_str(a) && crate::is_str(b) {
        return crate::w_str_get_wtf8(a).as_bytes() == crate::w_str_get_wtf8(b).as_bytes();
    }
    // Bytes keys — compare byte contents.
    if crate::bytesobject::is_bytes(a) && crate::bytesobject::is_bytes(b) {
        return crate::bytesobject::w_bytes_data(a) == crate::bytesobject::w_bytes_data(b);
    }
    // Tuple keys — element-wise compare via dict_keys_equal recursively.
    if crate::is_tuple(a) && crate::is_tuple(b) {
        let la = crate::w_tuple_len(a);
        let lb = crate::w_tuple_len(b);
        if la != lb {
            return false;
        }
        for i in 0..la {
            let ea = crate::w_tuple_getitem(a, i as i64).unwrap_or(std::ptr::null_mut());
            let eb = crate::w_tuple_getitem(b, i as i64).unwrap_or(std::ptr::null_mut());
            if !dict_keys_equal(ea, eb) {
                return false;
            }
        }
        return true;
    }
    // frozenset / set: element-wise containment via the same equality.
    if crate::is_frozenset(a) && crate::is_frozenset(b) {
        let ai = crate::w_set_items(a);
        let bi = crate::w_set_items(b);
        if ai.len() != bi.len() {
            return false;
        }
        return ai
            .iter()
            .all(|&x| bi.iter().any(|&y| dict_keys_equal(x, y)));
    }
    false
}

/// Get a value by PyObjectRef key.
///
/// When `dict_storage_proxy` is attached, the storage is treated as
/// authoritative for str keys: lookup checks the storage FIRST, so a
/// transient proxy dict whose `entries` Vec carries a stale snapshot
/// (`dict_storage_to_dict` materialisation, `w_module_new`
/// pre-population that was later mutated by a STORE_GLOBAL on the
/// shared storage) returns the live value rather than the cached
/// stale one.  Non-str keys live only in the dstorage Vec because
/// `DictStorage` is str-keyed by construction.
///
/// PyPy parity: `pypy/interpreter/module.py:77 Module.getdict()`
/// returns the live `W_DictMultiObject` whose state IS the module's
/// dict — there is no stale snapshot to worry about because there is
/// only one map.  Pyre's split (dstorage Vec + DictStorage) mirrors
/// the same single-source-of-truth shape only when the storage side
/// wins for the key types it represents.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
/// `pypy/objspace/std/dictmultiobject.py:93-95 W_DictMultiObject.getitem`
/// — `w_dict.get_strategy().getitem(w_dict, w_key)`.  Dispatches
/// through the polymorphic strategy slot so module dicts go through
/// `ModuleDictStrategy::getitem` and regular dicts through
/// `ObjectDictStrategy::getitem` (which honours W_DictObject's
/// `dict_storage_proxy` storage-first contract via
/// `w_dict_lookup_object_strategy`).
pub unsafe fn w_dict_lookup(obj: PyObjectRef, key: PyObjectRef) -> Option<PyObjectRef> {
    w_dict_get_strategy(obj).getitem(obj, key)
}

/// True when a regular dict is still on EmptyDictStrategy or
/// EmptyKwargsDictStrategy.  PyPy's `EmptyDictStrategy.pop` does not
/// hash the key: it returns the provided default, or raises KeyError
/// when no default was passed (`dictmultiobject.py:783-787`).
#[inline]
pub unsafe fn w_dict_is_empty_strategy(obj: PyObjectRef) -> bool {
    if is_module_dict(obj) {
        return false;
    }
    let strategy = (*(obj as *const W_DictObject)).dstrategy;
    strategy_is(strategy, &crate::dictstrategy::EMPTY_DICT_STRATEGY)
        || strategy_is(strategy, &crate::dictstrategy::EMPTY_KWARGS_DICT_STRATEGY)
}

/// Fallible variant of [`w_dict_lookup`].  Propagates hash errors
/// via [`DictKeyError`]; the caller recovers the concrete exception
/// from the interpreter-side thread-local error slot.
///
/// `dictmultiobject.py:93-95 W_DictMultiObject.getitem` — the
/// strategy's `getitem` calls `space.hash_w(w_key)` which may raise.
pub unsafe fn w_dict_lookup_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<Option<PyObjectRef>, DictKeyError> {
    if is_module_dict(obj) {
        return w_module_dict_lookup_inner_checked(obj, key);
    }
    let strategy = (*(obj as *const W_DictObject)).dstrategy;
    if strategy_is(strategy, &crate::dictstrategy::EMPTY_DICT_STRATEGY)
        || strategy_is(strategy, &crate::dictstrategy::EMPTY_KWARGS_DICT_STRATEGY)
    {
        hash_key_checked(key)?;
        return Ok(None);
    }
    if strategy_is(strategy, &crate::dictstrategy::OBJECT_DICT_STRATEGY) {
        return w_dict_lookup_object_strategy_checked(obj, key);
    }
    if strategy_is(strategy, &crate::dictstrategy::BYTES_DICT_STRATEGY) {
        if crate::is_bytes(key) {
            return Ok(w_dict_lookup_bytes_strategy(obj, key));
        }
        if _never_equal_to_string(key) {
            return Ok(None);
        }
        strategy.switch_to_object_strategy(obj);
        return w_dict_lookup_object_strategy_checked(obj, key);
    }
    if strategy_is(strategy, &crate::dictstrategy::UNICODE_DICT_STRATEGY) {
        if crate::is_str(key) {
            return w_dict_lookup_object_strategy_checked(obj, key);
        }
        if _never_equal_to_string(key) {
            return Ok(None);
        }
        w_dict_set_strategy(obj, &crate::dictstrategy::OBJECT_DICT_STRATEGY);
        return w_dict_lookup_object_strategy_checked(obj, key);
    }
    if strategy_is(strategy, &crate::dictstrategy::INT_DICT_STRATEGY) {
        if crate::is_int(key) && !crate::is_bool(key) {
            return Ok(w_dict_lookup_int_strategy(obj, key));
        }
        if _never_equal_to_int(key) {
            return Ok(None);
        }
        strategy.switch_to_object_strategy(obj);
        return w_dict_lookup_object_strategy_checked(obj, key);
    }
    if strategy_is(strategy, &crate::identitydict::IDENTITY_DICT_STRATEGY) {
        if key_compares_by_identity(key) {
            return Ok(strategy.getitem(obj, key));
        }
        strategy.switch_to_object_strategy(obj);
        return w_dict_lookup_object_strategy_checked(obj, key);
    }
    if strategy_is(strategy, &crate::kwargsdict::KWARGS_DICT_STRATEGY) {
        if crate::is_str(key) {
            return Ok(strategy.getitem(obj, key));
        }
        strategy.switch_to_object_strategy(obj);
        return w_dict_lookup_object_strategy_checked(obj, key);
    }
    let result = strategy.getitem(obj, key);
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    Ok(result)
}

/// Internal helper: `ObjectDictStrategy::getitem` body for pyre's
/// W_DictObject — storage-first lookup for str keys when the
/// `dict_storage_proxy` is attached (storage owns the live value for
/// the keys it represents), else linear scan of the dstorage Vec by
/// `dict_keys_equal`.  Called only from the strategy trait impl to
/// avoid recursion through `w_dict_lookup`.  Retires alongside
/// `dict_storage_proxy` when Phase C-1 lands.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_lookup_object_strategy(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Option<PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    // The `&str`-keyed proxy storage cannot represent a lone surrogate, so a
    // surrogate key skips the fast path and falls through to the generic
    // object-keyed entries below (`w_str_get_value_opt` returns None).
    if !dict.dict_storage_proxy.is_null() {
        if let Some(ks) = key_as_utf8(key) {
            if let Some(v) = maybe_lookup_dict_storage(dict.dict_storage_proxy, ks) {
                return Some(v);
            }
        }
    }
    let entries = &*(dict.dstorage as *const indexmap::IndexMap<ObjectKey, PyObjectRef>);
    entries.get(&object_key_for(key)).copied()
}

pub unsafe fn w_dict_lookup_object_strategy_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<Option<PyObjectRef>, DictKeyError> {
    let dict = &*(obj as *const W_DictObject);
    if !dict.dict_storage_proxy.is_null() {
        if let Some(ks) = key_as_utf8(key) {
            if let Some(v) = maybe_lookup_dict_storage(dict.dict_storage_proxy, ks) {
                return Ok(Some(v));
            }
        }
    }
    let entries = &*(dict.dstorage as *const indexmap::IndexMap<ObjectKey, PyObjectRef>);
    let key = object_key_for_checked(key)?;
    let result = entries.get(&key).copied();
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    Ok(result)
}

/// Internal helper: `ModuleDictStrategy::getitem` body for pyre's
/// W_ModuleDictObject — `celldict.py:131-141`:
///   * post-`switch_to_object_strategy` path: walk unified entries
///     Vec via `dict_keys_equal`.
///   * str fast path on fresh ModuleDictStrategy: route to
///     `w_module_dict_getitem_str`.
///   * non-str + never-eq-str: fast-return None.
///   * non-str otherwise: promote then walk entries.
/// Called only from the strategy trait impl to avoid recursion
/// through `w_dict_lookup`.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_lookup_inner(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Option<PyObjectRef> {
    let proxy = (*(obj as *const W_ModuleDictObject)).dict_storage_proxy;
    if let Some(entries) = w_module_dict_object_storage(obj) {
        if let Some(&v) = entries.get(&object_key_for(key)) {
            return Some(v);
        }
        // `celldict.py:173-186 switch_to_object_strategy` replaces the
        // single `w_dict.dstorage` so the generic lookup and the str
        // fast path share the same view.  Pyre keeps the proxy on a
        // sibling slot, so consult it from the generic lookup too —
        // mirrors the str-side fallback in `w_module_dict_getitem_str`
        // (`:947-951`).
        if !proxy.is_null() {
            if let Some(ks) = key_as_utf8(key) {
                if let Some(v) = maybe_lookup_dict_storage(proxy, ks) {
                    return Some(v);
                }
            }
        }
        return None;
    }
    if let Some(ks) = key_as_utf8(key) {
        return w_module_dict_getitem_str(obj, ks);
    }
    if _never_equal_to_string(key) {
        return None;
    }
    w_module_dict_switch_to_object_strategy(obj);
    let entries = w_module_dict_object_storage(obj)?;
    if let Some(&v) = entries.get(&object_key_for(key)) {
        return Some(v);
    }
    if !proxy.is_null() {
        if let Some(ks) = key_as_utf8(key) {
            if let Some(v) = maybe_lookup_dict_storage(proxy, ks) {
                return Some(v);
            }
        }
    }
    None
}

pub unsafe fn w_module_dict_lookup_inner_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<Option<PyObjectRef>, DictKeyError> {
    let proxy = (*(obj as *const W_ModuleDictObject)).dict_storage_proxy;
    if let Some(entries) = w_module_dict_object_storage(obj) {
        let object_key = object_key_for_checked(key)?;
        let hit = entries.get(&object_key).copied();
        if take_dict_key_error() {
            return Err(DictKeyError);
        }
        if let Some(v) = hit {
            return Ok(Some(v));
        }
        if !proxy.is_null() {
            if let Some(ks) = key_as_utf8(key) {
                if let Some(v) = maybe_lookup_dict_storage(proxy, ks) {
                    return Ok(Some(v));
                }
            }
        }
        return Ok(None);
    }
    if let Some(ks) = key_as_utf8(key) {
        return Ok(w_module_dict_getitem_str(obj, ks));
    }
    if _never_equal_to_string(key) {
        return Ok(None);
    }
    w_module_dict_switch_to_object_strategy(obj);
    let entries = w_module_dict_object_storage(obj).ok_or(DictKeyError)?;
    let object_key = object_key_for_checked(key)?;
    let hit = entries.get(&object_key).copied();
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    if let Some(v) = hit {
        return Ok(Some(v));
    }
    Ok(None)
}

/// Set a value by PyObjectRef key.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
/// `pypy/objspace/std/dictmultiobject.py:97-99 W_DictMultiObject.setitem`
/// — `w_dict.get_strategy().setitem(w_dict, w_key, w_value)`.
/// Dispatches through the polymorphic strategy slot so module dicts
/// go through `ModuleDictStrategy::setitem` and regular dicts through
/// `ObjectDictStrategy::setitem` (which honours W_DictObject's
/// `dict_storage_proxy` storage-sync contract via
/// `w_dict_store_object_strategy`).
pub unsafe fn w_dict_store(obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
    w_dict_get_strategy(obj).setitem(obj, key, value)
}

/// Fallible variant of [`w_dict_store`].
///
/// `dictmultiobject.py:97-99 W_DictMultiObject.setitem` — the
/// strategy's `setitem` calls `space.hash_w(w_key)` which may raise.
pub unsafe fn w_dict_store_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), DictKeyError> {
    if is_module_dict(obj) {
        return w_module_dict_store_inner_checked(obj, key, value);
    }
    let strategy = (*(obj as *const W_DictObject)).dstrategy;
    if strategy_is(strategy, &crate::dictstrategy::EMPTY_DICT_STRATEGY) {
        crate::dictstrategy::EMPTY_DICT_STRATEGY.switch_to_correct_strategy(obj, key);
        return w_dict_store_checked(obj, key, value);
    }
    if strategy_is(strategy, &crate::dictstrategy::EMPTY_KWARGS_DICT_STRATEGY) {
        crate::dictstrategy::EMPTY_KWARGS_DICT_STRATEGY.switch_to_correct_strategy(obj, key);
        return w_dict_store_checked(obj, key, value);
    }
    if strategy_is(strategy, &crate::dictstrategy::OBJECT_DICT_STRATEGY) {
        return w_dict_store_object_strategy_checked(obj, key, value);
    }
    if strategy_is(strategy, &crate::dictstrategy::BYTES_DICT_STRATEGY) {
        if crate::is_bytes(key) {
            w_dict_store_bytes_strategy(obj, key, value);
            return Ok(());
        }
        strategy.switch_to_object_strategy(obj);
        return w_dict_store_object_strategy_checked(obj, key, value);
    }
    if strategy_is(strategy, &crate::dictstrategy::UNICODE_DICT_STRATEGY) {
        if crate::is_str(key) {
            return w_dict_store_object_strategy_checked(obj, key, value);
        }
        w_dict_set_strategy(obj, &crate::dictstrategy::OBJECT_DICT_STRATEGY);
        return w_dict_store_object_strategy_checked(obj, key, value);
    }
    if strategy_is(strategy, &crate::dictstrategy::INT_DICT_STRATEGY) {
        if crate::is_int(key) && !crate::is_bool(key) {
            w_dict_store_int_strategy(obj, key, value);
            return Ok(());
        }
        strategy.switch_to_object_strategy(obj);
        return w_dict_store_object_strategy_checked(obj, key, value);
    }
    if strategy_is(strategy, &crate::identitydict::IDENTITY_DICT_STRATEGY) {
        if key_compares_by_identity(key) {
            strategy.setitem(obj, key, value);
            return Ok(());
        }
        strategy.switch_to_object_strategy(obj);
        return w_dict_store_object_strategy_checked(obj, key, value);
    }
    if strategy_is(strategy, &crate::kwargsdict::KWARGS_DICT_STRATEGY) {
        if crate::is_str(key) {
            strategy.setitem(obj, key, value);
            return Ok(());
        }
        strategy.switch_to_object_strategy(obj);
        return w_dict_store_object_strategy_checked(obj, key, value);
    }
    strategy.setitem(obj, key, value);
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    Ok(())
}

/// `dictmultiobject.py:487-493 DictStrategy.setdefault` (base) +
/// `:749-753 EmptyDictStrategy.setdefault` +
/// `:1073-1079 AbstractTypedStrategy.setdefault` — strategy-dispatched
/// setdefault.
///
/// PRE-EXISTING-ADAPTATION: PyPy's `AbstractTypedStrategy.setdefault`
/// is a single `r_dict.setdefault` (one hash).  pyre's typed strategies
/// keep a `dict_storage_proxy` back-mirror that must be looked up first
/// (proxy holds str-key authority) and stored with a sync, so the typed
/// path runs the base two-step (`getitem` + `setitem`) — two hashes on
/// an insert-miss.  This converges to the single op once the proxy
/// retires (R3.5).  The Empty case routes through the checked store
/// (`EmptyDictStrategy.setdefault` → `w_dict.setitem`) so an unhashable
/// key raises TypeError before insertion.
///
/// # Safety
/// `obj`, `key`, `value` must be valid PyObjectRef.
pub unsafe fn w_dict_setdefault_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
) -> Result<PyObjectRef, DictKeyError> {
    if is_module_dict(obj) {
        // `celldict.py:92-105 ModuleDictStrategy.setdefault`: for a str
        // key, grab the cell and return its value if set, else store the
        // default through the cell; non-str keys switch to object
        // strategy.  `w_module_dict_*_inner_checked` carry the
        // str-vs-object dispatch + cell write; pyre fetches the cell in
        // the lookup and again in the store rather than reusing one cell
        // object (`_setitem_str_cell_known`) — a back-mirror artefact
        // that retires with `dict_storage_proxy` (R3.5).
        if let Some(existing) = w_module_dict_lookup_inner_checked(obj, key)? {
            return Ok(existing);
        }
        w_module_dict_store_inner_checked(obj, key, value)?;
        return Ok(value);
    }
    let strategy = (*(obj as *const W_DictObject)).dstrategy;
    // `dictmultiobject.py:749-753 EmptyDictStrategy.setdefault`:
    //   self.switch_to_correct_strategy(w_dict, w_key)
    //   w_dict.setitem(w_key, w_default)
    //   return w_default
    // `w_dict.setitem` is the public CHECKED store, so an unhashable
    // key raises TypeError *before* insertion (parity with the
    // empty-dict setitem rejecting unhashable).  Route through
    // `w_dict_store_checked` and propagate its `Result` directly:
    // `object_key_for_checked` consumes the hook error slot, so a
    // post-hoc `take_hash_error()` would observe no pending error.
    if strategy_is(strategy, &crate::dictstrategy::EMPTY_DICT_STRATEGY) {
        crate::dictstrategy::EMPTY_DICT_STRATEGY.switch_to_correct_strategy(obj, key);
        w_dict_store_checked(obj, key, value)?;
        return Ok(value);
    }
    if strategy_is(strategy, &crate::dictstrategy::EMPTY_KWARGS_DICT_STRATEGY) {
        crate::dictstrategy::EMPTY_KWARGS_DICT_STRATEGY.switch_to_correct_strategy(obj, key);
        w_dict_store_checked(obj, key, value)?;
        return Ok(value);
    }
    let result = strategy.setdefault(obj, key, value);
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    Ok(result)
}

/// `dictmultiobject.py:624-634 DictStrategy.pop` (base) +
/// `:783-787 EmptyDictStrategy.pop` +
/// `:1123-1138 AbstractTypedStrategy.pop` — strategy-dispatched pop.
///
/// PRE-EXISTING-ADAPTATION: PyPy's `AbstractTypedStrategy.pop` is a
/// single `r_dict.pop` (one hash).  pyre's typed strategies run the
/// base two-step (`getitem` + `delitem`) so the `dict_storage_proxy`
/// back-mirror stays in sync; this converges to the single op once the
/// proxy retires (R3.5).  `EmptyDictStrategy.pop` does NOT hash the key
/// (`:783-787` returns default / raises KeyError directly).
///
/// Returns `Ok(Some(value))` on hit, `Ok(None)` on miss (caller
/// handles default/KeyError), or `Err(DictKeyError)` on hash error.
///
/// # Safety
/// `obj`, `key` must be valid PyObjectRef.
pub unsafe fn w_dict_pop_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<Option<PyObjectRef>, DictKeyError> {
    if is_module_dict(obj) {
        match w_module_dict_lookup_inner_checked(obj, key)? {
            Some(val) => {
                w_module_dict_delitem_inner_checked(obj, key)?;
                Ok(Some(val))
            }
            None => Ok(None),
        }
    } else {
        let strategy = (*(obj as *const W_DictObject)).dstrategy;
        match strategy.pop(obj, key, None) {
            Ok(val) => {
                if take_dict_key_error() {
                    return Err(DictKeyError);
                }
                Ok(Some(val))
            }
            Err(()) => {
                if take_dict_key_error() {
                    return Err(DictKeyError);
                }
                Ok(None)
            }
        }
    }
}

/// Internal helper: `ObjectDictStrategy::setitem` body for pyre's
/// W_DictObject — walks the dstorage Vec by `dict_keys_equal`,
/// updates the matching entry or pushes a new one, fires the GC
/// write barrier, and syncs the change into the
/// `dict_storage_proxy` back-mirror.  Called only from the strategy
/// trait impl to avoid recursion through `w_dict_store`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_store_object_strategy(obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<ObjectKey, PyObjectRef>);
    entries.insert(object_key_for(key), value);
    dict_write_barrier(obj);
    maybe_sync_dict_storage_store(dict.dict_storage_proxy, key, value);
}

pub unsafe fn w_dict_store_object_strategy_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), DictKeyError> {
    let object_key = object_key_for_checked(key)?;
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<ObjectKey, PyObjectRef>);
    // Single setitem probe (matches `r_dict.setitem`'s one bucket scan).
    // When `space.eq_w` raises mid-probe the comparison reads as "not
    // equal", so `insert` finds no match and appends a fresh (spurious)
    // entry at the end; drop it with `pop` so the store leaves the dict
    // unchanged, matching r_dict raising at the comparison without
    // completing the store.  A no-raise store touches the bucket once.
    entries.insert(object_key, value);
    if take_dict_key_error() {
        entries.pop();
        return Err(DictKeyError);
    }
    dict_write_barrier(obj);
    maybe_sync_dict_storage_store(dict.dict_storage_proxy, key, value);
    Ok(())
}

/// Internal helper: `ModuleDictStrategy::setitem` body for pyre's
/// W_ModuleDictObject — `celldict.py:41-67`:
///   * str keys on a fresh ModuleDictStrategy → `setitem_str`
///   * non-str keys OR already-promoted → `switch_to_object_strategy`
///     and write into the unified dstorage Vec, syncing str keys
///     back to the proxy.
/// Called only from the strategy trait impl to avoid recursion
/// through `w_dict_store`.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_store_inner(obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
    if !w_module_dict_is_object_strategy(obj) {
        if let Some(ks) = key_as_utf8(key) {
            return w_module_dict_setitem_str(obj, ks, value);
        }
    }
    if !w_module_dict_is_object_strategy(obj) {
        w_module_dict_switch_to_object_strategy(obj);
    }
    let proxy = (*(obj as *const W_ModuleDictObject)).dict_storage_proxy;
    let entries = w_module_dict_object_storage_mut(obj);
    entries.insert(object_key_for(key), value);
    let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
    strategy.mutated();
    if crate::is_str(key) {
        maybe_sync_dict_storage_store(proxy, key, value);
    }
}

pub unsafe fn w_module_dict_store_inner_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), DictKeyError> {
    if !w_module_dict_is_object_strategy(obj) {
        if let Some(ks) = key_as_utf8(key) {
            w_module_dict_setitem_str(obj, ks, value);
            return Ok(());
        }
    }
    if !w_module_dict_is_object_strategy(obj) {
        w_module_dict_switch_to_object_strategy(obj);
    }
    let object_key = object_key_for_checked(key)?;
    let proxy = (*(obj as *const W_ModuleDictObject)).dict_storage_proxy;
    let entries = w_module_dict_object_storage_mut(obj);
    // Single setitem probe; on an `__eq__` raise mid-probe `insert` appends
    // a spurious entry, so drop it with `pop` and leave the dict unchanged
    // (see `w_dict_store_object_strategy_checked`).
    entries.insert(object_key, value);
    if take_dict_key_error() {
        entries.pop();
        return Err(DictKeyError);
    }
    let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
    strategy.mutated();
    if crate::is_str(key) {
        maybe_sync_dict_storage_store(proxy, key, value);
    }
    Ok(())
}

/// Write a str-keyed assignment back to the dict's backing DictStorage,
/// if any. Declared in pyre-interpreter and re-exported via an `extern`
/// hook registered at startup to avoid a circular dependency.
unsafe fn maybe_sync_dict_storage_store(ns_ptr: *mut u8, key: PyObjectRef, value: PyObjectRef) {
    if ns_ptr.is_null() {
        return;
    }
    // A lone-surrogate key cannot live in the `&str`-keyed proxy storage, so
    // there is nothing to sync (the object-strategy entries hold it instead).
    let Some(name) = key_as_utf8(key) else {
        return;
    };
    if let Some(hook) = DICT_STORAGE_STORE_HOOK
        .load(std::sync::atomic::Ordering::Acquire)
        .as_ref()
    {
        hook(ns_ptr, name, value);
    }
}

/// Mirror of `maybe_sync_dict_storage_store` for deletions.  When a dict
/// with a backing storage drops a str-keyed entry, propagate the
/// deletion so storage-keyed lookups (LOAD_GLOBAL builtins fallback)
/// stop seeing the stale entry.  PyPy keeps everything in one
/// `W_DictMultiObject` so this asymmetry is pyre-only.
unsafe fn maybe_sync_dict_storage_delete(ns_ptr: *mut u8, key_str: &str) {
    if ns_ptr.is_null() {
        return;
    }
    if let Some(hook) = DICT_STORAGE_DELETE_HOOK
        .load(std::sync::atomic::Ordering::Acquire)
        .as_ref()
    {
        hook(ns_ptr, key_str);
    }
}

/// Storage-proxy read-through.  PyPy keeps every dict-backed lookup
/// inside the same `W_DictMultiObject`, so reads see entries
/// regardless of which interpreter side wrote them.  Pyre splits the
/// dict's `entries` Vec from the `DictStorage` proxy, so a dict whose
/// authoritative state lives in the storage (`Module.w_dict` over
/// `space.builtin`'s storage, `globals()` view) must surface storage
/// entries on read.  Returns `None` when no proxy is attached or the
/// hook is unregistered.
unsafe fn maybe_lookup_dict_storage(ns_ptr: *mut u8, key_str: &str) -> Option<PyObjectRef> {
    if ns_ptr.is_null() {
        return None;
    }
    let ptr = DICT_STORAGE_LOOKUP_HOOK.load(std::sync::atomic::Ordering::Acquire);
    if ptr.is_null() {
        return None;
    }
    (*ptr)(ns_ptr, key_str)
}

type NamespaceStoreHook = unsafe fn(*mut u8, &str, PyObjectRef);
type NamespaceDeleteHook = unsafe fn(*mut u8, &str);
type NamespaceLookupHook = unsafe fn(*mut u8, &str) -> Option<PyObjectRef>;
type NamespaceItemsHook = unsafe fn(*mut u8) -> Vec<(String, PyObjectRef)>;

struct AtomicHookPtr(std::sync::atomic::AtomicPtr<NamespaceStoreHook>);

impl AtomicHookPtr {
    const fn new() -> Self {
        Self(std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()))
    }

    fn store(&self, hook: NamespaceStoreHook) {
        // Leak a boxed function pointer so the pointer lives for the entire
        // process lifetime; this matches PyPy's one-time interp init.
        // `flavor='raw'` because this is host-side dispatch state, not a
        // GC-managed Python object.
        let raw = crate::lltype::malloc_raw(hook);
        self.0.store(raw, std::sync::atomic::Ordering::Release);
    }

    fn load(&self, order: std::sync::atomic::Ordering) -> *const NamespaceStoreHook {
        self.0.load(order) as *const NamespaceStoreHook
    }
}

struct AtomicDeleteHookPtr(std::sync::atomic::AtomicPtr<NamespaceDeleteHook>);

impl AtomicDeleteHookPtr {
    const fn new() -> Self {
        Self(std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()))
    }

    fn store(&self, hook: NamespaceDeleteHook) {
        let raw = crate::lltype::malloc_raw(hook);
        self.0.store(raw, std::sync::atomic::Ordering::Release);
    }

    fn load(&self, order: std::sync::atomic::Ordering) -> *const NamespaceDeleteHook {
        self.0.load(order) as *const NamespaceDeleteHook
    }
}

struct AtomicLookupHookPtr(std::sync::atomic::AtomicPtr<NamespaceLookupHook>);

impl AtomicLookupHookPtr {
    const fn new() -> Self {
        Self(std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()))
    }

    fn store(&self, hook: NamespaceLookupHook) {
        let raw = crate::lltype::malloc_raw(hook);
        self.0.store(raw, std::sync::atomic::Ordering::Release);
    }

    fn load(&self, order: std::sync::atomic::Ordering) -> *const NamespaceLookupHook {
        self.0.load(order) as *const NamespaceLookupHook
    }
}

struct AtomicItemsHookPtr(std::sync::atomic::AtomicPtr<NamespaceItemsHook>);

impl AtomicItemsHookPtr {
    const fn new() -> Self {
        Self(std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()))
    }

    fn store(&self, hook: NamespaceItemsHook) {
        let raw = crate::lltype::malloc_raw(hook);
        self.0.store(raw, std::sync::atomic::Ordering::Release);
    }

    fn load(&self, order: std::sync::atomic::Ordering) -> *const NamespaceItemsHook {
        self.0.load(order) as *const NamespaceItemsHook
    }
}

static DICT_STORAGE_STORE_HOOK: AtomicHookPtr = AtomicHookPtr::new();
static DICT_STORAGE_DELETE_HOOK: AtomicDeleteHookPtr = AtomicDeleteHookPtr::new();
static DICT_STORAGE_LOOKUP_HOOK: AtomicLookupHookPtr = AtomicLookupHookPtr::new();
static DICT_STORAGE_ITEMS_HOOK: AtomicItemsHookPtr = AtomicItemsHookPtr::new();

/// Register the interpreter-level hook that writes (name, value) into a
/// DictStorage. Called once during interpreter startup.
pub fn register_dict_storage_store_hook(hook: NamespaceStoreHook) {
    DICT_STORAGE_STORE_HOOK.store(hook);
}

/// Register the interpreter-level hook that deletes `name` from a
/// DictStorage. Called once during interpreter startup.
pub fn register_dict_storage_delete_hook(hook: NamespaceDeleteHook) {
    DICT_STORAGE_DELETE_HOOK.store(hook);
}

/// Register the interpreter-level hook that looks up `name` in a
/// DictStorage and returns its value (or `None`).  Called once during
/// interpreter startup so dicts with a `dict_storage_proxy` surface
/// storage entries on read miss.
pub fn register_dict_storage_lookup_hook(hook: NamespaceLookupHook) {
    DICT_STORAGE_LOOKUP_HOOK.store(hook);
}

/// Register the interpreter-level hook that enumerates all str-keyed
/// `(name, value)` pairs from a DictStorage.  Used by `w_dict_len`,
/// `w_dict_items`, `w_dict_str_entries` and `w_dict_delitem_str` to keep
/// the full dict protocol (`len(module.__dict__)`, `module.__dict__.items()`,
/// `del module.__dict__[name]`) consistent with PyPy's
/// `Module.getdict()` returning the live W_DictMultiObject — pyre splits
/// the proxy off the dstorage Vec and would otherwise miss every storage
/// entry not yet mirrored into the W_DictObject.
pub fn register_dict_storage_items_hook(hook: NamespaceItemsHook) {
    DICT_STORAGE_ITEMS_HOOK.store(hook);
}

/// Read-side counterpart of `maybe_sync_dict_storage_store`: enumerate
/// the str-keyed entries currently in the backing storage.
///
/// Returns `None` when the items hook has not yet been registered (the
/// hookless case fires for direct `w_module_new` callers and for unit
/// tests that exercise dict surfaces before `register_dict_storage_*_hook`
/// runs).  Callers must distinguish "hook missing" (storage view is
/// indeterminate, fall back to dstorage Vec) from "hook installed, storage
/// empty" (authoritative empty result) — collapsing the two would silently
/// drop dstorage Vec str keys for proxied dicts whose hook arrives later
/// in the bootstrap.
unsafe fn maybe_items_dict_storage(ns_ptr: *mut u8) -> Option<Vec<(String, PyObjectRef)>> {
    if ns_ptr.is_null() {
        return Some(Vec::new());
    }
    let ptr = DICT_STORAGE_ITEMS_HOOK.load(std::sync::atomic::Ordering::Acquire);
    if ptr.is_null() {
        return None;
    }
    Some((*ptr)(ns_ptr))
}

/// Get the dict_storage_proxy pointer from a dict (used by interpreter for
/// live globals sync).
///
/// Both W_DictObject and W_ModuleDictObject carry a `dict_storage_proxy`
/// field (TODO until
/// `PyFrame.w_globals` to `PyObjectRef`); this returns whichever side
/// is attached so callers like `exec(..., module.__dict__)` can re-use
/// the existing backing storage instead of materialising a fresh
/// `DictStorage` that drifts from the module dict.
pub unsafe fn w_dict_get_dict_storage_proxy(obj: PyObjectRef) -> *mut u8 {
    if is_module_dict(obj) {
        return (*(obj as *const W_ModuleDictObject)).dict_storage_proxy;
    }
    (*(obj as *const W_DictObject)).dict_storage_proxy
}

/// Attach a `DictStorage` proxy to an existing dict so subsequent
/// mutations sync into the storage.  Used by interpreter-level
/// `pick_builtin` (`pypy/module/__builtin__/moduledef.py:102-103`)
/// to lift an arbitrary user-supplied `__builtins__` dict into pyre's
/// storage-keyed lookup model — counterpart of the no-op assignment
/// `module.Module(space, None, w_builtin)` does in PyPy by aliasing
/// `module.w_dict = w_builtin`.
pub unsafe fn w_dict_set_dict_storage_proxy(obj: PyObjectRef, ns: *mut u8) {
    if is_module_dict(obj) {
        (*(obj as *mut W_ModuleDictObject)).dict_storage_proxy = ns;
        return;
    }
    (*(obj as *mut W_DictObject)).dict_storage_proxy = ns;
}

/// True only for a regular `W_DictObject` on EmptyDictStrategy with no
/// storage-proxy back-mirror. `W_ModuleDictObject` also surfaces as a
/// Python `dict`, but it has a different Rust layout and must take the
/// polymorphic item-loop path.
///
/// Mirrors the destination test in
/// `dictmultiobject.py:1401 update1_dict_dict` while excluding pyre's
/// proxy-attached adaptation from the storage-transplant fast path.
pub unsafe fn w_dict_is_regular_empty_no_proxy(obj: PyObjectRef) -> bool {
    if is_module_dict(obj) {
        return false;
    }
    let dict = &*(obj as *const W_DictObject);
    dict.dstrategy.strategy_kind() == crate::dictstrategy::StrategyKind::Empty
        && dict.dict_storage_proxy.is_null()
}

/// Adopt a freshly copied regular-dict storage into an empty regular
/// destination for `dict.update(dict)`'s PyPy fast path.
///
/// PyPy `update1_dict_dict` performs:
/// `w_copy = w_data.get_strategy().copy(w_data); w_dict.set_strategy(...);
/// w_dict.dstorage = w_copy.dstorage`. Pyre keeps a regular empty dict's
/// placeholder storage allocated, so this helper drops that placeholder,
/// installs the copy's strategy/storage/len, and fires the explicit GC
/// write barrier that RPython field stores would get from the GC.
pub unsafe fn w_dict_adopt_regular_copy_for_empty_update(dst: PyObjectRef, w_copy: PyObjectRef) {
    debug_assert!(!is_module_dict(dst));
    debug_assert!(!is_module_dict(w_copy));

    let dst_dict = &mut *(dst as *mut W_DictObject);
    let copy_dict = &*(w_copy as *const W_DictObject);
    let old_dstorage = dst_dict.dstorage;

    dst_dict.dstrategy = copy_dict.dstrategy;
    dst_dict.dstorage = copy_dict.dstorage;
    dict_write_barrier(dst);

    if !old_dstorage.is_null() {
        drop(Box::from_raw(
            old_dstorage as *mut indexmap::IndexMap<ObjectKey, PyObjectRef>,
        ));
    }
}

/// Get a value by int key (convenience wrapper).  Wraps the raw i64
/// into a `W_IntObject` and dispatches through `w_dict_lookup` so the
/// strategy-slot path applies uniformly to W_DictObject and
/// W_ModuleDictObject.
pub unsafe fn w_dict_getitem(obj: PyObjectRef, key: i64) -> Option<PyObjectRef> {
    w_dict_lookup(obj, crate::w_int_new(key))
}

/// Set a value by int key (convenience wrapper).
pub unsafe fn w_dict_setitem(obj: PyObjectRef, key: i64, value: PyObjectRef) {
    w_dict_store(obj, crate::w_int_new(key), value)
}

/// `pypy/objspace/std/dictmultiobject.py:103-105 W_DictMultiObject.getitem_str`
/// — `w_dict.get_strategy().getitem_str(w_dict, key)`.  Dispatches
/// through the polymorphic strategy slot so module dicts go through
/// `ModuleDictStrategy::getitem_str` (which honours
/// `dict_storage_proxy` + the `mstrategy.caches` cell-cache) and
/// regular dicts through `ObjectDictStrategy::getitem_str` (which
/// honours W_DictObject's proxy via `w_dict_getitem_str_proxy_first`).
pub unsafe fn w_dict_getitem_str(obj: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    w_dict_get_strategy(obj).getitem_str(obj, key)
}

/// Internal helper: `ObjectDictStrategy::getitem_str` proxy-first
/// lookup body.  Kept as a free function so the strategy trait
/// impl can stay in `dictstrategy.rs` without re-implementing the
/// `dict_storage_proxy` walk.  Mirrors `w_dict_lookup`'s
/// storage-first contract for proxied W_DictObjects so stale
/// `entries` snapshots (e.g. `dict_storage_to_dict` materialisation)
/// don't shadow live storage updates.  Retires alongside
/// `dict_storage_proxy` when Phase C-1 lands.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.  The sole caller is
/// `ObjectDictStrategy::getitem_str`, which only fires when the
/// receiver's `dstrategy` slot is `OBJECT_DICT_STRATEGY`; module
/// dicts route through `ModuleDictStrategy::getitem_str` →
/// `w_module_dict_getitem_str` instead.
pub unsafe fn w_dict_getitem_str_proxy_first(obj: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    if !dict.dict_storage_proxy.is_null() {
        if let Some(v) = maybe_lookup_dict_storage(dict.dict_storage_proxy, key) {
            return Some(v);
        }
    }
    let entries = &*(dict.dstorage as *const indexmap::IndexMap<ObjectKey, PyObjectRef>);
    // Use the str key directly via `object_key_for`; under the
    // `dict_keys_equal` hash/eq pair, str-keyed lookups hash on the
    // str hash and compare on `w_str_get_value` equality.
    entries.get(&object_key_for(crate::w_str_new(key))).copied()
}

/// `pypy/objspace/std/dictmultiobject.py:111-112 W_DictMultiObject.setitem_str`
/// — `w_dict.get_strategy().setitem_str(w_dict, key, value)`.
/// Dispatches through the polymorphic strategy slot so module dicts
/// fan out via `ModuleDictStrategy::setitem_str` and regular dicts
/// via `ObjectDictStrategy::setitem_str`.
pub unsafe fn w_dict_setitem_str(obj: PyObjectRef, key: &str, value: PyObjectRef) {
    w_dict_get_strategy(obj).setitem_str(obj, key, value)
}

/// Set a value by str key WITHOUT firing the dict_storage_proxy
/// store hook.  Used by the storage→W_DictObject back-mirror so a
/// `dict_storage_store` on a storage that has a registered mirror
/// target updates the W_DictObject's dstorage Vec without
/// re-entering `maybe_sync_dict_storage_store` (which would feed
/// the same write right back into the storage and create an
/// observable double-invalidation of slot watchers).
///
/// PyPy keeps everything in one `W_DictMultiObject`, so the
/// asymmetric "dstorage Vec write must skip storage notification"
/// shape is pyre-only; the no-proxy variant is the structural
/// adapter for the bidirectional sync that PyPy gets for free.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_setitem_str_no_proxy(obj: PyObjectRef, key: &str, value: PyObjectRef) {
    if is_module_dict(obj) {
        // The "no-proxy" qualifier means "do not bounce the write
        // back to a sibling DictStorage".  Route through the
        // W_ModuleDictObject's matching no-proxy setter so storage's
        // mirror_target back-write doesn't loop.
        return w_module_dict_setitem_str_no_proxy(obj, key, value);
    }
    let dict = &mut *(obj as *mut W_DictObject);
    // `dictmultiobject.py:759-761 EmptyDictStrategy.setitem_str`
    // promotes the dict to UnicodeDictStrategy before writing; without
    // this, the strategy readers (length/items/getitem) return empty
    // even after entries have been pushed into the dstorage Vec.
    // UnicodeDictStrategy stores `Vec<(PyObjectRef, PyObjectRef)>`
    // (same shape as ObjectDictStrategy), so the existing raw Vec
    // walk below is correct once the strategy slot points at it.
    // Discriminate by `strategy_kind()`, never by `std::ptr::eq` on the
    // strategy address: the strategy singletons are zero-sized statics
    // whose addresses the compiler is free to coalesce, so pointer
    // identity cannot tell `Empty`/`Unicode`/`Object` apart.
    use crate::dictstrategy::StrategyKind;
    match dict.dstrategy.strategy_kind() {
        StrategyKind::Empty => {
            dict.dstrategy = &crate::dictstrategy::UNICODE_DICT_STRATEGY;
        }
        StrategyKind::Unicode | StrategyKind::Object => {}
        // Int / Bytes / Identity / Kwargs typed storage: PyPy's
        // `AbstractTypedStrategy.setitem_str` (`dictmultiobject.py:1069`)
        // promotes to ObjectDictStrategy before the str-key insert.
        // Polymorphic dispatch through the trait converts the typed
        // backing to `Vec<(PyObjectRef, PyObjectRef)>` so the walk
        // below sees the correct layout.
        _ => dict.dstrategy.switch_to_object_strategy(obj),
    }
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<ObjectKey, PyObjectRef>);
    let w_key = crate::w_str_new(key);
    entries.insert(object_key_for(w_key), value);
    dict_write_barrier(obj);
}

/// Remove an entry by str key WITHOUT firing the dict_storage_proxy
/// delete hook.  Counterpart of `w_dict_setitem_str_no_proxy`; see
/// that doc-comment for the back-mirror rationale.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_delitem_str_no_proxy(obj: PyObjectRef, key: &str) -> bool {
    if is_module_dict(obj) {
        return w_module_dict_delitem_str_no_proxy(obj, key).is_some();
    }
    let dict = &mut *(obj as *mut W_DictObject);
    // Mirror `w_dict_setitem_str_no_proxy`: if the dict is on a typed
    // strategy whose backing isn't `Vec<(PyObjectRef, PyObjectRef)>`,
    // promote to ObjectDictStrategy first so the raw Vec walk below
    // sees the right layout.  PyPy's typed `delitem` (`dictmultiobject.py:1081-1087`)
    // promotes on key-type mismatch; we promote on the str-keyed
    // back-mirror path for the same reason.
    // Discriminate by `strategy_kind()`; the strategy singletons are
    // zero-sized statics whose addresses can coalesce, so `std::ptr::eq`
    // on them is unreliable.
    use crate::dictstrategy::StrategyKind;
    match dict.dstrategy.strategy_kind() {
        StrategyKind::Empty | StrategyKind::Unicode | StrategyKind::Object => {}
        _ => dict.dstrategy.switch_to_object_strategy(obj),
    }
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<ObjectKey, PyObjectRef>);
    let w_key = crate::w_str_new(key);
    entries.shift_remove(&object_key_for(w_key)).is_some()
}

/// WTF-8 keyed sibling of `w_dict_getitem_str` — routes through the
/// generic object-key lookup so a lone-surrogate key resolves on the
/// `ObjectDictStrategy` entries map.  `w_str_get_value` (used by the
/// Unicode strategy's str fast path) panics on a lone surrogate, so the
/// str-keyed wrapper cannot be used for such names.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_getitem_wtf8(
    obj: PyObjectRef,
    key: &rustpython_wtf8::Wtf8,
) -> Option<PyObjectRef> {
    let w_key = crate::w_str_from_wtf8(key.to_wtf8_buf());
    w_dict_lookup(obj, w_key)
}

/// WTF-8 keyed equivalent of `space.setitem_str` — `setitem_str` is itself
/// a fast path of `space.setitem`, so a key that is valid UTF-8 takes the
/// str fast path (keeping an ASCII/Unicode dict on its strategy) and a
/// lone-surrogate key wraps into a `W_StrObject` and routes through the
/// general `w_dict_store` (`space.setitem`).  Unlike the back-mirror
/// `_no_proxy` helper this carries no module-dict exception, matching
/// `space.setitem_str`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_setitem_wtf8(
    obj: PyObjectRef,
    key: &rustpython_wtf8::Wtf8,
    value: PyObjectRef,
) {
    match key.as_str() {
        Ok(s) => w_dict_setitem_str(obj, s, value),
        Err(_) => w_dict_store(obj, crate::w_str_from_wtf8(key.to_wtf8_buf()), value),
    }
}

/// WTF-8 keyed sibling of `w_dict_setitem_str_no_proxy` — lets a
/// lone-surrogate namespace key reach the back-mirror `W_DictObject`.
/// The dict is forced onto `ObjectDictStrategy` (ObjectKey-keyed via
/// hashed WTF-8 bytes); the Unicode strategy's str fast paths would
/// hit `w_str_get_value`, which panics on a lone surrogate.  Only the
/// regular (Instance) `W_DictObject` shape back-mirrors a type
/// namespace, so the module-dict case is a no-op.
///
/// # Safety
/// `obj` must point to a valid regular `W_DictObject`.
pub unsafe fn w_dict_setitem_wtf8_no_proxy(
    obj: PyObjectRef,
    key: &rustpython_wtf8::Wtf8,
    value: PyObjectRef,
) {
    if is_module_dict(obj) {
        return;
    }
    let dict = &mut *(obj as *mut W_DictObject);
    // A lone-surrogate key can only live on `ObjectDictStrategy`; force
    // the switch unless already there.  Detect via `strategy_kind()` —
    // `std::ptr::eq` on the zero-sized strategy statics is unreliable.
    if dict.dstrategy.strategy_kind() != crate::dictstrategy::StrategyKind::Object {
        dict.dstrategy.switch_to_object_strategy(obj);
    }
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<ObjectKey, PyObjectRef>);
    let w_key = crate::w_str_from_wtf8(key.to_wtf8_buf());
    entries.insert(object_key_for(w_key), value);
    dict_write_barrier(obj);
}

/// WTF-8 keyed sibling of `w_dict_delitem_str_no_proxy`.
///
/// # Safety
/// `obj` must point to a valid regular `W_DictObject`.
pub unsafe fn w_dict_delitem_wtf8_no_proxy(obj: PyObjectRef, key: &rustpython_wtf8::Wtf8) -> bool {
    if is_module_dict(obj) {
        return false;
    }
    let dict = &mut *(obj as *mut W_DictObject);
    // Detect via `strategy_kind()`; the strategy singletons are
    // zero-sized statics whose addresses can coalesce, so `std::ptr::eq`
    // on them is unreliable (an `Object`-strategy dict could otherwise
    // be misread as `Empty` and skip the removal entirely).
    use crate::dictstrategy::StrategyKind;
    match dict.dstrategy.strategy_kind() {
        StrategyKind::Empty => return false,
        StrategyKind::Object => {}
        _ => dict.dstrategy.switch_to_object_strategy(obj),
    }
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<ObjectKey, PyObjectRef>);
    let w_key = crate::w_str_from_wtf8(key.to_wtf8_buf());
    entries.shift_remove(&object_key_for(w_key)).is_some()
}

/// `pypy/objspace/std/dictmultiobject.py:469-471 W_DictMultiObject.descr_delitem`
/// — `space.delitem(w_dict, w_key)` which calls
/// `w_dict.get_strategy().delitem(w_dict, w_key)`.  PyPy has no
/// `delitem_str` fast path on the trait (only `setitem_str` and
/// `getitem_str`); pyre's str-keyed convenience wrapper routes
/// through `w_dict_delitem(obj, w_str_new(key))` for parity.
///
/// Returns `true` when the key was present in either the strategy
/// storage or the `dict_storage_proxy` back-mirror (PRE-EXISTING-
/// ADAPTATION storage view).
pub unsafe fn w_dict_delitem_str(obj: PyObjectRef, key: &str) -> bool {
    let w_key = crate::w_str_new(key);
    w_dict_delitem(obj, w_key)
}

/// Remove an entry by arbitrary key (str or non-str).  Returns `true`
/// when the key was present.  Mirrors `pypy/objspace/std/dictmultiobject.py
/// W_DictMultiObject.delitem` — PyPy's flat strategy walks the entries
/// list, comparing each key by `space.eq_w` (the `dict_keys_equal`
/// helper here), so `del d[1]` on an int key removes the entry.  The
/// previous str-only `w_dict_delitem_str` left non-str entries
/// untouched, breaking `dict.pop(int_key)`'s after-pop deletion.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
/// `pypy/objspace/std/dictmultiobject.py:101-102 W_DictMultiObject.delitem`
/// — `w_dict.get_strategy().delitem(w_dict, w_key)`.  Dispatches
/// through the polymorphic strategy slot so module dicts go through
/// `ModuleDictStrategy::delitem` and regular dicts through
/// `ObjectDictStrategy::delitem` (which honours W_DictObject's
/// `dict_storage_proxy` storage-sync contract via
/// `w_dict_delitem_object_strategy`).
pub unsafe fn w_dict_delitem(obj: PyObjectRef, key: PyObjectRef) -> bool {
    w_dict_get_strategy(obj).delitem(obj, key)
}

/// Fallible variant of [`w_dict_delitem`].
///
/// `dictmultiobject.py:101-102 W_DictMultiObject.delitem` — the
/// strategy's `delitem` calls `space.hash_w(w_key)` which may raise.
pub unsafe fn w_dict_delitem_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<bool, DictKeyError> {
    if is_module_dict(obj) {
        return w_module_dict_delitem_inner_checked(obj, key);
    }
    let strategy = (*(obj as *const W_DictObject)).dstrategy;
    if strategy_is(strategy, &crate::dictstrategy::EMPTY_DICT_STRATEGY)
        || strategy_is(strategy, &crate::dictstrategy::EMPTY_KWARGS_DICT_STRATEGY)
    {
        hash_key_checked(key)?;
        return Ok(false);
    }
    if strategy_is(strategy, &crate::dictstrategy::OBJECT_DICT_STRATEGY) {
        return w_dict_delitem_object_strategy_checked(obj, key);
    }
    if strategy_is(strategy, &crate::dictstrategy::BYTES_DICT_STRATEGY) {
        if crate::is_bytes(key) {
            return Ok(w_dict_delitem_bytes_strategy(obj, key));
        }
        strategy.switch_to_object_strategy(obj);
        return w_dict_delitem_object_strategy_checked(obj, key);
    }
    if strategy_is(strategy, &crate::dictstrategy::UNICODE_DICT_STRATEGY) {
        if crate::is_str(key) {
            return w_dict_delitem_object_strategy_checked(obj, key);
        }
        w_dict_set_strategy(obj, &crate::dictstrategy::OBJECT_DICT_STRATEGY);
        return w_dict_delitem_object_strategy_checked(obj, key);
    }
    if strategy_is(strategy, &crate::dictstrategy::INT_DICT_STRATEGY) {
        if crate::is_int(key) && !crate::is_bool(key) {
            return Ok(w_dict_delitem_int_strategy(obj, key));
        }
        strategy.switch_to_object_strategy(obj);
        return w_dict_delitem_object_strategy_checked(obj, key);
    }
    if strategy_is(strategy, &crate::identitydict::IDENTITY_DICT_STRATEGY) {
        if key_compares_by_identity(key) {
            return Ok(strategy.delitem(obj, key));
        }
        strategy.switch_to_object_strategy(obj);
        return w_dict_delitem_object_strategy_checked(obj, key);
    }
    if strategy_is(strategy, &crate::kwargsdict::KWARGS_DICT_STRATEGY) {
        strategy.switch_to_object_strategy(obj);
        return w_dict_delitem_object_strategy_checked(obj, key);
    }
    let removed = strategy.delitem(obj, key);
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    Ok(removed)
}

/// Internal helper: `ObjectDictStrategy::delitem` body for pyre's
/// W_DictObject — dstorage Vec scan + remove + len decrement + GC
/// reporting via the `dict_storage_proxy` storage sync (str keys
/// only).  Called only from the strategy trait impl to avoid
/// recursion through `w_dict_delitem`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_delitem_object_strategy(obj: PyObjectRef, key: PyObjectRef) -> bool {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<ObjectKey, PyObjectRef>);
    let mut hit = false;
    if entries.shift_remove(&object_key_for(key)).is_some() {
        hit = true;
    }
    // String-key delete must also flow into the storage proxy so
    // `del module.__dict__[name]` and `globals().pop(name)` clear the
    // backing namespace too.
    if !dict.dict_storage_proxy.is_null() {
        if let Some(key_str) = key_as_utf8(key) {
            if !hit && maybe_lookup_dict_storage(dict.dict_storage_proxy, key_str).is_some() {
                hit = true;
            }
            maybe_sync_dict_storage_delete(dict.dict_storage_proxy, key_str);
        }
    }
    hit
}

pub unsafe fn w_dict_delitem_object_strategy_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<bool, DictKeyError> {
    let object_key = object_key_for_checked(key)?;
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<ObjectKey, PyObjectRef>);
    let mut hit = false;
    // `shift_remove` removes only on a positive match; an `__eq__` raise
    // reads as "not equal" (and short-circuits the rest of the probe), so
    // the key is never found and the dict is left unchanged.  Reporting the
    // error before touching the storage proxy is therefore the
    // first-exception, no-mutation path that `r_dict.delitem` raises on.
    if entries.shift_remove(&object_key).is_some() {
        hit = true;
    }
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    if !dict.dict_storage_proxy.is_null() {
        if let Some(key_str) = key_as_utf8(key) {
            if !hit && maybe_lookup_dict_storage(dict.dict_storage_proxy, key_str).is_some() {
                hit = true;
            }
            maybe_sync_dict_storage_delete(dict.dict_storage_proxy, key_str);
        }
    }
    Ok(hit)
}

/// Internal helper: `ModuleDictStrategy::delitem` body for pyre's
/// W_ModuleDictObject — `celldict.py:106-126`:
///   * post-`switch_to_object_strategy`: walk unified dstorage Vec
///     via `dict_keys_equal`, remove, mutated(), sync str keys
///     back through proxy.
///   * str fast path on fresh ModuleDictStrategy → delitem_str.
///   * non-str + never-eq-str: fast-return false.
///   * non-str otherwise: switch_to_object_strategy then walk.
/// Called only from the strategy trait impl to avoid recursion
/// through `w_dict_delitem`.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_delitem_inner(obj: PyObjectRef, key: PyObjectRef) -> bool {
    if w_module_dict_is_object_strategy(obj) {
        let proxy = (*(obj as *const W_ModuleDictObject)).dict_storage_proxy;
        let entries = w_module_dict_object_storage_mut(obj);
        if entries.shift_remove(&object_key_for(key)).is_some() {
            let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
            strategy.mutated();
            if let Some(ks) = key_as_utf8(key) {
                maybe_sync_dict_storage_delete(proxy, ks);
            }
            return true;
        }
        // Entries missed — fall through to the proxy mirror just as
        // `w_module_dict_delitem_str_internal:1010-1013` does on the
        // str-fast-path side.  `celldict.py:106-126` invalidates the
        // cell version for any successful str delete; `mutated()` is
        // what triggers GlobalCache invalidation in pyre, so fire it
        // alongside the proxy-only removal.
        if !proxy.is_null() {
            if let Some(key_str) = key_as_utf8(key) {
                if maybe_lookup_dict_storage(proxy, key_str).is_some() {
                    maybe_sync_dict_storage_delete(proxy, key_str);
                    let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
                    strategy.mutated();
                    return true;
                }
            }
        }
        return false;
    }
    if let Some(ks) = key_as_utf8(key) {
        return w_module_dict_delitem_str(obj, ks).is_some();
    }
    if _never_equal_to_string(key) {
        return false;
    }
    w_module_dict_switch_to_object_strategy(obj);
    let proxy = (*(obj as *const W_ModuleDictObject)).dict_storage_proxy;
    let entries = w_module_dict_object_storage_mut(obj);
    if entries.shift_remove(&object_key_for(key)).is_some() {
        let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
        strategy.mutated();
        if let Some(ks) = key_as_utf8(key) {
            maybe_sync_dict_storage_delete(proxy, ks);
        }
        return true;
    }
    false
}

pub unsafe fn w_module_dict_delitem_inner_checked(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<bool, DictKeyError> {
    if w_module_dict_is_object_strategy(obj) {
        let object_key = object_key_for_checked(key)?;
        let proxy = (*(obj as *const W_ModuleDictObject)).dict_storage_proxy;
        let entries = w_module_dict_object_storage_mut(obj);
        let removed = entries.shift_remove(&object_key).is_some();
        if take_dict_key_error() {
            return Err(DictKeyError);
        }
        if removed {
            let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
            strategy.mutated();
            if let Some(ks) = key_as_utf8(key) {
                maybe_sync_dict_storage_delete(proxy, ks);
            }
            return Ok(true);
        }
        if !proxy.is_null() {
            if let Some(key_str) = key_as_utf8(key) {
                if maybe_lookup_dict_storage(proxy, key_str).is_some() {
                    maybe_sync_dict_storage_delete(proxy, key_str);
                    let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
                    strategy.mutated();
                    return Ok(true);
                }
            }
        }
        return Ok(false);
    }
    if let Some(ks) = key_as_utf8(key) {
        return Ok(w_module_dict_delitem_str(obj, ks).is_some());
    }
    if _never_equal_to_string(key) {
        return Ok(false);
    }
    w_module_dict_switch_to_object_strategy(obj);
    let object_key = object_key_for_checked(key)?;
    let proxy = (*(obj as *const W_ModuleDictObject)).dict_storage_proxy;
    let entries = w_module_dict_object_storage_mut(obj);
    let removed = entries.shift_remove(&object_key).is_some();
    if take_dict_key_error() {
        return Err(DictKeyError);
    }
    if removed {
        let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
        strategy.mutated();
        if let Some(ks) = key_as_utf8(key) {
            maybe_sync_dict_storage_delete(proxy, ks);
        }
        return Ok(true);
    }
    Ok(false)
}

/// `pypy/objspace/std/dictmultiobject.py:148-152 W_DictMultiObject.descr_clear`
/// — `w_dict.get_strategy().clear(w_dict)`.  Dispatches through the
/// polymorphic strategy slot.  Pyre wraps the dispatch with the
/// `dict_storage_proxy` flush bookkeeping (`maybe_sync_dict_storage_delete`)
/// so that `module.__dict__.clear()` / `globals().clear()` also empty
/// the legacy str-keyed `DictStorage` shadow.  The proxy flush is a
/// TODO; once `PyFrame.w_globals` migrates to
/// `PyObjectRef`, the shadow goes away and only the
/// strategy.clear() body remains here.
pub unsafe fn w_dict_clear(obj: PyObjectRef) {
    let proxy = w_dict_get_dict_storage_proxy(obj);
    // Snapshot str keys before strategy.clear so we can flush them
    // out of the backing DictStorage in step with the W_DictMultiObject
    // clear.  `dict.clear()` on `module.__dict__` must drain the
    // frame-side `w_globals` carrier too.
    let str_keys: Vec<String> = if proxy.is_null() {
        Vec::new()
    } else if let Some(items) = maybe_items_dict_storage(proxy) {
        items.into_iter().map(|(k, _)| k).collect()
    } else {
        Vec::new()
    };
    w_dict_get_strategy(obj).clear(obj);
    for k in str_keys {
        maybe_sync_dict_storage_delete(proxy, &k);
    }
}

/// Internal helper: `ObjectDictStrategy::clear` body for pyre's
/// W_DictObject — truncates the dstorage Vec and resets the JIT
/// inline length cache.  Called only from the strategy trait impl
/// to avoid recursion through `w_dict_clear`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_clear_object_strategy(obj: PyObjectRef) {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<ObjectKey, PyObjectRef>);
    entries.clear();
}

/// Internal helper: `ModuleDictStrategy::clear` body for pyre's
/// W_ModuleDictObject — branches on `is_object_strategy` and drains
/// whichever storage half is live.  Called only from the strategy
/// trait impl to avoid recursion through `w_dict_clear`.  The proxy
/// flush stays in the public `w_dict_clear` wrapper.
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_clear_inner(obj: PyObjectRef) {
    if w_module_dict_is_object_strategy(obj) {
        w_module_dict_object_storage_mut(obj).clear();
        let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
        strategy.mutated();
    } else {
        let strategy = &mut *(*(obj as *mut W_ModuleDictObject)).mstrategy;
        let storage = &mut *(*(obj as *mut W_ModuleDictObject)).dstorage;
        strategy.clear(storage);
    }
}

/// `pypy/objspace/std/dictmultiobject.py:107-109 W_DictMultiObject.length`
/// — `return self.get_strategy().length(self)`.  Dispatches through
/// the polymorphic strategy slot.
pub unsafe fn w_dict_len(obj: PyObjectRef) -> usize {
    w_dict_get_strategy(obj).length(obj)
}

/// Internal helper: `ObjectDictStrategy::length` body for pyre's
/// W_DictObject — `len(self.unerase(w_dict.dstorage))` per
/// `dictmultiobject.py:1226 length`, reconciling with the
/// `dict_storage_proxy` str-keyed view when attached.  Called only
/// from the strategy trait impl to avoid recursion through
/// `w_dict_len`.
///
/// Storage-authoritative for str keys when proxy is attached:
/// returns the storage's str-key count plus any non-str-keyed
/// `entries` Vec slots (storage is str-keyed by construction).  This
/// avoids the stale-cache double-count `dict_storage_to_dict` would
/// otherwise produce when a STORE_GLOBAL through the shared storage
/// replaces a pre-existing entry — the dstorage Vec might still hold
/// the old version, but storage owns the live count.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_length_object_strategy(obj: PyObjectRef) -> usize {
    let dict = &*(obj as *const W_DictObject);
    let entries = &*(dict.dstorage as *const indexmap::IndexMap<ObjectKey, PyObjectRef>);
    if dict.dict_storage_proxy.is_null() {
        return entries.len();
    }
    let Some(storage_items) = maybe_items_dict_storage(dict.dict_storage_proxy) else {
        return entries.len();
    };
    // The proxy str view covers valid-UTF-8 str keys; count the IndexMap
    // entries it cannot represent (non-str and lone-surrogate str keys).
    let extra = entries
        .keys()
        .filter(|k| !(crate::is_str(k.obj) && crate::w_str_get_value_opt(k.obj).is_some()))
        .count();
    storage_items.len() + extra
}

/// Iterate over all (key, value) pairs without type assumptions.
///
/// Storage-authoritative for str keys when proxy is attached: emits
/// the storage's str-keyed entries first, then any non-str-keyed
/// `entries` Vec slots.  Stale str entries cached in the dstorage Vec
/// (e.g. `dict_storage_to_dict` snapshot taken before a STORE_GLOBAL
/// on the shared storage) are dropped in favour of the storage's
/// live values.
pub unsafe fn w_dict_items(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    w_dict_get_strategy(obj).items(obj)
}

/// `dictmultiobject.py:585-587 W_DictMultiObject.descr_copy` —
/// `w_dict.copy()` delegates to `strategy.copy(w_dict)` so typed
/// strategies preserve their backing shape (`:1152
/// AbstractTypedStrategy.copy` → fresh W_DictObject with same strategy
/// + cloned typed dstorage).
///
/// `dict_storage_proxy`-attached dicts (pyre back-mirror
/// TODO) take the union-walk fallback so str-key
/// entries that live only in the proxy survive the copy.  Once the
/// proxy retires this branch goes with it.
pub unsafe fn w_dict_copy(obj: PyObjectRef) -> PyObjectRef {
    if is_dict(obj) && !w_dict_get_dict_storage_proxy(obj).is_null() {
        let new_dict = w_dict_new();
        for (k, v) in w_dict_items(obj) {
            w_dict_store(new_dict, k, v);
        }
        return new_dict;
    }
    w_dict_get_strategy(obj).copy(obj)
}

/// Internal helper: `IntDictStrategy::setitem` body — direct
/// linear-scan write on the native `Vec<(i64, PyObjectRef)>` storage,
/// matching `dictmultiobject.py:1061-1064` (`self.unerase
/// (w_dict.dstorage)[self.unwrap(w_key)] = w_value`).  Caller must
/// have already verified `is_correct_type(w_key)`.
///
/// `dict_storage_proxy` mirror is intentionally skipped — the proxy
/// is str-key-authoritative (`dict_storage_to_dict_kind` module flow
/// + `w_module_dict_*` paths), so int-keyed writes carry no proxy
/// counterpart.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on
/// [`crate::dictstrategy::INT_DICT_STRATEGY`]; `key` must be a
/// plain `W_IntObject` (not bool).
pub unsafe fn w_dict_store_int_strategy(obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<i64, PyObjectRef>);
    let k = crate::w_int_get_value(key);
    entries.insert(k, value);
    dict_write_barrier(obj);
}

/// Internal helper: `IntDictStrategy::getitem` body —
/// `dictmultiobject.py:1098 self.unerase(w_dict.dstorage).get(self.unwrap(w_key), None)`.
/// Caller must have already verified `is_correct_type(w_key)`.
///
/// # Safety
/// Same as [`w_dict_store_int_strategy`].
pub unsafe fn w_dict_lookup_int_strategy(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Option<PyObjectRef> {
    let entries = w_dict_int_storage(obj);
    let k = crate::w_int_get_value(key);
    entries.get(&k).copied()
}

/// Internal helper: `IntDictStrategy::delitem` body —
/// `dictmultiobject.py:1083 del self.unerase(w_dict.dstorage)[self.unwrap(w_key)]`.
/// Returns `true` if a key was removed.
///
/// # Safety
/// Same as [`w_dict_store_int_strategy`].
pub unsafe fn w_dict_delitem_int_strategy(obj: PyObjectRef, key: PyObjectRef) -> bool {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<i64, PyObjectRef>);
    let k = crate::w_int_get_value(key);
    // shift_remove preserves insertion order, matching CPython 3.7+ /
    // PyPy3 dict semantics where deleting an entry leaves the
    // remaining entries in their original relative order.
    if entries.shift_remove(&k).is_some() {
        true
    } else {
        false
    }
}

/// Internal helper: `IntDictStrategy::length` body —
/// `dictmultiobject.py:1090 len(self.unerase(w_dict.dstorage))`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on
/// [`crate::dictstrategy::INT_DICT_STRATEGY`].
pub unsafe fn w_dict_length_int_strategy(obj: PyObjectRef) -> usize {
    w_dict_int_storage(obj).len()
}

/// Internal helper: `IntDictStrategy::items` body —
/// `dictmultiobject.py:1113-1117 items` walks the typed storage and
/// wraps each i64 key via `self.wrap` (`:1342 wrap=newint`).
///
/// # Safety
/// Same as [`w_dict_length_int_strategy`].
pub unsafe fn w_dict_items_int_strategy(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    w_dict_int_storage(obj)
        .iter()
        .map(|(&k, &v)| (crate::w_int_new(k), v))
        .collect()
}

/// Internal helper: `IntDictStrategy::clear` body —
/// `dictmultiobject.py:1141 self.unerase(w_dict.dstorage).clear()`.
///
/// # Safety
/// Same as [`w_dict_length_int_strategy`].
pub unsafe fn w_dict_clear_int_strategy(obj: PyObjectRef) {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<i64, PyObjectRef>);
    entries.clear();
}

/// Internal helper: `BytesDictStrategy::setitem` body —
/// `dictmultiobject.py:1061-1064` direct typed-storage write.  Caller
/// must have already verified `is_correct_type(w_key)`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on
/// [`crate::dictstrategy::BYTES_DICT_STRATEGY`]; `key` must be a
/// `W_BytesObject`.
pub unsafe fn w_dict_store_bytes_strategy(obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<Vec<u8>, PyObjectRef>);
    let k = crate::w_bytes_data(key).to_vec();
    entries.insert(k, value);
    dict_write_barrier(obj);
}

/// Internal helper: `BytesDictStrategy::getitem` body —
/// `dictmultiobject.py:1098`.  Caller must have already verified
/// `is_correct_type(w_key)`.
///
/// # Safety
/// Same as [`w_dict_store_bytes_strategy`].
pub unsafe fn w_dict_lookup_bytes_strategy(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Option<PyObjectRef> {
    let entries = w_dict_bytes_storage(obj);
    let k = crate::w_bytes_data(key);
    entries.get(k).copied()
}

/// Internal helper: `BytesDictStrategy::delitem` body —
/// `dictmultiobject.py:1083`.  Returns `true` if a key was removed.
///
/// # Safety
/// Same as [`w_dict_store_bytes_strategy`].
pub unsafe fn w_dict_delitem_bytes_strategy(obj: PyObjectRef, key: PyObjectRef) -> bool {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<Vec<u8>, PyObjectRef>);
    let k = crate::w_bytes_data(key);
    if entries.shift_remove(k).is_some() {
        true
    } else {
        false
    }
}

/// Internal helper: `BytesDictStrategy::length` body.
///
/// # Safety
/// Same as [`w_dict_bytes_storage`].
pub unsafe fn w_dict_length_bytes_strategy(obj: PyObjectRef) -> usize {
    w_dict_bytes_storage(obj).len()
}

/// Internal helper: `BytesDictStrategy::items` body —
/// `dictmultiobject.py:1113-1117` with `wrap = newbytes` per
/// `:1234-1235`.
///
/// # Safety
/// Same as [`w_dict_bytes_storage`].
pub unsafe fn w_dict_items_bytes_strategy(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    w_dict_bytes_storage(obj)
        .iter()
        .map(|(k, v)| (crate::w_bytes_from_bytes(k.as_slice()), *v))
        .collect()
}

/// Internal helper: `BytesDictStrategy::clear` body.
///
/// # Safety
/// Same as [`w_dict_bytes_storage`].
pub unsafe fn w_dict_clear_bytes_strategy(obj: PyObjectRef) {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *(dict.dstorage as *mut indexmap::IndexMap<Vec<u8>, PyObjectRef>);
    entries.clear();
}

/// Internal helper: `ObjectDictStrategy::items` body for pyre's
/// W_DictObject — clones the dstorage Vec or merges with the
/// `dict_storage_proxy` str-keyed view.  Called only from the
/// strategy trait impl to avoid recursion through `w_dict_items`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_items_object_strategy(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    let dict = &*(obj as *const W_DictObject);
    let entries = &*(dict.dstorage as *const indexmap::IndexMap<ObjectKey, PyObjectRef>);
    if dict.dict_storage_proxy.is_null() {
        return entries.iter().map(|(k, &v)| (k.obj, v)).collect();
    }
    let Some(storage_items) = maybe_items_dict_storage(dict.dict_storage_proxy) else {
        return entries.iter().map(|(k, &v)| (k.obj, v)).collect();
    };
    let mut out: Vec<(PyObjectRef, PyObjectRef)> = storage_items
        .into_iter()
        .map(|(name, value)| (crate::w_str_new(&name), value))
        .collect();
    for (key, &v) in entries.iter() {
        // The proxy str view above is authoritative for valid-UTF-8 str
        // keys; a lone-surrogate str key has no `&str` form and is dropped
        // there, so the IndexMap is its only source.  Add every IndexMap
        // entry the proxy view does not already cover (non-str keys and
        // surrogate str keys).
        let covered_by_proxy =
            crate::is_str(key.obj) && crate::w_str_get_value_opt(key.obj).is_some();
        if !covered_by_proxy {
            out.push((key.obj, v));
        }
    }
    out
}

/// Internal helper: `ModuleDictStrategy::items` body for pyre's
/// W_ModuleDictObject — branches on `is_object_strategy` and emits
/// from whichever storage half is live.  Called only from the
/// strategy trait impl to avoid recursion through `w_dict_items`.
///
/// Reconciles with the `dict_storage_proxy` str-keyed view when
/// attached (proxy is authoritative for str keys; local non-str
/// entries from `object_storage` are preserved), mirroring
/// `w_dict_items_object_strategy` (:1672) and aligning with the
/// proxy fallback path in `w_module_dict_getitem_str` (:730).
///
/// # Safety
/// `obj` must point to a valid `W_ModuleDictObject`.
pub unsafe fn w_module_dict_items_inner(obj: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
    let proxy = (*(obj as *const W_ModuleDictObject)).dict_storage_proxy;
    let local: Vec<(PyObjectRef, PyObjectRef)> =
        if let Some(entries) = w_module_dict_object_storage(obj) {
            entries.iter().map(|(k, &v)| (k.obj, v)).collect()
        } else {
            let strategy = &*w_module_dict_get_strategy(obj);
            let storage = &*w_module_dict_get_storage(obj);
            strategy
                .getiterkeys(storage)
                .zip(strategy.getitervalues(storage))
                .map(|(k, v)| (crate::w_str_new(k), v))
                .collect()
        };
    if proxy.is_null() {
        return local;
    }
    let Some(storage_items) = maybe_items_dict_storage(proxy) else {
        return local;
    };
    // Proxy str-keyed entries first (authoritative for str keys);
    // then local non-str entries from `object_storage`.
    let mut out: Vec<(PyObjectRef, PyObjectRef)> = storage_items
        .into_iter()
        .map(|(name, value)| (crate::w_str_new(&name), value))
        .collect();
    for &(k, v) in local.iter() {
        // See `w_dict_items_object_strategy`: keep IndexMap keys the proxy
        // str view cannot represent — non-str and lone-surrogate str keys.
        let covered_by_proxy = crate::is_str(k) && crate::w_str_get_value_opt(k).is_some();
        if !covered_by_proxy {
            out.push((k, v));
        }
    }
    out
}

/// Iterate over (key_str, value) pairs. Keys must be str objects.
///
/// Pyre-only convenience wrapper around `w_dict_items` that drops
/// non-str entries and unwraps the str keys.  Dispatches through
/// the strategy slot via `w_dict_items`, so W_DictObject and
/// W_ModuleDictObject both round-trip their dict_storage_proxy /
/// celldict cell-cache walks uniformly.
///
/// Keys that carry a lone surrogate (not valid UTF-8) are skipped:
/// the remaining `&str`-keyed consumers (dict_storage_store, module
/// `__dir__`, builtins-module iteration) cannot yet represent a
/// surrogate key, so skipping them here avoids the [`w_str_get_value`]
/// panic.  The keyword-argument ABI no longer uses this helper — it
/// threads the byte-ish key through [`w_dict_str_entries_wtf8`].
pub unsafe fn w_dict_str_entries(obj: PyObjectRef) -> Vec<(String, PyObjectRef)> {
    w_dict_items(obj)
        .into_iter()
        .filter_map(|(k, v)| {
            if crate::is_str(k) {
                crate::w_str_get_value_opt(k).map(|s| (s.to_string(), v))
            } else {
                None
            }
        })
        .collect()
}

/// Iterate over (key_wtf8, value) pairs, preserving lone-surrogate keys.
///
/// The surrogate-preserving counterpart of [`w_dict_str_entries`], used
/// by the keyword-argument ABI (`call_with_kwargs`, `bind_kwargs_to_signature`,
/// the builtin `__pyre_kw__` readers).  `Arguments` keeps keyword names as
/// byte-ish `str` (`keywords: [str]`, `argument.py`), so a `**{'\udc80': v}`
/// key survives as a `Wtf8Buf` rather than being dropped.
///
/// Non-str keys are dropped here, which is correct for the name-enumeration
/// callers (`dir()`, dict merge).  It is NOT, however, the `**kwargs`-unpack
/// contract: `argument.py` `_do_combine_starstarargs_wrapped` RAISES
/// `TypeError("keywords must be strings, not '%T'")` on a non-str key rather
/// than skipping it, so a caller on that path (`CALL_FUNCTION_EX`) must enforce
/// the TypeError itself — this helper does not.
pub unsafe fn w_dict_str_entries_wtf8(
    obj: PyObjectRef,
) -> Vec<(rustpython_wtf8::Wtf8Buf, PyObjectRef)> {
    w_dict_items(obj)
        .into_iter()
        .filter_map(|(k, v)| {
            if crate::is_str(k) {
                Some((crate::w_str_get_wtf8(k).to_owned(), v))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::{w_int_get_value, w_int_new};
    use crate::w_str_new;

    /// Install the single hash path on this test thread.  pyre-object
    /// cannot reach `pyre-interpreter`'s `space.hash_w`, so its own dict
    /// tests register the deterministic builtin structural hash; the
    /// `register_hash_w_hook` cell is thread-local and libtest spawns a
    /// fresh thread per `#[test]`, so each dict-building test installs it.
    fn install_test_hash_hook() {
        crate::dict_eq_hook::register_hash_w_hook(builtin_structural_hash);
    }

    #[test]
    fn test_dict_int_key() {
        let dict = w_dict_new();
        unsafe {
            assert!(is_dict(dict));
            w_dict_setitem(dict, 1, w_int_new(100));
            assert_eq!(w_int_get_value(w_dict_getitem(dict, 1).unwrap()), 100);
        }
    }

    #[test]
    fn test_dict_str_key() {
        install_test_hash_hook();
        let dict = w_dict_new();
        unsafe {
            w_dict_setitem_str(dict, "hello", w_int_new(42));
            assert_eq!(
                w_int_get_value(w_dict_getitem_str(dict, "hello").unwrap()),
                42
            );
            assert!(w_dict_getitem_str(dict, "world").is_none());
        }
    }

    #[test]
    fn test_dict_pyobj_key() {
        install_test_hash_hook();
        let dict = w_dict_new();
        let key = crate::w_str_new("test");
        unsafe {
            w_dict_store(dict, key, w_int_new(99));
            assert_eq!(w_int_get_value(w_dict_lookup(dict, key).unwrap()), 99);
        }
    }

    /// Force every key into one bucket so a store probes the existing
    /// entry by equality.
    unsafe fn constant_collision_hash(_obj: PyObjectRef) -> i64 {
        42
    }

    use std::sync::atomic::{AtomicU32, Ordering};
    static RAISING_EQ_CALLS: AtomicU32 = AtomicU32::new(0);

    /// An `eq_w` hook that always raises: it records the call, signals the
    /// error flag, and returns `false` (the Rust `Eq` callback cannot raise).
    unsafe fn raising_eq(a: PyObjectRef, _b: PyObjectRef) -> bool {
        RAISING_EQ_CALLS.fetch_add(1, Ordering::Relaxed);
        crate::dict_eq_hook::signal_eq_error(a);
        false
    }

    /// A user `__eq__` that raises during a colliding store must leave the
    /// dict unchanged and surface the first error, matching r_dict raising
    /// mid-probe without completing setitem.  Verifies: the store reports an
    /// error, the spuriously appended entry is dropped (k1 intact, k2 never
    /// stored), and `__eq__` runs exactly once (the probe short-circuits
    /// after the first raise).
    #[test]
    fn test_dict_store_raising_eq_leaves_dict_unchanged() {
        install_test_hash_hook();
        unsafe {
            crate::dict_eq_hook::register_hash_w_hook(constant_collision_hash);
            let dict = w_dict_new();
            let k1 = w_str_new("k1");
            let k2 = w_str_new("k2");
            // Seed one entry while the bucket is empty (no comparison runs).
            w_dict_store(dict, k1, w_int_new(1));

            RAISING_EQ_CALLS.store(0, Ordering::Relaxed);
            crate::dict_eq_hook::register_eq_w_hook(raising_eq);
            let result = w_dict_store_checked(dict, k2, w_int_new(2));
            crate::dict_eq_hook::clear_eq_w_hook();

            assert!(result.is_err());
            // The probe stopped at the first raising comparison.
            assert_eq!(RAISING_EQ_CALLS.load(Ordering::Relaxed), 1);
            // The dict is unchanged: k1 intact, k2 never inserted.
            assert_eq!(w_int_get_value(w_dict_lookup(dict, k1).unwrap()), 1);
            assert!(w_dict_lookup(dict, k2).is_none());
            // The error flag was consumed by the store, not left dangling.
            assert!(!crate::dict_eq_hook::take_eq_error());
        }
    }

    #[test]
    fn test_dict_overwrite() {
        let dict = w_dict_new();
        unsafe {
            w_dict_setitem(dict, 1, w_int_new(10));
            w_dict_setitem(dict, 1, w_int_new(20));
            assert_eq!(w_dict_len(dict), 1);
        }
    }

    #[test]
    fn w_dict_gc_type_id_matches_descr() {
        assert_eq!(W_DICT_GC_TYPE_ID, 29);
        assert_eq!(
            <W_DictObject as crate::lltype::GcType>::type_id(),
            W_DICT_GC_TYPE_ID
        );
        assert_eq!(
            <W_DictObject as crate::lltype::GcType>::SIZE,
            W_DICT_OBJECT_SIZE
        );
    }

    /// `dict_storage_proxy` attached but `register_dict_storage_items_hook`
    /// has not yet been called (direct `w_module_new` callers,
    /// hookless unit/integration tests).  The pre-fix code returned an
    /// empty storage view from `maybe_items_dict_storage` and combined it
    /// only with the *non-str* dstorage Vec slots, which silently dropped
    /// every str key written through `w_dict_setitem_str`.  The current
    /// behaviour treats the missing hook as "storage view indeterminate"
    /// and falls back to the dstorage Vec, matching PyPy's
    /// `W_DictMultiObject` single-source-of-truth semantics during the
    /// hookless bootstrap window.
    ///
    /// pyre-object alone has no caller of
    /// `register_dict_storage_items_hook`, so within `cargo test -p
    /// pyre-object` the hook stays null for the lifetime of the test
    /// process — the assertion is therefore stable here.
    #[test]
    fn test_w_dict_proxied_hookless_falls_back_to_entries_vec() {
        // "hookless" here means the dict_storage *items* hook is absent;
        // the hash hook is still required to bucket str keys.
        install_test_hash_hook();
        let dict = w_dict_new();
        unsafe {
            // Non-null sentinel; the hook never fires because no hook
            // has been registered, so the pointer's pointee is never
            // dereferenced.
            let sentinel: *mut u8 = 0xdead_beef_usize as *mut u8;
            w_dict_set_dict_storage_proxy(dict, sentinel);

            w_dict_setitem_str(dict, "alpha", w_int_new(1));
            w_dict_setitem_str(dict, "beta", w_int_new(2));

            assert_eq!(
                w_dict_len(dict),
                2,
                "hookless proxied dict must expose the dstorage Vec count, not 0",
            );

            let items = w_dict_items(dict);
            assert_eq!(items.len(), 2);
            let mut keys: Vec<&str> = items
                .iter()
                .map(|&(k, _)| crate::w_str_get_value(k))
                .collect();
            keys.sort();
            assert_eq!(keys, vec!["alpha", "beta"]);

            let mut str_entries = w_dict_str_entries(dict);
            str_entries.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(str_entries.len(), 2);
            assert_eq!(str_entries[0].0, "alpha");
            assert_eq!(w_int_get_value(str_entries[0].1), 1);
            assert_eq!(str_entries[1].0, "beta");
            assert_eq!(w_int_get_value(str_entries[1].1), 2);
        }
    }

    #[test]
    fn module_dict_basic_roundtrip() {
        let md = w_module_dict_new();
        unsafe {
            assert!(is_module_dict(md));
            // PyPy surfaces both as `dict` (`dictmultiobject.py:67
            // allocate_instance(..., space.w_dict)`), so the
            // user-visible `is_dict` covers the module-dict layout.
            assert!(is_dict(md));
            w_module_dict_setitem_str(md, "x", w_int_new(1));
            w_module_dict_setitem_str(md, "y", w_int_new(2));
            assert_eq!(w_module_dict_length(md), 2);
            assert_eq!(
                w_int_get_value(w_module_dict_getitem_str(md, "x").unwrap()),
                1,
            );
            assert_eq!(
                w_int_get_value(w_module_dict_getitem_str(md, "y").unwrap()),
                2,
            );
            let removed = w_module_dict_delitem_str(md, "x").unwrap();
            assert_eq!(w_int_get_value(removed), 1);
            assert_eq!(w_module_dict_length(md), 1);
            assert!(w_module_dict_getitem_str(md, "x").is_none());
        }
    }

    #[test]
    fn module_dict_routes_through_w_dict_dispatch() {
        // Confirms `w_dict_*` public ops dispatch through the
        // `W_ModuleDictObject` branch when given a module dict, so
        // existing callers that take `PyObjectRef` continue to work
        // without knowing which layout backs the dict.
        let md = w_module_dict_new();
        unsafe {
            w_dict_setitem_str(md, "alpha", w_int_new(11));
            w_dict_setitem_str(md, "beta", w_int_new(22));
            assert_eq!(w_dict_len(md), 2);
            assert_eq!(
                w_int_get_value(w_dict_getitem_str(md, "alpha").unwrap()),
                11,
            );
            assert_eq!(
                w_int_get_value(w_dict_lookup(md, w_str_new("beta")).unwrap()),
                22,
            );
            assert!(w_dict_delitem_str(md, "alpha"));
            assert!(w_dict_getitem_str(md, "alpha").is_none());
            assert_eq!(w_dict_len(md), 1);
        }
    }

    #[test]
    fn module_dict_gc_type_id_matches_descr() {
        // Trip-wire mirroring the W_CELL / W_INT / W_FLOAT guards —
        // panics on startup if the constant here drifts from the id
        // that `pyre/pyre-jit/src/eval.rs` asserts at JitDriver init.
        assert_eq!(W_MODULE_DICT_GC_TYPE_ID, 48);
        assert_eq!(
            <W_ModuleDictObject as crate::lltype::GcType>::type_id(),
            W_MODULE_DICT_GC_TYPE_ID
        );
        assert_eq!(
            <W_ModuleDictObject as crate::lltype::GcType>::SIZE,
            W_MODULE_DICT_OBJECT_SIZE
        );
    }
}
