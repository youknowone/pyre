//! W_SetObject — Python `set` type.
//!
//! PyPy equivalent: pypy/objspace/std/setobject.py
//!
//! Stores arbitrary PyObjectRef elements in a hashed IndexMap of ObjectKey,
//! reusing the dict object strategy's hashing and equality semantics. PyPy
//! carries multiple set strategies (EmptySet, IntegerSet, etc.); pyre starts
//! with a single strategy while bringing the type online.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use indexmap::map::{RawEntryApiV1, raw_entry_v1::RawEntryMut};
use pyre_macros::pyre_class;
use std::hash::BuildHasher;

pub static SET_TYPE: PyType = crate::pyobject::new_pytype("set");
pub static FROZENSET_TYPE: PyType = crate::pyobject::new_pytype("frozenset");

/// setobject.py `W_SetIterObject`.  Unlike the old sequence-iterator
/// adapter this keeps the live set, so a size change is observed by next().
#[pyre_class("set_iterator", static_name = "SET_ITERATOR")]
pub struct W_SetIterObject {
    pub w_set: PyObjectRef,
    pub startlen: usize,
    pub index: usize,
}

pub fn w_set_iter_new(w_set: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_set);
    let startlen = unsafe { w_set_len(w_set) };
    W_SetIterObject::allocate_stable(W_SetIterObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_set,
        startlen,
        index: 0,
    })
}

#[inline]
pub unsafe fn is_set_iterator(obj: PyObjectRef) -> bool {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return false;
    }
    !obj.is_null() && (*obj).ob_type == &SET_ITERATOR_TYPE as *const PyType
}

#[inline]
pub unsafe fn w_set_iter_get_set(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_SetIterObject)).w_set
}

#[inline]
pub unsafe fn w_set_iter_set_set(obj: PyObjectRef, w_set: PyObjectRef) {
    (*(obj as *mut W_SetIterObject)).w_set = w_set;
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

#[inline]
pub unsafe fn w_set_iter_get_startlen(obj: PyObjectRef) -> usize {
    (*(obj as *const W_SetIterObject)).startlen
}

#[inline]
pub unsafe fn w_set_iter_set_startlen(obj: PyObjectRef, startlen: usize) {
    (*(obj as *mut W_SetIterObject)).startlen = startlen;
}

#[inline]
pub unsafe fn w_set_iter_get_index(obj: PyObjectRef) -> usize {
    (*(obj as *const W_SetIterObject)).index
}

#[inline]
pub unsafe fn w_set_iter_set_index(obj: PyObjectRef, index: usize) {
    (*(obj as *mut W_SetIterObject)).index = index;
}

/// Python set object.
///
/// Layout: `[ob_type | items | len]`. `items` points to a GC-managed hashed
/// [`SetItemsStorage`], matching PyPy's `sstorage` field: its `r_dict` table is
/// an `rdict.py:210` `GcStruct("dicttable")`, and `setobject.py:875` replaces
/// that GC pointer field with `get_storage_copy()`.
#[repr(C)]
pub struct W_SetObject {
    pub ob_header: PyObject,
    pub items: *mut SetItemsStorage,
    pub len: usize,
    /// setobject.py:584 `W_FrozensetObject.hash = DEFAULT_HASH`.
    pub hash: i64,
}

/// GC type id assigned to `W_SetObject` at JitDriver init time.
pub const W_SET_GC_TYPE_ID: u32 = 30;

/// GC-managed element table shared by `set` and `frozenset` bodies.
pub type SetItemsStorage = indexmap::IndexMap<crate::dictmultiobject::ObjectKey, ()>;

/// Runtime-assigned GC type id for [`SetItemsStorage`]. Like the bigint
/// payload id, this is published by `pyre-jit::eval` after the fixed-constant
/// type registrations and is never embedded in a JIT allocation descriptor.
static SET_ITEMS_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for [`SetItemsStorage`].
pub fn set_set_items_gc_type_id(id: u32) {
    SET_ITEMS_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for [`SetItemsStorage`].
#[majit_macros::dont_look_inside]
pub fn set_items_gc_type_id() -> u32 {
    SET_ITEMS_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Fixed payload size (`framework.py:811`).
pub const W_SET_OBJECT_SIZE: usize = std::mem::size_of::<W_SetObject>();

impl crate::lltype::GcType for W_SetObject {
    fn type_id() -> u32 {
        W_SET_GC_TYPE_ID
    }
    const SIZE: usize = W_SET_OBJECT_SIZE;
}

#[inline]
pub unsafe fn is_set(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SET_TYPE) }
}

#[inline]
pub unsafe fn is_frozenset(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FROZENSET_TYPE) }
}

#[inline]
pub unsafe fn is_set_or_frozenset(obj: PyObjectRef) -> bool {
    unsafe { is_set(obj) || is_frozenset(obj) }
}

/// Fire the GC write barrier for a set whose element storage just gained
/// a possibly-young element. `set_object_custom_trace` only forwards the
/// `items` slots when the set is reached by a collection; an old-gen set
/// that stored a young element is reached on a minor GC only if it sits in
/// the remembered set, so the barrier must run after every insert. Mirrors
/// `dict_write_barrier`.
#[inline]
fn set_write_barrier(obj: PyObjectRef) {
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// Allocate an empty `set`.
pub fn w_set_new() -> PyObjectRef {
    let items =
        crate::gc_storage::gc_alloc_storage_box(SetItemsStorage::new(), set_items_gc_type_id());
    let header = PyObject {
        ob_type: &SET_TYPE as *const PyType,
        w_class: get_instantiate(&SET_TYPE),
    };
    // Allocate the body in GC old-gen (mark-sweep, non-moving) so it
    // carries TRACK_YOUNG_PTRS, mirroring `w_list_new` / `w_tuple_new`.
    // `w_set_add` stores possibly-young elements into `items`; the write
    // barrier (`set_write_barrier`) only remembers the set on a minor
    // collection when the body is an old-gen object, so a body allocated
    // through the plain `malloc_typed` (no TRACK_YOUNG_PTRS) would leave
    // young elements unforwarded and collected. Falls back to
    // `malloc_typed` when no GC hook is installed (unit tests).
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(W_SET_GC_TYPE_ID, W_SET_OBJECT_SIZE);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(
                raw as *mut W_SetObject,
                W_SetObject {
                    ob_header: header,
                    items,
                    len: 0,
                    hash: -1,
                },
            );
        }
        raw as PyObjectRef
    } else {
        crate::lltype::malloc_typed(W_SetObject {
            ob_header: header,
            items,
            len: 0,
            hash: -1,
        }) as PyObjectRef
    }
}

/// Allocate an empty `frozenset`.
///
/// Same body as [`w_set_new`] with the constant `&FROZENSET_TYPE` baked
/// into `ob_type`; see that constructor for the GC old-gen rationale.
pub fn w_frozenset_new() -> PyObjectRef {
    let items =
        crate::gc_storage::gc_alloc_storage_box(SetItemsStorage::new(), set_items_gc_type_id());
    let header = PyObject {
        ob_type: &FROZENSET_TYPE as *const PyType,
        w_class: get_instantiate(&FROZENSET_TYPE),
    };
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(W_SET_GC_TYPE_ID, W_SET_OBJECT_SIZE);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(
                raw as *mut W_SetObject,
                W_SetObject {
                    ob_header: header,
                    items,
                    len: 0,
                    hash: -1,
                },
            );
        }
        raw as PyObjectRef
    } else {
        crate::lltype::malloc_typed(W_SetObject {
            ob_header: header,
            items,
            len: 0,
            hash: -1,
        }) as PyObjectRef
    }
}

/// Allocate a populated set from a slice of elements (deduped).
pub fn w_set_from_items(items: &[PyObjectRef]) -> PyObjectRef {
    let s = w_set_new();
    for &item in items {
        unsafe { w_set_add(s, item) };
    }
    s
}

/// Allocate a populated frozenset from a slice of elements (deduped).
pub fn w_frozenset_from_items(items: &[PyObjectRef]) -> PyObjectRef {
    let s = w_frozenset_new();
    for &item in items {
        unsafe { w_set_add(s, item) };
    }
    s
}

/// Insert an element. No-op when already present.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_add(obj: PyObjectRef, item: PyObjectRef) {
    let key = crate::dictmultiobject::object_key_for(item);
    let _ = w_set_insert_key_checked(obj, key);
}

/// Insert an element keyed on a `space.hash_w` digest the caller already
/// holds, propagating an `eq_w` raise from the bucket probe.
///
/// `setobject.py newset` builds the backing `r_dict` with both
/// `space.eq_w` and `space.hash_w`, so one `add` hashes the element once and
/// compares it with `eq_w`, and either callback raising aborts the store.
/// A user `__hash__` is a collection point that can move both `obj` and
/// `item`, so the hash is taken by the caller while they are still rooted and
/// the digest handed down here; `hash` must be the `space.hash_w` result for
/// `item` (see [`object_key_hashed`](crate::dictmultiobject::object_key_hashed)).
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_add_hashed_checked(
    obj: PyObjectRef,
    item: PyObjectRef,
    hash: i64,
) -> Result<(), SetUpdateError> {
    w_set_insert_key_checked(obj, crate::dictmultiobject::object_key_hashed(item, hash))
}

/// Run a set operation with the callback-free guard raised, so
/// `dict_keys_equal` answers from its builtin type ladder and no user
/// `__eq__` observes or mutates the set mid-probe.
///
/// Returns `None` when a comparison escaped the ladder: `op` withholds its
/// mutation in that case, so the set is untouched and the caller re-runs the
/// operation by scanning entries without holding a table borrow across a
/// callback.
#[inline]
unsafe fn callback_free_set_op<T>(
    op: impl FnOnce() -> T,
) -> Option<Result<T, crate::dictmultiobject::DictKeyError>> {
    crate::dict_eq_hook::begin_callback_free_probe();
    let result = op();
    if crate::dict_eq_hook::end_callback_free_probe() {
        return None;
    }
    if crate::dictmultiobject::take_dict_key_error() {
        return Some(Err(crate::dictmultiobject::DictKeyError));
    }
    Some(Ok(result))
}

/// Find `key` by scanning same-hash entries of a *captured* storage box one at
/// a time.  `items` is the box the caller snapshotted at operation entry and
/// pinned, mirroring `AbstractUnwrappedSetStrategy`'s `d =
/// self.unerase(w_set.sstorage)` (`setobject.py:934`): the whole probe runs
/// against that box, so if a probing `__eq__` swaps the set's live storage (a
/// `clear` → `switch_to_empty_strategy`, `:922`) the scan completes against the
/// now-orphaned snapshot instead of chasing the fresh table.
///
/// The storage box has a stable address (`gc_alloc_storage_box` →
/// `try_gc_alloc_stable_raw`), so `items` never moves; the caller's pin only
/// keeps an orphaned box alive across the callbacks.  The table borrow ends
/// before equality can call user code; a callback that grows or reorders the
/// captured box restarts the scan (`ll_dict_lookup` paranoia,
/// `rordereddict.py:1058`).  Capacity is exact here, not a proxy: within one
/// box's lifetime `IndexMap` capacity only grows, and a `clear` swaps the box
/// rather than resetting this one.
unsafe fn scan_set_key_reentrant(
    items: *mut SetItemsStorage,
    mut key: crate::dictmultiobject::ObjectKey,
) -> Result<(Option<usize>, crate::dictmultiobject::ObjectKey), crate::dictmultiobject::DictKeyError>
{
    'restart: loop {
        let table_capacity = (*items).capacity();
        let mut i = 0;
        loop {
            let Some((stored_hash, stored_obj)) = (*items)
                .get_index(i)
                .map(|(stored, _)| (stored.hash, stored.obj))
            else {
                return Ok((None, key));
            };

            if stored_hash == key.hash {
                let _roots = crate::gc_roots::push_roots();
                let stored_slot = crate::gc_roots::shadow_stack_len();
                crate::gc_roots::pin_root(stored_obj);
                let key_slot = crate::gc_roots::shadow_stack_len();
                crate::gc_roots::pin_root(key.obj);

                let equal = crate::dictmultiobject::dict_keys_equal(stored_obj, key.obj);
                let stored_obj = crate::gc_roots::shadow_stack_get(stored_slot);
                key.obj = crate::gc_roots::shadow_stack_get(key_slot);
                if crate::dictmultiobject::take_dict_key_error() {
                    return Err(crate::dictmultiobject::DictKeyError);
                }
                // Validate the paranoia condition before acting on the result:
                // `ll_dict_lookup` restarts even when the comparison answered
                // `true`, because a callback that reallocated the buffer or moved
                // the candidate leaves the matched index stale
                // (`rordereddict.py:1058`).
                let disturbed = (*items).capacity() != table_capacity
                    // `entries.valid(index) && entries[index].key == checkingkey`.
                    || !(*items).get_index(i).is_some_and(|(stored, _)| {
                        stored.hash == stored_hash && stored.obj == stored_obj
                    });
                if disturbed {
                    continue 'restart;
                }
                if equal {
                    return Ok((Some(i), key));
                }
            }
            i += 1;
        }
    }
}

/// Snapshot the set's storage box and pin it for a reentrant probe, mirroring
/// PyPy's capture-before-probe (`d = self.unerase(w_set.sstorage)`).  The pin
/// keeps the box alive even if a probing `__eq__` swaps the set's live storage
/// (`w_set_clear`); the returned pointer is used for the whole operation.
#[inline]
unsafe fn capture_set_items(obj: PyObjectRef) -> *mut SetItemsStorage {
    let items = (*(obj as *const W_SetObject)).items;
    crate::gc_roots::pin_root(items as PyObjectRef);
    items
}

/// Store a key that carries its own digest, propagating an `eq_w` raise from
/// the bucket probe.
///
/// `setobject.py _intersect_unwrapped` places a key it took from another
/// set with `setitem_with_hash(result, key, keyhash, None)`, i.e. under the
/// digest the key already carries rather than one taken afresh.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `key.hash` must be the
/// `space.hash_w` digest of `key.obj`.
pub unsafe fn w_set_insert_key_checked(
    obj: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError> {
    w_set_insert_key_reentrant(obj, key)
}

/// Membership test for a key that carries its own digest, propagating an
/// `eq_w` raise from the bucket probe.
///
/// `setobject.py _intersect_unwrapped` probes the other side with
/// `contains_with_hash(d_other, key, keyhash)`, reusing the digest the key was
/// stored under instead of hashing it again.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `key.hash` must be the
/// `space.hash_w` digest of `key.obj`.
pub unsafe fn w_set_contains_key_checked(
    obj: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<bool, crate::dictmultiobject::DictKeyError> {
    if let Some(result) = callback_free_set_op(|| {
        let s = &*(obj as *const W_SetObject);
        (*s.items).contains_key(&key)
    }) {
        return result;
    }
    let _roots = crate::gc_roots::push_roots();
    let items = capture_set_items(obj);
    let (found, _) = scan_set_key_reentrant(items, key)?;
    Ok(found.is_some())
}

/// Remove a key that carries its own digest, propagating an `eq_w` raise from
/// the bucket probe. Returns true when an element was removed.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `key.hash` must be the
/// `space.hash_w` digest of `key.obj`.
pub unsafe fn w_set_discard_key_checked(
    obj: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<bool, crate::dictmultiobject::DictKeyError> {
    if let Some(result) = callback_free_set_op(|| {
        let s = &mut *(obj as *mut W_SetObject);
        let index = (*s.items).get_index_of(&key);
        if crate::dict_eq_hook::callback_free_probe_broken() {
            return false;
        }
        match index {
            Some(index) => {
                (*s.items).shift_remove_index(index);
                s.len = (*s.items).len();
                s.hash = -1;
                true
            }
            None => false,
        }
    }) {
        return result;
    }

    let _roots = crate::gc_roots::push_roots();
    let items = capture_set_items(obj);
    let (found, _) = scan_set_key_reentrant(items, key)?;
    if let Some(index) = found {
        // Remove from the captured box; a `clear` during the probe orphans it,
        // leaving the live storage untouched (`discard` of an absent element).
        (*items).shift_remove_index(index);
        let s = &mut *(obj as *mut W_SetObject);
        s.len = (*s.items).len();
        s.hash = -1;
        return Ok(true);
    }
    Ok(false)
}

/// Membership test.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_contains(obj: PyObjectRef, item: PyObjectRef) -> bool {
    let key = crate::dictmultiobject::object_key_for(item);
    w_set_contains_key_checked(obj, key).unwrap_or(false)
}

/// Fallible variant of [`w_set_contains`].
///
/// `setobject.py EmptySetStrategy.has_key` — the element is hashed
/// even when the set is empty, so an unhashable element raises rather than
/// reading as absent.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_contains_checked(
    obj: PyObjectRef,
    item: PyObjectRef,
) -> Result<bool, crate::dictmultiobject::DictKeyError> {
    let key = crate::dictmultiobject::object_key_for_checked(item)?;
    w_set_contains_key_checked(obj, key)
}

/// Remove an element if present. Returns true when removed.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_discard(obj: PyObjectRef, item: PyObjectRef) -> bool {
    let key = crate::dictmultiobject::object_key_for(item);
    w_set_discard_key_checked(obj, key).unwrap_or(false)
}

/// Fallible variant of [`w_set_discard`].
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_discard_checked(
    obj: PyObjectRef,
    item: PyObjectRef,
) -> Result<bool, crate::dictmultiobject::DictKeyError> {
    let key = crate::dictmultiobject::object_key_for_checked(item)?;
    w_set_discard_key_checked(obj, key)
}

/// Remove every element.
///
/// `setobject.py W_BaseSetObject.clear` swaps the strategy to empty
/// (`switch_to_empty_strategy`, `:922`), which installs a *fresh* storage box
/// rather than emptying the current one.  A probe that captured the old box
/// before the clear (`scan_set_key_reentrant`) therefore keeps running against
/// that orphaned snapshot, matching PyPy: the element it later inserts lands in
/// the dropped box and is lost, and a membership test completes as if the clear
/// had not happened.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_clear(obj: PyObjectRef) {
    let s = &mut *(obj as *mut W_SetObject);
    s.items =
        crate::gc_storage::gc_alloc_storage_box(SetItemsStorage::new(), set_items_gc_type_id());
    s.len = 0;
    s.hash = -1;
    set_write_barrier(obj);
}

/// Remove and return an arbitrary stored element without hashing it again.
///
/// `setobject.py ObjectSetStrategy.popitem` delegates to the backing
/// dictionary's `popitem`; the key already occupies a bucket, so no user
/// `__hash__` or `__eq__` callback occurs while it leaves the set.
///
/// # Safety
/// `obj` must point to a valid mutable `W_SetObject`.
pub unsafe fn w_set_popitem(obj: PyObjectRef) -> Option<PyObjectRef> {
    let s = &mut *(obj as *mut W_SetObject);
    let entries = &mut *s.items;
    let (key, ()) = entries.pop()?;
    s.len -= 1;
    s.hash = -1;
    Some(key.obj)
}

/// Take over a copy of another set's storage, keeping the digest each element
/// was stored under.
///
/// `setobject.py:875` assigns a fresh storage table to the GC pointer field
/// (`w_set.sstorage = w_other.get_storage_copy()`), while
/// `ObjectSetStrategy.get_storage_copy` (`setobject.py:963`) creates that table
/// with `self.erase(d.copy())`. PyPy's underlying table is the GC-managed
/// `rdict.py:210` `GcStruct("dicttable")`. Do the same field reassignment here,
/// rather than overwriting the old table's pointee. Copying the buckets is
/// what makes the operand's elements reach the new set without handing them
/// to a user `__hash__` (or `__eq__`) a second time.
///
/// Cloning does not call back into user code, and the storage box uses the
/// non-collecting stable old-generation allocator, so there is no collection
/// point between reading `src`'s table and installing the new field value.
///
/// # Safety
/// `dst` and `src` must point to valid `W_SetObject`s.
pub unsafe fn w_set_copy_storage_from(dst: PyObjectRef, src: PyObjectRef) {
    let d = &mut *(dst as *mut W_SetObject);
    let copied = (*(*(src as *const W_SetObject)).items).clone();
    d.items = crate::gc_storage::gc_alloc_storage_box(copied, set_items_gc_type_id());
    d.len = (*d.items).len();
    d.hash = -1;
    set_write_barrier(dst);
}

/// Remove a set operand's elements, keeping the digests it holds.
///
/// `setobject.py _difference_update_unwrapped` — the operand's keys
/// are deleted out of self under the digests they already carry
/// (`delitem_with_hash`), and a missing one is not an error.
///
/// `:1032-1034` gives the two sides sharing one storage its own branch: that is
/// `s -= s`, which empties self. It also cannot be done by the walk below —
/// removing renumbers the very storage being walked, so every second element
/// would be stepped over.
///
/// # Safety
/// `dst` and `src` must point to valid `W_SetObject`s.
pub unsafe fn w_set_difference_update_from_set(
    dst: PyObjectRef,
    src: PyObjectRef,
) -> Result<(), SetUpdateError> {
    if std::ptr::eq(
        (*(dst as *const W_SetObject)).items,
        (*(src as *const W_SetObject)).items,
    ) {
        w_set_clear(dst);
        set_write_barrier(dst);
        return Ok(());
    }
    // setobject.py:1051-1054 — small_set -= big_set computes a fresh
    // difference by walking the smaller self storage, then replaces self's
    // storage wholesale. Besides the complexity bound, this preserves the
    // exact contains-with-hash callback direction of the upstream strategy.
    if w_set_len(dst) < w_set_len(src) {
        let result = w_set_new();
        let dst_items = (*(dst as *const W_SetObject)).items;
        let dst_len = (*dst_items).len();
        let mut i = 0;
        while i < dst_len {
            let Some(key) = w_set_key_at(dst, i) else {
                return Err(SetUpdateError::ChangedSize);
            };
            if !w_set_contains_key_for_update(src, key)? {
                if (*(dst as *const W_SetObject)).items != dst_items
                    || (*dst_items).len() != dst_len
                {
                    return Err(SetUpdateError::ChangedSize);
                }
                // The comparison may clear or otherwise shorten `dst`.
                // CPython's set probe restarts when `entry->key` changes;
                // once this live index disappeared there is no surviving
                // entry to copy into the difference result.
                let Some(key) = w_set_key_at(dst, i) else {
                    return Err(SetUpdateError::ChangedSize);
                };
                w_set_insert_key_checked(result, key)?;
            }
            i += 1;
        }
        w_set_copy_storage_from(dst, result);
        return Ok(());
    }
    // `src` is a distinct storage, so removing from `dst` cannot renumber it.
    let src_items = (*(src as *const W_SetObject)).items;
    let src_len = (*src_items).len();
    let mut i = 0;
    while i < src_len {
        let Some(key) = w_set_key_at(src, i) else {
            return Err(SetUpdateError::ChangedSize);
        };
        w_set_remove_key_for_update(dst, key)?;
        if (*(src as *const W_SetObject)).items != src_items || (*src_items).len() != src_len {
            return Err(SetUpdateError::ChangedSize);
        }
        i += 1;
    }
    Ok(())
}

/// Merge a set operand's storage in, keeping the digests it holds.
///
/// `setobject.py ObjectSetStrategy.update` takes `d_obj.update(
/// d_other)` when the operand shares this strategy — labelled "optimization
/// only" upstream, but it is also what keeps a set operand's elements from
/// being handed to a user `__hash__` a second time. Elements equal across the
/// two sides still meet in a bucket, so `eq_w` runs and can raise.
///
/// # Safety
/// `dst` and `src` must point to valid `W_SetObject`s.
pub unsafe fn w_set_update_from_set(
    dst: PyObjectRef,
    src: PyObjectRef,
) -> Result<(), SetUpdateError> {
    if std::ptr::eq(
        (*(dst as *const W_SetObject)).items,
        (*(src as *const W_SetObject)).items,
    ) {
        return Ok(());
    }
    // Both tables are captured once for the whole merge — `update` unerases
    // `d_obj` up front (`setobject.py:1396`) and `d_obj.update(d_other)` runs
    // `ll_dict_update(dic1, dic2)` on those two tables (`rordereddict.py:1379`).
    // A callback that clears either set swaps its live storage; the merge keeps
    // reading the captured source and inserting into the captured destination,
    // both now orphaned snapshots.
    //
    // `src`'s keys are still read one index at a time rather than collected: an
    // `eq_w` raised from the bucket probe below can move every element, and
    // the collector rewrites the `obj` slots inside the two tables in place
    // (`set_object_custom_trace`) — a `Vec` of keys lifted out of them would
    // not be walked and would be left holding stale pointers.
    let _roots = crate::gc_roots::push_roots();
    let dst_items = capture_set_items(dst);
    let src_items = capture_set_items(src);
    let mut i = 0;
    loop {
        let Some((&key, _)) = (*src_items).get_index(i) else {
            break;
        };
        w_set_insert_key_into(dst, dst_items, key)?;
        i += 1;
    }
    Ok(())
}

/// Build the elements on exactly one of the two sides as a fresh set.
///
/// `_symmetric_difference_unwrapped` (`setobject.py:1062-1074`) unerases both
/// tables once — `d_this` and `d_other` — then walks the other side first and
/// this side second, probing each stored `(key, hash)` pair against the
/// *captured* opposite table and placing survivors into a fresh `d_new` under
/// the digest they already carry.  Because both captures happen up front, an
/// `eq_w` that clears either set mid-walk orphans that table without steering
/// the walk or the membership probes onto the replacement storage; the caller
/// then installs the result wholesale (`w_set.sstorage = storage`, `:1114`).
///
/// # Safety
/// `w_set` and `w_other` must point to valid `W_SetObject`s.
pub unsafe fn w_set_symmetric_difference_storage(
    w_set: PyObjectRef,
    w_other: PyObjectRef,
) -> Result<PyObjectRef, SetUpdateError> {
    let _roots = crate::gc_roots::push_roots();
    let d_new = w_set_new();
    // The fresh set is only reachable from this frame while the probes below
    // run user code; pin it (set bodies are non-moving, so no reload).
    crate::gc_roots::pin_root(d_new);
    let d_this = capture_set_items(w_set);
    let d_other = capture_set_items(w_other);
    for (walk, probe) in [(d_other, d_this), (d_this, d_other)] {
        let mut i = 0;
        loop {
            let Some((&key, _)) = (*walk).get_index(i) else {
                break;
            };
            let (found, key) = scan_set_key_reentrant(probe, key).map_err(SetUpdateError::Key)?;
            if found.is_none() {
                w_set_insert_key_checked(d_new, key)?;
            }
            i += 1;
        }
    }
    Ok(d_new)
}

/// Failure modes of the PyPy `ObjectSetStrategy.update` table merge.
pub enum SetUpdateError {
    /// An equality callback raised; its concrete exception is parked in the
    /// interpreter's pending dict-key error slot.
    Key(crate::dictmultiobject::DictKeyError),
    /// A callback changed one of the tables while it was being traversed.
    ChangedSize,
}

/// Insert one cached-hash key without holding an IndexMap borrow across user
/// `eq_w`.
///
/// The set's storage box is captured and pinned once at entry
/// (`capture_set_items`), mirroring PyPy's `d = self.unerase(w_set.sstorage)`
/// (`setobject.py:942`): the membership probe and the follow-up insert both run
/// against that box.  If a probing `__eq__` clears the set, the fresh box
/// installed by `w_set_clear` replaces the live storage while this insert still
/// targets the orphaned snapshot, so the element lands in the dropped box and
/// is lost — the behaviour PyPy exhibits when `add` captures its storage before
/// a re-entrant `clear`.  The scan drops the table borrow before each
/// comparison and inserts through an already-proven vacant raw entry, so
/// insertion performs no second user callback.
unsafe fn w_set_insert_key_reentrant(
    dst: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError> {
    let _roots = crate::gc_roots::push_roots();
    let items = capture_set_items(dst);
    w_set_insert_key_into(dst, items, key)
}

/// Probe-and-insert half of [`w_set_insert_key_reentrant`] against a storage
/// box the caller already captured and pinned.  `w_set_update_from_set` passes
/// the box it captured for the whole merge so every source key targets the same
/// (possibly orphaned) table, the way `ll_dict_update` keeps inserting into its
/// captured `dic1`.
unsafe fn w_set_insert_key_into(
    dst: PyObjectRef,
    items: *mut SetItemsStorage,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError> {
    let (found, key) = scan_set_key_reentrant(items, key).map_err(SetUpdateError::Key)?;
    if found.is_some() {
        return Ok(());
    }
    let entries = &mut *items;
    let hash = entries.hasher().hash_one(&key);
    match entries.raw_entry_mut_v1().from_hash(hash, |_| false) {
        RawEntryMut::Vacant(entry) => {
            entry.insert_hashed_nocheck(hash, key, ());
        }
        RawEntryMut::Occupied(_) => unreachable!("a never-matching raw probe is vacant"),
    }
    let set = &mut *(dst as *mut W_SetObject);
    set.len = (*set.items).len();
    set.hash = -1;
    set_write_barrier(dst);
    Ok(())
}

/// Membership half of `contains_with_hash` for a mutation-sensitive set
/// operation.  As with the update inserter above, no IndexMap borrow crosses
/// `eq_w`; a size-changing callback invalidates the traversal explicitly.
unsafe fn w_set_contains_key_for_update(
    probe: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<bool, SetUpdateError> {
    let items = (*(probe as *const W_SetObject)).items;
    let len = (*items).len();
    let mut i = 0;
    while i < len {
        let Some((&stored, _)) = (*items).get_index(i) else {
            return Err(SetUpdateError::ChangedSize);
        };
        if stored.hash == key.hash {
            let equal = crate::dictmultiobject::dict_keys_equal(stored.obj, key.obj);
            if crate::dictmultiobject::take_dict_key_error() {
                return Err(SetUpdateError::Key(crate::dictmultiobject::DictKeyError));
            }
            if (*(probe as *const W_SetObject)).items != items || (*items).len() != len {
                return Err(SetUpdateError::ChangedSize);
            }
            if equal {
                return Ok(true);
            }
        }
        i += 1;
    }
    Ok(false)
}

/// `delitem_with_hash` for a mutation-sensitive difference update.  Find the
/// matching bucket without lending IndexMap across Python code, then delete
/// the proven index without another equality callback.
unsafe fn w_set_remove_key_for_update(
    dst: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError> {
    let items = (*(dst as *const W_SetObject)).items;
    let len = (*items).len();
    let mut found = None;
    let mut i = 0;
    while i < len {
        let Some((&stored, _)) = (*items).get_index(i) else {
            return Err(SetUpdateError::ChangedSize);
        };
        if stored.hash == key.hash {
            let equal = crate::dictmultiobject::dict_keys_equal(stored.obj, key.obj);
            if crate::dictmultiobject::take_dict_key_error() {
                return Err(SetUpdateError::Key(crate::dictmultiobject::DictKeyError));
            }
            if (*(dst as *const W_SetObject)).items != items || (*items).len() != len {
                return Err(SetUpdateError::ChangedSize);
            }
            if equal {
                found = Some(i);
                break;
            }
        }
        i += 1;
    }
    if let Some(index) = found {
        (*items).shift_remove_index(index);
        let set = &mut *(dst as *mut W_SetObject);
        set.len -= 1;
        set.hash = -1;
    }
    Ok(())
}

/// Number of elements in the set.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_len(obj: PyObjectRef) -> usize {
    (*(obj as *const W_SetObject)).len
}

#[inline]
pub unsafe fn w_set_capacity(obj: PyObjectRef) -> usize {
    let s = &*(obj as *const W_SetObject);
    (*s.items).capacity()
}

/// Cached frozenset hash; `-1` is the uncomputed sentinel.
#[inline]
pub unsafe fn w_frozenset_cached_hash(obj: PyObjectRef) -> Option<i64> {
    let hash = (*(obj as *const W_SetObject)).hash;
    (hash != -1).then_some(hash)
}

#[inline]
pub unsafe fn w_frozenset_set_cached_hash(obj: PyObjectRef, hash: i64) {
    (*(obj as *mut W_SetObject)).hash = hash;
}

/// Digests already carried by the r_dict keys. Python 3.14 frozenset hashing
/// consumes these instead of invoking each element's `__hash__` again.
pub unsafe fn w_set_stored_hashes(obj: PyObjectRef) -> Vec<i64> {
    let s = &*(obj as *const W_SetObject);
    (*s.items).keys().map(|key| key.hash).collect()
}

/// The key at `index`, carrying the digest it was stored under, or `None` once
/// `index` reaches the end.
///
/// `setobject.py iterkeys_with_hash` walks a storage handing out
/// `(key, keyhash)` pairs so the walk's consumer can place or probe the key
/// without hashing it again. Reading one index at a time lets a caller whose
/// loop body reaches user code (an `eq_w` from a bucket probe) re-read the key
/// afterwards: the collector rewrites the `obj` slots inside the table in place
/// (`set_object_custom_trace`), so a key read before that point can be stale
/// while the table itself stays correct.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_key_at(
    obj: PyObjectRef,
    index: usize,
) -> Option<crate::dictmultiobject::ObjectKey> {
    let s = &*(obj as *const W_SetObject);
    (*s.items).get_index(index).map(|(&key, _)| key)
}

/// Snapshot the contained elements as a `Vec`.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_items(obj: PyObjectRef) -> Vec<PyObjectRef> {
    let s = &*(obj as *const W_SetObject);
    (*s.items).keys().map(|key| key.obj).collect()
}

/// Walk, in place, every element `PyObjectRef` slot of a set for GC root
/// forwarding.  Forwards each `ObjectKey.obj` slot: `ObjectKey.hash` is
/// identity-stable across a GC move, so writing the relocated pointer through
/// the key's `obj` slot keeps the bucket index valid.  Alloc-free — unlike
/// [`w_set_items`], which materialises a `Vec`.  The port of
/// `set_object_custom_trace`.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_walk_gc_refs(obj: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
    let set = &mut *(obj as *mut W_SetObject);
    if set.items.is_null() {
        return;
    }
    let entries = &mut *set.items;
    for (key, _) in entries.iter_mut() {
        let key_ptr = key as *const crate::dictmultiobject::ObjectKey
            as *mut crate::dictmultiobject::ObjectKey;
        visitor(std::ptr::addr_of_mut!((*key_ptr).obj) as *mut PyObjectRef);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::w_int_new;

    fn install_test_hash_hook() {
        unsafe fn hash_int(obj: PyObjectRef) -> i64 {
            crate::w_int_get_value(obj)
        }

        unsafe fn hash_str(_ptr: *const u8, _len: usize) -> i64 {
            0
        }

        crate::dict_eq_hook::register_hash_w_hook(hash_int);
        crate::dict_eq_hook::register_hash_str_hook(hash_str);
    }

    #[test]
    fn add_dedupes() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(2));
            assert_eq!(w_set_len(s), 2);
            assert!(w_set_contains(s, w_int_new(1)));
            assert!(w_set_contains(s, w_int_new(2)));
            assert!(!w_set_contains(s, w_int_new(3)));
        }
    }

    #[test]
    fn discard_removes() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(2));
            assert!(w_set_discard(s, w_int_new(1)));
            assert!(!w_set_discard(s, w_int_new(99)));
            assert_eq!(w_set_len(s), 1);
            assert!(w_set_contains(s, w_int_new(2)));
        }
    }

    #[test]
    fn frozenset_distinct_type() {
        let s = w_set_new();
        let fs = w_frozenset_new();
        unsafe {
            assert!(is_set(s));
            assert!(!is_frozenset(s));
            assert!(is_frozenset(fs));
            assert!(!is_set(fs));
        }
    }

    #[test]
    fn w_set_gc_type_id_matches_descr() {
        assert_eq!(W_SET_GC_TYPE_ID, 30);
        assert_eq!(
            <W_SetObject as crate::lltype::GcType>::type_id(),
            W_SET_GC_TYPE_ID
        );
        assert_eq!(
            <W_SetObject as crate::lltype::GcType>::SIZE,
            W_SET_OBJECT_SIZE
        );
    }
}
