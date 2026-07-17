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
use pyre_macros::pyre_class;

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
    W_SetIterObject::allocate(W_SetIterObject {
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
    !obj.is_null() && (*obj).ob_type == &SET_ITERATOR_TYPE as *const PyType
}

#[inline]
pub unsafe fn w_set_iter_get_set(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_SetIterObject)).w_set
}

#[inline]
pub unsafe fn w_set_iter_set_set(obj: PyObjectRef, w_set: PyObjectRef) {
    (*(obj as *mut W_SetIterObject)).w_set = w_set;
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
/// Layout: `[ob_type | items | len]`. `items` is a heap-owned hashed
/// `IndexMap<ObjectKey, ()>`, mirroring the dict object strategy while
/// keeping the struct trivially `Copy`-friendly for the JIT raw-pointer model.
#[repr(C)]
pub struct W_SetObject {
    pub ob_header: PyObject,
    pub items: *mut indexmap::IndexMap<crate::dictmultiobject::ObjectKey, ()>,
    pub len: usize,
    /// setobject.py:584 `W_FrozensetObject.hash = DEFAULT_HASH`.
    pub hash: i64,
}

/// GC type id assigned to `W_SetObject` at JitDriver init time.
pub const W_SET_GC_TYPE_ID: u32 = 30;

/// Fixed payload size (`framework.py:811`).
pub const W_SET_OBJECT_SIZE: usize = std::mem::size_of::<W_SetObject>();

impl crate::lltype::GcType for W_SetObject {
    fn type_id() -> u32 {
        W_SET_GC_TYPE_ID
    }
    const SIZE: usize = W_SET_OBJECT_SIZE;
}

/// Free the off-GC item container owned by a `W_SetObject`.
///
/// # Safety
/// `obj` must point at a valid `W_SetObject` whose `items` Box is not aliased
/// by another owner.
pub unsafe fn w_set_dealloc_items(obj: PyObjectRef) {
    let raw = &mut *(obj as *mut W_SetObject);
    if !raw.items.is_null() {
        drop(Box::from_raw(raw.items));
        raw.items = std::ptr::null_mut();
    }
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
    let items = crate::lltype::malloc_raw(indexmap::IndexMap::<
        crate::dictmultiobject::ObjectKey,
        (),
    >::new());
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
    let items = crate::lltype::malloc_raw(indexmap::IndexMap::<
        crate::dictmultiobject::ObjectKey,
        (),
    >::new());
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
    let s = &mut *(obj as *mut W_SetObject);
    let entries = &mut *s.items;
    let key = crate::dictmultiobject::object_key_for(item);
    if entries.insert(key, ()).is_none() {
        s.len += 1;
        s.hash = -1;
        set_write_barrier(obj);
    }
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
) -> Result<(), crate::dictmultiobject::DictKeyError> {
    w_set_insert_key_checked(obj, crate::dictmultiobject::object_key_hashed(item, hash))
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
) -> Result<(), crate::dictmultiobject::DictKeyError> {
    let s = &mut *(obj as *mut W_SetObject);
    let entries = &mut *s.items;
    // Single insert probe, matching `r_dict.setitem`'s one bucket scan: an
    // `eq_w` that raises mid-probe reads as "not equal", so `insert` appends a
    // spurious entry at the end; drop it so the add leaves the set unchanged.
    let appended = entries.insert(key, ()).is_none();
    if crate::dictmultiobject::take_dict_key_error() {
        if appended {
            entries.pop();
        }
        return Err(crate::dictmultiobject::DictKeyError);
    }
    if appended {
        s.len += 1;
        s.hash = -1;
        set_write_barrier(obj);
    }
    Ok(())
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
    let s = &*(obj as *const W_SetObject);
    let entries = &*s.items;
    let found = entries.contains_key(&key);
    if crate::dictmultiobject::take_dict_key_error() {
        return Err(crate::dictmultiobject::DictKeyError);
    }
    Ok(found)
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
    let s = &mut *(obj as *mut W_SetObject);
    let entries = &mut *s.items;
    let removed = entries.shift_remove(&key).is_some();
    if crate::dictmultiobject::take_dict_key_error() {
        return Err(crate::dictmultiobject::DictKeyError);
    }
    if removed {
        s.len -= 1;
        s.hash = -1;
    }
    Ok(removed)
}

/// Membership test.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_contains(obj: PyObjectRef, item: PyObjectRef) -> bool {
    let s = &*(obj as *const W_SetObject);
    let entries = &*s.items;
    entries.contains_key(&crate::dictmultiobject::object_key_for(item))
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
    let s = &*(obj as *const W_SetObject);
    let entries = &*s.items;
    let found = entries.contains_key(&key);
    if crate::dictmultiobject::take_dict_key_error() {
        return Err(crate::dictmultiobject::DictKeyError);
    }
    Ok(found)
}

/// Remove an element if present. Returns true when removed.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_discard(obj: PyObjectRef, item: PyObjectRef) -> bool {
    let s = &mut *(obj as *mut W_SetObject);
    let entries = &mut *s.items;
    if entries
        .shift_remove(&crate::dictmultiobject::object_key_for(item))
        .is_some()
    {
        s.len -= 1;
        s.hash = -1;
        true
    } else {
        false
    }
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
    let s = &mut *(obj as *mut W_SetObject);
    let entries = &mut *s.items;
    let removed = entries.shift_remove(&key).is_some();
    if crate::dictmultiobject::take_dict_key_error() {
        return Err(crate::dictmultiobject::DictKeyError);
    }
    if removed {
        s.len -= 1;
        s.hash = -1;
    }
    Ok(removed)
}

/// Remove every element.
///
/// `setobject.py W_BaseSetObject.clear` — the storage is dropped in one
/// go, so no element is looked up (and so re-hashed) on the way out.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_clear(obj: PyObjectRef) {
    let s = &mut *(obj as *mut W_SetObject);
    (*s.items).clear();
    s.len = 0;
    s.hash = -1;
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
/// `setobject.py set_strategy_and_setdata` hands a set operand's
/// storage to the new set (`w_set.sstorage = w_iterable.get_storage_copy()`)
/// rather than walking its elements back in, and `setobject.py
/// ObjectSetStrategy.get_storage_copy` makes that copy with `d.copy()` — one
/// bulk copy of the table, not an element-by-element refill. Copying the
/// buckets is what makes the operand's elements reach the new set without
/// being handed to a user `__hash__` (or `__eq__`) a second time.
///
/// Because nothing here allocates or calls back into user code, there is no
/// collection point between reading `src`'s table and installing it.
///
/// # Safety
/// `dst` and `src` must point to valid `W_SetObject`s.
pub unsafe fn w_set_copy_storage_from(dst: PyObjectRef, src: PyObjectRef) {
    let d = &mut *(dst as *mut W_SetObject);
    *d.items = (*(*(src as *const W_SetObject)).items).clone();
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
) -> Result<(), crate::dictmultiobject::DictKeyError> {
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
        let mut i = 0;
        while let Some(key) = w_set_key_at(dst, i) {
            if !w_set_contains_key_checked(src, key)? {
                let key = w_set_key_at(dst, i).expect("probing the other set cannot shorten self");
                w_set_insert_key_checked(result, key)?;
            }
            i += 1;
        }
        w_set_copy_storage_from(dst, result);
        return Ok(());
    }
    // `src` is a distinct storage, so removing from `dst` cannot renumber it.
    let mut i = 0;
    while let Some(key) = w_set_key_at(src, i) {
        w_set_discard_key_checked(dst, key)?;
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
) -> Result<(), crate::dictmultiobject::DictKeyError> {
    if std::ptr::eq(
        (*(dst as *const W_SetObject)).items,
        (*(src as *const W_SetObject)).items,
    ) {
        return Ok(());
    }
    // `src`'s keys are read one index at a time rather than collected: an
    // `eq_w` raised from the bucket probe below can move every element, and
    // the collector rewrites the `obj` slots inside the two tables in place
    // (`set_object_custom_trace`) — a `Vec` of keys lifted out of them would
    // not be walked and would be left holding stale pointers.
    let mut i = 0;
    while i < (*(*(src as *const W_SetObject)).items).len() {
        let Some((&key, _)) = (*(*(src as *const W_SetObject)).items).get_index(i) else {
            break;
        };
        w_set_insert_key_checked(dst, key)?;
        i += 1;
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
