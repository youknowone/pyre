//! The ordered dict of `rpython/rtyper/lltypesystem/rordereddict.py`.
//!
//! Every RPython `dict` translates to this structure, and it is what a
//! `W_DictObject` strategy erases its storage to.  The shape is upstream's: a
//! power-of-two `indexes` table holding entry numbers biased by
//! [`VALID_OFFSET`], plus an insertion-ordered `entries` array whose dead slots
//! are tombstones rather than holes.
//!
//! The consequence that matters is [`RDict::remove`].  `ll_dict_delitem`
//! (rordereddict.py) writes [`DELETED`] into the one index slot that named
//! the entry and marks the entry dead; no other entry moves, so a delete is
//! O(1) and draining a dict is linear.  Entries are compacted only where
//! upstream compacts them — `ll_dict_remove_deleted_items` (803), reached from
//! `ll_dict_grow` (755) and `_ll_dict_resize_to` (735).
//!
//! Insertion order is carried by the `entries` array, so it survives deletes
//! without any compaction: a re-inserted key appends at the end, exactly as it
//! does upstream and in a `dict`.
//!
//! # Slots are not positions
//!
//! `num_ever_used_items` and [`RDict::len`] (`num_live_items`) diverge when a
//! slot is a tombstone.  `entries` is `DICTENTRYARRAY` (`get_ll_dict`): a
//! GC array whose `length` is the allocation, null when that length is 0.
//! Every index this type hands out or accepts — [`RDict::index_of`],
//! [`RDict::get_slot`], [`RDict::remove_slot`] — is a **slot**, an index into
//! `entries`, never the n-th live pair.  Walk a dict with `0..d.entry_slots()`
//! and skip the slots whose `f_valid` is false, which is what `_ll_dictnext`
//! does.

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};

pub use crate::rordereddict_entries::{
    Entry, EntryDummy, GcEntries, GcEntriesType, GcRefOffsets, alloc_entries,
    entries_allocated_len, entries_item_ptr, set_bytes_key_pyobject_entries_gc_type_id,
    set_i64_pyobject_entries_gc_type_id, set_identity_key_pyobject_entries_gc_type_id,
    set_object_key_pyobject_entries_gc_type_id, set_object_key_unit_entries_gc_type_id,
    set_str_key_pyobject_entries_gc_type_id,
};

/// An index slot naming no entry, and one whose entry has been deleted.
///
/// `FREE` ends a probe; `DELETED` does not, because the key being looked for
/// may have been stored past it (rordereddict.py).
pub const FREE: u32 = 0;
/// See [`FREE`].
pub const DELETED: u32 = 1;
/// The bias an entry number carries inside the index table, so that entry 0 is
/// distinguishable from [`FREE`] (rordereddict.py).
pub const VALID_OFFSET: u32 = 2;

/// `DICT_INITSIZE` (rordereddict.py).
const DICT_INITSIZE: usize = 16;
/// `PERTURB_SHIFT` (rordereddict.py).
const PERTURB_SHIFT: u32 = 5;

/// Walk live `d.entries` slots with `ll_getitem_fast`, not
/// `Enumerate` / `FilterMap`.
pub struct LiveIter<'a, K, V> {
    entries: &'a [Entry<K, V>],
    front: usize,
    back: usize,
}

impl<'a, K, V> LiveIter<'a, K, V> {
    fn new(entries: &'a [Entry<K, V>]) -> Self {
        Self {
            entries,
            front: 0,
            back: entries.len(),
        }
    }

    fn entry_at(&self, i: usize) -> Option<(&'a K, &'a V)> {
        let e = &self.entries[i];
        if e.f_valid {
            Some((&e.key, &e.value))
        } else {
            None
        }
    }
}

impl<'a, K, V> Iterator for LiveIter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        while self.front < self.back {
            let i = self.front;
            self.front += 1;
            if let Some(item) = self.entry_at(i) {
                return Some(item);
            }
        }
        None
    }
}

impl<K, V> DoubleEndedIterator for LiveIter<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while self.front < self.back {
            self.back -= 1;
            if let Some(item) = self.entry_at(self.back) {
                return Some(item);
            }
        }
        None
    }
}

pub struct LiveSlotIter<'a, K, V> {
    inner: LiveIter<'a, K, V>,
}

impl<'a, K, V> Iterator for LiveSlotIter<'a, K, V> {
    type Item = (usize, &'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        while self.inner.front < self.inner.back {
            let i = self.inner.front;
            self.inner.front += 1;
            if let Some((k, v)) = self.inner.entry_at(i) {
                return Some((i, k, v));
            }
        }
        None
    }
}

impl<K, V> DoubleEndedIterator for LiveSlotIter<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while self.inner.front < self.inner.back {
            self.inner.back -= 1;
            let i = self.inner.back;
            if let Some((k, v)) = self.inner.entry_at(i) {
                return Some((i, k, v));
            }
        }
        None
    }
}

pub struct LiveKeys<'a, K, V> {
    inner: LiveIter<'a, K, V>,
}

impl<'a, K, V> Iterator for LiveKeys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }
}

impl<K, V> DoubleEndedIterator for LiveKeys<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(k, _)| k)
    }
}

pub struct LiveValues<'a, K, V> {
    inner: LiveIter<'a, K, V>,
}

impl<'a, K, V> Iterator for LiveValues<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }
}

impl<K, V> DoubleEndedIterator for LiveValues<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(_, v)| v)
    }
}

pub struct LiveIterMut<'a, K, V> {
    entries: *mut Entry<K, V>,
    len: usize,
    front: usize,
    back: usize,
    _mark: std::marker::PhantomData<&'a mut Entry<K, V>>,
}

// `entries` is derived from the `&'a mut [Entry<K, V>]` handed to
// `new`, so the iterator holds that slice's unique borrow, and `front` / `back`
// only ever move toward each other — no index is yielded twice and the two ends
// cannot hand out the same item. That makes the raw pointer carry exactly the
// thread-safety of `slice::IterMut`, whose bounds these mirror; storing it as a
// pointer rather than a `FilterMap<slice::IterMut<_>>` is what dropped the auto
// traits the iterator used to infer.
unsafe impl<K: Send, V: Send> Send for LiveIterMut<'_, K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for LiveIterMut<'_, K, V> {}

impl<'a, K, V> LiveIterMut<'a, K, V> {
    fn new(entries: &'a mut [Entry<K, V>]) -> Self {
        let len = entries.len();
        Self {
            entries: entries.as_mut_ptr(),
            len,
            front: 0,
            back: len,
            _mark: std::marker::PhantomData,
        }
    }

    fn entry_at(&mut self, i: usize) -> Option<(&'a K, &'a mut V)> {
        debug_assert!(i < self.len);
        let e = unsafe { &mut *self.entries.add(i) };
        if e.f_valid {
            Some((&e.key, &mut e.value))
        } else {
            None
        }
    }
}

impl<'a, K, V> Iterator for LiveIterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        while self.front < self.back {
            let i = self.front;
            self.front += 1;
            if let Some(item) = self.entry_at(i) {
                return Some(item);
            }
        }
        None
    }
}

impl<K, V> DoubleEndedIterator for LiveIterMut<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while self.front < self.back {
            self.back -= 1;
            if let Some(item) = self.entry_at(self.back) {
                return Some(item);
            }
        }
        None
    }
}

pub struct LiveValuesMut<'a, K, V> {
    inner: LiveIterMut<'a, K, V>,
}

impl<'a, K, V> Iterator for LiveValuesMut<'a, K, V> {
    type Item = &'a mut V;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }
}

impl<K, V> DoubleEndedIterator for LiveValuesMut<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(_, v)| v)
    }
}

/// Residual `mem::replace` of an entry value (`ll_dict_setitem` overwrite).
#[majit_macros::dont_look_inside]
pub(crate) fn replace_value<V>(slot: &mut V, value: V) -> V {
    std::mem::replace(slot, value)
}

/// A borrowed key that can be compared against a `K` without building one.
///
/// The same shape as `indexmap::Equivalent`, so a lookup type written for the
/// `IndexMap` storage carries over unchanged.
pub trait Equivalent<K: ?Sized> {
    fn equivalent(&self, key: &K) -> bool;
}

impl<Q: ?Sized + Eq, K: ?Sized + Borrow<Q>> Equivalent<K> for Q {
    #[inline]
    fn equivalent(&self, key: &K) -> bool {
        *self == *key.borrow()
    }
}

/// See the module docs.
pub struct RDict<K, V, S = RandomState> {
    /// `d.indexes`.  A power of two, or empty before the first insert
    /// (`ll_dict_create_initial_index`, rordereddict.py).  Upstream picks
    /// a byte/short/int/long element width from the entry count; a single
    /// `u32` covers every dict that fits in memory here.
    indexes: Vec<u32>,
    /// `d.entries` (`DICTENTRYARRAY`). Null is `len(d.entries) == 0`.
    /// [`Self::num_ever_used_items`] is how far a slot walk reads; the array's
    /// `length` is the allocation.
    entries: *mut GcEntries<K, V>,
    /// `d.num_live_items`.
    num_live_items: usize,
    /// `d.num_ever_used_items`.
    num_ever_used_items: usize,
    /// `d.resize_counter`.  Signed because upstream tests `rc <= 0` after
    /// subtracting (rordereddict.py:684).
    resize_counter: isize,
    /// Bumped whenever the entries buffer is replaced or its contents move: a
    /// reindex, a compaction, a `clear`, or a growth that reallocates.
    ///
    /// Stands in for the first two clauses of `d.paranoia`, `entries !=
    /// d.entries or indexes != d.indexes` (rordereddict.py:1058-1060) — an
    /// identity test on two GC pointers that a `Vec` does not offer directly.
    /// !! A growth counts: `ll_dict_grow` hands `d.entries` a **new** array
    /// (`_overallocate_entries_len`, 745), so a probe holding a slot number
    /// across a comparison that merely *inserted* has to see the change.
    ///
    /// It is read by the *caller* — `scan_dict_key_reentrant` and setobject's
    /// three scans, which re-derive this table from the container object
    /// between steps and so can trust what they read.  See [`Self::lookup`]
    /// for why the check cannot live here.
    generation: u32,
    hash_builder: S,
}

// The entries array is a GC or immortal allocation, not a Rust owner, so the
// auto traits follow `K`, `V`, and `S` the way the old `Vec<Entry>` did.
unsafe impl<K: Send, V: Send, S: Send> Send for RDict<K, V, S> {}
unsafe impl<K: Sync, V: Sync, S: Sync> Sync for RDict<K, V, S> {}

impl<K, V, S: Default> RDict<K, V, S> {
    pub fn new() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K, V, S: Default> Default for RDict<K, V, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, S> RDict<K, V, S> {
    /// Sized for `capacity` entries without a reindex, which is what
    /// `_ll_dict_resize_to` would have picked once they were all in
    /// (rordereddict.py).  Sizing only the entry array would leave the
    /// index table to grow from `DICT_INITSIZE`, reindexing the whole run of a
    /// strategy switch several times over.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self
    where
        (K, V): GcEntriesType,
    {
        let mut d = Self::with_hasher(hash_builder);
        if capacity == 0 {
            return d;
        }
        // `_ll_malloc_entries` for the estimate `ll_newdict_size` allocates.
        d.entries = alloc_entries::<K, V>(capacity);
        let mut size = DICT_INITSIZE;
        while size <= (capacity + 1) * 2 {
            size *= 2;
        }
        d.indexes = vec![FREE; size];
        d.resize_counter = (size * 2) as isize;
        d
    }

    pub fn with_hasher(hash_builder: S) -> Self {
        Self {
            indexes: Vec::new(),
            entries: std::ptr::null_mut(),
            num_live_items: 0,
            num_ever_used_items: 0,
            resize_counter: 0,
            generation: 0,
            hash_builder,
        }
    }

    /// `d.num_live_items` — the number of pairs, not the number of slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.num_live_items
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_live_items == 0
    }

    /// `len(d.entries)`. Null is 0.
    #[inline]
    fn allocated_len(&self) -> usize {
        entries_allocated_len(self.entries)
    }

    #[inline]
    fn entry_ptr(&self) -> *mut Entry<K, V> {
        entries_item_ptr(self.entries)
    }

    /// `setinteriorfield` write barrier on the entries array. The guarded
    /// form: the array may be a pre-hook immortal allocation.
    #[inline]
    fn barrier_entries(&self) {
        if !self.entries.is_null() {
            crate::gc_hook::try_gc_write_barrier(self.entries as *mut u8);
        }
    }

    /// The `entries` field slot, one GcRef (`d.entries`).
    #[inline]
    pub fn entries_slot(&mut self) -> *mut *mut u8 {
        &raw mut self.entries as *mut *mut u8
    }

    /// `ll_valid_from_flag` (`entries[i].f_valid`).
    #[inline]
    fn entry_valid(&self, slot: usize) -> bool {
        unsafe { (*self.entry_ptr().add(slot)).f_valid }
    }

    /// `ll_getitem_nonneg` / `ll_getitem_fast` on `d.entries`.
    #[inline]
    fn entry_at(&self, slot: usize) -> &Entry<K, V> {
        unsafe { &*self.entry_ptr().add(slot) }
    }

    /// `ll_setitem_fast` on `d.entries`. The barrier runs before the
    /// reference is handed out; the caller stores immediately.
    #[inline]
    fn entry_at_mut(&mut self, slot: usize) -> &mut Entry<K, V> {
        self.barrier_entries();
        unsafe { &mut *self.entry_ptr().add(slot) }
    }

    /// `ll_mark_deleted_in_flag`, then `must_clear_key` / `must_clear_value`
    /// (`_ll_dict_del_entry`): `f_valid = False` and the key and value are
    /// reset to [`EntryDummy::dummy`].
    #[inline]
    fn mark_deleted(&mut self, slot: usize) -> (K, V)
    where
        K: EntryDummy + Copy,
        V: EntryDummy + Copy,
    {
        let e = self.entry_at_mut(slot);
        let key = e.key;
        let value = e.value;
        e.f_valid = false;
        e.key = K::dummy();
        e.value = V::dummy();
        e.f_hash = 0;
        (key, value)
    }

    /// Live prefix `d.entries[0:num_ever_used_items]`.
    #[inline]
    fn used_entries(&self) -> &[Entry<K, V>] {
        let n = self.num_ever_used_items;
        if n == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.entry_ptr(), n) }
        }
    }

    #[inline]
    fn used_entries_mut(&mut self) -> &mut [Entry<K, V>] {
        self.barrier_entries();
        self.used_entries_mut_for_trace()
    }

    /// The live prefix without the write barrier: the collector rewrites a
    /// moved reference in place while it traces, and `gc_trace` emits no
    /// barrier for that write.
    #[inline]
    fn used_entries_mut_for_trace(&mut self) -> &mut [Entry<K, V>] {
        let n = self.num_ever_used_items;
        if n == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.entry_ptr(), n) }
        }
    }

    /// `ll_getitem_nonneg` on `d.indexes`.
    #[inline]
    fn index_at(&self, i: usize) -> u32 {
        self.indexes[i]
    }

    /// `ll_setitem_fast` on `d.indexes`.
    #[inline]
    fn set_index_at(&mut self, i: usize, value: u32) {
        self.indexes[i] = value;
    }

    /// The first live slot at or after `from`, which is `_ll_dictnext`'s scan
    /// (rordereddict.py): "while i < num_ever_used_items: if
    /// entries.valid(i)".  A cursor stays valid across an unrelated delete
    /// because nothing renumbers; it goes stale only when [`Self::generation`]
    /// moves.
    #[inline]
    pub fn next_valid_slot(&self, from: usize) -> Option<usize> {
        (from..self.num_ever_used_items).find(|&i| self.entry_valid(i))
    }

    /// [`Self::next_valid_slot`] descending: the last live slot strictly below
    /// `before`, which is `ll_dictiter_reversed`'s walk.  A fresh reverse walk
    /// starts at `usize::MAX`.
    #[inline]
    pub fn prev_valid_slot(&self, before: usize) -> Option<usize> {
        (0..before.min(self.num_ever_used_items))
            .rev()
            .find(|&i| self.entry_valid(i))
    }

    /// `d.num_ever_used_items` — one past the highest slot ever filled, and so
    /// the bound of a slot walk.  Dead slots below it have `f_valid` false.
    #[inline]
    /// `_ll_dictnext` (rordereddict.py) — the entry at or after `from`,
    /// paired with the slot holding it.
    pub fn next_entry(&self, from: usize) -> Option<(usize, &K, &V)> {
        let slot = self.next_valid_slot(from)?;
        if !self.entry_valid(slot) {
            return None;
        }
        let e = self.entry_at(slot);
        Some((slot, &e.key, &e.value))
    }

    pub fn entry_slots(&self) -> usize {
        self.num_ever_used_items
    }

    /// Length of the probe table. A host `usize`, copied out before a
    /// callback so a later [`Self::from_preserved_slots`] does not read
    /// `self` again.
    pub fn index_len(&self) -> usize {
        self.indexes.len()
    }

    #[inline]
    pub fn is_valid_slot(&self, slot: usize) -> bool {
        slot < self.num_ever_used_items && self.entry_valid(slot)
    }

    /// Changes whenever a compaction or reindex moves entries; see the field.
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn capacity(&self) -> usize {
        self.allocated_len()
    }

    /// `ll_dict_clear`. The old array is left to the collector (`_ll_free_entries`
    /// is a no-op); the dict holds the empty (null) array.
    pub fn clear(&mut self) {
        if self.num_ever_used_items == 0 {
            return;
        }
        self.entries = std::ptr::null_mut();
        self.num_ever_used_items = 0;
        // "we can't remove the index here, because it is possible that crazy
        // Python code calls d.clear() from the method __eq__() called from
        // ll_dict_lookup(d).  Instead, stick to the rule that once a dictionary
        // has got an index, it will always have one."
        self.indexes = vec![FREE; DICT_INITSIZE];
        self.num_live_items = 0;
        self.resize_counter = (DICT_INITSIZE * 2) as isize;
        self.generation = self.generation.wrapping_add(1);
    }

    #[inline]
    pub fn get_slot(&self, slot: usize) -> Option<(&K, &V)> {
        if slot >= self.num_ever_used_items {
            return None;
        }
        if !self.entry_valid(slot) {
            return None;
        }
        let e = self.entry_at(slot);
        Some((&e.key, &e.value))
    }

    #[inline]
    pub fn get_slot_mut(&mut self, slot: usize) -> Option<(&K, &mut V)> {
        if slot >= self.num_ever_used_items {
            return None;
        }
        if !self.entry_valid(slot) {
            return None;
        }
        let e = self.entry_at_mut(slot);
        Some((&e.key, &mut e.value))
    }

    pub fn iter(&self) -> LiveIter<'_, K, V> {
        LiveIter::new(self.used_entries())
    }

    /// Pairs with their slot numbers, for a caller that must name an entry
    /// again after the walk.
    pub fn iter_slots(&self) -> LiveSlotIter<'_, K, V> {
        LiveSlotIter {
            inner: LiveIter::new(self.used_entries()),
        }
    }

    pub fn iter_mut(&mut self) -> LiveIterMut<'_, K, V> {
        LiveIterMut::new(self.used_entries_mut())
    }

    pub fn keys(&self) -> LiveKeys<'_, K, V> {
        LiveKeys {
            inner: LiveIter::new(self.used_entries()),
        }
    }

    pub fn values(&self) -> LiveValues<'_, K, V> {
        LiveValues {
            inner: LiveIter::new(self.used_entries()),
        }
    }

    pub fn values_mut(&mut self) -> LiveValuesMut<'_, K, V> {
        LiveValuesMut {
            inner: LiveIterMut::new(self.used_entries_mut()),
        }
    }

    /// [`iter_mut`](Self::iter_mut) for a GC walk that updates references in
    /// place during a collection. Takes no write barrier; a mutator store
    /// goes through `iter_mut`.
    pub fn iter_mut_for_trace(&mut self) -> LiveIterMut<'_, K, V> {
        LiveIterMut::new(self.used_entries_mut_for_trace())
    }

    /// [`values_mut`](Self::values_mut) for a GC walk; see
    /// [`iter_mut_for_trace`](Self::iter_mut_for_trace).
    pub fn values_mut_for_trace(&mut self) -> LiveValuesMut<'_, K, V> {
        LiveValuesMut {
            inner: LiveIterMut::new(self.used_entries_mut_for_trace()),
        }
    }

    /// The slot the next insert will fill, i.e. `d.num_ever_used_items`.
    #[inline]
    fn next_slot(&self) -> u32 {
        self.num_ever_used_items as u32
    }

    #[inline]
    fn probe_next(i: usize, perturb: u64, mask: usize) -> usize {
        // `i = (i << 2) + i + perturb + 1` on r_uint (rordereddict.py:1104).
        (i.wrapping_shl(2))
            .wrapping_add(i)
            .wrapping_add(perturb as usize)
            .wrapping_add(1)
            & mask
    }
}

impl<K, V, S> RDict<K, V, S>
where
    K: Hash + Eq + Copy + EntryDummy,
    V: Copy + EntryDummy,
    S: BuildHasher,
    (K, V): GcEntriesType,
{
    /// `d.keyhash` / `fnkeyhash` (rordereddict.py). The hasher is a
    /// residual: RandomState::build_hasher is not a translation subject.
    #[inline]
    #[majit_macros::dont_look_inside]
    fn hash_of<Q: Hash + ?Sized>(&self, key: &Q) -> u64 {
        let mut state = self.hash_builder.build_hasher();
        key.hash(&mut state);
        state.finish()
    }

    /// `ll_dict_lookup(d, key, hash, FLAG_LOOKUP)` (rordereddict.py).
    ///
    /// Returns the slot holding `key`.
    ///
    /// `@jit.oopspec('ordereddict.lookup')` (`ll_dict_lookup`,
    /// `ll_call_lookup_function`). The spec is the bare name: Rust's
    /// parameters are `(self, hash, key)`, and `global_marker_str`
    /// harvests only the spec string.
    ///
    /// # A comparison that mutates this dict
    ///
    /// `ll_dict_lookup` carries a `d.paranoia` branch (1093-1098) that restarts
    /// the probe when the comparison "did major nasty stuff to the dict", and
    /// this deliberately does not reproduce it, because it cannot: `&self`
    /// promises the compiler that nothing writes through it for the borrow's
    /// life, so a re-read meant to notice the write is free to be folded away.
    /// A restart written here reads as a guarantee and is not one — measured
    /// working at `opt-level=0` and losing a present key at `opt-level=1`.
    ///
    /// The guarantee lives one level up instead, where it can: a probe that
    /// might run user code is wrapped in `callback_free_dict_op!`, which asks
    /// afterwards whether a callback ran and **discards the answer** if one
    /// did, redoing the operation through
    /// `dictmultiobject::scan_dict_key_reentrant` — a walk that re-derives its
    /// pointers from the dict object at every step, so no stale borrow exists
    /// to fold.  What this owes that path is only that a reshape mid-probe
    /// cannot panic, hence the checked reads below; the value they produce is
    /// thrown away.
    #[majit_macros::oopspec("ordereddict.lookup")]
    fn lookup<Q>(&self, hash: u64, key: &Q) -> Option<usize>
    where
        Q: Equivalent<K> + ?Sized,
    {
        if self.indexes.is_empty() {
            return None;
        }
        let mask = self.indexes.len() - 1;
        let mut i = (hash as usize) & mask;
        let mut perturb = hash;
        loop {
            if i >= self.indexes.len() {
                return None;
            }
            let index = self.index_at(i);
            if index == FREE {
                return None;
            }
            if index >= VALID_OFFSET {
                let slot = (index - VALID_OFFSET) as usize;
                if slot < self.num_ever_used_items && self.entry_valid(slot) {
                    let e = self.entry_at(slot);
                    if e.f_hash == hash && key.equivalent(&e.key) {
                        return Some(slot);
                    }
                }
            }
            i = Self::probe_next(i, perturb, mask);
            perturb >>= PERTURB_SHIFT;
        }
    }

    /// `ll_dict_lookup(d, key, hash, FLAG_STORE)`.
    ///
    /// `Ok` is the slot already holding the key, `Err` the index slot a new
    /// entry should claim — the first [`DELETED`] one seen, else the [`FREE`]
    /// one that ended the probe (rordereddict.py).
    ///
    /// Carries [`Self::lookup`]'s note on a comparison that mutates the dict.
    fn lookup_for_store<Q>(&self, hash: u64, key: &Q) -> Result<usize, usize>
    where
        Q: Equivalent<K> + ?Sized,
    {
        debug_assert!(!self.indexes.is_empty());
        let mask = self.indexes.len() - 1;
        let mut i = (hash as usize) & mask;
        let mut perturb = hash;
        let mut deleted_slot: Option<usize> = None;
        loop {
            if i >= self.indexes.len() {
                return Err(deleted_slot.unwrap_or(0));
            }
            let index = self.index_at(i);
            if index == FREE {
                return Err(deleted_slot.unwrap_or(i));
            }
            if index == DELETED {
                if deleted_slot.is_none() {
                    deleted_slot = Some(i);
                }
            } else {
                let slot = (index - VALID_OFFSET) as usize;
                if slot < self.num_ever_used_items && self.entry_valid(slot) {
                    let e = self.entry_at(slot);
                    if e.f_hash == hash && key.equivalent(&e.key) {
                        return Ok(slot);
                    }
                }
            }
            i = Self::probe_next(i, perturb, mask);
            perturb >>= PERTURB_SHIFT;
        }
    }

    /// `rordereddict.py::ll_dict_store_clean` — probe for a [`FREE`]
    /// slot only, valid when no key can already be present.
    fn insert_clean(&mut self, hash: u64, slot: u32) {
        let mask = self.indexes.len() - 1;
        let mut i = (hash as usize) & mask;
        let mut perturb = hash;
        while self.index_at(i) != FREE {
            i = Self::probe_next(i, perturb, mask);
            perturb >>= PERTURB_SHIFT;
        }
        self.set_index_at(i, slot + VALID_OFFSET);
    }

    /// `ll_dict_reindex` (rordereddict.py).
    fn reindex(&mut self, new_size: usize) {
        debug_assert!(new_size.is_power_of_two());
        self.indexes = vec![FREE; new_size];
        self.resize_counter = (new_size * 2) as isize - (self.num_live_items * 3) as isize;
        for slot in 0..self.num_ever_used_items {
            if !self.entry_valid(slot) {
                continue;
            }
            let hash = self.entry_at(slot).f_hash;
            self.insert_clean(hash, slot as u32);
        }
        self.generation = self.generation.wrapping_add(1);
    }

    /// `rpython/rtyper/lltypesystem/rordereddict.py::ll_dict_remove_deleted_items`
    /// — drop the
    /// tombstones, renumbering the survivors, then reindex at the same size.
    /// Upstream makes compaction opaque to tracing (not to translation).
    /// `ll_dict_remove_deleted_items`. Below 25% live, allocate a new array;
    /// otherwise reuse this one, write the live items down, and zero the tail
    /// (`must_clear_key` / `must_clear_value`).
    #[majit_macros::dont_look_inside]
    fn remove_deleted_items(&mut self)
    where
        K: Copy + EntryDummy,
        V: Copy + EntryDummy,
        (K, V): GcEntriesType,
    {
        let old_len = self.allocated_len();
        let shrink = self.num_live_items < old_len / 4;
        let newitems = if shrink {
            alloc_entries::<K, V>(overallocate_entries_len(self.num_live_items))
        } else {
            // One barrier for the in-place writes (`llop.gc_writebarrier`).
            self.barrier_entries();
            self.entries
        };
        let src_base = self.entry_ptr();
        let dst_base = entries_item_ptr(newitems);
        let isrclimit = self.num_ever_used_items;
        let mut idst = 0usize;
        for isrc in 0..isrclimit {
            let src = unsafe { *src_base.add(isrc) };
            if !src.f_valid {
                continue;
            }
            if !newitems.is_null() {
                crate::gc_hook::try_gc_write_barrier(newitems as *mut u8);
            }
            unsafe {
                std::ptr::write(dst_base.add(idst), src);
            }
            idst += 1;
        }
        debug_assert_eq!(self.num_live_items, idst);
        if newitems == self.entries {
            let dead = Entry {
                key: K::dummy(),
                f_valid: false,
                value: V::dummy(),
                f_hash: 0,
            };
            while idst < isrclimit {
                if !newitems.is_null() {
                    crate::gc_hook::try_gc_write_barrier(newitems as *mut u8);
                }
                unsafe {
                    std::ptr::write(dst_base.add(idst), dead);
                }
                idst += 1;
            }
        } else {
            crate::gc_hook::try_gc_write_barrier_managed(newitems as *mut u8);
            self.entries = newitems;
        }
        self.num_ever_used_items = self.num_live_items;
        let size = self.indexes.len();
        self.reindex(size);
    }

    /// `ll_dict_resize` into `_ll_dict_resize_to`.
    ///
    /// `num_extra` is what makes the table *quadruple* rather than double:
    /// `(num_live_items + num_live_items + 1) * 2` until the dict is large
    /// enough for the 30000 cap to bite.  Both call sites reach the table
    /// through `ll_dict_resize`, and neither passes a `num_extra` of its own.
    fn resize(&mut self) {
        let num_extra = (self.num_live_items + 1).min(30000);
        let new_estimate = (self.num_live_items + num_extra) * 2;
        let mut new_size = DICT_INITSIZE;
        while new_size <= new_estimate {
            new_size *= 2;
        }
        if new_size < self.indexes.len() {
            self.remove_deleted_items();
        } else {
            self.reindex(new_size);
        }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_of(key);
        let slot = self.lookup(hash, key)?;
        // A callback that reshaped the dict mid-probe leaves `slot` stale;
        // `callback_free_dict_op!` discards this answer, so the read only
        // has to stay in bounds (see `lookup`).
        if slot >= self.num_ever_used_items || !self.entry_valid(slot) {
            return None;
        }
        Some(&self.entry_at(slot).value)
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_of(key);
        let slot = self.lookup(hash, key)?;
        if slot >= self.num_ever_used_items || !self.entry_valid(slot) {
            return None;
        }
        Some(&mut self.entry_at_mut(slot).value)
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_of(key);
        self.lookup(hash, key).is_some()
    }

    /// The slot holding `key`; see the module docs on slots versus positions.
    pub fn index_of<Q>(&self, key: &Q) -> Option<usize>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_of(key);
        self.lookup(hash, key)
    }

    /// `ll_dict_setitem_with_hash` (rordereddict.py) — probe, then hand the
    /// probe's answer to [`Self::setitem_lookup_done`].
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = self.hash_of(&key);
        if self.indexes.is_empty() {
            self.reindex(DICT_INITSIZE);
        }
        let index_slot = match self.lookup_for_store(hash, &key) {
            Ok(slot) => {
                assert!(self.entry_valid(slot), "valid slot");
                let e = self.entry_at_mut(slot);
                return Some(replace_value(&mut e.value, value));
            }
            Err(index_slot) => index_slot,
        };
        self.setitem_lookup_done(hash, Some(index_slot), key, value);
        None
    }

    /// Place a key the caller has already proven absent, running no key
    /// comparison of its own.
    ///
    /// The `i < 0` arm of `_ll_dict_setitem_lookup_done` entered without a
    /// preceding `FLAG_STORE` probe: with no index slot to reuse, placement
    /// goes through `ll_call_insert_clean_function` (rordereddict.py),
    /// which probes on the digest alone.
    ///
    /// A caller that is wrong about absence gets a second entry under the same
    /// key, and lookups then answer with whichever the probe reaches first.
    /// This is for a probe whose comparisons ran somewhere they could be
    /// undone — a set membership scan that must not hand the table a
    /// comparison able to re-enter it.
    pub fn insert_known_absent(&mut self, key: K, value: V) {
        let hash = self.hash_of(&key);
        if self.indexes.is_empty() {
            self.reindex(DICT_INITSIZE);
        }
        self.setitem_lookup_done(hash, None, key, value);
    }

    /// The `i < 0` arm of `_ll_dict_setitem_lookup_done` (rordereddict.py):
    /// grow or compact, then claim an index slot for a fresh entry.
    /// `index_slot` is the one a `FLAG_STORE` probe ended on, `None` when the
    /// caller never probed.
    fn setitem_lookup_done(&mut self, hash: u64, index_slot: Option<usize>, key: K, value: V) {
        let mut reindexed = false;
        // `if len(d.entries) == d.num_ever_used_items: ll_dict_grow(d)`.
        if self.num_ever_used_items == self.allocated_len() {
            reindexed = self.dict_grow();
        }
        let mut rc = self.resize_counter - 3;
        if rc <= 0 {
            self.resize();
            reindexed = true;
            rc = self.resize_counter - 3;
        }
        // `_ll_dict_setitem_lookup_done`: after growth/resize, publish the
        // index and then initialize the preallocated entry, with no allocation
        // between them. Only a reindex invalidates the original probe's slot.
        match index_slot.filter(|_| !reindexed) {
            Some(index_slot) => self.set_index_at(index_slot, self.next_slot() + VALID_OFFSET),
            None => {
                let slot = self.next_slot();
                self.insert_clean(hash, slot);
            }
        }
        self.resize_counter = rc;
        let slot = self.num_ever_used_items;
        self.barrier_entries();
        unsafe {
            std::ptr::write(
                self.entry_ptr().add(slot),
                Entry {
                    key,
                    f_valid: true,
                    value,
                    f_hash: hash,
                },
            );
        }
        self.num_ever_used_items += 1;
        self.num_live_items += 1;
    }

    /// `ll_dict_grow`. Returns whether the index was rebuilt (compaction).
    /// A larger array is `rgc.ll_arraycopy` of `num_ever_used_items` items;
    /// the old array is left to the collector.
    fn dict_grow(&mut self) -> bool
    where
        K: Copy + EntryDummy,
        V: Copy + EntryDummy,
        (K, V): GcEntriesType,
    {
        if self.num_live_items < self.num_ever_used_items / 2 {
            self.remove_deleted_items();
            return true;
        }
        let new_allocated = overallocate_entries_len(self.allocated_len());
        let newitems = alloc_entries::<K, V>(new_allocated);
        let n = self.num_ever_used_items;
        if n > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(self.entry_ptr(), entries_item_ptr(newitems), n);
            }
        }
        crate::gc_hook::try_gc_write_barrier_managed(newitems as *mut u8);
        self.entries = newitems;
        self.generation = self.generation.wrapping_add(1);
        false
    }

    /// `rordereddict.py::ll_dict_delete_by_entry_index` — re-probe from
    /// the entry's own digest for the one index slot naming it.
    fn delete_by_entry_index(&mut self, hash: u64, slot: usize) {
        let mask = self.indexes.len() - 1;
        let target = slot as u32 + VALID_OFFSET;
        let mut i = (hash as usize) & mask;
        let mut perturb = hash;
        while self.index_at(i) != target {
            // `ll_dict_delete_by_entry_index` checks for FREE, not a probe
            // count: the perturb prefix may revisit slots before becoming a
            // full-period walk, so a valid search can exceed indexes.len().
            debug_assert_ne!(self.index_at(i), FREE, "no index slot names entry {slot}");
            i = Self::probe_next(i, perturb, mask);
            perturb >>= PERTURB_SHIFT;
        }
        self.set_index_at(i, DELETED);
    }

    /// `ll_dict_pop` (rordereddict.py).  Order-preserving and O(1): the
    /// name says `remove` and not `shift_remove` because nothing shifts.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: EntryDummy,
        V: EntryDummy,
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_of(key);
        let slot = self.lookup(hash, key)?;
        let value = self.take_slot(hash, slot).1;
        Some(value)
    }

    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: EntryDummy,
        V: EntryDummy,
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_of(key);
        let slot = self.lookup(hash, key)?;
        Some(self.take_slot(hash, slot))
    }

    /// Delete the pair at `slot`, which must be valid.
    pub fn remove_slot(&mut self, slot: usize) -> Option<(K, V)>
    where
        K: EntryDummy,
        V: EntryDummy,
    {
        if slot >= self.num_ever_used_items {
            return None;
        }
        if !self.entry_valid(slot) {
            return None;
        }
        let hash = self.entry_at(slot).f_hash;
        Some(self.take_slot(hash, slot))
    }

    fn take_slot(&mut self, hash: u64, slot: usize) -> (K, V)
    where
        K: EntryDummy,
        V: EntryDummy,
    {
        self.delete_by_entry_index(hash, slot);
        let (key, value) = self.mark_deleted(slot);
        self.num_live_items -= 1;

        if self.num_live_items == 0 {
            // `_ll_dict_del`: the dict is empty. Reset the ever-used count.
            // The array stays; `ll_dict_clear` is what drops it.
            self.num_ever_used_items = 0;
        } else if slot + 1 == self.num_ever_used_items {
            self.num_ever_used_items -= 1;
            while self.num_ever_used_items > 0 && !self.entry_valid(self.num_ever_used_items - 1) {
                self.num_ever_used_items -= 1;
            }
        }
        if self.num_live_items + DICT_INITSIZE <= self.allocated_len() / 8 {
            self.resize();
        }
        (key, value)
    }

    /// `ll_dict_popitem` (rordereddict.py) — the last live pair.
    pub fn pop(&mut self) -> Option<(K, V)>
    where
        K: EntryDummy,
        V: EntryDummy,
    {
        let mut slot = self.num_ever_used_items;
        loop {
            if slot == 0 {
                return None;
            }
            slot -= 1;
            if self.entry_valid(slot) {
                return self.remove_slot(slot);
            }
        }
    }

    /// `move_to_end` (pypy/objspace/std/dictmultiobject.py) for a slot the
    /// caller already located.  Answers whether anything moved: a key already
    /// at the wanted end is a no-op, and the caller must not bump its
    /// iterator-invalidation state for one.
    ///
    /// To the back: delete and re-insert, which appends — upstream's
    /// `internal_delitem` + `setitem` pair.  To the front there is no cheap
    /// move; upstream calls its own path "a *very slow* fall-back" and rebuilds
    /// the dict, so this does too.
    pub fn move_slot_to_end(&mut self, slot: usize, last: bool) -> bool
    where
        K: EntryDummy,
        V: EntryDummy,
    {
        if slot >= self.num_ever_used_items {
            return false;
        }
        if !self.entry_valid(slot) {
            return false;
        }
        let hash = self.entry_at(slot).f_hash;
        if last {
            if slot + 1 == self.num_ever_used_items {
                return false;
            }
            let (k, v) = self.take_slot(hash, slot);
            self.insert(k, v);
        } else {
            if self.next_valid_slot(0) == Some(slot) {
                return false;
            }
            let (k, v) = self.take_slot(hash, slot);
            let mut rest = Vec::with_capacity(self.num_live_items);
            for i in 0..self.num_ever_used_items {
                if self.entry_valid(i) {
                    let e = self.entry_at(i);
                    rest.push((e.key, e.value));
                }
            }
            // The old array is left to the collector, as `ll_dict_clear` does.
            self.entries = std::ptr::null_mut();
            self.num_ever_used_items = 0;
            self.indexes.clear();
            self.num_live_items = 0;
            self.resize_counter = 0;
            self.generation = self.generation.wrapping_add(1);
            self.insert(k, v);
            for (k, v) in rest {
                self.insert(k, v);
            }
        }
        true
    }

    /// [`Self::move_slot_to_end`] by key; answers `None` when the key is absent.
    pub fn move_to_end<Q>(&mut self, key: &Q, last: bool) -> Option<bool>
    where
        K: EntryDummy,
        V: EntryDummy,
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_of(key);
        let slot = self.lookup(hash, key)?;
        Some(self.move_slot_to_end(slot, last))
    }

    /// Copy this table's entry-slot layout under a new key type.
    ///
    /// Reads this table at the call. `W_BaseSetObject.switch_to_object_strategy`
    /// cannot use that: `getdict_w` runs `hash_w` first, and the callback can
    /// detach this storage. The switch snapshots [`Self::index_len`] and the
    /// live flags, then builds with [`Self::from_preserved_slots`].
    /// `W_SetIterObject.slot` is an index into this array, and a tombstone
    /// must keep its index so slot `N` still names the same element after the
    /// switch (`_ll_dict_del_entry` does not renumber). `keys.len()` is
    /// `num_ever_used_items`; `keys[i]` is the image of a live slot `i` and is
    /// ignored for a tombstone. The index is `ll_dict_reindex` on the new
    /// keys' hashes, not a copy of the old probe table.
    pub fn map_keys_preserving_layout<K2, S2>(&self, keys: &[K2]) -> RDict<K2, V, S2>
    where
        K2: Hash + Eq + Copy + EntryDummy,
        S2: BuildHasher + Default,
        (K2, V): GcEntriesType,
    {
        let n = self.num_ever_used_items;
        debug_assert_eq!(keys.len(), n);
        if n == 0 {
            return RDict::with_hasher(S2::default());
        }
        let mut dst = RDict::with_hasher(S2::default());
        dst.entries = alloc_entries::<K2, V>(n);
        let mut live = 0usize;
        for slot in 0..n {
            if !self.entry_valid(slot) {
                continue;
            }
            let value = self.entry_at(slot).value;
            let key = keys[slot];
            let hash = dst.hash_of(&key);
            dst.barrier_entries();
            unsafe {
                std::ptr::write(
                    dst.entry_ptr().add(slot),
                    Entry {
                        key,
                        f_valid: true,
                        value,
                        f_hash: hash,
                    },
                );
            }
            live += 1;
        }
        debug_assert_eq!(live, self.num_live_items);
        dst.num_live_items = live;
        dst.num_ever_used_items = n;
        let index_size = if self.indexes.len().is_power_of_two() && !self.indexes.is_empty() {
            self.indexes.len()
        } else {
            DICT_INITSIZE
        };
        dst.reindex(index_size);
        crate::gc_hook::try_gc_write_barrier_managed(dst.entries as *mut u8);
        dst
    }

    /// [`Self::map_keys_preserving_layout`] from a slot image copied earlier.
    ///
    /// `live[i]` is `entries[i].f_valid` at the snapshot. Tombstone slots stay
    /// zero-filled, so their indexes do not move. `index_len` is
    /// [`Self::index_len`] from that same moment. Values are [`EntryDummy`];
    /// set storage stores `()`.
    pub fn from_preserved_slots(keys: &[K], live: &[bool], index_len: usize) -> Self
    where
        K: Hash + Eq + Copy + EntryDummy,
        V: Copy + EntryDummy,
        S: BuildHasher + Default,
        (K, V): GcEntriesType,
    {
        let n = keys.len();
        debug_assert_eq!(live.len(), n);
        if n == 0 {
            return Self::with_hasher(S::default());
        }
        let mut dst = Self::with_hasher(S::default());
        dst.entries = alloc_entries::<K, V>(n);
        let mut nlive = 0usize;
        for slot in 0..n {
            if !live[slot] {
                continue;
            }
            let key = keys[slot];
            let hash = dst.hash_of(&key);
            dst.barrier_entries();
            unsafe {
                std::ptr::write(
                    dst.entry_ptr().add(slot),
                    Entry {
                        key,
                        f_valid: true,
                        value: V::dummy(),
                        f_hash: hash,
                    },
                );
            }
            nlive += 1;
        }
        dst.num_live_items = nlive;
        dst.num_ever_used_items = n;
        let index_size = if index_len.is_power_of_two() && index_len != 0 {
            index_len
        } else {
            DICT_INITSIZE
        };
        dst.reindex(index_size);
        crate::gc_hook::try_gc_write_barrier_managed(dst.entries as *mut u8);
        dst
    }

    pub fn reserve(&mut self, additional: usize)
    where
        K: Copy,
        (K, V): GcEntriesType,
    {
        let entries_capacity = self.allocated_len();
        let need = self.num_ever_used_items.saturating_add(additional);
        if need > entries_capacity {
            let new_n = need.max(overallocate_entries_len(entries_capacity));
            let newitems = alloc_entries::<K, V>(new_n);
            let n = self.num_ever_used_items;
            if n > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(self.entry_ptr(), entries_item_ptr(newitems), n);
                }
            }
            crate::gc_hook::try_gc_write_barrier_managed(newitems as *mut u8);
            self.entries = newitems;
            self.generation = self.generation.wrapping_add(1);
        }
        let want = (self.num_live_items + additional) * 2;
        let mut new_size = DICT_INITSIZE;
        while new_size <= want {
            new_size *= 2;
        }
        if new_size > self.indexes.len() {
            self.reindex(new_size);
        }
    }
}

/// `_overallocate_entries_len` (rordereddict.py) — "the growth pattern is:
/// 0, 8, 17, 27, 38, 50, 64, 80, 98, ...".
fn overallocate_entries_len(baselen: usize) -> usize {
    baselen + (baselen >> 3) + 8
}

impl<K, V, S> FromIterator<(K, V)> for RDict<K, V, S>
where
    K: Hash + Eq + Copy + EntryDummy,
    V: Copy + EntryDummy,
    S: BuildHasher + Default,
    (K, V): GcEntriesType,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut d = Self::with_hasher(S::default());
        for (k, v) in iter {
            d.insert(k, v);
        }
        d
    }
}

impl<K, V, S> Extend<(K, V)> for RDict<K, V, S>
where
    K: Hash + Eq + Copy + EntryDummy,
    V: Copy + EntryDummy,
    S: BuildHasher,
    (K, V): GcEntriesType,
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

/// `ll_dict_copy`: allocate `len(d.entries)` items, copy the used prefix, barrier.
impl<K, V, S> Clone for RDict<K, V, S>
where
    K: Copy,
    V: Copy,
    S: Clone,
    (K, V): GcEntriesType,
{
    fn clone(&self) -> Self {
        let n = self.allocated_len();
        let entries = if n == 0 {
            std::ptr::null_mut()
        } else {
            let entries = alloc_entries::<K, V>(n);
            let used = self.num_ever_used_items;
            if used > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.entry_ptr(),
                        entries_item_ptr(entries),
                        used,
                    );
                }
            }
            crate::gc_hook::try_gc_write_barrier_managed(entries as *mut u8);
            entries
        };
        Self {
            indexes: self.indexes.clone(),
            entries,
            num_live_items: self.num_live_items,
            num_ever_used_items: self.num_ever_used_items,
            resize_counter: self.resize_counter,
            generation: self.generation,
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<K, V, S> Drop for RDict<K, V, S> {
    fn drop(&mut self) {
        // `_ll_free_entries` does not free the array. It is GC-owned, or an
        // immortal `malloc_typed` fallback. Only `indexes` is a Rust `Vec`.
        self.entries = std::ptr::null_mut();
    }
}

impl<K: std::fmt::Debug, V: std::fmt::Debug, S> std::fmt::Debug for RDict<K, V, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<'a, K, V, S> IntoIterator for &'a RDict<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = LiveIter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Owned walk of the live items. `K` and `V` are `Copy`; the entries array
/// is not freed (`_ll_free_entries`).
pub struct OwnedLiveIter<K, V> {
    ptr: *const Entry<K, V>,
    index: usize,
    end: usize,
    _mark: std::marker::PhantomData<(K, V)>,
}

unsafe impl<K: Send, V: Send> Send for OwnedLiveIter<K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for OwnedLiveIter<K, V> {}

impl<K: Copy, V: Copy> Iterator for OwnedLiveIter<K, V> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.end {
            let i = self.index;
            self.index += 1;
            let entry = unsafe { &*self.ptr.add(i) };
            if entry.f_valid {
                return Some((entry.key, entry.value));
            }
        }
        None
    }
}

impl<K, V, S> IntoIterator for RDict<K, V, S>
where
    K: Copy,
    V: Copy,
{
    type Item = (K, V);
    type IntoIter = OwnedLiveIter<K, V>;
    fn into_iter(self) -> Self::IntoIter {
        let iter = OwnedLiveIter {
            ptr: self.entry_ptr(),
            index: 0,
            end: self.num_ever_used_items,
            _mark: std::marker::PhantomData,
        };
        // Drop nulls `entries` and drops `indexes`; it does not free the array.
        iter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perturb_probes_can_revisit_slots_before_covering_the_table() {
        #[derive(Default)]
        struct CollisionHasher;
        impl Hasher for CollisionHasher {
            fn finish(&self) -> u64 {
                0x3a18_dd5a_e3c4_5eb9
            }
            fn write(&mut self, _: &[u8]) {}
        }
        let mut d: RDict<u64, u64, std::hash::BuildHasherDefault<CollisionHasher>> = RDict::new();
        for key in 0..10 {
            d.insert(key, key);
        }
        assert_eq!(d.indexes.len(), 16);
        check_invariants(&d);
        // The tenth distinct slot is reached after 20 probes, despite the
        // index table having only 16 slots. The perturb prefix revisits slots.
        assert_eq!(d.remove(&9), Some(9));
        d.reindex(16);
        check_invariants(&d);
        for key in 0..9 {
            assert_eq!(d.remove(&key), Some(key));
        }
        check_invariants(&d);
    }

    // a reference model: insertion-ordered Vec + linear lookup
    #[derive(Default)]
    struct Model {
        items: Vec<(u64, u64)>,
    }
    impl Model {
        fn insert(&mut self, k: u64, v: u64) -> Option<u64> {
            for e in self.items.iter_mut() {
                if e.0 == k {
                    return Some(std::mem::replace(&mut e.1, v));
                }
            }
            self.items.push((k, v));
            None
        }
        fn remove(&mut self, k: u64) -> Option<u64> {
            let i = self.items.iter().position(|e| e.0 == k)?;
            Some(self.items.remove(i).1)
        }
        fn pop(&mut self) -> Option<(u64, u64)> {
            self.items.pop()
        }
        fn move_to_end(&mut self, k: u64, last: bool) -> bool {
            let Some(i) = self.items.iter().position(|e| e.0 == k) else {
                return false;
            };
            let e = self.items.remove(i);
            if last {
                self.items.push(e);
            } else {
                self.items.insert(0, e);
            }
            true
        }
    }

    fn check_invariants<S: BuildHasher>(d: &RDict<u64, u64, S>) {
        // every live slot is found by its own key, and at the slot it lives in
        let mut live = 0;
        for slot in 0..d.entry_slots() {
            if let Some((k, _)) = d.get_slot(slot) {
                live += 1;
                assert_eq!(d.index_of(k), Some(slot), "slot {slot} not reachable");
            }
        }
        assert_eq!(live, d.len(), "num_live_items disagrees with the entries");
        // the index table names each live slot exactly once, and no dead one
        if !d.indexes.is_empty() {
            let mut named = vec![0usize; d.entry_slots()];
            for &ix in d.indexes.iter() {
                if ix >= VALID_OFFSET {
                    let slot = (ix - VALID_OFFSET) as usize;
                    assert!(
                        slot < d.entry_slots(),
                        "index names slot {slot} past the end"
                    );
                    assert!(d.is_valid_slot(slot), "index names dead slot {slot}");
                    named[slot] += 1;
                }
            }
            for slot in 0..d.entry_slots() {
                if d.is_valid_slot(slot) {
                    assert_eq!(named[slot], 1, "slot {slot} named {} times", named[slot]);
                }
            }
            assert!(d.indexes.len().is_power_of_two());
            assert!(
                d.indexes.iter().any(|&ix| ix == FREE),
                "index table has no FREE slot left"
            );
        }
        // the trailing slot is never a tombstone
        if d.entry_slots() > 0 {
            assert!(d.is_valid_slot(d.entry_slots() - 1), "trailing tombstone");
        }
    }

    fn same(d: &RDict<u64, u64>, m: &Model) {
        assert_eq!(d.len(), m.items.len(), "len");
        let got: Vec<(u64, u64)> = d.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(got, m.items, "order/contents");
        for (k, v) in m.items.iter() {
            assert_eq!(d.get(k), Some(v), "get({k})");
        }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn insert_get_remove_keeps_insertion_order() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..100 {
            assert_eq!(d.insert(i, i * 10), None);
        }
        assert_eq!(d.len(), 100);
        for i in 0..100 {
            assert_eq!(d.get(&i), Some(&(i * 10)));
        }
        // delete every other key: the survivors keep their order
        for i in (0..100).step_by(2) {
            assert_eq!(d.remove(&i), Some(i * 10));
        }
        check_invariants(&d);
        let got: Vec<u64> = d.keys().copied().collect();
        assert_eq!(got, (0..100).filter(|i| i % 2 == 1).collect::<Vec<_>>());
    }

    #[test]
    fn insert_clean_after_more_probes_than_index_slots() {
        // Unlike the deletion regression, keep all ten entries live for
        // both clean insertion and reindexing. The tenth distinct slot for
        // this hash needs 18 probe steps in a 16-slot index table.
        #[derive(Default)]
        struct CollisionHasher;
        impl Hasher for CollisionHasher {
            fn write(&mut self, _: &[u8]) {}
            fn finish(&self) -> u64 {
                0x3c6e_f372_fe94_f82a
            }
        }
        let mut d: RDict<u64, u64, std::hash::BuildHasherDefault<CollisionHasher>> =
            RDict::default();
        for k in 0..10 {
            d.insert_known_absent(k, k);
        }
        assert_eq!(d.indexes.len(), 16);
        check_invariants(&d);
        // Reindexing uses the same clean-insert probe, without changing
        // the hash or the number of free index slots in this fixture.
        d.reindex(16);
        check_invariants(&d);
        assert_eq!(
            d.keys().copied().collect::<Vec<_>>(),
            (0..10).collect::<Vec<_>>()
        );
    }

    #[test]
    fn insert_after_reindex_still_names_the_new_slot() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..32 {
            d.insert(i, i);
        }
        for i in 0..24 {
            d.remove(&i);
        }
        for i in 100..116 {
            d.insert(i, i);
        }
        check_invariants(&d);
        assert_eq!(d.get(&115), Some(&115));
        assert_eq!(d.remove(&115), Some(115));
        check_invariants(&d);
    }

    #[test]
    fn reinsert_after_delete_appends_at_the_end() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..5 {
            d.insert(i, i);
        }
        d.remove(&1);
        d.insert(1, 99);
        let got: Vec<u64> = d.keys().copied().collect();
        assert_eq!(got, vec![0, 2, 3, 4, 1]);
        assert_eq!(d.get(&1), Some(&99));
        check_invariants(&d);
    }

    #[test]
    fn overwrite_keeps_the_original_position() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..5 {
            d.insert(i, i);
        }
        assert_eq!(d.insert(1, 99), Some(1));
        let got: Vec<u64> = d.keys().copied().collect();
        assert_eq!(got, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn draining_a_dict_frees_its_slots() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..20_000 {
            d.insert(i, i);
        }
        for i in 0..20_000 {
            assert_eq!(d.remove(&i), Some(i));
        }
        assert!(d.is_empty());
        assert_eq!(d.entry_slots(), 0);
        check_invariants(&d);
        // and the dict still works afterwards
        d.insert(7, 7);
        assert_eq!(d.get(&7), Some(&7));
        check_invariants(&d);
    }

    #[test]
    fn deleting_from_the_front_compacts_eventually() {
        // deleting the *oldest* key never trims the tail, so this is the case that
        // relies on the resize-time compaction
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..10_000 {
            d.insert(i, i);
        }
        for i in 0..9_000 {
            d.remove(&i);
        }
        assert_eq!(d.len(), 1_000);
        assert!(
            d.entry_slots() < 4_000,
            "tombstones never compacted: {} slots for {} pairs",
            d.entry_slots(),
            d.len()
        );
        check_invariants(&d);
    }

    #[test]
    fn pop_takes_the_last_live_pair() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..10 {
            d.insert(i, i);
        }
        d.remove(&9);
        d.remove(&8);
        assert_eq!(d.pop(), Some((7, 7)));
        assert_eq!(d.len(), 7);
        check_invariants(&d);
        while d.pop().is_some() {}
        assert!(d.is_empty());
    }

    #[test]
    fn move_to_end_both_ways() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..5 {
            d.insert(i, i);
        }
        assert_eq!(d.move_to_end(&1, true), Some(true));
        assert_eq!(d.keys().copied().collect::<Vec<_>>(), vec![0, 2, 3, 4, 1]);
        assert_eq!(d.move_to_end(&3, false), Some(true));
        assert_eq!(d.keys().copied().collect::<Vec<_>>(), vec![3, 0, 2, 4, 1]);
        assert_eq!(d.move_to_end(&99, true), None);
        assert_eq!(d.move_to_end(&1, true), Some(false), "already last");
        assert_eq!(d.move_to_end(&3, false), Some(false), "already first");
        check_invariants(&d);
    }

    #[test]
    fn remove_slot_and_index_of_agree() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..50 {
            d.insert(i, i);
        }
        let slot = d.index_of(&30).unwrap();
        assert_eq!(d.get_slot(slot).map(|(k, _)| *k), Some(30));
        assert_eq!(d.remove_slot(slot), Some((30, 30)));
        assert_eq!(d.remove_slot(slot), None);
        assert_eq!(d.index_of(&30), None);
        check_invariants(&d);
    }

    #[test]
    fn differential_against_the_model() {
        for seed in 1..=8u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let mut d: RDict<u64, u64> = RDict::new();
            let mut m = Model::default();
            for step in 0..8_000 {
                let k = rng.next() % 300;
                match rng.next() % 10 {
                    0..=4 => {
                        let v = rng.next();
                        assert_eq!(d.insert(k, v), m.insert(k, v), "insert at step {step}");
                    }
                    5..=7 => {
                        assert_eq!(d.remove(&k), m.remove(k), "remove at step {step}");
                    }
                    8 => {
                        assert_eq!(d.pop(), m.pop(), "pop at step {step}");
                    }
                    _ => {
                        let last = rng.next() % 2 == 0;
                        assert_eq!(
                            d.move_to_end(&k, last).is_some(),
                            m.move_to_end(k, last),
                            "move_to_end at step {step}"
                        );
                    }
                }
                if step % 97 == 0 {
                    same(&d, &m);
                    check_invariants(&d);
                }
            }
            same(&d, &m);
            check_invariants(&d);
        }
    }

    #[test]
    fn clear_then_reuse() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..100 {
            d.insert(i, i);
        }
        d.clear();
        assert!(d.is_empty());
        assert_eq!(d.entry_slots(), 0);
        assert_eq!(d.get(&1), None);
        for i in 0..100 {
            d.insert(i, i + 1);
        }
        assert_eq!(d.len(), 100);
        assert_eq!(d.get(&99), Some(&100));
        check_invariants(&d);
    }

    #[test]
    fn borrowed_lookup_key() {
        let mut d: RDict<u64, u64> = RDict::default();
        d.insert(1, 1);
        d.insert(2, 2);
        assert_eq!(d.get(&1), Some(&1));
        assert_eq!(d.remove(&2), Some(2));
        assert_eq!(d.get(&2), None);
    }

    #[test]
    fn iteration_helpers_skip_tombstones() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..10 {
            d.insert(i, i);
        }
        d.remove(&3);
        d.remove(&5);
        assert_eq!(d.iter().count(), 8);
        assert_eq!(d.values().copied().sum::<u64>(), 45 - 3 - 5);
        for v in d.values_mut() {
            *v += 1;
        }
        assert_eq!(d.get(&4), Some(&5));
        let slots: Vec<usize> = d.iter_slots().map(|(s, _, _)| s).collect();
        assert_eq!(slots, vec![0, 1, 2, 4, 6, 7, 8, 9]);
    }

    #[test]
    fn slot_cursors_walk_both_ways() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..10 {
            d.insert(i, i);
        }
        d.remove(&0);
        d.remove(&3);
        d.remove(&9);
        let mut fwd = Vec::new();
        let mut cur = 0;
        while let Some(s) = d.next_valid_slot(cur) {
            fwd.push(*d.get_slot(s).unwrap().0);
            cur = s + 1;
        }
        assert_eq!(fwd, vec![1, 2, 4, 5, 6, 7, 8]);
        let mut rev = Vec::new();
        let mut cur = usize::MAX;
        while let Some(s) = d.prev_valid_slot(cur) {
            rev.push(*d.get_slot(s).unwrap().0);
            cur = s;
        }
        let mut expect = fwd.clone();
        expect.reverse();
        assert_eq!(rev, expect);
        assert_eq!(fwd.len(), d.len());
    }

    #[test]
    fn into_iterator_impls_skip_tombstones() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..6 {
            d.insert(i, i * 2);
        }
        d.remove(&2);
        let borrowed: Vec<(u64, u64)> = (&d).into_iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(borrowed, vec![(0, 0), (1, 2), (3, 6), (4, 8), (5, 10)]);
        let owned: Vec<(u64, u64)> = d.into_iter().collect();
        assert_eq!(owned, borrowed);
    }

    #[test]
    fn iterators_are_double_ended() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..6 {
            d.insert(i, i);
        }
        d.remove(&2);
        assert_eq!(
            d.iter().rev().map(|(k, _)| *k).collect::<Vec<_>>(),
            vec![5, 4, 3, 1, 0]
        );
        assert_eq!(
            d.keys().rev().copied().collect::<Vec<_>>(),
            vec![5, 4, 3, 1, 0]
        );
        assert_eq!(
            d.values().rev().copied().collect::<Vec<_>>(),
            vec![5, 4, 3, 1, 0]
        );
        assert_eq!(
            d.iter_slots().rev().map(|(s, _, _)| s).collect::<Vec<_>>(),
            vec![5, 4, 3, 1, 0]
        );
        for v in d.values_mut().rev().take(1) {
            *v = 99;
        }
        assert_eq!(d.get(&5), Some(&99));
    }

    /// `ll_dict_resize` quadruples: `num_extra` is `num_live_items + 1`, not 1,
    /// so `new_estimate` is `(2n + 1) * 2` and each new index table is four
    /// times the last.  Passing 1 halves the estimate and turns the cadence
    /// into a doubling — same final size, but every intermediate table is
    /// rebuilt, which is a full rehash of every live key.
    #[test]
    fn resize_quadruples_the_index_table() {
        let mut d: RDict<u64, u64> = RDict::new();
        let mut sizes = vec![];
        for i in 0..20_000u64 {
            d.insert(i, i);
            if sizes.last() != Some(&d.indexes.len()) {
                sizes.push(d.indexes.len());
            }
        }
        assert!(sizes.len() >= 4, "too few resizes to judge: {sizes:?}");
        for pair in sizes.windows(2) {
            assert_eq!(pair[1], pair[0] * 4, "not a quadrupling: {sizes:?}");
        }
    }

    /// `ll_dict_clear` *replaces* the entry array, so
    /// the old one is released; and it keeps an index table, because "once a
    /// dictionary has got an index, it will always have one".
    #[test]
    fn clear_releases_the_entries_and_keeps_an_index() {
        let mut d: RDict<u64, u64> = RDict::new();
        for i in 0..1000u64 {
            d.insert(i, i);
        }
        assert!(d.capacity() >= 1000);
        d.clear();
        assert_eq!(d.capacity(), 0, "the entry buffer was kept");
        assert_eq!(d.indexes.len(), DICT_INITSIZE);
        assert_eq!(d.resize_counter, (DICT_INITSIZE * 2) as isize);
        assert_eq!(d.len(), 0);
        assert_eq!(d.entry_slots(), 0);
        d.insert(7, 7);
        assert_eq!(d.get(&7), Some(&7));
        assert_eq!(d.len(), 1);
    }

    /// `entries != d.entries` (rordereddict.py:1058) fires for a *growth*, not
    /// only for a compaction: `ll_dict_grow` hands `d.entries` a new array, and
    /// a probe holding a slot number across a comparison that merely inserted
    /// has to notice.  Missing this answered a mutating `__eq__` without the
    /// restart it is owed.
    #[test]
    fn a_growth_that_reallocates_bumps_the_generation() {
        let mut d: RDict<u64, u64> = RDict::new();
        let mut growths = 0;
        for i in 0..64u64 {
            let (capacity, generation) = (d.capacity(), d.generation);
            d.insert(i, i);
            if d.capacity() != capacity {
                growths += 1;
                assert_ne!(d.generation, generation, "grew at insert {i}");
            }
        }
        assert!(growths >= 2, "the entries buffer never grew");
    }

    #[test]
    fn with_capacity_sizes_the_index_table_too() {
        let mut d: RDict<u64, u64, RandomState> =
            RDict::with_capacity_and_hasher(1000, RandomState::new());
        let start = d.generation();
        for i in 0..1000 {
            d.insert(i, i);
        }
        assert_eq!(d.len(), 1000);
        assert_eq!(
            d.generation(),
            start,
            "a sized dict reindexed while filling"
        );
        check_invariants(&d);
        // and the zero case still starts empty
        let e: RDict<u64, u64, RandomState> =
            RDict::with_capacity_and_hasher(0, RandomState::new());
        assert!(e.is_empty());
        assert_eq!(e.entry_slots(), 0);
    }

    // a comparison that re-enters and reshapes the dict
    //
    // The container promises only that this cannot panic and cannot leave the
    // table inconsistent; the *answer* is allowed to be wrong, because the
    // caller that can reach this (`callback_free_dict_op!`) throws it away and
    // redoes the operation through `scan_dict_key_reentrant`.  See
    // `RDict::lookup`.
    thread_local! {
        static REENTER: std::cell::Cell<*mut RDict<Nasty, u64>> =
            const { std::cell::Cell::new(std::ptr::null_mut()) };
        static FIRED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    #[derive(Clone, Copy, Debug)]
    struct Nasty(u64);

    impl EntryDummy for Nasty {
        fn dummy() -> Self {
            Nasty(0)
        }
    }

    impl GcEntriesType for (Nasty, u64) {
        fn entries_gc_type_id() -> u32 {
            0
        }
    }

    impl Hash for Nasty {
        fn hash<H: Hasher>(&self, state: &mut H) {
            // every key into one bucket run, so a probe has to walk
            state.write_u64(0);
        }
    }
    impl PartialEq for Nasty {
        fn eq(&self, other: &Self) -> bool {
            let d = REENTER.with(|c| c.replace(std::ptr::null_mut()));
            let mine = self.0;
            if !d.is_null() {
                FIRED.with(|c| c.set(c.get() + 1));
                unsafe {
                    for i in 100..160 {
                        (*d).insert(Nasty(i), i);
                    }
                    (*d).remove(&Nasty(101));
                }
                // `other` may point into the entry array the burst above
                // reallocated, so do not read it after
                return false;
            }
            mine == other.0
        }
    }
    impl Eq for Nasty {}

    #[test]
    fn a_reshaping_comparison_leaves_the_table_consistent() {
        let mut d: RDict<Nasty, u64> = RDict::new();
        for i in 0..40 {
            d.insert(Nasty(i), i);
        }
        let raw: *mut RDict<Nasty, u64> = &mut d;
        REENTER.with(|c| c.set(raw));
        let _ = d.get(&Nasty(39));
        assert_eq!(
            FIRED.with(|c| c.get()),
            1,
            "the reentrant comparison never ran"
        );
        // the answer above is not promised; the table is
        assert_eq!(d.len(), 99);
        assert_eq!(d.entry_slots(), 100);
        let mut named = vec![0usize; d.entry_slots()];
        for &ix in d.indexes.iter() {
            if ix >= VALID_OFFSET {
                let slot = (ix - VALID_OFFSET) as usize;
                assert!(d.is_valid_slot(slot), "index names dead slot {slot}");
                named[slot] += 1;
            }
        }
        for slot in 0..d.entry_slots() {
            if d.is_valid_slot(slot) {
                assert_eq!(named[slot], 1, "slot {slot} named {} times", named[slot]);
            }
        }
        // and every key is still reachable once the mutation is over
        for i in 0..40 {
            assert_eq!(d.get(&Nasty(i)).copied(), Some(i), "lost original key {i}");
        }
        for i in 102..160 {
            assert_eq!(d.get(&Nasty(i)).copied(), Some(i), "lost inserted key {i}");
        }
        assert_eq!(d.get(&Nasty(101)).copied(), None);
    }

    #[test]
    fn a_deleted_slot_is_invalid_and_cleared() {
        let mut d: RDict<u64, u64> = RDict::new();
        d.insert(1, 10);
        d.insert(2, 20);
        d.insert(3, 30);
        assert_eq!(d.remove(&2), Some(20));
        assert!(!d.entry_valid(1));
        assert_eq!(d.entry_at(1).key, 0);
        assert_eq!(d.entry_at(1).value, 0);
        let got: Vec<(u64, u64)> = d.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(got, vec![(1, 10), (3, 30)]);
    }

    #[test]
    fn one_hash_for_every_key_still_finds_them() {
        let mut d: RDict<Nasty, u64> = RDict::new();
        for i in 0..100 {
            d.insert(Nasty(i), i);
        }
        d.remove(&Nasty(50));
        for i in 0..100 {
            let want = if i == 50 { None } else { Some(i) };
            assert_eq!(d.get(&Nasty(i)).copied(), want, "colliding key {i}");
        }
    }
}
