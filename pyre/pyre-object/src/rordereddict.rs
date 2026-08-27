//! The ordered dict of `rpython/rtyper/lltypesystem/rordereddict.py`.
//!
//! Every RPython `dict` translates to this structure, and it is what a
//! `W_DictObject` strategy erases its storage to.  The shape is upstream's: a
//! power-of-two `indexes` table holding entry numbers biased by
//! [`VALID_OFFSET`], plus an insertion-ordered `entries` array whose dead slots
//! are tombstones rather than holes.
//!
//! The consequence that matters is [`RDict::remove`].  `ll_dict_delitem`
//! (rordereddict.py:855) writes [`DELETED`] into the one index slot that named
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
//! `entries.len()` is upstream's `num_ever_used_items` and [`RDict::len`] is
//! `num_live_items`; a tombstone makes them diverge.  Every index this type
//! hands out or accepts — [`RDict::index_of`], [`RDict::get_slot`],
//! [`RDict::remove_slot`] — is a **slot**, an index into `entries`, never the
//! n-th live pair.  Walk a dict with `0..d.entry_slots()` and skip the `None`s,
//! which is what `_ll_dictnext` (1373) does.

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};

/// An index slot naming no entry, and one whose entry has been deleted.
///
/// `FREE` ends a probe; `DELETED` does not, because the key being looked for
/// may have been stored past it (rordereddict.py:1029-1031).
pub const FREE: u32 = 0;
/// See [`FREE`].
pub const DELETED: u32 = 1;
/// The bias an entry number carries inside the index table, so that entry 0 is
/// distinguishable from [`FREE`] (rordereddict.py:1033).
pub const VALID_OFFSET: u32 = 2;

/// `DICT_INITSIZE` (rordereddict.py:1152).
const DICT_INITSIZE: usize = 16;
/// `PERTURB_SHIFT` (rordereddict.py:1021).
const PERTURB_SHIFT: u32 = 5;

/// One `d.entries` slot's payload — the key, its value, and the digest
/// `ENTRY.f_hash` caches, so a reindex and a probe both read the digest
/// instead of recomputing it.
///
/// Public only because it names the iterator types; nothing outside can read
/// or build one.
#[derive(Clone, Debug)]
pub struct Entry<K, V> {
    hash: u64,
    key: K,
    value: V,
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
    /// (`ll_dict_create_initial_index`, rordereddict.py:718).  Upstream picks
    /// a byte/short/int/long element width from the entry count; a single
    /// `u32` covers every dict that fits in memory here.
    indexes: Vec<u32>,
    /// `d.entries`.  `len()` is `num_ever_used_items`; a `None` is a slot
    /// `entries.valid(i)` answers false for.
    entries: Vec<Option<Entry<K, V>>>,
    /// `d.num_live_items`.
    num_live_items: usize,
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
    /// (rordereddict.py:735).  Sizing only the entry array would leave the
    /// index table to grow from `DICT_INITSIZE`, reindexing the whole run of a
    /// strategy switch several times over.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        let mut d = Self::with_hasher(hash_builder);
        if capacity == 0 {
            return d;
        }
        d.entries.reserve(capacity);
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
            entries: Vec::new(),
            num_live_items: 0,
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

    /// The first live slot at or after `from`, which is `_ll_dictnext`'s scan
    /// (rordereddict.py:1373): "while i < num_ever_used_items: if
    /// entries.valid(i)".  A cursor stays valid across an unrelated delete
    /// because nothing renumbers; it goes stale only when [`Self::generation`]
    /// moves.
    #[inline]
    pub fn next_valid_slot(&self, from: usize) -> Option<usize> {
        (from..self.entries.len()).find(|&i| self.entries[i].is_some())
    }

    /// [`Self::next_valid_slot`] descending: the last live slot strictly below
    /// `before`, which is `ll_dictiter_reversed`'s walk.  A fresh reverse walk
    /// starts at `usize::MAX`.
    #[inline]
    pub fn prev_valid_slot(&self, before: usize) -> Option<usize> {
        (0..before.min(self.entries.len()))
            .rev()
            .find(|&i| self.entries[i].is_some())
    }

    /// `d.num_ever_used_items` — one past the highest slot ever filled, and so
    /// the bound of a slot walk.  Dead slots below it read as `None`.
    #[inline]
    /// `_ll_dictnext` (rordereddict.py:1373) — the entry at or after `from`,
    /// paired with the slot holding it.
    pub fn next_entry(&self, from: usize) -> Option<(usize, &K, &V)> {
        let slot = self.next_valid_slot(from)?;
        let e = self.entries[slot].as_ref()?;
        Some((slot, &e.key, &e.value))
    }

    pub fn entry_slots(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_valid_slot(&self, slot: usize) -> bool {
        matches!(self.entries.get(slot), Some(Some(_)))
    }

    /// Changes whenever a compaction or reindex moves entries; see the field.
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    pub fn clear(&mut self) {
        self.indexes.clear();
        self.entries.clear();
        self.num_live_items = 0;
        self.resize_counter = 0;
        self.generation = self.generation.wrapping_add(1);
    }

    #[inline]
    pub fn get_slot(&self, slot: usize) -> Option<(&K, &V)> {
        match self.entries.get(slot) {
            Some(Some(e)) => Some((&e.key, &e.value)),
            _ => None,
        }
    }

    #[inline]
    pub fn get_slot_mut(&mut self, slot: usize) -> Option<(&K, &mut V)> {
        match self.entries.get_mut(slot) {
            Some(Some(e)) => Some((&e.key, &mut e.value)),
            _ => None,
        }
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&K, &V)> {
        self.entries
            .iter()
            .filter_map(|e| e.as_ref().map(|e| (&e.key, &e.value)))
    }

    /// Pairs with their slot numbers, for a caller that must name an entry
    /// again after the walk.
    pub fn iter_slots(&self) -> impl DoubleEndedIterator<Item = (usize, &K, &V)> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.as_ref().map(|e| (i, &e.key, &e.value)))
    }

    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = (&K, &mut V)> {
        self.entries
            .iter_mut()
            .filter_map(|e| e.as_mut().map(|e| (&e.key, &mut e.value)))
    }

    pub fn keys(&self) -> impl DoubleEndedIterator<Item = &K> {
        self.entries
            .iter()
            .filter_map(|e| e.as_ref().map(|e| &e.key))
    }

    pub fn values(&self) -> impl DoubleEndedIterator<Item = &V> {
        self.entries
            .iter()
            .filter_map(|e| e.as_ref().map(|e| &e.value))
    }

    pub fn values_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut V> {
        self.entries
            .iter_mut()
            .filter_map(|e| e.as_mut().map(|e| &mut e.value))
    }

    /// The slot the next insert will fill, i.e. `d.num_ever_used_items`.
    #[inline]
    fn next_slot(&self) -> u32 {
        self.entries.len() as u32
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

impl<K: Hash + Eq, V, S: BuildHasher> RDict<K, V, S> {
    #[inline]
    fn hash_of<Q: Hash + ?Sized>(&self, key: &Q) -> u64 {
        let mut state = self.hash_builder.build_hasher();
        key.hash(&mut state);
        state.finish()
    }

    /// `ll_dict_lookup(d, key, hash, FLAG_LOOKUP)` (rordereddict.py:1038).
    ///
    /// Returns the slot holding `key`.
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
            let index = *self.indexes.get(i)?;
            if index == FREE {
                return None;
            }
            if index >= VALID_OFFSET {
                let slot = (index - VALID_OFFSET) as usize;
                if let Some(Some(e)) = self.entries.get(slot) {
                    if e.hash == hash && key.equivalent(&e.key) {
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
    /// one that ended the probe (rordereddict.py:1110-1117).
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
            let Some(&index) = self.indexes.get(i) else {
                return Err(deleted_slot.unwrap_or(0));
            };
            if index == FREE {
                return Err(deleted_slot.unwrap_or(i));
            }
            if index == DELETED {
                if deleted_slot.is_none() {
                    deleted_slot = Some(i);
                }
            } else {
                let slot = (index - VALID_OFFSET) as usize;
                if let Some(Some(e)) = self.entries.get(slot) {
                    if e.hash == hash && key.equivalent(&e.key) {
                        return Ok(slot);
                    }
                }
            }
            i = Self::probe_next(i, perturb, mask);
            perturb >>= PERTURB_SHIFT;
        }
    }

    /// `ll_dict_store_clean` (rordereddict.py:1128) — probe for a [`FREE`]
    /// slot only, valid when no key can already be present.
    fn insert_clean(&mut self, hash: u64, slot: u32) {
        let mask = self.indexes.len() - 1;
        let mut i = (hash as usize) & mask;
        let mut perturb = hash;
        let mut probes = 0;
        while self.indexes[i] != FREE {
            i = Self::probe_next(i, perturb, mask);
            perturb >>= PERTURB_SHIFT;
            probes += 1;
            debug_assert!(probes <= self.indexes.len(), "no FREE slot to insert into");
        }
        self.indexes[i] = slot + VALID_OFFSET;
    }

    /// `ll_dict_reindex` (rordereddict.py:1000).
    fn reindex(&mut self, new_size: usize) {
        debug_assert!(new_size.is_power_of_two());
        self.indexes = vec![FREE; new_size];
        self.resize_counter = (new_size * 2) as isize - (self.num_live_items * 3) as isize;
        for slot in 0..self.entries.len() {
            let hash = match &self.entries[slot] {
                Some(e) => e.hash,
                None => continue,
            };
            self.insert_clean(hash, slot as u32);
        }
        self.generation = self.generation.wrapping_add(1);
    }

    /// `ll_dict_remove_deleted_items` (rordereddict.py:803) — drop the
    /// tombstones, renumbering the survivors, then reindex at the same size.
    fn remove_deleted_items(&mut self) {
        let shrink = self.num_live_items < self.entries.capacity() / 4;
        self.entries.retain(|e| e.is_some());
        debug_assert_eq!(self.entries.len(), self.num_live_items);
        if shrink {
            // "At least 75% of the allocated entries are dead, so shrink the
            // memory allocated as well as doing a compaction."
            self.entries
                .shrink_to(overallocate_entries_len(self.num_live_items));
        }
        let size = self.indexes.len();
        self.reindex(size);
    }

    /// `_ll_dict_resize_to(d, num_extra=1)` (rordereddict.py:735).
    fn resize(&mut self) {
        let new_estimate = (self.num_live_items + 1) * 2;
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
        self.entries[slot].as_ref().map(|e| &e.value)
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_of(key);
        let slot = self.lookup(hash, key)?;
        self.entries[slot].as_mut().map(|e| &mut e.value)
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

    /// `ll_dict_setitem_with_hash` (rordereddict.py:668) — probe, then hand the
    /// probe's answer to [`Self::setitem_lookup_done`].
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = self.hash_of(&key);
        if self.indexes.is_empty() {
            self.reindex(DICT_INITSIZE);
        }
        let index_slot = match self.lookup_for_store(hash, &key) {
            Ok(slot) => {
                let e = self.entries[slot].as_mut().expect("valid slot");
                return Some(std::mem::replace(&mut e.value, value));
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
    /// goes through `ll_call_insert_clean_function` (rordereddict.py:699),
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

    /// The `i < 0` arm of `_ll_dict_setitem_lookup_done` (rordereddict.py:675):
    /// grow or compact, then claim an index slot for a fresh entry.
    /// `index_slot` is the one a `FLAG_STORE` probe ended on, `None` when the
    /// caller never probed.
    fn setitem_lookup_done(&mut self, hash: u64, index_slot: Option<usize>, key: K, value: V) {
        let mut reindexed = false;
        // `if len(d.entries) == d.num_ever_used_items: ll_dict_grow(d)` — the
        // entries array is full, and `ll_dict_grow` (755) compacts instead of
        // growing when over half of it is dead.
        if self.entries.len() == self.entries.capacity()
            && self.num_live_items < self.entries.len() / 2
        {
            self.remove_deleted_items();
            reindexed = true;
        }
        let mut rc = self.resize_counter - 3;
        if rc <= 0 {
            self.resize();
            reindexed = true;
            rc = self.resize_counter - 3;
        }
        // A reindex rebuilt the whole table, so the probe's slot names nothing.
        match index_slot.filter(|_| !reindexed) {
            Some(index_slot) => self.indexes[index_slot] = self.next_slot() + VALID_OFFSET,
            None => {
                let slot = self.next_slot();
                self.insert_clean(hash, slot);
            }
        }
        self.resize_counter = rc;
        // `ll_dict_grow` replaces `d.entries` outright when the array is full,
        // so a growth here is the same event `entries != d.entries` reports.
        //
        // !! Read the *capacity*, not the data pointer: a `Vec` growth goes
        // through `realloc`, and an allocator that extends the block in place
        // hands back the address it already had — measured, this passed on
        // dynasm and cranelift and left wasm answering a mutating `__eq__`
        // without its restart.  Capacity changes on every growth whatever the
        // allocator does.
        let entries_capacity = self.entries.capacity();
        self.entries.push(Some(Entry { hash, key, value }));
        if self.entries.capacity() != entries_capacity {
            self.generation = self.generation.wrapping_add(1);
        }
        self.num_live_items += 1;
    }

    /// `ll_call_delete_by_entry_index` (rordereddict.py:1157) — re-probe from
    /// the entry's own digest for the one index slot naming it.
    fn delete_by_entry_index(&mut self, hash: u64, slot: usize) {
        let mask = self.indexes.len() - 1;
        let target = slot as u32 + VALID_OFFSET;
        let mut i = (hash as usize) & mask;
        let mut perturb = hash;
        let mut probes = 0;
        while self.indexes[i] != target {
            i = Self::probe_next(i, perturb, mask);
            perturb >>= PERTURB_SHIFT;
            probes += 1;
            debug_assert!(
                probes <= self.indexes.len(),
                "no index slot names entry {slot}"
            );
        }
        self.indexes[i] = DELETED;
    }

    /// `ll_dict_pop` (rordereddict.py:1497).  Order-preserving and O(1): the
    /// name says `remove` and not `shift_remove` because nothing shifts.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_of(key);
        let slot = self.lookup(hash, key)?;
        let value = self.take_slot(hash, slot).1;
        Some(value)
    }

    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_of(key);
        let slot = self.lookup(hash, key)?;
        Some(self.take_slot(hash, slot))
    }

    /// Delete the pair at `slot`, which must be valid.
    pub fn remove_slot(&mut self, slot: usize) -> Option<(K, V)> {
        let hash = match self.entries.get(slot) {
            Some(Some(e)) => e.hash,
            _ => return None,
        };
        Some(self.take_slot(hash, slot))
    }

    fn take_slot(&mut self, hash: u64, slot: usize) -> (K, V) {
        self.delete_by_entry_index(hash, slot);
        let entry = self.entries[slot].take().expect("valid slot");
        self.num_live_items -= 1;

        if self.num_live_items == 0 {
            self.entries.clear();
        } else if slot == self.entries.len() - 1 {
            while matches!(self.entries.last(), Some(None)) {
                self.entries.pop();
            }
        }
        if self.num_live_items + DICT_INITSIZE <= self.entries.capacity() / 8 {
            self.resize();
        }
        (entry.key, entry.value)
    }

    /// `ll_dict_popitem` (rordereddict.py:1488) — the last live pair.
    pub fn pop(&mut self) -> Option<(K, V)> {
        let mut slot = self.entries.len();
        loop {
            if slot == 0 {
                return None;
            }
            slot -= 1;
            if self.entries[slot].is_some() {
                return self.remove_slot(slot);
            }
        }
    }

    /// `move_to_end` (pypy/objspace/std/dictmultiobject.py:598) for a slot the
    /// caller already located.  Answers whether anything moved: a key already
    /// at the wanted end is a no-op, and the caller must not bump its
    /// iterator-invalidation state for one.
    ///
    /// To the back: delete and re-insert, which appends — upstream's
    /// `internal_delitem` + `setitem` pair.  To the front there is no cheap
    /// move; upstream calls its own path "a *very slow* fall-back" and rebuilds
    /// the dict, so this does too.
    pub fn move_slot_to_end(&mut self, slot: usize, last: bool) -> bool {
        let hash = match self.entries.get(slot) {
            Some(Some(e)) => e.hash,
            _ => return false,
        };
        if last {
            if slot + 1 == self.entries.len() {
                return false;
            }
            let (k, v) = self.take_slot(hash, slot);
            self.insert(k, v);
        } else {
            if self.next_valid_slot(0) == Some(slot) {
                return false;
            }
            let (k, v) = self.take_slot(hash, slot);
            let rest: Vec<(K, V)> = self
                .entries
                .drain(..)
                .flatten()
                .map(|e| (e.key, e.value))
                .collect();
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
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_of(key);
        let slot = self.lookup(hash, key)?;
        Some(self.move_slot_to_end(slot, last))
    }

    pub fn reserve(&mut self, additional: usize) {
        let entries_capacity = self.entries.capacity();
        self.entries.reserve(additional);
        if self.entries.capacity() != entries_capacity {
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

/// `_overallocate_entries_len` (rordereddict.py:745) — "the growth pattern is:
/// 0, 8, 17, 27, 38, 50, 64, 80, 98, ...".
fn overallocate_entries_len(baselen: usize) -> usize {
    baselen + (baselen >> 3) + 8
}

impl<K: Hash + Eq, V, S: BuildHasher + Default> FromIterator<(K, V)> for RDict<K, V, S> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut d = Self::with_hasher(S::default());
        for (k, v) in iter {
            d.insert(k, v);
        }
        d
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> Extend<(K, V)> for RDict<K, V, S> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K: Clone, V: Clone, S: Clone> Clone for RDict<K, V, S> {
    fn clone(&self) -> Self {
        Self {
            indexes: self.indexes.clone(),
            entries: self.entries.clone(),
            num_live_items: self.num_live_items,
            resize_counter: self.resize_counter,
            generation: self.generation,
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<K: std::fmt::Debug, V: std::fmt::Debug, S> std::fmt::Debug for RDict<K, V, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<'a, K, V, S> IntoIterator for &'a RDict<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = std::iter::FilterMap<
        std::slice::Iter<'a, Option<Entry<K, V>>>,
        fn(&'a Option<Entry<K, V>>) -> Option<(&'a K, &'a V)>,
    >;
    fn into_iter(self) -> Self::IntoIter {
        fn pair<K, V>(e: &Option<Entry<K, V>>) -> Option<(&K, &V)> {
            e.as_ref().map(|e| (&e.key, &e.value))
        }
        self.entries.iter().filter_map(pair as fn(_) -> _)
    }
}

impl<K, V, S> IntoIterator for RDict<K, V, S> {
    type Item = (K, V);
    type IntoIter = std::iter::Map<
        std::iter::Flatten<std::vec::IntoIter<Option<Entry<K, V>>>>,
        fn(Entry<K, V>) -> (K, V),
    >;
    fn into_iter(self) -> Self::IntoIter {
        fn pair<K, V>(e: Entry<K, V>) -> (K, V) {
            (e.key, e.value)
        }
        self.entries.into_iter().flatten().map(pair as fn(_) -> _)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- a reference model: insertion-ordered Vec + linear lookup ----
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
        let mut d: RDict<String, u64> = RDict::default();
        d.insert("alpha".to_string(), 1);
        d.insert("beta".to_string(), 2);
        assert_eq!(d.get("alpha"), Some(&1));
        assert_eq!(d.remove("beta"), Some(2));
        assert_eq!(d.get("beta"), None);
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
            let (capacity, generation) = (d.entries.capacity(), d.generation);
            d.insert(i, i);
            if d.entries.capacity() != capacity {
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

    // ---- a comparison that re-enters and reshapes the dict ----
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
