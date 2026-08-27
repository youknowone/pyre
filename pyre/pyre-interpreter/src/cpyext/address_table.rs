//! Hashing for the side tables this layer keys on a mirror address.
//!
//! Every one of them stores what a block at a fixed address is: its size, the
//! items it handed out, the bytes it exposes.  A mirror address is already a
//! well-spread 64-bit value, so the SipHash the standard hasher runs over it
//! buys nothing and costs more than the lookup it guards -- and these tables
//! are read on the reference-counting path, once per `incref`/`decref`, which
//! is the hottest thing a C extension does.
//!
//! This is the address counterpart of
//! [`pyre_object::dictmultiobject::ObjectKeyHasher`], and takes its mangling
//! from `majit_gc::address_dict::AddressHasher`, which does the same job for
//! the collector's own address tables.

use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Odd 64-bit multiplier (2^64 / golden ratio).
const SPREAD_MULTIPLIER: u64 = 0x9E37_79B9_7F4A_7C15;

#[derive(Default)]
pub(super) struct AddressHasher(u64);

impl AddressHasher {
    /// A block address is aligned, so its trailing bits are zero and a table
    /// indexing on those alone would pile every key into one bucket;
    /// `rpython/memory/support.py:10-14 mangle_hash` folds the high bits down
    /// over them.  `HashMap` reads the hash a second way -- the top 7 bits
    /// become the entry's control byte -- and a heap address leaves those
    /// zero, so the multiply carries the mangled value back over the whole
    /// word.  The low half still depends only on the low half, so the bucket
    /// index keeps the mangled distribution.
    #[inline]
    fn spread(value: u64) -> u64 {
        (value ^ (value >> 4)).wrapping_mul(SPREAD_MULTIPLIER)
    }
}

impl Hasher for AddressHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    /// These tables are `usize`-keyed, so `write_usize` is the path taken.
    /// Keep a deterministic fallback for the `Hasher` contract.
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut value = 0_u64;
        for (shift, byte) in bytes.iter().copied().take(8).enumerate() {
            value |= (byte as u64) << (shift * 8);
        }
        self.0 = Self::spread(value);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.0 = Self::spread(value as u64);
    }
}

/// Const-constructible, which is what lets these tables be `static`.
pub(super) type AddressBuildHasher = BuildHasherDefault<AddressHasher>;
pub(super) type AddressMap<V> = std::collections::HashMap<usize, V, AddressBuildHasher>;
pub(super) type AddressSet = std::collections::HashSet<usize, AddressBuildHasher>;

/// A collection whose entry count is what decides whether it holds anything.
pub(super) trait Populated {
    fn population(&self) -> usize;
}

impl<V> Populated for AddressMap<V> {
    fn population(&self) -> usize {
        self.len()
    }
}

impl Populated for AddressSet {
    fn population(&self) -> usize {
        self.len()
    }
}

/// A side table keyed on a mirror address, publishing how much it holds.
///
/// Tearing a mirror down asks every one of these tables to release whatever it
/// keyed by that address, and for most programs almost none of them hold
/// anything: an extension that never builds a `datetime` leaves that table
/// empty for the whole run, and one that never hands C a borrowed reference
/// leaves `BORROWED` empty.  Asking is otherwise a mutex and a probe each,
/// once per teardown, so the count is published outside the lock and an empty
/// table answers [`AddressTable::is_empty`] without taking it.
///
/// The count is republished when a guard is dropped rather than by each
/// mutating call, which is what lets every caller keep using the collection
/// directly through the guard.
pub(super) struct AddressTable<C> {
    entries: super::ForkMutex<C>,
    population: AtomicUsize,
}

// SAFETY: the payload is reachable only through `entries`, whose own `Sync`
// carries the same bound.
unsafe impl<C: Send> Sync for AddressTable<C> {}

impl<C: Populated> AddressTable<C> {
    pub(super) const fn new(entries: C) -> Self {
        Self {
            entries: super::ForkMutex::new(entries),
            population: AtomicUsize::new(0),
        }
    }

    pub(super) fn lock(&self) -> AddressTableGuard<'_, C> {
        AddressTableGuard {
            entries: self.entries.lock(),
            population: &self.population,
        }
    }

    /// Whether the table holds nothing, without taking its lock.
    ///
    /// A teardown reaching this has the mirror's last reference, so no other
    /// thread is entering the table under that address; the acquire pairs with
    /// the release a guard makes before it unlocks, so a count read here is
    /// one some thread actually published.
    pub(super) fn is_empty(&self) -> bool {
        self.population.load(Ordering::Acquire) == 0
    }

    /// # Safety
    /// Only in a forked child, where the thread that held the lock is gone.
    pub(super) unsafe fn reinit_after_fork(&self) {
        unsafe { self.entries.reinit_after_fork() };
    }
}

impl<V> AddressTable<AddressMap<V>> {
    /// Take what this table keyed by `address`, if anything.
    ///
    /// This is the teardown spelling: a mirror reaching it has no references
    /// left, so no other thread is entering the table under its address, and a
    /// table that reads empty stays empty for every key that matters here.
    pub(super) fn take(&self, address: usize) -> Option<V> {
        if self.is_empty() {
            return None;
        }
        self.lock().remove(&address)
    }
}

impl AddressTable<AddressSet> {
    /// Drop `address` from this table, answering whether it was there.
    ///
    /// The empty case is decided as it is in [`AddressTable::take`].
    pub(super) fn discard(&self, address: usize) -> bool {
        if self.is_empty() {
            return false;
        }
        self.lock().remove(&address)
    }
}

/// Holds a table open, and republishes its population on the way out.
pub(super) struct AddressTableGuard<'a, C: Populated> {
    entries: parking_lot::MutexGuard<'a, C>,
    population: &'a AtomicUsize,
}

impl<C: Populated> Drop for AddressTableGuard<'_, C> {
    fn drop(&mut self) {
        self.population
            .store(self.entries.population(), Ordering::Release);
    }
}

impl<C: Populated> std::ops::Deref for AddressTableGuard<'_, C> {
    type Target = C;

    fn deref(&self) -> &C {
        &self.entries
    }
}

impl<C: Populated> std::ops::DerefMut for AddressTableGuard<'_, C> {
    fn deref_mut(&mut self) -> &mut C {
        &mut self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    fn hash_of(address: usize) -> u64 {
        let mut hasher = AddressHasher::default();
        address.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn aligned_addresses_differ_in_the_control_byte() {
        // Sixteen-byte-aligned blocks are the shape a mirror allocator hands
        // out; SipHash is not needed to tell them apart, but the top bits do
        // have to move, or every entry carries the same control byte.
        let top_bits: std::collections::HashSet<u64> =
            (0..64).map(|i| hash_of(0x1_0000 + i * 16) >> 57).collect();
        assert!(top_bits.len() > 32, "control bytes collapsed: {top_bits:?}");
    }

    #[test]
    fn a_key_hashes_to_one_value() {
        assert_eq!(hash_of(0x1_0000), hash_of(0x1_0000));
        assert_ne!(hash_of(0x1_0000), hash_of(0x1_0010));
    }
}
