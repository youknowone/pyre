//! RPython `AddressDict` storage adapters.
//!
//! `rpython/memory/lldict.py:_hash` hashes aligned addresses with
//! `mangle_hash(i) == i ^ (i >> 4)`.  Using Rust's default SipHash for these
//! collector-internal address tables changes both their translated shape and
//! the cost of every lookup on GC hot paths.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub(crate) struct AddressHasher {
    hash: u64,
}

/// Odd 64-bit multiplier (2^64 / golden ratio) used to carry the mangled
/// value's entropy into the high bits.  See [`AddressHasher::spread`].
const SPREAD_MULTIPLIER: u64 = 0x9E37_79B9_7F4A_7C15;

impl AddressHasher {
    /// `mangle_hash` states its own purpose: an aligned address has zeros in
    /// its *trailing* bits, so a table that indexes on those bits alone would
    /// pile every key into the same bucket (`rpython/memory/support.py:10-14`).
    /// `lldict` reuses `rdict`, which consumes the hash only that way, so the
    /// xor-shift is the whole job there.
    ///
    /// `HashMap`/`HashSet` read the hash a second way: each entry also stores
    /// a control byte taken from the **top 7 bits**, matched against the
    /// query's before any key is compared.  A heap address leaves those bits
    /// zero and the xor-shift moves nothing into them, so every entry in the
    /// table carries the same control byte, the match reports every occupied
    /// slot in the group as a candidate, and a lookup that should have ended
    /// at the control byte instead compares keys one by one.  Multiplying by
    /// an odd constant spreads the mangled value over the whole word, which is
    /// the same intent applied to the bits this container actually reads; the
    /// low half still depends only on the low half, so the bucket index keeps
    /// the mangled distribution.
    #[inline]
    fn spread(value: u64) -> u64 {
        (value ^ (value >> 4)).wrapping_mul(SPREAD_MULTIPLIER)
    }
}

impl Hasher for AddressHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, bytes: &[u8]) {
        // AddressMap/AddressSet only use usize keys, whose Hash implementation
        // calls write_usize. Keep a deterministic fallback for the Hasher
        // contract and for direct tests.
        let mut value = 0_u64;
        for (shift, byte) in bytes.iter().copied().take(8).enumerate() {
            value |= (byte as u64) << (shift * 8);
        }
        self.hash = Self::spread(value);
    }

    fn write_usize(&mut self, value: usize) {
        self.hash = Self::spread(value as u64);
    }
}

pub(crate) type AddressMap<V> = HashMap<usize, V, BuildHasherDefault<AddressHasher>>;
pub(crate) type AddressSet = HashSet<usize, BuildHasherDefault<AddressHasher>>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{Hash, Hasher};

    fn hash_of(address: usize) -> u64 {
        let mut hasher = AddressHasher::default();
        address.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn address_hasher_starts_from_rpython_mangle_hash() {
        let address = 0x1234_5678usize;
        assert_eq!(
            hash_of(address),
            ((address ^ (address >> 4)) as u64).wrapping_mul(SPREAD_MULTIPLIER)
        );
    }

    #[test]
    fn address_hasher_gives_aligned_heap_addresses_distinct_control_bytes() {
        // Sixteen-byte-aligned addresses out of one heap region: what the
        // collector's tables are keyed on. The table's control byte is the top
        // 7 bits of the hash, which the bare mangle leaves zero for every one
        // of them.
        let addresses: Vec<usize> = (0..256).map(|i| 0x0001_2000_0000 + i * 0x40).collect();
        let mangled: HashSet<u8> = addresses
            .iter()
            .map(|&a| (((a ^ (a >> 4)) as u64) >> 57) as u8)
            .collect();
        assert_eq!(mangled.len(), 1, "premise: the bare mangle collapses");

        let spread: HashSet<u8> = addresses
            .iter()
            .map(|&a| (hash_of(a) >> 57) as u8)
            .collect();
        assert!(
            spread.len() > 64,
            "control bytes still collapse: {} distinct",
            spread.len()
        );
    }
}
