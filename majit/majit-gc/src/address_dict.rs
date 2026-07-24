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
        self.hash = value ^ (value >> 4);
    }

    fn write_usize(&mut self, value: usize) {
        self.hash = (value ^ (value >> 4)) as u64;
    }
}

pub(crate) type AddressMap<V> = HashMap<usize, V, BuildHasherDefault<AddressHasher>>;
pub(crate) type AddressSet = HashSet<usize, BuildHasherDefault<AddressHasher>>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{Hash, Hasher};

    #[test]
    fn address_hasher_matches_rpython_mangle_hash() {
        let address = 0x1234_5678usize;
        let mut hasher = AddressHasher::default();
        address.hash(&mut hasher);
        assert_eq!(hasher.finish(), (address ^ (address >> 4)) as u64);
    }
}
