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
