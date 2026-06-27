//! Tagged-int primitive — `rpython/rlib/rerased.py` /
//! `rpython/rtyper/lltypesystem/rtagged.py` applied to the runtime
//! `int` representation.
//!
//! A small `int` can be stored as an *immediate* inside a `PyObjectRef`
//! slot — the odd bit pattern `(value << 1) | 1` — instead of a heap
//! `W_IntObject`. Heap pointers are 8-byte aligned (`W_IntObject`
//! carries an `i64`), so the low bit distinguishes an immediate from a
//! real pointer; `None`/`True`/`False` are even-aligned statics and are
//! never tagged.
//!
//! This module is the structural primitive only: it has **zero call
//! sites** and is inert behind [`CAN_BE_TAGGED`]. The maker
//! (`intobject::w_int_new`), the readers/dispatch chokepoints, the GC
//! collector skip, and the enablement flip land in later slices; the
//! enablement itself is gated on the symbolic-valuestack work (#73).
//!
//! The bit layout mirrors the already-ported rtyper helper
//! `majit/majit-translate/src/translator/rtyper/lltypesystem/rtagged.rs`
//! (`ll_int_to_unboxed` = `value * 2 + 1`, `ll_unboxed_to_int` =
//! `n >> 1`, `is_unboxed_instance` = `(n & 1) != 0`).

use crate::pyobject::PyObjectRef;

/// `rpython/rtyper/lltypesystem/rtagged.py:64-96` static `can_be_tagged`
/// gate, collapsed to the single runtime `int` class. Defaults `false`,
/// mirroring `rpython/config/translationoption.py:185 taggedpointers`
/// (off by default), so every consumer chokepoint short-circuits to the
/// untagged path and this primitive stays inert until the enablement
/// slice. `rerased.py:1-3`: the point is to avoid putting `& 1` tag
/// checks on every object — they are gated on this static.
pub const CAN_BE_TAGGED: bool = false;

/// `value` fits the tagged immediate range, i.e. `value << 1` does not
/// overflow `i64`. Callers range-check with this before [`tag_int`];
/// it mirrors the `checked_mul`/`checked_add` overflow guard in
/// `rtagged.rs::ll_int_to_unboxed`.
#[inline]
pub fn fits_tagged(value: i64) -> bool {
    value >= (i64::MIN >> 1) && value <= (i64::MAX >> 1)
}

/// `ll_int_to_unboxed` — reinterpret `(value << 1) | 1` as a pointer.
///
/// The caller must have checked [`fits_tagged`]; the tagging arithmetic
/// lives here, never at the `Signed<->Ptr` cast boundaries (those stay
/// identity reinterpret-casts).
#[inline]
pub fn tag_int(value: i64) -> PyObjectRef {
    debug_assert!(fits_tagged(value), "tag_int: value out of taggable range");
    (((value << 1) | 1) as usize) as PyObjectRef
}

/// `ll_unboxed_to_int` — recover the payload with an arithmetic (sign
/// preserving) `>> 1`. Caller must have established [`is_tagged_int`].
#[inline]
pub fn untag_int(p: PyObjectRef) -> i64 {
    (p as usize as i64) >> 1
}

/// `is_unboxed_instance` — the low bit distinguishes a tagged immediate
/// from a real (even-aligned) heap pointer.
#[inline]
pub fn is_tagged_int(p: PyObjectRef) -> bool {
    (p as usize) & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_untagged() {
        // The enablement gate is off, matching the `taggedpointers`
        // default; consumers must short-circuit to the heap-box path.
        assert!(!CAN_BE_TAGGED);
    }

    #[test]
    fn tag_round_trips_signed_payload() {
        for v in [-1_000_000i64, -42, -1, 0, 1, 21, 1_000_000] {
            let p = tag_int(v);
            assert!(is_tagged_int(p));
            assert_eq!(untag_int(p), v);
        }
    }

    #[test]
    fn tag_round_trips_range_boundaries() {
        let lo = i64::MIN >> 1;
        let hi = i64::MAX >> 1;
        assert!(fits_tagged(lo) && fits_tagged(hi));
        assert_eq!(untag_int(tag_int(lo)), lo);
        assert_eq!(untag_int(tag_int(hi)), hi);
    }

    #[test]
    fn fits_tagged_rejects_top_bit_values() {
        assert!(!fits_tagged(i64::MAX));
        assert!(!fits_tagged(i64::MIN));
        assert!(fits_tagged(i64::MAX >> 1));
        assert!(fits_tagged(i64::MIN >> 1));
    }

    #[test]
    fn even_aligned_and_null_pointers_are_not_tagged() {
        // A null pointer (address 0) and any 8-byte-aligned heap pointer
        // have a clear low bit, so a real `PyObjectRef` never reads as
        // tagged.
        assert!(!is_tagged_int(std::ptr::null_mut()));
        let raw = Box::into_raw(Box::new(0u64)) as PyObjectRef;
        assert!(!is_tagged_int(raw));
        unsafe { drop(Box::from_raw(raw as *mut u64)) };
    }
}
