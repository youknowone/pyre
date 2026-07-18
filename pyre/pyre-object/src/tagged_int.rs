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
//! The tag path is live behind [`CAN_BE_TAGGED`] (#22 enablement): the
//! maker (`intobject::w_int_new`) returns small ints as immediates, the
//! readers/dispatch chokepoints `& 1`-precheck before any `ob_type`
//! deref, and the GC collector skips tagged immediates
//! (`taggedpointers`, wired through `pyre-jit`'s `build_gc`).
//!
//! The bit layout mirrors the already-ported rtyper helper
//! `majit/majit-translate/src/translator/rtyper/lltypesystem/rtagged.rs`
//! (`ll_int_to_unboxed` = `value * 2 + 1`, `ll_unboxed_to_int` =
//! `n >> 1`, `is_unboxed_instance` = `(n & 1) != 0`).

use crate::pyobject::PyObjectRef;

/// `rpython/rtyper/lltypesystem/rtagged.py:64-96` static `can_be_tagged`
/// gate, collapsed to the single runtime `int` class. Enabled (#22),
/// mirroring `rpython/config/translationoption.py:185 taggedpointers`
/// turned on, so every consumer chokepoint takes the `& 1` tag precheck
/// and the maker emits small ints as immediates. `rerased.py:1-3`: the
/// point is to avoid putting `& 1` tag checks on every object — they are
/// gated on this static, which is kept in lockstep with the GC
/// `taggedpointers` config (`pyre-jit` `build_gc`).
pub const CAN_BE_TAGGED: bool = false;

/// `value` fits the tagged immediate range, i.e. the payload survives
/// `<< 1` within pointer width. Callers range-check with this before
/// [`tag_int`]; it mirrors the `checked_mul`/`checked_add` overflow
/// guard in `rtagged.rs::ll_int_to_unboxed`. On 64-bit this is the full
/// `i64::MIN>>1 ..= i64::MAX>>1` range; on wasm32 it narrows to a
/// 31-bit signed payload.
#[inline]
pub fn fits_tagged(value: i64) -> bool {
    // The tagged immediate `(value << 1) | 1` is stored in a pointer-width
    // slot, so the payload must survive `<< 1` within `isize` (sign
    // preserving). On 64-bit this is the full `i64::MIN>>1 ..= i64::MAX>>1`
    // range; on wasm32 it narrows to a 31-bit signed payload.
    const LO: i64 = (isize::MIN >> 1) as i64;
    const HI: i64 = (isize::MAX >> 1) as i64;
    value >= LO && value <= HI
}

/// `ll_int_to_unboxed` — reinterpret `(value << 1) | 1` as a pointer.
///
/// The caller must have checked [`fits_tagged`]; the tagging arithmetic
/// lives here, never at the `Signed<->Ptr` cast boundaries (those stay
/// identity reinterpret-casts).
#[inline]
pub fn tag_int(value: i64) -> PyObjectRef {
    debug_assert!(fits_tagged(value), "tag_int: value out of taggable range");
    ((((value as isize) << 1) | 1) as usize) as PyObjectRef
}

/// `ll_unboxed_to_int` — recover the payload with an arithmetic (sign
/// preserving) `>> 1`. Caller must have established [`is_tagged_int`].
#[inline]
pub fn untag_int(p: PyObjectRef) -> i64 {
    ((p as usize as isize) >> 1) as i64
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
        // The scaffolding is inert behind `CAN_BE_TAGGED`: tagging mainline
        // ints is off by default (mirrors `taggedpointers` off by default).
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
        // Derive the boundary from pointer width so the round-trip is
        // correct on any target (full i64>>1 on 64-bit, 31-bit on wasm32).
        let lo = (isize::MIN >> 1) as i64;
        let hi = (isize::MAX >> 1) as i64;
        assert!(fits_tagged(lo) && fits_tagged(hi));
        assert_eq!(untag_int(tag_int(lo)), lo);
        assert_eq!(untag_int(tag_int(hi)), hi);
    }

    #[test]
    fn fits_tagged_rejects_top_bit_values() {
        let lo = (isize::MIN >> 1) as i64;
        let hi = (isize::MAX >> 1) as i64;
        assert!(fits_tagged(hi));
        assert!(fits_tagged(lo));
        // hi = i64::MAX>>1 < i64::MAX and lo = i64::MIN>>1 > i64::MIN on
        // both 32- and 64-bit, so the `+1`/`-1` cannot overflow.
        assert!(!fits_tagged(hi + 1));
        assert!(!fits_tagged(lo - 1));
    }

    #[test]
    fn fits_tagged_matches_pointer_width() {
        // The taggable range tracks pointer width: the top payload fits,
        // one past it does not.
        assert!(fits_tagged((isize::MAX >> 1) as i64));
        assert!(!fits_tagged(((isize::MAX >> 1) as i64) + 1));
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
