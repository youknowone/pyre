//! Untraced-path bodies for the intrinsics `#[jit_interp]` rewrites.
//!
//! A `#[jit_interp]` mainloop runs at two tiers. While tracing, the macro
//! rewrites a call to one of the names below into a trace op — the body is
//! never entered. At the interpreter tier the same source calls a real Rust
//! function, and the two tiers must agree bit-for-bit or a compiled loop
//! answers differently from the interpreter that fed it. This module is that
//! function, once, so an interpreter does not hand-write it per module.
//!
//! Every body is `#[inline]`: the interpreter tier calls them per opcode, and
//! reaching them from a consumer crate crosses a boundary a hand-written copy
//! did not.
//!
//! **The macro matches the LAST PATH SEGMENT of the call expression**, not the
//! definition site. All three spellings therefore lower identically:
//!
//! ```ignore
//! use majit_metainterp::intrinsics::majit_raw_load_i64;
//! majit_raw_load_i64(base, ea);                                // imported
//! majit_metainterp::intrinsics::majit_raw_load_i64(base, ea);  // qualified
//! fn majit_raw_load_i64(base: i64, ea: i64) -> i64 { .. }      // local
//! ```
//!
//! `majit_uint_mul_high` is the one exception and is documented at its
//! definition: it has no hard-coded name in the macro and is reached only
//! through a `native_int_binops` alias.
//!
//! # Addresses and safety
//!
//! Raw-memory intrinsics take their address as an `i64`, the `support`
//! `AddressAsInt` convention, and `ea` is a BYTE offset from it — as
//! `rawstorage.py` `raw_storage_getitem` takes an `index` it `ptradd`s onto a
//! `CCHARP`.
//!
//! ⚠ These are safe `fn`s that dereference a caller-supplied integer, and that
//! is forced, not chosen: the lowerer matches a bare call expression and has no
//! rule for an `unsafe` block, so an `unsafe fn` spelling would stop lowering
//! and silently leave the traced tier calling into the interpreter body. The
//! obligation is the caller's on every one of them — `base + ea` must address
//! a live, correctly sized allocation for the whole call, and the traced tier
//! repeats the access with no check of its own.

/// `rawstorage.py` `raw_storage_getitem` at one byte, sign-extended.
///
/// Loads widen into the 64-bit int register bank, so the intrinsic's own
/// signedness — not the register's — decides whether the high bits are the
/// sign or zero. The traced tier reads the same width and signedness off the
/// array descr the lowerer attaches.
#[inline]
pub fn majit_raw_load_i8(base: i64, ea: i64) -> i64 {
    unsafe { core::ptr::read_unaligned(raw_addr(base, ea) as *const i8) as i64 }
}

/// `rawstorage.py` `raw_storage_getitem` at one byte, zero-extended.
#[inline]
pub fn majit_raw_load_u8(base: i64, ea: i64) -> i64 {
    unsafe { core::ptr::read_unaligned(raw_addr(base, ea) as *const u8) as i64 }
}

/// `rawstorage.py` `raw_storage_getitem` at two bytes, sign-extended.
#[inline]
pub fn majit_raw_load_i16(base: i64, ea: i64) -> i64 {
    unsafe { core::ptr::read_unaligned(raw_addr(base, ea) as *const i16) as i64 }
}

/// `rawstorage.py` `raw_storage_getitem` at two bytes, zero-extended.
#[inline]
pub fn majit_raw_load_u16(base: i64, ea: i64) -> i64 {
    unsafe { core::ptr::read_unaligned(raw_addr(base, ea) as *const u16) as i64 }
}

/// `rawstorage.py` `raw_storage_getitem` at four bytes, sign-extended.
#[inline]
pub fn majit_raw_load_i32(base: i64, ea: i64) -> i64 {
    unsafe { core::ptr::read_unaligned(raw_addr(base, ea) as *const i32) as i64 }
}

/// `rawstorage.py` `raw_storage_getitem` at four bytes, zero-extended.
#[inline]
pub fn majit_raw_load_u32(base: i64, ea: i64) -> i64 {
    unsafe { core::ptr::read_unaligned(raw_addr(base, ea) as *const u32) as i64 }
}

/// `rawstorage.py` `raw_storage_getitem` at eight bytes.
#[inline]
pub fn majit_raw_load_i64(base: i64, ea: i64) -> i64 {
    unsafe { core::ptr::read_unaligned(raw_addr(base, ea) as *const i64) }
}

/// `rawstorage.py` `raw_storage_getitem` at eight bytes, unsigned descr.
///
/// At the register width there is no extension gap, so this reads the same
/// bits as [`majit_raw_load_i64`]. It exists because the lowerer accepts the
/// spelling and stamps an unsigned descr for it.
#[inline]
pub fn majit_raw_load_u64(base: i64, ea: i64) -> i64 {
    unsafe { core::ptr::read_unaligned(raw_addr(base, ea) as *const u64) as i64 }
}

/// `rawstorage.py` `raw_storage_getitem` at eight bytes into the FLOAT bank.
///
/// The only load whose result is a float register; the lowerer stamps a raw
/// float array descr for it rather than a width/signedness pair.
#[inline]
pub fn majit_raw_load_f(base: i64, ea: i64) -> f64 {
    unsafe { core::ptr::read_unaligned(raw_addr(base, ea) as *const f64) }
}

/// `rawstorage.py` `raw_storage_setitem` at one byte.
///
/// The stored value arrives in an int register and is truncated to the
/// intrinsic's width. Signedness cannot change which bits land, so the signed
/// and unsigned spellings at a given width write identically; they differ only
/// in the descr the traced tier carries.
#[inline]
pub fn majit_raw_store_i8(base: i64, ea: i64, val: i64) {
    unsafe { core::ptr::write_unaligned(raw_addr(base, ea) as *mut i8, val as i8) }
}

/// `rawstorage.py` `raw_storage_setitem` at one byte, unsigned descr.
#[inline]
pub fn majit_raw_store_u8(base: i64, ea: i64, val: i64) {
    unsafe { core::ptr::write_unaligned(raw_addr(base, ea) as *mut u8, val as u8) }
}

/// `rawstorage.py` `raw_storage_setitem` at two bytes.
#[inline]
pub fn majit_raw_store_i16(base: i64, ea: i64, val: i64) {
    unsafe { core::ptr::write_unaligned(raw_addr(base, ea) as *mut i16, val as i16) }
}

/// `rawstorage.py` `raw_storage_setitem` at two bytes, unsigned descr.
#[inline]
pub fn majit_raw_store_u16(base: i64, ea: i64, val: i64) {
    unsafe { core::ptr::write_unaligned(raw_addr(base, ea) as *mut u16, val as u16) }
}

/// `rawstorage.py` `raw_storage_setitem` at four bytes.
#[inline]
pub fn majit_raw_store_i32(base: i64, ea: i64, val: i64) {
    unsafe { core::ptr::write_unaligned(raw_addr(base, ea) as *mut i32, val as i32) }
}

/// `rawstorage.py` `raw_storage_setitem` at four bytes, unsigned descr.
#[inline]
pub fn majit_raw_store_u32(base: i64, ea: i64, val: i64) {
    unsafe { core::ptr::write_unaligned(raw_addr(base, ea) as *mut u32, val as u32) }
}

/// `rawstorage.py` `raw_storage_setitem` at eight bytes.
#[inline]
pub fn majit_raw_store_i64(base: i64, ea: i64, val: i64) {
    unsafe { core::ptr::write_unaligned(raw_addr(base, ea) as *mut i64, val) }
}

/// `rawstorage.py` `raw_storage_setitem` at eight bytes, unsigned descr.
#[inline]
pub fn majit_raw_store_u64(base: i64, ea: i64, val: i64) {
    unsafe { core::ptr::write_unaligned(raw_addr(base, ea) as *mut u64, val as u64) }
}

/// `longlong2float.py` `float2longlong` — a float's 64-bit pattern read as an
/// int, no value change.
///
/// Lowers to `convert_float_bytes_to_longlong`. A branchless float select uses
/// this pair to stay bit-exact where an arithmetic blend cannot.
#[inline]
pub fn majit_f64_to_bits(x: f64) -> i64 {
    x.to_bits() as i64
}

/// `longlong2float.py` `longlong2float` — the inverse bitcast, lowering to
/// `convert_longlong_bytes_to_float`.
#[inline]
pub fn majit_bits_to_f64(x: i64) -> f64 {
    f64::from_bits(x as u64)
}

/// Unsigned `<` over the int bank, lowering to the `uint_lt` resop.
///
/// A `rarithmetic.py` `r_uint` value travels in an ordinary int register as its
/// raw 64-bit pattern, and the lowerer collapses every Rust comparison operator
/// to its SIGNED opcode. An explicit intrinsic is therefore the only way to
/// select the unsigned comparison from the tracing frontend; a bare `<`
/// disagrees with this body for any operand at or above `2^63`.
#[inline]
pub fn majit_uint_lt(a: i64, b: i64) -> i64 {
    ((a as u64) < (b as u64)) as i64
}

/// Unsigned `<=` over the int bank, lowering to the `uint_le` resop. See
/// [`majit_uint_lt`].
#[inline]
pub fn majit_uint_le(a: i64, b: i64) -> i64 {
    ((a as u64) <= (b as u64)) as i64
}

/// Unsigned `/` over the int bank — `rint.py` `ll_uint_py_div`.
///
/// This lowers to the `int.udiv` oopspec residual call rather than to a trace
/// opcode: `UINT_FLOORDIV` was removed from the resop set, and unsigned
/// division routes through that elidable call instead.
///
/// The caller must guarantee `b != 0`, exactly the precondition
/// `ll_uint_py_div_zer` wraps: this body divides unconditionally and the
/// compiled tier does too.
#[inline]
pub fn majit_uint_div(a: i64, b: i64) -> i64 {
    ((a as u64) / (b as u64)) as i64
}

/// Unsigned `%` over the int bank — `rint.py` `ll_uint_py_mod`, lowering to the
/// `int.umod` oopspec residual call. Carries [`majit_uint_div`]'s `b != 0`
/// precondition.
#[inline]
pub fn majit_uint_mod(a: i64, b: i64) -> i64 {
    ((a as u64) % (b as u64)) as i64
}

/// `rarithmetic.py` `uint_mul_high` — the high 64 bits of the 128-bit unsigned
/// product, zero exactly when `a * b` fits in a `u64`. That makes it the
/// unsigned multiply-overflow test.
///
/// ⚠ UNLIKE EVERY OTHER NAME HERE, this one is not hard-coded in the lowerer.
/// It is reached only through a `native_int_binops` alias, which matches the
/// call's FULL path against the configured one:
///
/// ```ignore
/// use majit_metainterp::intrinsics::majit_uint_mul_high;
/// // #[jit_interp(.., native_int_binops = { majit_uint_mul_high => UintMulHigh })]
/// ```
///
/// So the alias key and the call site must be spelled the same way — an
/// imported bare name in both, or the same qualified path in both. Configured
/// under one spelling and called under the other, the call lowers as an
/// ordinary residual and the `u128` below is what the trace executes.
#[inline]
pub fn majit_uint_mul_high(a: i64, b: i64) -> i64 {
    (((a as u64 as u128) * (b as u64 as u128)) >> 64) as u64 as i64
}

/// The effective address a raw-memory intrinsic touches.
///
/// Wrapping rather than checked because the pair is an address and a byte
/// offset that the caller has already resolved; an overflow here would be a
/// caller defect that a panic in the interpreter tier would report and the
/// traced tier, which computes the same sum in a register, would not.
#[inline]
fn raw_addr(base: i64, ea: i64) -> usize {
    (base as usize).wrapping_add(ea as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch buffer addressed the way an intrinsic addresses it, with no
    /// shared reference aliasing the writes: the store tests below mutate
    /// through this address.
    fn base_of(buf: &mut [u8]) -> i64 {
        buf.as_mut_ptr() as usize as i64
    }

    #[test]
    fn raw_loads_extend_by_the_intrinsics_own_signedness() {
        // Every byte is 0xff, so a signed load reads -1 at any width and an
        // unsigned one reads that width's mask — whatever the byte order is.
        let mut cell = [0xffu8; 8];
        let base = base_of(&mut cell);
        assert_eq!(majit_raw_load_i8(base, 0), -1);
        assert_eq!(majit_raw_load_u8(base, 0), 0xff);
        assert_eq!(majit_raw_load_i16(base, 0), -1);
        assert_eq!(majit_raw_load_u16(base, 0), 0xffff);
        assert_eq!(majit_raw_load_i32(base, 0), -1);
        assert_eq!(majit_raw_load_u32(base, 0), 0xffff_ffff);
        assert_eq!(majit_raw_load_i64(base, 0), -1);
        assert_eq!(majit_raw_load_u64(base, 0), -1);
    }

    #[test]
    fn ea_is_a_byte_offset_not_an_element_index() {
        let cells: [i64; 3] = [10, 20, 30];
        let base = cells.as_ptr() as usize as i64;
        assert_eq!(majit_raw_load_i64(base, 0), 10);
        assert_eq!(majit_raw_load_i64(base, 16), 30);

        // Byte-addressed at one-byte width, so the offset is the index there.
        let mut bytes: [u8; 4] = [7, 8, 9, 10];
        let byte_base = base_of(&mut bytes);
        for (i, expected) in bytes.iter().enumerate() {
            assert_eq!(majit_raw_load_u8(byte_base, i as i64), i64::from(*expected));
        }
    }

    #[test]
    fn raw_stores_truncate_to_their_width() {
        let mut buf = [0u8; 8];
        let base = base_of(&mut buf);

        // A value wider than the intrinsic keeps only the low byte, and the
        // neighbouring byte is untouched.
        majit_raw_store_u8(base, 0, 0x1ff);
        assert_eq!(majit_raw_load_u8(base, 0), 0xff);
        assert_eq!(majit_raw_load_u8(base, 1), 0);

        // Signedness cannot change which bits land: two bytes of 0xff either
        // way, and nothing beyond them.
        majit_raw_store_i64(base, 0, 0);
        majit_raw_store_i16(base, 0, -1);
        assert_eq!(majit_raw_load_u16(base, 0), 0xffff);
        assert_eq!(majit_raw_load_u16(base, 2), 0);
        majit_raw_store_i64(base, 0, 0);
        majit_raw_store_u16(base, 0, 0xffff);
        assert_eq!(majit_raw_load_u16(base, 0), 0xffff);

        // Full width writes the whole cell, and the load side reads it back.
        majit_raw_store_i64(base, 0, -1);
        assert_eq!(majit_raw_load_i64(base, 0), -1);
    }

    #[test]
    fn float_load_and_bitcasts_round_trip() {
        let cells: [f64; 2] = [1.5, -0.25];
        let base = cells.as_ptr() as usize as i64;
        assert_eq!(majit_raw_load_f(base, 8), -0.25);
        assert_eq!(majit_bits_to_f64(majit_f64_to_bits(-0.25)), -0.25);

        // The bitcast is a pattern, not a value conversion: the float load and
        // the int-bank load of the same eight bytes hold the same bits.
        assert_eq!(
            majit_f64_to_bits(majit_raw_load_f(base, 8)),
            majit_raw_load_i64(base, 8)
        );

        // NaN survives, which is why a branchless float select uses the pair
        // instead of an arithmetic blend.
        assert!(majit_bits_to_f64(majit_f64_to_bits(f64::NAN)).is_nan());
    }

    #[test]
    fn unsigned_intrinsics_disagree_with_the_signed_operators() {
        // -1 is the largest u64, which is where every signed operator is wrong
        // — the reason these intrinsics exist at all.
        assert_eq!(majit_uint_lt(-1, 1), 0);
        assert!(-1 < 1);
        assert_eq!(majit_uint_le(-1, -1), 1);
        assert_eq!(majit_uint_div(-1, 2), (u64::MAX / 2) as i64);
        assert_eq!(majit_uint_mod(-1, 10), (u64::MAX % 10) as i64);

        // A zero high word is exactly "the product fits in a u64".
        assert_eq!(majit_uint_mul_high(3, 5), 0);
        assert_eq!(majit_uint_mul_high(-1, 2), 1);
    }
}
