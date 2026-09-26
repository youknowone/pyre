//! `rpython/rlib/rarithmetic.py` — machine-integer helpers the interpreter calls.

/// `(a * b) % c`, with `c > 0`, result nonnegative.
///
/// `rarithmetic.py mulmod`. 64-bit `long` uses the 128-bit product
/// (`check_support_int128`), then `intmask` truncates back to a signed long.
/// Floor division makes a negative product's remainder nonnegative.
#[inline(never)]
#[majit_macros::dont_look_inside]
pub fn mulmod(a: i64, b: i64, c: i64) -> i64 {
    debug_assert!(c > 0);
    let product = (a as i128).wrapping_mul(b as i128);
    let modulus = c as i128;
    let mut remainder = product % modulus;
    if remainder < 0 {
        remainder += modulus;
    }
    remainder as i64
}
