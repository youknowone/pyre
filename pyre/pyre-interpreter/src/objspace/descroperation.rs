//! pypy/objspace/descroperation.py — binary/unary operation dispatch.
//!
//! The ObjSpace mediates all operations on Python objects. This module
//! contains the dispatch layer that routes `+`, `-`, `*`, `//`, `%`,
//! `**`, `<<`, `>>`, `&`, `|`, `^`, comparisons, and unary `+`/`-`/`~`
//! through type-specific fast paths and then the dunder protocol.
#![allow(non_camel_case_types, non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use malachite_bigint::BigInt;
use num_integer::Integer;
use num_traits::ToPrimitive;

use pyre_object::strobject::is_str;
use pyre_object::*;
use rustpython_wtf8::Wtf8Buf;

use crate::baseobjspace::{
    getattr, getitem, is_true, issubtype_w, lookup, lookup_in_type, lookup_in_type_where,
    lookup_where_with_method_cache, p_abstract_issubclass_w, unwrap_cell,
};
pub use crate::{PyError, PyErrorKind, PyResult};

// ── BigInt helpers ──────────────────────────────────────────────────

/// Extract a BigInt from an int or long object.

unsafe fn as_bigint(obj: PyObjectRef) -> BigInt {
    // `is_int` is true for a bool (`BOOL_TYPE`), so test `is_bool` first; a bool
    // reads through `w_bool_get_value`, not the int accessor.
    if is_bool(obj) {
        BigInt::from(w_bool_get_value(obj) as i64)
    } else if is_int(obj) {
        BigInt::from(w_int_get_value(obj))
    } else {
        w_long_get_value(obj).clone()
    }
}

/// Box a BigInt result, demoting to W_IntObject if it fits in i64.

fn bigint_result(value: BigInt) -> PyObjectRef {
    match value.to_i64() {
        Some(v) => w_int_new(v),
        None => w_long_new(value),
    }
}

#[majit_macros::elidable]
fn bigint_add(a: BigInt, b: BigInt) -> BigInt {
    a + b
}

#[majit_macros::elidable]
fn bigint_sub(a: BigInt, b: BigInt) -> BigInt {
    a - b
}

#[majit_macros::elidable]
fn bigint_mul(a: BigInt, b: BigInt) -> BigInt {
    a * b
}

#[majit_macros::elidable]
fn bigint_and(a: BigInt, b: BigInt) -> BigInt {
    a & b
}

#[majit_macros::elidable]
fn bigint_or(a: BigInt, b: BigInt) -> BigInt {
    a | b
}

#[majit_macros::elidable]
fn bigint_xor(a: BigInt, b: BigInt) -> BigInt {
    a ^ b
}

#[majit_macros::elidable]
fn bigint_lshift(a: BigInt, shift: usize) -> BigInt {
    a << shift
}

#[majit_macros::elidable]
fn bigint_rshift(a: BigInt, shift: usize) -> BigInt {
    a >> shift
}

#[majit_macros::elidable]
fn bigint_to_f64(a: BigInt) -> f64 {
    a.to_f64().unwrap_or(f64::INFINITY)
}

#[majit_macros::elidable]
fn float_copysign(mag: f64, sign: f64) -> f64 {
    if sign.is_sign_negative() {
        -mag.abs()
    } else {
        mag.abs()
    }
}

#[majit_macros::elidable]
fn bigint_neg(a: BigInt) -> BigInt {
    -a
}

#[majit_macros::elidable]
fn bigint_invert(a: BigInt) -> BigInt {
    !a
}

#[majit_macros::elidable]
pub(crate) fn bigint_eq(a: BigInt, b: BigInt) -> bool {
    a == b
}

#[majit_macros::elidable]
fn bigint_lt(a: BigInt, b: BigInt) -> bool {
    a < b
}

#[majit_macros::elidable]
fn bigint_gt(a: BigInt, b: BigInt) -> bool {
    a > b
}

#[majit_macros::elidable]
fn bigint_mod(a: BigInt, b: BigInt) -> BigInt {
    a % b
}

/// longobject.py:62-70 `_truediv` → rbigint.truediv parity.
/// Produces the correctly-rounded IEEE 754 double for a/b.
/// Port of CPython `Objects/longobject.c long_true_divide`.
#[majit_macros::elidable]
fn bigint_truediv(a: BigInt, b: BigInt) -> Result<f64, PyError> {
    use malachite_bigint::Sign;

    if b.sign() == Sign::NoSign {
        return Err(PyError::zero_division("division by zero"));
    }
    if a.sign() == Sign::NoSign {
        return Ok(0.0);
    }

    let negate = (a.sign() == Sign::Minus) != (b.sign() == Sign::Minus);
    let a_abs = if a.sign() == Sign::Minus { -a } else { a };
    let b_abs = if b.sign() == Sign::Minus { -b } else { b };

    let a_bits = a_abs.bits() as i64;
    let b_bits = b_abs.bits() as i64;

    // f64 exponent range: [-1022, 1023]. If a/b would exceed 2^1024, overflow.
    if a_bits - b_bits > 1024 {
        return Err(PyError::new(
            PyErrorKind::OverflowError,
            "integer division result too large for a float",
        ));
    }
    // If a/b would underflow to 0 (ratio < 2^-1075 where 1075 = 1022+53):
    if b_bits - a_bits > 1075 {
        return Ok(if negate { -0.0 } else { 0.0 });
    }

    // Shift a so that a_shifted / b has exactly 54 significant bits
    // (53 mantissa + 1 rounding bit).
    const MANT_DIG: i64 = 54; // DBL_MANT_DIG + 1
    let shift = MANT_DIG - a_bits + b_bits;
    let a_shifted = if shift >= 0 {
        a_abs << (shift as usize)
    } else {
        &a_abs >> ((-shift) as usize)
    };

    let (q, r) = a_shifted.div_rem(&b_abs);
    let mut q_bits = q.bits() as i64;

    // Adjust if quotient is one bit too large (55 bits instead of 54)
    let (q, r, extra_shift) = if q_bits == MANT_DIG + 1 {
        let q2 = &q >> 1usize;
        let r2 = &a_shifted - &q2 * &b_abs * BigInt::from(2);
        (q2, r2, 1i64)
    } else {
        (q, r, 0i64)
    };
    q_bits = q.bits() as i64;

    // Round-half-to-even using 2*r vs b comparison (correct for odd b
    // where b>>1 would lose the low bit).
    let r_abs = if r.sign() == Sign::Minus { -r } else { r };
    let two_r = &r_abs << 1usize;
    let round_up = if two_r > b_abs {
        true
    } else if two_r == b_abs {
        &q % BigInt::from(2) != BigInt::from(0)
    } else {
        false
    };
    let q_final = if round_up { q + BigInt::from(1) } else { q };

    let mantissa = match q_final.to_u64() {
        Some(v) => v,
        None => {
            return Err(PyError::new(
                PyErrorKind::OverflowError,
                "integer division result too large for a float",
            ));
        }
    };

    let exponent = a_bits - b_bits - MANT_DIG + 1 + extra_shift;
    let result = (mantissa as f64) * (2.0_f64).powi(exponent as i32);

    if result.is_infinite() {
        return Err(PyError::new(
            PyErrorKind::OverflowError,
            "integer division result too large for a float",
        ));
    }

    Ok(if negate { -result } else { result })
}

// ── Arithmetic operations ─────────────────────────────────────────────

/// Integer addition fast path.
///
/// The JIT will specialize this via:
///   GuardClass(a, &INT_TYPE)
///   GuardClass(b, &INT_TYPE)
///   GetfieldGcI(a, intval_offset) → va
///   GetfieldGcI(b, intval_offset) → vb
///   IntAdd(va, vb) → result
///   New(W_IntObject) + SetfieldGcI(result)

unsafe fn int_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    match va.checked_add(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(bigint_add(BigInt::from(va), BigInt::from(vb)))),
    }
}

unsafe fn int_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    match va.checked_sub(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(bigint_sub(BigInt::from(va), BigInt::from(vb)))),
    }
}

unsafe fn int_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    match va.checked_mul(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(bigint_mul(BigInt::from(va), BigInt::from(vb)))),
    }
}

unsafe fn int_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb == 0 {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    // Python floor division: rounds toward negative infinity.
    // i64::MIN / -1 overflows → fall back to BigInt.
    let q = match va.checked_div(vb) {
        Some(q) => q,
        None => return Ok(bigint_result(BigInt::from(va).div_floor(&BigInt::from(vb)))),
    };
    let r = va % vb;
    // Adjust: if remainder is nonzero and signs of operands differ, subtract 1.
    let q = if r != 0 && (r ^ vb) < 0 { q - 1 } else { q };
    Ok(w_int_new(q))
}

unsafe fn int_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb == 0 {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    // Python modulo: result has the same sign as the divisor.
    let r = va % vb;
    let r = if r != 0 && (r ^ vb) < 0 { r + vb } else { r };
    Ok(w_int_new(r))
}

// ── Long (BigInt) arithmetic operations ─────────────────────────────

unsafe fn long_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(bigint_add(as_bigint(a), as_bigint(b))))
}

unsafe fn long_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(bigint_sub(as_bigint(a), as_bigint(b))))
}

unsafe fn long_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(bigint_mul(as_bigint(a), as_bigint(b))))
}

unsafe fn long_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if bigint_eq(vb.clone(), BigInt::from(0)) {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    Ok(bigint_result(as_bigint(a).div_floor(&vb)))
}

unsafe fn long_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if bigint_eq(vb.clone(), BigInt::from(0)) {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    Ok(bigint_result(as_bigint(a).mod_floor(&vb)))
}

// ── Float arithmetic operations ──────────────────────────────────────

/// Coerce an operand to f64. Works for int, long, and float objects.
unsafe fn as_float(obj: PyObjectRef) -> f64 {
    if is_float(obj) {
        w_float_get_value(obj)
    } else if is_int(obj) {
        w_int_get_value(obj) as f64
    } else {
        // long → f64 (may lose precision for very large values)
        w_long_get_value(obj).to_f64().unwrap_or(f64::INFINITY)
    }
}

/// True if both operands are numeric and at least one is float.

unsafe fn is_float_pair(a: PyObjectRef, b: PyObjectRef) -> bool {
    let a_num = is_int(a) || is_float(a) || is_long(a);
    let b_num = is_int(b) || is_float(b) || is_long(b);
    a_num && b_num && (is_float(a) || is_float(b))
}

unsafe fn float_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_float_new(as_float(a) + as_float(b)))
}

unsafe fn float_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_float_new(as_float(a) - as_float(b)))
}

unsafe fn float_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_float_new(as_float(a) * as_float(b)))
}

unsafe fn float_truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_float(b);
    if vb == 0.0 {
        return Err(PyError::zero_division("float division by zero"));
    }
    Ok(w_float_new(as_float(a) / vb))
}

/// floatobject.py:508-512: descr_floordiv → _divmod_w()[0].
unsafe fn float_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (floordiv, _mod) = float_divmod_w(as_float(a), as_float(b))?;
    Ok(w_float_new(floordiv))
}

/// floatobject.py:520-540: descr_mod with math_fmod + sign correction.
unsafe fn float_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let x = as_float(a);
    let y = as_float(b);
    if y == 0.0 {
        // floatobject.py:526
        return Err(PyError::zero_division("float modulo"));
    }
    let mut m = x % y; // fmod
    if m != 0.0 {
        // floatobject.py:529-531: ensure remainder has same sign as denominator
        if (y < 0.0) != (m < 0.0) {
            m += y;
        }
    } else {
        // floatobject.py:536-538: signed zero — copysign(0.0, y)
        m = float_copysign(0.0, y);
    }
    Ok(w_float_new(m))
}

/// floatobject.py:758-793: _divmod_w.
fn float_divmod_w(x: f64, y: f64) -> Result<(f64, f64), PyError> {
    if y == 0.0 {
        // floatobject.py:761
        return Err(PyError::zero_division("float modulo"));
    }
    let mut m = x % y; // fmod
    // floatobject.py:767: div = (x - mod) / y
    let mut div = (x - m) / y;
    if m != 0.0 {
        // floatobject.py:769-771: sign correction
        if (y < 0.0) != (m < 0.0) {
            m += y;
            div -= 1.0;
        }
    } else {
        // floatobject.py:776-778: signed zero
        // "mod *= mod" hides "+0" from optimizer, then negate if y < 0
        m = m * m; // hide from optimizer
        if y < 0.0 {
            m = -m;
        }
    }
    // floatobject.py:784-790: snap quotient to nearest integral value
    let floordiv = if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 { f + 1.0 } else { f }
    } else {
        // floatobject.py:789-790: zero with sign of true quotient
        let d = div * div; // hide from optimizer
        d * x / y
    };
    Ok((floordiv, m))
}

// ── Power ────────────────────────────────────────────────────────────

unsafe fn int_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb < 0 {
        // intobject.py:415-419 _pow_nomod raises ValueError for iw < 0,
        // descr_pow catches it and routes through float pow — which
        // carries the ZeroDivisionError guard from floatobject.py:910-913.
        return Ok(w_float_new(float_pow_raw(va as f64, vb as f64)?));
    }
    // intobject.py:415 / longobject.py:229: x ** 0 == 1 for any x.
    if vb == 0 {
        return Ok(w_int_new(1));
    }
    // longobject.py:224-231: rbigint.pow handles arbitrary exponents.
    // Rust BigInt::pow takes u32; short-circuit trivial bases so that
    // e.g. `1 ** huge` returns 1 instead of MemoryError.
    match va {
        0 => return Ok(w_int_new(0)),
        1 => return Ok(w_int_new(1)),
        -1 => return Ok(w_int_new(if vb % 2 == 0 { 1 } else { -1 })),
        _ => {}
    }
    let vb = match u32::try_from(vb) {
        Ok(v) => v,
        Err(_) => return Err(PyError::memory_error("exponent too large")),
    };
    match va.checked_pow(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(BigInt::from(va).pow(vb))),
    }
}

unsafe fn long_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if bigint_lt(vb.clone(), BigInt::from(0)) {
        let fa = as_float(a);
        let fb = as_float(b);
        return Ok(w_float_new(float_pow_raw(fa, fb)?));
    }
    // longobject.py:229: `if not exp_bigint: return int_pow(0)` → 1.
    if vb.sign() == malachite_bigint::Sign::NoSign {
        return Ok(w_int_new(1));
    }
    // longobject.py:224-231: rbigint.pow handles arbitrary exponents.
    // Short-circuit trivial bases before the u32 narrowing so that
    // 1 ** huge, (-1) ** huge, 0 ** huge succeed.
    let va = as_bigint(a);
    if va.sign() == malachite_bigint::Sign::NoSign {
        return Ok(w_int_new(0));
    }
    if va == BigInt::from(1) {
        return Ok(w_int_new(1));
    }
    if va == BigInt::from(-1) {
        let even = vb.clone() % BigInt::from(2) == BigInt::from(0);
        return Ok(w_int_new(if even { 1 } else { -1 }));
    }
    let exp = match vb.to_u32() {
        Some(v) => v,
        None => return Err(PyError::memory_error("exponent too large")),
    };
    Ok(bigint_result(va.pow(exp)))
}

// ── Shift operations ─────────────────────────────────────────────────

unsafe fn int_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    // `i64::checked_shl` only fails when the shift amount is >= 64, so it
    // happily returns a wrapped result when the VALUE overflows (e.g.
    // `(10**18) << 4`). Detect real value overflow by computing the shift
    // in BigInt and demoting to i64 only when the result fits.
    let big = bigint_lshift(BigInt::from(va), vb as usize);
    Ok(bigint_result(big))
}

unsafe fn int_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // intobject.py:393-403 `_rshift(space, a, b)`:
    //   if r_uint(b) >= LONG_BIT:
    //       if b < 0: raise ValueError("negative shift count")
    //       # b >= LONG_BIT
    //       if a == 0: return wrapint(space, a)
    //       a = -1 if a < 0 else 0
    //   else: a = a >> b
    let va = int_value(a);
    let vb = int_value(b);
    if vb < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    if vb >= 64 {
        return Ok(w_int_new(if va < 0 { -1 } else { 0 }));
    }
    Ok(w_int_new(va >> vb))
}

unsafe fn long_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if bigint_lt(vb.clone(), BigInt::from(0)) {
        return Err(PyError::value_error("negative shift count"));
    }
    // longobject.py:375-380: shift overflows → 0 if base is zero,
    // OverflowError otherwise.
    let shift = match vb.to_usize() {
        Some(v) => v,
        None => {
            let va = as_bigint(a);
            if va.sign() == malachite_bigint::Sign::NoSign {
                return Ok(w_int_new(0));
            }
            return Err(PyError::overflow_error("shift count too large"));
        }
    };
    Ok(bigint_result(bigint_lshift(as_bigint(a), shift)))
}

unsafe fn long_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if bigint_lt(vb.clone(), BigInt::from(0)) {
        return Err(PyError::value_error("negative shift count"));
    }
    // longobject.py:393-397: shift overflows → positive yields 0,
    // negative yields -1 (all bits shifted out).
    let shift = match vb.to_usize() {
        Some(v) => v,
        None => {
            let va = as_bigint(a);
            return Ok(w_int_new(if va.sign() == malachite_bigint::Sign::Minus {
                -1
            } else {
                0
            }));
        }
    };
    Ok(bigint_result(bigint_rshift(as_bigint(a), shift)))
}

// ── bool-as-int helpers ──────────────────────────────────────────────

/// True when obj is int or bool (bool is a subclass of int in Python).
#[inline]
pub(crate) unsafe fn is_int_like(obj: PyObjectRef) -> bool {
    is_int(obj) || is_bool(obj)
}

/// Extract i64 from an int or bool object.
#[inline]
pub(crate) unsafe fn int_value(obj: PyObjectRef) -> i64 {
    if is_bool(obj) {
        w_bool_get_value(obj) as i64
    } else {
        w_int_get_value(obj)
    }
}

// ── Bitwise operations ───────────────────────────────────────────────

// W_IntObject.descr_and/or/xor — always int; the bool result is produced
// by W_BoolObject's own descr_and/or/xor, not by the int path.
unsafe fn int_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(int_value(a) & int_value(b)))
}

unsafe fn int_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(int_value(a) | int_value(b)))
}

unsafe fn int_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(int_value(a) ^ int_value(b)))
}

unsafe fn long_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(bigint_and(as_bigint(a), as_bigint(b))))
}

unsafe fn long_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(bigint_or(as_bigint(a), as_bigint(b))))
}

unsafe fn long_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(bigint_xor(as_bigint(a), as_bigint(b))))
}

// ── String operations ────────────────────────────────────────────────

/// Concatenate two str objects.

pub(crate) unsafe fn str_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // Read the surrogate-aware WTF-8 view so concatenating a
    // surrogateescape/surrogatepass-decoded string does not go through
    // `w_str_get_value` (which rejects lone surrogates).
    let sa = w_str_get_wtf8(a);
    let sb = w_str_get_wtf8(b);
    let mut result = Wtf8Buf::with_capacity(sa.len() + sb.len());
    result.push_wtf8(sa);
    result.push_wtf8(sb);
    Ok(w_str_from_wtf8(result))
}

/// Extract a non-negative repeat count from an int or long, raising
/// OverflowError with `msg` for positive bigints that exceed usize.
unsafe fn repeat_count(n: PyObjectRef, msg: &str) -> Result<usize, PyError> {
    if is_long(n) {
        let big = as_bigint(n);
        match big.to_usize() {
            Some(v) => Ok(v),
            None if bigint_lt(big, BigInt::from(0)) => Ok(0),
            None => Err(PyError::new(PyErrorKind::OverflowError, msg)),
        }
    } else {
        let nv = w_int_get_value(n);
        Ok(if nv < 0 { 0 } else { nv as usize })
    }
}

/// unicodeobject.py:619-621 descr_mul
pub(crate) unsafe fn str_repeat(s: PyObjectRef, n: PyObjectRef) -> PyResult {
    // Repeat at the WTF-8 byte level — a repetition of valid WTF-8 is valid
    // WTF-8 — so a surrogate-bearing string repeats without going through
    // `w_str_get_value`.
    let bytes = w_str_get_wtf8(s).as_bytes();
    let count = repeat_count(n, "new string is too long")?;
    let total = bytes
        .len()
        .checked_mul(count)
        .ok_or_else(|| PyError::new(PyErrorKind::OverflowError, "new string is too long"))?;
    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(total)
        .map_err(|_| PyError::new(PyErrorKind::MemoryError, ""))?;
    for _ in 0..count {
        out.extend_from_slice(bytes);
    }
    let buf = Wtf8Buf::from_bytes(out).expect("repetition of WTF-8 is WTF-8");
    Ok(w_str_from_wtf8(buf))
}

pub(crate) unsafe fn list_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let len_a = w_list_len(a);
    let len_b = w_list_len(b);
    let mut items = Vec::with_capacity(len_a + len_b);
    for i in 0..len_a {
        if let Some(item) = w_list_getitem(a, i as i64) {
            items.push(item);
        }
    }
    for i in 0..len_b {
        if let Some(item) = w_list_getitem(b, i as i64) {
            items.push(item);
        }
    }
    Ok(w_list_new(items))
}

pub(crate) unsafe fn tuple_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let len_a = w_tuple_len(a);
    let len_b = w_tuple_len(b);
    let mut items = Vec::with_capacity(len_a + len_b);
    for i in 0..len_a {
        if let Some(item) = w_tuple_getitem(a, i as i64) {
            items.push(item);
        }
    }
    for i in 0..len_b {
        if let Some(item) = w_tuple_getitem(b, i as i64) {
            items.push(item);
        }
    }
    Ok(w_tuple_new(items))
}

/// listobject.py:638-641 descr_mul
pub(crate) unsafe fn list_repeat(list: PyObjectRef, n: PyObjectRef) -> PyResult {
    let count = repeat_count(n, "list is too large")?;
    let len = w_list_len(list);
    let cap = len
        .checked_mul(count)
        .ok_or_else(|| PyError::new(PyErrorKind::OverflowError, "list is too large"))?;
    let mut items: Vec<PyObjectRef> = Vec::new();
    items
        .try_reserve_exact(cap)
        .map_err(|_| PyError::new(PyErrorKind::MemoryError, ""))?;
    for _ in 0..count {
        for i in 0..len {
            if let Some(item) = w_list_getitem(list, i as i64) {
                items.push(item);
            }
        }
    }
    Ok(w_list_new(items))
}

// ── Comparison operations ─────────────────────────────────────────────

unsafe fn int_lt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) < int_value(b)))
}

unsafe fn int_le(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) <= int_value(b)))
}

unsafe fn int_gt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) > int_value(b)))
}

unsafe fn int_ge(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) >= int_value(b)))
}

unsafe fn int_eq(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) == int_value(b)))
}

unsafe fn int_ne(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) != int_value(b)))
}

unsafe fn float_lt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) < as_float(b)))
}

unsafe fn float_le(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) <= as_float(b)))
}

unsafe fn float_gt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) > as_float(b)))
}

unsafe fn float_ge(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) >= as_float(b)))
}

unsafe fn float_eq(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) == as_float(b)))
}

unsafe fn float_ne(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(as_float(a) != as_float(b)))
}

// ── Public dispatch API ───────────────────────────────────────────────

/// Try to call a dunder method on an instance for binary ops.
///
/// PyPy: descroperation.py `_binop_impl` →
///   1. Try `a.__op__(b)` (forward)
///   2. If not found or returns NotImplemented, try `b.__rop__(a)` (reverse)
///
/// descroperation.py:432-437 parity: `space.get_and_call_function` raises
/// OperationError; the NotImplemented return value alone triggers the
/// fallback. We mirror that by propagating PyError immediately — the
/// pyre `call_function` shim stashes exceptions in PENDING_CALL_ERROR
/// so we must consume them via `call_function_impl_result` to match.
unsafe fn try_instance_binop(a: PyObjectRef, b: PyObjectRef, dunder: &str) -> Option<PyResult> {
    let a_is_inst = is_instance(a);
    let b_is_inst = is_instance(b);

    // PyPy: descroperation.py _binop_impl
    // If b's type is a proper subtype of a's type, try reverse first.
    // This matches Python's "subclass reflected op takes priority" rule.
    let try_reverse_first = if a_is_inst && b_is_inst {
        if let Some(rdunder) = reverse_dunder(dunder) {
            let a_type = w_instance_get_type(a);
            let b_type = w_instance_get_type(b);
            !std::ptr::eq(a_type, b_type)
                && issubtype_cached(b_type, a_type)
                && lookup_in_type_where(b_type, rdunder).is_some()
        } else {
            false
        }
    } else {
        false
    };

    if try_reverse_first {
        let rdunder = reverse_dunder(dunder).unwrap();
        let w_type = w_instance_get_type(b);
        if let Some(method) = lookup_in_type_where(w_type, rdunder) {
            match crate::call::call_function_impl_result(method, &[b, a]) {
                Ok(result) => {
                    if !is_not_implemented(result) {
                        return Some(Ok(result));
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }

    // Forward: a.__op__(b)
    if a_is_inst {
        let w_type = w_instance_get_type(a);
        if let Some(method) = lookup_in_type_where(w_type, dunder) {
            match crate::call::call_function_impl_result(method, &[a, b]) {
                Ok(result) => {
                    if !is_not_implemented(result) {
                        return Some(Ok(result));
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }

    // Reverse: b.__rop__(a) — only if not already tried above
    if !try_reverse_first && b_is_inst {
        if let Some(rdunder) = reverse_dunder(dunder) {
            let w_type = w_instance_get_type(b);
            if let Some(method) = lookup_in_type_where(w_type, rdunder) {
                match crate::call::call_function_impl_result(method, &[b, a]) {
                    Ok(result) => {
                        if !is_not_implemented(result) {
                            return Some(Ok(result));
                        }
                    }
                    Err(e) => return Some(Err(e)),
                }
            }
        }
    }

    None
}

/// `descroperation.py _binop_impl` typedef-driven fallback for
/// non-instance LHS / RHS — pyre's `try_instance_binop` only fires
/// when at least one operand is `is_instance`, but built-in W_Root
/// types (dict_view, exception, generator, …) also expose dunder
/// methods through their typedef.  This helper does the same
/// forward + reverse MRO lookup but routes through
/// `crate::typedef::r#type` instead of `w_instance_get_type`, so
/// `dict_keys() | set()` etc. find the typedef-installed `__or__`
/// and friends.  Returns `None` when neither side defines the
/// method (caller falls through to the existing TypeError path).
unsafe fn try_typedef_binop(a: PyObjectRef, b: PyObjectRef, dunder: &str) -> Option<PyResult> {
    if let Some(a_type) = crate::typedef::r#type(a) {
        if let Some(method) = lookup_in_type_where(a_type, dunder) {
            match crate::call::call_function_impl_result(method, &[a, b]) {
                Ok(result) => {
                    if !is_not_implemented(result) {
                        return Some(Ok(result));
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
    if let Some(rdunder) = reverse_dunder(dunder) {
        if let Some(b_type) = crate::typedef::r#type(b) {
            if let Some(method) = lookup_in_type_where(b_type, rdunder) {
                match crate::call::call_function_impl_result(method, &[b, a]) {
                    Ok(result) => {
                        if !is_not_implemented(result) {
                            return Some(Ok(result));
                        }
                    }
                    Err(e) => return Some(Err(e)),
                }
            }
        }
    }
    None
}

/// Check if w_type is a subtype of cls using cached MRO.
unsafe fn issubtype_cached(w_type: PyObjectRef, cls: PyObjectRef) -> bool {
    let mro_ptr = w_type_get_mro(w_type);
    if !mro_ptr.is_null() {
        return (*mro_ptr).iter().any(|&t| std::ptr::eq(t, cls));
    }
    false
}

/// Map forward dunder to reverse dunder.
/// PyPy: descroperation.py `_make_binop_impl` generates both directions.
fn reverse_dunder(dunder: &str) -> Option<&'static str> {
    Some(match dunder {
        // Arithmetic — PyPy: descroperation.py _make_binop_impl
        "__add__" => "__radd__",
        "__sub__" => "__rsub__",
        "__mul__" => "__rmul__",
        "__truediv__" => "__rtruediv__",
        "__floordiv__" => "__rfloordiv__",
        "__mod__" => "__rmod__",
        "__pow__" => "__rpow__",
        "__lshift__" => "__rlshift__",
        "__rshift__" => "__rrshift__",
        "__and__" => "__rand__",
        "__or__" => "__ror__",
        "__xor__" => "__rxor__",
        // Comparison reflected — PyPy: descroperation.py _cmp_dispatch
        "__lt__" => "__gt__",
        "__le__" => "__ge__",
        "__gt__" => "__lt__",
        "__ge__" => "__le__",
        "__eq__" => "__eq__",
        "__ne__" => "__ne__",
        _ => return None,
    })
}

/// Try to call a unary dunder on an instance.
///
/// PyPy: `ObjSpace.call_function(space.lookup(w_obj, dunder), w_obj)`
/// The Python-level OperationError must propagate to the caller; use the
/// Result-returning call path so PENDING_CALL_ERROR is consumed.
unsafe fn try_instance_unaryop(a: PyObjectRef, dunder: &str) -> Option<PyResult> {
    if is_instance(a) {
        if let Some(method) = lookup(a, dunder) {
            return Some(crate::call::call_function_impl_result(method, &[a]));
        }
    }
    None
}

/// True when `obj`'s type defines `dunder` in a class other than the
/// builtin type object `tp` — i.e. a subclass overrides it.  Builtin
/// `str`/`list`/`tuple` install `__add__`/`__radd__` on their own type;
/// an inherited (non-overridden) lookup resolves back to `tp`.
unsafe fn dunder_overridden(obj: PyObjectRef, dunder: &str, tp: PyObjectRef) -> bool {
    match crate::typedef::r#type(obj).and_then(|t| lookup_where_with_method_cache(t, dunder)) {
        Some((src, _)) => !std::ptr::eq(src, tp),
        None => false,
    }
}

/// Builtin sequence base selected by [`needs_seq_binop_dispatch`].  The
/// caller passes this discriminant instead of a `&STR_TYPE`/… static so
/// the type-static load stays inside the residual helper, off the traced
/// `add` graph.
#[derive(Clone, Copy)]
enum SeqBase {
    Str,
    List,
    Tuple,
}

/// descroperation.py:708 `binop_impl` shortcut — the builtin sequence
/// fast path (`str`/`list`/`tuple` concat) bypasses `__op__`/`__rop__`
/// dispatch unless one operand is a subclass that actually overrides the
/// forward or reflected special method (descroperation.py:664 "unicode +
/// string subclass").  Returns `false` when no override exists so the
/// caller concatenates directly — this also avoids re-entering the
/// builtin `__add__` slot, which would recurse back into `add`.
///
/// `dont_look_inside`: the builtin-base type static is loaded here, so a
/// traced caller emits a residual call instead of carrying an
/// unresolvable `LoadStatic` into its JitCode.
#[majit_macros::dont_look_inside]
unsafe fn needs_seq_binop_dispatch(
    a: PyObjectRef,
    b: PyObjectRef,
    base: SeqBase,
    fwd: &str,
    rev: &str,
) -> bool {
    let tp: *const pyre_object::PyType = match base {
        SeqBase::Str => &pyre_object::STR_TYPE,
        SeqBase::List => &pyre_object::LIST_TYPE,
        SeqBase::Tuple => &pyre_object::TUPLE_TYPE,
    };
    let Some(t) = crate::typedef::gettypefor(tp) else {
        return false;
    };
    dunder_overridden(a, fwd, t)
        || dunder_overridden(a, rev, t)
        || dunder_overridden(b, fwd, t)
        || dunder_overridden(b, rev, t)
}

/// `bytes`/`bytearray` analog of `needs_seq_binop_dispatch`.  The two
/// builtin types share one `+` branch, so each operand is judged against
/// its own builtin base (`bytes` vs `bytearray`).  `dont_look_inside`
/// keeps the base type-static loads in the residual call, off the traced
/// `add` graph.
#[majit_macros::dont_look_inside]
unsafe fn needs_bytes_binop_dispatch(a: PyObjectRef, b: PyObjectRef, fwd: &str, rev: &str) -> bool {
    bytes_operand_overrides(a, fwd, rev) || bytes_operand_overrides(b, fwd, rev)
}

/// True when `obj`'s type overrides `fwd`/`rev` relative to its builtin
/// base (`bytes` or `bytearray`).  Only reached from the residual
/// `needs_bytes_binop_dispatch`, so the type-static loads never enter a
/// traced graph.
unsafe fn bytes_operand_overrides(obj: PyObjectRef, fwd: &str, rev: &str) -> bool {
    let tp: *const pyre_object::PyType = if pyre_object::bytesobject::is_bytes(obj) {
        &pyre_object::bytesobject::BYTES_TYPE
    } else {
        &pyre_object::bytearrayobject::BYTEARRAY_TYPE
    };
    let Some(t) = crate::typedef::gettypefor(tp) else {
        return false;
    };
    dunder_overridden(obj, fwd, t) || dunder_overridden(obj, rev, t)
}

/// The builtin numeric base type backing `obj`'s storage — `int` for
/// int/long, `float` for float, `None` for a non-numeric operand.
unsafe fn numeric_base_type(obj: PyObjectRef) -> Option<*const pyre_object::PyType> {
    if is_int(obj) || is_long(obj) {
        Some(&pyre_object::INT_TYPE)
    } else if is_float(obj) {
        Some(&pyre_object::FLOAT_TYPE)
    } else {
        None
    }
}

/// True when numeric operand `obj` (int/long/float storage) has a Python
/// class that overrides `dunder` relative to its builtin base — int/long
/// against `int`, float against `float`.  Mirrors [`dunder_overridden`]
/// with the numeric base selected per operand.  Only an *overriding*
/// subclass routes through dispatch: a non-overriding subclass keeps the
/// Rust fast path, which both matches the builtin result and avoids
/// re-entering the inherited slot (it would recurse back into this op).
unsafe fn numeric_operand_overrides(obj: PyObjectRef, dunder: &str, rdunder: &str) -> bool {
    let Some(base) = numeric_base_type(obj) else {
        return false;
    };
    let Some(t) = crate::typedef::gettypefor(base) else {
        return false;
    };
    dunder_overridden(obj, dunder, t) || dunder_overridden(obj, rdunder, t)
}

/// descroperation.py:708 `binop_impl` shortcut — the builtin numeric
/// (int/long/float) fast path bypasses `__op__`/`__rop__` dispatch unless
/// an operand is a subclass that actually overrides the forward or
/// reflected special method.  The seq/bytes analogs are
/// [`needs_seq_binop_dispatch`]/[`needs_bytes_binop_dispatch`].
///
/// `dont_look_inside`: the type-static + typeobject-registry loads stay in
/// this residual helper, off the traced numeric graph; the hot int path is
/// specialized separately via `guard_class` in the JIT.
#[majit_macros::dont_look_inside]
unsafe fn needs_numeric_binop_dispatch(
    a: PyObjectRef,
    b: PyObjectRef,
    fwd: &str,
    rev: &str,
) -> bool {
    numeric_operand_overrides(a, fwd, rev) || numeric_operand_overrides(b, fwd, rev)
}

/// Unary analog: true when numeric operand `a` overrides the unary
/// special `dunder` relative to its builtin base.
#[majit_macros::dont_look_inside]
unsafe fn needs_numeric_unaryop_dispatch(a: PyObjectRef, dunder: &str) -> bool {
    let Some(base) = numeric_base_type(a) else {
        return false;
    };
    let Some(t) = crate::typedef::gettypefor(base) else {
        return false;
    };
    dunder_overridden(a, dunder, t)
}

/// Call the overriding unary special on a numeric subclass operand before
/// the Rust fast path.  Returns `None` when `a` is an exact builtin
/// numeric or does not override `dunder`, so the caller falls through.
unsafe fn try_numeric_unaryop_override(a: PyObjectRef, dunder: &str) -> Option<PyResult> {
    if !needs_numeric_unaryop_dispatch(a, dunder) {
        return None;
    }
    let t = crate::typedef::r#type(a)?;
    let method = lookup_in_type_where(t, dunder)?;
    Some(crate::call::call_function_impl_result(method, &[a]))
}

pub fn add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if needs_numeric_binop_dispatch(a, b, "__add__", "__radd__") {
            if let Some(result) = try_dispatch_binary_special(a, b, "__add__", "__radd__")? {
                return Ok(result);
            }
        }
        if is_int_like(a) && is_int_like(b) {
            return int_add(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_add(a, b);
        }
        if is_float_pair(a, b) {
            return float_add(a, b);
        }
        if is_str(a) && is_str(b) {
            // descroperation.py:664 "unicode + string subclass" — a str
            // subclass overriding `__add__`/`__radd__` must reach the
            // reflected dispatch; otherwise concat directly.
            if needs_seq_binop_dispatch(a, b, SeqBase::Str, "__add__", "__radd__") {
                if let Some(result) = try_dispatch_binary_special(a, b, "__add__", "__radd__")? {
                    return Ok(result);
                }
            }
            return str_concat(a, b);
        }
        if is_list(a) && is_list(b) {
            if needs_seq_binop_dispatch(a, b, SeqBase::List, "__add__", "__radd__") {
                if let Some(result) = try_dispatch_binary_special(a, b, "__add__", "__radd__")? {
                    return Ok(result);
                }
            }
            return list_concat(a, b);
        }
        if is_tuple(a) && is_tuple(b) {
            if needs_seq_binop_dispatch(a, b, SeqBase::Tuple, "__add__", "__radd__") {
                if let Some(result) = try_dispatch_binary_special(a, b, "__add__", "__radd__")? {
                    return Ok(result);
                }
            }
            return tuple_concat(a, b);
        }
        if pyre_object::bytesobject::is_bytes_like(a) && pyre_object::bytesobject::is_bytes_like(b)
        {
            if needs_bytes_binop_dispatch(a, b, "__add__", "__radd__") {
                if let Some(result) = try_dispatch_binary_special(a, b, "__add__", "__radd__")? {
                    return Ok(result);
                }
            }
            let a_data = pyre_object::bytesobject::bytes_like_data(a);
            let b_data = pyre_object::bytesobject::bytes_like_data(b);
            let mut result = a_data.to_vec();
            result.extend_from_slice(b_data);
            // bytes + bytes → bytes; anything with bytearray → bytearray
            return Ok(
                if pyre_object::bytesobject::is_bytes(a) && pyre_object::bytesobject::is_bytes(b) {
                    pyre_object::bytesobject::w_bytes_from_bytes(&result)
                } else {
                    pyre_object::bytearrayobject::w_bytearray_from_bytes(&result)
                },
            );
        }
        // Forward `__add__` + reflected `__radd__` per
        // `descroperation.py:_make_binop_impl` — try_dispatch_binary_special
        // already implements the reflected-first reordering rule for
        // subclass operands.  Fall back to the legacy instance-only
        // `__add__` path when neither side is a typedef-bearing class.
        if let Some(result) = try_dispatch_binary_special(a, b, "__add__", "__radd__")? {
            return Ok(result);
        }
        if let Some(result) = try_instance_binop(a, b, "__add__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        // Sequence concatenation slot (sq_concat) reports a distinct
        // message when the left operand is a sequence — `unicode_concatenate`
        // / `list_concat` / `tuple_concat`.
        if is_str(a) || is_list(a) || is_tuple(a) {
            return Err(PyError::type_error(format!(
                "can only concatenate {a_name} (not \"{b_name}\") to {a_name}"
            )));
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for +: '{}' and '{}'",
            a_name, b_name,
        )))
    }
}

pub fn sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if needs_numeric_binop_dispatch(a, b, "__sub__", "__rsub__") {
            if let Some(result) = try_dispatch_binary_special(a, b, "__sub__", "__rsub__")? {
                return Ok(result);
            }
        }
        if is_int_like(a) && is_int_like(b) {
            return int_sub(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_sub(a, b);
        }
        if is_float_pair(a, b) {
            return float_sub(a, b);
        }
        // set / frozenset difference — PyPy: setobject.py W_BaseSetObject.descr_sub.
        // descr_sub returns NotImplemented for a non-set rhs, so `-` requires
        // both operands to be sets (the `difference` method takes iterables).
        if pyre_object::is_set_or_frozenset(a) && pyre_object::is_set_or_frozenset(b) {
            let other_items = crate::builtins::collect_iterable(b)?;
            let probe = pyre_object::w_set_from_items(&other_items);
            let result: Vec<PyObjectRef> = pyre_object::w_set_items(a)
                .into_iter()
                .filter(|&item| !pyre_object::w_set_contains(probe, item))
                .collect();
            return Ok(if pyre_object::is_frozenset(a) {
                pyre_object::w_frozenset_from_items(&result)
            } else {
                pyre_object::w_set_from_items(&result)
            });
        }
        if let Some(result) = try_dispatch_binary_special(a, b, "__sub__", "__rsub__")? {
            return Ok(result);
        }
        if let Some(result) = try_instance_binop(a, b, "__sub__") {
            return result;
        }
        // Built-in W_Root types (dict_view, …) expose `__sub__` /
        // `__rsub__` through their typedef.
        if let Some(result) = try_typedef_binop(a, b, "__sub__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for -: '{a_name}' and '{b_name}'"
        )))
    }
}

pub fn mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if needs_numeric_binop_dispatch(a, b, "__mul__", "__rmul__") {
            if let Some(result) = try_dispatch_binary_special(a, b, "__mul__", "__rmul__")? {
                return Ok(result);
            }
        }
        if is_int_like(a) && is_int_like(b) {
            return int_mul(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_mul(a, b);
        }
        if is_float_pair(a, b) {
            return float_mul(a, b);
        }
        if is_str(a) && is_int_or_long(b) {
            return str_repeat(a, b);
        }
        if is_int_or_long(a) && is_str(b) {
            return str_repeat(b, a);
        }
        if is_list(a) && is_int_or_long(b) {
            return list_repeat(a, b);
        }
        if is_int_or_long(a) && is_list(b) {
            return list_repeat(b, a);
        }
        // tupleobject.py descr_mul
        if is_tuple(a) && is_int_or_long(b) {
            let n = repeat_count(b, "tuple is too large")?;
            let len = w_tuple_len(a);
            let cap = len
                .checked_mul(n)
                .ok_or_else(|| PyError::new(PyErrorKind::OverflowError, "tuple is too large"))?;
            let mut items: Vec<PyObjectRef> = Vec::new();
            items
                .try_reserve_exact(cap)
                .map_err(|_| PyError::new(PyErrorKind::MemoryError, ""))?;
            for _ in 0..n {
                for i in 0..len {
                    if let Some(item) = w_tuple_getitem(a, i as i64) {
                        items.push(item);
                    }
                }
            }
            return Ok(w_tuple_new(items));
        }
        if is_int_or_long(a) && is_tuple(b) {
            return mul(b, a);
        }
        // bytesobject.py descr_mul / bytearrayobject.py descr_mul
        if pyre_object::bytesobject::is_bytes_like(a) && is_int_or_long(b) {
            let data = pyre_object::bytesobject::bytes_like_data(a);
            let n = repeat_count(b, "repeated bytes are too long")?;
            let cap = data.len().checked_mul(n).ok_or_else(|| {
                PyError::new(PyErrorKind::OverflowError, "repeated bytes are too long")
            })?;
            let mut buf: Vec<u8> = Vec::new();
            buf.try_reserve_exact(cap)
                .map_err(|_| PyError::new(PyErrorKind::MemoryError, ""))?;
            for _ in 0..n {
                buf.extend_from_slice(data);
            }
            return Ok(if pyre_object::bytesobject::is_bytes(a) {
                pyre_object::bytesobject::w_bytes_from_bytes(&buf)
            } else {
                pyre_object::bytearrayobject::w_bytearray_from_bytes(&buf)
            });
        }
        if is_int_or_long(a) && pyre_object::bytesobject::is_bytes_like(b) {
            return mul(b, a);
        }
        if let Some(result) = try_dispatch_binary_special(a, b, "__mul__", "__rmul__")? {
            return Ok(result);
        }
        if let Some(result) = try_instance_binop(a, b, "__mul__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        // Sequence repetition slot (sq_repeat): a sequence on either side
        // with a non-int multiplier reports the non-int's type.
        let a_seq =
            is_str(a) || is_list(a) || is_tuple(a) || pyre_object::bytesobject::is_bytes_like(a);
        let b_seq =
            is_str(b) || is_list(b) || is_tuple(b) || pyre_object::bytesobject::is_bytes_like(b);
        if a_seq {
            return Err(PyError::type_error(format!(
                "can't multiply sequence by non-int of type '{b_name}'"
            )));
        }
        if b_seq {
            return Err(PyError::type_error(format!(
                "can't multiply sequence by non-int of type '{a_name}'"
            )));
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for *: '{a_name}' and '{b_name}'"
        )))
    }
}

pub fn floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if needs_numeric_binop_dispatch(a, b, "__floordiv__", "__rfloordiv__") {
            if let Some(result) =
                try_dispatch_binary_special(a, b, "__floordiv__", "__rfloordiv__")?
            {
                return Ok(result);
            }
        }
        if is_int_like(a) && is_int_like(b) {
            return int_floordiv(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_floordiv(a, b);
        }
        if is_float_pair(a, b) {
            return float_floordiv(a, b);
        }
        if let Some(result) = try_dispatch_binary_special(a, b, "__floordiv__", "__rfloordiv__")? {
            return Ok(result);
        }
        if let Some(result) = try_instance_binop(a, b, "__floordiv__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for //: '{a_name}' and '{b_name}'"
        )))
    }
}

pub fn mod_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if needs_numeric_binop_dispatch(a, b, "__mod__", "__rmod__") {
            if let Some(result) = try_dispatch_binary_special(a, b, "__mod__", "__rmod__")? {
                return Ok(result);
            }
        }
        if is_int_like(a) && is_int_like(b) {
            return int_mod(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_mod(a, b);
        }
        if is_float_pair(a, b) {
            return float_mod(a, b);
        }
        // str % args — PyPy: unicodeobject.py mod__String_ANY
        if is_str(a) {
            return crate::objspace::std::formatting::str_format_percent(a, b);
        }
        if let Some(result) = try_dispatch_binary_special(a, b, "__mod__", "__rmod__")? {
            return Ok(result);
        }
        if let Some(result) = try_instance_binop(a, b, "__mod__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for %: '{a_name}' and '{b_name}'"
        )))
    }
}

/// True division (`/` operator) — always produces a float result.
///
/// intobject.py:332-345 `_truediv` raises "division by zero" for int/int;
/// floatobject.py:519 `_floatdiv` raises "float division by zero" once
/// any operand is a float.
/// longobject.py:62-70 `_truediv` catches OverflowError from
/// `rbigint.truediv` and reissues it as
/// "integer division result too large for a float".
pub fn truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if needs_numeric_binop_dispatch(a, b, "__truediv__", "__rtruediv__") {
            if let Some(result) = try_dispatch_binary_special(a, b, "__truediv__", "__rtruediv__")?
            {
                return Ok(result);
            }
        }
        let a_num = is_int(a) || is_float(a) || is_long(a);
        let b_num = is_int(b) || is_float(b) || is_long(b);
        if a_num && b_num {
            if is_float(a) || is_float(b) {
                return float_truediv(a, b);
            }
            if !is_long(b) && as_float(b) == 0.0 {
                return Err(PyError::zero_division("division by zero"));
            }
            if is_long(a) || is_long(b) {
                let r = bigint_truediv(as_bigint(a), as_bigint(b))?;
                return Ok(w_float_new(r));
            }
            return Ok(w_float_new(as_float(a) / as_float(b)));
        }
        if let Some(result) = try_dispatch_binary_special(a, b, "__truediv__", "__rtruediv__")? {
            return Ok(result);
        }
        if let Some(result) = try_instance_binop(a, b, "__truediv__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for /: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Power operation dispatch (`**` operator).

pub fn pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if needs_numeric_binop_dispatch(a, b, "__pow__", "__rpow__") {
            if let Some(result) = try_dispatch_binary_special(a, b, "__pow__", "__rpow__")? {
                return Ok(result);
            }
        }
        if is_int_like(a) && is_int_like(b) {
            return int_pow(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_pow(a, b);
        }
        if is_float_pair(a, b) {
            return float_pow_impl(as_float(a), as_float(b));
        }
        if let Some(result) = try_dispatch_binary_special(a, b, "__pow__", "__rpow__")? {
            return Ok(result);
        }
        if let Some(result) = try_instance_binop(a, b, "__pow__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for **: '{a_name}' and '{b_name}'"
        )))
    }
}

// ── Numeric type-slot builtins ────────────────────────────────────────
//
// The `int`/`float` numeric special methods (`int.__add__`,
// `float.__and__`, …) resolve to these concrete computations, not to the
// operator dispatch above.  The operator (`add`, `and_`, …) drives the
// forward+reflected protocol and, when an operand is a numeric subclass
// that overrides the special method, re-dispatches through that operand's
// type slot.  Wiring the slot back to the operator would re-enter it and
// recurse without bound; wiring the slot to the concrete computation
// terminates after computing the result or returning NotImplemented for
// the reflected method to handle.  The `is_int_or_long`/`is_float` macro
// guards in typedef.rs pre-filter the operand kinds, so the trailing
// NotImplemented is reached only defensively.

pub(crate) fn add_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_add(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_add(a, b);
        }
        if is_float_pair(a, b) {
            return float_add(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn sub_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_sub(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_sub(a, b);
        }
        if is_float_pair(a, b) {
            return float_sub(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn mul_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_mul(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_mul(a, b);
        }
        if is_float_pair(a, b) {
            return float_mul(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn truediv_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        let a_num = is_int(a) || is_float(a) || is_long(a);
        let b_num = is_int(b) || is_float(b) || is_long(b);
        if a_num && b_num {
            if is_float(a) || is_float(b) {
                return float_truediv(a, b);
            }
            if !is_long(b) && as_float(b) == 0.0 {
                return Err(PyError::zero_division("division by zero"));
            }
            if is_long(a) || is_long(b) {
                let r = bigint_truediv(as_bigint(a), as_bigint(b))?;
                return Ok(w_float_new(r));
            }
            return Ok(w_float_new(as_float(a) / as_float(b)));
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn floordiv_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_floordiv(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_floordiv(a, b);
        }
        if is_float_pair(a, b) {
            return float_floordiv(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn mod_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_mod(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_mod(a, b);
        }
        if is_float_pair(a, b) {
            return float_mod(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn pow_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_pow(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_pow(a, b);
        }
        if is_float_pair(a, b) {
            return float_pow_impl(as_float(a), as_float(b));
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn divmod_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        let lhs_num = is_int(a) || is_long(a) || is_float(a);
        let rhs_num = is_int(b) || is_long(b) || is_float(b);
        if lhs_num && rhs_num {
            let q = floordiv(a, b)?;
            let r = mod_(a, b)?;
            return Ok(w_tuple_new(vec![q, r]));
        }
    }
    Ok(w_not_implemented())
}

pub(crate) fn lshift_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_lshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_lshift(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn rshift_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_rshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_rshift(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn and_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        // int.__and__ — bool operands are treated as ints; the bool-typed
        // result is produced by bool.__and__ (init_bool_type), not here.
        if is_int(a) && is_int(b) {
            return int_bitand(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitand(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn or_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return int_bitor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitor(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn xor_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int(a) && is_int(b) {
            return int_bitxor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitxor(a, b);
        }
        Ok(w_not_implemented())
    }
}

// ── descroperation helpers — pypy/objspace/descroperation.py ──────────
//
// These helpers implement the standard "forward + reverse with
// NotImplemented fallback" dispatch that PyPy generates from
// `_make_binop_impl` / `_make_descr_unaryop`. They were originally in
// `baseobjspace` (not in `builtins`) because they are space-level
// semantics shared between the builtin module, the weakproxy wrappers,
// and any future opcode dispatch — every caller needs the same rule
// or NotImplemented from the forward path silently swallows the
// reflected operand.

/// `space.lookup(w_obj, dunder)` — descroperation.py.
pub(crate) unsafe fn lookup_type_special(obj: PyObjectRef, dunder: &str) -> Option<PyObjectRef> {
    crate::typedef::r#type(obj).and_then(|tp| lookup_in_type(tp, dunder))
}

/// descroperation.py `_binop_impl` — when `type(rhs)` is a proper
/// subtype of `type(lhs)` and defines the reflected dunder, the
/// reflected operand is tried first.
pub(crate) unsafe fn should_try_reverse_first(
    lhs: PyObjectRef,
    rhs: PyObjectRef,
    rdunder: &str,
) -> bool {
    let Some(lhs_type) = crate::typedef::r#type(lhs) else {
        return false;
    };
    let Some(rhs_type) = crate::typedef::r#type(rhs) else {
        return false;
    };
    !std::ptr::eq(lhs_type, rhs_type)
        && issubtype_w(rhs_type, lhs_type)
        && lookup_in_type(rhs_type, rdunder).is_some()
}

/// Call a special method and treat NotImplemented as "no result", per
/// descroperation.py `_check_notimplemented`.
pub(crate) fn try_call_special(
    method: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<Option<PyObjectRef>, PyError> {
    let result = crate::call::call_function_impl_result(method, args)?;
    if unsafe { is_not_implemented(result) } {
        Ok(None)
    } else {
        Ok(Some(result))
    }
}

/// descroperation.py:648 `_call_binop_impl` — resolve the forward
/// (`dunder`) and reflected (`rdunder`) special methods through
/// `lookup_where`, decide whether to try the reflected operand first by
/// comparing the two defining classes, then invoke forward-then-reverse.
pub(crate) fn try_dispatch_binary_special(
    lhs: PyObjectRef,
    rhs: PyObjectRef,
    dunder: &str,
    rdunder: &str,
) -> Result<Option<PyObjectRef>, PyError> {
    // descroperation.py:687 `seq_bug_compat = (symbol == '+' or symbol == '*')`.
    let seq_bug_compat = dunder == "__add__" || dunder == "__mul__";
    unsafe {
        let Some(w_typ1) = crate::typedef::r#type(lhs) else {
            return Ok(None);
        };
        let Some(w_typ2) = crate::typedef::r#type(rhs) else {
            return Ok(None);
        };
        let (w_left_src, mut w_left_impl) = match lookup_where_with_method_cache(w_typ1, dunder) {
            Some((src, imp)) => (Some(src), Some(imp)),
            None => (None, None),
        };
        let mut w_obj1 = lhs;
        let mut w_obj2 = rhs;
        let mut w_right_impl: Option<PyObjectRef> = None;
        // descroperation.py:652 — same type means the reflected method is
        // never considered.
        if !std::ptr::eq(w_typ1, w_typ2) {
            let (w_right_src, wri) = match lookup_where_with_method_cache(w_typ2, rdunder) {
                Some((src, imp)) => (Some(src), Some(imp)),
                None => (None, None),
            };
            w_right_impl = wri;
            // descroperation.py:662 — both `__op__` and `__rop__` are
            // found, in different MRO classes.
            if let (Some(rsrc), Some(lsrc)) = (w_right_src, w_left_src) {
                if !std::ptr::eq(lsrc, rsrc) {
                    // descroperation.py:667-670.
                    let prefer_reverse = (seq_bug_compat
                        && crate::baseobjspace::flag_sequence_bug_compat(w_typ1)
                        && !crate::baseobjspace::flag_sequence_bug_compat(w_typ2))
                        || issubtype_w(w_typ2, w_typ1);
                    // descroperation.py:671-672.
                    if prefer_reverse
                        && !p_abstract_issubclass_w(lsrc, rsrc)?
                        && !p_abstract_issubclass_w(w_typ1, rsrc)?
                    {
                        std::mem::swap(&mut w_obj1, &mut w_obj2);
                        std::mem::swap(&mut w_left_impl, &mut w_right_impl);
                    }
                }
            }
        }
        // descroperation.py:676 — _invoke_binop(w_left_impl, w_obj1, w_obj2).
        if let Some(method) = w_left_impl {
            if let Some(result) = try_call_special(method, &[w_obj1, w_obj2])? {
                return Ok(Some(result));
            }
        }
        // descroperation.py:679 — _invoke_binop(w_right_impl, w_obj2, w_obj1).
        if let Some(method) = w_right_impl {
            if let Some(result) = try_call_special(method, &[w_obj2, w_obj1])? {
                return Ok(Some(result));
            }
        }
        Ok(None)
    }
}

/// descroperation.py:825 `inplace_impl` — try the in-place special
/// (`__iadd__` etc.) on the lhs.  Returns `None` when the type has no
/// such method or the call yields `NotImplemented`, so the caller falls
/// back to the corresponding binary operation.
pub(crate) fn try_inplace_special(
    lhs: PyObjectRef,
    rhs: PyObjectRef,
    idunder: &str,
    rdunder: Option<&str>,
    seq_bug_compat: bool,
) -> Result<Option<PyObjectRef>, PyError> {
    // descroperation.py:826 — only when the lhs in-place method exists.
    if let Some(method) = unsafe { lookup_type_special(lhs, idunder) } {
        // descroperation.py:831 seq_bug_compat — for `+=` / `*=` where the
        // lhs is a builtin sequence and the rhs is not, try the rhs
        // reflected method before the lhs in-place method.
        if seq_bug_compat {
            if let Some(rd) = rdunder {
                if let (Some(lhs_type), Some(rhs_type)) =
                    (crate::typedef::r#type(lhs), crate::typedef::r#type(rhs))
                {
                    if crate::baseobjspace::flag_sequence_bug_compat(lhs_type)
                        && !crate::baseobjspace::flag_sequence_bug_compat(rhs_type)
                    {
                        if let Some(rmethod) = unsafe { lookup_type_special(rhs, rd) } {
                            if let Some(result) = try_call_special(rmethod, &[rhs, lhs])? {
                                return Ok(Some(result));
                            }
                        }
                    }
                }
            }
        }
        if let Some(result) = try_call_special(method, &[lhs, rhs])? {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

/// descroperation.py:399 `def pow(space, w_obj1, w_obj2, w_obj3)` —
/// the same forward/reverse dance as the binary version but threading
/// the third (modulo) operand through to both arms.
pub(crate) fn try_dispatch_ternary_special(
    lhs: PyObjectRef,
    rhs: PyObjectRef,
    third: PyObjectRef,
    dunder: &str,
    rdunder: &str,
) -> Result<Option<PyObjectRef>, PyError> {
    let try_reverse_first = unsafe { should_try_reverse_first(lhs, rhs, rdunder) };
    if try_reverse_first {
        if let Some(method) = unsafe { lookup_type_special(rhs, rdunder) } {
            if let Some(result) = try_call_special(method, &[rhs, lhs, third])? {
                return Ok(Some(result));
            }
        }
    }
    if let Some(method) = unsafe { lookup_type_special(lhs, dunder) } {
        if let Some(result) = try_call_special(method, &[lhs, rhs, third])? {
            return Ok(Some(result));
        }
    }
    if !try_reverse_first {
        if let Some(method) = unsafe { lookup_type_special(rhs, rdunder) } {
            if let Some(result) = try_call_special(method, &[rhs, lhs, third])? {
                return Ok(Some(result));
            }
        }
    }
    Ok(None)
}

/// `(int|long) ** (int|long) % (int|long)` fast path used by `space.pow`
/// when a modulus is supplied — longobject.py `int_pow`.
pub(crate) fn try_int_long_pow_with_modulo(
    base: PyObjectRef,
    exp: PyObjectRef,
    modulus: PyObjectRef,
) -> Result<Option<PyObjectRef>, PyError> {
    unsafe {
        if !is_int_or_long(base) || !is_int_or_long(exp) || !is_int_or_long(modulus) {
            return Ok(None);
        }

        let base = crate::builtins::obj_to_bigint(base);
        let exp = crate::builtins::obj_to_bigint(exp);
        let modulus = crate::builtins::obj_to_bigint(modulus);

        if bigint_eq(modulus.clone(), BigInt::from(0)) {
            return Err(PyError::value_error("pow() 3rd argument cannot be 0"));
        }
        if bigint_lt(exp.clone(), BigInt::from(0)) {
            // 3-arg pow with a negative exponent: raise the modular
            // inverse of `base` to `-exp` (longobject.c long_pow →
            // long_invmod).  The inverse exists only when `base` is
            // coprime to the modulus.
            let negative_modulus = bigint_lt(modulus.clone(), BigInt::from(0));
            let abs_modulus = if negative_modulus {
                bigint_neg(modulus.clone())
            } else {
                modulus.clone()
            };
            let base_residue = base.mod_floor(&abs_modulus);
            let egcd = base_residue.extended_gcd(&abs_modulus);
            if egcd.gcd != BigInt::from(1) {
                return Err(PyError::value_error(
                    "base is not invertible for the given modulus",
                ));
            }
            let inverse = egcd.x.mod_floor(&abs_modulus);
            let pos_exp = bigint_neg(exp.clone());
            let mut result = inverse.modpow(&pos_exp, &abs_modulus);
            if negative_modulus && bigint_gt(result.clone(), BigInt::from(0)) {
                result = bigint_sub(result, abs_modulus);
            }
            return Ok(Some(box_bigint_result(result)));
        }
        if bigint_eq(exp.clone(), BigInt::from(0)) {
            return Ok(Some(box_bigint_result(bigint_mod(
                BigInt::from(1),
                modulus,
            ))));
        }

        let negative_modulus = bigint_lt(modulus.clone(), BigInt::from(0));
        let abs_modulus = if negative_modulus {
            bigint_neg(modulus.clone())
        } else {
            modulus.clone()
        };
        let mut result = base.modpow(&exp, &abs_modulus);
        if negative_modulus && bigint_gt(result.clone(), BigInt::from(0)) {
            result = bigint_sub(result, abs_modulus);
        }
        Ok(Some(box_bigint_result(result)))
    }
}

pub(crate) fn box_bigint_result(value: BigInt) -> PyObjectRef {
    use num_traits::ToPrimitive;
    if let Some(small) = value.to_i64() {
        w_int_new(small)
    } else {
        w_long_new(value)
    }
}

pub(crate) fn binary_builtin_type_error(
    opname: &str,
    lhs: PyObjectRef,
    rhs: PyObjectRef,
) -> PyError {
    let lhs_name = unsafe {
        match crate::typedef::r#type(lhs) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => (*(*lhs).ob_type).name.to_string(),
        }
    };
    let rhs_name = unsafe {
        match crate::typedef::r#type(rhs) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => (*(*rhs).ob_type).name.to_string(),
        }
    };
    PyError::type_error(format!(
        "unsupported operand type(s) for {opname}: '{lhs_name}' and '{rhs_name}'"
    ))
}

/// 3-arg `pow(a, b, c)` dispatch — pypy/objspace/descroperation.py:399
/// `def pow(space, w_obj1, w_obj2, w_obj3)`. Tries the int/long modulus
/// fast path, then forward `__pow__` and reverse `__rpow__` with the
/// usual NotImplemented fallback. PyPy's MethodTable lists `pow` with
/// arity=3 (`('pow', '**', 3, ['__pow__', '__rpow__'])`) so a 3-arg
/// space op exists for the proxy wrapper to call.
pub fn pow3(base: PyObjectRef, exp: PyObjectRef, modulus: PyObjectRef) -> PyResult {
    let base = unwrap_cell(base);
    let exp = unwrap_cell(exp);
    let modulus = unwrap_cell(modulus);
    if unsafe { is_none(modulus) } {
        return pow(base, exp);
    }
    if let Some(result) = try_int_long_pow_with_modulo(base, exp, modulus)? {
        return Ok(result);
    }
    if let Some(result) = try_dispatch_ternary_special(base, exp, modulus, "__pow__", "__rpow__")? {
        return Ok(result);
    }
    Err(binary_builtin_type_error("pow()", base, exp))
}

/// `divmod(a, b)` dispatch — pypy/interpreter/baseobjspace.py:2159
/// `('divmod', 'divmod', 2, ['__divmod__', '__rdivmod__'])`. Numeric
/// fast path then forward + reverse special-method dispatch with the
/// standard NotImplemented fallback.
pub fn divmod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        let lhs_num = is_int(a) || is_long(a) || is_float(a);
        let rhs_num = is_int(b) || is_long(b) || is_float(b);
        if lhs_num && rhs_num {
            let q = floordiv(a, b)?;
            let r = mod_(a, b)?;
            return Ok(w_tuple_new(vec![q, r]));
        }
    }
    if let Some(result) = try_dispatch_binary_special(a, b, "__divmod__", "__rdivmod__")? {
        return Ok(result);
    }
    Err(binary_builtin_type_error("divmod()", a, b))
}

pub fn float_pow_raw(x: f64, y: f64) -> Result<f64, PyError> {
    // floatobject.py:800-801
    if y == 2.0 {
        return Ok(x * x);
    }
    // floatobject.py:803-804
    if y == 0.0 {
        return Ok(1.0);
    }
    // floatobject.py:806-807
    if x.is_nan() {
        return Ok(x);
    }
    // floatobject.py:809-814
    if y.is_nan() {
        return Ok(if x == 1.0 { 1.0 } else { y });
    }
    // floatobject.py:815-827
    if y.is_infinite() {
        let ax = x.abs();
        if ax == 1.0 {
            return Ok(1.0);
        }
        return Ok(if (y > 0.0) == (ax > 1.0) {
            f64::INFINITY
        } else {
            0.0
        });
    }
    // floatobject.py:828-842
    if x.is_infinite() {
        let y_is_odd = y.abs() % 2.0 == 1.0;
        return Ok(if y > 0.0 {
            if y_is_odd { x } else { x.abs() }
        } else if y_is_odd {
            float_copysign(0.0, x)
        } else {
            0.0
        });
    }
    // floatobject.py:844-847
    if x == 0.0 && y < 0.0 {
        return Err(PyError::zero_division(
            "0.0 cannot be raised to a negative power",
        ));
    }
    // floatobject.py:849-862
    let mut negate_result = false;
    let mut bx = x;
    if bx < 0.0 {
        if y.floor() != y {
            return Err(PyError::value_error(
                "negative number cannot be raised to a fractional power",
            ));
        }
        bx = -bx;
        negate_result = y.abs() % 2.0 == 1.0;
    }
    // floatobject.py:864-869
    if bx == 1.0 {
        return Ok(if negate_result { -1.0 } else { 1.0 });
    }
    // floatobject.py:871-877
    let z = bx.powf(y);
    if z.is_infinite() && !bx.is_infinite() {
        return Err(PyError::overflow_error("float power"));
    }
    // floatobject.py:879-881
    Ok(if negate_result { -z } else { z })
}

/// floatobject.py:562 `W_FloatObject.descr_pow` boxing wrapper over `_pow`.
/// Calls `float_pow_raw` and boxes the raw result into W_FloatObject.
fn float_pow_impl(x: f64, y: f64) -> PyResult {
    use pyre_object::w_float_new;
    Ok(w_float_new(float_pow_raw(x, y)?))
}

/// Left shift dispatch (`<<` operator).

pub fn lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if needs_numeric_binop_dispatch(a, b, "__lshift__", "__rlshift__") {
            if let Some(result) = try_dispatch_binary_special(a, b, "__lshift__", "__rlshift__")? {
                return Ok(result);
            }
        }
        if is_int_like(a) && is_int_like(b) {
            return int_lshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_lshift(a, b);
        }
        if let Some(result) = try_dispatch_binary_special(a, b, "__lshift__", "__rlshift__")? {
            return Ok(result);
        }
        if let Some(result) = try_instance_binop(a, b, "__lshift__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for <<: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Right shift dispatch (`>>` operator).

pub fn rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if needs_numeric_binop_dispatch(a, b, "__rshift__", "__rrshift__") {
            if let Some(result) = try_dispatch_binary_special(a, b, "__rshift__", "__rrshift__")? {
                return Ok(result);
            }
        }
        if is_int_like(a) && is_int_like(b) {
            return int_rshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_rshift(a, b);
        }
        if let Some(result) = try_dispatch_binary_special(a, b, "__rshift__", "__rrshift__")? {
            return Ok(result);
        }
        if let Some(result) = try_instance_binop(a, b, "__rshift__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for >>: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Bitwise AND dispatch (`&` operator).

pub fn and_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if needs_numeric_binop_dispatch(a, b, "__and__", "__rand__") {
            if let Some(result) = try_dispatch_binary_special(a, b, "__and__", "__rand__")? {
                return Ok(result);
            }
        }
        // boolobject.py:74 W_BoolObject.descr_and — both operands bool
        // → space.newbool(op(a, b)). MRO ensures this runs before the
        // W_IntObject.descr_and fallback in int_bitand.
        if is_bool(a) && is_bool(b) {
            return Ok(pyre_object::bool_descr_and(a, b));
        }
        if is_int(a) && is_int(b) {
            return int_bitand(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitand(a, b);
        }
        // set / frozenset intersection — PyPy: setobject.py W_BaseSetObject.descr_and.
        // descr_and returns NotImplemented for a non-set rhs, so `&` requires
        // both operands to be sets (the `intersection` method takes iterables).
        if pyre_object::is_set_or_frozenset(a) && pyre_object::is_set_or_frozenset(b) {
            let other_items = crate::builtins::collect_iterable(b)?;
            let probe = pyre_object::w_set_from_items(&other_items);
            let result: Vec<PyObjectRef> = pyre_object::w_set_items(a)
                .into_iter()
                .filter(|&item| pyre_object::w_set_contains(probe, item))
                .collect();
            return Ok(if pyre_object::is_frozenset(a) {
                pyre_object::w_frozenset_from_items(&result)
            } else {
                pyre_object::w_set_from_items(&result)
            });
        }
        if let Some(result) = try_dispatch_binary_special(a, b, "__and__", "__rand__")? {
            return Ok(result);
        }
        if let Some(result) = try_instance_binop(a, b, "__and__") {
            return result;
        }
        // Built-in W_Root types (dict_view, …) expose `__and__` /
        // `__rand__` through their typedef but are not is_instance
        // — fall back to typedef-driven MRO dispatch before TypeError.
        if let Some(result) = try_typedef_binop(a, b, "__and__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for &: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Check if an object can participate in `X | Y` union syntax.
///
/// PyPy equivalent: _unionable() in _pypy_generic_alias.py
#[inline]
pub(crate) fn unionable(obj: PyObjectRef) -> bool {
    unsafe {
        is_none(obj)
            || is_type(obj)
            || pyre_object::is_union(obj)
            || pyre_object::is_generic_alias(obj)
    }
}

/// Bitwise OR dispatch (`|` operator).

pub fn or_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    // `pypy/objspace/std/dictproxyobject.py:51 descr_or` /
    // `pypy/objspace/std/dictproxyobject.py:60 descr_ror` —
    // mappingproxy `|` dispatches by copying the proxy's wrapped
    // mapping then `update`-ing with the other operand.  Pre-unwrap
    // each side so the dict-arm below sees plain dicts and produces
    // the same merge result.  The proxy-on-rhs case mirrors
    // `descr_ror` (proxy wraps the rhs operand inside `__or__`).
    let a = unsafe {
        if pyre_object::is_dict_proxy(a) {
            pyre_object::w_dict_proxy_get_mapping(a)
        } else {
            a
        }
    };
    let b = unsafe {
        if pyre_object::is_dict_proxy(b) {
            pyre_object::w_dict_proxy_get_mapping(b)
        } else {
            b
        }
    };
    unsafe {
        if needs_numeric_binop_dispatch(a, b, "__or__", "__ror__") {
            if let Some(result) = try_dispatch_binary_special(a, b, "__or__", "__ror__")? {
                return Ok(result);
            }
        }
        // boolobject.py:75 W_BoolObject.descr_or — both operands bool
        // → space.newbool(op(a, b)).
        if is_bool(a) && is_bool(b) {
            return Ok(pyre_object::bool_descr_or(a, b));
        }
        if is_int(a) && is_int(b) {
            return int_bitor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitor(a, b);
        }
        // set / frozenset union — PyPy: setobject.py W_BaseSetObject.descr_or.
        // descr_or returns NotImplemented unless w_other is a set/frozenset,
        // so the binary `|` operator requires both operands to be sets
        // (the `union` method accepts arbitrary iterables, descr_or does not).
        if pyre_object::is_set_or_frozenset(a) && pyre_object::is_set_or_frozenset(b) {
            let mut items = pyre_object::w_set_items(a);
            for item in crate::builtins::collect_iterable(b)? {
                items.push(item);
            }
            return Ok(if pyre_object::is_frozenset(a) {
                pyre_object::w_frozenset_from_items(&items)
            } else {
                pyre_object::w_set_from_items(&items)
            });
        }
        // dict | dict — PEP 584 merge. PyPy: dictmultiobject.py descr_or.
        // Returns a new dict built from `a`'s items, then updated with `b`'s.
        if pyre_object::is_dict(a) && pyre_object::is_dict(b) {
            let new_dict = pyre_object::w_dict_new();
            for (k, v) in pyre_object::w_dict_items(a) {
                pyre_object::w_dict_store(new_dict, k, v);
            }
            for (k, v) in pyre_object::w_dict_items(b) {
                pyre_object::w_dict_store(new_dict, k, v);
            }
            return Ok(new_dict);
        }
        if let Some(result) = try_instance_binop(a, b, "__or__") {
            return result;
        }
        // type | type — PEP 604 union types (Python 3.10+)
        // PyPy: typeobject.py descr_or → _pypy_generic_alias._create_union,
        // which collapses identical operands (`int | int` is `int`).
        if unionable(a) && unionable(b) {
            return crate::genericalias::create_union(a, b);
        }
        if let Some(result) = try_instance_binop(a, b, "__ror__") {
            return result;
        }
        // Built-in W_Root types (dict_view, …) expose `__or__` /
        // `__ror__` through their typedef.
        if let Some(result) = try_typedef_binop(a, b, "__or__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for |: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Bitwise XOR dispatch (`^` operator).

pub fn xor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if needs_numeric_binop_dispatch(a, b, "__xor__", "__rxor__") {
            if let Some(result) = try_dispatch_binary_special(a, b, "__xor__", "__rxor__")? {
                return Ok(result);
            }
        }
        if is_bool(a) && is_bool(b) {
            return Ok(pyre_object::bool_descr_xor(a, b));
        }
        if is_int(a) && is_int(b) {
            return int_bitxor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitxor(a, b);
        }
        // set / frozenset symmetric difference — `pypy/objspace/std/
        // setobject.py W_BaseSetObject.descr_xor`.  Mirrors `and_`'s
        // intersection arm: walk both sides, keep elements present in
        // exactly one set.  Result type follows the left operand
        // (frozenset stays frozenset).
        if pyre_object::is_set_or_frozenset(a) && pyre_object::is_set_or_frozenset(b) {
            let mut out: Vec<PyObjectRef> = pyre_object::w_set_items(a)
                .into_iter()
                .filter(|&item| !pyre_object::w_set_contains(b, item))
                .collect();
            for item in pyre_object::w_set_items(b) {
                if !pyre_object::w_set_contains(a, item) {
                    out.push(item);
                }
            }
            return Ok(if pyre_object::is_frozenset(a) {
                pyre_object::w_frozenset_from_items(&out)
            } else {
                pyre_object::w_set_from_items(&out)
            });
        }
        if let Some(result) = try_dispatch_binary_special(a, b, "__xor__", "__rxor__")? {
            return Ok(result);
        }
        if let Some(result) = try_instance_binop(a, b, "__xor__") {
            return result;
        }
        // Built-in W_Root types (dict_view, …) expose `__xor__` /
        // `__rxor__` through their typedef.
        if let Some(result) = try_typedef_binop(a, b, "__xor__") {
            return result;
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for ^: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Comparison operation dispatch.

pub fn compare(a: PyObjectRef, b: PyObjectRef, op: CompareOp) -> PyResult {
    let a = unwrap_cell(a);
    let b = unwrap_cell(b);
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return match op {
                CompareOp::Lt => int_lt(a, b),
                CompareOp::Le => int_le(a, b),
                CompareOp::Gt => int_gt(a, b),
                CompareOp::Ge => int_ge(a, b),
                CompareOp::Eq => int_eq(a, b),
                CompareOp::Ne => int_ne(a, b),
            };
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            let va = as_bigint(a);
            let vb = as_bigint(b);
            return Ok(w_bool_from(match op {
                CompareOp::Lt => va < vb,
                CompareOp::Le => va <= vb,
                CompareOp::Gt => va > vb,
                CompareOp::Ge => va >= vb,
                CompareOp::Eq => va == vb,
                CompareOp::Ne => va != vb,
            }));
        }
        if is_float_pair(a, b) {
            return match op {
                CompareOp::Lt => float_lt(a, b),
                CompareOp::Le => float_le(a, b),
                CompareOp::Gt => float_gt(a, b),
                CompareOp::Ge => float_ge(a, b),
                CompareOp::Eq => float_eq(a, b),
                CompareOp::Ne => float_ne(a, b),
            };
        }
        if is_str(a) && is_str(b) {
            // Compare the WTF-8 bytes: for surrogate-free strings this is the
            // UTF-8 byte order (= code point order), and WTF-8 keeps lone
            // surrogates in code-point order too, so a surrogate-bearing
            // string compares correctly without going through
            // `w_str_get_value`.
            let sa = w_str_get_wtf8(a).as_bytes();
            let sb = w_str_get_wtf8(b).as_bytes();
            return Ok(w_bool_from(match op {
                CompareOp::Lt => sa < sb,
                CompareOp::Le => sa <= sb,
                CompareOp::Gt => sa > sb,
                CompareOp::Ge => sa >= sb,
                CompareOp::Eq => sa == sb,
                CompareOp::Ne => sa != sb,
            }));
        }
        // bytesobject.py W_BytesObject.descr_eq / _lt / ... and the
        // bytearray counterparts — lexicographic comparison on the raw
        // bytes.  bytes and bytearray compare by content
        // (b"a" == bytearray(b"a")), so both operands route through
        // bytes_like_data.
        if pyre_object::bytesobject::is_bytes_like(a) && pyre_object::bytesobject::is_bytes_like(b)
        {
            let da = pyre_object::bytesobject::bytes_like_data(a);
            let db = pyre_object::bytesobject::bytes_like_data(b);
            return Ok(w_bool_from(match op {
                CompareOp::Lt => da < db,
                CompareOp::Le => da <= db,
                CompareOp::Gt => da > db,
                CompareOp::Ge => da >= db,
                CompareOp::Eq => da == db,
                CompareOp::Ne => da != db,
            }));
        }
        // Tuple lexicographic comparison — PyPy: tupleobject.py descr_lt / _eq / etc.
        if is_tuple(a) && is_tuple(b) {
            let la = w_tuple_len(a);
            let lb = w_tuple_len(b);
            let min_len = la.min(lb);
            for i in 0..min_len {
                let ea = w_tuple_getitem(a, i as i64).unwrap_or(PY_NULL);
                let eb = w_tuple_getitem(b, i as i64).unwrap_or(PY_NULL);
                let eq = match compare(ea, eb, CompareOp::Eq) {
                    Ok(r) => is_true(r),
                    Err(_) => false,
                };
                if !eq {
                    return compare(ea, eb, op);
                }
            }
            return Ok(w_bool_from(match op {
                CompareOp::Lt => la < lb,
                CompareOp::Le => la <= lb,
                CompareOp::Gt => la > lb,
                CompareOp::Ge => la >= lb,
                CompareOp::Eq => la == lb,
                CompareOp::Ne => la != lb,
            }));
        }
        // dict equality — `pypy/objspace/std/dictobject.py
        // W_DictMultiObject.descr_eq` is order-independent: same length
        // AND each key-value pair in `a` exists with equal value in `b`.
        // CPython only defines == / != for dicts (no ordering), so we
        // restrict to those ops; other ops fall through to the dunder
        // dispatch which currently raises TypeError, matching the
        // unimplemented `__lt__` etc. on plain dict.
        if is_dict(a) && is_dict(b) && matches!(op, CompareOp::Eq | CompareOp::Ne) {
            let la = pyre_object::w_dict_len(a);
            let lb = pyre_object::w_dict_len(b);
            let mut equal = la == lb;
            if equal {
                for (k, v) in pyre_object::w_dict_items(a) {
                    match pyre_object::w_dict_lookup(b, k) {
                        Some(other_v) => {
                            let same = compare(v, other_v, CompareOp::Eq)
                                .map(|r| is_true(r))
                                .unwrap_or(false);
                            if !same {
                                equal = false;
                                break;
                            }
                        }
                        None => {
                            equal = false;
                            break;
                        }
                    }
                }
            }
            return Ok(w_bool_from(match op {
                CompareOp::Eq => equal,
                CompareOp::Ne => !equal,
                _ => unreachable!(),
            }));
        }
        // `dictmultiobject.py:1619-1623 _is_set_like` parity — when
        // one side is a set/frozenset and the other is a set-like
        // dict_view (Keys / Items), the comparison reduces to the
        // set-set arm with the dict_view materialised through its
        // snapshot.  Without this arm, `set == d.keys()` would fall
        // through to `object.__eq__`'s identity check and return
        // False even when the contents match.
        if (pyre_object::is_set_or_frozenset(a) || pyre_object::is_set_or_frozenset(b))
            && (pyre_object::dictviewobject::is_dict_view(a)
                || pyre_object::dictviewobject::is_dict_view(b))
        {
            let view_set_like = |obj: PyObjectRef| -> bool {
                if pyre_object::is_set_or_frozenset(obj) {
                    return true;
                }
                if pyre_object::dictviewobject::is_dict_view(obj) {
                    let kind = pyre_object::dictviewobject::w_dict_view_get_kind(obj);
                    return matches!(
                        kind,
                        pyre_object::dictviewobject::DictViewKind::Keys
                            | pyre_object::dictviewobject::DictViewKind::Items
                    );
                }
                false
            };
            if view_set_like(a) && view_set_like(b) {
                let a_items = if pyre_object::is_set_or_frozenset(a) {
                    pyre_object::w_set_items(a)
                } else {
                    crate::type_methods::dict_view_snapshot(a)
                };
                let b_items = if pyre_object::is_set_or_frozenset(b) {
                    pyre_object::w_set_items(b)
                } else {
                    crate::type_methods::dict_view_snapshot(b)
                };
                let a_set = pyre_object::w_set_from_items(&a_items);
                let b_set = pyre_object::w_set_from_items(&b_items);
                let la = pyre_object::w_set_len(a_set);
                let lb = pyre_object::w_set_len(b_set);
                let a_subset_b = || {
                    pyre_object::w_set_items(a_set)
                        .into_iter()
                        .all(|item| pyre_object::w_set_contains(b_set, item))
                };
                let b_subset_a = || {
                    pyre_object::w_set_items(b_set)
                        .into_iter()
                        .all(|item| pyre_object::w_set_contains(a_set, item))
                };
                return Ok(w_bool_from(match op {
                    CompareOp::Eq => la == lb && a_subset_b(),
                    CompareOp::Ne => la != lb || !a_subset_b(),
                    CompareOp::Le => la <= lb && a_subset_b(),
                    CompareOp::Lt => la < lb && a_subset_b(),
                    CompareOp::Ge => la >= lb && b_subset_a(),
                    CompareOp::Gt => la > lb && b_subset_a(),
                }));
            }
        }
        // set / frozenset comparison — subset / superset / equality.
        // PyPy: setobject.py W_BaseSetObject.descr_eq, descr_le, descr_lt
        if pyre_object::is_set_or_frozenset(a) && pyre_object::is_set_or_frozenset(b) {
            let la = pyre_object::w_set_len(a);
            let lb = pyre_object::w_set_len(b);
            let a_subset_b = || {
                pyre_object::w_set_items(a)
                    .into_iter()
                    .all(|item| pyre_object::w_set_contains(b, item))
            };
            let b_subset_a = || {
                pyre_object::w_set_items(b)
                    .into_iter()
                    .all(|item| pyre_object::w_set_contains(a, item))
            };
            return Ok(w_bool_from(match op {
                CompareOp::Eq => la == lb && a_subset_b(),
                CompareOp::Ne => la != lb || !a_subset_b(),
                CompareOp::Le => la <= lb && a_subset_b(),
                CompareOp::Lt => la < lb && a_subset_b(),
                CompareOp::Ge => la >= lb && b_subset_a(),
                CompareOp::Gt => la > lb && b_subset_a(),
            }));
        }
        // List lexicographic comparison — same logic as tuple.
        if is_list(a) && is_list(b) {
            let la = pyre_object::w_list_len(a);
            let lb = pyre_object::w_list_len(b);
            let min_len = la.min(lb);
            for i in 0..min_len {
                let ea = pyre_object::w_list_getitem(a, i as i64).unwrap_or(PY_NULL);
                let eb = pyre_object::w_list_getitem(b, i as i64).unwrap_or(PY_NULL);
                let eq = match compare(ea, eb, CompareOp::Eq) {
                    Ok(r) => is_true(r),
                    Err(_) => false,
                };
                if !eq {
                    return compare(ea, eb, op);
                }
            }
            return Ok(w_bool_from(match op {
                CompareOp::Lt => la < lb,
                CompareOp::Le => la <= lb,
                CompareOp::Gt => la > lb,
                CompareOp::Ge => la >= lb,
                CompareOp::Eq => la == lb,
                CompareOp::Ne => la != lb,
            }));
        }
        // range value comparison — rangeobject.py W_Range.descr_eq:
        // two ranges are equal iff they generate the same sequence
        // (equal length, and for non-empty ranges equal start and —
        // for length > 1 — equal step).  Only `==` / `!=` are defined;
        // ordering falls through to the dunder dispatch (TypeError).
        if pyre_object::is_w_range(a)
            && pyre_object::is_w_range(b)
            && matches!(op, CompareOp::Eq | CompareOp::Ne)
        {
            let equal = pyre_object::w_range_eq(a, b);
            return Ok(w_bool_from(match op {
                CompareOp::Eq => equal,
                CompareOp::Ne => !equal,
                _ => unreachable!(),
            }));
        }
        // Instance dunder dispatch for comparison
        let dunder = match op {
            CompareOp::Lt => "__lt__",
            CompareOp::Le => "__le__",
            CompareOp::Gt => "__gt__",
            CompareOp::Ge => "__ge__",
            CompareOp::Eq => "__eq__",
            CompareOp::Ne => "__ne__",
        };
        if let Some(result) = try_instance_binop(a, b, dunder) {
            return result;
        }
        // `dictmultiobject.py:1628-1656 SetLikeDictView` — dict_keys
        // / dict_items expose `__eq__` / `__ne__` / `__lt__` / etc.
        // through the typedef.  `try_instance_binop` only fires for
        // is_instance-shaped objects, so dict views (a separate
        // W_Root type) need an explicit MRO-driven dispatch here.
        // Reflected: if RHS is a dict view, try `b.dunder(a)` —
        // PyPy's `_is_set_like(other)` short-circuits the LHS-side
        // descr_eq when the other is set-like, so the reflected call
        // path is the one that succeeds for `set == d.keys()`.
        if let Some(a_type) = crate::typedef::r#type(a) {
            if let Some(method) = lookup_in_type_where(a_type, dunder) {
                if let Ok(result) = crate::call::call_function_impl_result(method, &[a, b]) {
                    if !is_not_implemented(result) {
                        return Ok(result);
                    }
                }
            }
        }
        if let Some(rdunder) = reverse_dunder(dunder) {
            if let Some(b_type) = crate::typedef::r#type(b) {
                if let Some(method) = lookup_in_type_where(b_type, rdunder) {
                    if let Ok(result) = crate::call::call_function_impl_result(method, &[b, a]) {
                        if !is_not_implemented(result) {
                            return Ok(result);
                        }
                    }
                }
            }
        }
        // Identity comparison fallback for == and !=
        if matches!(op, CompareOp::Eq) {
            return Ok(w_bool_from(std::ptr::eq(a, b)));
        }
        if matches!(op, CompareOp::Ne) {
            return Ok(w_bool_from(!std::ptr::eq(a, b)));
        }
        let a_name = (*(*a).ob_type).name;
        let b_name = (*(*b).ob_type).name;
        let op_symbol = op.symbol();
        Err(PyError::type_error(format!(
            "'{op_symbol}' not supported between instances of '{a_name}' and '{b_name}'"
        )))
    }
}

/// Comparison operator enum (mirrors RustPython's ComparisonOperator).
#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CompareOp {
    /// The Python operator symbol (`<`, `<=`, `>`, `>=`, `==`, `!=`) used in
    /// the "not supported between instances of" TypeError message.
    pub fn symbol(self) -> &'static str {
        match self {
            CompareOp::Lt => "<",
            CompareOp::Le => "<=",
            CompareOp::Gt => ">",
            CompareOp::Ge => ">=",
            CompareOp::Eq => "==",
            CompareOp::Ne => "!=",
        }
    }
}

/// Unary positive (`+a`).

pub fn pos(a: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    unsafe {
        if let Some(result) = try_numeric_unaryop_override(a, "__pos__") {
            return result;
        }
        if is_int(a) || is_bool(a) {
            return Ok(w_int_new(int_value(a)));
        }
        if is_long(a) {
            return Ok(bigint_result(w_long_get_value(a).clone()));
        }
        if is_float(a) {
            return Ok(w_float_new(w_float_get_value(a)));
        }
        if let Some(result) = try_instance_unaryop(a, "__pos__") {
            return result;
        }
        if a.is_null() {
            return Err(PyError::type_error(
                "bad operand type for unary +: 'NoneType'",
            ));
        }
        Err(PyError::type_error(format!(
            "bad operand type for unary +: '{}'",
            (*(*a).ob_type).name,
        )))
    }
}

/// Unary negation.

pub fn neg(a: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    unsafe {
        if let Some(result) = try_numeric_unaryop_override(a, "__neg__") {
            return result;
        }
        if is_int(a) || is_bool(a) {
            let v = int_value(a);
            return match v.checked_neg() {
                Some(r) => Ok(w_int_new(r)),
                None => Ok(w_long_new(bigint_neg(BigInt::from(v)))),
            };
        }
        if is_long(a) {
            return Ok(bigint_result(bigint_neg(w_long_get_value(a).clone())));
        }
        if is_float(a) {
            return Ok(w_float_new(-w_float_get_value(a)));
        }
        // Instance __neg__
        if let Some(result) = try_instance_unaryop(a, "__neg__") {
            return result;
        }
        if a.is_null() {
            return Err(PyError::type_error(
                "bad operand type for unary -: 'NoneType'",
            ));
        }
        Err(PyError::type_error(format!(
            "bad operand type for unary -: '{}'",
            (*(*a).ob_type).name,
        )))
    }
}

/// Unary bitwise inversion.

pub fn invert(a: PyObjectRef) -> PyResult {
    let a = unwrap_cell(a);
    unsafe {
        if let Some(result) = try_numeric_unaryop_override(a, "__invert__") {
            return result;
        }
        if is_int(a) || is_bool(a) {
            return Ok(w_int_new(!int_value(a)));
        }
        if is_long(a) {
            return Ok(bigint_result(bigint_invert(w_long_get_value(a).clone())));
        }
        if let Some(result) = try_instance_unaryop(a, "__invert__") {
            return result;
        }
        Err(PyError::type_error(format!(
            "bad operand type for unary ~: '{}'",
            (*(*a).ob_type).name,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_add() {
        let a = w_int_new(3);
        let b = w_int_new(4);
        let result = add(a, b).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 7) };
    }

    #[test]
    fn test_int_compare() {
        let a = w_int_new(5);
        let b = w_int_new(10);
        let result = compare(a, b, CompareOp::Lt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_zero_division() {
        let a = w_int_new(5);
        let b = w_int_new(0);
        assert!(floordiv(a, b).is_err());
    }

    #[test]
    fn test_truthiness() {
        assert!(is_true(w_int_new(1)));
        assert!(!is_true(w_int_new(0)));
        assert!(!is_true(w_none()));
        assert!(is_true(w_bool_from(true)));
        assert!(!is_true(w_bool_from(false)));
    }

    #[test]
    fn test_int_add_overflow() {
        let a = w_int_new(i64::MAX);
        let b = w_int_new(1);
        let result = add(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) + BigInt::from(1)
            );
        }
    }

    #[test]
    fn test_int_sub_overflow() {
        let a = w_int_new(i64::MIN);
        let b = w_int_new(1);
        let result = sub(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MIN) - BigInt::from(1)
            );
        }
    }

    #[test]
    fn test_int_mul_overflow() {
        let a = w_int_new(i64::MAX);
        let b = w_int_new(2);
        let result = mul(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) * BigInt::from(2)
            );
        }
    }

    #[test]
    fn test_long_add() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(100);
        let result = add(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) + BigInt::from(101)
            );
        }
    }

    #[test]
    fn test_long_demote_to_int() {
        // long + long that fits back in i64 → W_IntObject
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(-1);
        let result = add(a, b).unwrap();
        unsafe {
            assert!(is_int(result));
            assert_eq!(w_int_get_value(result), i64::MAX);
        }
    }

    #[test]
    fn test_negate_min_int() {
        let a = w_int_new(i64::MIN);
        let result = neg(a).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), -BigInt::from(i64::MIN));
        }
    }

    #[test]
    fn test_invert_int() {
        let result = invert(w_int_new(6)).unwrap();
        unsafe {
            assert!(is_int(result));
            assert_eq!(w_int_get_value(result), !6);
        }
    }

    #[test]
    fn test_long_compare() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(i64::MAX);
        let result = compare(a, b, CompareOp::Gt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_long_truthiness() {
        assert!(is_true(w_long_new(
            BigInt::from(i64::MAX) + BigInt::from(1)
        )));
        assert!(!is_true(w_long_new(BigInt::from(0))));
    }

    #[test]
    fn test_int_pow() {
        let result = pow(w_int_new(2), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_pow_overflow() {
        let result = pow(w_int_new(2), w_int_new(63)).unwrap();
        unsafe {
            // 2^63 overflows i64, should be long
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(2).pow(63));
        }
    }

    #[test]
    fn test_int_pow_negative_exponent() {
        let result = pow(w_int_new(2), w_int_new(-1)).unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 0.5);
        }
    }

    #[test]
    fn test_int_lshift() {
        let result = lshift(w_int_new(1), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_lshift_overflow() {
        let result = lshift(w_int_new(1), w_int_new(64)).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(1) << 64);
        }
    }

    #[test]
    fn test_int_rshift() {
        let result = rshift(w_int_new(1024), w_int_new(3)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 128) };
    }

    #[test]
    fn test_negative_shift_count() {
        assert!(lshift(w_int_new(1), w_int_new(-1)).is_err());
        assert!(rshift(w_int_new(1), w_int_new(-1)).is_err());
    }

    #[test]
    fn test_int_bitand() {
        let result = and_(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0x0F) };
    }

    #[test]
    fn test_int_bitor() {
        let result = or_(w_int_new(0xF0), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xFF) };
    }

    #[test]
    fn test_int_bitxor() {
        let result = xor(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xF0) };
    }

    #[test]
    fn test_long_bitand() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(0xFF);
        let result = and_(a, b).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0) };
    }

    #[test]
    fn test_invert_long() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let result = invert(a).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                !(BigInt::from(i64::MAX) + BigInt::from(1))
            );
        }
    }
}
