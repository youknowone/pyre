//! ObjSpace — Python object operation dispatch.
//!
//! The ObjSpace mediates all operations on Python objects. This is the layer
//! where type-specific fast paths live, and where the JIT inserts `GuardClass`
//! to specialize operations.

// Suppress unsafe-in-unsafe-fn warnings; our unsafe fns are inherently
// working with raw pointers throughout and wrapping every call in an
// additional unsafe block adds noise without safety benefit.
#![allow(unsafe_op_in_unsafe_fn)]

use malachite_bigint::BigInt;
use num_integer::Integer;
use num_traits::ToPrimitive;

use std::cell::RefCell;
use std::collections::HashMap;

use pyre_object::strobject::is_str;
use pyre_object::*;
pub use pyre_runtime::{PyError, PyErrorKind, PyResult};

// ── BigInt helpers ──────────────────────────────────────────────────

/// Extract a BigInt from an int or long object.
unsafe fn as_bigint(obj: PyObjectRef) -> BigInt {
    if is_int(obj) {
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
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    match va.checked_add(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(BigInt::from(va) + BigInt::from(vb))),
    }
}

unsafe fn int_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    match va.checked_sub(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(BigInt::from(va) - BigInt::from(vb))),
    }
}

unsafe fn int_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    match va.checked_mul(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(BigInt::from(va) * BigInt::from(vb))),
    }
}

unsafe fn int_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    if vb == 0 {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    // i64::MIN / -1 overflows
    match va.checked_div_euclid(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(bigint_result(BigInt::from(va).div_floor(&BigInt::from(vb)))),
    }
}

unsafe fn int_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    if vb == 0 {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    Ok(w_int_new(va.rem_euclid(vb)))
}

// ── Long (BigInt) arithmetic operations ─────────────────────────────

unsafe fn long_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(as_bigint(a) + as_bigint(b)))
}

unsafe fn long_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(as_bigint(a) - as_bigint(b)))
}

unsafe fn long_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(as_bigint(a) * as_bigint(b)))
}

unsafe fn long_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if vb == BigInt::from(0) {
        return Err(PyError::zero_division("integer division or modulo by zero"));
    }
    Ok(bigint_result(as_bigint(a).div_floor(&vb)))
}

unsafe fn long_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if vb == BigInt::from(0) {
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

unsafe fn float_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_float(b);
    if vb == 0.0 {
        return Err(PyError::zero_division("float floor division by zero"));
    }
    Ok(w_float_new((as_float(a) / vb).floor()))
}

unsafe fn float_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_float(b);
    if vb == 0.0 {
        return Err(PyError::zero_division("float modulo"));
    }
    let va = as_float(a);
    // Python modulo: result has the sign of the divisor
    let r = va % vb;
    let result = if r != 0.0 && ((r > 0.0) != (vb > 0.0)) {
        r + vb
    } else {
        r
    };
    Ok(w_float_new(result))
}

// ── Power ────────────────────────────────────────────────────────────

unsafe fn int_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    if vb < 0 {
        // Negative exponent → float result
        return Ok(w_float_new((va as f64).powf(vb as f64)));
    }
    let vb = vb as u32;
    match va.checked_pow(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(BigInt::from(va).pow(vb))),
    }
}

unsafe fn long_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if vb < BigInt::from(0) {
        let fa = as_float(a);
        let fb = as_float(b);
        return Ok(w_float_new(fa.powf(fb)));
    }
    let exp = vb.to_u32().unwrap_or(u32::MAX);
    Ok(bigint_result(as_bigint(a).pow(exp)))
}

// ── Shift operations ─────────────────────────────────────────────────

unsafe fn int_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    if vb < 0 {
        return Err(PyError::type_error("negative shift count"));
    }
    let vb = vb as u32;
    if vb < 63 {
        if let Some(r) = va.checked_shl(vb) {
            return Ok(w_int_new(r));
        }
    }
    Ok(w_long_new(BigInt::from(va) << vb))
}

unsafe fn int_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = w_int_get_value(a);
    let vb = w_int_get_value(b);
    if vb < 0 {
        return Err(PyError::type_error("negative shift count"));
    }
    let vb = vb as u32;
    Ok(w_int_new(va >> vb.min(63)))
}

unsafe fn long_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if vb < BigInt::from(0) {
        return Err(PyError::type_error("negative shift count"));
    }
    let shift = vb.to_u32().unwrap_or(u32::MAX);
    Ok(bigint_result(as_bigint(a) << shift))
}

unsafe fn long_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_bigint(b);
    if vb < BigInt::from(0) {
        return Err(PyError::type_error("negative shift count"));
    }
    let shift = vb.to_u32().unwrap_or(u32::MAX);
    Ok(bigint_result(as_bigint(a) >> shift))
}

// ── Bitwise operations ───────────────────────────────────────────────

unsafe fn int_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(w_int_get_value(a) & w_int_get_value(b)))
}

unsafe fn int_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(w_int_get_value(a) | w_int_get_value(b)))
}

unsafe fn int_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(w_int_get_value(a) ^ w_int_get_value(b)))
}

unsafe fn long_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(as_bigint(a) & as_bigint(b)))
}

unsafe fn long_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(as_bigint(a) | as_bigint(b)))
}

unsafe fn long_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(bigint_result(as_bigint(a) ^ as_bigint(b)))
}

// ── String operations ────────────────────────────────────────────────

/// Concatenate two str objects.
unsafe fn str_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let sa = w_str_get_value(a);
    let sb = w_str_get_value(b);
    let mut result = String::with_capacity(sa.len() + sb.len());
    result.push_str(sa);
    result.push_str(sb);
    Ok(w_str_new(&result))
}

/// Repeat a str object `n` times.
unsafe fn str_repeat(s: PyObjectRef, n: PyObjectRef) -> PyResult {
    let sv = w_str_get_value(s);
    let nv = w_int_get_value(n);
    let count = if nv < 0 { 0 } else { nv as usize };
    Ok(w_str_new(&sv.repeat(count)))
}

// ── Comparison operations ─────────────────────────────────────────────

unsafe fn int_lt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(w_int_get_value(a) < w_int_get_value(b)))
}

unsafe fn int_le(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(w_int_get_value(a) <= w_int_get_value(b)))
}

unsafe fn int_gt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(w_int_get_value(a) > w_int_get_value(b)))
}

unsafe fn int_ge(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(w_int_get_value(a) >= w_int_get_value(b)))
}

unsafe fn int_eq(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(w_int_get_value(a) == w_int_get_value(b)))
}

unsafe fn int_ne(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(w_int_get_value(a) != w_int_get_value(b)))
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

/// Binary operation dispatch.
///
/// Checks types and dispatches to the appropriate fast path.
/// The JIT traces through this function, recording `GuardClass` on operand types.
pub fn py_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_add(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_add(a, b);
        }
        if is_float_pair(a, b) {
            return float_add(a, b);
        }
        if is_str(a) && is_str(b) {
            return str_concat(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for +: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

pub fn py_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_sub(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_sub(a, b);
        }
        if is_float_pair(a, b) {
            return float_sub(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for -: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

pub fn py_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_mul(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_mul(a, b);
        }
        if is_float_pair(a, b) {
            return float_mul(a, b);
        }
        if is_str(a) && is_int(b) {
            return str_repeat(a, b);
        }
        if is_int(a) && is_str(b) {
            return str_repeat(b, a);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for *: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

pub fn py_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_floordiv(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_floordiv(a, b);
        }
        if is_float_pair(a, b) {
            return float_floordiv(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for //: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

pub fn py_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_mod(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_mod(a, b);
        }
        if is_float_pair(a, b) {
            return float_mod(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for %: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// True division (`/` operator) — always produces a float result.
pub fn py_truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let a_num = is_int(a) || is_float(a) || is_long(a);
        let b_num = is_int(b) || is_float(b) || is_long(b);
        if a_num && b_num {
            return float_truediv(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for /: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Power operation dispatch (`**` operator).
pub fn py_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_pow(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_pow(a, b);
        }
        if is_float_pair(a, b) {
            return Ok(w_float_new(as_float(a).powf(as_float(b))));
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for **: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Left shift dispatch (`<<` operator).
pub fn py_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_lshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_lshift(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for <<: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Right shift dispatch (`>>` operator).
pub fn py_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_rshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_rshift(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for >>: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Bitwise AND dispatch (`&` operator).
pub fn py_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_bitand(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitand(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for &: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Bitwise OR dispatch (`|` operator).
pub fn py_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_bitor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitor(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for |: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Bitwise XOR dispatch (`^` operator).
pub fn py_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_bitxor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitxor(a, b);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for ^: '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
        )))
    }
}

/// Comparison operation dispatch.
pub fn py_compare(a: PyObjectRef, b: PyObjectRef, op: CompareOp) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
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
            let sa = w_str_get_value(a);
            let sb = w_str_get_value(b);
            return Ok(w_bool_from(match op {
                CompareOp::Lt => sa < sb,
                CompareOp::Le => sa <= sb,
                CompareOp::Gt => sa > sb,
                CompareOp::Ge => sa >= sb,
                CompareOp::Eq => sa == sb,
                CompareOp::Ne => sa != sb,
            }));
        }
        Err(PyError::type_error(format!(
            "'{op:?}' not supported between instances of '{}' and '{}'",
            (*(*a).ob_type).tp_name,
            (*(*b).ob_type).tp_name,
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

/// Test if an object is truthy (for branch conditions).
///
/// Python truthiness rules:
/// - None → false
/// - bool → its value
/// - int → nonzero
pub fn py_is_true(obj: PyObjectRef) -> bool {
    unsafe {
        if is_bool(obj) {
            return w_bool_get_value(obj);
        }
        if is_int(obj) {
            return w_int_get_value(obj) != 0;
        }
        if is_long(obj) {
            return *w_long_get_value(obj) != BigInt::from(0);
        }
        if is_float(obj) {
            return w_float_get_value(obj) != 0.0;
        }
        if is_str(obj) {
            return w_str_len(obj) != 0;
        }
        if is_list(obj) {
            return w_list_len(obj) > 0;
        }
        if is_tuple(obj) {
            return w_tuple_len(obj) > 0;
        }
        if is_dict(obj) {
            return w_dict_len(obj) > 0;
        }
        if is_none(obj) {
            return false;
        }
        true // default: objects are truthy
    }
}

/// Unary negation.
pub fn py_negative(a: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) {
            let v = w_int_get_value(a);
            return match v.checked_neg() {
                Some(r) => Ok(w_int_new(r)),
                None => Ok(w_long_new(-BigInt::from(v))),
            };
        }
        if is_long(a) {
            return Ok(bigint_result(-w_long_get_value(a).clone()));
        }
        if is_float(a) {
            return Ok(w_float_new(-w_float_get_value(a)));
        }
        Err(PyError::type_error(format!(
            "bad operand type for unary -: '{}'",
            (*(*a).ob_type).tp_name,
        )))
    }
}

/// Unary bitwise inversion.
pub fn py_invert(a: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) {
            return Ok(w_int_new(!w_int_get_value(a)));
        }
        if is_long(a) {
            return Ok(bigint_result(!w_long_get_value(a).clone()));
        }
        Err(PyError::type_error(format!(
            "bad operand type for unary ~: '{}'",
            (*(*a).ob_type).tp_name,
        )))
    }
}

// ── Subscript operations ─────────────────────────────────────────────

/// Get item by index: `obj[index]`.
///
/// Dispatches based on the type of `obj`.
pub fn py_getitem(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    unsafe {
        if is_list(obj) {
            if !is_int(index) {
                return Err(PyError::type_error("list indices must be integers"));
            }
            let idx = w_int_get_value(index);
            match w_list_getitem(obj, idx) {
                Some(val) => Ok(val),
                None => Err(PyError {
                    kind: PyErrorKind::IndexError,
                    message: "list index out of range".to_string(),
                }),
            }
        } else if is_tuple(obj) {
            if !is_int(index) {
                return Err(PyError::type_error("tuple indices must be integers"));
            }
            let idx = w_int_get_value(index);
            match w_tuple_getitem(obj, idx) {
                Some(val) => Ok(val),
                None => Err(PyError {
                    kind: PyErrorKind::IndexError,
                    message: "tuple index out of range".to_string(),
                }),
            }
        } else if is_dict(obj) {
            if !is_int(index) {
                return Err(PyError::type_error("dict keys must be integers in Phase 1"));
            }
            let key = w_int_get_value(index);
            match w_dict_getitem(obj, key) {
                Some(val) => Ok(val),
                None => Err(PyError {
                    kind: PyErrorKind::KeyError,
                    message: format!("{key}"),
                }),
            }
        } else {
            Err(PyError::type_error(format!(
                "'{}' object is not subscriptable",
                (*(*obj).ob_type).tp_name,
            )))
        }
    }
}

/// Set item by index: `obj[index] = value`.
pub fn py_setitem(obj: PyObjectRef, index: PyObjectRef, value: PyObjectRef) -> PyResult {
    unsafe {
        if is_list(obj) {
            if !is_int(index) {
                return Err(PyError::type_error("list indices must be integers"));
            }
            let idx = w_int_get_value(index);
            if w_list_setitem(obj, idx, value) {
                Ok(w_none())
            } else {
                Err(PyError {
                    kind: PyErrorKind::IndexError,
                    message: "list assignment index out of range".to_string(),
                })
            }
        } else if is_dict(obj) {
            if !is_int(index) {
                return Err(PyError::type_error("dict keys must be integers in Phase 1"));
            }
            let key = w_int_get_value(index);
            w_dict_setitem(obj, key, value);
            Ok(w_none())
        } else {
            Err(PyError::type_error(format!(
                "'{}' object does not support item assignment",
                (*(*obj).ob_type).tp_name,
            )))
        }
    }
}

/// Get the length of a container: `len(obj)`.
pub fn py_len(obj: PyObjectRef) -> PyResult {
    unsafe {
        if is_list(obj) {
            Ok(w_int_new(w_list_len(obj) as i64))
        } else if is_tuple(obj) {
            Ok(w_int_new(w_tuple_len(obj) as i64))
        } else if is_dict(obj) {
            Ok(w_int_new(w_dict_len(obj) as i64))
        } else if is_str(obj) {
            Ok(w_int_new(w_str_len(obj) as i64))
        } else {
            Err(PyError::type_error(format!(
                "object of type '{}' has no len()",
                (*(*obj).ob_type).tp_name,
            )))
        }
    }
}

// ── Attribute operations ──────────────────────────────────────────────

thread_local! {
    /// Side table mapping object addresses to their instance __dict__.
    ///
    /// Every object can have attributes stored here. This avoids modifying
    /// the repr(C) layout of existing object types.
    static ATTR_TABLE: RefCell<HashMap<usize, HashMap<String, PyObjectRef>>> =
        RefCell::new(HashMap::new());
}

/// Get an attribute from an object: `obj.name`.
///
/// Looks up the attribute in the per-object side table.
pub fn py_getattr(obj: PyObjectRef, name: &str) -> PyResult {
    ATTR_TABLE.with(|table| {
        let table = table.borrow();
        let key = obj as usize;
        if let Some(dict) = table.get(&key) {
            if let Some(&value) = dict.get(name) {
                return Ok(value);
            }
        }
        unsafe {
            Err(PyError {
                kind: PyErrorKind::AttributeError,
                message: format!(
                    "'{}' object has no attribute '{name}'",
                    (*(*obj).ob_type).tp_name,
                ),
            })
        }
    })
}

/// Set an attribute on an object: `obj.name = value`.
///
/// Stores the attribute in the per-object side table.
pub fn py_setattr(obj: PyObjectRef, name: &str, value: PyObjectRef) -> PyResult {
    ATTR_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        let key = obj as usize;
        table
            .entry(key)
            .or_default()
            .insert(name.to_string(), value);
    });
    Ok(w_none())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_add() {
        let a = w_int_new(3);
        let b = w_int_new(4);
        let result = py_add(a, b).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 7) };
    }

    #[test]
    fn test_int_compare() {
        let a = w_int_new(5);
        let b = w_int_new(10);
        let result = py_compare(a, b, CompareOp::Lt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_zero_division() {
        let a = w_int_new(5);
        let b = w_int_new(0);
        assert!(py_floordiv(a, b).is_err());
    }

    #[test]
    fn test_truthiness() {
        assert!(py_is_true(w_int_new(1)));
        assert!(!py_is_true(w_int_new(0)));
        assert!(!py_is_true(w_none()));
        assert!(py_is_true(w_bool_from(true)));
        assert!(!py_is_true(w_bool_from(false)));
    }

    #[test]
    fn test_int_add_overflow() {
        let a = w_int_new(i64::MAX);
        let b = w_int_new(1);
        let result = py_add(a, b).unwrap();
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
        let result = py_sub(a, b).unwrap();
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
        let result = py_mul(a, b).unwrap();
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
        let result = py_add(a, b).unwrap();
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
        let result = py_add(a, b).unwrap();
        unsafe {
            assert!(is_int(result));
            assert_eq!(w_int_get_value(result), i64::MAX);
        }
    }

    #[test]
    fn test_negate_min_int() {
        let a = w_int_new(i64::MIN);
        let result = py_negative(a).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), -BigInt::from(i64::MIN));
        }
    }

    #[test]
    fn test_invert_int() {
        let result = py_invert(w_int_new(6)).unwrap();
        unsafe {
            assert!(is_int(result));
            assert_eq!(w_int_get_value(result), !6);
        }
    }

    #[test]
    fn test_long_compare() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(i64::MAX);
        let result = py_compare(a, b, CompareOp::Gt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_long_truthiness() {
        assert!(py_is_true(w_long_new(
            BigInt::from(i64::MAX) + BigInt::from(1)
        )));
        assert!(!py_is_true(w_long_new(BigInt::from(0))));
    }

    #[test]
    fn test_int_pow() {
        let result = py_pow(w_int_new(2), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_pow_overflow() {
        let result = py_pow(w_int_new(2), w_int_new(63)).unwrap();
        unsafe {
            // 2^63 overflows i64, should be long
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(2).pow(63));
        }
    }

    #[test]
    fn test_int_pow_negative_exponent() {
        let result = py_pow(w_int_new(2), w_int_new(-1)).unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 0.5);
        }
    }

    #[test]
    fn test_int_lshift() {
        let result = py_lshift(w_int_new(1), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_lshift_overflow() {
        let result = py_lshift(w_int_new(1), w_int_new(64)).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(1) << 64);
        }
    }

    #[test]
    fn test_int_rshift() {
        let result = py_rshift(w_int_new(1024), w_int_new(3)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 128) };
    }

    #[test]
    fn test_negative_shift_count() {
        assert!(py_lshift(w_int_new(1), w_int_new(-1)).is_err());
        assert!(py_rshift(w_int_new(1), w_int_new(-1)).is_err());
    }

    #[test]
    fn test_int_bitand() {
        let result = py_bitand(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0x0F) };
    }

    #[test]
    fn test_int_bitor() {
        let result = py_bitor(w_int_new(0xF0), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xFF) };
    }

    #[test]
    fn test_int_bitxor() {
        let result = py_bitxor(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xF0) };
    }

    #[test]
    fn test_long_bitand() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(0xFF);
        let result = py_bitand(a, b).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0) };
    }

    #[test]
    fn test_invert_long() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let result = py_invert(a).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                !(BigInt::from(i64::MAX) + BigInt::from(1))
            );
        }
    }

    #[test]
    fn test_setattr_getattr() {
        let obj = w_int_new(42);
        py_setattr(obj, "name", w_int_new(100)).unwrap();
        let result = py_getattr(obj, "name").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 100) };
    }

    #[test]
    fn test_getattr_missing() {
        let obj = w_int_new(1);
        let err = py_getattr(obj, "missing").unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::AttributeError));
    }

    #[test]
    fn test_setattr_overwrite() {
        let obj = w_int_new(42);
        py_setattr(obj, "x", w_int_new(1)).unwrap();
        py_setattr(obj, "x", w_int_new(2)).unwrap();
        let result = py_getattr(obj, "x").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 2) };
    }

    #[test]
    fn test_py_contains_manual_list() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        let needle = w_int_new(1);
        unsafe {
            assert!(is_list(list), "should be list, got type: {}", (*(*list).ob_type).tp_name);
        }
        let result = super::py_contains(list, needle).expect("py_contains failed");
        assert!(result, "1 should be in [1, 2, 3]");
    }
}

/// `in` operator: check if `needle` is in `haystack`.
/// PyPy: space.contains_w(haystack, needle)
pub fn py_contains(haystack: PyObjectRef, needle: PyObjectRef) -> Result<bool, PyError> {
    use pyre_object::*;
    unsafe {
        if is_list(haystack) {
            let len = w_list_len(haystack);
            for i in 0..len {
                if let Some(item) = w_list_getitem(haystack, i as i64) {
                    if py_eq_bool(item, needle) {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        if is_tuple(haystack) {
            let len = w_tuple_len(haystack);
            for i in 0..len {
                if let Some(item) = w_tuple_getitem(haystack, i as i64) {
                    if py_eq_bool(item, needle) {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        if is_str(haystack) && is_str(needle) {
            let h = w_str_get_value(haystack);
            let n = w_str_get_value(needle);
            return Ok(h.contains(n));
        }
    }
    // Fallback: try iterating with py_getitem(obj, i) for i=0,1,...
    unsafe {
        let mut i = 0i64;
        loop {
            match py_getitem(haystack, pyre_object::w_int_new(i)) {
                Ok(item) => {
                    if py_eq_bool(item, needle) { return Ok(true); }
                    i += 1;
                }
                Err(_) => return Ok(false), // IndexError → not found
            }
        }
    }
}

/// Compare two objects for equality (returns bool, not PyObjectRef).
fn py_eq_bool(a: PyObjectRef, b: PyObjectRef) -> bool {
    if a == b {
        return true;
    }
    unsafe {
        use pyre_object::*;
        if is_int(a) && is_int(b) {
            return w_int_get_value(a) == w_int_get_value(b);
        }
        if is_str(a) && is_str(b) {
            return w_str_get_value(a) == w_str_get_value(b);
        }
    }
    py_compare(a, b, CompareOp::Eq)
        .map(|r| py_is_true(r))
        .unwrap_or(false)
}

/// Delete item: `del obj[index]`
pub fn py_delitem(obj: PyObjectRef, index: PyObjectRef) -> Result<(), PyError> {
    use pyre_object::*;
    unsafe {
        if is_list(obj) && is_int(index) {
            let i = w_int_get_value(index);
            let len = w_list_len(obj) as i64;
            let idx = if i < 0 { len + i } else { i };
            if idx >= 0 && idx < len {
                // For Phase 1: set to PY_NULL (proper removal needs list mutation API)
                w_list_setitem(obj, idx, PY_NULL);
                return Ok(());
            }
            return Err(PyError::type_error("list index out of range"));
        }
    }
    Err(PyError::type_error("object does not support item deletion"))
}

/// Convert object to string representation (str()).
pub fn py_str(obj: PyObjectRef) -> String {
    use pyre_object::*;
    unsafe {
        if is_str(obj) {
            return w_str_get_value(obj).to_string();
        }
        if is_int(obj) {
            return w_int_get_value(obj).to_string();
        }
        if is_none(obj) {
            return "None".to_string();
        }
        if is_bool(obj) {
            return if w_bool_get_value(obj) {
                "True"
            } else {
                "False"
            }
            .to_string();
        }
        if is_float(obj) {
            let v = w_float_get_value(obj);
            if v == v.floor() && v.is_finite() {
                return format!("{v:.1}");
            }
            return v.to_string();
        }
    }
    "<object>".to_string()
}

/// Convert object to repr string (repr()).
pub fn py_repr(obj: PyObjectRef) -> String {
    use pyre_object::*;
    unsafe {
        if is_str(obj) {
            return format!("'{}'", w_str_get_value(obj));
        }
        if is_int(obj) {
            return w_int_get_value(obj).to_string();
        }
        if is_none(obj) {
            return "None".to_string();
        }
        if is_bool(obj) {
            return if w_bool_get_value(obj) {
                "True"
            } else {
                "False"
            }
            .to_string();
        }
        if is_float(obj) {
            let v = w_float_get_value(obj);
            if v == v.floor() && v.is_finite() {
                return format!("{v:.1}");
            }
            return v.to_string();
        }
    }
    "<object>".to_string()
}
