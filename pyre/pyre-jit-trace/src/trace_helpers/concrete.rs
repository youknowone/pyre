//! Concrete-value computation performed during tracing — the analog of
//! `executor.py` plus `floatobject.py` / `intobject.py`'s `descr_*`
//! arithmetic. Computes the shadow Python result alongside the emitted IR.

use super::*;
use pyre_interpreter::bytecode::{BinaryOperator, ComparisonOperator};

/// Compute concrete float binary operation result.
/// Returns None for unsupported ops or division by zero.
///
/// floatobject.py: descr_add/sub/mul/truediv/floordiv/mod/pow
pub fn concrete_float_binop(op: BinaryOperator, lhs: f64, rhs: f64) -> Option<f64> {
    match op {
        BinaryOperator::Add | BinaryOperator::InplaceAdd => Some(lhs + rhs),
        BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => Some(lhs - rhs),
        BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => Some(lhs * rhs),
        BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide if rhs != 0.0 => {
            Some(lhs / rhs)
        }
        // floatobject.py:508: descr_floordiv → _divmod_w()[0]
        BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide if rhs != 0.0 => {
            Some(float_divmod_w(lhs, rhs).0)
        }
        // floatobject.py:520-540: descr_mod → fmod + sign correction + copysign
        BinaryOperator::Remainder | BinaryOperator::InplaceRemainder if rhs != 0.0 => {
            Some(float_mod(lhs, rhs))
        }
        // floatobject.py:561 descr_pow → _pow()
        BinaryOperator::Power | BinaryOperator::InplacePower => float_pow(lhs, rhs),
        _ => None,
    }
}

/// floatobject.py:520-540: float modulo with sign correction.
/// Uses fmod + denominator sign correction + copysign for zero remainder.
fn float_mod(x: f64, y: f64) -> f64 {
    let mut m = x % y; // C fmod semantics (same as Rust %)
    if m != 0.0 {
        // ensure the remainder has the same sign as the denominator
        if (y < 0.0) != (m < 0.0) {
            m += y;
        }
    } else {
        // the remainder is zero: ensure it has the same sign as the denominator
        m = f64::copysign(0.0, y);
    }
    m
}

/// floatobject.py:758-793: _divmod_w — returns (floordiv, mod).
fn float_divmod_w(x: f64, y: f64) -> (f64, f64) {
    let mut m = x % y; // fmod
    let mut div = (x - m) / y;
    if m != 0.0 {
        if (y < 0.0) != (m < 0.0) {
            m += y;
            div -= 1.0;
        }
    } else {
        m = m * m; // hide from optimizer
        if y < 0.0 {
            m = -m;
        }
    }
    // snap quotient to nearest integral value
    let floordiv = if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 { f + 1.0 } else { f }
    } else {
        let d = div * div; // hide from optimizer
        d * x / y
    };
    (floordiv, m)
}

/// floatobject.py:799-881: _pow with special-case handling.
/// Returns None for domain errors (negative base with fractional exponent).
fn float_pow(x: f64, y: f64) -> Option<f64> {
    if y == 2.0 {
        return Some(x * x);
    }
    if y == 0.0 {
        return Some(1.0);
    }
    if x.is_nan() {
        return Some(x);
    }
    if y.is_nan() {
        return Some(if x == 1.0 { 1.0 } else { y });
    }
    if y.is_infinite() {
        let ax = x.abs();
        if ax == 1.0 {
            return Some(1.0);
        } else if (y > 0.0) == (ax > 1.0) {
            return Some(f64::INFINITY);
        } else {
            return Some(0.0);
        }
    }
    if x.is_infinite() {
        let y_is_odd = y.abs() % 2.0 == 1.0;
        if y > 0.0 {
            return Some(if y_is_odd { x } else { x.abs() });
        } else {
            return Some(if y_is_odd { f64::copysign(0.0, x) } else { 0.0 });
        }
    }
    if x == 0.0 && y < 0.0 {
        return None; // ZeroDivisionError
    }
    let mut negate_result = false;
    let mut bx = x;
    if bx < 0.0 {
        if y.floor() != y {
            return None; // PowDomainError → ValueError
        }
        bx = -bx;
        negate_result = y.abs() % 2.0 == 1.0;
    }
    if bx == 1.0 {
        return Some(if negate_result { -1.0 } else { 1.0 });
    }
    let z = bx.powf(y);
    if z.is_infinite() || z.is_nan() {
        return None;
    }
    Some(if negate_result { -z } else { z })
}

/// Compute concrete int binary operation result.
/// Returns None for overflow, division by zero, or unsupported ops.
pub fn concrete_int_binop(op: BinaryOperator, lhs: i64, rhs: i64) -> Option<i64> {
    match op {
        BinaryOperator::Add | BinaryOperator::InplaceAdd => lhs.checked_add(rhs),
        BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => lhs.checked_sub(rhs),
        BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => lhs.checked_mul(rhs),
        BinaryOperator::Remainder | BinaryOperator::InplaceRemainder if rhs != 0 => {
            Some(((lhs % rhs) + rhs) % rhs)
        }
        BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide if rhs != 0 => {
            lhs.checked_div(rhs).map(|d| {
                if (lhs ^ rhs) < 0 && d * rhs != lhs {
                    d - 1
                } else {
                    d
                }
            })
        }
        BinaryOperator::And | BinaryOperator::InplaceAnd => Some(lhs & rhs),
        BinaryOperator::Or | BinaryOperator::InplaceOr => Some(lhs | rhs),
        BinaryOperator::Xor | BinaryOperator::InplaceXor => Some(lhs ^ rhs),
        BinaryOperator::Lshift | BinaryOperator::InplaceLshift => {
            // intobject.py:205: negative shift → ValueError
            if rhs < 0 {
                return None;
            }
            let shift = rhs as u32;
            if shift >= 64 {
                return None;
            }
            let r = lhs.wrapping_shl(shift);
            if r.wrapping_shr(shift) != lhs {
                None
            } else {
                Some(r)
            }
        }
        BinaryOperator::Rshift | BinaryOperator::InplaceRshift => {
            // intobject.py:224: negative shift → ValueError("negative shift count")
            if rhs < 0 {
                return None;
            }
            let shift = rhs as u32;
            if shift >= 64 {
                Some(if lhs < 0 { -1 } else { 0 })
            } else {
                Some(lhs >> shift)
            }
        }
        _ => None,
    }
}

/// Concrete binary computation on Python objects.
///
/// Delegates to baseobjspace dispatch which handles:
///   is_int × is_int → int_add (checked, overflow → long)
///   is_int_or_long × is_int_or_long → long_add (BigInt)
///   is_float_pair → float_add (as_float coercion: int|float|long → f64)
///   str, list, tuple, bytearray, dunder dispatch
pub fn concrete_binary_value(
    op: BinaryOperator,
    lhs_obj: pyre_object::PyObjectRef,
    rhs_obj: pyre_object::PyObjectRef,
) -> crate::state::ConcreteValue {
    use crate::state::ConcreteValue;
    if lhs_obj.is_null() || rhs_obj.is_null() {
        return ConcreteValue::Null;
    }
    // Delegate to the interpreter's baseobjspace dispatch.
    // This handles all type combinations correctly including long,
    // overflow promotion, str/list/tuple concat, dunder methods.
    let result = pyre_interpreter::opcode_ops::binary_value(lhs_obj, rhs_obj, op);
    match result {
        Ok(r) => ConcreteValue::from_pyobj(r),
        Err(_) => ConcreteValue::Null,
    }
}

/// Concrete comparison computation on Python objects.
///
/// Delegates to baseobjspace::compare which handles:
///   is_int × is_int → int comparison
///   is_int_or_long × is_int_or_long → BigInt comparison
///   is_float_pair → float comparison (as_float coercion)
///   str comparison, dunder dispatch
pub fn concrete_compare_value(
    op: ComparisonOperator,
    lhs_obj: pyre_object::PyObjectRef,
    rhs_obj: pyre_object::PyObjectRef,
) -> crate::state::ConcreteValue {
    use crate::state::ConcreteValue;
    if lhs_obj.is_null() || rhs_obj.is_null() {
        return ConcreteValue::Null;
    }
    let result = pyre_interpreter::opcode_ops::compare_value(lhs_obj, rhs_obj, op);
    match result {
        Ok(r) => ConcreteValue::from_pyobj(r),
        Err(_) => ConcreteValue::Null,
    }
}
