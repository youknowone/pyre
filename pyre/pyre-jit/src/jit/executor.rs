//! Concrete execution for tracing — RPython `executor.execute()` parity.
//!
//! Each function computes a concrete result from concrete arguments,
//! without recording any IR operations. This mirrors RPython's executor
//! module which dispatches to `BlackholeInterpreter.bhimpl_*` methods
//! for concrete computation during tracing.

use super::state::ConcreteValue;

// ── Integer arithmetic (RPython bhimpl_int_*) ──

pub fn int_add(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_add(y)),
        _ => ConcreteValue::Null,
    }
}

pub fn int_sub(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_sub(y)),
        _ => ConcreteValue::Null,
    }
}

pub fn int_mul(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_mul(y)),
        _ => ConcreteValue::Null,
    }
}

pub fn int_floordiv(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) if y != 0 => {
            let d = x.wrapping_div(y);
            ConcreteValue::Int(if (x ^ y) < 0 && d * y != x { d - 1 } else { d })
        }
        _ => ConcreteValue::Null,
    }
}

pub fn int_mod(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) if y != 0 => ConcreteValue::Int(((x % y) + y) % y),
        _ => ConcreteValue::Null,
    }
}

pub fn int_and(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x & y),
        _ => ConcreteValue::Null,
    }
}

pub fn int_or(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x | y),
        _ => ConcreteValue::Null,
    }
}

pub fn int_xor(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x ^ y),
        _ => ConcreteValue::Null,
    }
}

pub fn int_lshift(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_shl(y as u32)),
        _ => ConcreteValue::Null,
    }
}

pub fn int_rshift(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_shr(y as u32)),
        _ => ConcreteValue::Null,
    }
}

pub fn int_neg(a: ConcreteValue) -> ConcreteValue {
    match a.getint() {
        Some(x) => ConcreteValue::Int(x.wrapping_neg()),
        _ => ConcreteValue::Null,
    }
}

pub fn int_invert(a: ConcreteValue) -> ConcreteValue {
    match a.getint() {
        Some(x) => ConcreteValue::Int(!x),
        _ => ConcreteValue::Null,
    }
}

// ── Integer comparison (RPython bhimpl_int_lt etc.) ──

pub fn int_lt(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x < y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn int_le(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x <= y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn int_eq(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x == y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn int_ne(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x != y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn int_gt(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x > y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn int_ge(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x >= y) as i64),
        _ => ConcreteValue::Null,
    }
}

// ── Float arithmetic (RPython bhimpl_float_*) ──

pub fn float_add(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Float(x + y),
        _ => ConcreteValue::Null,
    }
}

pub fn float_sub(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Float(x - y),
        _ => ConcreteValue::Null,
    }
}

pub fn float_mul(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Float(x * y),
        _ => ConcreteValue::Null,
    }
}

pub fn float_truediv(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) if y != 0.0 => ConcreteValue::Float(x / y),
        _ => ConcreteValue::Null,
    }
}

pub fn float_floordiv(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) if y != 0.0 => ConcreteValue::Float((x / y).floor()),
        _ => ConcreteValue::Null,
    }
}

pub fn float_mod(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) if y != 0.0 => ConcreteValue::Float(x % y),
        _ => ConcreteValue::Null,
    }
}

// ── Float comparison ──

pub fn float_lt(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x < y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn float_le(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x <= y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn float_eq(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x == y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn float_ne(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x != y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn float_gt(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x > y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn float_ge(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x >= y) as i64),
        _ => ConcreteValue::Null,
    }
}
