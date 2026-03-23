//! Concrete execution for tracing — RPython `executor.execute()` parity.
//!
//! Each function computes a concrete result from concrete arguments,
//! without recording any IR operations. This mirrors RPython's executor
//! module which dispatches to `BlackholeInterpreter.bhimpl_*` methods
//! for concrete computation during tracing.

use super::state::ConcreteValue;

// ── Integer arithmetic (RPython bhimpl_int_*) ──

pub fn bhimpl_int_add(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_add(y)),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_sub(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_sub(y)),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_mul(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_mul(y)),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_floordiv(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) if y != 0 => {
            let d = x.wrapping_div(y);
            ConcreteValue::Int(if (x ^ y) < 0 && d * y != x { d - 1 } else { d })
        }
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_mod(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) if y != 0 => ConcreteValue::Int(((x % y) + y) % y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_and(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x & y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_or(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x | y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_xor(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x ^ y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_lshift(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_shl(y as u32)),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_rshift(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int(x.wrapping_shr(y as u32)),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_neg(a: ConcreteValue) -> ConcreteValue {
    match a.getint() {
        Some(x) => ConcreteValue::Int(x.wrapping_neg()),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_invert(a: ConcreteValue) -> ConcreteValue {
    match a.getint() {
        Some(x) => ConcreteValue::Int(!x),
        _ => ConcreteValue::Null,
    }
}

// ── Integer comparison (RPython bhimpl_int_lt etc.) ──

pub fn bhimpl_int_lt(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x < y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_le(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x <= y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_eq(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x == y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_ne(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x != y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_gt(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x > y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_int_ge(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getint(), b.getint()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x >= y) as i64),
        _ => ConcreteValue::Null,
    }
}

// ── Float arithmetic (RPython bhimpl_float_*) ──

pub fn bhimpl_float_add(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Float(x + y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_sub(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Float(x - y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_mul(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Float(x * y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_truediv(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) if y != 0.0 => ConcreteValue::Float(x / y),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_floordiv(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) if y != 0.0 => ConcreteValue::Float((x / y).floor()),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_mod(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) if y != 0.0 => ConcreteValue::Float(x % y),
        _ => ConcreteValue::Null,
    }
}

// ── Float comparison ──

pub fn bhimpl_float_lt(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x < y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_le(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x <= y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_eq(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x == y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_ne(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x != y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_gt(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x > y) as i64),
        _ => ConcreteValue::Null,
    }
}

pub fn bhimpl_float_ge(a: ConcreteValue, b: ConcreteValue) -> ConcreteValue {
    match (a.getfloat(), b.getfloat()) {
        (Some(x), Some(y)) => ConcreteValue::Int((x >= y) as i64),
        _ => ConcreteValue::Null,
    }
}

// ── Unified dispatch (RPython executor.execute() parity) ──

/// Dispatch concrete execution by opcode.
/// RPython: executor.execute(cpu, metainterp, opnum, descr, *argboxes)
pub fn execute_opcode(opcode: majit_ir::OpCode, args: &[ConcreteValue]) -> ConcreteValue {
    use majit_ir::OpCode;
    match opcode {
        // Integer arithmetic
        OpCode::IntAdd | OpCode::IntAddOvf => {
            if args.len() >= 2 { bhimpl_int_add(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntSub | OpCode::IntSubOvf => {
            if args.len() >= 2 { bhimpl_int_sub(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntMul | OpCode::IntMulOvf => {
            if args.len() >= 2 { bhimpl_int_mul(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntFloorDiv => {
            if args.len() >= 2 { bhimpl_int_floordiv(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntMod => {
            if args.len() >= 2 { bhimpl_int_mod(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntAnd => {
            if args.len() >= 2 { bhimpl_int_and(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntOr => {
            if args.len() >= 2 { bhimpl_int_or(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntXor => {
            if args.len() >= 2 { bhimpl_int_xor(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntLshift => {
            if args.len() >= 2 { bhimpl_int_lshift(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntRshift => {
            if args.len() >= 2 { bhimpl_int_rshift(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntNeg => {
            if args.len() >= 1 { bhimpl_int_neg(args[0]) } else { ConcreteValue::Null }
        }
        OpCode::IntInvert => {
            if args.len() >= 1 { bhimpl_int_invert(args[0]) } else { ConcreteValue::Null }
        }
        // Integer comparison
        OpCode::IntLt => {
            if args.len() >= 2 { bhimpl_int_lt(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntLe => {
            if args.len() >= 2 { bhimpl_int_le(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntEq => {
            if args.len() >= 2 { bhimpl_int_eq(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntNe => {
            if args.len() >= 2 { bhimpl_int_ne(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntGt => {
            if args.len() >= 2 { bhimpl_int_gt(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::IntGe => {
            if args.len() >= 2 { bhimpl_int_ge(args[0], args[1]) } else { ConcreteValue::Null }
        }
        // Float arithmetic
        OpCode::FloatAdd => {
            if args.len() >= 2 { bhimpl_float_add(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::FloatSub => {
            if args.len() >= 2 { bhimpl_float_sub(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::FloatMul => {
            if args.len() >= 2 { bhimpl_float_mul(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::FloatTrueDiv => {
            if args.len() >= 2 { bhimpl_float_truediv(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::FloatFloorDiv => {
            if args.len() >= 2 { bhimpl_float_floordiv(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::FloatMod => {
            if args.len() >= 2 { bhimpl_float_mod(args[0], args[1]) } else { ConcreteValue::Null }
        }
        // Float comparison
        OpCode::FloatLt => {
            if args.len() >= 2 { bhimpl_float_lt(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::FloatLe => {
            if args.len() >= 2 { bhimpl_float_le(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::FloatEq => {
            if args.len() >= 2 { bhimpl_float_eq(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::FloatNe => {
            if args.len() >= 2 { bhimpl_float_ne(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::FloatGt => {
            if args.len() >= 2 { bhimpl_float_gt(args[0], args[1]) } else { ConcreteValue::Null }
        }
        OpCode::FloatGe => {
            if args.len() >= 2 { bhimpl_float_ge(args[0], args[1]) } else { ConcreteValue::Null }
        }
        // Unknown opcode — no concrete result
        _ => ConcreteValue::Null,
    }
}
