//! Blackhole interpreter: evaluates IR operations with concrete values.
//!
//! When a guard fails in compiled code, the blackhole interpreter can replay
//! the remaining operations from the guard point to the end of the trace,
//! using concrete values from the DeadFrame.
//!
//! This is the RPython equivalent of `rpython/jit/metainterp/blackhole.py`.

use std::collections::HashMap;

use crate::resume::{MaterializedVirtual, ResolvedPendingFieldWrite, ResumeData};
use majit_ir::{Op, OpCode, OpRef};

/// Trait for blackhole memory access: the interpreter supplies
/// concrete load/store implementations so that the blackhole can
/// actually execute field access ops instead of returning placeholders.
///
/// Mirrors RPython's `_execute_*` methods in `blackhole.py` that
/// delegate to the CPU's raw memory operations.
pub trait BlackholeMemory {
    /// Load a GC-managed field from `base + offset`.
    fn gc_load_i(&self, base: i64, offset: i64) -> i64 {
        let _ = (base, offset);
        0
    }
    /// Load a GC ref field.
    fn gc_load_r(&self, base: i64, offset: i64) -> i64 {
        let _ = (base, offset);
        0
    }
    /// Load a float field (returned as bits).
    fn gc_load_f(&self, base: i64, offset: i64) -> i64 {
        let _ = (base, offset);
        0
    }
    /// Store an int value into a GC object field.
    fn gc_store(&self, base: i64, offset: i64, value: i64) {
        let _ = (base, offset, value);
    }
    /// Load from array at `base + index * scale + offset`.
    fn gc_load_indexed_i(&self, base: i64, index: i64, scale: i64, offset: i64) -> i64 {
        let _ = (base, index, scale, offset);
        0
    }
    fn gc_load_indexed_r(&self, base: i64, index: i64, scale: i64, offset: i64) -> i64 {
        let _ = (base, index, scale, offset);
        0
    }
    fn gc_load_indexed_f(&self, base: i64, index: i64, scale: i64, offset: i64) -> i64 {
        let _ = (base, index, scale, offset);
        0
    }
    /// Store into array at `base + index * scale + offset`.
    fn gc_store_indexed(&self, base: i64, index: i64, scale: i64, offset: i64, value: i64) {
        let _ = (base, index, scale, offset, value);
    }
    /// Get array length.
    fn arraylen(&self, base: i64) -> i64 {
        let _ = base;
        0
    }
    /// Get string length.
    fn strlen(&self, base: i64) -> i64 {
        let _ = base;
        0
    }
    /// Call a function pointer with integer args, returning an integer.
    fn call_i(&self, func: i64, args: &[i64]) -> i64 {
        let _ = (func, args);
        0
    }
    /// Call a function pointer returning void.
    fn call_n(&self, func: i64, args: &[i64]) {
        let _ = (func, args);
    }
}

/// Default no-op memory implementation (returns 0 for all loads).
pub struct DefaultBlackholeMemory;
impl BlackholeMemory for DefaultBlackholeMemory {}

/// Exception state tracked during blackhole execution.
///
/// Mirrors RPython's exception tracking in the meta-interpreter.
/// Guards like GUARD_EXCEPTION and GUARD_NO_EXCEPTION check this state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExceptionState {
    /// The exception class pointer (0 = no exception pending).
    pub exc_class: i64,
    /// The exception value pointer.
    pub exc_value: i64,
}

impl ExceptionState {
    /// Whether an exception is currently pending.
    pub fn is_pending(&self) -> bool {
        self.exc_class != 0
    }

    /// Set a pending exception.
    pub fn set(&mut self, exc_class: i64, exc_value: i64) {
        self.exc_class = exc_class;
        self.exc_value = exc_value;
    }

    /// Clear the pending exception and return (class, value).
    pub fn clear(&mut self) -> (i64, i64) {
        let cls = self.exc_class;
        let val = self.exc_value;
        self.exc_class = 0;
        self.exc_value = 0;
        (cls, val)
    }
}

/// Result of blackhole execution.
pub enum BlackholeResult {
    /// Reached a Finish operation with output values.
    Finish(Vec<i64>),
    /// Reached a Jump — loop back to header with these values.
    Jump(Vec<i64>),
    /// A guard failed during blackhole execution.
    GuardFailed {
        guard_index: usize,
        fail_values: Vec<i64>,
    },
    /// A guard failed and resume-data virtuals were materialized for recovery.
    GuardFailedWithVirtuals {
        guard_index: usize,
        fail_values: Vec<i64>,
        materialized_virtuals: Vec<MaterializedVirtual>,
        pending_field_writes: Vec<ResolvedPendingFieldWrite>,
    },
    /// Abort: encountered an unhandled operation.
    Abort(String),
}

/// Evaluate IR operations sequentially with concrete i64 values.
///
/// `values` maps OpRef indices to their concrete values.
/// `constants` maps constant pool indices to values.
/// `ops` is the sequence of IR operations to evaluate.
/// `start_index` is the index in `ops` to start execution from.
pub fn blackhole_execute(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
) -> BlackholeResult {
    blackhole_execute_with_exception(
        ops,
        constants,
        initial_values,
        start_index,
        ExceptionState::default(),
    )
}

/// Evaluate with a concrete memory backend for load/store operations.
pub fn blackhole_execute_with_memory(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    memory: &dyn BlackholeMemory,
) -> BlackholeResult {
    blackhole_execute_full(
        ops,
        constants,
        initial_values,
        start_index,
        ExceptionState::default(),
        memory,
    )
    .0
}

fn blackhole_execute_full(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    initial_exception: ExceptionState,
    memory: &dyn BlackholeMemory,
) -> (BlackholeResult, ExceptionState) {
    let mut values: HashMap<u32, i64> = initial_values.clone();
    let mut exc_state = initial_exception;

    for (&k, &v) in constants {
        values.entry(k).or_insert(v);
    }

    for op_idx in start_index..ops.len() {
        let op = &ops[op_idx];
        let result = execute_one_with_memory(op, &values, &mut exc_state, memory);

        match result {
            OpResult::Value(v) => {
                if !op.pos.is_none() {
                    values.insert(op.pos.0, v);
                }
            }
            OpResult::Void => {}
            OpResult::Finish(args) => {
                let vals: Vec<i64> = args.iter().map(|&r| resolve(&values, r)).collect();
                return (BlackholeResult::Finish(vals), exc_state);
            }
            OpResult::Jump(args) => {
                let vals: Vec<i64> = args.iter().map(|&r| resolve(&values, r)).collect();
                return (BlackholeResult::Jump(vals), exc_state);
            }
            OpResult::GuardFailed => {
                let fail_values = if let Some(ref fail_args) = op.fail_args {
                    fail_args.iter().map(|&r| resolve(&values, r)).collect()
                } else {
                    vec![]
                };
                return (
                    BlackholeResult::GuardFailed {
                        guard_index: op_idx,
                        fail_values,
                    },
                    exc_state,
                );
            }
            OpResult::Unsupported(msg) => {
                return (BlackholeResult::Abort(msg), exc_state);
            }
        }
    }

    (
        BlackholeResult::Abort("reached end of ops without finish/jump".to_string()),
        exc_state,
    )
}

/// Dispatch an op using real memory access when a BlackholeMemory backend is available.
fn execute_one_with_memory(
    op: &Op,
    values: &HashMap<u32, i64>,
    exc: &mut ExceptionState,
    memory: &dyn BlackholeMemory,
) -> OpResult {
    match op.opcode {
        // Memory access ops with real backend
        OpCode::GcLoadI => {
            let base = resolve(values, op.args[0]);
            let offset = resolve(values, op.args[1]);
            OpResult::Value(memory.gc_load_i(base, offset))
        }
        OpCode::GcLoadR => {
            let base = resolve(values, op.args[0]);
            let offset = resolve(values, op.args[1]);
            OpResult::Value(memory.gc_load_r(base, offset))
        }
        OpCode::GcLoadF => {
            let base = resolve(values, op.args[0]);
            let offset = resolve(values, op.args[1]);
            OpResult::Value(memory.gc_load_f(base, offset))
        }
        OpCode::GcStore => {
            let base = resolve(values, op.args[0]);
            let offset = resolve(values, op.args[1]);
            let value = resolve(values, op.args[2]);
            memory.gc_store(base, offset, value);
            OpResult::Void
        }
        OpCode::GcLoadIndexedI => {
            let base = resolve(values, op.args[0]);
            let index = resolve(values, op.args[1]);
            let scale = resolve(values, op.args[2]);
            let offset = resolve(values, op.args[3]);
            OpResult::Value(memory.gc_load_indexed_i(base, index, scale, offset))
        }
        OpCode::GcLoadIndexedR => {
            let base = resolve(values, op.args[0]);
            let index = resolve(values, op.args[1]);
            let scale = resolve(values, op.args[2]);
            let offset = resolve(values, op.args[3]);
            OpResult::Value(memory.gc_load_indexed_r(base, index, scale, offset))
        }
        OpCode::GcLoadIndexedF => {
            let base = resolve(values, op.args[0]);
            let index = resolve(values, op.args[1]);
            let scale = resolve(values, op.args[2]);
            let offset = resolve(values, op.args[3]);
            OpResult::Value(memory.gc_load_indexed_f(base, index, scale, offset))
        }
        OpCode::GcStoreIndexed => {
            let base = resolve(values, op.args[0]);
            let index = resolve(values, op.args[1]);
            let value = resolve(values, op.args[2]);
            let scale = resolve(values, op.args[3]);
            let offset = resolve(values, op.args[4]);
            memory.gc_store_indexed(base, index, scale, offset, value);
            OpResult::Void
        }
        OpCode::ArraylenGc => {
            let base = resolve(values, op.args[0]);
            OpResult::Value(memory.arraylen(base))
        }
        OpCode::Strlen | OpCode::Unicodelen => {
            let base = resolve(values, op.args[0]);
            OpResult::Value(memory.strlen(base))
        }
        // Call with real dispatch
        OpCode::CallI | OpCode::CallPureI => {
            let func = resolve(values, op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| resolve(values, r)).collect();
            OpResult::Value(memory.call_i(func, &args))
        }
        OpCode::CallN | OpCode::CallPureN => {
            let func = resolve(values, op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| resolve(values, r)).collect();
            memory.call_n(func, &args);
            OpResult::Void
        }
        // Fall through to the default execute_one for everything else
        _ => execute_one(op, values, exc),
    }
}

pub(crate) fn blackhole_execute_with_state(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    initial_exception: ExceptionState,
) -> (BlackholeResult, ExceptionState) {
    let mut values: HashMap<u32, i64> = initial_values.clone();
    let mut exc_state = initial_exception;

    // Merge constants into values
    for (&k, &v) in constants {
        values.entry(k).or_insert(v);
    }

    for op_idx in start_index..ops.len() {
        let op = &ops[op_idx];
        let result = execute_one(op, &values, &mut exc_state);

        match result {
            OpResult::Value(v) => {
                if !op.pos.is_none() {
                    values.insert(op.pos.0, v);
                }
            }
            OpResult::Void => {}
            OpResult::Finish(args) => {
                let vals: Vec<i64> = args.iter().map(|&r| resolve(&values, r)).collect();
                return (BlackholeResult::Finish(vals), exc_state);
            }
            OpResult::Jump(args) => {
                let vals: Vec<i64> = args.iter().map(|&r| resolve(&values, r)).collect();
                return (BlackholeResult::Jump(vals), exc_state);
            }
            OpResult::GuardFailed => {
                let fail_values = if let Some(ref fail_args) = op.fail_args {
                    fail_args.iter().map(|&r| resolve(&values, r)).collect()
                } else {
                    vec![]
                };
                return (
                    BlackholeResult::GuardFailed {
                        guard_index: op_idx,
                        fail_values,
                    },
                    exc_state,
                );
            }
            OpResult::Unsupported(msg) => {
                return (BlackholeResult::Abort(msg), exc_state);
            }
        }
    }

    (
        BlackholeResult::Abort("reached end of ops without finish/jump".to_string()),
        exc_state,
    )
}

/// Evaluate IR operations sequentially with concrete i64 values and an
/// already-pending exception state.
pub fn blackhole_execute_with_exception(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    initial_exception: ExceptionState,
) -> BlackholeResult {
    blackhole_execute_with_state(
        ops,
        constants,
        initial_values,
        start_index,
        initial_exception,
    )
    .0
}

fn resolve(values: &HashMap<u32, i64>, opref: OpRef) -> i64 {
    values.get(&opref.0).copied().unwrap_or(0)
}

enum OpResult {
    Value(i64),
    Void,
    Finish(Vec<OpRef>),
    Jump(Vec<OpRef>),
    GuardFailed,
    Unsupported(String),
}

fn execute_one(op: &Op, values: &HashMap<u32, i64>, exc: &mut ExceptionState) -> OpResult {
    match op.opcode {
        // ── Control flow ──
        OpCode::Label => OpResult::Void,
        OpCode::Finish => OpResult::Finish(op.args.to_vec()),
        OpCode::Jump => OpResult::Jump(op.args.to_vec()),

        // ── Integer arithmetic ──
        OpCode::IntAdd => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_add(b))
        }
        OpCode::IntSub => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_sub(b))
        }
        OpCode::IntMul => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_mul(b))
        }
        OpCode::IntFloorDiv => {
            let (a, b) = binop(values, op);
            if b == 0 {
                OpResult::Value(0)
            } else {
                OpResult::Value(a.wrapping_div(b))
            }
        }
        OpCode::IntMod => {
            let (a, b) = binop(values, op);
            if b == 0 {
                OpResult::Value(0)
            } else {
                OpResult::Value(a.wrapping_rem(b))
            }
        }
        OpCode::IntAnd => {
            let (a, b) = binop(values, op);
            OpResult::Value(a & b)
        }
        OpCode::IntOr => {
            let (a, b) = binop(values, op);
            OpResult::Value(a | b)
        }
        OpCode::IntXor => {
            let (a, b) = binop(values, op);
            OpResult::Value(a ^ b)
        }
        OpCode::IntLshift => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_shl(b as u32))
        }
        OpCode::IntRshift => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_shr(b as u32))
        }
        OpCode::UintRshift => {
            let (a, b) = binop(values, op);
            OpResult::Value((a as u64).wrapping_shr(b as u32) as i64)
        }
        OpCode::IntNeg => {
            let a = unop(values, op);
            OpResult::Value(a.wrapping_neg())
        }
        OpCode::IntInvert => {
            let a = unop(values, op);
            OpResult::Value(!a)
        }
        OpCode::IntSignext => {
            let (a, b) = binop(values, op);
            // Sign extend from b bytes to i64
            let bits = b * 8;
            let shift = 64 - bits;
            OpResult::Value((a << shift) >> shift)
        }

        // ── Integer comparisons ──
        OpCode::IntLt => {
            let (a, b) = binop(values, op);
            OpResult::Value((a < b) as i64)
        }
        OpCode::IntLe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a <= b) as i64)
        }
        OpCode::IntGe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a >= b) as i64)
        }
        OpCode::IntGt => {
            let (a, b) = binop(values, op);
            OpResult::Value((a > b) as i64)
        }
        OpCode::IntEq => {
            let (a, b) = binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::IntNe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a != b) as i64)
        }
        OpCode::UintLt => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64) < (b as u64)) as i64)
        }
        OpCode::UintLe => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64) <= (b as u64)) as i64)
        }
        OpCode::UintGe => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64) >= (b as u64)) as i64)
        }
        OpCode::UintGt => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64) > (b as u64)) as i64)
        }
        OpCode::IntIsZero => {
            let a = unop(values, op);
            OpResult::Value((a == 0) as i64)
        }
        OpCode::IntIsTrue => {
            let a = unop(values, op);
            OpResult::Value((a != 0) as i64)
        }
        OpCode::IntForceGeZero => {
            let a = unop(values, op);
            OpResult::Value(a.max(0))
        }
        OpCode::IntBetween => {
            // int_between(a, b, c) => a <= b < c
            let a = resolve(values, op.args[0]);
            let b = resolve(values, op.args[1]);
            let c = resolve(values, op.args[2]);
            OpResult::Value((a <= b && b < c) as i64)
        }

        // ── Float operations ──
        OpCode::FloatAdd => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a + b) as i64)
        }
        OpCode::FloatSub => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a - b) as i64)
        }
        OpCode::FloatMul => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a * b) as i64)
        }
        OpCode::FloatTrueDiv => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a / b) as i64)
        }
        OpCode::FloatNeg => {
            let a = float_unop(values, op);
            OpResult::Value(f64::to_bits(-a) as i64)
        }
        OpCode::FloatAbs => {
            let a = float_unop(values, op);
            OpResult::Value(f64::to_bits(a.abs()) as i64)
        }
        OpCode::CastFloatToInt => {
            let a = float_unop(values, op);
            OpResult::Value(a as i64)
        }
        OpCode::CastIntToFloat => {
            let a = unop(values, op);
            OpResult::Value(f64::to_bits(a as f64) as i64)
        }

        // ── Guards ──
        OpCode::GuardTrue | OpCode::VecGuardTrue => {
            let a = unop(values, op);
            if a != 0 {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardFalse | OpCode::VecGuardFalse => {
            let a = unop(values, op);
            if a == 0 {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardValue => {
            let (a, b) = binop(values, op);
            if a == b {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardNonnull => {
            let a = unop(values, op);
            if a != 0 {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardIsnull => {
            let a = unop(values, op);
            if a == 0 {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardClass
        | OpCode::GuardNonnullClass
        | OpCode::GuardSubclass
        | OpCode::GuardCompatible => {
            let (a, b) = binop(values, op);
            if a == b {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }
        OpCode::GuardNoOverflow => {
            // In blackhole, overflow checks are moot (we use wrapping)
            OpResult::Void
        }
        OpCode::GuardOverflow => OpResult::Void,
        OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
            // In blackhole, check if a call set an exception (simulated force).
            if exc.is_pending() {
                OpResult::GuardFailed
            } else {
                OpResult::Void
            }
        }
        OpCode::GuardNotInvalidated | OpCode::GuardFutureCondition => OpResult::Void,
        OpCode::GuardAlwaysFails => OpResult::GuardFailed,
        OpCode::GuardNoException => {
            if exc.is_pending() {
                OpResult::GuardFailed
            } else {
                OpResult::Void
            }
        }
        OpCode::GuardException => {
            // Guard expects an exception of a specific class.
            // arg(0) is the expected exception class.
            if exc.is_pending() {
                let expected_class = resolve(values, op.args[0]);
                if exc.exc_class == expected_class {
                    // Match — return the exception value and clear exception state.
                    let (_, val) = exc.clear();
                    return OpResult::Value(val);
                }
            }
            OpResult::GuardFailed
        }
        OpCode::GuardGcType => {
            // In blackhole, skip (no GC type checking)
            OpResult::Void
        }
        OpCode::GuardIsObject => {
            let a = unop(values, op);
            if a != 0 {
                OpResult::Void
            } else {
                OpResult::GuardFailed
            }
        }

        // ── SameAs / Copy ──
        OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF => {
            let a = unop(values, op);
            OpResult::Value(a)
        }

        // ── No-op markers ──
        OpCode::Keepalive
        | OpCode::ForceSpill
        | OpCode::VirtualRefFinish
        | OpCode::RecordExactClass
        | OpCode::RecordExactValueR
        | OpCode::RecordExactValueI
        | OpCode::RecordKnownResult
        | OpCode::QuasiimmutField
        | OpCode::AssertNotNone
        | OpCode::IncrementDebugCounter => OpResult::Void,

        // ── ForceToken ──
        OpCode::ForceToken => {
            // Return a dummy token in blackhole mode
            OpResult::Value(0)
        }

        // ── Exception operations ──
        OpCode::SaveException => {
            // Return the pending exception value.
            OpResult::Value(exc.exc_value)
        }
        OpCode::SaveExcClass => {
            // Return the pending exception class.
            OpResult::Value(exc.exc_class)
        }
        OpCode::RestoreException => {
            // Restore exception state from (class, value) args.
            let cls = resolve(values, op.args[0]);
            let val = resolve(values, op.args[1]);
            exc.set(cls, val);
            OpResult::Void
        }
        OpCode::CheckMemoryError => {
            // If the allocation returned null, set a MemoryError exception.
            let ptr = resolve(values, op.args[0]);
            if ptr == 0 {
                // Set a generic memory error (class=1 by convention).
                exc.set(1, 0);
            }
            OpResult::Void
        }

        // ── Overflow arithmetic ──
        OpCode::IntAddOvf => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_add(b))
        }
        OpCode::IntSubOvf => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_sub(b))
        }
        OpCode::IntMulOvf => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_mul(b))
        }

        // ── Float comparisons ──
        OpCode::FloatLt => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a < b) as i64)
        }
        OpCode::FloatLe => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a <= b) as i64)
        }
        OpCode::FloatGt => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a > b) as i64)
        }
        OpCode::FloatGe => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a >= b) as i64)
        }
        OpCode::FloatEq => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::FloatNe => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a != b) as i64)
        }

        // ── Additional float operations ──
        OpCode::FloatFloorDiv => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits((a / b).floor()) as i64)
        }
        OpCode::FloatMod => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a % b) as i64)
        }

        // ── VirtualRef (pass through in blackhole) ──
        OpCode::VirtualRefI | OpCode::VirtualRefR => {
            let a = unop(values, op);
            OpResult::Value(a)
        }

        // ── Call operations (pass through with concrete values) ──
        // In blackhole mode, calls should re-execute with concrete args.
        // For now, we handle CALL_PURE variants (can evaluate if all args known).
        OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF => {
            // Cannot actually call in blackhole — result depends on runtime.
            // Return 0 as a placeholder. Full blackhole would re-execute the call.
            OpResult::Value(0)
        }
        OpCode::CallPureN => OpResult::Void,

        // Regular calls and residual calls — blackhole cannot execute these
        // without access to the actual function pointers. In a full implementation,
        // the blackhole would use the constant pool to resolve the function
        // pointer and call it. For now, treat as opaque.
        OpCode::CallI | OpCode::CallR | OpCode::CallF => {
            // Placeholder: full blackhole would actually invoke the function
            OpResult::Value(0)
        }
        OpCode::CallMayForceI | OpCode::CallMayForceR | OpCode::CallMayForceF => OpResult::Value(0),
        OpCode::CallN | OpCode::CallMayForceN => OpResult::Void,

        OpCode::CallReleaseGilI | OpCode::CallReleaseGilR | OpCode::CallReleaseGilF => {
            OpResult::Value(0)
        }
        OpCode::CallReleaseGilN => OpResult::Void,

        // ── Memory access (raw) ──
        // In a full blackhole, these would dereference actual pointers.
        // For now, return 0 as placeholder.
        OpCode::GetfieldGcI
        | OpCode::GetfieldGcR
        | OpCode::GetfieldGcF
        | OpCode::GetfieldRawI
        | OpCode::GetfieldRawR
        | OpCode::GetfieldRawF
        | OpCode::GetfieldGcPureI
        | OpCode::GetfieldGcPureR
        | OpCode::GetfieldGcPureF => OpResult::Value(0),
        OpCode::SetfieldGc | OpCode::SetfieldRaw => OpResult::Void,

        // ── Array access ──
        OpCode::GetarrayitemGcI
        | OpCode::GetarrayitemGcR
        | OpCode::GetarrayitemGcF
        | OpCode::GetarrayitemRawI
        | OpCode::GetarrayitemRawR
        | OpCode::GetarrayitemRawF
        | OpCode::GetarrayitemGcPureI
        | OpCode::GetarrayitemGcPureR
        | OpCode::GetarrayitemGcPureF => OpResult::Value(0),
        OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => OpResult::Void,

        // ── Array/string length ──
        OpCode::ArraylenGc => OpResult::Value(0),
        OpCode::Strlen | OpCode::Unicodelen => OpResult::Value(0),

        // ── Allocation (no-op in blackhole, objects already materialized) ──
        OpCode::New | OpCode::NewWithVtable => OpResult::Value(0),
        OpCode::NewArray | OpCode::NewArrayClear => OpResult::Value(0),
        OpCode::Newstr | OpCode::Newunicode => OpResult::Value(0),

        // ── String/char access ──
        OpCode::Strgetitem | OpCode::Unicodegetitem => OpResult::Value(0),
        OpCode::Strsetitem | OpCode::Unicodesetitem => OpResult::Void,
        OpCode::Strhash | OpCode::Unicodehash => OpResult::Value(0),

        // ── Interior field access ──
        OpCode::GetinteriorfieldGcI | OpCode::GetinteriorfieldGcR | OpCode::GetinteriorfieldGcF => {
            OpResult::Value(0)
        }
        OpCode::SetinteriorfieldGc | OpCode::SetinteriorfieldRaw => OpResult::Void,

        // ── Raw memory ──
        OpCode::RawStore => OpResult::Void,
        OpCode::RawLoadI | OpCode::RawLoadF => OpResult::Value(0),

        // ── GC write barriers (no-op in blackhole) ──
        OpCode::CondCallGcWb | OpCode::CondCallGcWbArray | OpCode::ZeroArray => OpResult::Void,

        // ── Nursery allocation (no-op in blackhole) ──
        OpCode::CallMallocNursery
        | OpCode::CallMallocNurseryVarsize
        | OpCode::CallMallocNurseryVarsizeFrame
        | OpCode::NurseryPtrIncrement => OpResult::Value(0),

        // ── Pointer comparisons/casts ──
        OpCode::PtrEq | OpCode::InstancePtrEq => {
            let (a, b) = binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::PtrNe | OpCode::InstancePtrNe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a != b) as i64)
        }
        OpCode::CastPtrToInt => {
            let a = unop(values, op);
            OpResult::Value(a)
        }
        OpCode::CastIntToPtr | OpCode::CastOpaquePtr => {
            let a = unop(values, op);
            OpResult::Value(a)
        }

        // ── CALL_ASSEMBLER (delegate to regular execution) ──
        OpCode::CallAssemblerI | OpCode::CallAssemblerR | OpCode::CallAssemblerF => {
            OpResult::Value(0)
        }
        OpCode::CallAssemblerN => OpResult::Void,

        // ── Cond call (conditional function call) ──
        OpCode::CondCallValueI | OpCode::CondCallValueR => OpResult::Value(0),
        OpCode::CondCallN => OpResult::Void,

        // ── Thread-local ref ──
        OpCode::ThreadlocalrefGet => OpResult::Value(0),

        // ── Loopinvariant calls ──
        OpCode::CallLoopinvariantI | OpCode::CallLoopinvariantR | OpCode::CallLoopinvariantF => {
            OpResult::Value(0)
        }
        OpCode::CallLoopinvariantN => OpResult::Void,

        // ── GC loads ──
        OpCode::GcLoadI | OpCode::GcLoadR | OpCode::GcLoadF => OpResult::Value(0),
        OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF => {
            OpResult::Value(0)
        }
        OpCode::GcStore | OpCode::GcStoreIndexed => OpResult::Void,

        // ── Vec loads/stores ──
        OpCode::VecLoadI | OpCode::VecLoadF => OpResult::Value(0),
        OpCode::VecStore => OpResult::Void,

        // ── String/unicode copy ──
        OpCode::Copystrcontent | OpCode::Copyunicodecontent => OpResult::Void,

        // ── Misc conversions ──
        OpCode::UintMulHigh => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u128 * b as u128) >> 64) as i64)
        }
        OpCode::CastFloatToSinglefloat => {
            let a = float_unop(values, op);
            let f32_val = a as f32;
            OpResult::Value(f32_val.to_bits() as i64)
        }
        OpCode::CastSinglefloatToFloat => {
            let a = unop(values, op);
            let f32_val = f32::from_bits(a as u32);
            OpResult::Value(f64::to_bits(f32_val as f64) as i64)
        }
        OpCode::ConvertFloatBytesToLonglong => {
            let a = unop(values, op);
            OpResult::Value(a) // f64 bits already stored as i64
        }
        OpCode::ConvertLonglongBytesToFloat => {
            let a = unop(values, op);
            OpResult::Value(a) // i64 bits reinterpreted as f64
        }

        // ── Debug / portal frame markers ──
        OpCode::DebugMergePoint
        | OpCode::EnterPortalFrame
        | OpCode::LeavePortalFrame
        | OpCode::JitDebug => OpResult::Void,

        // ── Escape ops (testing) ──
        OpCode::EscapeI | OpCode::EscapeR | OpCode::EscapeF => OpResult::Value(0),
        OpCode::EscapeN => OpResult::Void,

        // ── LoadFromGcTable / LoadEffectiveAddress ──
        OpCode::LoadFromGcTable | OpCode::LoadEffectiveAddress => OpResult::Value(0),

        other => OpResult::Unsupported(format!("blackhole: unsupported opcode {:?}", other)),
    }
}

fn binop(values: &HashMap<u32, i64>, op: &Op) -> (i64, i64) {
    let a = resolve(values, op.args[0]);
    let b = resolve(values, op.args[1]);
    (a, b)
}

fn unop(values: &HashMap<u32, i64>, op: &Op) -> i64 {
    resolve(values, op.args[0])
}

fn float_binop(values: &HashMap<u32, i64>, op: &Op) -> (f64, f64) {
    let a = f64::from_bits(resolve(values, op.args[0]) as u64);
    let b = f64::from_bits(resolve(values, op.args[1]) as u64);
    (a, b)
}

fn float_unop(values: &HashMap<u32, i64>, op: &Op) -> f64 {
    f64::from_bits(resolve(values, op.args[0]) as u64)
}

/// Blackhole execution with virtual object materialization.
///
/// When a guard fails, some values in the DeadFrame may correspond to
/// virtual objects that were never allocated. This function:
/// 1. Materializes virtual objects from resume data
/// 2. Provides the materialized objects as `BlackholeResult::GuardFailedWithVirtuals`
///
/// `allocator_fn` is called for each virtual that needs heap allocation.
/// It receives a `MaterializedVirtual` and returns the allocated object address.
///
/// Mirrors RPython's `_prepare_virtuals()` in resume.py.
pub fn blackhole_with_virtuals(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    resume_data: Option<&ResumeData>,
) -> BlackholeResult {
    let result = blackhole_execute(ops, constants, initial_values, start_index);

    // If guard failed and we have resume data with virtuals, materialize them
    if let BlackholeResult::GuardFailed {
        guard_index,
        ref fail_values,
    } = result
    {
        if let Some(rd) = resume_data {
            if !rd.virtuals.is_empty() || !rd.pending_fields.is_empty() {
                let materialized = rd.materialize_virtuals(fail_values);
                let pending_field_writes =
                    ResumeData::resolve_pending_field_writes(&rd.pending_fields, fail_values);
                return BlackholeResult::GuardFailedWithVirtuals {
                    guard_index,
                    fail_values: fail_values.clone(),
                    materialized_virtuals: materialized,
                    pending_field_writes,
                };
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resume::{
        PendingFieldInfo as ResumePendingFieldInfo, VirtualFieldSource,
        VirtualInfo as ResumeVirtualInfo,
    };
    use majit_ir::OpRef;

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut op = Op::new(opcode, args);
        op.pos = OpRef(pos);
        op
    }

    #[test]
    fn test_blackhole_simple_arithmetic() {
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 10i64);
        initial.insert(1, 20i64);

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish(vals) => assert_eq!(vals, vec![30]),
            other => panic!(
                "expected Finish, got {:?}",
                match other {
                    BlackholeResult::Abort(s) => s,
                    _ => "other".to_string(),
                }
            ),
        }
    }

    #[test]
    fn test_blackhole_guard_passes() {
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 1i64);

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish(vals) => assert_eq!(vals, vec![1]),
            _ => panic!("expected Finish"),
        }
    }

    #[test]
    fn test_blackhole_exception_guard_no_exception_passes() {
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GuardNoException, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 42i64);

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish(vals) => assert_eq!(vals, vec![42]),
            _ => panic!("expected Finish when no exception is pending"),
        }
    }

    #[test]
    fn test_blackhole_initial_exception_fails_guard_no_exception_without_restore() {
        let mut guard_op = mk_op(OpCode::GuardNoException, &[], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::smallvec![OpRef(0)]);

        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 42i64);

        match blackhole_execute_with_exception(
            &ops,
            &HashMap::new(),
            &initial,
            0,
            ExceptionState {
                exc_class: 100,
                exc_value: 200,
            },
        ) {
            BlackholeResult::GuardFailed { .. } => {}
            _ => panic!("expected GuardFailed when initial exception is pending"),
        }
    }

    #[test]
    fn test_blackhole_initial_exception_satisfies_guard_exception_without_restore() {
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GuardException, &[OpRef(0)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 100i64);

        match blackhole_execute_with_exception(
            &ops,
            &HashMap::new(),
            &initial,
            0,
            ExceptionState {
                exc_class: 100,
                exc_value: 200,
            },
        ) {
            BlackholeResult::Finish(vals) => assert_eq!(vals, vec![200]),
            _ => panic!("expected Finish when initial exception class matches"),
        }
    }

    #[test]
    fn test_blackhole_exception_save_exc_class() {
        // RestoreException sets exception state, then SaveExcClass reads it.
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(
                OpCode::RestoreException,
                &[OpRef(0), OpRef(1)],
                OpRef::NONE.0,
            ),
            mk_op(OpCode::SaveExcClass, &[], 2),
            mk_op(OpCode::SaveException, &[], 3),
            mk_op(OpCode::Finish, &[OpRef(2), OpRef(3)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 100i64); // exc_class
        initial.insert(1, 200i64); // exc_value

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish(vals) => assert_eq!(vals, vec![100, 200]),
            _ => panic!("expected Finish"),
        }
    }

    #[test]
    fn test_blackhole_guard_no_exception_fails_with_exception() {
        // RestoreException then GuardNoException should fail.
        let mut guard_op = mk_op(OpCode::GuardNoException, &[], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::smallvec![OpRef(0)]);

        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(
                OpCode::RestoreException,
                &[OpRef(0), OpRef(1)],
                OpRef::NONE.0,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 100i64);
        initial.insert(1, 200i64);

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::GuardFailed { .. } => {} // expected
            _ => panic!("expected GuardFailed when exception is pending"),
        }
    }

    #[test]
    fn test_blackhole_guard_exception_matches() {
        // Set exception, then GuardException with matching class should pass.
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(
                OpCode::RestoreException,
                &[OpRef(0), OpRef(1)],
                OpRef::NONE.0,
            ),
            // GuardException expects class in arg(0)
            mk_op(OpCode::GuardException, &[OpRef(0)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 100i64); // exc_class
        initial.insert(1, 200i64); // exc_value

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish(vals) => {
                // GuardException should return the exc_value
                assert_eq!(vals, vec![200]);
            }
            _ => panic!("expected Finish when exception class matches"),
        }
    }

    #[test]
    fn test_blackhole_guard_fails() {
        let mut guard_op = mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::smallvec![OpRef(0)]);

        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 0i64);

        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::GuardFailed { fail_values, .. } => {
                assert_eq!(fail_values, vec![0]);
            }
            _ => panic!("expected GuardFailed"),
        }
    }

    #[test]
    fn test_blackhole_with_virtuals_materializes_resume_virtuals() {
        let mut guard_op = mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::smallvec![OpRef(0)]);

        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 0i64);

        let resume_data = ResumeData {
            frames: vec![],
            virtuals: vec![ResumeVirtualInfo::VStruct {
                type_id: 0,
                descr_index: 7,
                fields: vec![(3, VirtualFieldSource::Constant(55))],
            }],
            pending_fields: Vec::new(),
        };

        match blackhole_with_virtuals(&ops, &HashMap::new(), &initial, 0, Some(&resume_data)) {
            BlackholeResult::GuardFailedWithVirtuals {
                guard_index,
                fail_values,
                materialized_virtuals,
                pending_field_writes,
            } => {
                assert_eq!(guard_index, 1);
                assert_eq!(fail_values, vec![0]);
                assert_eq!(materialized_virtuals.len(), 1);
                assert!(pending_field_writes.is_empty());
                match &materialized_virtuals[0] {
                    MaterializedVirtual::Struct {
                        type_id,
                        descr_index,
                        fields,
                    } => {
                        assert_eq!(*type_id, 0);
                        assert_eq!(*descr_index, 7);
                        assert_eq!(fields, &vec![(3, 55)]);
                    }
                    other => panic!("unexpected materialized virtual: {other:?}"),
                }
            }
            _ => panic!("expected GuardFailedWithVirtuals"),
        }
    }

    #[test]
    fn test_blackhole_with_virtuals_surfaces_pending_field_writes() {
        let mut guard_op = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::smallvec![OpRef(0)]);
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 1i64);

        let resume_data = ResumeData {
            frames: vec![],
            virtuals: vec![],
            pending_fields: vec![ResumePendingFieldInfo {
                descr_index: 9,
                target: VirtualFieldSource::FailArg(0),
                value: VirtualFieldSource::Constant(77),
                item_index: Some(2),
            }],
        };

        match blackhole_with_virtuals(&ops, &HashMap::new(), &initial, 0, Some(&resume_data)) {
            BlackholeResult::GuardFailedWithVirtuals {
                guard_index,
                fail_values,
                materialized_virtuals,
                pending_field_writes,
            } => {
                assert_eq!(guard_index, 1);
                assert_eq!(fail_values, vec![1]);
                assert!(materialized_virtuals.is_empty());
                assert_eq!(
                    pending_field_writes,
                    vec![crate::resume::ResolvedPendingFieldWrite {
                        descr_index: 9,
                        target: 1,
                        value: 77,
                        item_index: Some(2),
                    }]
                );
            }
            _ => panic!("expected GuardFailedWithVirtuals"),
        }
    }
}
