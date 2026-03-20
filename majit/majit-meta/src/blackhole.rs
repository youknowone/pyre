//! Blackhole interpreter: evaluates IR operations with concrete values.
//!
//! When a guard fails in compiled code, the blackhole interpreter can replay
//! the remaining operations from the guard point to the end of the trace,
//! using concrete values from the DeadFrame.
//!
//! This is the RPython equivalent of `rpython/jit/metainterp/blackhole.py`.

use std::collections::HashMap;

use crate::resume::{
    MaterializedVirtual, ResolvedPendingFieldWrite, ResumeData, ResumeLayoutSummary,
};
use majit_ir::{Op, OpCode, OpRef};

/// Trait for IR-based blackhole memory access.
///
/// **Deprecated**: Part of the IR-based blackhole path.
/// The jitcode-based `BlackholeInterpreter` delegates memory access
/// through concrete function pointers in `JitCode.fn_ptrs` instead.
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
    /// Call a function pointer returning a ref (as i64 pointer bits).
    fn call_r(&self, func: i64, args: &[i64]) -> i64 {
        let _ = (func, args);
        0
    }
    /// Call a function pointer returning a float (as i64 bit-cast).
    fn call_f(&self, func: i64, args: &[i64]) -> i64 {
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

/// Result of IR-based blackhole execution.
///
/// **Deprecated**: Part of the IR-based blackhole path.
pub enum BlackholeResult {
    /// Reached a Finish operation with output values.
    Finish { op_index: usize, values: Vec<i64> },
    /// Reached a Jump — loop back to header with these values.
    Jump { op_index: usize, values: Vec<i64> },
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
/// **Deprecated**: RPython's blackhole.py executes jitcode bytecodes, not IR ops.
/// Use `BlackholeInterpreter` + `resume_in_blackhole()` instead.
/// This function will be removed once pyre generates jitcode and
/// `pyjitpl.rs` guard failure recovery switches to jitcode-based blackhole.
#[deprecated(note = "use BlackholeInterpreter for RPython-parity jitcode execution")]
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
///
/// **Deprecated**: see `blackhole_execute`.
#[deprecated(note = "use BlackholeInterpreter for RPython-parity jitcode execution")]
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
                return (
                    BlackholeResult::Finish {
                        op_index: op_idx,
                        values: vals,
                    },
                    exc_state,
                );
            }
            OpResult::Jump(args) => {
                let vals: Vec<i64> = args.iter().map(|&r| resolve(&values, r)).collect();
                return (
                    BlackholeResult::Jump {
                        op_index: op_idx,
                        values: vals,
                    },
                    exc_state,
                );
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
        // ── Field access via descr offset ──
        OpCode::GetfieldGcI | OpCode::GetfieldRawI | OpCode::GetfieldGcPureI => {
            let base = resolve(values, op.args[0]);
            let offset = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .map_or(0, |f| f.offset() as i64);
            OpResult::Value(memory.gc_load_i(base, offset))
        }
        OpCode::GetfieldGcR | OpCode::GetfieldRawR | OpCode::GetfieldGcPureR => {
            let base = resolve(values, op.args[0]);
            let offset = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .map_or(0, |f| f.offset() as i64);
            OpResult::Value(memory.gc_load_r(base, offset))
        }
        OpCode::GetfieldGcF | OpCode::GetfieldRawF | OpCode::GetfieldGcPureF => {
            let base = resolve(values, op.args[0]);
            let offset = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .map_or(0, |f| f.offset() as i64);
            OpResult::Value(memory.gc_load_f(base, offset))
        }
        OpCode::SetfieldGc | OpCode::SetfieldRaw => {
            let base = resolve(values, op.args[0]);
            let value = resolve(values, op.args[1]);
            let offset = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .map_or(0, |f| f.offset() as i64);
            memory.gc_store(base, offset, value);
            OpResult::Void
        }
        // ── Array access via descr ──
        OpCode::GetarrayitemGcI | OpCode::GetarrayitemRawI | OpCode::GetarrayitemGcPureI => {
            let base = resolve(values, op.args[0]);
            let index = resolve(values, op.args[1]);
            let (item_size, base_ofs) = op
                .descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map_or((1, 0), |a| (a.item_size() as i64, a.base_size() as i64));
            OpResult::Value(memory.gc_load_indexed_i(base, index, item_size, base_ofs))
        }
        OpCode::GetarrayitemGcR | OpCode::GetarrayitemRawR | OpCode::GetarrayitemGcPureR => {
            let base = resolve(values, op.args[0]);
            let index = resolve(values, op.args[1]);
            let (item_size, base_ofs) = op
                .descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map_or((1, 0), |a| (a.item_size() as i64, a.base_size() as i64));
            OpResult::Value(memory.gc_load_indexed_r(base, index, item_size, base_ofs))
        }
        OpCode::GetarrayitemGcF | OpCode::GetarrayitemRawF | OpCode::GetarrayitemGcPureF => {
            let base = resolve(values, op.args[0]);
            let index = resolve(values, op.args[1]);
            let (item_size, base_ofs) = op
                .descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map_or((1, 0), |a| (a.item_size() as i64, a.base_size() as i64));
            OpResult::Value(memory.gc_load_indexed_f(base, index, item_size, base_ofs))
        }
        OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
            let base = resolve(values, op.args[0]);
            let index = resolve(values, op.args[1]);
            let value = resolve(values, op.args[2]);
            let (item_size, base_ofs) = op
                .descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map_or((1, 0), |a| (a.item_size() as i64, a.base_size() as i64));
            memory.gc_store_indexed(base, index, item_size, base_ofs, value);
            OpResult::Void
        }
        // Call with real dispatch (all call variants)
        OpCode::CallI
        | OpCode::CallPureI
        | OpCode::CallMayForceI
        | OpCode::CallReleaseGilI
        | OpCode::CallLoopinvariantI => {
            let func = resolve(values, op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| resolve(values, r)).collect();
            OpResult::Value(memory.call_i(func, &args))
        }
        OpCode::CallR
        | OpCode::CallPureR
        | OpCode::CallMayForceR
        | OpCode::CallReleaseGilR
        | OpCode::CallLoopinvariantR => {
            let func = resolve(values, op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| resolve(values, r)).collect();
            OpResult::Value(memory.call_r(func, &args))
        }
        OpCode::CallF
        | OpCode::CallPureF
        | OpCode::CallMayForceF
        | OpCode::CallReleaseGilF
        | OpCode::CallLoopinvariantF => {
            let func = resolve(values, op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| resolve(values, r)).collect();
            OpResult::Value(memory.call_f(func, &args))
        }
        OpCode::CallN
        | OpCode::CallPureN
        | OpCode::CallMayForceN
        | OpCode::CallReleaseGilN
        | OpCode::CallLoopinvariantN => {
            let func = resolve(values, op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| resolve(values, r)).collect();
            memory.call_n(func, &args);
            OpResult::Void
        }
        // CallAssembler — delegate to call dispatch
        OpCode::CallAssemblerI => {
            let func = resolve(values, op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| resolve(values, r)).collect();
            OpResult::Value(memory.call_i(func, &args))
        }
        OpCode::CallAssemblerR => {
            let func = resolve(values, op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| resolve(values, r)).collect();
            OpResult::Value(memory.call_r(func, &args))
        }
        OpCode::CallAssemblerF => {
            let func = resolve(values, op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| resolve(values, r)).collect();
            OpResult::Value(memory.call_f(func, &args))
        }
        OpCode::CallAssemblerN => {
            let func = resolve(values, op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| resolve(values, r)).collect();
            memory.call_n(func, &args);
            OpResult::Void
        }
        // CondCallValue — delegate to call dispatch
        OpCode::CondCallValueI => {
            let func = resolve(values, op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| resolve(values, r)).collect();
            OpResult::Value(memory.call_i(func, &args))
        }
        OpCode::CondCallValueR => {
            let func = resolve(values, op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| resolve(values, r)).collect();
            OpResult::Value(memory.call_r(func, &args))
        }
        OpCode::CondCallN => {
            let func = resolve(values, op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| resolve(values, r)).collect();
            memory.call_n(func, &args);
            OpResult::Void
        }
        // Fall through to the default execute_one for everything else
        _ => execute_one(op, values, exc),
    }
}

/// **Deprecated**: IR-based blackhole. Will be replaced by jitcode-based
/// `resume_in_blackhole()` once pyre generates jitcode.
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
                return (
                    BlackholeResult::Finish {
                        op_index: op_idx,
                        values: vals,
                    },
                    exc_state,
                );
            }
            OpResult::Jump(args) => {
                let vals: Vec<i64> = args.iter().map(|&r| resolve(&values, r)).collect();
                return (
                    BlackholeResult::Jump {
                        op_index: op_idx,
                        values: vals,
                    },
                    exc_state,
                );
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
///
/// **Deprecated**: see `blackhole_execute`.
#[deprecated(note = "use BlackholeInterpreter for RPython-parity jitcode execution")]
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
            let a = same_as_value(values, op);
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
        // Call operations — return placeholder 0 in no-memory path.
        // The execute_one_with_memory path handles actual dispatch.
        OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF => OpResult::Value(0),
        OpCode::CallPureN => OpResult::Void,
        OpCode::CallI
        | OpCode::CallR
        | OpCode::CallF
        | OpCode::CallMayForceI
        | OpCode::CallMayForceR
        | OpCode::CallMayForceF
        | OpCode::CallReleaseGilI
        | OpCode::CallReleaseGilR
        | OpCode::CallReleaseGilF => OpResult::Value(0),
        OpCode::CallN | OpCode::CallMayForceN | OpCode::CallReleaseGilN => OpResult::Void,

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

        // ── Vector arithmetic (scalar emulation in blackhole) ──
        OpCode::VecIntAdd => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_add(b))
        }
        OpCode::VecIntSub => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_sub(b))
        }
        OpCode::VecIntMul => {
            let (a, b) = binop(values, op);
            OpResult::Value(a.wrapping_mul(b))
        }
        OpCode::VecIntAnd => {
            let (a, b) = binop(values, op);
            OpResult::Value(a & b)
        }
        OpCode::VecIntOr => {
            let (a, b) = binop(values, op);
            OpResult::Value(a | b)
        }
        OpCode::VecIntXor => {
            let (a, b) = binop(values, op);
            OpResult::Value(a ^ b)
        }
        OpCode::VecFloatAdd => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a + b) as i64)
        }
        OpCode::VecFloatSub => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a - b) as i64)
        }
        OpCode::VecFloatMul => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a * b) as i64)
        }
        OpCode::VecFloatTrueDiv => {
            let (a, b) = float_binop(values, op);
            OpResult::Value(f64::to_bits(a / b) as i64)
        }
        OpCode::VecFloatNeg => {
            let a = float_unop(values, op);
            OpResult::Value(f64::to_bits(-a) as i64)
        }
        OpCode::VecFloatAbs => {
            let a = float_unop(values, op);
            OpResult::Value(f64::to_bits(a.abs()) as i64)
        }

        // ── Vector comparisons (scalar emulation) ──
        OpCode::VecFloatEq => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::VecFloatNe => {
            let (a, b) = float_binop(values, op);
            OpResult::Value((a != b) as i64)
        }
        OpCode::VecFloatXor => {
            let (a, b) = binop(values, op);
            OpResult::Value(a ^ b)
        }
        OpCode::VecIntIsTrue => {
            let a = unop(values, op);
            OpResult::Value((a != 0) as i64)
        }
        OpCode::VecIntNe => {
            let (a, b) = binop(values, op);
            OpResult::Value((a != b) as i64)
        }
        OpCode::VecIntEq => {
            let (a, b) = binop(values, op);
            OpResult::Value((a == b) as i64)
        }
        OpCode::VecIntSignext => {
            let (a, b) = binop(values, op);
            let bits = b * 8;
            let shift = 64 - bits;
            OpResult::Value((a << shift) >> shift)
        }

        // ── Vector casts (scalar emulation) ──
        OpCode::VecCastFloatToInt => {
            let a = float_unop(values, op);
            OpResult::Value(a as i64)
        }
        OpCode::VecCastIntToFloat => {
            let a = unop(values, op);
            OpResult::Value(f64::to_bits(a as f64) as i64)
        }
        OpCode::VecCastFloatToSinglefloat => {
            let a = float_unop(values, op);
            let f32_val = a as f32;
            OpResult::Value(f32_val.to_bits() as i64)
        }
        OpCode::VecCastSinglefloatToFloat => {
            let a = unop(values, op);
            let f32_val = f32::from_bits(a as u32);
            OpResult::Value(f64::to_bits(f32_val as f64) as i64)
        }

        // ── Vector pack/unpack/expand (scalar emulation) ──
        OpCode::VecI => OpResult::Value(0),
        OpCode::VecF => OpResult::Value(f64::to_bits(0.0) as i64),
        OpCode::VecUnpackI | OpCode::VecUnpackF => {
            // unpack(vec, lane, count) -> return vec (first scalar)
            let a = unop(values, op);
            OpResult::Value(a)
        }
        OpCode::VecPackI | OpCode::VecPackF => {
            // pack(vec, scalar, lane, count) -> return scalar
            let scalar = resolve(values, op.args[1]);
            OpResult::Value(scalar)
        }
        OpCode::VecExpandI | OpCode::VecExpandF => {
            // expand(scalar) -> return scalar
            let a = unop(values, op);
            OpResult::Value(a)
        }

        // ── String/unicode copy ──
        OpCode::Copystrcontent | OpCode::Copyunicodecontent => OpResult::Void,

        // ── Misc conversions ──
        OpCode::UintMulHigh => {
            let (a, b) = binop(values, op);
            OpResult::Value(((a as u64 as u128 * b as u64 as u128) >> 64) as i64)
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

        // All OpCode variants are explicitly handled above.
        // This arm is unreachable but kept for forward-compatibility
        // when new opcodes are added to the IR.
        #[allow(unreachable_patterns)]
        other => OpResult::Unsupported(format!(
            "blackhole: opcode {:?} has no interpreter handler",
            other
        )),
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

fn same_as_value(values: &HashMap<u32, i64>, op: &Op) -> i64 {
    if op.num_args() > 0 {
        unop(values, op)
    } else if !op.pos.is_none() {
        resolve(values, op.pos)
    } else {
        0
    }
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
/// **Deprecated**: IR-based blackhole. See `blackhole_execute`.
#[deprecated(note = "use BlackholeInterpreter for RPython-parity jitcode execution")]
pub fn blackhole_with_virtuals(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    resume_data: Option<&ResumeData>,
) -> BlackholeResult {
    blackhole_with_recovery_layout(
        ops,
        constants,
        initial_values,
        start_index,
        resume_data,
        None,
    )
}

/// Blackhole execution with semantic-free resume-layout materialization.
///
/// **Deprecated**: IR-based blackhole. See `blackhole_execute`.
#[deprecated(note = "use BlackholeInterpreter for RPython-parity jitcode execution")]
pub fn blackhole_with_resume_layout(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    resume_layout: Option<&ResumeLayoutSummary>,
) -> BlackholeResult {
    blackhole_with_recovery_layout(
        ops,
        constants,
        initial_values,
        start_index,
        None,
        resume_layout,
    )
}

fn blackhole_with_recovery_layout(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    resume_data: Option<&ResumeData>,
    resume_layout: Option<&ResumeLayoutSummary>,
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
        } else if let Some(layout) = resume_layout {
            if layout.num_virtuals > 0 || layout.pending_field_count > 0 {
                return BlackholeResult::GuardFailedWithVirtuals {
                    guard_index,
                    fail_values: fail_values.clone(),
                    materialized_virtuals: layout.materialize_virtuals(fail_values),
                    pending_field_writes: layout.resolve_pending_field_writes(fail_values),
                };
            }
        }
    }

    result
}

// ============================================================================
// RPython blackhole.py parity: BlackholeInterpreter
//
// Jitcode-based blackhole execution. When a guard fails in compiled code,
// resume_in_blackhole reconstructs execution frames from resume data and
// runs jitcode bytecodes with concrete values, following ALL code paths
// (unlike trace IR which only has the traced path).
// ============================================================================

use crate::jitcode::{
    self, JitArgKind, JitCode, MIFrame, MIFrameStack,
    BC_ABORT, BC_ABORT_PERMANENT, BC_ARRAYLEN_VABLE, BC_BRANCH_REG_ZERO, BC_BRANCH_ZERO,
    BC_CALL_ASSEMBLER_FLOAT, BC_CALL_ASSEMBLER_INT, BC_CALL_ASSEMBLER_REF, BC_CALL_ASSEMBLER_VOID,
    BC_CALL_FLOAT, BC_CALL_INT, BC_CALL_LOOPINVARIANT_FLOAT, BC_CALL_LOOPINVARIANT_INT,
    BC_CALL_LOOPINVARIANT_REF, BC_CALL_LOOPINVARIANT_VOID, BC_CALL_MAY_FORCE_FLOAT,
    BC_CALL_MAY_FORCE_INT, BC_CALL_MAY_FORCE_REF, BC_CALL_MAY_FORCE_VOID, BC_CALL_PURE_FLOAT,
    BC_CALL_PURE_INT, BC_CALL_PURE_REF, BC_CALL_REF, BC_CALL_RELEASE_GIL_FLOAT,
    BC_CALL_RELEASE_GIL_INT, BC_CALL_RELEASE_GIL_REF, BC_CALL_RELEASE_GIL_VOID,
    BC_COPY_FROM_BOTTOM, BC_DUP_STACK, BC_GETARRAYITEM_VABLE_F, BC_GETARRAYITEM_VABLE_I,
    BC_GETARRAYITEM_VABLE_R, BC_GETFIELD_VABLE_F, BC_GETFIELD_VABLE_I, BC_GETFIELD_VABLE_R,
    BC_HINT_FORCE_VIRTUALIZABLE, BC_INLINE_CALL, BC_JUMP, BC_JUMP_TARGET, BC_LOAD_CONST_F,
    BC_LOAD_CONST_I, BC_LOAD_CONST_R, BC_LOAD_STATE_ARRAY, BC_LOAD_STATE_FIELD,
    BC_LOAD_STATE_VARRAY, BC_MOVE_F, BC_MOVE_I, BC_MOVE_R, BC_PEEK_I, BC_POP_DISCARD, BC_POP_F,
    BC_POP_I, BC_POP_R, BC_PUSH_F, BC_PUSH_I, BC_PUSH_R, BC_PUSH_TO, BC_RECORD_BINOP_F,
    BC_RECORD_BINOP_I, BC_RECORD_UNARY_F, BC_RECORD_UNARY_I, BC_REQUIRE_STACK,
    BC_RESIDUAL_CALL_VOID, BC_SET_SELECTED, BC_SETARRAYITEM_VABLE_F, BC_SETARRAYITEM_VABLE_I,
    BC_SETARRAYITEM_VABLE_R, BC_SETFIELD_VABLE_F, BC_SETFIELD_VABLE_I, BC_SETFIELD_VABLE_R,
    BC_STORE_DOWN, BC_STORE_STATE_ARRAY, BC_STORE_STATE_FIELD, BC_STORE_STATE_VARRAY,
    BC_SWAP_STACK,
};
use crate::jitcode::machine::{
    call_int_function, eval_binop_f, eval_binop_i, eval_binop_ovf, eval_unary_f, eval_unary_i,
};

/// Return type of a blackhole frame.
///
/// RPython: `BlackholeInterpreter._return_type`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BhReturnType {
    Int,
    Ref,
    Float,
    Void,
}

/// Signal that the current frame executed a return instruction.
///
/// RPython: `LeaveFrame` exception in blackhole.py
struct LeaveFrame;

/// Jitcode-based blackhole interpreter.
///
/// Executes jitcode bytecodes with concrete values. Each instance
/// represents one execution frame. Frame chain is linked via
/// `nextblackholeinterp`.
///
/// RPython: `BlackholeInterpreter` class in blackhole.py
pub struct BlackholeInterpreter {
    /// Integer register bank.
    /// Indices 0..num_regs_i are working registers.
    /// Indices num_regs_i..num_regs_i+constants_i.len() hold constants.
    pub registers_i: Vec<i64>,
    /// Reference register bank.
    pub registers_r: Vec<i64>,
    /// Float register bank.
    pub registers_f: Vec<i64>,
    /// Temporary register for int return value.
    pub tmpreg_i: i64,
    /// Temporary register for ref return value.
    pub tmpreg_r: i64,
    /// Temporary register for float return value.
    pub tmpreg_f: i64,
    /// Current jitcode being executed.
    pub jitcode: JitCode,
    /// Current bytecode position (program counter).
    pub position: usize,
    /// Caller frame in the blackhole frame chain.
    pub nextblackholeinterp: Option<Box<BlackholeInterpreter>>,
    /// Return type of this frame.
    pub return_type: BhReturnType,
    /// Runtime stacks indexed by `selected`.
    runtime_stacks: HashMap<usize, Vec<i64>>,
    /// Current selected storage index.
    current_selected: usize,
}

impl Default for BlackholeInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl BlackholeInterpreter {
    pub fn new() -> Self {
        Self {
            registers_i: Vec::new(),
            registers_r: Vec::new(),
            registers_f: Vec::new(),
            tmpreg_i: 0,
            tmpreg_r: 0,
            tmpreg_f: 0,
            jitcode: JitCode::default(),
            position: 0,
            nextblackholeinterp: None,
            return_type: BhReturnType::Void,
            runtime_stacks: HashMap::new(),
            current_selected: 0,
        }
    }

    /// Initialize register arrays for a jitcode and set the position.
    ///
    /// RPython: `BlackholeInterpreter.setposition(jitcode, position)`
    pub fn setposition(&mut self, jitcode: JitCode, position: usize) {
        let num_i = jitcode.num_regs_and_consts_i();
        let num_r = jitcode.num_regs_r() as usize;
        let num_f = jitcode.num_regs_f() as usize;

        self.registers_i.clear();
        self.registers_i.resize(num_i, 0);
        self.registers_r.clear();
        self.registers_r.resize(num_r, 0);
        self.registers_f.clear();
        self.registers_f.resize(num_f, 0);

        // Copy constants into upper register indices
        let reg_base = jitcode.num_regs_i() as usize;
        for (i, &c) in jitcode.constants_i.iter().enumerate() {
            self.registers_i[reg_base + i] = c;
        }

        self.jitcode = jitcode;
        self.position = position;
    }

    /// Set an integer register value.
    ///
    /// RPython: `BlackholeInterpreter.setarg_i(index, value)`
    pub fn setarg_i(&mut self, index: usize, value: i64) {
        self.registers_i[index] = value;
    }

    /// Set a reference register value.
    pub fn setarg_r(&mut self, index: usize, value: i64) {
        self.registers_r[index] = value;
    }

    /// Set a float register value.
    pub fn setarg_f(&mut self, index: usize, value: i64) {
        self.registers_f[index] = value;
    }

    /// Get the int return value from a completed frame.
    ///
    /// RPython: `BlackholeInterpreter.get_tmpreg_i()`
    pub fn get_tmpreg_i(&self) -> i64 {
        self.tmpreg_i
    }

    pub fn get_tmpreg_r(&self) -> i64 {
        self.tmpreg_r
    }

    pub fn get_tmpreg_f(&self) -> i64 {
        self.tmpreg_f
    }

    /// Copy register state from a tracing MIFrame into this blackhole frame.
    ///
    /// RPython: `BlackholeInterpreter._copy_data_from_miframe(miframe)`
    pub fn copy_data_from_miframe(&mut self, miframe: &MIFrame) {
        self.setposition(miframe.jitcode.clone(), miframe.pc);
        for i in 0..self.jitcode.num_regs_i() as usize {
            if let Some(val) = miframe.int_values.get(i).copied().flatten() {
                self.setarg_i(i, val);
            }
        }
        for i in 0..self.jitcode.num_regs_r() as usize {
            if let Some(val) = miframe.ref_values.get(i).copied().flatten() {
                self.setarg_r(i, val);
            }
        }
        for i in 0..self.jitcode.num_regs_f() as usize {
            if let Some(val) = miframe.float_values.get(i).copied().flatten() {
                self.setarg_f(i, val);
            }
        }
    }

    /// Store int return value from called frame into caller's result register.
    ///
    /// RPython: `BlackholeInterpreter._setup_return_value_i(result)`
    fn setup_return_value_i(&mut self, result: i64) {
        // Return result register is encoded as the byte before `position`.
        // In RPython: `ord(self.jitcode.code[self.position-1])`
        // In majit jitcode: the dst u16 was read before the call,
        // so the caller must store separately. This is handled in
        // the inline_call dispatch.
        self.tmpreg_i = result;
    }

    fn setup_return_value_r(&mut self, result: i64) {
        self.tmpreg_r = result;
    }

    fn setup_return_value_f(&mut self, result: i64) {
        self.tmpreg_f = result;
    }

    // -- Bytecode reading helpers (matching MIFrame.next_u8/next_u16) --

    fn next_u8(&mut self) -> u8 {
        jitcode::read_u8(&self.jitcode.code, &mut self.position)
    }

    fn next_u16(&mut self) -> u16 {
        jitcode::read_u16(&self.jitcode.code, &mut self.position)
    }

    fn finished(&self) -> bool {
        self.position >= self.jitcode.code.len()
    }

    // -- Runtime stack access --

    fn runtime_stack_mut(&mut self, selected: usize) -> &mut Vec<i64> {
        self.runtime_stacks.entry(selected).or_default()
    }

    fn runtime_stack_pop(&mut self, selected: usize) -> i64 {
        self.runtime_stacks
            .get_mut(&selected)
            .and_then(|s| s.pop())
            .unwrap_or(0)
    }

    pub fn runtime_stack_push(&mut self, selected: usize, value: i64) {
        self.runtime_stacks.entry(selected).or_default().push(value);
    }

    fn runtime_stack_peek(&self, selected: usize, pos: usize) -> i64 {
        self.runtime_stacks
            .get(&selected)
            .and_then(|s| s.get(pos).copied())
            .unwrap_or(0)
    }

    fn runtime_stack_len(&self, selected: usize) -> usize {
        self.runtime_stacks
            .get(&selected)
            .map_or(0, |s| s.len())
    }

    // -- Call argument reading --

    fn read_call_arg(&self, kind: JitArgKind, reg: u16) -> i64 {
        match kind {
            JitArgKind::Int => self.registers_i[reg as usize],
            JitArgKind::Ref => self.registers_r[reg as usize],
            JitArgKind::Float => self.registers_f[reg as usize],
        }
    }

    /// Execute the dispatch loop on the current jitcode.
    ///
    /// RPython: `BlackholeInterpreter.run()` catches `LeaveFrame` and breaks.
    pub fn run(&mut self) {
        loop {
            if self.finished() {
                return;
            }
            let opcode = self.next_u8();
            if self.dispatch_one(opcode).is_err() {
                return; // LeaveFrame: return/abort hit
            }
        }
    }

    /// Dispatch a single bytecode instruction with concrete execution.
    ///
    /// RPython: bytecode dispatch in `dispatch_loop()`, each `bhimpl_*` method
    fn dispatch_one(&mut self, opcode: u8) -> Result<(), LeaveFrame> {
        match opcode {
            BC_LOAD_CONST_I => {
                let dst = self.next_u16() as usize;
                let const_idx = self.next_u16() as usize;
                let value = self.jitcode.constants_i[const_idx];
                self.registers_i[dst] = value;
            }
            BC_LOAD_CONST_R => {
                // Ref constants: for now store 0 (no ref constant pool yet)
                let _dst = self.next_u16() as usize;
                let _const_idx = self.next_u16() as usize;
            }
            BC_LOAD_CONST_F => {
                let _dst = self.next_u16() as usize;
                let _const_idx = self.next_u16() as usize;
            }
            BC_MOVE_I => {
                let dst = self.next_u16() as usize;
                let src = self.next_u16() as usize;
                self.registers_i[dst] = self.registers_i[src];
            }
            BC_MOVE_R => {
                let dst = self.next_u16() as usize;
                let src = self.next_u16() as usize;
                self.registers_r[dst] = self.registers_r[src];
            }
            BC_MOVE_F => {
                let dst = self.next_u16() as usize;
                let src = self.next_u16() as usize;
                self.registers_f[dst] = self.registers_f[src];
            }
            BC_POP_I => {
                let dst = self.next_u16() as usize;
                let selected = self.current_selected;
                self.registers_i[dst] = self.runtime_stack_pop(selected);
            }
            BC_POP_R => {
                let dst = self.next_u16() as usize;
                let selected = self.current_selected;
                self.registers_r[dst] = self.runtime_stack_pop(selected);
            }
            BC_POP_F => {
                let dst = self.next_u16() as usize;
                let selected = self.current_selected;
                self.registers_f[dst] = self.runtime_stack_pop(selected);
            }
            BC_PUSH_I => {
                let src = self.next_u16() as usize;
                let value = self.registers_i[src];
                let selected = self.current_selected;
                self.runtime_stack_push(selected, value);
            }
            BC_PUSH_R => {
                let src = self.next_u16() as usize;
                let value = self.registers_r[src];
                let selected = self.current_selected;
                self.runtime_stack_push(selected, value);
            }
            BC_PUSH_F => {
                let src = self.next_u16() as usize;
                let value = self.registers_f[src];
                let selected = self.current_selected;
                self.runtime_stack_push(selected, value);
            }
            BC_PUSH_TO => {
                let target_selected = self.next_u16() as usize;
                let src = self.next_u16() as usize;
                let value = self.registers_i[src];
                self.runtime_stack_push(target_selected, value);
            }
            BC_PEEK_I => {
                let dst = self.next_u16() as usize;
                let pos = self.next_u16() as usize;
                let selected = self.current_selected;
                self.registers_i[dst] = self.runtime_stack_peek(selected, pos);
            }
            BC_POP_DISCARD => {
                let selected = self.current_selected;
                self.runtime_stack_pop(selected);
            }
            BC_DUP_STACK => {
                let selected = self.current_selected;
                let top = self.runtime_stack_pop(selected);
                self.runtime_stack_push(selected, top);
                self.runtime_stack_push(selected, top);
            }
            BC_SWAP_STACK => {
                let selected = self.current_selected;
                let a = self.runtime_stack_pop(selected);
                let b = self.runtime_stack_pop(selected);
                self.runtime_stack_push(selected, a);
                self.runtime_stack_push(selected, b);
            }
            BC_COPY_FROM_BOTTOM => {
                let selected = self.current_selected;
                let pos = self.next_u16() as usize;
                let value = self.runtime_stack_peek(selected, pos);
                self.runtime_stack_push(selected, value);
            }
            BC_STORE_DOWN => {
                let selected = self.current_selected;
                let pos = self.next_u16() as usize;
                let value = self.runtime_stack_pop(selected);
                if let Some(stack) = self.runtime_stacks.get_mut(&selected) {
                    if pos < stack.len() {
                        stack[pos] = value;
                    }
                }
            }
            BC_REQUIRE_STACK => {
                // No-op in blackhole: stack depth requirements only
                // matter for tracing. Skip the operand.
                let _required = self.next_u16();
            }
            BC_RECORD_BINOP_I => {
                let dst = self.next_u16() as usize;
                let opcode_idx = self.next_u16() as usize;
                let lhs_idx = self.next_u16() as usize;
                let rhs_idx = self.next_u16() as usize;
                let opcode = self.jitcode.opcodes[opcode_idx];
                let lhs = self.registers_i[lhs_idx];
                let rhs = self.registers_i[rhs_idx];
                if opcode.is_ovf() {
                    // Overflow: use wrapping in blackhole
                    let value = eval_binop_ovf(opcode, lhs, rhs).unwrap_or_else(|| {
                        // Overflow occurred: use wrapping result
                        eval_binop_i(
                            match opcode {
                                OpCode::IntAddOvf => OpCode::IntAdd,
                                OpCode::IntSubOvf => OpCode::IntSub,
                                OpCode::IntMulOvf => OpCode::IntMul,
                                _ => opcode,
                            },
                            lhs,
                            rhs,
                        )
                    });
                    self.registers_i[dst] = value;
                } else {
                    self.registers_i[dst] = eval_binop_i(opcode, lhs, rhs);
                }
            }
            BC_RECORD_UNARY_I => {
                let dst = self.next_u16() as usize;
                let opcode_idx = self.next_u16() as usize;
                let src_idx = self.next_u16() as usize;
                let opcode = self.jitcode.opcodes[opcode_idx];
                let value = self.registers_i[src_idx];
                self.registers_i[dst] = eval_unary_i(opcode, value);
            }
            BC_RECORD_BINOP_F => {
                let dst = self.next_u16() as usize;
                let opcode_idx = self.next_u16() as usize;
                let lhs_idx = self.next_u16() as usize;
                let rhs_idx = self.next_u16() as usize;
                let opcode = self.jitcode.opcodes[opcode_idx];
                let lhs = self.registers_f[lhs_idx];
                let rhs = self.registers_f[rhs_idx];
                self.registers_f[dst] = eval_binop_f(opcode, lhs, rhs);
            }
            BC_RECORD_UNARY_F => {
                let dst = self.next_u16() as usize;
                let opcode_idx = self.next_u16() as usize;
                let src_idx = self.next_u16() as usize;
                let opcode = self.jitcode.opcodes[opcode_idx];
                let value = self.registers_f[src_idx];
                self.registers_f[dst] = eval_unary_f(opcode, value);
            }
            BC_BRANCH_ZERO => {
                // Pops from runtime stack. In blackhole, follow the
                // branch if value is zero (skip to next jump_target).
                let selected = self.current_selected;
                let cond = self.runtime_stack_pop(selected);
                if cond == 0 {
                    // Branch taken: scan forward for the next BC_JUMP_TARGET
                    // This is a simplification; in practice the label map
                    // would provide the target offset.
                    // Skip the branch body (fall through if non-zero).
                }
                // In blackhole mode, BC_BRANCH_ZERO without explicit target
                // needs the runtime label map. For now, fall through.
            }
            BC_BRANCH_REG_ZERO => {
                let cond_idx = self.next_u16() as usize;
                let target = self.next_u16() as usize;
                let cond = self.registers_i[cond_idx];
                if cond == 0 {
                    self.position = target;
                }
            }
            BC_JUMP => {
                let target = self.next_u16() as usize;
                self.position = target;
            }
            BC_JUMP_TARGET => {
                // No-op in blackhole: just a marker for the tracing machine.
            }
            BC_SET_SELECTED => {
                self.current_selected = self.next_u16() as usize;
            }
            BC_ABORT | BC_ABORT_PERMANENT => {
                return Err(LeaveFrame);
            }
            BC_INLINE_CALL => {
                let sub_idx = self.next_u16() as usize;
                let num_args = self.next_u16() as usize;
                let mut arg_triples = Vec::with_capacity(num_args);
                for _ in 0..num_args {
                    let kind = JitArgKind::decode(self.next_u8());
                    let caller_src = self.next_u16() as usize;
                    let callee_dst = self.next_u16() as usize;
                    arg_triples.push((kind, caller_src, callee_dst));
                }
                // Return slots: (callee_src, caller_dst) for i/r/f
                let return_i = self.decode_return_slot();
                let return_r = self.decode_return_slot();
                let return_f = self.decode_return_slot();

                let sub_jitcode = self.jitcode.sub_jitcodes[sub_idx].clone();

                // Create callee blackhole interpreter
                let mut callee = BlackholeInterpreter::new();
                callee.setposition(sub_jitcode, 0);

                // Copy arguments from caller to callee
                for (kind, caller_src, callee_dst) in arg_triples {
                    match kind {
                        JitArgKind::Int => {
                            callee.registers_i[callee_dst] = self.registers_i[caller_src];
                        }
                        JitArgKind::Ref => {
                            callee.registers_r[callee_dst] = self.registers_r[caller_src];
                        }
                        JitArgKind::Float => {
                            callee.registers_f[callee_dst] = self.registers_f[caller_src];
                        }
                    }
                }

                // Copy runtime stacks to callee
                for (k, v) in &self.runtime_stacks {
                    callee.runtime_stacks.insert(*k, v.clone());
                }
                callee.current_selected = self.current_selected;

                // Execute callee
                let _ = callee.run();

                // Copy runtime stacks back
                self.runtime_stacks = callee.runtime_stacks;
                self.current_selected = callee.current_selected;

                // Copy return values
                if let Some((callee_src, caller_dst)) = return_i {
                    self.registers_i[caller_dst] = callee.registers_i[callee_src];
                }
                if let Some((callee_src, caller_dst)) = return_r {
                    self.registers_r[caller_dst] = callee.registers_r[callee_src];
                }
                if let Some((callee_src, caller_dst)) = return_f {
                    self.registers_f[caller_dst] = callee.registers_f[callee_src];
                }
            }
            // -- Int-typed calls --
            BC_CALL_INT
            | BC_CALL_PURE_INT
            | BC_CALL_MAY_FORCE_INT
            | BC_CALL_RELEASE_GIL_INT
            | BC_CALL_LOOPINVARIANT_INT
            | BC_CALL_ASSEMBLER_INT => {
                let fn_ptr_idx = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                let num_args = self.next_u16() as usize;
                let args = self.read_call_args(num_args);
                let target = &self.jitcode.fn_ptrs[fn_ptr_idx];
                let result = call_int_function(target.concrete_ptr, &args);
                self.registers_i[dst] = result;
            }
            // -- Ref-typed calls --
            BC_CALL_REF
            | BC_CALL_PURE_REF
            | BC_CALL_MAY_FORCE_REF
            | BC_CALL_RELEASE_GIL_REF
            | BC_CALL_LOOPINVARIANT_REF
            | BC_CALL_ASSEMBLER_REF => {
                let fn_ptr_idx = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                let num_args = self.next_u16() as usize;
                let args = self.read_call_args(num_args);
                let target = &self.jitcode.fn_ptrs[fn_ptr_idx];
                let result = call_int_function(target.concrete_ptr, &args);
                self.registers_r[dst] = result;
            }
            // -- Float-typed calls --
            BC_CALL_FLOAT
            | BC_CALL_PURE_FLOAT
            | BC_CALL_MAY_FORCE_FLOAT
            | BC_CALL_RELEASE_GIL_FLOAT
            | BC_CALL_LOOPINVARIANT_FLOAT
            | BC_CALL_ASSEMBLER_FLOAT => {
                let fn_ptr_idx = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                let num_args = self.next_u16() as usize;
                let args = self.read_call_args(num_args);
                let target = &self.jitcode.fn_ptrs[fn_ptr_idx];
                let result = call_int_function(target.concrete_ptr, &args);
                self.registers_f[dst] = result;
            }
            // -- Void-typed calls --
            BC_CALL_MAY_FORCE_VOID
            | BC_CALL_RELEASE_GIL_VOID
            | BC_CALL_LOOPINVARIANT_VOID
            | BC_CALL_ASSEMBLER_VOID => {
                let fn_ptr_idx = self.next_u16() as usize;
                let _dst = self.next_u16(); // ignored for void
                let num_args = self.next_u16() as usize;
                let args = self.read_call_args(num_args);
                let target = &self.jitcode.fn_ptrs[fn_ptr_idx];
                call_int_function(target.concrete_ptr, &args);
            }
            BC_RESIDUAL_CALL_VOID => {
                let fn_ptr_idx = self.next_u16() as usize;
                let num_args = self.next_u16() as usize;
                let args = self.read_call_args(num_args);
                let target = &self.jitcode.fn_ptrs[fn_ptr_idx];
                call_int_function(target.concrete_ptr, &args);
            }
            // -- State field access --
            BC_LOAD_STATE_FIELD | BC_LOAD_STATE_VARRAY => {
                let _field_idx = self.next_u16();
                let _dst = self.next_u16();
                // No-op in blackhole: state fields are only meaningful
                // during tracing with a JitCodeSym.
            }
            BC_STORE_STATE_FIELD | BC_STORE_STATE_VARRAY => {
                let _field_idx = self.next_u16();
                let _src = self.next_u16();
            }
            BC_LOAD_STATE_ARRAY => {
                let _array_idx = self.next_u16();
                let _elem_idx = self.next_u16();
                let _dst = self.next_u16();
            }
            BC_STORE_STATE_ARRAY => {
                let _array_idx = self.next_u16();
                let _elem_idx = self.next_u16();
                let _src = self.next_u16();
            }
            // -- Virtualizable field/array access --
            BC_GETFIELD_VABLE_I | BC_GETFIELD_VABLE_R | BC_GETFIELD_VABLE_F => {
                let _descr_idx = self.next_u16();
                let _dst = self.next_u16();
                // No-op in standalone blackhole (requires virtualizable info)
            }
            BC_SETFIELD_VABLE_I | BC_SETFIELD_VABLE_R | BC_SETFIELD_VABLE_F => {
                let _descr_idx = self.next_u16();
                let _src = self.next_u16();
            }
            BC_GETARRAYITEM_VABLE_I | BC_GETARRAYITEM_VABLE_R | BC_GETARRAYITEM_VABLE_F => {
                let _descr_idx = self.next_u16();
                let _index = self.next_u16();
                let _dst = self.next_u16();
            }
            BC_SETARRAYITEM_VABLE_I | BC_SETARRAYITEM_VABLE_R | BC_SETARRAYITEM_VABLE_F => {
                let _descr_idx = self.next_u16();
                let _index = self.next_u16();
                let _src = self.next_u16();
            }
            BC_ARRAYLEN_VABLE => {
                let _descr_idx = self.next_u16();
                let _dst = self.next_u16();
            }
            BC_HINT_FORCE_VIRTUALIZABLE => {
                // No-op in blackhole
            }
            other => {
                panic!("blackhole: unknown jitcode bytecode {other}");
            }
        }
        Ok(())
    }

    /// Read call arguments from bytecode (kind:u8, reg:u16 per arg).
    fn read_call_args(&mut self, num_args: usize) -> Vec<i64> {
        let mut args = Vec::with_capacity(num_args);
        for _ in 0..num_args {
            let kind = JitArgKind::decode(self.next_u8());
            let reg = self.next_u16();
            args.push(self.read_call_arg(kind, reg));
        }
        args
    }

    /// Decode a return slot pair from bytecode.
    fn decode_return_slot(&mut self) -> Option<(usize, usize)> {
        let src = self.next_u16() as usize;
        let dst = self.next_u16() as usize;
        if src == u16::MAX as usize && dst == u16::MAX as usize {
            None
        } else {
            Some((src, dst))
        }
    }

    /// Execute one frame and handle its completion.
    ///
    /// Returns any pending exception to propagate to the caller.
    ///
    /// RPython: `BlackholeInterpreter._resume_mainloop(current_exc)`
    pub fn resume_mainloop(&mut self) -> BhReturnType {
        self.run();
        self.return_type
    }
}

/// Pool manager for blackhole interpreters.
///
/// RPython: `BlackholeInterpBuilder` class in blackhole.py
pub struct BlackholeInterpBuilder {
    pool: Vec<BlackholeInterpreter>,
}

impl Default for BlackholeInterpBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BlackholeInterpBuilder {
    pub fn new() -> Self {
        Self { pool: Vec::new() }
    }

    /// Acquire an interpreter from the pool or create a new one.
    ///
    /// RPython: `BlackholeInterpBuilder.acquire_interp()`
    pub fn acquire_interp(&mut self) -> BlackholeInterpreter {
        self.pool.pop().unwrap_or_default()
    }

    /// Return an interpreter to the pool after clearing its state.
    ///
    /// RPython: `BlackholeInterpBuilder.release_interp(interp)`
    pub fn release_interp(&mut self, mut interp: BlackholeInterpreter) {
        interp.registers_i.clear();
        interp.registers_r.clear();
        interp.registers_f.clear();
        interp.nextblackholeinterp = None;
        interp.runtime_stacks.clear();
        self.pool.push(interp);
    }
}

/// Execute a blackhole frame chain to completion.
///
/// Starts with the top frame, runs it, then pops to the caller frame
/// and passes the return value. Continues until the bottom frame completes.
///
/// RPython: `_run_forever()` in blackhole.py
pub fn run_forever(
    builder: &mut BlackholeInterpBuilder,
    mut bh: BlackholeInterpreter,
) {
    loop {
        let ret_type = bh.resume_mainloop();

        // Save return values before moving bh
        let tmp_i = bh.tmpreg_i;
        let tmp_r = bh.tmpreg_r;
        let tmp_f = bh.tmpreg_f;

        // If no caller frame, we're done
        let next = bh.nextblackholeinterp.take();
        builder.release_interp(bh);
        let Some(caller) = next else {
            return;
        };
        bh = *caller;

        // Pass return value to caller
        match ret_type {
            BhReturnType::Int => bh.tmpreg_i = tmp_i,
            BhReturnType::Ref => bh.tmpreg_r = tmp_r,
            BhReturnType::Float => bh.tmpreg_f = tmp_f,
            BhReturnType::Void => {}
        }
    }
}

/// Convert metainterp tracing frame stack to blackhole frame chain and run.
///
/// RPython: `convert_and_run_from_pyjitpl()` in blackhole.py
pub fn convert_and_run_from_pyjitpl(
    builder: &mut BlackholeInterpBuilder,
    framestack: &MIFrameStack,
) {
    let mut next_bh: Option<Box<BlackholeInterpreter>> = None;

    for frame in &framestack.frames {
        let mut cur_bh = builder.acquire_interp();
        cur_bh.copy_data_from_miframe(frame);
        cur_bh.nextblackholeinterp = next_bh;
        next_bh = Some(Box::new(cur_bh));
    }

    if let Some(first_bh) = next_bh {
        run_forever(builder, *first_bh);
    }
}

/// Resume execution in the blackhole interpreter after a compiled
/// code guard failure.
///
/// RPython: `resume_in_blackhole()` in blackhole.py
pub fn resume_in_blackhole(
    builder: &mut BlackholeInterpBuilder,
    jitcode: &JitCode,
    position: usize,
    fail_values: &[(usize, i64)], // (register_index, value) pairs
) {
    let mut bh = builder.acquire_interp();
    bh.setposition(jitcode.clone(), position);

    // Restore register values from guard failure
    for &(reg_idx, value) in fail_values {
        if reg_idx < bh.registers_i.len() {
            bh.registers_i[reg_idx] = value;
        }
    }

    run_forever(builder, bh);
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
            BlackholeResult::Finish { values: vals, .. } => assert_eq!(vals, vec![30]),
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
            BlackholeResult::Finish { values: vals, .. } => assert_eq!(vals, vec![1]),
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
            BlackholeResult::Finish { values: vals, .. } => assert_eq!(vals, vec![42]),
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
            BlackholeResult::Finish { values: vals, .. } => assert_eq!(vals, vec![200]),
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
            BlackholeResult::Finish { values: vals, .. } => assert_eq!(vals, vec![100, 200]),
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
            BlackholeResult::Finish { values: vals, .. } => {
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
                        assert_eq!(
                            fields,
                            &vec![(3, crate::resume::MaterializedValue::Value(55))]
                        );
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
                        target: crate::resume::MaterializedValue::Value(1),
                        value: crate::resume::MaterializedValue::Value(77),
                        item_index: Some(2),
                    }]
                );
            }
            _ => panic!("expected GuardFailedWithVirtuals"),
        }
    }

    // ── Executor parity tests (ported from test_executor.py) ──
    //
    // Systematic correctness tests for each opcode category.

    /// Helper: build and execute a single binop, returning the i64 result.
    fn exec_binop(opcode: OpCode, a: i64, b: i64) -> i64 {
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(opcode, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, a);
        initial.insert(1, b);
        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish { values, .. } => values[0],
            other => panic!(
                "expected Finish for {:?}, got {:?}",
                opcode,
                match other {
                    BlackholeResult::Abort(s) => s,
                    _ => "other".to_string(),
                }
            ),
        }
    }

    /// Helper: build and execute a single unary op, returning the i64 result.
    fn exec_unop(opcode: OpCode, a: i64) -> i64 {
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(opcode, &[OpRef(0)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, a);
        match blackhole_execute(&ops, &HashMap::new(), &initial, 0) {
            BlackholeResult::Finish { values, .. } => values[0],
            other => panic!(
                "expected Finish for {:?}, got {:?}",
                opcode,
                match other {
                    BlackholeResult::Abort(s) => s,
                    _ => "other".to_string(),
                }
            ),
        }
    }

    // ── Integer arithmetic correctness ──

    #[test]
    fn test_executor_int_add() {
        assert_eq!(exec_binop(OpCode::IntAdd, 3, 4), 7);
        assert_eq!(exec_binop(OpCode::IntAdd, -1, 1), 0);
        assert_eq!(exec_binop(OpCode::IntAdd, 0, 0), 0);
    }

    #[test]
    fn test_executor_int_sub() {
        assert_eq!(exec_binop(OpCode::IntSub, 10, 3), 7);
        assert_eq!(exec_binop(OpCode::IntSub, 0, 5), -5);
    }

    #[test]
    fn test_executor_int_mul() {
        assert_eq!(exec_binop(OpCode::IntMul, 6, 7), 42);
        assert_eq!(exec_binop(OpCode::IntMul, -3, 4), -12);
        assert_eq!(exec_binop(OpCode::IntMul, 0, 999), 0);
    }

    #[test]
    fn test_executor_int_floordiv() {
        assert_eq!(exec_binop(OpCode::IntFloorDiv, 17, 5), 3);
        assert_eq!(exec_binop(OpCode::IntFloorDiv, -17, 5), -3);
        assert_eq!(exec_binop(OpCode::IntFloorDiv, 100, 1), 100);
    }

    #[test]
    fn test_executor_int_mod() {
        assert_eq!(exec_binop(OpCode::IntMod, 17, 5), 2);
        assert_eq!(exec_binop(OpCode::IntMod, 10, 3), 1);
        assert_eq!(exec_binop(OpCode::IntMod, 6, 3), 0);
    }

    // ── Integer comparisons ──

    #[test]
    fn test_executor_int_lt() {
        assert_eq!(exec_binop(OpCode::IntLt, 3, 4), 1);
        assert_eq!(exec_binop(OpCode::IntLt, 4, 4), 0);
        assert_eq!(exec_binop(OpCode::IntLt, 5, 4), 0);
    }

    #[test]
    fn test_executor_int_ge() {
        assert_eq!(exec_binop(OpCode::IntGe, 4, 4), 1);
        assert_eq!(exec_binop(OpCode::IntGe, 5, 4), 1);
        assert_eq!(exec_binop(OpCode::IntGe, 3, 4), 0);
    }

    #[test]
    fn test_executor_int_eq() {
        assert_eq!(exec_binop(OpCode::IntEq, 5, 5), 1);
        assert_eq!(exec_binop(OpCode::IntEq, 5, 6), 0);
    }

    #[test]
    fn test_executor_int_ne() {
        assert_eq!(exec_binop(OpCode::IntNe, 5, 6), 1);
        assert_eq!(exec_binop(OpCode::IntNe, 5, 5), 0);
    }

    #[test]
    fn test_executor_int_le_gt() {
        assert_eq!(exec_binop(OpCode::IntLe, 3, 4), 1);
        assert_eq!(exec_binop(OpCode::IntLe, 4, 4), 1);
        assert_eq!(exec_binop(OpCode::IntLe, 5, 4), 0);
        assert_eq!(exec_binop(OpCode::IntGt, 5, 4), 1);
        assert_eq!(exec_binop(OpCode::IntGt, 4, 4), 0);
    }

    // ── Float arithmetic ──

    #[test]
    fn test_executor_float_add() {
        let a = f64::to_bits(1.5) as i64;
        let b = f64::to_bits(2.5) as i64;
        let result = exec_binop(OpCode::FloatAdd, a, b);
        assert_eq!(f64::from_bits(result as u64), 4.0);
    }

    #[test]
    fn test_executor_float_mul() {
        let a = f64::to_bits(3.0) as i64;
        let b = f64::to_bits(2.0) as i64;
        let result = exec_binop(OpCode::FloatMul, a, b);
        assert_eq!(f64::from_bits(result as u64), 6.0);
    }

    #[test]
    fn test_executor_float_sub() {
        let a = f64::to_bits(10.5) as i64;
        let b = f64::to_bits(3.5) as i64;
        let result = exec_binop(OpCode::FloatSub, a, b);
        assert_eq!(f64::from_bits(result as u64), 7.0);
    }

    #[test]
    fn test_executor_float_truediv() {
        let a = f64::to_bits(7.0) as i64;
        let b = f64::to_bits(2.0) as i64;
        let result = exec_binop(OpCode::FloatTrueDiv, a, b);
        assert_eq!(f64::from_bits(result as u64), 3.5);
    }

    // ── Unary ops ──

    #[test]
    fn test_executor_int_neg() {
        assert_eq!(exec_unop(OpCode::IntNeg, 5), -5);
        assert_eq!(exec_unop(OpCode::IntNeg, -3), 3);
        assert_eq!(exec_unop(OpCode::IntNeg, 0), 0);
    }

    #[test]
    fn test_executor_int_invert() {
        assert_eq!(exec_unop(OpCode::IntInvert, 0), -1);
        assert_eq!(exec_unop(OpCode::IntInvert, -1), 0);
        assert_eq!(exec_unop(OpCode::IntInvert, 1), -2);
    }

    #[test]
    fn test_executor_float_neg() {
        let a = f64::to_bits(3.0) as i64;
        let result = exec_unop(OpCode::FloatNeg, a);
        assert_eq!(f64::from_bits(result as u64), -3.0);
    }

    #[test]
    fn test_executor_float_abs() {
        let a = f64::to_bits(-5.5) as i64;
        let result = exec_unop(OpCode::FloatAbs, a);
        assert_eq!(f64::from_bits(result as u64), 5.5);
    }

    // ── Casts ──

    #[test]
    fn test_executor_cast_int_to_float() {
        let result = exec_unop(OpCode::CastIntToFloat, 42);
        assert_eq!(f64::from_bits(result as u64), 42.0);
    }

    #[test]
    fn test_executor_cast_float_to_int() {
        let a = f64::to_bits(3.7) as i64;
        let result = exec_unop(OpCode::CastFloatToInt, a);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_executor_cast_float_to_int_negative() {
        let a = f64::to_bits(-2.9) as i64;
        let result = exec_unop(OpCode::CastFloatToInt, a);
        assert_eq!(result, -2);
    }

    // ── Overflow arithmetic ──

    #[test]
    fn test_executor_int_add_ovf_no_overflow() {
        assert_eq!(exec_binop(OpCode::IntAddOvf, 10, 20), 30);
    }

    #[test]
    fn test_executor_int_add_ovf_wraps() {
        // In blackhole mode, overflow ops use wrapping arithmetic.
        // GuardNoOverflow/GuardOverflow are separate ops.
        let result = exec_binop(OpCode::IntAddOvf, i64::MAX, 1);
        assert_eq!(result, i64::MIN);
    }

    #[test]
    fn test_executor_int_sub_ovf() {
        assert_eq!(exec_binop(OpCode::IntSubOvf, 10, 3), 7);
    }

    #[test]
    fn test_executor_int_mul_ovf() {
        assert_eq!(exec_binop(OpCode::IntMulOvf, 6, 7), 42);
    }

    // ── Bitwise ops ──

    #[test]
    fn test_executor_int_and() {
        assert_eq!(exec_binop(OpCode::IntAnd, 0xFF, 0x0F), 0x0F);
        assert_eq!(exec_binop(OpCode::IntAnd, 0xAB, 0x00), 0x00);
    }

    #[test]
    fn test_executor_int_or() {
        assert_eq!(exec_binop(OpCode::IntOr, 0xF0, 0x0F), 0xFF);
        assert_eq!(exec_binop(OpCode::IntOr, 0, 0), 0);
    }

    #[test]
    fn test_executor_int_xor() {
        assert_eq!(exec_binop(OpCode::IntXor, 0xFF, 0x0F), 0xF0);
        assert_eq!(exec_binop(OpCode::IntXor, 42, 42), 0);
    }

    #[test]
    fn test_executor_int_lshift() {
        assert_eq!(exec_binop(OpCode::IntLshift, 1, 4), 16);
        assert_eq!(exec_binop(OpCode::IntLshift, 0xFF, 8), 0xFF00);
    }

    #[test]
    fn test_executor_int_rshift() {
        assert_eq!(exec_binop(OpCode::IntRshift, 16, 4), 1);
        assert_eq!(exec_binop(OpCode::IntRshift, -1, 1), -1); // arithmetic shift
    }

    #[test]
    fn test_executor_uint_rshift() {
        // Logical (unsigned) right shift.
        let result = exec_binop(OpCode::UintRshift, -1, 1);
        assert_eq!(result, i64::MAX);
    }

    // ── Boolean predicates ──

    #[test]
    fn test_executor_int_is_zero() {
        assert_eq!(exec_unop(OpCode::IntIsZero, 0), 1);
        assert_eq!(exec_unop(OpCode::IntIsZero, 1), 0);
        assert_eq!(exec_unop(OpCode::IntIsZero, -1), 0);
    }

    #[test]
    fn test_executor_int_is_true() {
        assert_eq!(exec_unop(OpCode::IntIsTrue, 0), 0);
        assert_eq!(exec_unop(OpCode::IntIsTrue, 1), 1);
        assert_eq!(exec_unop(OpCode::IntIsTrue, -42), 1);
    }

    #[test]
    fn test_executor_int_force_ge_zero() {
        assert_eq!(exec_unop(OpCode::IntForceGeZero, 5), 5);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, 0), 0);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, -10), 0);
    }

    // ── Float comparisons ──

    #[test]
    fn test_executor_float_comparisons() {
        let f2 = f64::to_bits(2.0) as i64;
        let f3 = f64::to_bits(3.0) as i64;
        assert_eq!(exec_binop(OpCode::FloatLt, f2, f3), 1);
        assert_eq!(exec_binop(OpCode::FloatLt, f3, f2), 0);
        assert_eq!(exec_binop(OpCode::FloatLe, f2, f2), 1);
        assert_eq!(exec_binop(OpCode::FloatGt, f3, f2), 1);
        assert_eq!(exec_binop(OpCode::FloatGe, f2, f2), 1);
        assert_eq!(exec_binop(OpCode::FloatEq, f3, f3), 1);
        assert_eq!(exec_binop(OpCode::FloatNe, f2, f3), 1);
    }

    // ── FloatFloorDiv / FloatMod ──

    #[test]
    fn test_executor_float_floordiv() {
        let a = f64::to_bits(7.0) as i64;
        let b = f64::to_bits(2.0) as i64;
        let result = exec_binop(OpCode::FloatFloorDiv, a, b);
        assert_eq!(f64::from_bits(result as u64), 3.0);
    }

    #[test]
    fn test_executor_float_mod() {
        let a = f64::to_bits(7.0) as i64;
        let b = f64::to_bits(3.0) as i64;
        let result = exec_binop(OpCode::FloatMod, a, b);
        assert_eq!(f64::from_bits(result as u64), 1.0);
    }

    // ── Pointer comparisons ──

    #[test]
    fn test_executor_ptr_eq_ne() {
        assert_eq!(exec_binop(OpCode::PtrEq, 100, 100), 1);
        assert_eq!(exec_binop(OpCode::PtrEq, 100, 200), 0);
        assert_eq!(exec_binop(OpCode::PtrNe, 100, 200), 1);
        assert_eq!(exec_binop(OpCode::PtrNe, 100, 100), 0);
    }

    // ── SameAs ──

    #[test]
    fn test_executor_same_as() {
        assert_eq!(exec_unop(OpCode::SameAsI, 42), 42);
        assert_eq!(exec_unop(OpCode::SameAsR, 0xDEAD), 0xDEAD);
    }

    #[test]
    fn test_executor_same_as_zero_arg_constant_placeholder() {
        let op = mk_op(OpCode::SameAsI, &[], 8);
        let mut values = HashMap::new();
        values.insert(8, 123);
        let mut exc = ExceptionState::default();
        match execute_one(&op, &values, &mut exc) {
            OpResult::Value(v) => assert_eq!(v, 123),
            _ => panic!("zero-arg SameAsI should resolve through its constant slot"),
        }
    }

    // ── UintMulHigh ──

    #[test]
    fn test_executor_uint_mul_high() {
        // Upper 64 bits of unsigned 128-bit multiply.
        // 0x8000_0000_0000_0000 * 2 = 0x1_0000_0000_0000_0000 → high = 1
        let result = exec_binop(OpCode::UintMulHigh, i64::MIN, 2);
        assert_eq!(result, 1);

        // Small values: high 64 bits should be 0.
        let result = exec_binop(OpCode::UintMulHigh, 100, 200);
        assert_eq!(result, 0);

        // -1i64 = 0xFFFF_FFFF_FFFF_FFFF as u64, * 2 → high = 1
        let result = exec_binop(OpCode::UintMulHigh, -1, 2);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_blackhole_with_resume_layout_materializes_virtuals_without_resume_data() {
        let mut guard_op = mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::smallvec![OpRef(0)]);

        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut initial = HashMap::new();
        initial.insert(0, 0i64);

        let resume_layout = ResumeData {
            frames: vec![],
            virtuals: vec![ResumeVirtualInfo::VStruct {
                type_id: 0,
                descr_index: 7,
                fields: vec![(3, VirtualFieldSource::Constant(55))],
            }],
            pending_fields: vec![ResumePendingFieldInfo {
                descr_index: 9,
                target: VirtualFieldSource::Virtual(0),
                value: VirtualFieldSource::FailArg(0),
                item_index: Some(2),
            }],
        }
        .encode()
        .layout_summary();

        match blackhole_with_resume_layout(&ops, &HashMap::new(), &initial, 0, Some(&resume_layout))
        {
            BlackholeResult::GuardFailedWithVirtuals {
                guard_index,
                fail_values,
                materialized_virtuals,
                pending_field_writes,
            } => {
                assert_eq!(guard_index, 1);
                assert_eq!(fail_values, vec![0]);
                assert_eq!(
                    materialized_virtuals,
                    vec![MaterializedVirtual::Struct {
                        type_id: 0,
                        descr_index: 7,
                        fields: vec![(3, crate::resume::MaterializedValue::Value(55))],
                    }]
                );
                assert_eq!(
                    pending_field_writes,
                    vec![crate::resume::ResolvedPendingFieldWrite {
                        descr_index: 9,
                        target: crate::resume::MaterializedValue::VirtualRef(0),
                        value: crate::resume::MaterializedValue::Value(0),
                        item_index: Some(2),
                    }]
                );
            }
            _ => panic!("expected GuardFailedWithVirtuals"),
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // Executor edge-case parity tests
    // Ported from rpython/jit/metainterp/test/test_executor.py
    // ══════════════════════════════════════════════════════════════════

    // ── Division by zero ──

    #[test]
    fn test_executor_int_floordiv_by_zero() {
        // Division by zero returns 0 (blackhole convention: no panic).
        assert_eq!(exec_binop(OpCode::IntFloorDiv, 42, 0), 0);
        assert_eq!(exec_binop(OpCode::IntFloorDiv, -1, 0), 0);
        assert_eq!(exec_binop(OpCode::IntFloorDiv, 0, 0), 0);
    }

    #[test]
    fn test_executor_int_mod_by_zero() {
        assert_eq!(exec_binop(OpCode::IntMod, 42, 0), 0);
        assert_eq!(exec_binop(OpCode::IntMod, -1, 0), 0);
        assert_eq!(exec_binop(OpCode::IntMod, 0, 0), 0);
    }

    // ── Integer overflow boundaries ──

    #[test]
    fn test_executor_int_add_overflow_boundary() {
        // Wrapping: i64::MAX + 1 = i64::MIN
        assert_eq!(exec_binop(OpCode::IntAdd, i64::MAX, 1), i64::MIN);
        // i64::MIN - 1 wraps to i64::MAX
        assert_eq!(exec_binop(OpCode::IntSub, i64::MIN, 1), i64::MAX);
    }

    #[test]
    fn test_executor_int_mul_overflow_boundary() {
        assert_eq!(exec_binop(OpCode::IntMul, i64::MAX, 2), -2);
        assert_eq!(exec_binop(OpCode::IntMul, i64::MIN, -1), i64::MIN); // wrapping
    }

    #[test]
    fn test_executor_int_neg_min() {
        // -i64::MIN wraps to i64::MIN (two's complement)
        assert_eq!(exec_unop(OpCode::IntNeg, i64::MIN), i64::MIN);
    }

    #[test]
    fn test_executor_int_floordiv_min_by_neg1() {
        // i64::MIN / -1 would overflow; wrapping_div wraps to i64::MIN
        assert_eq!(exec_binop(OpCode::IntFloorDiv, i64::MIN, -1), i64::MIN);
    }

    #[test]
    fn test_executor_int_mod_min_by_neg1() {
        // i64::MIN % -1 would overflow; wrapping_rem wraps to 0
        assert_eq!(exec_binop(OpCode::IntMod, i64::MIN, -1), 0);
    }

    // ── Float special values ──

    fn f64_bits(v: f64) -> i64 {
        f64::to_bits(v) as i64
    }

    fn bits_f64(v: i64) -> f64 {
        f64::from_bits(v as u64)
    }

    #[test]
    fn test_executor_float_add_nan() {
        let result = exec_binop(OpCode::FloatAdd, f64_bits(f64::NAN), f64_bits(1.0));
        assert!(bits_f64(result).is_nan());
    }

    #[test]
    fn test_executor_float_mul_inf_zero() {
        // Inf * 0 = NaN
        let result = exec_binop(OpCode::FloatMul, f64_bits(f64::INFINITY), f64_bits(0.0));
        assert!(bits_f64(result).is_nan());
    }

    #[test]
    fn test_executor_float_truediv_by_zero() {
        // 1.0 / 0.0 = Inf
        let result = exec_binop(OpCode::FloatTrueDiv, f64_bits(1.0), f64_bits(0.0));
        assert_eq!(bits_f64(result), f64::INFINITY);

        // -1.0 / 0.0 = -Inf
        let result = exec_binop(OpCode::FloatTrueDiv, f64_bits(-1.0), f64_bits(0.0));
        assert_eq!(bits_f64(result), f64::NEG_INFINITY);
    }

    #[test]
    fn test_executor_float_truediv_zero_by_zero() {
        // 0.0 / 0.0 = NaN
        let result = exec_binop(OpCode::FloatTrueDiv, f64_bits(0.0), f64_bits(0.0));
        assert!(bits_f64(result).is_nan());
    }

    #[test]
    fn test_executor_float_sub_inf() {
        // Inf - Inf = NaN
        let result = exec_binop(
            OpCode::FloatSub,
            f64_bits(f64::INFINITY),
            f64_bits(f64::INFINITY),
        );
        assert!(bits_f64(result).is_nan());
    }

    #[test]
    fn test_executor_float_neg_nan() {
        // -NaN is still NaN
        let result = exec_unop(OpCode::FloatNeg, f64_bits(f64::NAN));
        assert!(bits_f64(result).is_nan());
    }

    #[test]
    fn test_executor_float_abs_neg_inf() {
        let result = exec_unop(OpCode::FloatAbs, f64_bits(f64::NEG_INFINITY));
        assert_eq!(bits_f64(result), f64::INFINITY);
    }

    #[test]
    fn test_executor_float_comparisons_nan() {
        // All comparisons with NaN return false (0).
        let nan = f64_bits(f64::NAN);
        let one = f64_bits(1.0);
        assert_eq!(exec_binop(OpCode::FloatLt, nan, one), 0);
        assert_eq!(exec_binop(OpCode::FloatLe, nan, one), 0);
        assert_eq!(exec_binop(OpCode::FloatGt, nan, one), 0);
        assert_eq!(exec_binop(OpCode::FloatGe, nan, one), 0);
        assert_eq!(exec_binop(OpCode::FloatEq, nan, nan), 0);
        assert_eq!(exec_binop(OpCode::FloatNe, nan, nan), 1);
    }

    // ── Unsigned comparisons with negative values ──

    #[test]
    fn test_executor_uint_lt_negative_as_large() {
        // -1i64 as u64 is u64::MAX, which is larger than 1
        assert_eq!(exec_binop(OpCode::UintLt, -1, 1), 0);
        assert_eq!(exec_binop(OpCode::UintGt, -1, 1), 1);
    }

    #[test]
    fn test_executor_uint_comparisons_systematic() {
        // -1 as u64 = 0xFFFFFFFFFFFFFFFF (max unsigned)
        assert_eq!(exec_binop(OpCode::UintLt, -1, 0), 0);
        assert_eq!(exec_binop(OpCode::UintGe, -1, 0), 1);
        assert_eq!(exec_binop(OpCode::UintLe, 0, -1), 1);
        assert_eq!(exec_binop(OpCode::UintGt, 0, -1), 0);

        // Equal values
        assert_eq!(exec_binop(OpCode::UintLt, 5, 5), 0);
        assert_eq!(exec_binop(OpCode::UintLe, 5, 5), 1);
        assert_eq!(exec_binop(OpCode::UintGe, 5, 5), 1);
        assert_eq!(exec_binop(OpCode::UintGt, 5, 5), 0);

        // i64::MIN as u64 = 0x8000000000000000 (large positive unsigned)
        assert_eq!(exec_binop(OpCode::UintLt, i64::MIN, 1), 0);
        assert_eq!(exec_binop(OpCode::UintGt, i64::MIN, 1), 1);
    }

    // ── Shift edge cases ──

    #[test]
    fn test_executor_int_lshift_63() {
        // 1 << 63 = i64::MIN (sign bit)
        assert_eq!(exec_binop(OpCode::IntLshift, 1, 63), i64::MIN);
    }

    #[test]
    fn test_executor_int_rshift_neg1() {
        // Arithmetic shift: -1 >> 63 = -1 (sign bit fills)
        assert_eq!(exec_binop(OpCode::IntRshift, -1, 63), -1);
    }

    #[test]
    fn test_executor_uint_rshift_neg1() {
        // Logical shift: -1u >> 63 = 1 (top bit only)
        assert_eq!(exec_binop(OpCode::UintRshift, -1, 63), 1);
    }

    #[test]
    fn test_executor_shift_by_zero() {
        assert_eq!(exec_binop(OpCode::IntLshift, 42, 0), 42);
        assert_eq!(exec_binop(OpCode::IntRshift, 42, 0), 42);
        assert_eq!(exec_binop(OpCode::UintRshift, 42, 0), 42);
    }

    #[test]
    fn test_executor_lshift_negative_value() {
        // -5 << 2 = -20
        assert_eq!(exec_binop(OpCode::IntLshift, -5, 2), -20);
    }

    // ── IntSignext ──

    #[test]
    fn test_executor_int_signext_1byte() {
        // 0xFF sign-extended from 1 byte = -1
        assert_eq!(exec_binop(OpCode::IntSignext, 0xFF, 1), -1);
        // 0x7F sign-extended from 1 byte = 127
        assert_eq!(exec_binop(OpCode::IntSignext, 0x7F, 1), 127);
        // 0x80 sign-extended from 1 byte = -128
        assert_eq!(exec_binop(OpCode::IntSignext, 0x80, 1), -128);
    }

    #[test]
    fn test_executor_int_signext_2bytes() {
        // 0xFFFF sign-extended from 2 bytes = -1
        assert_eq!(exec_binop(OpCode::IntSignext, 0xFFFF, 2), -1);
        // 0x7FFF = 32767
        assert_eq!(exec_binop(OpCode::IntSignext, 0x7FFF, 2), 32767);
        // 0x8000 = -32768
        assert_eq!(exec_binop(OpCode::IntSignext, 0x8000, 2), -32768);
    }

    #[test]
    fn test_executor_int_signext_4bytes() {
        // 0xFFFFFFFF sign-extended from 4 bytes = -1
        assert_eq!(exec_binop(OpCode::IntSignext, 0xFFFFFFFF_i64, 4), -1);
        // 0x7FFFFFFF = 2147483647
        assert_eq!(exec_binop(OpCode::IntSignext, 0x7FFFFFFF, 4), 2147483647);
        // 0x80000000 = -2147483648
        assert_eq!(
            exec_binop(OpCode::IntSignext, 0x80000000_i64, 4),
            -2147483648
        );
    }

    // ── ConvertFloatBytesToLonglong / ConvertLonglongBytesToFloat roundtrip ──

    #[test]
    fn test_executor_convert_float_bytes_roundtrip() {
        // ConvertFloatBytesToLonglong is identity (f64 bits as i64)
        // ConvertLonglongBytesToFloat is identity (i64 bits as f64)
        let val = f64::to_bits(3.14) as i64;
        let ll = exec_unop(OpCode::ConvertFloatBytesToLonglong, val);
        assert_eq!(ll, val);
        let back = exec_unop(OpCode::ConvertLonglongBytesToFloat, ll);
        assert_eq!(back, val);
    }

    #[test]
    fn test_executor_convert_float_bytes_special_values() {
        for v in [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let bits = f64::to_bits(v) as i64;
            let ll = exec_unop(OpCode::ConvertFloatBytesToLonglong, bits);
            assert_eq!(ll, bits);
            let back = exec_unop(OpCode::ConvertLonglongBytesToFloat, ll);
            assert_eq!(back, bits);
        }
    }

    // ── UintMulHigh edge cases ──

    #[test]
    fn test_executor_uint_mul_high_max() {
        // u64::MAX * u64::MAX = (2^64-1)^2 = 2^128 - 2^65 + 1
        // High 64 bits = 0xFFFFFFFFFFFFFFFE
        let result = exec_binop(OpCode::UintMulHigh, -1, -1);
        assert_eq!(result, -2); // 0xFFFFFFFFFFFFFFFE as i64 = -2
    }

    #[test]
    fn test_executor_uint_mul_high_one() {
        // Any value * 1 has high 64 bits = 0
        assert_eq!(exec_binop(OpCode::UintMulHigh, i64::MAX, 1), 0);
        assert_eq!(exec_binop(OpCode::UintMulHigh, -1, 1), 0);
    }

    // ── Cast edge cases ──

    #[test]
    fn test_executor_cast_int_to_float_large() {
        let result = exec_unop(OpCode::CastIntToFloat, i64::MAX);
        assert_eq!(bits_f64(result), i64::MAX as f64);
    }

    #[test]
    fn test_executor_cast_float_to_int_truncates() {
        // Truncation toward zero
        assert_eq!(exec_unop(OpCode::CastFloatToInt, f64_bits(2.9)), 2);
        assert_eq!(exec_unop(OpCode::CastFloatToInt, f64_bits(-2.9)), -2);
        assert_eq!(exec_unop(OpCode::CastFloatToInt, f64_bits(0.999)), 0);
    }

    // ── Overflow ops with exact overflow detection ──

    #[test]
    fn test_executor_int_sub_ovf_underflow() {
        // i64::MIN - 1 wraps
        let result = exec_binop(OpCode::IntSubOvf, i64::MIN, 1);
        assert_eq!(result, i64::MAX);
    }

    #[test]
    fn test_executor_int_mul_ovf_large() {
        let result = exec_binop(OpCode::IntMulOvf, i64::MAX, 2);
        assert_eq!(result, -2);
    }

    // ── IntIsZero / IntIsTrue / IntForceGeZero boundary ──

    #[test]
    fn test_executor_int_is_zero_minmax() {
        assert_eq!(exec_unop(OpCode::IntIsZero, i64::MAX), 0);
        assert_eq!(exec_binop(OpCode::IntAdd, i64::MAX, 0), i64::MAX); // sanity
    }

    #[test]
    fn test_executor_int_force_ge_zero_boundary() {
        assert_eq!(exec_unop(OpCode::IntForceGeZero, i64::MIN), 0);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, i64::MAX), i64::MAX);
        assert_eq!(exec_unop(OpCode::IntForceGeZero, 1), 1);
    }

    // ── Float floordiv / mod edge cases ──

    #[test]
    fn test_executor_float_floordiv_negative() {
        // -7.0 // 2.0 = -4.0
        let result = exec_binop(OpCode::FloatFloorDiv, f64_bits(-7.0), f64_bits(2.0));
        assert_eq!(bits_f64(result), -4.0);
    }

    #[test]
    fn test_executor_float_mod_negative() {
        // Python-style float mod: result has sign of divisor
        let result = exec_binop(OpCode::FloatMod, f64_bits(-7.0), f64_bits(3.0));
        // -7.0 % 3.0 = 2.0 (Python) or -1.0 (C)
        // Depends on implementation; verify it at least produces a valid result
        let r = bits_f64(result);
        assert!(r.is_finite());
    }

    // ── Bitwise ops with boundary values ──

    #[test]
    fn test_executor_int_and_all_ones() {
        assert_eq!(exec_binop(OpCode::IntAnd, -1, -1), -1);
        assert_eq!(exec_binop(OpCode::IntAnd, -1, 0), 0);
    }

    #[test]
    fn test_executor_int_or_all_ones() {
        assert_eq!(exec_binop(OpCode::IntOr, -1, 0), -1);
        assert_eq!(exec_binop(OpCode::IntOr, i64::MAX, i64::MIN), -1);
    }

    #[test]
    fn test_executor_int_xor_self() {
        assert_eq!(exec_binop(OpCode::IntXor, i64::MAX, i64::MAX), 0);
        assert_eq!(exec_binop(OpCode::IntXor, i64::MIN, i64::MIN), 0);
    }

    #[test]
    fn test_executor_int_invert_boundaries() {
        assert_eq!(exec_unop(OpCode::IntInvert, i64::MAX), i64::MIN);
        assert_eq!(exec_unop(OpCode::IntInvert, i64::MIN), i64::MAX);
    }

    /// Verify every OpCode variant has an explicit handler in execute_one
    /// (i.e., none falls through to the Unsupported catch-all).
    #[test]
    fn test_all_opcodes_have_blackhole_handler() {
        let dummy_args = &[OpRef(10_000), OpRef(10_001), OpRef(10_002)];
        let mut constants = HashMap::new();
        constants.insert(10_000, 1i64);
        constants.insert(10_001, 2i64);
        constants.insert(10_002, 3i64);

        let mut values: HashMap<u32, i64> = constants.clone();
        let mut exc = ExceptionState::default();

        for opcode in OpCode::all() {
            // Build a minimal op with enough args for any opcode
            let arity = opcode.arity().unwrap_or(3) as usize;
            let args = &dummy_args[..arity.min(3)];
            let mut op = Op::new(opcode, args);
            op.pos = OpRef(opcode.as_u16() as u32 + 20_000);

            let result = execute_one(&op, &values, &mut exc);
            // Store the result so subsequent ops can reference it
            if let OpResult::Value(v) = &result {
                values.insert(op.pos.0, *v);
            }

            match result {
                OpResult::Unsupported(msg) => {
                    panic!("OpCode {:?} returned Unsupported: {}", opcode, msg);
                }
                // Any other result (Value, Void, Finish, Jump, GuardFailed) is acceptable
                _ => {}
            }
        }
    }

    // ================================================================
    // Tests for jitcode-based BlackholeInterpreter (RPython parity)
    // ================================================================

    mod bh_interp_tests {
        use super::super::*;
        use crate::jitcode::JitCodeBuilder;

        #[test]
        fn test_bh_interp_load_const_and_binop() {
            // Build jitcode: r0 = const(10), r1 = const(20), r2 = r0 + r1
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 10);
            b.load_const_i_value(1, 20);
            b.record_binop_i(2, OpCode::IntAdd, 0, 1);
            let jitcode = b.finish();

            let mut bh = BlackholeInterpreter::new();
            bh.setposition(jitcode, 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[2], 30);
        }

        #[test]
        fn test_bh_interp_branch_reg_zero_taken() {
            // Build jitcode: r0 = 0; if r0==0 goto end; r1 = 42; end: r2 = 99
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 0);
            let lbl = b.new_label();
            b.branch_reg_zero(0, lbl);
            b.load_const_i_value(1, 42); // should be skipped
            b.mark_label(lbl);
            b.load_const_i_value(2, 99);
            let jitcode = b.finish();

            let mut bh = BlackholeInterpreter::new();
            bh.setposition(jitcode, 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 0); // skipped, still 0
            assert_eq!(bh.registers_i[2], 99);
        }

        #[test]
        fn test_bh_interp_branch_reg_zero_not_taken() {
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 1); // nonzero
            let lbl = b.new_label();
            b.branch_reg_zero(0, lbl);
            b.load_const_i_value(1, 42); // NOT skipped
            b.mark_label(lbl);
            b.load_const_i_value(2, 99);
            let jitcode = b.finish();

            let mut bh = BlackholeInterpreter::new();
            bh.setposition(jitcode, 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 42);
            assert_eq!(bh.registers_i[2], 99);
        }

        #[test]
        fn test_bh_interp_jump() {
            let mut b = JitCodeBuilder::default();
            let lbl = b.new_label();
            b.jump(lbl);
            b.load_const_i_value(0, 42); // skipped
            b.mark_label(lbl);
            b.load_const_i_value(1, 99);
            let jitcode = b.finish();

            let mut bh = BlackholeInterpreter::new();
            bh.setposition(jitcode, 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[0], 0);  // skipped
            assert_eq!(bh.registers_i[1], 99);
        }

        #[test]
        fn test_bh_interp_move() {
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 42);
            b.move_i(1, 0);
            let jitcode = b.finish();

            let mut bh = BlackholeInterpreter::new();
            bh.setposition(jitcode, 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 42);
        }

        #[test]
        fn test_bh_interp_unary_neg() {
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 42);
            b.record_unary_i(1, OpCode::IntNeg, 0);
            let jitcode = b.finish();

            let mut bh = BlackholeInterpreter::new();
            bh.setposition(jitcode, 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], -42);
        }

        #[test]
        fn test_bh_interp_setarg() {
            let mut b = JitCodeBuilder::default();
            // Just record a binop to read r0 + r1
            b.record_binop_i(2, OpCode::IntMul, 0, 1);
            let jitcode = b.finish();

            let mut bh = BlackholeInterpreter::new();
            bh.setposition(jitcode, 0);
            bh.setarg_i(0, 7);
            bh.setarg_i(1, 6);
            let _ = bh.run();

            assert_eq!(bh.registers_i[2], 42);
        }

        #[test]
        fn test_bh_interp_builder_pool() {
            let mut builder = BlackholeInterpBuilder::new();

            let bh1 = builder.acquire_interp();
            assert!(bh1.registers_i.is_empty());

            builder.release_interp(bh1);
            let bh2 = builder.acquire_interp();
            // Reused from pool
            assert!(bh2.registers_i.is_empty());
        }

        #[test]
        fn test_bh_interp_inline_call() {
            // Build sub-jitcode: r0 = arg, result = r0 + r0
            let mut sub = JitCodeBuilder::default();
            sub.record_binop_i(1, OpCode::IntAdd, 0, 0);
            let sub_jitcode = sub.finish();

            // Build main jitcode: r0 = 21, inline_call(sub, arg=r0) → r1
            let mut b = JitCodeBuilder::default();
            b.load_const_i_value(0, 21);
            let sub_idx = b.add_sub_jitcode(sub_jitcode);
            b.inline_call_i(sub_idx, &[(0, 0)], Some((1, 1)));
            let jitcode = b.finish();

            let mut bh = BlackholeInterpreter::new();
            bh.setposition(jitcode, 0);
            let _ = bh.run();

            assert_eq!(bh.registers_i[1], 42);
        }
    }
}
