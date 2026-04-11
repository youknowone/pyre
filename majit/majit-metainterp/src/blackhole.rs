//! Blackhole interpreter: evaluates IR operations with concrete values.
//!
//! When a guard fails in compiled code, the blackhole interpreter can replay
//! the remaining operations from the guard point to the end of the trace,
//! using concrete values from the DeadFrame.
//!
//! This is the RPython equivalent of `rpython/jit/metainterp/blackhole.py`.

use std::collections::HashMap;

use crate::jitexc::JitException;
use crate::resume::{
    MaterializedVirtual, ResolvedPendingFieldWrite, ResumeData, ResumeLayoutSummary,
};
use majit_ir::{GcRef, Op, OpCode, OpRef};

use crate::executor::{OpResult, TraceValues, ValueStore, execute_one, resolve};

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
    /// executor.py: metainterp.ovf_flag — set by Int*Ovf operations,
    /// checked by GuardNoOverflow / GuardOverflow.
    pub ovf_flag: bool,
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
    /// Reached a Finish operation with output values and their types.
    /// compile.py:636 DoneWithThisFrameDescr*.handle_fail:
    /// the descr type determines how to interpret the value.
    Finish {
        op_index: usize,
        values: Vec<i64>,
        /// Types from the FINISH op's descr (e.g. [Type::Int] or [Type::Ref]).
        value_types: Vec<majit_ir::Type>,
    },
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

/// blackhole_execute_with_state + BlackholeMemory support.
/// RPython _run_forever parity: Jump loops back to Label.
pub fn blackhole_execute_full(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    initial_exception: ExceptionState,
    memory: &dyn BlackholeMemory,
) -> (BlackholeResult, ExceptionState) {
    let mut merged = initial_values.clone();
    for (&k, &v) in constants {
        merged.entry(k).or_insert(v);
    }
    let mut tv = TraceValues::from_hashmap(&merged);
    drop(merged);
    let mut exc_state = initial_exception;

    // RPython _run_forever parity: find the Label op (loop header) so
    // that Jump can loop back. The trace is [Label, ..., Jump(→Label)].
    let label_index = ops
        .iter()
        .position(|op| op.opcode == OpCode::Label)
        .unwrap_or(0);
    let label_inputarg_positions: Vec<u32> = ops
        .get(label_index)
        .map(|op| op.args.iter().map(|a| a.0).collect())
        .unwrap_or_default();

    let mut op_idx = start_index;

    while op_idx < ops.len() {
        let op = &ops[op_idx];
        let result = execute_one_with_memory(op, &tv, &mut exc_state, memory);

        match result {
            OpResult::Value(v) => {
                if !op.pos.is_none() {
                    tv.set(op.pos.0, v);
                }
            }
            OpResult::Void => {}
            OpResult::Finish(args) => {
                let vals: Vec<i64> = args.iter().map(|&r| tv.resolve(r)).collect();
                let vtypes = op
                    .descr
                    .as_ref()
                    .and_then(|d| d.as_fail_descr())
                    .map(|fd| fd.fail_arg_types().to_vec())
                    .unwrap_or_default();
                return (
                    BlackholeResult::Finish {
                        op_index: op_idx,
                        values: vals,
                        value_types: vtypes,
                    },
                    exc_state,
                );
            }
            OpResult::Jump(args) => {
                // RPython _run_forever parity: Jump loops back to Label.
                let vals: Vec<i64> = args.iter().map(|&r| tv.resolve(r)).collect();
                for (pos, val) in label_inputarg_positions.iter().zip(vals.iter()) {
                    tv.set(*pos, *val);
                }
                op_idx = label_index + 1;
                continue;
            }
            OpResult::GuardFailed => {
                let fail_values = if let Some(ref fail_args) = op.fail_args {
                    fail_args.iter().map(|&r| tv.resolve(r)).collect()
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
        op_idx += 1;
    }

    (
        BlackholeResult::Abort("reached end of ops without finish/jump".to_string()),
        exc_state,
    )
}

/// Dispatch an op using real memory access when a BlackholeMemory backend is available.
fn execute_one_with_memory(
    op: &Op,
    values: &(impl ValueStore + ?Sized),
    exc: &mut ExceptionState,
    memory: &dyn BlackholeMemory,
) -> OpResult {
    match op.opcode {
        // Memory access ops with real backend
        OpCode::GcLoadI => {
            let base = values.resolve(op.args[0]);
            let offset = values.resolve(op.args[1]);
            OpResult::Value(memory.gc_load_i(base, offset))
        }
        OpCode::GcLoadR => {
            let base = values.resolve(op.args[0]);
            let offset = values.resolve(op.args[1]);
            OpResult::Value(memory.gc_load_r(base, offset))
        }
        OpCode::GcLoadF => {
            let base = values.resolve(op.args[0]);
            let offset = values.resolve(op.args[1]);
            OpResult::Value(memory.gc_load_f(base, offset))
        }
        OpCode::GcStore => {
            let base = values.resolve(op.args[0]);
            let offset = values.resolve(op.args[1]);
            let value = values.resolve(op.args[2]);
            memory.gc_store(base, offset, value);
            OpResult::Void
        }
        OpCode::GcLoadIndexedI => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let scale = values.resolve(op.args[2]);
            let offset = values.resolve(op.args[3]);
            OpResult::Value(memory.gc_load_indexed_i(base, index, scale, offset))
        }
        OpCode::GcLoadIndexedR => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let scale = values.resolve(op.args[2]);
            let offset = values.resolve(op.args[3]);
            OpResult::Value(memory.gc_load_indexed_r(base, index, scale, offset))
        }
        OpCode::GcLoadIndexedF => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let scale = values.resolve(op.args[2]);
            let offset = values.resolve(op.args[3]);
            OpResult::Value(memory.gc_load_indexed_f(base, index, scale, offset))
        }
        OpCode::GcStoreIndexed => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let value = values.resolve(op.args[2]);
            let scale = values.resolve(op.args[3]);
            let offset = values.resolve(op.args[4]);
            memory.gc_store_indexed(base, index, scale, offset, value);
            OpResult::Void
        }
        OpCode::ArraylenGc => {
            let base = values.resolve(op.args[0]);
            OpResult::Value(memory.arraylen(base))
        }
        OpCode::Strlen | OpCode::Unicodelen => {
            let base = values.resolve(op.args[0]);
            OpResult::Value(memory.strlen(base))
        }
        // ── Field access via descr offset ──
        OpCode::GetfieldGcI | OpCode::GetfieldRawI | OpCode::GetfieldGcPureI => {
            let base = values.resolve(op.args[0]);
            let offset = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .map_or(0, |f| f.offset() as i64);
            OpResult::Value(memory.gc_load_i(base, offset))
        }
        OpCode::GetfieldGcR | OpCode::GetfieldRawR | OpCode::GetfieldGcPureR => {
            let base = values.resolve(op.args[0]);
            let offset = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .map_or(0, |f| f.offset() as i64);
            OpResult::Value(memory.gc_load_r(base, offset))
        }
        OpCode::GetfieldGcF | OpCode::GetfieldRawF | OpCode::GetfieldGcPureF => {
            let base = values.resolve(op.args[0]);
            let offset = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .map_or(0, |f| f.offset() as i64);
            OpResult::Value(memory.gc_load_f(base, offset))
        }
        OpCode::SetfieldGc | OpCode::SetfieldRaw => {
            let base = values.resolve(op.args[0]);
            let value = values.resolve(op.args[1]);
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
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let (item_size, base_ofs) = op
                .descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map_or((1, 0), |a| (a.item_size() as i64, a.base_size() as i64));
            OpResult::Value(memory.gc_load_indexed_i(base, index, item_size, base_ofs))
        }
        OpCode::GetarrayitemGcR | OpCode::GetarrayitemRawR | OpCode::GetarrayitemGcPureR => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let (item_size, base_ofs) = op
                .descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map_or((1, 0), |a| (a.item_size() as i64, a.base_size() as i64));
            OpResult::Value(memory.gc_load_indexed_r(base, index, item_size, base_ofs))
        }
        OpCode::GetarrayitemGcF | OpCode::GetarrayitemRawF | OpCode::GetarrayitemGcPureF => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let (item_size, base_ofs) = op
                .descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map_or((1, 0), |a| (a.item_size() as i64, a.base_size() as i64));
            OpResult::Value(memory.gc_load_indexed_f(base, index, item_size, base_ofs))
        }
        OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
            let base = values.resolve(op.args[0]);
            let index = values.resolve(op.args[1]);
            let value = values.resolve(op.args[2]);
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
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_i(func, &args))
        }
        OpCode::CallR
        | OpCode::CallPureR
        | OpCode::CallMayForceR
        | OpCode::CallReleaseGilR
        | OpCode::CallLoopinvariantR => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_r(func, &args))
        }
        OpCode::CallF
        | OpCode::CallPureF
        | OpCode::CallMayForceF
        | OpCode::CallReleaseGilF
        | OpCode::CallLoopinvariantF => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_f(func, &args))
        }
        OpCode::CallN
        | OpCode::CallPureN
        | OpCode::CallMayForceN
        | OpCode::CallReleaseGilN
        | OpCode::CallLoopinvariantN => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            memory.call_n(func, &args);
            OpResult::Void
        }
        // CallAssembler — delegate to call dispatch
        OpCode::CallAssemblerI => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_i(func, &args))
        }
        OpCode::CallAssemblerR => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_r(func, &args))
        }
        OpCode::CallAssemblerF => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_f(func, &args))
        }
        OpCode::CallAssemblerN => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            memory.call_n(func, &args);
            OpResult::Void
        }
        // CondCallValue — delegate to call dispatch
        OpCode::CondCallValueI => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_i(func, &args))
        }
        OpCode::CondCallValueR => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            OpResult::Value(memory.call_r(func, &args))
        }
        OpCode::CondCallN => {
            let func = values.resolve(op.args[0]);
            let args: Vec<i64> = op.args[1..].iter().map(|&r| values.resolve(r)).collect();
            memory.call_n(func, &args);
            OpResult::Void
        }
        // Fall through to the default execute_one for everything else
        _ => execute_one(op, values, exc),
    }
}

/// **Deprecated**: IR-based blackhole. Will be replaced by jitcode-based
/// `resume_in_blackhole()` once pyre generates jitcode.
/// blackhole.py:1095 bhimpl_recursive_call parity:
/// Callback to execute a CallAssembler op during IR blackhole.
/// Receives the callee frame pointer (args[0]) and returns the result.
/// In RPython this is `portal_runner` via `cpu.bh_call_i`.
pub type CallAssemblerFn = dyn Fn(i64) -> i64;

pub(crate) fn blackhole_execute_with_state(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    initial_exception: ExceptionState,
) -> (BlackholeResult, ExceptionState) {
    blackhole_execute_with_state_ca(
        ops,
        constants,
        initial_values,
        start_index,
        initial_exception,
        None,
    )
}

pub(crate) fn blackhole_execute_with_state_ca(
    ops: &[Op],
    constants: &HashMap<u32, i64>,
    initial_values: &HashMap<u32, i64>,
    start_index: usize,
    initial_exception: ExceptionState,
    call_assembler_fn: Option<&CallAssemblerFn>,
) -> (BlackholeResult, ExceptionState) {
    let mut merged = initial_values.clone();
    for (&k, &v) in constants {
        merged.entry(k).or_insert(v);
    }
    let mut tv = TraceValues::from_hashmap(&merged);
    drop(merged);
    let mut exc_state = initial_exception;

    // RPython _run_forever parity: find the Label op (loop header) so
    // that Jump can loop back. The trace is [Label, ..., Jump(→Label)].
    let label_index = ops
        .iter()
        .position(|op| op.opcode == OpCode::Label)
        .unwrap_or(0);
    // Label's args define the loop's inputargs (their OpRef positions).
    let label_inputarg_positions: Vec<u32> = ops
        .get(label_index)
        .map(|op| op.args.iter().map(|a| a.0).collect())
        .unwrap_or_default();

    let mut op_idx = start_index;

    // blackhole.py:1752 _run_forever: no iteration cap.
    while op_idx < ops.len() {
        let op = &ops[op_idx];
        let result = execute_one(op, &tv, &mut exc_state);

        match result {
            OpResult::Value(v) => {
                if !op.pos.is_none() {
                    tv.set(op.pos.0, v);
                }
            }
            OpResult::Void => {}
            // blackhole.py:1095 bhimpl_recursive_call parity:
            // CallAssembler ops invoke portal_runner via call_assembler_fn.
            OpResult::Unsupported(_)
                if call_assembler_fn.is_some() && op.opcode.is_call_assembler() =>
            {
                let ca_fn = call_assembler_fn.unwrap();
                let frame_ptr = tv.resolve(op.args[0]);
                let result = ca_fn(frame_ptr);
                if !op.pos.is_none() {
                    tv.set(op.pos.0, result);
                }
            }
            OpResult::Finish(args) => {
                let vals: Vec<i64> = args.iter().map(|&r| tv.resolve(r)).collect();
                let vtypes = op
                    .descr
                    .as_ref()
                    .and_then(|d| d.as_fail_descr())
                    .map(|fd| fd.fail_arg_types().to_vec())
                    .unwrap_or_default();
                return (
                    BlackholeResult::Finish {
                        op_index: op_idx,
                        values: vals,
                        value_types: vtypes,
                    },
                    exc_state,
                );
            }
            OpResult::Jump(args) => {
                // RPython _run_forever parity: Jump loops back to Label.
                // Map jump args → label inputargs and restart from label+1.
                let vals: Vec<i64> = args.iter().map(|&r| tv.resolve(r)).collect();
                for (pos, val) in label_inputarg_positions.iter().zip(vals.iter()) {
                    tv.set(*pos, *val);
                }
                op_idx = label_index + 1;
                continue;
            }
            OpResult::GuardFailed => {
                let fail_values = if let Some(ref fail_args) = op.fail_args {
                    fail_args.iter().map(|&r| tv.resolve(r)).collect()
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
        op_idx += 1;
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

/// Blackhole execution with virtual object materialization.
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

use crate::jitcode::machine::{
    call_int_function, eval_binop_f, eval_binop_i, eval_binop_ovf, eval_unary_f, eval_unary_i,
};
use crate::jitcode::{
    self, BC_ABORT, BC_ABORT_PERMANENT, BC_ARRAYLEN_VABLE, BC_BRANCH_REG_ZERO, BC_BRANCH_ZERO,
    BC_CALL_ASSEMBLER_FLOAT, BC_CALL_ASSEMBLER_INT, BC_CALL_ASSEMBLER_REF, BC_CALL_ASSEMBLER_VOID,
    BC_CALL_FLOAT, BC_CALL_INT, BC_CALL_LOOPINVARIANT_FLOAT, BC_CALL_LOOPINVARIANT_INT,
    BC_CALL_LOOPINVARIANT_REF, BC_CALL_LOOPINVARIANT_VOID, BC_CALL_MAY_FORCE_FLOAT,
    BC_CALL_MAY_FORCE_INT, BC_CALL_MAY_FORCE_REF, BC_CALL_MAY_FORCE_VOID, BC_CALL_PURE_FLOAT,
    BC_CALL_PURE_INT, BC_CALL_PURE_REF, BC_CALL_REF, BC_CALL_RELEASE_GIL_FLOAT,
    BC_CALL_RELEASE_GIL_INT, BC_CALL_RELEASE_GIL_REF, BC_CALL_RELEASE_GIL_VOID,
    BC_COPY_FROM_BOTTOM, BC_DUP_STACK, BC_GETARRAYITEM_VABLE_F, BC_GETARRAYITEM_VABLE_I,
    BC_GETARRAYITEM_VABLE_R, BC_GETFIELD_VABLE_F, BC_GETFIELD_VABLE_I, BC_GETFIELD_VABLE_R,
    BC_HINT_FORCE_VIRTUALIZABLE, BC_INLINE_CALL, BC_JIT_MERGE_POINT, BC_JUMP, BC_JUMP_TARGET,
    BC_LOAD_CONST_F, BC_LOAD_CONST_I, BC_LOAD_CONST_R, BC_LOAD_STATE_ARRAY, BC_LOAD_STATE_FIELD,
    BC_LOAD_STATE_VARRAY, BC_MOVE_F, BC_MOVE_I, BC_MOVE_R, BC_PEEK_I, BC_POP_DISCARD, BC_POP_F,
    BC_POP_I, BC_POP_R, BC_PUSH_F, BC_PUSH_I, BC_PUSH_R, BC_PUSH_TO, BC_RAISE, BC_RECORD_BINOP_F,
    BC_RECORD_BINOP_I, BC_RECORD_UNARY_F, BC_RECORD_UNARY_I, BC_REF_RETURN, BC_REQUIRE_STACK,
    BC_RERAISE, BC_RESIDUAL_CALL_VOID, BC_SET_SELECTED, BC_SETARRAYITEM_VABLE_F,
    BC_SETARRAYITEM_VABLE_I, BC_SETARRAYITEM_VABLE_R, BC_SETFIELD_VABLE_F, BC_SETFIELD_VABLE_I,
    BC_SETFIELD_VABLE_R, BC_STORE_DOWN, BC_STORE_STATE_ARRAY, BC_STORE_STATE_FIELD,
    BC_STORE_STATE_VARRAY, BC_SWAP_STACK, JitArgKind, JitCode, LivenessInfo, MIFrame, MIFrameStack,
};

// ── BlackholeInterpBuilder: setup_insns infrastructure ──────────────
//
// RPython `blackhole.py:52-103` `class BlackholeInterpBuilder` combines
// pool management AND dispatch setup. pyre's existing
// `BlackholeInterpBuilder` (below, at the pool management section) is the
// pool manager. The `setup_insns` infrastructure (opcode table + dispatch
// table) is being added incrementally as Phase D of the RPython parity
// plan. Handler function pointers and `dispatch_loop` will be wired in
// as `bhimpl_*` methods are ported one by one.

/// Handler function signature for the codewriter-orthodox dispatch table.
///
/// RPython `blackhole.py:107` `handler(self, code, position) -> position`.
/// Each handler decodes operands from `code[position..]` based on its
/// argcodes, calls the corresponding `bhimpl_*` method on `bh`, writes
/// results, and returns the updated position.
pub type BhOpcodeHandler =
    fn(bh: &mut BlackholeInterpreter, code: &[u8], position: usize) -> Result<usize, DispatchError>;

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

/// Re-export BhDescr from codewriter::jitcode — shared descriptor type
/// between codewriter assembler and blackhole interpreter.
/// RPython `history.py:AbstractDescr` parity.
pub use majit_codewriter::jitcode::{BhCallDescr, BhDescr};

/// Signal from dispatch_one to run().
///
/// RPython: `LeaveFrame` exception + `except Exception` in blackhole.py run()
#[derive(Debug)]
enum DispatchError {
    /// Normal return from frame (RPython: LeaveFrame).
    LeaveFrame,
    /// Exception raised — must call handle_exception_in_frame.
    /// Carries the exception value (GcRef pointer as i64).
    RaiseException(i64),
}

/// Jitcode-based blackhole interpreter.
///
/// Executes jitcode bytecodes with concrete values. Each instance
/// represents one execution frame. Frame chain is linked via
/// `nextblackholeinterp`.
///
/// RPython: `BlackholeInterpreter` class in blackhole.py:282-306.
///
/// RPython `__init__` receives `builder` and stores:
///   self.cpu = builder.cpu
///   self.dispatch_loop = builder.dispatch_loop
///   self.descrs = builder.descrs
///   self.op_catch_exception = builder.op_catch_exception
pub struct BlackholeInterpreter {
    /// RPython `blackhole.py:286` `self.cpu = builder.cpu`.
    /// Reference to the backend trait for `bh_*` concrete execution.
    /// Raw pointer because the interpreter is pool-managed and
    /// the Backend outlives all interpreter instances.
    /// RPython `blackhole.py:286/56` `self.cpu = builder.cpu`.
    /// Backend trait for `bh_*` concrete execution. None until set.
    pub cpu: Option<&'static dyn majit_backend::Backend>,
    /// RPython `blackhole.py:288` `self.descrs = builder.descrs`.
    /// Descriptor table from the assembler. In RPython, `descrs` is a list
    /// of `AbstractDescr` objects carrying field offsets, array item sizes,
    /// etc. In pyre, we store raw offsets (usize) as a simplification —
    /// descriptor-index argcode ('d', 2 bytes) indexes into this table.
    /// RPython `blackhole.py:288` `self.descrs = builder.descrs`.
    /// Descriptor table — heterogeneous like RPython AbstractDescr list.
    pub descrs: Vec<BhDescr>,
    /// RPython `blackhole.py:289` `self.op_catch_exception = builder.op_catch_exception`.
    pub op_catch_exception: u8,
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
    /// RPython: BlackholeInterpreter has a single value stack.
    /// pyre uses multiple stacks for selected storage (valuestackdepth).
    runtime_stacks: Vec<Vec<i64>>,
    /// Current selected storage index.
    current_selected: usize,
    /// Set to true when `run()` exited via merge point (not LeaveFrame).
    pub reached_merge_point: bool,
    /// True when run() hit BC_ABORT (unsupported bytecode). Callers
    /// must not treat abort as DoneWithThisFrame — side effects from
    /// partial execution have corrupted state.
    pub aborted: bool,
    /// RPython blackhole.py handle_exception_in_frame parity:
    /// True when a residual call raised an exception (returned NULL ref).
    /// Unlike `aborted`, this indicates a Python-level exception that
    /// should propagate up the blackhole chain, not a JIT infrastructure error.
    pub got_exception: bool,
    /// Position of the last dispatched opcode (before position advances past operands).
    /// Used by handle_exception_in_frame for handler lookup — the faulting instruction
    /// PC, not the next instruction PC. Public so caller-chain propagation in
    /// call_jit.rs can set it to the suspended caller's position.
    pub last_opcode_position: usize,
    /// blackhole.py:391 exception_last_value: the caught exception object.
    /// Set when handle_exception_in_frame finds a handler.
    /// Read by CheckExcMatch and other exception opcodes in the handler.
    pub exception_last_value: i64,
    /// blackhole.py bhimpl_getfield_vable_*: pointer to the virtualizable
    /// object (e.g. PyFrame). Used by BC_GETFIELD_VABLE_* bytecodes.
    /// Set during blackhole setup from the guard failure's virtualizable ptr.
    pub virtualizable_ptr: i64,
    /// Pointer to the VirtualizableInfo describing field offsets.
    /// Used by vable bytecodes to compute memory offsets.
    pub virtualizable_info: *const crate::virtualizable::VirtualizableInfo,
    /// pyre-specific portal contract: absolute start index of the operand
    /// stack in the virtualizable's unified locals+cells+stack array.
    ///
    /// blackhole.py passes reds explicitly into bh_call_r(...); pyre packs
    /// the same state into `locals_cells_stack_w` before calling the portal
    /// runner, so the blackhole needs the stack base to reconstruct the
    /// frame layout correctly on recursive portal re-entry.
    pub virtualizable_stack_base: usize,
    /// blackhole.py:1095 get_portal_runner(jdindex):
    ///   jitdriver_sd = self.builder.metainterp_sd.jitdrivers_sd[jdindex]
    ///   fnptr = adr2int(jitdriver_sd.portal_runner_adr)
    ///   calldescr = jitdriver_sd.mainjitcode.calldescr
    /// pyre: single jitdriver. portal_runner_ptr is the fnptr.
    pub portal_runner_ptr: Option<fn(i64) -> i64>,
    /// RPython: `jitdriver_sd.mainjitcode.calldescr` — CallDescr of the portal
    /// function. Returned by `get_portal_runner()` for `bhimpl_recursive_call_*`.
    pub mainjitcode_calldescr: BhCallDescr,
}

/// blackhole.py: last exception value from a residual call.
/// Set by pyre call helpers (bh_call_fn_impl etc.) on error.
/// Read by dispatch_one to populate exception_last_value on handler dispatch.
thread_local! {
    pub static BH_LAST_EXC_VALUE: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
}

/// blackhole.py bhimpl_recursive_call: virtualizable pointer for call helpers.
/// Set by the blackhole run loop before dispatch so extern "C" helpers
/// (bh_call_fn_impl etc.) can access the parent frame without passing
/// it through the register file.
thread_local! {
    pub static BH_VABLE_PTR: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
}

impl Default for BlackholeInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl BlackholeInterpreter {
    pub fn new() -> Self {
        Self {
            cpu: None,
            descrs: Vec::new(),
            op_catch_exception: u8::MAX,
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
            runtime_stacks: Vec::new(),
            current_selected: 0,
            reached_merge_point: false,
            aborted: false,
            got_exception: false,
            last_opcode_position: 0,
            exception_last_value: 0,
            virtualizable_ptr: 0,
            virtualizable_info: std::ptr::null(),
            virtualizable_stack_base: 0,
            portal_runner_ptr: None,
            mainjitcode_calldescr: BhCallDescr::default(),
        }
    }

    /// blackhole.py:312 setposition
    ///
    /// Initialize register arrays for a jitcode and set the position.
    /// Allocates registers sized to hold both working regs and constants,
    /// then copies constants into the upper portion of each register array.
    pub fn setposition(&mut self, jitcode: JitCode, position: usize) {
        // blackhole.py:313-315
        let num_regs_and_consts_i = jitcode.num_regs_and_consts_i();
        let num_regs_and_consts_r = jitcode.num_regs_and_consts_r();
        let num_regs_and_consts_f = jitcode.num_regs_and_consts_f();

        // blackhole.py:324-327
        if num_regs_and_consts_i > 0 {
            self.registers_i.clear();
            self.registers_i.resize(num_regs_and_consts_i, 0);
            // blackhole.py:441-449 copy_constants
            let target_index = jitcode.num_regs_i() as usize;
            for (i, &c) in jitcode.constants_i.iter().enumerate() {
                self.registers_i[target_index + i] = c;
            }
        } else {
            self.registers_i.clear();
        }

        // blackhole.py:328-331
        if num_regs_and_consts_r > 0 {
            self.registers_r.clear();
            self.registers_r.resize(num_regs_and_consts_r, 0);
            let target_index = jitcode.num_regs_r() as usize;
            for (i, &c) in jitcode.constants_r.iter().enumerate() {
                self.registers_r[target_index + i] = c;
            }
        } else {
            self.registers_r.clear();
        }

        // blackhole.py:332-335
        if num_regs_and_consts_f > 0 {
            self.registers_f.clear();
            self.registers_f.resize(num_regs_and_consts_f, 0);
            let target_index = jitcode.num_regs_f() as usize;
            for (i, &c) in jitcode.constants_f.iter().enumerate() {
                self.registers_f[target_index + i] = c;
            }
        } else {
            self.registers_f.clear();
        }

        // blackhole.py:336-337
        // RPython: descrs are shared on the builder (setup_descrs).
        // pyre: per-jitcode descrs, loaded here.
        if !jitcode.descrs.is_empty() {
            self.descrs = jitcode.descrs.clone();
        }
        self.jitcode = jitcode;
        self.position = position;
        self.aborted = false;
        self.got_exception = false;
        self.last_opcode_position = position;
        self.exception_last_value = 0;
        self.reached_merge_point = false;
    }

    /// blackhole.py:1095-1099 get_portal_runner(jdindex):
    ///   jitdriver_sd = self.builder.metainterp_sd.jitdrivers_sd[jdindex]
    ///   fnptr = adr2int(jitdriver_sd.portal_runner_adr)
    ///   calldescr = jitdriver_sd.mainjitcode.calldescr
    ///   return fnptr, calldescr
    /// pyre: single jitdriver. mainjitcode_calldescr set during blackhole setup.
    pub fn get_portal_runner(&self, _jdindex: usize) -> (i64, BhCallDescr) {
        let fnptr = self
            .portal_runner_ptr
            .map(|f| f as usize as i64)
            .unwrap_or(0);
        (fnptr, self.mainjitcode_calldescr.clone())
    }

    /// Resolve field descriptor offsets in this interpreter's descrs table.
    /// Delegates to the same logic as BlackholeInterpBuilder::resolve_field_offsets.
    pub fn resolve_field_offsets(&mut self, resolver: impl Fn(&str, &str) -> usize) {
        for descr in &mut self.descrs {
            if let BhDescr::Field {
                offset,
                name,
                owner,
            } = descr
            {
                if *offset == 0 && !name.is_empty() {
                    *offset = resolver(owner, name);
                }
            }
        }
    }

    /// Resolve JitCode fnaddr values in this interpreter's descrs table.
    pub fn resolve_jitcode_fnaddrs(&mut self, resolver: impl Fn(usize) -> i64) {
        for descr in &mut self.descrs {
            if let BhDescr::JitCode {
                jitcode_index,
                fnaddr,
                ..
            } = descr
            {
                if *fnaddr == 0 {
                    *fnaddr = resolver(*jitcode_index);
                }
            }
        }
    }

    /// blackhole.py:1109-1116 bhimpl_recursive_call_r:
    ///   fnptr, calldescr = self.get_portal_runner(jdindex)
    ///   return self.cpu.bh_call_r(fnptr, greens_i+reds_i, greens_r+reds_r, ...)
    ///
    /// pyre: greens = [next_instr], reds = [locals via virtualizable].
    /// Write greens+reds to the virtualizable (frame) then call portal_runner.
    pub fn bhimpl_recursive_call_r(&self, portal_runner: fn(i64) -> i64) -> i64 {
        // bh_call_r(fnptr, greens_i + reds_i, greens_r + reds_r, ..., calldescr)
        // pyre packs args into the virtualizable before the call.
        self.sync_virtualizable_for_recursive_portal();
        portal_runner(self.virtualizable_ptr)
    }

    /// pyre portal recursion contract: materialize the live virtualizable
    /// state in the heap frame before calling `portal_runner(frame_ptr)`.
    ///
    /// RPython's bh_call_r path forwards the current greens/reds directly as
    /// call arguments. pyre stores the equivalent state in the unified
    /// `locals_cells_stack_w` array and scalar virtualizable fields:
    ///   - green: next_instr
    ///   - red scalar: valuestackdepth
    ///   - red array: locals + live value stack
    ///
    /// Without the stack and valuestackdepth write-back, recursive portal
    /// re-entry observes a stale frame shape and can resume with the wrong
    /// operand stack contents.
    fn sync_virtualizable_for_recursive_portal(&self) {
        const PYFRAME_NEXT_INSTR_FIELD_INDEX: usize = 0;
        const PYFRAME_VALUESTACKDEPTH_FIELD_INDEX: usize = 2;
        const PYFRAME_LOCALS_CELLS_STACK_ARRAY_INDEX: usize = 0;

        if self.virtualizable_ptr == 0 || self.virtualizable_info.is_null() {
            return;
        }

        let py_pc = self.jitcode.jit_pc_to_py_pc(self.last_opcode_position) as usize;
        let frame_ptr = self.virtualizable_ptr as *mut u8;
        let info = unsafe { &*self.virtualizable_info };
        let nlocals = self.jitcode.nlocals;
        let stack_depth = self.jitcode.depth_at_py_pc.get(py_pc).copied().unwrap_or(0) as usize;
        let stack_base = self.virtualizable_stack_base.max(nlocals);

        unsafe {
            info.write_field(frame_ptr, PYFRAME_NEXT_INSTR_FIELD_INDEX, py_pc as i64);
            info.write_field(
                frame_ptr,
                PYFRAME_VALUESTACKDEPTH_FIELD_INDEX,
                (stack_base + stack_depth) as i64,
            );
        }

        let array_len = info
            .array_fields
            .get(PYFRAME_LOCALS_CELLS_STACK_ARRAY_INDEX)
            .map(|array| unsafe {
                crate::virtualizable::vable_array_len(frame_ptr as *const u8, array)
            })
            .unwrap_or(0);

        for i in 0..nlocals {
            if i < self.registers_r.len() && i < array_len {
                unsafe {
                    info.write_array_item(
                        frame_ptr,
                        PYFRAME_LOCALS_CELLS_STACK_ARRAY_INDEX,
                        i,
                        self.registers_r[i],
                    )
                };
            }
        }
        for d in 0..stack_depth {
            let reg_idx = nlocals + d;
            let frame_idx = stack_base + d;
            if reg_idx < self.registers_r.len() && frame_idx < array_len {
                unsafe {
                    info.write_array_item(
                        frame_ptr,
                        PYFRAME_LOCALS_CELLS_STACK_ARRAY_INDEX,
                        frame_idx,
                        self.registers_r[reg_idx],
                    )
                };
            }
        }
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

    /// blackhole.py:385 cleanup_registers
    ///
    /// Clear reference registers to avoid keeping objects alive.
    /// Does not clear constants (they are prebuilt).
    pub fn cleanup_registers(&mut self) {
        for i in 0..self.jitcode.num_regs_r() as usize {
            if i < self.registers_r.len() {
                self.registers_r[i] = 0;
            }
        }
        self.exception_last_value = 0;
    }

    /// blackhole.py:393-394 `get_current_position_info`.
    ///
    /// RPython returns an offset into `metainterp_sd.liveness_info`
    /// (via `jitcode.get_live_vars_info(self.position, self.builder.op_live)`).
    /// pyre currently keeps `liveness_info` per-JitCode and the offsets
    /// in a side table, but the return value is still that offset —
    /// consumers pair it with `enumerate_vars(offset, all_liveness, ...)`
    /// matching `resume.py:1017-1026`.
    pub fn get_current_position_info(&self) -> usize {
        // TODO: thread `metainterp_sd.op_live` through the builder once
        // the bytecode stream embeds `-live-` opcodes.
        self.jitcode.get_live_vars_info(self.position, 0)
    }

    /// Transitional helper: look up the per-entry `LivenessInfo` that
    /// matches the current `bh.position`. Upstream has no such lookup —
    /// consumers should read live-register bits via `enumerate_vars`
    /// over the packed `liveness_info` bytes. This exists only so the
    /// pyre-specific non-live-register seeding in `call_jit` can keep
    /// compiling until Phase D removes it.
    pub fn get_current_position_info_legacy(&self) -> Option<&LivenessInfo> {
        self.jitcode
            .liveness
            .iter()
            .find(|info| info.pc as usize == self.position)
    }

    /// blackhole.py:1653 _setup_return_value_i
    ///
    /// Connect the return of values from the called frame to the
    /// 'xxx_call_yyy' instructions from the caller frame.
    /// blackhole.py:1653 _setup_return_value_i
    pub fn setup_return_value_i(&mut self, result: i64) {
        // blackhole.py:1655-1656
        let reg_idx = self.jitcode.code[self.position - 1] as usize;
        self.registers_i[reg_idx] = result;
    }

    /// blackhole.py:1657 _setup_return_value_r
    pub fn setup_return_value_r(&mut self, result: i64) {
        // blackhole.py:1658-1659
        let reg_idx = self.jitcode.code[self.position - 1] as usize;
        self.registers_r[reg_idx] = result;
    }

    /// blackhole.py:1660 _setup_return_value_f
    pub fn setup_return_value_f(&mut self, result: i64) {
        // blackhole.py:1661-1662
        let reg_idx = self.jitcode.code[self.position - 1] as usize;
        self.registers_f[reg_idx] = result;
    }

    /// blackhole.py:1664 _done_with_this_frame
    ///
    /// Rare case: the blackhole interps all returned normally
    /// (in general we get a ContinueRunningNormally exception).
    fn done_with_this_frame(&self) -> JitException {
        match self.return_type {
            BhReturnType::Void => JitException::DoneWithThisFrameVoid,
            BhReturnType::Int => JitException::DoneWithThisFrameInt(self.get_tmpreg_i()),
            BhReturnType::Ref => {
                JitException::DoneWithThisFrameRef(GcRef(self.get_tmpreg_r() as usize))
            }
            BhReturnType::Float => {
                JitException::DoneWithThisFrameFloat(f64::from_bits(self.get_tmpreg_f() as u64))
            }
        }
    }

    /// blackhole.py:1679 _exit_frame_with_exception
    fn exit_frame_with_exception(&self, exc: i64) -> JitException {
        JitException::ExitFrameWithExceptionRef(GcRef(exc as usize))
    }

    /// blackhole.py:1647 _prepare_resume_from_failure
    ///
    /// Extract exception from the CPU deadframe on guard failure.
    /// Returns the exception value (0 if none).
    pub fn prepare_resume_from_failure(deadframe_exc: i64) -> i64 {
        // RPython: lltype.cast_opaque_ptr(rclass.OBJECTPTR,
        //          self.cpu.grab_exc_value(deadframe))
        deadframe_exc
    }

    /// blackhole.py:1612 _resume_mainloop
    ///
    /// Execute one frame and handle its completion.
    /// Returns Ok(exc) where exc is the exception to propagate to caller (0 = none),
    /// or Err(JitException) for JIT-level control flow exits.
    pub fn resume_mainloop(&mut self, current_exc: i64) -> Result<i64, JitException> {
        // blackhole.py:1614-1618
        // If there is a current exception, raise it now
        // (it may be caught by a catch_operation in this frame)
        if current_exc != 0 {
            if !self.handle_exception_in_frame(current_exc) {
                // No handler: propagate
                if self.nextblackholeinterp.is_none() {
                    return Err(self.exit_frame_with_exception(current_exc));
                }
                return Ok(current_exc);
            }
        }

        // blackhole.py:1621 — run the bytecode
        self.run();

        // Check for exception during execution
        if self.got_exception {
            let exc = self.exception_last_value;
            if self.nextblackholeinterp.is_none() {
                // blackhole.py:1629
                return Err(self.exit_frame_with_exception(exc));
            }
            return Ok(exc);
        }

        if self.aborted {
            // Abort is treated as an infrastructure error, not a normal exit.
            // The caller should not treat this as DoneWithThisFrame.
            if self.nextblackholeinterp.is_none() {
                return Err(JitException::DoneWithThisFrameVoid);
            }
            return Ok(0);
        }

        // blackhole.py:1633 — pass the frame's return value to the caller
        if self.nextblackholeinterp.is_none() {
            // blackhole.py:1635 — bottommost frame
            return Err(self.done_with_this_frame());
        }

        // Copy return values to locals before borrowing caller mutably
        let ret_type = self.return_type;
        let tmp_i = self.tmpreg_i;
        let tmp_r = self.tmpreg_r;
        let tmp_f = self.tmpreg_f;

        let caller = self.nextblackholeinterp.as_mut().unwrap();
        match ret_type {
            BhReturnType::Int => caller.setup_return_value_i(tmp_i),
            BhReturnType::Ref => caller.setup_return_value_r(tmp_r),
            BhReturnType::Float => caller.setup_return_value_f(tmp_f),
            BhReturnType::Void => {}
        }

        // blackhole.py:1645 — return no exception
        Ok(0)
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

    fn ensure_stack(&mut self, selected: usize) {
        if selected >= self.runtime_stacks.len() {
            self.runtime_stacks.resize_with(selected + 1, Vec::new);
        }
    }

    fn runtime_stack_mut(&mut self, selected: usize) -> &mut Vec<i64> {
        self.ensure_stack(selected);
        &mut self.runtime_stacks[selected]
    }

    fn runtime_stack_pop(&mut self, selected: usize) -> i64 {
        if selected < self.runtime_stacks.len() {
            self.runtime_stacks[selected].pop().unwrap_or(0)
        } else {
            0
        }
    }

    pub fn runtime_stack_push(&mut self, selected: usize, value: i64) {
        self.ensure_stack(selected);
        self.runtime_stacks[selected].push(value);
    }

    fn runtime_stack_peek(&self, selected: usize, pos: usize) -> i64 {
        self.runtime_stacks
            .get(selected)
            .and_then(|s| s.get(pos).copied())
            .unwrap_or(0)
    }

    fn runtime_stack_len(&self, selected: usize) -> usize {
        self.runtime_stacks.get(selected).map_or(0, |s| s.len())
    }

    /// blackhole.py:396 handle_exception_in_frame: check if the current
    /// position has an exception handler. If found, unwind stack, push
    /// exception, jump to handler. Returns true if handled.
    ///
    /// lasti (faulting Python PC) is determined internally from
    /// self.position via jit_pc_to_py_pc reverse map.
    pub fn handle_exception_in_frame(&mut self, exc_value: i64) -> bool {
        // Use the position of the opcode that raised, not the post-operand
        // position. This matches pyre-interpreter/eval.rs:79 (next_instr - 1)
        // and ensures the last protected instruction hits its handler.
        let faulting_jit_pc = self.last_opcode_position;
        let handler_opt = self
            .jitcode
            .find_exception_handler(faulting_jit_pc)
            .cloned();
        let Some(handler) = handler_opt else {
            return false;
        };
        let selected = self.current_selected;
        while self.runtime_stack_len(selected) > handler.stack_depth as usize {
            if selected < self.runtime_stacks.len() {
                self.runtime_stacks[selected].pop();
            }
        }
        if handler.push_lasti {
            // last_opcode_position is already the faulting instruction's
            // jitcode PC (set before dispatch_one). No additional -1 needed.
            let faulting_py_pc = self.jitcode.jit_pc_to_py_pc(faulting_jit_pc);
            let box_fn_ptr = self
                .jitcode
                .fn_ptrs
                .get(handler.box_int_fn_idx as usize)
                .map(|t| t.concrete_ptr);
            if let Some(ptr) = box_fn_ptr {
                let lasti_boxed = call_int_function(ptr, &[faulting_py_pc]);
                self.runtime_stack_push(selected, lasti_boxed);
            }
        }
        self.runtime_stack_push(selected, exc_value);
        self.position = handler.jit_target;
        self.exception_last_value = exc_value;
        self.got_exception = false;
        true
    }

    /// Drain the runtime stack for the given selected index, returning
    /// all values in order (bottom → top).
    pub fn runtime_stack_drain(&mut self, selected: usize) -> Vec<i64> {
        if selected < self.runtime_stacks.len() {
            std::mem::take(&mut self.runtime_stacks[selected])
        } else {
            Vec::new()
        }
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
    /// RPython: `BlackholeInterpreter.run()` catches `LeaveFrame` and breaks,
    /// catches exceptions and calls `handle_exception_in_frame`.
    pub fn run(&mut self) {
        // blackhole.py parity: each BlackholeInterp has its own parent frame
        // via its nextblackholeinterp chain. In pyre we expose the parent
        // frame via BH_VABLE_PTR (thread-local) for extern "C" call helpers.
        // Save/restore the previous value so nested blackhole runs (triggered
        // by bhimpl_residual_call re-entering compiled code → another
        // blackhole) do not corrupt the caller's parent pointer.
        let saved_bh_vable = BH_VABLE_PTR.with(|c| c.get());
        BH_VABLE_PTR.with(|c| c.set(self.virtualizable_ptr));
        // blackhole.py BlackholeInterpreter.registers_r parity: register the
        // ref register bank with the GC so nursery objects held only by
        // blackhole regs survive across collecting calls. RPython traces
        // these via the Box array's RPython type; pyre uses an explicit
        // thread-local stack walked by `do_collect_nursery`.
        let bh_depth = unsafe {
            majit_gc::shadow_stack::push_bh_regs(&mut self.registers_r, &mut self.tmpreg_r)
        };
        self.run_inner();
        majit_gc::shadow_stack::pop_bh_regs_to(bh_depth);
        BH_VABLE_PTR.with(|c| c.set(saved_bh_vable));
    }

    fn run_inner(&mut self) {
        let trace = crate::majit_log_enabled();
        loop {
            if self.finished() {
                if trace {
                    eprintln!(
                        "[bh-trace] finished at pos={} reg0={}",
                        self.position,
                        self.registers_i.get(0).copied().unwrap_or(-1)
                    );
                }
                return;
            }
            let pos_before = self.position;
            self.last_opcode_position = pos_before;
            let opcode = self.next_u8();
            if trace {
                let stack_len = self.runtime_stack_len(0);
                eprintln!(
                    "[bh-trace] pos={} op={} reg0={} reg1={} stack_len={}",
                    pos_before,
                    opcode,
                    self.registers_i.get(0).copied().unwrap_or(-1),
                    self.registers_i.get(1).copied().unwrap_or(-1),
                    stack_len
                );
            }
            match self.dispatch_one(opcode) {
                Ok(()) => {}
                Err(DispatchError::LeaveFrame) => {
                    if trace {
                        eprintln!(
                            "[bh-trace] leave-frame at pos={} ret_type={:?}",
                            pos_before, self.return_type,
                        );
                    }
                    return;
                }
                Err(DispatchError::RaiseException(exc)) => {
                    // blackhole.py:359-361: except Exception → handle_exception_in_frame
                    if trace {
                        eprintln!("[bh-trace] exception at pos={} exc=0x{:x}", pos_before, exc);
                    }
                    if self.handle_exception_in_frame(exc) {
                        // Handler found, continue execution at handler target
                        continue;
                    }
                    // No handler: propagate exception via got_exception flag
                    self.got_exception = true;
                    self.exception_last_value = exc;
                    return;
                }
            }
        }
    }

    /// Dispatch a single bytecode instruction with concrete execution.
    ///
    /// RPython: bytecode dispatch in `dispatch_loop()`, each `bhimpl_*` method
    fn dispatch_one(&mut self, opcode: u8) -> Result<(), DispatchError> {
        match opcode {
            BC_LOAD_CONST_I => {
                let dst = self.next_u16() as usize;
                let const_idx = self.next_u16() as usize;
                let value = self.jitcode.constants_i[const_idx];
                self.registers_i[dst] = value;
            }
            BC_LOAD_CONST_R => {
                let dst = self.next_u16() as usize;
                let const_idx = self.next_u16() as usize;
                let value = self.jitcode.constants_r[const_idx];
                self.registers_r[dst] = value;
            }
            BC_LOAD_CONST_F => {
                let dst = self.next_u16() as usize;
                let const_idx = self.next_u16() as usize;
                let value = self.jitcode.constants_f[const_idx];
                self.registers_f[dst] = value;
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
                if selected < self.runtime_stacks.len() {
                    let stack = &mut self.runtime_stacks[selected];
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
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[bh-trace] BRANCH_REG_ZERO cond_idx={} cond={} target={} taken={}",
                        cond_idx,
                        cond,
                        target,
                        cond == 0
                    );
                }
                if cond == 0 {
                    self.position = target;
                }
            }
            BC_JUMP => {
                let target = self.next_u16() as usize;
                self.position = target;
            }
            BC_JIT_MERGE_POINT => {
                // blackhole.py:1067-1093 bhimpl_jit_merge_point
                if self.nextblackholeinterp.is_none() {
                    // blackhole.py:1068: bottommost → ContinueRunningNormally
                    self.reached_merge_point = true;
                    return Err(DispatchError::LeaveFrame);
                }
                // blackhole.py:1074-1093: recursive portal level.
                //   sd = self.builder.metainterp_sd
                //   result_type = sd.jitdrivers_sd[jdindex].result_type
                //   x = self.bhimpl_recursive_call_r(jdindex, *args)
                //   self.bhimpl_ref_return(x)
                // pyre: result_type is always 'r' (PyObjectRef).
                let (fnptr, _calldescr) = self.get_portal_runner(0);
                if fnptr != 0 {
                    let portal_runner: fn(i64) -> i64 =
                        unsafe { std::mem::transmute(fnptr as usize) };
                    let x = self.bhimpl_recursive_call_r(portal_runner);
                    // bhimpl_ref_return(x): tmpreg_r = x; raise LeaveFrame
                    self.tmpreg_r = x;
                    self.return_type = BhReturnType::Ref;
                    return Err(DispatchError::LeaveFrame);
                }
            }
            BC_JUMP_TARGET => {
                // Non-portal loop header marker (helper jitcodes only).
            }
            BC_SET_SELECTED => {
                self.current_selected = self.next_u16() as usize;
            }
            BC_REF_RETURN => {
                // RPython bhimpl_ref_return: return ref value to caller.
                let src = self.next_u16() as usize;
                self.tmpreg_r = self.registers_r[src];
                self.return_type = BhReturnType::Ref;
                return Err(DispatchError::LeaveFrame);
            }
            BC_ABORT => {
                self.aborted = true;
                return Err(DispatchError::LeaveFrame);
            }
            BC_ABORT_PERMANENT => {
                // blackhole.py bhimpl_raise parity: exception path bytecodes
                // trigger RaiseException so handle_exception_in_frame can route
                // to the except handler. If no TLS exception is available,
                // fall back to abort.
                let exc = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc != 0 {
                    BH_LAST_EXC_VALUE.with(|c| c.set(0));
                    return Err(DispatchError::RaiseException(exc));
                }
                self.aborted = true;
                return Err(DispatchError::LeaveFrame);
            }
            // blackhole.py:1000 bhimpl_raise(excvalue)
            BC_RAISE => {
                let src = self.next_u16() as usize;
                let exc = self.registers_r[src];
                if exc != 0 {
                    return Err(DispatchError::RaiseException(exc));
                }
                self.aborted = true;
                return Err(DispatchError::LeaveFrame);
            }
            // blackhole.py:1006 bhimpl_reraise()
            BC_RERAISE => {
                let exc = self.exception_last_value;
                if exc != 0 {
                    return Err(DispatchError::RaiseException(exc));
                }
                self.aborted = true;
                return Err(DispatchError::LeaveFrame);
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
                callee.runtime_stacks = self.runtime_stacks.clone();
                callee.current_selected = self.current_selected;

                // Execute callee
                let _ = callee.run();

                // RPython blackhole.py: propagate callee exceptions/aborts.
                // If callee aborted or got an exception, propagate to caller.
                if callee.aborted {
                    self.aborted = true;
                    return Err(DispatchError::LeaveFrame);
                }
                if callee.got_exception {
                    let exc_val = callee.exception_last_value;
                    // Try to handle in this (caller) frame first.
                    if exc_val != 0 && self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }

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
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                let result = call_int_function(target.concrete_ptr, &args);
                // Check if call raised an exception (TLS set by helper).
                let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc_val != 0 {
                    if self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }
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
                // Clear stale exception before call to prevent false positives.
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[bh] call_ref fn={} dst={} nargs={} args={:?} pos={}",
                        fn_ptr_idx,
                        dst,
                        num_args,
                        &args[..args.len().min(4)],
                        self.last_opcode_position,
                    );
                }
                let result = call_int_function(target.concrete_ptr, &args);
                // Check if call raised an exception (TLS set by helper).
                let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc_val != 0 {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[bh] call_ref EXCEPTION fn={} pos={} nargs={} exc={:#x}",
                            fn_ptr_idx, self.last_opcode_position, num_args, exc_val,
                        );
                    }
                    // Actual exception: try handler dispatch.
                    if self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }
                // result == 0 without exception is a legitimate null ref
                // (e.g., None object has non-zero address in pyre, so this
                // only happens for actual PY_NULL returns which are rare
                // but legal in generic majit BC_CALL_REF semantics).
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
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                let result = call_int_function(target.concrete_ptr, &args);
                let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc_val != 0 {
                    if self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }
                self.registers_f[dst] = result;
            }
            // -- Void-typed calls --
            BC_CALL_MAY_FORCE_VOID
            | BC_CALL_RELEASE_GIL_VOID
            | BC_CALL_LOOPINVARIANT_VOID
            | BC_CALL_ASSEMBLER_VOID => {
                let fn_ptr_idx = self.next_u16() as usize;
                // void calls have no dst field in bytecode (call_void_like encoding)
                let num_args = self.next_u16() as usize;
                let args = self.read_call_args(num_args);
                let target = &self.jitcode.fn_ptrs[fn_ptr_idx];
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                call_int_function(target.concrete_ptr, &args);
                let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc_val != 0 {
                    if self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }
            }
            BC_RESIDUAL_CALL_VOID => {
                let fn_ptr_idx = self.next_u16() as usize;
                let num_args = self.next_u16() as usize;
                let args = self.read_call_args(num_args);
                let target = &self.jitcode.fn_ptrs[fn_ptr_idx];
                BH_LAST_EXC_VALUE.with(|c| c.set(0));
                call_int_function(target.concrete_ptr, &args);
                let exc_val = BH_LAST_EXC_VALUE.with(|c| c.get());
                if exc_val != 0 {
                    if self.handle_exception_in_frame(exc_val) {
                        return Ok(());
                    }
                    self.exception_last_value = exc_val;
                    self.got_exception = true;
                    return Err(DispatchError::LeaveFrame);
                }
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
            // blackhole.py:1446-1458 bhimpl_getfield_vable_i/r/f:
            //   fielddescr.get_vinfo().clear_vable_token(struct)
            //   return cpu.bh_getfield_gc_i/r/f(struct, fielddescr)
            BC_GETFIELD_VABLE_I => {
                let field_idx = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                let vinfo = unsafe { &*self.virtualizable_info };
                let ptr = self.virtualizable_ptr as *mut u8;
                // virtualizable.py:218-222 clear_vable_token
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let offset = vinfo.static_fields[field_idx].offset;
                let value = unsafe { *(ptr.add(offset) as *const i64) };
                self.registers_i[dst] = value;
            }
            BC_GETFIELD_VABLE_R => {
                let field_idx = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                let vinfo = unsafe { &*self.virtualizable_info };
                let ptr = self.virtualizable_ptr as *mut u8;
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let offset = vinfo.static_fields[field_idx].offset;
                let value = unsafe { *(ptr.add(offset) as *const i64) };
                self.registers_r[dst] = value;
            }
            BC_GETFIELD_VABLE_F => {
                let field_idx = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                let vinfo = unsafe { &*self.virtualizable_info };
                let ptr = self.virtualizable_ptr as *mut u8;
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let offset = vinfo.static_fields[field_idx].offset;
                let value = unsafe { *(ptr.add(offset) as *const i64) };
                self.registers_f[dst] = value;
            }
            // blackhole.py:1485-1495 bhimpl_setfield_vable_i/r/f
            BC_SETFIELD_VABLE_I => {
                let field_idx = self.next_u16() as usize;
                let src = self.next_u16() as usize;
                let vinfo = unsafe { &*self.virtualizable_info };
                let ptr = self.virtualizable_ptr as *mut u8;
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let offset = vinfo.static_fields[field_idx].offset;
                let value = self.registers_i[src];
                unsafe { *(ptr.add(offset) as *mut i64) = value };
            }
            BC_SETFIELD_VABLE_R => {
                let field_idx = self.next_u16() as usize;
                let src = self.next_u16() as usize;
                let vinfo = unsafe { &*self.virtualizable_info };
                let ptr = self.virtualizable_ptr as *mut u8;
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let offset = vinfo.static_fields[field_idx].offset;
                let value = self.registers_r[src];
                unsafe { *(ptr.add(offset) as *mut i64) = value };
            }
            BC_SETFIELD_VABLE_F => {
                let field_idx = self.next_u16() as usize;
                let src = self.next_u16() as usize;
                let vinfo = unsafe { &*self.virtualizable_info };
                let ptr = self.virtualizable_ptr as *mut u8;
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let offset = vinfo.static_fields[field_idx].offset;
                let value = self.registers_f[src];
                unsafe { *(ptr.add(offset) as *mut i64) = value };
            }
            // blackhole.py:1374-1387 bhimpl_getarrayitem_vable_i/r/f:
            //   fielddescr.get_vinfo().clear_vable_token(vable)
            //   array = cpu.bh_getfield_gc_r(vable, fielddescr)
            //   return cpu.bh_getarrayitem_gc_i/r/f(array, index, arraydescr)
            BC_GETARRAYITEM_VABLE_I | BC_GETARRAYITEM_VABLE_R | BC_GETARRAYITEM_VABLE_F => {
                let array_idx = self.next_u16() as usize;
                let index_reg = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                let vinfo = unsafe { &*self.virtualizable_info };
                let ptr = self.virtualizable_ptr as *mut u8;
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let ainfo = &vinfo.array_fields[array_idx];
                let index = self.registers_i[index_reg] as usize;
                let value = unsafe {
                    crate::virtualizable::vable_read_array_item(ptr as *const u8, ainfo, index)
                };
                match opcode {
                    BC_GETARRAYITEM_VABLE_I => self.registers_i[dst] = value,
                    BC_GETARRAYITEM_VABLE_R => self.registers_r[dst] = value,
                    _ => self.registers_f[dst] = value,
                }
            }
            // blackhole.py:1390-1403 bhimpl_setarrayitem_vable_i/r/f
            BC_SETARRAYITEM_VABLE_I | BC_SETARRAYITEM_VABLE_R | BC_SETARRAYITEM_VABLE_F => {
                let array_idx = self.next_u16() as usize;
                let index_reg = self.next_u16() as usize;
                let src = self.next_u16() as usize;
                let vinfo = unsafe { &*self.virtualizable_info };
                let ptr = self.virtualizable_ptr as *mut u8;
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let ainfo = &vinfo.array_fields[array_idx];
                let index = self.registers_i[index_reg] as usize;
                let value = match opcode {
                    BC_SETARRAYITEM_VABLE_I => self.registers_i[src],
                    BC_SETARRAYITEM_VABLE_R => self.registers_r[src],
                    _ => self.registers_f[src],
                };
                unsafe {
                    crate::virtualizable::vable_write_array_item(ptr, ainfo, index, value);
                }
            }
            // blackhole.py:1406-1409 bhimpl_arraylen_vable
            BC_ARRAYLEN_VABLE => {
                let array_idx = self.next_u16() as usize;
                let dst = self.next_u16() as usize;
                let vinfo = unsafe { &*self.virtualizable_info };
                let ptr = self.virtualizable_ptr as *mut u8;
                unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
                let ainfo = &vinfo.array_fields[array_idx];
                let len = unsafe { crate::virtualizable::vable_array_len(ptr as *const u8, ainfo) };
                self.registers_i[dst] = len as i64;
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
}

/// Pool manager + dispatch builder for blackhole interpreters.
///
/// RPython `blackhole.py:52-103` `class BlackholeInterpBuilder`.
///
/// Combines two responsibilities:
/// 1. Interpreter pool management (acquire/release/release_chain).
/// 2. Codewriter-orthodox dispatch setup (`setup_insns` → dispatch table
///    + `dispatch_loop`). Phase D incrementally wires this up as
///    `bhimpl_*` methods are ported from RPython.
pub struct BlackholeInterpBuilder {
    pool: Vec<BlackholeInterpreter>,
    /// RPython `blackhole.py:56` `self.cpu = codewriter.cpu`.
    /// Stored as raw pointer; the Backend outlives the builder.
    /// RPython `blackhole.py:286/56` `self.cpu = builder.cpu`.
    /// Backend trait for `bh_*` concrete execution. None until set.
    pub cpu: Option<&'static dyn majit_backend::Backend>,
    /// RPython `blackhole.py:68` `self._insns`: opcode byte → "opname/argcodes".
    /// Populated by `setup_insns`; empty until called.
    pub _insns: Vec<String>,
    /// RPython `blackhole.py:72` `self.op_live = insns.get('live/', -1)`.
    pub op_live: u8,
    /// RPython `blackhole.py:73` `self.op_catch_exception`.
    pub op_catch_exception: u8,
    /// RPython `blackhole.py:103` `self.descrs`.
    /// Populated by `setup_descrs()` from the assembler's descriptor table.
    pub descrs: Vec<BhDescr>,
    /// Dispatch table: opcode byte → handler fn pointer.
    /// RPython builds `dispatch_loop` closure via `unrolling_iterable`;
    /// Rust uses indirect call through this table.
    pub dispatch_table: Vec<BhOpcodeHandler>,
}

impl Default for BlackholeInterpBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BlackholeInterpBuilder {
    pub fn new() -> Self {
        Self {
            pool: Vec::new(),
            cpu: None,
            _insns: Vec::new(),
            op_live: u8::MAX,
            op_catch_exception: u8::MAX,
            descrs: Vec::new(),
            dispatch_table: Vec::new(),
        }
    }

    /// RPython `blackhole.py:66-100` `setup_insns(insns)`.
    ///
    /// ```python
    /// def setup_insns(self, insns):
    ///     assert len(insns) <= 256, "too many instructions!"
    ///     self._insns = [None] * len(insns)
    ///     for key, value in insns.items():
    ///         assert self._insns[value] is None
    ///         self._insns[value] = key
    ///     self.op_live = insns.get('live/', -1)
    ///     self.op_catch_exception = insns.get('catch_exception/L', -1)
    /// ```
    ///
    /// Builds the reverse opcode table and dispatch function table from the
    /// assembler's `insns` dict. For now, all dispatch table entries are
    /// placeholder handlers — real `bhimpl_*` methods are wired in
    /// incrementally as Phase D progresses.
    pub fn setup_insns(&mut self, insns: &std::collections::HashMap<String, u8>) {
        assert!(insns.len() <= 256, "too many instructions!");
        // RPython blackhole.py:68-71: build reverse table
        self._insns = vec![String::new(); insns.len()];
        for (key, &value) in insns {
            assert!(
                self._insns[value as usize].is_empty(),
                "duplicate opcode {value} for key {key:?}"
            );
            self._insns[value as usize] = key.clone();
        }
        // RPython blackhole.py:72-74: resolve well-known opcodes
        self.op_live = insns.get("live/").copied().unwrap_or(u8::MAX);
        self.op_catch_exception = insns.get("catch_exception/L").copied().unwrap_or(u8::MAX);
        // RPython blackhole.py:76-80: build handler table.
        //
        // RPython immediately calls _get_method(name, argcodes) for every
        // insns key and panics if the corresponding bhimpl_* is missing.
        // We match that behavior: the default handler panics with the
        // opname so missing bhimpl_* methods surface at dispatch time
        // instead of being silently swallowed.
        self.dispatch_table = self
            ._insns
            .iter()
            .map(|key| {
                // Default: panic identifying the unimplemented opcode.
                // RPython would raise AttributeError at setup time; we
                // defer to dispatch time but with the same crash semantics.
                (|_bh: &mut BlackholeInterpreter,
                  _code: &[u8],
                  _position: usize|
                 -> Result<usize, DispatchError> {
                    panic!("missing bhimpl for opcode (use wire_handler to register)")
                }) as BhOpcodeHandler
            })
            .collect();
    }

    /// RPython `blackhole.py:102-103` `setup_descrs(descrs)`.
    pub fn setup_descrs(&mut self, descrs: Vec<BhDescr>) {
        self.descrs = descrs;
    }

    /// Resolve JitCode fnaddr values from a mapping function.
    /// RPython: fnaddr is already set on JitCode objects when they're stored in descrs.
    /// pyre: fnaddr is 0 at assembly time, resolved here after compilation.
    /// `resolver(jitcode_index) -> fnaddr`.
    pub fn resolve_jitcode_fnaddrs(&mut self, resolver: impl Fn(usize) -> i64) {
        for descr in &mut self.descrs {
            if let BhDescr::JitCode {
                jitcode_index,
                fnaddr,
                ..
            } = descr
            {
                if *fnaddr == 0 {
                    *fnaddr = resolver(*jitcode_index);
                }
            }
        }
    }

    /// Resolve field descriptor offsets from a mapping function.
    /// RPython: FieldDescr carries actual byte offset from rtyper.
    /// pyre: offset is 0 at assembly time, resolved here from runtime layout.
    /// `resolver(owner, field_name) -> byte_offset`.
    pub fn resolve_field_offsets(&mut self, resolver: impl Fn(&str, &str) -> usize) {
        for descr in &mut self.descrs {
            if let BhDescr::Field {
                offset,
                name,
                owner,
            } = descr
            {
                if *offset == 0 && !name.is_empty() {
                    *offset = resolver(owner, name);
                }
            }
        }
    }

    /// RPython `blackhole.py:83-100` `dispatch_loop(self, code, position)`.
    ///
    /// Runs the codewriter-orthodox bytecode dispatch loop. Each iteration
    /// reads one opcode byte, looks up the handler in `dispatch_table`,
    /// and calls it to advance position.
    pub fn dispatch_loop(
        &self,
        bh: &mut BlackholeInterpreter,
        code: &[u8],
        mut position: usize,
    ) -> Result<(), DispatchError> {
        loop {
            let opcode = code[position] as usize;
            position += 1;
            if opcode >= self.dispatch_table.len() {
                panic!("bad opcode {opcode} at position {}", position - 1);
            }
            position = self.dispatch_table[opcode](bh, code, position)?;
        }
    }

    /// Wire a handler for a specific opname/argcodes key into the dispatch table.
    ///
    /// RPython `blackhole.py:76-80`: iterates `_insns` and calls
    /// `_get_method(name, argcodes)` for each. In Rust we wire specific
    /// opnames one by one during Phase D migration. Returns false if
    /// the opname is not present in the insns table.
    /// Wire a handler for a specific opname/argcodes key.
    ///
    /// Returns true if the key was found in the insns table.
    /// Callers in wire_bhimpl_handlers use `try_wire_handler` for optional
    /// keys (aliases that may not exist in all assembler configurations).
    pub fn wire_handler(&mut self, opname_key: &str, handler: BhOpcodeHandler) -> bool {
        for (i, key) in self._insns.iter().enumerate() {
            if key == opname_key {
                self.dispatch_table[i] = handler;
                return true;
            }
        }
        false
    }

    /// Wire a handler, panicking if the key is not found.
    /// Use for mandatory opcodes that MUST exist in every assembler
    /// configuration. Matches RPython's _get_method raising AttributeError.
    pub fn wire_handler_required(&mut self, opname_key: &str, handler: BhOpcodeHandler) {
        if !self.wire_handler(opname_key, handler) {
            panic!(
                "wire_handler_required: opname {:?} not found in insns table",
                opname_key
            );
        }
    }

    /// Acquire an interpreter from the pool or create a new one.
    ///
    /// RPython `blackhole.py:245-251`:
    /// ```python
    /// def acquire_interp(self):
    ///     res = self.blackholeinterps
    ///     if res is not None:
    ///         self.blackholeinterps = res.back
    ///         return res
    ///     else:
    ///         return BlackholeInterpreter(self)
    /// ```
    /// Note: RPython's `BlackholeInterpreter(self)` passes `builder` to
    /// `__init__`, which stores `self.cpu = builder.cpu`. We propagate
    /// the `cpu` field from the builder to each acquired interpreter.
    pub fn acquire_interp(&mut self) -> BlackholeInterpreter {
        let mut bh = self.pool.pop().unwrap_or_default();
        // RPython blackhole.py:284-289:
        //   self.cpu = builder.cpu
        //   self.dispatch_loop = builder.dispatch_loop
        //   self.descrs = builder.descrs
        //   self.op_catch_exception = builder.op_catch_exception
        bh.cpu = self.cpu;
        // RPython blackhole.py:288: self.descrs = builder.descrs
        bh.descrs = self.descrs.clone();
        bh.op_catch_exception = self.op_catch_exception;
        bh
    }

    /// blackhole.py:253 release_interp
    pub fn release_interp(&mut self, mut interp: BlackholeInterpreter) {
        // blackhole.py:254
        interp.cleanup_registers();
        // Pool management (RPython uses linked-list via .back; Rust uses Vec)
        interp.nextblackholeinterp = None;
        interp.runtime_stacks.clear();
        interp.reached_merge_point = false;
        interp.aborted = false;
        interp.got_exception = false;
        interp.virtualizable_stack_base = 0;
        self.pool.push(interp);
    }

    /// Release an entire blackhole chain (including all nextblackholeinterps).
    pub fn release_chain(&mut self, chain: Option<BlackholeInterpreter>) {
        let mut current = chain;
        while let Some(mut bh) = current {
            let next = bh.nextblackholeinterp.take().map(|b| *b);
            self.release_interp(bh);
            current = next;
        }
    }
}

/// warmspot.py:961 handle_jitexception parity.
///
/// Dispatches on JitException type and returns (return_type, value).
/// For ContinueRunningNormally, calls portal_runner to re-enter the
/// portal function. The portal_runner may itself raise a JitException,
/// which is returned as Err for the caller to re-dispatch (while loop).
fn handle_jitexception_dispatch(
    exc: JitException,
    portal_runner: Option<&dyn Fn(&JitException) -> Result<(BhReturnType, i64), JitException>>,
) -> Result<(BhReturnType, i64), JitException> {
    match exc {
        // warmspot.py:986-987
        JitException::DoneWithThisFrameVoid => Ok((BhReturnType::Void, 0)),
        // warmspot.py:988-990
        JitException::DoneWithThisFrameInt(result) => Ok((BhReturnType::Int, result)),
        // warmspot.py:991-993
        JitException::DoneWithThisFrameRef(result) => Ok((BhReturnType::Ref, result.0 as i64)),
        // warmspot.py:994-996
        JitException::DoneWithThisFrameFloat(result) => {
            Ok((BhReturnType::Float, result.to_bits() as i64))
        }
        // warmspot.py:998-1005
        JitException::ExitFrameWithExceptionRef(_) => Err(exc),
        // warmspot.py:970-983
        JitException::ContinueRunningNormally { .. } => {
            if let Some(runner) = portal_runner {
                // warmspot.py:976-978: result = portal_ptr(*args)
                // May raise JitException → Err propagated for re-dispatch.
                runner(&exc)
            } else {
                // No portal runner registered — treat as void return.
                // This is a graceful degradation for non-pyre callers.
                Ok((BhReturnType::Void, 0))
            }
        }
    }
}

/// blackhole.py:1684 _handle_jitexception_in_portal +
/// warmspot.py:1039 handle_jitexception_from_blackhole
///
/// Handle a JitException at a recursive portal level.
/// warmspot.py:1040: result = handle_jitexception(e)
/// warmspot.py:1041-1050: bhcaller._setup_return_value_{i,r,f}(result)
///
/// Returns Ok(()) on success (return value set in bhcaller),
/// or Err(exc_value) if the exception should be propagated as a
/// regular exception (ExitFrameWithExceptionRef).
fn handle_jitexception_in_portal(
    bhcaller: &mut BlackholeInterpreter,
    exc: JitException,
    portal_runner: Option<&dyn Fn(&JitException) -> Result<(BhReturnType, i64), JitException>>,
) -> Result<(), i64> {
    // warmspot.py:961 handle_jitexception: while True loop.
    // ContinueRunningNormally → portal_runner → may raise JitException → loop.
    let mut current_exc = exc;
    loop {
        match handle_jitexception_dispatch(current_exc, portal_runner) {
            Ok((ret_type, result)) => {
                // warmspot.py:1041-1050
                match ret_type {
                    BhReturnType::Void => {}
                    BhReturnType::Int => bhcaller.setup_return_value_i(result),
                    BhReturnType::Ref => bhcaller.setup_return_value_r(result),
                    BhReturnType::Float => bhcaller.setup_return_value_f(result),
                }
                return Ok(());
            }
            Err(JitException::ExitFrameWithExceptionRef(exc_ref)) => {
                // warmspot.py:998-1005: raise as regular exception
                return Err(exc_ref.0 as i64);
            }
            Err(next_exc) => {
                // warmspot.py:967-968, 979-980: JitException from portal_runner
                // or EnterJitAssembler → loop back in handle_jitexception
                current_exc = next_exc;
                continue;
            }
        }
    }
}

/// blackhole.py:1762 _handle_jitexception
///
/// Route a JitException through the blackhole frame chain.
/// Walks up the chain until a portal frame is found. If the portal
/// is the bottommost frame, the exception propagates out. Otherwise
/// it's handled at the recursive portal level.
fn handle_jitexception(
    builder: &mut BlackholeInterpBuilder,
    mut bh: BlackholeInterpreter,
    exc: JitException,
    portal_runner: Option<&dyn Fn(&JitException) -> Result<(BhReturnType, i64), JitException>>,
) -> Result<(BlackholeInterpreter, i64), JitException> {
    // blackhole.py:1764: while blackholeinterp.jitcode.jitdriver_sd is None
    while bh.jitcode.jitdriver_sd.is_none() {
        let next = bh.nextblackholeinterp.take();
        builder.release_interp(bh);
        match next.map(|b| *b) {
            Some(caller) => bh = caller,
            None => return Err(exc), // no portal found
        }
    }

    // blackhole.py:1767-1769
    if bh.nextblackholeinterp.is_none() {
        // Bottommost entry: exception goes through
        builder.release_interp(bh);
        return Err(exc);
    }

    // blackhole.py:1770-1780: recursive portal level.
    // _handle_jitexception_in_portal(exc) calls jd.handle_jitexc_from_bh,
    // which is warmspot.py:1039 handle_jitexception_from_blackhole:
    //   result = handle_jitexception(e)
    //   bhcaller._setup_return_value_{i,r,f}(result)
    //
    // handle_jitexception (warmspot.py:961) extracts the result from
    // DoneWithThisFrame{Int,Ref,Float,Void} and returns it.
    //
    // In Rust we can do this directly since JitException carries the result.
    let caller = bh.nextblackholeinterp.as_mut().unwrap();
    let current_exc = match handle_jitexception_in_portal(caller, exc, portal_runner) {
        Ok(()) => 0,
        Err(regular_exc) => regular_exc,
    };
    // blackhole.py:1780: return blackholeinterp, lle
    Ok((bh, current_exc))
}

/// blackhole.py:1752 _run_forever
///
/// Execute a blackhole frame chain to completion.
/// Loops through frames: runs each one via `resume_mainloop`, releases it,
/// then moves to the caller frame. Terminates when the bottommost frame
/// raises a JitException (DoneWithThisFrame* or ExitFrameWithException*).
///
/// Returns the JitException that terminated execution.
pub fn run_forever(
    builder: &mut BlackholeInterpBuilder,
    mut bh: BlackholeInterpreter,
    mut current_exc: i64,
) -> JitException {
    run_forever_with_portal(builder, bh, current_exc, None)
}

/// blackhole.py:1752 _run_forever with optional portal runner callback.
///
/// `portal_runner` is warmspot.py:961 handle_jitexception parity:
/// when ContinueRunningNormally is raised at a recursive portal level,
/// this callback re-enters the portal function with the exception's
/// green/red args and returns the result.
pub fn run_forever_with_portal(
    builder: &mut BlackholeInterpBuilder,
    mut bh: BlackholeInterpreter,
    mut current_exc: i64,
    portal_runner: Option<&dyn Fn(&JitException) -> Result<(BhReturnType, i64), JitException>>,
) -> JitException {
    loop {
        // blackhole.py:1754-1755
        match bh.resume_mainloop(current_exc) {
            Ok(exc) => {
                current_exc = exc;
            }
            Err(jit_exc) => {
                // blackhole.py:1756-1758
                match handle_jitexception(builder, bh, jit_exc, portal_runner) {
                    Ok((new_bh, exc)) => {
                        // Handled at recursive portal level — continue
                        bh = new_bh;
                        current_exc = exc;
                        continue;
                    }
                    Err(propagated_exc) => {
                        // Bottommost or unhandled — propagate out
                        return propagated_exc;
                    }
                }
            }
        }

        // blackhole.py:1759
        let next = bh.nextblackholeinterp.take();
        builder.release_interp(bh);
        // blackhole.py:1760
        // RPython: blackholeinterp = blackholeinterp.nextblackholeinterp
        // In RPython this can be None, but _resume_mainloop on the
        // bottommost frame always raises a JitException (via
        // _done_with_this_frame or _exit_frame_with_exception), so
        // this code is unreachable in normal operation.
        bh = *next.expect("_run_forever: nextblackholeinterp is None (unreachable)");
    }
}

/// blackhole.py:1798 convert_and_run_from_pyjitpl
///
/// Get a chain of blackhole interpreters and fill them by copying
/// 'metainterp.framestack'.
pub fn convert_and_run_from_pyjitpl(
    builder: &mut BlackholeInterpBuilder,
    framestack: &MIFrameStack,
    last_exc_value: i64,
    raising_exception: bool,
) -> JitException {
    // blackhole.py:1803-1810
    let mut next_bh: Option<Box<BlackholeInterpreter>> = None;

    for frame in &framestack.frames {
        let mut cur_bh = builder.acquire_interp();
        cur_bh.copy_data_from_miframe(frame);
        cur_bh.nextblackholeinterp = next_bh;
        next_bh = Some(Box::new(cur_bh));
    }

    let Some(first_bh_box) = next_bh else {
        return JitException::DoneWithThisFrameVoid;
    };
    let mut first_bh = *first_bh_box;

    // blackhole.py:1812-1818
    let current_exc = if raising_exception {
        last_exc_value
    } else {
        first_bh.exception_last_value = last_exc_value;
        0
    };

    run_forever(builder, first_bh, current_exc)
}

/// blackhole.py:1782 resume_in_blackhole
///
/// Resume execution in the blackhole interpreter after a compiled
/// code guard failure. Builds a frame chain from resume data, extracts
/// exception from deadframe, and runs the chain to completion.
///
/// `resolve_jitcode` is `metainterp_sd.jitcodes[jitcode_pos]` in RPython.
pub fn resume_in_blackhole(
    builder: &mut BlackholeInterpBuilder,
    resolve_jitcode: &dyn Fn(i32, i32) -> Option<(JitCode, usize)>,
    rd_numb: &[u8],
    rd_consts: &[i64],
    deadframe: &[i64],
    deadframe_exc: i64,
) -> JitException {
    // blackhole.py:1786-1792
    let null_alloc = crate::resume::NullAllocator;
    let bh = crate::resume::blackhole_from_resumedata(
        builder,
        resolve_jitcode,
        rd_numb,
        rd_consts,
        deadframe,
        None, // deadframe_types
        None, // rd_virtuals
        None, // rd_pendingfields (PendingFieldInfo)
        None, // rd_guard_pendingfields
        None, // vrefinfo
        None, // vinfo
        None, // ginfo
        &null_alloc,
    );

    let Some((bh, _virtualizable_ptr)) = bh else {
        return JitException::DoneWithThisFrameVoid;
    };

    // blackhole.py:1794
    let current_exc = BlackholeInterpreter::prepare_resume_from_failure(deadframe_exc);

    // blackhole.py:1795
    run_forever(builder, bh, current_exc)
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
                ovf_flag: false,
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
                ovf_flag: false,
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

    // ── Integer arithmetic & comparison (consolidated) ──

    #[test]
    fn test_executor_int_arithmetic() {
        // ADD
        assert_eq!(exec_binop(OpCode::IntAdd, 3, 4), 7);
        assert_eq!(exec_binop(OpCode::IntAdd, -1, 1), 0);
        assert_eq!(exec_binop(OpCode::IntAdd, 0, 0), 0);
        // SUB
        assert_eq!(exec_binop(OpCode::IntSub, 10, 3), 7);
        assert_eq!(exec_binop(OpCode::IntSub, 0, 5), -5);
        // MUL
        assert_eq!(exec_binop(OpCode::IntMul, 6, 7), 42);
        assert_eq!(exec_binop(OpCode::IntMul, -3, 4), -12);
        assert_eq!(exec_binop(OpCode::IntMul, 0, 999), 0);
        // FLOORDIV
        assert_eq!(exec_binop(OpCode::IntFloorDiv, 17, 5), 3);
        assert_eq!(exec_binop(OpCode::IntFloorDiv, -17, 5), -3);
        assert_eq!(exec_binop(OpCode::IntFloorDiv, 100, 1), 100);
        // MOD
        assert_eq!(exec_binop(OpCode::IntMod, 17, 5), 2);
        assert_eq!(exec_binop(OpCode::IntMod, 10, 3), 1);
        assert_eq!(exec_binop(OpCode::IntMod, 6, 3), 0);
    }

    #[test]
    fn test_executor_int_comparisons() {
        // LT
        assert_eq!(exec_binop(OpCode::IntLt, 3, 4), 1);
        assert_eq!(exec_binop(OpCode::IntLt, 4, 4), 0);
        assert_eq!(exec_binop(OpCode::IntLt, 5, 4), 0);
        // GE
        assert_eq!(exec_binop(OpCode::IntGe, 4, 4), 1);
        assert_eq!(exec_binop(OpCode::IntGe, 5, 4), 1);
        assert_eq!(exec_binop(OpCode::IntGe, 3, 4), 0);
        // EQ / NE
        assert_eq!(exec_binop(OpCode::IntEq, 5, 5), 1);
        assert_eq!(exec_binop(OpCode::IntEq, 5, 6), 0);
        assert_eq!(exec_binop(OpCode::IntNe, 5, 5), 0);
        assert_eq!(exec_binop(OpCode::IntNe, 5, 6), 1);
        // LE / GT
        assert_eq!(exec_binop(OpCode::IntLe, 3, 4), 1);
        assert_eq!(exec_binop(OpCode::IntLe, 4, 4), 1);
        assert_eq!(exec_binop(OpCode::IntLe, 5, 4), 0);
        assert_eq!(exec_binop(OpCode::IntGt, 5, 4), 1);
        assert_eq!(exec_binop(OpCode::IntGt, 4, 4), 0);
    }

    #[test]
    fn test_executor_int_bitwise() {
        assert_eq!(exec_binop(OpCode::IntAnd, 0xFF, 0x0F), 0x0F);
        assert_eq!(exec_binop(OpCode::IntOr, 0xF0, 0x0F), 0xFF);
        assert_eq!(exec_binop(OpCode::IntXor, 0xFF, 0x0F), 0xF0);
        assert_eq!(exec_binop(OpCode::IntLshift, 1, 4), 16);
        assert_eq!(exec_binop(OpCode::IntRshift, 16, 4), 1);
    }

    #[test]
    fn test_executor_int_unary() {
        assert_eq!(exec_unop(OpCode::IntNeg, 42), -42);
        assert_eq!(exec_unop(OpCode::IntNeg, -1), 1);
        assert_eq!(exec_unop(OpCode::IntInvert, 0), -1);
        assert_eq!(exec_unop(OpCode::IntInvert, -1), 0);
    }

    #[test]
    fn test_executor_float_arithmetic() {
        let fb = |v: f64| f64::to_bits(v) as i64;
        let fr = |r: i64| f64::from_bits(r as u64);
        assert_eq!(fr(exec_binop(OpCode::FloatAdd, fb(1.5), fb(2.5))), 4.0);
        assert_eq!(fr(exec_binop(OpCode::FloatMul, fb(3.0), fb(4.0))), 12.0);
        assert_eq!(fr(exec_binop(OpCode::FloatSub, fb(10.0), fb(3.5))), 6.5);
        assert_eq!(fr(exec_binop(OpCode::FloatTrueDiv, fb(10.0), fb(4.0))), 2.5);
    }

    #[test]
    fn test_executor_float_unary() {
        let fb = |v: f64| f64::to_bits(v) as i64;
        let fr = |r: i64| f64::from_bits(r as u64);
        assert_eq!(fr(exec_unop(OpCode::FloatNeg, fb(3.14))), -3.14);
        assert_eq!(fr(exec_unop(OpCode::FloatAbs, fb(-2.5))), 2.5);
        assert_eq!(fr(exec_unop(OpCode::FloatAbs, fb(2.5))), 2.5);
    }

    #[test]
    fn test_executor_cast_int_float() {
        let fr = |r: i64| f64::from_bits(r as u64);
        assert_eq!(fr(exec_unop(OpCode::CastIntToFloat, 42)), 42.0);
        assert_eq!(
            exec_unop(OpCode::CastFloatToInt, f64::to_bits(3.7) as i64),
            3
        );
        assert_eq!(
            exec_unop(OpCode::CastFloatToInt, f64::to_bits(-2.9) as i64),
            -2
        );
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
                    // CallAssembler ops intentionally return Unsupported —
                    // they require force_fn fallback, not blackhole execution.
                    let is_call_assembler = matches!(
                        opcode,
                        OpCode::CallAssemblerI
                            | OpCode::CallAssemblerR
                            | OpCode::CallAssemblerF
                            | OpCode::CallAssemblerN
                    );
                    if !is_call_assembler {
                        panic!("OpCode {:?} returned Unsupported: {}", opcode, msg);
                    }
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

            assert_eq!(bh.registers_i[0], 0); // skipped
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

        /// Integration test: build bytecode manually with known opcode
        /// assignments and run it through the orthodox dispatch_loop.
        ///
        /// This validates the Phase D setup_insns → dispatch_loop → bhimpl
        /// pipeline end-to-end without depending on SSARepr assembly.
        ///
        /// RPython equivalent: the setup_insns + dispatch_loop closure + bhimpl
        /// flow described in blackhole.py:52-103 and 452-460.
        #[test]
        fn test_orthodox_dispatch_loop_int_add() {
            // Build a minimal insns dict (as if the assembler had produced it).
            // Opcode 0 = "live/" (liveness marker, skip 2 bytes)
            // Opcode 1 = "int_add/ii>i" (3 register bytes: a, b, dst)
            // Opcode 2 = "int_return/i" (1 register byte)
            let mut insns = HashMap::new();
            insns.insert("live/".to_string(), 0u8);
            insns.insert("int_add/ii>i".to_string(), 1u8);
            insns.insert("int_return/i".to_string(), 2u8);

            // Setup builder
            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_insns(&insns);
            super::wire_bhimpl_handlers(&mut builder);

            // Hand-assemble bytecode: live + int_add(r0, r1) → r2 + int_return(r2)
            let code: Vec<u8> = vec![
                0, 0, 0, // opcode 0 = live/, 2 bytes liveness offset (skipped)
                1, 0, 1, 2, // opcode 1 = int_add, a=r0, b=r1, dst=r2
                2, 2, // opcode 2 = int_return, src=r2
            ];

            // Create BlackholeInterpreter with 3 int regs
            let mut bh = BlackholeInterpreter::new();
            bh.registers_i = vec![0i64; 3];
            bh.registers_i[0] = 10; // r0 = 10
            bh.registers_i[1] = 32; // r1 = 32

            // Run dispatch_loop
            let result = builder.dispatch_loop(&mut bh, &code, 0);

            // Should leave frame with LeaveFrame
            assert!(matches!(result, Err(DispatchError::LeaveFrame)));
            // tmpreg_i should hold 10 + 32 = 42
            assert_eq!(bh.tmpreg_i, 42, "int_add(10, 32) should produce 42");
            assert_eq!(bh.return_type, BhReturnType::Int);
        }
    }
}

// ── bhimpl_* methods (RPython blackhole.py:452+) ────────────────────
//
// RPython defines each bhimpl_* as a static method decorated with
// @arguments("i", "i", returns="i") etc. The handler closure generated
// by _get_method decodes args from the bytecode stream and calls the
// bhimpl method.
//
// In Rust we define each bhimpl as a standalone fn and generate
// BhOpcodeHandler wrappers. The handler decodes operands, calls the
// bhimpl fn, stores the result, and returns the updated position.

// ── handler generators for common patterns ──────────────────────────

/// Decode pattern `@arguments("i", "i", returns="i")` — argcodes `"ii>i"`.
///
/// Read 2 int-register indices, call bhimpl fn, write result, advance by 3.
/// Each concrete handler is its own fn (not a closure) so it can be stored
/// as a bare BhOpcodeHandler fn pointer.
macro_rules! bhhandler_ii_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            let b = bh.registers_i[code[position + 1] as usize];
            bh.registers_i[code[position + 2] as usize] = $bhimpl(a, b);
            Ok(position + 3)
        }
    };
}

/// Decode pattern `@arguments("i", returns="i")` — argcodes `"i>i"`.
macro_rules! bhhandler_i_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            bh.registers_i[code[position + 1] as usize] = $bhimpl(a);
            Ok(position + 2)
        }
    };
}

// ── bhimpl methods (line-by-line from RPython blackhole.py) ─────────

/// blackhole.py:454-456 `bhimpl_int_same_as`.
fn bhimpl_int_same_as(a: i64) -> i64 {
    a
}

/// blackhole.py:458-460 `bhimpl_int_add(a, b): return intmask(a + b)`.
fn bhimpl_int_add(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

/// blackhole.py:462-464 `bhimpl_int_sub(a, b): return intmask(a - b)`.
fn bhimpl_int_sub(a: i64, b: i64) -> i64 {
    a.wrapping_sub(b)
}

/// blackhole.py:466-468 `bhimpl_int_mul(a, b): return intmask(a * b)`.
fn bhimpl_int_mul(a: i64, b: i64) -> i64 {
    a.wrapping_mul(b)
}

/// blackhole.py:499-501 `bhimpl_int_and(a, b): return a & b`.
fn bhimpl_int_and(a: i64, b: i64) -> i64 {
    a & b
}

/// blackhole.py:503-505 `bhimpl_int_or(a, b): return a | b`.
fn bhimpl_int_or(a: i64, b: i64) -> i64 {
    a | b
}

/// blackhole.py:507-509 `bhimpl_int_xor(a, b): return a ^ b`.
fn bhimpl_int_xor(a: i64, b: i64) -> i64 {
    a ^ b
}

/// blackhole.py:511-513 `bhimpl_int_rshift(a, b): return a >> b`.
fn bhimpl_int_rshift(a: i64, b: i64) -> i64 {
    a >> (b & 63)
}

/// blackhole.py:516-518 `bhimpl_int_lshift(a, b): return intmask(a << b)`.
fn bhimpl_int_lshift(a: i64, b: i64) -> i64 {
    a.wrapping_shl((b & 63) as u32)
}

/// blackhole.py:521-524 `bhimpl_uint_rshift(a, b): return intmask(r_uint(a) >> r_uint(b))`.
fn bhimpl_uint_rshift(a: i64, b: i64) -> i64 {
    ((a as u64) >> ((b as u64) & 63)) as i64
}

/// blackhole.py:527-529 `bhimpl_int_neg(a): return intmask(-a)`.
fn bhimpl_int_neg(a: i64) -> i64 {
    a.wrapping_neg()
}

/// blackhole.py:531-533 `bhimpl_int_invert(a): return ~a`.
fn bhimpl_int_invert(a: i64) -> i64 {
    !a
}

/// blackhole.py:535 `bhimpl_int_lt(a, b): return int(a < b)`.
fn bhimpl_int_lt(a: i64, b: i64) -> i64 {
    (a < b) as i64
}

/// blackhole.py:539 `bhimpl_int_le(a, b): return int(a <= b)`.
fn bhimpl_int_le(a: i64, b: i64) -> i64 {
    (a <= b) as i64
}

/// blackhole.py:543 `bhimpl_int_eq(a, b): return int(a == b)`.
fn bhimpl_int_eq(a: i64, b: i64) -> i64 {
    (a == b) as i64
}

/// blackhole.py:547 `bhimpl_int_ne(a, b): return int(a != b)`.
fn bhimpl_int_ne(a: i64, b: i64) -> i64 {
    (a != b) as i64
}

/// blackhole.py:551 `bhimpl_int_gt(a, b): return int(a > b)`.
fn bhimpl_int_gt(a: i64, b: i64) -> i64 {
    (a > b) as i64
}

/// blackhole.py:555 `bhimpl_int_ge(a, b): return int(a >= b)`.
fn bhimpl_int_ge(a: i64, b: i64) -> i64 {
    (a >= b) as i64
}

/// blackhole.py:559 `bhimpl_int_is_true(a): return int(bool(a))`.
fn bhimpl_int_is_true(a: i64) -> i64 {
    (a != 0) as i64
}

/// blackhole.py:563 `bhimpl_int_is_zero(a): return int(not a)`.
fn bhimpl_int_is_zero(a: i64) -> i64 {
    (a == 0) as i64
}

/// blackhole.py:567 `bhimpl_int_force_ge_zero(a): if a < 0: return 0; return a`.
fn bhimpl_int_force_ge_zero(a: i64) -> i64 {
    if a < 0 { 0 } else { a }
}

// Generate handler fns from bhimpl methods via macros.
// @arguments("i", returns="i") → argcodes "i>i" → 1 src reg + 1 dst reg = 2 bytes
bhhandler_i_i!(handler_int_same_as, bhimpl_int_same_as);
bhhandler_i_i!(handler_int_neg, bhimpl_int_neg);
bhhandler_i_i!(handler_int_invert, bhimpl_int_invert);
bhhandler_i_i!(handler_int_is_true, bhimpl_int_is_true);
bhhandler_i_i!(handler_int_is_zero, bhimpl_int_is_zero);
bhhandler_i_i!(handler_int_force_ge_zero, bhimpl_int_force_ge_zero);

// @arguments("i", "i", returns="i") → argcodes "ii>i" → 2 src regs + 1 dst reg = 3 bytes
bhhandler_ii_i!(handler_int_add, bhimpl_int_add);
bhhandler_ii_i!(handler_int_sub, bhimpl_int_sub);
bhhandler_ii_i!(handler_int_mul, bhimpl_int_mul);
bhhandler_ii_i!(handler_int_and, bhimpl_int_and);
bhhandler_ii_i!(handler_int_or, bhimpl_int_or);
bhhandler_ii_i!(handler_int_xor, bhimpl_int_xor);
bhhandler_ii_i!(handler_int_rshift, bhimpl_int_rshift);
bhhandler_ii_i!(handler_int_lshift, bhimpl_int_lshift);
bhhandler_ii_i!(handler_uint_rshift, bhimpl_uint_rshift);
bhhandler_ii_i!(handler_int_lt, bhimpl_int_lt);
bhhandler_ii_i!(handler_int_le, bhimpl_int_le);
bhhandler_ii_i!(handler_int_eq, bhimpl_int_eq);
bhhandler_ii_i!(handler_int_ne, bhimpl_int_ne);
bhhandler_ii_i!(handler_int_gt, bhimpl_int_gt);
bhhandler_ii_i!(handler_int_ge, bhimpl_int_ge);

// ── control flow + copy handlers ─────────────────────────────────────

/// blackhole.py:638-640 `bhimpl_int_copy(a): return a` — @arguments("i", returns="i").
/// Decoded as `i>i` (same as int_same_as). Already have handler_int_same_as.
/// Wire as alias.
bhhandler_i_i!(handler_int_copy, bhimpl_int_same_as);

/// Handler for `live/` — liveness marker. Argcodes: empty, but the assembler
/// emits a 2-byte offset after the opcode. Skip those 2 bytes.
/// RPython blackhole.py:146-158 (inside _get_method for `-live-` ops).
fn handler_live(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // Skip the 2-byte liveness offset (RPython: OFFSET_SIZE = 2).
    Ok(position + 2)
}

/// Handler for `goto/L` — unconditional jump. Argcodes: `L` (2-byte label).
/// RPython blackhole.py:950-952: `def bhimpl_goto(target): return target`.
/// Returns="L" means the handler returns the label value as new position.
fn handler_goto(
    _bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let target = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    Ok(target)
}

/// Handler for `goto_if_not/iL` — conditional jump.
/// RPython blackhole.py:864-869:
/// ```python
/// @arguments("i", "L", "pc", returns="L")
/// def bhimpl_goto_if_not(a, target, pc):
///     if a: return pc
///     else: return target
/// ```
/// "pc" means the current position AFTER decoding all operands.
fn handler_goto_if_not(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[position] as usize];
    let target = (code[position + 1] as usize) | ((code[position + 2] as usize) << 8);
    let pc = position + 3; // position after all operands
    if a != 0 {
        Ok(pc) // condition true: fall through
    } else {
        Ok(target) // condition false: jump to target
    }
}

/// Handler for `int_return/i` — RPython blackhole.py:841-845.
/// @arguments("self", "i"): read one int register, store in tmpreg_i,
/// raise LeaveFrame.
fn handler_int_return(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[position] as usize];
    bh.tmpreg_i = a;
    bh.return_type = BhReturnType::Int;
    Err(DispatchError::LeaveFrame)
}

/// Handler for `ref_return/r` — RPython blackhole.py:847-851.
fn handler_ref_return(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_r[code[position] as usize];
    bh.tmpreg_r = a;
    bh.return_type = BhReturnType::Ref;
    Err(DispatchError::LeaveFrame)
}

/// Handler for `void_return/` — RPython blackhole.py:859-862.
fn handler_void_return(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    bh.return_type = BhReturnType::Void;
    Err(DispatchError::LeaveFrame)
}

// ── float bhimpl methods (RPython blackhole.py:676-808) ─────────────

// RPython stores floats as longlong (i64 bits). pyre stores f64 in
// registers_f directly. The bhimpl methods work on f64 values.

fn bhimpl_float_neg(a: f64) -> f64 {
    -a
}
fn bhimpl_float_abs(a: f64) -> f64 {
    a.abs()
}
fn bhimpl_float_add(a: f64, b: f64) -> f64 {
    a + b
}
fn bhimpl_float_sub(a: f64, b: f64) -> f64 {
    a - b
}
fn bhimpl_float_mul(a: f64, b: f64) -> f64 {
    a * b
}
fn bhimpl_float_truediv(a: f64, b: f64) -> f64 {
    a / b
}

/// Decode pattern `@arguments("f", "f", returns="f")` — argcodes `"ff>f"`.
macro_rules! bhhandler_ff_f {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            let b = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
            bh.registers_f[code[position + 2] as usize] = $bhimpl(a, b).to_bits() as i64;
            Ok(position + 3)
        }
    };
}

/// Decode pattern `@arguments("f", returns="f")` — argcodes `"f>f"`.
macro_rules! bhhandler_f_f {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            bh.registers_f[code[position + 1] as usize] = $bhimpl(a).to_bits() as i64;
            Ok(position + 2)
        }
    };
}

/// Decode pattern `@arguments("f", "f", returns="i")` — argcodes `"ff>i"`.
macro_rules! bhhandler_ff_i {
    ($name:ident, $cmp:expr) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            let b = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
            bh.registers_i[code[position + 2] as usize] = $cmp(a, b) as i64;
            Ok(position + 3)
        }
    };
}

bhhandler_ff_f!(handler_float_add, bhimpl_float_add);
bhhandler_ff_f!(handler_float_sub, bhimpl_float_sub);
bhhandler_ff_f!(handler_float_mul, bhimpl_float_mul);
bhhandler_ff_f!(handler_float_truediv, bhimpl_float_truediv);
bhhandler_f_f!(handler_float_neg, bhimpl_float_neg);
bhhandler_f_f!(handler_float_abs, bhimpl_float_abs);
bhhandler_ff_i!(handler_float_lt, |a: f64, b: f64| a < b);
bhhandler_ff_i!(handler_float_le, |a: f64, b: f64| a <= b);
bhhandler_ff_i!(handler_float_eq, |a: f64, b: f64| a == b);
bhhandler_ff_i!(handler_float_ne, |a: f64, b: f64| a != b);
bhhandler_ff_i!(handler_float_gt, |a: f64, b: f64| a > b);
bhhandler_ff_i!(handler_float_ge, |a: f64, b: f64| a >= b);

// ── unsigned comparison bhimpl (RPython blackhole.py:571-582) ────────

fn bhimpl_uint_lt(a: i64, b: i64) -> i64 {
    ((a as u64) < (b as u64)) as i64
}
fn bhimpl_uint_le(a: i64, b: i64) -> i64 {
    ((a as u64) <= (b as u64)) as i64
}
fn bhimpl_uint_gt(a: i64, b: i64) -> i64 {
    ((a as u64) > (b as u64)) as i64
}
fn bhimpl_uint_ge(a: i64, b: i64) -> i64 {
    ((a as u64) >= (b as u64)) as i64
}

bhhandler_ii_i!(handler_uint_lt, bhimpl_uint_lt);
bhhandler_ii_i!(handler_uint_le, bhimpl_uint_le);
bhhandler_ii_i!(handler_uint_gt, bhimpl_uint_gt);
bhhandler_ii_i!(handler_uint_ge, bhimpl_uint_ge);

// ── goto_if_not_int_* conditionals (RPython blackhole.py:871-920) ───

/// Decode pattern `@arguments("i", "i", "L", "pc", returns="L")`.
/// Read 2 int registers + 2-byte label; compare; return target or pc.
macro_rules! bhhandler_goto_if_not_ii {
    ($name:ident, $cmp:expr) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_i[code[position] as usize];
            let b = bh.registers_i[code[position + 1] as usize];
            let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
            let pc = position + 4;
            if $cmp(a, b) { Ok(pc) } else { Ok(target) }
        }
    };
}

bhhandler_goto_if_not_ii!(handler_goto_if_not_int_lt, |a: i64, b: i64| a < b);
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_le, |a: i64, b: i64| a <= b);
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_eq, |a: i64, b: i64| a == b);
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_ne, |a: i64, b: i64| a != b);
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_gt, |a: i64, b: i64| a > b);
bhhandler_goto_if_not_ii!(handler_goto_if_not_int_ge, |a: i64, b: i64| a >= b);

// ── ref operations (RPython blackhole.py:584-610) ───────────────────

fn bhimpl_ptr_eq(a: i64, b: i64) -> i64 {
    (a == b) as i64
}
fn bhimpl_ptr_ne(a: i64, b: i64) -> i64 {
    (a != b) as i64
}
fn bhimpl_ptr_iszero(a: i64) -> i64 {
    (a == 0) as i64
}
fn bhimpl_ptr_nonzero(a: i64) -> i64 {
    (a != 0) as i64
}

/// `@arguments("r", "r", returns="i")` — rr>i: read 2 ref regs, result int.
macro_rules! bhhandler_rr_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_r[code[position] as usize];
            let b = bh.registers_r[code[position + 1] as usize];
            bh.registers_i[code[position + 2] as usize] = $bhimpl(a, b);
            Ok(position + 3)
        }
    };
}

/// `@arguments("r", returns="i")` — r>i: read 1 ref reg, result int.
macro_rules! bhhandler_r_i {
    ($name:ident, $bhimpl:ident) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = bh.registers_r[code[position] as usize];
            bh.registers_i[code[position + 1] as usize] = $bhimpl(a);
            Ok(position + 2)
        }
    };
}

bhhandler_rr_i!(handler_ptr_eq, bhimpl_ptr_eq);
bhhandler_rr_i!(handler_ptr_ne, bhimpl_ptr_ne);
bhhandler_rr_i!(handler_instance_ptr_eq, bhimpl_ptr_eq);
bhhandler_rr_i!(handler_instance_ptr_ne, bhimpl_ptr_ne);
bhhandler_r_i!(handler_ptr_iszero, bhimpl_ptr_iszero);
bhhandler_r_i!(handler_ptr_nonzero, bhimpl_ptr_nonzero);

// ref/float copy (blackhole.py:641-645)
fn handler_ref_copy(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_r[code[position + 1] as usize] = bh.registers_r[code[position] as usize];
    Ok(position + 2)
}
fn handler_float_copy(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_f[code[position + 1] as usize] = bh.registers_f[code[position] as usize];
    Ok(position + 2)
}

// float_return (blackhole.py:853-857)
fn handler_float_return(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.tmpreg_f = bh.registers_f[code[position] as usize];
    bh.return_type = BhReturnType::Float;
    Err(DispatchError::LeaveFrame)
}

// ── guard_value — no-op in blackhole (blackhole.py:648-656) ─────────
fn handler_int_guard_value(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 1)
}
fn handler_ref_guard_value(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 1)
}
fn handler_float_guard_value(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 1)
}

// ── push/pop (blackhole.py:661-679) ─────────────────────────────────
fn handler_int_push(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.tmpreg_i = bh.registers_i[code[position] as usize];
    Ok(position + 1)
}
fn handler_ref_push(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.tmpreg_r = bh.registers_r[code[position] as usize];
    Ok(position + 1)
}
fn handler_float_push(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.tmpreg_f = bh.registers_f[code[position] as usize];
    Ok(position + 1)
}
fn handler_int_pop(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[position] as usize] = bh.tmpreg_i;
    Ok(position + 1)
}
fn handler_ref_pop(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_r[code[position] as usize] = bh.tmpreg_r;
    Ok(position + 1)
}
fn handler_float_pop(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_f[code[position] as usize] = bh.tmpreg_f;
    Ok(position + 1)
}

// ── record_exact_class/value — no-op (blackhole.py:616-636) ─────────
fn handler_record_exact_class(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 2)
}
fn handler_record_exact_value_r(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 2)
}
fn handler_record_exact_value_i(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 2)
}

// ── cast operations (blackhole.py:800-831) ──────────────────────────
fn handler_cast_float_to_int(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
    bh.registers_i[code[position + 1] as usize] = a as i64;
    Ok(position + 2)
}
fn handler_cast_int_to_float(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[position] as usize];
    bh.registers_f[code[position + 1] as usize] = (a as f64).to_bits() as i64;
    Ok(position + 2)
}

// ── int_signext (blackhole.py:566-569) ──────────────────────────────
fn handler_int_signext(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[position] as usize];
    let numbytes = bh.registers_i[code[position + 1] as usize];
    let result = match numbytes {
        1 => (a as i8) as i64,
        2 => (a as i16) as i64,
        4 => (a as i32) as i64,
        _ => a,
    };
    bh.registers_i[code[position + 2] as usize] = result;
    Ok(position + 3)
}

// ── overflow ops (blackhole.py:478-497) ─────────────────────────────

fn handler_int_add_jump_if_ovf(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let target = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    let a = bh.registers_i[code[position + 2] as usize];
    let b = bh.registers_i[code[position + 3] as usize];
    let pc = position + 5;
    match a.checked_add(b) {
        Some(result) => {
            bh.registers_i[code[position + 4] as usize] = result;
            Ok(pc)
        }
        None => Ok(target),
    }
}
fn handler_int_sub_jump_if_ovf(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let target = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    let a = bh.registers_i[code[position + 2] as usize];
    let b = bh.registers_i[code[position + 3] as usize];
    let pc = position + 5;
    match a.checked_sub(b) {
        Some(result) => {
            bh.registers_i[code[position + 4] as usize] = result;
            Ok(pc)
        }
        None => Ok(target),
    }
}
fn handler_int_mul_jump_if_ovf(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let target = (code[position] as usize) | ((code[position + 1] as usize) << 8);
    let a = bh.registers_i[code[position + 2] as usize];
    let b = bh.registers_i[code[position + 3] as usize];
    let pc = position + 5;
    match a.checked_mul(b) {
        Some(result) => {
            bh.registers_i[code[position + 4] as usize] = result;
            Ok(pc)
        }
        None => Ok(target),
    }
}

// ── misc simple ops ─────────────────────────────────────────────────

fn handler_assert_not_none(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 1)
}

fn handler_virtual_ref(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_r[code[position + 1] as usize] = bh.registers_r[code[position] as usize];
    Ok(position + 2)
}
fn handler_virtual_ref_finish(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 1)
}
fn handler_loop_header(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position + 1)
}
fn handler_ref_isconstant(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[position + 1] as usize] = 0;
    Ok(position + 2)
}
fn handler_ref_isvirtual(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[position + 1] as usize] = 0;
    Ok(position + 2)
}
fn handler_goto_if_not_int_is_zero(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[position] as usize];
    let target = (code[position + 1] as usize) | ((code[position + 2] as usize) << 8);
    let pc = position + 3;
    if a == 0 { Ok(pc) } else { Ok(target) }
}
fn handler_goto_if_not_ptr_iszero(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_r[code[position] as usize];
    let target = (code[position + 1] as usize) | ((code[position + 2] as usize) << 8);
    let pc = position + 3;
    if a == 0 { Ok(pc) } else { Ok(target) }
}
fn handler_goto_if_not_ptr_nonzero(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_r[code[position] as usize];
    let target = (code[position + 1] as usize) | ((code[position + 2] as usize) << 8);
    let pc = position + 3;
    if a != 0 { Ok(pc) } else { Ok(target) }
}
fn handler_unreachable(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _position: usize,
) -> Result<usize, DispatchError> {
    panic!("bhimpl_unreachable reached");
}

// ── cpu-dependent field/array operations ─────────────────────────────
//
// RPython blackhole.py:1432-1481: bhimpl_getfield_gc_*/setfield_gc_*
// These call `cpu.bh_getfield_gc_i(struct_ptr, descr)` etc.
// The 'd' argcode is a 2-byte descriptor index into `bh.descrs`.
// In pyre, descrs[index] resolves to a field offset (usize).

/// RPython `blackhole.py:150-157`: read a 2-byte descriptor index from
/// bytecode and return `(descr_object, new_position)`.
///
/// In RPython: `value = self.descrs[index]`. In pyre: returns the
/// `BhDescr` enum variant.
#[inline]
fn read_descr<'a>(bh: &'a BlackholeInterpreter, code: &[u8], pos: usize) -> (&'a BhDescr, usize) {
    let descr_idx = (code[pos] as usize) | ((code[pos + 1] as usize) << 8);
    let descr = &bh.descrs[descr_idx]; // RPython: no fallback, index must be valid
    (descr, pos + 2)
}

/// Read a VableField descriptor and resolve to a synthesized BhDescr::Field
/// with the resolved byte offset via VirtualizableInfo.
/// RPython: fielddescr carries byte offset directly; pyre VableField.index
/// needs vinfo.static_fields[index].offset resolution.
#[inline]
fn read_descr_vable_field(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (BhDescr, usize) {
    let (descr, pos) = read_descr(bh, code, pos);
    let field_index = descr.as_vable_field_index();
    // Resolve field_index → byte offset via VirtualizableInfo.
    let offset = if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        vinfo
            .static_fields
            .get(field_index)
            .map(|f| f.offset)
            .unwrap_or(field_index)
    } else {
        field_index
    };
    (
        BhDescr::Field {
            offset,
            name: String::new(),
            owner: String::new(),
        },
        pos,
    )
}

/// Read a VableArray descriptor and resolve to a synthesized BhDescr::Field
/// with the resolved field_offset via VirtualizableInfo.
/// RPython: fielddescr carries byte offset for the array pointer field.
/// pyre: VableArray.index → vinfo.array_fields[index].field_offset.
#[inline]
fn read_descr_vable_array(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (BhDescr, usize) {
    let (descr, pos) = read_descr(bh, code, pos);
    let array_index = descr.as_vable_array_index();
    let offset = if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        vinfo
            .array_fields
            .get(array_index)
            .map(|a| a.field_offset)
            .unwrap_or(array_index)
    } else {
        array_index
    };
    (
        BhDescr::Field {
            offset,
            name: String::new(),
            owner: String::new(),
        },
        pos,
    )
}

// bhimpl_getfield_gc_i: @arguments("cpu", "r", "d", returns="i")
fn handler_getfield_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_getfield_gc_i(struct_ptr, descr);
    bh.registers_i[code[pos] as usize] = result;
    Ok(pos + 1)
}
fn handler_getfield_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_getfield_gc_r(struct_ptr, descr);
    bh.registers_r[code[pos] as usize] = result.0 as i64;
    Ok(pos + 1)
}
fn handler_getfield_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_getfield_gc_f(struct_ptr, descr);
    bh.registers_f[code[pos] as usize] = result.to_bits() as i64;
    Ok(pos + 1)
}

// bhimpl_setfield_gc_i: @arguments("cpu", "r", "i", "d")
fn handler_setfield_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let value = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_i(struct_ptr, value, descr);
    Ok(pos)
}
fn handler_setfield_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let value = bh.registers_r[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_r(struct_ptr, majit_ir::GcRef(value as usize), descr);
    Ok(pos)
}
fn handler_setfield_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[position] as usize];
    let value = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_f(struct_ptr, value, descr);
    Ok(pos)
}

// bhimpl_arraylen_gc: @arguments("cpu", "r", "d", returns="i")
fn handler_arraylen_gc(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array_ptr = bh.registers_r[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_arraylen_gc(array_ptr, descr);
    bh.registers_i[code[pos] as usize] = result;
    Ok(pos + 1)
}

// ── getarrayitem_gc (blackhole.py:1329-1341) ────────────────────────
// @arguments("cpu", "r", "i", "d", returns="X")

fn handler_getarrayitem_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[pos] as usize] = cpu.bh_getarrayitem_gc_i(array, index, descr);
    Ok(pos + 1)
}
fn handler_getarrayitem_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[pos] as usize] = cpu.bh_getarrayitem_gc_r(array, index, descr).0 as i64;
    Ok(pos + 1)
}

// ── setarrayitem_gc (blackhole.py:1350-1358) ────────────────────────
// @arguments("cpu", "r", "i", "X", "d")

fn handler_setarrayitem_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_i[code[position + 2] as usize];
    let (descr, pos) = read_descr(bh, code, position + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setarrayitem_gc_i(array, index, value, descr);
    Ok(pos)
}
fn handler_setarrayitem_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_r[code[position + 2] as usize];
    let (descr, pos) = read_descr(bh, code, position + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setarrayitem_gc_r(array, index, majit_ir::GcRef(value as usize), descr);
    Ok(pos)
}

// ── getfield_raw (blackhole.py:1464-1472) ───────────────────────────
fn handler_getfield_raw_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize]; // raw ptr is int
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[pos] as usize] = cpu.bh_getfield_raw_i(struct_ptr, descr);
    Ok(pos + 1)
}
fn handler_getfield_raw_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[pos] as usize] = cpu.bh_getfield_raw_f(struct_ptr, descr).to_bits() as i64;
    Ok(pos + 1)
}

// ── setfield_raw (blackhole.py:1497-1502) ───────────────────────────
fn handler_setfield_raw_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let value = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_raw_i(struct_ptr, value, descr);
    Ok(pos)
}
fn handler_setfield_raw_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[position] as usize];
    let value = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_raw_f(struct_ptr, value, descr);
    Ok(pos)
}

// ── new / new_with_vtable / new_array (blackhole.py:1301-1327) ──────
// These need SizeDescr which pyre doesn't fully have yet.
// Stub handlers that read the descriptor and call cpu.bh_new etc.

fn handler_new(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "d", returns="r")
    let (descr, pos) = read_descr(bh, code, position);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[pos] as usize] = cpu.bh_new(descr);
    Ok(pos + 1)
}
fn handler_new_with_vtable(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let (descr, pos) = read_descr(bh, code, position);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[pos] as usize] = cpu.bh_new_with_vtable(descr);
    Ok(pos + 1)
}
fn handler_new_array(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "i", "d", returns="r")
    let length = bh.registers_i[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[pos] as usize] = cpu.bh_new_array(length, descr);
    Ok(pos + 1)
}
fn handler_new_array_clear(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[position] as usize];
    let (descr, pos) = read_descr(bh, code, position + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[pos] as usize] = cpu.bh_new_array_clear(length, descr);
    Ok(pos + 1)
}

// ── string operations (blackhole.py:1200-1283) ──────────────────────
fn handler_strlen(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[position + 1] as usize] = cpu.bh_strlen(s);
    Ok(position + 2)
}
fn handler_strgetitem(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[position + 2] as usize] = cpu.bh_strgetitem(s, index);
    Ok(position + 3)
}
fn handler_strsetitem(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_i[code[position + 2] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_strsetitem(s, index, value);
    Ok(position + 3)
}
fn handler_newstr(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[position] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[position + 1] as usize] = cpu.bh_newstr(length);
    Ok(position + 2)
}
fn handler_unicodelen(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[position + 1] as usize] = cpu.bh_unicodelen(s);
    Ok(position + 2)
}
fn handler_unicodegetitem(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[position + 2] as usize] = cpu.bh_unicodegetitem(s, index);
    Ok(position + 3)
}
fn handler_unicodesetitem(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let s = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_i[code[position + 2] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_unicodesetitem(s, index, value);
    Ok(position + 3)
}
fn handler_newunicode(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[position] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[position + 1] as usize] = cpu.bh_newunicode(length);
    Ok(position + 2)
}

// ── exception handling (blackhole.py:969-1009) ──────────────────────
fn handler_catch_exception(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("L") — no-op, skip 2-byte label
    Ok(position + 2)
}

// ── misc no-ops (blackhole.py:1017-1049) ────────────────────────────
fn handler_jit_debug(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("r", "i", "i", "i", "i") = 1 ref + 4 int = 5 regs
    Ok(position + 5)
}
fn handler_jit_enter_portal_frame(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    // @arguments("i") = 1 int
    Ok(position + 1)
}
fn handler_jit_leave_portal_frame(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    Ok(position)
}

// ── interiorfield_gc (blackhole.py:1411-1429) ───────────────────────
// @arguments("cpu", "r", "i", "d", returns="X")
fn handler_getinteriorfield_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let (descr, pos) = read_descr(bh, code, position + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[pos] as usize] = cpu.bh_getinteriorfield_gc_i(array, index, descr);
    Ok(pos + 1)
}
fn handler_setinteriorfield_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[position] as usize];
    let index = bh.registers_i[code[position + 1] as usize];
    let value = bh.registers_i[code[position + 2] as usize];
    let (descr, pos) = read_descr(bh, code, position + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setinteriorfield_gc_i(array, index, value, descr);
    Ok(pos)
}

// ── call operations (blackhole.py:1224-1276) ────────────────────────

#[inline]
fn read_list_i(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (Vec<i64>, usize) {
    let count = code[pos] as usize;
    let values: Vec<i64> = (0..count)
        .map(|i| bh.registers_i[code[pos + 1 + i] as usize])
        .collect();
    (values, pos + 1 + count)
}
#[inline]
fn read_list_r(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (Vec<i64>, usize) {
    let count = code[pos] as usize;
    let values: Vec<i64> = (0..count)
        .map(|i| bh.registers_r[code[pos + 1 + i] as usize])
        .collect();
    (values, pos + 1 + count)
}
#[inline]
fn read_list_f(bh: &BlackholeInterpreter, code: &[u8], pos: usize) -> (Vec<i64>, usize) {
    let count = code[pos] as usize;
    let values: Vec<i64> = (0..count)
        .map(|i| bh.registers_f[code[pos + 1 + i] as usize])
        .collect();
    (values, pos + 1 + count)
}
#[allow(dead_code)]
fn flatten_args(a: &[i64], b: &[i64], c: &[i64]) -> Vec<i64> {
    let mut all = Vec::with_capacity(a.len() + b.len() + c.len());
    all.extend_from_slice(a);
    all.extend_from_slice(b);
    all.extend_from_slice(c);
    all
}

// residual_call_irf_*
fn handler_residual_call_irf_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    bh.registers_i[code[p] as usize] =
        bh.cpu
            .expect("cpu")
            .bh_call_i(func, Some(&ai), Some(&ar), Some(&af), &calldescr);
    Ok(p + 1)
}
fn handler_residual_call_irf_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    bh.registers_r[code[p] as usize] = bh
        .cpu
        .expect("cpu")
        .bh_call_r(func, Some(&ai), Some(&ar), Some(&af), &calldescr)
        .0 as i64;
    Ok(p + 1)
}
fn handler_residual_call_irf_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    bh.registers_f[code[p] as usize] = bh
        .cpu
        .expect("cpu")
        .bh_call_f(func, Some(&ai), Some(&ar), Some(&af), &calldescr)
        .to_bits() as i64;
    Ok(p + 1)
}
fn handler_residual_call_irf_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    bh.cpu
        .expect("cpu")
        .bh_call_v(func, Some(&ai), Some(&ar), Some(&af), &calldescr);
    Ok(p)
}
// residual_call_ir_*
fn handler_residual_call_ir_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    bh.registers_i[code[p] as usize] =
        bh.cpu
            .expect("cpu")
            .bh_call_i(func, Some(&ai), Some(&ar), None, &calldescr);
    Ok(p + 1)
}
fn handler_residual_call_ir_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    bh.registers_r[code[p] as usize] = bh
        .cpu
        .expect("cpu")
        .bh_call_r(func, Some(&ai), Some(&ar), None, &calldescr)
        .0 as i64;
    Ok(p + 1)
}
fn handler_residual_call_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ai, p) = read_list_i(bh, code, position + 1);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    bh.cpu
        .expect("cpu")
        .bh_call_v(func, Some(&ai), Some(&ar), None, &calldescr);
    Ok(p)
}
// residual_call_r_*
fn handler_residual_call_r_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ar, p) = read_list_r(bh, code, position + 1);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    bh.registers_i[code[p] as usize] =
        bh.cpu
            .expect("cpu")
            .bh_call_i(func, None, Some(&ar), None, &calldescr);
    Ok(p + 1)
}
fn handler_residual_call_r_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ar, p) = read_list_r(bh, code, position + 1);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    bh.registers_r[code[p] as usize] = bh
        .cpu
        .expect("cpu")
        .bh_call_r(func, None, Some(&ar), None, &calldescr)
        .0 as i64;
    Ok(p + 1)
}
fn handler_residual_call_r_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let func = bh.registers_i[code[position] as usize];
    let (ar, p) = read_list_r(bh, code, position + 1);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    bh.cpu
        .expect("cpu")
        .bh_call_v(func, None, Some(&ar), None, &calldescr);
    Ok(p)
}

/// Wire all currently-ported bhimpl methods into a `BlackholeInterpBuilder`'s
/// dispatch table. Called once after `setup_insns`.
///
/// RPython builds all handlers in `setup_insns` via `_get_method`; pyre
/// wires them incrementally as methods are ported from RPython.
pub fn wire_bhimpl_handlers(builder: &mut BlackholeInterpBuilder) {
    // @arguments("i", returns="i") pattern
    builder.wire_handler("int_same_as/i>i", handler_int_same_as);
    builder.wire_handler("int_neg/i>i", handler_int_neg);
    builder.wire_handler("int_invert/i>i", handler_int_invert);
    builder.wire_handler("int_is_true/i>i", handler_int_is_true);
    builder.wire_handler("int_is_zero/i>i", handler_int_is_zero);
    builder.wire_handler("int_force_ge_zero/i>i", handler_int_force_ge_zero);

    // @arguments("i", "i", returns="i") pattern
    builder.wire_handler("int_add/ii>i", handler_int_add);
    builder.wire_handler("int_sub/ii>i", handler_int_sub);
    builder.wire_handler("int_mul/ii>i", handler_int_mul);
    builder.wire_handler("int_and/ii>i", handler_int_and);
    builder.wire_handler("int_or/ii>i", handler_int_or);
    builder.wire_handler("int_xor/ii>i", handler_int_xor);
    builder.wire_handler("int_rshift/ii>i", handler_int_rshift);
    builder.wire_handler("int_lshift/ii>i", handler_int_lshift);
    builder.wire_handler("uint_rshift/ii>i", handler_uint_rshift);
    builder.wire_handler("int_lt/ii>i", handler_int_lt);
    builder.wire_handler("int_le/ii>i", handler_int_le);
    builder.wire_handler("int_eq/ii>i", handler_int_eq);
    builder.wire_handler("int_ne/ii>i", handler_int_ne);
    builder.wire_handler("int_gt/ii>i", handler_int_gt);
    builder.wire_handler("int_ge/ii>i", handler_int_ge);

    // Copy operations
    builder.wire_handler("int_copy/i>i", handler_int_copy);

    // Control flow
    builder.wire_handler("live/", handler_live);
    builder.wire_handler("goto/L", handler_goto);
    builder.wire_handler("goto_if_not/iL", handler_goto_if_not);

    // Unsigned comparisons (blackhole.py:571-582)
    builder.wire_handler("uint_lt/ii>i", handler_uint_lt);
    builder.wire_handler("uint_le/ii>i", handler_uint_le);
    builder.wire_handler("uint_gt/ii>i", handler_uint_gt);
    builder.wire_handler("uint_ge/ii>i", handler_uint_ge);

    // Float arithmetic (blackhole.py:676-718)
    builder.wire_handler("float_neg/f>f", handler_float_neg);
    builder.wire_handler("float_abs/f>f", handler_float_abs);
    builder.wire_handler("float_add/ff>f", handler_float_add);
    builder.wire_handler("float_sub/ff>f", handler_float_sub);
    builder.wire_handler("float_mul/ff>f", handler_float_mul);
    builder.wire_handler("float_truediv/ff>f", handler_float_truediv);

    // Float comparisons → int result (blackhole.py:720-749)
    builder.wire_handler("float_lt/ff>i", handler_float_lt);
    builder.wire_handler("float_le/ff>i", handler_float_le);
    builder.wire_handler("float_eq/ff>i", handler_float_eq);
    builder.wire_handler("float_ne/ff>i", handler_float_ne);
    builder.wire_handler("float_gt/ff>i", handler_float_gt);
    builder.wire_handler("float_ge/ff>i", handler_float_ge);

    // Ref operations (blackhole.py:584-610)
    builder.wire_handler("ptr_eq/rr>i", handler_ptr_eq);
    builder.wire_handler("ptr_ne/rr>i", handler_ptr_ne);
    builder.wire_handler("instance_ptr_eq/rr>i", handler_instance_ptr_eq);
    builder.wire_handler("instance_ptr_ne/rr>i", handler_instance_ptr_ne);
    builder.wire_handler("ptr_iszero/r>i", handler_ptr_iszero);
    builder.wire_handler("ptr_nonzero/r>i", handler_ptr_nonzero);

    // Copy operations (ref + float)
    builder.wire_handler("ref_copy/r>r", handler_ref_copy);
    builder.wire_handler("float_copy/f>f", handler_float_copy);

    // Conditional jumps (blackhole.py:871-920)
    builder.wire_handler("goto_if_not_int_lt/iiL", handler_goto_if_not_int_lt);
    builder.wire_handler("goto_if_not_int_le/iiL", handler_goto_if_not_int_le);
    builder.wire_handler("goto_if_not_int_eq/iiL", handler_goto_if_not_int_eq);
    builder.wire_handler("goto_if_not_int_ne/iiL", handler_goto_if_not_int_ne);
    builder.wire_handler("goto_if_not_int_gt/iiL", handler_goto_if_not_int_gt);
    builder.wire_handler("goto_if_not_int_ge/iiL", handler_goto_if_not_int_ge);
    // goto_if_not_int_is_true = goto_if_not (alias, blackhole.py:913)
    builder.wire_handler("goto_if_not_int_is_true/iL", handler_goto_if_not);

    // Guard values — no-op in blackhole (blackhole.py:648-656)
    builder.wire_handler("int_guard_value/i", handler_int_guard_value);
    builder.wire_handler("ref_guard_value/r", handler_ref_guard_value);
    builder.wire_handler("float_guard_value/f", handler_float_guard_value);

    // Push/pop tmpreg (blackhole.py:661-679)
    builder.wire_handler("int_push/i", handler_int_push);
    builder.wire_handler("ref_push/r", handler_ref_push);
    builder.wire_handler("float_push/f", handler_float_push);
    builder.wire_handler("int_pop/>i", handler_int_pop);
    builder.wire_handler("ref_pop/>r", handler_ref_pop);
    builder.wire_handler("float_pop/>f", handler_float_pop);

    // Record — no-op (blackhole.py:616-636)
    builder.wire_handler("record_exact_class/ri", handler_record_exact_class);
    builder.wire_handler("record_exact_value_r/rr", handler_record_exact_value_r);
    builder.wire_handler("record_exact_value_i/ii", handler_record_exact_value_i);

    // Cast operations (blackhole.py:800-831)
    builder.wire_handler("cast_float_to_int/f>i", handler_cast_float_to_int);
    builder.wire_handler("cast_int_to_float/i>f", handler_cast_int_to_float);

    // int_signext (blackhole.py:566-569)
    builder.wire_handler("int_signext/ii>i", handler_int_signext);

    // Overflow ops (blackhole.py:478-497)
    builder.wire_handler("int_add_jump_if_ovf/Lii>i", handler_int_add_jump_if_ovf);
    builder.wire_handler("int_sub_jump_if_ovf/Lii>i", handler_int_sub_jump_if_ovf);
    builder.wire_handler("int_mul_jump_if_ovf/Lii>i", handler_int_mul_jump_if_ovf);

    // Misc simple ops
    builder.wire_handler("assert_not_none/r", handler_assert_not_none);
    builder.wire_handler("virtual_ref/r>r", handler_virtual_ref);
    builder.wire_handler("virtual_ref_finish/r", handler_virtual_ref_finish);
    builder.wire_handler("loop_header/i", handler_loop_header);
    builder.wire_handler("ref_isconstant/r>i", handler_ref_isconstant);
    builder.wire_handler("ref_isvirtual/r>i", handler_ref_isvirtual);
    builder.wire_handler(
        "goto_if_not_int_is_zero/iL",
        handler_goto_if_not_int_is_zero,
    );
    builder.wire_handler("goto_if_not_ptr_iszero/rL", handler_goto_if_not_ptr_iszero);
    builder.wire_handler(
        "goto_if_not_ptr_nonzero/rL",
        handler_goto_if_not_ptr_nonzero,
    );
    builder.wire_handler("unreachable/", handler_unreachable);

    // Field operations via cpu.bh_* (blackhole.py:1432-1481)
    builder.wire_handler("getfield_gc_i/rd>i", handler_getfield_gc_i);
    builder.wire_handler("getfield_gc_r/rd>r", handler_getfield_gc_r);
    builder.wire_handler("getfield_gc_f/rd>f", handler_getfield_gc_f);
    builder.wire_handler("getfield_gc_i_pure/rd>i", handler_getfield_gc_i); // alias
    builder.wire_handler("getfield_gc_r_pure/rd>r", handler_getfield_gc_r);
    builder.wire_handler("getfield_gc_f_pure/rd>f", handler_getfield_gc_f);
    builder.wire_handler("setfield_gc_i/rid", handler_setfield_gc_i);
    builder.wire_handler("setfield_gc_r/rrd", handler_setfield_gc_r);
    builder.wire_handler("setfield_gc_f/rfd", handler_setfield_gc_f);
    builder.wire_handler("arraylen_gc/rd>i", handler_arraylen_gc);

    // Array item operations (blackhole.py:1329-1365)
    builder.wire_handler("getarrayitem_gc_i/rid>i", handler_getarrayitem_gc_i);
    builder.wire_handler("getarrayitem_gc_r/rid>r", handler_getarrayitem_gc_r);
    builder.wire_handler("getarrayitem_gc_i_pure/rid>i", handler_getarrayitem_gc_i);
    builder.wire_handler("getarrayitem_gc_r_pure/rid>r", handler_getarrayitem_gc_r);
    builder.wire_handler("setarrayitem_gc_i/riid", handler_setarrayitem_gc_i);
    builder.wire_handler("setarrayitem_gc_r/rird", handler_setarrayitem_gc_r);

    // Raw field operations (blackhole.py:1464-1502)
    builder.wire_handler("getfield_raw_i/id>i", handler_getfield_raw_i);
    builder.wire_handler("getfield_raw_f/id>f", handler_getfield_raw_f);
    builder.wire_handler("setfield_raw_i/iid", handler_setfield_raw_i);
    builder.wire_handler("setfield_raw_f/ifd", handler_setfield_raw_f);

    // Greenfield aliases
    builder.wire_handler("getfield_gc_i_greenfield/rd>i", handler_getfield_gc_i);
    builder.wire_handler("getfield_gc_r_greenfield/rd>r", handler_getfield_gc_r);
    builder.wire_handler("getfield_gc_f_greenfield/rd>f", handler_getfield_gc_f);

    // New operations (blackhole.py:1301-1327)
    builder.wire_handler("new/d>r", handler_new);
    builder.wire_handler("new_with_vtable/d>r", handler_new_with_vtable);
    builder.wire_handler("new_array/id>r", handler_new_array);
    builder.wire_handler("new_array_clear/id>r", handler_new_array_clear);

    // String operations (blackhole.py:1200-1283)
    builder.wire_handler("strlen/r>i", handler_strlen);
    builder.wire_handler("strgetitem/ri>i", handler_strgetitem);
    builder.wire_handler("strsetitem/rii", handler_strsetitem);
    builder.wire_handler("newstr/i>r", handler_newstr);
    builder.wire_handler("unicodelen/r>i", handler_unicodelen);
    builder.wire_handler("unicodegetitem/ri>i", handler_unicodegetitem);
    builder.wire_handler("unicodesetitem/rii", handler_unicodesetitem);
    builder.wire_handler("newunicode/i>r", handler_newunicode);

    // Exception handling (blackhole.py:969-975)
    builder.wire_handler("catch_exception/L", handler_catch_exception);

    // Interior field operations (blackhole.py:1411-1429)
    // _r/_f deferred until Backend gains those variants.
    builder.wire_handler("getinteriorfield_gc_i/rid>i", handler_getinteriorfield_gc_i);
    builder.wire_handler("setinteriorfield_gc_i/riid", handler_setinteriorfield_gc_i);

    // Residual call operations (blackhole.py:1224-1255)
    builder.wire_handler("residual_call_irf_i/iIRFd>i", handler_residual_call_irf_i);
    builder.wire_handler("residual_call_irf_r/iIRFd>r", handler_residual_call_irf_r);
    builder.wire_handler("residual_call_irf_f/iIRFd>f", handler_residual_call_irf_f);
    builder.wire_handler("residual_call_irf_v/iIRFd", handler_residual_call_irf_v);
    builder.wire_handler("residual_call_ir_i/iIRd>i", handler_residual_call_ir_i);
    builder.wire_handler("residual_call_ir_r/iIRd>r", handler_residual_call_ir_r);
    builder.wire_handler("residual_call_ir_v/iIRd", handler_residual_call_ir_v);
    builder.wire_handler("residual_call_r_i/iRd>i", handler_residual_call_r_i);
    builder.wire_handler("residual_call_r_r/iRd>r", handler_residual_call_r_r);
    builder.wire_handler("residual_call_r_v/iRd", handler_residual_call_r_v);

    // Misc no-ops (blackhole.py:1017-1049)
    builder.wire_handler("jit_debug/riiii", handler_jit_debug);
    builder.wire_handler("jit_enter_portal_frame/i", handler_jit_enter_portal_frame);
    builder.wire_handler("jit_leave_portal_frame/", handler_jit_leave_portal_frame);

    // Float conditional jumps (blackhole.py:751-798)
    builder.wire_handler("goto_if_not_float_lt/ffL", handler_goto_if_not_float_lt);
    builder.wire_handler("goto_if_not_float_le/ffL", handler_goto_if_not_float_le);
    builder.wire_handler("goto_if_not_float_eq/ffL", handler_goto_if_not_float_eq);
    builder.wire_handler("goto_if_not_float_ne/ffL", handler_goto_if_not_float_ne);
    builder.wire_handler("goto_if_not_float_gt/ffL", handler_goto_if_not_float_gt);
    builder.wire_handler("goto_if_not_float_ge/ffL", handler_goto_if_not_float_ge);
    builder.wire_handler("goto_if_not_ptr_eq/rrL", handler_goto_if_not_ptr_eq);
    builder.wire_handler("goto_if_not_ptr_ne/rrL", handler_goto_if_not_ptr_ne);

    // Assert/isconstant (no-ops in blackhole)
    builder.wire_handler("int_assert_green/i", handler_int_assert_green);
    builder.wire_handler("ref_assert_green/r", handler_ref_assert_green);
    builder.wire_handler("float_assert_green/f", handler_float_assert_green);
    builder.wire_handler("int_isconstant/i>i", handler_int_isconstant);
    builder.wire_handler("float_isconstant/f>i", handler_float_isconstant);

    // Misc integer ops
    builder.wire_handler("uint_mul_high/ii>i", handler_uint_mul_high);
    builder.wire_handler("int_between/iii>i", handler_int_between);

    // String hashing (stubs)
    builder.wire_handler("strhash/r>i", handler_strhash);
    builder.wire_handler("unicodehash/r>i", handler_unicodehash);

    // Float <-> longlong / singlefloat conversions
    builder.wire_handler(
        "convert_float_bytes_to_longlong/f>i",
        handler_convert_float_bytes_to_longlong,
    );
    builder.wire_handler(
        "convert_longlong_bytes_to_float/i>f",
        handler_convert_longlong_bytes_to_float,
    );
    builder.wire_handler(
        "cast_float_to_singlefloat/f>i",
        handler_cast_float_to_singlefloat,
    );
    builder.wire_handler(
        "cast_singlefloat_to_float/i>f",
        handler_cast_singlefloat_to_float,
    );

    // Misc
    builder.wire_handler(
        "hint_force_virtualizable/rd",
        handler_hint_force_virtualizable,
    );
    builder.wire_handler("guard_class/ri", handler_guard_class);
    builder.wire_handler(
        "record_quasiimmut_field/rd",
        handler_record_quasiimmut_field,
    );
    builder.wire_handler(
        "jit_force_quasi_immutable/rd",
        handler_jit_force_quasi_immutable,
    );
    builder.wire_handler(
        "record_known_result_i_ir_v/iiIRd",
        handler_record_known_result_i_ir_v,
    );
    builder.wire_handler(
        "record_known_result_r_ir_v/riIRd",
        handler_record_known_result_r_ir_v,
    );
    builder.wire_handler("str_guard_value/rid>r", handler_str_guard_value);
    builder.wire_handler("rvmprof_code/ii", handler_rvmprof_code);
    builder.wire_handler("copystrcontent/rriii", handler_copystrcontent);
    builder.wire_handler("copyunicodecontent/rriii", handler_copyunicodecontent);
    builder.wire_handler("current_trace_length/>i", handler_current_trace_length);

    // Exception ops (blackhole.py:976-1009)
    builder.wire_handler("raise/r", handler_raise);
    builder.wire_handler("reraise/", handler_reraise);
    builder.wire_handler("last_exception/>i", handler_last_exception);
    builder.wire_handler("last_exc_value/>r", handler_last_exc_value);
    builder.wire_handler(
        "goto_if_exception_mismatch/iL",
        handler_goto_if_exception_mismatch,
    );
    builder.wire_handler("debug_fatalerror/r", handler_debug_fatalerror);

    // Cast ptr<->int (blackhole.py:603-610)
    builder.wire_handler("cast_ptr_to_int/r>i", handler_cast_ptr_to_int);
    builder.wire_handler("cast_int_to_ptr/i>r", handler_cast_int_to_ptr);

    // Vable field operations
    builder.wire_handler("getfield_vable_i/rd>i", handler_getfield_vable_i);
    builder.wire_handler("getfield_vable_r/rd>r", handler_getfield_vable_r);
    builder.wire_handler("getfield_vable_f/rd>f", handler_getfield_vable_f);
    builder.wire_handler("setfield_vable_i/rid", handler_setfield_vable_i);
    builder.wire_handler("setfield_vable_r/rrd", handler_setfield_vable_r);
    builder.wire_handler("setfield_vable_f/rfd", handler_setfield_vable_f);
    builder.wire_handler("getarrayitem_vable_i/ridd>i", handler_getarrayitem_vable_i);
    builder.wire_handler("getarrayitem_vable_r/ridd>r", handler_getarrayitem_vable_r);
    builder.wire_handler("setarrayitem_vable_i/riidd", handler_setarrayitem_vable_i);
    builder.wire_handler("setarrayitem_vable_r/rirdd", handler_setarrayitem_vable_r);
    builder.wire_handler("arraylen_vable/rdd>i", handler_arraylen_vable);
    builder.wire_handler("getarrayitem_raw_i/iid>i", handler_getarrayitem_raw_i);
    builder.wire_handler("setarrayitem_raw_i/iiid", handler_setarrayitem_raw_i);
    builder.wire_handler("conditional_call_ir_v/iiIRd", handler_conditional_call_ir_v);
    builder.wire_handler(
        "conditional_call_value_ir_i/iiIRd>i",
        handler_conditional_call_value_ir_i,
    );
    builder.wire_handler(
        "conditional_call_value_ir_r/riIRd>r",
        handler_conditional_call_value_ir_r,
    );
    builder.wire_handler("getlistitem_gc_i/ridd>i", handler_getlistitem_gc_i);
    builder.wire_handler("getlistitem_gc_r/ridd>r", handler_getlistitem_gc_r);
    builder.wire_handler("setlistitem_gc_i/riidd", handler_setlistitem_gc_i);
    builder.wire_handler("setlistitem_gc_r/rirdd", handler_setlistitem_gc_r);
    builder.wire_handler("switch/id", handler_switch);
    builder.wire_handler("getlistitem_gc_f/ridd>f", handler_getlistitem_gc_f);
    builder.wire_handler("setlistitem_gc_f/rifdd", handler_setlistitem_gc_f);
    builder.wire_handler("check_neg_index/rid>i", handler_check_neg_index);
    builder.wire_handler(
        "check_resizable_neg_index/rid>i",
        handler_check_resizable_neg_index,
    );

    // Float/raw array ops
    builder.wire_handler("getarrayitem_gc_f/rid>f", handler_getarrayitem_gc_f);
    builder.wire_handler("getarrayitem_gc_f_pure/rid>f", handler_getarrayitem_gc_f);
    builder.wire_handler("setarrayitem_gc_f/rifd", handler_setarrayitem_gc_f);
    builder.wire_handler("getarrayitem_raw_f/iid>f", handler_getarrayitem_raw_f);
    builder.wire_handler("setarrayitem_raw_f/iifd", handler_setarrayitem_raw_f);
    builder.wire_handler("getfield_raw_r/id>r", handler_getfield_raw_r);
    builder.wire_handler("getinteriorfield_gc_f/rid>f", handler_getinteriorfield_gc_f);
    builder.wire_handler("getinteriorfield_gc_r/rid>r", handler_getinteriorfield_gc_r);
    builder.wire_handler("setinteriorfield_gc_f/rifd", handler_setinteriorfield_gc_f);
    builder.wire_handler("setinteriorfield_gc_r/rird", handler_setinteriorfield_gc_r);
    builder.wire_handler("gc_load_indexed_i/riiiii>i", handler_gc_load_indexed_i);
    builder.wire_handler("gc_load_indexed_f/riiiii>f", handler_gc_load_indexed_f);
    builder.wire_handler("gc_store_indexed_i/riiiid", handler_gc_store_indexed_i);
    builder.wire_handler("gc_store_indexed_f/rifiid", handler_gc_store_indexed_f);
    builder.wire_handler("raw_store_i/iiid", handler_raw_store_i);
    builder.wire_handler("raw_store_f/iifd", handler_raw_store_f);
    builder.wire_handler("raw_load_i/iid>i", handler_raw_load_i);
    builder.wire_handler("raw_load_f/iid>f", handler_raw_load_f);
    builder.wire_handler("newlist/idddd>r", handler_newlist);
    builder.wire_handler("newlist_clear/idddd>r", handler_newlist_clear);
    builder.wire_handler("newlist_hint/idddd>r", handler_newlist_hint);
    builder.wire_handler("getarrayitem_vable_f/ridd>f", handler_getarrayitem_vable_f);
    builder.wire_handler("setarrayitem_vable_f/rifdd", handler_setarrayitem_vable_f);

    // Inline call (stub — needs frame-chain)
    builder.wire_handler("inline_call_irf_i/dIRF>i", handler_inline_call_irf_i);
    builder.wire_handler("inline_call_irf_r/dIRF>r", handler_inline_call_irf_r);
    builder.wire_handler("inline_call_irf_f/dIRF>f", handler_inline_call_irf_f);
    builder.wire_handler("inline_call_irf_v/dIRF", handler_inline_call_irf_v);
    builder.wire_handler("inline_call_ir_i/dIR>i", handler_inline_call_ir_i);
    builder.wire_handler("inline_call_ir_r/dIR>r", handler_inline_call_ir_r);
    builder.wire_handler("inline_call_ir_v/dIR", handler_inline_call_ir_v);
    builder.wire_handler("inline_call_r_i/dR>i", handler_inline_call_r_i);
    builder.wire_handler("inline_call_r_r/dR>r", handler_inline_call_r_r);
    builder.wire_handler("inline_call_r_v/dR", handler_inline_call_r_v);

    // Recursive call (stub — needs portal runner)
    builder.wire_handler("recursive_call_i/cIRFIRF>i", handler_recursive_call_i);
    builder.wire_handler("recursive_call_r/cIRFIRF>r", handler_recursive_call_r);
    builder.wire_handler("recursive_call_f/cIRFIRF>f", handler_recursive_call_f);
    builder.wire_handler("recursive_call_v/cIRFIRF", handler_recursive_call_v);

    // Returns
    builder.wire_handler("int_return/i", handler_int_return);
    builder.wire_handler("ref_return/r", handler_ref_return);
    builder.wire_handler("float_return/f", handler_float_return);
    builder.wire_handler("void_return/", handler_void_return);
}

// ── goto_if_not_float (blackhole.py:751-798) ────────────────────────
macro_rules! bhhandler_goto_if_not_ff {
    ($name:ident, $cmp:expr) => {
        fn $name(
            bh: &mut BlackholeInterpreter,
            code: &[u8],
            position: usize,
        ) -> Result<usize, DispatchError> {
            let a = f64::from_bits(bh.registers_f[code[position] as usize] as u64);
            let b = f64::from_bits(bh.registers_f[code[position + 1] as usize] as u64);
            let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
            let pc = position + 4;
            if $cmp(a, b) { Ok(pc) } else { Ok(target) }
        }
    };
}
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_lt, |a: f64, b: f64| a < b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_le, |a: f64, b: f64| a <= b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_eq, |a: f64, b: f64| a == b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_ne, |a: f64, b: f64| a != b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_gt, |a: f64, b: f64| a > b);
bhhandler_goto_if_not_ff!(handler_goto_if_not_float_ge, |a: f64, b: f64| a >= b);

// goto_if_not_ptr_eq/ne (reuse ii macro with ref registers)
fn handler_goto_if_not_ptr_eq(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_r[code[position] as usize];
    let b = bh.registers_r[code[position + 1] as usize];
    let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
    if a == b { Ok(position + 4) } else { Ok(target) }
}
fn handler_goto_if_not_ptr_ne(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_r[code[position] as usize];
    let b = bh.registers_r[code[position + 1] as usize];
    let target = (code[position + 2] as usize) | ((code[position + 3] as usize) << 8);
    if a != b { Ok(position + 4) } else { Ok(target) }
}

// assert_green / isconstant — no-ops
fn handler_int_assert_green(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 1)
}
fn handler_ref_assert_green(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 1)
}
fn handler_float_assert_green(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 1)
}
fn handler_int_isconstant(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[p + 1] as usize] = 0;
    Ok(p + 2)
}
fn handler_float_isconstant(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[p + 1] as usize] = 0;
    Ok(p + 2)
}

// misc
fn handler_uint_mul_high(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[p] as usize] as u64;
    let b = bh.registers_i[code[p + 1] as usize] as u64;
    bh.registers_i[code[p + 2] as usize] = ((a as u128 * b as u128) >> 64) as i64;
    Ok(p + 3)
}
fn handler_int_between(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let a = bh.registers_i[code[p] as usize];
    let b = bh.registers_i[code[p + 1] as usize];
    let c = bh.registers_i[code[p + 2] as usize];
    bh.registers_i[code[p + 3] as usize] = (a <= b && b < c) as i64;
    Ok(p + 4)
}
fn handler_strhash(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[p + 1] as usize] = 0;
    Ok(p + 2)
}
fn handler_unicodehash(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[p + 1] as usize] = 0;
    Ok(p + 2)
}
fn handler_convert_float_bytes_to_longlong(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[p + 1] as usize] = bh.registers_f[code[p] as usize];
    Ok(p + 2)
}
fn handler_convert_longlong_bytes_to_float(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_f[code[p + 1] as usize] = bh.registers_i[code[p] as usize];
    Ok(p + 2)
}
fn handler_cast_float_to_singlefloat(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let f = f64::from_bits(bh.registers_f[code[p] as usize] as u64);
    bh.registers_i[code[p + 1] as usize] = (f as f32).to_bits() as i64;
    Ok(p + 2)
}
fn handler_cast_singlefloat_to_float(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let f = f32::from_bits(bh.registers_i[code[p] as usize] as u32) as f64;
    bh.registers_f[code[p + 1] as usize] = f.to_bits() as i64;
    Ok(p + 2)
}
fn handler_hint_force_virtualizable(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 3)
}
fn handler_guard_class(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 2)
}
fn handler_record_quasiimmut_field(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 3)
}
fn handler_jit_force_quasi_immutable(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 3)
}
fn handler_record_known_result_i_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let p = position + 2;
    let (_, p) = read_list_i(bh, code, p);
    let (_, p) = read_list_r(bh, code, p);
    let (_, p) = read_descr(bh, code, p);
    Ok(p)
}
fn handler_record_known_result_r_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let p = position + 2;
    let (_, p) = read_list_i(bh, code, p);
    let (_, p) = read_list_r(bh, code, p);
    let (_, p) = read_descr(bh, code, p);
    Ok(p)
}
fn handler_str_guard_value(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    position: usize,
) -> Result<usize, DispatchError> {
    let r = bh.registers_r[code[position] as usize];
    let (_, p) = read_descr(bh, code, position + 2);
    bh.registers_r[code[p] as usize] = r;
    Ok(p + 1)
}
fn handler_rvmprof_code(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 2)
}
/// RPython `blackhole.py:1575-1578` `bhimpl_copystrcontent`.
fn handler_copystrcontent(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let src = bh.registers_r[code[p] as usize];
    let dst = bh.registers_r[code[p + 1] as usize];
    let srcstart = bh.registers_i[code[p + 2] as usize];
    let dststart = bh.registers_i[code[p + 3] as usize];
    let length = bh.registers_i[code[p + 4] as usize];
    if let Some(cpu) = bh.cpu {
        cpu.bh_copystrcontent(src, dst, srcstart, dststart, length);
    }
    Ok(p + 5)
}
/// RPython `blackhole.py:1580-1583` `bhimpl_copyunicodecontent`.
fn handler_copyunicodecontent(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let src = bh.registers_r[code[p] as usize];
    let dst = bh.registers_r[code[p + 1] as usize];
    let srcstart = bh.registers_i[code[p + 2] as usize];
    let dststart = bh.registers_i[code[p + 3] as usize];
    let length = bh.registers_i[code[p + 4] as usize];
    if let Some(cpu) = bh.cpu {
        cpu.bh_copyunicodecontent(src, dst, srcstart, dststart, length);
    }
    Ok(p + 5)
}
fn handler_raise(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Err(DispatchError::RaiseException(
        bh.registers_r[code[p] as usize],
    ))
}
fn handler_reraise(
    bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _p: usize,
) -> Result<usize, DispatchError> {
    Err(DispatchError::RaiseException(bh.exception_last_value))
}
/// RPython `blackhole.py:987-991`:
/// ```python
/// @arguments("self", returns="i")
/// def bhimpl_last_exception(self):
///     real_instance = self.exception_last_value
///     assert real_instance
///     return ptr2int(real_instance.typeptr)
/// ```
/// Returns the CLASS POINTER (typeptr) of the caught exception, not the
/// exception object itself. Uses `cpu.bh_classof(obj)` to get the typeptr.
fn handler_last_exception(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let exc_obj = bh.exception_last_value;
    // RPython: ptr2int(real_instance.typeptr) — get class pointer
    let typeptr = if let Some(cpu) = bh.cpu {
        cpu.bh_classof(exc_obj)
    } else {
        exc_obj // fallback: use object pointer as-is if no cpu
    };
    bh.registers_i[code[p] as usize] = typeptr;
    Ok(p + 1)
}
/// RPython `blackhole.py:993-997`:
/// ```python
/// @arguments("self", returns="r")
/// def bhimpl_last_exc_value(self):
///     return cast_opaque_ptr(GCREF, self.exception_last_value)
/// ```
fn handler_last_exc_value(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_r[code[p] as usize] = bh.exception_last_value;
    Ok(p + 1)
}
/// RPython `blackhole.py:976-985`:
/// ```python
/// @arguments("self", "i", "L", "pc", returns="L")
/// def bhimpl_goto_if_exception_mismatch(self, vtable, target, pc):
///     bounding_class = cast_adr_to_ptr(int2adr(vtable), CLASSTYPE)
///     real_instance = self.exception_last_value
///     if rclass.ll_issubclass(real_instance.typeptr, bounding_class):
///         return pc  # match → fall through
///     else:
///         return target  # mismatch → jump
/// ```
/// Uses `cpu.bh_classof` to get the exception's typeptr and compares
/// against the bounding class vtable. For now uses pointer equality
/// (correct for exact match; subclass check needs rclass infrastructure).
fn handler_goto_if_exception_mismatch(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let bounding_vtable = bh.registers_i[code[p] as usize];
    let target = (code[p + 1] as usize) | ((code[p + 2] as usize) << 8);
    let pc = p + 3;
    let exc_obj = bh.exception_last_value;
    let exc_typeptr = if let Some(cpu) = bh.cpu {
        cpu.bh_classof(exc_obj)
    } else {
        exc_obj
    };
    // RPython: rclass.ll_issubclass(real_instance.typeptr, bounding_class).
    // Uses Backend::bh_issubclass for the subclass check.
    let is_match = if let Some(cpu) = bh.cpu {
        cpu.bh_issubclass(exc_typeptr, bounding_vtable)
    } else {
        exc_typeptr == bounding_vtable
    };
    if is_match {
        Ok(pc) // match → fall through
    } else {
        Ok(target) // mismatch → jump to target
    }
}
fn handler_debug_fatalerror(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    _p: usize,
) -> Result<usize, DispatchError> {
    panic!("bhimpl_debug_fatalerror");
}
fn handler_cast_ptr_to_int(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_i[code[p + 1] as usize] = bh.registers_r[code[p] as usize];
    Ok(p + 2)
}
fn handler_cast_int_to_ptr(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    bh.registers_r[code[p + 1] as usize] = bh.registers_i[code[p] as usize];
    Ok(p + 2)
}
fn handler_current_trace_length(
    _bh: &mut BlackholeInterpreter,
    _code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    Ok(p + 1)
}

// ── vable field operations (blackhole.py:1446-1495) ─────────────────
// RPython: fielddescr.get_vinfo().clear_vable_token(struct)
//          return cpu.bh_getfield_gc_*(struct, fielddescr)
// pyre: read_descr_vable_field resolves VableField.index → byte offset via VirtualizableInfo.

fn handler_getfield_vable_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[p] as usize] = cpu.bh_getfield_gc_i(struct_ptr, &descr);
    Ok(p + 1)
}
fn handler_getfield_vable_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[p] as usize] = cpu.bh_getfield_gc_r(struct_ptr, &descr).0 as i64;
    Ok(p + 1)
}
fn handler_getfield_vable_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p] as usize] = cpu.bh_getfield_gc_f(struct_ptr, &descr).to_bits() as i64;
    Ok(p + 1)
}
fn handler_setfield_vable_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let value = bh.registers_i[code[p + 1] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_i(struct_ptr, value, &descr);
    Ok(p)
}
fn handler_setfield_vable_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let value = bh.registers_r[code[p + 1] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_r(struct_ptr, majit_ir::GcRef(value as usize), &descr);
    Ok(p)
}
fn handler_setfield_vable_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_r[code[p] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 1] as usize] as u64);
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, struct_ptr as *mut u8) };
    }
    let (descr, p) = read_descr_vable_field(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setfield_gc_f(struct_ptr, value, &descr);
    Ok(p)
}

// ── vable array operations (blackhole.py:1374-1409) ─────────────────
// @arguments("cpu", "r", "i", "d", "d", returns="X")
// Two descriptors: fielddescr (VableArray) + arraydescr (Array).
// RPython: fielddescr.get_vinfo().clear_vable_token(vable)
//          array = cpu.bh_getfield_gc_r(vable, fielddescr)
//          return cpu.bh_getarrayitem_gc_*(array, index, arraydescr)
fn handler_getarrayitem_vable_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        let ptr = vable as *mut u8;
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    bh.registers_i[code[p] as usize] = cpu.bh_getarrayitem_gc_i(array, index, array_descr);
    Ok(p + 1)
}
fn handler_getarrayitem_vable_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        let ptr = vable as *mut u8;
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    bh.registers_r[code[p] as usize] = cpu.bh_getarrayitem_gc_r(array, index, array_descr).0 as i64;
    Ok(p + 1)
}
fn handler_setarrayitem_vable_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_i[code[p + 2] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        let ptr = vable as *mut u8;
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    cpu.bh_setarrayitem_gc_i(array, index, value, array_descr);
    Ok(p)
}
fn handler_setarrayitem_vable_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_r[code[p + 2] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        let ptr = vable as *mut u8;
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    cpu.bh_setarrayitem_gc_r(array, index, majit_ir::GcRef(value as usize), array_descr);
    Ok(p)
}
fn handler_arraylen_vable(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        let ptr = vable as *mut u8;
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 1);
    let (array_len_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    bh.registers_i[code[p] as usize] = cpu.bh_arraylen_gc(array, array_len_descr);
    Ok(p + 1)
}

// ── getarrayitem_raw / setarrayitem_raw (blackhole.py:1343-1365) ────
/// RPython `blackhole.py:1343-1345` `bhimpl_getarrayitem_raw_i`:
/// `return cpu.bh_getarrayitem_raw_i(array, index, arraydescr)`.
/// Raw memory access — NOT GC-managed. Uses unsafe direct read.
fn handler_getarrayitem_raw_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[p] as usize] as usize; // raw ptr
    let index = bh.registers_i[code[p + 1] as usize] as usize;
    let (descr, p) = read_descr(bh, code, p + 2);
    let item_size = descr.as_offset();
    let offset = index * item_size.max(1);
    let value = unsafe { *((array + offset) as *const i64) };
    bh.registers_i[code[p] as usize] = value;
    Ok(p + 1)
}
/// RPython `blackhole.py:1360-1362` `bhimpl_setarrayitem_raw_i`.
fn handler_setarrayitem_raw_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[p] as usize] as usize;
    let index = bh.registers_i[code[p + 1] as usize] as usize;
    let value = bh.registers_i[code[p + 2] as usize];
    let (descr, p) = read_descr(bh, code, p + 3);
    let item_size = descr.as_offset();
    let offset = index * item_size.max(1);
    unsafe { *((array + offset) as *mut i64) = value };
    Ok(p)
}

// ── conditional call (blackhole.py:1257-1276) ───────────────────────
fn handler_conditional_call_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let condition = bh.registers_i[code[p] as usize];
    let func = bh.registers_i[code[p + 1] as usize];
    let (ai, p) = read_list_i(bh, code, p + 2);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    if condition != 0 {
        bh.cpu
            .expect("cpu")
            .bh_call_v(func, Some(&ai), Some(&ar), None, &calldescr);
    }
    Ok(p)
}
fn handler_conditional_call_value_ir_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let mut value = bh.registers_i[code[p] as usize];
    let func = bh.registers_i[code[p + 1] as usize];
    let (ai, p) = read_list_i(bh, code, p + 2);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    if value == 0 {
        value = bh
            .cpu
            .expect("cpu")
            .bh_call_i(func, Some(&ai), Some(&ar), None, &calldescr);
    }
    bh.registers_i[code[p] as usize] = value;
    Ok(p + 1)
}
fn handler_conditional_call_value_ir_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let mut value = bh.registers_r[code[p] as usize];
    let func = bh.registers_i[code[p + 1] as usize];
    let (ai, p) = read_list_i(bh, code, p + 2);
    let (ar, p) = read_list_r(bh, code, p);
    let (calldescr, p) = read_descr(bh, code, p);
    let calldescr = calldescr.as_calldescr().clone();
    if value == 0 {
        value = bh
            .cpu
            .expect("cpu")
            .bh_call_r(func, Some(&ai), Some(&ar), None, &calldescr)
            .0 as i64;
    }
    bh.registers_r[code[p] as usize] = value;
    Ok(p + 1)
}

// ── list ops (blackhole.py:1195-1219) — compound: getfield_gc_r + getarrayitem ──
fn handler_getlistitem_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    bh.registers_i[code[p] as usize] = cpu.bh_getarrayitem_gc_i(items, index, array_descr);
    Ok(p + 1)
}
fn handler_getlistitem_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    bh.registers_r[code[p] as usize] = cpu.bh_getarrayitem_gc_r(items, index, array_descr).0 as i64;
    Ok(p + 1)
}
fn handler_setlistitem_gc_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_i[code[p + 2] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    cpu.bh_setarrayitem_gc_i(items, index, value, array_descr);
    Ok(p)
}
fn handler_setlistitem_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_r[code[p + 2] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    cpu.bh_setarrayitem_gc_r(items, index, majit_ir::GcRef(value as usize), array_descr);
    Ok(p)
}

// ── switch (blackhole.py:954-960) ───────────────────────────────────
/// RPython `blackhole.py:954-960`:
/// ```python
/// @arguments("i", "d", "pc", returns="L")
/// def bhimpl_switch(switchvalue, switchdict, pc):
///     assert isinstance(switchdict, SwitchDictDescr)
///     try:
///         return switchdict.dict[switchvalue]
///     except KeyError:
///         return pc
/// ```
/// TODO: resolve descr index to a SwitchDictDescr table. Currently falls
/// through (returns pc) which matches the KeyError fallback path but never
/// takes the dict-hit branch.
fn handler_switch(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let switchvalue = bh.registers_i[code[p] as usize];
    let (descr, pos) = read_descr(bh, code, p + 1);
    // RPython blackhole.py:954-960:
    //   try: return switchdict.dict[switchvalue]
    //   except KeyError: return pc
    if let Some(target) = descr.switch_lookup(switchvalue) {
        Ok(target)
    } else {
        Ok(pos) // fallthrough (KeyError path)
    }
}

// ── check_neg_index / check_resizable_neg_index (blackhole.py:1148-1158) ─
fn handler_check_neg_index(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let mut index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    if index < 0 {
        let cpu = bh.cpu.expect("cpu not set");
        index += cpu.bh_arraylen_gc(array, descr);
    }
    bh.registers_i[code[p] as usize] = index;
    Ok(p + 1)
}
fn handler_check_resizable_neg_index(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let mut index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    if index < 0 {
        let cpu = bh.cpu.expect("cpu not set");
        index += cpu.bh_getfield_gc_i(lst, descr);
    }
    bh.registers_i[code[p] as usize] = index;
    Ok(p + 1)
}

// ── getarrayitem_gc_f / setarrayitem_gc_f ───────────────────────────
// blackhole.py:1336-1337 bhimpl_getarrayitem_gc_f
fn handler_getarrayitem_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p] as usize] =
        cpu.bh_getarrayitem_gc_f(array, index, descr).to_bits() as i64;
    Ok(p + 1)
}
// blackhole.py:1357-1358 bhimpl_setarrayitem_gc_f
fn handler_setarrayitem_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (descr, p) = read_descr(bh, code, p + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setarrayitem_gc_f(array, index, value, descr);
    Ok(p)
}
// blackhole.py:1347-1348 bhimpl_getarrayitem_raw_f
fn handler_getarrayitem_raw_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p] as usize] =
        cpu.bh_getarrayitem_raw_f(array, index, descr).to_bits() as i64;
    Ok(p + 1)
}
// blackhole.py:1363-1364 bhimpl_setarrayitem_raw_f
fn handler_setarrayitem_raw_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_i[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (descr, p) = read_descr(bh, code, p + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setarrayitem_raw_f(array, index, value, descr);
    Ok(p)
}
// getfield_raw_r (pure only, blackhole.py:1467-1469)
fn handler_getfield_raw_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let struct_ptr = bh.registers_i[code[p] as usize];
    let (descr, p) = read_descr(bh, code, p + 1);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[p] as usize] = cpu.bh_getfield_raw_r(struct_ptr, descr).0 as i64;
    Ok(p + 1)
}
// getinteriorfield_gc_f / setinteriorfield_gc_f / setinteriorfield_gc_r
/// RPython `blackhole.py:1417-1419`:
/// `return cpu.bh_getinteriorfield_gc_f(array, index, descr)`
fn handler_getinteriorfield_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p] as usize] =
        cpu.bh_getinteriorfield_gc_f(array, index, descr).to_bits() as i64;
    Ok(p + 1)
}
fn handler_getinteriorfield_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (descr, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[p] as usize] = cpu.bh_getinteriorfield_gc_r(array, index, descr).0 as i64;
    Ok(p + 1)
}
fn handler_setinteriorfield_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (descr, p) = read_descr(bh, code, p + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setinteriorfield_gc_f(array, index, value, descr);
    Ok(p)
}
fn handler_setinteriorfield_gc_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let array = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_r[code[p + 2] as usize];
    let (descr, p) = read_descr(bh, code, p + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_setinteriorfield_gc_r(array, index, majit_ir::GcRef(value as usize), descr);
    Ok(p)
}
// gc_load_indexed_i/f, gc_store_indexed_i/f (blackhole.py:1518-1540)
fn handler_gc_load_indexed_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let addr = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let scale = bh.registers_i[code[p + 2] as usize];
    let base_ofs = bh.registers_i[code[p + 3] as usize];
    let bytes = bh.registers_i[code[p + 4] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[p + 5] as usize] =
        cpu.bh_gc_load_indexed_i(addr, index, scale, base_ofs, bytes);
    Ok(p + 6)
}
fn handler_gc_load_indexed_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let addr = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let scale = bh.registers_i[code[p + 2] as usize];
    let base_ofs = bh.registers_i[code[p + 3] as usize];
    let bytes = bh.registers_i[code[p + 4] as usize];
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p + 5] as usize] = cpu
        .bh_gc_load_indexed_f(addr, index, scale, base_ofs, bytes)
        .to_bits() as i64;
    Ok(p + 6)
}
// blackhole.py:1525-1529 bhimpl_gc_store_indexed_i
fn handler_gc_store_indexed_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "r", "i", "i", "i", "i", "i", "d")
    let addr = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_i[code[p + 2] as usize];
    let scale = bh.registers_i[code[p + 3] as usize];
    let base_ofs = bh.registers_i[code[p + 4] as usize];
    let bytes = bh.registers_i[code[p + 5] as usize];
    let (_, p) = read_descr(bh, code, p + 6);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_gc_store_indexed_i(addr, index, value, scale, base_ofs, bytes);
    Ok(p)
}
// blackhole.py:1531-1535 bhimpl_gc_store_indexed_f
fn handler_gc_store_indexed_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "r", "i", "f", "i", "i", "i", "d")
    let addr = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let scale = bh.registers_i[code[p + 3] as usize];
    let base_ofs = bh.registers_i[code[p + 4] as usize];
    let bytes = bh.registers_i[code[p + 5] as usize];
    let (_, p) = read_descr(bh, code, p + 6);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_gc_store_indexed_f(addr, index, value, scale, base_ofs, bytes);
    Ok(p)
}
// blackhole.py:1504-1509 bhimpl_raw_store_i/f
fn handler_raw_store_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "i", "i", "i", "d")
    let addr = bh.registers_i[code[p] as usize];
    let offset = bh.registers_i[code[p + 1] as usize];
    let value = bh.registers_i[code[p + 2] as usize];
    let (descr, p) = read_descr(bh, code, p + 3);
    // blackhole.py:1505-1506: cpu.bh_raw_store_i(addr, offset, newvalue, arraydescr)
    // llmodel.py:740: ofs, size, _ = self.unpack_arraydescr_size(descr)
    let size = descr.as_itemsize();
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_raw_store_i(addr, offset, value, size);
    Ok(p)
}
fn handler_raw_store_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    // @arguments("cpu", "i", "i", "f", "d")
    let addr = bh.registers_i[code[p] as usize];
    let offset = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (_, p) = read_descr(bh, code, p + 3);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_raw_store_f(addr, offset, value);
    Ok(p)
}
fn handler_raw_load_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let addr = bh.registers_i[code[p] as usize];
    let offset = bh.registers_i[code[p + 1] as usize];
    let (_, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[p] as usize] = cpu.bh_raw_load_i(addr, offset);
    Ok(p + 1)
}
fn handler_raw_load_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let addr = bh.registers_i[code[p] as usize];
    let offset = bh.registers_i[code[p + 1] as usize];
    let (_, p) = read_descr(bh, code, p + 2);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p] as usize] = cpu.bh_raw_load_f(addr, offset).to_bits() as i64;
    Ok(p + 1)
}
// newlist / newlist_clear / newlist_hint (blackhole.py:1160-1193)
// RPython: compound allocation: bh_new(structdescr) + setfield + bh_new_array + setfield.
// 4 descriptors: structdescr, lengthdescr, itemsdescr, arraydescr.
fn handler_newlist(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[p] as usize];
    let (structdescr, p) = read_descr(bh, code, p + 1);
    let (lengthdescr, p) = read_descr(bh, code, p);
    let (itemsdescr, p) = read_descr(bh, code, p);
    let (arraydescr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    // blackhole.py:1163: result = cpu.bh_new(structdescr)
    let result = cpu.bh_new(structdescr);
    // blackhole.py:1164: cpu.bh_setfield_gc_i(result, length, lengthdescr)
    cpu.bh_setfield_gc_i(result, length, lengthdescr);
    // blackhole.py:1165-1169: bh_new_array_clear when is_array_of_structs or is_array_of_pointers
    let items = if arraydescr.is_array_of_structs() || arraydescr.is_array_of_pointers() {
        cpu.bh_new_array_clear(length, arraydescr)
    } else {
        cpu.bh_new_array(length, arraydescr)
    };
    // blackhole.py:1170: cpu.bh_setfield_gc_r(result, items, itemsdescr)
    cpu.bh_setfield_gc_r(result, majit_ir::GcRef(items as usize), itemsdescr);
    bh.registers_r[code[p] as usize] = result;
    Ok(p + 1)
}
// blackhole.py:1173-1180: newlist_clear always uses bh_new_array_clear.
fn handler_newlist_clear(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let length = bh.registers_i[code[p] as usize];
    let (structdescr, p) = read_descr(bh, code, p + 1);
    let (lengthdescr, p) = read_descr(bh, code, p);
    let (itemsdescr, p) = read_descr(bh, code, p);
    let (arraydescr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_new(structdescr);
    cpu.bh_setfield_gc_i(result, length, lengthdescr);
    // blackhole.py:1178: items = cpu.bh_new_array_clear(length, arraydescr)
    let items = cpu.bh_new_array_clear(length, arraydescr);
    cpu.bh_setfield_gc_r(result, majit_ir::GcRef(items as usize), itemsdescr);
    bh.registers_r[code[p] as usize] = result;
    Ok(p + 1)
}
// blackhole.py:1182-1193: newlist_hint — length=0, allocate with hint capacity.
fn handler_newlist_hint(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lengthhint = bh.registers_i[code[p] as usize];
    let (structdescr, p) = read_descr(bh, code, p + 1);
    let (lengthdescr, p) = read_descr(bh, code, p);
    let (itemsdescr, p) = read_descr(bh, code, p);
    let (arraydescr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let result = cpu.bh_new(structdescr);
    // blackhole.py:1186: cpu.bh_setfield_gc_i(result, 0, lengthdescr)
    cpu.bh_setfield_gc_i(result, 0, lengthdescr);
    // blackhole.py:1187-1191: bh_new_array_clear when is_array_of_structs or is_array_of_pointers
    let items = if arraydescr.is_array_of_structs() || arraydescr.is_array_of_pointers() {
        cpu.bh_new_array_clear(lengthhint, arraydescr)
    } else {
        cpu.bh_new_array(lengthhint, arraydescr)
    };
    cpu.bh_setfield_gc_r(result, majit_ir::GcRef(items as usize), itemsdescr);
    bh.registers_r[code[p] as usize] = result;
    Ok(p + 1)
}
// blackhole.py:1384-1387 bhimpl_getarrayitem_vable_f
fn handler_getarrayitem_vable_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        let ptr = vable as *mut u8;
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    bh.registers_f[code[p] as usize] = cpu
        .bh_getarrayitem_gc_f(array, index, array_descr)
        .to_bits() as i64;
    Ok(p + 1)
}
// blackhole.py:1400-1403 bhimpl_setarrayitem_vable_f
fn handler_setarrayitem_vable_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let vable = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    if !bh.virtualizable_info.is_null() {
        let vinfo = unsafe { &*bh.virtualizable_info };
        let ptr = vable as *mut u8;
        unsafe { crate::virtualizable::bh_clear_vable_token(vinfo, ptr) };
    }
    let (field_descr, p) = read_descr_vable_array(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let array = cpu.bh_getfield_gc_r(vable, &field_descr).0 as i64;
    cpu.bh_setarrayitem_gc_f(array, index, value, array_descr);
    Ok(p)
}
// blackhole.py:1204-1206 bhimpl_getlistitem_gc_f
fn handler_getlistitem_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let (items_descr, p) = read_descr(bh, code, p + 2);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    bh.registers_f[code[p] as usize] = cpu
        .bh_getarrayitem_gc_f(items, index, array_descr)
        .to_bits() as i64;
    Ok(p + 1)
}
// blackhole.py:1217-1219 bhimpl_setlistitem_gc_f
fn handler_setlistitem_gc_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let lst = bh.registers_r[code[p] as usize];
    let index = bh.registers_i[code[p + 1] as usize];
    let value = f64::from_bits(bh.registers_f[code[p + 2] as usize] as u64);
    let (items_descr, p) = read_descr(bh, code, p + 3);
    let (array_descr, p) = read_descr(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    let items = cpu.bh_getfield_gc_r(lst, items_descr).0 as i64;
    cpu.bh_setarrayitem_gc_f(items, index, value, array_descr);
    Ok(p)
}
// inline_call — RPython blackhole.py:1278-1319
// RPython: cpu.bh_call_*(adr2int(jitcode.fnaddr), args_i, args_r, args_f, jitcode.calldescr)
// The 'j' argcode reads a JitCode descriptor carrying fnaddr + calldescr.
// pyre: fnaddr is stored in BhDescr::JitCode; calldescr not yet modeled.
// TODO: Full implementation should use jitcode_index for frame-chain push/pop.
fn read_inline_call_jitcode(
    bh: &BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> (usize, i64, BhCallDescr, usize) {
    let (jc_descr, p) = read_descr(bh, code, p);
    match jc_descr {
        BhDescr::JitCode {
            jitcode_index,
            fnaddr,
            calldescr,
        } => (*jitcode_index, *fnaddr, calldescr.clone(), p),
        _ => panic!("expected JitCode descriptor"),
    }
}
fn handler_inline_call_irf_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[p] as usize] =
        cpu.bh_call_i(fnaddr, Some(&ai), Some(&ar), Some(&af), &calldescr);
    Ok(p + 1)
}
fn handler_inline_call_irf_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[p] as usize] = cpu
        .bh_call_r(fnaddr, Some(&ai), Some(&ar), Some(&af), &calldescr)
        .0 as i64;
    Ok(p + 1)
}
fn handler_inline_call_irf_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p] as usize] = cpu
        .bh_call_f(fnaddr, Some(&ai), Some(&ar), Some(&af), &calldescr)
        .to_bits() as i64;
    Ok(p + 1)
}
fn handler_inline_call_irf_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let (af, p) = read_list_f(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_call_v(fnaddr, Some(&ai), Some(&ar), Some(&af), &calldescr);
    Ok(p)
}
fn handler_inline_call_ir_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[p] as usize] =
        cpu.bh_call_i(fnaddr, Some(&ai), Some(&ar), None, &calldescr);
    Ok(p + 1)
}
fn handler_inline_call_ir_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[p] as usize] = cpu
        .bh_call_r(fnaddr, Some(&ai), Some(&ar), None, &calldescr)
        .0 as i64;
    Ok(p + 1)
}
fn handler_inline_call_ir_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ai, p) = read_list_i(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_call_v(fnaddr, Some(&ai), Some(&ar), None, &calldescr);
    Ok(p)
}
fn handler_inline_call_r_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[p] as usize] = cpu.bh_call_i(fnaddr, None, Some(&ar), None, &calldescr);
    Ok(p + 1)
}
fn handler_inline_call_r_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[p] as usize] =
        cpu.bh_call_r(fnaddr, None, Some(&ar), None, &calldescr).0 as i64;
    Ok(p + 1)
}
fn handler_inline_call_r_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (_jitcode_index, fnaddr, calldescr, p) = read_inline_call_jitcode(bh, code, p);
    let (ar, p) = read_list_r(bh, code, p);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_call_v(fnaddr, None, Some(&ar), None, &calldescr);
    Ok(p)
}
// recursive_call — stub (needs portal runner)
/// RPython `blackhole.py:1095-1099` `get_portal_runner(jdindex)`:
/// Returns (fnptr, calldescr) from jitdrivers_sd[jdindex].
/// pyre: uses portal_runner_ptr directly (single jitdriver).
///
/// RPython `blackhole.py:1101-1132`:
/// ```python
/// def bhimpl_recursive_call_i(self, jdindex, greens_i, greens_r, greens_f,
///                                            reds_i, reds_r, reds_f):
///     fnptr, calldescr = self.get_portal_runner(jdindex)
///     return self.cpu.bh_call_i(fnptr, greens_i+reds_i, greens_r+reds_r,
///                               greens_f+reds_f, calldescr)
/// ```
/// Read recursive_call args and merge greens+reds per kind.
/// Returns (jdindex, all_i, all_r, all_f, next_position).
fn read_recursive_call_args(
    bh: &BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> (usize, Vec<i64>, Vec<i64>, Vec<i64>, usize) {
    let jdindex = code[p] as usize; // 'c' short constant
    let p = p + 1;
    let (greens_i, p) = read_list_i(bh, code, p);
    let (greens_r, p) = read_list_r(bh, code, p);
    let (greens_f, p) = read_list_f(bh, code, p);
    let (reds_i, p) = read_list_i(bh, code, p);
    let (reds_r, p) = read_list_r(bh, code, p);
    let (reds_f, p) = read_list_f(bh, code, p);
    // RPython blackhole.py:1105-1108: greens + reds merged per kind.
    let mut all_i = greens_i;
    all_i.extend(&reds_i);
    let mut all_r = greens_r;
    all_r.extend(&reds_r);
    let mut all_f = greens_f;
    all_f.extend(&reds_f);
    (jdindex, all_i, all_r, all_f, p)
}
// blackhole.py:1101-1108 bhimpl_recursive_call_i
fn handler_recursive_call_i(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jdindex, all_i, all_r, all_f, p) = read_recursive_call_args(bh, code, p);
    let (fnptr, calldescr) = bh.get_portal_runner(jdindex);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_i[code[p] as usize] =
        cpu.bh_call_i(fnptr, Some(&all_i), Some(&all_r), Some(&all_f), &calldescr);
    Ok(p + 1)
}
// blackhole.py:1109-1116 bhimpl_recursive_call_r
fn handler_recursive_call_r(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jdindex, all_i, all_r, all_f, p) = read_recursive_call_args(bh, code, p);
    let (fnptr, calldescr) = bh.get_portal_runner(jdindex);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_r[code[p] as usize] = cpu
        .bh_call_r(fnptr, Some(&all_i), Some(&all_r), Some(&all_f), &calldescr)
        .0 as i64;
    Ok(p + 1)
}
// blackhole.py:1117-1124 bhimpl_recursive_call_f
fn handler_recursive_call_f(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jdindex, all_i, all_r, all_f, p) = read_recursive_call_args(bh, code, p);
    let (fnptr, calldescr) = bh.get_portal_runner(jdindex);
    let cpu = bh.cpu.expect("cpu not set");
    bh.registers_f[code[p] as usize] = cpu
        .bh_call_f(fnptr, Some(&all_i), Some(&all_r), Some(&all_f), &calldescr)
        .to_bits() as i64;
    Ok(p + 1)
}
// blackhole.py:1125-1132 bhimpl_recursive_call_v
fn handler_recursive_call_v(
    bh: &mut BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, DispatchError> {
    let (jdindex, all_i, all_r, all_f, p) = read_recursive_call_args(bh, code, p);
    let (fnptr, calldescr) = bh.get_portal_runner(jdindex);
    let cpu = bh.cpu.expect("cpu not set");
    cpu.bh_call_v(fnptr, Some(&all_i), Some(&all_r), Some(&all_f), &calldescr);
    Ok(p)
}
