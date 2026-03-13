//! JitState implementation for pyre.
//!
//! `PyreJitState` bridges the interpreter's `PyFrame` with majit's JIT
//! infrastructure. It extracts live values from the frame, restores them
//! after compiled code runs, and provides the meta/sym types for tracing.

use pyre_bytecode::bytecode::OpArgState;

use majit_ir::{GcRef, OpRef, Type, Value};
use majit_meta::JitState;

use pyre_object::PyObjectRef;

/// Interpreter state exposed to the JIT framework.
///
/// Built from `PyFrame` before calling `back_edge`, and synced back
/// after compiled code runs.
pub struct PyreJitState {
    /// Current instruction index.
    pub next_instr: usize,
    /// Fast local variables (from frame.locals_w).
    pub locals: Vec<PyObjectRef>,
    /// Namespace keys in deterministic (sorted) order.
    pub ns_keys: Vec<String>,
    /// Namespace values corresponding to ns_keys.
    pub ns_values: Vec<PyObjectRef>,
    /// Value stack snapshot.
    pub stack: Vec<PyObjectRef>,
    /// Current stack depth.
    pub stack_depth: usize,
}

/// Meta information for a trace — describes the shape of the code being traced.
#[derive(Clone)]
pub struct PyreMeta {
    /// Instruction index at the merge point (green key).
    pub merge_pc: usize,
    /// Number of fast local variable slots.
    pub num_locals: usize,
    /// Sorted namespace keys tracked by this trace.
    pub ns_keys: Vec<String>,
    /// Stack depth at the merge point.
    pub stack_depth: usize,
}

/// Symbolic state during tracing.
///
/// Maps interpreter values to IR `OpRef`s during trace recording.
pub struct PyreSym {
    /// OpRefs for each fast local variable.
    pub locals: Vec<OpRef>,
    /// Sorted namespace keys (mirrors PyreMeta.ns_keys).
    pub ns_keys: Vec<String>,
    /// OpRefs for namespace values.
    pub ns_values: Vec<OpRef>,
    /// OpRefs for stack values.
    pub stack: Vec<OpRef>,
    /// Instruction decoding state for ExtendedArg handling.
    pub arg_state: OpArgState,
}

/// Environment context — currently unused.
pub struct PyreEnv;

impl PyreMeta {
    /// Total number of live values tracked (locals + namespace + stack).
    pub fn num_live(&self) -> usize {
        self.num_locals + self.ns_keys.len() + self.stack_depth
    }
}

impl JitState for PyreJitState {
    type Meta = PyreMeta;
    type Sym = PyreSym;
    type Env = PyreEnv;

    fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {
        PyreMeta {
            merge_pc: self.next_instr,
            num_locals: self.locals.len(),
            ns_keys: self.ns_keys.clone(),
            stack_depth: self.stack_depth,
        }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        let mut live =
            Vec::with_capacity(self.locals.len() + self.ns_values.len() + self.stack_depth);
        for &local in &self.locals {
            live.push(local as i64);
        }
        for &val in &self.ns_values {
            live.push(val as i64);
        }
        for i in 0..self.stack_depth {
            live.push(self.stack[i] as i64);
        }
        live
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        let mut values =
            Vec::with_capacity(self.locals.len() + self.ns_values.len() + self.stack_depth);
        for &local in &self.locals {
            values.push(if local.is_null() {
                Value::Int(0)
            } else {
                Value::Ref(GcRef(local as usize))
            });
        }
        for &val in &self.ns_values {
            values.push(Value::Ref(GcRef(val as usize)));
        }
        for i in 0..self.stack_depth {
            let val = self.stack[i];
            values.push(Value::Ref(GcRef(val as usize)));
        }
        values
    }

    fn create_sym(meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        let mut next = 0u32;
        let locals: Vec<OpRef> = (0..meta.num_locals)
            .map(|_| {
                let r = OpRef(next);
                next += 1;
                r
            })
            .collect();
        let ns_values: Vec<OpRef> = (0..meta.ns_keys.len())
            .map(|_| {
                let r = OpRef(next);
                next += 1;
                r
            })
            .collect();
        let stack: Vec<OpRef> = (0..meta.stack_depth)
            .map(|_| {
                let r = OpRef(next);
                next += 1;
                r
            })
            .collect();
        PyreSym {
            locals,
            ns_keys: meta.ns_keys.clone(),
            ns_values,
            stack,
            arg_state: OpArgState::default(),
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        self.next_instr == meta.merge_pc
            && self.locals.len() == meta.num_locals
            && self.ns_keys == meta.ns_keys
            && self.stack_depth == meta.stack_depth
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        let mut idx = 0;
        for local in &mut self.locals {
            *local = values[idx] as PyObjectRef;
            idx += 1;
        }
        for val in &mut self.ns_values {
            *val = values[idx] as PyObjectRef;
            idx += 1;
        }
        for i in 0..self.stack_depth {
            self.stack[i] = values[idx] as PyObjectRef;
            idx += 1;
        }
    }

    fn restore_values(&mut self, meta: &Self::Meta, values: &[Value]) {
        let mut idx = 0;
        for local in &mut self.locals {
            *local = value_to_ptr(&values[idx]);
            idx += 1;
        }
        for val in &mut self.ns_values {
            *val = value_to_ptr(&values[idx]);
            idx += 1;
        }
        let stack_depth = meta.stack_depth;
        for i in 0..stack_depth {
            self.stack[i] = value_to_ptr(&values[idx]);
            idx += 1;
        }
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        let mut args = Vec::with_capacity(sym.locals.len() + sym.ns_values.len() + sym.stack.len());
        args.extend_from_slice(&sym.locals);
        args.extend_from_slice(&sym.ns_values);
        args.extend_from_slice(&sym.stack);
        args
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        let mut args = Vec::with_capacity(sym.locals.len() + sym.ns_values.len() + sym.stack.len());
        for &op in &sym.locals {
            args.push((op, Type::Ref));
        }
        for &op in &sym.ns_values {
            args.push((op, Type::Ref));
        }
        for &op in &sym.stack {
            args.push((op, Type::Ref));
        }
        args
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        sym.locals.len() == meta.num_locals
            && sym.ns_values.len() == meta.ns_keys.len()
            && sym.stack.len() == meta.stack_depth
    }
}

fn value_to_ptr(value: &Value) -> PyObjectRef {
    match value {
        Value::Ref(gc_ref) => gc_ref.0 as PyObjectRef,
        Value::Int(n) => *n as PyObjectRef,
        _ => std::ptr::null_mut(),
    }
}

// ── Virtualizable configuration ──────────────────────────────────────
//
// PyPy's `pypy/interpreter/pyframe.py` declares:
//
//     _virtualizable_ = ['locals_stack_w[*]', 'valuestackdepth',
//                         'last_instr', ...]
//
// Our Rust equivalent uses explicit byte offsets instead of name-based
// introspection. The JIT optimizer's Virtualize pass uses this info
// to keep frame fields in CPU registers, eliminating heap accesses
// for LoadFast/StoreFast and stack push/pop during compiled code.
//
// `build_pyframe_virtualizable_info()` lives in `pyre-interp/src/frame.rs`
// alongside the PyFrame struct and its offset constants, because
// `pyre-jit` cannot depend on `pyre-interp` (reverse dependency).
// The driver registration call happens in `pyre-interp/src/eval.rs`.
