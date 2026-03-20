/// JIT optimization pipeline.
///
/// Translated from rpython/jit/metainterp/optimizeopt/.
///
/// The optimizer chains multiple passes, each implementing the Optimization trait.
/// Operations flow through the chain: IntBounds → Rewrite → Virtualize → String →
/// Pure → Guard → Simplify → Heap (configurable).
pub mod bridgeopt;
pub mod earlyforce;
pub mod guard;
pub mod heap;
pub mod info;
pub mod intbounds;
pub mod intdiv;
pub mod intutils;
pub mod optimizer;
pub mod pure;
pub mod rewrite;
pub mod shortpreamble;
pub mod simplify;
pub mod unroll;
pub mod vector;
pub mod virtualize;
pub mod virtualstate;
pub mod vstring;
pub mod walkvirtual;

use std::collections::HashMap;

use info::PtrInfo;
use majit_ir::{Op, OpRef, Value};
use std::collections::VecDeque;

pub(crate) fn majit_log_enabled() -> bool {
    std::env::var_os("MAJIT_LOG").is_some()
}

/// Result of an optimization pass processing an operation.
#[derive(Debug)]
pub enum OptimizationResult {
    /// Emit this operation (possibly modified).
    Emit(Op),
    /// Replace with a different operation.
    Replace(Op),
    /// Remove the operation entirely.
    Remove,
    /// Pass the operation to the next pass unchanged.
    PassOn,
}

/// Context provided to optimization passes.
///
/// Holds the shared state that passes read from and write to.
pub struct OptContext {
    /// The output operation list being built.
    pub new_operations: Vec<Op>,
    /// Constants known at optimization time (op -> value).
    pub constants: Vec<Option<Value>>,
    /// Forwarding chain: maps old OpRef to replacement OpRef.
    pub forwarding: Vec<OpRef>,
    /// Number of input arguments, used to offset emitted op positions
    /// so that variable indices don't collide with input arg indices.
    num_inputs: u32,
    /// Next unique op position for newly emitted or queued extra operations.
    next_pos: u32,
    /// Extra operations requested by the current pass. The optimizer drains
    /// these through the remaining downstream passes, matching RPython
    /// send_extra_operation()/emit_operation behavior.
    extra_operations: VecDeque<Op>,
    /// info.py: per-OpRef pointer info, shared across all passes.
    ///
    /// RPython attaches info objects directly to operations via
    /// `op.set_forwarded(info)`. majit uses an indexed Vec instead.
    /// All passes can read/write this to share virtual/class/nonnull info.
    pub ptr_info: Vec<Option<PtrInfo>>,
    /// Known lower bounds for integer-typed OpRefs, shared across passes.
    ///
    /// heap.py: arrayinfo.getlenbound().make_gt_const(index) records that
    /// an ARRAYLEN_GC result >= index+1. intbounds.py uses this to
    /// eliminate redundant length guards.
    pub int_lower_bounds: HashMap<OpRef, i64>,
    /// RPython unroll.py: virtual structures at JUMP for preamble peeling.
    pub exported_jump_virtuals: Vec<crate::optimizer::ExportedJumpVirtual>,
}

impl OptContext {
    pub fn new(estimated_ops: usize) -> Self {
        OptContext {
            new_operations: Vec::with_capacity(estimated_ops),
            constants: Vec::new(),
            forwarding: Vec::new(),
            num_inputs: 0,
            next_pos: 0,
            extra_operations: VecDeque::new(),
            ptr_info: Vec::new(),
            int_lower_bounds: HashMap::new(),
            exported_jump_virtuals: Vec::new(),
        }
    }

    pub fn with_num_inputs(estimated_ops: usize, num_inputs: usize) -> Self {
        OptContext {
            new_operations: Vec::with_capacity(estimated_ops),
            constants: Vec::new(),
            forwarding: Vec::new(),
            num_inputs: num_inputs as u32,
            next_pos: num_inputs as u32,
            extra_operations: VecDeque::new(),
            ptr_info: Vec::new(),
            int_lower_bounds: HashMap::new(),
            exported_jump_virtuals: Vec::new(),
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs as usize
    }

    pub(crate) fn reserve_pos(&mut self) -> OpRef {
        self.next_pos = self
            .next_pos
            .max(self.num_inputs + self.new_operations.len() as u32);
        while self
            .constants
            .get(self.next_pos as usize)
            .is_some_and(|value| value.is_some())
        {
            self.next_pos += 1;
        }
        let pos_ref = OpRef(self.next_pos);
        self.next_pos += 1;
        pos_ref
    }

    /// Emit an operation to the output.
    ///
    /// If the op has no pos assigned (NONE), sets it to `num_inputs + idx`
    /// so the backend's variable numbering stays consistent.
    pub fn emit(&mut self, mut op: Op) -> OpRef {
        if op.pos.is_none() {
            op.pos = self.reserve_pos();
        } else {
            self.next_pos = self.next_pos.max(op.pos.0.saturating_add(1));
        }
        let pos_ref = op.pos;
        self.new_operations.push(op);
        pos_ref
    }

    /// Queue an extra operation to be processed through the remaining
    /// downstream passes instead of being appended directly.
    pub fn emit_through_passes(&mut self, mut op: Op) -> OpRef {
        if op.pos.is_none() {
            op.pos = self.reserve_pos();
        } else {
            self.next_pos = self.next_pos.max(op.pos.0.saturating_add(1));
        }
        let pos_ref = op.pos;
        self.extra_operations.push_back(op);
        pos_ref
    }

    pub(crate) fn pop_extra_operation(&mut self) -> Option<Op> {
        self.extra_operations.pop_front()
    }

    #[cfg(test)]
    pub(crate) fn flush_extra_operations_raw(&mut self) {
        while let Some(op) = self.extra_operations.pop_front() {
            self.emit(op);
        }
    }

    /// Record that `old` should be replaced by `new` wherever it appears.
    pub fn replace_op(&mut self, old: OpRef, new: OpRef) {
        if old == new {
            return; // avoid self-referencing forwarding loop
        }
        let idx = old.0 as usize;
        if idx >= self.forwarding.len() {
            self.forwarding.resize(idx + 1, OpRef::NONE);
        }
        self.forwarding[idx] = new;
    }

    /// Follow the forwarding chain to get the current replacement for `opref`.
    pub fn get_replacement(&self, mut opref: OpRef) -> OpRef {
        loop {
            let idx = opref.0 as usize;
            if idx >= self.forwarding.len() {
                return opref;
            }
            let next = self.forwarding[idx];
            if next.is_none() {
                return opref;
            }
            opref = next;
        }
    }

    /// Record that an operation produces a known constant value.
    pub fn make_constant(&mut self, opref: OpRef, value: Value) {
        let idx = opref.0 as usize;
        if idx >= self.constants.len() {
            self.constants.resize(idx + 1, None);
        }
        self.constants[idx] = Some(value);
    }

    /// Get the constant value for an operation, if known.
    pub fn get_constant(&self, opref: OpRef) -> Option<&Value> {
        let opref = self.get_replacement(opref);
        let idx = opref.0 as usize;
        self.constants.get(idx).and_then(|v| v.as_ref())
    }

    /// Get constant integer value, if known.
    pub fn get_constant_int(&self, opref: OpRef) -> Option<i64> {
        self.get_constant(opref).and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
    }

    /// Record a constant-folded value and return its OpRef.
    ///
    /// If `opref` is not already a known constant, records the value.
    /// Returns `opref` (which is now known to be this constant).
    pub fn find_or_record_constant_int(&mut self, opref: OpRef, value: i64) -> OpRef {
        self.make_constant(opref, Value::Int(value));
        opref
    }

    /// Get constant float value, if known.
    pub fn get_constant_float(&self, opref: OpRef) -> Option<f64> {
        self.get_constant(opref).and_then(|v| match v {
            Value::Float(f) => Some(*f),
            _ => None,
        })
    }

    /// optimizer.py: make_equal_to(op, value)
    /// Replace an op's result with a known value (forwarding + constant).
    pub fn make_equal_to(&mut self, opref: OpRef, target: OpRef) {
        self.replace_op(opref, target);
    }

    /// optimizer.py: get_box_replacement(opref)
    /// Same as get_replacement — follows forwarding chain.
    pub fn get_box_replacement(&self, opref: OpRef) -> OpRef {
        self.get_replacement(opref)
    }

    /// Number of emitted operations so far.
    pub fn num_emitted(&self) -> usize {
        self.new_operations.len()
    }

    /// Get the last emitted operation, if any.
    pub fn last_emitted_operation(&self) -> Option<&Op> {
        self.new_operations.last()
    }

    /// optimizer.py: get_constant_box(opref)
    /// Get a constant Value for an OpRef, or None if not constant.
    pub fn get_constant_box(&self, opref: OpRef) -> Option<Value> {
        self.get_constant(opref).cloned()
    }

    /// optimizer.py: clear_newoperations()
    /// Clear the output operation list (used when restarting optimization).
    pub fn clear_newoperations(&mut self) {
        self.new_operations.clear();
        self.extra_operations.clear();
        self.next_pos = self.num_inputs;
    }

    /// Get a mutable reference to the last emitted operation.
    pub fn last_emitted_operation_mut(&mut self) -> Option<&mut Op> {
        self.new_operations.last_mut()
    }

    // ── info.py: per-OpRef pointer info ──

    /// info.py: getptrinfo(op) — get PtrInfo for an OpRef.
    pub fn get_ptr_info(&self, opref: OpRef) -> Option<&PtrInfo> {
        let opref = self.get_replacement(opref);
        self.ptr_info.get(opref.0 as usize).and_then(|v| v.as_ref())
    }

    /// info.py: getptrinfo(op) — mutable variant.
    pub fn get_ptr_info_mut(&mut self, opref: OpRef) -> Option<&mut PtrInfo> {
        let opref = self.get_replacement(opref);
        self.ptr_info
            .get_mut(opref.0 as usize)
            .and_then(|v| v.as_mut())
    }

    /// info.py: op.set_forwarded(info) — set PtrInfo for an OpRef.
    pub fn set_ptr_info(&mut self, opref: OpRef, info: PtrInfo) {
        let idx = opref.0 as usize;
        if idx >= self.ptr_info.len() {
            self.ptr_info.resize(idx + 1, None);
        }
        self.ptr_info[idx] = Some(info);
    }

    /// optimizer.py: replace_op_with(old, new_op, ctx)
    /// Replace old opref AND emit the new op.
    pub fn replace_op_with(&mut self, old: OpRef, new_op: Op) -> OpRef {
        let new_ref = self.emit(new_op);
        self.replace_op(old, new_ref);
        new_ref
    }

    /// Check if an opref has been replaced (forwarded).
    pub fn is_replaced(&self, opref: OpRef) -> bool {
        let idx = opref.0 as usize;
        if idx < self.forwarding.len() {
            !self.forwarding[idx].is_none()
        } else {
            false
        }
    }
}

/// An optimization pass.
///
/// optimizer.py: Optimization base class.
pub trait Optimization {
    /// Process an operation. Called for each operation in the trace.
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult;

    /// Called once before optimization starts.
    fn setup(&mut self) {}

    /// Called after all operations have been processed.
    fn flush(&mut self) {}

    /// Name of this pass (for debugging).
    fn name(&self) -> &'static str;

    /// RPython unroll.py: set Phase 2 flatten mode (only OptVirtualize uses this).
    fn set_flatten_virtuals_at_jump(&mut self, _enabled: bool) {}

    /// optimizer.py: produce_potential_short_preamble_ops(sb)
    /// Contribute operations to the short preamble builder.
    /// Called after preamble optimization to collect ops that bridges need to replay.
    fn produce_potential_short_preamble_ops(&self, _sb: &mut crate::shortpreamble::ShortBoxes) {
        // Default: no contribution
    }

    /// optimizer.py: is_virtual(opref)
    /// Whether an opref refers to a virtual object (for this pass).
    fn is_virtual(&self, _opref: OpRef) -> bool {
        false
    }
}
