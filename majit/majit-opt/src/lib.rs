/// JIT optimization pipeline.
///
/// Translated from rpython/jit/metainterp/optimizeopt/.
///
/// The optimizer chains multiple passes, each implementing the Optimization trait.
/// Operations flow through the chain: IntBounds → Rewrite → Virtualize → String →
/// Pure → Guard → Simplify → Heap (configurable).
pub mod bridgeopt;
pub mod dependency;
pub mod earlyforce;
pub mod guard;
pub mod heap;
pub mod info;
pub mod intbounds;
pub mod intdiv;
pub mod intutils;
pub mod optimize;
pub mod optimizer;
pub mod pure;
pub mod rewrite;
pub mod schedule;
pub mod shortpreamble;
pub mod simplify;
pub mod unroll;
pub mod vector;
pub mod version;
pub mod virtualize;
pub mod virtualstate;
pub mod vstring;
pub mod walkvirtual;

use std::collections::{HashMap, HashSet};

use crate::intutils::IntBound;
use info::PtrInfo;
use majit_ir::{Op, OpCode, OpRef, Value};
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

#[derive(Clone, Debug, PartialEq)]
pub enum ImportedShortPureArg {
    OpRef(OpRef),
    Const(Value),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedShortPureOp {
    pub opcode: OpCode,
    pub descr_idx: Option<u32>,
    pub args: Vec<ImportedShortPureArg>,
    pub result: OpRef,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedShortAlias {
    pub result: OpRef,
    pub same_as_source: OpRef,
    pub same_as_opcode: OpCode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedShortSource {
    pub result: OpRef,
    pub source: OpRef,
}

#[derive(Clone, Debug)]
pub struct TrackedPreambleUse {
    pub result: OpRef,
    pub produced: crate::shortpreamble::ProducedShortOp,
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
    /// RPython unroll.py: widened integer knowledge imported from the preamble.
    /// OptIntBounds intersects these with freshly discovered facts in phase 2.
    pub imported_int_bounds: HashMap<OpRef, IntBound>,
    /// RPython shortpreamble.py / heap.py: imported cached field reads from the
    /// preamble. Phase 2 uses these to seed Heap's read cache without
    /// re-emitting preamble heap reads.
    pub imported_short_fields: HashMap<(OpRef, u32), OpRef>,
    /// RPython shortpreamble.py / heap.py: imported cached constant-index array
    /// reads from the preamble.
    pub imported_short_arrayitems: HashMap<(OpRef, u32, i64), OpRef>,
    /// RPython shortpreamble.py / pure.py: imported pure-operation results from
    /// the preamble. Phase 2 uses these as cross-iteration CSE facts.
    pub imported_short_pure_ops: Vec<ImportedShortPureOp>,
    /// RPython shortpreamble.py: invented SameAs names preserved from exported
    /// short boxes. Phase 2 can later re-materialize these aliases when
    /// building the short preamble for bridges.
    pub imported_short_aliases: Vec<ImportedShortAlias>,
    /// Original preamble result box for each imported short-box result.
    /// This preserves RPython PreambleOp.op identity so phase 2 assembly
    /// can remap any surviving synthetic imported boxes back to the
    /// corresponding preamble value.
    pub imported_short_sources: Vec<ImportedShortSource>,
    /// RPython shortpreamble.py / rewrite.py: imported CALL_LOOPINVARIANT
    /// results keyed by constant function pointer.
    pub imported_loop_invariant_results: HashMap<i64, OpRef>,
    /// RPython shortpreamble.py: active phase-2 short preamble builder.
    /// Tracks which imported short facts are actually consumed by the body.
    pub imported_short_preamble_builder: Option<crate::shortpreamble::ShortPreambleBuilder>,
    /// Dedup imported short fact uses so the builder stays in first-use order.
    imported_short_preamble_used: HashSet<OpRef>,
    /// RPython unroll.py: potential_extra_ops populated by force_op_from_preamble
    /// and later consumed by optimizer.force_box().
    potential_extra_ops: HashMap<OpRef, TrackedPreambleUse>,
    /// RPython unroll.py: live ExtendedShortPreambleBuilder while replaying an
    /// existing target token's short preamble.
    active_short_preamble_producer: Option<crate::shortpreamble::ExtendedShortPreambleBuilder>,
    /// RPython unroll.py: virtual structures at JUMP for preamble peeling.
    pub exported_jump_virtuals: Vec<crate::optimizer::ExportedJumpVirtual>,
    /// RPython shortpreamble.py: pass-collected preamble producers aligned to
    /// the exported loop-header inputargs.
    pub exported_short_boxes: Vec<crate::shortpreamble::PreambleOp>,
    /// RPython import_state: maps original inputarg index → fresh virtual head OpRef.
    /// Used by ensure_linked_list_head to return the imported virtual.
    pub imported_virtual_heads: Vec<(usize, OpRef)>,
    /// RPython optimizer.py: `patchguardop` — the last GUARD_FUTURE_CONDITION op.
    /// Used by unroll to attach resume data to extra guards from short preamble.
    pub patchguardop: Option<Op>,
    /// RPython unroll.py:454-457: virtual state captured BEFORE force at JUMP.
    /// Used by export_state to produce a VirtualState that includes virtuals
    /// (which are forced by the time exported_loop_state is computed).
    pub pre_force_virtual_state: Option<crate::virtualstate::VirtualState>,
    /// JUMP args captured BEFORE force (corresponding to pre_force_virtual_state).
    pub pre_force_jump_args: Option<Vec<OpRef>>,
    /// RPython optimizer.py: end_args after force_box_for_end_of_preamble().
    /// export_state() prefers this over a raw get_replacement() snapshot.
    pub preamble_end_args: Option<Vec<OpRef>>,
    /// Field OpRefs of virtual args before force. Maps virtual OpRef →
    /// [(field_idx, field_value_ref)]. Used by make_inputargs to flatten
    /// virtuals into label args after force has destroyed PtrInfo.
    pub pre_force_field_refs: HashMap<OpRef, Vec<(u32, OpRef)>>,
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
            imported_int_bounds: HashMap::new(),
            imported_short_fields: HashMap::new(),
            imported_short_arrayitems: HashMap::new(),
            imported_short_pure_ops: Vec::new(),
            imported_short_aliases: Vec::new(),
            imported_short_sources: Vec::new(),
            imported_loop_invariant_results: HashMap::new(),
            imported_short_preamble_builder: None,
            imported_short_preamble_used: HashSet::new(),
            potential_extra_ops: HashMap::new(),
            active_short_preamble_producer: None,
            exported_jump_virtuals: Vec::new(),
            exported_short_boxes: Vec::new(),
            imported_virtual_heads: Vec::new(),
            patchguardop: None,
            pre_force_virtual_state: None,
            pre_force_jump_args: None,
            preamble_end_args: None,
            pre_force_field_refs: HashMap::new(),
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
            imported_int_bounds: HashMap::new(),
            imported_short_fields: HashMap::new(),
            imported_short_arrayitems: HashMap::new(),
            imported_short_pure_ops: Vec::new(),
            imported_short_aliases: Vec::new(),
            imported_short_sources: Vec::new(),
            imported_loop_invariant_results: HashMap::new(),
            imported_short_preamble_builder: None,
            imported_short_preamble_used: HashSet::new(),
            potential_extra_ops: HashMap::new(),
            active_short_preamble_producer: None,
            exported_jump_virtuals: Vec::new(),
            exported_short_boxes: Vec::new(),
            imported_virtual_heads: Vec::new(),
            patchguardop: None,
            pre_force_virtual_state: None,
            pre_force_jump_args: None,
            preamble_end_args: None,
            pre_force_field_refs: HashMap::new(),
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs as usize
    }

    /// Allocate a fresh OpRef position (for imported virtual heads).
    pub fn alloc_op_position(&mut self) -> OpRef {
        self.reserve_pos()
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

    pub fn initialize_imported_short_preamble_builder(
        &mut self,
        label_args: &[OpRef],
        short_inputargs: &[OpRef],
        exported_short_boxes: &[crate::shortpreamble::PreambleOp],
    ) {
        let produced: Vec<(OpRef, crate::shortpreamble::ProducedShortOp)> = exported_short_boxes
            .iter()
            .map(|entry| {
                (
                    entry.op.pos,
                    crate::shortpreamble::ProducedShortOp {
                        kind: entry.kind.clone(),
                        preamble_op: entry.op.clone(),
                        invented_name: entry.invented_name,
                        same_as_source: entry.same_as_source,
                    },
                )
            })
            .collect();
        self.imported_short_preamble_builder = Some(
            crate::shortpreamble::ShortPreambleBuilder::new(label_args, &produced, short_inputargs),
        );
        self.imported_short_preamble_used.clear();
    }

    pub fn note_imported_short_use(&mut self, result: OpRef) {
        let _ = self.force_op_from_preamble(result);
    }

    pub fn force_op_from_preamble(&mut self, result: OpRef) -> OpRef {
        let result = self.get_replacement(result);
        let is_constant = self.get_constant(result).is_some();
        if self.imported_short_preamble_used.insert(result) {
            if let Some(builder) = self.imported_short_preamble_builder.as_mut() {
                let tracked = builder.use_box(result).is_some();
                if tracked && !is_constant {
                    if let Some(produced) = builder.produced_short_op(result) {
                        self.potential_extra_ops
                            .insert(result, TrackedPreambleUse { result, produced });
                    }
                }
            }
        }
        result
    }

    pub fn take_potential_extra_op(&mut self, result: OpRef) -> Option<TrackedPreambleUse> {
        self.potential_extra_ops.remove(&result)
    }

    pub fn activate_short_preamble_producer(
        &mut self,
        builder: crate::shortpreamble::ExtendedShortPreambleBuilder,
    ) {
        self.active_short_preamble_producer = Some(builder);
    }

    pub fn active_short_preamble_producer_mut(
        &mut self,
    ) -> Option<&mut crate::shortpreamble::ExtendedShortPreambleBuilder> {
        self.active_short_preamble_producer.as_mut()
    }

    pub fn build_active_short_preamble(&self) -> Option<crate::shortpreamble::ShortPreamble> {
        self.active_short_preamble_producer
            .as_ref()
            .map(|builder| builder.build_short_preamble_struct())
    }

    pub fn take_active_short_preamble_producer(
        &mut self,
    ) -> Option<crate::shortpreamble::ExtendedShortPreambleBuilder> {
        self.active_short_preamble_producer.take()
    }

    pub fn build_imported_short_preamble(&self) -> Option<crate::shortpreamble::ShortPreamble> {
        self.imported_short_preamble_builder
            .as_ref()
            .map(|builder| builder.build_short_preamble_struct())
    }

    pub fn used_imported_short_aliases(&self) -> Vec<ImportedShortAlias> {
        self.imported_short_preamble_builder
            .as_ref()
            .map(|builder| {
                builder
                    .extra_same_as()
                    .iter()
                    .map(|op| ImportedShortAlias {
                        result: op.pos,
                        same_as_source: op.arg(0),
                        same_as_opcode: op.opcode,
                    })
                    .collect()
            })
            .unwrap_or_default()
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
        self.imported_int_bounds.clear();
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

    /// optimizer.py: produce_potential_short_preamble_ops(sb)
    /// Contribute operations to the short preamble builder.
    /// Called after preamble optimization to collect ops that bridges need to replay.
    fn produce_potential_short_preamble_ops(&self, _sb: &mut crate::shortpreamble::ShortBoxes) {
        // Default: no contribution
    }

    /// RPython unroll.py: exported_infos also carries widened IntBound knowledge.
    fn export_arg_int_bounds(
        &self,
        _args: &[OpRef],
        _ctx: &OptContext,
    ) -> HashMap<OpRef, IntBound> {
        HashMap::new()
    }

    /// optimizer.py: is_virtual(opref)
    /// Whether an opref refers to a virtual object (for this pass).
    fn is_virtual(&self, _opref: OpRef) -> bool {
        false
    }
}
