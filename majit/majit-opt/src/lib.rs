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
use majit_ir::{DescrRef, Op, OpCode, OpRef, Value};
use std::collections::VecDeque;

pub(crate) fn majit_log_enabled() -> bool {
    std::env::var_os("MAJIT_LOG").is_some()
}

/// compile.py: ResumeAtPositionDescr — type tag for guards created during
/// loop unrolling / short preamble inlining.
#[derive(Debug)]
struct OptResumeAtPositionDescr;

impl majit_ir::Descr for OptResumeAtPositionDescr {
    fn is_resume_at_position(&self) -> bool {
        true
    }
}

/// Create a ResumeAtPositionDescr for optimizer-generated guards.
pub fn make_resume_at_position_descr() -> DescrRef {
    std::sync::Arc::new(OptResumeAtPositionDescr)
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

#[derive(Clone, Debug)]
pub struct ImportedShortPureOp {
    pub opcode: OpCode,
    pub descr: Option<DescrRef>,
    pub args: Vec<ImportedShortPureArg>,
    pub result: OpRef,
}

impl PartialEq for ImportedShortPureOp {
    fn eq(&self, other: &Self) -> bool {
        self.opcode == other.opcode
            && self.descr.as_ref().map(|d| d.index()) == other.descr.as_ref().map(|d| d.index())
            && self.args == other.args
            && self.result == other.result
    }
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

/// optimizer.py:787-789: constant_fold — allocate an immutable object at
/// compile time when all fields are constants. The callback receives the
/// SizeDescr size_bytes, and returns a raw pointer (GcRef) to freshly
/// allocated memory. The optimizer writes field values directly.
pub type ConstantFoldAllocFn = Box<dyn Fn(usize) -> majit_ir::GcRef>;

// --- Phase A: new forwarding types (additive, not yet wired) ---

/// Unified forwarding slot. RPython stores forwarded as one of:
/// - another op (Op), a PtrInfo (Info), or a Const (Constant).
/// This replaces the separate `forwarding`, `ptr_info`, `constants` vecs
/// once migration is complete.
#[derive(Clone)]
pub enum Forwarded {
    None,
    Op(OpRef),
    Info(InfoRef),
    Constant(Value),
}

/// Handle into `InfoArena`. Lightweight copy type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InfoRef(pub u32);

/// Arena allocator for `PtrInfo` objects. Provides stable `InfoRef` handles
/// that do not move when the arena grows.
pub struct InfoArena {
    infos: Vec<PtrInfo>,
}

impl InfoArena {
    pub fn new() -> Self {
        Self { infos: Vec::new() }
    }

    pub fn alloc(&mut self, info: PtrInfo) -> InfoRef {
        let idx = self.infos.len();
        self.infos.push(info);
        InfoRef(idx as u32)
    }

    pub fn get(&self, r: InfoRef) -> &PtrInfo {
        &self.infos[r.0 as usize]
    }

    pub fn get_mut(&mut self, r: InfoRef) -> &mut PtrInfo {
        &mut self.infos[r.0 as usize]
    }
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
    /// RPython: Box._forwarded — used for within-phase forwarding only.
    pub forwarding: Vec<OpRef>,
    /// RPython: mapping dict in inline_short_preamble — separate from _forwarded.
    /// Maps Phase 1 source OpRefs to Phase 2 short arg OpRefs.
    /// Consulted by get_replacement BEFORE the forwarding chain.
    pub short_preamble_mapping: HashMap<OpRef, OpRef>,
    /// Number of input arguments, used to offset emitted op positions
    /// so that variable indices don't collide with input arg indices.
    num_inputs: u32,
    /// Next unique op position for newly emitted or queued extra operations.
    pub(crate) next_pos: u32,
    /// Extra operations requested by the current pass. The optimizer drains
    /// these through the remaining downstream passes, matching RPython
    /// send_extra_operation()/emit_operation behavior.
    extra_operations: VecDeque<Op>,
    /// RPython emit_extra(op, emit=False) parity: ops queued to be
    /// processed starting from a specific pass index (skipping earlier passes).
    /// Used by heap's force_lazy_set to route ops through remaining passes
    /// without re-entering the heap pass itself.
    pub(crate) extra_operations_after: VecDeque<(usize, Op)>,
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
    /// Real descriptors for imported cached field reads.
    pub imported_short_field_descrs: HashMap<(OpRef, u32), DescrRef>,
    /// RPython shortpreamble.py / heap.py: imported cached constant-index array
    /// reads from the preamble.
    pub imported_short_arrayitems: HashMap<(OpRef, u32, i64), OpRef>,
    /// Real descriptors for imported cached constant-index array reads.
    pub imported_short_arrayitem_descrs: HashMap<(OpRef, u32, i64), DescrRef>,
    /// RPython shortpreamble.py / pure.py: imported pure-operation results from
    /// the preamble. Phase 2 uses these as cross-iteration CSE facts.
    pub imported_short_pure_ops: Vec<ImportedShortPureOp>,
    /// RPython shortpreamble.py: invented SameAs names preserved from exported
    /// short boxes. Phase 2 can later re-materialize these aliases when
    /// building the short preamble for bridges.
    pub imported_short_aliases: Vec<ImportedShortAlias>,
    /// (base_len, short_args): virtual field values start at base_len
    /// within short_args. Used by install_imported_virtuals.
    pub imported_virtual_args: Option<(usize, Vec<OpRef>)>,
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
    /// Phase-2 loop-body mode from optimizer.skip_flush.
    /// RPython unroll.py relies on this distinction so virtualize can keep
    /// body-side allocations concrete when guard recovery cannot rebuild them.
    pub skip_flush_mode: bool,
    /// Index of the pass currently executing propagate_forward.
    /// Used by passes to call send_extra_operation_after(self_idx, ..)
    /// matching RPython's emit_extra(op, emit=False) which routes to
    /// self.next_optimization.
    pub current_pass_idx: usize,
    /// Field OpRefs of virtual args before force. Maps virtual OpRef →
    /// [(field_idx, field_value_ref)]. Used by make_inputargs to flatten
    /// virtuals into label args after force has destroyed PtrInfo.
    pub pre_force_field_refs: HashMap<OpRef, Vec<(u32, OpRef)>>,
    /// optimizer.py: pendingfields — deferred SetfieldGc/SetarrayitemGc ops
    /// where the stored value is virtual. Set by OptHeap.emitting_operation()
    /// before a guard, consumed by emit_with_guard_check() to encode into
    /// the guard's rd_pendingfields.
    pub pending_for_guard: Vec<Op>,
    /// optimizer.py:787: constant_fold allocator callback.
    /// When set, the optimizer can fold immutable virtuals filled with
    /// constants into compile-time constant pointers (info.py:140-145).
    pub constant_fold_alloc: Option<ConstantFoldAllocFn>,
    /// True while optimizer.py:_emit_operation equivalent is forcing args
    /// just before final emission. In this phase, virtual forcing must emit
    /// directly into new_operations instead of re-entering the pass chain.
    pub in_final_emission: bool,
    /// Phase A: unified forwarding table (additive, not yet wired).
    pub forwarded: Vec<Forwarded>,
    /// Phase A: arena for PtrInfo objects (additive, not yet wired).
    pub info_arena: InfoArena,
}

impl OptContext {
    pub fn new(estimated_ops: usize) -> Self {
        OptContext {
            new_operations: Vec::with_capacity(estimated_ops),
            constants: Vec::new(),
            forwarding: Vec::new(),
            short_preamble_mapping: HashMap::new(),
            num_inputs: 0,
            next_pos: 0,
            extra_operations: VecDeque::new(),
            extra_operations_after: VecDeque::new(),
            ptr_info: Vec::new(),
            int_lower_bounds: HashMap::new(),
            imported_int_bounds: HashMap::new(),
            imported_short_fields: HashMap::new(),
            imported_short_field_descrs: HashMap::new(),
            imported_short_arrayitems: HashMap::new(),
            imported_short_arrayitem_descrs: HashMap::new(),
            imported_short_pure_ops: Vec::new(),
            imported_short_aliases: Vec::new(),
            imported_virtual_args: None,
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
            skip_flush_mode: false,
            current_pass_idx: 0,
            pre_force_field_refs: HashMap::new(),
            in_final_emission: false,
            pending_for_guard: Vec::new(),
            constant_fold_alloc: None,
            // (import_boxes removed)
            forwarded: Vec::new(),
            info_arena: InfoArena::new(),
        }
    }

    pub fn with_num_inputs(estimated_ops: usize, num_inputs: usize) -> Self {
        OptContext {
            new_operations: Vec::with_capacity(estimated_ops),
            constants: Vec::new(),
            forwarding: Vec::new(),
            short_preamble_mapping: HashMap::new(),
            num_inputs: num_inputs as u32,
            next_pos: num_inputs as u32,
            extra_operations: VecDeque::new(),
            extra_operations_after: VecDeque::new(),
            ptr_info: Vec::new(),
            int_lower_bounds: HashMap::new(),
            imported_int_bounds: HashMap::new(),
            imported_short_fields: HashMap::new(),
            imported_short_field_descrs: HashMap::new(),
            imported_short_arrayitems: HashMap::new(),
            imported_short_arrayitem_descrs: HashMap::new(),
            imported_short_pure_ops: Vec::new(),
            imported_short_aliases: Vec::new(),
            imported_virtual_args: None,
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
            skip_flush_mode: false,
            current_pass_idx: 0,
            pre_force_field_refs: HashMap::new(),
            in_final_emission: false,
            pending_for_guard: Vec::new(),
            constant_fold_alloc: None,
            // (import_boxes removed)
            forwarded: Vec::new(),
            info_arena: InfoArena::new(),
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
        } else if self.new_operations.iter().any(|e| e.pos == op.pos) {
            // RPython Box parity: reassign position to avoid collision.
            op.pos = self.reserve_pos();
        } else {
            self.next_pos = self.next_pos.max(op.pos.0.saturating_add(1));
        }
        let pos_ref = op.pos;
        // Clear any stale forwarding for this pos. Phase 2 import may set
        // forwarding for heap cache entries whose source OpRef coincides with
        // a trace op's pos. When the trace op is re-emitted in Phase 2, the
        // new result must NOT be aliased to the imported value.
        let idx = pos_ref.0 as usize;
        if idx < self.forwarding.len() && !self.forwarding[idx].is_none() {
            self.forwarding[idx] = OpRef::NONE;
        }
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

    /// RPython emit_extra(op, emit=False) parity: queue an operation to
    /// be processed through passes AFTER the calling pass. Skips earlier
    /// passes (including the caller) to avoid re-absorption loops.
    /// `after_pass_idx`: index of the calling pass (op starts from idx+1).
    pub fn emit_through_passes_after(&mut self, after_pass_idx: usize, mut op: Op) -> OpRef {
        if op.pos.is_none() {
            op.pos = self.reserve_pos();
        } else {
            self.next_pos = self.next_pos.max(op.pos.0.saturating_add(1));
        }
        let pos_ref = op.pos;
        self.extra_operations_after
            .push_back((after_pass_idx + 1, op));
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

    pub fn initialize_imported_short_preamble_builder_from_exported_ops(
        &mut self,
        short_args: &[OpRef],
        short_inputargs: &[OpRef],
        exported_short_ops: &[crate::unroll::ExportedShortOp],
    ) -> bool {
        use crate::shortpreamble::{PreambleOpKind, ProducedShortOp, ShortPreambleBuilder};
        use crate::unroll::{ExportedShortArg, ExportedShortOp, ExportedShortResult};

        let mut produced: Vec<(OpRef, ProducedShortOp)> =
            Vec::with_capacity(exported_short_ops.len());
        let mut produced_results: Vec<OpRef> = Vec::with_capacity(exported_short_ops.len());
        let mut temporary_results: Vec<Option<OpRef>> = Vec::new();
        let mut imported_constants: HashMap<OpRef, OpRef> = HashMap::new();
        let imported_result_for_source = |source: OpRef, this: &Self| {
            this.imported_short_sources
                .iter()
                .find_map(|entry| (entry.source == source).then_some(entry.result))
        };

        let mut resolve_result = |result: &ExportedShortResult, this: &mut Self| match result {
            ExportedShortResult::Slot(slot) => short_args.get(*slot).copied(),
            ExportedShortResult::Temporary(index) => {
                if *index >= temporary_results.len() {
                    temporary_results.resize(index + 1, None);
                }
                let slot = &mut temporary_results[*index];
                Some(*slot.get_or_insert_with(|| this.alloc_op_position()))
            }
        };

        for entry in exported_short_ops {
            match entry {
                ExportedShortOp::Pure {
                    source,
                    opcode,
                    descr,
                    args,
                    result,
                    invented_name,
                    same_as_source,
                } => {
                    if opcode.is_call() {
                        return false;
                    }
                    let Some(result_opref) = imported_result_for_source(*source, self)
                        .or_else(|| resolve_result(result, self))
                    else {
                        return false;
                    };
                    let mut resolved_args = Vec::with_capacity(args.len());
                    for arg in args {
                        let resolved = match arg {
                            ExportedShortArg::Slot(slot) => short_args.get(*slot).copied(),
                            ExportedShortArg::Produced(index) => {
                                produced_results.get(*index).copied()
                            }
                            ExportedShortArg::Const { source, value } => {
                                let opref =
                                    imported_constants.entry(*source).or_insert_with(|| {
                                        let opref = self.alloc_op_position();
                                        self.make_constant(opref, value.clone());
                                        opref
                                    });
                                Some(*opref)
                            }
                        };
                        let Some(resolved) = resolved else {
                            return false;
                        };
                        resolved_args.push(resolved);
                    }
                    let mut op = Op::new(*opcode, &resolved_args);
                    op.pos = result_opref;
                    op.descr = descr.clone();
                    let produced_op = ProducedShortOp {
                        kind: PreambleOpKind::Pure,
                        preamble_op: op,
                        invented_name: *invented_name,
                        same_as_source: *same_as_source,
                    };
                    produced.push((*source, produced_op.clone()));
                    if *source != result_opref {
                        produced.push((result_opref, produced_op));
                    }
                    produced_results.push(result_opref);
                }
                ExportedShortOp::HeapField {
                    source,
                    object_slot,
                    descr,
                    result_type,
                    result,
                    invented_name,
                    same_as_source,
                } => {
                    let Some(result_opref) = imported_result_for_source(*source, self)
                        .or_else(|| resolve_result(result, self))
                    else {
                        return false;
                    };
                    let Some(&obj) = short_args.get(*object_slot) else {
                        return false;
                    };
                    let opcode = match result_type {
                        majit_ir::Type::Int => OpCode::GetfieldGcI,
                        majit_ir::Type::Ref => OpCode::GetfieldGcR,
                        majit_ir::Type::Float => OpCode::GetfieldGcF,
                        majit_ir::Type::Void => return false,
                    };
                    let mut op = Op::new(opcode, &[obj]);
                    op.pos = result_opref;
                    op.descr = Some(descr.clone());
                    let produced_op = ProducedShortOp {
                        kind: PreambleOpKind::Heap,
                        preamble_op: op,
                        invented_name: *invented_name,
                        same_as_source: *same_as_source,
                    };
                    produced.push((*source, produced_op.clone()));
                    if *source != result_opref {
                        produced.push((result_opref, produced_op));
                    }
                    produced_results.push(result_opref);
                }
                _ => return false,
            }
        }

        self.imported_short_preamble_builder = Some(ShortPreambleBuilder::new(
            short_args,
            &produced,
            short_inputargs,
        ));
        self.imported_short_preamble_used.clear();
        true
    }

    pub fn note_imported_short_use(&mut self, result: OpRef) {
        let _ = self.force_op_from_preamble(result);
    }

    pub(crate) fn imported_short_source(&self, result: OpRef) -> OpRef {
        self.imported_short_sources
            .iter()
            .find_map(|entry| (entry.result == result).then_some(entry.source))
            .unwrap_or(result)
    }

    pub fn force_op_from_preamble(&mut self, result: OpRef) -> OpRef {
        let result = self.get_replacement(result);
        let preamble_result = self.imported_short_source(result);
        let is_constant = self.get_constant(preamble_result).is_some();
        if self.imported_short_preamble_used.insert(preamble_result) {
            let tracked = if let Some(builder) = self.active_short_preamble_producer.as_mut() {
                builder.use_box(preamble_result).is_some()
            } else if let Some(builder) = self.imported_short_preamble_builder.as_mut() {
                builder.use_box(preamble_result).is_some()
            } else {
                false
            };
            if tracked && !is_constant {
                if let Some(builder) = self.imported_short_preamble_builder.as_ref() {
                    if let Some(produced) = builder.produced_short_op(preamble_result) {
                        self.potential_extra_ops.insert(
                            preamble_result,
                            TrackedPreambleUse {
                                result: preamble_result,
                                produced,
                            },
                        );
                    }
                } else if let Some(builder) = self.active_short_preamble_producer.as_ref() {
                    if let Some(produced) = builder.produced_short_op(preamble_result) {
                        self.potential_extra_ops.insert(
                            preamble_result,
                            TrackedPreambleUse {
                                result: preamble_result,
                                produced,
                            },
                        );
                    }
                }
            }
        }
        preamble_result
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

    pub(crate) fn has_extra_operations(&self) -> bool {
        !self.extra_operations.is_empty()
    }

    pub(crate) fn flush_extra_operations_raw(&mut self) {
        while let Some(op) = self.extra_operations.pop_front() {
            self.emit(op);
        }
    }

    /// Record that `old` should be replaced by `new` wherever it appears.
    /// RPython: make_equal_to — within-phase forwarding only.
    pub fn replace_op(&mut self, old: OpRef, new: OpRef) {
        if old == new {
            return;
        }
        let idx = old.0 as usize;
        if idx >= self.forwarding.len() {
            self.forwarding.resize(idx + 1, OpRef::NONE);
        }
        self.forwarding[idx] = new;
        // Sync to forwarded table.
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        self.forwarded[idx] = Forwarded::Op(new);
    }

    /// RPython Box identity: set per-value forwarding for import_state.
    /// This does NOT use the flat forwarding table, preventing chains.
    pub fn set_import_box(&mut self, source: OpRef, target: OpRef) {
        // Use replace_op: forwarding[source] = target.
        // With ptr_info stop in get_replacement, import_boxes is no longer
        // needed — target has ptr_info set by apply_exported_info, so
        // get_replacement(source) → target (stops at ptr_info).
        self.replace_op(source, target);
    }

    /// RPython get_box_replacement: follow forwarding chain, stop when the
    /// NEXT position has ptr_info set (it's a terminal, like RPython's
    /// get_box_replacement stopping at _forwarded=Info).
    pub fn get_replacement(&self, mut opref: OpRef) -> OpRef {
        // RPython: mapping dict lookup (inline_short_preamble).
        // Cross-phase mapping is separate from _forwarded chain.
        if let Some(&mapped) = self.short_preamble_mapping.get(&opref) {
            opref = mapped;
        }
        loop {
            let idx = opref.0 as usize;
            if idx >= self.forwarding.len() {
                return opref;
            }
            let next = self.forwarding[idx];
            if next.is_none() {
                return opref;
            }
            // RPython: stop if NEXT has Info (ptr_info set → terminal).
            let next_idx = next.0 as usize;
            if next_idx < self.ptr_info.len() {
                if self.ptr_info[next_idx].is_some() {
                    return next;
                }
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

    /// Whether `opref` has a known constant value.
    pub fn is_constant(&self, opref: OpRef) -> bool {
        self.get_constant(opref).is_some()
    }

    /// Get constant integer value, if known.
    pub fn get_constant_int(&self, opref: OpRef) -> Option<i64> {
        self.get_constant(opref).and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
    }

    /// Create a new constant int OpRef.
    pub fn make_constant_int(&mut self, value: i64) -> OpRef {
        let pos = self.alloc_op_position();
        self.make_constant(pos, Value::Int(value));
        pos
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

    /// Look up the operation that produces a given OpRef.
    /// Searches emitted operations and input ops.
    /// Used for pattern matching nested operations (e.g., int_add(int_add(x, C1), C2)).
    /// Returns a clone to avoid borrow conflicts with mutable ctx methods.
    pub fn get_producing_op(&self, opref: OpRef) -> Option<Op> {
        let opref = self.get_replacement(opref);
        self.new_operations
            .iter()
            .find(|op| op.pos == opref)
            .cloned()
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

    /// Extract known class from PtrInfo, if available.
    /// RPython: preamble_info.get_known_class(cpu) — used by guard pass
    /// to eliminate redundant GuardClass/GuardNonnullClass in Phase 2.
    pub fn get_known_class(&self, opref: OpRef) -> Option<majit_ir::GcRef> {
        match self.get_ptr_info(opref)? {
            PtrInfo::KnownClass { class_ptr, .. } => Some(*class_ptr),
            PtrInfo::Instance(info) => info.known_class,
            _ => None,
        }
    }

    /// info.py: getptrinfo(op) — mutable variant.
    pub fn get_ptr_info_mut(&mut self, opref: OpRef) -> Option<&mut PtrInfo> {
        let opref = self.get_replacement(opref);
        self.ptr_info
            .get_mut(opref.0 as usize)
            .and_then(|v| v.as_mut())
    }

    /// info.py: op.set_forwarded(info) — set PtrInfo for an OpRef.
    /// Ensure a PtrInfo exists for the given OpRef. Creates an empty
    /// Instance if none exists, so that set_field can store values.
    pub fn ensure_ptr_info(&mut self, opref: OpRef) {
        let idx = opref.0 as usize;
        if idx >= self.ptr_info.len() {
            self.ptr_info.resize(idx + 1, None);
        }
        if self.ptr_info[idx].is_none() {
            self.ptr_info[idx] = Some(PtrInfo::instance(None, None));
        }
    }

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
    fn flush(&mut self, _ctx: &mut OptContext) {}

    /// Emit any remaining lazy sets directly (not through pass chain).
    /// Called after flush+drain to emit lazy_sets that were re-stored
    /// during drain. Virtual values should already be forced at this point.
    fn emit_remaining_lazy_directly(&mut self, _ctx: &mut OptContext) {}

    /// Emit only virtualizable lazy SetfieldGc ops.
    /// Called before JUMP in Phase 2 so compiled code writes virtualizable
    /// fields (head/size) to memory for guard failure recovery.
    fn flush_virtualizable(&mut self, _ctx: &mut OptContext) {}

    /// Mark this pass as Phase 2 (loop body). Phase 2 should not fully
    /// virtualize New() ops because guard recovery_layout is not yet
    /// populated. Default: no-op.
    fn set_phase2(&mut self, _phase2: bool) {}

    /// Name of this pass (for debugging).
    fn name(&self) -> &'static str;

    /// optimizer.py: produce_potential_short_preamble_ops(sb)
    /// Contribute operations to the short preamble builder.
    /// Called after preamble optimization to collect ops that bridges need to replay.
    fn produce_potential_short_preamble_ops(&self, _sb: &mut crate::shortpreamble::ShortBoxes) {
        // Default: no contribution
    }

    /// Export all cached field entries from this pass.
    /// Used to propagate heap cache from Phase 1 to Phase 2 via ExportedState.
    fn export_cached_fields(&self) -> Vec<(OpRef, u32, OpRef)> {
        Vec::new()
    }

    /// heap.py: deserialize_optheap — import cached fields into this pass.
    fn import_cached_fields(&mut self, _entries: &[(OpRef, u32, OpRef)]) {}

    /// bridgeopt.py:113-122: serialize_optrewrite — export loopinvariant results.
    fn export_loopinvariant_results(&self) -> Vec<(i64, OpRef)> {
        Vec::new()
    }

    /// bridgeopt.py:173-185: deserialize_optrewrite — import loopinvariant results.
    fn import_loopinvariant_results(&mut self, _entries: &[(i64, OpRef)]) {}

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

    /// RPython optimizer.py: emitting_operation(op)
    /// Called before any operation is emitted to the output, regardless of
    /// which pass emits it. This enables passes like OptHeap to force lazy
    /// sets before guards, even when the guard is emitted by an earlier pass.
    fn emitting_operation(&mut self, _op: &Op, _ctx: &mut OptContext) {}
}
