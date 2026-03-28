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
pub mod renamer;
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

use crate::optimizeopt::intutils::IntBound;
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
    pub produced: crate::optimizeopt::shortpreamble::ProducedShortOp,
}

/// optimizer.py:787-789: constant_fold — allocate an immutable object at
/// compile time when all fields are constants. The callback receives the
/// SizeDescr size_bytes, and returns a raw pointer (GcRef) to freshly
/// allocated memory. The optimizer writes field values directly.
pub type ConstantFoldAllocFn = Box<dyn Fn(usize) -> majit_ir::GcRef>;

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
    /// Phase 2 imported virtuals (from Phase 1 export). Used by
    /// store_final_boxes_in_guard to create GuardVirtualEntry for
    /// NONE positions inherited from Phase 1 virtualization.
    pub imported_virtuals: Vec<crate::optimizeopt::optimizer::ImportedVirtual>,
    /// Phase 2 imported label args (OpRefs in Phase 2 namespace).
    pub imported_label_args: Option<Vec<OpRef>>,
    /// RPython shortpreamble.py: active phase-2 short preamble builder.
    /// Tracks which imported short facts are actually consumed by the body.
    pub imported_short_preamble_builder:
        Option<crate::optimizeopt::shortpreamble::ShortPreambleBuilder>,
    /// RPython optimizer.py: quasi_immutable_deps — collected during optimization.
    /// (object_ptr, field_index) pairs identifying specific quasi-immutable
    /// slots the trace depends on. After compilation, per-slot watchers
    /// are registered.
    pub quasi_immutable_deps: HashSet<(u64, u32)>,
    /// Dedup imported short fact uses so the builder stays in first-use order.
    imported_short_preamble_used: HashSet<OpRef>,
    /// RPython unroll.py: potential_extra_ops populated by force_op_from_preamble
    /// and later consumed by optimizer.force_box().
    potential_extra_ops: HashMap<OpRef, TrackedPreambleUse>,
    /// RPython unroll.py: live ExtendedShortPreambleBuilder while replaying an
    /// existing target token's short preamble.
    active_short_preamble_producer:
        Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder>,
    /// RPython unroll.py: virtual structures at JUMP for preamble peeling.
    pub exported_jump_virtuals: Vec<crate::optimizeopt::optimizer::ExportedJumpVirtual>,
    /// RPython shortpreamble.py: pass-collected preamble producers aligned to
    /// the exported loop-header inputargs.
    pub exported_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
    /// RPython import_state: maps original inputarg index → fresh virtual head OpRef.
    /// Used by ensure_linked_list_head to return the imported virtual.
    pub imported_virtual_heads: Vec<(usize, OpRef)>,
    /// RPython optimizer.py: `patchguardop` — the last GUARD_FUTURE_CONDITION op.
    /// Used by unroll to attach resume data to extra guards from short preamble.
    pub patchguardop: Option<Op>,
    /// RPython unroll.py:454-457: virtual state captured BEFORE force at JUMP.
    /// Used by export_state to produce a VirtualState that includes virtuals
    /// (which are forced by the time exported_loop_state is computed).
    pub pre_force_virtual_state: Option<crate::optimizeopt::virtualstate::VirtualState>,
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
    /// resume.py parity: per-guard snapshot boxes from tracing time.
    /// Used by emit() to call store_final_boxes_in_guard inline (RPython
    /// calls this during optimization, not post-assembly).
    pub snapshot_boxes: HashMap<i32, Vec<OpRef>>,
    /// ConstantPool type map for BoxEnv.is_const() during inline numbering.
    pub constant_types_for_numbering: HashMap<u32, majit_ir::Type>,
}

/// resume.py:192-226 parity — BoxEnv for optimizer context.
///
/// Wraps an immutable reference to OptContext, implementing the BoxEnv
/// trait so that ResumeDataLoopMemo.number() can tag boxes during
/// store_final_boxes_in_guard.
pub struct OptBoxEnv<'a> {
    pub ctx: &'a OptContext,
}

impl<'a> majit_ir::BoxEnv for OptBoxEnv<'a> {
    fn get_box_replacement(&self, opref: OpRef) -> OpRef {
        self.ctx.get_replacement(opref)
    }

    fn is_const(&self, opref: OpRef) -> bool {
        // RPython resume.py:204: isinstance(box, Const)
        // Checks both optimizer constant map AND PtrInfo::Constant (ConstPtrInfo).
        if self.ctx.is_constant(opref) {
            return true;
        }
        // info.py: ConstPtrInfo.is_constant() → True
        matches!(
            self.ctx.get_ptr_info(opref),
            Some(crate::optimizeopt::info::PtrInfo::Constant(_))
        )
    }

    fn get_const(&self, opref: OpRef) -> (i64, majit_ir::Type) {
        // RPython ConstPtr parity: check numbering type overrides first.
        // ob_type constants are stored as Value::Int(ptr) in the constant map
        // but their true type is Ref (from numbering_type_overrides).
        let type_override = self.ctx.constant_types_for_numbering.get(&opref.0).copied();
        match self.ctx.get_constant(opref) {
            Some(Value::Int(v)) => (*v, type_override.unwrap_or(majit_ir::Type::Int)),
            Some(Value::Float(f)) => (f.to_bits() as i64, majit_ir::Type::Float),
            Some(Value::Ref(r)) => (r.0 as i64, majit_ir::Type::Ref),
            _ => {
                // info.py: ConstPtrInfo — GcRef constant stored in PtrInfo
                if let Some(crate::optimizeopt::info::PtrInfo::Constant(gcref)) =
                    self.ctx.get_ptr_info(opref)
                {
                    (gcref.0 as i64, majit_ir::Type::Ref)
                } else {
                    (0, majit_ir::Type::Int)
                }
            }
        }
    }

    fn get_type(&self, opref: OpRef) -> majit_ir::Type {
        // Check constant type first
        if let Some(val) = self.ctx.get_constant(opref) {
            return val.get_type();
        }
        // RPython: box.type — check constant_types_for_numbering (includes
        // inputarg types and constant pool types registered by the tracer).
        if let Some(&tp) = self.ctx.constant_types_for_numbering.get(&opref.0) {
            return tp;
        }
        // Check emitted op result type (most accurate for concrete values)
        let resolved = self.ctx.get_replacement(opref);
        for op in &self.ctx.new_operations {
            if op.pos == resolved {
                return op.result_type();
            }
        }
        // PtrInfo presence → Ref type (for non-emitted ops like input args)
        if self.ctx.get_ptr_info(opref).is_some() {
            return majit_ir::Type::Ref;
        }
        // info.py: ConstPtrInfo → Ref
        if let Some(crate::optimizeopt::info::PtrInfo::Constant(_)) = self.ctx.get_ptr_info(opref) {
            return majit_ir::Type::Ref;
        }
        majit_ir::Type::Int
    }

    fn is_virtual_ref(&self, opref: OpRef) -> bool {
        matches!(
            self.ctx.get_ptr_info(opref),
            Some(PtrInfo::Virtual(_) | PtrInfo::VirtualStruct(_))
        )
    }

    fn is_virtual_raw(&self, _opref: OpRef) -> bool {
        // pyre doesn't have Int-typed virtual objects
        false
    }

    fn get_virtual_fields(&self, opref: OpRef) -> Option<majit_ir::VirtualFieldsInfo> {
        let resolved = self.ctx.get_replacement(opref);
        let info = self.ctx.get_ptr_info(resolved)?;
        match info {
            PtrInfo::Virtual(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: vi.known_class,
                field_oprefs: vi
                    .fields
                    .iter()
                    .map(|(_, vref)| self.ctx.get_replacement(*vref))
                    .collect(),
            }),
            PtrInfo::VirtualStruct(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: None,
                field_oprefs: vi
                    .fields
                    .iter()
                    .map(|(_, vref)| self.ctx.get_replacement(*vref))
                    .collect(),
            }),
            _ => None,
        }
    }
}

impl OptContext {
    /// RPython optimizer.py: add to quasi_immutable_deps
    pub fn add_quasi_immutable_dep(&mut self, dep: (u64, u32)) {
        self.quasi_immutable_deps.insert(dep);
    }

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
            imported_virtuals: Vec::new(),
            imported_label_args: None,
            patchguardop: None,
            pre_force_virtual_state: None,
            pre_force_jump_args: None,
            preamble_end_args: None,
            skip_flush_mode: false,
            current_pass_idx: 0,

            in_final_emission: false,
            pending_for_guard: Vec::new(),
            constant_fold_alloc: None,
            quasi_immutable_deps: HashSet::new(),
            snapshot_boxes: HashMap::new(),
            constant_types_for_numbering: HashMap::new(),
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
            imported_virtuals: Vec::new(),
            imported_label_args: None,
            patchguardop: None,
            pre_force_virtual_state: None,
            pre_force_jump_args: None,
            preamble_end_args: None,
            skip_flush_mode: false,
            current_pass_idx: 0,

            in_final_emission: false,
            pending_for_guard: Vec::new(),
            constant_fold_alloc: None,
            quasi_immutable_deps: HashSet::new(),
            snapshot_boxes: HashMap::new(),
            constant_types_for_numbering: HashMap::new(),
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
        // RPython parity: emit() does NOT clear forwarding.
        // In RPython, Box._forwarded is never cleared by emit — each Box
        // has unique identity. The forwarding set by import_box must
        // survive body op emission for consumer switchover to work.
        // Phase 2 body defines its own result at pos — supersede any cross-phase
        // short_preamble_mapping entry, so get_replacement returns the body's
        // value rather than the preamble import.
        self.short_preamble_mapping.remove(&pos_ref);

        // RPython optimizer.py:_emit_operation → store_final_boxes_in_guard:
        // produce rd_numb inline at guard emission time, not post-assembly.
        if op.opcode.is_guard() {
            self.number_guard_inline(&mut op);
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
        exported_short_boxes: &[crate::optimizeopt::shortpreamble::PreambleOp],
    ) {
        let produced: Vec<(OpRef, crate::optimizeopt::shortpreamble::ProducedShortOp)> =
            exported_short_boxes
                .iter()
                .map(|entry| {
                    (
                        entry.op.pos,
                        crate::optimizeopt::shortpreamble::ProducedShortOp {
                            kind: entry.kind.clone(),
                            preamble_op: entry.op.clone(),
                            invented_name: entry.invented_name,
                            same_as_source: entry.same_as_source,
                        },
                    )
                })
                .collect();
        self.imported_short_preamble_builder = Some(
            crate::optimizeopt::shortpreamble::ShortPreambleBuilder::new(
                label_args,
                &produced,
                short_inputargs,
            ),
        );
        self.imported_short_preamble_used.clear();
    }

    pub fn initialize_imported_short_preamble_builder_from_exported_ops(
        &mut self,
        short_args: &[OpRef],
        short_inputargs: &[OpRef],
        exported_short_ops: &[crate::optimizeopt::unroll::ExportedShortOp],
    ) -> bool {
        use crate::optimizeopt::shortpreamble::{
            PreambleOpKind, ProducedShortOp, ShortPreambleBuilder,
        };
        use crate::optimizeopt::unroll::{ExportedShortArg, ExportedShortOp, ExportedShortResult};

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
                    object,
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
                    let obj = match object {
                        ExportedShortArg::Slot(slot) => short_args.get(*slot).copied(),
                        ExportedShortArg::Const { source, .. } => Some(*source),
                        ExportedShortArg::Produced(_) => None,
                    };
                    let Some(obj) = obj else {
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
        // In RPython, preamble_result maps to the original preamble Box.
        // In pyre, the preamble source OpRef can be overridden by Phase 2
        // body processing (e.g., Virtualize sets forwarding on the same
        // position). Use the imported result directly to avoid stale
        // forwarding from body ops.
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
        // Return the imported result (not preamble source) to avoid stale
        // forwarding from Phase 2 body ops that reuse preamble OpRef positions.
        result
    }

    pub fn take_potential_extra_op(&mut self, result: OpRef) -> Option<TrackedPreambleUse> {
        self.potential_extra_ops.remove(&result)
    }

    pub fn activate_short_preamble_producer(
        &mut self,
        builder: crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder,
    ) {
        self.active_short_preamble_producer = Some(builder);
    }

    pub fn active_short_preamble_producer_mut(
        &mut self,
    ) -> Option<&mut crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder> {
        self.active_short_preamble_producer.as_mut()
    }

    pub fn build_active_short_preamble(
        &self,
    ) -> Option<crate::optimizeopt::shortpreamble::ShortPreamble> {
        self.active_short_preamble_producer
            .as_ref()
            .map(|builder| builder.build_short_preamble_struct())
    }

    pub fn take_active_short_preamble_producer(
        &mut self,
    ) -> Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder> {
        self.active_short_preamble_producer.take()
    }

    pub fn build_imported_short_preamble(
        &self,
    ) -> Option<crate::optimizeopt::shortpreamble::ShortPreamble> {
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
    /// RPython set_forwarded parity: setting forwarding REPLACES any Info.
    pub fn replace_op(&mut self, old: OpRef, new: OpRef) {
        if old == new {
            return;
        }
        let idx = old.0 as usize;
        if idx >= self.forwarding.len() {
            self.forwarding.resize(idx + 1, OpRef::NONE);
        }
        self.forwarding[idx] = new;
        // Clear ptr_info: this position is now a transit (forwarding).
        if idx < self.ptr_info.len() {
            self.ptr_info[idx] = None;
        }
        // Phase 2 body is defining the value for `old` — supersede any
        // cross-phase mapping so get_replacement follows forwarding instead.
        self.short_preamble_mapping.remove(&old);
    }

    /// RPython unroll.py: source.set_forwarded(target)
    /// Sets forwarding from Phase 2 source to Phase 1 export target.
    /// apply_exported_info_recursive then sets PtrInfo on the TARGET
    /// (via get_replacement), matching RPython's setinfo_from_preamble.
    pub fn set_import_box(&mut self, source: OpRef, target: OpRef) {
        self.replace_op(source, target);
    }

    /// RPython get_box_replacement: follow forwarding chain, stop when the
    /// NEXT position has ptr_info set (it's a terminal, like RPython's
    /// get_box_replacement stopping at _forwarded=Info).
    /// info.py:111-118: mark_last_guard — record the last guard position
    /// on the PtrInfo for an OpRef. RPython: opinfo.mark_last_guard(optimizer).
    pub fn mark_last_guard(&mut self, opref: OpRef) {
        let pos = match self.new_operations.last() {
            Some(op) if op.opcode.is_guard() => (self.new_operations.len() - 1) as i32,
            _ => return,
        };
        if let Some(info) = self.get_ptr_info_mut(opref) {
            info.set_last_guard_pos(pos);
        }
    }

    /// info.py:100-103: get_last_guard — retrieve the last guard op via PtrInfo.
    pub fn get_last_guard(&self, opref: OpRef) -> Option<&Op> {
        let pos = self.get_ptr_info(opref)?.get_last_guard_pos()?;
        self.new_operations.get(pos)
    }

    /// RPython get_box_replacement: follow the forwarding chain until
    /// we reach a terminal. With ptr_info/forwarding mutual exclusion,
    /// a position has EITHER forwarding (transit) OR ptr_info (terminal),
    /// never both. A terminal position has no forwarding → the loop
    /// ends naturally when forwarding[idx] is NONE.
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

    /// RPython optimizer.py:722-752 store_final_boxes_in_guard inline.
    /// Called from emit() for every guard during optimization. Produces
    /// rd_numb via memo.number() using the CURRENT optimizer state
    /// (replacement chain, constants, virtual info).
    fn number_guard_inline(&self, op: &mut Op) {
        use majit_ir::resumedata::{self, ResumeDataLoopMemo, Snapshot};

        // RPython parity: store_final_boxes_in_guard (in emit_with_guard_check)
        // already produced rd_numb + liveboxes. Don't overwrite.
        if op.rd_numb.is_some() {
            return;
        }

        // RPython: every guard has a snapshot (from capture_resumedata).
        // Use tracing-time snapshot if available, otherwise build from
        // current fail_args (for guards created by the optimizer itself,
        // e.g. GUARD_NOT_INVALIDATED from quasi-immutable field access).
        // For guards without tracing-time snapshot (optimizer-created guards
        // like GUARD_NOT_INVALIDATED from quasi-immut), produce a simple
        // 1:1 TAGBOX rd_numb without modifying fail_args.
        if op.rd_resume_position < 0 || !self.snapshot_boxes.contains_key(&op.rd_resume_position) {
            if let Some(ref fa) = op.fail_args {
                let fa_len = fa.len();
                let mut ns = resumedata::NumberingState::new(fa_len + 8);
                ns.append_int(0); // slot 0: size (patched)
                ns.append_int(fa_len as i32); // slot 1: num_failargs
                ns.append_int(0); // vable_array len
                ns.append_int(0); // vref_array len
                ns.append_int(0); // jitcode_index
                ns.append_int(0); // pc
                for (i, &opref) in fa.iter().enumerate() {
                    if opref.is_none() {
                        // TODO(TAGVIRTUAL): resume.py _number_virtuals encodes
                        // virtual slots as TAGVIRTUAL(vidx). Currently NULLREF;
                        // requires investigation of spectral_norm/fannkuch
                        // regression before enabling.
                        ns.append_short(resumedata::NULLREF);
                    } else {
                        let t = resumedata::tag(i as i32, resumedata::TAGBOX)
                            .unwrap_or(resumedata::NULLREF);
                        ns.append_short(t);
                    }
                }
                ns.patch_current_size(0);
                op.rd_numb = Some(ns.create_numbering());
                op.rd_consts = Some(Vec::new());
            }
            return;
        }

        let snapshot_boxes = self.snapshot_boxes[&op.rd_resume_position].clone();

        // resume.py:201-202 get_box_replacement parity:
        // Pass ORIGINAL (unresolved) snapshot boxes. _number_boxes calls
        // env.get_box_replacement per-box, which resolves through the
        // replacement chain while preserving virtual identity.
        let snapshot = Snapshot::single_frame(0, snapshot_boxes.clone());

        // BoxEnv bridging current optimizer state.
        struct InlineBoxEnv<'a> {
            ctx: &'a OptContext,
            /// OpRef.0 values that are known to be virtual from fail_args analysis.
            virtual_oprefs: &'a HashSet<u32>,
        }
        impl majit_ir::BoxEnv for InlineBoxEnv<'_> {
            fn get_box_replacement(&self, opref: OpRef) -> OpRef {
                // resume.py:201-202 box.get_box_replacement()
                let repl = self.ctx.get_replacement(opref);
                if repl.is_none() && !opref.is_none() {
                    return opref;
                }
                repl
            }
            fn is_const(&self, opref: OpRef) -> bool {
                if self.ctx.is_constant(opref) {
                    // RPython parity: null Ref (GcRef(0)) is not a valid
                    // constant for resume data — would produce null Ref
                    // at runtime via TAGCONST(0, Ref).
                    if let Some(Value::Ref(r)) = self.ctx.get_constant(opref) {
                        if r.0 == 0 {
                            return false;
                        }
                    }
                    return true;
                }
                false
            }
            fn get_const(&self, opref: OpRef) -> (i64, majit_ir::Type) {
                if let Some(val) = self.ctx.get_constant(opref) {
                    match val {
                        Value::Int(i) => (*i, majit_ir::Type::Int),
                        Value::Float(f) => (f.to_bits() as i64, majit_ir::Type::Float),
                        Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                        Value::Void => (0, majit_ir::Type::Void),
                    }
                } else {
                    (0, majit_ir::Type::Int)
                }
            }
            fn get_type(&self, opref: OpRef) -> majit_ir::Type {
                if let Some(val) = self.ctx.get_constant(opref) {
                    return match val {
                        Value::Int(_) => majit_ir::Type::Int,
                        Value::Float(_) => majit_ir::Type::Float,
                        Value::Ref(_) => majit_ir::Type::Ref,
                        Value::Void => majit_ir::Type::Void,
                    };
                }
                for o in &self.ctx.new_operations {
                    if o.pos == opref {
                        return o.result_type();
                    }
                }
                majit_ir::Type::Ref
            }
            fn is_virtual_ref(&self, opref: OpRef) -> bool {
                // resume.py:210-216 is_virtual check.
                // First: check if this opref was identified as virtual
                // from the fail_args NONE pattern.
                if self.virtual_oprefs.contains(&opref.0) {
                    return true;
                }
                // Second: walk replacement chain for PtrInfo::Virtual.
                let mut check = opref;
                for _ in 0..20 {
                    if self
                        .ctx
                        .ptr_info
                        .get(check.0 as usize)
                        .and_then(|v| v.as_ref())
                        .is_some_and(|info| info.is_virtual())
                    {
                        return true;
                    }
                    let next = self.ctx.get_replacement(check);
                    if next == check || next.is_none() {
                        break;
                    }
                    check = next;
                }
                false
            }
            fn is_virtual_raw(&self, _opref: OpRef) -> bool {
                false
            }
        }

        // resume.py parity: identify virtual slots from fail_args.
        // RPython's optimizer keeps PtrInfo::Virtual on replacement boxes;
        // pyre's Phase 2 detaches virtual info from Phase 1 inputargs.
        // Build a set of snapshot oprefs that are virtual (fail_args=NONE).
        let virtual_oprefs: HashSet<u32> = op
            .fail_args
            .as_ref()
            .map(|fa| {
                fa.iter()
                    .enumerate()
                    .filter(|(i, opref)| {
                        *i >= 3
                            && opref.is_none()
                            && *i < snapshot_boxes.len()
                            && !snapshot_boxes[*i].is_none()
                    })
                    .map(|(i, _)| snapshot_boxes[i].0)
                    .collect()
            })
            .unwrap_or_default();

        let env = InlineBoxEnv {
            ctx: self,
            virtual_oprefs: &virtual_oprefs,
        };
        let mut memo = ResumeDataLoopMemo::new();
        let Ok(mut numb_state) = memo.number(&snapshot, &env) else {
            return;
        };

        // resume.py:406-417: extract TAGBOX entries → liveboxes.
        let n = (numb_state.liveboxes.len() as i32 - numb_state.num_virtuals) as usize;
        let mut liveboxes: Vec<OpRef> = vec![OpRef::NONE; n];
        let mut virtual_boxes: Vec<(OpRef, i32)> = Vec::new();
        for (&opref_id, &tagged) in &numb_state.liveboxes {
            let (idx, tagbits) = resumedata::untag(tagged);
            if tagbits == resumedata::TAGBOX && (idx as usize) < liveboxes.len() {
                liveboxes[idx as usize] = OpRef(opref_id);
            } else if tagbits == resumedata::TAGVIRTUAL {
                virtual_boxes.push((OpRef(opref_id), idx));
            }
        }

        // Map vidx → snapshot frame position. TAGVIRTUAL(vidx) was emitted
        // at a specific position in the snapshot — that's the frame slot index.
        let mut vidx_to_frame_pos: std::collections::HashMap<i32, usize> =
            std::collections::HashMap::new();
        for (snap_pos, &snap_opref) in snapshot_boxes.iter().enumerate() {
            if !snap_opref.is_none() {
                // Resolve through replacement chain (same as _number_boxes does
                // via env.get_box_replacement) to match numb_state.liveboxes keys.
                let resolved = self.get_replacement(snap_opref);
                if let Some(&tagged) = numb_state.liveboxes.get(&resolved.0) {
                    let (v, tagbits) = resumedata::untag(tagged);
                    if tagbits == resumedata::TAGVIRTUAL {
                        vidx_to_frame_pos.entry(v).or_insert(snap_pos);
                    }
                }
            }
        }

        // resume.py:419-426 + 444 _number_virtuals: append virtual field boxes.
        let mut virtual_entries: Vec<majit_ir::GuardVirtualEntry> = Vec::new();
        for (vbox, vidx) in &virtual_boxes {
            let vinfo_opt = self.get_ptr_info(*vbox).cloned();
            let (descr, known_class, fields_data) = match vinfo_opt {
                Some(crate::optimizeopt::info::PtrInfo::Virtual(ref vi)) => {
                    (vi.descr.clone(), vi.known_class, vi.fields.clone())
                }
                Some(crate::optimizeopt::info::PtrInfo::VirtualStruct(ref vi)) => {
                    (vi.descr.clone(), None, vi.fields.clone())
                }
                _ => continue,
            };
            let base_idx = liveboxes.len();
            let mut fields = Vec::new();
            for (fi, (field_idx, value_ref)) in fields_data.iter().enumerate() {
                let resolved_val = self.get_replacement(*value_ref);
                liveboxes.push(resolved_val);
                fields.push((*field_idx, base_idx + fi));
            }
            // fail_arg_index = snapshot frame position (not vidx)
            let frame_pos = vidx_to_frame_pos
                .get(vidx)
                .copied()
                .unwrap_or(*vidx as usize);
            virtual_entries.push(majit_ir::GuardVirtualEntry {
                fail_arg_index: frame_pos,
                descr,
                known_class,
                fields,
            });
        }

        // resume.py:447,450-451: patch and store.
        // resume.py:447: patch num_failargs
        numb_state.patch(1, liveboxes.len() as i32);
        // compile.py:875: descr.store_final_boxes(guard_op, newboxes)
        op.store_final_boxes(liveboxes);
        if !virtual_entries.is_empty() {
            op.rd_virtuals = Some(virtual_entries);
        }

        // resume.py:449 _add_optimizer_sections: serialize optimizer knowledge.
        // RPython appends class/heap/loopinvariant data to numb_state inline.
        // majit stores this in per_guard_knowledge (Optimizer struct) and
        // deserializes via deserialize_optimizer_knowledge at bridge time.
        // Functionally equivalent — same data used by bridge compilation.

        // resume.py:450-451: storage.rd_numb, storage.rd_consts
        op.rd_numb = Some(numb_state.create_numbering());
        op.rd_consts = Some(memo.consts().to_vec());
    }

    /// Get the IntBound for an OpRef, if known from imported bounds or constants.
    pub fn get_int_bound(&self, opref: OpRef) -> Option<crate::optimizeopt::intutils::IntBound> {
        let opref = self.get_replacement(opref);
        // Check imported bounds first (from Phase 1 / preamble)
        if let Some(bound) = self.imported_int_bounds.get(&opref) {
            return Some(bound.clone());
        }
        // Constants have exact bounds
        if let Some(c) = self.get_constant_int(opref) {
            return Some(crate::optimizeopt::intutils::IntBound::from_constant(c));
        }
        None
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

    /// RPython box.type parity: find the result type of the operation
    /// that produces this OpRef. Returns None if the OpRef is an
    /// inputarg or was not produced by any emitted operation.
    pub fn get_op_result_type(&self, opref: OpRef) -> Option<majit_ir::Type> {
        for op in self.new_operations.iter().rev() {
            if op.pos == opref && op.result_type() != majit_ir::Type::Void {
                return Some(op.result_type());
            }
        }
        None
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

    /// RPython set_forwarded parity: setting Info on a box REPLACES
    /// any forwarding. In pyre, ptr_info and forwarding must be
    /// mutually exclusive — the last writer wins.
    pub fn set_ptr_info(&mut self, opref: OpRef, info: PtrInfo) {
        let idx = opref.0 as usize;
        if idx >= self.ptr_info.len() {
            self.ptr_info.resize(idx + 1, None);
        }
        self.ptr_info[idx] = Some(info);
        // RPython set_forwarded parity: _forwarded is a single field that
        // holds EITHER an Op (forwarding) OR an Info (terminal).
        // Setting Info replaces any existing forwarding.
        if idx < self.forwarding.len() {
            self.forwarding[idx] = OpRef::NONE;
        }
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
    /// RPython passes `optimizer` for PtrInfo access. We pass `ctx`.
    fn produce_potential_short_preamble_ops(
        &self,
        _sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        _ctx: &OptContext,
    ) {
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
