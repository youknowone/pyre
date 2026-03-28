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
    /// Const arg with source OpRef for matching in force_preamble_op.
    /// RPython: Const Box has identity; get_box_replacement returns itself.
    Const(Value, OpRef),
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
    /// info.py:716-721: ConstPtrInfo._get_info — const_infos stores
    /// StructPtrInfo for constant GC objects, keyed by pointer address.
    /// RPython: optheap.const_infos[ref] = StructPtrInfo(descr)
    pub const_infos: HashMap<usize, crate::optimizeopt::info::PtrInfo>,
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
    /// Per-frame box counts for multi-frame snapshots.
    /// opencoder.py:819 capture_resumedata encodes multiple frames;
    /// this tracks the boundary between callee and caller sections.
    pub snapshot_frame_sizes: HashMap<i32, Vec<usize>>,
    /// Per-guard virtualizable boxes from tracing-time snapshots.
    pub snapshot_vable_boxes: HashMap<i32, Vec<OpRef>>,
    /// Per-guard per-frame (jitcode_index, pc) from tracing-time snapshots.
    pub snapshot_frame_pcs: HashMap<i32, Vec<(i32, i32)>>,
    /// ConstantPool type map for BoxEnv.is_const() during inline numbering.
    pub constant_types_for_numbering: HashMap<u32, majit_ir::Type>,
    /// optimizer.py:644,679 _last_guard_op — index of the last guard in
    /// new_operations that had full resume data built. Consecutive guards
    /// share resume data via _copy_resume_data_from (ResumeGuardCopiedDescr).
    last_guard_idx: Option<usize>,
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
        self.ctx.get_box_replacement(opref)
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
        let resolved = self.ctx.get_box_replacement(opref);
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
        let resolved = self.ctx.get_box_replacement(opref);
        let info = self.ctx.get_ptr_info(resolved)?;
        match info {
            PtrInfo::Virtual(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: vi.known_class,
                field_oprefs: vi
                    .fields
                    .iter()
                    .map(|(_, vref)| self.ctx.get_box_replacement(*vref))
                    .collect(),
            }),
            PtrInfo::VirtualStruct(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: None,
                field_oprefs: vi
                    .fields
                    .iter()
                    .map(|(_, vref)| self.ctx.get_box_replacement(*vref))
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
            const_infos: HashMap::new(),
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
            snapshot_frame_sizes: HashMap::new(),
            snapshot_vable_boxes: HashMap::new(),
            snapshot_frame_pcs: HashMap::new(),
            constant_types_for_numbering: HashMap::new(),
            last_guard_idx: None,
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
            const_infos: HashMap::new(),
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
            snapshot_frame_sizes: HashMap::new(),
            snapshot_vable_boxes: HashMap::new(),
            snapshot_frame_pcs: HashMap::new(),
            constant_types_for_numbering: HashMap::new(),
            last_guard_idx: None,
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

        // RPython optimizer.py:652-686 emit_guard_operation — guard resume
        // data sharing via _copy_resume_data_from / ResumeGuardCopiedDescr.
        if op.opcode.is_guard() {
            self.emit_guard_operation(&mut op);
        } else {
            // optimizer.py:639-644: side-effectful non-guard ops clear sharing.
            // optimizer.py:705-711: is_call_pure_pure_canraise — CallPure that
            // can_raise(ignore_memoryerror=True) counts as side-effectful even
            // though has_no_side_effect is true for call_pure opcodes.
            let dominated_by_side_effect = if (op.opcode.has_no_side_effect()
                || op.opcode.is_ovf()
                || op.opcode.is_jit_debug())
                && !Self::is_call_pure_pure_canraise(&op)
            {
                false
            } else {
                true
            };
            if dominated_by_side_effect {
                self.last_guard_idx = None;
            }
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

    /// unroll.py:26-39: force_op_from_preamble
    /// Calls use_box (shortpreamble.py:382-407) then registers in
    /// potential_extra_ops for later force_box consumption.
    pub fn force_op_from_preamble(&mut self, result: OpRef) -> OpRef {
        // Check imported short identity BEFORE get_box_replacement
        // (RPython checks isinstance on the raw input, line 27).
        let preamble_source = self.imported_short_source(result);
        let is_constant = self.get_constant(preamble_source).is_some();
        if self.imported_short_preamble_used.insert(preamble_source) {
            // unroll.py:32: use_box(op, preamble_op.preamble_op, self)
            let tracked = if let Some(builder) = self.active_short_preamble_producer.as_mut() {
                builder.use_box(preamble_source).is_some()
            } else if let Some(builder) = self.imported_short_preamble_builder.as_mut() {
                builder.use_box(preamble_source).is_some()
            } else {
                false
            };
            // unroll.py:33-37: if not constant → potential_extra_ops[op] = preamble_op
            if tracked && !is_constant {
                let produced = self
                    .imported_short_preamble_builder
                    .as_ref()
                    .and_then(|b| b.produced_short_op(preamble_source))
                    .or_else(|| {
                        self.active_short_preamble_producer
                            .as_ref()
                            .and_then(|b| b.produced_short_op(preamble_source))
                    });
                if let Some(produced) = produced {
                    // unroll.py:34-35: if invented_name: op = get_box_replacement(op)
                    let key = if produced.invented_name {
                        self.get_box_replacement(preamble_source)
                    } else {
                        preamble_source
                    };
                    self.potential_extra_ops.insert(
                        key,
                        TrackedPreambleUse {
                            result: preamble_source,
                            produced,
                        },
                    );
                }
            }
        }
        // unroll.py:38: return preamble_op.op — raw imported op, no
        // forwarding resolve. Callers store this in cache; subsequent
        // reads go through get_box_replacement to resolve.
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

    /// resoperation.py:57-68 get_box_replacement: follow the forwarding
    /// chain (op._forwarded) until we reach a terminal. RPython: walks
    /// op → op._forwarded → ... until None or Info instance.
    /// In majit: forwarding[idx] == NONE means terminal.
    ///
    /// NEVER consults mapping dicts — RPython's get_box_replacement only
    /// follows the _forwarded chain on the box itself.
    pub fn get_box_replacement(&self, mut opref: OpRef) -> OpRef {
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

    /// resume.py:157 getconst parity for synthetic rd_numb encoding.
    /// Matches OptBoxEnv::get_const: checks constant_types_for_numbering
    /// override, PtrInfo::Constant, and constant pool.
    /// Returns None if opref is not a constant.
    pub fn getconst_for_numbering(&self, opref: OpRef) -> Option<(i64, majit_ir::Type)> {
        let type_override = self.constant_types_for_numbering.get(&opref.0).copied();
        // Check constant pool (through replacement chain).
        if let Some(val) = self.get_constant(opref) {
            let (raw, tp) = match val {
                Value::Int(v) => (*v, type_override.unwrap_or(majit_ir::Type::Int)),
                Value::Float(f) => (f.to_bits() as i64, majit_ir::Type::Float),
                Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                _ => return None,
            };
            return Some((raw, tp));
        }
        // Check raw constants (before replacement).
        if let Some(val) = self
            .constants
            .get(opref.0 as usize)
            .and_then(|v| v.as_ref())
        {
            let (raw, tp) = match val {
                Value::Int(v) => (*v, type_override.unwrap_or(majit_ir::Type::Int)),
                Value::Float(f) => (f.to_bits() as i64, majit_ir::Type::Float),
                Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                _ => return None,
            };
            return Some((raw, tp));
        }
        // info.py: ConstPtrInfo — GcRef constant stored in PtrInfo.
        if let Some(crate::optimizeopt::info::PtrInfo::Constant(gcref)) = self.get_ptr_info(opref) {
            return Some((gcref.0 as i64, majit_ir::Type::Ref));
        }
        None
    }

    /// Get the constant value for an operation, if known.
    pub fn get_constant(&self, opref: OpRef) -> Option<&Value> {
        let opref = self.get_box_replacement(opref);
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

    /// optimizer.py:705-711: is_call_pure_pure_canraise — a CallPure op whose
    /// effectinfo says check_can_raise(ignore_memoryerror=True). These ops are
    /// formally side-effect-free (has_no_side_effect), but their potential to
    /// raise means they break guard resume-data sharing.
    fn is_call_pure_pure_canraise(op: &Op) -> bool {
        if !op.opcode.is_call_pure() {
            return false;
        }
        let Some(ref descr) = op.descr else {
            return false;
        };
        let Some(cd) = descr.as_call_descr() else {
            return false;
        };
        cd.effect_info().check_can_raise(true)
    }

    /// optimizer.py:652-686 emit_guard_operation — decide whether to share
    /// resume data from the previous guard (_copy_resume_data_from) or build
    /// new resume data (store_final_boxes_in_guard / number_guard_inline).
    fn emit_guard_operation(&mut self, op: &mut Op) {
        let opnum = op.opcode;

        // optimizer.py:655-664: GUARD_(NO_)EXCEPTION following a guard that
        // is NOT GUARD_NOT_FORCED — give up sharing.
        if (opnum == OpCode::GuardNoException || opnum == OpCode::GuardException) {
            if let Some(idx) = self.last_guard_idx {
                if self.new_operations[idx].opcode != OpCode::GuardNotForced
                    && self.new_operations[idx].opcode != OpCode::GuardNotForced2
                {
                    self.last_guard_idx = None;
                }
            }
        }

        // optimizer.py:665-670: GUARD_ALWAYS_FAILS must never share.
        if opnum == OpCode::GuardAlwaysFails {
            self.last_guard_idx = None;
        }

        // optimizer.py:672-683: _copy_resume_data_from vs store_final_boxes_in_guard.
        // RPython condition: self._last_guard_op and guard_op.getdescr() is None.
        // In RPython, getdescr() is None before store_final_boxes_in_guard
        // processes the guard (which sets the descr + replaces fail_args with
        // normalized liveboxes). In majit, number_guard_inline produces rd_numb
        // and normalizes fail_args to liveboxes (store_final_boxes parity).
        // rd_numb.is_some() means already processed.
        //
        // GUARD_NOT_FORCED never uses copied descr (compile.py:926 assert).
        // optimizer.py:672 — fail_args.is_none() is kept because majit's
        // number_guard_inline doesn't yet normalize fail_args to liveboxes
        // (store_final_boxes parity). This requires coordinated changes to
        // compile.rs resume_layout construction + backend jitframe sizing.
        let can_share = self.last_guard_idx.is_some()
            && op.rd_numb.is_none()
            && op.fail_args.is_none()
            && opnum != OpCode::GuardNotForced
            && opnum != OpCode::GuardNotForced2;

        if can_share {
            let idx = self.last_guard_idx.unwrap();
            // _copy_resume_data_from: share resume data from last guard.
            op.rd_numb = self.new_operations[idx].rd_numb.clone();
            op.rd_consts = self.new_operations[idx].rd_consts.clone();
            op.rd_virtuals_info = self.new_operations[idx].rd_virtuals_info.clone();
            op.rd_virtuals = self.new_operations[idx].rd_virtuals.clone();
            op.rd_pendingfields = self.new_operations[idx].rd_pendingfields.clone();
            op.fail_args = self.new_operations[idx].fail_args.clone();
            // Don't update last_guard_idx — copied guards don't become sources.
        } else {
            // store_final_boxes_in_guard: build new resume data.
            self.number_guard_inline(op);
            self.last_guard_idx = Some(self.new_operations.len());
            // optimizer.py:680-683: force_box on fail_args for unrolling.
            // Mirrors Optimizer.force_box contract: resolve replacement,
            // handle tracked preamble ops, force virtuals.
            if let Some(ref fa) = op.fail_args {
                let fargs: Vec<OpRef> = fa.iter().copied().collect();
                for farg in fargs {
                    if !farg.is_none() {
                        self.force_box_inline(farg);
                    }
                }
            }
        }

        // optimizer.py:684-685: GUARD_EXCEPTION clears sharing.
        if opnum == OpCode::GuardException {
            self.last_guard_idx = None;
        }
    }

    /// optimizer.py:345-364 force_box — inline equivalent for
    /// emit_guard_operation's fail_arg forcing (optimizer.py:680-683).
    /// Mirrors Optimizer.force_box contract: resolve imported short source,
    /// handle tracked preamble ops, then force virtuals to concrete.
    fn force_box_inline(&mut self, opref: OpRef) -> OpRef {
        let preamble_source = self.imported_short_source(opref);
        let resolved = self.get_box_replacement(opref);
        let tracked = self
            .take_potential_extra_op(resolved)
            .or_else(|| self.take_potential_extra_op(opref))
            .or_else(|| {
                (preamble_source != resolved && preamble_source != opref)
                    .then(|| self.take_potential_extra_op(preamble_source))
                    .flatten()
            });
        if let Some(tracked) = tracked {
            if let Some(builder) = self.active_short_preamble_producer_mut() {
                builder.add_tracked_preamble_op(tracked.result, &tracked.produced);
            } else if let Some(builder) = self.imported_short_preamble_builder.as_mut() {
                builder.add_tracked_preamble_op(tracked.result, &tracked.produced);
            }
        }
        if let Some(mut info) = self.get_ptr_info(resolved).cloned() {
            if info.is_virtual() {
                let forced = info.force_to_ops_direct(resolved, self);
                return self.get_box_replacement(forced);
            }
        }
        resolved
    }

    /// RPython optimizer.py:722-752 store_final_boxes_in_guard inline.
    /// Called from emit() for every guard during optimization. Produces
    /// rd_numb via memo.number() using the CURRENT optimizer state
    /// (replacement chain, constants, virtual info).
    /// resume.py ResumeDataVirtualAdder.finish() parity:
    /// Generate rd_numb + rd_consts + rd_virtuals for a guard.
    /// Called from store_final_boxes_in_guard in optimizer.rs.
    /// Uses snapshot data (vable_boxes, frame_pcs, multi-frame) when available.
    pub fn finalize_guard_resume_data(&self, op: &mut Op) {
        self.number_guard_inline_impl(op);
    }

    fn number_guard_inline(&self, op: &mut Op) {
        // RPython parity: store_final_boxes_in_guard already produced
        // rd_numb via finalize_guard_resume_data. Assert completeness.
        if op.rd_numb.is_some() {
            return;
        }
        if op.fail_args.is_none() {
            return;
        }
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] WARNING: guard {:?} at {:?} missing rd_numb",
                op.opcode, op.pos,
            );
        }
    }

    fn number_guard_inline_impl(&self, op: &mut Op) {
        use majit_ir::resumedata::{self, ResumeDataLoopMemo, Snapshot};

        // RPython parity: store_final_boxes_in_guard (in emit_with_guard_check)
        // already produced rd_numb + liveboxes. Don't overwrite — UNLESS the
        // guard has virtual placeholders (OpRef::NONE) in fail_args that need
        // fresh TAGVIRTUAL encoding. Phase 2 guards inherit Phase 1 rd_numb
        // which has wrong slot indices for Phase 2 fail_args.
        if op.rd_numb.is_some() {
            let has_virtual_slots = op
                .fail_args
                .as_ref()
                .map_or(false, |fa| fa.iter().any(|r| r.is_none()));
            if !has_virtual_slots {
                return;
            }
            // Clear stale rd_numb/rd_virtuals_info to regenerate from
            // current fail_args with correct TAGVIRTUAL encoding.
            op.rd_numb = None;
            op.rd_virtuals_info = None;
        }

        // RPython parity: every guard has a snapshot from capture_resumedata.
        // Guards without snapshot (optimizer-created, rd_resume_position < 0)
        // share resume data from _copy_resume_data_from. If they reach here
        // without a snapshot, build a minimal rd_numb directly from fail_args
        // using the same format as memo.number() but without replacement/
        // virtual resolution (the fail_args are already final from
        // store_final_boxes_in_guard).
        if op.rd_resume_position < 0 || !self.snapshot_boxes.contains_key(&op.rd_resume_position) {
            if let Some(ref fa) = op.fail_args {
                use majit_ir::resumedata;
                let fa_len = fa.len();
                let mut ns = resumedata::NumberingState::new(fa_len + 8);
                ns.append_int(0); // size (patched)
                ns.append_int(fa_len as i32); // num_failargs
                ns.append_int(0); // vable_array len
                ns.append_int(0); // vref_array len
                ns.append_int(0); // jitcode_index
                ns.append_int(0); // pc
                ns.append_int(fa_len as i32); // slot_count
                for (i, &opref) in fa.iter().enumerate() {
                    if opref.is_none() {
                        // TAGVIRTUAL if rd_virtuals has entry for this slot.
                        let vidx = op
                            .rd_virtuals
                            .as_ref()
                            .and_then(|entries| entries.iter().position(|e| e.fail_arg_index == i));
                        if let Some(vidx) = vidx {
                            let t = resumedata::tag(vidx as i32, resumedata::TAGVIRTUAL)
                                .unwrap_or(resumedata::NULLREF);
                            ns.append_short(t);
                        } else {
                            ns.append_short(resumedata::NULLREF);
                        }
                    } else if let Some((raw, tp)) = self.getconst_for_numbering(opref) {
                        // resume.py getconst: TAGINT for small ints, TAGCONST for rest.
                        if tp == majit_ir::Type::Int {
                            if let Ok(tagged) = resumedata::tag(raw as i32, resumedata::TAGINT) {
                                ns.append_short(tagged);
                                continue;
                            }
                        }
                        if tp == majit_ir::Type::Ref && raw == 0 {
                            ns.append_short(resumedata::NULLREF);
                            continue;
                        }
                        let mut rd_consts_local: Vec<(i64, majit_ir::Type)> =
                            op.rd_consts.take().unwrap_or_default();
                        let existing = rd_consts_local
                            .iter()
                            .position(|(v, t)| *v == raw && *t == tp);
                        let idx = existing.unwrap_or_else(|| {
                            let j = rd_consts_local.len();
                            rd_consts_local.push((raw, tp));
                            j
                        });
                        op.rd_consts = Some(rd_consts_local);
                        let t = resumedata::tag(
                            (idx + resumedata::TAG_CONST_OFFSET as usize) as i32,
                            resumedata::TAGCONST,
                        )
                        .unwrap_or(resumedata::NULLREF);
                        ns.append_short(t);
                    } else {
                        let t = resumedata::tag(i as i32, resumedata::TAGBOX)
                            .unwrap_or(resumedata::NULLREF);
                        ns.append_short(t);
                    }
                }
                ns.patch_current_size(0);
                op.rd_numb = Some(ns.create_numbering());
                if op.rd_consts.is_none() {
                    op.rd_consts = Some(Vec::new());
                }
            }
            return;
        }
        let snapshot_boxes = self.snapshot_boxes[&op.rd_resume_position].clone();
        let vable_oprefs = self
            .snapshot_vable_boxes
            .get(&op.rd_resume_position)
            .cloned()
            .unwrap_or_default();
        let frame_pcs = self
            .snapshot_frame_pcs
            .get(&op.rd_resume_position)
            .cloned()
            .unwrap_or_default();

        // resume.py:201-202 get_box_replacement parity:
        // Pass ORIGINAL (unresolved) snapshot boxes. _number_boxes calls
        // env.get_box_replacement per-box, which resolves through the
        // replacement chain while preserving virtual identity.
        let frame_sizes = self.snapshot_frame_sizes.get(&op.rd_resume_position);
        let mut snapshot = if let Some(sizes) = frame_sizes.filter(|s| s.len() > 1) {
            // Multi-frame: split snapshot_boxes into per-frame chunks.
            let mut frames = Vec::new();
            let mut offset = 0;
            for (i, &size) in sizes.iter().enumerate() {
                let end = (offset + size).min(snapshot_boxes.len());
                let frame_boxes: Vec<OpRef> = snapshot_boxes[offset..end].to_vec();
                let (jitcode_index, pc) = frame_pcs.get(i).copied().unwrap_or((0, 0));
                frames.push((jitcode_index, pc, frame_boxes));
                offset = end;
            }
            Snapshot::multi_frame(frames)
        } else {
            let pc = frame_pcs.first().map(|&(_, pc)| pc).unwrap_or(0);
            Snapshot::single_frame(pc, snapshot_boxes.clone())
        };
        // pyjitpl.py:2588: vable_array stores virtualizable_boxes.
        // Currently kept empty for numbering because pyre's recovery
        // path reads frame/ni/vsd from fail_args[0..3], not from
        // vable_array. Populating vable_array would shift TAGBOX
        // indices and break the fail_args ↔ deadframe correspondence.
        // TODO: populate once recovery reads frame from vable_array.
        let _ = vable_oprefs;

        // BoxEnv bridging current optimizer state.
        struct InlineBoxEnv<'a> {
            ctx: &'a OptContext,
        }
        impl majit_ir::BoxEnv for InlineBoxEnv<'_> {
            fn get_box_replacement(&self, opref: OpRef) -> OpRef {
                // resume.py:201-202 box.get_box_replacement()
                let repl = self.ctx.get_box_replacement(opref);
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
                // resume.py:210-216: info = getptrinfo(box)
                //                    is_virtual = (info is not None and info.is_virtual())
                // RPython uses ONLY PtrInfo.is_virtual() — no fallback.
                // virtual_oprefs hint (from fail_args NONE) is checked only
                // when PtrInfo confirms virtual. This prevents placeholder
                // rd_virtuals entries (RPython resume.py:498 asserts
                // info.is_virtual() for every virtual in _number_virtuals).
                matches!(
                    self.ctx.get_ptr_info(opref),
                    Some(
                        crate::optimizeopt::info::PtrInfo::Virtual(_)
                            | crate::optimizeopt::info::PtrInfo::VirtualStruct(_),
                    )
                )
            }
            fn is_virtual_raw(&self, _opref: OpRef) -> bool {
                false
            }
        }

        let env = InlineBoxEnv { ctx: self };
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
                let resolved = self.get_box_replacement(snap_opref);
                if let Some(&tagged) = numb_state.liveboxes.get(&resolved.0) {
                    let (v, tagbits) = resumedata::untag(tagged);
                    if tagbits == resumedata::TAGVIRTUAL {
                        vidx_to_frame_pos.entry(v).or_insert(snap_pos);
                    }
                }
            }
        }

        // resume.py:490-506 _number_virtuals: create rd_virtuals indexed by
        // the TAGVIRTUAL number assigned in _number_boxes. RPython:
        //   virtuals = [None] * length
        //   for virtualbox, fieldboxes in vfieldboxes.iteritems():
        //       num, _ = untag(self.liveboxes[virtualbox])
        //       assert info.is_virtual()   # resume.py:498
        //       virtuals[num] = vinfo
        //
        // resume.py:490-506 _number_virtuals + _gettagged parity.
        // Worklist loop: nested virtual field values get TAGVIRTUAL and are
        // added to the worklist for recursive processing (resume.py:495).
        let mut rd_virt_info: Vec<majit_ir::RdVirtualInfo> =
            vec![majit_ir::RdVirtualInfo::Empty; numb_state.num_virtuals as usize];
        let mut worklist: std::collections::VecDeque<(OpRef, i32)> =
            virtual_boxes.into_iter().collect();
        while let Some((vbox, vidx)) = worklist.pop_front() {
            let idx = vidx as usize;
            if idx >= rd_virt_info.len() {
                rd_virt_info.resize(idx + 1, majit_ir::RdVirtualInfo::Empty);
            }
            let vinfo_opt = self.get_ptr_info(vbox).cloned();
            // resume.py:560-568 _gettagged(box) parity.
            let mut gettagged = |value_ref: OpRef| -> i16 {
                let resolved_val = self.get_box_replacement(value_ref);
                // resume.py:561: None → UNINITIALIZED
                if resolved_val.is_none() {
                    return resumedata::UNINITIALIZED_TAG;
                }
                if self.is_constant(resolved_val) {
                    if let Some(val) = self.get_constant(resolved_val) {
                        let (c, tp) = match val {
                            Value::Int(i) => (*i, majit_ir::Type::Int),
                            Value::Float(f) => (f.to_bits() as i64, majit_ir::Type::Float),
                            Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                            Value::Void => (0, majit_ir::Type::Void),
                        };
                        return memo.getconst(c, tp);
                    }
                    return resumedata::NULLREF;
                }
                if let Some(&existing_tag) = numb_state.liveboxes.get(&resolved_val.0) {
                    return existing_tag;
                }
                // resume.py:495 parity: nested virtual → TAGVIRTUAL + worklist
                if self
                    .get_ptr_info(resolved_val)
                    .is_some_and(|info| info.is_virtual())
                {
                    let nested_vidx = rd_virt_info.len() as i32;
                    rd_virt_info.push(majit_ir::RdVirtualInfo::Empty);
                    let tagged = resumedata::tag(nested_vidx, resumedata::TAGVIRTUAL)
                        .unwrap_or(resumedata::NULLREF);
                    numb_state.liveboxes.insert(resolved_val.0, tagged);
                    worklist.push_back((resolved_val, nested_vidx));
                    return tagged;
                }
                // Non-virtual → TAGBOX
                let fa_idx = liveboxes.len();
                liveboxes.push(resolved_val);
                let tagged = resumedata::tag(fa_idx as i32, resumedata::TAGBOX)
                    .unwrap_or(resumedata::NULLREF);
                numb_state.liveboxes.insert(resolved_val.0, tagged);
                tagged
            };
            // resume.py:326-338: dispatch by virtual type.
            let entry = match vinfo_opt {
                Some(crate::optimizeopt::info::PtrInfo::Virtual(ref vi)) => {
                    let fielddescr_indices: Vec<u32> =
                        vi.fields.iter().map(|(idx, _)| *idx).collect();
                    let field_offsets: Vec<usize> = vi
                        .fields
                        .iter()
                        .map(|(fi, _)| {
                            vi.field_descrs
                                .iter()
                                .find(|(di, _)| *di == *fi)
                                .and_then(|(_, d)| d.as_field_descr().map(|fd| fd.offset()))
                                .unwrap_or((*fi as usize + 1) * 8)
                        })
                        .collect();
                    let fieldnums: Vec<i16> =
                        vi.fields.iter().map(|(_, vr)| gettagged(*vr)).collect();
                    majit_ir::RdVirtualInfo::Instance {
                        descr_index: vi.descr.index(),
                        known_class: vi.known_class.map(|gc| gc.as_usize() as i64),
                        fielddescr_indices,
                        field_offsets,
                        fieldnums,
                    }
                }
                Some(crate::optimizeopt::info::PtrInfo::VirtualStruct(ref vi)) => {
                    let fielddescr_indices: Vec<u32> =
                        vi.fields.iter().map(|(idx, _)| *idx).collect();
                    let field_offsets: Vec<usize> = vi
                        .fields
                        .iter()
                        .map(|(fi, _)| {
                            vi.field_descrs
                                .iter()
                                .find(|(di, _)| *di == *fi)
                                .and_then(|(_, d)| d.as_field_descr().map(|fd| fd.offset()))
                                .unwrap_or((*fi as usize + 1) * 8)
                        })
                        .collect();
                    let fieldnums: Vec<i16> =
                        vi.fields.iter().map(|(_, vr)| gettagged(*vr)).collect();
                    majit_ir::RdVirtualInfo::Struct {
                        descr_index: vi.descr.index(),
                        fielddescr_indices,
                        field_offsets,
                        fieldnums,
                    }
                }
                Some(crate::optimizeopt::info::PtrInfo::VirtualArray(ref vi)) => {
                    let fieldnums: Vec<i16> = vi.items.iter().map(|vr| gettagged(*vr)).collect();
                    // resume.py:656: arraydescr element kind
                    let kind = vi
                        .descr
                        .as_array_descr()
                        .map(|ad| match ad.item_type() {
                            majit_ir::Type::Float => 2u8,
                            majit_ir::Type::Int => 1u8,
                            _ => 0u8, // Ref
                        })
                        .unwrap_or(0);
                    majit_ir::RdVirtualInfo::Array {
                        descr_index: vi.descr.index(),
                        clear: vi.clear,
                        kind,
                        fieldnums,
                    }
                }
                Some(crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(ref vi)) => {
                    let fielddescr_indices: Vec<u32> = vi
                        .element_fields
                        .first()
                        .map(|ef| ef.iter().map(|(idx, _)| *idx).collect())
                        .unwrap_or_default();
                    let mut fieldnums = Vec::new();
                    for ef in &vi.element_fields {
                        for (_, vr) in ef {
                            fieldnums.push(gettagged(*vr));
                        }
                    }
                    majit_ir::RdVirtualInfo::ArrayStruct {
                        descr_index: vi.descr.index(),
                        size: vi.element_fields.len(),
                        fielddescr_indices,
                        fieldnums,
                    }
                }
                Some(crate::optimizeopt::info::PtrInfo::VirtualRawBuffer(ref vi)) => {
                    let offsets: Vec<usize> = vi.entries.iter().map(|(o, _, _)| *o).collect();
                    let entry_sizes: Vec<usize> =
                        vi.entries.iter().map(|(_, len, _)| *len).collect();
                    let fieldnums: Vec<i16> =
                        vi.entries.iter().map(|(_, _, vr)| gettagged(*vr)).collect();
                    majit_ir::RdVirtualInfo::RawBuffer {
                        size: vi.size,
                        offsets,
                        entry_sizes,
                        fieldnums,
                    }
                }
                _ => continue,
            };
            rd_virt_info[idx] = entry;
        }

        // resume.py:447,450-451: patch and store.
        numb_state.patch(1, liveboxes.len() as i32);
        op.store_final_boxes(liveboxes);
        // Store rd_virtuals_info directly (indexed by vidx, RPython parity).
        // Bypass op.rd_virtuals (GuardVirtualEntry path) — rd_virtuals_info
        // is the authoritative source, indexed consistently with rd_numb.
        if !rd_virt_info.is_empty() {
            op.rd_virtuals_info = Some(rd_virt_info);
            op.rd_virtuals = None;
        }

        // resume.py:450-451: storage.rd_numb, storage.rd_consts
        op.rd_numb = Some(numb_state.create_numbering());
        op.rd_consts = Some(memo.consts().to_vec());
    }

    /// Get the IntBound for an OpRef, if known from imported bounds or constants.
    pub fn get_int_bound(&self, opref: OpRef) -> Option<crate::optimizeopt::intutils::IntBound> {
        let opref = self.get_box_replacement(opref);
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

    /// Look up the operation that produces a given OpRef.
    /// Searches emitted operations and input ops.
    /// Used for pattern matching nested operations (e.g., int_add(int_add(x, C1), C2)).
    /// Returns a clone to avoid borrow conflicts with mutable ctx methods.
    pub fn get_producing_op(&self, opref: OpRef) -> Option<Op> {
        let opref = self.get_box_replacement(opref);
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
        self.const_infos.clear();
    }

    /// Get a mutable reference to the last emitted operation.
    pub fn last_emitted_operation_mut(&mut self) -> Option<&mut Op> {
        self.new_operations.last_mut()
    }

    // ── info.py: per-OpRef pointer info ──

    /// info.py: getptrinfo(op) — get PtrInfo for an OpRef.
    pub fn get_ptr_info(&self, opref: OpRef) -> Option<&PtrInfo> {
        let opref = self.get_box_replacement(opref);
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
        let opref = self.get_box_replacement(opref);
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

    /// info.py:716-721: ConstPtrInfo._get_info(descr, optheap)
    /// For constant GC objects, get or create a StructPtrInfo in const_infos.
    /// Returns None if opref is not a Ref constant.
    pub fn get_const_info_mut(
        &mut self,
        opref: OpRef,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        let addr = match self.get_constant(opref) {
            Some(majit_ir::Value::Ref(r)) if !r.is_null() => r.0,
            _ => return None,
        };
        use std::collections::hash_map::Entry;
        match self.const_infos.entry(addr) {
            Entry::Occupied(e) => Some(e.into_mut()),
            Entry::Vacant(e) => {
                Some(e.insert(crate::optimizeopt::info::PtrInfo::instance(None, None)))
            }
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

    /// shortpreamble.py:112-126: PureOp.produce_op / LoopInvariantOp.produce_op
    /// Transfer imported PreambleOp entries from OptContext to this pass.
    /// RPython calls `opt.optimizer.optpure` directly during produce_op.
    /// In majit, the Optimization trait mediates this transfer.
    fn install_preamble_pure_ops(&mut self, _ctx: &OptContext) {}

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
