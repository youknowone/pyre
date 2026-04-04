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
use majit_ir::{DescrRef, GcRef, Op, OpCode, OpRef, Value};
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
    fn clone_descr(&self) -> Option<DescrRef> {
        Some(std::sync::Arc::new(OptResumeAtPositionDescr))
    }
    fn clone_as_loop_version_descr(&self) -> Option<DescrRef> {
        Some(crate::fail_descr::make_compile_loop_version_descr(
            0,
            crate::resume::ResumeData {
                vable_array: Vec::new(),
                frames: Vec::new(),
                virtuals: Vec::new(),
                pending_fields: Vec::new(),
            },
        ))
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
    /// resoperation.py: _forwarded — unified forwarding + PtrInfo.
    /// Forwarded::Op(target) = forwarding, Forwarded::Info(info) = terminal.
    pub forwarded: Vec<crate::optimizeopt::info::Forwarded>,
    /// RPython: mapping dict in inline_short_preamble — separate from _forwarded.
    /// Maps Phase 1 source OpRefs to Phase 2 short arg OpRefs.
    /// Number of input arguments, used to offset emitted op positions
    /// so that variable indices don't collide with input arg indices.
    num_inputs: u32,
    /// Next unique op position for newly emitted or queued extra operations.
    pub(crate) next_pos: u32,
    /// Next unique constant position (OpRef >= CONST_BASE).
    pub(crate) next_const_pos: u32,
    /// RPython emit_extra(op, emit=False) parity: ops queued to be
    /// processed starting from a specific pass index (skipping earlier passes).
    /// Used by heap's force_lazy_set to route ops through remaining passes
    /// without re-entering the heap pass itself.
    pub(crate) extra_operations_after: VecDeque<(usize, Op)>,
    // ptr_info merged into forwarded (Forwarded::Info variant)
    /// Known lower bounds for integer-typed OpRefs, shared across passes.
    ///
    /// heap.py: arrayinfo.getlenbound().make_gt_const(index) records that
    /// an ARRAYLEN_GC result >= index+1. intbounds.py uses this to
    /// eliminate redundant length guards.
    pub int_lower_bounds: HashMap<OpRef, i64>,
    /// optimizer.py:415-426 parity: per-OpRef IntBound from the IntBounds pass.
    /// Synced at the end of each IntBounds.propagate_forward so that later
    /// passes (Rewrite, Pure, etc.) calling make_constant can validate that
    /// the constant is within the known range.
    pub int_bounds: Vec<Option<IntBound>>,
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
    /// store_final_boxes_in_guard to resolve NONE positions
    /// inherited from Phase 1 virtualization.
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
    pub(crate) potential_extra_ops: HashMap<OpRef, TrackedPreambleUse>,
    /// RPython unroll.py: live ExtendedShortPreambleBuilder while replaying an
    /// existing target token's short preamble.
    active_short_preamble_producer:
        Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder>,
    /// RPython shortpreamble.py: pass-collected preamble producers aligned to
    /// the exported loop-header inputargs.
    pub exported_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
    /// RPython import_state: maps original inputarg index → fresh virtual head OpRef.
    /// Used by ensure_linked_list_head to return the imported virtual.
    pub imported_virtual_heads: Vec<(usize, OpRef)>,
    /// optimizer.py: `can_replace_guards` — disable guard replacement during
    /// bridge compilation. Defaults to true for preamble.
    pub can_replace_guards: bool,
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
    /// before a guard, consumed by emit_operation() to encode into
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
    /// RPython box.type parity: each snapshot Box carries its type.
    /// Used by InlineBoxEnv.get_type() to avoid fallback to Int for
    /// unresolved OpRefs (resume.py:210 box.type == 'r' vs 'i').
    pub snapshot_box_types: HashMap<u32, majit_ir::Type>,
    /// compile.rs value_types parity: OpRef → Type for all defined values
    /// (inputargs + operation results). Used by store_final_boxes_in_guard
    /// to infer fail_arg types correctly for OpRefs from earlier phases.
    pub value_types: HashMap<u32, majit_ir::Type>,
    /// optimizer.py:644,679 _last_guard_op — index of the last guard in
    /// new_operations that had full resume data built. Consecutive guards
    /// share resume data via _copy_resume_data_from (ResumeGuardCopiedDescr).
    last_guard_idx: Option<usize>,
    /// Last rd_resume_position with a valid snapshot. Used as fallback
    /// for optimizer-created guards that can't share from a previous guard.
    /// resume.py parity: RPython guards always get a snapshot via
    /// capture_resumedata; pyre tracks the nearest valid position.
    pub last_seen_snapshot_pos: Option<i32>,
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
        // True Const = constant pool entry (>= CONST_BASE) or PtrInfo::Constant.
        // NOT optimizer-known values from make_constant() on operation results.
        if opref.is_constant() {
            return true;
        }
        // optimizer.py:432: make_constant → Forwarded::Const terminal.
        // resume.py:204: isinstance(box, Const)
        if matches!(
            self.ctx.forwarded.get(opref.0 as usize),
            Some(crate::optimizeopt::info::Forwarded::Const(_))
        ) {
            return true;
        }
        // info.py: ConstPtrInfo.is_constant() → True
        matches!(
            self.ctx.get_ptr_info(opref),
            Some(crate::optimizeopt::info::PtrInfo::Constant(_))
        )
    }

    fn get_const(&self, opref: OpRef) -> (i64, majit_ir::Type) {
        // optimizer.py:432: make_constant → Forwarded::Const — check first.
        if let Some(crate::optimizeopt::info::Forwarded::Const(val)) =
            self.ctx.forwarded.get(opref.0 as usize)
        {
            let (raw, tp) = match val {
                Value::Int(v) => (*v, majit_ir::Type::Int),
                Value::Float(f) => (f.to_bits() as i64, majit_ir::Type::Float),
                Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                Value::Void => (0, majit_ir::Type::Int),
            };
            let tp = self
                .ctx
                .constant_types_for_numbering
                .get(&opref.0)
                .copied()
                .unwrap_or(tp);
            return (raw, tp);
        }
        // RPython ConstPtr parity: check numbering type overrides first.
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
        // RPython box.type parity: snapshot Box carries its type.
        if let Some(&tp) = self.ctx.snapshot_box_types.get(&opref.0) {
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
        majit_ir::Type::Int
    }

    fn is_virtual_ref(&self, opref: OpRef) -> bool {
        // resume.py:210-216: info = getptrinfo(box); is_virtual = info.is_virtual()
        self.ctx
            .get_ptr_info(opref)
            .is_some_and(|info| info.is_virtual())
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
            PtrInfo::VirtualArray(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: None,
                field_oprefs: vi
                    .items
                    .iter()
                    .map(|vref| self.ctx.get_box_replacement(*vref))
                    .collect(),
            }),
            PtrInfo::VirtualArrayStruct(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: None,
                field_oprefs: vi
                    .element_fields
                    .iter()
                    .flat_map(|ef| {
                        ef.iter()
                            .map(|(_, vref)| self.ctx.get_box_replacement(*vref))
                    })
                    .collect(),
            }),
            PtrInfo::VirtualRawBuffer(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: None,
                known_class: None,
                field_oprefs: vi
                    .entries
                    .iter()
                    .map(|(_, _, vref)| self.ctx.get_box_replacement(*vref))
                    .collect(),
            }),
            _ => None,
        }
    }

    fn make_rd_virtual_info(
        &self,
        opref: OpRef,
        fieldnums: Vec<i16>,
    ) -> Option<majit_ir::RdVirtualInfo> {
        let resolved = self.ctx.get_box_replacement(opref);
        let info = self.ctx.get_ptr_info(resolved)?;
        match info {
            PtrInfo::Virtual(vi) => {
                let fielddescrs: Vec<majit_ir::FieldDescrInfo> = vi
                    .fields
                    .iter()
                    .map(|(fi, _)| {
                        let fd = vi
                            .field_descrs
                            .iter()
                            .find(|(di, _)| *di == *fi)
                            .and_then(|(_, d)| d.as_field_descr());
                        majit_ir::FieldDescrInfo {
                            index: *fi,
                            offset: fd.map(|f| f.offset()).unwrap_or(0),
                            field_type: fd.map(|f| f.field_type()).unwrap_or(majit_ir::Type::Int),
                            field_size: fd.map(|f| f.field_size()).unwrap_or(8),
                        }
                    })
                    .collect();
                let descr_size = vi.descr.as_size_descr().map(|s| s.size()).unwrap_or(0);
                Some(majit_ir::RdVirtualInfo::VirtualInfo {
                    descr: Some(vi.descr.clone()),
                    type_id: vi.descr.as_size_descr().map(|sd| sd.type_id()).unwrap_or(0),
                    descr_index: vi.descr.index(),
                    known_class: vi
                        .known_class
                        .map(|gc| gc.as_usize() as i64)
                        .or_else(|| vi.descr.as_size_descr().map(|sd| sd.vtable() as i64))
                        .filter(|&v| v != 0),
                    fielddescrs,
                    fieldnums,
                    descr_size,
                })
            }
            PtrInfo::VirtualStruct(vi) => {
                let fielddescrs: Vec<majit_ir::FieldDescrInfo> = vi
                    .fields
                    .iter()
                    .map(|(fi, _)| {
                        let fd = vi
                            .field_descrs
                            .iter()
                            .find(|(di, _)| *di == *fi)
                            .and_then(|(_, d)| d.as_field_descr());
                        majit_ir::FieldDescrInfo {
                            index: *fi,
                            offset: fd.map(|f| f.offset()).unwrap_or(0),
                            field_type: fd.map(|f| f.field_type()).unwrap_or(majit_ir::Type::Int),
                            field_size: fd.map(|f| f.field_size()).unwrap_or(8),
                        }
                    })
                    .collect();
                let sd = vi.descr.as_size_descr();
                let descr_size = sd.map(|s| s.size()).unwrap_or(0);
                let tid = sd.map(|s| s.type_id()).unwrap_or(0);
                Some(majit_ir::RdVirtualInfo::VStructInfo {
                    typedescr: Some(vi.descr.clone()),
                    type_id: tid,
                    descr_index: vi.descr.index(),
                    fielddescrs,
                    fieldnums,
                    descr_size,
                })
            }
            PtrInfo::VirtualArray(vi) => {
                let kind = vi
                    .descr
                    .as_array_descr()
                    .map(|ad| match ad.item_type() {
                        majit_ir::Type::Float => 2u8,
                        majit_ir::Type::Int => 1u8,
                        _ => 0u8,
                    })
                    .unwrap_or(0);
                if vi.clear {
                    Some(majit_ir::RdVirtualInfo::VArrayInfoClear {
                        descr_index: vi.descr.index(),
                        kind,
                        fieldnums,
                    })
                } else {
                    Some(majit_ir::RdVirtualInfo::VArrayInfoNotClear {
                        descr_index: vi.descr.index(),
                        kind,
                        fieldnums,
                    })
                }
            }
            PtrInfo::VirtualArrayStruct(vi) => {
                let fielddescr_indices: Vec<u32> = vi
                    .element_fields
                    .first()
                    .map(|ef| ef.iter().map(|(idx, _)| *idx).collect())
                    .unwrap_or_default();
                let is = vi
                    .descr
                    .as_array_descr()
                    .map(|ad| ad.item_size())
                    .unwrap_or(0);
                let mut fo = Vec::new();
                let mut fs = Vec::new();
                let mut ft = Vec::new();
                for fd in &vi.fielddescrs {
                    if let Some(ifd) = fd.as_interior_field_descr() {
                        let fld = ifd.field_descr();
                        fo.push(fld.offset());
                        fs.push(fld.field_size());
                        ft.push(match fld.field_type() {
                            majit_ir::Type::Float => 2u8,
                            majit_ir::Type::Int => 1u8,
                            _ => 0u8,
                        });
                    } else {
                        fo.push(fo.len() * 8);
                        fs.push(8);
                        ft.push(0);
                    }
                }
                if ft.is_empty() {
                    ft = vec![0u8; fielddescr_indices.len()];
                }
                Some(majit_ir::RdVirtualInfo::VArrayStructInfo {
                    descr_index: vi.descr.index(),
                    size: vi.element_fields.len(),
                    fielddescr_indices,
                    field_types: ft,
                    item_size: is,
                    field_offsets: fo,
                    field_sizes: fs,
                    fieldnums,
                })
            }
            PtrInfo::VirtualRawBuffer(vi) => {
                let offsets: Vec<usize> = vi.entries.iter().map(|(o, _, _)| *o).collect();
                let entry_sizes: Vec<usize> = vi.entries.iter().map(|(_, len, _)| *len).collect();
                Some(majit_ir::RdVirtualInfo::VRawBufferInfo {
                    size: vi.size,
                    offsets,
                    entry_sizes,
                    fieldnums,
                })
            }
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
            forwarded: Vec::new(),
            num_inputs: 0,
            next_pos: 0,
            next_const_pos: OpRef::CONST_BASE,
            extra_operations_after: VecDeque::new(),
            int_lower_bounds: HashMap::new(),
            int_bounds: Vec::new(),
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
            exported_short_boxes: Vec::new(),
            imported_virtual_heads: Vec::new(),
            imported_virtuals: Vec::new(),
            imported_label_args: None,
            can_replace_guards: true,
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
            snapshot_box_types: HashMap::new(),
            value_types: HashMap::new(),
            last_guard_idx: None,
            last_seen_snapshot_pos: None,
        }
    }

    pub fn with_num_inputs(estimated_ops: usize, num_inputs: usize) -> Self {
        OptContext {
            new_operations: Vec::with_capacity(estimated_ops),
            constants: Vec::new(),
            forwarded: Vec::new(),
            num_inputs: num_inputs as u32,
            next_pos: num_inputs as u32,
            next_const_pos: OpRef::CONST_BASE,
            extra_operations_after: VecDeque::new(),
            int_lower_bounds: HashMap::new(),
            int_bounds: Vec::new(),
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
            exported_short_boxes: Vec::new(),
            imported_virtual_heads: Vec::new(),
            imported_virtuals: Vec::new(),
            imported_label_args: None,
            can_replace_guards: true,
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
            snapshot_box_types: HashMap::new(),
            value_types: HashMap::new(),
            last_guard_idx: None,
            last_seen_snapshot_pos: None,
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs as usize
    }

    /// Allocate a fresh OpRef position (for imported virtual heads).
    pub fn alloc_op_position(&mut self) -> OpRef {
        self.reserve_pos()
    }

    /// Allocate a fresh OpRef in the constant namespace (>= CONST_BASE).
    pub(crate) fn reserve_const_pos(&mut self) -> OpRef {
        while self
            .constants
            .get(self.next_const_pos as usize)
            .is_some_and(|value| value.is_some())
        {
            self.next_const_pos += 1;
        }
        debug_assert!(
            self.next_const_pos >= OpRef::CONST_BASE,
            "reserve_const_pos allocated non-constant OpRef below constant namespace: {}",
            self.next_const_pos
        );
        let pos_ref = OpRef(self.next_const_pos);
        self.next_const_pos += 1;
        pos_ref
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
        debug_assert!(
            self.next_pos < OpRef::CONST_BASE,
            "reserve_pos in constant namespace: {}",
            self.next_pos
        );
        let pos_ref = OpRef(self.next_pos);
        self.next_pos += 1;
        pos_ref
    }

    /// Emit an operation to the output.
    ///
    /// If the op has no pos assigned (NONE), sets it to `num_inputs + idx`
    /// so the backend's variable numbering stays consistent.
    pub fn emit(&mut self, mut op: Op) -> OpRef {
        if op.pos.is_none() || op.pos.0 >= OpRef::CONST_BASE {
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

        // Track OpRef → Type for fail_arg_types inference
        // (compile.rs value_types parity).
        if !op.pos.is_none() && op.result_type() != majit_ir::Type::Void {
            self.value_types.insert(op.pos.0, op.result_type());
        }
        self.new_operations.push(op);
        pos_ref
    }

    /// RPython emit_extra(op, emit=False) parity: queue an operation to
    /// be processed through passes AFTER the calling pass. Skips earlier
    /// passes (including the caller) to avoid re-absorption loops.
    /// `after_pass_idx`: index of the calling pass (op starts from idx+1).
    pub fn emit_extra(&mut self, after_pass_idx: usize, mut op: Op) -> OpRef {
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
        let mut builder = crate::optimizeopt::shortpreamble::ShortPreambleBuilder::new(
            label_args,
            &produced,
            short_inputargs,
        );
        // RPython parity: populate known_constants from OptContext
        // so use_box_recursive can skip constant OpRef args.
        for (idx, val) in self.constants.iter().enumerate() {
            if val.is_some() {
                builder.note_known_constant(OpRef(idx as u32));
            }
        }
        self.imported_short_preamble_builder = Some(builder);
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

        let mut builder = ShortPreambleBuilder::new(short_args, &produced, short_inputargs);
        // RPython parity: populate known_constants from OptContext
        // so add_op_to_short can resolve constant OpRef args.
        for (idx, val) in self.constants.iter().enumerate() {
            if val.is_some() {
                builder.note_known_constant(OpRef(idx as u32));
            }
        }
        self.imported_short_preamble_builder = Some(builder);
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
    /// Calls use_box (shortpreamble.py:382-407) with info/guard replay,
    /// then registers in potential_extra_ops for later force_box.
    pub fn force_op_from_preamble(&mut self, result: OpRef) -> OpRef {
        // unroll.py:27: check isinstance BEFORE get_box_replacement
        let preamble_source = self.imported_short_source(result);
        let is_constant = self.get_constant(preamble_source).is_some();
        if self.imported_short_preamble_used.insert(preamble_source) {
            // shortpreamble.py:389,396,406: info.make_guards → self.short
            let (arg_guards, result_guards) = self.collect_use_box_guards(preamble_source);

            // unroll.py:32: use_box(op, preamble_op.preamble_op, self)
            let tracked = if let Some(mut builder) = self.active_short_preamble_producer.take() {
                // shortpreamble.py:478-481: ExtendedShortPreambleBuilder.use_box
                let ok = builder
                    .use_box(preamble_source, &arg_guards, &result_guards)
                    .is_some();
                self.active_short_preamble_producer = Some(builder);
                ok
            } else if let Some(mut builder) = self.imported_short_preamble_builder.take() {
                let ok = builder
                    .use_box(preamble_source, &arg_guards, &result_guards)
                    .is_some();
                self.imported_short_preamble_builder = Some(builder);
                ok
            } else {
                false
            };
            // shortpreamble.py:404: setinfo_from_preamble(box, info, None)
            if let Some(info) = self.get_ptr_info(preamble_source).cloned() {
                self.setinfo_from_preamble(result, &info, None);
            }

            // unroll.py:33-37: potential_extra_ops[op] = preamble_op
            //
            // shortpreamble.py:471-477: add_preamble_op — track used ops.
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
        result
    }

    /// shortpreamble.py:383-396,401-406: collect guards from PtrInfo
    /// of preamble_op's args and result.
    fn collect_use_box_guards(&mut self, preamble_source: OpRef) -> (Vec<Op>, Vec<Op>) {
        let produced = self
            .imported_short_preamble_builder
            .as_ref()
            .and_then(|b| b.produced_short_op(preamble_source))
            .or_else(|| {
                self.active_short_preamble_producer
                    .as_ref()
                    .and_then(|b| b.produced_short_op(preamble_source))
            });
        let Some(produced) = produced else {
            return (Vec::new(), Vec::new());
        };

        // shortpreamble.py:383-396: guards for InputArg args only
        let short_inputargs: Vec<OpRef> = self
            .imported_short_preamble_builder
            .as_ref()
            .map(|b| b.short_inputargs().to_vec())
            .or_else(|| {
                self.active_short_preamble_producer
                    .as_ref()
                    .map(|b| b.short_inputargs().to_vec())
            })
            .unwrap_or_default();

        // Collect (arg, PtrInfo clone) pairs to avoid borrow conflicts
        let arg_infos: Vec<(OpRef, PtrInfo)> = produced
            .preamble_op
            .args
            .iter()
            .filter(|arg| short_inputargs.contains(arg))
            .filter_map(|&arg| self.get_ptr_info(arg).cloned().map(|info| (arg, info)))
            .collect();
        let result_info = self
            .get_ptr_info(produced.preamble_op.pos)
            .cloned()
            .map(|info| (produced.preamble_op.pos, info));

        // Now generate guards — can call &mut self for constant allocation
        let mut arg_guards = Vec::new();
        let mut const_pool = Vec::new();
        for (arg, info) in &arg_infos {
            info.make_guards(*arg, &mut arg_guards, &mut const_pool);
        }
        let mut result_guards = Vec::new();
        if let Some((result_ref, info)) = &result_info {
            info.make_guards(*result_ref, &mut result_guards, &mut const_pool);
        }
        // Replace placeholder OpRefs with stable constant pool OpRefs.
        // Use 10600+ range to avoid interfering with alloc_op_position
        // (which advances next_pos and shifts all subsequent OpRefs).
        let mut remap: std::collections::HashMap<OpRef, OpRef> = std::collections::HashMap::new();
        for (i, (placeholder, value)) in const_pool.into_iter().enumerate() {
            let real = OpRef(10600 + i as u32);
            self.make_constant(real, value);
            remap.insert(placeholder, real);
        }
        if !remap.is_empty() {
            for guard in arg_guards.iter_mut().chain(result_guards.iter_mut()) {
                for arg in &mut guard.args {
                    if let Some(&real) = remap.get(arg) {
                        *arg = real;
                    }
                }
            }
        }
        (arg_guards, result_guards)
    }

    /// unroll.py:53-98: setinfo_from_preamble(op, preamble_info, exported_infos)
    /// RPython uses sequential `if` (not elif) so multiple properties accumulate.
    /// `exported_infos`: None from use_box path (shortpreamble.py:404),
    /// Some from import_state path. When None, virtual branch does NOT recurse.
    fn setinfo_from_preamble(
        &mut self,
        op: OpRef,
        preamble_info: &PtrInfo,
        exported_infos: Option<&HashMap<OpRef, crate::optimizeopt::unroll::ExportedValueInfo>>,
    ) {
        let op = self.get_box_replacement(op);
        // unroll.py:55: if op.get_forwarded() is not None: return
        if self.get_ptr_info(op).is_some() || self.is_replaced(op) {
            return;
        }
        // unroll.py:57: if op.is_constant(): return
        if self.is_constant(op) {
            return;
        }

        // unroll.py:60-64: virtual — set_forwarded + recurse, then return
        if preamble_info.is_virtual() {
            self.set_ptr_info(op, preamble_info.clone());
            if let Some(infos) = exported_infos {
                let items: Vec<OpRef> = match preamble_info {
                    PtrInfo::Virtual(v) => v.fields.iter().map(|(_, r)| *r).collect(),
                    PtrInfo::VirtualArray(a) => a.items.iter().copied().collect(),
                    PtrInfo::VirtualStruct(s) => s.fields.iter().map(|(_, r)| *r).collect(),
                    PtrInfo::VirtualArrayStruct(a) => a
                        .element_fields
                        .iter()
                        .flat_map(|row| row.iter().map(|(_, r)| *r))
                        .collect(),
                    PtrInfo::VirtualRawBuffer(r) => r.entries.iter().map(|(_, _, v)| *v).collect(),
                    _ => Vec::new(),
                };
                self.setinfo_from_preamble_list(&items, infos);
            }
            return;
        }

        // unroll.py:65-68: constant — return early
        if let PtrInfo::Constant(gcref) = preamble_info {
            self.make_constant(op, Value::Ref(*gcref));
            return;
        }

        // --- Sequential checks (RPython: NOT elif, all accumulate) ---

        // unroll.py:69-74: Struct/Instance with descr → set_forwarded
        if preamble_info.get_descr().is_some() {
            if let PtrInfo::Struct(sinfo) = preamble_info {
                self.set_ptr_info(op, PtrInfo::struct_ptr(sinfo.descr.clone()));
            }
            if let PtrInfo::Instance(iinfo) = preamble_info {
                self.set_ptr_info(op, PtrInfo::instance(iinfo.descr.clone(), None));
            }
        }

        // unroll.py:75-77: known_class → make_constant_class(op, class, False)
        if let Some(cls) = preamble_info.get_known_class() {
            // optimizer.py:137-156: updates existing InstancePtrInfo._known_class
            let resolved = self.get_box_replacement(op);
            let is_instance = matches!(self.get_ptr_info(resolved), Some(PtrInfo::Instance(_)));
            if is_instance {
                if let Some(PtrInfo::Instance(iinfo)) = self.get_ptr_info_mut(resolved) {
                    iinfo.known_class = Some(*cls);
                }
            } else {
                self.set_ptr_info(op, PtrInfo::known_class(*cls, true));
            }
        }

        // unroll.py:79-84: ArrayPtrInfo → set_forwarded(ArrayPtrInfo(descr, lenbound))
        if let PtrInfo::Array(ainfo) = preamble_info {
            self.set_ptr_info(
                op,
                PtrInfo::array(ainfo.descr.clone(), ainfo.lenbound.clone()),
            );
        }

        // unroll.py:85-89: StrPtrInfo — clone lenbound
        if let PtrInfo::Str(sinfo) = preamble_info {
            let mut new_info = crate::optimizeopt::info::StrPtrInfo {
                lenbound: sinfo.lenbound.clone(),
                mode: sinfo.mode,
                length: -1,
                last_guard_pos: -1,
            };
            if new_info.lenbound.is_none() {
                new_info.lenbound = Some(crate::optimizeopt::intutils::IntBound::nonnegative());
            }
            self.set_ptr_info(op, PtrInfo::Str(new_info));
            return;
        }

        // unroll.py:91-92: is_nonnull → make_nonnull
        if preamble_info.is_nonnull() {
            self.make_nonnull(op);
        }
    }

    /// unroll.py:41-51: setinfo_from_preamble_list(lst, infos)
    /// Recursively propagate PtrInfo for virtual field items using
    /// exported_infos as source of truth (not current OptContext).
    fn setinfo_from_preamble_list(
        &mut self,
        items: &[OpRef],
        exported_infos: &HashMap<OpRef, crate::optimizeopt::unroll::ExportedValueInfo>,
    ) {
        for &item in items {
            if item.is_none() {
                continue;
            }
            // unroll.py:45-46: i = infos.get(item, None)
            if let Some(info) = exported_infos.get(&item) {
                // unroll.py:47: self.setinfo_from_preamble(item, i, infos)
                if let Some(ref ptr_info) = info.ptr_info {
                    self.setinfo_from_preamble(item, ptr_info, Some(exported_infos));
                } else if let Some(cls) = info.known_class {
                    // ExportedValueInfo has known_class but no full PtrInfo.
                    // RPython: the info IS the PtrInfo (always non-None when key exists).
                    let synth = PtrInfo::known_class(cls, info.nonnull);
                    self.setinfo_from_preamble(item, &synth, Some(exported_infos));
                } else if info.nonnull {
                    let synth = PtrInfo::nonnull();
                    self.setinfo_from_preamble(item, &synth, Some(exported_infos));
                }
                // int_bound: unroll.py:93-96 — handled by setinfo_from_preamble
                // when we add IntBound dispatch there.
            } else {
                // unroll.py:49: item.set_forwarded(None)
                // "let's not inherit stuff we don't know anything about"
                self.clear_forwarded(item);
            }
        }
    }

    /// unroll.py:49: item.set_forwarded(None)
    fn clear_forwarded(&mut self, opref: OpRef) {
        use crate::optimizeopt::info::Forwarded;
        let idx = opref.0 as usize;
        if idx < self.forwarded.len() {
            self.forwarded[idx] = Forwarded::None;
        }
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
        self.active_short_preamble_producer.as_ref().map(|builder| {
            builder.build_short_preamble_struct(&HashMap::new(), &self.constant_types_for_numbering)
        })
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
            .map(|builder| {
                // RPython parity: extract constant pool from OptContext.
                // In RPython, Const objects in short preamble ops survive
                // across compilations via GC tracing. In majit, we must
                // snapshot the constant pool so build_short_preamble_struct
                // can capture referenced constants.
                let loop_constants: HashMap<u32, i64> = self
                    .constants
                    .iter()
                    .enumerate()
                    .filter_map(|(i, v)| {
                        v.as_ref().map(|val| {
                            let raw = match val {
                                majit_ir::Value::Int(v) => *v,
                                majit_ir::Value::Float(f) => f.to_bits() as i64,
                                majit_ir::Value::Ref(r) => r.0 as i64,
                                majit_ir::Value::Void => 0,
                            };
                            (i as u32, raw)
                        })
                    })
                    .collect();
                let mut loop_constant_types = self.constant_types_for_numbering.clone();
                for (i, v) in self.constants.iter().enumerate() {
                    if let Some(val) = v {
                        let tp = match val {
                            majit_ir::Value::Int(_) => majit_ir::Type::Int,
                            majit_ir::Value::Float(_) => majit_ir::Type::Float,
                            majit_ir::Value::Ref(_) => majit_ir::Type::Ref,
                            majit_ir::Value::Void => majit_ir::Type::Void,
                        };
                        loop_constant_types.entry(i as u32).or_insert(tp);
                    }
                }
                builder.build_short_preamble_struct(&loop_constants, &loop_constant_types)
            })
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

    /// resoperation.py:240-242: set_forwarded(forwarded_to) — store the
    /// forwarding target on this box. Info is set separately by setinfo /
    /// setinfo_from_preamble.
    ///
    /// NOTE: This function now matches RPython's set_forwarded semantics,
    /// but the full import/export cycle that calls it (import_state,
    /// cross-slot fresh allocation) still operates on majit's flat OpRef
    /// model, not RPython's per-Box identity. See XXX comments in
    /// unroll.rs:import_state and optimizer.rs:imported_loop_state.
    pub fn replace_op(&mut self, old: OpRef, new: OpRef) {
        if old == new {
            return;
        }
        use crate::optimizeopt::info::Forwarded;
        let idx = old.0 as usize;
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        if new.is_none() {
            self.forwarded[idx] = Forwarded::None;
        } else {
            self.forwarded[idx] = Forwarded::Op(new);
        }
    }

    /// RPython unroll.py: source.set_forwarded(target)
    /// Sets forwarding from Phase 2 source to Phase 1 export target.
    /// setinfo_from_preamble_recursive then sets PtrInfo on the TARGET
    /// (via get_replacement).
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
    ///
    /// RPython invariant: get_box_replacement NEVER returns None.
    /// `_forwarded = None` means "no forwarding" (terminal), NOT
    /// "forwarded to None". Forwarded::Op(OpRef::NONE) is treated
    /// as terminal — the chain stops at the current opref.
    ///
    /// NEVER consults mapping dicts — RPython's get_box_replacement only
    /// follows the _forwarded chain on the box itself.
    pub fn get_box_replacement(&self, mut opref: OpRef) -> OpRef {
        use crate::optimizeopt::info::Forwarded;
        loop {
            let idx = opref.0 as usize;
            if idx >= self.forwarded.len() {
                return opref;
            }
            match &self.forwarded[idx] {
                Forwarded::Op(next) if !next.is_none() => opref = *next,
                _ => return opref,
            }
        }
    }

    /// Store a constant value WITHOUT setting Forwarded::Const.
    /// Used for pre-populating backend constants and call_pure_results.
    pub fn seed_constant(&mut self, opref: OpRef, value: Value) {
        let idx = opref.0 as usize;
        if idx >= self.constants.len() {
            self.constants.resize(idx + 1, None);
        }
        self.constants[idx] = Some(value);
    }

    /// optimizer.py:99-113: getintbound(op) — get or create IntBound for
    /// an int-typed box. Lazy: creates unbounded on first access and stores
    /// it in forwarded[].
    pub fn getintbound(&mut self, opref: OpRef) -> crate::optimizeopt::intutils::IntBound {
        use crate::optimizeopt::info::Forwarded;
        let replaced = self.get_box_replacement(opref);
        // optimizer.py:102-103: if isinstance(op, ConstInt): return from_constant
        if let Some(Value::Int(v)) = self.get_constant(replaced) {
            return crate::optimizeopt::intutils::IntBound::from_constant(*v as i64);
        }
        let idx = replaced.0 as usize;
        if idx < self.forwarded.len() {
            match &self.forwarded[idx] {
                // optimizer.py:106-107: isinstance(fw, IntBound) → return fw
                Forwarded::IntBound(b) => return b.clone(),
                // optimizer.py:108-109: fw is not None but not IntBound
                // (rare: RawBufferPtrInfo etc.) → return unbounded, do NOT
                // overwrite existing forwarding.
                Forwarded::None => {}
                _ => return crate::optimizeopt::intutils::IntBound::unbounded(),
            }
        }
        // optimizer.py:110-112: fw is None → create unbounded and store
        let intbound = crate::optimizeopt::intutils::IntBound::unbounded();
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        self.forwarded[idx] = Forwarded::IntBound(intbound.clone());
        intbound
    }

    /// optimizer.py:115-125: setintbound(op, bound) — intersect existing
    /// bound with new bound, or set if none exists.
    pub fn setintbound(&mut self, opref: OpRef, bound: &crate::optimizeopt::intutils::IntBound) {
        use crate::optimizeopt::info::Forwarded;
        let replaced = self.get_box_replacement(opref);
        if self.get_constant(replaced).is_some() {
            return;
        }
        let idx = replaced.0 as usize;
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        match &mut self.forwarded[idx] {
            Forwarded::IntBound(cur) => {
                let _ = cur.intersect(bound);
            }
            fwd @ Forwarded::None => {
                *fwd = Forwarded::IntBound(bound.clone());
            }
            _ => {
                // Already has Op/Info/Const forwarding — don't overwrite.
                // RPython: if cur is not IntBound, set_forwarded replaces it,
                // but that case is rare (RawBufferPtrInfo).
            }
        }
    }

    /// optimizer.py:410-432 make_constant(box, constbox).
    /// Forwarded::Const(value) is the terminal — get_box_replacement stops
    /// here and returns the opref. is_const detects Forwarded::Const.
    /// optimizer.py:410-432: make_constant(box, constbox)
    pub fn make_constant(&mut self, opref: OpRef, value: Value) {
        use crate::optimizeopt::info::Forwarded;
        // optimizer.py:412: box = get_box_replacement(box)
        let replaced = self.get_box_replacement(opref);
        // optimizer.py:415-426: safety check — if box.get_forwarded() is
        // IntBound and the constant is Int, validate contains() + make_eq_const().
        // RPython checks ONE authoritative source: box._forwarded.
        if let Value::Int(intval) = value {
            let ridx = replaced.0 as usize;
            if ridx < self.forwarded.len() {
                if let Forwarded::IntBound(bound) = &mut self.forwarded[ridx] {
                    if !bound.contains(intval as i64) {
                        std::panic::panic_any(crate::optimizeopt::optimize::InvalidLoop(
                            "constant int is outside the range allowed for that box",
                        ));
                    }
                    // optimizer.py:424-426: info.make_eq_const(value)
                    let _ = bound.make_eq_const(intval as i64);
                }
            }
        }
        // optimizer.py:427: if box.is_constant(): return
        if replaced.is_constant()
            || matches!(
                self.forwarded.get(replaced.0 as usize),
                Some(Forwarded::Const(_))
            )
        {
            return;
        }
        // optimizer.py:429-431 + info.py:194-198,533-538,718-736:
        // copy_fields_to_const — create a concrete (non-virtual) info on
        // the const and copy only the field/item cache.
        // ConstPtrInfo._get_info → StructPtrInfo(descr)
        // ConstPtrInfo._get_array_info → ArrayPtrInfo(descr)
        if let Value::Ref(gcref) = value {
            if let Some(Forwarded::Info(info)) = self.forwarded.get(replaced.0 as usize) {
                use crate::optimizeopt::info::{ArrayPtrInfo, PtrInfo, StructPtrInfo};
                let key = gcref.as_usize();
                match info {
                    // AbstractStructPtrInfo hierarchy → concrete StructPtrInfo
                    PtrInfo::Instance(v) if !v.fields.is_empty() => {
                        if let Some(descr) = v.descr.clone() {
                            let ci = self.const_infos.entry(key).or_insert_with(|| {
                                PtrInfo::Struct(StructPtrInfo {
                                    descr,
                                    fields: Vec::new(),
                                    field_descrs: Vec::new(),
                                    preamble_fields: Vec::new(),
                                    last_guard_pos: -1,
                                })
                            });
                            if let PtrInfo::Struct(s) = ci {
                                s.fields = v.fields.clone();
                            }
                        }
                    }
                    PtrInfo::Struct(v) if !v.fields.is_empty() => {
                        let ci = self.const_infos.entry(key).or_insert_with(|| {
                            PtrInfo::Struct(StructPtrInfo {
                                descr: v.descr.clone(),
                                fields: Vec::new(),
                                field_descrs: Vec::new(),
                                preamble_fields: Vec::new(),
                                last_guard_pos: -1,
                            })
                        });
                        if let PtrInfo::Struct(s) = ci {
                            s.fields = v.fields.clone();
                        }
                    }
                    PtrInfo::Virtual(v) if !v.fields.is_empty() => {
                        let ci = self.const_infos.entry(key).or_insert_with(|| {
                            PtrInfo::Struct(StructPtrInfo {
                                descr: v.descr.clone(),
                                fields: Vec::new(),
                                field_descrs: Vec::new(),
                                preamble_fields: Vec::new(),
                                last_guard_pos: -1,
                            })
                        });
                        if let PtrInfo::Struct(s) = ci {
                            s.fields = v.fields.clone();
                        }
                    }
                    PtrInfo::VirtualStruct(v) if !v.fields.is_empty() => {
                        let ci = self.const_infos.entry(key).or_insert_with(|| {
                            PtrInfo::Struct(StructPtrInfo {
                                descr: v.descr.clone(),
                                fields: Vec::new(),
                                field_descrs: Vec::new(),
                                preamble_fields: Vec::new(),
                                last_guard_pos: -1,
                            })
                        });
                        if let PtrInfo::Struct(s) = ci {
                            s.fields = v.fields.clone();
                        }
                    }
                    // ArrayPtrInfo hierarchy → concrete ArrayPtrInfo
                    PtrInfo::Array(v) if !v.items.is_empty() => {
                        let ci = self.const_infos.entry(key).or_insert_with(|| {
                            PtrInfo::Array(ArrayPtrInfo {
                                descr: v.descr.clone(),
                                lenbound: v.lenbound.clone(),
                                items: Vec::new(),
                                last_guard_pos: -1,
                            })
                        });
                        if let PtrInfo::Array(a) = ci {
                            a.items = v.items.clone();
                        }
                    }
                    PtrInfo::VirtualArray(v) if !v.items.is_empty() => {
                        let ci = self.const_infos.entry(key).or_insert_with(|| {
                            PtrInfo::Array(ArrayPtrInfo {
                                descr: v.descr.clone(),
                                lenbound: IntBound::from_constant(v.items.len() as i64),
                                items: Vec::new(),
                                last_guard_pos: -1,
                            })
                        });
                        if let PtrInfo::Array(a) = ci {
                            a.items = v.items.clone();
                        }
                    }
                    _ => {}
                }
            }
        }
        // Store in constants array for get_constant() callers.
        let idx = replaced.0 as usize;
        if idx >= self.constants.len() {
            self.constants.resize(idx + 1, None);
        }
        self.constants[idx] = Some(value.clone());
        // optimizer.py:432: box.set_forwarded(constbox)
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        self.forwarded[idx] = Forwarded::Const(value);
    }

    /// resume.py:157 getconst parity for synthetic rd_numb encoding.
    /// Matches OptBoxEnv::get_const: checks constant_types_for_numbering
    /// override, PtrInfo::Constant, and constant pool.
    /// Returns None if opref is not a constant.
    pub fn getconst(&self, opref: OpRef) -> Option<(i64, majit_ir::Type)> {
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
    /// new resume data (store_final_boxes_in_guard).
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

        // optimizer.py:672: `self._last_guard_op and guard_op.getdescr() is None`
        // getdescr() is None only for optimizer-created guards (no descr
        // from tracing, no ResumeAtPositionDescr from unroll).
        // compile.py:925-926: GUARD_NOT_FORCED* must never share —
        // invent_fail_descr_for_op asserts copied_from_descr is None.
        let can_share = self.last_guard_idx.is_some()
            && op.descr.is_none()
            && opnum != OpCode::GuardNotForced
            && opnum != OpCode::GuardNotForced2;

        if can_share {
            let idx = self.last_guard_idx.unwrap();
            // _copy_resume_data_from: share resume data from last guard.
            op.rd_numb = self.new_operations[idx].rd_numb.clone();
            op.rd_consts = self.new_operations[idx].rd_consts.clone();
            op.rd_virtuals = self.new_operations[idx].rd_virtuals.clone();
            op.rd_pendingfields = self.new_operations[idx].rd_pendingfields.clone();
            op.fail_args = self.new_operations[idx].fail_args.clone();
            // Don't update last_guard_idx — copied guards don't become sources.
        } else {
            // optimizer.py:678: store_final_boxes_in_guard
            self.store_final_boxes_in_guard(op);
            self.last_guard_idx = Some(self.new_operations.len());
            // optimizer.py:680-683: force_box on fail_args for unrolling.
            // Mirrors Optimizer.force_box contract: resolve replacement,
            // handle tracked preamble ops, force virtuals.
            if let Some(ref fa) = op.fail_args {
                let fargs: Vec<OpRef> = fa.iter().copied().collect();
                for farg in fargs {
                    if !farg.is_none() {
                        // regalloc.py:1206: Const objects skip forcing.
                        // Constant OpRefs may collide with virtual positions;
                        // forcing would corrupt the virtual's PtrInfo.
                        let resolved = self.get_box_replacement(farg);
                        if !self.is_constant(resolved) {
                            self.force_box_inline(farg);
                        }
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
                let forced = info.force_box(resolved, self);
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
        self.store_final_boxes_in_guard(op);
    }

    fn store_final_boxes_in_guard(&self, op: &mut Op) {
        use crate::resume::{ResumeDataLoopMemo, Snapshot};

        // resume.py:397: assert not storage.rd_numb
        // RPython's finish() is called exactly once per guard.
        // If rd_numb is already set (from _copy_resume_data_from / shared
        // guard path), the numbering is already correct — return immediately.
        if op.rd_numb.is_some() {
            return;
        }

        // resume.py:396-397:
        //   resume_position = self.guard_op.rd_resume_position
        //   assert resume_position >= 0
        // RPython: every guard has a valid rd_resume_position set by either
        // capture_resumedata (tracer guards) or patchguardop copy
        // (unroll.py:336/409). No fallback — the position is always set
        // before store_final_boxes_in_guard runs.
        let resume_pos = op.rd_resume_position;
        let has_snapshot = resume_pos >= 0 && self.snapshot_boxes.contains_key(&resume_pos);
        if !has_snapshot {
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[jit][drop] no-snapshot guard {:?} pos={:?} resume_pos={}",
                    op.opcode, op.pos, op.rd_resume_position,
                );
            }
            return;
        }

        // RPython parity: snapshot path handles ALL guards with snapshots,
        // including guards with rd_virtuals. The snapshot uses original boxes
        // and PtrInfo to correctly assign TAGVIRTUAL via _number_boxes.
        // _number_virtuals then builds rd_virtuals from PtrInfo.
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
        // ni/vsd are constants (TAGINT/TAGCONST) so they don't affect
        // TAGBOX numbering. The same OpRefs also appear in fail_args —
        // _number_boxes deduplicates via liveboxes HashMap.
        snapshot.vable_array = vable_oprefs;

        // resume.py:389-452: delegate to ResumeDataVirtualAdder.finish()
        let env = OptBoxEnv { ctx: self };
        let mut memo = ResumeDataLoopMemo::new();
        let Ok(numb_state) = memo.number(&snapshot, &env) else {
            return;
        };

        // resume.py:428-445, 520-558: pending_setfields are passed to finish()
        // which handles register_box, visitor_walk_recursive, and tagging.
        let mut pending_slice = op.rd_pendingfields.take().unwrap_or_default();

        let (rd_numb, rd_consts, rd_virtuals, liveboxes) =
            memo.finish(numb_state, &env, &mut pending_slice, None);

        // fail_arg_types — majit-specific (RPython Box.type is intrinsic).
        // Full type resolution cascade: constants, value_types, ops, snapshots.
        let new_types: Vec<majit_ir::Type> = liveboxes
            .iter()
            .map(|opref| {
                if opref.is_none() {
                    return majit_ir::Type::Ref;
                }
                let resolved = self.get_box_replacement(*opref);
                if let Some(&tp) = self.constant_types_for_numbering.get(&resolved.0) {
                    return tp;
                }
                if resolved != *opref {
                    if let Some(&tp) = self.constant_types_for_numbering.get(&opref.0) {
                        return tp;
                    }
                }
                if let Some(val) = self.get_constant(resolved) {
                    return val.get_type();
                }
                if let Some(&tp) = self.value_types.get(&resolved.0) {
                    return tp;
                }
                if *opref != resolved {
                    if let Some(&tp) = self.value_types.get(&opref.0) {
                        return tp;
                    }
                }
                if let Some(tp) = self.get_op_result_type(resolved) {
                    return tp;
                }
                if let Some(&tp) = self.snapshot_box_types.get(&resolved.0) {
                    return tp;
                }
                if resolved != *opref {
                    if let Some(&tp) = self.snapshot_box_types.get(&opref.0) {
                        return tp;
                    }
                }
                if self.get_ptr_info(resolved).is_some() {
                    return majit_ir::Type::Ref;
                }
                for o in self.new_operations.iter() {
                    if o.pos == resolved && o.result_type() != majit_ir::Type::Void {
                        return o.result_type();
                    }
                    if o.pos == *opref && o.result_type() != majit_ir::Type::Void {
                        return o.result_type();
                    }
                }
                panic!(
                    "fail_arg_types: unknown type for OpRef({}) resolved=OpRef({})",
                    opref.0, resolved.0
                )
            })
            .collect();

        op.store_final_boxes(liveboxes);
        op.fail_arg_types = Some(new_types);
        if !rd_virtuals.is_empty() {
            op.rd_virtuals = Some(rd_virtuals);
        }

        // Restore pending fields (now tagged by finish())
        if !pending_slice.is_empty() {
            op.rd_pendingfields = Some(pending_slice);
        }

        op.rd_numb = Some(rd_numb);
        op.rd_consts = Some(rd_consts);
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

    pub fn make_constant_int(&mut self, value: i64) -> OpRef {
        let pos = self.reserve_const_pos();
        self.make_constant(pos, Value::Int(value));
        pos
    }

    pub fn make_constant_ref(&mut self, value: GcRef) -> OpRef {
        let pos = self.reserve_const_pos();
        self.make_constant(pos, Value::Ref(value));
        pos
    }

    pub fn make_constant_float(&mut self, value: f64) -> OpRef {
        let pos = self.reserve_const_pos();
        self.make_constant(pos, Value::Float(value));
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

    /// optimizer.py:783-790: constant_fold(op).
    /// Calls protect_speculative_operation, then execute_nonspec_const.
    /// Returns None on SpeculativeError (fold skipped).
    pub fn constant_fold(&self, op: &Op) -> Option<Value> {
        self.protect_speculative_operation(op)?;
        self.execute_nonspec_const(op)
    }

    /// optimizer.py:791-840: protect_speculative_operation(op).
    /// Validates that constant GcRef args are safe to dereference.
    /// Returns None (SpeculativeError) if validation fails.
    fn protect_speculative_operation(&self, op: &Op) -> Option<()> {
        // llmodel.py:555-567: protect_speculative_field.
        // When supports_guard_gc_type is false (majit has no GC type registry),
        // only null check is performed (llmodel.py:556-557).
        if op.opcode.is_getfield() {
            let gcref = match self.get_constant_box(op.arg(0))? {
                Value::Ref(r) => r,
                _ => return None,
            };
            if gcref.is_null() {
                return None; // SpeculativeError
            }
        }
        Some(())
    }

    /// executor.py:555 execute_nonspec_const → _execute_arglist →
    /// do_getfield_gc_i → cpu.bh_getfield_gc_i → llmodel.py:467
    /// read_int_at_mem → llop.raw_load(TYPE, gcref, ofs).
    ///
    /// RPython's constant_fold is ultimately a direct memory read.
    /// Safety is ensured by protect_speculative_operation (null + type
    /// check) BEFORE this function is called.
    ///
    /// GC safety: Ref constants are rooted on the shadow stack via
    /// ConstantPool (gcreftracer.py parity). GC updates shadow stack
    /// entries in place; refresh_from_gc propagates to the HashMap
    /// before optimization reads them. Constants are live during
    /// optimization (no Python allocations trigger GC).
    fn execute_nonspec_const(&self, op: &Op) -> Option<Value> {
        if !op.opcode.is_getfield() {
            return None;
        }
        let arg0 = op.arg(0);
        let resolved = self.get_box_replacement(arg0);
        if !resolved.is_constant() {
            return None;
        }
        let gcref = match self.get_constant_box(arg0)? {
            Value::Ref(r) => r,
            _ => return None,
        };
        if gcref.is_null() {
            return None;
        }
        let descr = op.descr.as_ref()?;
        let fd = descr.as_field_descr()?;
        let addr = gcref.0 + fd.offset();
        // llmodel.py:467-478 read_int_at_mem / read_ref_at_mem dispatch.
        match (fd.field_type(), fd.field_size()) {
            (majit_ir::Type::Int, 8) => Some(Value::Int(unsafe { *(addr as *const i64) })),
            (majit_ir::Type::Int, 4) => {
                if fd.is_field_signed() {
                    Some(Value::Int(unsafe { *(addr as *const i32) as i64 }))
                } else {
                    Some(Value::Int(unsafe { *(addr as *const u32) as i64 }))
                }
            }
            (majit_ir::Type::Int, 2) => {
                if fd.is_field_signed() {
                    Some(Value::Int(unsafe { *(addr as *const i16) as i64 }))
                } else {
                    Some(Value::Int(unsafe { *(addr as *const u16) as i64 }))
                }
            }
            (majit_ir::Type::Int, 1) => Some(Value::Int(unsafe { *(addr as *const u8) as i64 })),
            (majit_ir::Type::Float, 8) => {
                let bits = unsafe { *(addr as *const u64) };
                Some(Value::Float(f64::from_bits(bits)))
            }
            (majit_ir::Type::Ref, _) => {
                let ptr = unsafe { *(addr as *const usize) };
                Some(Value::Ref(majit_ir::GcRef(ptr)))
            }
            _ => None,
        }
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
        self.next_pos = self.num_inputs;
        self.int_bounds.clear();
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
        use crate::optimizeopt::info::Forwarded;
        let r = self.get_box_replacement(opref);
        match self.forwarded.get(r.0 as usize)? {
            Forwarded::Info(info) => Some(info),
            _ => None,
        }
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
        use crate::optimizeopt::info::Forwarded;
        let r = self.get_box_replacement(opref);
        match self.forwarded.get_mut(r.0 as usize)? {
            Forwarded::Info(info) => Some(info),
            _ => None,
        }
    }

    /// info.py: op.set_forwarded(info) — set PtrInfo for an OpRef.
    /// Ensure a PtrInfo exists for the given OpRef. Creates an empty
    /// Instance if none exists, so that set_field can store values.
    pub fn ensure_ptr_info(&mut self, opref: OpRef) {
        use crate::optimizeopt::info::Forwarded;
        let idx = opref.0 as usize;
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        if matches!(self.forwarded[idx], Forwarded::None) {
            self.forwarded[idx] = Forwarded::Info(PtrInfo::instance(None, None));
        }
    }

    /// Set PtrInfo without clearing forwarding.
    /// RPython parity: set PtrInfo at the terminal of opref's forwarding chain.
    /// In RPython, `box.set_forwarded(info)` sets info on the Box directly.
    /// `get_box_replacement(box)` then returns the terminal Box which has the info.
    /// In majit, we follow the Op chain to the terminal OpRef and set Info there.
    fn ensure_ptr_info_preserve_forwarding(&mut self, opref: OpRef, info: PtrInfo) {
        use crate::optimizeopt::info::Forwarded;
        let terminal = self.get_box_replacement(opref);
        let idx = terminal.0 as usize;
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        if matches!(self.forwarded[idx], Forwarded::None) {
            self.forwarded[idx] = Forwarded::Info(info);
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
    /// optimizer.py:437-448: make_nonnull — record that a Ref box is nonnull.
    /// Only sets NonNull if no existing PtrInfo is present.
    pub fn make_nonnull(&mut self, opref: OpRef) {
        let resolved = self.get_box_replacement(opref);
        if resolved.is_constant() {
            return;
        }
        // optimizer.py:446: if opinfo is not None: assert opinfo.is_nonnull(); return
        if self.get_ptr_info(resolved).is_some() {
            return;
        }
        // optimizer.py:448: op.set_forwarded(info.NonNullPtrInfo())
        self.set_ptr_info(resolved, PtrInfo::NonNull { last_guard_pos: -1 });
    }

    /// optimizer.py:453-459: make_nonnull_str — record StrPtrInfo on a string box.
    /// Only sets if no existing PtrInfo is present or existing is not StrPtrInfo.
    pub fn make_nonnull_str(&mut self, opref: OpRef, mode: u8) {
        let resolved = self.get_box_replacement(opref);
        if resolved.is_constant() {
            return;
        }
        if let Some(PtrInfo::Str(_)) = self.get_ptr_info(resolved) {
            return;
        }
        self.set_ptr_info(
            resolved,
            PtrInfo::Str(crate::optimizeopt::info::StrPtrInfo {
                lenbound: None,
                mode,
                length: -1,
                last_guard_pos: -1,
            }),
        );
    }

    pub fn set_ptr_info(&mut self, opref: OpRef, info: PtrInfo) {
        use crate::optimizeopt::info::Forwarded;
        let idx = opref.0 as usize;
        if idx >= self.forwarded.len() {
            self.forwarded.resize(idx + 1, Forwarded::None);
        }
        self.forwarded[idx] = Forwarded::Info(info);
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
        use crate::optimizeopt::info::Forwarded;
        let idx = opref.0 as usize;
        if idx < self.forwarded.len() {
            matches!(self.forwarded[idx], Forwarded::Op(_))
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
    ///
    /// `self_pass_idx` is this pass's own index in the optimizer pipeline.
    /// RPython uses `self.next_optimization` to route lazy-set emissions
    /// starting AFTER the current pass. In majit, pass this index to
    /// `emit_extra` to achieve the same behavior.
    fn emitting_operation(&mut self, _op: &Op, _ctx: &mut OptContext, _self_pass_idx: usize) {}
}
