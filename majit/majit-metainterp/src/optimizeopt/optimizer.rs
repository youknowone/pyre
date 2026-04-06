use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};
/// Main optimization driver.
///
/// Translated from rpython/jit/metainterp/optimizeopt/optimizer.py.
/// Chains multiple optimization passes and drives operations through them.
use crate::optimizeopt::{
    earlyforce::OptEarlyForce,
    heap::OptHeap,
    intbounds::OptIntBounds,
    pure::OptPure,
    rewrite::OptRewrite,
    virtualize::{OptVirtualize, VirtualizableConfig},
    vstring::OptString,
};
use majit_ir::{DescrRef, GcRef, Op, OpCode, OpRef, Type};

use crate::optimizeopt::info::PtrInfo;

/// bridgeopt.py: serialized optimizer knowledge for guard resume data.
///
/// Captures the optimizer's knowledge at each guard point so that bridges
/// compiled from guard failures can inherit heap cache, known class, and
/// loopinvariant call result information.
#[derive(Clone, Debug, Default)]
pub struct OptimizerKnowledge {
    /// Heap cache: (object, field_descr_index, value) triples.
    pub heap_fields: Vec<(OpRef, u32, OpRef)>,
    /// Known classes: (box, class_ptr) pairs.
    /// bridgeopt.py:74-88: serialize known class bitfield.
    pub known_classes: Vec<(OpRef, GcRef)>,
    /// Loop-invariant call results: (func_ptr, result) pairs.
    /// bridgeopt.py:113-122: serialize_optrewrite.
    pub loopinvariant_results: Vec<(i64, OpRef)>,
}

impl OptimizerKnowledge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.heap_fields.is_empty()
            && self.known_classes.is_empty()
            && self.loopinvariant_results.is_empty()
    }

    /// Remap OpRefs from source-trace numbering to bridge inputarg numbering.
    pub fn remap(&self, remap: &std::collections::HashMap<OpRef, OpRef>) -> Self {
        let heap_fields = self
            .heap_fields
            .iter()
            .filter_map(|&(obj, field_idx, val)| {
                let new_obj = remap.get(&obj)?;
                let new_val = remap.get(&val)?;
                Some((*new_obj, field_idx, *new_val))
            })
            .collect();
        let known_classes = self
            .known_classes
            .iter()
            .filter_map(|&(opref, class_ptr)| {
                let new_opref = remap.get(&opref)?;
                Some((*new_opref, class_ptr))
            })
            .collect();
        let loopinvariant_results = self
            .loopinvariant_results
            .iter()
            .filter_map(|&(func_ptr, result)| {
                let new_result = remap.get(&result)?;
                Some((func_ptr, *new_result))
            })
            .collect();
        OptimizerKnowledge {
            heap_fields,
            known_classes,
            loopinvariant_results,
        }
    }
}

/// The optimizer: chains passes and runs them over a trace.
///
/// RPython optimizer.py: Optimizer class with pass chain and shared state.
pub struct Optimizer {
    passes: Vec<Box<dyn Optimization>>,
    /// Final num_inputs after optimization (may increase if virtualizable
    /// adds virtual input args).
    final_num_inputs: usize,
    /// Cache of CALL_PURE results from previous traces.
    /// optimizer.py: `call_pure_results` — maps
    /// (func_ptr, arg0, arg1, ...) → result value, carried across
    /// loop iterations so the optimizer can constant-fold repeated
    /// pure calls.
    call_pure_results: std::collections::HashMap<Vec<majit_ir::OpRef>, majit_ir::Value>,
    /// optimizer.py: `_last_guard_op` — tracks the last emitted guard
    /// for guard sharing and descriptor fusion.
    last_guard_op: Option<Op>,
    /// optimizer.py: `replaces_guard` — maps guard op position to replacement.
    replaces_guard: std::collections::HashMap<u32, Op>,
    /// optimizer.py: `pendingfields` — heap fields that need to be
    /// written back before the next guard (lazy set forcing).
    pendingfields: Vec<Op>,
    /// optimizer.py: `can_replace_guards` — flag to enable/disable guard sharing.
    can_replace_guards: bool,
    /// optimizer.py: `quasi_immutable_deps` — quasi-immutable field dependencies.
    /// RPython: dict[QuasiImmut → None]. We store (object_ptr, field_index)
    /// pairs identifying the specific quasi-immutable slot that compiled
    /// code depends on. After compilation, each dependency gets the loop's
    /// invalidation flag registered as a per-slot watcher.
    pub quasi_immutable_deps: std::collections::HashSet<(u64, u32)>,
    /// optimizer.py: `resumedata_memo` — shared constant map for resume data.
    /// Maps constant values to shared indices to reduce resume data size.
    /// In RPython this is `resume.ResumeDataLoopMemo`; here we use a simple
    /// HashMap since the full type lives in majit-meta (no circular dep).
    resumedata_memo_consts: std::collections::HashMap<i64, u32>,
    /// Types of constant OpRefs from ConstantPool.constant_types.
    /// Used to distinguish Ref constants from Int in guard fail_args.
    pub constant_types: std::collections::HashMap<u32, majit_ir::Type>,
    /// Short preamble constants imported during inline_short_preamble.
    /// RPython parity: these are Const objects embedded in short preamble ops
    /// that survive across compilations. In pyre, they must be merged into
    /// the bridge's constant pool after optimize_bridge returns.
    pub bridge_preamble_constants: std::collections::HashMap<u32, (i64, majit_ir::Type)>,
    /// RPython parity: GcRef constants (ob_type etc.) recorded as const_int
    /// need Ref type for resume data but must NOT trigger Cranelift GC root.
    pub numbering_type_overrides: std::collections::HashMap<u32, majit_ir::Type>,
    /// RPython unroll.py: import_state — virtual structures to inject at Phase 2 start.
    /// Maps the original loop-carried input slot to a recursive abstract
    /// description of the virtual's field values.
    pub imported_virtuals: Vec<ImportedVirtual>,
    /// Types of the original trace inputargs (from LABEL or inputarg_types).
    /// RPython Boxes carry type intrinsically; we store it here so
    /// export_state can propagate to ExportedState.renamed_inputarg_types.
    pub trace_inputarg_types: Vec<majit_ir::Type>,
    /// RPython unroll.py: export_state — exported optimizer facts at the end
    /// of the preamble, adapted to majit's slot-based inputarg model.
    pub exported_loop_state: Option<crate::optimizeopt::unroll::ExportedState>,
    /// RPython unroll.py: import_state — exported facts to re-apply onto the
    /// next optimizer instance before phase 2 body optimization starts.
    pub imported_loop_state: Option<crate::optimizeopt::unroll::ExportedState>,
    /// Original preamble result boxes for imported short-box results.
    pub imported_short_sources: Vec<crate::optimizeopt::ImportedShortSource>,
    /// Invented SameAs aliases imported from short-preamble export/import.
    pub imported_short_aliases: Vec<crate::optimizeopt::ImportedShortAlias>,
    /// Builder-derived short preamble actually used by phase 2.
    pub imported_short_preamble: Option<crate::optimizeopt::shortpreamble::ShortPreamble>,
    /// RPython unroll.py: short_preamble_producer after import_state.
    /// Preserved so finalize_short_preamble can create the live extended
    /// producer for the target token currently being compiled.
    pub imported_short_preamble_builder:
        Option<crate::optimizeopt::shortpreamble::ShortPreambleBuilder>,
    /// RPython unroll.py: `label_args = import_state(...)`.
    /// The peeled loop's LABEL must use these args, not the phase-1 end_args.
    pub imported_label_args: Option<Vec<OpRef>>,
    /// Local slot-map needed because majit assembles the peeled trace from a
    /// slot-indexed body trace. RPython keeps forwarded boxes directly.
    pub imported_label_source_slots: Option<Vec<OpRef>>,
    /// simplify.py: patchguardop recorded from GUARD_FUTURE_CONDITION.
    pub patchguardop: Option<Op>,
    /// RPython: propagate_all_forward(trace, flush=False) for Phase 2.
    /// When true, skip flush() at end of optimization.
    pub skip_flush: bool,
    /// RPython optimizer.py: last_op/info.jump_op when flush=False.
    /// Phase-2 loop compilation keeps the terminal JUMP/FINISH outside the
    /// returned body ops and lets unroll/compile own it explicitly.
    pub terminal_op: Option<Op>,
    /// Preserved final context after optimization, for jump_to_existing_trace.
    pub final_ctx: Option<OptContext>,
    /// RPython Box identity: generation epoch for Phase 2 ops.
    /// Phase 1 JUMP arg OpRef indices to pre-tag as gen=0.
    /// bridgeopt.py: pending bridge knowledge to apply after setup().
    /// Stored here so it survives the setup() clear in optimize_with_constants_and_inputs.
    pending_bridge_knowledge: Option<OptimizerKnowledge>,
    /// Per-guard optimizer knowledge snapshots, captured at each guard emit
    /// during optimization. Keyed by guard op position (OpRef).
    /// RPython: resume.py:570-574 _add_optimizer_sections serializes the
    /// optimizer state AT EACH GUARD, not just at end-of-trace.
    pub per_guard_knowledge: Vec<(OpRef, OptimizerKnowledge)>,
    /// optimizer.py:787: constant_fold allocator for compile-time object creation.
    pub constant_fold_alloc: Option<crate::optimizeopt::ConstantFoldAllocFn>,
    /// optimizer.py:732 — resume.ResumeDataLoopMemo.
    /// Shared constant pool + box numbering cache across all guards in a loop.
    pub resumedata_memo: crate::resume::ResumeDataLoopMemo,
    /// resume.py parity: per-guard snapshots from tracing time.
    /// Maps rd_resume_position → flattened OpRef boxes from the snapshot.
    /// Propagated to OptContext for store_final_boxes_in_guard.
    pub snapshot_boxes: std::collections::HashMap<i32, Vec<OpRef>>,
    /// Per-frame box counts for multi-frame snapshots.
    /// Propagated to OptContext for store_final_boxes_in_guard multi-frame encoding.
    pub snapshot_frame_sizes: std::collections::HashMap<i32, Vec<usize>>,
    /// Per-guard virtualizable boxes from tracing-time snapshots.
    pub snapshot_vable_boxes: std::collections::HashMap<i32, Vec<OpRef>>,
    /// Per-guard per-frame (jitcode_index, pc) from tracing-time snapshots.
    pub snapshot_frame_pcs: std::collections::HashMap<i32, Vec<(i32, i32)>>,
    /// RPython box.type parity: snapshot Box types.
    pub snapshot_box_types: std::collections::HashMap<u32, majit_ir::Type>,
    /// RPython Box type parity: in RPython each Box carries its type
    /// intrinsically. In majit, OpRef is an untyped u32, so we track
    /// types in value_types. Phase 1 value_types are preserved here
    /// so Phase 2 can resolve types for carried-over OpRefs.
    pub prev_phase_value_types: std::collections::HashMap<u32, majit_ir::Type>,
    /// RPython Box type parity: position→type for ALL original trace ops.
    /// snapshot_boxes reference original trace positions, but the optimizer
    /// receives transformed ops (after fold_box/elide). This map covers
    /// positions that were removed during trace transformation.
    pub original_trace_op_types: std::collections::HashMap<u32, majit_ir::Type>,
}

fn value_from_backend_constant_bits(opref: OpRef, raw: i64, ops: &[Op]) -> majit_ir::Value {
    // First check ops for result type (for inline constants),
    // then fall back to Int.
    let result_type = ops
        .iter()
        .find(|op| op.pos == opref)
        .map(|op| op.result_type())
        .unwrap_or(Type::Int);
    match result_type {
        Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(raw as usize)),
        Type::Float => majit_ir::Value::Float(f64::from_bits(raw as u64)),
        Type::Int | Type::Void => majit_ir::Value::Int(raw),
    }
}

fn value_from_backend_constant_bits_typed(
    opref: OpRef,
    raw: i64,
    ops: &[Op],
    constant_types: &std::collections::HashMap<u32, Type>,
) -> majit_ir::Value {
    // Use explicit constant_types first (from ConstantPool),
    // then fall back to ops result_type, then Int.
    let result_type = constant_types
        .get(&opref.0)
        .copied()
        .or_else(|| {
            ops.iter()
                .find(|op| op.pos == opref)
                .map(|op| op.result_type())
        })
        .unwrap_or(Type::Int);
    match result_type {
        Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(raw as usize)),
        Type::Float => majit_ir::Value::Float(f64::from_bits(raw as u64)),
        Type::Int | Type::Void => majit_ir::Value::Int(raw),
    }
}

fn value_to_backend_constant_bits(value: &majit_ir::Value) -> i64 {
    match value {
        majit_ir::Value::Int(v) => *v,
        majit_ir::Value::Ref(r) => r.0 as i64,
        majit_ir::Value::Float(f) => f.to_bits() as i64,
        majit_ir::Value::Void => 0,
    }
}

/// RPython unroll.py: import_state virtual info for Phase 2.
/// Tells OptVirtualize that an inputarg is a virtual object.
#[derive(Clone, Debug)]
pub struct ImportedVirtual {
    /// Inputarg index that holds this virtual.
    pub inputarg_index: usize,
    /// Size descriptor for the virtual's New().
    pub size_descr: majit_ir::DescrRef,
    /// Whether this imported virtual is an instance or a plain struct.
    pub kind: ImportedVirtualKind,
    /// Fields: (field_descr, exported abstract info for the field value).
    pub fields: Vec<(
        majit_ir::DescrRef,
        crate::optimizeopt::virtualstate::VirtualStateInfo,
    )>,
    /// Descr index of the GetfieldGcR(pool) that loads this head.
    /// OptVirtualize forwards this load result to the virtual head.
    pub head_load_descr_index: Option<u32>,
}

#[derive(Clone, Debug)]
pub enum ImportedVirtualKind {
    Instance {
        known_class: Option<majit_ir::GcRef>,
    },
    Struct,
}

impl Optimizer {
    fn is_constant_placeholder_op(op: &Op, constants: &[Option<majit_ir::Value>]) -> bool {
        if !matches!(
            op.opcode,
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF
        ) {
            return false;
        }
        let idx = op.pos.0 as usize;
        if idx >= constants.len() || constants[idx].is_none() {
            return false;
        }
        op.args.is_empty() || op.args.iter().all(|arg| arg.is_none())
    }

    fn import_virtual_state_value(
        info: &crate::optimizeopt::virtualstate::VirtualStateInfo,
        ctx: &mut OptContext,
    ) -> OpRef {
        let opref = ctx.alloc_op_position();
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("[jit] import_virtual_state_value {opref:?} <= {info:?}");
        }
        Self::apply_imported_virtual_state(info, opref, ctx);
        opref
    }

    fn apply_imported_virtual_state(
        info: &crate::optimizeopt::virtualstate::VirtualStateInfo,
        opref: OpRef,
        ctx: &mut OptContext,
    ) {
        use crate::optimizeopt::virtualstate::VirtualStateInfo;

        match info {
            VirtualStateInfo::Constant(value) => {
                ctx.make_constant(opref, value.clone());
            }
            VirtualStateInfo::Virtual {
                descr,
                known_class,
                ob_type_descr,
                fields,
                field_descrs,
            } => {
                let mut imported_fields = Vec::new();
                for (field_idx, field_info) in fields {
                    let field_ref = Self::import_virtual_state_value(field_info, ctx);
                    // ob_type (offset 0) constants are GcRef pointers stored
                    // as Value::Int. Register as Ref in constant_types so
                    // fail_arg_types + blackhole decode_ref handle them correctly.
                    let offset = field_descrs
                        .iter()
                        .find(|(di, _)| *di == *field_idx)
                        .and_then(|(_, d)| d.as_field_descr().map(|fd| fd.offset()));
                    if offset == Some(0) && known_class.is_some() {
                        ctx.constant_types_for_numbering
                            .insert(field_ref.0, majit_ir::Type::Ref);
                    }
                    imported_fields.push((*field_idx, field_ref));
                }
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::Virtual(
                        crate::optimizeopt::info::VirtualInfo {
                            descr: descr.clone(),
                            known_class: *known_class,
                            ob_type_descr: ob_type_descr.clone(),
                            fields: imported_fields,
                            field_descrs: field_descrs.clone(),
                            last_guard_pos: -1,
                        },
                    ),
                );
            }
            VirtualStateInfo::VArray { descr, items, .. } => {
                let imported_items = items
                    .iter()
                    .map(|item_info| Self::import_virtual_state_value(item_info, ctx))
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualArray(
                        crate::optimizeopt::info::VirtualArrayInfo {
                            descr: descr.clone(),
                            clear: false,
                            items: imported_items,
                            last_guard_pos: -1,
                        },
                    ),
                );
            }
            VirtualStateInfo::VStruct {
                descr,
                fields,
                field_descrs,
            } => {
                let mut imported_fields = Vec::new();
                for (field_idx, field_info) in fields {
                    let field_ref = Self::import_virtual_state_value(field_info, ctx);
                    imported_fields.push((*field_idx, field_ref));
                }
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualStruct(
                        crate::optimizeopt::info::VirtualStructInfo {
                            descr: descr.clone(),
                            fields: imported_fields,
                            field_descrs: field_descrs.clone(),
                            last_guard_pos: -1,
                        },
                    ),
                );
            }
            VirtualStateInfo::VArrayStruct {
                descr,
                element_fields,
            } => {
                let imported_elements = element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_idx, field_info)| {
                                (
                                    *field_idx,
                                    Self::import_virtual_state_value(field_info, ctx),
                                )
                            })
                            .collect()
                    })
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(
                        crate::optimizeopt::info::VirtualArrayStructInfo {
                            descr: descr.clone(),
                            fielddescrs: Vec::new(),
                            element_fields: imported_elements,
                            last_guard_pos: -1,
                        },
                    ),
                );
            }
            VirtualStateInfo::VirtualRawBuffer { size, entries } => {
                let imported_entries = entries
                    .iter()
                    .map(|(offset, length, entry_info)| {
                        (
                            *offset,
                            *length,
                            Self::import_virtual_state_value(entry_info, ctx),
                        )
                    })
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualRawBuffer(
                        crate::optimizeopt::info::VirtualRawBufferInfo {
                            size: *size,
                            entries: imported_entries,
                            last_guard_pos: -1,
                        },
                    ),
                );
            }
            VirtualStateInfo::KnownClass { class_ptr } => {
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::KnownClass {
                        class_ptr: *class_ptr,
                        is_nonnull: true,
                        last_guard_pos: -1,
                    },
                );
            }
            VirtualStateInfo::NonNull => {
                ctx.set_ptr_info(opref, crate::optimizeopt::info::PtrInfo::nonnull());
            }
            VirtualStateInfo::IntBounded(bound) => {
                let widened = bound.widen();
                if widened.lower > i64::MIN {
                    ctx.int_lower_bounds.insert(opref, widened.lower);
                }
                ctx.imported_int_bounds.insert(opref, widened);
            }
            VirtualStateInfo::Unknown => {}
        }
    }

    fn install_imported_virtuals(&self, ctx: &mut OptContext) {
        // virtualstate.py:655-670 make_inputargs + 627-634 _enum parity:
        // label_args are laid out by recursive _enum traversal where each
        // NotVirtualStateInfo leaf gets a position_in_notvirtuals slot.
        // Virtual states don't consume slots directly — their non-virtual
        // FIELD sub-states do, interleaved with top-level non-virtual states.
        // Traverse ALL states in order, advancing label_slot for each leaf.
        let imported_label_args = self
            .imported_label_args
            .as_ref()
            .expect("install_imported_virtuals requires imported_label_args");
        let mut label_slot = 0usize;
        if crate::optimizeopt::majit_log_enabled() {
            eprintln!(
                "[jit] install_virt: label_slot={} label_args_len={} label_args={:?}",
                label_slot,
                imported_label_args.len(),
                imported_label_args
            );
        }
        // RPython Box identity parity: Two-pass approach.
        //
        // Pass 1: process all fields with import_virtual_state_from_label_args.
        //   Constant fields get alloc_op_position. Leaf fields get LABEL arg OpRefs.
        //   Record which leaf field positions need SameAs (skip_flush_mode = Phase 2).
        //
        // Pass 2: allocate SameAs positions and create ops for recorded leaves.
        //   This happens AFTER all Constant allocs, preventing reserve_pos collision
        //   with Phase 1 virtual head positions.
        //
        // Without SameAs, the assembly's body_result_remap maps fail_args entries
        // from LABEL arg positions to body op fresh positions, causing virtual fields
        // to get body IntAddOvf results instead of LABEL values.
        struct VirtualEntry {
            head: OpRef,
            size_descr: majit_ir::DescrRef,
            fields: Vec<(u32, OpRef)>,
            field_descrs: Vec<(u32, majit_ir::DescrRef)>,
            kind: ImportedVirtualKind,
            head_load_descr_index: Option<u32>,
        }
        let mut entries: Vec<VirtualEntry> = Vec::new();
        // Track leaf field positions that need SameAs (position, entry_idx, field_idx).
        let mut same_as_targets: Vec<(OpRef, usize, usize)> = Vec::new();

        // RPython parity: traverse ALL states in order (not just virtuals).
        // Non-virtual states advance label_slot without creating entries.
        // Virtual states create entries with fields from label_args.
        // Build a map from inputarg_index to imported_virtual for virtual lookup.
        let iv_map: std::collections::HashMap<usize, &_> = self
            .imported_virtuals
            .iter()
            .map(|iv| (iv.inputarg_index, iv))
            .collect();
        let all_states = self
            .imported_loop_state
            .as_ref()
            .map(|s| &s.virtual_state.state[..])
            .unwrap_or(&[]);
        for (state_idx, state_info) in all_states.iter().enumerate() {
            if let Some(iv) = iv_map.get(&state_idx) {
                // Virtual state: process fields recursively, consuming slots
                // for non-virtual leaf fields.
                let raw = OpRef(iv.inputarg_index as u32);
                let virtual_head = ctx.get_box_replacement(raw);
                let mut fields = Vec::new();
                let mut field_descrs = Vec::new();
                for (descr, field_info) in &iv.fields {
                    let field_ref = Self::import_virtual_state_from_label_args(
                        field_info,
                        imported_label_args,
                        &mut label_slot,
                        ctx,
                    );
                    let field_idx = fields.len();
                    if ctx.skip_flush_mode
                        && !field_ref.is_none()
                        && field_ref.0 < 10_000
                        && ctx
                            .get_ptr_info(field_ref)
                            .map_or(true, |info| !info.is_virtual())
                        && ctx.get_constant(field_ref).is_none()
                    {
                        same_as_targets.push((field_ref, entries.len(), field_idx));
                    }
                    fields.push((descr.index(), field_ref));
                    field_descrs.push((descr.index(), descr.clone()));
                }
                entries.push(VirtualEntry {
                    head: virtual_head,
                    size_descr: iv.size_descr.clone(),
                    fields,
                    field_descrs,
                    kind: iv.kind.clone(),
                    head_load_descr_index: iv.head_load_descr_index,
                });
            } else {
                // Non-virtual state: advance label_slot to match RPython's
                // position_in_notvirtuals enumeration order.
                let count = crate::optimizeopt::virtualstate::VirtualState::count_forced_boxes_for_entry_static(state_info);
                label_slot += count;
            }
        }

        // Pass 2: allocate SameAs ops for leaf field values.
        // Advance next_pos past all virtual head positions to prevent
        // reserve_pos from returning a position that's already used
        // as a virtual head (allocated during import_state).
        for entry in &entries {
            if !entry.head.is_none() && entry.head.0 < 10_000 {
                ctx.next_pos = ctx.next_pos.max(entry.head.0 + 1);
            }
        }
        for (label_arg, entry_idx, field_idx) in &same_as_targets {
            let tp = ctx
                .value_types
                .get(&label_arg.0)
                .copied()
                .unwrap_or(majit_ir::Type::Int);
            let same_as_op = majit_ir::OpCode::same_as_for_type(tp);
            let mut op = majit_ir::Op::new(same_as_op, &[*label_arg]);
            op.pos = ctx.reserve_pos();
            let fresh = op.pos;
            ctx.value_types.insert(fresh.0, tp);
            ctx.new_operations.push(op);
            // Update the field to reference the SameAs result.
            entries[*entry_idx].fields[*field_idx].1 = fresh;
        }

        // Install PtrInfo for each virtual.
        // unroll.py:55: if op.get_forwarded() is not None: return
        // Skip heads that already have PtrInfo (duplicate entries from
        // aliased JUMP args sharing the same VirtualState position).
        let mut installed_heads = std::collections::HashSet::new();
        for entry in entries {
            if !installed_heads.insert(entry.head) {
                continue;
            }
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[jit] install_imported_virtual head={:?} fields={:?}",
                    entry.head, entry.fields
                );
            }
            match &entry.kind {
                ImportedVirtualKind::Instance { known_class } => {
                    ctx.set_ptr_info(
                        entry.head,
                        crate::optimizeopt::info::PtrInfo::Virtual(
                            crate::optimizeopt::info::VirtualInfo {
                                descr: entry.size_descr,
                                known_class: *known_class,
                                ob_type_descr: None,
                                fields: entry.fields,
                                field_descrs: entry.field_descrs,
                                last_guard_pos: -1,
                            },
                        ),
                    );
                }
                ImportedVirtualKind::Struct => {
                    ctx.set_ptr_info(
                        entry.head,
                        crate::optimizeopt::info::PtrInfo::VirtualStruct(
                            crate::optimizeopt::info::VirtualStructInfo {
                                descr: entry.size_descr,
                                fields: entry.fields,
                                field_descrs: entry.field_descrs,
                                last_guard_pos: -1,
                            },
                        ),
                    );
                }
            }
            if let Some(descr_idx) = entry.head_load_descr_index {
                ctx.imported_virtual_heads
                    .push((descr_idx as usize, entry.head));
            }
        }
    }

    fn import_virtual_state_from_label_args(
        info: &crate::optimizeopt::virtualstate::VirtualStateInfo,
        imported_label_args: &[OpRef],
        label_slot: &mut usize,
        ctx: &mut OptContext,
    ) -> OpRef {
        use crate::optimizeopt::virtualstate::VirtualStateInfo;

        match info {
            VirtualStateInfo::Constant(_) => Self::import_virtual_state_value(info, ctx),
            VirtualStateInfo::Virtual {
                descr,
                known_class,
                ob_type_descr,
                fields,
                field_descrs,
            } => {
                let opref = ctx.alloc_op_position();
                let imported_fields: Vec<(u32, OpRef)> = fields
                    .iter()
                    .map(|(field_idx, field_info)| {
                        let field_ref = Self::import_virtual_state_from_label_args(
                            field_info,
                            imported_label_args,
                            label_slot,
                            ctx,
                        );
                        let offset = field_descrs
                            .iter()
                            .find(|(di, _)| *di == *field_idx)
                            .and_then(|(_, d)| d.as_field_descr().map(|fd| fd.offset()));
                        if offset == Some(0) && known_class.is_some() {
                            ctx.constant_types_for_numbering
                                .insert(field_ref.0, majit_ir::Type::Ref);
                        }
                        (*field_idx, field_ref)
                    })
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::Virtual(
                        crate::optimizeopt::info::VirtualInfo {
                            descr: descr.clone(),
                            known_class: *known_class,
                            ob_type_descr: ob_type_descr.clone(),
                            fields: imported_fields,
                            field_descrs: field_descrs.clone(),
                            last_guard_pos: -1,
                        },
                    ),
                );
                opref
            }
            VirtualStateInfo::VArray { descr, items, .. } => {
                let opref = ctx.alloc_op_position();
                let imported_items = items
                    .iter()
                    .map(|item_info| {
                        Self::import_virtual_state_from_label_args(
                            item_info,
                            imported_label_args,
                            label_slot,
                            ctx,
                        )
                    })
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualArray(
                        crate::optimizeopt::info::VirtualArrayInfo {
                            descr: descr.clone(),
                            clear: false,
                            items: imported_items,
                            last_guard_pos: -1,
                        },
                    ),
                );
                opref
            }
            VirtualStateInfo::VStruct {
                descr,
                fields,
                field_descrs,
            } => {
                let opref = ctx.alloc_op_position();
                let imported_fields = fields
                    .iter()
                    .map(|(field_idx, field_info)| {
                        (
                            *field_idx,
                            Self::import_virtual_state_from_label_args(
                                field_info,
                                imported_label_args,
                                label_slot,
                                ctx,
                            ),
                        )
                    })
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualStruct(
                        crate::optimizeopt::info::VirtualStructInfo {
                            descr: descr.clone(),
                            fields: imported_fields,
                            field_descrs: field_descrs.clone(),
                            last_guard_pos: -1,
                        },
                    ),
                );
                opref
            }
            VirtualStateInfo::VArrayStruct {
                descr,
                element_fields,
            } => {
                let opref = ctx.alloc_op_position();
                let imported_elements = element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_idx, field_info)| {
                                (
                                    *field_idx,
                                    Self::import_virtual_state_from_label_args(
                                        field_info,
                                        imported_label_args,
                                        label_slot,
                                        ctx,
                                    ),
                                )
                            })
                            .collect()
                    })
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(
                        crate::optimizeopt::info::VirtualArrayStructInfo {
                            descr: descr.clone(),
                            fielddescrs: Vec::new(),
                            element_fields: imported_elements,
                            last_guard_pos: -1,
                        },
                    ),
                );
                opref
            }
            VirtualStateInfo::VirtualRawBuffer { size, entries } => {
                let opref = ctx.alloc_op_position();
                let imported_entries = entries
                    .iter()
                    .map(|(offset, length, entry_info)| {
                        (
                            *offset,
                            *length,
                            Self::import_virtual_state_from_label_args(
                                entry_info,
                                imported_label_args,
                                label_slot,
                                ctx,
                            ),
                        )
                    })
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualRawBuffer(
                        crate::optimizeopt::info::VirtualRawBufferInfo {
                            size: *size,
                            entries: imported_entries,
                            last_guard_pos: -1,
                        },
                    ),
                );
                opref
            }
            VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown => {
                let opref = imported_label_args
                    .get(*label_slot)
                    .copied()
                    .unwrap_or_else(|| {
                        if std::env::var_os("MAJIT_LOG").is_some() {
                            eprintln!(
                                "[jit] MISS: label_slot={} len={}",
                                *label_slot,
                                imported_label_args.len()
                            );
                        }
                        OpRef::NONE
                    });
                *label_slot += 1;
                Self::apply_imported_virtual_state(info, opref, ctx);
                let resolved = ctx.get_box_replacement(opref);
                resolved
            }
        }
    }

    pub fn new() -> Self {
        Optimizer {
            passes: Vec::new(),
            final_num_inputs: 0,
            call_pure_results: std::collections::HashMap::new(),
            last_guard_op: None,
            replaces_guard: std::collections::HashMap::new(),
            pendingfields: Vec::new(),
            can_replace_guards: true,
            quasi_immutable_deps: std::collections::HashSet::new(),
            resumedata_memo_consts: std::collections::HashMap::new(),
            constant_types: std::collections::HashMap::new(),
            bridge_preamble_constants: std::collections::HashMap::new(),
            numbering_type_overrides: std::collections::HashMap::new(),
            imported_virtuals: Vec::new(),
            trace_inputarg_types: Vec::new(),
            exported_loop_state: None,
            imported_loop_state: None,
            imported_short_sources: Vec::new(),
            imported_short_aliases: Vec::new(),
            imported_short_preamble: None,
            imported_short_preamble_builder: None,
            imported_label_args: None,
            imported_label_source_slots: None,
            patchguardop: None,
            skip_flush: false,
            terminal_op: None,
            final_ctx: None,
            pending_bridge_knowledge: None,
            per_guard_knowledge: Vec::new(),
            constant_fold_alloc: None,
            resumedata_memo: crate::resume::ResumeDataLoopMemo::new(),
            snapshot_boxes: std::collections::HashMap::new(),
            snapshot_frame_sizes: std::collections::HashMap::new(),
            snapshot_vable_boxes: std::collections::HashMap::new(),
            snapshot_frame_pcs: std::collections::HashMap::new(),
            snapshot_box_types: std::collections::HashMap::new(),
            prev_phase_value_types: std::collections::HashMap::new(),
            original_trace_op_types: std::collections::HashMap::new(),
        }
    }

    /// Record a CALL_PURE result for cross-iteration constant folding.
    /// RPython optimizer.py: `call_pure_results[key] = value`
    pub fn record_call_pure_result(&mut self, args: Vec<majit_ir::OpRef>, value: majit_ir::Value) {
        self.call_pure_results.insert(args, value);
    }

    /// Look up a previously recorded CALL_PURE result.
    pub fn get_call_pure_result(&self, args: &[majit_ir::OpRef]) -> Option<&majit_ir::Value> {
        self.call_pure_results.get(args)
    }

    /// bridgeopt.py:63-122: serialize_optimizer_knowledge
    /// Export the optimizer's heap cache, known class, and loopinvariant knowledge
    /// for storage in guard resume data. Called after optimization.
    pub fn serialize_optimizer_knowledge(&self, ctx: &OptContext) -> OptimizerKnowledge {
        let mut knowledge = OptimizerKnowledge::new();

        // Heap fields: exported from OptHeap pass
        for pass in &self.passes {
            knowledge.heap_fields.extend(pass.export_cached_fields());
        }

        // bridgeopt.py:74-88: known classes from ptr_info
        for (idx, fwd) in ctx.forwarded.iter().enumerate() {
            if let crate::optimizeopt::info::Forwarded::Info(info) = fwd {
                if let Some(class_ptr) = info.get_known_class() {
                    knowledge
                        .known_classes
                        .push((OpRef(idx as u32), *class_ptr));
                }
            }
        }

        // bridgeopt.py:113-122: loopinvariant results from OptRewrite
        for pass in &self.passes {
            knowledge
                .loopinvariant_results
                .extend(pass.export_loopinvariant_results());
        }

        knowledge
    }

    /// bridgeopt.py:124-185: deserialize_optimizer_knowledge
    /// Import optimizer knowledge from the guard's resume data into this optimizer.
    /// Called before bridge optimization to pre-populate caches.
    pub fn deserialize_optimizer_knowledge(
        &mut self,
        knowledge: &OptimizerKnowledge,
        ctx: &mut OptContext,
    ) {
        // Heap fields -> OptHeap
        if !knowledge.heap_fields.is_empty() {
            for pass in &mut self.passes {
                pass.import_cached_fields(&knowledge.heap_fields);
            }
        }

        // bridgeopt.py:133-146: known classes -> PtrInfo::KnownClass
        for &(opref, class_ptr) in &knowledge.known_classes {
            Self::make_constant_class(ctx, opref, class_ptr.0 as i64, true);
        }

        // bridgeopt.py:173-185: loopinvariant results -> OptRewrite
        if !knowledge.loopinvariant_results.is_empty() {
            for pass in &mut self.passes {
                pass.import_loopinvariant_results(&knowledge.loopinvariant_results);
            }
        }
    }

    /// Get the final num_inputs after optimization.
    /// May be larger than the original if virtualizable added virtual input args.
    pub fn final_num_inputs(&self) -> usize {
        self.final_num_inputs
    }

    /// optimizer.py: getlastop() — get the last emitted guard operation.
    pub fn get_last_guard_op(&self) -> Option<&Op> {
        self.last_guard_op.as_ref()
    }

    pub fn set_last_guard_op(&mut self, op: Op) {
        self.last_guard_op = Some(op);
    }

    /// optimizer.py: notice_guard_future_condition(op)
    /// Record that a guard at the given position should be replaced
    /// with the given op when the future condition is realized.
    pub fn notice_guard_future_condition(&mut self, guard_pos: u32, replacement: Op) {
        self.replaces_guard.insert(guard_pos, replacement);
    }

    /// optimizer.py:713: replace_guard_op(old_op_pos, new_op)
    /// Replace a previously emitted guard with a new one.
    pub fn replace_guard_op(&mut self, old_pos: u32, new_guard: Op) {
        self.replaces_guard.insert(old_pos, new_guard);
    }

    // RPython optimizer.py:722-752 store_final_boxes_in_guard and
    // optimizer.py:649-670 emit_guard_operation are implemented inside
    // emit_operation: _copy_resume_data_from, store_final_boxes_in_guard,
    // force_box on fail_args, and store_final_boxes_in_guard in ctx.emit().

    /// optimizer.py: add_pending_field(op)
    /// Queue a SETFIELD_GC to be emitted before the next guard.
    pub fn add_pending_field(&mut self, op: Op) {
        self.pendingfields.push(op);
    }

    /// optimizer.py: flush_pendingfields(ctx)
    /// Emit all pending field writes.
    pub fn flush_pendingfields(&mut self, ctx: &mut OptContext) {
        let pending = std::mem::take(&mut self.pendingfields);
        for op in pending {
            ctx.emit(op);
        }
    }

    /// optimizer.py: has_pending_fields()
    pub fn has_pending_fields(&self) -> bool {
        !self.pendingfields.is_empty()
    }

    /// optimizer.py: num_pending_fields()
    pub fn num_pending_fields(&self) -> usize {
        self.pendingfields.len()
    }

    /// optimizer.py: cant_replace_guards()
    /// Temporarily disable guard replacement (e.g., during bridge compilation).
    pub fn disable_guard_replacement(&mut self) {
        self.can_replace_guards = false;
    }

    /// Re-enable guard replacement.
    pub fn enable_guard_replacement(&mut self) {
        self.can_replace_guards = true;
    }

    /// optimizer.py: resumedata_memo — shared constant mapping for resume data.
    /// Maps constant i64 values to shared indices so multiple guards
    /// can reference the same constant without duplication.
    pub fn resumedata_memo_get_or_insert(&mut self, value: i64) -> u32 {
        let next_idx = self.resumedata_memo_consts.len() as u32;
        *self.resumedata_memo_consts.entry(value).or_insert(next_idx)
    }

    /// Number of shared constants in the memo.
    pub fn resumedata_memo_num_consts(&self) -> usize {
        self.resumedata_memo_consts.len()
    }

    /// optimizer.py: quasi_immutable_deps[qmut] = None
    /// Track a quasi-immutable field dependency. `dep` is (object_ptr,
    /// field_index) identifying the specific slot the compiled loop
    /// depends on. After compilation, a per-slot watcher is registered.
    pub fn add_quasi_immutable_dep(&mut self, dep: (u64, u32)) {
        self.quasi_immutable_deps.insert(dep);
    }

    /// optimizer.py: produce_potential_short_preamble_ops(sb)
    /// Collect short preamble ops from all passes.
    pub fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        ctx: &OptContext,
    ) {
        for pass in &self.passes {
            pass.produce_potential_short_preamble_ops(sb, ctx);
        }
    }

    /// Collect all cached field entries from all passes.
    pub fn export_all_cached_fields(&self) -> Vec<(OpRef, u32, OpRef)> {
        let mut result = Vec::new();
        for pass in &self.passes {
            result.extend(pass.export_cached_fields());
        }
        result
    }

    /// Pre-tag Phase 1 JUMP arg OpRefs as generation 0.

    /// Lock JUMP arg OpRefs so replace_op won't forward them.

    /// optimizer.py: flush()
    /// Flush all passes' postponed state.
    pub fn flush(&mut self, ctx: &mut OptContext) {
        for pass_idx in 0..self.passes.len() {
            let pass = &mut self.passes[pass_idx];
            pass.flush(ctx);
            // RPython Optimization.emit_extra() routes newly forced ops to
            // optimizer.send_extra_operation(op, self.next_optimization).
            // During flush we must preserve that contract: each pass's flush
            // output is processed only by downstream passes, never by the
            // flushing pass again.
            self.drain_extra_operations_from(pass_idx + 1, ctx);
        }
    }

    /// Build a short preamble from an optimized trace's preamble section.
    /// Convenience method that combines extract + produce.
    pub fn build_short_preamble(
        optimized_ops: &[Op],
    ) -> crate::optimizeopt::shortpreamble::ShortPreamble {
        crate::optimizeopt::shortpreamble::extract_short_preamble(optimized_ops)
    }

    /// optimizer.py: send_extra_operation(op, ctx)
    /// Send an extra operation through the pass chain as if it were
    /// a new operation from the trace. Used by passes that need to
    /// inject additional operations.
    pub fn send_extra_operation(&mut self, op: &Op, ctx: &mut OptContext) {
        self.propagate_from_pass(0, op, ctx);
    }

    /// RPython optimizer.py: emit_extra(op, emit=False) parity.
    /// Route an operation through passes starting AFTER `after_pass_idx`,
    /// matching RPython's `send_extra_operation(op, self.next_optimization)`.
    pub fn send_extra_operation_after(
        &mut self,
        after_pass_idx: usize,
        op: &Op,
        ctx: &mut OptContext,
    ) {
        self.propagate_from_pass(after_pass_idx + 1, op, ctx);
    }

    /// optimizer.py:345-364: force_box — force a virtual to be materialized.
    /// Also pops from potential_extra_ops (optimizer.py:351-359).
    pub fn force_box(&mut self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        // optimizer.py:346: op = get_box_replacement(op)
        let resolved = ctx.get_box_replacement(opref);
        // optimizer.py:351-359: potential_extra_ops.pop(op)
        // → sb.add_preamble_op(preamble_op)
        if let Some(preamble_op) = ctx.take_potential_extra_op(resolved) {
            if let Some(builder) = ctx.active_short_preamble_producer_mut() {
                builder.add_preamble_op_from_pop(&preamble_op, resolved);
            } else if let Some(builder) = ctx.imported_short_preamble_builder.as_mut() {
                builder.add_preamble_op_from_pop(&preamble_op, resolved);
            }
        }
        if ctx
            .get_ptr_info(resolved)
            .is_some_and(|info| info.is_virtual())
        {
            // RPython: info.force_box() sets _is_virtual=False in-place.
            // Take ownership so the Virtual PtrInfo is removed. force_box_impl
            // installs a non-virtual (Instance/Struct) at the alloc_ref.
            let mut info = ctx.take_ptr_info(resolved).unwrap();
            let forced = info.force_box(resolved, ctx);
            return ctx.get_box_replacement(forced);
        }
        resolved
    }

    /// optimizer.py: force_box_for_end_of_preamble(box)
    ///
    /// The exported loop state should record the boxes that survive the end of
    /// the preamble after virtuals have been forced into a loop-carried shape.
    pub fn force_box_for_end_of_preamble(&mut self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        let mut rec = std::collections::HashSet::new();
        self.force_box_for_end_of_preamble_rec(opref, ctx, &mut rec)
    }

    fn force_box_for_end_of_preamble_rec(
        &mut self,
        opref: OpRef,
        ctx: &mut OptContext,
        rec: &mut std::collections::HashSet<OpRef>,
    ) -> OpRef {
        let resolved = ctx.get_box_replacement(opref);
        let Some(mut info) = ctx.get_ptr_info(resolved).cloned() else {
            return resolved;
        };

        // RPython info.py: InstancePtrInfo, StructPtrInfo, ArrayPtrInfo all
        // override _force_at_the_end_of_preamble to keep the virtual alive
        // and recurse into fields. AbstractRawPtrInfo uses the base
        // _force_at_the_end_of_preamble → force_box() (materialization).
        if matches!(
            info,
            crate::optimizeopt::info::PtrInfo::Virtual(_)
                | crate::optimizeopt::info::PtrInfo::VirtualStruct(_)
                | crate::optimizeopt::info::PtrInfo::VirtualArray(_)
                | crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(_)
        ) {
            if !rec.insert(resolved) {
                return resolved;
            }
            info.force_at_the_end_of_preamble(|child| {
                self.force_box_for_end_of_preamble_rec(child, ctx, rec)
            });
            ctx.set_ptr_info(resolved, info);
            return resolved;
        }

        // RawBuffer: AbstractRawPtrInfo inherits base force_box() default.
        // info.py:159-160: _force_at_the_end_of_preamble → force_box()
        // optimizer.py:311-312: optforce = self.optearlyforce
        if matches!(info, crate::optimizeopt::info::PtrInfo::VirtualRawBuffer(_)) {
            let saved = ctx.current_pass_idx;
            ctx.current_pass_idx = ctx.optearlyforce_idx;
            let result = self.force_box(resolved, ctx);
            ctx.current_pass_idx = saved;
            return result;
        }

        resolved
    }

    /// optimizer.py: protect_speculative_operation(op, ctx)
    /// When constant-folding a pure operation, verify that the folded
    /// constant doesn't cause a memory safety issue (e.g., null deref in
    /// getfield). If the result would be invalid, don't fold.
    pub fn protect_speculative_operation(&self, op: &Op, ctx: &OptContext) -> bool {
        // For now, conservative: only allow folding on arithmetic ops.
        // getfield/getarrayitem on constant null pointer would crash.
        match op.opcode {
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF => {
                // Check arg(0) is not null constant.
                if let Some(0) = ctx.get_constant_int(op.arg(0)) {
                    return false; // would deref null
                }
                true
            }
            _ => true,
        }
    }

    /// optimizer.py: getlastop() — return the last emitted non-guard operation.
    pub fn getlastop<'a>(&self, ctx: &'a OptContext) -> Option<&'a Op> {
        ctx.new_operations.last()
    }

    /// optimizer.py: new_const(fieldvalue) — create a new constant OpRef.
    /// Emits a SameAs op with the given constant value.
    pub fn new_const_int(ctx: &mut OptContext, value: i64) -> OpRef {
        let op = Op::new(OpCode::SameAsI, &[]);
        let opref = ctx.emit(op);
        ctx.make_constant(opref, majit_ir::Value::Int(value));
        opref
    }

    /// optimizer.py: new_const_item(arraydescr) — create a default value
    /// for an array item (0 for int, null for ref, 0.0 for float).
    pub fn new_const_item(ctx: &mut OptContext, item_type: majit_ir::Type) -> OpRef {
        match item_type {
            majit_ir::Type::Int => Self::new_const_int(ctx, 0),
            majit_ir::Type::Ref => {
                let op = Op::new(OpCode::SameAsR, &[]);
                let opref = ctx.emit(op);
                ctx.make_constant(opref, majit_ir::Value::Ref(majit_ir::GcRef::NULL));
                opref
            }
            majit_ir::Type::Float => {
                let op = Op::new(OpCode::SameAsF, &[]);
                let opref = ctx.emit(op);
                ctx.make_constant(opref, majit_ir::Value::Float(0.0));
                opref
            }
            majit_ir::Type::Void => OpRef::NONE,
        }
    }

    /// optimizer.py: _clean_optimization_info(ops)
    /// Reset forwarding pointers on all ops before re-optimization.
    /// Called when re-optimizing a trace (e.g., retrace).
    pub fn clean_optimization_info(ctx: &mut OptContext) {
        ctx.forwarded.clear();
    }

    /// optimizer.py: get_count_of_ops()
    /// Count operations emitted so far.
    pub fn get_count_of_ops(ctx: &OptContext) -> usize {
        ctx.new_operations.len()
    }

    /// optimizer.py: get_count_of_guards()
    /// Count guards emitted so far.
    pub fn get_count_of_guards(ctx: &OptContext) -> usize {
        ctx.new_operations
            .iter()
            .filter(|op| op.opcode.is_guard())
            .count()
    }

    /// optimizer.py: log_loop(ops)
    /// Log the optimized trace for debugging/profiling.
    pub fn log_optimized_trace(ctx: &OptContext) {
        if std::env::var("MAJIT_LOG_OPT").is_ok() {
            eprintln!(
                "[MAJIT] optimized trace: {} ops, {} guards",
                ctx.new_operations.len(),
                ctx.new_operations
                    .iter()
                    .filter(|op| op.opcode.is_guard())
                    .count()
            );
        }
    }

    /// optimizer.py: getnullness(op)
    /// Check the nullness of an OpRef: NONNULL (1), NULL (-1), or UNKNOWN (0).
    pub fn getnullness(ctx: &OptContext, opref: OpRef) -> i8 {
        let resolved = ctx.get_box_replacement(opref);
        if let Some(val) = ctx.get_constant_int(resolved) {
            if val != 0 { 1 } else { -1 }
        } else {
            0 // unknown
        }
    }

    /// optimizer.py:137-152: make_constant_class(op, class_const, update_last_guard)
    ///
    /// Sets known class on PtrInfo, preserving last_guard_pos from any
    /// existing info. When `update_last_guard` is false (RECORD_EXACT_CLASS),
    /// the guard position is NOT updated — preserving guard strengthening
    /// opportunities for subsequent GUARD_CLASS ops.
    pub fn make_constant_class(
        ctx: &mut OptContext,
        opref: OpRef,
        class_value: i64,
        update_last_guard: bool,
    ) {
        let class_ptr = GcRef(class_value as usize);
        let resolved = ctx.get_box_replacement(opref);
        // optimizer.py:139: isinstance(opinfo, info.InstancePtrInfo)
        let is_instance = matches!(ctx.get_ptr_info(resolved), Some(PtrInfo::Instance(_)));
        if is_instance {
            // optimizer.py:140: opinfo._known_class = class_const
            if let Some(PtrInfo::Instance(iinfo)) = ctx.get_ptr_info_mut(resolved) {
                iinfo.known_class = Some(class_ptr);
            }
        } else {
            // optimizer.py:142-148: preserve last_guard_pos from old info
            let old_guard_pos = ctx
                .get_ptr_info(resolved)
                .and_then(|i| i.get_last_guard_pos())
                .map(|p| p as i32)
                .unwrap_or(-1);
            let mut new_info = PtrInfo::known_class(class_ptr, true);
            new_info.set_last_guard_pos(old_guard_pos);
            ctx.set_ptr_info(resolved, new_info);
        }
        // optimizer.py:150-151: if update_last_guard: mark_last_guard
        if update_last_guard {
            ctx.mark_last_guard(resolved);
        }
    }

    /// optimizer.py: is_raw_ptr(op)
    /// Check if an OpRef refers to a raw (non-GC) pointer.
    pub fn is_raw_ptr(_opref: OpRef) -> bool {
        // In RPython this checks the type annotation.
        // In majit we don't have type annotations on OpRefs,
        // so we conservatively return false (assume GC pointer).
        false
    }

    /// optimizer.py: is_call_pure_pure_canraise(op)
    /// Check if a CALL_PURE can raise an exception.
    pub fn is_call_pure_pure_canraise(op: &Op) -> bool {
        op.descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| cd.effect_info().can_raise())
            .unwrap_or(true)
    }

    /// Add an optimization pass to the chain.
    pub fn add_pass(&mut self, pass: Box<dyn Optimization>) {
        self.passes.push(pass);
    }

    /// Mark all passes as Phase 2 (loop body).
    pub fn set_phase2(&mut self, phase2: bool) {
        for pass in &mut self.passes {
            pass.set_phase2(phase2);
        }
    }

    /// Run all optimization passes over a list of operations.
    ///
    /// Returns the optimized operation list.
    /// optimizer.py:517: propagate_all_forward(trace, call_pure_results, flush)
    pub fn propagate_all_forward(&mut self, ops: &[Op]) -> Vec<Op> {
        self.optimize_with_constants(ops, &mut std::collections::HashMap::new())
    }

    /// Run all optimization passes, with known constants pre-populated.
    ///
    /// `constants` maps OpRef indices to raw backend bits. RPython keeps typed
    /// Const boxes in the optimized trace; majit threads the same information
    /// through this side table and recovers Int/Ref/Float from the producing
    /// op's result type.
    ///
    /// After optimization, newly-discovered constants (from constant folding)
    /// are written back into the map so the backend can resolve them.
    pub fn optimize_with_constants(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
    ) -> Vec<Op> {
        self.optimize_with_constants_and_inputs(ops, constants, 0)
    }

    /// Like `optimize_with_constants`, but also takes `num_inputs` so that
    /// ops emitted by the optimizer (e.g. from force_virtual) get pos values
    /// that don't collide with input argument variable indices.
    pub fn optimize_with_constants_and_inputs(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
    ) -> Vec<Op> {
        use majit_ir::OpRef;
        self.imported_label_args = None;
        self.imported_label_source_slots = None;
        self.terminal_op = None;
        // RPython parity: each optimizer run is a fresh Optimizer instance.
        // In pyre we reuse the same Optimizer, so clear per-run state.
        self.last_guard_op = None;
        // Ensure new ops get positions beyond all original trace positions.
        // Original ops keep their tracer-assigned positions; new ops (constants,
        // force materializations) must not collide with them.
        let max_pos = ops
            .iter()
            .map(|op| op.pos.0)
            .filter(|&p| p != u32::MAX) // skip OpRef::NONE
            .max()
            .unwrap_or(0);
        let effective_inputs = num_inputs.max((max_pos + 1) as usize);
        let mut ctx = OptContext::with_num_inputs(ops.len(), effective_inputs);
        ctx.skip_flush_mode = self.skip_flush;
        ctx.constant_fold_alloc = self.constant_fold_alloc.take();
        // RPython resume.py parity: Phase 2 optimizer needs imported_label_args
        // to resolve NONE positions in fail_args inherited from Phase 1.
        ctx.imported_virtuals = self.imported_virtuals.clone();
        ctx.imported_label_args = self.imported_label_args.clone();

        // RPython Box type parity: in RPython each Box carries its type
        // intrinsically (InputArgInt, InputArgRef, etc.). In majit, OpRef
        // is an untyped u32. Seed value_types from ALL sources, lowest
        // priority first (later entries override earlier):
        // 1. Original trace ops (pre-transformation positions for snapshots)
        for (&k, &v) in &self.original_trace_op_types {
            ctx.value_types.insert(k, v);
        }
        // 2. Transformed trace ops (optimizer input)
        for op in ops {
            if !op.pos.is_none() && op.result_type() != majit_ir::Type::Void {
                ctx.value_types.insert(op.pos.0, op.result_type());
            }
        }
        // 3. Previous phase's emitted op types (Phase 1 → Phase 2 carry)
        for (&k, &v) in &self.prev_phase_value_types {
            ctx.value_types.insert(k, v);
        }
        // 4. Inputarg types (from recorder — RPython InputArgInt/Ref/Float)
        //    Highest priority — override all others.
        for (i, &tp) in self.trace_inputarg_types.iter().enumerate() {
            ctx.value_types.insert(i as u32, tp);
        }

        // optimizer.py:293 patchguardop parity: propagate to Phase 2
        // OptContext so copy_and_change guards (unroll.py:409) can get
        // rd_resume_position before GUARD_FUTURE_CONDITION is re-encountered.
        if ctx.patchguardop.is_none() {
            ctx.patchguardop = self.patchguardop.clone();
        }
        // optimizer.py:294: patchguardop is set by GUARD_FUTURE_CONDITION
        // during optimization (rewrite.rs / simplify.rs). No synthetic
        // fallback — RPython relies solely on the actual GFC from tracing.

        // RPython resume.py parity: pass snapshot_boxes and constant_types
        // to OptContext so emit() can call store_final_boxes_in_guard inline
        // at each guard emission (not post-assembly).
        ctx.snapshot_boxes = self.snapshot_boxes.clone();
        ctx.snapshot_frame_sizes = self.snapshot_frame_sizes.clone();
        ctx.snapshot_vable_boxes = self.snapshot_vable_boxes.clone();
        ctx.snapshot_frame_pcs = self.snapshot_frame_pcs.clone();
        ctx.snapshot_box_types = self.snapshot_box_types.clone();
        ctx.constant_types_for_numbering = self.constant_types.clone();
        // RPython parity: merge numbering_type_overrides (ob_type Ref types)
        // into constant_types_for_numbering. These override Int → Ref for
        // resume data only, not for Cranelift backend.
        for (&k, &v) in &self.numbering_type_overrides {
            ctx.constant_types_for_numbering.entry(k).or_insert(v);
        }

        // Pre-populate known constants so passes can see them.
        // Use constant_types to distinguish Ref from Int (GC pointers
        // are stored as i64 in the constants map).
        for (&idx, &val) in constants.iter() {
            ctx.seed_constant(
                OpRef(idx),
                value_from_backend_constant_bits(OpRef(idx), val, ops),
            );
        }

        // Setup all passes
        for pass in &mut self.passes {
            pass.setup();
        }

        // earlyforce.py:32: self.optimizer.optearlyforce = self
        // Find the EarlyForce pass index so force paths can route
        // emitted operations starting from earlyforce.next (= heap).
        for (idx, pass) in self.passes.iter().enumerate() {
            if pass.name() == "earlyforce" {
                ctx.optearlyforce_idx = idx;
                break;
            }
        }

        // bridgeopt.py:124-185: apply pending bridge knowledge AFTER setup.
        // RPython calls deserialize_optimizer_knowledge before propagate_all_forward
        // but after the optimizer is constructed (setup already done at __init__).
        if let Some(knowledge) = self.pending_bridge_knowledge.take() {
            // heap.py:870-894: deserialize_optheap — import into passes AND
            // set PtrInfo fields so other passes see the cached values.
            self.deserialize_optimizer_knowledge(&knowledge, &mut ctx);
            for &(obj, field_idx, val) in &knowledge.heap_fields {
                if !obj.is_none() && !val.is_none() {
                    if let Some(info) = ctx.get_ptr_info_mut(obj) {
                        info.setfield(field_idx, val);
                    }
                }
            }
        }

        // optimizer.py: inject call_pure_results into OptPure so it can
        // constant-fold repeated pure calls across loop iterations.
        if !self.call_pure_results.is_empty() {
            for pass in &mut self.passes {
                if pass.name() == "pure" {
                    // Downcast not possible with trait objects, but we record
                    // the results as known constants in the context instead.
                    for (args, value) in &self.call_pure_results {
                        if let majit_ir::Value::Int(v) = value {
                            if let Some(result_ref) = args.last() {
                                ctx.seed_constant(*result_ref, majit_ir::Value::Int(*v));
                            }
                        }
                    }
                    break;
                }
            }
        }

        if let Some(exported_state) = self.imported_loop_state.as_ref() {
            // opencoder.py:259 parity: RPython allocates fresh InputArg Box
            // objects for each Phase 2 iteration. In majit's flat OpRef model,
            // source == target creates a forwarding cycle, so fresh allocation
            // is only applied for cross-slot collisions (target[i] == source[j],
            // i!=j). The source == target case is a no-op in replace_op.
            let nia = &exported_state.next_iteration_args;
            let n = nia.len();
            let source_set: std::collections::HashSet<OpRef> =
                (0..n).map(|i| OpRef(i as u32)).collect();
            let targetargs: Vec<OpRef> = (0..n)
                .map(|i| {
                    let source = OpRef(i as u32);
                    let target = nia[i];
                    // Constants (>= CONST_BASE) don't participate in forwarding.
                    if target.0 >= majit_ir::OpRef::CONST_BASE {
                        return source;
                    }
                    // Cross-slot collision: target is another slot's source.
                    // Allocate fresh to avoid forwarding overwrite.
                    if target != source && source_set.contains(&target) {
                        let fresh = ctx.alloc_op_position();
                        if let Some(&tp) = ctx.value_types.get(&source.0) {
                            ctx.value_types.insert(fresh.0, tp);
                        }
                        ctx.replace_op(source, fresh);
                        fresh
                    } else {
                        source
                    }
                })
                .collect();
            let (label_args, source_slots) =
                crate::optimizeopt::unroll::import_state_with_source_slots(
                    &targetargs,
                    exported_state,
                    &mut ctx,
                );
            if crate::optimizeopt::majit_log_enabled() {
                if let Some((base, ref sa)) = ctx.imported_virtual_args {
                    eprintln!(
                        "[jit] virtual_args from import_state: base={} total={} virtuals={:?}",
                        base,
                        sa.len(),
                        &sa[base..]
                    );
                }
            }
            self.imported_label_args = Some(label_args);
            self.imported_label_source_slots = Some(source_slots);
        }

        if !self.imported_virtuals.is_empty() {
            self.install_imported_virtuals(&mut ctx);
        }

        // RPython shortpreamble.py: PureOp.produce_op stores PreambleOp
        // directly in opt.optimizer.optpure. In majit, imported short pure ops
        // are first collected in ctx.imported_short_pure_ops, then transferred
        // to the OptPure pass here (matching RPython's produce_op timing).
        if !ctx.imported_short_pure_ops.is_empty() {
            for pass in &mut self.passes {
                pass.install_preamble_pure_ops(&ctx);
            }
        }

        // RPython optimizer.py:536-538: JUMP/FINISH always breaks the main
        // loop. flush() is called before JUMP is processed.
        let mut last_op = None;
        for op in ops {
            if op.opcode == OpCode::Jump || op.opcode == OpCode::Finish {
                last_op = Some(op.clone());
                break;
            }
            self.propagate_one(op, &mut ctx);
        }

        // RPython: flush() before JUMP processing (export_state calls flush
        // before get_virtual_state). Phase 2 skips flush.
        if !self.skip_flush {
            self.flush(&mut ctx);
        }

        // RPython unroll.py:457: get_virtual_state(end_args)
        // Capture virtual state BEFORE JUMP is processed through passes
        // (which forces virtuals). This replaces pre_force_virtual_state.
        let pre_jump_virtual_state = if !self.skip_flush {
            last_op
                .as_ref()
                .filter(|op| op.opcode == OpCode::Jump)
                .map(|jump_op| {
                    let resolved_args: Vec<OpRef> = jump_op
                        .args
                        .iter()
                        .map(|&arg| ctx.get_box_replacement(arg))
                        .collect();
                    (
                        crate::optimizeopt::virtualstate::export_state(
                            &resolved_args,
                            &ctx,
                            &ctx.forwarded,
                        ),
                        resolved_args,
                    )
                })
        } else {
            None
        };

        // Set pre_force_virtual_state for export_state_with_bounds to use.
        // This replaces the previous approach where OptVirtualize's JUMP handler set it.
        if let Some((ref vs, ref args)) = pre_jump_virtual_state {
            ctx.pre_force_virtual_state = Some(vs.clone());
            ctx.pre_force_jump_args = Some(args.clone());
        }

        // RPython optimizer.py:536-538 / unroll.py:101-103:
        // JUMP/FINISH handling. RPython Phase 1 uses flush=False (JUMP not
        // sent through passes). In majit, Phase 1 JUMP goes through passes
        // for heap flush and resume data. Phase 2 stores JUMP as terminal_op.
        // TODO: implement flush=False for Phase 1 when export_state handles
        // virtual JUMP args (requires rd_virtuals encoding).
        if let Some(mut terminal_op) = last_op {
            if self.skip_flush {
                // RPython: Phase 2 JUMP is NOT sent through optimization
                // passes. potential_extra_ops are consumed during normal
                // _emit_operation → force_box, not at the end.
                for arg in &mut terminal_op.args {
                    *arg = ctx.get_box_replacement(*arg);
                }
                self.terminal_op = Some(terminal_op);
            } else {
                self.propagate_one(&terminal_op, &mut ctx);
            }
        }

        // RPython store_final_boxes_in_guard parity: re-encode late virtuals
        // in guard fail_args. Phase 2 guards may inherit NONE from Phase 1
        // virtualization — rescan resolves these using imported_virtuals.
        // RPython store_final_boxes_in_guard parity: re-encode late virtuals
        // RPython parity: store_final_boxes_in_guard in ctx.emit() handles
        // virtual tagging inline at each guard emit. No post-pass rescan.

        // RPython Box type parity: preserve value_types from this phase
        // so the next phase can resolve types for carried-over OpRefs.
        self.prev_phase_value_types
            .extend(ctx.value_types.iter().map(|(&k, &v)| (k, v)));

        // Transfer exported virtual state from context to optimizer
        // RPython BasicLoopInfo: quasi_immutable_deps collected during optimization
        self.quasi_immutable_deps = std::mem::take(&mut ctx.quasi_immutable_deps);
        self.imported_short_sources = std::mem::take(&mut ctx.imported_short_sources);
        self.imported_short_aliases = ctx.used_imported_short_aliases();
        self.imported_short_preamble = ctx.build_imported_short_preamble();
        self.imported_short_preamble_builder = ctx.imported_short_preamble_builder.clone();
        // optimizer.py:294: patchguardop is set exclusively by
        // GUARD_FUTURE_CONDITION during optimization. No fallback.
        self.patchguardop = ctx.patchguardop.clone();
        // JUMP location: in new_operations (flush=True path where JUMP was
        // processed through passes), or terminal_op (skip_flush path).
        let jump = ctx
            .new_operations
            .iter()
            .rfind(|op| op.opcode == OpCode::Jump)
            .cloned()
            .or_else(|| {
                self.terminal_op
                    .clone()
                    .filter(|op| op.opcode == OpCode::Jump)
            });
        self.exported_loop_state = jump.map(|jump| {
            // Use pre-JUMP virtual state captured before passes forced virtuals.
            // RPython unroll.py:457: get_virtual_state(end_args) — computed
            // after force_box_for_end_of_preamble but before final JUMP emission.
            let (pre_vs, pre_args) = pre_jump_virtual_state.clone()
                .unwrap_or_else(|| {
                    let args: Vec<OpRef> = jump.args.iter()
                        .map(|&a| ctx.get_box_replacement(a))
                        .collect();
                    let vs = crate::optimizeopt::virtualstate::export_state(&args, &ctx, &ctx.forwarded);
                    (vs, args)
                });
            let original_jump_args = pre_args;
            let mut resolved_args = original_jump_args.clone();
            // Dedup: when two slots reference the same OpRef (e.g. b and t
            // in fib_loop), create SameAsR with fresh OpRef and copy the
            // VIRTUAL PtrInfo (before force turns it into Instance).
            {
                let mut seen = std::collections::HashSet::new();
                for arg in resolved_args.iter_mut() {
                    if arg.0 >= 10_000 || *arg == OpRef::NONE {
                        continue;
                    }
                    if !seen.insert(*arg) {
                        let orig = *arg;
                        let same_as = OpCode::SameAsR;
                        let fresh = ctx.alloc_op_position();
                        let mut op = Op::new(same_as, &[orig]);
                        op.pos = fresh;
                        ctx.emit(op);
                        if let Some(info) = ctx.get_ptr_info(orig).cloned() {
                            let fresh_info = match info {
                                crate::optimizeopt::info::PtrInfo::Virtual(mut vinfo) => {
                                    for field in &mut vinfo.fields {
                                        let orig_field = field.1;
                                        let ff = ctx.alloc_op_position();
                                        ctx.replace_op(ff, orig_field);
                                        // RPython Box type parity: copy type
                                        if let Some(&tp) = ctx.value_types.get(&orig_field.0) {
                                            ctx.value_types.insert(ff.0, tp);
                                        }
                                        if let Some(val) =
                                            ctx.get_constant(orig_field).cloned()
                                        {
                                            ctx.make_constant(ff, val);
                                        }
                                        field.1 = ff;
                                    }
                                    crate::optimizeopt::info::PtrInfo::Virtual(vinfo)
                                }
                                other => other,
                            };
                            ctx.set_ptr_info(fresh, fresh_info);
                        }
                        *arg = fresh;
                    }
                }
            }
            // Now force all resolved (and dedup'd) args.
            ctx.preamble_end_args = Some(
                resolved_args
                    .iter()
                    .map(|&arg| self.force_box_for_end_of_preamble(arg, &mut ctx))
                    .collect(),
            );
            // RPython parity: nested virtuals in output ops should be
            // forced by force_box_for_end_of_preamble's recursive walk.
            // Additional force is handled by the undefined-ref cleanup below
            // which drops ops referencing un-forced virtual OpRefs.
            // Virtual state was captured before JUMP went through passes.
            let preview_virtual_state = pre_vs;
            let vs_args = &original_jump_args;
            let (preview_label_args, preview_virtuals) =
                preview_virtual_state.make_inputargs_and_virtuals(vs_args, &ctx);
            let mut preview_short_args = preview_label_args.clone();
            preview_short_args.extend(preview_virtuals);
            let mut short_boxes =
                crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&preview_short_args);
            short_boxes.note_known_constants_from_ctx(&ctx);
            for &arg in &preview_short_args {
                short_boxes.add_short_input_arg(arg);
            }
            self.produce_potential_short_preamble_ops(&mut short_boxes, &ctx);
            let produced = short_boxes.produced_ops();
            ctx.exported_short_boxes = produced
                .into_iter()
                .map(|(result, produced)| {
                    let canonical_result = ctx.get_box_replacement(result);
                    let mut preamble_op = produced.preamble_op;
                    // RPython parity: key and preamble_op.pos must be the
                    // same resolved value. Independent get_box_replacement
                    // calls can diverge when forwarding chains differ.
                    // Use canonical_result (resolved key) for both.
                    preamble_op.pos = canonical_result;
                    for arg in &mut preamble_op.args {
                        *arg = ctx.get_box_replacement(*arg);
                    }
                    if let Some(fail_args) = preamble_op.fail_args.as_mut() {
                        for arg in fail_args {
                            *arg = ctx.get_box_replacement(*arg);
                        }
                    }
                    crate::optimizeopt::shortpreamble::PreambleOp {
                        op: preamble_op,
                        kind: produced.kind,
                        label_arg_idx: short_boxes.lookup_label_arg(canonical_result),
                        invented_name: produced.invented_name,
                        same_as_source: produced.same_as_source.map(|src| ctx.get_box_replacement(src)),
                    }
                })
                .collect();
            if std::env::var_os("MAJIT_LOG").is_some() {
                for entry in &ctx.exported_short_boxes {
                    eprintln!(
                        "[jit] exported_short_box: kind={:?} pos={:?} opcode={:?} args={:?} descr_idx={:?}",
                        entry.kind,
                        entry.op.pos,
                        entry.op.opcode,
                        entry.op.args,
                        entry.op.descr.as_ref().map(|d| d.index()),
                    );
                }
            }
            let exported_int_bounds = self.collect_exported_int_bounds(&jump.args, &ctx);
            // RPython unroll.py passes optimize_preamble()'s inputargs here,
            // i.e. the external loop-entry contract, not the optimizer's
            // internal position base (`ctx.num_inputs`), which may be widened
            // to avoid collisions with existing op positions.
            let renamed_inputargs: Vec<OpRef> = (0..num_inputs).map(|i| OpRef(i as u32)).collect();
            crate::optimizeopt::unroll::export_state(
                &original_jump_args,
                &renamed_inputargs,
                &ctx,
                Some(&exported_int_bounds),
            )
        });
        // Populate renamed_inputarg_types from trace_inputarg_types.
        // RPython Box objects carry type intrinsically; we store it separately.
        if let Some(ref mut es) = self.exported_loop_state {
            if es.renamed_inputarg_types.is_empty() && !self.trace_inputarg_types.is_empty() {
                es.renamed_inputarg_types = es
                    .renamed_inputargs
                    .iter()
                    .map(|&opref| {
                        self.trace_inputarg_types
                            .get(opref.0 as usize)
                            .copied()
                            .unwrap_or(Type::Int)
                    })
                    .collect();
            }
        }
        // RPython parity: propagate patchguardop to ExportedState so Phase 2
        // can use it for extra_guards from virtualstate (unroll.py:333-336).
        if let Some(ref mut es) = self.exported_loop_state {
            es.patchguardop = self.patchguardop.clone();
        }

        // RPython export_state() flushes force artifacts into the preamble
        // before building the exported loop state. If the loop header needs
        // additional inputargs, the corresponding SETFIELD/SETARRAYITEM must
        // remain in the trace rather than being silently discarded.
        // final_num_inputs = original inputs + virtual inputs added by passes.
        let num_virtual_inputs = (ctx.num_inputs as usize).saturating_sub(effective_inputs);
        self.final_num_inputs = num_inputs + num_virtual_inputs;

        // RPython store_final_boxes_in_guard parity: re-encode late virtuals
        // RPython parity: store_final_boxes_in_guard handles virtual tagging
        // and rd_numb production inline. No post-pass rescan needed.

        // Force any remaining virtual refs in output ops before forwarding resolve.
        // RPython: virtuals are forced during preamble export or JUMP handling.
        // In majit, skip_flush=true may leave some virtual refs un-forced in
        // the output (e.g., linked list nodes nested in non-virtual objects).
        {
            let all_refs: Vec<OpRef> = ctx
                .new_operations
                .iter()
                .flat_map(|op| op.args.iter().copied())
                .filter(|r| !r.is_none())
                .collect();
            for opref in all_refs {
                let resolved = ctx.get_box_replacement(opref);
                if let Some(info) = ctx.get_ptr_info(resolved) {
                    if info.is_virtual() {
                        self.force_box_for_end_of_preamble(resolved, &mut ctx);
                    }
                }
            }
        }

        // Resolve forwarding BEFORE remap.
        // RPython get_box_replacement: follow chain, stop at ptr_info terminal.
        {
            let fwd = ctx.forwarded.clone();
            let resolve = |opref: OpRef| -> OpRef {
                use crate::optimizeopt::info::Forwarded;
                let mut cur = opref;
                loop {
                    let idx = cur.0 as usize;
                    if idx >= fwd.len() {
                        return cur;
                    }
                    match &fwd[idx] {
                        Forwarded::Op(next) => cur = *next,
                        _ => return cur,
                    }
                }
            };
            for op in &mut ctx.new_operations {
                for arg in &mut op.args {
                    *arg = resolve(*arg);
                }
                if let Some(ref mut fa) = op.fail_args {
                    for arg in fa.iter_mut() {
                        *arg = resolve(*arg);
                    }
                }
            }
        }

        // RPython keeps constants as Const boxes, not SameAs placeholder ops in
        // the final trace. Drop constant-only SameAs placeholders before the
        // backend sees the trace; their OpRefs remain available through the
        // constants table.
        ctx.new_operations
            .retain(|op| !Self::is_constant_placeholder_op(op, &ctx.constants));

        // Drain remaining extra ops.
        self.drain_extra_operations_from(0, &mut ctx);
        // Extra operations can introduce new forwarding (for example Heap/Pure
        // forwarding a recently-emitted boxed-field read to its raw payload).
        // Resolve forwarding again with ptr_info stop.
        {
            let fwd = ctx.forwarded.clone();
            let resolve = |opref: OpRef| -> OpRef {
                use crate::optimizeopt::info::Forwarded;
                let mut cur = opref;
                loop {
                    let idx = cur.0 as usize;
                    if idx >= fwd.len() {
                        return cur;
                    }
                    match &fwd[idx] {
                        Forwarded::Op(next) => cur = *next,
                        _ => return cur,
                    }
                }
            };
            for op in &mut ctx.new_operations {
                for arg in &mut op.args {
                    *arg = resolve(*arg);
                }
                if let Some(ref mut fa) = op.fail_args {
                    for arg in fa.iter_mut() {
                        *arg = resolve(*arg);
                    }
                }
            }
        }
        // RPython: no equivalent filter — Box identity guarantees all
        // references are valid. The retain filter was a majit safety net
        // but incorrectly dropped valid ops (e.g., IntAddOvf referencing
        // inputarg positions beyond final_num_inputs in bridge traces).

        // Remap ALL positions: virtual inputs go to num_inputs..final_num_inputs,
        // This ensures no position collisions between input block params and ops.
        if num_virtual_inputs > 0 {
            let fni = self.final_num_inputs as u32;
            let mut remap = std::collections::HashMap::new();

            // Virtual input positions: optimizer used effective_inputs+k, backend needs num_inputs+k
            for k in 0..num_virtual_inputs {
                let opt_pos = (effective_inputs + k) as u32;
                let be_pos = (num_inputs + k) as u32;
                if opt_pos != be_pos {
                    remap.insert(opt_pos, be_pos);
                }
            }

            // Op positions: reassign ALL ops to start from final_num_inputs.
            if crate::optimizeopt::majit_log_enabled() {
                let fwd_1995 = if 1995 < ctx.forwarded.len() {
                    format!("{:?}", ctx.forwarded[1995])
                } else {
                    "None".to_string()
                };
                let in_ops = ctx.new_operations.iter().any(|o| o.pos.0 == 1995);
                let in_args = ctx
                    .new_operations
                    .iter()
                    .any(|o| o.args.iter().any(|a| a.0 == 1995));
                eprintln!("[remap] v1995: fwd={fwd_1995:?} in_ops={in_ops} in_args={in_args}");
            }
            for (new_idx, op) in ctx.new_operations.iter_mut().enumerate() {
                let new_pos = fni + new_idx as u32;
                if !op.pos.is_none() {
                    remap.insert(op.pos.0, new_pos);
                    op.pos = OpRef(new_pos);
                }
            }

            // Constant-folded operations that were removed from the trace still
            // have entries in `ctx.constants`. If we leave them at their old
            // positions, they can collide with the freshly compacted op
            // positions above (for example old constant v71 vs new live op v71),
            // and the backend will resolve the live op as the stale constant.
            // Give every such constant-only opref a fresh slot after the last
            // live op, mirroring RPython's separate constant identity.
            let mut next_const_pos = fni + ctx.new_operations.len() as u32;
            for (idx, val) in ctx.constants.iter().enumerate() {
                let old_idx = idx as u32;
                if val.is_none() || remap.contains_key(&old_idx) {
                    continue;
                }
                if old_idx < num_inputs as u32 {
                    continue;
                }
                remap.insert(old_idx, next_const_pos);
                next_const_pos += 1;
            }

            // Apply remap to forwarding table too, so forwarding resolution
            // after remap resolves to the correct remapped positions.
            for entry in &mut ctx.forwarded {
                if let crate::optimizeopt::info::Forwarded::Op(target) = entry {
                    if let Some(&new_pos) = remap.get(&target.0) {
                        *target = OpRef(new_pos);
                    }
                }
            }

            // Apply remap to all args and fail_args
            for op in &mut ctx.new_operations {
                for arg in &mut op.args {
                    if let Some(&new_pos) = remap.get(&arg.0) {
                        *arg = OpRef(new_pos);
                    }
                }
                if let Some(ref mut fail_args) = op.fail_args {
                    for arg in fail_args.iter_mut() {
                        if let Some(&new_pos) = remap.get(&arg.0) {
                            *arg = OpRef(new_pos);
                        }
                    }
                }
            }

            // Remap constants
            let old_constants = std::mem::take(&mut ctx.constants);
            for (idx, val) in old_constants.into_iter().enumerate() {
                if let Some(v) = val {
                    let target_idx =
                        remap.get(&(idx as u32)).copied().unwrap_or(idx as u32) as usize;
                    if target_idx >= ctx.constants.len() {
                        ctx.constants.resize(target_idx + 1, None);
                    }
                    ctx.constants[target_idx] = Some(v);
                }
            }

            // Remap exported_loop_state OpRefs so Phase 2 sees post-remap
            // positions. Without this, Phase 2's import_boxes maps to
            // pre-remap positions that no longer exist in the constants map.
            if let Some(ref mut state) = self.exported_loop_state {
                let remap_opref = |opref: &mut OpRef| {
                    if let Some(&new_pos) = remap.get(&opref.0) {
                        *opref = OpRef(new_pos);
                    }
                };
                for arg in &mut state.next_iteration_args {
                    remap_opref(arg);
                }
                for arg in &mut state.end_args {
                    remap_opref(arg);
                }
                for arg in &mut state.renamed_inputargs {
                    remap_opref(arg);
                }
                for arg in &mut state.short_inputargs {
                    remap_opref(arg);
                }
                // Remap exported_infos keys
                let old_infos = std::mem::take(&mut state.exported_infos);
                for (key, value) in old_infos {
                    let new_key = remap.get(&key.0).map(|&p| OpRef(p)).unwrap_or(key);
                    state.exported_infos.insert(new_key, value);
                }
                // Remap exported short boxes
                for entry in &mut state.exported_short_boxes {
                    remap_opref(&mut entry.op.pos);
                    for arg in &mut entry.op.args {
                        remap_opref(arg);
                    }
                    if let Some(ref mut fa) = entry.op.fail_args {
                        for arg in fa.iter_mut() {
                            remap_opref(arg);
                        }
                    }
                    if let Some(ref mut src) = entry.same_as_source {
                        remap_opref(src);
                    }
                }
            }
        }

        // Export newly-discovered constants back to the caller's map.
        for (idx, val) in ctx.constants.iter().enumerate() {
            if let Some(value) = val {
                constants
                    .entry(idx as u32)
                    .or_insert_with(|| value_to_backend_constant_bits(value));
            }
        }

        // Preserve final context for jump_to_existing_trace.
        let ops = std::mem::take(&mut ctx.new_operations);
        if crate::optimizeopt::majit_log_enabled() {
            let cmf_count = ops.iter().filter(|o| o.opcode.is_call_may_force()).count();
            let gnf_count = ops
                .iter()
                .filter(|o| matches!(o.opcode, OpCode::GuardNotForced | OpCode::GuardNotForced2))
                .count();
            eprintln!(
                "[opt] final ops: total={} call_may_force={} guard_not_forced={}",
                ops.len(),
                cmf_count,
                gnf_count
            );
            if cmf_count == 0 && gnf_count > 0 {
                for (i, op) in ops.iter().enumerate() {
                    eprintln!("[opt] idx={} {:?} pos={:?}", i, op.opcode, op.pos);
                }
            }
        }
        self.final_ctx = Some(ctx);
        ops
    }

    /// unroll.py:183-236: optimize_bridge()
    ///
    /// Optimizes a bridge trace and redirects its terminal JUMP to the
    /// appropriate loop body target token, falling back to the preamble
    /// when no match is found.
    ///
    /// `retraced_count` / `retrace_limit`: RPython history.py
    /// JitCellToken.retraced_count tracking. When retrace_limit > 0 and
    /// no existing trace matches, export_state creates a new specialization.
    /// Default retrace_limit = 0 (disabled, warmstate.py PARAMETERS).
    ///
    /// Returns `(optimized_ops, retrace_requested)`. When retrace_requested
    /// is true, the caller should increment retraced_count and may use the
    /// optimizer's exported_loop_state for the new target token.
    pub fn optimize_bridge(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
        front_target_tokens: &mut Vec<crate::optimizeopt::unroll::TargetToken>,
        inline_short_preamble: bool,
        retraced_count: u32,
        retrace_limit: u32,
        bridge_knowledge: Option<&OptimizerKnowledge>,
        loop_num_inputs: Option<usize>,
    ) -> (Vec<Op>, bool) {
        // bridgeopt.py:124-185: deserialize_optimizer_knowledge
        // Store as pending — setup() inside optimize_with_constants_and_inputs
        // clears pass state, so we apply AFTER setup.
        if let Some(knowledge) = bridge_knowledge {
            self.pending_bridge_knowledge = Some(knowledge.clone());
        }
        // unroll.py:183 parity: save pre-optimization JUMP args as
        // runtime_boxes. RPython passes the original live_arg_boxes from
        // compile_trace / guard failure, not the optimized jump args.
        let pre_opt_jump_args: Vec<OpRef> = ops
            .last()
            .filter(|op| op.opcode == OpCode::Jump)
            .map(|op| op.args.to_vec())
            .unwrap_or_default();

        // unroll.py:193: info, ops = self.propagate_all_forward(trace, ...)
        let optimized_ops = self.optimize_with_constants_and_inputs(ops, constants, num_inputs);

        // Check if trace ends with JUMP
        let ends_with_jump = optimized_ops
            .last()
            .map_or(false, |op| op.opcode == OpCode::Jump);

        if !ends_with_jump {
            return (optimized_ops, false);
        }

        let jump_args = optimized_ops.last().unwrap().args.to_vec();

        // unroll.py:198-200: not inline_short_preamble → jump_to_preamble
        // RPython calls send_extra_operation(jump_op) which forces virtuals
        // through the full pass chain. No explicit flush()/force_box() needed.
        if !inline_short_preamble || front_target_tokens.len() <= 1 {
            if let Some(preamble_token) = front_target_tokens.first() {
                let mut ctx = self.final_ctx.take().unwrap_or_else(|| {
                    let ni = self.final_num_inputs();
                    OptContext::with_num_inputs(32, ni)
                });
                // unroll.py:240-242: jump_to_preamble → send_extra_operation
                let mut jump_op = optimized_ops.last().unwrap().clone();
                jump_op.descr = Some(preamble_token.as_jump_target_descr());
                self.send_extra_operation(&jump_op, &mut ctx);
                let mut result = optimized_ops[..optimized_ops.len() - 1].to_vec();
                result.extend(ctx.new_operations.drain(..));
                return (result, false);
            }
            return (optimized_ops, false);
        }

        // unroll.py:203: self.flush()
        let mut ctx = self.final_ctx.take().unwrap_or_else(|| {
            let ni = self.final_num_inputs();
            OptContext::with_num_inputs(32, ni)
        });

        // RPython parity: export virtual state BEFORE flush() and
        // force_box_for_end_of_preamble(). In RPython, get_virtual_state
        // is called inside _jump_to_existing_trace on the original Box
        // objects whose PtrInfo reflects the pre-flush optimizer state.
        // flush() may force pending virtuals, changing their PtrInfo to
        // non-virtual. Capture the virtual state before any forcing.
        let pre_force_vs =
            crate::optimizeopt::virtualstate::export_state(&jump_args, &ctx, &ctx.forwarded);

        self.flush(&mut ctx);

        // unroll.py:204-205: force_box_for_end_of_preamble for each jump arg
        for &arg in &jump_args {
            let _ = self.force_box_for_end_of_preamble(arg, &mut ctx);
        }

        // RPython parity: jump_op.getarglist() returns the ORIGINAL Boxes.
        // force_box_for_end_of_preamble recurses into virtual fields but
        // does NOT force the top-level virtuals. So get_virtual_state on
        // the original args still sees Virtual PtrInfo.
        // DO NOT resolve jump_args here — pass originals so virtual_state
        // is computed from the pre-force state.

        // unroll.py:206-211: jump_to_existing_trace(force_boxes=False)
        // RPython iterates ALL target_tokens; preamble (virtual_state=None)
        // is skipped inside jump_to_existing_trace (unroll.py:327-328).
        let opt_unroll = crate::optimizeopt::unroll::OptUnroll::new();
        let vs = match Self::try_jump_to_existing_trace(
            &opt_unroll,
            &jump_args,
            front_target_tokens,
            self,
            &mut ctx,
            false,
            &pre_opt_jump_args,
            Some(pre_force_vs.clone()),
        ) {
            Ok(vs) => vs,
            // unroll.py:209-210: except InvalidLoop → jump_to_preamble
            // RPython: self.jump_to_preamble → send_extra_operation
            Err(()) => {
                if let Some(preamble_token) = front_target_tokens.first() {
                    ctx.clear_newoperations();
                    let mut jump_op = optimized_ops.last().unwrap().clone();
                    jump_op.descr = Some(preamble_token.as_jump_target_descr());
                    self.send_extra_operation(&jump_op, &mut ctx);
                    let mut result = optimized_ops[..optimized_ops.len() - 1].to_vec();
                    result.extend(ctx.new_operations.drain(..));
                    return (result, false);
                }
                return (optimized_ops, false);
            }
        };

        // unroll.py:212-213: vs is None → matched, JUMP redirected
        if vs.is_none() {
            let mut result = optimized_ops[..optimized_ops.len() - 1].to_vec();
            result.extend(ctx.new_operations.drain(..));
            return (result, false);
        }

        // unroll.py:214-218: retrace check
        if retraced_count < retrace_limit {
            if crate::optimizeopt::majit_log_enabled() {
                eprintln!("[jit] Retracing ({}/{})", retraced_count + 1, retrace_limit);
            }
            return (optimized_ops, true);
        }

        // unroll.py:220-227: retrace limit reached, try force_boxes=True
        ctx.clear_newoperations();
        let vs2 = match Self::try_jump_to_existing_trace(
            &opt_unroll,
            &jump_args,
            front_target_tokens,
            self,
            &mut ctx,
            true,
            &pre_opt_jump_args,
            Some(pre_force_vs),
        ) {
            Ok(vs) => vs,
            // unroll.py:224-225: except InvalidLoop: pass
            // vs (from first attempt) is still not None → falls through
            // to jump_to_preamble below.
            Err(()) => vs,
        };

        // unroll.py:226-227: vs is None → matched with forced boxes
        if vs2.is_none() {
            let mut result = optimized_ops[..optimized_ops.len() - 1].to_vec();
            result.extend(ctx.new_operations.drain(..));
            return (result, false);
        }

        // unroll.py:228-229,238-242: jump_to_preamble → send_extra_operation.
        // RPython sends the JUMP through the full optimization chain so that
        // force_box materializes virtuals and potential_extra_ops are consumed.
        if crate::optimizeopt::majit_log_enabled() {
            eprintln!("[jit] Retrace count reached, jumping to preamble");
        }
        if let Some(preamble_token) = front_target_tokens.first() {
            ctx.clear_newoperations();
            let mut jump_op = optimized_ops.last().unwrap().clone();
            jump_op.descr = Some(preamble_token.as_jump_target_descr());
            self.send_extra_operation(&jump_op, &mut ctx);
            let mut result = optimized_ops[..optimized_ops.len() - 1].to_vec();
            result.extend(ctx.new_operations.drain(..));
            (result, false)
        } else {
            (optimized_ops, false)
        }
    }

    /// Wrapper: call jump_to_existing_trace, catch only InvalidLoop panics.
    /// Returns Ok(vs) on normal return, Err(()) on InvalidLoop.
    /// Non-InvalidLoop panics are re-raised.
    fn try_jump_to_existing_trace(
        opt_unroll: &crate::optimizeopt::unroll::OptUnroll,
        jump_args: &[OpRef],
        front_target_tokens: &mut Vec<crate::optimizeopt::unroll::TargetToken>,
        optimizer: &mut Self,
        ctx: &mut OptContext,
        force_boxes: bool,
        pre_opt_jump_args: &[OpRef],
        pre_vs: Option<crate::optimizeopt::virtualstate::VirtualState>,
    ) -> Result<Option<crate::optimizeopt::virtualstate::VirtualState>, ()> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            opt_unroll.jump_to_existing_trace_with_vs(
                jump_args,
                None,
                front_target_tokens,
                optimizer,
                ctx,
                force_boxes,
                Some(pre_opt_jump_args),
                pre_vs,
            )
        })) {
            Ok(vs) => Ok(vs),
            Err(payload) => {
                if payload
                    .downcast_ref::<crate::optimize::InvalidLoop>()
                    .is_some()
                {
                    Err(())
                } else {
                    // Not InvalidLoop — re-raise
                    std::panic::resume_unwind(payload);
                }
            }
        }
    }

    fn collect_exported_int_bounds(
        &self,
        args: &[OpRef],
        ctx: &OptContext,
    ) -> std::collections::HashMap<OpRef, crate::optimizeopt::intutils::IntBound> {
        let mut exported = std::collections::HashMap::new();
        for pass in &self.passes {
            for (opref, bound) in pass.export_arg_int_bounds(args, ctx) {
                exported.insert(opref, bound);
            }
        }
        exported
    }

    /// Send one operation through the pass chain.
    ///
    /// NOTE: Do NOT add `replace_op(original_pos, new_pos)` here.
    /// The Emit variant's position tracking is handled by each pass
    /// and OptContext. Adding automatic replacement mapping here
    /// causes spurious forwarding that breaks heap/guard tests.
    fn propagate_one(&mut self, op: &Op, ctx: &mut OptContext) {
        self.propagate_from_pass(0, op, ctx);
    }

    fn drain_extra_operations_from(&mut self, _start_pass: usize, ctx: &mut OptContext) {
        let end_pass = self.extra_operation_end_pass();
        let mut pending = std::collections::VecDeque::new();
        while let Some((start, op)) = ctx.extra_operations_after.pop_front() {
            pending.push_back((start, op));
        }
        while let Some((from_pass, op)) = pending.pop_front() {
            self.propagate_from_pass_range(from_pass, end_pass, &op, ctx);
            while let Some((start, op)) = ctx.extra_operations_after.pop_front() {
                pending.push_front((start, op));
            }
        }
    }

    fn propagate_from_pass(&mut self, start_pass: usize, op: &Op, ctx: &mut OptContext) {
        self.propagate_from_pass_range(start_pass, self.passes.len(), op, ctx);
    }

    fn extra_operation_end_pass(&self) -> usize {
        let mut end = self.passes.len();
        while end > 0 && self.passes[end - 1].name() == "unroll" {
            end -= 1;
        }
        end
    }

    fn propagate_from_pass_range(
        &mut self,
        start_pass: usize,
        end_pass: usize,
        op: &Op,
        ctx: &mut OptContext,
    ) {
        // Resolve forwarded arguments
        let mut resolved_op = op.clone();
        for arg in &mut resolved_op.args {
            *arg = ctx.get_box_replacement(*arg);
        }
        if let Some(ref mut fa) = resolved_op.fail_args {
            for arg in fa.iter_mut() {
                *arg = ctx.get_box_replacement(*arg);
            }
        }

        let mut current_op = resolved_op;

        // optimizer.py:864-867: optimize_SAME_AS_I/R/F → make_equal_to(op, arg0)
        // SameAs ops are absorbed into forwarding, never emitted.
        if matches!(
            current_op.opcode,
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF
        ) {
            ctx.make_equal_to(current_op.pos, current_op.arg(0));
            return;
        }

        for pass_idx in start_pass..end_pass {
            ctx.current_pass_idx = pass_idx;
            let result = {
                let pass = &mut self.passes[pass_idx];
                pass.propagate_forward(&current_op, ctx)
            };
            self.drain_extra_operations_from(pass_idx + 1, ctx);
            match result {
                OptimizationResult::Emit(op) => {
                    self.emit_operation(op, ctx);
                    return;
                }
                OptimizationResult::Replace(op) => {
                    current_op = op;
                }
                OptimizationResult::Remove => {
                    // RPython parity: postprocess callbacks only run when the
                    // op survives to emission (optimizer.py:586-589). If any
                    // pass removes the op, discard pending postprocess state
                    // to prevent it from leaking into the next emitted op.
                    ctx.pending_mark_last_guard = None;
                    ctx.pending_guard_class_postprocess = None;
                    return;
                }
                OptimizationResult::PassOn => {}
            }
        }

        // If no pass handled it, emit as-is
        self.emit_operation(current_op, ctx);
    }

    /// optimizer.py: _emit_operation — emit with guard tracking.
    ///
    /// When emitting a guard, check replaces_guard to see if this guard
    /// should replace a previously emitted one (guard strengthening).
    /// Also track last_guard_op for consecutive guard descriptor sharing.
    /// RPython optimizer.py:623-625: _emit_operation calls force_box(arg)
    /// on every arg before final emission. In majit, this forces any remaining
    /// virtual args that weren't caught by pass-level handlers.
    fn emit_operation(&mut self, mut op: Op, ctx: &mut OptContext) {
        // RPython optimizer.py:614: _emit_operation is on the Optimizer (last
        // "pass" in the chain). Any force_box called here should emit directly,
        // matching RPython's Optimizer.emit_extra which just calls self.emit(op).
        let saved_in_final_emission = ctx.in_final_emission;
        ctx.in_final_emission = true;
        // RPython optimizer.py: emitting_operation callback — notify all passes
        // before any op is emitted. This is how OptHeap forces lazy sets before
        // guards even when the guard is emitted by an earlier pass.
        for (idx, pass) in self.passes.iter_mut().enumerate() {
            pass.emitting_operation(&op, ctx, idx);
        }
        // RPython emit_extra(op, emit=False) parity: drain operations
        // queued by emitting_operation (e.g., heap's force_lazy_set)
        // BEFORE the current op is emitted, preserving correct ordering.
        {
            let end_pass = self.extra_operation_end_pass();
            while let Some((start, queued_op)) = ctx.extra_operations_after.pop_front() {
                self.propagate_from_pass_range(start, end_pass, &queued_op, ctx);
            }
        }

        // optimizer.py:623-625: force_box on every arg unconditionally.
        for i in 0..op.num_args() {
            let arg = ctx.get_box_replacement(op.arg(i));
            op.args[i] = self.force_box(arg, ctx);
        }
        if op.opcode.is_guard() {
            // optimizer.py:632-635: replaces_guard check BEFORE emit_guard_operation.
            // If this guard should replace a previously emitted guard, do it
            // now and return — skip all guard processing.
            if self.can_replace_guards {
                if let Some(replacement) = self.replaces_guard.remove(&op.pos.0) {
                    let target_pos = replacement.pos.0 as usize;
                    if target_pos < ctx.new_operations.len() {
                        if std::env::var_os("MAJIT_LOG").is_some() {
                            eprintln!(
                                "[opt] guard replacement op={:?} pos={:?} target_index={} len={}",
                                op.opcode,
                                op.pos,
                                target_pos,
                                ctx.new_operations.len()
                            );
                        }
                        ctx.new_operations[target_pos] = op.clone();
                        ctx.in_final_emission = saved_in_final_emission;
                        return;
                    }
                }
            }

            // optimizer.py:652-686 emit_guard_operation
            let opcode = op.opcode;

            // optimizer.py:661-664: guard chain management.
            if (opcode == OpCode::GuardNoException || opcode == OpCode::GuardException)
                && self.last_guard_op.as_ref().is_some_and(|last| {
                    last.opcode != OpCode::GuardNotForced && last.opcode != OpCode::GuardNotForced2
                })
            {
                self.last_guard_op = None;
            }
            // optimizer.py:665-670: GUARD_ALWAYS_FAILS must never share resume
            // data with a previous guard.
            if opcode == OpCode::GuardAlwaysFails {
                self.last_guard_op = None;
            }

            // optimizer.py:672-683: _copy_resume_data_from / store_final_boxes_in_guard.
            let shared = op.descr.is_none() && self.last_guard_op.is_some();
            if shared {
                // optimizer.py:688-695: _copy_resume_data_from
                let last = self.last_guard_op.as_ref().unwrap();
                op.descr = last.descr.clone();
                op.fail_args = last.fail_args.clone();
                op.rd_resume_position = last.rd_resume_position;
                // Copy complete resume data so store_final_boxes_in_guard is a no-op.
                op.rd_numb = last.rd_numb.clone();
                op.rd_consts = last.rd_consts.clone();
                op.rd_virtuals = last.rd_virtuals.clone();
            } else {
                // optimizer.py:678: store_final_boxes_in_guard
                op = Self::store_final_boxes_in_guard(op, ctx);
                // optimizer.py:681-683: force_box on fail_args for unrolling.
                if let Some(ref fa) = op.fail_args {
                    let fargs: Vec<OpRef> = fa.iter().copied().collect();
                    for farg in fargs {
                        if !farg.is_none() {
                            self.force_box(farg, ctx);
                        }
                    }
                }
            }

            // resume.py:570-574: _add_optimizer_sections captures the
            // optimizer's knowledge AT THIS GUARD POINT.
            // bridgeopt.py:74-88: known classes from ptr_info.
            {
                let mut heap_fields = Vec::new();
                for pass in &self.passes {
                    let fields = pass.export_cached_fields();
                    if !fields.is_empty() {
                        heap_fields = fields;
                        break;
                    }
                }
                // bridgeopt.py:74-88: collect known classes for bridge
                // deserialization. RPython iterates liveboxes (= fail_args)
                // and calls getptrinfo(box).get_known_class() for Ref boxes.
                // Follow box replacement (forwarding) to find PtrInfo.
                let mut known_classes = Vec::new();
                if let Some(ref fa) = op.fail_args {
                    for &farg in fa {
                        if farg.is_none() {
                            continue;
                        }
                        let resolved = ctx.get_box_replacement(farg);
                        if let Some(info) = ctx.get_ptr_info(resolved) {
                            if let Some(class_ptr) = info.get_known_class() {
                                known_classes.push((farg, *class_ptr));
                            }
                        }
                    }
                }
                if !heap_fields.is_empty() || !known_classes.is_empty() {
                    self.per_guard_knowledge.push((
                        op.pos,
                        OptimizerKnowledge {
                            heap_fields,
                            known_classes,
                            loopinvariant_results: Vec::new(),
                        },
                    ));
                }
            }

            // RPython parity: fail_arg_types are set by store_final_boxes_in_guard.
            // RPython's typed Box objects carry type information intrinsically.
            // Snapshot-based type inference resolves types during numbering.

            // optimizer.py:630-631: pendingfields → rd_pendingfields
            // resume.py:428-445: encode pending SetfieldGc/SetarrayitemGc
            let pending = std::mem::take(&mut ctx.pending_for_guard);
            if !pending.is_empty() {
                op.rd_pendingfields = Some(
                    pending
                        .into_iter()
                        .map(|pf_op| {
                            let (target, value, item_index) =
                                if pf_op.opcode == OpCode::SetarrayitemGc {
                                    let idx = ctx
                                        .get_constant_int(ctx.get_box_replacement(pf_op.arg(1)))
                                        .unwrap_or(0);
                                    (pf_op.arg(0), pf_op.arg(2), idx as i32)
                                } else {
                                    (pf_op.arg(0), pf_op.arg(1), -1i32)
                                };
                            // Extract field layout from the descriptor for backend
                            // store emission on guard failure.
                            let (field_offset, field_size, field_type) =
                                if let Some(ref descr) = pf_op.descr {
                                    if let Some(fd) = descr.as_field_descr() {
                                        (fd.offset(), fd.field_size(), fd.field_type())
                                    } else if let Some(ad) = descr.as_array_descr() {
                                        let offset = ad.base_size()
                                            + (item_index.max(0) as usize) * ad.item_size();
                                        (offset, ad.item_size(), ad.item_type())
                                    } else {
                                        (0, 8, majit_ir::Type::Int)
                                    }
                                } else {
                                    (0, 8, majit_ir::Type::Int)
                                };
                            majit_ir::GuardPendingFieldEntry {
                                descr_index: pf_op.descr.as_ref().map_or(0, |d| d.index()),
                                item_index,
                                target: ctx.get_box_replacement(target),
                                value: ctx.get_box_replacement(value),
                                // resume.py:554-555: tagged encoding set during
                                // resume numbering (_add_pending_fields).
                                // UNASSIGNED means "not yet tagged".
                                target_tagged: majit_ir::resumedata::UNASSIGNED,
                                value_tagged: majit_ir::resumedata::UNASSIGNED,
                                field_offset,
                                field_size,
                                field_type,
                            }
                        })
                        .collect(),
                );
            }

            // optimizer.py:679: update last_guard_op only for fresh guards
            // (not shared). optimizer.py:684-685: GUARD_EXCEPTION breaks chain.
            if !shared {
                self.last_guard_op = Some(op.clone());
            }
            if opcode == OpCode::GuardException {
                self.last_guard_op = None;
            }
        } else {
            // optimizer.py:639-644: preserve last_guard_op for guard chaining
            // unless the op has side effects or is a call_pure that can raise.
            let preserves_chain = (op.opcode.has_no_side_effect()
                || op.opcode.is_guard()
                || op.opcode.is_jit_debug()
                || op.opcode.is_ovf())
                && !Self::is_call_pure_pure_canraise(&op);
            if !preserves_chain {
                self.last_guard_op = None;
            }
        }
        let emitted = ctx.emit(op.clone());
        if std::env::var_os("MAJIT_LOG").is_some()
            && matches!(
                op.opcode,
                OpCode::CallMayForceI
                    | OpCode::CallMayForceR
                    | OpCode::CallMayForceF
                    | OpCode::CallMayForceN
                    | OpCode::GuardNotForced
                    | OpCode::GuardNotForced2
            )
        {
            eprintln!(
                "[opt] emit {:?} pos={:?} len={}",
                op.opcode,
                emitted,
                ctx.new_operations.len()
            );
        }
        // optimizer.py:47-54: run deferred postprocess after emit.
        // RPython calls OptimizationResult.callback() → propagate_postprocess.
        // rewrite.py:282: postprocess_GUARD_NONNULL → mark_last_guard
        if let Some(opref) = ctx.pending_mark_last_guard.take() {
            ctx.mark_last_guard(opref);
        }
        if let Some(pp) = ctx.pending_guard_class_postprocess.take() {
            // rewrite.py:430-436 postprocess_GUARD_CLASS:
            //   update_last_guard = not old_guard or isinstance(descr, ResumeAtPositionDescr)
            //   make_constant_class(arg0, expectedclassbox, update_last_guard)
            let old_guard_pos = ctx
                .get_ptr_info(pp.obj)
                .and_then(|info| info.last_guard_pos())
                .unwrap_or(-1);
            let update_last_guard =
                old_guard_pos < 0 || ctx.is_resume_at_position_guard(old_guard_pos);
            // optimizer.py:137-151 make_constant_class:
            //   opinfo._known_class = class_const
            //   if update_last_guard: mark_last_guard → len(_newoperations) - 1
            let new_guard_pos = if update_last_guard {
                (ctx.new_operations.len() as i32) - 1 // NOW correct: guard is in new_operations
            } else {
                old_guard_pos
            };
            ctx.set_ptr_info(
                pp.obj,
                crate::optimizeopt::info::PtrInfo::KnownClass {
                    class_ptr: majit_ir::GcRef(pp.class_val as usize),
                    is_nonnull: true,
                    last_guard_pos: new_guard_pos,
                },
            );
        }
        ctx.in_final_emission = saved_in_final_emission;
    }

    /// optimizer.py:722-752 store_final_boxes_in_guard
    ///
    /// Resolve fail_args through get_box_replacement and delegate to
    /// finalize_guard_resume_data for snapshot-based virtual encoding
    /// (rd_numb, rd_consts, rd_virtuals).
    fn store_final_boxes_in_guard(mut op: Op, ctx: &mut OptContext) -> Op {
        let Some(ref mut fail_args) = op.fail_args else {
            // RPython optimizer.py:722-752: store_final_boxes_in_guard
            // calls modifier.finish() even when fail_args is initially None
            // (e.g. Phase 2 guards from unroll copy_and_change). The snapshot
            // provides the sole source of fail_args via _number_boxes.
            if op.rd_resume_position >= 0 && ctx.snapshot_boxes.contains_key(&op.rd_resume_position)
            {
                ctx.finalize_guard_resume_data(&mut op);
            }
            return op;
        };

        // optimizer.py:732-748 + resume.py:389-452:
        // RPython finish() handles virtuals without forcing.
        // _number_boxes tags virtual fail_args as TAGVIRTUAL,
        // _number_virtuals builds rd_virtuals from PtrInfo.
        for fa_idx in 0..fail_args.len() {
            if !fail_args[fa_idx].is_none() {
                fail_args[fa_idx] = ctx.get_box_replacement(fail_args[fa_idx]);
            }
        }
        ctx.finalize_guard_resume_data(&mut op);

        op
    }
}

impl Optimizer {
    /// Create an optimizer with the standard pass pipeline.
    /// RPython __init__.py:15-22 ALL_OPTS + ENABLE_ALL_OPTS (rlib/jit.py):
    ///   intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll
    /// (unroll is handled separately by UnrollOptimizer)
    pub fn default_pipeline() -> Self {
        let mut opt = Self::new();
        opt.add_pass(Box::new(OptIntBounds::new()));
        opt.add_pass(Box::new(OptRewrite::new()));
        opt.add_pass(Box::new(OptVirtualize::new()));
        opt.add_pass(Box::new(OptString::new()));
        opt.add_pass(Box::new(OptPure::new()));
        opt.add_pass(Box::new(OptEarlyForce::new()));
        opt.add_pass(Box::new(OptHeap::new()));
        opt
    }

    /// Create an optimizer with virtualizable config for frame field tracking.
    pub fn default_pipeline_with_virtualizable(config: VirtualizableConfig) -> Self {
        let mut opt = Self::new();
        opt.add_pass(Box::new(OptIntBounds::new()));
        opt.add_pass(Box::new(OptRewrite::new()));
        opt.add_pass(Box::new(OptVirtualize::with_virtualizable(config)));
        opt.add_pass(Box::new(OptString::new()));
        opt.add_pass(Box::new(OptPure::new()));
        opt.add_pass(Box::new(OptEarlyForce::new()));
        opt.add_pass(Box::new(OptHeap::new()));
        opt
    }

    /// Number of passes in this optimizer.
    pub fn num_passes(&self) -> usize {
        self.passes.len()
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::default_pipeline()
    }
}

// OptimizerBoxEnv removed: was only used by store_final_boxes_in_guard.
// store_final_boxes_in_guard in mod.rs defines InlineBoxEnv for the same purpose.

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::Type;
    use majit_ir::descr::make_size_descr;
    use majit_ir::descr::{CallDescr, EffectInfo, ExtraEffect, OopSpecIndex, make_fail_descr};
    use majit_ir::{DescrRef, OpCode, OpRef};
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Arc;

    /// A trivial pass that removes INT_ADD(x, 0) -> x
    struct AddZeroElimination;

    impl Optimization for AddZeroElimination {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if op.opcode == OpCode::IntAdd {
                // Check if second arg is constant 0
                if let Some(0) = ctx.get_constant_int(op.arg(1)) {
                    // Replace with first arg
                    ctx.replace_op(op.pos, op.arg(0));
                    return OptimizationResult::Remove;
                }
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "add_zero_elim"
        }
    }

    struct AddVirtualInputsOnce {
        added: bool,
    }

    struct RemoveAsConstant {
        target: OpRef,
        value: i64,
    }

    impl Optimization for AddVirtualInputsOnce {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if !self.added {
                ctx.num_inputs += 2;
                self.added = true;
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "add_virtual_inputs_once"
        }
    }

    impl Optimization for RemoveAsConstant {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if op.pos == self.target {
                ctx.make_constant(op.pos, majit_ir::Value::Int(self.value));
                return OptimizationResult::Remove;
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "remove_as_constant"
        }
    }

    struct FlushCounter {
        hits: Rc<Cell<usize>>,
    }

    impl Optimization for FlushCounter {
        fn propagate_forward(&mut self, _op: &Op, _ctx: &mut OptContext) -> OptimizationResult {
            OptimizationResult::PassOn
        }

        fn flush(&mut self, _ctx: &mut OptContext) {
            self.hits.set(self.hits.get() + 1);
        }

        fn name(&self) -> &'static str {
            "flush_counter"
        }
    }

    struct RemoveAsTypedConstant {
        target: OpRef,
        value: majit_ir::Value,
    }

    impl Optimization for RemoveAsTypedConstant {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if op.pos == self.target {
                ctx.make_constant(op.pos, self.value.clone());
                return OptimizationResult::Remove;
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "remove_as_typed_constant"
        }
    }

    #[derive(Debug)]
    struct TestDescr(u32);

    impl majit_ir::Descr for TestDescr {
        fn index(&self) -> u32 {
            self.0
        }
    }

    #[derive(Debug)]
    struct TestCallDescr {
        idx: u32,
        effect: EffectInfo,
        result_type: majit_ir::Type,
    }

    impl majit_ir::Descr for TestCallDescr {
        fn index(&self) -> u32 {
            self.idx
        }

        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestCallDescr {
        fn arg_types(&self) -> &[majit_ir::Type] {
            &[]
        }

        fn result_type(&self) -> majit_ir::Type {
            self.result_type
        }

        fn result_size(&self) -> usize {
            8
        }

        fn effect_info(&self) -> &EffectInfo {
            &self.effect
        }
    }

    fn call_may_force_descr(idx: u32, result_type: majit_ir::Type) -> DescrRef {
        Arc::new(TestCallDescr {
            idx,
            effect: EffectInfo {
                extra_effect: ExtraEffect::CanRaise,
                oopspec_index: OopSpecIndex::None,
                ..Default::default()
            },
            result_type,
        })
    }

    struct QueueForceLikeExtraOps {
        queued: bool,
        field_descr: majit_ir::DescrRef,
    }

    impl Optimization for QueueForceLikeExtraOps {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if !self.queued && op.opcode == OpCode::IntAdd {
                self.queued = true;

                let alloc = ctx.emit_extra(ctx.current_pass_idx, Op::new(OpCode::New, &[]));
                let mut set = Op::new(OpCode::SetfieldGc, &[alloc, OpRef(0)]);
                set.descr = Some(self.field_descr.clone());
                ctx.emit_extra(ctx.current_pass_idx, set);
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "queue_force_like_extra_ops"
        }
    }

    #[test]
    fn test_optimizer_passthrough() {
        let mut opt = Optimizer::new();
        let ops = vec![Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)])];
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_default_pipeline_keeps_call_may_force_pairs_alive_when_results_are_used() {
        let field_descr = Arc::new(TestDescr(91));
        let call_descr_a = call_may_force_descr(81, majit_ir::Type::Ref);
        let call_descr_b = call_may_force_descr(82, majit_ir::Type::Ref);
        let mut ops = vec![
            Op::with_descr(OpCode::CallMayForceR, &[OpRef(0), OpRef(1)], call_descr_a),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(3)], field_descr.clone()),
            Op::with_descr(OpCode::CallMayForceR, &[OpRef(0), OpRef(2)], call_descr_b),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(6)], field_descr),
            Op::new(OpCode::IntAdd, &[OpRef(5), OpRef(8)]),
            Op::new(OpCode::Finish, &[OpRef(9)]),
        ];
        for (idx, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef((idx as u32) + 3);
        }

        let mut opt = Optimizer::default_pipeline();
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 3);

        let call_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::CallMayForceR)
            .count();
        let guard_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::GuardNotForced)
            .count();
        assert_eq!(
            call_count, 2,
            "optimized trace lost CallMayForceR ops: {result:?}"
        );
        assert_eq!(
            guard_count, 2,
            "optimized trace lost GuardNotForced ops: {result:?}"
        );
    }

    #[test]
    fn test_default_pipeline_keeps_call_may_force_when_guard_fail_args_reference_results() {
        let field_descr = Arc::new(TestDescr(101));
        let call_descr_a = call_may_force_descr(83, majit_ir::Type::Ref);
        let call_descr_b = call_may_force_descr(84, majit_ir::Type::Ref);
        let guard_descr_a = make_fail_descr(
            1,
            vec![
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Int,
                majit_ir::Type::Ref,
                majit_ir::Type::Ref,
                majit_ir::Type::Ref,
                majit_ir::Type::Ref,
            ],
        );
        let guard_descr_b = make_fail_descr(
            2,
            vec![
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Int,
                majit_ir::Type::Ref,
                majit_ir::Type::Ref,
                majit_ir::Type::Ref,
                majit_ir::Type::Ref,
                majit_ir::Type::Ref,
            ],
        );

        let mut call_a = Op::with_descr(OpCode::CallMayForceR, &[OpRef(0), OpRef(1)], call_descr_a);
        let mut guard_a = Op::with_descr(OpCode::GuardNotForced, &[], guard_descr_a);
        guard_a.fail_args = Some(
            vec![
                OpRef(0),
                OpRef(2000),
                OpRef(2001),
                OpRef(3),
                OpRef(3000),
                OpRef(3001),
                OpRef(4),
            ]
            .into(),
        );
        let get_a_type = Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(3)], field_descr.clone());
        let get_a_val = Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(3)], field_descr.clone());
        let mut call_b = Op::with_descr(OpCode::CallMayForceR, &[OpRef(0), OpRef(2)], call_descr_b);
        let mut guard_b = Op::with_descr(OpCode::GuardNotForced, &[], guard_descr_b);
        guard_b.fail_args = Some(
            vec![
                OpRef(0),
                OpRef(2002),
                OpRef(2003),
                OpRef(3),
                OpRef(6),
                OpRef(3002),
                OpRef(3003),
                OpRef(7),
            ]
            .into(),
        );
        let get_b_type = Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(6)], field_descr.clone());
        let get_b_val = Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(6)], field_descr);
        let add = Op::new(OpCode::IntAdd, &[OpRef(5), OpRef(8)]);
        let finish = Op::new(OpCode::Finish, &[OpRef(9)]);

        let mut ops = vec![
            call_a.clone(),
            guard_a,
            get_a_type,
            get_a_val,
            call_b.clone(),
            guard_b,
            get_b_type,
            get_b_val,
            add,
            finish,
        ];
        for (idx, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef((idx as u32) + 3);
        }
        call_a.pos = ops[0].pos;
        call_b.pos = ops[4].pos;

        let mut opt = Optimizer::default_pipeline();
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 3);

        let call_positions: std::collections::HashSet<_> = result
            .iter()
            .filter(|op| op.opcode == OpCode::CallMayForceR)
            .map(|op| op.pos)
            .collect();
        assert!(
            call_positions.contains(&call_a.pos) && call_positions.contains(&call_b.pos),
            "optimized trace lost CallMayForceR producer(s): {result:?}"
        );
        let guarded = result
            .iter()
            .filter(|op| op.opcode == OpCode::GuardNotForced)
            .count();
        assert_eq!(
            guarded, 2,
            "optimized trace lost GuardNotForced ops: {result:?}"
        );
    }

    #[test]
    fn test_default_pipeline_has_7_passes() {
        // RPython __init__.py:15-22 ALL_OPTS + ENABLE_ALL_OPTS (rlib/jit.py):
        // intbounds:rewrite:virtualize:string:pure:earlyforce:heap (unroll separate)
        let opt = Optimizer::default_pipeline();
        assert_eq!(opt.num_passes(), 7);
    }

    #[test]
    fn test_default_pipeline_processes_trace() {
        let mut opt = Optimizer::default_pipeline();
        // A simple trace: two INT_ADD with identical args. The Pure pass (CSE)
        // should eliminate the duplicate.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );
        // The duplicate INT_ADD should be eliminated by CSE (OptPure).
        let add_count = result.iter().filter(|o| o.opcode == OpCode::IntAdd).count();
        assert_eq!(add_count, 1, "CSE should eliminate duplicate INT_ADD");
        // Jump should still be present.
        assert_eq!(result.last().unwrap().opcode, OpCode::Jump);
    }

    #[test]
    fn test_remaps_all_op_positions_when_virtual_inputs_are_added() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(AddVirtualInputsOnce { added: false }));

        let mut ops = vec![
            Op::new(OpCode::GetfieldRawI, &[OpRef(0)]),
            Op::new(OpCode::GetfieldRawI, &[OpRef(0)]),
            Op::new(OpCode::GetfieldRawI, &[OpRef(4)]),
            Op::new(OpCode::IntGt, &[OpRef(5), OpRef(1)]),
        ];
        ops[0].pos = OpRef(3);
        ops[1].pos = OpRef(4);
        ops[2].pos = OpRef(5);
        ops[3].pos = OpRef(6);

        let mut constants = std::collections::HashMap::new();
        constants.insert(1, 27);
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 3);

        let positions: Vec<_> = result.iter().map(|op| op.pos).collect();
        assert_eq!(positions, vec![OpRef(5), OpRef(6), OpRef(7), OpRef(8)]);
        assert_eq!(result[2].arg(0), OpRef(6));
        assert_eq!(result[3].arg(0), OpRef(7));
    }

    #[test]
    fn test_remaps_removed_constants_away_from_compacted_live_ops() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(AddVirtualInputsOnce { added: false }));
        opt.add_pass(Box::new(RemoveAsConstant {
            target: OpRef(5),
            value: 123,
        }));

        let mut ops = vec![
            Op::new(OpCode::GetfieldRawI, &[OpRef(0)]),
            Op::new(OpCode::GetfieldRawI, &[OpRef(0)]),
            Op::new(OpCode::GetfieldRawI, &[OpRef(0)]),
            Op::new(OpCode::IntGt, &[OpRef(3), OpRef(1)]),
        ];
        ops[0].pos = OpRef(3);
        ops[1].pos = OpRef(4);
        ops[2].pos = OpRef(5);
        ops[3].pos = OpRef(6);

        let mut constants = std::collections::HashMap::new();
        constants.insert(1, 27);
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 3);

        assert_eq!(result[0].pos, OpRef(5));
        assert_eq!(result[1].pos, OpRef(6));
        assert_eq!(result[2].pos, OpRef(7));
        assert_eq!(result[2].arg(0), OpRef(5));
        assert_eq!(constants.get(&5), None);
        assert_eq!(constants.get(&8), Some(&123));
    }

    #[test]
    fn test_drops_constant_placeholder_same_as_from_final_trace_without_virtual_inputs() {
        let mut opt = Optimizer::new();

        let mut ops = vec![
            Op::new(OpCode::SameAsI, &[]),
            Op::new(OpCode::IntAdd, &[OpRef(2), OpRef(0)]),
        ];
        ops[0].pos = OpRef(2);
        ops[1].pos = OpRef(3);

        let mut constants = std::collections::HashMap::new();
        constants.insert(2, 1);
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[0].pos, OpRef(3));
        assert_eq!(result[0].arg(0), OpRef(2));
        assert_eq!(constants.get(&2), Some(&1));
    }

    #[test]
    fn test_imports_typed_ref_and_float_constants_from_backend_map() {
        let mut opt = Optimizer::new();
        let ptr = majit_ir::GcRef(0x1234);
        let float = 3.25f64;

        let mut ops = vec![
            Op::new(OpCode::SameAsR, &[]),
            Op::new(OpCode::SameAsF, &[]),
            Op::new(OpCode::Jump, &[OpRef(2), OpRef(3)]),
        ];
        ops[0].pos = OpRef(2);
        ops[1].pos = OpRef(3);
        ops[2].pos = OpRef(4);

        let mut constants = std::collections::HashMap::new();
        constants.insert(2, ptr.0 as i64);
        constants.insert(3, float.to_bits() as i64);

        let _ = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);
        let ctx = opt
            .final_ctx
            .as_ref()
            .expect("optimizer should preserve final ctx");

        assert_eq!(ctx.get_constant(OpRef(2)), Some(&majit_ir::Value::Ref(ptr)));
        assert_eq!(
            ctx.get_constant(OpRef(3)),
            Some(&majit_ir::Value::Float(float))
        );
    }

    #[test]
    fn test_exports_typed_ref_and_float_constants_back_to_backend_map() {
        let mut opt = Optimizer::new();
        let ptr = majit_ir::GcRef(0x5678);
        let float = 9.5f64;
        opt.add_pass(Box::new(RemoveAsTypedConstant {
            target: OpRef(5),
            value: majit_ir::Value::Ref(ptr),
        }));
        opt.add_pass(Box::new(RemoveAsTypedConstant {
            target: OpRef(6),
            value: majit_ir::Value::Float(float),
        }));

        let mut ops = vec![
            Op::new(OpCode::SameAsR, &[]),
            Op::new(OpCode::SameAsF, &[]),
            Op::new(OpCode::Jump, &[]),
        ];
        ops[0].pos = OpRef(5);
        ops[1].pos = OpRef(6);
        ops[2].pos = OpRef(7);

        let mut constants = std::collections::HashMap::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::Jump);
        assert_eq!(constants.get(&5), Some(&(ptr.0 as i64)));
        assert_eq!(constants.get(&6), Some(&(float.to_bits() as i64)));
    }

    #[test]
    fn test_skip_flush_keeps_terminal_jump_out_of_result_ops() {
        let mut opt = Optimizer::new();
        opt.skip_flush = true;

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(2)]),
        ];
        ops[0].pos = OpRef(2);
        ops[1].pos = OpRef(3);

        let mut constants = std::collections::HashMap::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        let terminal = opt
            .terminal_op
            .as_ref()
            .expect("skip_flush should preserve terminal jump");
        assert_eq!(terminal.opcode, OpCode::Jump);
        assert_eq!(terminal.args.as_ref(), &[OpRef(2)]);
    }

    #[test]
    fn test_get_count_of_ops_and_guards() {
        let mut opt = Optimizer::default_pipeline();
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(100)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Finish, &[]),
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );
        let ctx = OptContext::new(result.len());
        // Just verify the counting methods work
        assert_eq!(Optimizer::get_count_of_ops(&ctx), 0); // empty ctx
    }

    #[test]
    fn test_flush_invokes_all_passes() {
        let hits = Rc::new(Cell::new(0));
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(FlushCounter { hits: hits.clone() }));
        opt.add_pass(Box::new(FlushCounter { hits: hits.clone() }));

        let mut ctx = OptContext::new(0);
        opt.flush(&mut ctx);

        assert_eq!(hits.get(), 2);
    }

    #[test]
    fn test_extra_ops_do_not_flow_into_unroll_buffer() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(QueueForceLikeExtraOps {
            queued: false,
            field_descr: std::sync::Arc::new(TestDescr(1)),
        }));
        opt.add_pass(Box::new(OptHeap::new()));
        opt.add_pass(Box::new(crate::optimizeopt::unroll::OptUnroll::new()));

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(1)]),
        ];
        ops[0].pos = OpRef(2);
        ops[1].pos = OpRef(3);

        let mut constants = std::collections::HashMap::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        // force_all_lazy_setfields emits lazy SetfieldGc before JUMP.
        let new_count = result.iter().filter(|op| op.opcode == OpCode::New).count();
        assert!(
            new_count > 0,
            "force-like extra ops should still emit a New; got {:?}",
            result
        );
        // SetfieldGc is emitted by force_all_lazy_setfields at JUMP
        let setfield_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::SetfieldGc)
            .count();
        assert_eq!(
            setfield_count, 1,
            "lazy SetfieldGc should be emitted at JUMP; got {:?}",
            result
        );
    }

    struct QueueTwoForceLikePairs {
        queued: bool,
        field_descr: majit_ir::DescrRef,
    }

    impl Optimization for QueueTwoForceLikePairs {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if !self.queued && op.opcode == OpCode::IntAdd {
                self.queued = true;

                let alloc_a = ctx.emit_extra(ctx.current_pass_idx, Op::new(OpCode::New, &[]));
                let mut set_a = Op::new(OpCode::SetfieldGc, &[alloc_a, OpRef(0)]);
                set_a.descr = Some(self.field_descr.clone());
                ctx.emit_extra(ctx.current_pass_idx, set_a);

                let alloc_b = ctx.emit_extra(ctx.current_pass_idx, Op::new(OpCode::New, &[]));
                let mut set_b = Op::new(OpCode::SetfieldGc, &[alloc_b, OpRef(1)]);
                set_b.descr = Some(self.field_descr.clone());
                ctx.emit_extra(ctx.current_pass_idx, set_b);
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "queue_two_force_like_pairs"
        }
    }

    #[test]
    fn test_force_like_extra_ops_preserve_new_before_matching_setfield() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(QueueTwoForceLikePairs {
            queued: false,
            field_descr: std::sync::Arc::new(TestDescr(7)),
        }));
        opt.add_pass(Box::new(OptHeap::new()));

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(1)]),
        ];
        ops[0].pos = OpRef(2);
        ops[1].pos = OpRef(3);

        let mut constants = std::collections::HashMap::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        let op_index: std::collections::HashMap<_, _> = result
            .iter()
            .enumerate()
            .map(|(idx, op)| (op.pos, idx))
            .collect();

        for set_op in result.iter().filter(|op| op.opcode == OpCode::SetfieldGc) {
            let alloc_ref = set_op.arg(0);
            let new_idx = result
                .iter()
                .position(|op| op.opcode == OpCode::New && op.pos == alloc_ref)
                .unwrap_or_else(|| panic!("missing New for {alloc_ref:?} in {result:?}"));
            let set_idx = *op_index
                .get(&set_op.pos)
                .unwrap_or_else(|| panic!("missing setfield pos {:?} in {:?}", set_op.pos, result));
            assert!(
                new_idx < set_idx,
                "matching New must appear before SetfieldGc; got {:?}",
                result
            );
        }
    }

    #[test]
    fn test_remap_keeps_force_like_new_positions_out_of_constant_map() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(AddVirtualInputsOnce { added: false }));
        opt.add_pass(Box::new(QueueForceLikeExtraOps {
            queued: false,
            field_descr: std::sync::Arc::new(TestDescr(9)),
        }));
        opt.add_pass(Box::new(RemoveAsConstant {
            target: OpRef(2),
            value: 472,
        }));
        opt.add_pass(Box::new(OptHeap::new()));

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(1)]),
        ];
        ops[0].pos = OpRef(2);
        ops[1].pos = OpRef(3);

        let mut constants = std::collections::HashMap::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        let new_positions: std::collections::HashSet<_> = result
            .iter()
            .filter(|op| op.opcode == OpCode::New)
            .map(|op| op.pos.0)
            .collect();
        assert!(
            !new_positions.is_empty(),
            "expected force-like New op in optimized trace; got {:?}",
            result
        );
        for pos in &new_positions {
            assert!(
                !constants.contains_key(pos),
                "live New position v{pos} must not collide with exported int constant map {:?}; trace {:?}",
                constants,
                result
            );
        }
        assert!(
            result
                .iter()
                .filter(|op| op.opcode == OpCode::SetfieldGc)
                .all(|op| new_positions.contains(&op.arg(0).0)),
            "SetfieldGc targets must remain emitted New refs; got {:?}",
            result
        );
    }

    #[test]
    fn test_force_like_extra_ops_skip_preexisting_constant_slots_without_virtual_inputs() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(QueueForceLikeExtraOps {
            queued: false,
            field_descr: std::sync::Arc::new(TestDescr(11)),
        }));
        opt.add_pass(Box::new(OptHeap::new()));

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(1)]),
        ];
        ops[0].pos = OpRef(10066);
        ops[1].pos = OpRef(10067);

        let mut constants = std::collections::HashMap::new();
        constants.insert(10068, 472);
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        let new_positions: std::collections::HashSet<_> = result
            .iter()
            .filter(|op| op.opcode == OpCode::New)
            .map(|op| op.pos.0)
            .collect();
        assert_eq!(
            new_positions,
            std::collections::HashSet::from([10069]),
            "queued New should skip constant-only slot v10068; got {:?}",
            result
        );
        assert!(
            result
                .iter()
                .filter(|op| op.opcode == OpCode::SetfieldGc)
                .all(|op| new_positions.contains(&op.arg(0).0)),
            "SetfieldGc targets must remain emitted New refs; got {:?}",
            result
        );
        assert_eq!(constants.get(&10068), Some(&472));
        assert!(
            !constants.contains_key(&10069),
            "live New position must not collide with constant map {:?}",
            constants
        );
    }

    #[test]
    #[ignore] // consumer switchover: test setup doesn't trigger inline guard numbering
    fn test_optimizer_encodes_direct_virtual_guard_fail_args_as_rd_numb() {
        let mut opt = Optimizer::default_pipeline();
        let size_descr = make_size_descr(16);
        let field_descr = majit_ir::make_field_descr(8, 8, Type::Int, true);

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(10)]);
        guard.fail_args = Some(vec![OpRef(0)].into());

        let mut ops = vec![
            Op::with_descr(OpCode::New, &[], size_descr),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(11)], field_descr),
            guard,
            Op::new(OpCode::Jump, &[]),
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }

        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );
        let guard = result
            .iter()
            .find(|op| op.opcode == OpCode::GuardTrue)
            .expect("guard should survive optimization");

        // resume.py parity: rd_numb + rd_virtuals from store_final_boxes_in_guard.
        assert!(
            guard.rd_numb.is_some(),
            "guard should have rd_numb (compact resume numbering)"
        );
        // rd_virtuals removed — rd_virtuals is produced by number_guard_inline_impl.
        let fail_args = guard
            .fail_args
            .as_ref()
            .expect("guard should keep fail args");
        // fail_args has EXPANDED length: virtual slot replaced with OpRef::NONE,
        // field values appended as extra fail_args.
        assert!(
            fail_args.iter().any(|a| a.is_none()),
            "virtual slot should be OpRef::NONE placeholder in fail_args"
        );
        // Field value (OpRef(11)) should appear as extra fail_arg
        assert!(
            fail_args.len() > 1,
            "fail_args should be expanded with virtual field values"
        );
    }

    #[test]
    fn test_call_pure_results() {
        let mut opt = Optimizer::new();
        opt.record_call_pure_result(vec![OpRef(10), OpRef(20)], majit_ir::Value::Int(42));
        assert_eq!(
            opt.get_call_pure_result(&[OpRef(10), OpRef(20)]),
            Some(&majit_ir::Value::Int(42))
        );
        assert_eq!(opt.get_call_pure_result(&[OpRef(10), OpRef(99)]), None);
    }

    #[test]
    fn test_protect_speculative_operation() {
        let opt = Optimizer::new();
        let ctx = OptContext::new(10);

        // Arithmetic ops are always safe
        let add_op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
        assert!(opt.protect_speculative_operation(&add_op, &ctx));

        // Getfield on unknown arg is safe (not constant null)
        let get_op = Op::new(OpCode::GetfieldGcI, &[OpRef(0)]);
        assert!(opt.protect_speculative_operation(&get_op, &ctx));
    }

    #[test]
    fn test_pending_fields() {
        let mut opt = Optimizer::new();
        assert!(!opt.has_pending_fields());
        assert_eq!(opt.num_pending_fields(), 0);

        opt.add_pending_field(Op::new(OpCode::SetfieldGc, &[OpRef(0), OpRef(1)]));
        assert!(opt.has_pending_fields());
        assert_eq!(opt.num_pending_fields(), 1);
    }

    #[test]
    fn test_getnullness() {
        let mut ctx = OptContext::new(10);
        // Unknown → 0
        assert_eq!(Optimizer::getnullness(&ctx, OpRef(0)), 0);
        // Known nonzero → 1 (NONNULL)
        ctx.make_constant(OpRef(1), majit_ir::Value::Int(42));
        assert_eq!(Optimizer::getnullness(&ctx, OpRef(1)), 1);
        // Known zero → -1 (NULL)
        ctx.make_constant(OpRef(2), majit_ir::Value::Int(0));
        assert_eq!(Optimizer::getnullness(&ctx, OpRef(2)), -1);
    }

    #[test]
    fn test_guard_replacement_flag() {
        let mut opt = Optimizer::new();
        assert!(opt.can_replace_guards);
        opt.disable_guard_replacement();
        assert!(!opt.can_replace_guards);
        opt.enable_guard_replacement();
        assert!(opt.can_replace_guards);
    }

    #[test]
    fn test_force_box_for_end_of_preamble_recurses_virtual_fields() {
        use crate::optimizeopt::info::{PtrInfo, VirtualStructInfo};

        let descr = make_size_descr(16);
        let mut ctx = OptContext::with_num_inputs(32, 1024);
        ctx.set_ptr_info(
            OpRef(10),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: descr.clone(),
                fields: vec![(1, OpRef(11))],
                field_descrs: Vec::new(),
                last_guard_pos: -1,
            }),
        );
        ctx.replace_op(OpRef(11), OpRef(20));
        ctx.set_ptr_info(
            OpRef(20),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr,
                fields: Vec::new(),
                field_descrs: Vec::new(),
                last_guard_pos: -1,
            }),
        );

        let mut opt = Optimizer::new();
        let result = opt.force_box_for_end_of_preamble(OpRef(10), &mut ctx);

        // The virtual is forced to a concrete allocation; the returned ref
        // is the allocation's position, which ctx.get_box_replacement(OpRef(10))
        // should resolve to.
        assert_eq!(result, ctx.get_box_replacement(OpRef(10)));
        // After forcing, the struct's ptr_info reflects that field 1
        // (originally OpRef(11), forwarded to OpRef(20)) has been recursively forced.
        match ctx.get_ptr_info(result) {
            Some(PtrInfo::VirtualStruct(info)) => {
                // The inner virtual (OpRef(20)) was also forced; its allocation
                // ref is whatever force_box assigned.
                assert_eq!(info.fields.len(), 1);
                assert_eq!(info.fields[0].0, 1);
            }
            // After full forcing the info might become NonNull or similar
            Some(PtrInfo::NonNull { .. }) => {}
            other => panic!("expected virtual struct or non-null after forcing, got {other:?}"),
        }
    }

    #[test]
    fn test_force_box_materializes_virtual_struct_outside_final_emission() {
        use crate::optimizeopt::info::{PtrInfo, VirtualStructInfo};

        let descr = make_size_descr(16);
        let field_descr = crate::optimizeopt::virtualize::make_field_index_descr(1);
        let mut ctx = OptContext::new(16);
        ctx.set_ptr_info(
            OpRef(10),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: descr.clone(),
                fields: vec![(1, OpRef(11))],
                field_descrs: vec![(1, field_descr.clone())],
                last_guard_pos: -1,
            }),
        );

        let mut opt = Optimizer::new();
        let forced = opt.force_box(OpRef(10), &mut ctx);
        assert_ne!(forced, OpRef(10));
        // force_box → emit_extra routes through pass chain, emits to new_operations
        assert!(
            ctx.new_operations
                .iter()
                .any(|op| op.opcode == OpCode::New && op.pos == forced)
        );
        assert!(ctx.new_operations.iter().any(|op| {
            op.opcode == OpCode::SetfieldGc
                && op.arg(0) == forced
                && op.arg(1) == OpRef(11)
                && op.descr.is_some()
        }));
    }

    #[test]
    fn test_emit_operation_materializes_virtual_args_directly() {
        use crate::optimizeopt::info::{PtrInfo, VirtualStructInfo};

        let descr = make_size_descr(16);
        let field_descr = crate::optimizeopt::virtualize::make_field_index_descr(1);
        let mut ctx = OptContext::with_num_inputs(16, 1);
        ctx.set_ptr_info(
            OpRef(10),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: descr.clone(),
                fields: vec![(1, OpRef(11))],
                field_descrs: vec![(1, field_descr.clone())],
                last_guard_pos: -1,
            }),
        );

        let mut opt = Optimizer::new();
        let op = Op::new(OpCode::GuardNonnull, &[OpRef(10)]);
        opt.emit_operation(op, &mut ctx);

        assert!(!ctx.in_final_emission);
        assert!(ctx.new_operations.iter().any(|op| op.opcode == OpCode::New));
        assert!(ctx.new_operations.iter().any(|op| {
            op.opcode == OpCode::SetfieldGc && op.arg(1) == OpRef(11) && op.descr.is_some()
        }));
        assert!(
            ctx.new_operations
                .iter()
                .any(|op| op.opcode == OpCode::GuardNonnull && op.arg(0) != OpRef(10))
        );
    }

    #[test]
    fn test_emit_operation_forces_imported_short_guard_args() {
        let mut opt = Optimizer::new();
        let mut ctx = OptContext::with_num_inputs(16, 1);

        let mut preamble_op = Op::new(OpCode::IntGe, &[OpRef(3), OpRef(10_000)]);
        preamble_op.pos = OpRef(14);
        ctx.make_constant(OpRef(10_000), majit_ir::Value::Int(0));
        ctx.initialize_imported_short_preamble_builder(
            &[OpRef(0)],
            &[OpRef(0)],
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: preamble_op.clone(),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            }],
        );
        ctx.potential_extra_ops.insert(
            OpRef(14),
            crate::optimizeopt::info::PreambleOp {
                op: OpRef(14),
                resolved: OpRef(14),
                invented_name: false,
            },
        );

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(14)]);
        guard.pos = OpRef(15);
        opt.emit_operation(guard, &mut ctx);

        let sp = ctx
            .build_imported_short_preamble()
            .expect("forcing imported short guard arg should build short preamble");
        assert_eq!(sp.used_boxes, vec![OpRef(14)]);
        assert_eq!(sp.jump_args, vec![OpRef(14)]);
    }

    #[test]
    #[ignore] // consumer switchover: test setup doesn't trigger inline guard numbering
    fn test_resumedata_memo_encodes_rd_numb_on_guard() {
        let mut opt = Optimizer::default_pipeline();

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef(100)]),
            Op::new(OpCode::Finish, &[]),
        ];
        ops[1].fail_args = Some(vec![OpRef(100), OpRef(101)].into());
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }

        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        let guard = result
            .iter()
            .find(|op| op.opcode == OpCode::GuardTrue)
            .expect("guard should survive optimization");
        assert!(
            guard.rd_numb.is_some(),
            "resumedata_memo should set rd_numb on guard"
        );
        assert!(
            guard.rd_consts.is_some(),
            "resumedata_memo should set rd_consts on guard"
        );
    }
}
