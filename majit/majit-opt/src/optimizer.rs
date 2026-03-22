use crate::{OptContext, Optimization, OptimizationResult};
/// Main optimization driver.
///
/// Translated from rpython/jit/metainterp/optimizeopt/optimizer.py.
/// Chains multiple optimization passes and drives operations through them.
use crate::{
    guard::GuardStrengthenOpt,
    heap::OptHeap,
    intbounds::OptIntBounds,
    pure::OptPure,
    rewrite::OptRewrite,
    simplify::OptSimplify,
    virtualize::{OptVirtualize, VirtualizableConfig},
    vstring::OptString,
};
use majit_ir::{GuardVirtualEntry, Op, OpCode, OpRef, Type};

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
    quasi_immutable_deps: std::collections::HashSet<u32>,
    /// optimizer.py: `resumedata_memo` — shared constant map for resume data.
    /// Maps constant values to shared indices to reduce resume data size.
    /// In RPython this is `resume.ResumeDataLoopMemo`; here we use a simple
    /// HashMap since the full type lives in majit-meta (no circular dep).
    resumedata_memo_consts: std::collections::HashMap<i64, u32>,
    /// RPython unroll.py: virtual structures found in JUMP args during preamble.
    /// Populated by OptVirtualize.export_virtual_for_preamble().
    pub exported_jump_virtuals: Vec<ExportedJumpVirtual>,
    /// RPython unroll.py: import_state — virtual structures to inject at Phase 2 start.
    /// Maps the original loop-carried input slot to a recursive abstract
    /// description of the virtual's field values.
    pub imported_virtuals: Vec<ImportedVirtual>,
    /// RPython unroll.py: export_state — exported optimizer facts at the end
    /// of the preamble, adapted to majit's slot-based inputarg model.
    pub exported_loop_state: Option<crate::unroll::ExportedState>,
    /// RPython unroll.py: import_state — exported facts to re-apply onto the
    /// next optimizer instance before phase 2 body optimization starts.
    pub imported_loop_state: Option<crate::unroll::ExportedState>,
    /// Original preamble result boxes for imported short-box results.
    pub imported_short_sources: Vec<crate::ImportedShortSource>,
    /// Invented SameAs aliases imported from short-preamble export/import.
    pub imported_short_aliases: Vec<crate::ImportedShortAlias>,
    /// Builder-derived short preamble actually used by phase 2.
    pub imported_short_preamble: Option<crate::shortpreamble::ShortPreamble>,
    /// RPython unroll.py: short_preamble_producer after import_state.
    /// Preserved so finalize_short_preamble can create the live extended
    /// producer for the target token currently being compiled.
    pub imported_short_preamble_builder: Option<crate::shortpreamble::ShortPreambleBuilder>,
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
}

fn value_from_backend_constant_bits(opref: OpRef, raw: i64, ops: &[Op]) -> majit_ir::Value {
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
    pub fields: Vec<(majit_ir::DescrRef, crate::virtualstate::VirtualStateInfo)>,
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

/// RPython unroll.py: ExportedState virtual field info.
/// Records a virtual object's structure at JUMP for preamble peeling phase 2.
#[derive(Clone, Debug)]
pub struct ExportedJumpVirtual {
    /// Index in JUMP args where this virtual was.
    pub jump_arg_index: usize,
    /// Size descriptor for New().
    pub size_descr: majit_ir::DescrRef,
    /// Whether this exported virtual is an instance or a plain struct.
    pub kind: ImportedVirtualKind,
    /// Descr index of the pool GetfieldGcR that loaded this head.
    pub head_load_descr_index: Option<u32>,
    /// Fields: (field_descr, exported abstract info for the field value).
    pub fields: Vec<(majit_ir::DescrRef, crate::virtualstate::VirtualStateInfo)>,
}

impl Optimizer {
    fn is_constant_placeholder_op(
        op: &Op,
        constants: &[Option<majit_ir::Value>],
    ) -> bool {
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
        info: &crate::virtualstate::VirtualStateInfo,
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
        info: &crate::virtualstate::VirtualStateInfo,
        opref: OpRef,
        ctx: &mut OptContext,
    ) {
        use crate::virtualstate::VirtualStateInfo;

        match info {
            VirtualStateInfo::Constant(value) => {
                ctx.make_constant(opref, value.clone());
            }
            VirtualStateInfo::Virtual {
                descr,
                known_class,
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
                    crate::info::PtrInfo::Virtual(crate::info::VirtualInfo {
                        descr: descr.clone(),
                        known_class: *known_class,
                        fields: imported_fields,
                        field_descrs: field_descrs.clone(),
                    }),
                );
            }
            VirtualStateInfo::VirtualArray { descr, items, .. } => {
                let imported_items = items
                    .iter()
                    .map(|item_info| Self::import_virtual_state_value(item_info, ctx))
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::info::PtrInfo::VirtualArray(crate::info::VirtualArrayInfo {
                        descr: descr.clone(),
                        items: imported_items,
                    }),
                );
            }
            VirtualStateInfo::VirtualStruct {
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
                    crate::info::PtrInfo::VirtualStruct(crate::info::VirtualStructInfo {
                        descr: descr.clone(),
                        fields: imported_fields,
                        field_descrs: field_descrs.clone(),
                    }),
                );
            }
            VirtualStateInfo::VirtualArrayStruct {
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
                    crate::info::PtrInfo::VirtualArrayStruct(crate::info::VirtualArrayStructInfo {
                        descr: descr.clone(),
                        element_fields: imported_elements,
                    }),
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
                    crate::info::PtrInfo::VirtualRawBuffer(crate::info::VirtualRawBufferInfo {
                        size: *size,
                        entries: imported_entries,
                    }),
                );
            }
            VirtualStateInfo::KnownClass { class_ptr } => {
                ctx.set_ptr_info(
                    opref,
                    crate::info::PtrInfo::KnownClass {
                        class_ptr: *class_ptr,
                        is_nonnull: true,
                    },
                );
            }
            VirtualStateInfo::NonNull => {
                ctx.set_ptr_info(opref, crate::info::PtrInfo::NonNull);
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
        // RPython: virtual field values come from the virtuals portion of
        // short_args (= label_args + virtuals). Use imported_virtual_args
        // which has (base_len, short_args) to start from the right offset.
        // RPython: virtual field values are at positions AFTER the
        // non-virtual args in the flattened boxes array from make_inputargs.
        // label_slot starts at the number of non-virtual state entries.
        let imported_label_args = self
            .imported_label_args
            .as_ref()
            .expect("install_imported_virtuals requires imported_label_args");
        // Count non-virtual top-level states to determine the starting slot
        // for virtual field values. This matches RPython's position_in_notvirtuals.
        let num_notvirtual = if let Some(ref exported) = self.imported_loop_state {
            exported.virtual_state.state.iter()
                .filter(|s| !s.is_virtual())
                .map(|s| crate::virtualstate::VirtualState::count_forced_boxes_for_entry_static(s))
                .sum()
        } else {
            0
        };
        let mut label_slot = num_notvirtual;
        if crate::majit_log_enabled() {
            eprintln!("[jit] install_virt: label_slot={} label_args_len={} label_args={:?}",
                label_slot, imported_label_args.len(), imported_label_args);
        }
        for iv in &self.imported_virtuals {
            let virtual_head = ctx.get_replacement(OpRef(iv.inputarg_index as u32));
            let mut fields = Vec::new();
            let mut field_descrs = Vec::new();
            for (descr, field_info) in &iv.fields {
                let field_ref = Self::import_virtual_state_from_label_args(
                    field_info,
                    imported_label_args,
                    &mut label_slot,
                    ctx,
                );
                fields.push((descr.index(), field_ref));
                field_descrs.push((descr.index(), descr.clone()));
            }
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[jit] install_imported_virtual head={virtual_head:?} fields={:?}",
                    fields
                );
            }
            match &iv.kind {
                ImportedVirtualKind::Instance { known_class } => {
                    ctx.set_ptr_info(
                        virtual_head,
                        crate::info::PtrInfo::Virtual(crate::info::VirtualInfo {
                            descr: iv.size_descr.clone(),
                            known_class: *known_class,
                            fields,
                            field_descrs,
                        }),
                    );
                }
                ImportedVirtualKind::Struct => {
                    ctx.set_ptr_info(
                        virtual_head,
                        crate::info::PtrInfo::VirtualStruct(crate::info::VirtualStructInfo {
                            descr: iv.size_descr.clone(),
                            fields,
                            field_descrs,
                        }),
                    );
                }
            }
            if let Some(descr_idx) = iv.head_load_descr_index {
                ctx.imported_virtual_heads
                    .push((descr_idx as usize, virtual_head));
            }
        }
    }

    fn import_virtual_state_from_label_args(
        info: &crate::virtualstate::VirtualStateInfo,
        imported_label_args: &[OpRef],
        label_slot: &mut usize,
        ctx: &mut OptContext,
    ) -> OpRef {
        use crate::virtualstate::VirtualStateInfo;

        match info {
            VirtualStateInfo::Constant(_) => Self::import_virtual_state_value(info, ctx),
            VirtualStateInfo::Virtual {
                descr,
                known_class,
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
                    crate::info::PtrInfo::Virtual(crate::info::VirtualInfo {
                        descr: descr.clone(),
                        known_class: *known_class,
                        fields: imported_fields,
                        field_descrs: field_descrs.clone(),
                    }),
                );
                opref
            }
            VirtualStateInfo::VirtualArray { descr, items, .. } => {
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
                    crate::info::PtrInfo::VirtualArray(crate::info::VirtualArrayInfo {
                        descr: descr.clone(),
                        items: imported_items,
                    }),
                );
                opref
            }
            VirtualStateInfo::VirtualStruct {
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
                    crate::info::PtrInfo::VirtualStruct(crate::info::VirtualStructInfo {
                        descr: descr.clone(),
                        fields: imported_fields,
                        field_descrs: field_descrs.clone(),
                    }),
                );
                opref
            }
            VirtualStateInfo::VirtualArrayStruct {
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
                    crate::info::PtrInfo::VirtualArrayStruct(crate::info::VirtualArrayStructInfo {
                        descr: descr.clone(),
                        element_fields: imported_elements,
                    }),
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
                    crate::info::PtrInfo::VirtualRawBuffer(crate::info::VirtualRawBufferInfo {
                        size: *size,
                        entries: imported_entries,
                    }),
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
                            eprintln!("[jit] MISS: label_slot={} len={}", *label_slot, imported_label_args.len());
                        }
                        OpRef::NONE
                    });
                *label_slot += 1;
                Self::apply_imported_virtual_state(info, opref, ctx);
                ctx.get_replacement(opref)
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
            exported_jump_virtuals: Vec::new(),
            imported_virtuals: Vec::new(),
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

    /// Get the final num_inputs after optimization.
    /// May be larger than the original if virtualizable added virtual input args.
    pub fn final_num_inputs(&self) -> usize {
        self.final_num_inputs
    }

    /// optimizer.py: getlastop() — get the last emitted guard operation.
    pub fn get_last_guard_op(&self) -> Option<&Op> {
        self.last_guard_op.as_ref()
    }

    /// optimizer.py: notice_guard_future_condition(op)
    /// Record that a guard at the given position should be replaced
    /// with the given op when the future condition is realized.
    pub fn notice_guard_future_condition(&mut self, guard_pos: u32, replacement: Op) {
        self.replaces_guard.insert(guard_pos, replacement);
    }

    /// optimizer.py: replace_guard(old_guard_pos, new_guard)
    /// Replace a previously emitted guard with a new one.
    pub fn replace_guard(&mut self, old_pos: u32, new_guard: Op) {
        self.replaces_guard.insert(old_pos, new_guard);
    }

    /// optimizer.py: store_final_boxes_in_guard(guard_op, pendingfields)
    ///
    /// In RPython, this stores pending field writes into the guard's resume
    /// data (`rd_pendingfields`) so they can be replayed on guard failure.
    /// In majit, pending fields are currently emitted as ops before the guard
    /// (see `emit_guard_operation`) or kept deferred in heap.rs lazy sets.
    /// This method is reserved for future resume data integration.
    /// optimizer.py: store_final_boxes_in_guard(op)
    ///
    /// RPython parity: walk fail_args, detect virtual objects, expand
    /// fail_args with virtual field values, record blueprints in rd_virtuals.
    /// resume.py: ResumeDataVirtualAdder.finish() equivalent.
    pub fn store_final_boxes_in_guard(&mut self, guard_op: &mut Op, ctx: &OptContext) {
        let pending = std::mem::take(&mut self.pendingfields);
        if guard_op.fail_args.is_none() {
            guard_op.fail_args = Some(Default::default());
        }
        let _ = pending;

        let Some(ref fail_args) = guard_op.fail_args else { return };
        let original_len = fail_args.len();
        let mut extra_args: Vec<OpRef> = Vec::new();
        let mut virtual_entries: Vec<GuardVirtualEntry> = Vec::new();

        for fa_idx in 0..original_len {
            let opref = fail_args[fa_idx];
            if opref.is_none() {
                continue;
            }
            let resolved = ctx.get_replacement(opref);
            let Some(info) = ctx.get_ptr_info(resolved) else { continue };

            match info.clone() {
                crate::info::PtrInfo::VirtualStruct(vinfo) => {
                    let base_idx = original_len + extra_args.len();
                    let mut fields = Vec::new();
                    for (i, (field_idx, value_ref)) in vinfo.fields.iter().enumerate() {
                        let resolved_val = ctx.get_replacement(*value_ref);
                        extra_args.push(resolved_val);
                        fields.push((*field_idx, base_idx + i));
                    }
                    virtual_entries.push(GuardVirtualEntry {
                        fail_arg_index: fa_idx,
                        descr: vinfo.descr.clone(),
                        known_class: None,
                        fields,
                    });
                }
                crate::info::PtrInfo::Virtual(vinfo) => {
                    let base_idx = original_len + extra_args.len();
                    let mut fields = Vec::new();
                    for (i, (field_idx, value_ref)) in vinfo.fields.iter().enumerate() {
                        let resolved_val = ctx.get_replacement(*value_ref);
                        extra_args.push(resolved_val);
                        fields.push((*field_idx, base_idx + i));
                    }
                    virtual_entries.push(GuardVirtualEntry {
                        fail_arg_index: fa_idx,
                        descr: vinfo.descr.clone(),
                        known_class: vinfo.known_class,
                        fields,
                    });
                }
                _ => {}
            }
        }

        if !extra_args.is_empty() {
            let fa = guard_op.fail_args.as_mut().unwrap();
            fa.extend(extra_args);
        }
        if !virtual_entries.is_empty() {
            guard_op.rd_virtuals = Some(virtual_entries);
        }
    }

    /// optimizer.py: emit_guard_operation(op, pendingfields)
    /// Emit a guard with resume data sharing when possible.
    ///
    /// If the previous guard has compatible resume data (same fail_args)
    /// AND the current guard has no descriptor, share the previous guard's
    /// descriptor. Otherwise, store fresh resume data.
    pub fn emit_guard_operation(&mut self, mut guard_op: Op, ctx: &mut OptContext) {
        let opcode = guard_op.opcode;

        // optimizer.py: guard_(no)_exception after non-GUARD_NOT_FORCED
        // breaks the sharing chain.
        if (opcode == OpCode::GuardNoException || opcode == OpCode::GuardException) {
            if let Some(ref last) = self.last_guard_op {
                if last.opcode != OpCode::GuardNotForced && last.opcode != OpCode::GuardNotForced2 {
                    self.last_guard_op = None;
                }
            }
        }

        // optimizer.py: try to share resume data with last guard
        if self.can_replace_guards {
            if let Some(ref last_guard) = self.last_guard_op {
                if guard_op.descr.is_none() {
                    // _copy_resume_data_from: copy descriptor and fail_args
                    guard_op.descr = last_guard.descr.clone();
                    guard_op.fail_args = last_guard.fail_args.clone();
                    // Emit without updating last_guard_op (shared)
                    ctx.emit(guard_op);
                    return;
                }
            }
        }

        // Flush pending fields before emitting a fresh guard
        let pending = std::mem::take(&mut self.pendingfields);
        for op in pending {
            ctx.emit(op);
        }

        // RPython parity: store_final_boxes_in_guard — expand fail_args
        // with virtual field values and record blueprints in rd_virtuals.
        // TODO: re-enable when cranelift backend handles extended fail_args.
        // self.store_final_boxes_in_guard(&mut guard_op, ctx);

        // Store this guard as the new sharing source
        let emitted = ctx.emit(guard_op.clone());
        guard_op.pos = emitted;
        self.last_guard_op = Some(guard_op);
    }

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

    /// optimizer.py: add_quasi_immutable_dep(descr_idx)
    /// Track a quasi-immutable field dependency.
    pub fn add_quasi_immutable_dep(&mut self, descr_idx: u32) {
        self.quasi_immutable_deps.insert(descr_idx);
    }

    /// optimizer.py: produce_potential_short_preamble_ops(sb)
    /// Collect short preamble ops from all passes.
    pub fn produce_potential_short_preamble_ops(&self, sb: &mut crate::shortpreamble::ShortBoxes) {
        for pass in &self.passes {
            pass.produce_potential_short_preamble_ops(sb);
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

    /// heap.py: deserialize_optheap — import preamble heap cache into Phase 2.
    pub fn import_all_cached_fields(&mut self, entries: &[(OpRef, u32, OpRef)]) {
        for pass in &mut self.passes {
            pass.import_cached_fields(entries);
        }
    }

    /// Flush only virtualizable lazy SetfieldGc ops from the heap pass.
    /// Called before JUMP in Phase 2 (skip_flush=true) so compiled code
    /// writes head/size to memory for guard failure recovery.
    fn flush_virtualizable_lazy_sets(&mut self, ctx: &mut OptContext) {
        for pass in &mut self.passes {
            if pass.name() == "heap" {
                pass.flush_virtualizable(ctx);
                break;
            }
        }
    }

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
    pub fn build_short_preamble(optimized_ops: &[Op]) -> crate::shortpreamble::ShortPreamble {
        crate::shortpreamble::extract_short_preamble(optimized_ops)
    }

    /// optimizer.py: send_extra_operation(op, ctx)
    /// Send an extra operation through the pass chain as if it were
    /// a new operation from the trace. Used by passes that need to
    /// inject additional operations.
    pub fn send_extra_operation(&mut self, op: &Op, ctx: &mut OptContext) {
        self.propagate_from_pass(0, op, ctx);
    }

    /// optimizer.py: force_box(opref, ctx) — force a virtual to be materialized.
    /// If the opref refers to a virtual object, emit the allocation and field writes.
    /// Returns the concrete OpRef (unchanged if not virtual).
    pub fn force_box(&mut self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        // Follow forwarding chain first.
        let resolved = ctx.get_replacement(opref);
        let preamble_source = ctx.imported_short_source(resolved);
        let tracked = ctx
            .take_potential_extra_op(resolved)
            .or_else(|| ctx.take_potential_extra_op(opref))
            .or_else(|| {
                (preamble_source != resolved && preamble_source != opref)
                    .then(|| ctx.take_potential_extra_op(preamble_source))
                    .flatten()
            });
        if let Some(tracked) = tracked {
            if let Some(builder) = ctx.active_short_preamble_producer_mut() {
                // RPython unroll.py: force_op_from_preamble() routes the
                // tracked preamble use through short_preamble_producer.use_box(...)
                // before recording it for jump replay. This matters for ovf
                // ops, where the short builder must materialize both the
                // overflowing operation and its guard together.
                let _ = builder.use_box(tracked.result);
                builder.add_tracked_preamble_op(tracked.result, &tracked.produced);
            } else if let Some(builder) = ctx.imported_short_preamble_builder.as_mut() {
                let _ = builder.use_box(tracked.result);
                builder.add_tracked_preamble_op(tracked.result, &tracked.produced);
            }
        }
        if let Some(mut info) = ctx.get_ptr_info(resolved).cloned() {
            if info.is_virtual() {
                let forced = if ctx.in_final_emission {
                    info.force_to_ops_direct(resolved, ctx)
                } else {
                    info.force_to_ops(resolved, ctx)
                };
                return ctx.get_replacement(forced);
            }
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
        let resolved = ctx.get_replacement(opref);
        let Some(mut info) = ctx.get_ptr_info(resolved).cloned() else {
            return resolved;
        };

        if matches!(info, crate::info::PtrInfo::Virtual(_)) {
            if !rec.insert(resolved) {
                return resolved;
            }
            // RPython AbstractVirtualPtrInfo._force_at_the_end_of_preamble():
            // top-level instance-like virtuals fall back to force_box().
            let forced_ref = info.force_to_ops_direct(resolved, ctx);
            if let Some(new_info) = ctx.get_ptr_info(forced_ref).cloned() {
                let mut updated = new_info;
                updated.force_at_the_end_of_preamble(|child| {
                    self.force_box_for_end_of_preamble_rec(child, ctx, rec)
                });
                ctx.set_ptr_info(forced_ref, updated);
            }
            return forced_ref;
        }

        if matches!(
            info,
            crate::info::PtrInfo::VirtualStruct(_)
                | crate::info::PtrInfo::VirtualArray(_)
                | crate::info::PtrInfo::VirtualArrayStruct(_)
                | crate::info::PtrInfo::VirtualRawBuffer(_)
        ) {
            if !rec.insert(resolved) {
                return resolved;
            }
            // RPython AbstractStructPtrInfo/ArrayPtrInfo override
            // _force_at_the_end_of_preamble() to recurse into children while
            // leaving the top-level virtual in the exported virtual state.
            info.force_at_the_end_of_preamble(|child| {
                self.force_box_for_end_of_preamble_rec(child, ctx, rec)
            });
            ctx.set_ptr_info(resolved, info);
            return resolved;
        }

        if info.is_virtual() {
            return self.force_box(resolved, ctx);
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
        ctx.forwarding.clear();
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
        let resolved = ctx.get_replacement(opref);
        if let Some(val) = ctx.get_constant_int(resolved) {
            if val != 0 { 1 } else { -1 }
        } else {
            0 // unknown
        }
    }

    /// optimizer.py: make_constant_class(op, class_const)
    /// Record that an OpRef has a known class (type pointer).
    /// This is used by GUARD_CLASS to propagate class info.
    pub fn make_constant_class(ctx: &mut OptContext, opref: OpRef, class_value: i64) {
        // In RPython this creates an InstancePtrInfo with _known_class.
        // In majit we record it as a fact the guard pass can use.
        // The class value is stored so downstream passes can check it.
        let _ = (ctx, opref, class_value);
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
    pub fn optimize(&mut self, ops: &[Op]) -> Vec<Op> {
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

        // Pre-populate known constants so passes can see them.
        for (&idx, &val) in constants.iter() {
            ctx.make_constant(OpRef(idx), value_from_backend_constant_bits(OpRef(idx), val, ops));
        }

        // Setup all passes
        for pass in &mut self.passes {
            pass.setup();
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
                                ctx.make_constant(*result_ref, majit_ir::Value::Int(*v));
                            }
                        }
                    }
                    break;
                }
            }
        }

        if let Some(exported_state) = self.imported_loop_state.as_ref() {
            // RPython unroll.py passes the actual old-label targetargs into
            // import_state(). In majit's phase-2 entry there is no concrete
            // Label op yet, so we synthesize placeholder inputargs matching
            // the exported next-iteration contract rather than blindly using
            // `num_inputs`, which can diverge after bridge/retrace rewriting.
            let targetargs: Vec<OpRef> = (0..exported_state.next_iteration_args.len())
                .map(|i| OpRef(i as u32))
                .collect();
            let (label_args, source_slots) =
                crate::unroll::import_state_with_source_slots(&targetargs, exported_state, &mut ctx);
            // short_args (label_args + virtuals) was already stored in
            // ctx.imported_virtual_args by import_state(). Don't re-compute
            // — make_inputargs_and_virtuals returns empty after forwarding.
            if crate::majit_log_enabled() {
                if let Some((base, ref sa)) = ctx.imported_virtual_args {
                    eprintln!("[jit] virtual_args from import_state: base={} total={} virtuals={:?}",
                        base, sa.len(), &sa[base..]);
                }
            }
            self.imported_label_args = Some(label_args);
            self.imported_label_source_slots = Some(source_slots);
        }

        if !self.imported_virtuals.is_empty() {
            self.install_imported_virtuals(&mut ctx);
        }

        // RPython optimizer.py:536-556: JUMP/FINISH is separated from
        // the main loop. With flush=False (Phase 2), JUMP is NOT processed
        // through the passes — it's returned as last_op for the caller.
        let mut last_op = None;
        for op in ops {
            if self.skip_flush && op.opcode == OpCode::Jump {
                // Phase 2: virtualizable lazy sets must be emitted before
                // JUMP so the compiled code writes head/size to memory.
                // RPython: flush(flush=False) still forces virtualizable fields.
                self.flush_virtualizable_lazy_sets(&mut ctx);
                last_op = Some(op.clone());
                break;
            }
            // Finish must go through passes even with skip_flush=true:
            // OptVirtualize.optimize_escaping_op forces virtual args,
            // emitting New + SetfieldGc. Without this, Finish receives
            // an empty New() with no fields set.
            if self.skip_flush && op.opcode == OpCode::Finish {
                self.propagate_one(op, &mut ctx);
                // The Finish op was replaced by propagate_one; grab it.
                if let Some(finish_op) = ctx.new_operations.iter().rev()
                    .find(|o| o.opcode == OpCode::Finish)
                {
                    last_op = Some(finish_op.clone());
                }
                break;
            }
            self.propagate_one(op, &mut ctx);
        }

        // RPython: JUMP goes through all passes BEFORE flush.
        // OptVirtualize captures pre_force_virtual_state during JUMP.
        // Flush destroys virtuals, so JUMP must run first.
        if let Some(mut terminal_op) = last_op {
            for arg in &mut terminal_op.args {
                *arg = ctx.get_replacement(*arg);
            }
            if self.skip_flush {
                self.terminal_op = Some(terminal_op);
            } else {
                self.propagate_one(&terminal_op, &mut ctx);
            }
        }
        // Phase 2 leaves lazy sets pending so virtuals aren't forced.
        if !self.skip_flush {
            self.flush(&mut ctx);
        }

        // Transfer exported virtual state from context to optimizer
        let exported_jump_virtuals = std::mem::take(&mut ctx.exported_jump_virtuals);
        self.imported_short_sources = std::mem::take(&mut ctx.imported_short_sources);
        self.imported_short_aliases = ctx.used_imported_short_aliases();
        self.imported_short_preamble = ctx.build_imported_short_preamble();
        self.imported_short_preamble_builder = ctx.imported_short_preamble_builder.clone();
        self.patchguardop = ctx.patchguardop.clone();
        let jump = ctx
            .new_operations
            .iter()
            .rfind(|op| op.opcode == OpCode::Jump)
            .cloned();
        if crate::majit_log_enabled() {
            eprintln!("[jit] export: pre_force_vs={} pre_force_args={}",
                ctx.pre_force_virtual_state.is_some(),
                ctx.pre_force_jump_args.is_some());
        }
        self.exported_loop_state = jump.map(|jump| {
            let original_jump_args = ctx
                .pre_force_jump_args
                .clone()
                .unwrap_or_else(|| jump.args.to_vec());
            ctx.preamble_end_args = Some(
                original_jump_args
                    .iter()
                    .map(|&arg| self.force_box_for_end_of_preamble(arg, &mut ctx))
                    .collect(),
            );
            // RPython parity: nested virtuals in output ops should be
            // forced by force_box_for_end_of_preamble's recursive walk.
            // Additional force is handled by the undefined-ref cleanup below
            // which drops ops referencing un-forced virtual OpRefs.
            // RPython unroll.py:454-457: use virtual state from BEFORE force.
            // OptVirtualize captures this in the JUMP handler.
            let preview_virtual_state = ctx.pre_force_virtual_state.clone().unwrap_or_else(|| {
                crate::virtualstate::export_state(&jump.args, &ctx, &ctx.ptr_info)
            });
            // Use pre-force args for make_inputargs so virtual entries can
            // look up their field values from the still-virtual PtrInfo.
            let vs_args = ctx.pre_force_jump_args.as_deref().unwrap_or(&jump.args);
            let (preview_label_args, preview_virtuals) =
                preview_virtual_state.make_inputargs_and_virtuals(vs_args, &ctx);
            let mut preview_short_args = preview_label_args.clone();
            preview_short_args.extend(preview_virtuals);
            let mut short_boxes =
                crate::shortpreamble::ShortBoxes::with_label_args(&preview_short_args);
            short_boxes.note_known_constants_from_ctx(&ctx);
            for &arg in &preview_short_args {
                short_boxes.add_short_input_arg(arg);
            }
            self.produce_potential_short_preamble_ops(&mut short_boxes);
            let produced = short_boxes.produced_ops();
            ctx.exported_short_boxes = produced
                .into_iter()
                .map(|(result, produced)| {
                    let canonical_result = ctx.get_replacement(result);
                    let mut preamble_op = produced.preamble_op;
                    preamble_op.pos = ctx.get_replacement(preamble_op.pos);
                    for arg in &mut preamble_op.args {
                        *arg = ctx.get_replacement(*arg);
                    }
                    if let Some(fail_args) = preamble_op.fail_args.as_mut() {
                        for arg in fail_args {
                            *arg = ctx.get_replacement(*arg);
                        }
                    }
                    crate::shortpreamble::PreambleOp {
                        op: preamble_op,
                        kind: produced.kind,
                        label_arg_idx: short_boxes.lookup_label_arg(canonical_result),
                        invented_name: produced.invented_name,
                        same_as_source: produced.same_as_source.map(|src| ctx.get_replacement(src)),
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
            crate::unroll::export_state(
                &original_jump_args,
                &renamed_inputargs,
                &ctx,
                Some(&exported_int_bounds),
            )
        });
        self.exported_jump_virtuals = exported_jump_virtuals;

        // RPython export_state() flushes force artifacts into the preamble
        // before building the exported loop state. If the loop header needs
        // additional inputargs, the corresponding SETFIELD/SETARRAYITEM must
        // remain in the trace rather than being silently discarded.
        while let Some(extra) = ctx.pop_extra_operation() {
            self.propagate_one(&extra, &mut ctx);
        }

        // final_num_inputs = original inputs + virtual inputs added by passes.
        let num_virtual_inputs = (ctx.num_inputs as usize).saturating_sub(effective_inputs);
        self.final_num_inputs = num_inputs + num_virtual_inputs;

        // Force any remaining virtual refs in output ops before forwarding resolve.
        // RPython: virtuals are forced during preamble export or JUMP handling.
        // In majit, skip_flush=true may leave some virtual refs un-forced in
        // the output (e.g., linked list nodes nested in non-virtual objects).
        {
            let all_refs: Vec<OpRef> = ctx.new_operations.iter()
                .flat_map(|op| op.args.iter().copied()
                    .chain(op.fail_args.iter().flat_map(|fa| fa.iter().copied())))
                .filter(|r| !r.is_none())
                .collect();
            for opref in all_refs {
                let resolved = ctx.get_replacement(opref);
                if let Some(info) = ctx.get_ptr_info(resolved) {
                    if info.is_virtual() {
                        self.force_box_for_end_of_preamble(resolved, &mut ctx);
                    }
                }
            }
            // Drain any force artifacts into the output.
            while let Some(extra) = ctx.pop_extra_operation() {
                self.propagate_one(&extra, &mut ctx);
            }
        }

        // Resolve forwarding BEFORE remap so that all refs point to concrete
        // positions that remap can then reassign.
        {
            let fwd = ctx.forwarding.clone();
            let resolve = |opref: OpRef| -> OpRef {
                let mut cur = opref;
                loop {
                    let idx = cur.0 as usize;
                    if idx >= fwd.len() { return cur; }
                    let next = fwd[idx];
                    if next.is_none() || next == cur { return cur; }
                    cur = next;
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

        // Drop ops whose args reference undefined OpRefs (un-forced virtual
        // remnants). Include imported OpRefs (from import_state forwarding and
        // short preamble) in the defined set — RPython doesn't have this
        // cleanup at all, but we keep it to catch genuinely dangling refs.
        {
            let defined: std::collections::HashSet<u32> = {
                let mut s: std::collections::HashSet<u32> = (0..self.final_num_inputs as u32).collect();
                for op in &ctx.new_operations {
                    if !op.pos.is_none() && op.result_type() != Type::Void {
                        s.insert(op.pos.0);
                    }
                }
                for (k, val) in ctx.constants.iter().enumerate() {
                    if val.is_some() {
                        s.insert(k as u32);
                    }
                }
                // Include all OpRefs reachable via forwarding (import_state
                // maps original input args to preamble OpRefs through forwarding).
                for (i, &fwd) in ctx.forwarding.iter().enumerate() {
                    if !fwd.is_none() && fwd != OpRef(i as u32) {
                        s.insert(fwd.0);
                    }
                }
                s
            };
            ctx.new_operations.retain(|op| {
                op.args.iter().all(|arg| arg.is_none() || defined.contains(&arg.0))
            });
        }

        // Drain remaining extra ops + re-check for undefined refs.
        self.drain_extra_operations_from(0, &mut ctx);
        // Extra operations can introduce new forwarding (for example Heap/Pure
        // forwarding a recently-emitted boxed-field read to its raw payload).
        // Resolve forwarding again before the second undefined-ref filter so
        // late-emitted users don't keep stale producer OpRefs.
        {
            let fwd = ctx.forwarding.clone();
            let resolve = |opref: OpRef| -> OpRef {
                let mut cur = opref;
                loop {
                    let idx = cur.0 as usize;
                    if idx >= fwd.len() {
                        return cur;
                    }
                    let next = fwd[idx];
                    if next.is_none() || next == cur {
                        return cur;
                    }
                    cur = next;
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
        {
            let defined: std::collections::HashSet<u32> = {
                let mut s: std::collections::HashSet<u32> = (0..self.final_num_inputs as u32).collect();
                for op in &ctx.new_operations {
                    if !op.pos.is_none() && op.result_type() != Type::Void {
                        s.insert(op.pos.0);
                    }
                }
                for (k, val) in ctx.constants.iter().enumerate() {
                    if val.is_some() { s.insert(k as u32); }
                }
                for (i, &fwd) in ctx.forwarding.iter().enumerate() {
                    if !fwd.is_none() && fwd != OpRef(i as u32) {
                        s.insert(fwd.0);
                    }
                }
                s
            };
            ctx.new_operations.retain(|op| {
                op.args.iter().all(|arg| arg.is_none() || defined.contains(&arg.0))
            });
        }

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
            if crate::majit_log_enabled() {
                let fwd_1995 = if 1995 < ctx.forwarding.len() { ctx.forwarding[1995] } else { OpRef::NONE };
                let in_ops = ctx.new_operations.iter().any(|o| o.pos.0 == 1995);
                let in_args = ctx.new_operations.iter().any(|o| o.args.iter().any(|a| a.0 == 1995));
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
            for entry in &mut ctx.forwarding {
                if !entry.is_none() {
                    if let Some(&new_pos) = remap.get(&entry.0) {
                        *entry = OpRef(new_pos);
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
        if crate::majit_log_enabled() {
            let cmf_count = ops.iter().filter(|o| o.opcode.is_call_may_force()).count();
            let gnf_count = ops.iter().filter(|o| matches!(o.opcode, OpCode::GuardNotForced | OpCode::GuardNotForced2)).count();
            eprintln!("[opt] final ops: total={} call_may_force={} guard_not_forced={}", ops.len(), cmf_count, gnf_count);
            if cmf_count == 0 && gnf_count > 0 {
                for (i, op) in ops.iter().enumerate() {
                    eprintln!("[opt] idx={} {:?} pos={:?}", i, op.opcode, op.pos);
                }
            }
        }
        self.final_ctx = Some(ctx);
        ops
    }

    fn collect_exported_int_bounds(
        &self,
        args: &[OpRef],
        ctx: &OptContext,
    ) -> std::collections::HashMap<OpRef, crate::intutils::IntBound> {
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

    fn drain_extra_operations_from(&mut self, start_pass: usize, ctx: &mut OptContext) {
        let end_pass = self.extra_operation_end_pass();
        let mut pending = std::collections::VecDeque::new();
        while let Some(op) = ctx.pop_extra_operation() {
            pending.push_back(op);
        }
        while let Some(op) = pending.pop_front() {
            self.propagate_from_pass_range(start_pass, end_pass, &op, ctx);
            while let Some(child) = ctx.pop_extra_operation() {
                pending.push_front(child);
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
            *arg = ctx.get_replacement(*arg);
        }
        if let Some(ref mut fa) = resolved_op.fail_args {
            for arg in fa.iter_mut() {
                *arg = ctx.get_replacement(*arg);
            }
        }

        let mut current_op = resolved_op;

        for pass_idx in start_pass..end_pass {
            let pass_name = self.passes[pass_idx].name().to_string();
            let result = {
                let pass = &mut self.passes[pass_idx];
                pass.propagate_forward(&current_op, ctx)
            };
            self.drain_extra_operations_from(pass_idx + 1, ctx);
            match result {
                OptimizationResult::Emit(op) => {
                    self.emit_with_guard_check(op, ctx);
                    return;
                }
                OptimizationResult::Replace(op) => {
                    current_op = op;
                }
                OptimizationResult::Remove => {
                    return;
                }
                OptimizationResult::PassOn => {}
            }
        }

        // If no pass handled it, emit as-is
        self.emit_with_guard_check(current_op, ctx);
    }

    /// optimizer.py: _emit_operation — emit with guard tracking.
    ///
    /// When emitting a guard, check replaces_guard to see if this guard
    /// should replace a previously emitted one (guard strengthening).
    /// Also track last_guard_op for consecutive guard descriptor sharing.
    /// RPython optimizer.py:623-625: _emit_operation calls force_box(arg)
    /// on every arg before final emission. In majit, this forces any remaining
    /// virtual args that weren't caught by pass-level handlers.
    fn emit_with_guard_check(&mut self, mut op: Op, ctx: &mut OptContext) {
        // RPython optimizer.py: emitting_operation callback — notify all passes
        // before any op is emitted. This is how OptHeap forces lazy sets before
        // guards even when the guard is emitted by an earlier pass.
        // force_to_ops_direct emits directly to new_operations, so no drain needed.
        for pass in &mut self.passes {
            pass.emitting_operation(&op, ctx);
        }

        // RPython optimizer.py:623-625: _emit_operation calls force_box on
        // every arg before emission, not just virtual ones. This is also what
        // promotes imported short-preamble producers into the loop-header
        // contract when a later guard/JUMP still references them.
        if !matches!(op.opcode,
            OpCode::SetfieldGc | OpCode::SetfieldRaw
            | OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw)
        {
            ctx.in_final_emission = true;
            for i in 0..op.num_args() {
                let arg = ctx.get_replacement(op.arg(i));
                op.args[i] = self.force_box(arg, ctx);
            }
            ctx.in_final_emission = false;
        }
        if op.opcode.is_guard() {
            // optimizer.py: store_final_boxes_in_guard — encode virtual
            // objects in fail_args as rd_virtuals instead of forcing them.
            // This runs at the optimizer level (not per-pass) so ALL guards
            // get virtual encoding regardless of which pass emits them.
            op = Self::encode_guard_virtuals(op, ctx);

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
                                        .get_constant_int(ctx.get_replacement(pf_op.arg(1)))
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
                                descr_index: pf_op
                                    .descr
                                    .as_ref()
                                    .map_or(0, |d| d.index()),
                                item_index,
                                target: ctx.get_replacement(target),
                                value: ctx.get_replacement(value),
                                field_offset,
                                field_size,
                                field_type,
                            }
                        })
                        .collect(),
                );
            }

            // optimizer.py: if orig_op in replaces_guard → replace_guard_op
            if self.can_replace_guards {
                if let Some(replacement) = self.replaces_guard.remove(&op.pos.0) {
                    let target_pos = replacement.pos.0 as usize;
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!(
                            "[opt] guard replacement op={:?} pos={:?} replacement_pos={:?} target_index={} len={}",
                            op.opcode,
                            op.pos,
                            replacement.pos,
                            target_pos,
                            ctx.new_operations.len()
                        );
                    }
                    if target_pos < ctx.new_operations.len() {
                        ctx.new_operations[target_pos] = op.clone();
                        return;
                    }
                }
            }
            self.last_guard_op = Some(op.clone());
        } else if !op.opcode.has_no_side_effect()
            && !op.opcode.is_ovf()
            && !op.opcode.is_jit_debug()
        {
            // Side-effecting ops reset last_guard_op
            self.last_guard_op = None;
        }
        let emitted = ctx.emit(op.clone());
        if std::env::var_os("MAJIT_LOG").is_some()
            && matches!(op.opcode, OpCode::CallMayForceI | OpCode::CallMayForceR | OpCode::CallMayForceF | OpCode::CallMayForceN | OpCode::GuardNotForced | OpCode::GuardNotForced2)
        {
            eprintln!(
                "[opt] emit {:?} pos={:?} len={}",
                op.opcode,
                emitted,
                ctx.new_operations.len()
            );
        }
    }

    /// optimizer.py: store_final_boxes_in_guard — encode virtual objects
    /// in guard fail_args as rd_virtuals for lazy reconstruction on guard
    /// failure. RPython does this in the optimizer's store_final_boxes_in_guard,
    /// which runs for every guard regardless of which pass emits it.
    fn encode_guard_virtuals(op: Op, ctx: &mut OptContext) -> Op {
        Self::encode_guard_virtuals_impl(op, ctx)
    }

    #[allow(dead_code)]
    fn encode_guard_virtuals_impl(mut op: Op, ctx: &mut OptContext) -> Op {
        use crate::info::PtrInfo;
        use majit_ir::GuardVirtualEntry;

        let Some(ref mut fail_args) = op.fail_args else {
            return op;
        };

        let original_len = fail_args.len();
        let mut virtual_entries: Vec<GuardVirtualEntry> = Vec::new();
        let mut extra_fail_args: Vec<OpRef> = Vec::new();

        for fa_idx in 0..original_len {
            let resolved = ctx.get_replacement(fail_args[fa_idx]);
            let info = ctx.get_ptr_info(resolved).cloned();
            let Some(info) = info else {
                fail_args[fa_idx] = resolved;
                continue;
            };
            if !info.is_virtual() || matches!(info, PtrInfo::Virtualizable(_)) {
                fail_args[fa_idx] = resolved;
                continue;
            }

            // Encode virtual metadata
            let extra_start = original_len + extra_fail_args.len();
            match &info {
                PtrInfo::Virtual(vinfo) => {
                    let mut fields = Vec::with_capacity(vinfo.fields.len());
                    for &(field_idx, value_ref) in &vinfo.fields {
                        let mut final_ref = ctx.get_replacement(value_ref);
                        // Nested virtual field values must be forced to concrete.
                        // Cranelift backend does not support multi-level rd_virtuals.
                        if let Some(nested) = ctx.get_ptr_info(final_ref).cloned() {
                            if nested.is_virtual() && !matches!(nested, crate::info::PtrInfo::Virtualizable(_)) {
                                let mut nested_mut = nested;
                                let forced = nested_mut.force_to_ops_direct(final_ref, ctx);
                                final_ref = ctx.get_replacement(forced);
                            }
                        }
                        let fa_index = extra_start + extra_fail_args.len();
                        extra_fail_args.push(final_ref);
                        let descr_idx = vinfo
                            .field_descrs
                            .iter()
                            .find(|(idx, _)| *idx == field_idx)
                            .map(|(_, d)| d.index())
                            .unwrap_or(field_idx);
                        fields.push((descr_idx, fa_index));
                    }
                    virtual_entries.push(GuardVirtualEntry {
                        fail_arg_index: fa_idx,
                        descr: vinfo.descr.clone(),
                        known_class: vinfo.known_class,
                        fields,
                    });
                    fail_args[fa_idx] = OpRef::NONE;
                }
                PtrInfo::VirtualStruct(vinfo) => {
                    let mut fields = Vec::with_capacity(vinfo.fields.len());
                    for &(field_idx, value_ref) in &vinfo.fields {
                        let mut final_ref = ctx.get_replacement(value_ref);
                        if let Some(nested) = ctx.get_ptr_info(final_ref).cloned() {
                            if nested.is_virtual() && !matches!(nested, crate::info::PtrInfo::Virtualizable(_)) {
                                let mut nested_mut = nested;
                                let forced = nested_mut.force_to_ops_direct(final_ref, ctx);
                                final_ref = ctx.get_replacement(forced);
                            }
                        }
                        let fa_index = extra_start + extra_fail_args.len();
                        extra_fail_args.push(final_ref);
                        let descr_idx = vinfo
                            .field_descrs
                            .iter()
                            .find(|(idx, _)| *idx == field_idx)
                            .map(|(_, d)| d.index())
                            .unwrap_or(field_idx);
                        fields.push((descr_idx, fa_index));
                    }
                    virtual_entries.push(GuardVirtualEntry {
                        fail_arg_index: fa_idx,
                        descr: vinfo.descr.clone(),
                        known_class: None,
                        fields,
                    });
                    fail_args[fa_idx] = OpRef::NONE;
                }
                _ => {
                    // Unsupported virtual kind — leave as-is (will be forced
                    // by virtualize pass or emitted as concrete)
                    fail_args[fa_idx] = resolved;
                }
            }
        }

        if !extra_fail_args.is_empty() {
            // RPython parity: store_final_boxes_in_guard creates a new
            // ResumeGuardDescr with updated fail_arg_types matching the
            // expanded fail_args. We must update the descriptor's types
            // to include the extra field values.
            if let Some(ref descr) = op.descr {
                if let Some(fd) = descr.as_fail_descr() {
                    let mut types = fd.fail_arg_types().to_vec();
                    // Extra fail_args are field values — all Int for now
                    // (W_Int fields are vtable ptr and int value).
                    for _ in 0..extra_fail_args.len() {
                        types.push(majit_ir::Type::Int);
                    }
                    let new_descr =
                        majit_ir::SimpleFailDescr::new(descr.index(), fd.fail_index(), types);
                    op.descr = Some(std::sync::Arc::new(new_descr));
                }
            }
            for extra in extra_fail_args {
                fail_args.push(extra);
            }
        }

        if !virtual_entries.is_empty() {
            op.rd_virtuals = Some(virtual_entries);
        }
        op
    }
}

impl Optimizer {
    /// Create an optimizer with the standard pass pipeline.
    /// Order: IntBounds -> Rewrite -> Virtualize -> String -> Pure -> Guard -> Simplify -> Heap
    pub fn default_pipeline() -> Self {
        let mut opt = Self::new();
        opt.add_pass(Box::new(OptIntBounds::new()));
        opt.add_pass(Box::new(OptRewrite::new()));
        opt.add_pass(Box::new(OptVirtualize::new()));
        opt.add_pass(Box::new(OptString::new()));
        opt.add_pass(Box::new(OptPure::new()));
        opt.add_pass(Box::new(GuardStrengthenOpt::new()));
        opt.add_pass(Box::new(OptSimplify::new()));
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
        opt.add_pass(Box::new(GuardStrengthenOpt::new()));
        opt.add_pass(Box::new(OptSimplify::new()));
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

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::descr::{CallDescr, EffectInfo, ExtraEffect, OopSpecIndex, make_fail_descr};
    use majit_ir::Type;
    use majit_ir::descr::make_size_descr;
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

                let alloc = ctx.emit_through_passes(Op::new(OpCode::New, &[]));
                let mut set = Op::new(OpCode::SetfieldGc, &[alloc, OpRef(0)]);
                set.descr = Some(self.field_descr.clone());
                ctx.emit_through_passes(set);
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
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 1024);
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
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            3,
        );

        let call_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::CallMayForceR)
            .count();
        let guard_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::GuardNotForced)
            .count();
        assert_eq!(call_count, 2, "optimized trace lost CallMayForceR ops: {result:?}");
        assert_eq!(guard_count, 2, "optimized trace lost GuardNotForced ops: {result:?}");
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
        guard_a.fail_args = Some(vec![
            OpRef(0),
            OpRef(2000),
            OpRef(2001),
            OpRef(3),
            OpRef(3000),
            OpRef(3001),
            OpRef(4),
        ].into());
        let get_a_type = Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(3)], field_descr.clone());
        let get_a_val = Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(3)], field_descr.clone());
        let mut call_b = Op::with_descr(OpCode::CallMayForceR, &[OpRef(0), OpRef(2)], call_descr_b);
        let mut guard_b = Op::with_descr(OpCode::GuardNotForced, &[], guard_descr_b);
        guard_b.fail_args = Some(vec![
            OpRef(0),
            OpRef(2002),
            OpRef(2003),
            OpRef(3),
            OpRef(6),
            OpRef(3002),
            OpRef(3003),
            OpRef(7),
        ].into());
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
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            3,
        );

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
        assert_eq!(guarded, 2, "optimized trace lost GuardNotForced ops: {result:?}");
    }

    #[test]
    fn test_default_pipeline_has_8_passes() {
        let opt = Optimizer::default_pipeline();
        assert_eq!(opt.num_passes(), 8);
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
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 1024);
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
        let ctx = opt.final_ctx.as_ref().expect("optimizer should preserve final ctx");

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
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 1024);
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
        opt.add_pass(Box::new(crate::unroll::OptUnroll::new()));

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(1)]),
        ];
        ops[0].pos = OpRef(2);
        ops[1].pos = OpRef(3);

        let mut constants = std::collections::HashMap::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        // RPython parity: force-like extra ops emit New, but the
        // SetfieldGc goes through heap pass as lazy_set and is dropped
        // at JUMP (emit_extra → re-absorbed → lost).
        let new_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::New)
            .count();
        assert!(
            new_count > 0,
            "force-like extra ops should still emit a New; got {:?}",
            result
        );
        // SetfieldGc is absorbed by heap cache and dropped at JUMP
        let setfield_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::SetfieldGc)
            .count();
        assert_eq!(
            setfield_count, 0,
            "lazy SetfieldGc should be dropped at JUMP (RPython parity); got {:?}",
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

                let alloc_a = ctx.emit_through_passes(Op::new(OpCode::New, &[]));
                let mut set_a = Op::new(OpCode::SetfieldGc, &[alloc_a, OpRef(0)]);
                set_a.descr = Some(self.field_descr.clone());
                ctx.emit_through_passes(set_a);

                let alloc_b = ctx.emit_through_passes(Op::new(OpCode::New, &[]));
                let mut set_b = Op::new(OpCode::SetfieldGc, &[alloc_b, OpRef(1)]);
                set_b.descr = Some(self.field_descr.clone());
                ctx.emit_through_passes(set_b);
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
    fn test_optimizer_encodes_direct_virtual_guard_fail_args_as_rd_virtuals() {
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

        let result = opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 1024);
        let guard = result
            .iter()
            .find(|op| op.opcode == OpCode::GuardTrue)
            .expect("guard should survive optimization");
        let rd_virtuals = guard
            .rd_virtuals
            .as_ref()
            .expect("direct virtual fail arg should be encoded as rd_virtuals");
        let fail_args = guard
            .fail_args
            .as_ref()
            .expect("guard should keep fail args");

        assert_eq!(rd_virtuals.len(), 1);
        assert!(
            fail_args[0].is_none(),
            "virtual fail arg slot should be replaced"
        );
        assert_eq!(
            fail_args.len(),
            2,
            "field value should be appended as extra fail arg"
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
        use crate::info::{PtrInfo, VirtualStructInfo};

        let descr = make_size_descr(16);
        let mut ctx = OptContext::with_num_inputs(32, 1024);
        ctx.set_ptr_info(
            OpRef(10),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: descr.clone(),
                fields: vec![(1, OpRef(11))],
                field_descrs: Vec::new(),
            }),
        );
        ctx.replace_op(OpRef(11), OpRef(20));
        ctx.set_ptr_info(
            OpRef(20),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr,
                fields: Vec::new(),
                field_descrs: Vec::new(),
            }),
        );

        let mut opt = Optimizer::new();
        let result = opt.force_box_for_end_of_preamble(OpRef(10), &mut ctx);

        // The virtual is forced to a concrete allocation; the returned ref
        // is the allocation's position, which ctx.get_replacement(OpRef(10))
        // should resolve to.
        assert_eq!(result, ctx.get_replacement(OpRef(10)));
        // After forcing, the struct's ptr_info reflects that field 1
        // (originally OpRef(11), forwarded to OpRef(20)) has been recursively forced.
        match ctx.get_ptr_info(result) {
            Some(PtrInfo::VirtualStruct(info)) => {
                // The inner virtual (OpRef(20)) was also forced; its allocation
                // ref is whatever force_to_ops assigned.
                assert_eq!(info.fields.len(), 1);
                assert_eq!(info.fields[0].0, 1);
            }
            // After full forcing the info might become NonNull or similar
            Some(PtrInfo::NonNull) => {}
            other => panic!("expected virtual struct or non-null after forcing, got {other:?}"),
        }
    }

    #[test]
    fn test_force_box_materializes_virtual_struct_outside_final_emission() {
        use crate::info::{PtrInfo, VirtualStructInfo};

        let descr = make_size_descr(16);
        let field_descr = crate::virtualize::make_field_index_descr(1);
        let mut ctx = OptContext::new(16);
        ctx.set_ptr_info(
            OpRef(10),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: descr.clone(),
                fields: vec![(1, OpRef(11))],
                field_descrs: vec![(1, field_descr.clone())],
            }),
        );

        let mut opt = Optimizer::new();
        let forced = opt.force_box(OpRef(10), &mut ctx);
        assert_ne!(forced, OpRef(10));
        assert!(ctx.has_extra_operations());

        opt.drain_extra_operations_from(0, &mut ctx);

        assert!(ctx
            .new_operations
            .iter()
            .any(|op| op.opcode == OpCode::New && op.pos == forced));
        assert!(ctx.new_operations.iter().any(|op| {
            op.opcode == OpCode::SetfieldGc
                && op.arg(0) == forced
                && op.arg(1) == OpRef(11)
                && op.descr.is_some()
        }));
    }

    #[test]
    fn test_emit_with_guard_check_materializes_virtual_args_directly() {
        use crate::info::{PtrInfo, VirtualStructInfo};

        let descr = make_size_descr(16);
        let field_descr = crate::virtualize::make_field_index_descr(1);
        let mut ctx = OptContext::with_num_inputs(16, 1);
        ctx.set_ptr_info(
            OpRef(10),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: descr.clone(),
                fields: vec![(1, OpRef(11))],
                field_descrs: vec![(1, field_descr.clone())],
            }),
        );

        let mut opt = Optimizer::new();
        let op = Op::new(OpCode::GuardNonnull, &[OpRef(10)]);
        opt.emit_with_guard_check(op, &mut ctx);

        assert!(!ctx.in_final_emission);
        assert!(!ctx.has_extra_operations());
        assert!(ctx
            .new_operations
            .iter()
            .any(|op| op.opcode == OpCode::New));
        assert!(ctx.new_operations.iter().any(|op| {
            op.opcode == OpCode::SetfieldGc
                && op.arg(1) == OpRef(11)
                && op.descr.is_some()
        }));
        assert!(ctx
            .new_operations
            .iter()
            .any(|op| op.opcode == OpCode::GuardNonnull && op.arg(0) != OpRef(10)));
    }

    #[test]
    fn test_emit_with_guard_check_forces_imported_short_guard_args() {
        let mut opt = Optimizer::new();
        let mut ctx = OptContext::with_num_inputs(16, 1);

        let mut preamble_op = Op::new(OpCode::IntGe, &[OpRef(3), OpRef(10_000)]);
        preamble_op.pos = OpRef(14);
        ctx.make_constant(OpRef(10_000), majit_ir::Value::Int(0));
        ctx.initialize_imported_short_preamble_builder(
            &[OpRef(0)],
            &[OpRef(0)],
            &[crate::shortpreamble::PreambleOp {
                op: preamble_op.clone(),
                kind: crate::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            }],
        );
        ctx.potential_extra_ops.insert(
            OpRef(14),
            crate::TrackedPreambleUse {
                result: OpRef(14),
                produced: crate::shortpreamble::ProducedShortOp {
                    kind: crate::shortpreamble::PreambleOpKind::Pure,
                    preamble_op: preamble_op.clone(),
                    invented_name: false,
                    same_as_source: None,
                },
            },
        );

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(14)]);
        guard.pos = OpRef(15);
        opt.emit_with_guard_check(guard, &mut ctx);

        let sp = ctx
            .build_imported_short_preamble()
            .expect("forcing imported short guard arg should build short preamble");
        assert_eq!(sp.used_boxes, vec![OpRef(14)]);
        assert_eq!(sp.jump_args, vec![OpRef(14)]);
    }
}
