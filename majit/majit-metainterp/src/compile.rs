//! Trace compilation helpers.
//!
//! Mirrors RPython's `compile.py`: guard metadata building, exit layout
//! management, backend layout merging, and trace post-processing (unboxing).

use std::collections::HashMap;

use majit_backend::{
    CompiledTraceInfo, ExitFrameLayout, ExitRecoveryLayout, FailDescrLayout, TerminalExitLayout,
};
use majit_ir::{GcRef, InputArg, Op, OpCode, OpRef, Type, Value};

use crate::blackhole::ExceptionState;
use crate::pyjitpl::{CompiledTrace, StoredExitLayout, StoredResumeData};
use crate::resume::{
    ResumeDataLoopMemo, ResumeDataVirtualAdder, ResumeFrameLayoutSummary, ResumeLayoutSummary,
    ResumeValueSource,
};

/// Resolve the type of an OpRef in guard fail_args.
/// OpRef::NONE is a virtual slot placeholder (null GC ref).
fn fail_arg_type(
    opref: &OpRef,
    value_types: &HashMap<u32, Type>,
    constant_types: &HashMap<u32, Type>,
) -> Type {
    if *opref == OpRef::NONE {
        Type::Ref
    } else if opref.is_constant() {
        // RPython Box.type parity: constant type comes from constant_types
        // (set by optimizer), then value_types, then default Int for
        // numeric constants. Ref constants (GcRef/None) must appear in
        // constant_types to avoid being mis-tagged as Int.
        constant_types
            .get(&opref.0)
            .or_else(|| value_types.get(&opref.0))
            .copied()
            .unwrap_or(Type::Int)
    } else {
        value_types.get(&opref.0).copied().unwrap_or(Type::Ref)
    }
}

/// Derive slot_types from ExitValueSourceLayout + exit_types.
/// RPython resume.py:1410 load_next_value_of_type parity:
/// slot type is the DECLARED type of the variable.
/// ExitValue(idx) → exit_types[idx]; Constant → Int; others → Ref.
fn derive_slot_types(
    slots: &[majit_backend::ExitValueSourceLayout],
    exit_types: &[Type],
) -> Vec<Type> {
    slots
        .iter()
        .map(|slot| match slot {
            majit_backend::ExitValueSourceLayout::ExitValue(idx) => {
                exit_types.get(*idx).copied().unwrap_or(Type::Ref)
            }
            majit_backend::ExitValueSourceLayout::Constant(_) => Type::Int,
            _ => Type::Ref,
        })
        .collect()
}

// ── Compilation result types (compile.py) ───────────────────────────────

/// Static exit metadata for a compiled guard or finish point.
#[derive(Debug, Clone)]
pub struct CompiledExitLayout {
    /// compile.py:186 rd_loop_token: the green_key of the compiled loop
    /// that owns this guard. Used by handle_fail to find the owning
    /// compiled entry without scanning all entries.
    pub rd_loop_token: u64,
    pub trace_id: u64,
    pub fail_index: u32,
    pub source_op_index: Option<usize>,
    pub exit_types: Vec<Type>,
    pub is_finish: bool,
    pub gc_ref_slots: Vec<usize>,
    pub force_token_slots: Vec<usize>,
    pub recovery_layout: Option<ExitRecoveryLayout>,
    pub resume_layout: Option<ResumeLayoutSummary>,
    /// resume.py:450 — compact resume numbering for this guard.
    pub rd_numb: Option<Vec<u8>>,
    /// resume.py:451 — shared constant pool.
    pub rd_consts: Option<Vec<(i64, Type)>>,
    /// resume.py:488 — virtual object blueprints.
    pub rd_virtuals: Option<Vec<majit_ir::RdVirtualInfo>>,
    /// resume.py:858 rd_pendingfields — deferred heap writes.
    pub rd_pendingfields: Option<Vec<majit_ir::GuardPendingFieldEntry>>,
}

/// Typed result from running compiled code.
pub struct CompileResult<'a, M> {
    pub values: Vec<i64>,
    pub typed_values: Vec<Value>,
    pub meta: &'a M,
    pub fail_index: u32,
    pub trace_id: u64,
    pub is_finish: bool,
    pub exit_layout: CompiledExitLayout,
    pub savedata: Option<GcRef>,
    pub exception: ExceptionState,
    /// compile.py:741-745: ResumeGuardDescr.status read at guard failure.
    pub status: u64,
    /// compile.py:780: current_object_addr_as_int(self) — descriptor identity
    /// for GUARD_VALUE per-value hash. Raw pointer to the failed FailDescr.
    pub descr_addr: usize,
}

/// Raw (lightweight) result from running compiled code.
pub struct RawCompileResult<'a, M> {
    pub values: Vec<i64>,
    pub typed_values: Vec<Value>,
    pub meta: &'a M,
    pub fail_index: u32,
    pub trace_id: u64,
    pub is_finish: bool,
    pub exit_layout: CompiledExitLayout,
    pub savedata: Option<GcRef>,
    pub exception: ExceptionState,
    /// compile.py:741-745: ResumeGuardDescr.status read at guard failure.
    pub status: u64,
    /// compile.py:780: current_object_addr_as_int(self) — descriptor identity.
    pub descr_addr: usize,
}

/// Terminal exit layout for a FINISH or JUMP op.
#[derive(Debug, Clone)]
pub struct CompiledTerminalExitLayout {
    pub op_index: usize,
    pub exit_layout: CompiledExitLayout,
}

/// Full trace compilation layout with all exits.
#[derive(Debug, Clone)]
pub struct CompiledTraceLayout {
    pub trace_id: u64,
    pub exit_layouts: Vec<CompiledExitLayout>,
    pub terminal_exit_layouts: Vec<CompiledTerminalExitLayout>,
}

/// Artifacts extracted from a backend DeadFrame.
#[derive(Debug, Clone)]
pub struct DeadFrameArtifacts {
    pub values: Vec<i64>,
    pub typed_values: Vec<Value>,
    pub exit_layout: CompiledExitLayout,
    pub savedata: Option<GcRef>,
    pub exception: ExceptionState,
}

// ── Compilation helper functions ────────────────────────────────────────

/// Build guard metadata for a compiled trace.
///
/// The backend numbers every guard and finish in a single exit table, so this
/// helper mirrors that numbering and records only the guard entries that need
/// resume data plus the corresponding op index for blackhole fallback.
/// resume.py ResumeDataLoopMemo parity: `constant_types` maps OpRef.0 → Type.
/// Used by `getconst` to encode virtual field constants as TAGINT/TAGCONST
/// instead of TAGBOX.
pub(crate) fn build_guard_metadata(
    inputargs: &[InputArg],
    ops: &[majit_ir::Op],
    pc: u64,
    constant_types: &std::collections::HashMap<u32, Type>,
) -> (
    HashMap<u32, StoredResumeData>,
    HashMap<u32, usize>,
    HashMap<u32, StoredExitLayout>,
) {
    let mut result = HashMap::new();
    let mut guard_op_indices = HashMap::new();
    let mut exit_layouts = HashMap::new();
    let mut fail_index = 0u32;
    let mut resume_memo = ResumeDataLoopMemo::new();
    let mut value_types: HashMap<u32, Type> =
        inputargs.iter().map(|arg| (arg.index, arg.tp)).collect();
    // Merge constant types so fail_arg_type can resolve constant OpRefs.
    for (&idx, &tp) in constant_types {
        value_types.entry(idx).or_insert(tp);
    }

    for (op_idx, op) in ops.iter().enumerate() {
        if !op.pos.is_none() && op.result_type() != Type::Void {
            value_types.insert(op.pos.0, op.result_type());
        }

        let is_guard = op.opcode.is_guard();
        let is_finish = op.opcode == OpCode::Finish;
        if !is_guard && !is_finish {
            continue;
        }

        if is_guard {
            guard_op_indices.insert(fail_index, op_idx);
        }

        // RPython Box.type parity: exit_types from fail_args (liveboxes).
        // Prefer fail_arg_types when it already matches fail_args length.
        // Otherwise reconstruct per-arg types from the current OpRefs,
        // consulting constant_types for constant Ref boxes.
        let exit_types: Vec<Type> = if is_finish {
            op.args
                .iter()
                .map(|opref| value_types.get(&opref.0).copied().unwrap_or(Type::Int))
                .collect()
        } else if let Some(ref fail_args) = op.fail_args {
            // Derive exit_types from value_types (post-unbox) first,
            // falling back to fail_arg_types then fail_arg_type.
            let fa_types = op.fail_arg_types.as_ref();
            fail_args
                .iter()
                .enumerate()
                .map(|(i, opref)| {
                    if let Some(&tp) = value_types.get(&opref.0) {
                        return tp;
                    }
                    if let Some(types) = fa_types {
                        if let Some(&tp) = types.get(i) {
                            return tp;
                        }
                    }
                    fail_arg_type(opref, &value_types, constant_types)
                })
                .collect()
        } else if let Some(ref types) = op.fail_arg_types {
            types.clone()
        } else {
            inputargs.iter().map(|arg| arg.tp).collect()
        };
        let resume_layout;
        if is_guard {
            let mut builder = ResumeDataVirtualAdder::new();

            // store_final_boxes parity: when rd_numb is present, fail_args
            // are normalized to liveboxes only (no constants/virtuals).
            // Build resume_layout from rd_numb so that TAGCONST/TAGINT
            // slots produce Constant entries in the reconstructed state.
            // Multi-frame: push_frame per frame with correct pc.
            if let (Some(rd_numb_bytes), Some(rd_consts_data)) = (&op.rd_numb, &op.rd_consts) {
                use majit_ir::resumedata::{RebuiltValue, rebuild_from_numbering};
                let fvc = majit_ir::resumedata::get_frame_value_count_fn();
                let fvc_ref: Option<&dyn Fn(i32, i32) -> usize> =
                    fvc.as_ref().map(|f| f as &dyn Fn(i32, i32) -> usize);
                let (_num_failargs, vable_values, _vref_values, frames) =
                    rebuild_from_numbering(rd_numb_bytes, rd_consts_data, &exit_types, fvc_ref);
                let vable_array = vable_values
                    .iter()
                    .map(|val| match val {
                        RebuiltValue::Box(idx, _) => ResumeValueSource::FailArg(*idx),
                        RebuiltValue::Const(c, _tp) => ResumeValueSource::Constant(*c),
                        RebuiltValue::Int(i) => ResumeValueSource::Constant(*i as i64),
                        RebuiltValue::Virtual(vidx) => ResumeValueSource::Virtual(*vidx),
                        RebuiltValue::Unassigned => ResumeValueSource::Unavailable,
                    })
                    .collect::<Vec<_>>();
                builder.set_vable_array(vable_array);
                let add_slot =
                    |builder: &mut ResumeDataVirtualAdder, slot_idx: usize, val: &RebuiltValue| {
                        match val {
                            RebuiltValue::Box(idx, _) => {
                                builder.map_slot(slot_idx, *idx);
                            }
                            RebuiltValue::Const(c, _tp) => {
                                builder.set_slot_constant(slot_idx, *c);
                            }
                            RebuiltValue::Int(i) => {
                                builder.set_slot_constant(slot_idx, *i as i64);
                            }
                            RebuiltValue::Virtual(vidx) => {
                                builder.set_slot_virtual(slot_idx, *vidx);
                            }
                            RebuiltValue::Unassigned => {
                                builder.set_slot_uninitialized(slot_idx);
                            }
                        }
                    };
                // rd_numb encodes [callee(top), caller(parent)] order.
                // resume_layout expects [outer, ..., innermost] order.
                //
                // RPython resume.py keeps vable_array/vref_array/framestack
                // as separate sections. Do not merge vable_array entries into
                // the innermost frame slots here.
                for frame in frames.iter().rev() {
                    builder.push_frame(frame.jitcode_index, frame.pc as u64);
                    let mut slot_idx = 0usize;
                    for val in &frame.values {
                        add_slot(&mut builder, slot_idx, val);
                        slot_idx += 1;
                    }
                }
            } else {
                // No rd_numb: single frame, 1:1 mapping (fail_args[i] → state[i]).
                builder.push_frame(0, pc);
                let num_slots = op
                    .fail_args
                    .as_ref()
                    .map(|fa| fa.len())
                    .unwrap_or(exit_types.len());
                for slot_idx in 0..num_slots {
                    builder.map_slot(slot_idx, slot_idx);
                }
            }

            let stored = StoredResumeData::with_loop_memo(builder.build(), &mut resume_memo);
            resume_layout = Some(stored.layout.clone());
            result.insert(fail_index, stored);
        } else {
            resume_layout = None;
        }

        // Store rd_numb/rd_consts/rd_virtuals/rd_pendingfields for guard failure recovery.
        let rd_numb = op.rd_numb.clone();
        let rd_consts = op.rd_consts.clone();
        let rd_virtuals = op.rd_virtuals.clone();
        let rd_pendingfields = op.rd_pendingfields.clone();
        let recovery_layout = if op.rd_numb.is_some() {
            // Consumer switchover path: rd_numb contains the full frame encoding.
            // Build recovery_layout from rd_numb + rd_virtuals.
            use majit_backend::{ExitRecoveryLayout, ExitValueSourceLayout};
            let (num_failargs, vable_layout, vref_layout, frames_layout) =
                if let (Some(rd_numb_bytes), Some(rd_consts_data)) = (&op.rd_numb, &op.rd_consts) {
                    use majit_ir::resumedata::{RebuiltValue, rebuild_from_numbering};
                    let fvc = majit_ir::resumedata::get_frame_value_count_fn();
                    let fvc_ref: Option<&dyn Fn(i32, i32) -> usize> =
                        fvc.as_ref().map(|f| f as &dyn Fn(i32, i32) -> usize);
                    let (num_failargs, vable_values, vref_values, frames) =
                        rebuild_from_numbering(rd_numb_bytes, rd_consts_data, &exit_types, fvc_ref);
                    debug_assert!(
                        vref_values.len() & 1 == 0,
                        "vref_values length must be even, got {}",
                        vref_values.len(),
                    );
                    let to_exit_source = |val: &RebuiltValue| match val {
                        RebuiltValue::Box(idx, _) => ExitValueSourceLayout::ExitValue(*idx),
                        RebuiltValue::Virtual(vidx) => ExitValueSourceLayout::Virtual(*vidx),
                        RebuiltValue::Const(c, _tp) => ExitValueSourceLayout::Constant(*c),
                        RebuiltValue::Int(i) => ExitValueSourceLayout::Constant(*i as i64),
                        RebuiltValue::Unassigned => ExitValueSourceLayout::Uninitialized,
                    };
                    (
                        num_failargs,
                        vable_values.iter().map(to_exit_source).collect::<Vec<_>>(),
                        vref_values.iter().map(to_exit_source).collect::<Vec<_>>(),
                        frames
                            .iter()
                            .enumerate()
                            .rev()
                            .map(|(_orig_idx, frame)| {
                                let mut slots = Vec::new();
                                slots.extend(frame.values.iter().map(to_exit_source));
                                let slot_types = derive_slot_types(&slots, &exit_types);
                                majit_backend::ExitFrameLayout {
                                    trace_id: None,
                                    header_pc: Some(frame.pc as u64),
                                    source_guard: None,
                                    pc: frame.pc as u64,
                                    slots,
                                    slot_types: Some(slot_types),
                                }
                            })
                            .collect::<Vec<_>>(),
                    )
                } else {
                    (exit_types.len() as i32, vec![], vec![], vec![])
                };
            // Collect slots from ALL frames for virtual target_slot lookup.
            // RPython resolves virtuals across the entire frame stack, not
            // just the innermost frame (resume.py:1410).
            let frame_slots: Vec<ExitValueSourceLayout> = frames_layout
                .iter()
                .flat_map(|frame| frame.slots.iter().cloned())
                .collect();
            // resume.py:576-860 parity: resolve fieldnums tags for recovery.
            let rd_consts_ref = op.rd_consts.as_deref().unwrap_or(&[]);
            let resolve_tagged_source = |tagged: i16| -> ExitValueSourceLayout {
                let (val, tagbits) = majit_ir::resumedata::untag(tagged);
                match tagbits {
                    majit_ir::resumedata::TAGBOX => {
                        let idx = if val >= 0 {
                            val as usize
                        } else {
                            (num_failargs + val) as usize
                        };
                        ExitValueSourceLayout::ExitValue(idx)
                    }
                    majit_ir::resumedata::TAGVIRTUAL => {
                        ExitValueSourceLayout::Virtual(val as usize)
                    }
                    majit_ir::resumedata::TAGINT => ExitValueSourceLayout::Constant(val as i64),
                    majit_ir::resumedata::TAGCONST => {
                        let idx = (val - majit_ir::resumedata::TAG_CONST_OFFSET) as usize;
                        let c = rd_consts_ref.get(idx).map(|(v, _)| *v).unwrap_or(0);
                        ExitValueSourceLayout::Constant(c)
                    }
                    _ => ExitValueSourceLayout::Constant(0),
                }
            };
            let resolve_fieldnums = |fieldnums: &[i16],
                                     fielddescr_indices: &[u32]|
             -> Vec<(u32, ExitValueSourceLayout)> {
                fieldnums
                    .iter()
                    .enumerate()
                    .map(|(fi, &fnum)| {
                        let fdi = fielddescr_indices.get(fi).copied().unwrap_or(fi as u32);
                        (fdi, resolve_tagged_source(fnum))
                    })
                    .collect()
            };
            let virtual_layouts: Vec<majit_backend::ExitVirtualLayout> = op
                .rd_virtuals
                .as_ref()
                .map(|entries| {
                    entries
                        .iter()
                        .enumerate()
                        .map(|(vidx, entry)| {
                            let target_slot = frame_slots.iter().position(
                                |s| matches!(s, ExitValueSourceLayout::Virtual(v) if *v == vidx),
                            );
                            match entry {
                                majit_ir::RdVirtualInfo::VirtualInfo {
                                    descr,
                                    type_id,
                                    descr_index,
                                    known_class,
                                    fielddescrs,
                                    fieldnums,
                                    descr_size,
                                } => {
                                    let idx: Vec<u32> =
                                        fielddescrs.iter().map(|fd| fd.index).collect();
                                    majit_backend::ExitVirtualLayout::Object {
                                        descr: descr.clone(),
                                        type_id: *type_id,
                                        descr_index: *descr_index,
                                        known_class: *known_class,
                                        fields: resolve_fieldnums(fieldnums, &idx),
                                        target_slot,
                                        fielddescrs: fielddescrs.clone(),
                                        descr_size: *descr_size,
                                    }
                                }
                                majit_ir::RdVirtualInfo::VStructInfo {
                                    typedescr,
                                    type_id,
                                    descr_index,
                                    fielddescrs,
                                    fieldnums,
                                    descr_size,
                                } => {
                                    let idx: Vec<u32> =
                                        fielddescrs.iter().map(|fd| fd.index).collect();
                                    majit_backend::ExitVirtualLayout::Struct {
                                        typedescr: typedescr.clone(),
                                        type_id: *type_id,
                                        descr_index: *descr_index,
                                        fields: resolve_fieldnums(fieldnums, &idx),
                                        target_slot,
                                        fielddescrs: fielddescrs.clone(),
                                        descr_size: *descr_size,
                                    }
                                }
                                majit_ir::RdVirtualInfo::VArrayInfoClear {
                                    arraydescr: _,
                                    descr_index,
                                    kind,
                                    fieldnums,
                                }
                                | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
                                    arraydescr: _,
                                    descr_index,
                                    kind,
                                    fieldnums,
                                } => {
                                    let clear = matches!(
                                        entry,
                                        majit_ir::RdVirtualInfo::VArrayInfoClear { .. }
                                    );
                                    let items = fieldnums
                                        .iter()
                                        .map(|&fnum| resolve_tagged_source(fnum))
                                        .collect();
                                    majit_backend::ExitVirtualLayout::Array {
                                        descr_index: *descr_index,
                                        clear,
                                        kind: *kind,
                                        items,
                                    }
                                }
                                majit_ir::RdVirtualInfo::VArrayStructInfo {
                                    arraydescr,
                                    descr_index,
                                    fielddescrs,
                                    size,
                                    fielddescr_indices,
                                    fieldnums,
                                    ..
                                } => {
                                    let fpe = if *size > 0 {
                                        fieldnums.len() / *size
                                    } else {
                                        0
                                    };
                                    let element_fields = (0..*size)
                                        .map(|ei| {
                                            let s = ei * fpe;
                                            let e = (s + fpe).min(fieldnums.len());
                                            resolve_fieldnums(&fieldnums[s..e], fielddescr_indices)
                                        })
                                        .collect();
                                    majit_backend::ExitVirtualLayout::ArrayStruct {
                                        descr_index: *descr_index,
                                        arraydescr: arraydescr.clone(),
                                        fielddescrs: fielddescrs.clone(),
                                        element_fields,
                                    }
                                }
                                majit_ir::RdVirtualInfo::VRawBufferInfo {
                                    func,
                                    size,
                                    offsets,
                                    descrs,
                                    fieldnums,
                                } => {
                                    let values = fieldnums
                                        .iter()
                                        .map(|&fnum| resolve_tagged_source(fnum))
                                        .collect();
                                    majit_backend::ExitVirtualLayout::RawBuffer {
                                        func: *func,
                                        size: *size,
                                        offsets: offsets.clone(),
                                        descrs: descrs.clone(),
                                        values,
                                    }
                                }
                                majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
                                    // resume.py:717: VRawSliceInfo — base_buffer + offset.
                                    let base = fieldnums
                                        .first()
                                        .map(|&fnum| resolve_tagged_source(fnum))
                                        .unwrap_or(ExitValueSourceLayout::Constant(0));
                                    majit_backend::ExitVirtualLayout::RawSlice {
                                        offset: *offset,
                                        base,
                                    }
                                }
                                majit_ir::RdVirtualInfo::Empty => {
                                    panic!("[jit] rd_virtuals[{vidx}] is Empty");
                                }
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            // resume.py:926,993: rd_pendingfields → pending_field_layouts.
            let pending_field_layouts: Vec<majit_backend::ExitPendingFieldLayout> = op
                .rd_pendingfields
                .as_ref()
                .map(|entries| {
                    entries
                        .iter()
                        .map(|pf| {
                            // field_offset is precomputed for array items
                            // (base_size + item_index * item_size), so do NOT
                            // pass item_index to the consumer — it would
                            // double-count the index. Consumer uses plain
                            // target_ptr + field_offset for both struct and
                            // array paths.
                            majit_backend::ExitPendingFieldLayout {
                                descr_index: pf.descr_index,
                                item_index: None,
                                is_array_item: false,
                                target: resolve_tagged_source(pf.target_tagged),
                                value: resolve_tagged_source(pf.value_tagged),
                                field_offset: pf.field_offset,
                                field_size: pf.field_size,
                                field_type: pf.field_type,
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            Some(ExitRecoveryLayout {
                vable_array: vable_layout,
                vref_array: vref_layout,
                frames: frames_layout,
                virtual_layouts,
                pending_field_layouts,
            })
        } else {
            // No rd_numb: identity recovery layout.
            // Every guard has at minimum an identity mapping from
            // fail_args → frame slots, with exit_types as slot_types.
            let slots: Vec<majit_backend::ExitValueSourceLayout> = (0..exit_types.len())
                .map(majit_backend::ExitValueSourceLayout::ExitValue)
                .collect();
            Some(ExitRecoveryLayout {
                vable_array: vec![],
                vref_array: vec![],
                frames: vec![majit_backend::ExitFrameLayout {
                    trace_id: None,
                    header_pc: Some(pc),
                    source_guard: None,
                    pc,
                    slots,
                    slot_types: Some(exit_types.clone()),
                }],
                virtual_layouts: vec![],
                pending_field_layouts: vec![],
            })
        };

        exit_layouts.insert(
            fail_index,
            StoredExitLayout {
                source_op_index: Some(op_idx),
                gc_ref_slots: exit_types
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, tp)| (*tp == Type::Ref).then_some(slot))
                    .collect(),
                force_token_slots: Vec::new(),
                exit_types,
                is_finish,
                recovery_layout,
                resume_layout,
                rd_numb,
                rd_consts,
                rd_virtuals,
                rd_pendingfields,
            },
        );
        fail_index += 1;
    }

    (result, guard_op_indices, exit_layouts)
}

pub(crate) fn merge_backend_exit_layouts(
    exit_layouts: &mut HashMap<u32, StoredExitLayout>,
    backend_layouts: &[FailDescrLayout],
) {
    for layout in backend_layouts {
        let entry: &mut StoredExitLayout =
            exit_layouts
                .entry(layout.fail_index)
                .or_insert_with(|| StoredExitLayout {
                    source_op_index: layout.source_op_index,
                    exit_types: layout.fail_arg_types.clone(),
                    is_finish: layout.is_finish,
                    gc_ref_slots: layout.gc_ref_slots.clone(),
                    force_token_slots: layout.force_token_slots.clone(),
                    recovery_layout: layout.recovery_layout.clone(),
                    resume_layout: None,
                    rd_numb: None,
                    rd_consts: None,
                    rd_virtuals: None,
                    rd_pendingfields: None,
                });
        entry.source_op_index = layout.source_op_index;
        // Preserve exit_types from build_guard_metadata (which reconciles
        // optimizer types with inputarg types after unbox_call_assembler).
        // Backend fail_arg_types may have stale Ref for unboxed CA results.
        if entry.exit_types.is_empty() {
            entry.exit_types = layout.fail_arg_types.clone();
        }
        entry.is_finish = layout.is_finish;
        entry.gc_ref_slots = layout.gc_ref_slots.clone();
        entry.force_token_slots = layout.force_token_slots.clone();
        entry.recovery_layout = layout.recovery_layout.clone();

        // Merge backend frame_stack metadata into the stored resume layout.
        if let Some(frame_stack) = &layout.frame_stack {
            merge_frame_stack_into_resume_layout(entry, frame_stack);
        }
    }
    validate_exit_layouts(exit_layouts);
}

/// Validate that guard exit layouts with recovery_layout have complete
/// metadata. Called after merge_backend_exit_layouts to enforce:
/// if recovery_layout is present, header_pc and slot_types must be set.
///
/// Guards that HAVE recovery_layout (all production guards after backend
/// merge) must satisfy the full invariant. Guards without (only possible
/// in unit tests with mock backends) are warned but not fatal.
pub(crate) fn validate_exit_layouts(exit_layouts: &HashMap<u32, StoredExitLayout>) {
    for (&fail_index, layout) in exit_layouts {
        if layout.is_finish {
            continue;
        }
        let Some(ref recovery) = layout.recovery_layout else {
            // No recovery_layout — backend didn't provide one (test mock).
            // In production, identity_recovery_layout always creates it.
            continue;
        };
        for (fi, frame) in recovery.frames.iter().enumerate() {
            // header_pc and slot_types are filled by both
            // build_guard_metadata (metainterp) and identity_recovery_layout
            // (backend). After backend merge, both must be present.
            // Backend-provided layouts always have them (compiler.rs:4482,4488).
            // Metainterp-provided layouts fill them since Step 1.
            if frame.header_pc.is_none() || frame.slot_types.is_none() {
                // Backend test mocks may omit these — not fatal in tests.
                #[cfg(not(test))]
                {
                    debug_assert!(
                        frame.header_pc.is_some(),
                        "guard fail_index={fail_index} frame[{fi}] has no header_pc"
                    );
                    debug_assert!(
                        frame.slot_types.is_some(),
                        "guard fail_index={fail_index} frame[{fi}] has no slot_types"
                    );
                }
                continue;
            }
            let st = frame.slot_types.as_ref().unwrap();
            debug_assert_eq!(
                st.len(),
                frame.slots.len(),
                "guard fail_index={fail_index} frame[{fi}]: slot_types.len()={} != slots.len()={}",
                st.len(),
                frame.slots.len(),
            );
        }
    }
}

/// Merge backend-origin `frame_stack` metadata into a `StoredExitLayout`'s
/// resume layout, enriching or creating `frame_layouts` entries with slot
/// types from the backend's `ExitFrameLayout`.
pub(crate) fn merge_frame_stack_into_resume_layout(
    entry: &mut StoredExitLayout,
    frame_stack: &[ExitFrameLayout],
) {
    if frame_stack.is_empty() {
        return;
    }

    let frame_layouts: Vec<ResumeFrameLayoutSummary> = frame_stack
        .iter()
        .map(ResumeFrameLayoutSummary::from_exit_frame_layout)
        .collect();

    if let Some(ref mut resume_layout) = entry.resume_layout {
        // Merge slot types from frame_stack into existing frame_layouts.
        let shared = resume_layout.frame_layouts.len().min(frame_layouts.len());
        for offset in 0..shared {
            let resume_index = resume_layout.frame_layouts.len() - 1 - offset;
            let fs_index = frame_layouts.len() - 1 - offset;
            let target = &mut resume_layout.frame_layouts[resume_index];
            let source = &frame_layouts[fs_index];

            if target.trace_id.is_none() {
                target.trace_id = source.trace_id;
            }
            if target.header_pc.is_none() {
                target.header_pc = source.header_pc;
            }
            if target.source_guard.is_none() {
                target.source_guard = source.source_guard;
            }

            let needs_slot_types = target
                .slot_types
                .as_ref()
                .map_or(true, |types| types.len() != target.slot_layouts.len());
            if needs_slot_types
                && source
                    .slot_types
                    .as_ref()
                    .is_some_and(|types| types.len() == target.slot_layouts.len())
            {
                target.slot_types = source.slot_types.clone();
            }
        }

        // If the frame_stack has more frames than the existing resume layout,
        // prepend the extra outer frames.
        if frame_layouts.len() > resume_layout.frame_layouts.len() {
            let extra_count = frame_layouts.len() - resume_layout.frame_layouts.len();
            let mut new_frames = frame_layouts[..extra_count].to_vec();
            new_frames.append(&mut resume_layout.frame_layouts);
            resume_layout.frame_layouts = new_frames;
            resume_layout.num_frames = resume_layout.frame_layouts.len();
            resume_layout.frame_pcs = resume_layout.frame_layouts.iter().map(|f| f.pc).collect();
            resume_layout.frame_slot_counts = resume_layout
                .frame_layouts
                .iter()
                .map(|f| f.slot_layouts.len())
                .collect();
        }
    } else {
        // No existing resume layout; create one from the frame_stack.
        entry.resume_layout = Some(ResumeLayoutSummary {
            num_frames: frame_layouts.len(),
            frame_pcs: frame_layouts.iter().map(|f| f.pc).collect(),
            frame_slot_counts: frame_layouts.iter().map(|f| f.slot_layouts.len()).collect(),
            frame_layouts,
            num_virtuals: 0,
            virtual_kinds: Vec::new(),
            virtual_layouts: Vec::new(),
            pending_field_count: 0,
            pending_field_layouts: Vec::new(),
            const_pool_size: 0,
        });
    }
}

/// Enrich an `Option<ResumeLayoutSummary>` with backend-origin `frame_stack`
/// metadata at runtime, merging slot types and outer frames.
pub(crate) fn enrich_resume_layout_with_frame_stack(
    resume_layout: &mut Option<ResumeLayoutSummary>,
    frame_stack: Option<&[ExitFrameLayout]>,
) {
    let Some(frame_stack) = frame_stack else {
        return;
    };
    if frame_stack.is_empty() {
        return;
    }

    let frame_layouts: Vec<ResumeFrameLayoutSummary> = frame_stack
        .iter()
        .map(ResumeFrameLayoutSummary::from_exit_frame_layout)
        .collect();

    if let Some(layout) = resume_layout {
        let shared = layout.frame_layouts.len().min(frame_layouts.len());
        for offset in 0..shared {
            let resume_index = layout.frame_layouts.len() - 1 - offset;
            let fs_index = frame_layouts.len() - 1 - offset;
            let target = &mut layout.frame_layouts[resume_index];
            let source = &frame_layouts[fs_index];

            if target.trace_id.is_none() {
                target.trace_id = source.trace_id;
            }
            if target.header_pc.is_none() {
                target.header_pc = source.header_pc;
            }
            if target.source_guard.is_none() {
                target.source_guard = source.source_guard;
            }

            let needs_slot_types = target
                .slot_types
                .as_ref()
                .map_or(true, |types| types.len() != target.slot_layouts.len());
            if needs_slot_types
                && source
                    .slot_types
                    .as_ref()
                    .is_some_and(|types| types.len() == target.slot_layouts.len())
            {
                target.slot_types = source.slot_types.clone();
            }
        }

        if frame_layouts.len() > layout.frame_layouts.len() {
            let extra_count = frame_layouts.len() - layout.frame_layouts.len();
            let mut new_frames = frame_layouts[..extra_count].to_vec();
            new_frames.append(&mut layout.frame_layouts);
            layout.frame_layouts = new_frames;
            layout.num_frames = layout.frame_layouts.len();
            layout.frame_pcs = layout.frame_layouts.iter().map(|f| f.pc).collect();
            layout.frame_slot_counts = layout
                .frame_layouts
                .iter()
                .map(|f| f.slot_layouts.len())
                .collect();
        }
    } else {
        *resume_layout = Some(ResumeLayoutSummary {
            num_frames: frame_layouts.len(),
            frame_pcs: frame_layouts.iter().map(|f| f.pc).collect(),
            frame_slot_counts: frame_layouts.iter().map(|f| f.slot_layouts.len()).collect(),
            frame_layouts,
            num_virtuals: 0,
            virtual_kinds: Vec::new(),
            virtual_layouts: Vec::new(),
            pending_field_count: 0,
            pending_field_layouts: Vec::new(),
            const_pool_size: 0,
        });
    }
}

pub(crate) fn merge_backend_terminal_exit_layouts(
    terminal_exit_layouts: &mut HashMap<usize, StoredExitLayout>,
    backend_layouts: &[TerminalExitLayout],
) {
    for layout in backend_layouts {
        let entry = terminal_exit_layouts
            .entry(layout.op_index)
            .or_insert_with(|| StoredExitLayout {
                source_op_index: Some(layout.op_index),
                exit_types: layout.exit_types.clone(),
                is_finish: layout.is_finish,
                gc_ref_slots: layout.gc_ref_slots.clone(),
                force_token_slots: layout.force_token_slots.clone(),
                recovery_layout: layout.recovery_layout.clone(),
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            });
        entry.source_op_index = Some(layout.op_index);
        entry.exit_types = layout.exit_types.clone();
        entry.is_finish = layout.is_finish;
        entry.gc_ref_slots = layout.gc_ref_slots.clone();
        entry.force_token_slots = layout.force_token_slots.clone();
        entry.recovery_layout = layout.recovery_layout.clone();
    }
}

pub(crate) fn enrich_resume_layout_with_trace_metadata(
    layout: &mut ResumeLayoutSummary,
    trace_id: u64,
    inputargs: &[InputArg],
    trace_info: Option<&CompiledTraceInfo>,
    recovery_layout: Option<&ExitRecoveryLayout>,
) {
    if layout.frame_layouts.is_empty() {
        return;
    }

    if let Some(recovery_layout) = recovery_layout {
        let shared_frames = layout.frame_layouts.len().min(recovery_layout.frames.len());
        for offset in 0..shared_frames {
            let layout_index = layout.frame_layouts.len() - 1 - offset;
            let recovery_index = recovery_layout.frames.len() - 1 - offset;
            let recovery_frame = &recovery_layout.frames[recovery_index];
            let frame = &mut layout.frame_layouts[layout_index];
            if frame.trace_id.is_none() {
                frame.trace_id = recovery_frame.trace_id;
            }
            if frame.header_pc.is_none() {
                frame.header_pc = recovery_frame.header_pc;
            }
            if frame.source_guard.is_none() {
                frame.source_guard = recovery_frame.source_guard;
            }
            let needs_slot_types = match frame.slot_types.as_ref() {
                Some(slot_types) => slot_types.len() != frame.slot_layouts.len(),
                None => true,
            };
            if needs_slot_types
                && recovery_frame
                    .slot_types
                    .as_ref()
                    .is_some_and(|slot_types| slot_types.len() == frame.slot_layouts.len())
            {
                frame.slot_types = recovery_frame.slot_types.clone();
            }
        }
    }

    let last_index = layout.frame_layouts.len() - 1;
    let innermost = &mut layout.frame_layouts[last_index];
    if innermost.trace_id.is_none() {
        innermost.trace_id = Some(trace_id);
    }
    if innermost.header_pc.is_none() {
        innermost.header_pc = trace_info.map(|info| info.header_pc);
    }
    if innermost.source_guard.is_none() {
        innermost.source_guard = trace_info.and_then(|info| info.source_guard);
    }
    let needs_slot_types = match innermost.slot_types.as_ref() {
        Some(slot_types) => slot_types.len() != innermost.slot_layouts.len(),
        None => true,
    };
    if needs_slot_types && inputargs.len() == innermost.slot_layouts.len() {
        innermost.slot_types = Some(inputargs.iter().map(|arg| arg.tp).collect());
    }
}

pub(crate) fn build_trace_value_maps(
    inputargs: &[InputArg],
    ops: &[majit_ir::Op],
) -> (HashMap<u32, Type>, HashMap<u32, OpCode>) {
    let mut value_types: HashMap<u32, Type> =
        inputargs.iter().map(|arg| (arg.index, arg.tp)).collect();
    let mut producers = HashMap::new();
    for op in ops {
        if !op.pos.is_none() && op.result_type() != Type::Void {
            value_types.insert(op.pos.0, op.result_type());
            producers.insert(op.pos.0, op.opcode);
        }
    }
    (value_types, producers)
}

pub(crate) fn find_fail_index_for_exit_op(ops: &[majit_ir::Op], op_index: usize) -> Option<u32> {
    let mut fail_index = 0u32;
    for (idx, op) in ops.iter().enumerate() {
        if op.opcode.is_guard() || op.opcode == OpCode::Finish {
            if idx == op_index {
                return Some(fail_index);
            }
            fail_index += 1;
        }
    }
    None
}

pub(crate) fn infer_terminal_exit_layout(
    inputargs: &[InputArg],
    ops: &[majit_ir::Op],
    owning_key: u64,
    trace_id: u64,
    op_index: usize,
) -> Option<CompiledExitLayout> {
    let op = ops.get(op_index)?;
    let is_finish = op.opcode == OpCode::Finish;
    if !is_finish && op.opcode != OpCode::Jump {
        return None;
    }
    let fail_index = find_fail_index_for_exit_op(ops, op_index).unwrap_or(u32::MAX);
    let (value_types, producers) = build_trace_value_maps(inputargs, ops);
    let exit_types: Vec<Type> = op
        .args
        .iter()
        .map(|opref| value_types.get(&opref.0).copied().unwrap_or(Type::Int))
        .collect();
    let force_token_slots: Vec<usize> = op
        .args
        .iter()
        .enumerate()
        .filter_map(|(slot, opref)| {
            producers
                .get(&opref.0)
                .copied()
                .filter(|opcode| *opcode == OpCode::ForceToken)
                .map(|_| slot)
        })
        .collect();
    let gc_ref_slots: Vec<usize> = exit_types
        .iter()
        .enumerate()
        .filter_map(|(slot, tp)| {
            (*tp == Type::Ref && !force_token_slots.contains(&slot)).then_some(slot)
        })
        .collect();
    Some(CompiledExitLayout {
        rd_loop_token: owning_key, // compile.py:186
        trace_id,
        fail_index,
        source_op_index: Some(op_index),
        exit_types,
        is_finish,
        gc_ref_slots,
        force_token_slots,
        recovery_layout: None,
        resume_layout: None,
        rd_numb: None,
        rd_consts: None,
        rd_virtuals: None,
        rd_pendingfields: None,
    })
}

pub(crate) fn build_terminal_exit_layouts(
    inputargs: &[InputArg],
    ops: &[majit_ir::Op],
) -> HashMap<usize, StoredExitLayout> {
    let mut layouts = HashMap::new();
    for (op_index, op) in ops.iter().enumerate() {
        if op.opcode != OpCode::Finish && op.opcode != OpCode::Jump {
            continue;
        }
        if let Some(layout) = infer_terminal_exit_layout(inputargs, ops, 0, 0, op_index) {
            layouts.insert(
                op_index,
                StoredExitLayout {
                    source_op_index: Some(op_index),
                    exit_types: layout.exit_types,
                    is_finish: layout.is_finish,
                    gc_ref_slots: layout.gc_ref_slots,
                    force_token_slots: layout.force_token_slots,
                    recovery_layout: None,
                    resume_layout: None,
                    rd_numb: None,
                    rd_consts: None,
                    rd_virtuals: None,
                    rd_pendingfields: None,
                },
            );
        }
    }
    layouts
}

pub(crate) fn terminal_exit_layout_for_trace(
    trace: &CompiledTrace,
    owning_key: u64,
    trace_id: u64,
    op_index: usize,
) -> Option<CompiledExitLayout> {
    if let Some(layout) = trace.terminal_exit_layouts.get(&op_index) {
        return Some(layout.public(
            owning_key,
            trace_id,
            find_fail_index_for_exit_op(&trace.ops, op_index).unwrap_or(u32::MAX),
        ));
    }
    if let Some(fail_index) = find_fail_index_for_exit_op(&trace.ops, op_index) {
        if let Some(layout) = trace.exit_layouts.get(&fail_index) {
            return Some(layout.public(owning_key, trace_id, fail_index));
        }
    }
    infer_terminal_exit_layout(&trace.inputargs, &trace.ops, owning_key, trace_id, op_index)
}

pub(crate) fn decode_values_with_layout(
    raw_values: &[i64],
    layout: &CompiledExitLayout,
) -> Vec<Value> {
    layout
        .exit_types
        .iter()
        .enumerate()
        .map(|(index, tp)| {
            let raw = raw_values.get(index).copied().unwrap_or(0);
            match tp {
                Type::Int => Value::Int(raw),
                Type::Ref => Value::Ref(GcRef(raw as usize)),
                Type::Float => Value::Float(f64::from_bits(raw as u64)),
                Type::Void => Value::Void,
            }
        })
        .collect()
}

pub(crate) fn normalize_closing_jump_args(
    mut ops: Vec<Op>,
    constants: &std::collections::HashMap<u32, i64>,
    num_inputs: usize,
) -> Vec<Op> {
    let Some(label_args) = ops
        .iter()
        .rev()
        .find(|op| op.opcode == OpCode::Label)
        .map(|op| op.args.clone())
    else {
        return ops;
    };

    let defined: std::collections::HashSet<OpRef> = ops
        .iter()
        .filter(|op| op.result_type() != majit_ir::Type::Void && !op.pos.is_none())
        .map(|op| op.pos)
        .collect();

    let Some(jump) = ops.iter_mut().rfind(|op| op.opcode == OpCode::Jump) else {
        return ops;
    };

    for (idx, arg) in jump.args.iter_mut().enumerate() {
        if idx >= label_args.len() {
            break;
        }
        if constants.contains_key(&arg.0) {
            continue;
        }
        if (arg.0 as usize) < num_inputs {
            continue;
        }
        if defined.contains(arg) {
            continue;
        }
        *arg = label_args[idx];
    }

    ops
}

/// `rpython/jit/metainterp/compile.py:425-461`
/// `patch_new_loop_to_load_virtualizable_fields`.
///
/// ```python
/// def patch_new_loop_to_load_virtualizable_fields(loop, jitdriver_sd, vable):
///     vinfo = jitdriver_sd.virtualizable_info
///     extra_ops = []
///     inputargs = loop.inputargs
///     vable_box = inputargs[jitdriver_sd.index_of_virtualizable]
///     i = jitdriver_sd.num_red_args
///     loop.inputargs = inputargs[:i]
///     for descr in vinfo.static_field_descrs:
///         assert i < len(inputargs)
///         box = inputargs[i]
///         opnum = OpHelpers.getfield_for_descr(descr)
///         emit_op(extra_ops,
///                 ResOperation(opnum, [vable_box], descr=descr))
///         box.set_forwarded(extra_ops[-1])
///         i += 1
///     arrayindex = 0
///     for descr in vinfo.array_field_descrs:
///         arraylen = vinfo.get_array_length(vable, arrayindex)
///         arrayop = ResOperation(rop.GETFIELD_GC_R, [vable_box], descr=descr)
///         emit_op(extra_ops, arrayop)
///         arraydescr = vinfo.array_descrs[arrayindex]
///         assert i + arraylen <= len(inputargs)
///         for index in range(arraylen):
///             opnum = OpHelpers.getarrayitem_for_descr(arraydescr)
///             box = inputargs[i]
///             emit_op(extra_ops,
///                 ResOperation(opnum,
///                              [arrayop, ConstInt(index)],
///                              descr=arraydescr))
///             i += 1
///             box.set_forwarded(extra_ops[-1])
///         arrayindex += 1
///     assert i == len(inputargs)
///     for op in loop.operations:
///         emit_op(extra_ops, op)
///     loop.operations = extra_ops
/// ```
///
/// Called from `send_loop_to_backend` (compile.py:504-511) after the loop
/// has been optimized but before it is handed to the CPU backend. The
/// virtualizable's static and array fields ride through the optimizer as
/// expanded trace inputargs; this function strips them and reconstructs
/// each field at loop entry with a `GETFIELD_GC` / `GETARRAYITEM_GC` op
/// so the compiled loop's `len(inputargs) == num_red_args` matches
/// `execute_token`'s `clt._debug_nbargs` and CA's `op.args.len()`.
///
/// `vable_array_lengths` mirrors RPython's `vinfo.get_array_length(vable, i)`
/// reads: one length per array field (in `vinfo.array_fields` order), taken
/// from the concrete virtualizable at trace-start time.
pub(crate) fn patch_new_loop_to_load_virtualizable_fields(
    ops: &mut Vec<Op>,
    inputargs: &mut Vec<InputArg>,
    vinfo: &crate::virtualizable::VirtualizableInfo,
    vable_array_lengths: &[usize],
    num_red_args: usize,
    index_of_virtualizable: usize,
    constants: &mut HashMap<u32, i64>,
    constant_types: &mut HashMap<u32, Type>,
) {
    use majit_ir::{Op, OpCode, OpRef};

    assert!(
        index_of_virtualizable < num_red_args,
        "virtualizable must live inside the red args (pyjitpl.py:3589 index_of_virtualizable < num_red_args)"
    );
    if inputargs.len() <= num_red_args {
        // Already reduced or no virtualizable expansion in the trace.
        return;
    }

    // compile.py:429-430 — vable_box = inputargs[index_of_virtualizable].
    let vable_box = OpRef(inputargs[index_of_virtualizable].index);

    // Allocate fresh OpRefs above existing op positions AND inputarg indices.
    let max_op_pos = ops
        .iter()
        .filter_map(|op| (!op.pos.is_none()).then_some(op.pos.0))
        .max()
        .unwrap_or(0);
    let max_inputarg = inputargs.iter().map(|ia| ia.index).max().unwrap_or(0);
    let mut next_opref = max_op_pos.max(max_inputarg) + 1;

    // Allocate fresh const indices above the existing max.
    let mut next_const_idx = constants
        .keys()
        .map(|&k| OpRef(k).const_index())
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    let mut forwarding: HashMap<OpRef, OpRef> = HashMap::new();
    let mut extra_ops: Vec<Op> = Vec::new();
    let mut i = num_red_args;

    // compile.py:433-440 — GETFIELD_GC per static field.
    let static_descrs = vinfo.static_field_descrs();
    for (fi, field) in vinfo.static_fields.iter().enumerate() {
        assert!(
            i < inputargs.len(),
            "static field {fi} exceeds inputargs ({} <= {})",
            i,
            inputargs.len()
        );
        let descr = static_descrs
            .get(fi)
            .cloned()
            .expect("static_field_descrs must be populated by set_parent_descr");
        let opcode = match field.field_type {
            Type::Int => OpCode::GetfieldGcI,
            Type::Ref => OpCode::GetfieldGcR,
            Type::Float => OpCode::GetfieldGcF,
            Type::Void => panic!("virtualizable static field {fi} has Void type"),
        };
        let old_opref = OpRef(inputargs[i].index);
        let new_opref = OpRef(next_opref);
        next_opref += 1;
        let mut op = Op::new(opcode, &[vable_box]);
        op.pos = new_opref;
        op.descr = Some(descr);
        extra_ops.push(op);
        forwarding.insert(old_opref, new_opref);
        i += 1;
    }

    // compile.py:441-457 — GETFIELD_GC_R (array ptr) + GETARRAYITEM_GC per element.
    let array_descrs_list = vinfo.array_field_descrs();
    for (ai, array_field_descr) in array_descrs_list.iter().enumerate() {
        let array_len = vable_array_lengths.get(ai).copied().unwrap_or(0);
        assert!(
            i + array_len <= inputargs.len(),
            "array {ai} length {array_len} would overrun inputargs (i={i}, len={})",
            inputargs.len()
        );
        // GETFIELD_GC_R(vable_box, array_field_descr) → array pointer.
        let array_opref = OpRef(next_opref);
        next_opref += 1;
        let mut arr_load = Op::new(OpCode::GetfieldGcR, &[vable_box]);
        arr_load.pos = array_opref;
        arr_load.descr = Some(array_field_descr.clone());
        extra_ops.push(arr_load);

        let array_descr = vinfo
            .array_descrs
            .get(ai)
            .cloned()
            .expect("VirtualizableInfo.array_descrs must cover every array_field");
        let item_opcode = match vinfo.array_fields[ai].item_type {
            Type::Int => OpCode::GetarrayitemGcI,
            Type::Ref => OpCode::GetarrayitemGcR,
            Type::Float => OpCode::GetarrayitemGcF,
            Type::Void => panic!("virtualizable array {ai} has Void item_type"),
        };
        for index in 0..array_len {
            // compile.py:453 — ConstInt(index) for the array subscript.
            let const_opref = OpRef::from_const(next_const_idx);
            next_const_idx += 1;
            constants.insert(const_opref.0, index as i64);
            constant_types.insert(const_opref.0, Type::Int);

            let old_opref = OpRef(inputargs[i].index);
            let new_opref = OpRef(next_opref);
            next_opref += 1;
            let mut elem_op = Op::new(item_opcode, &[array_opref, const_opref]);
            elem_op.pos = new_opref;
            elem_op.descr = Some(array_descr.clone());
            extra_ops.push(elem_op);
            forwarding.insert(old_opref, new_opref);
            i += 1;
        }
    }

    assert!(
        i == inputargs.len(),
        "compile.py:458 assert i == len(inputargs) failed ({i} != {})",
        inputargs.len()
    );

    // compile.py:432 — loop.inputargs = inputargs[:num_red_args].
    inputargs.truncate(num_red_args);

    // compile.py:459-461 — emit_op walks existing ops through
    // get_box_replacement; in pyre we apply the forwarding map directly.
    for op in ops.iter_mut() {
        for arg in op.args.iter_mut() {
            if let Some(&new_ref) = forwarding.get(arg) {
                *arg = new_ref;
            }
        }
        if let Some(fa) = op.fail_args.as_mut() {
            for arg in fa.iter_mut() {
                if let Some(&new_ref) = forwarding.get(arg) {
                    *arg = new_ref;
                }
            }
        }
    }

    // Prepend extra_ops to loop operations (compile.py:459-461 reconstructs
    // loop.operations in the same order).
    //
    // In RPython the closing `JUMP` targets the `TargetToken`'s internal
    // `LABEL`, whose arity is SEPARATE from `loop.inputargs`; truncating
    // inputargs therefore never touches JUMP/LABEL inside operations. pyre's
    // current JUMP-terminated paths (compile_loop at pyjitpl/mod.rs:1936,
    // retry-without-unroll at 3094, simple-loop at 3803) inline a
    // `LABEL(inputargs)` whose arity is coupled to the inputargs array, so
    // this helper is not yet wired into those paths — see
    // `MetaInterp::patch_new_loop_to_load_virtualizable_fields` for the
    // finish-only call site and the JUMP-path TODO.
    let mut combined = extra_ops;
    combined.append(ops);
    *ops = combined;
}

/// RPython dependency.py requires GUARD_(NO_)OVERFLOW to be scheduled only
/// when there is a live preceding INT_*_OVF operation to consume.
/// intbounds.py:231-242: optimizer raises InvalidLoop for stray overflow
/// guards. This function is a post-optimization safety net: if any stray
/// guard survived, strip it to prevent backend panic.
pub(crate) fn strip_stray_overflow_guards(ops: Vec<Op>) -> Vec<Op> {
    use majit_ir::OpCode;

    let mut pending_ovf = false;
    let mut result = Vec::with_capacity(ops.len());
    for op in ops {
        match op.opcode {
            OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf => {
                pending_ovf = true;
                result.push(op);
            }
            OpCode::GuardNoOverflow | OpCode::GuardOverflow => {
                if pending_ovf {
                    result.push(op);
                }
                // else: stray guard — strip it (intbounds.py:231 InvalidLoop
                // should have caught it; this is a safety net).
                pending_ovf = false;
            }
            OpCode::Label | OpCode::Jump | OpCode::Finish => {
                pending_ovf = false;
                result.push(op);
            }
            _ => {
                result.push(op);
            }
        }
    }
    result
}

pub(crate) fn enrich_guard_resume_layouts_for_trace(
    resume_data: &mut HashMap<u32, StoredResumeData>,
    exit_layouts: &mut HashMap<u32, StoredExitLayout>,
    trace_id: u64,
    inputargs: &[InputArg],
    trace_info: Option<&CompiledTraceInfo>,
) {
    for (fail_index, stored) in resume_data.iter_mut() {
        let recovery_layout = exit_layouts
            .get(fail_index)
            .and_then(|layout| layout.recovery_layout.clone());
        enrich_resume_layout_with_trace_metadata(
            &mut stored.layout,
            trace_id,
            inputargs,
            trace_info,
            recovery_layout.as_ref(),
        );
        if let Some(exit_layout) = exit_layouts.get_mut(fail_index) {
            exit_layout.resume_layout = Some(stored.layout.clone());
        }
    }
}

pub(crate) fn patch_backend_guard_recovery_layouts_for_trace(
    backend: &mut dyn majit_backend::Backend,
    token: &majit_backend::JitCellToken,
    trace_id: u64,
    exit_layouts: &mut HashMap<u32, StoredExitLayout>,
) {
    for (&fail_index, exit_layout) in exit_layouts.iter_mut() {
        let Some(resume_layout) = exit_layout.resume_layout.as_ref() else {
            continue;
        };
        let recovery_layout = resume_layout
            .to_exit_recovery_layout_with_caller_prefix(exit_layout.recovery_layout.as_ref());
        if backend.update_fail_descr_recovery_layout(
            token,
            trace_id,
            fail_index,
            recovery_layout.clone(),
        ) {
            exit_layout.recovery_layout = Some(recovery_layout);
        }
    }
}

pub(crate) fn patch_backend_terminal_recovery_layouts_for_trace(
    backend: &mut dyn majit_backend::Backend,
    token: &majit_backend::JitCellToken,
    trace_id: u64,
    terminal_exit_layouts: &mut HashMap<usize, StoredExitLayout>,
) {
    for (&op_index, exit_layout) in terminal_exit_layouts.iter_mut() {
        let Some(resume_layout) = exit_layout.resume_layout.as_ref() else {
            continue;
        };
        let recovery_layout = resume_layout
            .to_exit_recovery_layout_with_caller_prefix(exit_layout.recovery_layout.as_ref());
        if backend.update_terminal_exit_recovery_layout(
            token,
            trace_id,
            op_index,
            recovery_layout.clone(),
        ) {
            exit_layout.recovery_layout = Some(recovery_layout);
        }
    }
}

#[derive(Debug)]
pub(crate) struct BridgeFailDescrProxy {
    pub(crate) fail_index: u32,
    pub(crate) trace_id: u64,
    pub(crate) fail_arg_types: Vec<Type>,
    pub(crate) gc_ref_slots: Vec<usize>,
    pub(crate) force_token_slots: Vec<usize>,
    pub(crate) is_finish: bool,
}

impl majit_ir::Descr for BridgeFailDescrProxy {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn majit_ir::FailDescr> {
        Some(self)
    }
}

impl majit_ir::FailDescr for BridgeFailDescrProxy {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        self.is_finish
    }
    fn trace_id(&self) -> u64 {
        self.trace_id
    }
    fn is_gc_ref_slot(&self, slot: usize) -> bool {
        self.gc_ref_slots.contains(&slot)
    }
    fn force_token_slots(&self) -> &[usize] {
        &self.force_token_slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fail_descr::make_fail_descr_with_index;
    use crate::resume::{ResumeDataLoopMemo, SimpleBoxEnv, Snapshot, SnapshotFrame};
    use majit_ir::{Op, OpCode, OpRef};

    #[test]
    fn test_build_guard_metadata_keeps_vable_array_out_of_frame_slots() {
        use majit_backend::ExitValueSourceLayout;

        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.types.insert(0, Type::Ref);
        env.types.insert(1, Type::Int);
        env.constants.insert(OpRef::from_const(1).0, (8, Type::Int));
        env.constants
            .insert(OpRef::from_const(2).0, (777, Type::Int)); // code object payload
        env.constants.insert(OpRef::from_const(3).0, (2, Type::Int));
        env.constants
            .insert(OpRef::from_const(4).0, (999, Type::Int)); // namespace payload

        let snapshot = Snapshot {
            vable_array: vec![
                OpRef(0),
                OpRef::from_const(1),
                OpRef::from_const(2),
                OpRef::from_const(3),
                OpRef::from_const(4),
            ],
            vref_array: vec![],
            framestack: vec![SnapshotFrame {
                jitcode_index: 0,
                pc: 8,
                boxes: vec![OpRef(1)],
            }],
        };
        let mut numb_state = memo.number(&snapshot, &env).unwrap();
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();
        let rd_consts = memo.consts().to_vec();

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(1)]);
        guard.descr = Some(make_fail_descr_with_index(0, 2));
        guard.fail_args = Some(smallvec::smallvec![OpRef(0), OpRef(1)]);
        guard.fail_arg_types = Some(vec![Type::Ref, Type::Int]);
        guard.rd_numb = Some(rd_numb);
        guard.rd_consts = Some(rd_consts);

        let (_resume_data, _guard_indices, exit_layouts) =
            build_guard_metadata(&inputargs, &[guard], 8, &HashMap::new());
        let exit = exit_layouts.get(&0).expect("guard exit layout");

        let resume_layout = exit.resume_layout.as_ref().expect("resume_layout");
        assert_eq!(resume_layout.frame_layouts.len(), 1);
        assert_eq!(
            resume_layout.frame_layouts[0]
                .slot_layouts
                .iter()
                .map(|slot| slot.fail_arg_index)
                .collect::<Vec<_>>(),
            vec![1]
        );

        let recovery = exit.recovery_layout.as_ref().expect("recovery_layout");
        assert_eq!(recovery.frames.len(), 1);
        assert_eq!(
            recovery.frames[0].slots,
            vec![ExitValueSourceLayout::ExitValue(1),]
        );
        assert_eq!(
            recovery.frames[0].slot_types.as_ref().unwrap(),
            &vec![Type::Int]
        );
    }
}
