//! Trace compilation helpers.
//!
//! Mirrors RPython's `compile.py`: guard metadata building, exit layout
//! management, backend layout merging, and trace post-processing (unboxing).

use std::collections::{HashMap, HashSet};

use majit_codegen::{
    Backend, CompiledTraceInfo, ExitFrameLayout, ExitRecoveryLayout, FailDescrLayout, JitCellToken,
    TerminalExitLayout,
};
use majit_ir::{GcRef, InputArg, Op, OpCode, OpRef, Type, Value};

use crate::blackhole::ExceptionState;
use crate::pyjitpl::{CompiledTrace, StoredExitLayout, StoredResumeData};
use crate::resume::{
    ResumeDataLoopMemo, ResumeDataVirtualAdder, ResumeFrameLayoutSummary, ResumeLayoutSummary,
};

/// Resolve the type of an OpRef in guard fail_args.
/// OpRef::NONE is a virtual slot placeholder (null GC ref).
fn fail_arg_type(opref: &OpRef, value_types: &HashMap<u32, Type>) -> Type {
    if *opref == OpRef::NONE {
        Type::Ref
    } else {
        value_types.get(&opref.0).copied().unwrap_or(Type::Int)
    }
}

// ── Compilation result types (compile.py) ───────────────────────────────

/// Static exit metadata for a compiled guard or finish point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledExitLayout {
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
    /// resume.py:488 — virtual object blueprints (descr_index, known_class, fieldnums).
    pub rd_virtuals_info: Option<Vec<(u32, Option<i64>, Vec<i16>)>>,
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
}

/// Terminal exit layout for a FINISH or JUMP op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledTerminalExitLayout {
    pub op_index: usize,
    pub exit_layout: CompiledExitLayout,
}

/// Full trace compilation layout with all exits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledTraceLayout {
    pub trace_id: u64,
    pub exit_layouts: Vec<CompiledExitLayout>,
    pub terminal_exit_layouts: Vec<CompiledTerminalExitLayout>,
}

/// Artifacts extracted from a backend DeadFrame.
#[derive(Debug, Clone, PartialEq)]
pub struct DeadFrameArtifacts {
    pub values: Vec<i64>,
    pub typed_values: Vec<Value>,
    pub exit_layout: CompiledExitLayout,
    pub savedata: Option<GcRef>,
    pub exception: ExceptionState,
}

// ── Compilation helper functions ────────────────────────────────────────

// number_guards_final removed: rd_numb is now produced inline during
// optimization via ctx.emit() → number_guard_inline (RPython parity).

/// Build guard metadata for a compiled trace.
///
/// The backend numbers every guard and finish in a single exit table, so this
/// helper mirrors that numbering and records only the guard entries that need
/// resume data plus the corresponding op index for blackhole fallback.
pub(crate) fn build_guard_metadata(
    inputargs: &[InputArg],
    ops: &[majit_ir::Op],
    pc: u64,
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

        let exit_types: Vec<Type> = if is_finish {
            op.args
                .iter()
                .map(|opref| value_types.get(&opref.0).copied().unwrap_or(Type::Int))
                .collect()
        } else if let Some(ref fail_args) = op.fail_args {
            fail_args
                .iter()
                .map(|opref| fail_arg_type(opref, &value_types))
                .collect()
        } else {
            inputargs.iter().map(|arg| arg.tp).collect()
        };

        let mut resume_layout = None;
        if is_guard {
            if let Some(ref fail_args) = op.fail_args {
                let mut builder = ResumeDataVirtualAdder::new();
                builder.push_frame(pc);

                for (slot_idx, _) in fail_args.iter().enumerate() {
                    builder.map_slot(slot_idx, slot_idx);
                }

                let stored = StoredResumeData::with_loop_memo(builder.build(), &mut resume_memo);
                resume_layout = Some(stored.layout.clone());
                result.insert(fail_index, stored);
            }
        }

        // Store rd_numb/rd_consts/rd_virtuals_info for guard failure recovery.
        let rd_numb = op.rd_numb.clone();
        let rd_consts = op.rd_consts.clone();
        let rd_virtuals_info = op.rd_virtuals_info.clone();

        // Build recovery_layout from rd_virtuals (GuardVirtualEntry).
        let recovery_layout = if let Some(ref entries) = op.rd_virtuals {
            use majit_codegen::{ExitRecoveryLayout, ExitValueSourceLayout, ExitVirtualLayout};
            let virtual_layouts: Vec<ExitVirtualLayout> = entries
                .iter()
                .map(|entry| ExitVirtualLayout::Struct {
                    type_id: entry.known_class.map_or(0, |gc| gc.as_usize() as u32),
                    descr_index: entry.descr.index(),
                    fields: entry
                        .fields
                        .iter()
                        .map(|(field_descr_idx, fail_arg_idx)| {
                            (
                                *field_descr_idx,
                                ExitValueSourceLayout::ExitValue(*fail_arg_idx),
                            )
                        })
                        .collect(),
                })
                .collect();
            // Convert op.rd_pendingfields to ExitPendingFieldLayout
            let pending_field_layouts: Vec<majit_codegen::ExitPendingFieldLayout> = op
                .rd_pendingfields
                .as_ref()
                .map(|entries| {
                    entries
                        .iter()
                        .filter_map(|entry| {
                            let fail_args = op.fail_args.as_ref()?;
                            let target_idx = fail_args.iter().position(|&a| a == entry.target)?;
                            let value_idx = fail_args.iter().position(|&a| a == entry.value);
                            let target = ExitValueSourceLayout::ExitValue(target_idx);
                            let value = match value_idx {
                                Some(idx) => ExitValueSourceLayout::ExitValue(idx),
                                None => ExitValueSourceLayout::Constant(entry.value.0 as i64),
                            };
                            Some(majit_codegen::ExitPendingFieldLayout {
                                descr_index: entry.descr_index,
                                item_index: if entry.item_index >= 0 {
                                    Some(entry.item_index as usize)
                                } else {
                                    None
                                },
                                is_array_item: entry.item_index >= 0,
                                target,
                                value,
                                field_offset: entry.field_offset,
                                field_size: entry.field_size,
                                field_type: entry.field_type,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            // resume.py:1042-1057 parity: build frame_slots from rd_numb.
            // rd_numb encodes TAGBOX/TAGCONST/TAGVIRTUAL per frame slot.
            //
            // Infrastructure constraint: encode_guard_virtuals_impl (which
            // has no RPython equivalent) replaces virtual OpRefs with NONE
            // in fail_args. The 1:1 TAGBOX path in number_guard_inline
            // then encodes them as NULLREF instead of TAGVIRTUAL. We
            // overlay rd_virtuals virtual_map to recover Virtual entries.
            // In RPython, finish() tags virtuals as TAGVIRTUAL directly.
            let frame_slots = if let (Some(rd_numb_bytes), Some(rd_consts_data)) =
                (&op.rd_numb, &op.rd_consts)
            {
                use majit_ir::resumedata::{RebuiltValue, rebuild_from_numbering};
                let (_num_failargs, frames) = rebuild_from_numbering(rd_numb_bytes, rd_consts_data);
                // Build virtual_map from rd_virtuals for NULLREF → Virtual.
                let virtual_map: std::collections::HashMap<usize, usize> = entries
                    .iter()
                    .enumerate()
                    .map(|(vidx, entry)| (entry.fail_arg_index, vidx))
                    .collect();
                let mut slots = Vec::new();
                let mut slot_idx = 0usize;
                for frame in &frames {
                    for val in &frame.values {
                        slots.push(match val {
                            RebuiltValue::Box(idx) => ExitValueSourceLayout::ExitValue(*idx),
                            RebuiltValue::Virtual(vidx) => ExitValueSourceLayout::Virtual(*vidx),
                            RebuiltValue::Const(c, tp) => {
                                // NULLREF (Const(0, Ref)) → check virtual_map.
                                if *c == 0 && *tp == majit_ir::Type::Ref {
                                    if let Some(&vidx) = virtual_map.get(&slot_idx) {
                                        ExitValueSourceLayout::Virtual(vidx)
                                    } else {
                                        ExitValueSourceLayout::Constant(*c)
                                    }
                                } else {
                                    ExitValueSourceLayout::Constant(*c)
                                }
                            }
                            RebuiltValue::Int(i) => ExitValueSourceLayout::Constant(*i as i64),
                            RebuiltValue::Unassigned => ExitValueSourceLayout::Uninitialized,
                        });
                        slot_idx += 1;
                    }
                }
                slots
            } else if let Some(ref fa) = op.fail_args {
                // Legacy fallback: fail_args + virtual_map overlay.
                let mut virtual_map = std::collections::HashMap::new();
                for (vidx, entry) in entries.iter().enumerate() {
                    virtual_map.insert(entry.fail_arg_index, vidx);
                }
                (0..fa.len())
                    .map(|i| {
                        if let Some(&vidx) = virtual_map.get(&i) {
                            ExitValueSourceLayout::Virtual(vidx)
                        } else {
                            ExitValueSourceLayout::ExitValue(i)
                        }
                    })
                    .collect()
            } else {
                vec![]
            };
            Some(ExitRecoveryLayout {
                frames: vec![majit_codegen::ExitFrameLayout {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: 0,
                    slots: frame_slots,
                    slot_types: None,
                }],
                virtual_layouts,
                pending_field_layouts: vec![],
            })
        } else {
            None
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
                rd_virtuals_info,
            },
        );
        fail_index += 1;
    }

    (result, guard_op_indices, exit_layouts)
}

pub(crate) fn retag_fail_descrs_from_trace_types(inputargs: &[InputArg], ops: &mut [majit_ir::Op]) {
    let (value_types, _) = build_trace_value_maps(inputargs, ops);
    let mut next_fail_index = 0u32;
    for op in ops.iter_mut() {
        let is_exit = op.opcode.is_guard() || op.opcode == OpCode::Finish;
        if !is_exit {
            continue;
        }
        let types: Vec<Type> = if op.opcode == OpCode::Finish {
            op.args
                .iter()
                .map(|opref| value_types.get(&opref.0).copied().unwrap_or(Type::Int))
                .collect()
        } else if let Some(ref fail_args) = op.fail_args {
            fail_args
                .iter()
                .map(|opref| fail_arg_type(opref, &value_types))
                .collect()
        } else {
            inputargs.iter().map(|arg| arg.tp).collect()
        };
        op.descr = Some(crate::fail_descr::make_fail_descr_typed_with_index(
            next_fail_index,
            types,
        ));
        next_fail_index += 1;
    }
}

pub(crate) fn merge_backend_exit_layouts(
    exit_layouts: &mut HashMap<u32, StoredExitLayout>,
    backend_layouts: &[FailDescrLayout],
) {
    for layout in backend_layouts {
        let entry = exit_layouts
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
                rd_virtuals_info: None,
            });
        entry.source_op_index = layout.source_op_index;
        entry.exit_types = layout.fail_arg_types.clone();
        entry.is_finish = layout.is_finish;
        entry.gc_ref_slots = layout.gc_ref_slots.clone();
        entry.force_token_slots = layout.force_token_slots.clone();
        entry.recovery_layout = layout.recovery_layout.clone();

        // Merge backend frame_stack metadata into the stored resume layout.
        if let Some(frame_stack) = &layout.frame_stack {
            merge_frame_stack_into_resume_layout(entry, frame_stack);
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
            num_fail_args: entry.exit_types.len(),
            fail_arg_positions: (0..entry.exit_types.len()).collect(),
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
    exit_types: &[Type],
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
            num_fail_args: exit_types.len(),
            fail_arg_positions: (0..exit_types.len()).collect(),
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
                rd_virtuals_info: None,
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
        rd_virtuals_info: None,
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
        if let Some(layout) = infer_terminal_exit_layout(inputargs, ops, 0, op_index) {
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
                    rd_virtuals_info: None,
                },
            );
        }
    }
    layouts
}

pub(crate) fn terminal_exit_layout_for_trace(
    trace: &CompiledTrace,
    trace_id: u64,
    op_index: usize,
) -> Option<CompiledExitLayout> {
    if let Some(layout) = trace.terminal_exit_layouts.get(&op_index) {
        return Some(layout.public(
            trace_id,
            find_fail_index_for_exit_op(&trace.ops, op_index).unwrap_or(u32::MAX),
        ));
    }
    if let Some(fail_index) = find_fail_index_for_exit_op(&trace.ops, op_index) {
        if let Some(layout) = trace.exit_layouts.get(&fail_index) {
            return Some(layout.public(trace_id, fail_index));
        }
    }
    infer_terminal_exit_layout(&trace.inputargs, &trace.ops, trace_id, op_index)
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

pub(crate) fn unbox_finish_result(
    mut ops: Vec<Op>,
    constants: &HashMap<u32, i64>,
    raw_int_box_helpers: &HashSet<i64>,
) -> (Vec<Op>, bool) {
    use majit_ir::OpCode;

    let finish_idx = match ops.iter().rposition(|op| op.opcode == OpCode::Finish) {
        Some(i) => i,
        None => return (ops, false),
    };
    let finish_arg = match ops[finish_idx].args.first().copied() {
        Some(a) => a,
        None => return (ops, false),
    };

    // Pattern 1: CallI(box_int_helper, raw_int)
    for idx in (0..finish_idx).rev() {
        let op = &ops[idx];
        if op.pos == finish_arg && op.opcode == OpCode::CallI {
            let helper_ptr = op
                .args
                .first()
                .and_then(|func| constants.get(&func.0))
                .copied();
            if op.args.len() >= 2
                && helper_ptr.is_some_and(|ptr| raw_int_box_helpers.contains(&ptr))
            {
                let raw_int = op.args[1];
                ops[finish_idx].args[0] = raw_int;
                ops[finish_idx].descr = Some(crate::make_fail_descr_typed(vec![Type::Int]));
                ops.remove(idx);
                return (ops, true);
            }
        }
    }

    // Pattern 2: New() + SetfieldGc chain
    //
    // Optimizer passes may reorder the `New` relative to its `SetfieldGc`
    // users in the final op list, so do not assume the field stores appear
    // textually after the allocation. Match by producer/result identity only.
    let new_idx = match ops[..finish_idx]
        .iter()
        .rposition(|op| op.pos == finish_arg && op.opcode == OpCode::New)
    {
        Some(i) => i,
        None => return (ops, false),
    };

    let mut raw_int = None;
    for op in &ops[..finish_idx] {
        if op.opcode == OpCode::SetfieldGc && op.args.first() == Some(&finish_arg) {
            if let Some(ref d) = op.descr {
                let ds = format!("{d:?}");
                if ds.contains("offset: 8") && ds.contains("signed: true") {
                    raw_int = op.args.get(1).copied();
                }
            }
        }
    }
    if let Some(raw_int) = raw_int {
        ops[finish_idx].args[0] = raw_int;
        ops[finish_idx].descr = Some(crate::make_fail_descr_typed(vec![Type::Int]));
        let mut to_remove = vec![new_idx];
        for (i, op) in ops[..finish_idx].iter().enumerate() {
            if op.opcode == OpCode::SetfieldGc && op.args.first() == Some(&finish_arg) {
                to_remove.push(i);
            }
        }
        to_remove.sort_unstable();
        to_remove.dedup();
        for &idx in to_remove.iter().rev() {
            ops.remove(idx);
        }
        return (ops, true);
    }
    (ops, false)
}

/// Strip caller-side unboxing after CallAssemblerI results.
pub(crate) fn unbox_call_assembler_results(mut ops: Vec<Op>) -> Vec<Op> {
    use majit_ir::OpCode;

    // Strip unboxing for CallAssemblerI results only.
    // CallAssemblerI returns raw int (compiled Finish is unboxed).
    // CallMayForceI now returns boxed — its force_fn re-boxes the result.
    let ca_results: Vec<OpRef> = ops
        .iter()
        .filter(|op| op.opcode == OpCode::CallAssemblerI)
        .map(|op| op.pos)
        .collect();

    if ca_results.is_empty() {
        return ops;
    }

    for ca_ref in &ca_results {
        let mut intval_refs: Vec<(usize, OpRef)> = Vec::new();
        let mut ops_to_remove: Vec<usize> = Vec::new();
        let mut ob_type_refs: Vec<(usize, OpRef)> = Vec::new();

        for (idx, op) in ops.iter().enumerate() {
            if !matches!(
                op.opcode,
                OpCode::GetfieldRawI | OpCode::GetfieldGcI | OpCode::GetfieldGcPureI
            ) || op.args.first() != Some(ca_ref)
            {
                continue;
            }
            if let Some(ref d) = op.descr {
                let ds = format!("{d:?}");
                if ds.contains("offset: 8")
                    && ds.contains("field_size: 8")
                    && ds.contains("signed: true")
                {
                    intval_refs.push((idx, op.pos));
                    ops_to_remove.push(idx);
                } else if ds.contains("offset: 0") {
                    ob_type_refs.push((idx, op.pos));
                }
            }
        }

        if !intval_refs.is_empty() {
            for (idx, ob_type_ref) in ob_type_refs {
                ops_to_remove.push(idx);
                for (idx2, op2) in ops.iter().enumerate() {
                    if op2.opcode == OpCode::GuardClass && op2.args.first() == Some(&ob_type_ref) {
                        ops_to_remove.push(idx2);
                    }
                }
            }
        }

        for &(_, intval_ref) in &intval_refs {
            for op in ops.iter_mut() {
                for arg in op.args.iter_mut() {
                    if *arg == intval_ref {
                        *arg = *ca_ref;
                    }
                }
                if let Some(ref mut fa) = op.fail_args {
                    for arg in fa.iter_mut() {
                        if *arg == intval_ref {
                            *arg = *ca_ref;
                        }
                    }
                }
            }
        }

        ops_to_remove.sort_unstable();
        ops_to_remove.dedup();
        for &idx in ops_to_remove.iter().rev() {
            if idx < ops.len() {
                ops.remove(idx);
            }
        }
    }

    ops
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
        // compile.py closes the loop against the label namespace. If a stale
        // pre-normalization box leaks into the final Jump, replace it with the
        // corresponding label slot before handing the trace to the backend.
        *arg = label_args[idx];
    }

    ops
}

/// Strip caller-side unboxing after CallMayForceI results from raw-int helpers.
///
/// Pattern:
///   v1 = CallMayForceI(raw_helper, ...)
///   v2 = GetfieldGcI(v1, ob_type)
///   GuardClass(v2, INT_TYPE)
///   v3 = GetfieldGcI(v1, intval)
/// becomes:
///   v1 = CallMayForceI(raw_helper, ...)
///   ... uses of v3 rewritten to v1 ...
pub(crate) fn unbox_raw_force_results(
    mut ops: Vec<Op>,
    constants: &HashMap<u32, i64>,
    raw_force_helpers: &HashSet<i64>,
) -> Vec<Op> {
    use majit_ir::OpCode;

    let force_results: Vec<OpRef> = ops
        .iter()
        .filter(|op| {
            op.opcode == OpCode::CallMayForceI
                && op
                    .args
                    .first()
                    .and_then(|func| constants.get(&func.0))
                    .is_some_and(|ptr| raw_force_helpers.contains(ptr))
        })
        .map(|op| op.pos)
        .collect();

    if force_results.is_empty() {
        return ops;
    }

    for force_ref in &force_results {
        let mut intval_refs: Vec<(usize, OpRef)> = Vec::new();
        let mut ops_to_remove: Vec<usize> = Vec::new();
        let mut ob_type_refs: Vec<(usize, OpRef)> = Vec::new();

        for (idx, op) in ops.iter().enumerate() {
            if op.opcode != OpCode::GetfieldGcI || op.args.first() != Some(force_ref) {
                continue;
            }
            if let Some(ref d) = op.descr {
                let ds = format!("{d:?}");
                if ds.contains("offset: 8")
                    && ds.contains("field_size: 8")
                    && ds.contains("signed: true")
                {
                    intval_refs.push((idx, op.pos));
                    ops_to_remove.push(idx);
                } else if ds.contains("offset: 0") {
                    ob_type_refs.push((idx, op.pos));
                }
            }
        }

        if !intval_refs.is_empty() {
            for (idx, ob_type_ref) in ob_type_refs {
                ops_to_remove.push(idx);
                for (idx2, op2) in ops.iter().enumerate() {
                    if op2.opcode == OpCode::GuardClass && op2.args.first() == Some(&ob_type_ref) {
                        ops_to_remove.push(idx2);
                    }
                }
            }
        }

        for &(_, intval_ref) in &intval_refs {
            for op in ops.iter_mut() {
                for arg in op.args.iter_mut() {
                    if *arg == intval_ref {
                        *arg = *force_ref;
                    }
                }
                if let Some(ref mut fa) = op.fail_args {
                    for arg in fa.iter_mut() {
                        if *arg == intval_ref {
                            *arg = *force_ref;
                        }
                    }
                }
            }
        }

        ops_to_remove.sort_unstable();
        ops_to_remove.dedup();
        for &idx in ops_to_remove.iter().rev() {
            if idx < ops.len() {
                ops.remove(idx);
            }
        }
    }

    ops
}

/// RPython dependency.py requires GUARD_(NO_)OVERFLOW to be scheduled only
/// when there is a live preceding INT_*_OVF operation to consume.
/// If a late optimizer/retrace path leaves a stray overflow guard in the final
/// trace, drop it conservatively before backend codegen.
pub(crate) fn strip_stray_overflow_guards(ops: Vec<Op>) -> Vec<Op> {
    use majit_ir::OpCode;

    let mut pending_ovf = false;
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        match op.opcode {
            OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf => {
                pending_ovf = true;
                out.push(op);
            }
            OpCode::GuardNoOverflow | OpCode::GuardOverflow => {
                if pending_ovf {
                    pending_ovf = false;
                    out.push(op);
                }
            }
            OpCode::Label | OpCode::Jump | OpCode::Finish => {
                pending_ovf = false;
                out.push(op);
            }
            _ => out.push(op),
        }
    }
    out
}

/// Fold boxing into create_frame: when a box helper result feeds directly
/// into create_frame as the last argument, replace with create_frame_raw_int.
///
/// Pattern: `vB = CallI(box_fn, raw) ... vF = CallI(create_frame, ..., vB)`
/// Result:  `vF = CallI(create_frame_raw, ..., raw)` + remove vB
pub(crate) fn fold_box_into_create_frame(
    mut ops: Vec<Op>,
    constants: &mut HashMap<u32, i64>,
    box_helpers: &HashSet<i64>,
    create_frame_raw_map: &HashMap<i64, i64>,
) -> Vec<Op> {
    use majit_ir::OpCode;

    #[derive(Clone)]
    struct Replacement {
        create_idx: usize,
        boxed_ref: OpRef,
        raw_val: OpRef,
        removable_indices: Vec<usize>,
    }

    if box_helpers.is_empty() || create_frame_raw_map.is_empty() {
        return ops;
    }

    let mut replacements: Vec<Replacement> = Vec::new();

    for (ci, create_op) in ops.iter().enumerate() {
        if create_op.opcode != OpCode::CallI {
            continue;
        }
        // Check if this is a known create_frame helper
        let create_fn_ptr = create_op
            .args
            .first()
            .and_then(|func| constants.get(&func.0))
            .copied();
        let raw_fn = create_fn_ptr.and_then(|p| create_frame_raw_map.get(&p).copied());
        let Some(raw_fn_ptr) = raw_fn else {
            continue;
        };

        // Last arg of create_frame should be a boxed value
        let last_arg = match create_op.args.last() {
            Some(a) => *a,
            None => continue,
        };

        let mut replacement = None;

        // Pattern 1: CallI(box_fn, raw)
        for (bi, box_op) in ops[..ci].iter().enumerate().rev() {
            if box_op.pos != last_arg {
                continue;
            }
            if box_op.opcode == OpCode::CallI {
                let box_fn_ptr = box_op
                    .args
                    .first()
                    .and_then(|func| constants.get(&func.0))
                    .copied();
                if box_fn_ptr.is_some_and(|p| box_helpers.contains(&p)) && box_op.args.len() >= 2 {
                    replacement = Some(Replacement {
                        create_idx: ci,
                        boxed_ref: last_arg,
                        raw_val: box_op.args[1],
                        removable_indices: vec![bi],
                    });
                }
                break;
            }

            // Pattern 2: New() + SetfieldGc(box, raw)
            if box_op.opcode == OpCode::New {
                let mut raw_val = None;
                let mut removable_indices = vec![bi];
                for (si, set_op) in ops[bi + 1..ci].iter().enumerate() {
                    let idx = bi + 1 + si;
                    if set_op.opcode != OpCode::SetfieldGc || set_op.args.first() != Some(&last_arg)
                    {
                        continue;
                    }
                    removable_indices.push(idx);
                    if let Some(ref d) = set_op.descr {
                        let ds = format!("{d:?}");
                        if ds.contains("offset: 8") && ds.contains("signed: true") {
                            raw_val = set_op.args.get(1).copied();
                        }
                    }
                }
                if let Some(raw_val) = raw_val {
                    replacement = Some(Replacement {
                        create_idx: ci,
                        boxed_ref: last_arg,
                        raw_val,
                        removable_indices,
                    });
                }
                break;
            }
        }

        if let Some(repl) = replacement {
            replacements.push(repl);
        }
    }

    // Apply replacements in reverse order
    for repl in replacements.iter().rev() {
        let create_fn_ptr = match ops[repl.create_idx]
            .args
            .first()
            .and_then(|func| constants.get(&func.0))
            .copied()
        {
            Some(p) => p,
            None => continue,
        };
        let raw_fn_ptr = match create_frame_raw_map.get(&create_fn_ptr) {
            Some(&p) => p,
            None => continue, // already replaced in a previous iteration
        };

        // Replace last arg of create_frame with raw_val
        let nargs = ops[repl.create_idx].args.len();
        ops[repl.create_idx].args[nargs - 1] = repl.raw_val;

        // Replace function pointer: create_frame → create_frame_raw_int
        let func_ref = ops[repl.create_idx].args[0];
        constants.insert(func_ref.0, raw_fn_ptr);

        // Remove the boxing ops only if they are now dead. The raw helper
        // rewrite is still valuable even when the boxed object remains live
        // for later virtual field reads.
        let still_used = ops.iter().enumerate().any(|(i, op)| {
            if repl.removable_indices.contains(&i) || i == repl.create_idx {
                return false;
            }
            op.args.contains(&repl.boxed_ref)
                || op
                    .fail_args
                    .as_ref()
                    .is_some_and(|fa| fa.contains(&repl.boxed_ref))
        });
        if !still_used {
            for &idx in repl.removable_indices.iter().rev() {
                ops.remove(idx);
            }
        }
    }

    ops
}

/// RPython parity: elide create_frame + drop_frame around CallAssemblerI
/// for self-recursive raw-int calls. The callee frame is created lazily
/// on guard failure (in force_fn), not before every recursive call.
///
/// Pattern:
///   v_frame = CallI(create_frame_raw_fn, caller_frame, raw_arg)
///   v_result = CallAssemblerI(v_frame, 0, 1, raw_arg)
///   CallN(drop_frame_fn, v_frame)
///
/// Result:
///   v_result = CallAssemblerI(caller_frame, 0, 1, raw_arg)
///   (CallI and CallN removed)
///
/// Guard failure path: force_fn receives caller_frame from inputs[0]
/// and raw_arg from inputs[3], creates callee frame lazily.
pub(crate) fn elide_create_frame_for_call_assembler(
    ops: Vec<Op>,
    _constants: &HashMap<u32, i64>,
    _create_frame_raw_map: &HashMap<i64, i64>,
) -> Vec<Op> {
    // RPython parity: elide frame creation for self-recursive calls.
    // Force_fn uses PENDING_FORCE_LOCAL0 for lazy callee frame creation.

    #[allow(unreachable_code)]
    {
        use majit_ir::OpCode;

        let mut ops = ops;
        let constants = _constants;
        let create_frame_raw_map = _create_frame_raw_map;

        if create_frame_raw_map.is_empty() {
            return ops;
        }

        // Collect all create_frame function pointers (keys AND values of the map)
        // Keys = original create_frame, Values = create_frame_raw variant
        let raw_frame_fns: HashSet<i64> = create_frame_raw_map
            .keys()
            .copied()
            .chain(create_frame_raw_map.values().copied())
            .collect();

        // Find CallI(create_frame_raw) → CallAssemblerI → CallN(drop) triples
        let mut removals: Vec<(usize, usize, OpRef, OpRef)> = Vec::new(); // (create_idx, drop_idx, frame_ref, caller_frame)

        for (ci, op) in ops.iter().enumerate() {
            if !matches!(op.opcode, OpCode::CallI | OpCode::CallR) {
                continue;
            }
            let fn_ptr = op.args.first().and_then(|f| constants.get(&f.0)).copied();
            let Some(fn_ptr) = fn_ptr else { continue };
            if !raw_frame_fns.contains(&fn_ptr) {
                continue;
            }
            // This is a create_frame_raw call
            let frame_ref = op.pos; // v_frame
            let caller_frame = if op.args.len() >= 2 {
                op.args[1]
            } else {
                continue;
            };

            // Find the CallAssemblerI that uses frame_ref as first arg
            let ca_idx = ops[ci + 1..].iter().position(|o| {
                matches!(
                    o.opcode,
                    OpCode::CallAssemblerI
                        | OpCode::CallAssemblerR
                        | OpCode::CallAssemblerF
                        | OpCode::CallAssemblerN
                ) && o.args.first() == Some(&frame_ref)
            });
            let Some(ca_offset) = ca_idx else { continue };
            let ca_idx = ci + 1 + ca_offset;

            // Find the CallN(drop_frame) that uses frame_ref
            let drop_idx = ops[ca_idx + 1..].iter().position(|o| {
                o.opcode == OpCode::CallN && o.args.len() >= 2 && o.args[1] == frame_ref
            });
            let Some(drop_offset) = drop_idx else {
                continue;
            };
            let drop_idx = ca_idx + 1 + drop_offset;

            // Check frame_ref is not used elsewhere (only in create, CA, drop)
            let other_use = ops.iter().enumerate().any(|(i, o)| {
                if i == ci || i == ca_idx || i == drop_idx {
                    return false;
                }
                o.args.contains(&frame_ref)
                    || o.fail_args
                        .as_ref()
                        .is_some_and(|fa| fa.contains(&frame_ref))
            });
            if other_use {
                continue;
            }

            removals.push((ci, drop_idx, frame_ref, caller_frame));
        }

        // Apply in reverse order
        for (create_idx, drop_idx, frame_ref, caller_frame) in removals.iter().rev() {
            // Replace CallAssemblerI's first arg: frame_ref → caller_frame
            for op in ops.iter_mut() {
                if matches!(
                    op.opcode,
                    OpCode::CallAssemblerI
                        | OpCode::CallAssemblerR
                        | OpCode::CallAssemblerF
                        | OpCode::CallAssemblerN
                ) && op.args.first() == Some(frame_ref)
                {
                    op.args[0] = *caller_frame;
                }
            }
            // Remove drop first (higher index), then create
            ops.remove(*drop_idx);
            // Adjust create_idx if drop was before it (shouldn't happen)
            let ci = if *drop_idx < *create_idx {
                create_idx - 1
            } else {
                *create_idx
            };
            ops.remove(ci);
        }

        ops
    }
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
    backend: &mut majit_codegen_cranelift::CraneliftBackend,
    token: &majit_codegen::JitCellToken,
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
    backend: &mut majit_codegen_cranelift::CraneliftBackend,
    token: &majit_codegen::JitCellToken,
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
    use majit_ir::{Op, OpCode, OpRef};

    #[test]
    fn test_retag_fail_descrs_from_trace_types_rebuilds_dense_exit_numbering() {
        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(1)]);
        guard.descr = Some(make_fail_descr_with_index(42, 2));
        guard.fail_args = Some(smallvec::smallvec![OpRef(0), OpRef(1)]);

        let mut finish = Op::new(OpCode::Finish, &[OpRef(0), OpRef(1)]);
        finish.descr = Some(make_fail_descr_with_index(77, 2));

        let mut ops = vec![Op::new(OpCode::Label, &[OpRef(0), OpRef(1)]), guard, finish];
        retag_fail_descrs_from_trace_types(&inputargs, &mut ops);

        assert_eq!(find_fail_index_for_exit_op(&ops, 1), Some(0));
        assert_eq!(find_fail_index_for_exit_op(&ops, 2), Some(1));
        assert_eq!(
            ops[1]
                .descr
                .as_ref()
                .unwrap()
                .as_fail_descr()
                .unwrap()
                .fail_arg_types(),
            &[Type::Ref, Type::Int]
        );
        assert_eq!(
            ops[2]
                .descr
                .as_ref()
                .unwrap()
                .as_fail_descr()
                .unwrap()
                .fail_arg_types(),
            &[Type::Ref, Type::Int]
        );
    }
}
