/// Loop unrolling pass (peel one iteration).
///
/// Detects loops ending with a `Jump` back-edge and peels one iteration:
/// the loop body is duplicated to create a "preamble" that executes once
/// before the main loop. Guards in the peeled preamble serve as initial
/// type checks, enabling downstream passes to remove redundant guards
/// from the main loop body.
///
/// The peeled structure looks like:
///
/// ```text
/// [peeled body]          ← first iteration (preamble), guards act as type checks
///   Label(...)           ← loop header
/// [original body]        ← main loop body
///   Jump(...)            ← back-edge to Label
/// ```
///
/// OpRefs in the peeled iteration are remapped to new positions so they
/// don't collide with the original ops.
use std::collections::HashMap;

use majit_ir::{GcRef, Op, OpCode, OpRef, Type, Value};

use crate::{OptContext, Optimization, OptimizationResult};

/// unroll.py: UnrollOptimizer — high-level loop optimization controller.
///
/// Wraps the streaming OptUnroll pass with RPython's UnrollOptimizer API:
/// - optimize_preamble: process and optimize the first iteration
/// - optimize_peeled_loop: optimize the main loop body
/// - optimize_bridge: compile a bridge into an unrolled loop
pub struct UnrollOptimizer {
    /// Maximum retrace attempts for a single loop location.
    /// unroll.py: retrace_limit from WarmEnterState.
    pub retrace_limit: u32,
    /// Number of guards in the bridge (for retrace_limit checks).
    pub bridge_guard_count: u32,
    /// The short preamble from the preamble optimization pass.
    pub short_preamble: Option<crate::shortpreamble::ShortPreamble>,
    /// The exported virtual state at the loop header.
    pub exported_state: Option<crate::virtualstate::VirtualState>,
    /// history.py: JitCellToken.target_tokens — compiled versions of this loop.
    /// Each TargetToken has its own virtual state and short preamble.
    pub target_tokens: Vec<TargetToken>,
}

impl UnrollOptimizer {
    pub fn new() -> Self {
        UnrollOptimizer {
            retrace_limit: 0, // rlib/jit.py:595 default
            bridge_guard_count: 0,
            short_preamble: None,
            exported_state: None,
            target_tokens: Vec::new(),
        }
    }

    /// unroll.py: optimize_preamble(trace, runtime_boxes)
    /// Optimize the preamble (first iteration) of a loop trace.
    /// Returns the optimized preamble ops + the peeled loop ops.
    pub fn optimize_preamble(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut optimizer = crate::optimizer::Optimizer::default_pipeline();
        optimizer.add_pass(Box::new(OptUnroll::new()));
        optimizer.optimize(ops)
    }

    /// unroll.py: optimize_peeled_loop(trace)
    /// Optimize the loop body AFTER preamble peeling.
    /// The peeled preamble has already established the type/class/bounds
    /// information; this method optimizes the repeating body.
    pub fn optimize_peeled_loop(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut optimizer = crate::optimizer::Optimizer::default_pipeline();
        optimizer.optimize(ops)
    }

    /// unroll.py: optimize_bridge(trace, runtime_boxes, call_pure_results)
    /// unroll.py: optimize_bridge(trace, runtime_boxes, call_pure_results)
    /// Optimize a bridge trace that enters an existing loop.
    ///
    /// If a short preamble is available, its ops are prepended to the bridge
    /// so the optimizer can re-establish the invariants that the loop body
    /// depends on (type checks, bounds, cached values).
    pub fn optimize_bridge(&mut self, bridge_ops: &[Op]) -> Vec<Op> {
        self.optimize_bridge_with_label_args(bridge_ops, &[])
    }

    /// Optimize a bridge with explicit label args for short preamble instantiation.
    pub fn optimize_bridge_with_label_args(
        &mut self,
        bridge_ops: &[Op],
        label_args: &[OpRef],
    ) -> Vec<Op> {
        let mut optimizer = crate::optimizer::Optimizer::default_pipeline();
        let mut full_ops = Vec::new();

        // unroll.py: inline short preamble ops at the start of the bridge.
        if let Some(ref sp) = self.short_preamble {
            if !sp.is_empty() && !label_args.is_empty() {
                let preamble_ops = sp.instantiate(label_args);
                full_ops.extend(preamble_ops);
            }
        }

        full_ops.extend_from_slice(bridge_ops);
        optimizer.optimize(&full_ops)
    }

    /// Whether we've exceeded the retrace limit for this loop location.
    /// unroll.py: checks retrace_limit in optimize_bridge.
    pub fn should_give_up(&self, retrace_count: u32) -> bool {
        retrace_count >= self.retrace_limit
    }

    /// unroll.py: export_state(target_token)
    /// Export the virtual state at the current loop header.
    pub fn set_exported_state(&mut self, state: crate::virtualstate::VirtualState) {
        self.exported_state = Some(state);
    }

    /// unroll.py: get_virtual_state()
    /// Get the exported virtual state for this loop.
    pub fn get_exported_state(&self) -> Option<&crate::virtualstate::VirtualState> {
        self.exported_state.as_ref()
    }

    /// unroll.py: set_short_preamble(sp)
    pub fn set_short_preamble(&mut self, sp: crate::shortpreamble::ShortPreamble) {
        self.short_preamble = Some(sp);
    }

    /// unroll.py: get_short_preamble()
    pub fn get_short_preamble(&self) -> Option<&crate::shortpreamble::ShortPreamble> {
        self.short_preamble.as_ref()
    }

    /// unroll.py:238-242: jump_to_preamble — redirect body JUMP to preamble.
    ///
    /// When jump_to_existing_trace fails (no compatible target_token),
    /// the body JUMP goes back to the preamble for the next iteration.
    /// This ensures the preamble guard provides loop exit.
    ///
    /// Returns modified body ops with JUMP retargeted to the preamble.
    /// RPython keeps the arglist intact and only changes the jump target.
    pub fn jump_to_preamble(body_ops: &[Op], _preamble_num_inputs: usize) -> Vec<Op> {
        body_ops.to_vec()
    }

    /// unroll.py: check_retrace_count(retrace_count, max_retrace_guards)
    /// Whether this bridge has too many guards for retrace to be worthwhile.
    pub fn too_many_guards_for_retrace(&self, retrace_count: u32, max_retrace_guards: u32) -> bool {
        if retrace_count >= self.retrace_limit {
            return true;
        }
        self.bridge_guard_count > max_retrace_guards
    }

    /// unroll.py: optimize_trace(trace, call_pure_results)
    /// Full trace optimization: peel → optimize preamble → optimize body.
    /// Returns the optimized peeled+body trace.
    pub fn optimize_trace(&mut self, ops: &[Op]) -> Vec<Op> {
        let result = self.optimize_preamble(ops);
        // After peeling, extract short preamble from the result.
        let sp = crate::shortpreamble::extract_short_preamble(&result);
        if !sp.is_empty() {
            self.short_preamble = Some(sp);
        }
        result
    }

    /// unroll.py: optimize_trace_with_constants
    /// Same as optimize_trace but with known constants.
    pub fn optimize_trace_with_constants(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
    ) -> Vec<Op> {
        let mut optimizer = crate::optimizer::Optimizer::default_pipeline();
        optimizer.add_pass(Box::new(OptUnroll::new()));
        let result = optimizer.optimize_with_constants(ops, constants);
        let sp = crate::shortpreamble::extract_short_preamble(&result);
        if !sp.is_empty() {
            self.short_preamble = Some(sp);
        }
        result
    }

    /// optimize_trace with constants AND explicit num_inputs.
    /// compile.py: compile_loop → optimize with preamble peeling.
    pub fn optimize_trace_with_constants_and_inputs(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
    ) -> (Vec<Op>, usize) {
        self.optimize_trace_with_constants_and_inputs_vable(ops, constants, num_inputs, None)
    }

    /// compile.py:275-308: compile_loop — 2-phase preamble peeling.
    /// compile.py:275-338: 2-phase preamble peeling (RPython parity).
    ///
    /// Phase 1 (optimize_preamble): full pipeline on trace → preamble_ops.
    /// export_state: capture the preamble's exported optimizer state.
    /// Phase 2 (optimize_peeled_loop): import_state + full pipeline → body_ops.
    /// Assembly: [preamble_no_jump] + Label(label_args) + [body_with_jump].
    pub fn optimize_trace_with_constants_and_inputs_vable(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
        vable_config: Option<crate::virtualize::VirtualizableConfig>,
    ) -> (Vec<Op>, usize) {
        // ── Phase 1: PreambleCompileData.optimize() ──
        // ── Phase 1: optimize_preamble (compile.py:275-276) ──
        let mut consts_p1 = constants.clone();
        let mut opt_p1 = match vable_config.as_ref() {
            Some(c) => crate::optimizer::Optimizer::default_pipeline_with_virtualizable(c.clone()),
            None => crate::optimizer::Optimizer::default_pipeline(),
        };
        let p1_ops = opt_p1.optimize_with_constants_and_inputs(ops, &mut consts_p1, num_inputs);
        let p1_ni = opt_p1.final_num_inputs();
        let jump_virtuals = std::mem::take(&mut opt_p1.exported_jump_virtuals);

        if jump_virtuals.is_empty() {
            *constants = consts_p1;
            let sp = opt_p1
                .exported_loop_state
                .as_ref()
                .map(|state| {
                    crate::shortpreamble::build_short_preamble_from_exported_boxes(
                        &state.end_args,
                        &state.exported_short_boxes,
                    )
                })
                .unwrap_or_else(|| crate::shortpreamble::extract_short_preamble(&p1_ops));
            // Store TargetToken even for non-peeled loops (for bridge reuse)
            if let Some(ref es) = opt_p1.exported_loop_state {
                let opt_unroll = OptUnroll::new();
                let tt = opt_unroll.finalize_short_preamble(
                    self.target_tokens.len() as u64,
                    es.virtual_state.clone(),
                    sp.clone(),
                    None,
                );
                self.target_tokens.push(tt);
            }
            if !sp.is_empty() { self.short_preamble = Some(sp); }
            return (p1_ops, p1_ni);
        }

        let exported_state = match opt_p1.exported_loop_state.clone() {
            Some(state) => state,
            None => {
                *constants = consts_p1;
                return (p1_ops, p1_ni);
            }
        };
        // RPython import_state/export_state preserve the original loop-header
        // contract. Virtual structure is restored through VirtualState, not by
        // flattening jump args into field-value slots.
        let exported_label_args = exported_state.end_args.clone();

        // ── Phase 2: optimize_peeled_loop (compile.py:291-292) ──
        // RPython import_state: phase 2 starts from the original trace input
        // contract. The optimizer state is imported from ExportedState; the
        // trace itself is not rewritten to add synthetic field inputargs.
        let body_num_inputs = exported_label_args.len();

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] preamble peeling: {} virtual(s), exported label_args={}",
                jump_virtuals.len(), exported_state.end_args.len(),
            );
        }

        let remapped_ops = ops.to_vec();
        let mut consts_p2 = consts_p1.clone();

        let mut opt_p2 = match vable_config.as_ref() {
            Some(c) => crate::optimizer::Optimizer::default_pipeline_with_virtualizable(c.clone()),
            None => crate::optimizer::Optimizer::default_pipeline(),
        };
        opt_p2.imported_loop_state = Some(exported_state.clone());

        if std::env::var_os("MAJIT_LOG").is_some() {
            let gc_before = remapped_ops.iter().filter(|o| o.opcode.is_guard()).count();
            eprintln!("[jit] phase 2 input: {} ops, {} guards, body_ni={}", remapped_ops.len(), gc_before, body_num_inputs);
        }

        let p2_ops = opt_p2.optimize_with_constants_and_inputs(
            &remapped_ops, &mut consts_p2, body_num_inputs,
        );
        let p2_ni = opt_p2.final_num_inputs();

        if std::env::var_os("MAJIT_LOG").is_some() {
            let nc = p2_ops.iter()
                .filter(|o| o.opcode == OpCode::New || o.opcode == OpCode::NewWithVtable).count();
            let gc = p2_ops.iter().filter(|o| o.opcode.is_guard()).count();
            eprintln!(
                "[jit] phase 2: {} ops, {} New, {} guards, p2_ni={}",
                p2_ops.len(),
                nc,
                gc,
                p2_ni
            );
        }

        // ── unroll.py:140-175: finalize + jump_to_existing_trace ──
        // Build the virtual state at end of Phase 2 body.
        let p2_exported = opt_p2.exported_loop_state.clone();
        let imported_short_preamble_builder = opt_p2.imported_short_preamble_builder.clone();
        let imported_short_aliases = opt_p2.imported_short_aliases.clone();
        let imported_short_sources = opt_p2.imported_short_sources.clone();

        if std::env::var_os("MAJIT_LOG").is_some() {
            let imported_map: HashMap<OpRef, OpRef> = imported_short_sources
                .iter()
                .map(|entry| (entry.result, entry.source))
                .collect();
            let mut defined: std::collections::HashSet<OpRef> =
                (0..body_num_inputs as u32).map(OpRef).collect();
            for op in &p2_ops {
                if !op.pos.is_none() {
                    defined.insert(op.pos);
                }
            }
            let mut unresolved = Vec::new();
            for op in &p2_ops {
                for &arg in &op.args {
                    if arg.is_none() || defined.contains(&arg) {
                        continue;
                    }
                    if unresolved.contains(&arg) {
                        continue;
                    }
                    unresolved.push(arg);
                }
                if let Some(fail_args) = &op.fail_args {
                    for &arg in fail_args {
                        if arg.is_none() || defined.contains(&arg) {
                            continue;
                        }
                        if unresolved.contains(&arg) {
                            continue;
                        }
                        unresolved.push(arg);
                    }
                }
            }
            eprintln!(
                "[jit] phase 2 imported_short_sources={}, unresolved_refs={:?}",
                imported_short_sources.len(),
                unresolved
            );
            for unresolved_ref in unresolved {
                if let Some(source) = imported_map.get(&unresolved_ref) {
                    eprintln!(
                        "[jit] unresolved imported ref {unresolved_ref:?} -> preamble {source:?}"
                    );
                }
            }
        }

        // finalize_short_preamble: create TargetToken for this loop version
        let initial_sp = opt_p2
            .imported_short_preamble
            .clone()
            .unwrap_or_else(|| {
                crate::shortpreamble::build_short_preamble_from_exported_boxes(
                    &exported_state.end_args,
                    &exported_state.exported_short_boxes,
                )
            });
        let opt_unroll = OptUnroll::new();
        let target_token = opt_unroll.finalize_short_preamble(
            self.target_tokens.len() as u64,
            exported_state.virtual_state.clone(),
            initial_sp.clone(),
            imported_short_preamble_builder.as_ref(),
        );
        self.target_tokens.push(target_token);

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] finalize_short_preamble: target_tokens={}",
                self.target_tokens.len()
            );
        }

        // ── unroll.py:151-175: jump_to_existing_trace / jump_to_preamble ──
        // Try to match the body's JUMP virtual state to an existing target.
        // RPython: new_virtual_state = jump_to_existing_trace(end_jump, ...)
        let mut body_ops = p2_ops;
        let jump_to_self = {
            let body_jump_args: Vec<OpRef> = body_ops
                .iter()
                .rfind(|o| o.opcode == OpCode::Jump)
                .map(|j| j.args.to_vec())
                .unwrap_or_default();
            let opt_unroll = OptUnroll::new();
            let mut jump_ctx = crate::OptContext::with_num_inputs(32, body_num_inputs);
            opt_unroll
                .jump_to_existing_trace(&body_jump_args, &mut self.target_tokens, &mut jump_ctx)
                .is_none() // None = jumped successfully
        };

        let sp = self
            .target_tokens
            .last()
            .and_then(|target| target.short_preamble.clone())
            .unwrap_or(initial_sp);
        let rewritten_jump_args = p2_exported.as_ref().map(|s| {
            let mut args = s.end_args.clone();
            args.extend(sp.used_boxes.iter().copied());
            args
        });
        if !sp.is_empty() {
            self.short_preamble = Some(sp.clone());
        }

        if !jump_to_self {
            // unroll.py:170-171: jump_to_preamble — body JUMP → preamble Label
            body_ops = Self::jump_to_preamble(&body_ops, num_inputs);
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!("[jit] jump_to_preamble: body JUMP preserved");
            }
        } else if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("[jit] jump_to_existing_trace: body JUMP → self-loop");
        }

        // ── Assembly (compile.py:310-338) ──
        let combined = assemble_peeled_trace(
            &p1_ops,
            &body_ops,
            &exported_label_args,
            &sp.used_boxes,
            rewritten_jump_args.as_deref(),
            p2_ni,
            &imported_short_aliases,
            &imported_short_sources,
        );
        *constants = consts_p2;
        (combined, p2_ni)
    }

    /// Count the guards in an optimized trace (for retrace_limit checks).
    pub fn count_guards(ops: &[Op]) -> u32 {
        ops.iter().filter(|op| op.opcode.is_guard()).count() as u32
    }

    /// unroll.py: _map_args(mapping, arglist)
    /// Remap a list of OpRefs through a forwarding mapping.
    /// Constants (OpRef >= 10000) are left unchanged.
    pub fn map_args(
        mapping: &std::collections::HashMap<OpRef, OpRef>,
        args: &[OpRef],
    ) -> Vec<OpRef> {
        args.iter()
            .map(|&arg| {
                if arg.0 >= 10_000 {
                    arg // constant, keep as-is
                } else {
                    mapping.get(&arg).copied().unwrap_or(arg)
                }
            })
            .collect()
    }

    /// unroll.py: _check_no_forwarding(lsts)
    /// Debug assertion: verify no OpRef in the lists has been forwarded.
    pub fn check_no_forwarding(ctx: &crate::OptContext, oprefs: &[OpRef]) -> bool {
        oprefs.iter().all(|&r| ctx.get_replacement(r) == r)
    }


    /// unroll.py: disable_retracing_if_max_retrace_guards(ops, target_token)
    /// If the trace has too many guards, disable retracing for this location.
    /// Returns true if retracing was disabled.
    pub fn disable_retracing_if_max_retrace_guards(ops: &[Op], max_retrace_guards: u32) -> bool {
        let guard_count = Self::count_guards(ops);
        guard_count > max_retrace_guards
    }

    /// unroll.py: get_virtual_state(args)
    /// Build a VirtualState from the optimizer's current knowledge about args.
    pub fn get_virtual_state(
        args: &[OpRef],
        ctx: &crate::OptContext,
        ptr_info: &[Option<crate::info::PtrInfo>],
    ) -> crate::virtualstate::VirtualState {
        crate::virtualstate::export_state(args, ctx, ptr_info)
    }
}

impl Default for UnrollOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// unroll.py: ExportedState — snapshot of optimizer state at the end of
/// the preamble, used to initialize the peeled loop body.
///
/// Contains the virtual state, short preamble boxes, arg mappings,
/// and exported infos needed to resume optimization after peeling.
#[derive(Clone, Debug)]
pub struct ExportedState {
    /// Label args at the end of the preamble (after forcing).
    pub end_args: Vec<OpRef>,
    /// Args for the next iteration (before forcing).
    pub next_iteration_args: Vec<OpRef>,
    /// Virtual state at the loop boundary.
    pub virtual_state: crate::virtualstate::VirtualState,
    /// unroll.py: exported_infos — optimizer knowledge from preamble.
    /// Maps OpRef → info for all args including virtual field contents.
    pub exported_infos: HashMap<OpRef, ExportedValueInfo>,
    /// RPython unroll.py: short_boxes exported from the preamble.
    /// Kept in a compact form that phase 2 can translate back into imported
    /// heap cache facts.
    pub exported_short_ops: Vec<ExportedShortOp>,
    /// RPython shortpreamble.py: produced short boxes in preamble order.
    /// This preserves the original preamble ops so the active path can build
    /// short preambles without re-extracting them from the peeled trace.
    pub exported_short_boxes: Vec<crate::shortpreamble::PreambleOp>,
    /// Short preamble builder for bridge entry.
    pub short_preamble: Option<crate::shortpreamble::ShortPreamble>,
    /// Renamed inputargs from the preamble.
    pub renamed_inputargs: Vec<OpRef>,
    /// Short inputargs for the short preamble.
    pub short_inputargs: Vec<OpRef>,
}

#[derive(Clone, Debug, Default)]
pub struct ExportedValueInfo {
    /// A constant carried by this slot.
    pub constant: Option<Value>,
    /// Full pointer info graph exported from the preamble.
    pub ptr_info: Option<crate::info::PtrInfo>,
    /// Non-virtual pointer shape restored from the preamble.
    pub ptr_kind: ExportedPtrKind,
    /// Descriptor for non-virtual pointer shapes.
    pub ptr_descr: Option<majit_ir::DescrRef>,
    /// Known class carried by this slot.
    pub known_class: Option<GcRef>,
    /// Array length knowledge for ArrayPtrInfo.
    pub array_lenbound: Option<crate::intutils::IntBound>,
    /// Widened integer knowledge restored into phase 2.
    pub int_bound: Option<crate::intutils::IntBound>,
    /// Whether this slot is known non-null.
    pub nonnull: bool,
    /// Lower bound learned for this integer slot.
    pub int_lower_bound: Option<i64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExportedPtrKind {
    #[default]
    None,
    Instance,
    Struct,
    Array,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExportedShortOp {
    Pure {
        source: OpRef,
        opcode: OpCode,
        descr_idx: Option<u32>,
        args: Vec<ExportedShortArg>,
        result: ExportedShortResult,
        invented_name: bool,
        same_as_source: Option<OpRef>,
    },
    HeapField {
        source: OpRef,
        object_slot: usize,
        descr_idx: u32,
        result_type: Type,
        result: ExportedShortResult,
        invented_name: bool,
        same_as_source: Option<OpRef>,
    },
    HeapArrayItem {
        source: OpRef,
        object_slot: usize,
        descr_idx: u32,
        index: i64,
        result_type: Type,
        result: ExportedShortResult,
        invented_name: bool,
        same_as_source: Option<OpRef>,
    },
    LoopInvariant {
        source: OpRef,
        func_ptr: i64,
        result_type: Type,
        result: ExportedShortResult,
        invented_name: bool,
        same_as_source: Option<OpRef>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExportedShortArg {
    Slot(usize),
    Const(Value),
    Produced(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExportedShortResult {
    Slot(usize),
    Temporary(usize),
}

impl ExportedState {
    /// unroll.py: ExportedState.__init__
    pub fn new(
        end_args: Vec<OpRef>,
        next_iteration_args: Vec<OpRef>,
        virtual_state: crate::virtualstate::VirtualState,
        exported_infos: HashMap<OpRef, ExportedValueInfo>,
        exported_short_ops: Vec<ExportedShortOp>,
        exported_short_boxes: Vec<crate::shortpreamble::PreambleOp>,
        renamed_inputargs: Vec<OpRef>,
        short_inputargs: Vec<OpRef>,
    ) -> Self {
        ExportedState {
            end_args,
            next_iteration_args,
            virtual_state,
            exported_infos,
            exported_short_ops,
            exported_short_boxes,
            short_preamble: None,
            renamed_inputargs,
            short_inputargs,
        }
    }

    /// unroll.py: final() — ExportedState is never final (loop continues).
    pub fn is_final(&self) -> bool {
        false
    }
}

/// history.py: TargetToken — describes one compiled version of a loop.
///
/// Each peeled loop body creates a TargetToken that records the virtual state
/// and short preamble needed for bridge entry. Multiple TargetTokens can exist
/// per loop (from retracing with different virtual states).
#[derive(Clone, Debug)]
pub struct TargetToken {
    /// RPython history.py: identity of this target token within the current
    /// JitCellToken.target_tokens list.
    pub token_id: u64,
    /// Virtual state at this loop entry point.
    /// Used by _jump_to_existing_trace to check compatibility.
    pub virtual_state: Option<crate::virtualstate::VirtualState>,
    /// Short preamble: ops to replay when entering from a bridge.
    pub short_preamble: Option<crate::shortpreamble::ShortPreamble>,
    /// RPython unroll.py: active ExtendedShortPreambleBuilder for the target
    /// token currently being finalized.
    pub short_preamble_producer: Option<crate::shortpreamble::ExtendedShortPreambleBuilder>,
    /// The exported state from the preamble (for retracing).
    pub exported_state: Option<ExportedState>,
    /// Number of times this target has been retraced.
    pub retraced_count: u32,
}

impl TargetToken {
    pub fn new() -> Self {
        TargetToken {
            token_id: 0,
            virtual_state: None,
            short_preamble: None,
            short_preamble_producer: None,
            exported_state: None,
            retraced_count: 0,
        }
    }
}

/// unroll.py: UnrollInfo(BasicLoopInfo) — return type from optimize_peeled_loop.
///
/// Carries the target_token, label_op, and extra_same_as needed to
/// finalize compilation after the peeled loop body is optimized.
#[derive(Clone, Debug)]
pub struct UnrollInfo {
    /// The target token for this loop's entry point.
    pub target_token: u64,
    /// Extra same_as ops added during finalization.
    pub extra_same_as: Vec<Op>,
    /// Quasi-immutable dependencies discovered during optimization.
    pub quasi_immutable_deps: std::collections::HashSet<u64>,
    /// Extra ops to insert before the label (from bridge inlining).
    pub extra_before_label: Vec<Op>,
}

impl UnrollInfo {
    /// unroll.py: final() — UnrollInfo is always final.
    pub fn is_final(&self) -> bool {
        true
    }
}

impl OptUnroll {
    /// unroll.py: export_state — capture optimizer state at end of preamble.
    ///
    /// After the preamble is optimized, snapshot:
    /// - end_args: forced versions of label args
    /// - virtual_state: abstract info for loop-carried values
    /// - short boxes: mapping of preamble ops to label args
    pub fn export_state(
        &self,
        original_label_args: &[OpRef],
        renamed_inputargs: &[OpRef],
        ctx: &OptContext,
    ) -> ExportedState {
        self.export_state_with_bounds(original_label_args, renamed_inputargs, ctx, None)
    }

    /// unroll.py:452-477: export_state implementation.
    fn export_state_with_bounds(
        &self,
        original_label_args: &[OpRef],
        renamed_inputargs: &[OpRef],
        ctx: &OptContext,
        exported_int_bounds: Option<&HashMap<OpRef, crate::intutils::IntBound>>,
    ) -> ExportedState {
        // unroll.py:454: end_args = [force_box_for_end_of_preamble(a) ...]
        let end_args: Vec<OpRef> = original_label_args
            .iter()
            .map(|&a| ctx.get_replacement(a))
            .collect();
        // unroll.py:457: use pre-force virtual state if available
        let virtual_state = ctx
            .pre_force_virtual_state
            .clone()
            .unwrap_or_else(|| crate::virtualstate::export_state(&end_args, ctx, &ctx.ptr_info));
        // unroll.py:459-461: infos = {}; for arg in end_args: _expand_info(arg, infos)
        let mut infos: HashMap<OpRef, ExportedValueInfo> = HashMap::new();
        for &arg in &end_args {
            self.expand_info(arg, ctx, exported_int_bounds, &mut infos);
        }
        // unroll.py:462-469: use pre-force args for make_inputargs
        let vs_args = ctx.pre_force_jump_args.as_deref().unwrap_or(&end_args);
        let (label_args, virtuals) = virtual_state.make_inputargs_and_virtuals(vs_args, ctx);
        // unroll.py:464-465: for arg in label_args: _expand_info(arg, infos)
        for &arg in &label_args {
            self.expand_info(arg, ctx, exported_int_bounds, &mut infos);
        }
        let mut short_args = label_args.clone();
        short_args.extend(virtuals);
        let exported_short_ops = self.collect_exported_short_ops(&short_args, ctx);
        let exported_short_boxes = ctx.exported_short_boxes.clone();

        ExportedState::new(
            label_args.clone(),
            end_args,
            virtual_state,
            infos,
            exported_short_ops,
            exported_short_boxes,
            renamed_inputargs.to_vec(),
            short_args,
        )
    }

    /// unroll.py:432-443: _expand_info
    fn expand_info(
        &self,
        arg: OpRef,
        ctx: &OptContext,
        exported_int_bounds: Option<&HashMap<OpRef, crate::intutils::IntBound>>,
        infos: &mut HashMap<OpRef, ExportedValueInfo>,
    ) {
        let resolved = ctx.get_replacement(arg);
        if infos.contains_key(&resolved) {
            return;
        }
        let info = self.collect_exported_info(resolved, ctx, exported_int_bounds);
        let is_virtual = matches!(ctx.get_ptr_info(resolved), Some(pi) if pi.is_virtual());
        infos.insert(resolved, info);
        if is_virtual {
            self.expand_infos_from_virtual(resolved, ctx, exported_int_bounds, infos);
        }
    }

    /// unroll.py:445-450: _expand_infos_from_virtual
    fn expand_infos_from_virtual(
        &self,
        opref: OpRef,
        ctx: &OptContext,
        exported_int_bounds: Option<&HashMap<OpRef, crate::intutils::IntBound>>,
        infos: &mut HashMap<OpRef, ExportedValueInfo>,
    ) {
        let fields: Vec<OpRef> = match ctx.get_ptr_info(opref) {
            Some(crate::info::PtrInfo::Virtual(v)) => v.fields.iter().map(|(_, r)| *r).collect(),
            Some(crate::info::PtrInfo::VirtualStruct(v)) => {
                v.fields.iter().map(|(_, r)| *r).collect()
            }
            Some(crate::info::PtrInfo::VirtualArray(v)) => v.items.clone(),
            _ => return,
        };
        for field in fields {
            if field.is_none() {
                continue;
            }
            self.expand_info(field, ctx, exported_int_bounds, infos);
        }
    }

    /// unroll.py:284-301: finalize_short_preamble — create a TargetToken
    /// and attach the short preamble to it. Called at the end of
    /// optimize_peeled_loop after the loop body is optimized.
    ///
    /// Returns the new TargetToken with virtual_state and short_preamble set.
    pub fn finalize_short_preamble(
        &self,
        token_id: u64,
        virtual_state: crate::virtualstate::VirtualState,
        short_preamble: crate::shortpreamble::ShortPreamble,
        short_preamble_builder: Option<&crate::shortpreamble::ShortPreambleBuilder>,
    ) -> TargetToken {
        let mut target_token = TargetToken::new();
        target_token.token_id = token_id;
        target_token.virtual_state = Some(virtual_state);
        target_token.short_preamble = Some(short_preamble);
        target_token.short_preamble_producer = short_preamble_builder.map(|builder| {
            crate::shortpreamble::ExtendedShortPreambleBuilder::new(token_id, builder)
        });
        target_token
    }

    /// unroll.py:320-362: _jump_to_existing_trace — check if any existing
    /// compiled trace (target_token) has a compatible virtual state.
    /// If so, generate extra guards, inline short preamble, and redirect jump.
    ///
    /// Returns None if jumped successfully, Some(virtual_state) otherwise.
    pub fn jump_to_existing_trace(
        &self,
        jump_args: &[OpRef],
        target_tokens: &mut [TargetToken],
        ctx: &mut OptContext,
    ) -> Option<crate::virtualstate::VirtualState> {
        let virtual_state = crate::virtualstate::export_state(jump_args, ctx, &ctx.ptr_info);
        let args: Vec<OpRef> = jump_args.iter().map(|&a| ctx.get_replacement(a)).collect();

        for target_token in target_tokens {
            let target_vs = match &target_token.virtual_state {
                Some(vs) => vs,
                None => continue,
            };

            // unroll.py:330-332: generate_guards — check compatibility
            if !target_vs.generalization_of(&virtual_state) {
                continue;
            }
            let extra_guards = target_vs.generate_guards(&virtual_state);
            for guard_req in &extra_guards {
                if let Some(guard_op) = guard_req.to_op(&args) {
                    ctx.emit(guard_op);
                }
            }

            // unroll.py:346-347: make_inputargs_and_virtuals
            let (target_args, virtuals) =
                target_vs.make_inputargs_and_virtuals(&args, ctx);
            let mut short_jump_args = target_args.clone();
            short_jump_args.extend(virtuals);

            // unroll.py:353-356: inline short preamble
            let mut extra = Vec::new();
            if let Some(sp) = target_token.short_preamble.clone() {
                if let Some(builder) = target_token.short_preamble_producer.as_mut() {
                    let mut label_args = sp.inputargs.clone();
                    label_args.extend(sp.used_boxes.iter().copied());
                    builder.setup(&sp, &label_args);
                    extra = Self::inline_short_preamble(
                        &short_jump_args,
                        &target_args,
                        &sp,
                        Some(builder),
                        ctx,
                    );
                    target_token.short_preamble = Some(builder.build_short_preamble_struct());
                } else {
                    extra = Self::inline_short_preamble(
                        &short_jump_args,
                        &target_args,
                        &sp,
                        None,
                        ctx,
                    );
                }
            }

            // unroll.py:357-359: emit JUMP to target
            let mut jump_args = target_args;
            jump_args.extend(extra);
            let jump = Op::new(OpCode::Jump, &jump_args);
            ctx.emit(jump);
            return None; // successfully jumped
        }

        Some(virtual_state)
    }

    /// unroll.py: inline_short_preamble — replay short preamble ops
    /// to re-populate the optimizer's cache when entering from a bridge.
    ///
    /// Maps short preamble input args to the jump args, then emits
    /// each short preamble op with remapped arguments.
    pub fn inline_short_preamble(
        jump_args: &[OpRef],
        _args_no_virtuals: &[OpRef],
        short_preamble: &crate::shortpreamble::ShortPreamble,
        mut short_preamble_producer: Option<&mut crate::shortpreamble::ExtendedShortPreambleBuilder>,
        ctx: &mut OptContext,
    ) -> Vec<OpRef> {
        let mut mapping: HashMap<OpRef, OpRef> = HashMap::new();

        for (i, &short_inputarg) in short_preamble.inputargs.iter().enumerate() {
            if let Some(&jump_arg) = jump_args.get(i) {
                mapping.insert(short_inputarg, jump_arg);
            }
        }

        let mut active_short = short_preamble.clone();
        let mut replay_index = 0;

        loop {
            while replay_index < active_short.ops.len() {
                let sp_op = &active_short.ops[replay_index];
                let mut new_op = sp_op.op.clone();
                // Remap args from arg_mapping (label idx → jump arg)
                for &(arg_pos, label_idx) in &sp_op.arg_mapping {
                    if arg_pos < new_op.args.len() && label_idx < jump_args.len() {
                        new_op.args[arg_pos] = jump_args[label_idx];
                    }
                }
                // Remap remaining args from the mapping table
                for arg in &mut new_op.args {
                    if let Some(&mapped) = mapping.get(arg) {
                        *arg = mapped;
                    }
                }
                if let Some(ref mut fail_args) = new_op.fail_args {
                    for &(fail_arg_pos, label_idx) in &sp_op.fail_arg_mapping {
                        if fail_arg_pos < fail_args.len() && label_idx < jump_args.len() {
                            fail_args[fail_arg_pos] = jump_args[label_idx];
                        }
                    }
                    for arg in fail_args.iter_mut() {
                        if let Some(&mapped) = mapping.get(arg) {
                            *arg = mapped;
                        }
                    }
                }
                let new_ref = ctx.emit(new_op.clone());
                mapping.insert(sp_op.op.pos, new_ref);
                replay_index += 1;
            }

            let Some(builder) = short_preamble_producer.as_deref_mut() else {
                break;
            };
            let old_len = active_short.ops.len();
            let old_jump_args = active_short.used_boxes.len();
            let current_used_boxes = active_short.used_boxes.clone();
            for used_box in current_used_boxes {
                let _ = builder.use_box(used_box);
            }
            active_short = builder.build_short_preamble_struct();
            if active_short.ops.len() == old_len && active_short.used_boxes.len() == old_jump_args {
                break;
            }
        }

        active_short
            .used_boxes
            .iter()
            .map(|&used_box| *mapping.get(&used_box).unwrap_or(&used_box))
            .collect()
    }

    /// unroll.py: import_state — restore optimizer state for peeled loop.
    ///
    /// Maps target args (from the new label) to the exported state's
    /// next_iteration_args, carrying forward type info and virtuals.
    /// unroll.py:479-504: import_state
    pub fn import_state(
        &self,
        targetargs: &[OpRef],
        exported_state: &ExportedState,
        ctx: &mut OptContext,
    ) -> Vec<OpRef> {
        assert_eq!(
            exported_state.next_iteration_args.len(),
            targetargs.len(),
            "import_state: next_iteration_args mismatch"
        );

        // unroll.py:483-490: forward args, apply exported info
        for (i, target) in exported_state.next_iteration_args.iter().enumerate() {
            let source = targetargs[i];
            ctx.replace_op(source, *target);
            // unroll.py:487: info = exported_state.exported_infos.get(target, None)
            if let Some(info) = exported_state.exported_infos.get(target) {
                self.apply_exported_info(source, info, &exported_state.exported_infos, ctx);
            }
        }

        // unroll.py:493-494: label_args = virtual_state.make_inputargs(targetargs)
        let (label_args, virtuals) = exported_state
            .virtual_state
            .make_inputargs_and_virtuals(targetargs, ctx);
        let mut short_args = label_args.clone();
        short_args.extend(virtuals);
        ctx.initialize_imported_short_preamble_builder(
            &short_args,
            &exported_state.exported_short_boxes,
        );
        self.import_short_preamble_ops(&short_args, exported_state, ctx);
        label_args
    }

    fn collect_exported_info(
        &self,
        opref: OpRef,
        ctx: &OptContext,
        exported_int_bounds: Option<&HashMap<OpRef, crate::intutils::IntBound>>,
    ) -> ExportedValueInfo {
        let resolved = ctx.get_replacement(opref);
        let constant = ctx.get_constant(resolved).cloned();
        let (ptr_kind, ptr_descr, known_class, array_lenbound, nonnull) = match ctx
            .get_ptr_info(resolved)
        {
            Some(crate::info::PtrInfo::Instance(info)) => (
                ExportedPtrKind::Instance,
                info.descr.clone(),
                info.known_class,
                None,
                true,
            ),
            Some(crate::info::PtrInfo::Struct(info)) => (
                ExportedPtrKind::Struct,
                Some(info.descr.clone()),
                None,
                None,
                true,
            ),
            Some(crate::info::PtrInfo::Array(info)) => (
                ExportedPtrKind::Array,
                Some(info.descr.clone()),
                None,
                Some(info.lenbound.clone()),
                true,
            ),
            Some(ptr) => (
                ExportedPtrKind::None,
                None,
                ptr.get_known_class().copied(),
                None,
                ptr.is_nonnull(),
            ),
            None => (ExportedPtrKind::None, None, None, None, false),
        };
        let int_bound = exported_int_bounds.and_then(|bounds| bounds.get(&resolved).cloned());
        let int_lower_bound = int_bound
            .as_ref()
            .map(|bound| bound.lower)
            .filter(|lower| *lower > i64::MIN)
            .or_else(|| ctx.int_lower_bounds.get(&resolved).copied());
        ExportedValueInfo {
            constant,
            ptr_info: ctx.get_ptr_info(resolved).cloned(),
            ptr_kind,
            ptr_descr,
            known_class,
            array_lenbound,
            int_bound,
            nonnull,
            int_lower_bound,
        }
    }

    fn apply_exported_info(
        &self,
        opref: OpRef,
        info: &ExportedValueInfo,
        exported_infos: &HashMap<OpRef, ExportedValueInfo>,
        ctx: &mut OptContext,
    ) {
        let mut seen = std::collections::HashSet::new();
        self.apply_exported_info_recursive(opref, info, exported_infos, ctx, &mut seen);
    }

    fn apply_exported_info_recursive(
        &self,
        opref: OpRef,
        info: &ExportedValueInfo,
        exported_infos: &HashMap<OpRef, ExportedValueInfo>,
        ctx: &mut OptContext,
        seen: &mut std::collections::HashSet<OpRef>,
    ) {
        let opref = ctx.get_replacement(opref);
        if !seen.insert(opref) {
            return;
        }
        if let Some(value) = &info.constant {
            ctx.make_constant(opref, value.clone());
            if let Value::Ref(ptr) = value {
                ctx.set_ptr_info(opref, crate::info::PtrInfo::Constant(*ptr));
            }
        }
        if let Some(ptr_info) = info.ptr_info.clone() {
            self.apply_exported_ptr_info(opref, ptr_info, exported_infos, ctx, seen);
        } else {
            match info.ptr_kind {
                ExportedPtrKind::Instance => {
                    ctx.set_ptr_info(
                        opref,
                        crate::info::PtrInfo::instance(info.ptr_descr.clone(), info.known_class),
                    );
                }
                ExportedPtrKind::Struct => {
                    if let Some(descr) = info.ptr_descr.clone() {
                        ctx.set_ptr_info(opref, crate::info::PtrInfo::struct_ptr(descr));
                    }
                }
                ExportedPtrKind::Array => {
                    if let Some(descr) = info.ptr_descr.clone() {
                        let lenbound = info
                            .array_lenbound
                            .clone()
                            .unwrap_or_else(crate::intutils::IntBound::nonnegative);
                        ctx.set_ptr_info(opref, crate::info::PtrInfo::array(descr, lenbound));
                    }
                }
                ExportedPtrKind::None => {
                    if let Some(class_ptr) = info.known_class {
                        ctx.set_ptr_info(
                            opref,
                            crate::info::PtrInfo::KnownClass {
                                class_ptr,
                                is_nonnull: true,
                            },
                        );
                    } else if info.nonnull {
                        ctx.set_ptr_info(opref, crate::info::PtrInfo::NonNull);
                    }
                }
            }
        }
        if let Some(bound) = info.int_bound.as_ref() {
            let widened = bound.widen();
            if widened.lower > i64::MIN {
                ctx.int_lower_bounds.insert(opref, widened.lower);
            }
            ctx.imported_int_bounds.insert(opref, widened);
        }
        if let Some(lower) = info.int_lower_bound {
            ctx.int_lower_bounds.insert(opref, lower);
        }
    }

    fn apply_exported_ptr_info(
        &self,
        opref: OpRef,
        ptr_info: crate::info::PtrInfo,
        exported_infos: &HashMap<OpRef, ExportedValueInfo>,
        ctx: &mut OptContext,
        seen: &mut std::collections::HashSet<OpRef>,
    ) {
        use crate::info::PtrInfo;

        match ptr_info {
            PtrInfo::Virtual(info) => {
                ctx.set_ptr_info(opref, PtrInfo::Virtual(info.clone()));
                for &(_, field_ref) in &info.fields {
                    if let Some(field_info) = exported_infos.get(&field_ref) {
                        self.apply_exported_info_recursive(
                            field_ref,
                            field_info,
                            exported_infos,
                            ctx,
                            seen,
                        );
                    }
                }
            }
            PtrInfo::VirtualStruct(info) => {
                ctx.set_ptr_info(opref, PtrInfo::VirtualStruct(info.clone()));
                for &(_, field_ref) in &info.fields {
                    if let Some(field_info) = exported_infos.get(&field_ref) {
                        self.apply_exported_info_recursive(
                            field_ref,
                            field_info,
                            exported_infos,
                            ctx,
                            seen,
                        );
                    }
                }
            }
            PtrInfo::VirtualArray(info) => {
                ctx.set_ptr_info(opref, PtrInfo::VirtualArray(info.clone()));
                for &item_ref in &info.items {
                    if let Some(item_info) = exported_infos.get(&item_ref) {
                        self.apply_exported_info_recursive(
                            item_ref,
                            item_info,
                            exported_infos,
                            ctx,
                            seen,
                        );
                    }
                }
            }
            PtrInfo::VirtualArrayStruct(info) => {
                ctx.set_ptr_info(opref, PtrInfo::VirtualArrayStruct(info.clone()));
                for fields in &info.element_fields {
                    for &(_, field_ref) in fields {
                        if let Some(field_info) = exported_infos.get(&field_ref) {
                            self.apply_exported_info_recursive(
                                field_ref,
                                field_info,
                                exported_infos,
                                ctx,
                                seen,
                            );
                        }
                    }
                }
            }
            PtrInfo::VirtualRawBuffer(info) => {
                ctx.set_ptr_info(opref, PtrInfo::VirtualRawBuffer(info.clone()));
                for &(_, _, entry_ref) in &info.entries {
                    if let Some(entry_info) = exported_infos.get(&entry_ref) {
                        self.apply_exported_info_recursive(
                            entry_ref,
                            entry_info,
                            exported_infos,
                            ctx,
                            seen,
                        );
                    }
                }
            }
            PtrInfo::Instance(info) => {
                ctx.set_ptr_info(opref, PtrInfo::Instance(info.clone()));
                for &(_, field_ref) in &info.fields {
                    if let Some(field_info) = exported_infos.get(&field_ref) {
                        self.apply_exported_info_recursive(
                            field_ref,
                            field_info,
                            exported_infos,
                            ctx,
                            seen,
                        );
                    }
                }
            }
            PtrInfo::Struct(info) => {
                ctx.set_ptr_info(opref, PtrInfo::Struct(info.clone()));
                for &(_, field_ref) in &info.fields {
                    if let Some(field_info) = exported_infos.get(&field_ref) {
                        self.apply_exported_info_recursive(
                            field_ref,
                            field_info,
                            exported_infos,
                            ctx,
                            seen,
                        );
                    }
                }
            }
            PtrInfo::Array(info) => {
                ctx.set_ptr_info(opref, PtrInfo::Array(info.clone()));
                for &item_ref in &info.items {
                    if let Some(item_info) = exported_infos.get(&item_ref) {
                        self.apply_exported_info_recursive(
                            item_ref,
                            item_info,
                            exported_infos,
                            ctx,
                            seen,
                        );
                    }
                }
            }
            PtrInfo::KnownClass { class_ptr, is_nonnull } => {
                ctx.set_ptr_info(opref, PtrInfo::KnownClass { class_ptr, is_nonnull });
            }
            PtrInfo::Constant(ptr) => {
                ctx.set_ptr_info(opref, PtrInfo::Constant(ptr));
            }
            PtrInfo::NonNull => {
                ctx.set_ptr_info(opref, PtrInfo::NonNull);
            }
            PtrInfo::Virtualizable(info) => {
                ctx.set_ptr_info(opref, PtrInfo::Virtualizable(info));
            }
        }
    }

    fn collect_exported_short_ops(
        &self,
        short_args: &[OpRef],
        ctx: &OptContext,
    ) -> Vec<ExportedShortOp> {
        let short_boxes = crate::shortpreamble::ShortBoxes::with_label_args(short_args);
        let mut produced_indices: HashMap<OpRef, usize> = HashMap::new();
        let mut next_temp = 0usize;
        let mut exported = Vec::new();
        for entry in &ctx.exported_short_boxes {
            let result = if let Some(slot) = short_boxes.lookup_label_arg(entry.op.pos) {
                ExportedShortResult::Slot(slot)
            } else {
                let temp = next_temp;
                next_temp += 1;
                ExportedShortResult::Temporary(temp)
            };
            let exported_op = match entry.kind {
                crate::shortpreamble::PreambleOpKind::Pure => {
                    let args = entry
                        .op
                        .args
                        .iter()
                        .map(|&arg| {
                            short_boxes
                                .lookup_label_arg(arg)
                                .map(ExportedShortArg::Slot)
                                .or_else(|| {
                                    produced_indices
                                        .get(&arg)
                                        .copied()
                                        .map(ExportedShortArg::Produced)
                                })
                                .or_else(|| ctx.get_constant(arg).cloned().map(ExportedShortArg::Const))
                        })
                        .collect::<Option<Vec<_>>>();
                    let Some(args) = args else {
                        continue;
                    };
                let opcode = if entry.op.opcode.is_call() {
                        OpCode::call_pure_for_type(entry.op.result_type())
                    } else {
                        entry.op.opcode
                    };
                    Some(ExportedShortOp::Pure {
                        source: entry.op.pos,
                        opcode,
                        descr_idx: entry.op.descr.as_ref().map(|d| d.index()),
                        args,
                        result,
                        invented_name: entry.invented_name,
                        same_as_source: entry.same_as_source,
                    })
                }
                crate::shortpreamble::PreambleOpKind::Heap { descr_idx } => {
                    let Some(object_slot) = short_boxes.lookup_label_arg(entry.op.arg(0)) else {
                        continue;
                    };
                    match entry.op.opcode {
                        OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF => {
                            Some(ExportedShortOp::HeapField {
                                source: entry.op.pos,
                                object_slot,
                                descr_idx,
                                result_type: entry.op.result_type(),
                                result,
                                invented_name: entry.invented_name,
                                same_as_source: entry.same_as_source,
                            })
                        }
                        OpCode::GetarrayitemGcI
                        | OpCode::GetarrayitemGcR
                        | OpCode::GetarrayitemGcF => {
                            let index = entry.op.arg(1).0 as i64;
                            Some(ExportedShortOp::HeapArrayItem {
                                source: entry.op.pos,
                                object_slot,
                                descr_idx,
                                index,
                                result_type: entry.op.result_type(),
                                result,
                                invented_name: entry.invented_name,
                                same_as_source: entry.same_as_source,
                            })
                        }
                        _ => None,
                    }
                }
                crate::shortpreamble::PreambleOpKind::LoopInvariant => {
                    let Some(func_ptr) = ctx.get_constant_int(entry.op.arg(0)) else {
                        continue;
                    };
                    Some(ExportedShortOp::LoopInvariant {
                        source: entry.op.pos,
                        func_ptr,
                        result_type: entry.op.result_type(),
                        result,
                        invented_name: entry.invented_name,
                        same_as_source: entry.same_as_source,
                    })
                }
                _ => None,
            };
            if let Some(exported_op) = exported_op {
                produced_indices.insert(entry.op.pos, exported.len());
                exported.push(exported_op);
            }
        }
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] collect_exported_short_ops: short_args={}, exported_short_boxes={}, exported_short_ops={}",
                short_args.len(),
                ctx.exported_short_boxes.len(),
                exported.len()
            );
        }
        exported
    }

    fn import_short_preamble_ops(
        &self,
        short_args: &[OpRef],
        exported_state: &ExportedState,
        ctx: &mut OptContext,
    ) {
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] import_short_preamble_ops: short_args={}, exported_short_ops={}",
                short_args.len(),
                exported_state.exported_short_ops.len()
            );
        }
        let mut produced_results = Vec::with_capacity(exported_state.exported_short_ops.len());
        for entry in &exported_state.exported_short_ops {
            let mut resolve_result = |result: &ExportedShortResult| match result {
                ExportedShortResult::Slot(slot) => short_args.get(*slot).copied(),
                ExportedShortResult::Temporary(_) => Some(ctx.alloc_op_position()),
            };
            match *entry {
                ExportedShortOp::Pure {
                    source,
                    opcode,
                    descr_idx,
                    ref args,
                    ref result,
                    invented_name,
                    same_as_source,
                } => {
                    let Some(result_opref) = resolve_result(result) else {
                        continue;
                    };
                    let args = args
                        .iter()
                        .map(|arg| match arg {
                            ExportedShortArg::Slot(slot) => {
                                short_args
                                    .get(*slot)
                                    .copied()
                                    .map(crate::ImportedShortPureArg::OpRef)
                            }
                            ExportedShortArg::Const(value) => {
                                Some(crate::ImportedShortPureArg::Const(*value))
                            }
                            ExportedShortArg::Produced(index) => produced_results
                                .get(*index)
                                .copied()
                                .map(crate::ImportedShortPureArg::OpRef),
                        })
                        .collect::<Option<Vec<_>>>();
                    let Some(args) = args else {
                        continue;
                    };
                    ctx.imported_short_pure_ops.push(crate::ImportedShortPureOp {
                        opcode,
                        descr_idx,
                        args,
                        result: result_opref,
                    });
                    ctx.imported_short_sources.push(crate::ImportedShortSource {
                        result: result_opref,
                        source: same_as_source.unwrap_or(source),
                    });
                    if invented_name {
                        if let Some(source) = same_as_source {
                            ctx.imported_short_aliases.push(crate::ImportedShortAlias {
                                result: result_opref,
                                same_as_source: source,
                                same_as_opcode: OpCode::same_as_for_type(opcode.result_type()),
                            });
                        }
                    }
                    produced_results.push(result_opref);
                }
                ExportedShortOp::HeapField {
                    source,
                    object_slot,
                    descr_idx,
                    result_type,
                    ref result,
                    invented_name,
                    same_as_source,
                } => {
                    let Some(&obj) = short_args.get(object_slot) else {
                        continue;
                    };
                    let Some(value) = resolve_result(result) else {
                        continue;
                    };
                    ctx.imported_short_fields.insert((obj, descr_idx), value);
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!(
                            "[jit] import_short_heap_field: obj={obj:?} descr_idx={descr_idx} value={value:?}"
                        );
                    }
                    ctx.imported_short_sources.push(crate::ImportedShortSource {
                        result: value,
                        source: same_as_source.unwrap_or(source),
                    });
                    if invented_name {
                        if let Some(source) = same_as_source {
                            ctx.imported_short_aliases.push(crate::ImportedShortAlias {
                                result: value,
                                same_as_source: source,
                                same_as_opcode: OpCode::same_as_for_type(result_type),
                            });
                        }
                    }
                    produced_results.push(value);
                }
                ExportedShortOp::HeapArrayItem {
                    source,
                    object_slot,
                    descr_idx,
                    index,
                    result_type,
                    ref result,
                    invented_name,
                    same_as_source,
                } => {
                    let Some(&obj) = short_args.get(object_slot) else {
                        continue;
                    };
                    let Some(value) = resolve_result(result) else {
                        continue;
                    };
                    ctx.imported_short_arrayitems
                        .insert((obj, descr_idx, index), value);
                    ctx.imported_short_sources.push(crate::ImportedShortSource {
                        result: value,
                        source: same_as_source.unwrap_or(source),
                    });
                    if invented_name {
                        if let Some(source) = same_as_source {
                            ctx.imported_short_aliases.push(crate::ImportedShortAlias {
                                result: value,
                                same_as_source: source,
                                same_as_opcode: OpCode::same_as_for_type(result_type),
                            });
                        }
                    }
                    produced_results.push(value);
                }
                ExportedShortOp::LoopInvariant {
                    source,
                    func_ptr,
                    result_type,
                    ref result,
                    invented_name,
                    same_as_source,
                } => {
                    let Some(value) = resolve_result(result) else {
                        continue;
                    };
                    ctx.imported_loop_invariant_results.insert(func_ptr, value);
                    ctx.imported_short_sources.push(crate::ImportedShortSource {
                        result: value,
                        source: same_as_source.unwrap_or(source),
                    });
                    if invented_name {
                        if let Some(source) = same_as_source {
                            ctx.imported_short_aliases.push(crate::ImportedShortAlias {
                                result: value,
                                same_as_source: source,
                                same_as_opcode: OpCode::same_as_for_type(result_type),
                            });
                        }
                    }
                    produced_results.push(value);
                }
            }
        }
    }
}

/// unroll.py: export_state — module-level entry point.
pub(crate) fn export_state(
    jump_args: &[OpRef],
    renamed_inputargs: &[OpRef],
    ctx: &OptContext,
    exported_int_bounds: Option<&HashMap<OpRef, crate::intutils::IntBound>>,
) -> ExportedState {
    OptUnroll::new().export_state_with_bounds(jump_args, renamed_inputargs, ctx, exported_int_bounds)
}

/// unroll.py: import_state — module-level entry point.
pub(crate) fn import_state(
    targetargs: &[OpRef],
    exported_state: &ExportedState,
    ctx: &mut OptContext,
) -> Vec<OpRef> {
    OptUnroll::new().import_state(targetargs, exported_state, ctx)
}

/// unroll.py: pick_virtual_state(my_vs, label_vs, target_tokens)
///
/// Given the current virtual state and available target tokens,
/// find a compatible target to jump to. Returns the target index
/// or None if no match.
/// RPython unroll.py: import_state + _generate_virtual.
///
// ── RPython-parity helper functions for 2-phase preamble peeling ──

/// unroll.py:479-504 import_state: build ImportedVirtual for Phase 2.
fn build_imported_virtuals(
    jump_virtuals: &[crate::optimizer::ExportedJumpVirtual],
) -> Vec<crate::optimizer::ImportedVirtual> {
    jump_virtuals
        .iter()
        .map(|virt| crate::optimizer::ImportedVirtual {
            inputarg_index: virt.jump_arg_index,
            size_descr: virt.size_descr.clone(),
            kind: virt.kind.clone(),
            fields: virt.fields.clone(),
            head_load_descr_index: virt.head_load_descr_index,
        })
        .collect()
}

/// compile.py:310-338: [preamble_no_jump] + Label(label_args) + [body_with_jump]
fn assemble_peeled_trace(
    p1_ops: &[Op],
    p2_ops: &[Op],
    label_args: &[OpRef],
    extra_label_args: &[OpRef],
    rewritten_jump_args: Option<&[OpRef]>,
    body_num_inputs: usize,
    imported_short_aliases: &[crate::ImportedShortAlias],
    imported_short_sources: &[crate::ImportedShortSource],
) -> Vec<Op> {
    let mut result = Vec::with_capacity(
        p1_ops.len() + p2_ops.len() + 1 + imported_short_aliases.len(),
    );
    let imported_short_sources: HashMap<OpRef, OpRef> = imported_short_sources
        .iter()
        .map(|entry| (entry.result, entry.source))
        .collect();

    // Preamble: everything except Jump
    for op in p1_ops {
        if op.opcode == OpCode::Jump { break; }
        result.push(op.clone());
    }

    // Extra SameAs aliases live at the end of the preamble, before the loop
    // label, matching compile.py's `loop_info.extra_same_as + [label_op]`.
    let mut max_pos = result
        .iter()
        .map(|op| op.pos.0)
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(0);
    let mut remap: HashMap<OpRef, OpRef> = HashMap::new();
    for alias in imported_short_aliases {
        let pos = OpRef((max_pos + 1).max(body_num_inputs as u32));
        max_pos = pos.0;
        let mut op = Op::new(alias.same_as_opcode, &[alias.same_as_source]);
        op.pos = pos;
        remap.insert(alias.result, pos);
        result.push(op);
    }

    // Label position
    let label_pos = (max_pos + 1).max(body_num_inputs as u32);
    let mut full_label_args = label_args.to_vec();
    full_label_args.extend(extra_label_args.iter().map(|arg| {
        remap
            .get(arg)
            .copied()
            .or_else(|| imported_short_sources.get(arg).copied())
            .unwrap_or(*arg)
    }));
    let mut label_op = Op::new(OpCode::Label, &full_label_args);
    label_op.pos = OpRef(label_pos);
    result.push(label_op);

    // Body: 2-pass remap (inputarg refs → label_args, op positions → after label)
    let body_base = label_pos + 1;

    // Pass 1: collect all remappings
    for (i, &la) in label_args.iter().enumerate() {
        if (i as u32) < body_num_inputs as u32 {
            remap.insert(OpRef(i as u32), la);
        }
    }
    for (idx, op) in p2_ops.iter().enumerate() {
        if op.pos.0 != u32::MAX {
            remap.insert(op.pos, OpRef(body_base + idx as u32));
        }
    }
    // Phase 2 ops may reference OpRefs not in the body (e.g., a forced
    // virtual's New at an intermediate position). If such a ref matches
    // a Label arg, map it to that Label arg's position in the preamble.
    // This handles the case where Phase 2's forwarding chain resolves to
    // a position that exists only in the preamble output.
    let label_set: HashMap<OpRef, OpRef> = label_args.iter().copied()
        .enumerate()
        .map(|(_, la)| (la, la))
        .collect();
    // Also map Phase 1 preamble positions that Phase 2 might reference
    for p1_op in p1_ops {
        if p1_op.opcode == OpCode::Jump { break; }
        if !p1_op.pos.is_none() && !remap.contains_key(&p1_op.pos) {
            // If this Phase 1 op produced a Label arg, use its preamble pos
            remap.insert(p1_op.pos, p1_op.pos);
        }
    }
    // imported_short_sources: Phase 2 result OpRef → Phase 1 source OpRef.
    // After forwarding resolution, body args may reference the SOURCE
    // (Phase 1 position) instead of the result. Map those to preamble.
    for (source_ref, _) in &imported_short_sources {
        if !remap.contains_key(source_ref) {
            // This source is a preamble op position; body can reference it
            remap.insert(*source_ref, *source_ref);
        }
    }

    // Pass 2: apply
    for (idx, op) in p2_ops.iter().enumerate() {
        let mut new_op = op.clone();
        if new_op.pos.0 != u32::MAX {
            new_op.pos = OpRef(body_base + idx as u32);
        }
        for arg in &mut new_op.args {
            if let Some(&m) = remap.get(arg) {
                *arg = m;
            } else if let Some(&source) = imported_short_sources.get(arg) {
                *arg = source;
            }
        }
        if new_op.opcode == OpCode::Jump {
            if let Some(forced_jump_args) = rewritten_jump_args {
                new_op.args = forced_jump_args
                    .iter()
                    .map(|arg| {
                        remap
                            .get(arg)
                            .copied()
                            .or_else(|| imported_short_sources.get(arg).copied())
                            .unwrap_or(*arg)
                    })
                    .collect();
            }
        }
        if let Some(ref mut fa) = new_op.fail_args {
            for a in fa.iter_mut() {
                if let Some(&m) = remap.get(a) {
                    *a = m;
                } else if let Some(&source) = imported_short_sources.get(a) {
                    *a = source;
                }
            }
        }
        result.push(new_op);
    }

    result
}


pub fn pick_virtual_state(
    my_vs: &crate::virtualstate::VirtualState,
    target_states: &[crate::virtualstate::VirtualState],
) -> Option<usize> {
    for (i, target_vs) in target_states.iter().enumerate() {
        if target_vs.generalization_of(my_vs) {
            return Some(i);
        }
    }
    None
}

pub struct OptUnroll {
    /// Buffer of ops received before the Jump back-edge.
    buffer: Vec<Op>,
    /// Whether a Jump was already seen (avoid double-unrolling).
    seen_jump: bool,
}

impl OptUnroll {
    pub fn new() -> Self {
        OptUnroll {
            buffer: Vec::new(),
            seen_jump: false,
        }
    }

    /// Peel one iteration of the loop body.
    ///
    /// Given the buffered ops (everything before the Jump), this emits:
    /// 1. The peeled (duplicated) body with remapped OpRefs
    /// 2. A Label op marking the loop header
    /// 3. The original body ops
    ///
    /// The caller is responsible for emitting the final Jump.
    fn peel_iteration(&self, jump_op: &Op, ctx: &mut OptContext) {
        if self.buffer.is_empty() {
            return;
        }

        // Build the OpRef remapping for peeled ops.
        // Each op in the buffer gets a new position in the peeled iteration.
        // The offset is: original positions map to new sequential positions
        // starting from the current emission point.
        let peel_base = ctx.new_operations.len() as u32;
        let mut ref_map: HashMap<OpRef, OpRef> = HashMap::new();

        // First pass: assign new positions for peeled ops.
        for (i, op) in self.buffer.iter().enumerate() {
            let new_pos = OpRef(peel_base + i as u32);
            ref_map.insert(op.pos, new_pos);
        }

        // Emit peeled iteration with remapped refs.
        for (i, op) in self.buffer.iter().enumerate() {
            let mut peeled = op.clone();
            peeled.pos = OpRef(peel_base + i as u32);

            // Remap argument references.
            for arg in &mut peeled.args {
                if let Some(&new_ref) = ref_map.get(arg) {
                    *arg = new_ref;
                }
                // Args referencing ops outside the buffer (e.g., input args)
                // are kept as-is.
            }

            // Remap fail_args references.
            if let Some(ref mut fa) = peeled.fail_args {
                for arg in fa.iter_mut() {
                    if let Some(&new_ref) = ref_map.get(arg) {
                        *arg = new_ref;
                    }
                }
            }

            ctx.emit(peeled);
        }

        // Emit Label between peeled and original body.
        // The Label's args match the Jump's args, forming the loop header.
        let label_pos = OpRef(peel_base + self.buffer.len() as u32);
        let mut label_op = Op::new(OpCode::Label, &jump_op.args);
        label_op.pos = label_pos;
        ctx.emit(label_op);

        // Emit original body ops with updated positions.
        let body_base = peel_base + self.buffer.len() as u32 + 1; // +1 for Label
        let mut orig_ref_map: HashMap<OpRef, OpRef> = HashMap::new();

        for (i, op) in self.buffer.iter().enumerate() {
            orig_ref_map.insert(op.pos, OpRef(body_base + i as u32));
        }

        for (i, op) in self.buffer.iter().enumerate() {
            let mut body_op = op.clone();
            body_op.pos = OpRef(body_base + i as u32);

            // Remap argument references within the body.
            for arg in &mut body_op.args {
                if let Some(&new_ref) = orig_ref_map.get(arg) {
                    *arg = new_ref;
                }
            }

            if let Some(ref mut fa) = body_op.fail_args {
                for arg in fa.iter_mut() {
                    if let Some(&new_ref) = orig_ref_map.get(arg) {
                        *arg = new_ref;
                    }
                }
            }

            ctx.emit(body_op);
        }
    }
}

impl Default for OptUnroll {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptUnroll {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // Only peel once per trace, and only for Jump (back-edge).
        if op.opcode == OpCode::Jump && !self.seen_jump {
            self.seen_jump = true;

            if self.buffer.is_empty() {
                // Empty loop body, nothing to peel.
                return OptimizationResult::Emit(op.clone());
            }

            // Perform the peeling: emit peeled body + Label + original body.
            self.peel_iteration(op, ctx);

            // Emit the final Jump with remapped args pointing to the
            // original body's ops.
            let body_base = ctx.new_operations.len() as u32 - self.buffer.len() as u32;
            let mut jump = op.clone();

            // Build ref map for Jump args (same as orig_ref_map).
            let mut orig_ref_map: HashMap<OpRef, OpRef> = HashMap::new();
            for (i, buffered_op) in self.buffer.iter().enumerate() {
                orig_ref_map.insert(buffered_op.pos, OpRef(body_base + i as u32));
            }

            for arg in &mut jump.args {
                if let Some(&new_ref) = orig_ref_map.get(arg) {
                    *arg = new_ref;
                }
            }

            jump.pos = OpRef(ctx.new_operations.len() as u32);
            return OptimizationResult::Emit(jump);
        }

        // For non-Jump ops (or after we've already unrolled), buffer them.
        if !self.seen_jump {
            self.buffer.push(op.clone());
            return OptimizationResult::Remove;
        }

        // After unrolling, pass everything through.
        OptimizationResult::Emit(op.clone())
    }

    fn setup(&mut self) {
        self.buffer.clear();
        self.seen_jump = false;
    }

    fn name(&self) -> &'static str {
        "unroll"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Optimizer;

    /// Assign sequential positions to ops starting from `base`.
    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(base + i as u32);
        }
    }

    fn run_unroll_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptUnroll::new()));
        opt.optimize(ops)
    }

    // ── Basic peeling ─────────────────────────────────────────────────

    #[test]
    fn test_no_jump_no_unroll() {
        // Without a Jump back-edge, the pass just buffers and nothing is emitted.
        // (In practice, traces always end with Jump or Finish.)
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Finish, &[OpRef(0)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // IntAdd gets buffered, Finish is not a Jump so it gets buffered too.
        // Nothing is emitted because there's no Jump to trigger peeling.
        // The buffered ops are lost (which is correct: no loop = no unrolling).
        assert!(
            result.is_empty(),
            "no Jump means no loop to unroll, ops are buffered but never emitted"
        );
    }

    #[test]
    fn test_empty_loop_body() {
        // Jump with no prior ops: nothing to peel.
        let mut ops = vec![Op::new(OpCode::Jump, &[])];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::Jump);
    }

    #[test]
    fn test_jump_to_preamble_preserves_jump_args() {
        let body_ops = vec![
            {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
                op.pos = OpRef(2);
                op
            },
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(2), OpRef(50)]),
        ];

        let result = UnrollOptimizer::jump_to_preamble(&body_ops, 1);
        assert_eq!(result[1].opcode, OpCode::Jump);
        assert_eq!(result[1].args.as_slice(), &[OpRef(0), OpRef(2), OpRef(50)]);
    }

    #[test]
    fn test_pick_virtual_state_uses_target_generalization_direction() {
        let my_vs = crate::virtualstate::VirtualState::new(vec![
            crate::virtualstate::VirtualStateInfo::NonNull,
        ]);
        let target_states = vec![
            crate::virtualstate::VirtualState::new(vec![
                crate::virtualstate::VirtualStateInfo::Unknown,
            ]),
            crate::virtualstate::VirtualState::new(vec![
                crate::virtualstate::VirtualStateInfo::KnownClass {
                    class_ptr: GcRef(0x1234),
                },
            ]),
        ];

        assert_eq!(pick_virtual_state(&my_vs, &target_states), Some(0));
    }

    #[test]
    fn test_simple_loop_peeled() {
        // A simple loop: one add op, then Jump.
        // Expected output: peeled_add, Label, original_add, Jump
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        assert_eq!(
            result.len(),
            4,
            "expected: peeled_add, Label, original_add, Jump"
        );
        assert_eq!(result[0].opcode, OpCode::IntAdd); // peeled
        assert_eq!(result[1].opcode, OpCode::Label);
        assert_eq!(result[2].opcode, OpCode::IntAdd); // original body
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    #[test]
    fn test_peeled_ops_have_different_positions() {
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntSub, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 2 peeled + Label + 2 original + Jump = 6
        assert_eq!(result.len(), 6);

        // All positions should be unique.
        let positions: Vec<OpRef> = result.iter().map(|op| op.pos).collect();
        for (i, pos) in positions.iter().enumerate() {
            for (j, other) in positions.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        pos, other,
                        "positions at index {} and {} should differ",
                        i, j
                    );
                }
            }
        }
    }

    // ── OpRef remapping ───────────────────────────────────────────────

    #[test]
    fn test_internal_refs_remapped_in_peeled_copy() {
        // op0: v0 = IntAdd(v100, v101)  -- uses input args
        // op1: v1 = IntMul(v0, v101)    -- uses result of op0
        // Jump()
        //
        // After peeling:
        // peeled_v0 = IntAdd(v100, v101)     -- input refs unchanged
        // peeled_v1 = IntMul(peeled_v0, v101) -- v0 remapped to peeled_v0
        // Label()
        // body_v0 = IntAdd(v100, v101)
        // body_v1 = IntMul(body_v0, v101)    -- v0 remapped to body_v0
        // Jump()
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntMul, &[OpRef(0), OpRef(101)]), // references op0
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        assert_eq!(result.len(), 6); // 2 peeled + Label + 2 body + Jump

        // Peeled iteration:
        let peeled_add = &result[0];
        let peeled_mul = &result[1];
        assert_eq!(peeled_add.opcode, OpCode::IntAdd);
        assert_eq!(peeled_mul.opcode, OpCode::IntMul);
        // peeled_mul should reference peeled_add's position, not original op0.
        assert_eq!(peeled_mul.args[0], peeled_add.pos);
        // Second arg (input ref) should be unchanged.
        assert_eq!(peeled_mul.args[1], OpRef(101));

        // Original body:
        let body_add = &result[3];
        let body_mul = &result[4];
        assert_eq!(body_add.opcode, OpCode::IntAdd);
        assert_eq!(body_mul.opcode, OpCode::IntMul);
        // body_mul should reference body_add's position.
        assert_eq!(body_mul.args[0], body_add.pos);
        assert_eq!(body_mul.args[1], OpRef(101));
    }

    #[test]
    fn test_external_refs_preserved() {
        // Refs to ops outside the buffer (input arguments) should not be remapped.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // Peeled add should still reference v100 and v101.
        assert_eq!(result[0].args[0], OpRef(100));
        assert_eq!(result[0].args[1], OpRef(101));

        // Body add should also reference v100 and v101.
        assert_eq!(result[2].args[0], OpRef(100));
        assert_eq!(result[2].args[1], OpRef(101));
    }

    // ── Guard preservation ────────────────────────────────────────────

    #[test]
    fn test_guards_duplicated_in_peel() {
        // Guards in the preamble serve as type checks.
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(100)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // peeled_guard, peeled_add, Label, body_guard, body_add, Jump
        assert_eq!(result.len(), 6);

        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardTrue)
            .count();
        assert_eq!(
            guard_count, 2,
            "guard should appear in both peeled and body"
        );
    }

    #[test]
    fn test_guard_fail_args_remapped() {
        // Guards with fail_args should have those refs remapped too.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            {
                let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(100)]);
                guard.fail_args = Some(vec![OpRef(0)].into()); // refs op0
                guard
            },
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // Check peeled guard's fail_args.
        let peeled_guard = &result[1];
        assert_eq!(peeled_guard.opcode, OpCode::GuardTrue);
        let peeled_add_pos = result[0].pos;
        assert_eq!(
            peeled_guard.fail_args.as_ref().unwrap()[0],
            peeled_add_pos,
            "peeled guard's fail_args should reference peeled add"
        );

        // Check body guard's fail_args.
        let body_guard = &result[4]; // after Label (idx 3) and body_add (idx 3)
        assert_eq!(body_guard.opcode, OpCode::GuardTrue);
        let body_add_pos = result[3].pos;
        assert_eq!(
            body_guard.fail_args.as_ref().unwrap()[0],
            body_add_pos,
            "body guard's fail_args should reference body add"
        );
    }

    // ── Jump args remapping ───────────────────────────────────────────

    #[test]
    fn test_jump_args_remapped_to_body() {
        // Jump args should reference the body's ops, not the original positions.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[OpRef(0)]), // carries v0 (the add result)
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // peeled_add, Label, body_add, Jump
        assert_eq!(result.len(), 4);

        let jump = result.last().unwrap();
        assert_eq!(jump.opcode, OpCode::Jump);

        let body_add_pos = result[2].pos;
        assert_eq!(
            jump.args[0], body_add_pos,
            "Jump arg should reference body add, not original"
        );
    }

    #[test]
    fn test_label_args_match_jump_args() {
        // The Label should carry the same args as the Jump.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(100)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        let label = result.iter().find(|o| o.opcode == OpCode::Label).unwrap();
        let jump_args = &ops.last().unwrap().args;
        assert_eq!(
            label.args.as_slice(),
            jump_args.as_slice(),
            "Label args should match original Jump args"
        );
    }

    // ── Multiple ops in loop body ─────────────────────────────────────

    #[test]
    fn test_multi_op_loop() {
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntSub, &[OpRef(0), OpRef(101)]),
            Op::new(OpCode::IntMul, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef(2)]),
            Op::new(OpCode::Jump, &[OpRef(2)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 4 peeled + Label + 4 body + Jump = 10
        assert_eq!(result.len(), 10);

        // Verify structure: peeled body, then Label, then body, then Jump.
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::IntSub);
        assert_eq!(result[2].opcode, OpCode::IntMul);
        assert_eq!(result[3].opcode, OpCode::GuardTrue);
        assert_eq!(result[4].opcode, OpCode::Label);
        assert_eq!(result[5].opcode, OpCode::IntAdd);
        assert_eq!(result[6].opcode, OpCode::IntSub);
        assert_eq!(result[7].opcode, OpCode::IntMul);
        assert_eq!(result[8].opcode, OpCode::GuardTrue);
        assert_eq!(result[9].opcode, OpCode::Jump);
    }

    // ── No double-unrolling ───────────────────────────────────────────

    #[test]
    fn test_second_jump_not_unrolled() {
        // If there are multiple Jumps (unusual, but defensive), only the first
        // triggers peeling.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
            Op::new(OpCode::IntSub, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // First Jump triggers peeling of the IntAdd.
        // After that, IntSub and second Jump pass through.
        let jump_count = result.iter().filter(|o| o.opcode == OpCode::Jump).count();
        assert_eq!(jump_count, 2, "both jumps should be in output");

        let label_count = result.iter().filter(|o| o.opcode == OpCode::Label).count();
        assert_eq!(label_count, 1, "only one Label from the first peeling");
    }

    // ── Setup resets state ────────────────────────────────────────────

    #[test]
    fn test_setup_resets_state() {
        let mut pass = OptUnroll::new();

        // Simulate some state.
        pass.buffer.push(Op::new(OpCode::IntAdd, &[OpRef(0)]));
        pass.seen_jump = true;

        pass.setup();

        assert!(pass.buffer.is_empty());
        assert!(!pass.seen_jump);
    }

    // ── Integration with optimizer ────────────────────────────────────

    #[test]
    fn test_unroll_standalone_optimizer() {
        // Run the unroll pass through the optimizer infrastructure.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptUnroll::new()));
        let result = opt.optimize(&ops);

        // Expect: peeled_add, peeled_guard, Label, body_add, body_guard, Jump = 6
        assert_eq!(result.len(), 6);

        // All ops should have valid (non-NONE) positions.
        for op in &result {
            assert!(
                !op.pos.is_none(),
                "op {:?} should have a valid pos",
                op.opcode
            );
        }
    }

    #[test]
    fn test_unroll_preserves_descr() {
        // Descriptors on ops should be preserved in both copies.
        use majit_ir::{Descr, DescrRef};
        use std::sync::Arc;

        #[derive(Debug)]
        struct TestDescr(u32);
        impl Descr for TestDescr {
            fn index(&self) -> u32 {
                self.0
            }
        }

        let descr: DescrRef = Arc::new(TestDescr(42));
        let mut ops = vec![
            Op::with_descr(OpCode::GuardTrue, &[OpRef(100)], descr.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // Both guards (peeled and body) should have the descriptor.
        let guards: Vec<&Op> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardTrue)
            .collect();
        assert_eq!(guards.len(), 2);
        for guard in &guards {
            assert!(guard.descr.is_some(), "guard should have a descriptor");
            assert_eq!(guard.descr.as_ref().unwrap().index(), 42);
        }
    }

    // ── Chain of references ───────────────────────────────────────────

    #[test]
    fn test_chain_of_refs_correctly_remapped() {
        // v0 = IntAdd(v100, v101)
        // v1 = IntAdd(v0, v100)
        // v2 = IntAdd(v1, v0)
        // Jump(v2)
        //
        // In the peeled copy, all internal refs must point to peeled positions.
        // In the body copy, all internal refs must point to body positions.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(100)]),
            Op::new(OpCode::IntAdd, &[OpRef(1), OpRef(0)]),
            Op::new(OpCode::Jump, &[OpRef(2)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 3 peeled + Label + 3 body + Jump = 8
        assert_eq!(result.len(), 8);

        // Peeled iteration refs:
        let p0 = result[0].pos;
        let p1 = result[1].pos;
        let p2 = result[2].pos;
        assert_eq!(result[1].args[0], p0, "peeled v1 should ref peeled v0");
        assert_eq!(result[2].args[0], p1, "peeled v2 should ref peeled v1");
        assert_eq!(result[2].args[1], p0, "peeled v2 should ref peeled v0");

        // Body refs:
        let b0 = result[4].pos;
        let b1 = result[5].pos;
        let b2 = result[6].pos;
        assert_eq!(result[5].args[0], b0, "body v1 should ref body v0");
        assert_eq!(result[6].args[0], b1, "body v2 should ref body v1");
        assert_eq!(result[6].args[1], b0, "body v2 should ref body v0");

        // Jump should reference body v2.
        let jump = &result[7];
        assert_eq!(jump.args[0], b2, "Jump should ref body v2");
    }

    #[test]
    fn test_unroll_optimizer_optimize_trace() {
        let mut unroll_opt = UnrollOptimizer::new();
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        assign_positions(&mut ops, 0);
        let result = unroll_opt.optimize_trace(&ops);
        // The optimizer processes the trace; result should not be empty
        assert!(!result.is_empty(), "optimize_trace should produce output");
    }

    #[test]
    fn test_unroll_optimizer_count_guards() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        assert_eq!(UnrollOptimizer::count_guards(&ops), 2);
    }

    #[test]
    fn test_unroll_optimizer_should_give_up() {
        let mut opt = UnrollOptimizer::new();
        opt.retrace_limit = 5;
        assert!(!opt.should_give_up(0));
        assert!(!opt.should_give_up(4));
        assert!(opt.should_give_up(5));
        assert!(opt.should_give_up(10));
    }

    #[test]
    fn test_unroll_optimizer_too_many_guards() {
        let mut opt = UnrollOptimizer::new();
        opt.retrace_limit = 5;
        opt.bridge_guard_count = 20;
        assert!(opt.too_many_guards_for_retrace(0, 15));
        assert!(!opt.too_many_guards_for_retrace(0, 25));
    }

    #[test]
    fn test_exported_state_captures_and_reimports_slot_facts() {
        use crate::info::PtrInfo;
        use crate::intutils::IntBound;
        use majit_ir::{ArrayDescr, Descr, GcRef, SizeDescr, Type};
        use std::sync::Arc;

        #[derive(Debug)]
        struct TestSizeDescr(u32);
        impl Descr for TestSizeDescr {
            fn index(&self) -> u32 {
                self.0
            }
            fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
                Some(self)
            }
        }
        impl SizeDescr for TestSizeDescr {
            fn size(&self) -> usize {
                16
            }
            fn type_id(&self) -> u32 {
                self.0
            }
            fn is_immutable(&self) -> bool {
                false
            }
            fn is_object(&self) -> bool {
                true
            }
            fn vtable(&self) -> usize {
                self.0 as usize
            }
        }

        #[derive(Debug)]
        struct TestArrayDescr(u32);
        impl Descr for TestArrayDescr {
            fn index(&self) -> u32 {
                self.0
            }
            fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
                Some(self)
            }
        }
        impl ArrayDescr for TestArrayDescr {
            fn base_size(&self) -> usize {
                24
            }
            fn item_size(&self) -> usize {
                8
            }
            fn type_id(&self) -> u32 {
                self.0
            }
            fn item_type(&self) -> Type {
                Type::Ref
            }
        }

        let mut ctx = crate::OptContext::with_num_inputs(8, 0);
        ctx.make_constant(OpRef(10), Value::Int(42));
        ctx.make_constant(OpRef(13), Value::Ref(GcRef(0x5678)));
        ctx.set_ptr_info(
            OpRef(14),
            PtrInfo::instance(Some(Arc::new(TestSizeDescr(91))), Some(GcRef(0x7777))),
        );
        ctx.set_ptr_info(
            OpRef(15),
            PtrInfo::array(Arc::new(TestArrayDescr(92)), IntBound::bounded(4, 32)),
        );
        ctx.set_ptr_info(
            OpRef(11),
            PtrInfo::KnownClass {
                class_ptr: GcRef(0x1234),
                is_nonnull: true,
            },
        );
        ctx.int_lower_bounds.insert(OpRef(12), 7);

        let opt = OptUnroll::new();
        let exported = opt.export_state(
            &[OpRef(10), OpRef(11), OpRef(12), OpRef(13), OpRef(14), OpRef(15)],
            &[],
            &ctx,
        );

        assert!(exported.exported_infos.len() >= 6);
        assert_eq!(exported.exported_infos[&OpRef(10)].constant, Some(Value::Int(42)));
        assert_eq!(exported.exported_infos[&OpRef(11)].known_class, Some(GcRef(0x1234)));
        assert_eq!(exported.exported_infos[&OpRef(12)].int_lower_bound, Some(7));
        assert_eq!(exported.exported_infos[&OpRef(13)].constant, Some(Value::Ref(GcRef(0x5678))));
        assert_eq!(exported.exported_infos[&OpRef(14)].ptr_kind, ExportedPtrKind::Instance);
        assert_eq!(exported.exported_infos[&OpRef(14)].ptr_descr.as_ref().map(|d| d.index()), Some(91));
        assert_eq!(exported.exported_infos[&OpRef(14)].known_class, Some(GcRef(0x7777)));
        assert_eq!(exported.exported_infos[&OpRef(15)].ptr_kind, ExportedPtrKind::Array);
        assert_eq!(exported.exported_infos[&OpRef(15)].ptr_descr.as_ref().map(|d| d.index()), Some(92));
        assert_eq!(
            exported.exported_infos[&OpRef(15)]
                .array_lenbound
                .as_ref()
                .map(|b| (b.lower, b.upper)),
            Some((4, 32))
        );

        let mut ctx2 = crate::OptContext::with_num_inputs(8, 6);
        let label_args = opt.import_state(
            &[OpRef(0), OpRef(1), OpRef(2), OpRef(3), OpRef(4), OpRef(5)],
            &exported,
            &mut ctx2,
        );

        // import_state forwards source → target (RPython set_forwarded)
        // so make_inputargs returns the forwarded (original) OpRefs
        assert_eq!(
            label_args,
            vec![OpRef(11), OpRef(12), OpRef(14), OpRef(15)]
        );
        // setinfo_from_preamble first resolves source -> target, so facts land
        // on the forwarded target boxes.
        assert_eq!(ctx2.get_constant_int(OpRef(10)), Some(42));
        match ctx2.get_ptr_info(OpRef(11)) {
            Some(PtrInfo::KnownClass { class_ptr, is_nonnull }) => {
                assert_eq!(class_ptr.as_usize(), 0x1234);
                assert!(*is_nonnull);
            }
            other => panic!("expected known-class ptr info, got {:?}", other),
        }
        assert_eq!(ctx2.int_lower_bounds.get(&OpRef(12)).copied(), Some(7));
        match ctx2.get_ptr_info(OpRef(13)) {
            Some(PtrInfo::Constant(ptr)) => assert_eq!(ptr.as_usize(), 0x5678),
            other => panic!("expected constant ptr info, got {:?}", other),
        }
        match ctx2.get_ptr_info(OpRef(14)) {
            Some(PtrInfo::Instance(info)) => {
                assert_eq!(info.descr.as_ref().map(|d| d.index()), Some(91));
                assert_eq!(info.known_class, Some(GcRef(0x7777)));
            }
            other => panic!("expected instance ptr info, got {:?}", other),
        }
        match ctx2.get_ptr_info(OpRef(15)) {
            Some(PtrInfo::Array(info)) => {
                assert_eq!(info.descr.index(), 92);
                assert_eq!((info.lenbound.lower, info.lenbound.upper), (4, 32));
            }
            other => panic!("expected array ptr info, got {:?}", other),
        }
    }

    #[test]
    fn test_exported_state_reimports_widened_intbounds() {
        use crate::intutils::IntBound;

        let ctx = crate::OptContext::with_num_inputs(4, 0);
        let mut exported_bounds = std::collections::HashMap::new();
        exported_bounds.insert(OpRef(21), IntBound::bounded(10, 20));

        let exported = export_state(
            &[OpRef(21)],
            &[],
            &ctx,
            Some(&exported_bounds),
        );

        assert_eq!(
            exported.exported_infos[&OpRef(21)]
                .int_bound
                .as_ref()
                .map(|b| (b.lower, b.upper)),
            Some((10, 20))
        );

        let mut ctx2 = crate::OptContext::with_num_inputs(4, 1);
        let label_args = import_state(&[OpRef(0)], &exported, &mut ctx2);
        assert_eq!(label_args, vec![OpRef(21)]);
        assert_eq!(
            ctx2.imported_int_bounds
                .get(&OpRef(21))
                .map(|b| (b.lower, b.upper)),
            Some((10, 20))
        );
        assert_eq!(ctx2.int_lower_bounds.get(&OpRef(21)).copied(), Some(10));
    }

    #[test]
    fn test_exported_state_reimports_short_heap_field_facts() {
        let mut ctx = crate::OptContext::with_num_inputs(4, 0);
        ctx.exported_short_boxes.push(crate::shortpreamble::PreambleOp {
            op: {
                let mut op = Op::new(OpCode::GetfieldGcI, &[OpRef(10)]);
                op.pos = OpRef(11);
                op
            },
            kind: crate::shortpreamble::PreambleOpKind::Heap { descr_idx: 55 },
            label_arg_idx: Some(1),
            invented_name: false,
            same_as_source: None,
        });

        let exported = export_state(&[OpRef(10), OpRef(11)], &[], &ctx, None);
        assert_eq!(
            exported.exported_short_ops,
            vec![ExportedShortOp::HeapField {
                source: OpRef(11),
                object_slot: 0,
                descr_idx: 55,
                result_type: Type::Int,
                result: ExportedShortResult::Slot(1),
                invented_name: false,
                same_as_source: None,
            }]
        );

        let mut ctx2 = crate::OptContext::with_num_inputs(4, 2);
        let label_args = import_state(&[OpRef(0), OpRef(1)], &exported, &mut ctx2);
        assert_eq!(label_args, vec![OpRef(10), OpRef(11)]);
        assert_eq!(
            ctx2.imported_short_fields.get(&(OpRef(10), 55)).copied(),
            Some(OpRef(11))
        );
    }

    #[test]
    fn test_exported_state_reimports_loopinvariant_short_fact() {
        let mut ctx = crate::OptContext::with_num_inputs(4, 0);
        ctx.make_constant(OpRef(10), Value::Int(0x1234));
        ctx.exported_short_boxes.push(crate::shortpreamble::PreambleOp {
            op: {
                let mut op = Op::new(OpCode::CallI, &[OpRef(10), OpRef(12)]);
                op.pos = OpRef(11);
                op
            },
            kind: crate::shortpreamble::PreambleOpKind::LoopInvariant,
            label_arg_idx: Some(1),
            invented_name: false,
            same_as_source: None,
        });

        let exported = export_state(&[OpRef(10), OpRef(11)], &[], &ctx, None);
        // label_args may exclude constants, so result_slot depends on
        // make_inputargs output — just check func_ptr is correct.
        assert_eq!(exported.exported_short_ops.len(), 1);
        match &exported.exported_short_ops[0] {
            ExportedShortOp::LoopInvariant { func_ptr, .. } => {
                assert_eq!(*func_ptr, 0x1234);
            }
            other => panic!("expected LoopInvariant, got {other:?}"),
        }

        let mut ctx2 = crate::OptContext::with_num_inputs(4, 2);
        let label_args = import_state(&[OpRef(0), OpRef(1)], &exported, &mut ctx2);
        // OpRef(10) is constant → excluded from label_args by make_inputargs
        assert_eq!(label_args, vec![OpRef(11)]);
        // result_slot=0 in label_args=[OpRef(11)] → OpRef(11)
        assert_eq!(
            ctx2.imported_loop_invariant_results.get(&0x1234).copied(),
            Some(OpRef(11))
        );
    }

    #[test]
    fn test_exported_state_reimports_short_pure_fact() {
        let mut ctx = crate::OptContext::with_num_inputs(6, 0);
        ctx.make_constant(OpRef(10), Value::Int(7));
        ctx.exported_short_boxes.push(crate::shortpreamble::PreambleOp {
            op: {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef(12), OpRef(10)]);
                op.pos = OpRef(11);
                op
            },
            kind: crate::shortpreamble::PreambleOpKind::Pure,
            label_arg_idx: Some(1),
            invented_name: false,
            same_as_source: None,
        });

        let exported = export_state(&[OpRef(12), OpRef(11)], &[], &ctx, None);
        assert_eq!(
            exported.exported_short_ops,
            vec![ExportedShortOp::Pure {
                source: OpRef(11),
                opcode: OpCode::IntAdd,
                descr_idx: None,
                args: vec![ExportedShortArg::Slot(0), ExportedShortArg::Const(Value::Int(7))],
                result: ExportedShortResult::Slot(1),
                invented_name: false,
                same_as_source: None,
            }]
        );

        let mut ctx2 = crate::OptContext::with_num_inputs(6, 2);
        let label_args = import_state(&[OpRef(0), OpRef(1)], &exported, &mut ctx2);
        assert_eq!(label_args, vec![OpRef(12), OpRef(11)]);
        assert_eq!(
            ctx2.imported_short_pure_ops,
            vec![crate::ImportedShortPureOp {
                opcode: OpCode::IntAdd,
                descr_idx: None,
                args: vec![
                    crate::ImportedShortPureArg::OpRef(OpRef(12)),
                    crate::ImportedShortPureArg::Const(Value::Int(7)),
                ],
                result: OpRef(11),
            }]
        );
    }

    #[test]
    fn test_exported_state_reimports_short_call_pure_fact() {
        let mut ctx = crate::OptContext::with_num_inputs(8, 0);
        ctx.make_constant(OpRef(10), Value::Int(0x1234));
        let call_descr = majit_ir::descr::make_call_descr_full(
            77,
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            8,
            majit_ir::EffectInfo::elidable(),
        );
        ctx.exported_short_boxes.push(crate::shortpreamble::PreambleOp {
            op: {
                let mut op = Op::new(OpCode::CallI, &[OpRef(10), OpRef(12)]);
                op.pos = OpRef(11);
                op.descr = Some(call_descr);
                op
            },
            kind: crate::shortpreamble::PreambleOpKind::Pure,
            label_arg_idx: Some(1),
            invented_name: false,
            same_as_source: None,
        });

        let exported = export_state(&[OpRef(12), OpRef(11)], &[], &ctx, None);
        assert_eq!(
            exported.exported_short_ops,
            vec![ExportedShortOp::Pure {
                source: OpRef(11),
                opcode: OpCode::CallPureI,
                descr_idx: Some(77),
                args: vec![
                    ExportedShortArg::Const(Value::Int(0x1234)),
                    ExportedShortArg::Slot(0),
                ],
                result: ExportedShortResult::Slot(1),
                invented_name: false,
                same_as_source: None,
            }]
        );

        let mut ctx2 = crate::OptContext::with_num_inputs(8, 2);
        let label_args = import_state(&[OpRef(0), OpRef(1)], &exported, &mut ctx2);
        assert_eq!(label_args, vec![OpRef(12), OpRef(11)]);
        assert_eq!(
            ctx2.imported_short_pure_ops,
            vec![crate::ImportedShortPureOp {
                opcode: OpCode::CallPureI,
                descr_idx: Some(77),
                args: vec![
                    crate::ImportedShortPureArg::Const(Value::Int(0x1234)),
                    crate::ImportedShortPureArg::OpRef(OpRef(12)),
                ],
                result: OpRef(11),
            }]
        );
    }

    #[test]
    fn test_exported_state_reimports_short_pure_dependency_chain() {
        let mut ctx = crate::OptContext::with_num_inputs(10, 0);
        ctx.make_constant(OpRef(10), Value::Int(7));
        ctx.exported_short_boxes.push(crate::shortpreamble::PreambleOp {
            op: {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef(12), OpRef(10)]);
                op.pos = OpRef(11);
                op
            },
            kind: crate::shortpreamble::PreambleOpKind::Pure,
            label_arg_idx: None,
            invented_name: false,
            same_as_source: None,
        });
        ctx.exported_short_boxes.push(crate::shortpreamble::PreambleOp {
            op: {
                let mut op = Op::new(OpCode::IntMul, &[OpRef(11), OpRef(13)]);
                op.pos = OpRef(14);
                op
            },
            kind: crate::shortpreamble::PreambleOpKind::Pure,
            label_arg_idx: Some(2),
            invented_name: false,
            same_as_source: None,
        });

        let exported = export_state(&[OpRef(12), OpRef(13), OpRef(14)], &[], &ctx, None);
        assert_eq!(
            exported.exported_short_ops,
            vec![
                ExportedShortOp::Pure {
                    source: OpRef(11),
                    opcode: OpCode::IntAdd,
                    descr_idx: None,
                    args: vec![
                        ExportedShortArg::Slot(0),
                        ExportedShortArg::Const(Value::Int(7)),
                    ],
                    result: ExportedShortResult::Temporary(0),
                    invented_name: false,
                    same_as_source: None,
                },
                ExportedShortOp::Pure {
                    source: OpRef(14),
                    opcode: OpCode::IntMul,
                    descr_idx: None,
                    args: vec![
                        ExportedShortArg::Produced(0),
                        ExportedShortArg::Slot(1),
                    ],
                    result: ExportedShortResult::Slot(2),
                    invented_name: false,
                    same_as_source: None,
                },
            ]
        );

        let mut ctx2 = crate::OptContext::with_num_inputs(10, 3);
        let label_args = import_state(&[OpRef(0), OpRef(1), OpRef(2)], &exported, &mut ctx2);
        assert_eq!(label_args, vec![OpRef(12), OpRef(13), OpRef(14)]);
        assert_eq!(ctx2.imported_short_pure_ops.len(), 2);
        let temp_result = ctx2.imported_short_pure_ops[0].result;
        assert_ne!(temp_result, OpRef(12));
        assert_ne!(temp_result, OpRef(13));
        assert_ne!(temp_result, OpRef(14));
        assert_eq!(
            ctx2.imported_short_pure_ops,
            vec![
                crate::ImportedShortPureOp {
                    opcode: OpCode::IntAdd,
                    descr_idx: None,
                    args: vec![
                        crate::ImportedShortPureArg::OpRef(OpRef(12)),
                        crate::ImportedShortPureArg::Const(Value::Int(7)),
                    ],
                    result: temp_result,
                },
                crate::ImportedShortPureOp {
                    opcode: OpCode::IntMul,
                    descr_idx: None,
                    args: vec![
                        crate::ImportedShortPureArg::OpRef(temp_result),
                        crate::ImportedShortPureArg::OpRef(OpRef(13)),
                    ],
                    result: OpRef(14),
                },
            ]
        );
    }

    #[test]
    fn test_imported_short_builder_tracks_used_dependency_chain() {
        let mut ctx = crate::OptContext::with_num_inputs(10, 0);
        ctx.make_constant(OpRef(10), Value::Int(7));
        ctx.exported_short_boxes.push(crate::shortpreamble::PreambleOp {
            op: {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef(12), OpRef(10)]);
                op.pos = OpRef(11);
                op
            },
            kind: crate::shortpreamble::PreambleOpKind::Pure,
            label_arg_idx: None,
            invented_name: false,
            same_as_source: None,
        });
        ctx.exported_short_boxes.push(crate::shortpreamble::PreambleOp {
            op: {
                let mut op = Op::new(OpCode::IntMul, &[OpRef(11), OpRef(13)]);
                op.pos = OpRef(14);
                op
            },
            kind: crate::shortpreamble::PreambleOpKind::Pure,
            label_arg_idx: Some(2),
            invented_name: false,
            same_as_source: None,
        });

        let exported = export_state(&[OpRef(12), OpRef(13), OpRef(14)], &[], &ctx, None);
        let mut ctx2 = crate::OptContext::with_num_inputs(10, 3);
        import_state(&[OpRef(0), OpRef(1), OpRef(2)], &exported, &mut ctx2);
        ctx2.note_imported_short_use(OpRef(14));
        let sp = ctx2.build_imported_short_preamble().unwrap();

        assert_eq!(sp.ops.len(), 2);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
        assert_eq!(sp.ops[1].op.opcode, OpCode::IntMul);
    }

    #[test]
    fn test_exported_state_reimports_invented_short_alias_metadata() {
        let mut ctx = crate::OptContext::with_num_inputs(6, 0);
        ctx.exported_short_boxes.push(crate::shortpreamble::PreambleOp {
            op: {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef(12), OpRef(13)]);
                op.pos = OpRef(30);
                op
            },
            kind: crate::shortpreamble::PreambleOpKind::Pure,
            label_arg_idx: None,
            invented_name: true,
            same_as_source: Some(OpRef(14)),
        });

        let exported = export_state(&[OpRef(12), OpRef(13), OpRef(14)], &[], &ctx, None);
        assert_eq!(
            exported.exported_short_ops,
            vec![ExportedShortOp::Pure {
                source: OpRef(30),
                opcode: OpCode::IntAdd,
                descr_idx: None,
                args: vec![ExportedShortArg::Slot(0), ExportedShortArg::Slot(1)],
                result: ExportedShortResult::Temporary(0),
                invented_name: true,
                same_as_source: Some(OpRef(14)),
            }]
        );

        let mut ctx2 = crate::OptContext::with_num_inputs(6, 3);
        import_state(&[OpRef(0), OpRef(1), OpRef(2)], &exported, &mut ctx2);
        assert_eq!(ctx2.imported_short_pure_ops.len(), 1);
        assert_eq!(
            ctx2.imported_short_aliases,
            vec![crate::ImportedShortAlias {
                result: ctx2.imported_short_pure_ops[0].result,
                same_as_source: OpRef(14),
                same_as_opcode: OpCode::SameAsI,
            }]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_emits_extra_same_as_before_label() {
        let p1_ops = vec![{
            let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
            op.pos = OpRef(3);
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::IntMul, &[OpRef(50), OpRef(0)]);
                op.pos = OpRef(1);
                op
            },
            Op::new(OpCode::Jump, &[OpRef(50)]),
        ];

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef(10)],
            &[],
            None,
            1,
            &[crate::ImportedShortAlias {
                result: OpRef(50),
                same_as_source: OpRef(10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &[crate::ImportedShortSource {
                result: OpRef(50),
                source: OpRef(10),
            }],
        );

        assert_eq!(combined[0].opcode, OpCode::IntAdd);
        assert_eq!(combined[1].opcode, OpCode::SameAsI);
        assert_eq!(combined[1].args.as_slice(), &[OpRef(10)]);
        assert_eq!(combined[2].opcode, OpCode::Label);
        assert_eq!(combined[3].opcode, OpCode::IntMul);
        assert_eq!(combined[3].args[0], combined[1].pos);
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].args[0], combined[1].pos);
    }

    #[test]
    fn test_inline_short_preamble_returns_extra_used_boxes() {
        let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
        op.pos = OpRef(7);
        let sp = crate::shortpreamble::ShortPreamble {
            ops: vec![crate::shortpreamble::ShortPreambleOp {
                op,
                arg_mapping: vec![(0, 0), (1, 1)],
                fail_arg_mapping: Vec::new(),
            }],
            inputargs: vec![OpRef(0), OpRef(1)],
            used_boxes: vec![OpRef(7)],
            exported_state: None,
        };

        let mut ctx = crate::OptContext::with_num_inputs(8, 2);
        let extra = OptUnroll::inline_short_preamble(
            &[OpRef(10), OpRef(11)],
            &[OpRef(10), OpRef(11)],
            &sp,
            None,
            &mut ctx,
        );

        assert_eq!(ctx.new_operations.len(), 1);
        assert_eq!(ctx.new_operations[0].opcode, OpCode::IntAdd);
        assert_eq!(ctx.new_operations[0].args.as_slice(), &[OpRef(10), OpRef(11)]);
        assert_eq!(extra, vec![ctx.new_operations[0].pos]);
    }

    #[test]
    fn test_assemble_peeled_trace_extends_label_with_used_boxes() {
        let p1_ops = vec![{
            let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
            op.pos = OpRef(3);
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::IntMul, &[OpRef(50), OpRef(0)]);
                op.pos = OpRef(1);
                op
            },
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(50)]),
        ];

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef(10)],
            &[OpRef(50)],
            None,
            1,
            &[crate::ImportedShortAlias {
                result: OpRef(50),
                same_as_source: OpRef(10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &[crate::ImportedShortSource {
                result: OpRef(50),
                source: OpRef(10),
            }],
        );

        assert_eq!(combined[2].opcode, OpCode::Label);
        assert_eq!(combined[2].args.as_slice(), &[OpRef(10), combined[1].pos]);
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].args.as_slice(), &[OpRef(10), combined[1].pos]);
    }

    #[test]
    fn test_import_state_rebuilds_nested_virtual_graph_from_exported_infos() {
        use crate::info::{PtrInfo, VirtualStructInfo};
        use majit_ir::{Type, Value};
        use majit_ir::descr::{make_field_descr_full, make_size_descr_full};

        let value_descr = make_field_descr_full(101, 0, 8, Type::Int, false);
        let next_descr = make_field_descr_full(102, 8, 8, Type::Ref, false);
        let node_descr = make_size_descr_full(200, 16, 0);

        let mut ctx = crate::OptContext::with_num_inputs(16, 0);
        ctx.make_constant(OpRef(12), Value::Int(3));
        ctx.make_constant(OpRef(13), Value::Int(2));
        ctx.set_ptr_info(OpRef(15), PtrInfo::NonNull);
        ctx.set_ptr_info(
            OpRef(11),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: node_descr.clone(),
                fields: vec![(value_descr.index(), OpRef(12)), (next_descr.index(), OpRef(15))],
                field_descrs: vec![
                    (value_descr.index(), value_descr.clone()),
                    (next_descr.index(), next_descr.clone()),
                ],
            }),
        );
        ctx.set_ptr_info(
            OpRef(10),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: node_descr.clone(),
                fields: vec![(value_descr.index(), OpRef(13)), (next_descr.index(), OpRef(11))],
                field_descrs: vec![
                    (value_descr.index(), value_descr.clone()),
                    (next_descr.index(), next_descr.clone()),
                ],
            }),
        );

        let exported = export_state(&[OpRef(10)], &[], &ctx, None);
        let mut ctx2 = crate::OptContext::with_num_inputs(16, 1);
        let label_args = import_state(&[OpRef(0)], &exported, &mut ctx2);

        assert_eq!(label_args, vec![OpRef(15)]);

        match ctx2.get_ptr_info(OpRef(10)) {
            Some(PtrInfo::VirtualStruct(info)) => {
                assert_eq!(info.fields, vec![(value_descr.index(), OpRef(13)), (next_descr.index(), OpRef(11))]);
            }
            other => panic!("expected virtual root, got {other:?}"),
        }
        match ctx2.get_ptr_info(OpRef(11)) {
            Some(PtrInfo::VirtualStruct(info)) => {
                assert_eq!(info.fields, vec![(value_descr.index(), OpRef(12)), (next_descr.index(), OpRef(15))]);
            }
            other => panic!("expected nested virtual, got {other:?}"),
        }
        match ctx2.get_ptr_info(OpRef(15)) {
            Some(PtrInfo::NonNull) => {}
            other => panic!("expected concrete tail info, got {other:?}"),
        }
    }
}
