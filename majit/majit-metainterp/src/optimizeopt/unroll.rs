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
use std::sync::{Arc, Mutex};

use majit_ir::{DescrRef, GcRef, Op, OpCode, OpRef, Type, Value};

use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

fn is_trace_constant_ref(opref: OpRef, constants: &HashMap<u32, i64>) -> bool {
    !opref.is_none() && constants.contains_key(&opref.0)
}

fn is_trace_runtime_ref(opref: OpRef, constants: &HashMap<u32, i64>) -> bool {
    !opref.is_none() && !is_trace_constant_ref(opref, constants)
}

/// unroll.py: UnrollOptimizer — high-level loop optimization controller.
///
/// Wraps the streaming OptUnroll pass with RPython's UnrollOptimizer API:
/// - optimize_preamble: process and optimize the first iteration
/// - optimize_peeled_loop: optimize the main loop body
pub struct UnrollOptimizer {
    /// The short preamble from the preamble optimization pass.
    pub short_preamble: Option<crate::optimizeopt::shortpreamble::ShortPreamble>,
    /// history.py: JitCellToken.target_tokens — compiled versions of this loop.
    /// Each TargetToken has its own virtual state and short preamble.
    pub target_tokens: Vec<TargetToken>,
    /// history.py: JitCellToken.retraced_count — number of times this loop
    /// has been retraced. Compared against retrace_limit to prevent infinite
    /// retracing.
    pub retraced_count: u32,
    /// warmstate.py: retrace_limit parameter. When retraced_count reaches
    /// this limit, jump_to_preamble is forced instead of creating a new
    /// target token.
    pub retrace_limit: u32,
    /// warmstate.py: max_retrace_guards parameter. If a compiled trace has
    /// more guards than this, retracing is permanently disabled.
    pub max_retrace_guards: u32,
    /// Constant type hints from ConstantPool, propagated to inner Optimizer.
    pub constant_types: std::collections::HashMap<u32, majit_ir::Type>,
    /// compile.py:362: pre-imported ExportedState for compile_retrace.
    /// When set, Phase 1 (preamble) is skipped and Phase 2 uses this state
    /// directly, matching UnrolledLoopData.optimize → optimize_peeled_loop.
    pub imported_state: Option<ExportedState>,
    // RPython compile.py:278-284: Phase 1 results for retrace_needed.
    // In RPython, Phase 1 and Phase 2 are separate calls, so Phase 1
    // results are naturally accessible. In pyre, Phase 1 results are
    // returned via the phase1_out output parameter to the caller's
    // stack frame (survives Phase 2 panic).
    /// resume.py parity: per-guard snapshot boxes from tracing time.
    /// Passed through to Phase 1 and Phase 2 optimizers for
    /// store_final_boxes_in_guard snapshot-based fail_args rebuild.
    pub snapshot_boxes: std::collections::HashMap<i32, Vec<majit_ir::OpRef>>,
    /// Per-frame box counts for multi-frame snapshots.
    pub snapshot_frame_sizes: std::collections::HashMap<i32, Vec<usize>>,
    /// Per-guard virtualizable boxes from tracing-time snapshots.
    pub snapshot_vable_boxes: std::collections::HashMap<i32, Vec<majit_ir::OpRef>>,
    /// Per-guard per-frame (jitcode_index, pc) from tracing-time snapshots.
    pub snapshot_frame_pcs: std::collections::HashMap<i32, Vec<(i32, i32)>>,
    /// RPython box.type parity: each snapshot Box carries its type from tracing.
    /// Used by InlineBoxEnv.get_type() for correct _number_boxes virtual detection.
    pub snapshot_box_types: std::collections::HashMap<u32, majit_ir::Type>,
    /// resume.py:570-574 _add_optimizer_sections: per-guard optimizer
    /// knowledge collected during optimization. Propagated to CompiledTrace
    /// for bridge compilation.
    pub per_guard_knowledge: Vec<(
        majit_ir::OpRef,
        crate::optimizeopt::optimizer::OptimizerKnowledge,
    )>,
    /// RPython parity: GcRef constants that need Ref type for resume data.
    pub numbering_type_overrides: std::collections::HashMap<u32, majit_ir::Type>,
    /// RPython Box type parity: trace inputarg types from recorder.
    /// Each RPython Box carries its type; in majit OpRef is untyped u32.
    /// Propagated to Phase 1 and Phase 2 Optimizer.trace_inputarg_types
    /// so value_types covers inputarg OpRefs.
    pub trace_inputarg_types: Vec<majit_ir::Type>,
    /// RPython Box type parity: position→type for ALL original trace ops.
    /// snapshot_boxes reference original trace positions, but the optimizer
    /// receives transformed ops (fold_box/elide). This map covers the gap.
    pub original_trace_op_types: std::collections::HashMap<u32, majit_ir::Type>,
    /// RPython Box type parity: Phase 1's emitted op types.
    /// Phase 2 references Phase 1 OpRefs via imported_label_args (NONE
    /// resolution). These OpRefs may not exist in Phase 2's input ops.
    phase1_value_types: std::collections::HashMap<u32, majit_ir::Type>,
    /// RPython: same Optimizer instance across phases keeps patchguardop.
    /// In majit, separate instances — forward explicitly.
    phase1_patchguardop: Option<majit_ir::Op>,
    /// Cross-phase fresh OpRef high water (majit-specific companion to
    /// RPython's `TraceIterator._index`).
    ///
    /// In RPython each `TraceIterator.next()` allocates a fresh `cls()`
    /// ResOperation whose Python identity distinguishes Phase 1 from
    /// Phase 2 boxes; majit's `OpRef(u32)` IS the identity, so Phase 2 must
    /// continue allocating *above* Phase 1's high water mark to keep the
    /// two phases' OpRef sets disjoint. After Phase 1 finishes,
    /// `next_global_opref` holds the smallest OpRef Phase 2 may emit; it
    /// is the `start_fresh` argument the next `TraceIterator::new` call
    /// (or bridge entry) should use. Initialized to 0.
    #[allow(dead_code)]
    pub(crate) next_global_opref: u32,
    /// RPython metainterp_sd.callinfocollection parity.
    /// Maps oopspec indices to (calldescr, func_ptr) for generate_modified_call.
    pub callinfocollection: Option<std::sync::Arc<majit_ir::descr::CallInfoCollection>>,
}

impl UnrollOptimizer {
    pub fn new() -> Self {
        UnrollOptimizer {
            short_preamble: None,
            target_tokens: Vec::new(),
            retraced_count: 0,
            constant_types: std::collections::HashMap::new(),
            numbering_type_overrides: std::collections::HashMap::new(),
            retrace_limit: 5,
            max_retrace_guards: 15,
            imported_state: None,
            snapshot_boxes: std::collections::HashMap::new(),
            snapshot_frame_sizes: std::collections::HashMap::new(),
            snapshot_vable_boxes: std::collections::HashMap::new(),
            snapshot_frame_pcs: std::collections::HashMap::new(),
            snapshot_box_types: std::collections::HashMap::new(),
            per_guard_knowledge: Vec::new(),
            trace_inputarg_types: Vec::new(),
            original_trace_op_types: std::collections::HashMap::new(),
            phase1_value_types: std::collections::HashMap::new(),
            phase1_patchguardop: None,
            next_global_opref: 0,
            callinfocollection: None,
        }
    }

    /// unroll.py: optimize_preamble(trace, runtime_boxes)
    /// Optimize the preamble (first iteration) of a loop trace.
    /// Returns the optimized preamble ops + the peeled loop ops.
    pub fn optimize_preamble(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::default_pipeline();
        optimizer.add_pass(Box::new(OptUnroll::new()));
        optimizer.propagate_all_forward(ops)
    }

    /// unroll.py: optimize_peeled_loop(trace)
    /// Optimize the loop body AFTER preamble peeling.
    /// The peeled preamble has already established the type/class/bounds
    /// information; this method optimizes the repeating body.
    pub fn optimize_peeled_loop(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::default_pipeline();
        optimizer.propagate_all_forward(ops)
    }

    /// unroll.py:238-242: jump_to_preamble(cell_token, jump_op).
    ///
    /// Redirect the closing JUMP to the preamble entry token
    /// (target_tokens[0], virtual_state=None). Only changes the
    /// descriptor, keeping arglist intact — RPython parity.
    pub fn jump_to_preamble(body_ops: &[Op], preamble_target: &TargetToken) -> Vec<Op> {
        assert!(
            preamble_target.virtual_state.is_none(),
            "jump_to_preamble expects the start/preamble target token"
        );
        let mut result = body_ops.to_vec();
        if let Some(jump) = result.iter_mut().rfind(|op| op.opcode == OpCode::Jump) {
            jump.descr = Some(preamble_target.as_jump_target_descr());
        }
        result
    }

    fn ensure_preamble_target_token(&mut self) {
        if self
            .target_tokens
            .first()
            .is_some_and(|token| token.virtual_state.is_none())
        {
            return;
        }
        self.target_tokens.insert(0, TargetToken::new_preamble(0));
    }

    /// unroll.py: optimize_trace(trace, call_pure_results)
    /// Full trace optimization: peel → optimize preamble → optimize body.
    /// Returns the optimized peeled+body trace.
    pub fn optimize_trace(&mut self, ops: &[Op]) -> Vec<Op> {
        let result = self.optimize_preamble(ops);
        // After peeling, extract short preamble from the result.
        let sp = crate::optimizeopt::shortpreamble::extract_short_preamble(&result);
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
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::default_pipeline();
        optimizer.add_pass(Box::new(OptUnroll::new()));
        let result = optimizer.optimize_with_constants(ops, constants);
        let sp = crate::optimizeopt::shortpreamble::extract_short_preamble(&result);
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
        vable_config: Option<crate::optimizeopt::virtualize::VirtualizableConfig>,
    ) -> (Vec<Op>, usize) {
        self.optimize_trace_with_constants_and_inputs_vable_out(
            ops,
            constants,
            num_inputs,
            vable_config,
            None,
        )
    }

    /// Same as optimize_trace_with_constants_and_inputs_vable but with an
    /// output parameter for Phase 1 results. RPython compile.py:278-294
    /// parity: Phase 1 results (preamble_ops + exported_state) are written
    /// to `phase1_out` before Phase 2 starts. If Phase 2 panics, the caller
    /// still has the Phase 1 results for retrace_needed.
    pub fn optimize_trace_with_constants_and_inputs_vable_out(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
        vable_config: Option<crate::optimizeopt::virtualize::VirtualizableConfig>,
        phase1_out: Option<&mut Option<(Vec<Op>, ExportedState)>>,
    ) -> (Vec<Op>, usize) {
        // compile.py:362: if imported_state is pre-set (compile_retrace path),
        // skip Phase 1 and go directly to Phase 2 with the imported state.
        let (mut exported_state, consts_p1, p1_ops) = if let Some(pre_imported) =
            self.imported_state.take()
        {
            // Retrace path: Phase 1 already done; preamble ops are in
            // the caller's partial_trace, not produced here.
            // RPython: same Optimizer persists patchguardop. Recover here.
            if self.phase1_patchguardop.is_none() {
                self.phase1_patchguardop = pre_imported.patchguardop.clone();
            }
            (pre_imported, constants.clone(), Vec::new())
        } else {
            // ── Phase 1: PreambleCompileData.optimize() ──
            // ── Phase 1: optimize_preamble (compile.py:275-276) ──
            let mut consts_p1 = constants.clone();
            let mut opt_p1 = match vable_config.as_ref() {
                Some(c) => {
                    crate::optimizeopt::optimizer::Optimizer::default_pipeline_with_virtualizable(
                        c.clone(),
                    )
                }
                None => crate::optimizeopt::optimizer::Optimizer::default_pipeline(),
            };
            opt_p1.constant_types = self.constant_types.clone();
            opt_p1.callinfocollection = self.callinfocollection.clone();
            opt_p1.numbering_type_overrides = self.numbering_type_overrides.clone();
            opt_p1.trace_inputarg_types = self.trace_inputarg_types.clone();
            opt_p1.original_trace_op_types = self.original_trace_op_types.clone();
            opt_p1.snapshot_boxes = self.snapshot_boxes.clone();
            opt_p1.snapshot_frame_sizes = self.snapshot_frame_sizes.clone();
            opt_p1.snapshot_vable_boxes = self.snapshot_vable_boxes.clone();
            opt_p1.snapshot_frame_pcs = self.snapshot_frame_pcs.clone();
            opt_p1.snapshot_box_types = self.snapshot_box_types.clone();
            // RPython optimize_preamble (unroll.py:101-103): flush=False.
            // JUMP/FINISH is NOT sent through the pass pipeline; it's
            // returned in info.jump_op for Phase 2 to consume.
            opt_p1.skip_flush = true;
            // RPython unroll.py:101-103 `optimize_preamble` calls
            // `propagate_all_forward(trace.get_iter())`. `trace.get_iter()`
            // is a fresh `TraceIterator` whose `next()` produces a freshly
            // allocated `cls()` ResOperation for every visited op.
            //
            // Phase 1 routes the input ops through `TraceIterator::new`
            // with `start_fresh = 0`. The recorder emits ops at
            // monotonically increasing positions starting from
            // `num_inputs` (recorder.rs `record_op` uses `op_count` for
            // BOTH inputargs and ops, and ops follow inputargs), and
            // `TraceIterator::next` allocates fresh OpRefs from `_fresh`
            // which is also seeded at `num_inputs` after the inputarg
            // pre-seed loop. Both void and non-void ops advance `_fresh`
            // (see opencoder.rs::next), so the freshly produced OpRef
            // sequence is bit-identical to the input — this wrap is a
            // structural alignment with RPython's `trace.get_iter()`
            // call site, not a functional change.
            let mut p1_iter = majit_trace::opencoder::TraceIterator::new(
                ops,
                0,
                ops.len(),
                None,
                num_inputs,
                0, // start_fresh = 0 — inputargs at [0..num_inputs)
            );
            let mut p1_ops_in: Vec<Op> = Vec::with_capacity(ops.len());
            while let Some(op) = p1_iter.next() {
                p1_ops_in.push(op);
            }
            let p1_ops =
                opt_p1.optimize_with_constants_and_inputs(&p1_ops_in, &mut consts_p1, num_inputs);
            // RPython parity: Phase 1 optimizer may discover new constants
            // via make_constant (e.g., constant-folded heap reads, guard
            // class pointers). These live in ctx.constants but not in
            // consts_p1 (which was only seeded from the input constants).
            // Merge them back so build_short_preamble_from_exported_boxes
            // can capture all constants referenced by short preamble ops.
            if let Some(ref final_ctx) = opt_p1.final_ctx {
                for (idx, val) in final_ctx.constants.iter().enumerate() {
                    if let Some(v) = val {
                        let raw = match v {
                            majit_ir::Value::Int(v) => *v,
                            majit_ir::Value::Float(f) => f.to_bits() as i64,
                            majit_ir::Value::Ref(r) => r.0 as i64,
                            majit_ir::Value::Void => 0,
                        };
                        consts_p1.entry(idx as u32).or_insert(raw);
                    }
                }
                for (&const_idx, v) in &final_ctx.const_pool {
                    let key = OpRef::from_const(const_idx).0;
                    let raw = match v {
                        majit_ir::Value::Int(v) => *v,
                        majit_ir::Value::Float(f) => f.to_bits() as i64,
                        majit_ir::Value::Ref(r) => r.0 as i64,
                        majit_ir::Value::Void => 0,
                    };
                    consts_p1.entry(key).or_insert(raw);
                }
            }
            let p1_ni = opt_p1.final_num_inputs();

            match opt_p1.exported_loop_state.take() {
                Some(mut state) => {
                    // Determine types of end_args from Phase 1's output ops.
                    {
                        let op_types: HashMap<OpRef, Type> = p1_ops
                            .iter()
                            .filter(|op| !op.pos.is_none())
                            .map(|op| (op.pos, op.opcode.result_type()))
                            .collect();
                        state.end_arg_types = state
                            .end_args
                            .iter()
                            .map(|&arg| op_types.get(&arg).copied().unwrap_or(Type::Ref))
                            .collect();
                    }
                    // Export Phase 1's heap cache for Phase 2.
                    state.preamble_heap_cache = opt_p1.export_all_cached_fields();
                    // opencoder.py:271 _index parity: Phase 2's TraceIterator
                    // must allocate fresh boxes ABOVE Phase 1's high water
                    // mark so the two phases' OpRef namespaces are disjoint
                    // (RPython relies on Python identity to distinguish them;
                    // majit relies on disjoint integer ranges). The high
                    // water is `final_ctx.next_pos` after Phase 1 emit, with
                    // a floor of `num_inputs` for empty traces.
                    self.next_global_opref = opt_p1
                        .final_ctx
                        .as_ref()
                        .map(|c| c.next_pos)
                        .unwrap_or(num_inputs as u32)
                        .max(num_inputs as u32);
                    // RPython Box type parity: Phase 1's emitted op types
                    // must be accessible to Phase 2 (imported_label_args
                    // reference Phase 1 OpRefs). Save Phase 1 value_types.
                    self.phase1_value_types = opt_p1.prev_phase_value_types.clone();
                    // RPython: same Optimizer instance keeps patchguardop.
                    if self.phase1_patchguardop.is_none() {
                        self.phase1_patchguardop = opt_p1.patchguardop.clone();
                    }
                    state.patchguardop = opt_p1.patchguardop.clone();
                    // resume.py:570-574: collect Phase 1 per-guard knowledge.
                    self.per_guard_knowledge
                        .extend(opt_p1.per_guard_knowledge.drain(..));
                    (state, consts_p1, p1_ops)
                }
                None => {
                    *constants = consts_p1;
                    // RPython: compile_loop uses flush=True — terminal op
                    // (Finish/Jump) goes through the pass pipeline normally.
                    // majit: flush=False stores it in terminal_op; restore it
                    // here for non-peeled traces that return directly.
                    let mut ops = p1_ops;
                    if let Some(terminal) = opt_p1.terminal_op.take() {
                        ops.push(terminal);
                    }
                    let loop_arity = closing_loop_contract_arity(&ops, p1_ni);
                    return (ops, loop_arity);
                }
            }
        };
        // Determine types of end_args from Phase 1's output ops.
        {
            let op_types: HashMap<OpRef, Type> = p1_ops
                .iter()
                .filter(|op| !op.pos.is_none())
                .map(|op| (op.pos, op.opcode.result_type()))
                .collect();
            exported_state.end_arg_types = exported_state
                .end_args
                .iter()
                .map(|&arg| op_types.get(&arg).copied().unwrap_or(Type::Ref))
                .collect();
        }
        // RPython parity: Phase 2 needs patchguardop from Phase 1's
        // GuardFutureCondition (unroll.py:333). Extract before dropping opt_p1.
        let p1_patchguardop = exported_state.patchguardop.clone();

        self.ensure_preamble_target_token();
        // ── Phase 2: optimize_peeled_loop (compile.py:291-292) ──
        let body_num_inputs = num_inputs;

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] preamble peeling: {} virtual(s), phase1 end_args={} p1_patchguardop={}",
                exported_state
                    .virtual_state
                    .state
                    .iter()
                    .filter(|s| s.is_virtual())
                    .count(),
                exported_state.end_args.len(),
                p1_patchguardop
                    .as_ref()
                    .map(|p| p.rd_resume_position)
                    .unwrap_or(-99),
            );
        }

        // opencoder.py:259-404 parity: Phase 2 uses the same ops as Phase 1.
        // RPython TraceIterator creates fresh Box objects per phase — each
        // iterator has its own _cache, so Phase 2 results never collide with
        // Phase 1. In majit, Phase 2 gets a separate OptContext, achieving
        // the same isolation.
        let mut consts_p2 = consts_p1.clone();

        let mut opt_p2 = match vable_config.as_ref() {
            Some(c) => {
                crate::optimizeopt::optimizer::Optimizer::default_pipeline_with_virtualizable(
                    c.clone(),
                )
            }
            None => crate::optimizeopt::optimizer::Optimizer::default_pipeline(),
        };
        opt_p2.constant_types = self.constant_types.clone();
        opt_p2.callinfocollection = self.callinfocollection.clone();
        opt_p2.numbering_type_overrides = self.numbering_type_overrides.clone();
        opt_p2.trace_inputarg_types = self.trace_inputarg_types.clone();
        opt_p2.original_trace_op_types = self.original_trace_op_types.clone();
        opt_p2.prev_phase_value_types = self.phase1_value_types.clone();
        opt_p2.snapshot_boxes = self.snapshot_boxes.clone();
        opt_p2.snapshot_frame_sizes = self.snapshot_frame_sizes.clone();
        opt_p2.snapshot_vable_boxes = self.snapshot_vable_boxes.clone();
        opt_p2.snapshot_frame_pcs = self.snapshot_frame_pcs.clone();
        opt_p2.snapshot_box_types = self.snapshot_box_types.clone();
        // RPython: same Optimizer instance keeps patchguardop across phases.
        // Phase 1 processes GUARD_FUTURE_CONDITION (from close_loop_args_at)
        // which sets patchguardop. optimizer.py:294 parity — no synthetic
        // fallback; the actual GFC provides rd_resume_position.
        opt_p2.patchguardop = self.phase1_patchguardop.clone();
        // gcreftracer.py parity: root GcRef values on the shadow stack.
        // RPython: single Python object — GC traces automatically.
        // Rust: LIFO shadow stack requires longer-lived roots at lower depth.
        //
        // Order: (1) phase1_out clone rooted first (survives beyond this
        // function for retrace — pyre-specific panic safety backup).
        // (2) original rooted second (lives until opt_p2 drops at Phase 2
        // end — shorter-lived, higher depth, dropped first).
        if let Some(out) = phase1_out {
            let mut backup = exported_state.clone();
            backup.root_all_gcrefs();
            *out = Some((p1_ops.clone(), backup));
        }
        exported_state.root_all_gcrefs();
        opt_p2.imported_loop_state = Some(exported_state);
        // Set imported_virtuals so Phase 2 intercepts GetfieldGcR(pool)
        // and sets up VirtualStruct PtrInfo for the imported head.
        // Virtual structure is derived from VirtualState (ExportedState).
        opt_p2.imported_virtuals =
            build_imported_virtuals_from_state(opt_p2.imported_loop_state.as_ref().unwrap());
        // RPython: propagate_all_forward(trace, flush=False) for Phase 2.
        // Don't flush lazy sets — virtuals remain virtual until JUMP handling.
        opt_p2.skip_flush = true;
        // RPython parity: Phase 2 DOES virtualize New(). Guard recovery uses
        // rd_virtuals (generated by finalize_guard_resume_data)
        // for virtual materialization on guard failure.
        // Previously disabled (set_phase2(true)) due to missing rd_virtuals;
        // now enabled after compile.rs rd_virtuals→rd_virtuals generation.
        // RPython parity: Phase 2 imports heap cache via short preamble
        // RPython: Phase 2 heap cache is populated by inline_short_preamble
        // replaying HeapOps through send_extra_operation.
        if std::env::var_os("MAJIT_LOG").is_some() {
            let gc_before = ops.iter().filter(|o| o.opcode.is_guard()).count();
            eprintln!(
                "[jit] phase 2 input: {} ops, {} guards, body_ni={}",
                ops.len(),
                gc_before,
                body_num_inputs
            );
        }

        // opencoder.py:249-406 TraceIterator parity for Phase 2.
        //
        // RPython's `optimize_peeled_loop` calls `trace.get_iter()` which
        // constructs a FRESH TraceIterator whose `__init__` allocates new
        // Box objects for every inputarg (inputarg_from_tp) and whose
        // `next()` allocates new cls() ResOperation instances for every
        // emitted op. Each iteration over the same trace produces a
        // completely disjoint set of Python identities; the cache
        // `_cache[raw_position]` records the per-iteration fresh box so
        // later references resolve to the iteration-local identity.
        //
        // majit's `OpRef(u32)` IS the identity, so "fresh per iteration"
        // means disjoint integer ranges. Phase 2 must not emit op results
        // at OpRefs that collide with Phase 1's emitted positions — doing
        // so reintroduces the box-identity collision that the reactive
        // check at `mod.rs::emit` (collision detection + forwarding
        // redirect) currently compensates for.
        //
        // This step (Commit D2 of the Box identity plan): run Phase 2
        // ops through TraceIterator with `start_index = next_global_opref`,
        // which gives BOTH inputargs AND op results fresh OpRefs in a
        // disjoint range. Phase 2 inputargs live at
        // `[next_global_opref..next_global_opref+body_num_inputs)` and
        // op results at `[next_global_opref+body_num_inputs..)`. This is
        // the RPython-literal model where each TraceIterator produces
        // freshly allocated InputArg and ResOp Python instances for
        // every iteration — `import_state`'s `assert source is not
        // target` (unroll.py:483) now holds by construction because the
        // Phase 2 source slot OpRefs are always distinct from any
        // Phase 1 end_arg OpRef.
        //
        // After optimization, Phase 2 output is post-translated back to
        // the shared `[0..body_num_inputs)` inputarg layout for the
        // final assembly (see the `shift_back` pass below). The
        // assembly and downstream consumers continue to assume the
        // shared-inputarg layout; only the Phase 2 optimizer internals
        // see the disjoint range.
        let phase2_inputarg_base = self.next_global_opref.max(body_num_inputs as u32);
        let mut iter = majit_trace::opencoder::TraceIterator::new(
            &ops,
            0,
            ops.len(),
            None,
            body_num_inputs,
            phase2_inputarg_base, // fresh inputargs at [phase2_inputarg_base..)
        );
        let mut p2_ops_in: Vec<Op> = Vec::with_capacity(ops.len());
        while let Some(op) = iter.next() {
            p2_ops_in.push(op);
        }
        let p2_high_water = iter._fresh;
        let p2_cache = iter._cache;
        // opencoder.py: `_cache[raw_pos]` holds the fresh per-iteration
        // box. Use it to translate Phase 2-side maps that were populated
        // against raw trace OpRefs so they match the fresh layout of
        // `p2_ops_in`. Inputarg positions (raw `[0..body_num_inputs)`)
        // are seeded to identity by `TraceIterator::new`, so any
        // reference to an inputarg OpRef translates to itself — only
        // op-result OpRefs are actually remapped.
        let translate_opref = |opref: OpRef| -> OpRef {
            if opref.is_none() || opref.is_constant() {
                return opref;
            }
            p2_cache
                .get(opref.0 as usize)
                .copied()
                .flatten()
                .unwrap_or(opref)
        };
        for boxes in opt_p2.snapshot_boxes.values_mut() {
            for r in boxes.iter_mut() {
                *r = translate_opref(*r);
            }
        }
        for boxes in opt_p2.snapshot_vable_boxes.values_mut() {
            for r in boxes.iter_mut() {
                *r = translate_opref(*r);
            }
        }
        // KEY translation: rebuild HashMap with translated keys.
        let translated_box_types: HashMap<u32, Type> = opt_p2
            .snapshot_box_types
            .iter()
            .map(|(&k, &v)| (translate_opref(OpRef(k)).0, v))
            .collect();
        opt_p2.snapshot_box_types = translated_box_types;
        let translated_op_types: HashMap<u32, Type> = opt_p2
            .original_trace_op_types
            .iter()
            .map(|(&k, &v)| (translate_opref(OpRef(k)).0, v))
            .collect();
        opt_p2.original_trace_op_types = translated_op_types;
        // Phase 1's emitted op types are already in Phase 1's emitted
        // namespace `[num_inputs..next_global_opref)`. Phase 2 body may
        // reference these via `imported_label_args`. They are NOT in the
        // `p2_cache` (only raw trace positions are), so leave
        // `phase1_value_types` untranslated.
        let mut p2_ops = opt_p2.optimize_with_constants_and_inputs_at(
            &p2_ops_in,
            &mut consts_p2,
            body_num_inputs,
            phase2_inputarg_base, // inputarg_base — Phase 2 inputargs at [phase2_inputarg_base..)
            p2_high_water,
        );
        // Post-translate Phase 2 output back to the shared-inputarg
        // layout expected by `assemble_peeled_trace_with_jump_args`.
        //
        // Phase 2's OptContext internally used disjoint inputarg OpRefs
        // at `[phase2_inputarg_base..phase2_inputarg_base+body_num_inputs)`
        // so the RPython `import_state` "source is not target"
        // (unroll.py:483) invariant held by construction. The final
        // assembled trace, however, uses shared body inputargs at
        // `[0..body_num_inputs)` so that the preamble and body share
        // the same inputarg slots — the `Label(label_args)` op at the
        // body boundary binds them to the preamble's end values, and
        // downstream consumers (Cranelift regalloc, resume data) see
        // a single consistent inputarg numbering.
        //
        // This shift only touches op args, fail_args, and the label /
        // source_slot vectors. Op positions (the emitted results) are
        // already in `[p2_high_water..)`, which is above both the
        // shared range `[0..body_num_inputs)` and the Phase 1 emitted
        // range `[num_inputs..next_global_opref)`, so they stay put.
        if phase2_inputarg_base > 0 {
            let end = phase2_inputarg_base + body_num_inputs as u32;
            let shift_back = |opref: OpRef| -> OpRef {
                if !opref.is_none()
                    && !opref.is_constant()
                    && opref.0 >= phase2_inputarg_base
                    && opref.0 < end
                {
                    OpRef(opref.0 - phase2_inputarg_base)
                } else {
                    opref
                }
            };
            for op in p2_ops.iter_mut() {
                for arg in op.args.iter_mut() {
                    *arg = shift_back(*arg);
                }
                if let Some(ref mut fa) = op.fail_args {
                    for arg in fa.iter_mut() {
                        *arg = shift_back(*arg);
                    }
                }
            }
            if let Some(ref mut label_args) = opt_p2.imported_label_args {
                for arg in label_args.iter_mut() {
                    *arg = shift_back(*arg);
                }
            }
            if let Some(ref mut source_slots) = opt_p2.imported_label_source_slots {
                for arg in source_slots.iter_mut() {
                    *arg = shift_back(*arg);
                }
            }
        }
        // Phase 2 may discover new constants via make_constant (e.g., guard
        // class pointers from collect_use_box_guards).
        // Merge back into consts_p2 so the backend can resolve them.
        if let Some(ref final_ctx) = opt_p2.final_ctx {
            for (idx, val) in final_ctx.constants.iter().enumerate() {
                if let Some(v) = val {
                    let raw = match v {
                        majit_ir::Value::Int(v) => *v,
                        majit_ir::Value::Float(f) => f.to_bits() as i64,
                        majit_ir::Value::Ref(r) => r.0 as i64,
                        majit_ir::Value::Void => 0,
                    };
                    consts_p2.entry(idx as u32).or_insert(raw);
                }
            }
        }
        let body_terminal_op = opt_p2.terminal_op.clone();
        let p2_ni = opt_p2.final_num_inputs();
        // resume.py:570-574: collect per-guard optimizer knowledge from both phases.
        self.per_guard_knowledge
            .extend(opt_p2.per_guard_knowledge.drain(..));
        let label_args = opt_p2
            .imported_label_args
            .clone()
            .expect("phase 2 missing import_state label_args");

        if std::env::var_os("MAJIT_LOG").is_some() {
            for op in &p2_ops {
                if op.opcode.is_guard() {
                    let rd_numb_len = op.rd_numb.as_ref().map(|v| v.len()).unwrap_or(0);
                    if let Some(ref fa) = op.fail_args {
                        let fa_raw: Vec<String> =
                            fa.iter().map(|a| format!("OpRef({})", a.0)).collect();
                        eprintln!(
                            "[jit] p2 guard {:?} pos={:?} resume_pos={} rd_numb={} fail_args_raw=[{}]",
                            op.opcode,
                            op.pos,
                            op.rd_resume_position,
                            rd_numb_len,
                            fa_raw.join(", ")
                        );
                    } else {
                        eprintln!(
                            "[jit] p2 guard {:?} pos={:?} resume_pos={} rd_numb={} fail_args_raw=<none>",
                            op.opcode, op.pos, op.rd_resume_position, rd_numb_len,
                        );
                    }
                }
            }
            let nc = p2_ops
                .iter()
                .filter(|o| o.opcode == OpCode::New || o.opcode == OpCode::NewWithVtable)
                .count();
            let gc = p2_ops.iter().filter(|o| o.opcode.is_guard()).count();
            eprintln!(
                "[jit] phase 2: {} ops, {} New, {} guards, p2_ni={}",
                p2_ops.len(),
                nc,
                gc,
                p2_ni
            );
            for (i, op) in p2_ops.iter().enumerate() {
                eprintln!(
                    "[jit] p2[{i}]: {:?} pos={:?} args={:?}",
                    op.opcode, op.pos, op.args
                );
            }
        }

        // ── unroll.py:140-175: finalize + jump_to_existing_trace ──
        let imported_short_preamble_builder = opt_p2.imported_short_preamble_builder.clone();
        let imported_short_aliases = opt_p2.imported_short_aliases.clone();
        let imported_short_sources = opt_p2.imported_short_sources.clone();

        // finalize_short_preamble: create TargetToken for this loop version
        // RPython parity: short preamble ops reference constant OpRefs from
        // the loop's constant pool. Pass consts_p1 so the ShortPreamble
        // captures (value, type) for each constant, enabling bridges to
        // re-register them in their own pool (RPython embeds Const objects
        // directly in op args; majit uses separate constant pool indices).
        // RPython parity: read back from the same ExportedState that
        // import_state used (Python reference semantics — one object).
        // exported_state was moved into opt_p2.imported_loop_state.
        // Extract needed fields before opt_p2 is borrowed mutably below.
        let (
            exported_vs,
            exported_end_args,
            exported_short_inputargs,
            exported_short_boxes,
            exported_renamed_inputargs,
            exported_renamed_inputarg_types,
        ) = {
            let es = opt_p2
                .imported_loop_state
                .as_ref()
                .expect("imported_loop_state must survive Phase 2");
            (
                es.virtual_state.clone(),
                es.end_args.clone(),
                es.short_inputargs.clone(),
                es.exported_short_boxes.clone(),
                es.renamed_inputargs.clone(),
                es.renamed_inputarg_types.clone(),
            )
        };
        let mut initial_sp = opt_p2.imported_short_preamble.clone().unwrap_or_else(|| {
            crate::optimizeopt::shortpreamble::build_short_preamble_from_exported_boxes(
                &exported_end_args,
                &exported_short_inputargs,
                &exported_short_boxes,
                &consts_p1,
                &self.constant_types,
            )
        });
        // shortpreamble.py:414-425 parity: store PtrInfo for each inputarg.
        // In RPython, preamble_op.set_forwarded(info) attaches PtrInfo to
        // the short_inputarg Box. In majit, we extract from Phase 2 final_ctx
        // so inline_short_preamble can propagate to jump_args.
        if let Some(ref final_ctx) = opt_p2.final_ctx {
            let mut infos = Vec::with_capacity(initial_sp.inputargs.len());
            for &inputarg in &initial_sp.inputargs {
                infos.push(final_ctx.get_ptr_info(inputarg).cloned());
            }
            initial_sp.inputarg_infos = infos;
        }
        let opt_unroll = OptUnroll::new();
        let target_token = opt_unroll.finalize_short_preamble(
            self.target_tokens.len() as u64,
            exported_vs,
            initial_sp.clone(),
            imported_short_preamble_builder.as_ref(),
        );
        self.target_tokens.push(target_token);

        // unroll.py:176-177: disable_retracing_if_max_retrace_guards
        if Self::disable_retracing_if_max_retrace_guards(&p2_ops, self.max_retrace_guards) {
            self.retraced_count = u32::MAX;
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[jit] too many guards (>{}), disabling retracing",
                    self.max_retrace_guards
                );
            }
        }

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] finalize_short_preamble: target_tokens={}",
                self.target_tokens.len()
            );
        }

        // ── unroll.py:207-230: jump_to_existing_trace / retrace_limit ──
        // Try to match the body's JUMP virtual state to an existing target.
        // RPython: new_virtual_state = jump_to_existing_trace(end_jump, ...)
        //
        // RPython parity: never skip jump_to_existing_trace based on
        // guard count. RPython's unroll.py always attempts
        // jump_to_existing_trace regardless of body size.
        let p2_guard_count = p2_ops.iter().filter(|o| o.opcode.is_guard()).count();
        let skip_jump_to_existing = false;
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] post-finalize: entering jump_to_existing_trace section (p2_guards={}, skip={})",
                p2_guard_count, skip_jump_to_existing
            );
        }
        let mut body_ops = p2_ops;
        let mut redirected_tail_ops = Vec::new();
        let jump_to_self = {
            let body_jump_args: Vec<OpRef> = body_terminal_op
                .as_ref()
                .map(|jump| jump.args.to_vec())
                .or_else(|| {
                    body_ops
                        .iter()
                        .rfind(|o| o.opcode == OpCode::Jump)
                        .map(|j| j.args.to_vec())
                })
                .unwrap_or_default();
            let mut current_label_args = label_args.clone();
            // RPython Box parity: each used_box is a distinct Box even
            // when two virtuals share the same OpRef. Allocate fresh
            // OpRefs for duplicates so the LABEL carries independent slots.
            {
                let mut seen_used = std::collections::HashSet::new();
                let mut next_fresh = current_label_args
                    .iter()
                    .chain(initial_sp.used_boxes.iter())
                    .map(|a| a.0)
                    .max()
                    .unwrap_or(0)
                    .saturating_add(1)
                    .max(body_num_inputs as u32 + 100);
                for &ub in &initial_sp.used_boxes {
                    if seen_used.insert(ub) {
                        current_label_args.push(ub);
                    } else {
                        current_label_args.push(OpRef(next_fresh));
                        next_fresh += 1;
                    }
                }
            }
            let opt_unroll = OptUnroll::new();
            // Use Phase 2's final context for virtual state matching.
            let mut jump_ctx = opt_p2.final_ctx.take().unwrap_or_else(|| {
                crate::optimizeopt::OptContext::with_num_inputs(32, body_num_inputs)
            });
            jump_ctx.clear_newoperations();

            // unroll.py:151-158: jump_to_existing_trace(force_boxes=False)
            // RPython: except InvalidLoop → jump_to_preamble immediately,
            // NO retry. The big comment at unroll.py:305-316 explains why
            // continuing after partial inlining is unsafe.
            let runtime_boxes = body_jump_args.clone();
            let mut invalid_loop = false;
            let mut jumped = if skip_jump_to_existing {
                false
            } else {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    opt_unroll
                        .jump_to_existing_trace(
                            &body_jump_args,
                            Some(&current_label_args),
                            &mut self.target_tokens,
                            &mut opt_p2,
                            &mut jump_ctx,
                            false,
                            Some(&runtime_boxes),
                        )
                        .is_none()
                })) {
                    Ok(result) => result,
                    Err(payload) => {
                        if payload
                            .downcast_ref::<crate::optimize::InvalidLoop>()
                            .is_some()
                        {
                            // unroll.py:154-158: except InvalidLoop →
                            // jump_to_preamble, skip retry
                            invalid_loop = true;
                            false
                        } else {
                            std::panic::resume_unwind(payload);
                        }
                    }
                }
            };
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[jit] jump_to_existing_trace(force_boxes=false) result: jumped={}, invalid_loop={}",
                    jumped, invalid_loop
                );
            }

            // unroll.py:154-158: on InvalidLoop, skip retry entirely
            if !jumped && !skip_jump_to_existing && !invalid_loop {
                // unroll.py:161-174: virtual state not matched, retry
                if self.retraced_count < self.retrace_limit {
                    self.retraced_count += 1;
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!(
                            "[jit] Retracing ({}/{})",
                            self.retraced_count, self.retrace_limit
                        );
                    }
                    // unroll.py:164-168: force_boxes=True, except InvalidLoop: pass
                    jump_ctx.clear_newoperations();
                    jumped = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        opt_unroll
                            .jump_to_existing_trace(
                                &body_jump_args,
                                Some(&current_label_args),
                                &mut self.target_tokens,
                                &mut opt_p2,
                                &mut jump_ctx,
                                true,
                                Some(&runtime_boxes),
                            )
                            .is_none()
                    })) {
                        Ok(result) => result,
                        Err(payload) => {
                            if payload
                                .downcast_ref::<crate::optimize::InvalidLoop>()
                                .is_some()
                            {
                                false // unroll.py:167-168: except InvalidLoop: pass
                            } else {
                                std::panic::resume_unwind(payload);
                            }
                        }
                    };
                } else {
                    // unroll.py:220-226: limit reached, try force_boxes=true
                    jump_ctx.clear_newoperations();
                    jumped = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        opt_unroll
                            .jump_to_existing_trace(
                                &body_jump_args,
                                Some(&current_label_args),
                                &mut self.target_tokens,
                                &mut opt_p2,
                                &mut jump_ctx,
                                true,
                                Some(&runtime_boxes),
                            )
                            .is_none()
                    })) {
                        Ok(result) => result,
                        Err(payload) => {
                            if payload
                                .downcast_ref::<crate::optimize::InvalidLoop>()
                                .is_some()
                            {
                                false // unroll.py:224-225: except InvalidLoop: pass
                            } else {
                                std::panic::resume_unwind(payload);
                            }
                        }
                    };
                    if !jumped {
                        // unroll.py:228: "Retrace count reached, jumping to preamble"
                        if std::env::var_os("MAJIT_LOG").is_some() {
                            eprintln!("[jit] Retrace count reached, jumping to preamble");
                        }
                        // jumped stays false → jump_to_preamble below
                    }
                }
            }
            if jumped && redirected_tail_ops.is_empty() {
                // Only take jump_ctx ops if we don't already have
                // a self-loop Jump from the retrace path.
                redirected_tail_ops = jump_ctx.new_operations;
                // Check if the redirected Jump targets the current body token
                // (last in target_tokens) or an external token from a previous
                // compilation.  The Cranelift backend compiles each trace as a
                // single function — cross-function jumps to external target
                // tokens are not supported.  Discard the redirected tail and
                // restore the body's original self-loop Jump instead.
                let current_body_descr_idx = self
                    .target_tokens
                    .last()
                    .map(|t| t.as_jump_target_descr().index());
                let redirected_jump_descr_idx = redirected_tail_ops
                    .iter()
                    .rfind(|o| o.opcode == OpCode::Jump)
                    .and_then(|o| o.descr.as_ref())
                    .map(|d| d.index());
                if redirected_jump_descr_idx != current_body_descr_idx {
                    // RPython parity: the Cranelift backend can't jump to
                    // code from a previous compilation (separate function).
                    // Fall back to jump_to_preamble, matching RPython's
                    // behavior when the target isn't reachable (unroll.py:228).
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!(
                            "[jit] jump_to_existing_trace: external target {:?} != body {:?}, falling back to preamble",
                            redirected_jump_descr_idx, current_body_descr_idx
                        );
                    }
                    redirected_tail_ops.clear();
                    jumped = false;
                }
            }
            jumped
        };

        let sp = self
            .target_tokens
            .last()
            .and_then(|target| target.short_preamble.clone())
            .unwrap_or(initial_sp);
        if !sp.is_empty() {
            self.short_preamble = Some(sp.clone());
        }
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] assembly_contract: label_args={:?} used_boxes={:?} jump_args={:?}",
                label_args, sp.used_boxes, sp.jump_args
            );
        }

        if !jump_to_self {
            // unroll.py:170-171: jump_to_preamble — body JUMP → preamble Label
            //
            // RPython parity: force_box_for_end_of_preamble (unroll.py:126-127)
            // re-boxes unboxed values before the JUMP so types match the
            // preamble inputargs. Without force_box, the body JUMP may pass
            // Float/Int values at Ref-typed positions, causing the preamble's
            // guard checks to dereference non-pointer values → segfault.
            //
            // Until force_box_for_end_of_preamble is implemented, reject
            // traces where the body JUMP types don't match preamble inputarg
            // types. The metainterp falls back to interpretation.
            let preamble_target = self
                .target_tokens
                .first()
                .expect("preamble target token must exist before jump_to_preamble")
                .clone();
            let preamble_arity = exported_renamed_inputargs.len();
            if std::env::var_os("MAJIT_LOG").is_some() {
                let body_jump_arity = body_terminal_op.as_ref().map(|j| j.args.len()).unwrap_or(0);
                eprintln!(
                    "[jit] jump_to_preamble: body_jump_args={} preamble_arity={} start_label_args={:?}",
                    body_jump_arity, preamble_arity, exported_renamed_inputargs,
                );
            }
            if let Some(mut end_jump) = body_terminal_op {
                end_jump.descr = Some(preamble_target.as_jump_target_descr());
                // Truncate Jump args to match preamble start Label arity.
                // Body Jump may have N+M args (short preamble), but preamble
                // Label only has N args. RPython backend ignores extras;
                // Cranelift requires exact match.
                if end_jump.args.len() > preamble_arity {
                    end_jump.args.truncate(preamble_arity);
                }
                body_ops = replace_terminal_jump(&body_ops, end_jump);
            } else {
                body_ops = Self::jump_to_preamble(&body_ops, &preamble_target);
            }
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!("[jit] jump_to_preamble: body JUMP retargeted to start descr");
            }
        } else if !redirected_tail_ops.is_empty() {
            body_ops = splice_redirected_tail(&body_ops, &redirected_tail_ops);
        } else if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("[jit] jump_to_existing_trace: body JUMP → self-loop");
        }

        // ── Assembly (compile.py:310-338) ──
        let mut combined = assemble_peeled_trace_with_jump_args(
            &p1_ops,
            &body_ops,
            &label_args,
            opt_p2.imported_label_source_slots.as_deref().unwrap_or(&[]),
            &exported_renamed_inputargs,
            &sp.used_boxes,
            &sp.jump_args,
            p2_ni,
            jump_to_self,
            &imported_short_aliases,
            &imported_short_sources,
            &consts_p2,
            self.target_tokens
                .first()
                .map(|target| target.as_jump_target_descr()),
            self.target_tokens
                .last()
                .map(|target| target.as_jump_target_descr()),
            &exported_end_args,
            &exported_renamed_inputarg_types,
        );
        // RPython Box parity: drop duplicate-position ops. In RPython
        // each Box is unique so collisions can't happen. Keep first.
        {
            let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
            combined.retain(|op| {
                if op.pos.is_none() || op.result_type() == Type::Void {
                    return true;
                }
                seen.insert(op.pos.0)
            });
        }
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("--- peeled trace (assembled) ---");
            eprint!("{}", majit_ir::format_trace(&combined, &consts_p2));
            let mut sorted_consts: Vec<_> =
                consts_p2.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
            sorted_consts.sort_by_key(|(k, _)| *k);
            eprintln!("[jit] consts_p2: {:?}", sorted_consts);
        }
        *constants = consts_p2;
        // Merge Phase 2 constant types back so build_guard_metadata
        // can resolve Phase 2 allocated constants for rd_virtuals.
        for (k, v) in &opt_p2.constant_types {
            self.constant_types.entry(*k).or_insert(*v);
        }
        (combined, p2_ni)
    }

    /// RPython compile.py uses the optimized loop state's inputargs contract,
    /// not a stale input counter. When majit falls back to the phase-1 trace,
    /// derive the live loop arity from the actual closing Label/Jump.
    pub fn closing_loop_contract_arity(ops: &[Op], fallback: usize) -> usize {
        closing_loop_contract_arity(ops, fallback)
    }

    /// Count the guards in an optimized trace (for retrace_limit checks).
    pub fn count_guards(ops: &[Op]) -> u32 {
        ops.iter().filter(|op| op.opcode.is_guard()).count() as u32
    }

    /// unroll.py: _map_args(mapping, arglist)
    /// Remap a list of OpRefs through a forwarding mapping.
    /// Constant OpRefs are left unchanged because they are not remapped.
    pub fn map_args(
        mapping: &std::collections::HashMap<OpRef, OpRef>,
        args: &[OpRef],
    ) -> Vec<OpRef> {
        args.iter()
            .map(|&arg| mapping.get(&arg).copied().unwrap_or(arg))
            .collect()
    }

    /// unroll.py: _check_no_forwarding(lsts)
    /// Debug assertion: verify no OpRef in the lists has been forwarded.
    pub fn check_no_forwarding(ctx: &crate::optimizeopt::OptContext, oprefs: &[OpRef]) -> bool {
        oprefs.iter().all(|&r| ctx.get_box_replacement(r) == r)
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
        ctx: &crate::optimizeopt::OptContext,
        forwarded: &[crate::optimizeopt::info::Forwarded],
    ) -> crate::optimizeopt::virtualstate::VirtualState {
        crate::optimizeopt::virtualstate::export_state(args, ctx, forwarded)
    }
}

fn closing_loop_contract_arity(ops: &[Op], fallback: usize) -> usize {
    ops.iter()
        .rev()
        .find_map(|op| match op.opcode {
            OpCode::Label | OpCode::Jump => Some(op.args.len()),
            _ => None,
        })
        .unwrap_or(fallback)
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
///
/// gcreftracer.py parity: GcRef values in exported_infos are rooted on
/// the shadow stack. ExportedState persists between Phase 1 (preamble)
/// and Phase 2 (body), during which GC can run and move objects.
#[derive(Debug)]
pub struct ExportedState {
    /// Label args at the end of the preamble (after forcing).
    pub end_args: Vec<OpRef>,
    /// Args for the next iteration (before forcing).
    pub next_iteration_args: Vec<OpRef>,
    /// Types of end_args as determined by Phase 1 optimization.
    /// Used by Phase 2 import_state to propagate unboxed types.
    pub end_arg_types: Vec<Type>,
    /// Phase 1 heap cache: (obj, descr_idx, cached_value) entries
    /// for ALL fields cached during preamble optimization.
    /// Phase 2 import_state uses this to pre-populate its heap cache
    /// for inputargs that the preamble already read fields from.
    pub preamble_heap_cache: Vec<(OpRef, u32, OpRef)>,
    /// Virtual state at the loop boundary.
    pub virtual_state: crate::optimizeopt::virtualstate::VirtualState,
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
    pub exported_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
    /// Short preamble builder for bridge entry.
    pub short_preamble: Option<crate::optimizeopt::shortpreamble::ShortPreamble>,
    /// Renamed inputargs from the preamble.
    pub renamed_inputargs: Vec<OpRef>,
    /// Types of renamed_inputargs. RPython Box objects carry type
    /// intrinsically; majit stores it separately.
    pub renamed_inputarg_types: Vec<Type>,
    /// Short inputargs for the short preamble.
    pub short_inputargs: Vec<OpRef>,
    /// RPython parity: patchguardop from Phase 1's GuardFutureCondition.
    /// Phase 2's extra_guards (from virtualstate) need rd_resume_position
    /// from this patchguardop (unroll.py:333-336).
    pub patchguardop: Option<majit_ir::Op>,
    /// Shadow stack rooting for GcRef values in exported_infos.
    /// (OpRef key, field kind, shadow stack index).
    rooted_refs: Vec<(OpRef, ExportedGcRefField, usize)>,
    /// Shadow stack depth at creation. release_roots pops to here.
    shadow_stack_base: usize,
}

/// majit's serialized form of RPython's per-Box `_forwarded` /
/// `PtrInfo`. RPython's `_expand_info` (unroll.py:432-450) builds a
/// `dict[Box -> info.PtrInfo]` directly because Python keeps the
/// PtrInfo alive across phases via Box identity. majit must serialize
/// because Phase 1's `OptContext` is dropped before Phase 2 starts.
///
/// Holds:
/// - `constant` — the side `Value` RPython carries on the box itself
/// - `ptr_info` — a clone of the live `PtrInfo` from Phase 1
/// - `int_bound` — full IntBound captured at export time, imported with
///   widen() in setinfo_from_preamble (unroll.py:93-96)
#[derive(Clone, Debug, Default)]
pub struct ExportedValueInfo {
    pub constant: Option<Value>,
    pub ptr_info: Option<crate::optimizeopt::info::PtrInfo>,
    pub int_bound: Option<crate::optimizeopt::intutils::IntBound>,
}

/// Identifies which GcRef-bearing field inside an ExportedState
/// is rooted at a particular shadow stack slot.
#[derive(Clone, Copy, Debug)]
enum ExportedGcRefField {
    /// exported_infos[OpRef].constant (Value::Ref)
    InfoConstantRef,
    /// exported_infos[OpRef].ptr_info = PtrInfo::Constant(GcRef)
    InfoPtrInfoConstant,
    /// exported_infos[OpRef].ptr_info = PtrInfo::Instance with known_class set
    /// (PyPy `InstancePtrInfo._known_class`; majit folds the former
    /// `KnownClass` variant into `Instance(descr=None, known_class=Some(_))`).
    InfoPtrInfoKnownClass,
    /// virtual_state.state[index] = KnownClass { class_ptr }
    VirtualStateKnownClass(usize),
    /// virtual_state.state[index] = Virtual { known_class }
    VirtualStateVirtualClass(usize),
    /// virtual_state.state[index] = Constant(Value::Ref)
    VirtualStateConstantRef(usize),
}

#[derive(Clone, Debug)]
pub enum ExportedShortOp {
    Pure {
        source: OpRef,
        opcode: OpCode,
        descr: Option<DescrRef>,
        args: Vec<ExportedShortArg>,
        result: ExportedShortResult,
        invented_name: bool,
        same_as_source: Option<OpRef>,
    },
    HeapField {
        source: OpRef,
        /// RPython shortpreamble.py: preamble_op.getarg(0).
        /// Slot(i) for label_arg, Const for promoted struct.
        object: ExportedShortArg,
        descr: DescrRef,
        result_type: Type,
        result: ExportedShortResult,
        invented_name: bool,
        same_as_source: Option<OpRef>,
    },
    HeapArrayItem {
        source: OpRef,
        /// Same as HeapField.object.
        object: ExportedShortArg,
        descr: DescrRef,
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
    Const { source: OpRef, value: Value },
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
        virtual_state: crate::optimizeopt::virtualstate::VirtualState,
        exported_infos: HashMap<OpRef, ExportedValueInfo>,
        exported_short_ops: Vec<ExportedShortOp>,
        exported_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
        renamed_inputargs: Vec<OpRef>,
        renamed_inputarg_types: Vec<Type>,
        short_inputargs: Vec<OpRef>,
    ) -> Self {
        ExportedState {
            end_args,
            next_iteration_args,
            end_arg_types: Vec::new(),
            preamble_heap_cache: Vec::new(),
            virtual_state,
            exported_infos,
            exported_short_ops,
            exported_short_boxes,
            short_preamble: None,
            renamed_inputargs,
            renamed_inputarg_types,
            short_inputargs,
            patchguardop: None,
            rooted_refs: Vec::new(),
            shadow_stack_base: majit_gc::shadow_stack::depth(),
        }
        // gcreftracer.py parity: RPython ExportedState is a Python object
        // whose GcRef fields are automatically traced by the GC. In Rust,
        // root_all_gcrefs() must be called at each storage site in LIFO
        // order (longer-lived copy rooted first → lower shadow stack depth).
        // new() does NOT auto-root because the LIFO ordering depends on
        // the caller's storage pattern.
    }

    /// Push all GcRef values from exported_infos and virtual_state to
    /// shadow stack. gcreftracer.py parity: GC can run between Phase 1
    /// and Phase 2, and between Phase 1 and retrace.
    ///
    /// Must be called explicitly after construction — not auto-called in
    /// new(). This enables LIFO-correct rooting: root the longer-lived
    /// copy first (lower shadow stack depth), then the shorter-lived copy.
    pub fn root_all_gcrefs(&mut self) {
        use crate::optimizeopt::info::PtrInfo;
        use crate::optimizeopt::virtualstate::VirtualStateInfo;
        self.shadow_stack_base = majit_gc::shadow_stack::depth();
        // ── exported_infos GcRef fields ──
        let mut keys: Vec<OpRef> = self.exported_infos.keys().copied().collect();
        keys.sort_by_key(|k| k.0);
        for key in keys {
            let info = &self.exported_infos[&key];
            if let Some(Value::Ref(gcref)) = info.constant {
                if !gcref.is_null() {
                    let ss_idx = majit_gc::shadow_stack::push(gcref);
                    self.rooted_refs
                        .push((key, ExportedGcRefField::InfoConstantRef, ss_idx));
                }
            }
            if let Some(ref pi) = info.ptr_info {
                match pi {
                    PtrInfo::Constant(gcref) if !gcref.is_null() => {
                        let ss_idx = majit_gc::shadow_stack::push(*gcref);
                        self.rooted_refs.push((
                            key,
                            ExportedGcRefField::InfoPtrInfoConstant,
                            ss_idx,
                        ));
                    }
                    PtrInfo::Instance(iinfo) => {
                        if let Some(gcref) = iinfo.known_class {
                            if !gcref.is_null() {
                                let ss_idx = majit_gc::shadow_stack::push(gcref);
                                self.rooted_refs.push((
                                    key,
                                    ExportedGcRefField::InfoPtrInfoKnownClass,
                                    ss_idx,
                                ));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        // ── virtual_state GcRef fields ──
        // VirtualStateInfo::KnownClass, Virtual{known_class}, Constant(Ref)
        let dummy_key = OpRef(u32::MAX);
        for (i, entry) in self.virtual_state.state.iter().enumerate() {
            match &**entry {
                VirtualStateInfo::KnownClass { class_ptr } if !class_ptr.is_null() => {
                    let ss_idx = majit_gc::shadow_stack::push(*class_ptr);
                    self.rooted_refs.push((
                        dummy_key,
                        ExportedGcRefField::VirtualStateKnownClass(i),
                        ss_idx,
                    ));
                }
                VirtualStateInfo::Virtual {
                    known_class: Some(gcref),
                    ..
                } if !gcref.is_null() => {
                    let ss_idx = majit_gc::shadow_stack::push(*gcref);
                    self.rooted_refs.push((
                        dummy_key,
                        ExportedGcRefField::VirtualStateVirtualClass(i),
                        ss_idx,
                    ));
                }
                VirtualStateInfo::Constant(Value::Ref(gcref)) if !gcref.is_null() => {
                    let ss_idx = majit_gc::shadow_stack::push(*gcref);
                    self.rooted_refs.push((
                        dummy_key,
                        ExportedGcRefField::VirtualStateConstantRef(i),
                        ss_idx,
                    ));
                }
                _ => {}
            }
        }
    }

    /// Update GcRef values from shadow stack — GC may have moved objects.
    ///
    /// VirtualStateInfo top-level entries are stored as
    /// `Rc<VirtualStateInfo>` (so two aliased jump args can share a single
    /// `Rc`); GcRef updates have to replace the entire `Rc` with a fresh
    /// one because the inner enum is immutable.
    ///
    /// **Aliasing preservation**: RPython virtualstate.py:712-728
    /// `VirtualStateConstructor.create_state` caches by Python object
    /// identity, so a GC pause never breaks the "two aliased jump args
    /// share one VirtualStateInfo" invariant. Rust's `Rc<...>` is immutable
    /// after construction, so each per-slot GcRef refresh would otherwise
    /// allocate an independent new `Rc` for every shared slot, breaking
    /// the `Rc::as_ptr` dedup that `build_sequential_slot_schedule` relies
    /// on. Snapshot the original `Rc::as_ptr` per slot, group slots that
    /// originally shared an `Rc`, and after the GcRef updates re-clone a
    /// single canonical `Rc` into every slot of each group so the
    /// post-refresh tree preserves the pre-GC aliasing.
    pub fn refresh_from_gc(&mut self) {
        use crate::optimizeopt::info::PtrInfo;
        use crate::optimizeopt::virtualstate::VirtualStateInfo;
        use std::rc::Rc;
        // virtualstate.py:712 cache parity: snapshot original Rc identities
        // so we can re-share post-update slots that aliased pre-GC.
        let original_ptrs: Vec<usize> = self
            .virtual_state
            .state
            .iter()
            .map(|rc| Rc::as_ptr(rc) as usize)
            .collect();
        let mut virtual_state_dirty = false;
        for &(key, ref field, ss_idx) in &self.rooted_refs {
            let updated = majit_gc::shadow_stack::get(ss_idx);
            match field {
                ExportedGcRefField::InfoConstantRef => {
                    if let Some(info) = self.exported_infos.get_mut(&key) {
                        info.constant = Some(Value::Ref(updated));
                    }
                }
                ExportedGcRefField::InfoPtrInfoConstant => {
                    if let Some(info) = self.exported_infos.get_mut(&key) {
                        info.ptr_info = Some(PtrInfo::Constant(updated));
                    }
                }
                ExportedGcRefField::InfoPtrInfoKnownClass => {
                    if let Some(info) = self.exported_infos.get_mut(&key) {
                        if let Some(PtrInfo::Instance(iinfo)) = &mut info.ptr_info {
                            iinfo.known_class = Some(updated);
                        }
                    }
                }
                ExportedGcRefField::VirtualStateKnownClass(i) => {
                    if let Some(slot) = self.virtual_state.state.get_mut(*i)
                        && matches!(&**slot, VirtualStateInfo::KnownClass { .. })
                    {
                        *slot = Rc::new(VirtualStateInfo::KnownClass { class_ptr: updated });
                        virtual_state_dirty = true;
                    }
                }
                ExportedGcRefField::VirtualStateVirtualClass(i) => {
                    if let Some(slot) = self.virtual_state.state.get_mut(*i) {
                        if let VirtualStateInfo::Virtual {
                            descr,
                            ob_type_descr,
                            fields,
                            field_descrs,
                            ..
                        } = &**slot
                        {
                            *slot = Rc::new(VirtualStateInfo::Virtual {
                                descr: descr.clone(),
                                known_class: Some(updated),
                                ob_type_descr: ob_type_descr.clone(),
                                fields: fields.clone(),
                                field_descrs: field_descrs.clone(),
                            });
                            virtual_state_dirty = true;
                        }
                    }
                }
                ExportedGcRefField::VirtualStateConstantRef(i) => {
                    if let Some(entry) = self.virtual_state.state.get_mut(*i) {
                        *entry = Rc::new(VirtualStateInfo::Constant(Value::Ref(updated)));
                        virtual_state_dirty = true;
                    }
                }
            }
        }
        if virtual_state_dirty {
            // Re-share slots that originally aliased: walk the snapshot
            // map and copy each group's first canonical Rc into every
            // peer slot, restoring the pre-GC `Rc::as_ptr` equivalences.
            let mut canonical_by_old: std::collections::HashMap<usize, Rc<VirtualStateInfo>> =
                std::collections::HashMap::new();
            for (slot_idx, &old_ptr) in original_ptrs.iter().enumerate() {
                let entry = canonical_by_old
                    .entry(old_ptr)
                    .or_insert_with(|| Rc::clone(&self.virtual_state.state[slot_idx]));
                if !Rc::ptr_eq(entry, &self.virtual_state.state[slot_idx]) {
                    self.virtual_state.state[slot_idx] = Rc::clone(entry);
                }
            }
            // Rc identities have shifted, so the dedup walkers'
            // numnotvirtuals / slot_schedule need to be recomputed.
            self.virtual_state.rebuild_slot_schedule();
        }
    }

    /// Release shadow stack roots.
    fn release_roots(&mut self) {
        if !self.rooted_refs.is_empty() {
            majit_gc::shadow_stack::pop_to(self.shadow_stack_base);
            self.rooted_refs.clear();
        }
    }

    /// unroll.py: final() — ExportedState is never final (loop continues).
    pub fn is_final(&self) -> bool {
        false
    }
}

impl Clone for ExportedState {
    /// Pure data clone — no shadow stack side effects.
    ///
    /// RPython has no clone (single Python object shared by reference).
    /// When a Rust clone is stored long-term (across potential GC points),
    /// the caller must call root_all_gcrefs() explicitly.
    fn clone(&self) -> Self {
        ExportedState {
            end_args: self.end_args.clone(),
            next_iteration_args: self.next_iteration_args.clone(),
            end_arg_types: self.end_arg_types.clone(),
            preamble_heap_cache: self.preamble_heap_cache.clone(),
            virtual_state: self.virtual_state.clone(),
            exported_infos: self.exported_infos.clone(),
            exported_short_ops: self.exported_short_ops.clone(),
            exported_short_boxes: self.exported_short_boxes.clone(),
            short_preamble: self.short_preamble.clone(),
            renamed_inputargs: self.renamed_inputargs.clone(),
            renamed_inputarg_types: self.renamed_inputarg_types.clone(),
            short_inputargs: self.short_inputargs.clone(),
            patchguardop: self.patchguardop.clone(),
            rooted_refs: Vec::new(),
            shadow_stack_base: majit_gc::shadow_stack::depth(),
        }
    }
}

impl Drop for ExportedState {
    fn drop(&mut self) {
        self.release_roots();
    }
}

impl PartialEq for ExportedShortOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                ExportedShortOp::Pure {
                    source: s1,
                    opcode: o1,
                    descr: d1,
                    args: a1,
                    result: r1,
                    invented_name: i1,
                    same_as_source: sa1,
                },
                ExportedShortOp::Pure {
                    source: s2,
                    opcode: o2,
                    descr: d2,
                    args: a2,
                    result: r2,
                    invented_name: i2,
                    same_as_source: sa2,
                },
            ) => {
                s1 == s2
                    && o1 == o2
                    && d1.as_ref().map(|d| d.index()) == d2.as_ref().map(|d| d.index())
                    && a1 == a2
                    && r1 == r2
                    && i1 == i2
                    && sa1 == sa2
            }
            (
                ExportedShortOp::HeapField {
                    source: s1,
                    object: o1,
                    descr: d1,
                    result_type: t1,
                    result: r1,
                    invented_name: i1,
                    same_as_source: sa1,
                },
                ExportedShortOp::HeapField {
                    source: s2,
                    object: o2,
                    descr: d2,
                    result_type: t2,
                    result: r2,
                    invented_name: i2,
                    same_as_source: sa2,
                },
            ) => {
                s1 == s2
                    && o1 == o2
                    && d1.index() == d2.index()
                    && t1 == t2
                    && r1 == r2
                    && i1 == i2
                    && sa1 == sa2
            }
            (
                ExportedShortOp::HeapArrayItem {
                    source: s1,
                    object: o1,
                    descr: d1,
                    index: x1,
                    result_type: t1,
                    result: r1,
                    invented_name: i1,
                    same_as_source: sa1,
                },
                ExportedShortOp::HeapArrayItem {
                    source: s2,
                    object: o2,
                    descr: d2,
                    index: x2,
                    result_type: t2,
                    result: r2,
                    invented_name: i2,
                    same_as_source: sa2,
                },
            ) => {
                s1 == s2
                    && o1 == o2
                    && d1.index() == d2.index()
                    && x1 == x2
                    && t1 == t2
                    && r1 == r2
                    && i1 == i2
                    && sa1 == sa2
            }
            (
                ExportedShortOp::LoopInvariant {
                    source: s1,
                    func_ptr: f1,
                    result_type: t1,
                    result: r1,
                    invented_name: i1,
                    same_as_source: sa1,
                },
                ExportedShortOp::LoopInvariant {
                    source: s2,
                    func_ptr: f2,
                    result_type: t2,
                    result: r2,
                    invented_name: i2,
                    same_as_source: sa2,
                },
            ) => s1 == s2 && f1 == f2 && t1 == t2 && r1 == r2 && i1 == i2 && sa1 == sa2,
            _ => false,
        }
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
    /// JitCellToken.target_tokens list. Debug-display only — backend identity
    /// is the `jump_target_descr` Arc address.
    pub token_id: u64,
    /// compile.py: start_descr — the preamble target token has no virtual
    /// state and lives at target_tokens[0].
    pub is_preamble_target: bool,
    /// Virtual state at this loop entry point.
    /// Used by _jump_to_existing_trace to check compatibility.
    pub virtual_state: Option<crate::optimizeopt::virtualstate::VirtualState>,
    /// Short preamble: ops to replay when entering from a bridge.
    pub short_preamble: Option<crate::optimizeopt::shortpreamble::ShortPreamble>,
    /// RPython unroll.py: active ExtendedShortPreambleBuilder for the target
    /// token currently being finalized.
    pub short_preamble_producer:
        Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder>,
    jump_target_descr: Arc<LoopTargetDescr>,
}

impl TargetToken {
    pub fn new() -> Self {
        TargetToken {
            token_id: 0,
            is_preamble_target: false,
            virtual_state: None,
            short_preamble: None,
            short_preamble_producer: None,
            jump_target_descr: Arc::new(LoopTargetDescr::new(0, false)),
        }
    }

    pub fn new_loop(token_id: u64) -> Self {
        let mut token = Self::new();
        token.token_id = token_id;
        token.jump_target_descr = Arc::new(LoopTargetDescr::new(token_id, false));
        token
    }

    pub fn new_preamble(token_id: u64) -> Self {
        let mut token = Self::new();
        token.token_id = token_id;
        token.is_preamble_target = true;
        token.jump_target_descr = Arc::new(LoopTargetDescr::new(token_id, true));
        token
    }

    pub fn as_jump_target_descr(&self) -> majit_ir::DescrRef {
        self.jump_target_descr.clone()
    }
}

#[derive(Debug, Default)]
struct LoopTargetDescrState {
    ll_loop_code: usize,
    target_arglocs: Vec<majit_ir::TargetArgLoc>,
}

#[derive(Debug)]
struct LoopTargetDescr {
    token_id: u64,
    is_preamble_target: bool,
    state: Mutex<LoopTargetDescrState>,
}

impl LoopTargetDescr {
    fn new(token_id: u64, is_preamble_target: bool) -> Self {
        Self {
            token_id,
            is_preamble_target,
            state: Mutex::new(LoopTargetDescrState::default()),
        }
    }
}

impl majit_ir::Descr for LoopTargetDescr {
    fn index(&self) -> u32 {
        self.token_id as u32
    }

    fn repr(&self) -> String {
        if self.is_preamble_target {
            format!("LoopTargetDescr(start:{})", self.token_id)
        } else {
            format!("LoopTargetDescr({})", self.token_id)
        }
    }

    fn as_loop_target_descr(&self) -> Option<&dyn majit_ir::LoopTargetDescr> {
        Some(self)
    }
}

impl majit_ir::LoopTargetDescr for LoopTargetDescr {
    fn token_id(&self) -> u64 {
        self.token_id
    }

    fn is_preamble_target(&self) -> bool {
        self.is_preamble_target
    }

    fn ll_loop_code(&self) -> usize {
        self.state.lock().unwrap().ll_loop_code
    }

    fn set_ll_loop_code(&self, loop_code: usize) {
        self.state.lock().unwrap().ll_loop_code = loop_code;
    }

    fn target_arglocs(&self) -> Vec<majit_ir::TargetArgLoc> {
        self.state.lock().unwrap().target_arglocs.clone()
    }

    fn set_target_arglocs(&self, arglocs: Vec<majit_ir::TargetArgLoc>) {
        self.state.lock().unwrap().target_arglocs = arglocs;
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
    pub quasi_immutable_deps: std::collections::HashSet<(u64, u32)>,
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
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
    ) -> ExportedState {
        self.export_state_with_bounds(original_label_args, renamed_inputargs, optimizer, ctx, None)
    }

    /// unroll.py:452-477: export_state implementation.
    fn export_state_with_bounds(
        &self,
        original_label_args: &[OpRef],
        renamed_inputargs: &[OpRef],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        exported_int_bounds: Option<&HashMap<OpRef, crate::optimizeopt::intutils::IntBound>>,
    ) -> ExportedState {
        // unroll.py:454: end_args = [force_at_the_end_of_preamble(a) ...]
        let end_args: Vec<OpRef> = ctx.preamble_end_args.clone().unwrap_or_else(|| {
            original_label_args
                .iter()
                .map(|&a| ctx.get_box_replacement(a))
                .collect()
        });
        // unroll.py:457 `virtual_state = self.get_virtual_state(end_args)`
        // — VS captured AFTER `force_box_for_end_of_preamble` and AFTER
        // `flush()`. The caller (`Optimizer::optimize_with_constants_and_inputs_at`)
        // already ran both passes before invoking us, so `end_args` is in
        // the same post-force, post-flush state RPython feeds in.
        let virtual_state =
            crate::optimizeopt::virtualstate::export_state(&end_args, ctx, &ctx.forwarded);
        // unroll.py:459-461: infos = {}; for arg in end_args: _expand_info(arg, infos)
        let mut infos: HashMap<OpRef, ExportedValueInfo> = HashMap::new();
        for &arg in &end_args {
            self.expand_info(arg, ctx, exported_int_bounds, &mut infos);
        }
        // unroll.py:462-463 `label_args, virtuals =
        //   virtual_state.make_inputargs_and_virtuals(end_args, self.optimizer)`.
        let (label_args, virtuals) = virtual_state
            .make_inputargs_and_virtuals(&end_args, optimizer, ctx, false)
            .expect("export_state make_inputargs_and_virtuals failed");
        // unroll.py:464-465: for arg in label_args: _expand_info(arg, infos)
        for &arg in &label_args {
            self.expand_info(arg, ctx, exported_int_bounds, &mut infos);
        }
        let mut short_args = label_args.clone();
        short_args.extend(virtuals);
        let exported_short_ops = self.collect_exported_short_ops(&short_args, ctx);
        let exported_short_boxes = ctx.exported_short_boxes.clone();

        // RPython unroll.py:467: next_iteration_args = end_args (post-force).
        // Aliased boxes (same resolved OpRef) are handled by export_state's
        // create_state cache + make_inputargs' position_in_notvirtuals dedup.
        // Types are populated by the caller from Optimizer.trace_inputarg_types
        // after export_state returns. Initialize empty here.
        ExportedState::new(
            label_args.clone(),
            end_args,
            virtual_state,
            infos,
            exported_short_ops,
            exported_short_boxes,
            renamed_inputargs.to_vec(),
            Vec::new(), // populated by caller
            short_args,
        )
    }

    /// unroll.py:432-443: _expand_info
    fn expand_info(
        &self,
        arg: OpRef,
        ctx: &OptContext,
        exported_int_bounds: Option<&HashMap<OpRef, crate::optimizeopt::intutils::IntBound>>,
        infos: &mut HashMap<OpRef, ExportedValueInfo>,
    ) {
        let resolved = ctx.get_box_replacement(arg);
        if infos.contains_key(&resolved) {
            // Also store under the original key so import_state can
            // find the info using the unresolved next_iteration_args key.
            if arg != resolved {
                if let Some(info) = infos.get(&resolved).cloned() {
                    infos.insert(arg, info);
                }
            }
            return;
        }
        let info = self.collect_exported_info(resolved, ctx, exported_int_bounds);
        let has_fields = matches!(ctx.get_ptr_info(resolved), Some(pi) if pi.is_virtual() || !pi.all_items().is_empty());
        infos.insert(resolved, info.clone());
        // Also store under the original (unresolved) key.
        if arg != resolved {
            infos.insert(arg, info);
        }
        if has_fields {
            self.expand_infos_from_virtual(resolved, ctx, exported_int_bounds, infos);
        }
    }

    /// unroll.py:445-450: _expand_infos_from_virtual
    fn expand_infos_from_virtual(
        &self,
        opref: OpRef,
        ctx: &OptContext,
        exported_int_bounds: Option<&HashMap<OpRef, crate::optimizeopt::intutils::IntBound>>,
        infos: &mut HashMap<OpRef, ExportedValueInfo>,
    ) {
        let fields: Vec<OpRef> = match ctx.get_ptr_info(opref) {
            Some(crate::optimizeopt::info::PtrInfo::Virtual(v)) => {
                v.fields.iter().map(|(_, r)| *r).collect()
            }
            Some(crate::optimizeopt::info::PtrInfo::VirtualStruct(v)) => {
                v.fields.iter().map(|(_, r)| *r).collect()
            }
            Some(crate::optimizeopt::info::PtrInfo::VirtualArray(v)) => v.items.clone(),
            Some(crate::optimizeopt::info::PtrInfo::Instance(v)) if !v.fields.is_empty() => {
                v.fields.iter().map(|(_, r)| *r).collect()
            }
            Some(crate::optimizeopt::info::PtrInfo::Struct(v)) if !v.fields.is_empty() => {
                v.fields.iter().map(|(_, r)| *r).collect()
            }
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
        virtual_state: crate::optimizeopt::virtualstate::VirtualState,
        short_preamble: crate::optimizeopt::shortpreamble::ShortPreamble,
        short_preamble_builder: Option<&crate::optimizeopt::shortpreamble::ShortPreambleBuilder>,
    ) -> TargetToken {
        let mut target_token = TargetToken::new();
        target_token.token_id = token_id;
        target_token.jump_target_descr = Arc::new(LoopTargetDescr::new(token_id, false));
        target_token.virtual_state = Some(virtual_state);
        target_token.short_preamble = Some(short_preamble);
        target_token.short_preamble_producer = short_preamble_builder.map(|builder| {
            crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder::new(token_id, builder)
        });
        target_token
    }

    /// unroll.py:320-362: _jump_to_existing_trace — check if any existing
    /// compiled trace (target_token) has a compatible virtual state.
    /// If so, generate extra guards, inline short preamble, and redirect jump.
    ///
    /// Returns None if jumped successfully, Some(virtual_state) otherwise.
    /// unroll.py:304-362: jump_to_existing_trace
    ///
    /// `runtime_boxes`: live values at the jump point, used by
    /// generate_guards to emit GUARD_VALUE when the runtime value
    /// matches a known constant. Pass None when unavailable (bridges).
    pub fn jump_to_existing_trace(
        &self,
        jump_args: &[OpRef],
        current_label_args: Option<&[OpRef]>,
        target_tokens: &mut [TargetToken],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
        runtime_boxes: Option<&[OpRef]>,
    ) -> Option<crate::optimizeopt::virtualstate::VirtualState> {
        self.jump_to_existing_trace_with_vs(
            jump_args,
            current_label_args,
            target_tokens,
            optimizer,
            ctx,
            force_boxes,
            runtime_boxes,
            None,
        )
    }

    /// Like jump_to_existing_trace, but with an optional pre-computed
    /// virtual_state. Used by optimize_bridge where force_at_the_end_of_preamble
    /// may change forwarding chains after the virtual state was exported.
    pub fn jump_to_existing_trace_with_vs(
        &self,
        jump_args: &[OpRef],
        current_label_args: Option<&[OpRef]>,
        target_tokens: &mut [TargetToken],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
        runtime_boxes: Option<&[OpRef]>,
        pre_vs: Option<crate::optimizeopt::virtualstate::VirtualState>,
    ) -> Option<crate::optimizeopt::virtualstate::VirtualState> {
        optimizer.disable_guard_replacement();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.jump_to_existing_trace_impl(
                jump_args,
                current_label_args,
                target_tokens,
                optimizer,
                ctx,
                force_boxes,
                runtime_boxes,
                pre_vs,
            )
        }));
        optimizer.enable_guard_replacement();
        match result {
            Ok(vs) => vs,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    fn jump_to_existing_trace_impl(
        &self,
        jump_args: &[OpRef],
        current_label_args: Option<&[OpRef]>,
        target_tokens: &mut [TargetToken],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
        runtime_boxes: Option<&[OpRef]>,
        pre_vs: Option<crate::optimizeopt::virtualstate::VirtualState>,
    ) -> Option<crate::optimizeopt::virtualstate::VirtualState> {
        let mut virtual_state = pre_vs.unwrap_or_else(|| {
            crate::optimizeopt::virtualstate::export_state(jump_args, ctx, &ctx.forwarded)
        });
        let mut args: Vec<OpRef> = jump_args
            .iter()
            .map(|&a| ctx.get_box_replacement(a))
            .collect();
        let mut first_target_attempt = true;

        for (tt_idx, target_token) in target_tokens.iter_mut().enumerate() {
            if crate::optimizeopt::majit_log_enabled() {
                eprintln!("[jit][jump_to_existing] trying target_token #{tt_idx}");
            }
            if !first_target_attempt {
                // RPython unroll.py leaves bogus ops from failed target-token
                // attempts at the end of the live trace, which is safe there.
                // majit's detached jump_ctx later splices the redirected tail
                // back into the body, so stale extra guards/short-preamble ops
                // from an earlier failed target must not leak into a later
                // successful redirect.
                ctx.clear_newoperations();
            }
            first_target_attempt = false;

            let target_vs = match &target_token.virtual_state {
                Some(vs) => vs,
                None => continue,
            };

            // RPython unroll.py:333: patchguardop = self.optimizer.patchguardop
            // Ensure ctx.patchguardop is set before generate_guards so that
            // extra guards get a valid rd_resume_position (resume.py:397).
            if ctx.patchguardop.is_none() {
                ctx.patchguardop = optimizer.patchguardop.clone();
            }

            // RPython unroll.py:315 parity: try generate_guards directly
            // instead of gating on generalization_of. If guards can't be
            // generated (VirtualStatesCantMatch), skip this target.
            let extra_guards = match target_vs.generate_guards(
                &virtual_state,
                &args,
                runtime_boxes,
                force_boxes,
            ) {
                Ok(guards) => guards,
                Err(()) => {
                    if std::env::var_os("MAJIT_LOG_JTET").is_some() {
                        eprintln!(
                            "[jit][jte] target_token #{tt_idx} generate_guards failed (force_boxes={force_boxes})",
                        );
                    }
                    continue;
                }
            };
            for guard_req in &extra_guards {
                if let Some(mut guard_op) = guard_req.to_op(&args, ctx) {
                    // unroll.py:336: guard.rd_resume_position = patchguardop.rd_resume_position
                    // RPython: patchguardop is always set (from GUARD_FUTURE_CONDITION).
                    if let Some(ref patch) = ctx.patchguardop {
                        guard_op.rd_resume_position = patch.rd_resume_position;
                    }
                    guard_op.descr = Some(crate::optimizeopt::make_resume_at_position_descr());
                    optimizer.send_extra_operation(&guard_op, ctx);
                }
            }

            // unroll.py:346-347: make_inputargs_and_virtuals
            // RPython: force_box emits New/SetfieldGc via emit_extra which
            // routes through passes AFTER Virtualize. The non-virtual PtrInfo
            // (Struct/Instance) on alloc_ref prevents re-absorption.
            let (target_args, virtuals) = match target_vs.make_inputargs_and_virtuals(
                &args,
                optimizer,
                ctx,
                force_boxes,
            ) {
                Ok(result) => result,
                Err(()) => {
                    if std::env::var_os("MAJIT_LOG_JTET").is_some() {
                        eprintln!(
                            "[jit][jte] target_token #{tt_idx} make_inputargs failed (force_boxes={force_boxes})",
                        );
                    }
                    if force_boxes {
                        args = jump_args
                            .iter()
                            .map(|&a| ctx.get_box_replacement(a))
                            .collect();
                        virtual_state = crate::optimizeopt::virtualstate::export_state(
                            &args,
                            ctx,
                            &ctx.forwarded,
                        );
                    }
                    continue;
                }
            };
            let mut short_jump_args = target_args.clone();
            short_jump_args.extend(virtuals);

            // Ensure jump_args carry PtrInfo from Phase 2 body.
            // RPython Box identity preserves info across forwarding.
            // In majit, forwarding target may lack PtrInfo — propagate
            // from the original label arg (before forwarding).
            if let Some(label) = current_label_args {
                for (i, &jump_arg) in short_jump_args.iter().enumerate() {
                    let resolved = ctx.get_box_replacement(jump_arg);
                    if ctx.get_ptr_info(resolved).is_none() {
                        // Try label arg at same index
                        if let Some(&label_arg) = label.get(i) {
                            if let Some(info) = ctx.get_ptr_info(label_arg).cloned() {
                                ctx.ensure_ptr_info_preserve_forwarding(resolved, info);
                            }
                        }
                    }
                }
            }

            // unroll.py:353-356: inline short preamble
            let mut extra = Vec::new();
            if let Some(sp) = target_token.short_preamble.clone() {
                if let Some(mut builder) = target_token.short_preamble_producer.take() {
                    if let Some(label_args) = current_label_args {
                        // shortpreamble.py:283-296 / 311-341 parity:
                        // setup() returns false when an op references an
                        // unresolvable Phase 1 OpRef. Treat this exactly
                        // like RPython's "produce_arg returned None →
                        // add_op_to_short returned None" path: drop the
                        // peeled trace and let the unroll caller raise
                        // InvalidLoop, falling back to jump_to_preamble.
                        if !builder.setup(&sp, label_args) {
                            target_token.short_preamble_producer = Some(builder);
                            std::panic::panic_any(crate::optimize::InvalidLoop(
                                "short preamble has unresolvable Phase 1 args",
                            ));
                        }
                        ctx.activate_short_preamble_producer(builder);
                        extra = Self::inline_short_preamble(
                            &short_jump_args,
                            &target_args,
                            &sp,
                            optimizer,
                            ctx,
                        );
                        if let Some(builder) = ctx.take_active_short_preamble_producer() {
                            // RPython parity: extract constant pool from
                            // OptContext. In RPython, Const objects in short
                            // preamble ops are GC-tracked and survive across
                            // compilations. build_short_preamble_struct scans
                            // all op args to capture referenced constants.
                            let mut loop_constants: HashMap<u32, i64> = HashMap::new();
                            let mut loop_constant_types: HashMap<u32, majit_ir::Type> =
                                optimizer.constant_types.clone();
                            for (&const_idx, val) in &ctx.const_pool {
                                let (raw, tp) = match val {
                                    majit_ir::Value::Int(v) => (*v, majit_ir::Type::Int),
                                    majit_ir::Value::Float(f) => {
                                        (f.to_bits() as i64, majit_ir::Type::Float)
                                    }
                                    majit_ir::Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                                    majit_ir::Value::Void => (0, majit_ir::Type::Void),
                                };
                                let opref = OpRef::from_const(const_idx);
                                loop_constants.insert(opref.0, raw);
                                loop_constant_types.entry(opref.0).or_insert(tp);
                            }
                            // Merge previous short preamble's constants.
                            // RPython's Const objects survive across compilations
                            // via GC tracing. In majit, we must carry forward
                            // constants from the previous build that may not
                            // exist in the current Phase 2 context.
                            for (&k, &(v, tp)) in &sp.constants {
                                loop_constants.entry(k).or_insert(v);
                                loop_constant_types.entry(k).or_insert(tp);
                            }
                            target_token.short_preamble =
                                Some(builder.build_short_preamble_struct(
                                    &loop_constants,
                                    &loop_constant_types,
                                ));
                            target_token.short_preamble_producer = Some(builder);
                        }
                    } else {
                        extra = Self::inline_short_preamble(
                            &short_jump_args,
                            &target_args,
                            &sp,
                            optimizer,
                            ctx,
                        );
                        target_token.short_preamble_producer = Some(builder);
                    }
                } else {
                    extra = Self::inline_short_preamble(
                        &short_jump_args,
                        &target_args,
                        &sp,
                        optimizer,
                        ctx,
                    );
                    if crate::optimizeopt::majit_log_enabled() {
                        eprintln!("[jit][jte-isp] done, extra_len={}", extra.len());
                    }
                }
            }

            // unroll.py:357-359: emit JUMP to target
            let mut jump_args = target_args;
            jump_args.extend(extra);
            let mut jump = Op::new(OpCode::Jump, &jump_args);
            jump.descr = Some(target_token.as_jump_target_descr());
            optimizer.send_extra_operation(&jump, ctx);
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
        args_no_virtuals: &[OpRef],
        short_preamble: &crate::optimizeopt::shortpreamble::ShortPreamble,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
    ) -> Vec<OpRef> {
        // RPython parity: in RPython, short preamble ops embed Const objects
        // (ConstPtr/ConstInt) that carry their own value and are GC-tracked.
        // _map_args skips Const args — they're always valid because the GC
        // keeps referenced objects alive via JitCellToken → TargetToken →
        // short_preamble → ResOperation → ConstPtr chain.
        //
        // In pyre, short preamble ops use OpRef indices from the loop's
        // constant pool. These OpRefs aren't in the bridge's context.
        // Register them as constants in the bridge's OptContext so the ops
        // can reference them correctly.
        for (&idx, &(val, tp)) in &short_preamble.constants {
            let value = match tp {
                majit_ir::Type::Int => majit_ir::Value::Int(val),
                majit_ir::Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(val as usize)),
                majit_ir::Type::Float => majit_ir::Value::Float(f64::from_bits(val as u64)),
                majit_ir::Type::Void => majit_ir::Value::Int(val),
            };
            ctx.make_constant(majit_ir::OpRef(idx), value);
            optimizer.constant_types.entry(idx).or_insert(tp);
            // Store for compile_bridge to merge into Cranelift's constants map.
            optimizer
                .bridge_preamble_constants
                .entry(idx)
                .or_insert((val, tp));
        }

        let mut mapping: HashMap<OpRef, OpRef> = HashMap::new();

        for (i, &short_inputarg) in short_preamble.inputargs.iter().enumerate() {
            if let Some(&jump_arg) = jump_args.get(i) {
                mapping.insert(short_inputarg, jump_arg);
                // RPython: jump_arg Box inherits info via identity.
                // In majit, propagate PtrInfo from short_inputarg (which
                // has info from Phase 1 export) to the resolved jump_arg.
                // shortpreamble.py:414-425 parity: propagate PtrInfo from
                // Phase 1 export to jump_args so guards are redundant.
                let resolved = ctx.get_box_replacement(jump_arg);
                if ctx.get_ptr_info(resolved).is_none() {
                    let info = ctx
                        .get_ptr_info(jump_arg)
                        .cloned()
                        .or_else(|| ctx.get_ptr_info(short_inputarg).cloned())
                        .or_else(|| {
                            short_preamble
                                .inputarg_infos
                                .get(i)
                                .and_then(|opt| opt.clone())
                        });
                    if let Some(info) = info {
                        ctx.ensure_ptr_info_preserve_forwarding(resolved, info);
                    }
                }
            }
        }

        // RPython parity: also map Phase 1 inputargs → jump_args.
        // Short ops may reference Phase 1 OpRefs (from produce_arg's
        // label_arg_positions check) that aren't in the current inputargs.
        // In RPython, renamed inputargs are stable across compilations,
        // so this situation doesn't arise.
        if let Some(ref phase1) = short_preamble.phase1_inputargs {
            for (i, &phase1_inputarg) in phase1.iter().enumerate() {
                if let Some(&jump_arg) = jump_args.get(i) {
                    mapping.entry(phase1_inputarg).or_insert(jump_arg);
                }
            }
        }

        let mut replay_index = 0;

        fn current_short_len(
            short_preamble: &crate::optimizeopt::shortpreamble::ShortPreamble,
            ctx: &OptContext,
        ) -> usize {
            ctx.active_short_preamble_producer
                .as_ref()
                .map(|builder| builder.short_ops_len())
                .unwrap_or_else(|| short_preamble.ops.len())
        }

        fn current_short_op(
            short_preamble: &crate::optimizeopt::shortpreamble::ShortPreamble,
            ctx: &OptContext,
            index: usize,
        ) -> Option<Op> {
            if let Some(builder) = ctx.active_short_preamble_producer.as_ref() {
                builder.short_op(index).cloned()
            } else {
                short_preamble.ops.get(index).map(|entry| entry.op.clone())
            }
        }

        fn current_short_jump_args(
            short_preamble: &crate::optimizeopt::shortpreamble::ShortPreamble,
            ctx: &OptContext,
        ) -> Vec<OpRef> {
            ctx.active_short_preamble_producer
                .as_ref()
                .map(|builder| builder.jump_args().to_vec())
                .unwrap_or_else(|| short_preamble.jump_args.clone())
        }

        // unroll.py:398-427: fix-point loop, runs only once in almost all cases.
        // RPython uses `while 1:` until convergence, but in practice it should
        // converge in very few iterations. Add a safety cap to prevent hangs.
        let mut fixpoint_iter = 0u32;
        loop {
            fixpoint_iter += 1;
            if fixpoint_iter > 20 {
                if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!(
                        "[jit][inline_short_preamble] fixpoint loop exceeded 20 iterations, breaking"
                    );
                }
                break;
            }
            // unroll.py:402: while i < len(short) - 1
            // Use LIVE length — newly added ops (from use_box during
            // send_extra_operation) are replayed in the same iteration.
            while replay_index < current_short_len(short_preamble, ctx) {
                let Some(sp_op) = current_short_op(short_preamble, ctx, replay_index) else {
                    break;
                };
                let mut new_op = sp_op.clone();
                // unroll.py:404: _map_args(mapping, sop.getarglist())
                // Const passes through unchanged, non-Const must be in mapping.
                for arg in &mut new_op.args {
                    // unroll.py:367: isinstance(box, Const) — true Const objects
                    // only (ConstInt/ConstPtr/ConstFloat). make_constant'd values
                    // are NOT Const objects — they are regular boxes with forwarded
                    // set to a Const, so they must go through the mapping.
                    if arg.is_constant() || short_preamble.constants.contains_key(&arg.0) {
                        continue;
                    }
                    // unroll.py:404: _map_args — non-Const must be in mapping.
                    // RPython: mapping is complete (seeded from short_inputargs →
                    // jump_args, extended by mapping[sop] = op). Missing keys
                    // indicate a structural mismatch (e.g., cross-loop bridge
                    // with incompatible short preamble). Raise InvalidLoop.
                    match mapping.get(arg) {
                        Some(&mapped) => *arg = mapped,
                        None => {
                            // RPython: _map_args raises KeyError for unmapped
                            // args. This is equivalent to InvalidLoop — the
                            // short preamble is structurally incompatible.
                            // Panic propagates through jump_to_existing_trace
                            // to the caller, which catches it and falls back
                            // to jump_to_preamble (unroll.py:154-158, 209-211).
                            if crate::optimizeopt::majit_log_enabled() {
                                eprintln!(
                                    "[jit] inline_short_preamble: unmapped arg {:?} in {:?} — InvalidLoop",
                                    arg, new_op.opcode
                                );
                            }
                            std::panic::panic_any(crate::optimize::InvalidLoop(
                                "inline_short_preamble: unmapped arg in short preamble",
                            ));
                        }
                    }
                }
                // unroll.py:405-414: unified guard/non-guard handling.
                // RPython: both guards and non-guards follow the same path:
                //   copy_and_change → mapping[sop] = op → send_extra_operation(op)
                if new_op.opcode.is_guard() {
                    // unroll.py:406-409: copy_and_change with ResumeAtPositionDescr
                    new_op.descr = Some(crate::optimizeopt::make_resume_at_position_descr());
                    new_op.fail_args = None;
                    new_op.rd_numb = None;
                    new_op.rd_consts = None;
                    new_op.rd_virtuals = None;
                    new_op.rd_pendingfields = None;
                    new_op.fail_arg_types = None;
                    // unroll.py:409: op.rd_resume_position = patchguardop.rd_resume_position
                    // RPython: patchguardop is always set (from GUARD_FUTURE_CONDITION).
                    if let Some(ref patch) = ctx.patchguardop {
                        new_op.rd_resume_position = patch.rd_resume_position;
                    }
                    // Re-register guard constant args from preamble's constant pool.
                    for &arg in &new_op.args {
                        if let Some(&(val, tp)) = short_preamble.constants.get(&arg.0) {
                            let value = match tp {
                                majit_ir::Type::Int => majit_ir::Value::Int(val),
                                majit_ir::Type::Ref => {
                                    majit_ir::Value::Ref(majit_ir::GcRef(val as usize))
                                }
                                majit_ir::Type::Float => {
                                    majit_ir::Value::Float(f64::from_bits(val as u64))
                                }
                                majit_ir::Type::Void => majit_ir::Value::Int(val),
                            };
                            ctx.make_constant(arg, value);
                        }
                    }
                } else if let Some(ref mut fail_args) = new_op.fail_args {
                    for arg in fail_args.iter_mut() {
                        if let Some(&mapped) = mapping.get(arg) {
                            *arg = mapped;
                        }
                    }
                }
                let new_ref = ctx.alloc_op_position();
                new_op.pos = new_ref;
                // unroll.py:412-414: mapping[sop] = op; i += 1; send_extra_operation(op)
                // RPython sets mapping BEFORE send_extra_operation.
                mapping.insert(sp_op.pos, new_ref);
                replay_index += 1;
                optimizer.send_extra_operation(&new_op, ctx);
            }

            // unroll.py:417-423: force all except virtuals.
            loop {
                let short_jump_args = current_short_jump_args(short_preamble, ctx);
                let num_short_jump_args = short_jump_args.len();
                // unroll.py:420: _map_args(mapping, short_jump_args)
                // Const passes through, non-Const requires mapping.
                let mapped_jump_args: Vec<OpRef> = short_jump_args
                    .iter()
                    .map(|jump_arg| {
                        let mapped = if short_preamble.constants.contains_key(&jump_arg.0)
                            || jump_arg.is_constant()
                        {
                            *jump_arg
                        } else {
                            mapping[jump_arg]
                        };
                        ctx.get_box_replacement(mapped)
                    })
                    .collect();
                // unroll.py:419-421
                for &arg in args_no_virtuals.iter().chain(mapped_jump_args.iter()) {
                    let _ = optimizer.force_box(arg, ctx);
                }
                if current_short_jump_args(short_preamble, ctx).len() == num_short_jump_args {
                    break;
                }
            }
            // unroll.py:424
            optimizer.flush(ctx);
            // unroll.py:426: done unless "short" has grown again
            if replay_index == current_short_len(short_preamble, ctx) {
                break;
            }
        }

        // RPython: get_box_replacement follows forwarding after mapping
        current_short_jump_args(short_preamble, ctx)
            .iter()
            .map(|&jump_arg| {
                let mapped = *mapping.get(&jump_arg).unwrap_or(&jump_arg);
                ctx.get_box_replacement(mapped)
            })
            .collect()
    }

    /// unroll.py:479-504 import_state — line-by-line port.
    ///
    /// ```python
    /// def import_state(self, targetargs, exported_state):
    ///     assert len(exported_state.next_iteration_args) == len(targetargs)
    ///     for i, target in enumerate(exported_state.next_iteration_args):
    ///         source = targetargs[i]
    ///         assert source is not target
    ///         source.set_forwarded(target)
    ///         info = exported_state.exported_infos.get(target, None)
    ///         if info is not None:
    ///             self.optimizer.setinfo_from_preamble(source, info,
    ///                                             exported_state.exported_infos)
    ///     label_args = exported_state.virtual_state.make_inputargs(
    ///         targetargs, self.optimizer)
    ///     self.short_preamble_producer = ShortPreambleBuilder(
    ///         label_args, exported_state.short_boxes,
    ///         exported_state.short_inputargs, exported_state.exported_infos,
    ///         self.optimizer)
    ///     for produced_op in exported_state.short_boxes:
    ///         produced_op.produce_op(self, exported_state.exported_infos)
    ///     return label_args
    /// ```
    ///
    /// In RPython the short-preamble setup is done by constructing
    /// `ShortPreambleBuilder` with `label_args` and looping over
    /// `exported_state.short_boxes` calling `produce_op`.  In majit the
    /// equivalent is `import_short_preamble_ops` (a serialized form of
    /// the per-PreambleOp `produce_op` body) plus
    /// `initialize_imported_short_preamble_builder` (a serialized form
    /// of the `ShortPreambleBuilder` constructor).  Both consume the
    /// same `short_args = label_args + virtuals` slot space that
    /// `export_state` used to build `ExportedShortOp` (see
    /// `collect_exported_short_ops`); the slot indices are the
    /// majit-only stand-in for RPython's Box-keyed
    /// `produced_short_boxes` dict.  The dual `_from_exported_ops`
    /// initializer is the preferred path; it falls back to
    /// `initialize_imported_short_preamble_builder` for short ops that
    /// the serialized form does not yet handle (Calls).
    pub fn import_state(
        &self,
        targetargs: &[OpRef],
        exported_state: &ExportedState,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
    ) -> Vec<OpRef> {
        // assert len(exported_state.next_iteration_args) == len(targetargs)
        assert_eq!(
            exported_state.next_iteration_args.len(),
            targetargs.len(),
            "import_state: next_iteration_args mismatch"
        );
        // for i, target in enumerate(exported_state.next_iteration_args):
        for (i, target) in exported_state.next_iteration_args.iter().enumerate() {
            // source = targetargs[i]
            let source = targetargs[i];
            // assert source is not target — see commit log for the
            // disjoint-namespace invariant from Step 2 Commit D2 that
            // makes this hold by construction in production callers.
            debug_assert!(source != *target, "import_state: source is target");
            // source.set_forwarded(target)
            ctx.replace_op(source, *target);
            // info = exported_state.exported_infos.get(target, None)
            // if info is not None:
            if let Some(info) = exported_state.exported_infos.get(target) {
                //     self.optimizer.setinfo_from_preamble(source, info,
                //                                     exported_state.exported_infos)
                self.setinfo_from_preamble(source, info, &exported_state.exported_infos, ctx);
            }
        }
        // label_args = exported_state.virtual_state.make_inputargs(
        //     targetargs, self.optimizer)
        let label_args = exported_state
            .virtual_state
            .make_inputargs(targetargs, optimizer, ctx, false)
            .expect("import_state make_inputargs failed (VirtualStatesCantMatch)");
        // self.short_preamble_producer = ShortPreambleBuilder(
        //     label_args, exported_state.short_boxes,
        //     exported_state.short_inputargs, exported_state.exported_infos,
        //     self.optimizer)
        //
        // majit's `ShortPreambleBuilder` constructor is split into
        // `initialize_imported_short_preamble_builder*`. The majit
        // serialization keys short ops by slot index over the combined
        // `label_args + virtuals` slot space (the same space
        // `collect_exported_short_ops` built against), so we compute the
        // virtuals tail inline before calling the initializer. The dual
        // initializer paths (`_from_exported_ops` preferred, legacy
        // fallback for short ops the serialization doesn't yet handle)
        // are an outstanding RPython divergence pending Call support in
        // `ExportedShortOp`.
        let virtuals: Vec<OpRef> = exported_state
            .virtual_state
            .state
            .iter()
            .enumerate()
            .filter(|(_, info)| info.is_virtual())
            .filter_map(|(i, _)| targetargs.get(i).copied())
            .collect();
        let mut short_args = label_args.clone();
        short_args.extend(virtuals);
        // for produced_op in exported_state.short_boxes:
        //     produced_op.produce_op(self, exported_state.exported_infos)
        self.import_short_preamble_ops(&short_args, exported_state, ctx);
        let from_exported = ctx.initialize_imported_short_preamble_builder_from_exported_ops(
            &short_args,
            &exported_state.short_inputargs,
            &exported_state.exported_short_ops,
        );
        if !from_exported {
            ctx.initialize_imported_short_preamble_builder(
                &label_args,
                &exported_state.short_inputargs,
                &exported_state.exported_short_boxes,
            );
        }
        // return label_args
        label_args
    }

    fn collect_exported_info(
        &self,
        opref: OpRef,
        ctx: &OptContext,
        exported_int_bounds: Option<&HashMap<OpRef, crate::optimizeopt::intutils::IntBound>>,
    ) -> ExportedValueInfo {
        let resolved = ctx.get_box_replacement(opref);
        // RPython export_state/_expand_info records properties on the Box
        // objects that survive to the next iteration. In majit, a loop box may
        // temporarily forward to a Const during the current iteration even
        // though the next iteration still needs a box identity. Export only
        // true Const objects here; forwarded-to-Const metadata would otherwise
        // over-specialize the imported targetarg and leak one iteration's
        // guard knowledge into the next.
        let constant = if opref.is_constant() {
            ctx.get_constant(resolved).cloned()
        } else {
            None
        };
        let ptr_info = match ctx.get_ptr_info(resolved).cloned() {
            Some(crate::optimizeopt::info::PtrInfo::Constant(_)) if !opref.is_constant() => None,
            other => other,
        };
        // unroll.py:432-443 _expand_info uses self.optimizer.getinfo(arg) which
        // dispatches by op.type ('r' → getptrinfo, 'i' → getintbound). The Rust
        // port stores int bounds in a separate `int_bound` field populated
        // earlier by `OptIntBounds::export_arg_int_bounds`, which already
        // filters by `opref_type(resolved) == Some(Int)`. We rely on that
        // filter so the lookup here cannot pull a bound for a ref/float box.
        let int_bound = exported_int_bounds.and_then(|bounds| bounds.get(&resolved).cloned());
        ExportedValueInfo {
            constant,
            ptr_info,
            int_bound,
        }
    }

    /// unroll.py:53-98 setinfo_from_preamble(op, preamble_info, exported_infos)
    ///
    /// majit thin wrapper that resolves the box, unpacks the `PtrInfo`
    /// out of `ExportedValueInfo`, and delegates to the canonical
    /// `OptContext::setinfo_from_preamble` (mod.rs), which is the
    /// line-by-line port of unroll.py:59-92.
    ///
    /// `ExportedValueInfo` is majit's serialized container around the
    /// PtrInfo plus a few side fields (constant, int_lower_bound) that
    /// RPython carries on the box itself. This wrapper is the only
    /// place that needs to know the serialization details.
    fn setinfo_from_preamble(
        &self,
        opref: OpRef,
        info: &ExportedValueInfo,
        exported_infos: &HashMap<OpRef, ExportedValueInfo>,
        ctx: &mut OptContext,
    ) {
        use crate::optimizeopt::info::PtrInfo;

        // unroll.py:53-54: op = get_box_replacement(op)
        let target = ctx.get_box_replacement(opref);
        // unroll.py:55-56: if op.get_forwarded() is not None: return
        if ctx.has_forwarding(target) {
            return;
        }
        // unroll.py:57-58: if op.is_constant(): return  # nothing we can learn
        if ctx.is_constant(target) {
            return;
        }
        // unroll.py:59-92: isinstance(preamble_info, info.PtrInfo)
        if let Some(ptr_info) = info.ptr_info.clone() {
            if ptr_info.is_constant() {
                if let Some(value) = &info.constant {
                    ctx.make_constant(target, value.clone());
                    if let Value::Ref(ptr) = value {
                        ctx.set_ptr_info(target, crate::optimizeopt::info::PtrInfo::Constant(*ptr));
                    }
                }
                return;
            }
            // Delegate the non-constant PtrInfo cases to the canonical
            // line-by-line port in OptContext.
            ctx.setinfo_from_preamble(target, &ptr_info, Some(exported_infos));
        }
        // unroll.py:93-96: IntBound import with widen().
        //
        //     elif isinstance(preamble_info, intutils.IntBound):
        //         loop_info = preamble_info.widen()
        //         intbound = self.getintbound(op)
        //         intbound.intersect(loop_info)
        //
        // widen() relaxes exact bounds to avoid over-specialization:
        //   lower < MININT/2 → MININT
        //   upper > MAXINT/2 → MAXINT
        //   tvalue/tmask → UNKNOWN
        // This ensures Phase 2 gets safe bounds (e.g., [0, MAXINT] for
        // a loop counter) without the exact preamble values.
        // RPython parity: imported preamble bounds become the box's
        // forwarded IntBound directly (optimizer.py:115-125 setintbound).
        if let Some(bound) = &info.int_bound {
            let widened = bound.widen();
            ctx.setintbound(target, &widened);
        }
    }

    fn collect_exported_short_ops(
        &self,
        short_args: &[OpRef],
        ctx: &OptContext,
    ) -> Vec<ExportedShortOp> {
        fn exported_const_arg(ctx: &OptContext, arg: OpRef) -> Option<ExportedShortArg> {
            // RPython shortpreamble.py: produce_arg() only serializes real
            // Const objects here. A loop box that merely has Forwarded::Const
            // metadata for the current iteration must stay a box; exporting it
            // as Const leaks one iteration's guard knowledge into the next.
            if !arg.is_constant() {
                return None;
            }
            ctx.get_constant(arg)
                .cloned()
                .map(|value| ExportedShortArg::Const { source: arg, value })
        }

        let short_boxes =
            crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(short_args);
        let mut produced_indices: HashMap<OpRef, usize> = HashMap::new();
        let mut next_temp = 0usize;
        let mut exported = Vec::new();
        for entry in &ctx.exported_short_boxes {
            let current_result = match entry.kind {
                crate::optimizeopt::shortpreamble::PreambleOpKind::Heap => entry.op.pos,
                _ => entry.op.pos,
            };
            let result = if let Some(slot) = short_boxes.lookup_label_arg(current_result) {
                ExportedShortResult::Slot(slot)
            } else {
                let temp = next_temp;
                next_temp += 1;
                ExportedShortResult::Temporary(temp)
            };
            let exported_op = match entry.kind {
                crate::optimizeopt::shortpreamble::PreambleOpKind::Pure => {
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
                                .or_else(|| exported_const_arg(ctx, arg))
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
                        descr: entry.op.descr.clone(),
                        args,
                        result,
                        invented_name: entry.invented_name,
                        same_as_source: entry.same_as_source,
                    })
                }
                crate::optimizeopt::shortpreamble::PreambleOpKind::Heap => {
                    // RPython shortpreamble.py:91-103: HeapOp.add_op_to_short
                    // preamble_arg = sb.produce_arg(sop.getarg(0))
                    // RPython produce_arg: label_arg → Slot, Const → Const
                    let struct_arg = entry.op.arg(0);
                    let object = short_boxes
                        .lookup_label_arg(struct_arg)
                        .map(ExportedShortArg::Slot)
                        .or_else(|| exported_const_arg(ctx, struct_arg));
                    let Some(object) = object else {
                        continue;
                    };
                    let Some(descr) = entry.op.descr.clone() else {
                        continue;
                    };
                    match entry.op.opcode {
                        OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF => {
                            Some(ExportedShortOp::HeapField {
                                source: current_result,
                                object,
                                descr,
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
                                source: current_result,
                                object: object,
                                descr,
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
                crate::optimizeopt::shortpreamble::PreambleOpKind::LoopInvariant => {
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
        fn imported_const_opref(
            ctx: &mut OptContext,
            imported_constants: &mut HashMap<OpRef, OpRef>,
            source: OpRef,
            value: &Value,
        ) -> OpRef {
            if source.is_constant() {
                ctx.seed_constant(source, value.clone());
                source
            } else {
                if let Some(&opref) = imported_constants.get(&source) {
                    return opref;
                }
                let opref = ctx.alloc_op_position();
                ctx.make_constant(opref, value.clone());
                imported_constants.insert(source, opref);
                opref
            }
        }

        fn resolve_exported_short_result(
            result: &ExportedShortResult,
            short_args: &[OpRef],
            temporary_results: &mut Vec<Option<OpRef>>,
            ctx: &mut OptContext,
        ) -> Option<OpRef> {
            match result {
                ExportedShortResult::Slot(slot) => short_args.get(*slot).copied(),
                // RPython keeps box identity across traces. In majit, raw OpRef
                // indices are trace-local, so imported temporary short results
                // must get a fresh local OpRef and carry `source` separately via
                // `imported_short_sources`. Reusing the old source OpRef lets
                // stale positions leak into assembled traces.
                ExportedShortResult::Temporary(index) => {
                    if *index >= temporary_results.len() {
                        temporary_results.resize(index + 1, None);
                    }
                    let slot = &mut temporary_results[*index];
                    Some(*slot.get_or_insert_with(|| ctx.alloc_op_position()))
                }
            }
        }

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] import_short_preamble_ops: short_args={}, exported_short_ops={}",
                short_args.len(),
                exported_state.exported_short_ops.len()
            );
        }
        let mut produced_results = Vec::with_capacity(exported_state.exported_short_ops.len());
        let mut temporary_results: Vec<Option<OpRef>> = Vec::new();
        let mut imported_constants: HashMap<OpRef, OpRef> = HashMap::new();
        for entry in &exported_state.exported_short_ops {
            match *entry {
                ExportedShortOp::Pure {
                    source,
                    opcode,
                    ref descr,
                    ref args,
                    ref result,
                    invented_name,
                    same_as_source,
                } => {
                    let Some(result_opref) = resolve_exported_short_result(
                        result,
                        short_args,
                        &mut temporary_results,
                        ctx,
                    ) else {
                        continue;
                    };
                    // RPython parity: shortpreamble.py:112-126 PureOp.produce_op
                    //
                    //     if invented_name:
                    //         op = self.orig_op.copy_and_change(...)
                    //         op.set_forwarded(self.res)
                    //     else:
                    //         op = self.res
                    //
                    // The `invented_name` case DOES set forwarding — it forwards
                    // the freshly-allocated copy of orig_op to `self.res` (the
                    // SameAs alias). The non-invented case reuses self.res
                    // directly, no forwarding needed.
                    //
                    // In majit's flat OpRef model:
                    // - `source` is the preamble op's position (== `self.res`
                    //   in RPython terms).
                    // - `result_opref` is the Phase 2 fresh slot that Phase 2
                    //   trace ops will reference.
                    //
                    // For invented_name we forward source → result_opref so
                    // that `force_op_from_preamble`'s key lookup
                    // (`get_box_replacement(preamble_source)` at mod.rs:1134)
                    // resolves to `result_opref`, matching RPython
                    // `unroll.py:35-36 get_box_replacement(op)` where op was
                    // set_forwarded to self.res. Without this, the PreambleOp
                    // is stored under `source` but `force_box(result_opref)`
                    // looks it up by `result_opref` and never finds the alias
                    // metadata.
                    //
                    // For non-invented we DO NOT forward: the old unconditional
                    // replace_op routed body `GuardTrue(v12)` through the Phase 1
                    // stale boolean instead of letting the body emit a fresh
                    // IntLt for the new iteration's current — breaks loop body
                    // semantics for per-iteration pure ops.
                    if invented_name {
                        ctx.replace_op(source, result_opref);
                    }
                    // Register type for the new OpRef (RPython Box.type parity).
                    ctx.value_types.insert(result_opref.0, opcode.result_type());
                    let args = args
                        .iter()
                        .map(|arg| match arg {
                            ExportedShortArg::Slot(slot) => short_args
                                .get(*slot)
                                .copied()
                                .map(crate::optimizeopt::ImportedShortPureArg::OpRef),
                            ExportedShortArg::Const { source, value } => {
                                let const_opref = imported_const_opref(
                                    ctx,
                                    &mut imported_constants,
                                    *source,
                                    value,
                                );
                                Some(crate::optimizeopt::ImportedShortPureArg::Const(
                                    *value,
                                    const_opref,
                                ))
                            }
                            ExportedShortArg::Produced(index) => produced_results
                                .get(*index)
                                .copied()
                                .map(crate::optimizeopt::ImportedShortPureArg::OpRef),
                        })
                        .collect::<Option<Vec<_>>>();
                    let Some(args) = args else {
                        continue;
                    };
                    // RPython PureOp.produce_op() preloads all pure ops
                    // (including OVF) into OptPure's CSE table. OptPure
                    // postpones OVF ops and only resolves them when the
                    // paired GuardNoOverflow arrives, so guard pairing
                    // is maintained even for imported OVF operations.
                    ctx.imported_short_pure_ops
                        .push(crate::optimizeopt::ImportedShortPureOp {
                            opcode,
                            descr: descr.clone(),
                            args,
                            result: result_opref,
                        });
                    ctx.imported_short_sources
                        .push(crate::optimizeopt::ImportedShortSource {
                            result: result_opref,
                            source,
                        });
                    if invented_name {
                        if let Some(source) = same_as_source {
                            ctx.imported_short_aliases.push(
                                crate::optimizeopt::ImportedShortAlias {
                                    result: result_opref,
                                    same_as_source: source,
                                    same_as_opcode: OpCode::same_as_for_type(opcode.result_type()),
                                },
                            );
                        }
                    }
                    produced_results.push(result_opref);
                }
                ExportedShortOp::HeapField {
                    source,
                    ref object,
                    ref descr,
                    result_type,
                    ref result,
                    invented_name,
                    same_as_source,
                } => {
                    // Resolve object arg before resolve_result to avoid borrow conflict.
                    let obj =
                        match object {
                            ExportedShortArg::Slot(slot) => short_args.get(*slot).copied(),
                            ExportedShortArg::Const { source, value } => Some(
                                imported_const_opref(ctx, &mut imported_constants, *source, value),
                            ),
                            ExportedShortArg::Produced(idx) => produced_results.get(*idx).copied(),
                        };
                    let Some(obj) = obj else {
                        continue;
                    };
                    // RPython shortpreamble.py:62-85: HeapOp.produce_op
                    // pop = PreambleOp(self.res, preamble_op, invented_name)
                    // opinfo.setfield(descr, struct, pop, optheap, cf)
                    //
                    // Allocate a fresh OpRef (PreambleOp.op identity isolation)
                    // and store PreambleOp in PtrInfo._fields.
                    // CachedField._getfield (heap.py:177-187) will detect it
                    // via take_preamble_field and call force_op_from_preamble.
                    let _slot_value = resolve_exported_short_result(
                        result,
                        short_args,
                        &mut temporary_results,
                        ctx,
                    );
                    let value = ctx.alloc_op_position();
                    // Register type for the new OpRef (RPython Box.type parity).
                    ctx.value_types.insert(value.0, result_type);
                    // Prefer parent-local index when the FieldDescr is wired
                    // up to a SizeDescr; otherwise (lib-test fixtures or
                    // descrs constructed without parent_descr) fall back to
                    // the raw descr.index() so cache lookups stay stable.
                    let descr_idx = descr
                        .as_field_descr()
                        .and_then(|field_descr| {
                            field_descr
                                .get_parent_descr()
                                .map(|_| field_descr.index_in_parent() as u32)
                        })
                        .unwrap_or_else(|| descr.index());
                    let obj_resolved = ctx.get_box_replacement(obj);
                    // shortpreamble.py:66-68: HeapOp.produce_op
                    // if g.getarg(0) in exported_infos:
                    //     setinfo_from_preamble(g.getarg(0), exported_infos[...])
                    if let Some(einfo) = exported_state.exported_infos.get(&obj) {
                        if let Some(ref pinfo) = einfo.ptr_info {
                            ctx.setinfo_from_preamble(
                                obj_resolved,
                                pinfo,
                                Some(&exported_state.exported_infos),
                            );
                        }
                    }
                    let pop = crate::optimizeopt::info::PreambleOp {
                        op: source,
                        resolved: value,
                        invented_name,
                    };
                    let mut getfield_op =
                        majit_ir::Op::new(OpCode::getfield_for_type(result_type), &[obj_resolved]);
                    getfield_op.descr = Some(descr.clone());
                    // RPython shortpreamble.py:72-79:
                    //   opinfo = ensure_ptr_info_arg0(g)
                    //   opinfo.setfield(..., pop, ...)
                    //
                    // Preserve the existing const-info lookup because majit's
                    // heap import still routes ConstPtr preamble fields via
                    // const_infos before the full ConstPtrInfo.setfield port
                    // lands.
                    if let Some(info) = ctx.get_const_info_mut(obj_resolved) {
                        info.set_preamble_field(descr_idx, pop.clone());
                    }
                    let mut struct_info = ctx.ensure_ptr_info_arg0(&getfield_op);
                    if let Some(info) = struct_info.as_mut() {
                        info.set_preamble_field(descr_idx, pop.clone());
                    }
                    if crate::optimizeopt::majit_log_enabled() {
                        eprintln!(
                            "[jit] import_short_heap_field: obj={obj:?} descr_idx={descr_idx} value={value:?} preamble_op={source:?}"
                        );
                    }
                    ctx.imported_short_sources
                        .push(crate::optimizeopt::ImportedShortSource {
                            result: value,
                            source,
                        });
                    if invented_name {
                        if let Some(source) = same_as_source {
                            ctx.imported_short_aliases.push(
                                crate::optimizeopt::ImportedShortAlias {
                                    result: value,
                                    same_as_source: source,
                                    same_as_opcode: OpCode::same_as_for_type(result_type),
                                },
                            );
                        }
                    }
                    produced_results.push(value);
                }
                ExportedShortOp::HeapArrayItem {
                    source,
                    ref object,
                    ref descr,
                    index,
                    result_type,
                    ref result,
                    invented_name,
                    same_as_source,
                } => {
                    let obj =
                        match object {
                            ExportedShortArg::Slot(slot) => short_args.get(*slot).copied(),
                            ExportedShortArg::Const { source, value } => Some(
                                imported_const_opref(ctx, &mut imported_constants, *source, value),
                            ),
                            ExportedShortArg::Produced(idx) => produced_results.get(*idx).copied(),
                        };
                    let Some(obj) = obj else {
                        continue;
                    };
                    // RPython shortpreamble.py:80-85: HeapOp.produce_op (array item)
                    // Fresh OpRef for PreambleOp.op identity isolation.
                    let _slot_value = resolve_exported_short_result(
                        result,
                        short_args,
                        &mut temporary_results,
                        ctx,
                    );
                    let value = ctx.alloc_op_position();
                    // Register type for the new OpRef (RPython Box.type parity).
                    ctx.value_types.insert(value.0, result_type);
                    let descr_idx = descr.index();
                    // shortpreamble.py:72,84 parity: canonicalize obj through
                    // get_box_replacement so the key matches heap.py lookup
                    // which also canonicalizes array via get_box_replacement.
                    let obj_resolved = ctx.get_box_replacement(obj);
                    let index_const = ctx.make_constant_int(index as i64);
                    let mut getarrayitem_op = majit_ir::Op::new(
                        OpCode::getarrayitem_for_type(result_type),
                        &[obj_resolved, index_const],
                    );
                    getarrayitem_op.descr = Some(descr.clone());
                    let mut array_info = ctx.ensure_ptr_info_arg0(&getarrayitem_op);
                    if let Some(crate::optimizeopt::info::PtrInfo::Array(info)) =
                        array_info.as_mut()
                    {
                        let _ = info.lenbound.make_gt_const(index as i64);
                        if index as usize >= info.items.len() {
                            info.items.resize(index as usize + 1, OpRef::NONE);
                        }
                        info.items[index as usize] = value;
                    }
                    ctx.imported_short_arrayitems
                        .insert((obj_resolved, descr_idx, index), value);
                    ctx.imported_short_arrayitem_descrs
                        .insert((obj_resolved, descr_idx, index), descr.clone());
                    ctx.imported_short_sources
                        .push(crate::optimizeopt::ImportedShortSource {
                            result: value,
                            source,
                        });
                    if invented_name {
                        if let Some(source) = same_as_source {
                            ctx.imported_short_aliases.push(
                                crate::optimizeopt::ImportedShortAlias {
                                    result: value,
                                    same_as_source: source,
                                    same_as_opcode: OpCode::same_as_for_type(result_type),
                                },
                            );
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
                    let Some(value) = resolve_exported_short_result(
                        result,
                        short_args,
                        &mut temporary_results,
                        ctx,
                    ) else {
                        continue;
                    };
                    ctx.imported_loop_invariant_results.insert(func_ptr, value);
                    ctx.imported_short_sources
                        .push(crate::optimizeopt::ImportedShortSource {
                            result: value,
                            source,
                        });
                    if invented_name {
                        if let Some(source) = same_as_source {
                            ctx.imported_short_aliases.push(
                                crate::optimizeopt::ImportedShortAlias {
                                    result: value,
                                    same_as_source: source,
                                    same_as_opcode: OpCode::same_as_for_type(result_type),
                                },
                            );
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
    optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
    ctx: &mut OptContext,
    exported_int_bounds: Option<&HashMap<OpRef, crate::optimizeopt::intutils::IntBound>>,
) -> ExportedState {
    OptUnroll::new().export_state_with_bounds(
        jump_args,
        renamed_inputargs,
        optimizer,
        ctx,
        exported_int_bounds,
    )
}

/// unroll.py:479-504 import_state — module-level entry point.
pub(crate) fn import_state(
    targetargs: &[OpRef],
    exported_state: &ExportedState,
    optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
    ctx: &mut OptContext,
) -> Vec<OpRef> {
    OptUnroll::new().import_state(targetargs, exported_state, optimizer, ctx)
}

/// majit-only sibling of `import_state`: derive the per-non-virtual
/// inputarg "source slot" array from the same `targetargs` /
/// VirtualState walk that `make_inputargs` performs.
///
/// `imported_label_source_slots` carries the *original* incoming Phase
/// 2 inputarg OpRef for each label_args entry (rather than the
/// box-replacement target).  `assemble_peeled_trace_with_jump_args`
/// uses it to map body source references back onto the correct LABEL
/// slot during peeled-loop assembly.  RPython does not need this
/// because Box identity is shared across the assembly boundary; in
/// majit the slot identity is encoded as a parallel OpRef array.
pub(crate) fn make_imported_label_source_slots(
    targetargs: &[OpRef],
    exported_state: &ExportedState,
    ctx: &OptContext,
) -> Vec<OpRef> {
    exported_state
        .virtual_state
        .make_inputarg_source_slots(targetargs, ctx)
}

/// unroll.py: pick_virtual_state(my_vs, label_vs, target_tokens)
///
/// Given the current virtual state and available target tokens,
/// find a compatible target to jump to. Returns the target index
/// or None if no match.
/// RPython unroll.py: import_state + _generate_virtual.
///
// ── RPython-parity helper functions for 2-phase preamble peeling ──

/// Derive ImportedVirtual entries from ExportedState's VirtualState.
/// Virtual structure is obtained from the VirtualState snapshot.
fn build_imported_virtuals_from_state(
    state: &ExportedState,
) -> Vec<crate::optimizeopt::optimizer::ImportedVirtual> {
    use crate::optimizeopt::virtualstate::VirtualStateInfo;

    /// virtualstate.py:158-165 AbstractVirtualStructStateInfo.generate_guards
    /// parity: walk `fielddescrs` in parent-local slot order, looking up
    /// the matching field state via descr.get_index() (= field_idx in pyre).
    fn ordered_fields(
        field_descrs: &[majit_ir::DescrRef],
        fields: &[(u32, std::rc::Rc<VirtualStateInfo>)],
    ) -> Vec<(majit_ir::DescrRef, VirtualStateInfo)> {
        field_descrs
            .iter()
            .filter_map(|field_descr| {
                let field_idx = field_descr
                    .as_field_descr()
                    .map(|fd| fd.index_in_parent() as u32)?;
                fields
                    .iter()
                    .find(|(idx, _)| *idx == field_idx)
                    .map(|(_, field_value)| (field_descr.clone(), (**field_value).clone()))
            })
            .collect()
    }

    let mut result = Vec::new();
    for (idx, info) in state.virtual_state.state.iter().enumerate() {
        match &**info {
            VirtualStateInfo::Virtual {
                descr,
                known_class,
                fields,
                field_descrs,
                ..
            } => {
                result.push(crate::optimizeopt::optimizer::ImportedVirtual {
                    inputarg_index: idx,
                    size_descr: descr.clone(),
                    kind: crate::optimizeopt::optimizer::ImportedVirtualKind::Instance {
                        known_class: *known_class,
                    },
                    fields: ordered_fields(field_descrs, fields),
                    head_load_descr_index: None,
                });
            }
            VirtualStateInfo::VStruct {
                descr,
                fields,
                field_descrs,
            } => {
                result.push(crate::optimizeopt::optimizer::ImportedVirtual {
                    inputarg_index: idx,
                    size_descr: descr.clone(),
                    kind: crate::optimizeopt::optimizer::ImportedVirtualKind::Struct,
                    fields: ordered_fields(field_descrs, fields),
                    head_load_descr_index: None,
                });
            }
            _ => {}
        }
    }
    result
}

/// compile.py:310-338: [preamble_no_jump] + Label(label_args) + [body_with_jump]
fn assemble_peeled_trace(
    p1_ops: &[Op],
    p2_ops: &[Op],
    label_args: &[OpRef],
    label_source_slots: &[OpRef],
    start_label_args: &[OpRef],
    extra_label_args: &[OpRef],
    body_num_inputs: usize,
    jump_to_self: bool,
    imported_short_aliases: &[crate::optimizeopt::ImportedShortAlias],
    imported_short_sources: &[crate::optimizeopt::ImportedShortSource],
    constants: &std::collections::HashMap<u32, i64>,
    start_label_descr: Option<DescrRef>,
    loop_label_descr: Option<DescrRef>,
) -> Vec<Op> {
    assemble_peeled_trace_with_jump_args(
        p1_ops,
        p2_ops,
        label_args,
        label_source_slots,
        start_label_args,
        extra_label_args,
        extra_label_args,
        body_num_inputs,
        jump_to_self,
        imported_short_aliases,
        imported_short_sources,
        constants,
        start_label_descr,
        loop_label_descr,
        &[], // no p1_end_args for simple assembly
        &[], // no inputarg_types for simple assembly
    )
}

fn assemble_peeled_trace_with_jump_args(
    p1_ops: &[Op],
    p2_ops: &[Op],
    label_args: &[OpRef],
    label_source_slots: &[OpRef],
    start_label_args: &[OpRef],
    extra_label_args: &[OpRef],
    extra_jump_args: &[OpRef],
    body_num_inputs: usize,
    jump_to_self: bool,
    imported_short_aliases: &[crate::optimizeopt::ImportedShortAlias],
    imported_short_sources: &[crate::optimizeopt::ImportedShortSource],
    constants: &std::collections::HashMap<u32, i64>,
    start_label_descr: Option<DescrRef>,
    loop_label_descr: Option<DescrRef>,
    p1_end_args: &[OpRef],
    inputarg_types: &[Type],
) -> Vec<Op> {
    let mut result =
        Vec::with_capacity(p1_ops.len() + p2_ops.len() + 1 + imported_short_aliases.len());
    let mut filtered_extra_label_args = Vec::new();
    let mut filtered_extra_jump_args = Vec::new();
    // RPython Box parity (compile.py:327, shortpreamble.py:436-439):
    //
    //   loop.operations = ([start_label] + preamble_ops + loop_info.extra_same_as +
    //                      loop_info.extra_before_label + [loop_info.label_op] + loop_ops)
    //
    //   op = preamble_op.op.get_box_replacement()
    //   if preamble_op.invented_name:
    //       self.extra_same_as.append(op)
    //   self.used_boxes.append(op)
    //   self.short_preamble_jump.append(preamble_op.preamble_op)
    //
    // `used_boxes` is appended to `label_op.arglist()` in its entirety, with
    // NO deduplication against the base label args. RPython's Box identity
    // makes every used_box a distinct Python object; even two boxes that
    // happen to carry the same runtime value are separate slots in the
    // label. In majit's flat OpRef model we carry this literally — do not
    // drop entries just because their OpRef coincides with a base label
    // arg, because that silently corrupts the loop arity (the matching
    // `extra_jump_args[idx]` value that the JUMP was supposed to deliver
    // into that slot disappears along with it).
    //
    // The only entries that legitimately drop out here are constants:
    // those become immediates in the body and carry no live-in slot.
    for (idx, &label_arg) in extra_label_args.iter().enumerate() {
        let jump_arg = extra_jump_args.get(idx).copied().unwrap_or(label_arg);
        if is_trace_constant_ref(label_arg, constants) {
            continue;
        }
        filtered_extra_label_args.push(label_arg);
        filtered_extra_jump_args.push(jump_arg);
    }

    let mut next_free_pos = |mut next: u32| {
        next = next.max(body_num_inputs as u32);
        while constants.contains_key(&next) {
            next += 1;
        }
        next
    };

    // RPython Box.type parity: SameAs opcode must match the source Box's type.
    // In the flat OpRef model, inputargs don't have a defining op in `result`,
    // so consult `inputarg_types` first and only then fall back to the emitted op.
    let derive_same_as_opcode = |source: OpRef, result: &[Op]| -> OpCode {
        if (source.0 as usize) < body_num_inputs {
            if let Some(tp) = inputarg_types.get(source.0 as usize) {
                return OpCode::same_as_for_type(*tp);
            }
        }
        result
            .iter()
            .find(|op| op.pos == source)
            .map(|op| OpCode::same_as_for_type(op.opcode.result_type()))
            .unwrap_or(OpCode::SameAsI)
    };

    if let Some(start_label_descr) = start_label_descr {
        let mut start_label = Op::new(OpCode::Label, start_label_args);
        start_label.pos = OpRef::NONE;
        start_label.descr = Some(start_label_descr);
        result.push(start_label);
    }

    // Preamble: everything except Jump
    for op in p1_ops {
        if op.opcode == OpCode::Jump {
            break;
        }
        result.push(op.clone());
    }

    // max_pos must account for ALL p1_ops positions, including SameAs
    // ops AFTER the Jump that weren't copied into result. These positions
    // are referenced by the body Label args and must not be reused.
    let result_max = result
        .iter()
        .map(|op| op.pos.0)
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(body_num_inputs as u32);
    let p1_all_max = p1_ops
        .iter()
        .map(|op| op.pos.0)
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(0);
    // Account for label_source_slots positions to prevent alias
    // position collisions. label_source_slots carry Phase 2 inputarg
    // identities that may be higher than preamble op positions.
    let source_slot_max = label_source_slots
        .iter()
        .map(|s| s.0)
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(0);
    let mut max_pos = result_max.max(p1_all_max).max(source_slot_max);
    max_pos = next_free_pos(max_pos.saturating_add(1));
    // Step 6 of the Box identity plan: the per-iteration alias /
    // rescue / dedup remap maps used to live here, splitting the
    // assembly's collision-compensation logic into three namespace
    // buckets. After Commit D1/D2 (Phase 2 disjoint OpRef ranges) the
    // body-use-before-def fall-throughs and label-arg duplicates that
    // populated `label_rescue_remap` / `label_dedup_remap` are empty
    // across the entire test suite. Both maps and the lookups that
    // consumed them have been removed.
    //
    // `imported_short_aliases` still drives SameAs emission for the
    // imported short-preamble alias result OpRefs (the alias is the
    // alias_result that body ops will reference; without the SameAs
    // op the preamble does not produce that OpRef). The previous code
    // recorded a fresh-position alias_remap entry; with disjoint
    // OpRef ranges the alias result already lives at a unique
    // position, so the SameAs op is emitted at the alias's existing
    // result OpRef directly.
    for alias in imported_short_aliases {
        let mut op = Op::new(alias.same_as_opcode, &[alias.same_as_source]);
        op.pos = alias.result;
        result.push(op);
    }

    // RPython compile.py:327 extra_same_as parity for non-invented imports.
    // shortpreamble.py PureOp/HeapOp.produce_op emits a SameAs (or the
    // original op) before the loop label so the body's reference to the
    // imported result OpRef has a defining op. The `imported_short_aliases`
    // loop above only handles compound alternates (`invented_name=true`);
    // single non-invented imported short ops (a Pure GetfieldGcPureI on a
    // non-constant object, a HeapField, …) record their `(result, source)`
    // mapping in `imported_short_sources` instead and never reach the alias
    // emission. Without an extra SameAs they leave Phase 2 body ops with an
    // undefined fresh OpRef (e.g. v56 = i.intval) that the body
    // use-before-def loop later promotes into the loop label, producing
    // an undefined Cranelift Variable on the head's fall-through into the
    // body label.
    //
    // Mirror RPython by emitting `SameAs(source)` at `pos=result` for every
    // non-invented entry whose source is a real (preamble-defined) OpRef.
    // Constant sources are skipped: the body sees them via the constants
    // pool, not via SSA. Entries already produced by the alias loop or by
    // the preamble itself are skipped to avoid duplicate definitions.
    {
        let mut already_defined: std::collections::HashSet<OpRef> =
            imported_short_aliases.iter().map(|a| a.result).collect();
        for op in &result {
            if !op.pos.is_none() && op.opcode != OpCode::Jump && op.result_type() != Type::Void {
                already_defined.insert(op.pos);
            }
        }
        for entry in imported_short_sources.iter() {
            if entry.result.is_none() || entry.result.is_constant() {
                continue;
            }
            if already_defined.contains(&entry.result) {
                continue;
            }
            if entry.source.is_none() || entry.source.is_constant() {
                continue;
            }
            // Skip when source already equals the result (no-op SameAs).
            if entry.source == entry.result {
                already_defined.insert(entry.result);
                continue;
            }
            let same_as_opcode = derive_same_as_opcode(entry.source, &result);
            let mut op = Op::new(same_as_opcode, &[entry.source]);
            op.pos = entry.result;
            result.push(op);
            already_defined.insert(entry.result);
        }
    }

    // Label position
    let label_pos = next_free_pos(max_pos);
    let mut full_label_args: Vec<OpRef> = label_args
        .iter()
        .copied()
        .filter(|arg| !is_trace_constant_ref(*arg, constants))
        .collect();

    // Collect preamble-defined OpRefs BEFORE adding extra label args,
    // so we can filter out virtual remnants (removed New ops).
    let preamble_defs: std::collections::HashSet<OpRef> = {
        let mut s: std::collections::HashSet<OpRef> =
            (0..body_num_inputs).map(|i| OpRef(i as u32)).collect();
        for op in &result {
            if !op.pos.is_none() && op.opcode != OpCode::Jump && op.result_type() != Type::Void {
                s.insert(op.pos);
            }
        }
        s
    };

    // Append non-constant extra label args. The caller (unroll pass)
    // determines which extra values the loop header needs; virtual
    // remnants have already been filtered by the caller.
    //
    // For extra args not defined in the preamble (Phase 2-only OpRefs),
    // emit a SameAs op mapping them from their preamble source so the
    // Cranelift fall-through can resolve them.
    let short_source_map: HashMap<OpRef, OpRef> = imported_short_sources
        .iter()
        .map(|s| (s.result, s.source))
        .collect();
    // Advance max_pos past every position already in use before allocating
    // fresh SameAs positions. In RPython, Box identity prevents collisions;
    // in the flat OpRef model, the assembly-allocated SameAs position must
    // not overlap base label_args, filtered_extra_label_args, p2_ops
    // positions/args/fail_args, or imported_short_sources.result fresh
    // slots — each of those may be referenced by the body. A single
    // colliding OpRef aliases two distinct values and corrupts the loop.
    for &la in &full_label_args {
        if is_trace_runtime_ref(la, constants) {
            max_pos = max_pos.max(la.0.saturating_add(1));
        }
    }
    for &la in &filtered_extra_label_args {
        if is_trace_runtime_ref(la, constants) {
            max_pos = max_pos.max(la.0.saturating_add(1));
        }
    }
    for op in p2_ops.iter() {
        if is_trace_runtime_ref(op.pos, constants) {
            max_pos = max_pos.max(op.pos.0.saturating_add(1));
        }
        for &arg in op.args.iter() {
            if is_trace_runtime_ref(arg, constants) {
                max_pos = max_pos.max(arg.0.saturating_add(1));
            }
        }
        if let Some(ref fa) = op.fail_args {
            for &arg in fa.iter() {
                if is_trace_runtime_ref(arg, constants) {
                    max_pos = max_pos.max(arg.0.saturating_add(1));
                }
            }
        }
    }
    for entry in imported_short_sources.iter() {
        if is_trace_runtime_ref(entry.result, constants) {
            max_pos = max_pos.max(entry.result.0.saturating_add(1));
        }
    }
    max_pos = next_free_pos(max_pos);

    // RPython compile.py:327 extra_same_as parity. Every extra label arg
    // must be defined before the label — either by a preamble op or by
    // an explicit SameAs bridge between preamble and label. In RPython
    // this is the `extra_same_as` list populated by
    // `ShortPreambleBuilder.add_preamble_op` (shortpreamble.py:436-439)
    // for every `invented_name` preamble op.
    //
    // majit's `imported_short_aliases` already emits the SameAs ops for
    // invented_name entries at the loop above (`for alias in
    // imported_short_aliases`). For any other `filtered_extra_label_args`
    // entry whose OpRef is not produced by the preamble — e.g. the fresh
    // value OpRef allocated by `import_short_preamble_ops` for a
    // non-invented HeapField import — emit a SameAs here so the
    // fall-through into the label sees it defined. Without this bridge,
    // Cranelift's first iteration would read an undefined OpRef for the
    // extra label slot.
    //
    // The source for the SameAs comes from `imported_short_sources` which
    // records `(result, source)` for every imported short preamble op.
    // If an extra label arg has no known source (synthetic test setups
    // or base-label-arg collisions), it is appended as-is and the caller
    // is responsible for ensuring it is defined elsewhere.
    let extra_label_start_idx = full_label_args.len();
    {
        let alias_results: std::collections::HashSet<OpRef> = imported_short_aliases
            .iter()
            .map(|alias| alias.result)
            .collect();
        for &la in filtered_extra_label_args.iter() {
            if preamble_defs.contains(&la) || alias_results.contains(&la) {
                continue;
            }
            let Some(source) = imported_short_sources
                .iter()
                .find(|entry| entry.result == la)
                .map(|entry| entry.source)
            else {
                continue;
            };
            let same_as_opcode = derive_same_as_opcode(source, &result);
            let mut op = Op::new(same_as_opcode, &[source]);
            op.pos = la;
            result.push(op);
        }
    }
    full_label_args.extend(filtered_extra_label_args.iter().copied());

    // RPython compile.py parity: after the loop label, only the loop-header
    // contract is live. When `splice_redirected_tail` glues a redirected
    // tail onto the body, the spliced output may mention OpRefs whose
    // defining op was removed (the section between the splice point and
    // the original Jump). Carry such "body use-before-def" references
    // through the label so the assembled body doesn't contain dangling
    // references.
    //
    // Step 6 of the Box identity plan: the previous code populated
    // `label_rescue_remap` (now deleted) with synthetic SameAs ops to
    // bridge these references. With Commit D1/D2 the carried OpRef is
    // already in a unique slot, so adding it directly to
    // `full_label_args` is sufficient — the JUMP's mapped_base_args
    // path will pick up the corresponding fresh value on the next
    // iteration.
    let mut carried_source_slots: std::collections::HashSet<OpRef> =
        if !label_source_slots.is_empty() && label_source_slots.len() == label_args.len() {
            label_source_slots.iter().copied().collect()
        } else {
            (0..label_args.len()).map(|i| OpRef(i as u32)).collect()
        };
    carried_source_slots.extend(filtered_extra_jump_args.iter().copied());
    // `label_set` tracks which OpRefs are already carried by the label so
    // that the body-use-before-def pass doesn't add the same OpRef twice
    // (which would inflate label arity without adding new information).
    // RPython's Box identity makes this a correctness filter: two
    // references to the same Box collapse into one live-in slot. This
    // is NOT the Issue 1 dedup — which drops distinct Boxes that happen
    // to share an OpRef — it is RPython parity: the same Box appears
    // once in the label arglist.
    let mut label_set: std::collections::HashSet<OpRef> = full_label_args.iter().copied().collect();
    {
        let mut seen_body_defs = std::collections::HashSet::new();
        for op in p2_ops {
            let all_refs = op
                .args
                .iter()
                .chain(op.fail_args.as_ref().into_iter().flat_map(|fa| fa.iter()));
            for &arg in all_refs {
                if !is_trace_runtime_ref(arg, constants) {
                    continue; // skip NONE and constants
                }
                if label_set.contains(&arg)
                    || carried_source_slots.contains(&arg)
                    || seen_body_defs.contains(&arg)
                {
                    continue;
                }
                full_label_args.push(arg);
                label_set.insert(arg);
            }
            if op.result_type() != Type::Void && !op.pos.is_none() {
                seen_body_defs.insert(op.pos);
            }
        }
    }

    // RPython Box identity parity: virtual field values remapped by
    // import_state's remap_field_ref need SameAs ops to connect fresh
    // label positions to their original Phase 1 sources. Also cover base
    // label_args whose forwarding chain leads to a Phase 1 end_arg.
    // In RPython, the JUMP's Box IS the label's Box — no mapping needed.
    // flat OpRef emits explicit SameAs ops as an equivalent bridge.
    for (i, &la) in full_label_args.iter().enumerate() {
        if la.is_none() || is_trace_constant_ref(la, constants) {
            continue;
        }
        if preamble_defs.contains(&la) {
            continue; // already defined by preamble
        }
        // Step 6 of the Box identity plan: the `imported_field_remap`
        // priority-1 lookup is gone (the map is dead after Commit D1/D2 —
        // Phase 2's disjoint OpRef range means virtual fields keep their
        // Phase 1 emitted positions and never need a fresh-position
        // SameAs to reach them). Only the base label-arg → Phase 1 JUMP
        // arg path remains.
        let source_opt = if i < p1_end_args.len() {
            let jump_arg = p1_end_args[i];
            if !jump_arg.is_none() && preamble_defs.contains(&jump_arg) && jump_arg != la {
                Some(jump_arg)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(source) = source_opt {
            if preamble_defs.contains(&source) {
                let same_as_opcode = derive_same_as_opcode(source, &result);
                let mut op = Op::new(same_as_opcode, &[source]);
                op.pos = la;
                result.push(op);
            }
        }
    }

    let mut label_op = Op::new(OpCode::Label, &full_label_args);
    label_op.pos = OpRef(label_pos);
    label_op.descr = loop_label_descr;
    result.push(label_op);

    // Body: 2-pass remap (inputarg refs -> label args, body results -> fresh boxes)
    let max_label_arg_pos = full_label_args
        .iter()
        .map(|a| a.0)
        .max()
        .unwrap_or(label_pos);
    // Fresh body positions must be higher than ALL existing positions:
    // label, label args, AND Phase 2 op positions (which may be higher
    // than label positions due to Phase 2 remap or redirect ops).
    let max_p2_pos = p2_ops
        .iter()
        .map(|op| op.pos.0)
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(0);
    let mut next_body_pos = next_free_pos(
        label_pos
            .max(max_label_arg_pos)
            .max(max_p2_pos)
            .saturating_add(1),
    );
    let mut input_remap: HashMap<OpRef, OpRef> = HashMap::new();
    let mut body_result_remap: HashMap<OpRef, OpRef> = HashMap::new();
    let visible_before_label: std::collections::HashSet<OpRef> = full_label_args
        .iter()
        .copied()
        .chain(preamble_defs.iter().copied())
        .collect();

    if !label_source_slots.is_empty() && label_source_slots.len() == label_args.len() {
        for (&source_slot, &la) in label_source_slots.iter().zip(label_args.iter()) {
            if !source_slot.is_none() {
                input_remap.insert(source_slot, la);
            }
        }
    } else {
        for (i, &la) in label_args.iter().enumerate() {
            if (i as u32) < body_num_inputs as u32 {
                input_remap.insert(OpRef(i as u32), la);
            }
        }
    }
    for (i, &source_slot) in filtered_extra_label_args.iter().enumerate() {
        if !source_slot.is_none() {
            if let Some(&extended_label_arg) = full_label_args.get(extra_label_start_idx + i) {
                input_remap.insert(source_slot, extended_label_arg);
            }
        }
    }
    for op in p2_ops.iter() {
        // Only map non-Void ops that actually produce a result.
        // Void ops (SetfieldGc, guards, Jump) don't define values at
        // their position — mapping them creates phantom OpRefs.
        if op.pos.0 != u32::MAX && op.result_type() != Type::Void {
            let fresh = OpRef(next_body_pos);
            next_body_pos = next_free_pos(next_body_pos.saturating_add(1));
            body_result_remap.insert(op.pos, fresh);
        }
    }

    let mut seen_body_defs = std::collections::HashSet::new();
    let mut label_scope_remap: HashMap<OpRef, OpRef> = HashMap::new();
    let mut current_inner_label_index: Option<usize> = None;
    let mut defs_since_inner_label: std::collections::HashSet<OpRef> =
        std::collections::HashSet::new();
    for (op_idx, op) in p2_ops.iter().enumerate() {
        let mut new_op = op.clone();
        let mut original_args = op.args.clone();
        if let Some(&mapped_pos) = body_result_remap.get(&op.pos) {
            new_op.pos = mapped_pos;
        }
        // Step 6 of the Box identity plan: the 6-way priority remap was
        // a band-aid for the flat OpRef model where multiple namespaces
        // (preamble emit, label dedup, alias import, body redefine)
        // could collide on the same OpRef. With Commit D1/D2's disjoint
        // Phase 2 namespace, the 3 reactive maps (alias / rescue / dedup)
        // are dead and have been removed. The remaining 3-way priority
        // is: inner label scope > body redefine > input remap.
        let remap_body_arg = |arg: OpRef,
                              label_scope_remap: &HashMap<OpRef, OpRef>,
                              input_remap: &HashMap<OpRef, OpRef>,
                              body_result_remap: &HashMap<OpRef, OpRef>,
                              seen_body_defs: &std::collections::HashSet<OpRef>,
                              visible_before_label: &std::collections::HashSet<OpRef>|
         -> OpRef {
            // 1. Inner label scope (takes full priority)
            if let Some(&mapped) = label_scope_remap.get(&arg) {
                return mapped;
            }
            // 2. Body-redefined values take priority over input_remap
            if let Some(&mapped) = body_result_remap.get(&arg) {
                if seen_body_defs.contains(&arg) || !visible_before_label.contains(&arg) {
                    return mapped;
                }
            }
            // 3. input_remap — body inputarg → label arg
            if let Some(&mapped) = input_remap.get(&arg) {
                return mapped;
            }
            arg
        };
        for arg in &mut new_op.args {
            *arg = remap_body_arg(
                *arg,
                &label_scope_remap,
                &input_remap,
                &body_result_remap,
                &seen_body_defs,
                &visible_before_label,
            );
        }
        if new_op.opcode == OpCode::Label {
            let mut seen_after_label_defs = std::collections::HashSet::new();
            let mut extra_inner_sources = Vec::new();
            let mut extra_inner_set = std::collections::HashSet::new();
            let label_arg_set: std::collections::HashSet<OpRef> = original_args
                .iter()
                .copied()
                .filter(|arg| !arg.is_none())
                .collect();
            for later_op in p2_ops.iter().skip(op_idx + 1) {
                for arg in later_op
                    .args
                    .iter()
                    .copied()
                    .chain(later_op.fail_args.iter().flatten().copied())
                {
                    if arg.is_none()
                        || constants.contains_key(&arg.0)
                        || label_arg_set.contains(&arg)
                        || extra_inner_set.contains(&arg)
                        || seen_after_label_defs.contains(&arg)
                    {
                        continue;
                    }
                    let available_before_label = visible_before_label.contains(&arg)
                        || seen_body_defs.contains(&arg)
                        || short_source_map.contains_key(&arg)
                        || input_remap.contains_key(&arg);
                    if available_before_label {
                        extra_inner_set.insert(arg);
                        extra_inner_sources.push(arg);
                    }
                }
                if later_op.result_type() != Type::Void && !later_op.pos.is_none() {
                    seen_after_label_defs.insert(later_op.pos);
                }
            }
            // RPython Box parity: do not dedup the inner-label extension
            // against its existing args. Each source_arg collected from
            // the later body is a distinct RPython Box; adding it as a
            // separate slot matches `label_op.initarglist(label_op.getarglist()
            // + sb.used_boxes)` (unroll.py:300) where RPython never filters
            // by value or by box coincidence. The outer collection above
            // already dedups by box identity (`extra_inner_set.contains`),
            // so each source_arg appears at most once — which is the
            // RPython-parity behavior for Box-keyed live-in sets.
            for &source_arg in &extra_inner_sources {
                let mapped_arg = remap_body_arg(
                    source_arg,
                    &label_scope_remap,
                    &input_remap,
                    &body_result_remap,
                    &seen_body_defs,
                    &visible_before_label,
                );
                new_op.args.push(mapped_arg);
                original_args.push(source_arg);
            }
            label_scope_remap.clear();
            for (&source_arg, &mapped_arg) in original_args.iter().zip(new_op.args.iter()) {
                if !source_arg.is_none() {
                    label_scope_remap.insert(source_arg, mapped_arg);
                    if let Some(imported_result) = imported_short_sources
                        .iter()
                        .find(|entry| entry.source == source_arg)
                        .map(|entry| entry.result)
                    {
                        label_scope_remap.insert(imported_result, mapped_arg);
                    }
                }
            }
        }
        if new_op.opcode == OpCode::Jump {
            let mapped_base_args: Vec<OpRef> = if !jump_to_self {
                let start_remap: HashMap<OpRef, OpRef> = start_label_args
                    .iter()
                    .enumerate()
                    .map(|(i, &arg)| (OpRef(i as u32), arg))
                    .collect();
                original_args
                    .iter()
                    .map(|arg| start_remap.get(arg).copied().unwrap_or(*arg))
                    .collect()
            } else {
                new_op.args.iter().copied().collect()
            };
            let target_label_args: Vec<OpRef> = current_inner_label_index
                .and_then(|label_idx| result.get(label_idx).map(|op| op.args.to_vec()))
                .unwrap_or_else(|| full_label_args.clone());
            let target_base_len = if current_inner_label_index.is_some() {
                original_args.len()
            } else {
                label_args.len()
            };
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[jit] assemble_jump: inner_label={:?} original_args={:?} mapped_base_args={:?} label_args={:?} label_source_slots={:?} filtered_extra_jump_args={:?}",
                    current_inner_label_index,
                    original_args,
                    mapped_base_args,
                    label_args,
                    label_source_slots,
                    filtered_extra_jump_args,
                );
            }
            let mut jump_args = mapped_base_args;
            if !jump_to_self {
                jump_args.truncate(start_label_args.len());
            }
            if jump_to_self {
                // RPython compile.py:334: assert jump.numargs() == label.numargs().
                // Truncate excess JUMP args (from forced virtuals in
                // jump_to_existing_trace) to match the LABEL arity.
                if jump_args.len() > target_label_args.len() {
                    jump_args.truncate(target_label_args.len());
                }
                // Pad if JUMP is shorter than the target label.
                while jump_args.len() < target_label_args.len() {
                    let extra_idx = jump_args.len().saturating_sub(target_base_len);
                    let extra_arg = if current_inner_label_index.is_some() {
                        target_label_args[jump_args.len()]
                    } else {
                        filtered_extra_jump_args
                            .get(extra_idx)
                            .copied()
                            .unwrap_or(target_label_args[jump_args.len()])
                    };
                    // Extra args from the label are assembly-allocated positions.
                    // They must NOT be remapped through body-scoped maps.
                    let remapped = if label_set.contains(&extra_arg) {
                        extra_arg
                    } else {
                        input_remap
                            .get(&extra_arg)
                            .copied()
                            .or_else(|| {
                                body_result_remap
                                    .get(&extra_arg)
                                    .copied()
                                    .and_then(|mapped| {
                                        if seen_body_defs.contains(&extra_arg)
                                            || !visible_before_label.contains(&extra_arg)
                                        {
                                            Some(mapped)
                                        } else {
                                            None
                                        }
                                    })
                            })
                            .unwrap_or(extra_arg)
                    };
                    jump_args.push(remapped);
                }
            }
            new_op.args = jump_args.into();
        }
        // RPython resume.py parity: fail_args are rebuilt from the
        // snapshot by store_final_boxes_in_guard. Snapshot-derived
        // values that reference label args must NOT be remapped to
        // body results, because the snapshot captures the state at
        // the guard point (or parent capture point), not the body's
        // final state.  Use body_result_remap only for values that
        // are body-defined AND not visible before the label.
        // Single-step priority lookup for fail_args (no chain iteration).
        // fail_args capture guard-point state — body_result_remap only for
        // values that are body-defined AND not visible before the label.
        if let Some(ref mut fa) = new_op.fail_args {
            for a in fa.iter_mut() {
                *a = label_scope_remap
                    .get(a)
                    .copied()
                    .or_else(|| input_remap.get(a).copied())
                    .or_else(|| {
                        body_result_remap.get(a).copied().filter(|_| {
                            seen_body_defs.contains(a) && !visible_before_label.contains(a)
                        })
                    })
                    .unwrap_or(*a);
            }
        }
        if let Some(label_idx) = current_inner_label_index {
            let mut extra_live_args = Vec::new();
            let label_args = &result[label_idx].args;
            for arg in new_op
                .args
                .iter()
                .copied()
                .chain(new_op.fail_args.iter().flatten().copied())
            {
                if arg.is_none()
                    || constants.contains_key(&arg.0)
                    || defs_since_inner_label.contains(&arg)
                    || label_args.contains(&arg)
                    || extra_live_args.contains(&arg)
                {
                    continue;
                }
                extra_live_args.push(arg);
            }
            if !extra_live_args.is_empty() {
                let existing: std::collections::HashSet<OpRef> =
                    result[label_idx].args.iter().copied().collect();
                result[label_idx].args.extend(
                    extra_live_args
                        .into_iter()
                        .filter(|arg| !existing.contains(arg)),
                );
            }
        }
        result.push(new_op);
        if op.opcode == OpCode::Label {
            current_inner_label_index = Some(result.len() - 1);
            defs_since_inner_label.clear();
        }
        if op.result_type() != Type::Void && !op.pos.is_none() {
            seen_body_defs.insert(op.pos);
            defs_since_inner_label.insert(result.last().unwrap().pos);
        }
    }

    result
}

fn splice_redirected_tail(body_ops: &[Op], redirected_tail_ops: &[Op]) -> Vec<Op> {
    let mut result = Vec::with_capacity(body_ops.len() + redirected_tail_ops.len());
    let split_idx = body_ops
        .iter()
        .rposition(|op| op.opcode == OpCode::Jump)
        .unwrap_or(body_ops.len());
    result.extend_from_slice(&body_ops[..split_idx]);
    result.extend_from_slice(redirected_tail_ops);
    result
}

fn replace_terminal_jump(body_ops: &[Op], jump_op: Op) -> Vec<Op> {
    let mut result = Vec::with_capacity(body_ops.len() + 1);
    let split_idx = body_ops
        .iter()
        .rposition(|op| op.opcode == OpCode::Jump)
        .unwrap_or(body_ops.len());
    result.extend_from_slice(&body_ops[..split_idx]);
    result.push(jump_op);
    result
}

pub fn pick_virtual_state(
    my_vs: &crate::optimizeopt::virtualstate::VirtualState,
    target_states: &[crate::optimizeopt::virtualstate::VirtualState],
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
    use crate::optimizeopt::optimizer::Optimizer;

    /// Assign sequential positions to ops starting from `base`.
    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(base + i as u32);
        }
    }

    fn run_unroll_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptUnroll::new()));
        // See `run_heap_opt` in heap.rs for the 1024-slot Ref seed
        // rationale — the preamble exporter needs an intrinsic type
        // for every renamed inputarg, which production derives from
        // the recorder's trace_inputarg_types.
        opt.trace_inputarg_types = vec![majit_ir::Type::Ref; 1024];
        opt.optimize_with_constants_and_inputs(ops, &mut std::collections::HashMap::new(), 1024)
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
        let preamble_target = TargetToken::new_preamble(7);

        let result = UnrollOptimizer::jump_to_preamble(&body_ops, &preamble_target);
        assert_eq!(result[1].opcode, OpCode::Jump);
        assert_eq!(result[1].args.as_slice(), &[OpRef(0), OpRef(2), OpRef(50)]);
        assert_eq!(
            result[1].descr.as_ref().map(|descr| descr.repr()),
            Some("LoopTargetDescr(start:7)".to_string())
        );
    }

    #[test]
    fn test_replace_terminal_jump_appends_when_body_prefix_has_no_jump() {
        let body_ops = vec![{
            let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
            op.pos = OpRef(2);
            op
        }];
        let mut jump = Op::new(OpCode::Jump, &[OpRef(2)]);
        jump.descr = Some(TargetToken::new_preamble(7).as_jump_target_descr());

        let result = replace_terminal_jump(&body_ops, jump);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::Jump);
        assert_eq!(result[1].args.as_slice(), &[OpRef(2)]);
        assert_eq!(
            result[1].descr.as_ref().map(|descr| descr.repr()),
            Some("LoopTargetDescr(start:7)".to_string())
        );
    }

    #[test]
    fn test_ensure_preamble_target_token_inserts_start_descr_first() {
        let mut unroll = UnrollOptimizer::new();
        let regular = TargetToken {
            token_id: 3,
            is_preamble_target: false,
            virtual_state: Some(crate::optimizeopt::virtualstate::VirtualState::new(vec![
                crate::optimizeopt::virtualstate::VirtualStateInfo::Unknown,
            ])),
            short_preamble: None,
            short_preamble_producer: None,
            jump_target_descr: Arc::new(LoopTargetDescr::new(3, false)),
        };
        unroll.target_tokens.push(regular);

        unroll.ensure_preamble_target_token();

        assert_eq!(unroll.target_tokens.len(), 2);
        assert!(unroll.target_tokens[0].is_preamble_target);
        assert!(unroll.target_tokens[0].virtual_state.is_none());
        assert_eq!(unroll.target_tokens[1].token_id, 3);
    }

    #[test]
    fn test_pick_virtual_state_uses_target_generalization_direction() {
        let my_vs = crate::optimizeopt::virtualstate::VirtualState::new(vec![
            crate::optimizeopt::virtualstate::VirtualStateInfo::NonNull,
        ]);
        let target_states = vec![
            crate::optimizeopt::virtualstate::VirtualState::new(vec![
                crate::optimizeopt::virtualstate::VirtualStateInfo::Unknown,
            ]),
            crate::optimizeopt::virtualstate::VirtualState::new(vec![
                crate::optimizeopt::virtualstate::VirtualStateInfo::KnownClass {
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
        opt.trace_inputarg_types = vec![majit_ir::Type::Ref; 1024];
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

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
        // IntAdd operates on Int-typed inputs — seed the inner phase1/2
        // optimizers' trace_inputarg_types via UnrollOptimizer so the
        // intbounds pass sees Int on OpRef(0), OpRef(1).
        unroll_opt.trace_inputarg_types = vec![majit_ir::Type::Int; 2];
        // Use optimize_trace_with_constants_and_inputs to properly set
        // num_inputs so input args don't collide with op positions.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        assign_positions(&mut ops, 2);
        let mut constants = std::collections::HashMap::new();
        let (result, _) =
            unroll_opt.optimize_trace_with_constants_and_inputs(&ops, &mut constants, 2);
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
    fn test_exported_state_reimports_widened_intbounds() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        use crate::optimizeopt::intutils::IntBound;

        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(4, 0);
        let mut exported_bounds = std::collections::HashMap::new();
        exported_bounds.insert(OpRef(21), IntBound::bounded(10, 20));

        let exported = export_state(
            &[OpRef(21)],
            &[],
            &mut optimizer,
            &mut ctx,
            Some(&exported_bounds),
        );

        assert_eq!(
            exported.exported_infos[&OpRef(21)]
                .int_bound
                .as_ref()
                .map(|b| (b.lower, b.upper)),
            Some((10, 20))
        );

        let mut ctx2 = crate::optimizeopt::OptContext::with_num_inputs(4, 1);
        let label_args = import_state(&[OpRef(0)], &exported, &mut optimizer, &mut ctx2);
        assert_eq!(label_args, vec![OpRef(21)]);
        // unroll.py:93-96: IntBound IS imported with widen() and stored
        // directly on the box's _forwarded slot via setintbound.
        // widen() relaxes bounds: lower < MININT/2 → MININT, upper > MAXINT/2 → MAXINT.
        // For [10, 20], both are within MININT/2..MAXINT/2 so widen() preserves them.
        let imported_bound = ctx2.getintbound(OpRef(21));
        assert_eq!((imported_bound.lower, imported_bound.upper), (10, 20));
    }

    #[test]
    fn test_export_state_uses_forced_end_args_snapshot() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(4, 1);
        ctx.preamble_end_args = Some(vec![OpRef(21)]);

        let exported = export_state(&[OpRef(0)], &[], &mut optimizer, &mut ctx, None);

        assert_eq!(exported.end_args, vec![OpRef(21)]);
    }

    #[test]
    fn test_exported_state_reimports_short_heap_field_facts() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(4, 0);
        let parent = majit_ir::make_size_descr_full(0xFFFF_1000, 16, 1);
        let field_descr: majit_ir::DescrRef = std::sync::Arc::new(
            majit_ir::SimpleFieldDescr::new(0, 0, 8, majit_ir::Type::Int, false)
                .with_signed(true)
                .with_parent_descr(parent.clone(), 0),
        );
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op =
                        Op::with_descr(OpCode::GetfieldGcI, &[OpRef(10)], field_descr.clone());
                    op.pos = OpRef(11);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Heap,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            });

        let exported = export_state(&[OpRef(10), OpRef(11)], &[], &mut optimizer, &mut ctx, None);
        assert_eq!(
            exported.exported_short_ops,
            vec![ExportedShortOp::HeapField {
                source: OpRef(11),
                object: ExportedShortArg::Slot(0),
                descr: field_descr,
                result_type: Type::Int,
                result: ExportedShortResult::Slot(1),
                invented_name: false,
                same_as_source: None,
            }]
        );

        let mut ctx2 = crate::optimizeopt::OptContext::with_num_inputs(4, 2);
        let label_args = import_state(&[OpRef(0), OpRef(1)], &exported, &mut optimizer, &mut ctx2);
        assert_eq!(label_args, vec![OpRef(10), OpRef(11)]);
        // RPython PreambleOp parity: PreambleOp stored in PtrInfo._fields.
        // No imported_short_fields for heap fields — PtrInfo is the single
        // source of truth, matching RPython's HeapOp.produce_op → opinfo.setfield.
        let obj_resolved = ctx2.get_box_replacement(OpRef(10));
        let pop = ctx2
            .get_ptr_info_mut(obj_resolved)
            .and_then(|info| info.take_preamble_field(0));
        assert!(pop.is_some(), "PreambleOp must be in PtrInfo._fields");
        let pop = pop.unwrap();
        assert_eq!(pop.op, OpRef(11)); // Phase 1 source
        assert_ne!(pop.resolved, OpRef(10)); // fresh, not label_arg
        assert_ne!(pop.resolved, OpRef(11));
    }

    #[test]
    fn test_exported_state_reimports_short_heap_ref_field_facts() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(4, 0);
        let parent = majit_ir::make_size_descr_full(0xFFFF_1001, 24, 2);
        let field_descr: majit_ir::DescrRef = std::sync::Arc::new(
            majit_ir::SimpleFieldDescr::new(56, 8, 8, majit_ir::Type::Ref, false)
                .with_parent_descr(parent.clone(), 0),
        );
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op =
                        Op::with_descr(OpCode::GetfieldGcR, &[OpRef(10)], field_descr.clone());
                    op.pos = OpRef(11);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Heap,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            });

        let exported = export_state(&[OpRef(10), OpRef(11)], &[], &mut optimizer, &mut ctx, None);
        assert_eq!(
            exported.exported_short_ops,
            vec![ExportedShortOp::HeapField {
                source: OpRef(11),
                object: ExportedShortArg::Slot(0),
                descr: field_descr,
                result_type: Type::Ref,
                result: ExportedShortResult::Slot(1),
                invented_name: false,
                same_as_source: None,
            }]
        );

        let mut ctx2 = crate::optimizeopt::OptContext::with_num_inputs(4, 2);
        let label_args = import_state(&[OpRef(0), OpRef(1)], &exported, &mut optimizer, &mut ctx2);
        assert_eq!(label_args, vec![OpRef(10), OpRef(11)]);
        // RPython PreambleOp parity: PreambleOp in PtrInfo._fields
        let obj_resolved = ctx2.get_box_replacement(OpRef(10));
        let pop = ctx2
            .get_ptr_info_mut(obj_resolved)
            .and_then(|info| info.take_preamble_field(0));
        assert!(pop.is_some());
        let pop = pop.unwrap();
        assert_eq!(pop.op, OpRef(11));
        assert_ne!(pop.resolved, OpRef(10));
        assert_ne!(pop.resolved, OpRef(11));
    }

    #[test]
    fn test_exported_short_heap_field_uses_preamble_result_as_source() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        use majit_ir::descr::make_field_descr_full;

        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(8, 0);
        let head_descr = make_field_descr_full(56, 0, 8, Type::Ref, false);
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op =
                        Op::with_descr(OpCode::GetfieldGcR, &[OpRef(10)], head_descr.clone());
                    op.pos = OpRef(26);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Heap,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            });

        let exported = export_state(&[OpRef(10)], &[], &mut optimizer, &mut ctx, None);
        assert_eq!(
            exported.exported_short_ops,
            vec![ExportedShortOp::HeapField {
                source: OpRef(26),
                object: ExportedShortArg::Slot(0),
                descr: head_descr,
                result_type: Type::Ref,
                result: ExportedShortResult::Temporary(0),
                invented_name: false,
                same_as_source: None,
            }]
        );
    }

    #[test]
    fn test_exported_state_reimports_short_pure_fact() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(6, 0);
        ctx.seed_constant(OpRef::from_const(10), Value::Int(7));
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntAdd, &[OpRef(12), OpRef::from_const(10)]);
                    op.pos = OpRef(11);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            });

        let exported = export_state(&[OpRef(12), OpRef(11)], &[], &mut optimizer, &mut ctx, None);
        assert_eq!(
            exported.exported_short_ops,
            vec![ExportedShortOp::Pure {
                source: OpRef(11),
                opcode: OpCode::IntAdd,
                descr: None,
                args: vec![
                    ExportedShortArg::Slot(0),
                    ExportedShortArg::Const {
                        source: OpRef::from_const(10),
                        value: Value::Int(7),
                    }
                ],
                result: ExportedShortResult::Slot(1),
                invented_name: false,
                same_as_source: None,
            }]
        );

        let mut ctx2 = crate::optimizeopt::OptContext::with_num_inputs(6, 2);
        let label_args = import_state(&[OpRef(0), OpRef(1)], &exported, &mut optimizer, &mut ctx2);
        assert_eq!(label_args, vec![OpRef(12), OpRef(11)]);
        assert_eq!(
            ctx2.get_constant(OpRef::from_const(10)),
            Some(&Value::Int(7))
        );
        assert_eq!(
            ctx2.imported_short_pure_ops,
            vec![crate::optimizeopt::ImportedShortPureOp {
                opcode: OpCode::IntAdd,
                descr: None,
                args: vec![
                    crate::optimizeopt::ImportedShortPureArg::OpRef(OpRef(12)),
                    crate::optimizeopt::ImportedShortPureArg::Const(
                        Value::Int(7),
                        OpRef::from_const(10)
                    ),
                ],
                result: OpRef(11),
            }]
        );
    }

    #[test]
    fn test_exported_state_reimports_short_call_pure_fact() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(8, 0);
        ctx.seed_constant(OpRef::from_const(10), Value::Int(0x1234));
        let call_descr = majit_ir::descr::make_call_descr_full(
            77,
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            8,
            majit_ir::EffectInfo::elidable(),
        );
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::CallI, &[OpRef::from_const(10), OpRef(12)]);
                    op.pos = OpRef(11);
                    op.descr = Some(call_descr.clone());
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            });

        let exported = export_state(&[OpRef(12), OpRef(11)], &[], &mut optimizer, &mut ctx, None);
        assert_eq!(
            exported.exported_short_ops,
            vec![ExportedShortOp::Pure {
                source: OpRef(11),
                opcode: OpCode::CallPureI,
                descr: Some(call_descr.clone()),
                args: vec![
                    ExportedShortArg::Const {
                        source: OpRef::from_const(10),
                        value: Value::Int(0x1234),
                    },
                    ExportedShortArg::Slot(0),
                ],
                result: ExportedShortResult::Slot(1),
                invented_name: false,
                same_as_source: None,
            }]
        );

        let mut ctx2 = crate::optimizeopt::OptContext::with_num_inputs(8, 2);
        let label_args = import_state(&[OpRef(0), OpRef(1)], &exported, &mut optimizer, &mut ctx2);
        assert_eq!(label_args, vec![OpRef(12), OpRef(11)]);
        assert_eq!(
            ctx2.get_constant(OpRef::from_const(10)),
            Some(&Value::Int(0x1234))
        );
        assert_eq!(
            ctx2.imported_short_pure_ops,
            vec![crate::optimizeopt::ImportedShortPureOp {
                opcode: OpCode::CallPureI,
                descr: Some(call_descr.clone()),
                args: vec![
                    crate::optimizeopt::ImportedShortPureArg::Const(
                        Value::Int(0x1234),
                        OpRef::from_const(10)
                    ),
                    crate::optimizeopt::ImportedShortPureArg::OpRef(OpRef(12)),
                ],
                result: OpRef(11),
            }]
        );
    }

    #[test]
    fn test_exported_state_reimports_short_pure_dependency_chain() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(10, 0);
        ctx.seed_constant(OpRef::from_const(10), Value::Int(7));
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntAdd, &[OpRef(12), OpRef::from_const(10)]);
                    op.pos = OpRef(11);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            });
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntMul, &[OpRef(11), OpRef(13)]);
                    op.pos = OpRef(14);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: Some(2),
                invented_name: false,
                same_as_source: None,
            });

        let exported = export_state(
            &[OpRef(12), OpRef(13), OpRef(14)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        assert_eq!(
            exported.exported_short_ops,
            vec![
                ExportedShortOp::Pure {
                    source: OpRef(11),
                    opcode: OpCode::IntAdd,
                    descr: None,
                    args: vec![
                        ExportedShortArg::Slot(0),
                        ExportedShortArg::Const {
                            source: OpRef::from_const(10),
                            value: Value::Int(7),
                        },
                    ],
                    result: ExportedShortResult::Temporary(0),
                    invented_name: false,
                    same_as_source: None,
                },
                ExportedShortOp::Pure {
                    source: OpRef(14),
                    opcode: OpCode::IntMul,
                    descr: None,
                    args: vec![ExportedShortArg::Produced(0), ExportedShortArg::Slot(1),],
                    result: ExportedShortResult::Slot(2),
                    invented_name: false,
                    same_as_source: None,
                },
            ]
        );

        // Use a large num_inputs so freshly allocated OpRef positions
        // don't collide with the OpRef values used in the exported state.
        let mut ctx2 = crate::optimizeopt::OptContext::with_num_inputs(10, 1024);
        let label_args = import_state(
            &[OpRef(0), OpRef(1), OpRef(2)],
            &exported,
            &mut optimizer,
            &mut ctx2,
        );
        assert_eq!(label_args, vec![OpRef(12), OpRef(13), OpRef(14)]);
        assert_eq!(
            ctx2.get_constant(OpRef::from_const(10)),
            Some(&Value::Int(7))
        );
        assert_eq!(ctx2.imported_short_pure_ops.len(), 2);
        // The temporary result is allocated by ctx.alloc_op_position();
        // its exact value depends on num_inputs, so read it dynamically.
        let temp_result = ctx2.imported_short_pure_ops[0].result;
        assert_eq!(
            ctx2.imported_short_pure_ops,
            vec![
                crate::optimizeopt::ImportedShortPureOp {
                    opcode: OpCode::IntAdd,
                    descr: None,
                    args: vec![
                        crate::optimizeopt::ImportedShortPureArg::OpRef(OpRef(12)),
                        crate::optimizeopt::ImportedShortPureArg::Const(
                            Value::Int(7),
                            OpRef::from_const(10)
                        ),
                    ],
                    result: temp_result,
                },
                crate::optimizeopt::ImportedShortPureOp {
                    opcode: OpCode::IntMul,
                    descr: None,
                    args: vec![
                        crate::optimizeopt::ImportedShortPureArg::OpRef(temp_result),
                        crate::optimizeopt::ImportedShortPureArg::OpRef(OpRef(13)),
                    ],
                    result: OpRef(14),
                },
            ]
        );
    }

    #[test]
    fn test_import_state_reimports_short_ref_constant_identity() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(8, 0);
        let ptr = GcRef(0x1234_5678);
        let field_descr = majit_ir::descr::make_field_descr_full(88, 0, 8, Type::Int, false);
        ctx.seed_constant(OpRef::from_const(23), Value::Ref(ptr));
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::with_descr(
                        OpCode::GetfieldGcPureI,
                        &[OpRef::from_const(23)],
                        field_descr.clone(),
                    );
                    op.pos = OpRef(11);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            });

        let exported = export_state(&[OpRef(12), OpRef(11)], &[], &mut optimizer, &mut ctx, None);
        assert_eq!(
            exported.exported_short_ops,
            vec![ExportedShortOp::Pure {
                source: OpRef(11),
                opcode: OpCode::GetfieldGcPureI,
                descr: Some(field_descr.clone()),
                args: vec![ExportedShortArg::Const {
                    source: OpRef::from_const(23),
                    value: Value::Ref(ptr),
                }],
                result: ExportedShortResult::Slot(1),
                invented_name: false,
                same_as_source: None,
            }]
        );

        let mut ctx2 = crate::optimizeopt::OptContext::with_num_inputs(8, 2);
        let label_args = import_state(&[OpRef(0), OpRef(1)], &exported, &mut optimizer, &mut ctx2);
        assert_eq!(label_args, vec![OpRef(12), OpRef(11)]);
        assert_eq!(
            ctx2.get_constant(OpRef::from_const(23)),
            Some(&Value::Ref(ptr))
        );
        assert_eq!(
            ctx2.imported_short_pure_ops,
            vec![crate::optimizeopt::ImportedShortPureOp {
                opcode: OpCode::GetfieldGcPureI,
                descr: Some(field_descr.clone()),
                args: vec![crate::optimizeopt::ImportedShortPureArg::Const(
                    Value::Ref(ptr),
                    OpRef::from_const(23)
                )],
                result: OpRef(11),
            }]
        );
    }

    #[test]
    fn test_imported_short_builder_tracks_used_dependency_chain() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(10, 0);
        ctx.seed_constant(OpRef::from_const(10), Value::Int(7));
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntAdd, &[OpRef(12), OpRef::from_const(10)]);
                    op.pos = OpRef(11);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            });
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntMul, &[OpRef(11), OpRef(13)]);
                    op.pos = OpRef(14);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: Some(2),
                invented_name: false,
                same_as_source: None,
            });

        let exported = export_state(
            &[OpRef(12), OpRef(13), OpRef(14)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        // Use a large num_inputs so freshly allocated OpRef positions
        // don't collide with the OpRef values used in the exported state.
        let mut ctx2 = crate::optimizeopt::OptContext::with_num_inputs(10, 1024);
        import_state(
            &[OpRef(0), OpRef(1), OpRef(2)],
            &exported,
            &mut optimizer,
            &mut ctx2,
        );
        ctx2.note_imported_short_use(OpRef(14));
        let sp = ctx2.build_imported_short_preamble().unwrap();

        assert_eq!(sp.ops.len(), 2);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
        assert_eq!(sp.ops[1].op.opcode, OpCode::IntMul);
    }

    #[test]
    fn test_force_op_from_preamble_only_adds_jump_box_after_force_box() {
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(10, 3);
        ctx.initialize_imported_short_preamble_builder(
            &[OpRef(0), OpRef(1), OpRef(2)],
            &[OpRef(10), OpRef(11), OpRef(12)],
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
                    op.pos = OpRef(20);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            }],
        );

        let forced = ctx.force_op_from_preamble(OpRef(20));
        assert_eq!(forced, OpRef(20));

        let sp = ctx.build_imported_short_preamble().unwrap();
        assert_eq!(sp.ops.len(), 1);
        assert!(sp.used_boxes.is_empty());

        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let _ = optimizer.force_box(OpRef(20), &mut ctx);

        let sp = ctx.build_imported_short_preamble().unwrap();
        assert_eq!(sp.used_boxes, vec![OpRef(20)]);
    }

    #[test]
    fn test_force_op_from_preamble_maps_imported_result_back_to_preamble_source() {
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(32, 4);
        ctx.initialize_imported_short_preamble_builder(
            &[OpRef(0), OpRef(1), OpRef(2), OpRef(3)],
            &[OpRef(10), OpRef(11), OpRef(12), OpRef(13)],
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::with_descr(
                        OpCode::GetfieldGcR,
                        &[OpRef(3)],
                        majit_ir::descr::make_field_descr_full(56, 0, 8, Type::Ref, false),
                    );
                    op.pos = OpRef(19);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Heap,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            }],
        );
        ctx.imported_short_sources
            .push(crate::optimizeopt::ImportedShortSource {
                result: OpRef(14),
                source: OpRef(19),
            });

        let forced = ctx.force_op_from_preamble(OpRef(14));
        assert_eq!(forced, OpRef(14));

        let sp = ctx.build_imported_short_preamble().unwrap();
        assert_eq!(sp.ops.len(), 1);
        assert!(sp.used_boxes.is_empty());
        assert!(sp.jump_args.is_empty());

        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let _ = optimizer.force_box(forced, &mut ctx);

        let sp = ctx.build_imported_short_preamble().unwrap();
        assert_eq!(sp.used_boxes, vec![OpRef(19)]);
        assert_eq!(sp.jump_args, vec![OpRef(19)]);
    }

    #[test]
    fn test_exported_state_reimports_invented_short_alias_metadata() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(6, 0);
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntAdd, &[OpRef(12), OpRef(13)]);
                    op.pos = OpRef(30);
                    op
                },
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: true,
                same_as_source: Some(OpRef(14)),
            });

        let exported = export_state(
            &[OpRef(12), OpRef(13), OpRef(14)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        assert_eq!(
            exported.exported_short_ops,
            vec![ExportedShortOp::Pure {
                source: OpRef(30),
                opcode: OpCode::IntAdd,
                descr: None,
                args: vec![ExportedShortArg::Slot(0), ExportedShortArg::Slot(1)],
                result: ExportedShortResult::Temporary(0),
                invented_name: true,
                same_as_source: Some(OpRef(14)),
            }]
        );

        let mut ctx2 = crate::optimizeopt::OptContext::with_num_inputs(6, 1024);
        import_state(
            &[OpRef(0), OpRef(1), OpRef(2)],
            &exported,
            &mut optimizer,
            &mut ctx2,
        );
        let imported_result = ctx2.imported_short_pure_ops[0].result;
        assert_ne!(imported_result, OpRef(30));
        let forced = ctx2.force_op_from_preamble(imported_result);
        // force_op_from_preamble may return the imported position (not necessarily 30)
        let _ = forced;
        assert_eq!(ctx2.imported_short_pure_ops.len(), 1);
        assert_eq!(
            ctx2.imported_short_aliases,
            vec![crate::optimizeopt::ImportedShortAlias {
                result: imported_result,
                same_as_source: OpRef(14),
                same_as_opcode: OpCode::SameAsI,
            }]
        );
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let _ = optimizer.force_box(forced, &mut ctx2);
        let aliases = ctx2.used_imported_short_aliases();
        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0].same_as_source, OpRef(14));
        assert_eq!(aliases[0].same_as_opcode, OpCode::SameAsI);
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
            &[OpRef(0)],
            &[],
            1,
            true,
            &[crate::optimizeopt::ImportedShortAlias {
                result: OpRef(50),
                same_as_source: OpRef(10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &[],
            &std::collections::HashMap::new(),
            None,
            None,
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
    fn test_assemble_peeled_trace_preserves_visible_label_arg_until_body_redef() {
        let p1_ops = vec![{
            let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
            op.pos = OpRef(11);
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::IntGe, &[OpRef(11), OpRef::from_const(0)]);
                op.pos = OpRef(4);
                op
            },
            {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef(11), OpRef::from_const(1)]);
                op.pos = OpRef(11);
                op
            },
            Op::new(OpCode::Jump, &[OpRef(11)]),
        ];

        let mut constants = std::collections::HashMap::new();
        constants.insert(OpRef::from_const(0).0, 2);
        constants.insert(OpRef::from_const(1).0, 1);

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef(11)],
            &[],
            &[OpRef(0)],
            &[],
            1,
            true,
            &[],
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(combined[1].args.as_slice(), &[OpRef(11)]);
        assert_eq!(combined[2].opcode, OpCode::IntGe);
        assert_eq!(combined[2].args[0], OpRef(11));
        assert_eq!(combined[3].opcode, OpCode::IntAdd);
        assert_eq!(combined[3].args[0], OpRef(11));
        assert_ne!(combined[3].pos, OpRef(11));
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].args[0], combined[3].pos);
    }

    #[test]
    fn test_assemble_peeled_trace_preserves_visible_preamble_box_over_body_collision() {
        let p1_ops = vec![{
            let mut op = Op::new(OpCode::GetfieldGcR, &[OpRef(3)]);
            op.pos = OpRef(19);
            op.descr = Some(majit_ir::descr::make_field_descr_full(
                56,
                0,
                8,
                Type::Ref,
                false,
            ));
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::SetfieldGc, &[OpRef(25), OpRef(19)]);
                op.descr = Some(majit_ir::descr::make_field_descr_full(
                    57,
                    8,
                    8,
                    Type::Ref,
                    false,
                ));
                op
            },
            {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef::from_const(1)]);
                op.pos = OpRef(19);
                op
            },
            Op::new(OpCode::Jump, &[OpRef(19)]),
        ];

        let mut constants = std::collections::HashMap::new();
        constants.insert(OpRef::from_const(1).0, 1);

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef(0)],
            &[],
            &[OpRef(0)],
            &[],
            1,
            true,
            &[],
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(combined[2].opcode, OpCode::SetfieldGc);
        assert_eq!(combined[2].args[1], OpRef(19));
        assert_eq!(combined[3].opcode, OpCode::IntAdd);
        assert_ne!(combined[3].pos, OpRef(19));
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].args[0], combined[3].pos);
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
            &[],
            &[OpRef(0)],
            &[OpRef(50)],
            1,
            true,
            &[crate::optimizeopt::ImportedShortAlias {
                result: OpRef(50),
                same_as_source: OpRef(10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &[],
            &std::collections::HashMap::new(),
            None,
            None,
        );

        assert_eq!(combined[2].opcode, OpCode::Label);
        assert_eq!(combined[2].args.as_slice(), &[OpRef(10), combined[1].pos]);
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].args.as_slice(), &[OpRef(10), combined[1].pos]);
    }

    #[test]
    fn test_assemble_peeled_trace_remaps_extra_label_source_slots() {
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::GetfieldGcPureI, &[OpRef(50)]);
                op.pos = OpRef(1);
                op.descr = Some(majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, true));
                op
            },
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(50)]),
        ];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef(10)],
            &[OpRef(0)],
            &[OpRef(0)],
            &[OpRef(50)],
            1,
            true,
            &[crate::optimizeopt::ImportedShortAlias {
                result: OpRef(50),
                same_as_source: OpRef(10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &[],
            &std::collections::HashMap::new(),
            None,
            None,
        );

        let label_idx = combined
            .iter()
            .position(|op| op.opcode == OpCode::Label)
            .expect("label");
        let label = &combined[label_idx];
        let extra_label_arg = label.args[1];
        assert_eq!(label.args.as_slice(), &[OpRef(10), extra_label_arg]);
        let body_getfield = &combined[label_idx + 1];
        assert_eq!(body_getfield.opcode, OpCode::GetfieldGcPureI);
        assert_eq!(body_getfield.args.as_slice(), &[extra_label_arg]);
    }

    #[test]
    fn test_assemble_peeled_trace_carries_body_value_used_before_local_def() {
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::GuardTrue, &[OpRef(64)]);
                op.fail_args = Some(vec![OpRef(64)].into());
                op
            },
            {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef::from_const(0)]);
                op.pos = OpRef(64);
                op
            },
            Op::new(OpCode::Jump, &[OpRef(64)]),
        ];
        let constants = std::collections::HashMap::from([(OpRef::from_const(0).0, 1_i64)]);

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef(10)],
            &[OpRef(0)],
            &[OpRef(0)],
            &[],
            1,
            true,
            &[],
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(combined[0].args.as_slice(), &[OpRef(10), OpRef(64)]);
        assert_eq!(combined[1].opcode, OpCode::GuardTrue);
        assert_eq!(combined[1].args.as_slice(), &[OpRef(64)]);
        assert_eq!(
            combined[1]
                .fail_args
                .as_ref()
                .expect("guard fail args")
                .as_slice(),
            &[OpRef(64)]
        );
        assert_eq!(combined[2].opcode, OpCode::IntAdd);
        assert_ne!(combined[2].pos, OpRef(64));
    }

    #[test]
    fn test_assemble_peeled_trace_maps_jump_to_preamble_via_start_label_contract() {
        let start_descr = TargetToken::new_preamble(0).as_jump_target_descr();
        let p2_ops = vec![{
            let mut jump = Op::new(
                OpCode::Jump,
                &[OpRef(0), OpRef(1), OpRef(2), OpRef(3), OpRef(4)],
            );
            jump.descr = Some(start_descr.clone());
            jump
        }];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef(10), OpRef(11), OpRef(12), OpRef(13), OpRef(14)],
            &[],
            &[OpRef(100), OpRef(101), OpRef(102)],
            &[OpRef(13), OpRef(14)],
            5,
            false,
            &[],
            &[],
            &std::collections::HashMap::new(),
            Some(start_descr),
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(combined[2].opcode, OpCode::Jump);
        assert_eq!(
            combined[2].args.as_slice(),
            &[OpRef(100), OpRef(101), OpRef(102)]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_skips_constant_slots_for_new_body_positions() {
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::New, &[]);
                op.pos = OpRef(1);
                op
            },
            Op::new(OpCode::SetfieldGc, &[OpRef(1), OpRef(0)]),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        let constants = std::collections::HashMap::from([(2_u32, 606_i64), (4_u32, 611_i64)]);

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef(10)],
            &[],
            &[OpRef(0)],
            &[],
            1,
            true,
            &[],
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(combined[1].opcode, OpCode::New);
        assert_ne!(combined[1].pos, OpRef(2));
        assert_ne!(combined[1].pos, OpRef(4));
        assert_eq!(combined[2].opcode, OpCode::SetfieldGc);
        assert_eq!(combined[2].args[0], combined[1].pos);
    }

    #[test]
    fn test_assemble_peeled_trace_skips_constant_extra_label_args() {
        let p2_ops = vec![Op::new(OpCode::Jump, &[OpRef(0)])];
        let constants = std::collections::HashMap::from([(7_u32, 606_i64)]);

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef(10)],
            &[OpRef(0)],
            &[OpRef(0)],
            &[OpRef(7), OpRef(8)],
            1,
            true,
            &[],
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(combined[0].args.as_slice(), &[OpRef(10), OpRef(8)]);
    }

    #[test]
    fn test_assemble_peeled_trace_maps_body_inputs_via_source_slots() {
        let mut constants = std::collections::HashMap::new();
        constants.insert(OpRef::from_const(0).0, 1);
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef(5), OpRef::from_const(0)]);
                op.pos = OpRef(20);
                op
            },
            Op::new(OpCode::Jump, &[OpRef(5)]),
        ];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef(200), OpRef(300)],
            &[OpRef(5), OpRef(0)],
            &[OpRef(0)],
            &[],
            6,
            true,
            &[],
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(combined[0].args.as_slice(), &[OpRef(200), OpRef(300)]);
        assert_eq!(combined[1].opcode, OpCode::IntAdd);
        assert_eq!(combined[1].args[0], OpRef(200));
        assert_eq!(combined[2].opcode, OpCode::Jump);
        assert_eq!(combined[2].args.as_slice(), &[OpRef(200), OpRef(300)]);
    }

    #[test]
    fn test_splice_redirected_tail_replaces_terminal_jump() {
        let body_ops = vec![
            {
                let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
                op.pos = OpRef(3);
                op
            },
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        let redirected_tail = vec![
            {
                let mut op = Op::new(OpCode::GuardTrue, &[OpRef(3)]);
                op.fail_args = Some(vec![OpRef(3)].into());
                op
            },
            Op::new(OpCode::Jump, &[OpRef(3), OpRef(4)]),
        ];

        let spliced = splice_redirected_tail(&body_ops, &redirected_tail);
        assert_eq!(spliced.len(), 3);
        assert_eq!(spliced[0].opcode, OpCode::IntAdd);
        assert_eq!(spliced[1].opcode, OpCode::GuardTrue);
        assert_eq!(spliced[2].opcode, OpCode::Jump);
        assert_eq!(spliced[2].args.as_slice(), &[OpRef(3), OpRef(4)]);
    }

    #[test]
    fn test_closing_loop_contract_arity_uses_actual_jump_contract() {
        let ops = vec![Op::new(OpCode::Jump, &[OpRef(0), OpRef(1), OpRef(2)])];

        assert_eq!(closing_loop_contract_arity(&ops, 5), 3);
    }
}
