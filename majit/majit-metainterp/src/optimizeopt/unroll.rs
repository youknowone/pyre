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
use indexmap::{IndexMap, IndexSet};
use majit_ir::operand::Operand;
use majit_ir::{DescrRef, GcRef, IndexMapExt, Op, OpCode, OpRef, Type, Value};

use crate::history::TargetToken;
use crate::optimizeopt::{
    OptContext, Optimization, OptimizationResult, SnapshotBoxes, SnapshotFramePcs,
    SnapshotFrameSizes, next_snapshot_pos, snapshot_get, snapshot_insert,
};
use crate::resume::SnapshotBox;

/// `unroll.py:119-123`:
///
/// ```python
/// try:
///     info, _ = self.propagate_all_forward(trace, call_pure_results, flush=False)
/// except SpeculativeError:
///     raise InvalidLoop("Speculative heap access would be ill-typed")
/// ```
///
/// Catch a `SpeculativeError` panic raised by
/// `OptContext::protect_speculative_operation` and rethrow it as an
/// `InvalidLoop` — the existing catch_unwind sites at the pyjitpl
/// layer already convert that into a "skip retrace" outcome.
/// unroll.py:119-123 / 122 `except SpeculativeError: raise InvalidLoop`.
///
/// Speculative-fold failures are now recorded as a deferred `InvalidLoop`
/// signal on `OptContext` (see `protect_speculative_operation` /
/// `constant_fold`) and surfaced as `Err(InvalidLoop)` by the optimizer
/// driver, so there is no `SpeculativeError` panic left to catch.  This
/// wrapper is retained as a structural marker of the upstream conversion
/// point; it simply forwards the closure's result.
fn with_speculative_to_invalid_loop<R, F>(f: F) -> R
where
    F: FnOnce() -> R,
{
    f()
}

fn is_trace_constant_ref(opref: OpRef, constants: &majit_ir::ConstMap<majit_ir::Value>) -> bool {
    if opref.is_none() {
        return false;
    }
    // history.py:189-220 — Const variants are constants by Box class.
    // Const OpRefs carry the value directly; the surviving raw `constants`
    // map (empty in production) is also treated as Const. A guard minted by
    // `jump_to_existing_trace` (IntBound.make_guards, unroll.py:333) can
    // reference a fresh const-namespace OpRef whose raw is not yet a key in
    // `constants`; the variant check is authoritative, while the map check
    // additionally catches body-namespace OpRefs forwarded to a constant.
    if opref.is_constant() {
        return true;
    }
    constants.contains_key(&opref.raw())
}

fn is_trace_runtime_ref(opref: OpRef, constants: &majit_ir::ConstMap<majit_ir::Value>) -> bool {
    !opref.is_none() && !is_trace_constant_ref(opref, constants)
}

fn callee_rca_virtual_state_summary(
    vs: &crate::optimizeopt::virtualstate::VirtualState,
) -> Vec<String> {
    fn walk(
        prefix: String,
        node: &crate::optimizeopt::virtualstate::VirtualStateInfoNode,
        out: &mut Vec<String>,
        seen: &mut IndexSet<usize>,
    ) {
        let key = node as *const _ as usize;
        let kind = match &node.info {
            crate::optimizeopt::virtualstate::VirtualStateInfo::Constant(value) => {
                format!("Constant({value:?})")
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::Virtual { fields, .. } => {
                format!("Virtual(fields={})", fields.len())
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::VArray { items, .. } => {
                format!("VArray(items={})", items.len())
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::VStruct { fields, .. } => {
                format!("VStruct(fields={})", fields.len())
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::VArrayStruct {
                element_fields,
                ..
            } => format!("VArrayStruct(elements={})", element_fields.len()),
            crate::optimizeopt::virtualstate::VirtualStateInfo::KnownClass { class_ptr } => {
                format!("KnownClass({class_ptr})")
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::NonNull => "NonNull".to_string(),
            crate::optimizeopt::virtualstate::VirtualStateInfo::IntBounded(bound) => {
                format!("IntBounded({bound:?})")
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::Unknown(tp) => {
                format!("Unknown({tp:?})")
            }
        };
        out.push(format!(
            "{prefix}: {kind} pos={} notvirt={}",
            node.position.get(),
            node.position_in_notvirtuals.get()
        ));
        if !seen.insert(key) {
            return;
        }
        match &node.info {
            crate::optimizeopt::virtualstate::VirtualStateInfo::Virtual { fields, .. }
            | crate::optimizeopt::virtualstate::VirtualStateInfo::VStruct { fields, .. } => {
                for (idx, child) in fields {
                    walk(format!("{prefix}.{idx}"), child, out, seen);
                }
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::VArray { items, .. } => {
                for (idx, child) in items.iter().enumerate() {
                    walk(format!("{prefix}[{idx}]"), child, out, seen);
                }
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::VArrayStruct {
                element_fields,
                ..
            } => {
                for (elem_idx, fields) in element_fields.iter().enumerate() {
                    for (field_idx, child) in fields {
                        walk(
                            format!("{prefix}[{elem_idx}].{field_idx}"),
                            child,
                            out,
                            seen,
                        );
                    }
                }
            }
            crate::optimizeopt::virtualstate::VirtualStateInfo::Constant(_)
            | crate::optimizeopt::virtualstate::VirtualStateInfo::KnownClass { .. }
            | crate::optimizeopt::virtualstate::VirtualStateInfo::NonNull
            | crate::optimizeopt::virtualstate::VirtualStateInfo::IntBounded(_)
            | crate::optimizeopt::virtualstate::VirtualStateInfo::Unknown(_) => {}
        }
    }

    let mut out = Vec::new();
    let mut seen = IndexSet::new();
    for (idx, node) in vs.state.iter().enumerate() {
        walk(format!("state[{idx}]"), node, &mut out, &mut seen);
    }
    out
}

/// Root any GcRef payload reachable from a single `_forwarded` slot (the
/// `AbstractInputArg.forwarded` / `AbstractResOp.forwarded` host,
/// resoperation.py:233-242 / :700). PyPy's Python GC walks `_forwarded`
/// transitively; pyre pins each GcRef on the shadow stack instead.
/// Used by `ExportedState::root_all_gcrefs` to keep
/// `PtrInfo::Constant` / `Const::Ref` payloads live across GC pauses.
/// (`PtrInfo::Instance.known_class` is an immortal vtable integer, not a
/// traced ref, so it is not rooted.)
fn root_forwarded_gcref(
    forwarded: &majit_ir::forwarding::Forwarded,
    info_constant_field: ExportedGcRefField,
    const_ref_field: ExportedGcRefField,
    dummy_key: Operand,
    rooted_refs: &mut Vec<(Operand, ExportedGcRefField, usize)>,
) {
    use crate::optimizeopt::info::{OpInfo, PtrInfo};
    if let majit_ir::forwarding::Forwarded::Info(OpInfo::Ptr(rc)) = forwarded {
        let info = rc.borrow();
        match &*info {
            PtrInfo::Constant(gcref) if !gcref.is_null() => {
                let ss_idx = majit_gc::shadow_stack::push(*gcref);
                rooted_refs.push((dummy_key.clone(), info_constant_field, ss_idx));
            }
            // PtrInfo::Instance.known_class is an immortal vtable integer
            // (ConstInt), never a traced ref — no rooting needed.
            _ => {}
        }
    } else if let majit_ir::forwarding::Forwarded::Const(majit_ir::Const::Ref(gcref)) = forwarded
        && !gcref.is_null()
    {
        let ss_idx = majit_gc::shadow_stack::push(*gcref);
        rooted_refs.push((dummy_key, const_ref_field, ss_idx));
    }
}

/// Mutate the `PtrInfo::Constant` payload of a `Forwarded::Info` cell in
/// place so any other handle sharing the `Rc<RefCell<PtrInfo>>` sees the
/// post-GC GcRef. Matches PyPy `_forwarded` Python object reference
/// semantics — the cell stays, only its content updates.
fn refresh_forwarded_ptrinfo_constant(
    forwarded: &std::cell::RefCell<majit_ir::forwarding::Forwarded>,
    updated: majit_ir::GcRef,
) {
    use crate::optimizeopt::info::{OpInfo, PtrInfo};
    let rc = match &*forwarded.borrow() {
        majit_ir::forwarding::Forwarded::Info(OpInfo::Ptr(rc))
            if matches!(&*rc.borrow(), PtrInfo::Constant(_)) =>
        {
            Some(rc.clone())
        }
        _ => None,
    };
    if let Some(rc) = rc {
        *rc.borrow_mut() = PtrInfo::Constant(updated);
    }
}

/// Overwrite a `Forwarded::Const(Const::Ref(_))` payload in place with the
/// post-GC GcRef. Matches PyPy `_forwarded` Python object reference
/// semantics — the chain terminal stays a Const, only its GcRef updates.
fn refresh_forwarded_const_ref(
    forwarded: &std::cell::RefCell<majit_ir::forwarding::Forwarded>,
    updated: majit_ir::GcRef,
) {
    let is_const_ref = matches!(
        &*forwarded.borrow(),
        majit_ir::forwarding::Forwarded::Const(majit_ir::Const::Ref(_))
    );
    if is_const_ref {
        *forwarded.borrow_mut() =
            majit_ir::forwarding::Forwarded::Const(majit_ir::Const::Ref(updated));
    }
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
    /// compile.py:362: pre-imported ExportedState for compile_retrace.
    /// When set, Phase 1 (preamble) is skipped and Phase 2 uses this state
    /// directly, matching UnrolledLoopData.optimize → optimize_peeled_loop.
    pub imported_state: Option<ExportedState>,
    /// Phase 1's finalized ExportedState, retained across the Phase 2 run
    /// so the caller of `optimize_trace_*` can consult the renamed inputarg
    /// types that the optimizer decided on. RPython does not need this
    /// (Box.type is intrinsic), but majit's `InputArg.tp` side table is
    /// otherwise disconnected from the optimizer's reduced LABEL.
    pub final_exported_state: Option<ExportedState>,
    /// Compact TargetToken LABEL source positions in the full Phase-1
    /// `end_args` vector. This is the small part of `ExportedState.end_args`
    /// needed by Cranelift's host-side direct LABEL entry after
    /// `final_exported_state` is reduced to type metadata.
    pub final_exported_label_source_positions: Option<Vec<usize>>,
    // RPython compile.py:278-284: Phase 1 results for retrace_needed.
    // In RPython, Phase 1 and Phase 2 are separate calls, so Phase 1
    // results are naturally accessible. In pyre, Phase 1 results are
    // returned via the phase1_out output parameter to the caller's
    // stack frame (survives Phase 2 panic).
    /// resume.py parity: per-guard snapshot boxes from tracing time.
    /// Passed through to Phase 1 and Phase 2 optimizers for
    /// store_final_boxes_in_guard snapshot-based fail_args rebuild.
    pub snapshot_boxes: SnapshotBoxes,
    /// Per-frame box counts for multi-frame snapshots.
    pub snapshot_frame_sizes: SnapshotFrameSizes,
    /// Per-guard virtualizable boxes from tracing-time snapshots.
    pub snapshot_vable_boxes: SnapshotBoxes,
    /// Per-guard virtualref boxes from tracing-time snapshots
    /// (resume.py:243-247 vref_array — _number_boxes consumes them
    /// after the virtualizable array).
    pub snapshot_vref_boxes: SnapshotBoxes,
    /// Per-guard per-frame (jitcode_index, pc) from tracing-time snapshots.
    pub snapshot_frame_pcs: SnapshotFramePcs,
    /// pyjitpl.py:2289 all_descrs: dense list indexed by descr_index.
    /// Threaded through inner Optimizer instances for inline registration.
    pub all_descrs: Vec<majit_ir::descr::DescrRef>,
    /// RPython Box type parity: trace inputarg types from recorder.
    /// Each RPython Box carries its type; in majit OpRef is untyped u32.
    /// Propagated to Phase 1 and Phase 2 Optimizer.trace_inputargs
    /// so value_types covers inputarg OpRefs.
    pub trace_inputargs: Vec<majit_ir::OpRef>,
    /// Phase 1 emit ops (filtered to non-NONE pos, non-Void type) carried
    /// into Phase 2 so that `OptContext::op_at` resolves Phase 1 OpRefs
    /// directly via `op.type_` (history.py:220 box.type parity).
    phase1_emit_ops: Vec<majit_ir::OpRc>,
    /// RPython: same Optimizer instance across phases keeps patchguardop.
    /// In majit, separate instances — forward explicitly.
    phase1_patchguardop: Option<majit_ir::Op>,
    /// Cross-phase fresh OpRef high water (majit-specific companion to
    /// RPython's `TraceIterator._index`).
    ///
    /// In RPython each `TraceIterator.next()` allocates a fresh `cls()`
    /// ResOperation whose Python identity distinguishes Phase 1 from
    /// Phase 2 boxes; majit's `OpRef::from_raw(u32)` IS the identity, so Phase 2 must
    /// continue allocating *above* Phase 1's high water mark to keep the
    /// two phases' OpRef sets disjoint. After Phase 1 finishes,
    /// `next_global_opref` holds the smallest OpRef Phase 2 may emit; it
    /// is the `start_fresh` argument the next `TraceIterator::new` call
    /// (or bridge entry) should use. Initialized to 0.
    #[allow(dead_code)]
    pub(crate) next_global_opref: u32,
    /// RPython metainterp_sd.callinfocollection parity.
    /// Maps oopspec indices to (calldescr, func_ptr) for generate_modified_call.
    pub callinfocollection: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    /// compile.py:221 + optimizer.py:530: call_pure_results from tracing.
    /// Passed through to the inner Optimizer for cross-iteration CALL_PURE folding.
    pub call_pure_results: indexmap::IndexMap<Vec<majit_ir::Value>, majit_ir::Value>,
    /// `optimizer.cpu` (model.py:39 `AbstractCPU`) backref, carried into
    /// the inner phase-1/phase-2 `Optimizer.cpu` at spawn time so
    /// `cpu.cls_of_box(runtime_box)` reads (virtualstate.py:601/:608/:620)
    /// and any future `bh_*` calls resolve to the same backend services.
    pub cpu: std::sync::Arc<dyn crate::cpu::Cpu>,
    /// Explicit `input_ops` seed for the phase optimizers, threaded from the
    /// compile caller. `Some(ops)` on the non-cut finish path =
    /// `preamble_data.base.operations()` — the recorder's `Rc<Op>` carrying
    /// the authoritative Phase-1 `_forwarded`. `Some(empty)` on the cut path:
    /// the cut trace's ops live in a remapped namespace, so no seed can
    /// resolve cut-op lookups anyway; an empty seed states that. `None`
    /// (retrace / fixtures) leaves `input_ops` empty.
    pub phase2_input_ops_seed: Option<Vec<majit_ir::OpRc>>,
    /// MetaInterp-owned compile snapshot root slot list. Stored as an address
    /// because the unroll optimizer owns the inner phase optimizers while the
    /// registered GC walker enters through MetaInterp.
    pub compile_snapshot_root_slots: Option<usize>,
    /// Snapshot-root slots that must stay rooted across every phase's
    /// `replace_compile_snapshot_roots`. pyjitpl installs the caller-owned
    /// original snapshot maps here so a moving GC during unroll forwards their
    /// inline `ConstPtr`s in place — the `InvalidLoop` retry re-clones those
    /// originals and would otherwise read a stale pre-move gcref. Prepended to
    /// each phase's slot list rather than overwritten.
    pub persistent_snapshot_root_slots: Vec<usize>,
}

impl UnrollOptimizer {
    pub fn new() -> Self {
        UnrollOptimizer {
            short_preamble: None,
            target_tokens: Vec::new(),
            retraced_count: 0,
            // unroll.py:215/265 reads
            // `warmrunnerdescr.memory_manager.{retrace_limit,max_retrace_guards}`.
            // Production callers (pyjitpl.rs:4109,5170) override
            // these via `warm_state.retrace_limit()` /
            // `.max_retrace_guards()` before driving the optimizer,
            // matching the upstream MemoryManager hookup. These
            // fallback defaults (5/15) are only consulted by test
            // fixtures that bypass pyjitpl — they match the
            // `rpython/jit/metainterp/optimizeopt/test/test_util.py`
            // `n = 5` / `n = 15` test values.
            retrace_limit: 5,
            max_retrace_guards: 15,
            imported_state: None,
            final_exported_state: None,
            final_exported_label_source_positions: None,
            snapshot_boxes: Vec::new(),
            snapshot_frame_sizes: Vec::new(),
            snapshot_vable_boxes: Vec::new(),
            snapshot_vref_boxes: Vec::new(),
            snapshot_frame_pcs: Vec::new(),
            all_descrs: Vec::new(),
            trace_inputargs: Vec::new(),
            phase1_emit_ops: Vec::new(),
            phase1_patchguardop: None,
            next_global_opref: 0,
            callinfocollection: None,
            call_pure_results: indexmap::IndexMap::new(),
            cpu: crate::cpu::default_cpu(),
            phase2_input_ops_seed: None,
            compile_snapshot_root_slots: None,
            persistent_snapshot_root_slots: Vec::new(),
        }
    }

    fn collect_snapshot_const_ptr_slots(maps: &mut [&mut SnapshotBoxes]) -> Vec<usize> {
        let mut slots = Vec::new();
        for map in maps {
            for slot in map.iter_mut() {
                if let Some(boxes) = slot {
                    for sb in boxes {
                        if let majit_ir::OpRef::ConstPtr(gcref) = sb.opref {
                            if !gcref.is_null() {
                                slots.push((&mut sb.opref as *mut majit_ir::OpRef) as usize);
                            }
                        }
                    }
                }
            }
        }
        slots
    }

    fn replace_compile_snapshot_roots(&self, slots: Vec<usize>) {
        if let Some(addr) = self.compile_snapshot_root_slots {
            // SAFETY: pyjitpl installs this address from
            // `MetaInterp.compile_snapshot_refs` for the duration of one
            // compile. Unroll phases run on the same thread as the registered
            // root walker.
            let mut all = self.persistent_snapshot_root_slots.clone();
            all.extend(slots);
            unsafe {
                *(addr as *mut Vec<usize>) = all;
            }
        }
    }

    fn clear_compile_snapshot_roots(&self) {
        self.replace_compile_snapshot_roots(Vec::new());
    }

    /// Optimize the preamble (first iteration) of a loop trace.
    /// Returns the optimized preamble ops + the peeled loop ops.
    ///
    /// Upstream `unroll.py:100-110 optimize_preamble` has no
    /// SpeculativeError catch — by construction the preamble's
    /// gcrefs are concrete runtime values from the recorded
    /// interpreter, so `cpu.protect_speculative_*` always passes.
    /// A SpeculativeError here indicates a genuine bug (concrete
    /// gcref failed type validation) and should propagate.
    pub fn optimize_preamble(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::default_pipeline();
        optimizer.add_pass(Box::new(OptUnroll::new()));
        optimizer.propagate_all_forward(ops)
    }

    /// unroll.py:112-123 `optimize_peeled_loop(trace)`
    /// Optimize the loop body AFTER preamble peeling.  The peeled
    /// preamble has already established the type/class/bounds
    /// information; this method optimizes the repeating body, with
    /// speculative imports that may fail type validation —
    /// `unroll.py:119-123 except SpeculativeError: raise InvalidLoop`.
    pub fn optimize_peeled_loop(&mut self, ops: &[Op]) -> Vec<Op> {
        with_speculative_to_invalid_loop(|| {
            let mut optimizer = crate::optimizeopt::optimizer::Optimizer::default_pipeline();
            optimizer.propagate_all_forward(ops)
        })
    }

    /// unroll.py:238-242: jump_to_preamble(cell_token, jump_op).
    ///
    /// Redirect the closing JUMP to the preamble entry token
    /// (target_tokens[0], virtual_state=None). Only changes the
    /// descriptor, keeping arglist intact — RPython parity.
    pub fn jump_to_preamble(
        body_ops: &[majit_ir::OpRc],
        preamble_target: &TargetToken,
    ) -> Vec<majit_ir::OpRc> {
        assert!(
            preamble_target.virtual_state.is_none(),
            "jump_to_preamble expects the start/preamble target token"
        );
        let mut result = body_ops.to_vec();
        if let Some(idx) = result.iter().rposition(|op| op.opcode == OpCode::Jump) {
            // Re-clone before the descr write: the input Rc may still be
            // registered in the optimizer's producer stores.
            let jump = (*result[idx]).clone();
            jump.setdescr(preamble_target.as_jump_target_descr());
            result[idx] = std::rc::Rc::new(jump);
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
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
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
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
        num_inputs: usize,
    ) -> (Vec<majit_ir::OpRc>, usize) {
        self.optimize_trace_with_constants_and_inputs_vable(ops, constants, num_inputs, None)
            .expect("optimize_trace_with_constants_and_inputs: unexpected InvalidLoop")
    }

    /// compile.py:275-308: compile_loop — 2-phase preamble peeling.
    /// compile.py:275-338: 2-phase preamble peeling (RPython parity).
    ///
    /// Phase 1 (optimize_preamble): full pipeline on trace → preamble_ops.
    /// export_state: capture the preamble's exported optimizer state.
    /// Phase 2 (optimize_peeled_loop): import_state + full pipeline → body_ops.
    /// Assembly: [preamble_no_jump] + Label(label_args) + [body_with_jump].
    pub(crate) fn optimize_trace_with_constants_and_inputs_vable(
        &mut self,
        ops: &[Op],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
        num_inputs: usize,
        vable_config: Option<crate::optimizeopt::virtualize::VirtualizableConfig>,
    ) -> Result<(Vec<majit_ir::OpRc>, usize), crate::optimize::InvalidLoop> {
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
    pub(crate) fn optimize_trace_with_constants_and_inputs_vable_out(
        &mut self,
        ops: &[Op],
        constants: &mut majit_ir::ConstMap<majit_ir::Value>,
        num_inputs: usize,
        vable_config: Option<crate::optimizeopt::virtualize::VirtualizableConfig>,
        phase1_out: Option<&mut Option<(Vec<majit_ir::OpRc>, ExportedState)>>,
    ) -> Result<(Vec<majit_ir::OpRc>, usize), crate::optimize::InvalidLoop> {
        // compile.py:362: if imported_state is pre-set (compile_retrace path),
        // skip Phase 1 and go directly to Phase 2 with the imported state.
        let (mut exported_state, consts_p1, p1_ops) = if let Some(pre_imported) =
            self.imported_state.take()
        {
            // RPython uses object identity for Boxes, so a TraceIterator over a
            // retrace can never numerically collide with the already optimized
            // partial preamble. Majit's OpRef is the identity; when Phase 1 is
            // skipped, recover the partial preamble high-water from the imported
            // state before allocating Phase 2 input/result OpRefs.
            self.next_global_opref = self
                .next_global_opref
                .max(num_inputs as u32)
                .max(pre_imported.opref_high_water());
            // Retrace path: Phase 1 already done; the preamble producers
            // live in the imported state's `partial_trace_operations`
            // (export records them there from `phase1_emit_ops`, #104). The
            // non-skip path populates `self.phase1_emit_ops` from `opt_p1`;
            // restore the same pool here so Phase 2's producer lookup
            // (`new_operations ∪ phase1_emit_ops ∪ resop_refs`) can reach a
            // const-folded preamble producer that no other store carries.
            // RPython keeps box `_forwarded` reachable across the retrace;
            // this is the pyre analog. (Readers dedup, so double-coverage
            // with `resop_refs` is idempotent.)
            self.phase1_emit_ops = pre_imported.partial_trace_operations.clone();
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
            opt_p1.all_descrs = std::mem::take(&mut self.all_descrs);
            opt_p1.callinfocollection = self.callinfocollection.clone();
            opt_p1.cpu = self.cpu.clone();
            opt_p1.trace_inputargs = self.trace_inputargs.clone();
            opt_p1.snapshot_boxes = self.snapshot_boxes.clone();
            opt_p1.snapshot_frame_sizes = self.snapshot_frame_sizes.clone();
            opt_p1.snapshot_vable_boxes = self.snapshot_vable_boxes.clone();
            opt_p1.snapshot_vref_boxes = self.snapshot_vref_boxes.clone();
            opt_p1.snapshot_frame_pcs = self.snapshot_frame_pcs.clone();
            self.replace_compile_snapshot_roots(Self::collect_snapshot_const_ptr_slots(&mut [
                &mut opt_p1.snapshot_boxes,
                &mut opt_p1.snapshot_vable_boxes,
                &mut opt_p1.snapshot_vref_boxes,
            ]));
            opt_p1.call_pure_results = self.call_pure_results.clone();
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
            // opencoder.py:264 `inputarg_from_tp(arg.type)` — fresh inputargs
            // are typed via `self.trace_inputargs`, the recorder-supplied
            // Box list (length must equal `num_inputs`). Derive the &[Type]
            // surface TraceIterator expects from each Box's variant tag
            // (resoperation.py:719/727/739).
            debug_assert_eq!(self.trace_inputargs.len(), num_inputs);
            let p1_inputarg_types: Vec<majit_ir::Type> = self
                .trace_inputargs
                .iter()
                .map(|op| op.ty().expect("inputarg OpRef must carry box.type"))
                .collect();
            // Wrap input ops as `Vec<OpRc>` so TraceIterator's `&[OpRc]`
            // surface receives shared identity (history.py:528). The
            // deep-clone here corresponds to PyPy's `cls()` per-op fresh
            // allocation inside `TraceIterator.next` (opencoder.py:399-401).
            let ops_oprc: Vec<majit_ir::OpRc> =
                ops.iter().map(|op| std::rc::Rc::new(op.clone())).collect();
            let mut p1_iter = crate::opencoder::TraceIterator::new(
                &ops_oprc,
                0,
                ops_oprc.len(),
                None,
                &p1_inputarg_types,
                0, // start_fresh = 0 — inputargs at [0..num_inputs)
            );
            let mut p1_ops_in: Vec<majit_ir::OpRc> = Vec::with_capacity(ops.len());
            while let Some(op) = p1_iter.next() {
                p1_ops_in.push(op);
            }
            let p1_iter_fresh_hw = p1_iter._fresh;
            // compile.py:275 `PreambleCompileData(trace, jumpargs, ...)` —
            // the recorded JUMP arglist is the preamble's `runtime_boxes`
            // (live_arg_boxes captured at the merge point). Capture it into a
            // local here; `opt_p1.runtime_boxes` cannot carry it because
            // `setup()` clears that field at the start of every optimize run
            // (optimizer.rs `self.runtime_boxes.clear()`). Threaded into the
            // exported state below so the peeled-loop close reads it as
            // `state.runtime_boxes` (unroll.py:105 → :153/166) rather than the
            // peeled body's own jump args.
            let recorded_jump_args: Vec<OpRef> = ops
                .iter()
                .rfind(|op| op.opcode == OpCode::Jump)
                .map(|op| op.getarglist().iter().map(|a| a.to_opref()).collect())
                .unwrap_or_default();
            // Hand opt_p1 the per-iter operand pool that p1_iter
            // allocated (slice 77b.A). trace.get_iter() per-call
            // inputarg_from_tp(...) / cls() — each phase optimizes against a
            // fresh Box identity set so _forwarded mutations cannot alias
            // across phases.
            //
            // Const operands are NOT cached: `get_box_replacement_box`
            // allocates a fresh const operand per call from `const_pool`
            // (`history.py:220` ConstInt(value) per-call-site parity).
            // opt_p1's entry path seeds `const_pool` from the shared
            // `constants` map (`optimizer.rs:1944`).
            // unroll.py:100-110 `optimize_preamble` has no
            // SpeculativeError catch — Phase 1 corresponds to the
            // preamble, whose gcrefs are concrete runtime values
            // from the recorded interpreter and never raise
            // SpeculativeError under correct construction.  A raise
            // here is a real bug and must propagate.
            // Phase 1's input ops are the same recorder ops the seed names,
            // so its `input_ops` can come from the seed too — only the read
            // source changes; `_forwarded` writes are untouched. No
            // forwarding yet at Phase 1 setup, so the seed is trivially the
            // authoritative source here.
            if let Some(seed) = &self.phase2_input_ops_seed {
                opt_p1.explicit_input_ops_seed = Some(seed.clone());
            }
            let p1_ops =
                opt_p1.run_optimize_from_inputs(&p1_ops_in, &mut consts_p1, num_inputs, false)?;
            // RPython parity: Phase 1 optimizer may discover new constants
            // via make_constant (e.g., constant-folded heap reads, guard
            // class pointers). These live on the operand's forwarded chain
            // (and in `ctx.const_pool` for const-namespace OpRefs) but
            // not in `consts_p1` (which was only seeded from the input
            // constants). Merge them back so
            // build_short_preamble_from_exported_boxes can capture all
            // constants referenced by short preamble ops.
            if let Some(ref final_ctx) = opt_p1.final_ctx {
                // history.py:220 box.type parity: every `Value` carries its
                // Const class identity intrinsically; no companion type map
                // needs threading alongside.
                crate::optimizeopt::optimizer::merge_backend_constants_from_ctx(
                    final_ctx,
                    &mut consts_p1,
                );
            }
            let p1_ni = opt_p1.final_num_inputs();

            match opt_p1.exported_loop_state.take() {
                Some(mut state) => {
                    // unroll.py:105 `export_state(..., runtime_boxes, ...)` —
                    // carry the recorded JUMP args into ExportedState so the
                    // peeled-loop close passes them to generate_guards as
                    // `state.runtime_boxes` (unroll.py:153/166).
                    state.runtime_boxes = recorded_jump_args.clone();
                    self.final_exported_label_source_positions =
                        Some(state.label_source_positions.clone());
                    // end_arg_types is already populated by
                    // `Optimizer::optimize_with_constants_and_inputs_at`
                    // using the optimizer-visible `ctx.opref_type()` (see
                    // optimizer.rs:2405-2416). Overwriting it here with
                    // `opcode.result_type()` + `Type::Ref` default would
                    // retype Phase 1 inputarg OpRefs (absent from `p1_ops`)
                    // as `Ref`, which later feeds the cross-type forward
                    // assertion in `OptContext::make_equal_to`.
                    // RPython Phase 1 → Phase 2 heap cache transfer:
                    // RPython does NOT serialize heap cache in ExportedState.
                    // HeapOps in the short preamble are replayed during Phase 2's
                    // inline_short_preamble, populating the heap cache naturally
                    // through OptHeap. serialize_optheap is only for bridgeopt.
                    // opencoder.py:271 _index parity: Phase 2's TraceIterator
                    // must allocate fresh boxes ABOVE Phase 1's high water
                    // mark so the two phases' OpRef namespaces are disjoint
                    // (RPython relies on Python identity to distinguish them;
                    // majit relies on disjoint integer ranges). The high
                    // water is `final_ctx.next_pos` after Phase 1 emit, with
                    // a floor of `num_inputs` for empty traces.
                    // Phase 1's emit-side high water (`final_ctx.next_pos`)
                    // reflects only the positions `ctx.reserve_pos` handed
                    // out; the per-iteration TraceIterator additionally
                    // allocates fresh OpRefs via its `_fresh` counter and
                    // those values can exceed `next_pos` for traces where
                    // Phase 1 dropped or folded many ops. Taking the max
                    // with `p1_iter_fresh_hw` keeps Phase 2's inputarg base
                    // strictly above every OpRef Phase 1 ever allocated,
                    // so Phase 2's own fresh OpRefs cannot collide with
                    // positions in `phase1_emit_ops` / typed snapshots.
                    self.next_global_opref = opt_p1
                        .final_ctx
                        .as_ref()
                        .map(|c| c.next_pos)
                        .unwrap_or(num_inputs as u32)
                        .max(num_inputs as u32)
                        .max(p1_iter_fresh_hw);
                    // RPython Box type parity: Phase 1's emit ops carry
                    // `op.type_` intrinsically; Phase 2's `op_at` reads it
                    // directly to resolve cross-phase OpRefs that appear as
                    // imported_label_args / fail_args / record_same_as
                    // sources (history.py:220 parity).
                    self.phase1_emit_ops = std::mem::take(&mut opt_p1.phase1_emit_ops);
                    // RPython: same Optimizer instance keeps patchguardop.
                    if self.phase1_patchguardop.is_none() {
                        self.phase1_patchguardop = opt_p1.patchguardop.clone();
                    }
                    state.patchguardop = opt_p1.patchguardop.clone();
                    self.all_descrs = std::mem::take(&mut opt_p1.all_descrs);
                    // Box.type parity: retain Phase 1's renamed_inputargs so
                    // the backend can read the reduced LABEL's declared types
                    // off each typed InputArg OpRef. Clone only the field we
                    // need to avoid keeping Phase 1 short preamble data alive
                    // longer than before.
                    let mut final_exported_state = ExportedState {
                        end_args: Vec::new(),
                        label_source_positions: state.label_source_positions.clone(),
                        next_iteration_args: Vec::new(),
                        end_arg_types: Vec::new(),
                        virtual_state: state.virtual_state.clone(),
                        exported_infos: indexmap::IndexMap::new(),
                        exported_short_boxes: Vec::new(),
                        short_boxes: Vec::new(),
                        short_box_const_values: indexmap::IndexMap::new(),
                        short_preamble: None,
                        renamed_inputargs: state.renamed_inputargs.clone(),
                        short_inputargs: Vec::new(),
                        short_inputarg_refs: Vec::new(),
                        runtime_boxes: Vec::new(),
                        patchguardop: None,
                        phase1_emit_high_water: self.next_global_opref,
                        partial_trace_inputargs: Vec::new(),
                        partial_trace_operations: Vec::new(),
                        short_box_producer_roots: Vec::new(),
                        rooted_refs: Vec::new(),
                        rooted_const_ptr_slots: Vec::new(),
                        shadow_stack_base: 0,
                    };
                    final_exported_state.root_all_gcrefs();
                    self.final_exported_state = Some(final_exported_state);
                    (state, consts_p1, p1_ops)
                }
                None => {
                    *constants = consts_p1;
                    // Take back the descriptor table grown during Phase 1's pass
                    // (`ensure_descr_index` appends at guard-resume serialization),
                    // mirroring the Some branch's restore above. Without this the
                    // non-peeled path drops those descriptors and publishes a stale
                    // `all_descrs` back over the global table.
                    self.all_descrs = std::mem::take(&mut opt_p1.all_descrs);
                    // RPython: compile_loop uses flush=True — terminal op
                    // (Finish/Jump) goes through the pass pipeline normally.
                    // majit: flush=False stores it in terminal_op; restore it
                    // here for non-peeled traces that return directly.
                    let mut ops = p1_ops;
                    if let Some(terminal) = opt_p1.terminal_op.take() {
                        ops.push(std::rc::Rc::new(terminal));
                    }
                    // `compile.py:245` `jitcell_token.target_tokens = [target_token]`
                    // / `:290` `jitcell_token.target_tokens = [start_descr]` —
                    // PyPy unconditionally publishes one preamble target token
                    // for any successful compile path (compile_simple_loop +
                    // compile_loop both).  Phase 2 is bypassed here (no
                    // exported_loop_state) but the loop still compiles, so
                    // mirror the unconditional preamble registration so that
                    // `JitCellToken.target_tokens` is non-empty at the
                    // `has_compiled_targets` (`pyjitpl.py:3898`) read site.
                    self.ensure_preamble_target_token();
                    let loop_arity = closing_loop_contract_arity(&ops, p1_ni);
                    self.clear_compile_snapshot_roots();
                    return Ok((ops, loop_arity));
                }
            }
        };
        self.clear_compile_snapshot_roots();
        // unroll.py:454 end_args carry type via Box; export_state already
        // populated `exported_state.end_arg_types` from ctx.
        // RPython parity: Phase 2 needs patchguardop from Phase 1's
        // GuardFutureCondition (unroll.py:333). Extract before dropping opt_p1.
        let p1_patchguardop = exported_state.patchguardop.clone();

        self.ensure_preamble_target_token();
        // ── Phase 2: optimize_peeled_loop (compile.py:291-292) ──
        let body_num_inputs = num_inputs;

        if crate::majit_log_enabled() {
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
                    .map(|p| p.rd_resume_position.get())
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
        opt_p2.all_descrs = std::mem::take(&mut self.all_descrs);
        opt_p2.callinfocollection = self.callinfocollection.clone();
        opt_p2.cpu = self.cpu.clone();
        // #217 Slice 3 — Phase 2 (peeled-loop pass) runtime value seed.
        // The remap is deferred until after the Phase 2 `TraceIterator`
        // builds its `_cache`, because both inputargs AND body op results
        // are reminted to fresh OpRefs (opencoder.py:259-267 / :399-401).
        // Using the iterator's `_cache` as the lookup table is the
        // RPython-orthodox equivalent of Box identity parallelism between
        // `boxes` and `runtime_boxes` (virtualstate.py:646-648): each
        // Phase 1 raw position maps to the same Phase 2 box the iterator
        // allocated for it.  See the remap block after `p2_cache` is
        // built below.
        // `trace_inputargs` stays a clone: it is re-read below at the Phase 2
        // TraceIterator setup (`p2_inputarg_types`) and by the earlier
        // `debug_assert_eq!` on its length.
        opt_p2.trace_inputargs = self.trace_inputargs.clone();
        // Move, not clone: Phase 2 is the last reader of these fields in this
        // function (no `self.`-qualified read past this point on any path —
        // the imported_state path writes `phase1_emit_ops` above then reads it
        // here; the non-peeled early-return arm returns before reaching here),
        // and the caller never reads `unroll_opt.{snapshot_*,phase1_emit_ops,
        // call_pure_results}` after the optimize call (the InvalidLoop retry
        // moves the caller's own `snapshot_map` locals, not these fields). A
        // `Vec` move copies only the (ptr,len,cap) header and leaves the inner
        // buffers — and the `*mut OpRef` const-ptr root slots collected into
        // `opt_p2` at the re-root below — at the same addresses. Mirrors the
        // move-not-clone precedent on the InvalidLoop retry path (pyjitpl.rs
        // `simple_opt.snapshot_boxes = snapshot_map`).
        opt_p2.phase1_emit_ops = std::mem::take(&mut self.phase1_emit_ops);
        opt_p2.snapshot_boxes = std::mem::take(&mut self.snapshot_boxes);
        opt_p2.snapshot_frame_sizes = std::mem::take(&mut self.snapshot_frame_sizes);
        opt_p2.snapshot_vable_boxes = std::mem::take(&mut self.snapshot_vable_boxes);
        opt_p2.snapshot_vref_boxes = std::mem::take(&mut self.snapshot_vref_boxes);
        opt_p2.snapshot_frame_pcs = std::mem::take(&mut self.snapshot_frame_pcs);
        opt_p2.call_pure_results = std::mem::take(&mut self.call_pure_results);
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
        if crate::majit_log_enabled() {
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
        // majit's `OpRef::from_raw(u32)` IS the identity, so "fresh per iteration"
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
        // opencoder.py:264 `inputarg_from_tp(arg.type)` — same per-arg types
        // as Phase 1 (Phase 2 walks the body half of the same trace).
        debug_assert_eq!(self.trace_inputargs.len(), body_num_inputs);
        let p2_inputarg_types: Vec<majit_ir::Type> = self
            .trace_inputargs
            .iter()
            .map(|op| op.ty().expect("inputarg OpRef must carry box.type"))
            .collect();
        // Wrap into `Vec<OpRc>` for TraceIterator's `&[OpRc]` surface.
        let ops_oprc: Vec<majit_ir::OpRc> =
            ops.iter().map(|op| std::rc::Rc::new(op.clone())).collect();
        let mut iter = crate::opencoder::TraceIterator::new(
            &ops_oprc,
            0,
            ops_oprc.len(),
            None,
            &p2_inputarg_types,
            phase2_inputarg_base, // fresh inputargs at [phase2_inputarg_base..)
        );
        let mut p2_ops_in: Vec<majit_ir::OpRc> = Vec::with_capacity(ops.len());
        while let Some(op) = iter.next() {
            p2_ops_in.push(op);
        }
        let p2_high_water = iter._fresh;
        let p2_cache = iter._cache;
        // opencoder.py:286-289 `_get(self, i)` parity. `p2_cache[raw_pos]`
        // holds the fresh per-iteration box for every Phase 1 input/op
        // position that Phase 2's TraceIterator walks. snapshot_boxes /
        // snapshot_vable_boxes are populated during Phase 1 against
        // those same Phase 1 positions, so each entry must hit the
        // cache. A miss means the snapshot references a Phase 1 OpRef
        // that does not appear in Phase 2's input ops (e.g. a stale
        // reference to a Phase 1-DCE'd op). The previous
        // `unwrap_or(opref)` silently leaked such stale OpRefs into
        // Phase 2 namespace; production probes (nbody / fannkuch /
        // fib_loop / fib_recursive / spectral_norm) show zero misses,
        // so promote to a strict panic matching RPython's _get assert.
        let translate_opref = |opref: OpRef| -> OpRef {
            if opref.is_none() || opref.is_constant() {
                return opref;
            }
            p2_cache
                .get(opref.raw() as usize)
                .and_then(|slot| slot.as_ref())
                .map(|b| b.to_opref())
                .unwrap_or_else(|| {
                    panic!(
                        "phase2 snapshot remap cache miss for {opref:?} \
                         (cache_len={} body_ni={} phase2_inputarg_base={})",
                        p2_cache.len(),
                        body_num_inputs,
                        phase2_inputarg_base,
                    )
                })
        };
        for boxes in opt_p2.snapshot_boxes.iter_mut().flatten() {
            for r in boxes.iter_mut() {
                *r = r.map_opref(translate_opref);
            }
        }
        for boxes in opt_p2.snapshot_vable_boxes.iter_mut().flatten() {
            for r in boxes.iter_mut() {
                *r = r.map_opref(translate_opref);
            }
        }
        for boxes in opt_p2.snapshot_vref_boxes.iter_mut().flatten() {
            for r in boxes.iter_mut() {
                *r = r.map_opref(translate_opref);
            }
        }
        self.replace_compile_snapshot_roots(Self::collect_snapshot_const_ptr_slots(&mut [
            &mut opt_p2.snapshot_boxes,
            &mut opt_p2.snapshot_vable_boxes,
            &mut opt_p2.snapshot_vref_boxes,
        ]));
        // Phase 1's emitted ops are already in Phase 1's emitted
        // namespace `[num_inputs..next_global_opref)`. Phase 2 body may
        // reference these via `imported_label_args`. They are NOT in the
        // `p2_cache` (only raw trace positions are), so leave
        // `phase1_emit_ops` untranslated.
        // Hand opt_p2 the per-iter operand pool that the Phase 2
        // iter allocated. Disjoint from opt_p1's pool — _forwarded mutations
        // recorded against Phase 1 boxes do not alias Phase 2 boxes for the
        // same OpRef raw index, fixing the Rc<Box> split-brain that broke
        // the first 77b attempt.
        //
        // Const operands: see opt_p1 plumb above — fresh per-call from
        // `const_pool`, no dedup.
        // unroll.py:119-123 — Phase 2 (peeled loop) raises
        // SpeculativeError on speculative-fold paths; convert
        // to InvalidLoop so the caller's catch handles it.
        // Seed Phase 2's `input_ops` from the recorder `Rc<Op>` (same `Rc`,
        // same Phase-1 `_forwarded`) so `input_ops` is the authoritative
        // producer store. A `Some(empty)` (cut path) leaves it empty.
        if let Some(seed) = &self.phase2_input_ops_seed {
            opt_p2.explicit_input_ops_seed = Some(seed.clone());
        }
        let p2_ops = with_speculative_to_invalid_loop(|| {
            opt_p2.optimize_with_constants_and_inputs_at(
                &p2_ops_in,
                &mut consts_p2,
                body_num_inputs,
                phase2_inputarg_base, // inputarg_base — Phase 2 inputargs at [phase2_inputarg_base..)
                p2_high_water,
                // Phase 2 ops are fresh-Rc `TraceIterator` wraps; with no
                // explicit seed threaded, `input_ops` stays empty and identity
                // comes from `bind_input_resops` / emit.
                false,
            )
        })?;
        self.clear_compile_snapshot_roots();
        // RPython optimizer.py:614-625 freezes op arguments during
        // `_emit_operation`; optimizer.py:598-612 may then install a Const
        // forwarding on the result, but it never retroactively rewrites the
        // already-emitted op. Keep Phase 2 output in that emit-time shape here.
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
        // D2 shift_back shim removed: the assembly now natively handles
        // disjoint Phase 2 inputarg OpRefs via the inputarg_base parameter.
        // Phase 2 inputarg OpRefs at [phase2_inputarg_base..+body_num_inputs)
        // flow directly into the assembly without translation.
        // Phase 2 may discover new constants via make_constant (e.g., guard
        // class pointers from collect_use_box_guards).
        // Merge back into consts_p2 so the backend can resolve them.
        if let Some(ref final_ctx) = opt_p2.final_ctx {
            // history.py:220 box.type parity: every `Value` carries its
            // Const class identity intrinsically.
            crate::optimizeopt::optimizer::merge_backend_constants_from_ctx(
                final_ctx,
                &mut consts_p2,
            );
        }
        let body_terminal_op = opt_p2.terminal_op.clone();
        let p2_ni = opt_p2.final_num_inputs();
        self.all_descrs = std::mem::take(&mut opt_p2.all_descrs);
        let label_args = opt_p2
            .imported_label_args
            .clone()
            .expect("phase 2 missing import_state label_args");

        if crate::majit_log_enabled() {
            for op in &p2_ops {
                if op.opcode.is_guard() {
                    let rd_numb_len = op.resolved_rd_numb().map(|s| s.len()).unwrap_or(0);
                    if let Some(fa) = op.getfailargs() {
                        let fa_raw: Vec<String> = fa
                            .iter()
                            .map(|a| format!("OpRef::from_raw({})", a.to_opref().raw()))
                            .collect();
                        eprintln!(
                            "[jit] p2 guard {:?} pos={:?} resume_pos={} rd_numb={} fail_args_raw=[{}]",
                            op.opcode,
                            op.pos.get(),
                            op.rd_resume_position.get(),
                            rd_numb_len,
                            fa_raw.join(", ")
                        );
                    } else {
                        eprintln!(
                            "[jit] p2 guard {:?} pos={:?} resume_pos={} rd_numb={} fail_args_raw=<none>",
                            op.opcode,
                            op.pos.get(),
                            op.rd_resume_position.get(),
                            rd_numb_len,
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
                    op.opcode,
                    op.pos.get(),
                    op.getarglist()
                );
            }
        }

        // ── unroll.py:140-175: finalize + jump_to_existing_trace ──
        let imported_short_aliases = opt_p2.imported_short_aliases.clone();
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
            exported_short_boxes_produced,
            exported_renamed_inputargs,
            exported_runtime_boxes,
        ) = {
            let es = opt_p2
                .imported_loop_state
                .as_ref()
                .expect("imported_loop_state must survive Phase 2");
            (
                es.virtual_state.clone(),
                es.end_args.clone(),
                es.short_inputargs.clone(),
                es.short_boxes.clone(),
                es.renamed_inputargs.clone(),
                es.runtime_boxes.clone(),
            )
        };
        // RPython unroll.py:124-141 performs an extra end-of-preamble forcing
        // pass on the closing JUMP args before finalize_short_preamble():
        //   for a in end_jump.getarglist():
        //       self.force_box_for_end_of_preamble(get_box_replacement(a))
        //   current_vs = self.optunroll.get_virtual_state(end_jump.getarglist())
        //   target_virtual_state = self.optunroll.pick_virtual_state(...)
        //   args = target_virtual_state.make_inputargs(..., force_boxes=True)
        //   for arg in args:
        //       self.force_box(arg)
        //
        // The optimizer-level imported_short_preamble snapshot is taken before
        // this step, so rebuild from the live Phase 2 context here instead of
        // trusting the stale copy.
        let body_jump_args: Vec<OpRef> = body_terminal_op
            .as_ref()
            .map(|jump| jump.getarglist().iter().map(|a| a.to_opref()).collect())
            .or_else(|| {
                p2_ops
                    .iter()
                    .rfind(|op| op.opcode == OpCode::Jump)
                    .map(|jump| jump.getarglist().iter().map(|a| a.to_opref()).collect())
            })
            .unwrap_or_default();
        let (imported_short_preamble_builder, rebuilt_imported_short_preamble) =
            if let Some(mut final_ctx) = opt_p2.final_ctx.take() {
                if !body_jump_args.is_empty() {
                    // unroll.py:126-127 parity:
                    //   for a in end_jump.getarglist():
                    //       self.force_box_for_end_of_preamble(get_box_replacement(a))
                    //
                    // Route through `force_box_for_end_of_preamble` (the
                    // per-box type-gating wrapper) rather than the inner
                    // dispatcher, matching optimizer.py:306-319.
                    let resolved_jump_args: Vec<OpRef> = body_jump_args
                        .iter()
                        .map(|&arg| final_ctx.get_replacement_opref(arg))
                        .collect();
                    for &arg in &resolved_jump_args {
                        let _ = opt_p2.force_box_for_end_of_preamble(arg, &mut final_ctx);
                    }
                    let forced_jump_args: Vec<OpRef> = body_jump_args
                        .iter()
                        .map(|&arg| final_ctx.get_replacement_opref(arg))
                        .collect();
                    let current_vs = crate::optimizeopt::virtualstate::export_state(
                        &forced_jump_args,
                        &final_ctx,
                    );
                    let mut target_states: Vec<crate::optimizeopt::virtualstate::VirtualState> =
                        self.target_tokens
                            .iter()
                            .filter_map(|token| token.virtual_state.clone())
                            .collect();
                    let target_virtual_state = if let Some(idx) =
                        pick_virtual_state(&current_vs, &target_states, &mut final_ctx)
                    {
                        target_states.swap_remove(idx)
                    } else {
                        exported_vs.clone()
                    };
                    if let Ok(args) = target_virtual_state.make_inputargs(
                        &forced_jump_args,
                        &mut opt_p2,
                        &mut final_ctx,
                        true,
                    ) {
                        for arg in args {
                            if !arg.is_none() {
                                let _ = opt_p2.force_box(arg, &mut final_ctx);
                            }
                        }
                    }
                }
                // TODO (B.6.5): RPython's
                // `force_op_from_preamble` (unroll.py:26-39) only seeds
                // `potential_extra_ops`. The orthodox path that fills
                // `used_boxes` / `short_preamble_jump` / `extra_same_as`
                // is `force_box -> potential_extra_ops.pop ->
                // add_preamble_op` (shortpreamble.py:432-440), which
                // RPython runs in `optimize_op` whenever a body op
                // forces an imported Box. RPython's Box identity then
                // keeps that Box live across the loop boundary
                // implicitly.
                //
                // majit's disjoint Phase 1 / Phase 2 OpRef namespaces
                // strip the Box-identity transparency. Body ops that
                // CSE-replace through `force_op_from_preamble_op` and
                // never trigger `force_box` leave their imported entry
                // in `potential_extra_ops`, so the LABEL is not
                // extended with the corresponding `used_box`. To
                // recover orthodox shape, walk body args + fail_args
                // in body-occurrence order before
                // `build_imported_short_preamble` and call the
                // orthodox `force_box` for every body OpRef that has a
                // pending `potential_extra_ops` entry. `force_box`
                // routes through `add_preamble_op_from_pop`
                // (optimizer.rs `force_box`), so used_boxes /
                // short_preamble_jump / extra_same_as fill via the
                // canonical RPython path.
                {
                    let mut visited_force: indexmap::IndexSet<OpRef> = indexmap::IndexSet::new();
                    for op in p2_ops.iter() {
                        if op.opcode == OpCode::Jump {
                            // The terminal jump's args are already
                            // run through `force_box_for_end_of_preamble`
                            // (unroll.py:126-127) above.
                            continue;
                        }
                        let arg_list = op.getarglist_copy();
                        let arg_iter = arg_list
                            .iter()
                            .map(|a| a.to_opref())
                            .chain(op.getfailargs().into_iter().flatten().map(|a| a.to_opref()));
                        for arg in arg_iter {
                            if !is_trace_runtime_ref(arg, &consts_p2) {
                                continue;
                            }
                            if visited_force.contains(&arg) {
                                continue;
                            }
                            visited_force.insert(arg);
                            let resolved = final_ctx.get_replacement_opref(arg);
                            let needs_force = final_ctx
                                .potential_extra_ops
                                .iter()
                                .any(|(k, _)| *k == arg || *k == resolved);
                            if needs_force {
                                let _ = opt_p2.force_box(arg, &mut final_ctx);
                            }
                        }
                    }
                }
                let rebuilt = final_ctx.build_imported_short_preamble();
                let builder = final_ctx.imported_short_preamble_builder.clone();
                opt_p2.final_ctx = Some(final_ctx);
                (builder, rebuilt)
            } else {
                (
                    opt_p2.imported_short_preamble_builder.clone(),
                    opt_p2.imported_short_preamble.clone(),
                )
            };
        let mut initial_sp = rebuilt_imported_short_preamble.unwrap_or_else(|| {
            // RPython unroll.py:497-504 `for produced_op in
            // exported_state.short_boxes` parity: consume the eagerly-derived
            // ProducedShortOp list stored on ExportedState at export time.
            crate::optimizeopt::shortpreamble::build_short_preamble_from_produced_boxes(
                &exported_end_args,
                &exported_short_inputargs,
                &exported_short_boxes_produced,
            )
        });
        // Per-slot ORIGINAL box (what each renamed `short_inputargs[i]`
        // replaces), recovered from the produced `InputArg` short boxes via
        // `label_arg_idx` (shortpreamble.py:417 keys the info lookup by
        // `produced_op.short_op.res`, the original). Shared by the two
        // consumers below. Every label/virtual slot produces an InputArg
        // entry, except a duplicate slot (one box appears twice in
        // `label_args + virtuals`: the `potential_ops[box]` overwrite keys the
        // single entry at its LAST slot — shortpreamble.py:259, mirrored by
        // `live_slot` in add_short_input_arg — so the earlier dead slot stays
        // None) and a const-folded slot (dropped at export, never surviving
        // into Phase 2).
        let mut slot_to_original: Vec<Option<OpRef>> = vec![None; initial_sp.inputargs.len()];
        for (_, produced) in &exported_short_boxes_produced {
            if !matches!(
                produced.kind,
                crate::optimizeopt::shortpreamble::PreambleOpKind::InputArg
            ) {
                continue;
            }
            if let Some(slot) = produced.label_arg_idx {
                if slot < slot_to_original.len() {
                    slot_to_original[slot] = Some(produced.res.to_opref());
                }
            }
        }
        // shortpreamble.py:416-425 parity: attach PtrInfo to each short
        // inputarg. RPython keys the info by the ORIGINAL res box
        // (`op = produced_op.short_op.res; info = exported_infos.get(op)`)
        // and forwards it onto the renamed `preamble_op`. The renamed short
        // inputarg carries no PtrInfo of its own, so the lookup MUST use the
        // original box, not the renamed one — otherwise a distinct renamed
        // identity yields None and the loop-carried info (e.g. KnownClass)
        // is dropped. A dead duplicate slot's box carries no info
        // (shortpreamble.py:414-417), so its None is correct.
        if let Some(ref final_ctx) = opt_p2.final_ctx {
            let mut infos = Vec::with_capacity(initial_sp.inputargs.len());
            for (i, inputarg) in initial_sp.inputargs.iter().enumerate() {
                // Phase-2 info of the original box this short inputarg renames.
                let original = slot_to_original[i];
                let info = original
                    .and_then(|o| final_ctx.get_box_replacement_operand_opt(o))
                    .or_else(|| final_ctx.get_box_replacement_operand_opt(*inputarg))
                    .as_ref()
                    .and_then(|o| final_ctx.peek_ptr_info(o));
                infos.push(info);
            }
            initial_sp.inputarg_infos = infos;
        }
        // shortpreamble.py:255-259 renamed-short_inputargs: in THIS import path
        // the short-preamble Label is the renamed `short_inputargs`, so seed
        // `phase1_inputargs` with the per-slot ORIGINALS (paired 1:1 with the
        // renamed Label / jump_args) as the second inline-mapping leg — see the
        // load-bearing analysis at the consuming site in inline_short_preamble.
        // NOTE (measured): the originals seeded here are empirically INERT — post
        // #217 produce_arg embeds the renamed boxes into the short ops, so no short
        // op references an original (0 of the 176 corpus consumptions of the
        // phase1 leg were original boxes; the load-bearing consumptions all come
        // from the build_short_preamble_struct producer, whose phase1 holds the
        // RENAMED boxes). The field cannot be dropped on that account because the
        // OTHER producer is load-bearing; see the convergence note at the consumer.
        // Dead duplicate / const-folded slots (`slot_to_original == None`) fall
        // back to the renamed Label box, leaving phase1[i] == inputargs[i] there
        // (no insert at the consumer). No-op when the Label already IS the
        // originals (the two box sets coincide).
        if initial_sp.phase1_inputargs.is_none() {
            let phase1: Vec<OpRef> = initial_sp
                .inputargs
                .iter()
                .enumerate()
                .map(|(i, label)| slot_to_original[i].unwrap_or(*label))
                .collect();
            let differs = phase1
                .iter()
                .zip(initial_sp.inputargs.iter())
                .any(|(orig, label)| orig != label);
            if differs {
                initial_sp.phase1_inputargs = Some(phase1);
            }
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
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] too many guards (>{}), disabling retracing",
                    self.max_retrace_guards
                );
            }
        }

        if crate::majit_log_enabled() {
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
        if crate::majit_log_enabled() {
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
                .map(|jump| jump.getarglist().iter().map(|a| a.to_opref()).collect())
                .or_else(|| {
                    body_ops
                        .iter()
                        .rfind(|o| o.opcode == OpCode::Jump)
                        .map(|j| j.getarglist().iter().map(|a| a.to_opref()).collect())
                })
                .unwrap_or_default();
            let mut current_label_args = label_args.clone();
            // RPython Box parity: each used_box is a distinct Box even
            // when two virtuals share the same OpRef. Allocate fresh
            // OpRefs for duplicates so the LABEL carries independent slots.
            {
                let mut seen_used: indexmap::IndexSet<OpRef> = indexmap::IndexSet::new();
                let mut next_fresh = current_label_args
                    .iter()
                    .copied()
                    .chain(initial_sp.used_boxes.iter().copied())
                    .map(|a| a.raw())
                    .max()
                    .unwrap_or(0)
                    .saturating_add(1)
                    .max(body_num_inputs as u32 + 100);
                for ub in &initial_sp.used_boxes {
                    let ub = *ub;
                    if !seen_used.contains(&ub) {
                        seen_used.insert(ub);
                        current_label_args.push(ub);
                    } else {
                        // shortpreamble.py:343-350 inputarg_from_tp parity:
                        // when the short preamble needs a fresh label slot,
                        // PyPy creates a new typed InputArg from the source
                        // box's `.type`. Carry that into duplicate used_box
                        // slots: typed parents produce a matching
                        // `InputArg{Int,Float,Ref}` alias.
                        //
                        // history.py:220 `box.type` is the authoritative
                        // source. Variant tag is the primary line-by-line
                        // equivalent (resoperation.py:29
                        // `AbstractValue.type`); legacy untyped paths fall
                        // through to Phase 2's `final_ctx.opref_type` and
                        // the recorder's `trace_inputargs`.
                        let tp = ub
                            .ty()
                            .or_else(|| {
                                opt_p2.final_ctx.as_ref().and_then(|ctx| ctx.opref_type(ub))
                            })
                            .or_else(|| {
                                let raw = ub.raw() as usize;
                                opt_p2.trace_inputargs.get(raw).and_then(|o| o.ty())
                            })
                            .unwrap_or_else(|| {
                                panic!(
                                    "duplicate short-preamble used_box {:?} has untyped variant; \
                                     RPython label/inputarg boxes carry Int/Ref/Float type \
                                     intrinsically (resoperation.py:1544 inputarg_from_tp)",
                                    ub
                                )
                            });
                        if matches!(tp, Type::Void) {
                            panic!(
                                "duplicate short-preamble used_box {:?} has void type; \
                                 RPython label/inputarg boxes must carry Int/Ref/Float \
                                 type (resoperation.py:1544 inputarg_from_tp)",
                                ub
                            );
                        }
                        let alias = OpRef::input_arg_typed(next_fresh, tp);
                        current_label_args.push(alias);
                        next_fresh += 1;
                    }
                }
            }
            let opt_unroll = OptUnroll::new();
            // Use Phase 2's final context for virtual state matching.
            let mut jump_ctx = opt_p2.final_ctx.take().unwrap_or_else(|| {
                // opencoder.py:259 inputarg_from_tp parity — Phase 2 inputargs
                // carry the same producer-side types as Phase 1's; fall back
                // to Ref when types are unavailable.
                let types: Vec<majit_ir::Type> = opt_p2
                    .trace_inputargs
                    .get(..body_num_inputs)
                    .map(|s| {
                        s.iter()
                            .map(|o| o.ty().unwrap_or(majit_ir::Type::Ref))
                            .collect()
                    })
                    .unwrap_or_else(|| vec![majit_ir::Type::Ref; body_num_inputs]);
                crate::optimizeopt::OptContext::with_inputarg_types(32, &types)
            });

            // unroll.py:151-158: jump_to_existing_trace(force_boxes=False)
            // RPython: except InvalidLoop → jump_to_preamble immediately,
            // NO retry. The big comment at unroll.py:305-316 explains why
            // continuing after partial inlining is unsafe.
            // unroll.py:153/166 passes `state.runtime_boxes` (the preamble's
            // recorded JUMP args, exported at unroll.py:105) to
            // jump_to_existing_trace, NOT the peeled body's own jump arglist.
            // `runtime_boxes[i]` is read un-forwarded via `getint`/`getref_base`
            // (the box's own observed value), while `boxes[i]` (= forwarded body
            // jump args) drives the virtual-state match (virtualstate.py:646
            // reads them as parallel-but-distinct lists).
            //
            // `exported_runtime_boxes` is the recorded JUMP arglist threaded
            // through ExportedState.runtime_boxes (set in the Phase-1 export
            // above, carried across a retrace import). Those Phase-1 OpRefs
            // resolve in Phase-2's jump_ctx via cross-phase find_producer_op
            // (phase1_emit_ops / input_ops) and carry the boxes' own observed
            // runtime values. Fall back to the body jump args only when the
            // channel is empty (a trace with no recorded JUMP), where the
            // generate_guards length assert would otherwise fail.
            let runtime_boxes = if exported_runtime_boxes.len() == body_jump_args.len() {
                exported_runtime_boxes.clone()
            } else {
                body_jump_args.clone()
            };
            let mut invalid_loop = false;
            let mut jumped = if skip_jump_to_existing {
                false
            } else {
                let did_jump = opt_unroll
                    .jump_to_existing_trace(
                        &body_jump_args,
                        Some(&current_label_args),
                        &mut self.target_tokens,
                        &mut opt_p2,
                        &mut jump_ctx,
                        false,
                        &runtime_boxes,
                    )
                    .is_none();
                if let Some(reason) = jump_ctx.take_invalid_loop() {
                    if crate::log_jtet_enabled() {
                        eprintln!(
                            "[jit][jte] InvalidLoop during force_boxes=false: {}",
                            reason.0
                        );
                    }
                    // unroll.py:154-158: except InvalidLoop →
                    // jump_to_preamble, skip retry
                    invalid_loop = true;
                    false
                } else {
                    did_jump
                }
            };
            if crate::majit_log_enabled() {
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
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] Retracing ({}/{})",
                            self.retraced_count, self.retrace_limit
                        );
                    }
                    // unroll.py:164-168: force_boxes=True, except InvalidLoop: pass
                    let did_jump = opt_unroll
                        .jump_to_existing_trace(
                            &body_jump_args,
                            Some(&current_label_args),
                            &mut self.target_tokens,
                            &mut opt_p2,
                            &mut jump_ctx,
                            true,
                            &runtime_boxes,
                        )
                        .is_none();
                    jumped = if let Some(reason) = jump_ctx.take_invalid_loop() {
                        if crate::log_jtet_enabled() {
                            eprintln!(
                                "[jit][jte] InvalidLoop during force_boxes=true retrace: {}",
                                reason.0
                            );
                        }
                        false // unroll.py:167-168: except InvalidLoop: pass
                    } else {
                        did_jump
                    };
                } else {
                    // unroll.py:220-226: limit reached, try force_boxes=true
                    let did_jump = opt_unroll
                        .jump_to_existing_trace(
                            &body_jump_args,
                            Some(&current_label_args),
                            &mut self.target_tokens,
                            &mut opt_p2,
                            &mut jump_ctx,
                            true,
                            &runtime_boxes,
                        )
                        .is_none();
                    jumped = if let Some(reason) = jump_ctx.take_invalid_loop() {
                        if crate::log_jtet_enabled() {
                            eprintln!(
                                "[jit][jte] InvalidLoop during force_boxes=true limit: {}",
                                reason.0
                            );
                        }
                        false // unroll.py:224-225: except InvalidLoop: pass
                    } else {
                        did_jump
                    };
                    if !jumped {
                        // unroll.py:228: "Retrace count reached, jumping to preamble"
                        crate::debug::log_one(
                            "jit-tracing",
                            "Retrace count reached, jumping to preamble",
                        );
                        // jumped stays false → jump_to_preamble below
                    }
                }
            }
            if jumped && redirected_tail_ops.is_empty() {
                // Only take jump_ctx ops if we don't already have
                // a self-loop Jump from the retrace path.
                redirected_tail_ops = std::mem::take(&mut jump_ctx.new_operations);
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
                    .and_then(|o| o.getdescr())
                    .map(|d| d.index());
                if redirected_jump_descr_idx != current_body_descr_idx {
                    // RPython parity: the Cranelift backend can't jump to
                    // code from a previous compilation (separate function).
                    // Fall back to jump_to_preamble, matching RPython's
                    // behavior when the target isn't reachable (unroll.py:228).
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] jump_to_existing_trace: external target {:?} != body {:?}, falling back to preamble",
                            redirected_jump_descr_idx, current_body_descr_idx
                        );
                    }
                    redirected_tail_ops.clear();
                    jumped = false;
                }
            }
            opt_p2.final_ctx = Some(jump_ctx);
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
        if crate::majit_log_enabled() {
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
            if crate::majit_log_enabled() {
                let body_jump_arity = body_terminal_op.as_ref().map(|j| j.num_args()).unwrap_or(0);
                eprintln!(
                    "[jit] jump_to_preamble: body_jump_args={} preamble_arity={} start_label_args={:?}",
                    body_jump_arity, preamble_arity, exported_renamed_inputargs,
                );
            }
            if let Some(mut end_jump) = body_terminal_op {
                end_jump.setdescr(preamble_target.as_jump_target_descr());
                if let Some(mut final_ctx) = opt_p2.final_ctx.take() {
                    // unroll.py:238-242 parity: jump_to_preamble retargets
                    // the live end_jump and routes it through
                    // send_extra_operation, preserving any force_box /
                    // partial-inline operations already appended to
                    // _newoperations.
                    //
                    // unroll.py:242 lets send_extra_operation raise
                    // InvalidLoop; this function returns Result, so `?`
                    // carries it out (final_ctx is abandoned with the
                    // discarded trace).
                    opt_p2.send_extra_operation(&end_jump, &mut final_ctx)?;
                    let redirected_tail_ops: Vec<majit_ir::OpRc> =
                        std::mem::take(&mut final_ctx.new_operations);
                    opt_p2.final_ctx = Some(final_ctx);
                    body_ops = splice_redirected_tail(&body_ops, &redirected_tail_ops);
                } else {
                    body_ops = replace_terminal_jump(&body_ops, end_jump);
                }
            } else {
                body_ops = Self::jump_to_preamble(&body_ops, &preamble_target);
            }
            crate::debug::log_one(
                "jit-tracing",
                "jump_to_preamble: body JUMP retargeted to start descr",
            );
        } else if !redirected_tail_ops.is_empty() {
            body_ops = splice_redirected_tail(&body_ops, &redirected_tail_ops);
        } else {
            crate::debug::log_one(
                "jit-tracing",
                "jump_to_existing_trace: body JUMP → self-loop",
            );
        }

        // ── Assembly (compile.py:310-338) ──
        let sp_used_boxes: Vec<OpRef> = sp.used_boxes.clone();
        let sp_jump_args: Vec<OpRef> = sp.jump_args.clone();
        let mut combined = assemble_peeled_trace_with_jump_args(
            &p1_ops,
            &body_ops,
            &label_args,
            &exported_renamed_inputargs,
            &sp_used_boxes,
            &sp_jump_args,
            p2_ni,
            phase2_inputarg_base,
            jump_to_self,
            &imported_short_aliases,
            &consts_p2,
            self.target_tokens
                .first()
                .map(|target| target.as_jump_target_descr()),
            self.target_tokens
                .last()
                .map(|target| target.as_jump_target_descr()),
            &exported_end_args,
            opt_p2
                .final_ctx
                .as_mut()
                .expect("assemble peeled trace requires Phase 2 OptContext for Box.type parity"),
        );
        // RPython Box parity: drop duplicate-position ops. In RPython
        // each Box is unique so collisions can't happen. Keep first.
        {
            let mut seen: indexmap::IndexSet<u32> = indexmap::IndexSet::new();
            combined.retain(|op| {
                if op.pos.get().is_none() || op.result_type() == Type::Void {
                    return true;
                }
                seen.insert(op.pos.get().raw())
            });
        }
        crate::optimizeopt::optimizer::sanitize_backend_constants_for_ops(
            combined.iter().map(|op| &**op),
            &mut consts_p2,
        );
        if crate::debug::have_debug_prints() {
            let _s = crate::debug::scope("jit-log-opt-loop");
            crate::debug::debug_print("--- peeled trace (assembled) ---");
            for line in majit_ir::format_trace(&combined, &consts_p2).lines() {
                crate::debug::debug_print(line);
            }
            let mut sorted_consts: Vec<_> = consts_p2
                .iter()
                .map(|(k, v)| (*k, v.clone()))
                .collect::<Vec<_>>();
            sorted_consts.sort_by_key(|(k, _)| *k);
            crate::debug::debug_print(&format!("consts_p2: {sorted_consts:?}"));
        }
        *constants = consts_p2;
        Ok((combined, p2_ni))
    }

    /// RPython compile.py uses the optimized loop state's inputargs contract,
    /// not a stale input counter. When majit falls back to the phase-1 trace,
    /// derive the live loop arity from the actual closing Label/Jump.
    pub fn closing_loop_contract_arity<T: AsRef<Op>>(ops: &[T], fallback: usize) -> usize {
        closing_loop_contract_arity(ops, fallback)
    }

    /// Count the guards in an optimized trace (for retrace_limit checks).
    pub fn count_guards<T: AsRef<Op>>(ops: &[T]) -> u32 {
        ops.iter()
            .filter(|op| op.as_ref().opcode.is_guard())
            .count() as u32
    }

    /// unroll.py: _map_args(mapping, arglist)
    /// Remap a list of OpRefs through a forwarding mapping.
    /// Constant OpRefs are left unchanged because they are not remapped.
    pub fn map_args(mapping: &indexmap::IndexMap<OpRef, OpRef>, args: &[OpRef]) -> Vec<OpRef> {
        args.iter()
            .map(|&arg| mapping.get(&arg).copied().unwrap_or(arg))
            .collect()
    }

    /// unroll.py: _check_no_forwarding(lsts)
    /// Debug assertion: verify no OpRef in the lists has been forwarded.
    pub fn check_no_forwarding(ctx: &crate::optimizeopt::OptContext, oprefs: &[OpRef]) -> bool {
        oprefs.iter().all(|&r| ctx.get_replacement_opref(r) == r)
    }

    /// unroll.py: disable_retracing_if_max_retrace_guards(ops, target_token)
    /// If the trace has too many guards, disable retracing for this location.
    /// Returns true if retracing was disabled.
    pub fn disable_retracing_if_max_retrace_guards<T: AsRef<Op>>(
        ops: &[T],
        max_retrace_guards: u32,
    ) -> bool {
        let guard_count = Self::count_guards(ops);
        guard_count > max_retrace_guards
    }

    /// unroll.py: get_virtual_state(args)
    /// Build a VirtualState from the optimizer's current knowledge about args.
    pub fn get_virtual_state(
        args: &[OpRef],
        ctx: &crate::optimizeopt::OptContext,
    ) -> crate::optimizeopt::virtualstate::VirtualState {
        crate::optimizeopt::virtualstate::export_state(args, ctx)
    }
}

fn closing_loop_contract_arity<T: AsRef<Op>>(ops: &[T], fallback: usize) -> usize {
    ops.iter()
        .map(|op| op.as_ref())
        .rev()
        .find_map(|op| match op.opcode {
            OpCode::Label | OpCode::Jump => Some(op.num_args()),
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
    /// Positions in `end_args` that feed the compact non-virtual LABEL args.
    pub label_source_positions: Vec<usize>,
    /// Args for the next iteration (before forcing). unroll.py:467
    /// `next_iteration_args = end_args` — the SAME canonical box objects
    /// (producer-bound / const [`Operand`]s) used as `exported_infos` keys,
    /// so the import-state lookup is an identity hit.
    pub next_iteration_args: Vec<Operand>,
    /// Types of end_args as determined by Phase 1 optimization.
    /// Used by Phase 2 import_state to propagate unboxed types.
    pub end_arg_types: Vec<Type>,
    /// Virtual state at the loop boundary.
    pub virtual_state: crate::optimizeopt::virtualstate::VirtualState,
    /// unroll.py:548 ExportedState.exported_infos — optimizer knowledge from preamble.
    /// Maps OpRef → info for all args including virtual field contents.
    ///
    /// RPython stores one of `PtrInfo` / `IntBound` / `FloatConstInfo` per box,
    /// dispatched via `isinstance` in `setinfo_from_preamble` (unroll.py:53-98).
    /// Majit uses the existing `OpInfo` enum (info.rs:137) as the discriminated
    /// union of these three cases. Keyed by box object identity ([`Operand`]
    /// `Eq` = producer / const-cell `Rc::ptr_eq`), matching RPython's plain
    /// box-keyed dict.
    pub exported_infos: indexmap::IndexMap<Operand, crate::optimizeopt::info::OpInfo>,
    /// RPython shortpreamble.py: produced short boxes in preamble order.
    /// This preserves the original preamble ops so the active path can build
    /// short preambles without re-extracting them from the peeled trace.
    pub exported_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
    /// RPython unroll.py:466-477 `short_boxes = sb.create_short_boxes(...)`
    /// stored directly on ExportedState. Consumer sites
    /// (`build_imported_short_preamble`, `import_short_preamble_state`)
    /// iterate this list verbatim. Pyre derives this eagerly from
    /// `exported_short_boxes + label_args + short_inputargs` at export time
    /// (`shortpreamble.py:269-270 ShortBoxes.create_short_boxes` parity).
    pub short_boxes: Vec<(OpRef, crate::optimizeopt::shortpreamble::ProducedShortOp)>,
    /// TODO: producer-side const value snapshot for any
    /// const-namespace OpRef referenced by `short_boxes` op args. RPython
    /// gets this for free because `Const` Box objects carry their value as
    /// an attribute and persist across optimization phases. In majit, OpRef
    /// is a flat trace-local index — `OpRef::from_const(N)` only resolves
    /// to a value through `OptContext::const_pool[N]`, and consumer-side ctx
    /// (Phase 2 / bridge) may not have the producer's slot N populated.
    ///
    /// The legacy `ExportedShortOp::Pure { args: Vec<ExportedShortArg> }`
    /// path captured constant args inline as `ExportedShortArg::Const
    /// { source, value }`. Phase B.1's polymorphic `ProducedShortOp::
    /// produce_op` reads raw OpRef args, so we lift the value snapshot to
    /// this sidecar map keyed by source OpRef. `produce_op` /
    /// `classify_short_arg` (shortpreamble.rs) read this map first when
    /// classifying a Const arg.
    ///
    /// Convergence path: this map disappears once consumer ctxs are
    /// guaranteed to have producer constants pre-seeded (production
    /// already does this at `optimizer.rs:1927`; bridges and unit tests
    /// remain the open cases). At that point `classify_short_arg` can
    /// read the box's `const_value()` exclusively.
    pub short_box_const_values: indexmap::IndexMap<OpRef, majit_ir::Value>,
    /// Short preamble builder for bridge entry.
    pub short_preamble: Option<crate::optimizeopt::shortpreamble::ShortPreamble>,
    /// Renamed inputargs from the preamble. Each OpRef is a typed
    /// `InputArg{Int,Float,Ref}` variant carrying its `.type` intrinsically
    /// (history.py:220), so consumers read the type via `OpRef::ty()` —
    /// RPython `info.renamed_inputargs` Box parity, no parallel type array.
    pub renamed_inputargs: Vec<OpRef>,
    /// Short inputargs for the short preamble — the renamed inputarg
    /// positions (shortpreamble.py:430 / unroll.py:480), shared with the
    /// renamed operands inside `short_boxes`.
    pub short_inputargs: Vec<OpRef>,
    /// Rooted `InputArgRc` carriers for `short_inputargs`, index-aligned.
    /// Keeping the strong Rc alive across the cross-peel export boundary
    /// preserves the renamed inputarg producers for consumers that need a
    /// bound operand view.
    pub short_inputarg_refs: Vec<majit_ir::InputArgRc>,
    /// unroll.py: runtime_boxes — live values at the original jump point.
    /// Threaded into Phase 2 import as `runtime_boxes` for guard generation.
    /// Default `Vec::new()` until the export site populates it; callers
    /// that need it write the field directly after `ExportedState::new`.
    pub runtime_boxes: Vec<OpRef>,
    /// RPython parity: patchguardop from Phase 1's GuardFutureCondition.
    /// Phase 2's extra_guards (from virtualstate) need rd_resume_position
    /// from this patchguardop (unroll.py:333-336).
    pub patchguardop: Option<majit_ir::Op>,
    /// `OptContext::next_pos` at end of Phase 1 — strict upper bound of
    /// every OpRef Phase 1 allocated, including intermediates folded /
    /// forwarded away. `reserve_pos_typed` skips `materialize_operand_at` on the
    /// zero-inputarg / retrace baselines (`optimizeopt/mod.rs:2026`),
    /// so capturing `ctx.next_pos` at export is the only reliable
    /// floor. Phase 2 / retrace seed their TraceIterator namespace
    /// strictly above this watermark via `opref_high_water()` to keep
    /// the OpRef set disjoint — RPython's object-identity Boxes get
    /// disjointness from Python identity for free; pyre's numeric
    /// OpRefs need an explicit position floor.
    pub phase1_emit_high_water: u32,
    /// `partial_trace.inputargs` (compile.py:362, history.py:509).
    /// Keeps the preamble's `AbstractInputArg` instances alive so that
    /// `compile_retrace`'s `optimize_peeled_loop` reads `_forwarded` off
    /// the same objects (resoperation.py:700
    /// `AbstractInputArg._forwarded`). Each entry is `Rc::clone` cheap;
    /// the vec is bounded by `ExportedState` lifetime. Populated only by
    /// `export_state_with_bounds`.
    pub(crate) partial_trace_inputargs: Vec<majit_ir::InputArgRc>,
    /// `partial_trace.operations` (compile.py:362, history.py:528).
    /// Keeps the preamble's emit `AbstractResOp` instances alive — a
    /// follow-up `compile_retrace` spawns a fresh optimizer that only
    /// sees the imported `ExportedState`, so the OpRc identities the
    /// preamble mutated `_forwarded` on must travel through here
    /// (resoperation.py:233-242 `_forwarded` host).
    pub(crate) partial_trace_operations: Vec<majit_ir::OpRc>,
    /// #173 producer-rooting: keeps the Phase-1 producer `Op` that each
    /// exported short-box `res` is bound to (`res.bound_op()`) alive into
    /// Phase 2. A short-box `res` carries only a `Weak<Op>`; the Phase-1
    /// OptContext that owns the strong `OpRc` drops at the peel boundary, so
    /// without this carry `res` (a `Weak<Op>`) upgrades to a dead producer at
    /// Phase 2 and its accessors panic. Populated at export from
    /// `res.bound_op()` while still alive; InputArg-kind res (`bound_inputarg`,
    /// not `bound_op`) contributes nothing here and is rooted via
    /// `short_inputarg_refs` instead.
    pub(crate) short_box_producer_roots: Vec<majit_ir::OpRc>,
    /// Shadow stack rooting for GcRef values in exported_infos.
    /// (operand key, field kind, shadow stack index). The key is the
    /// `exported_infos` operand key for `InfoPtrInfoConstant` entries; other
    /// field kinds carry the none/empty operand sentinel since they never key
    /// back into `exported_infos`.
    rooted_refs: Vec<(Operand, ExportedGcRefField, usize)>,
    /// Shadow stack slots for every inline `ConstPtr.value` reachable from
    /// this ExportedState's Rust object graph. RPython traces these fields as
    /// normal Const object attributes; pyre records the walk order and copies
    /// the forwarded values back in `refresh_from_gc`.
    rooted_const_ptr_slots: Vec<usize>,
    /// Shadow stack depth at creation. release_roots pops to here.
    shadow_stack_base: usize,
}

// unroll.py:529 `exported_infos - a mapping from ops to infos, including inputargs`
// The per-entry info type is now `OpInfo` (info.rs:137), the discriminated
// union matching RPython's `PtrInfo | IntBound | FloatConstInfo` dispatched
// via `isinstance` in `setinfo_from_preamble` (unroll.py:53-98). The earlier
// majit-only bundle `ExportedValueInfo { constant, ptr_info, int_bound }`
// is removed as part of the Box identity plan Phase D.

/// Identifies which GcRef-bearing field inside an ExportedState
/// is rooted at a particular shadow stack slot.
#[derive(Clone, Copy, Debug)]
enum ExportedGcRefField {
    /// exported_infos[OpRef].ptr_info = PtrInfo::Constant(GcRef)
    InfoPtrInfoConstant,
    /// virtual_state.state[index] = Constant(Value::Ref)
    VirtualStateConstantRef(usize),
    /// short_box_const_values[OpRef] = Value::Ref(...)
    ShortBoxConstValue(OpRef),
    /// `partial_trace_inputargs[index].forwarded = Info(OpInfo::Ptr(PtrInfo::Constant(_)))`.
    /// PyPy `InputArg._forwarded` host (resoperation.py:700).
    PartialTraceInputArgInfoPtrInfoConstant(usize),
    /// `partial_trace_inputargs[index].forwarded = Const(Const::Ref(_), _)`.
    PartialTraceInputArgConstRef(usize),
    /// `partial_trace_operations[index].forwarded = Info(OpInfo::Ptr(PtrInfo::Constant(_)))`.
    /// PyPy `AbstractResOp._forwarded` host (resoperation.py:233-242).
    PartialTraceOpInfoPtrInfoConstant(usize),
    /// `partial_trace_operations[index].forwarded = Const(Const::Ref(_), _)`.
    PartialTraceOpConstRef(usize),
}

impl ExportedState {
    /// unroll.py: ExportedState.__init__
    pub fn new(
        end_args: Vec<OpRef>,
        label_source_positions: Vec<usize>,
        next_iteration_args: Vec<Operand>,
        virtual_state: crate::optimizeopt::virtualstate::VirtualState,
        exported_infos: indexmap::IndexMap<Operand, crate::optimizeopt::info::OpInfo>,
        exported_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
        renamed_inputargs: Vec<OpRef>,
        short_inputargs: Vec<OpRef>,
        short_inputarg_refs: Vec<majit_ir::InputArgRc>,
    ) -> Self {
        // unroll.py:466-477 `sb.create_short_boxes(...)` parity: pyre
        // pre-derives the per-OpRef ProducedShortOp view and stores it
        // directly, matching RPython's `ExportedState.short_boxes`. The
        // label-arg → short-inputarg rename already happened at export time
        // inside `produce_arg` (shortpreamble.py:285/294), so this is a
        // plain GuardOverflow filter + transform of `exported_short_boxes`.
        let short_boxes =
            crate::optimizeopt::shortpreamble::produced_short_boxes_from_exported_boxes(
                &exported_short_boxes,
            );
        ExportedState {
            end_args,
            label_source_positions,
            // unroll.py:467 `next_iteration_args = end_args` — carry the literal
            // Phase-1 boxes (the same Rcs used as `exported_infos` keys) so the
            // import-state lookup is a ptr_eq hit. NOT `from_opref` (which would
            // mint fresh producer-less Rcs and sever the carry identity).
            next_iteration_args,
            end_arg_types: Vec::new(),
            virtual_state,
            exported_infos,
            exported_short_boxes,
            short_boxes,
            short_box_const_values: indexmap::IndexMap::new(),
            short_preamble: None,
            renamed_inputargs,
            short_inputargs,
            short_inputarg_refs,
            runtime_boxes: Vec::new(),
            patchguardop: None,
            phase1_emit_high_water: 0,
            partial_trace_inputargs: Vec::new(),
            partial_trace_operations: Vec::new(),
            short_box_producer_roots: Vec::new(),
            rooted_refs: Vec::new(),
            rooted_const_ptr_slots: Vec::new(),
            shadow_stack_base: majit_gc::shadow_stack::depth(),
        }
        // gcreftracer.py parity: RPython ExportedState is a Python object
        // whose GcRef fields are automatically traced by the GC. In Rust,
        // root_all_gcrefs() must be called at each storage site in LIFO
        // order (longer-lived copy rooted first → lower shadow stack depth).
        // new() does NOT auto-root because the LIFO ordering depends on
        // the caller's storage pattern.
    }

    /// Visit every GC reference slot reachable from this exported optimizer
    /// state. This is the Rust equivalent of PyPy keeping `ExportedState`
    /// inside the GC-traced Python object graph: ConstPtr boxes, op args,
    /// short preamble state, virtual state, and `_forwarded` payloads all
    /// expose their actual mutable storage.
    pub fn walk_const_ptr_refs_mut(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        fn visit_opref(opref: &mut OpRef, visitor: &mut dyn FnMut(&mut GcRef)) {
            if let OpRef::ConstPtr(gcref) = opref {
                visitor(gcref);
            }
        }

        fn visit_oprefs(refs: &mut [OpRef], visitor: &mut dyn FnMut(&mut GcRef)) {
            for r in refs {
                visit_opref(r, visitor);
            }
        }

        fn visit_operands(operands: &[Operand], visitor: &mut dyn FnMut(&mut GcRef)) {
            for o in operands {
                o.walk_const_ptr_refs(visitor);
            }
        }

        fn visit_value(value: &mut Value, visitor: &mut dyn FnMut(&mut GcRef)) {
            if let Value::Ref(gcref) = value {
                visitor(gcref);
            }
        }

        fn visit_op(op: &Op, visitor: &mut dyn FnMut(&mut GcRef)) {
            let mut pos = op.pos.get();
            visit_opref(&mut pos, visitor);
            op.pos.set(pos);
            op.walk_const_ptr_refs_mut(visitor);
        }

        fn visit_op_info(
            info: &mut crate::optimizeopt::info::OpInfo,
            visitor: &mut dyn FnMut(&mut GcRef),
        ) {
            if let crate::optimizeopt::info::OpInfo::Ptr(rc) = info {
                rc.borrow_mut().walk_const_ptr_refs_mut(visitor);
            }
        }

        fn visit_forwarded(
            forwarded: &std::cell::RefCell<majit_ir::forwarding::Forwarded>,
            visitor: &mut dyn FnMut(&mut GcRef),
        ) {
            let mut forwarded = forwarded.borrow_mut();
            match &mut *forwarded {
                majit_ir::forwarding::Forwarded::Info(info) => visit_op_info(info, visitor),
                majit_ir::forwarding::Forwarded::Const(majit_ir::Const::Ref(gcref)) => {
                    visitor(gcref)
                }
                _ => {}
            }
        }

        fn visit_preamble_op(
            entry: &mut crate::optimizeopt::shortpreamble::PreambleOp,
            visitor: &mut dyn FnMut(&mut GcRef),
        ) {
            visit_op(&entry.op, visitor);
            entry.res.walk_const_ptr_refs(visitor);
            if let Some(source) = entry.same_as_source.as_ref() {
                source.walk_const_ptr_refs(visitor);
            }
        }

        fn visit_produced_short_op(
            produced: &mut crate::optimizeopt::shortpreamble::ProducedShortOp,
            visitor: &mut dyn FnMut(&mut GcRef),
        ) {
            visit_op(&produced.preamble_op, visitor);
            produced.res.walk_const_ptr_refs(visitor);
            if let Some(source) = produced.same_as_source.as_ref() {
                source.walk_const_ptr_refs(visitor);
            }
        }

        visit_oprefs(&mut self.end_args, visitor);
        visit_operands(&self.next_iteration_args, visitor);
        self.virtual_state.walk_const_ptr_refs_mut(visitor);
        // Operand hashes by Rc pointer identity, so walking interior GcRefs
        // does not alter the key's hash — safe to borrow key immutably while
        // mutating values. Key GcRefs are walked through interior mutability
        // (Operand wraps Rc<RefCell<..>>).
        for (key, info) in self.exported_infos.iter_mut() {
            key.walk_const_ptr_refs(visitor);
            visit_op_info(info, visitor);
        }
        for entry in &mut self.exported_short_boxes {
            visit_preamble_op(entry, visitor);
        }
        for (key, produced) in &mut self.short_boxes {
            visit_opref(key, visitor);
            visit_produced_short_op(produced, visitor);
        }
        // The key stays an `OpRef`, not an `Operand` (unlike the migrated #108
        // sites): this is a value-keyed const lookup, and an operand is ordered
        // and compared by `Rc` identity (no `Ord`, `Eq` via `Rc::ptr_eq`), so
        // two `ConstPtr` boxes carrying the same gcref would be distinct keys
        // and the dedup the map relies on would break. `visit_opref` already
        // forwards the key's inline gcref canonically, and `visit_value`
        // forwards the stored `Value::Ref`, so the slot is GC-safe as-is.
        // OpRef keys may carry GcRef values whose hash changes after
        // forwarding; drain and reinsert to maintain index integrity.
        let consts: Vec<_> = self.short_box_const_values.drain(..).collect();
        for (mut key, mut value) in consts {
            visit_opref(&mut key, visitor);
            visit_value(&mut value, visitor);
            self.short_box_const_values.insert(key, value);
        }
        if let Some(short_preamble) = self.short_preamble.as_mut() {
            short_preamble.walk_const_ptr_refs_mut(visitor);
        }
        visit_oprefs(&mut self.renamed_inputargs, visitor);
        visit_oprefs(&mut self.short_inputargs, visitor);
        visit_oprefs(&mut self.runtime_boxes, visitor);
        if let Some(patchguardop) = self.patchguardop.as_ref() {
            visit_op(patchguardop, visitor);
        }
        for ia in &self.partial_trace_inputargs {
            visit_forwarded(&ia.forwarded, visitor);
        }
        for op in &self.partial_trace_operations {
            visit_op(op, visitor);
            visit_forwarded(&op.forwarded, visitor);
        }
    }

    pub fn has_shadow_roots(&self) -> bool {
        !self.rooted_refs.is_empty() || !self.rooted_const_ptr_slots.is_empty()
    }

    /// Smallest fresh OpRef that is guaranteed not to collide with any Box
    /// identity stored in this exported preamble state.
    ///
    /// RPython keeps these as object identities in `partial_trace` /
    /// `ExportedState`. Majit uses integer OpRefs, so compile_retrace must
    /// reconstruct this high-water mark before creating the fresh Phase 2
    /// TraceIterator namespace.
    fn opref_high_water(&self) -> u32 {
        let mut high = 0_u32;
        let mut visit = |opref: OpRef| {
            if !opref.is_none() && !opref.is_constant() {
                high = high.max(opref.raw().saturating_add(1));
            }
        };
        let visit_op = |op: &Op, visit: &mut dyn FnMut(OpRef)| {
            visit(op.pos.get());
            for arg in op.getarglist().iter() {
                visit(arg.to_opref());
            }
            if let Some(fail_args) = op.getfailargs() {
                for arg in fail_args {
                    visit(arg.to_opref());
                }
            }
        };
        for arg in &self.end_args {
            visit(*arg);
        }
        for arg in &self.next_iteration_args {
            visit(arg.to_opref());
        }
        for arg in &self.renamed_inputargs {
            visit(*arg);
        }
        for arg in &self.short_inputargs {
            visit(*arg);
        }
        // The original label/virtual positions that `short_inputargs` rename
        // are reached through the `exported_short_boxes` walk below: every
        // `ShortInputArg` entry carries its original as `preamble_op.res` and
        // as the `SameAs*` op's `pos` (visited via `visit_op`). Const-folded
        // slots never survive into Phase 2, so they need no high-water cover.
        for arg in &self.runtime_boxes {
            visit(*arg);
        }

        for (key, info) in &self.exported_infos {
            visit(key.to_opref());
            if let crate::optimizeopt::info::OpInfo::Ptr(rc) = info {
                let children = rc.borrow().visitor_walk_recursive();
                for child in children {
                    visit(child);
                }
            }
        }

        for preamble_op in &self.exported_short_boxes {
            visit_op(&preamble_op.op, &mut visit);
            visit(preamble_op.res.to_opref());
            if let Some(source) = preamble_op.same_as_source.as_ref() {
                visit(source.to_opref());
            }
        }

        if let Some(short_preamble) = &self.short_preamble {
            for r in short_preamble
                .inputargs
                .iter()
                .chain(short_preamble.used_boxes.iter())
                .chain(short_preamble.jump_args.iter())
            {
                visit(*r);
            }
            if let Some(phase1_inputargs) = &short_preamble.phase1_inputargs {
                for r in phase1_inputargs {
                    visit(*r);
                }
            }
            for short_op in &short_preamble.ops {
                visit_op(&short_op.op, &mut visit);
            }
            for ptr_info in short_preamble.inputarg_infos.iter().flatten() {
                for child in ptr_info.visitor_walk_recursive() {
                    visit(child);
                }
            }
        }

        // `phase1_emit_high_water` is `OptContext::next_pos` at end of
        // Phase 1 — covers every emit OpRef the recorder / optimizer
        // allocated, including intermediates forwarded/folded before
        // they could reach a structure-stored field. Phase 2 / retrace
        // must seed `start_fresh` strictly above this watermark to keep
        // the OpRef set disjoint.
        high = high.max(self.phase1_emit_high_water);
        // `partial_trace.inputargs` / `partial_trace.operations` cover
        // every preamble-pass `AbstractValue` whose `_forwarded` must
        // survive into `compile_retrace`; raise the high-water mark
        // above any position they reference so the retrace's fresh
        // OpRef namespace cannot collide with them.
        for ia in &self.partial_trace_inputargs {
            high = high.max((ia.index as u32).saturating_add(1));
        }
        for op in &self.partial_trace_operations {
            let pos = op.pos.get();
            if !pos.is_none() && !pos.is_constant() {
                high = high.max(pos.raw().saturating_add(1));
            }
        }

        high
    }

    /// Push all GcRef values from exported_infos and virtual_state to
    /// shadow stack. gcreftracer.py parity: GC can run between Phase 1
    /// and Phase 2, and between Phase 1 and retrace.
    ///
    /// Must be called explicitly after construction — not auto-called in
    /// new(). This enables LIFO-correct rooting: root the longer-lived
    /// copy first (lower shadow stack depth), then the shorter-lived copy.
    pub fn root_all_gcrefs(&mut self) {
        use crate::optimizeopt::info::{OpInfo, PtrInfo};
        use crate::optimizeopt::virtualstate::VirtualStateInfo;
        // Idempotency: a retrace can feed the same ExportedState back
        // through `imported_state` and call `root_all_gcrefs` again.
        // Without releasing the previous batch first, the earlier roots
        // get pinned (`release_roots` only pops to the most recent
        // `shadow_stack_base`).
        self.release_roots();
        self.shadow_stack_base = majit_gc::shadow_stack::depth();
        // ── exported_infos GcRef fields ──
        let mut keys: Vec<Operand> = self.exported_infos.keys().cloned().collect();
        // Sort for determinism via the key's resolved OpRef position. Const
        // variants (history.py:189-220) sort separately via the inline
        // payload — `.raw()` would panic on inline-Const OpRefs.
        keys.sort_by_key(|k| match k.to_opref() {
            OpRef::ConstInt(v) => (1u8, v as u64),
            OpRef::ConstFloat(v) => (1u8, v.to_bits()),
            OpRef::ConstPtr(v) => (1u8, v.0 as u64),
            other => (0u8, other.raw() as u64),
        });
        for key in keys {
            if let Some(OpInfo::Ptr(rc)) = self.exported_infos.get(&key) {
                let info = rc.borrow();
                match &*info {
                    // RPython ConstPtrInfo: GcRef constant stored directly in
                    // PtrInfo. Root the GcRef so GC keeps it alive.
                    PtrInfo::Constant(gcref) if !gcref.is_null() => {
                        let ss_idx = majit_gc::shadow_stack::push(*gcref);
                        self.rooted_refs.push((
                            key,
                            ExportedGcRefField::InfoPtrInfoConstant,
                            ss_idx,
                        ));
                    }
                    // InstancePtrInfo.known_class is an immortal vtable integer
                    // (ConstInt), never a traced ref — no rooting needed.
                    _ => {}
                }
            }
        }
        // ── virtual_state GcRef fields ──
        // VirtualStateInfo::KnownClass, Virtual{known_class}, Constant(Ref)
        // The rooted_refs `key` slot here just tags the field origin;
        // these entries dispatch on the field kind and never key back into
        // `exported_infos`, so use the `Operand::None` sentinel.
        let dummy_key = Operand::None;
        for (i, entry) in self.virtual_state.state.iter().enumerate() {
            match &entry.info {
                // KnownClass.class_ptr and Virtual.known_class are immortal
                // vtable integers (ConstInt), never traced refs — not rooted.
                VirtualStateInfo::Constant(Value::Ref(gcref)) if !gcref.is_null() => {
                    let ss_idx = majit_gc::shadow_stack::push(*gcref);
                    self.rooted_refs.push((
                        dummy_key.clone(),
                        ExportedGcRefField::VirtualStateConstantRef(i),
                        ss_idx,
                    ));
                }
                _ => {}
            }
        }
        // RPython Const boxes are GC-traced objects. The Rust producer-side
        // const snapshot must therefore root any Ref payload it carries.
        let mut const_keys: Vec<OpRef> = self.short_box_const_values.keys().copied().collect();
        const_keys.sort_by_key(|k| match k {
            OpRef::ConstInt(v) => (1u8, *v as u64),
            OpRef::ConstFloat(v) => (2u8, v.to_bits()),
            OpRef::ConstPtr(v) => (3u8, v.0 as u64),
            _ => (0u8, k.raw() as u64),
        });
        for key in const_keys {
            if let Some(Value::Ref(gcref)) = self.short_box_const_values.get(&key)
                && !gcref.is_null()
            {
                let ss_idx = majit_gc::shadow_stack::push(*gcref);
                self.rooted_refs.push((
                    Operand::None,
                    ExportedGcRefField::ShortBoxConstValue(key),
                    ss_idx,
                ));
            }
        }
        // ── partial_trace `_forwarded` GcRef fields ──
        // `partial_trace.inputargs` / `partial_trace.operations`
        // (compile.py:362) keep every preamble-pass `AbstractValue`
        // instance alive; their `_forwarded` slots may carry GcRef
        // payloads (`PtrInfo::Constant` or `Const::Ref`; `Instance.known_class`
        // is an immortal vtable integer, never a traced ref). Root each so
        // `compile_retrace` chain walks observe live handles.
        for (i, ia) in self.partial_trace_inputargs.iter().enumerate() {
            let forwarded = ia.forwarded.borrow().clone();
            root_forwarded_gcref(
                &forwarded,
                ExportedGcRefField::PartialTraceInputArgInfoPtrInfoConstant(i),
                ExportedGcRefField::PartialTraceInputArgConstRef(i),
                dummy_key.clone(),
                &mut self.rooted_refs,
            );
        }
        for (i, op) in self.partial_trace_operations.iter().enumerate() {
            let forwarded = op.forwarded.borrow().clone();
            root_forwarded_gcref(
                &forwarded,
                ExportedGcRefField::PartialTraceOpInfoPtrInfoConstant(i),
                ExportedGcRefField::PartialTraceOpConstRef(i),
                dummy_key.clone(),
                &mut self.rooted_refs,
            );
        }

        let mut rooted_const_ptr_slots = Vec::new();
        self.walk_const_ptr_refs_mut(&mut |slot| {
            let ss_idx = majit_gc::shadow_stack::push(*slot);
            rooted_const_ptr_slots.push(ss_idx);
        });
        self.rooted_const_ptr_slots = rooted_const_ptr_slots;
    }

    /// Update GcRef values from shadow stack — GC may have moved objects.
    ///
    /// VirtualStateInfo top-level entries are stored as
    /// `Rc<VirtualStateInfoNode>` (so two aliased jump args can share a single
    /// `Rc`); GcRef updates have to replace the entire `Rc` with a fresh
    /// one because the inner enum is immutable.
    ///
    /// **Aliasing preservation**: RPython virtualstate.py:712-728
    /// `VirtualStateConstructor.create_state` caches by Python object
    /// identity, so a GC pause never breaks the "two aliased jump args
    /// share one VirtualStateInfo" invariant. Rust's `Rc<...>` is immutable
    /// after construction, so each per-slot GcRef refresh would otherwise
    /// allocate an independent new `Rc` for every shared slot, breaking
    /// the `Rc::as_ptr` dedup the walker relies on. Snapshot the original
    /// `Rc::as_ptr` per slot, group slots that
    /// originally shared an `Rc`, and after the GcRef updates re-clone a
    /// single canonical `Rc` into every slot of each group so the
    /// post-refresh tree preserves the pre-GC aliasing.
    pub fn refresh_from_gc(&mut self) {
        use crate::optimizeopt::info::PtrInfo;
        use crate::optimizeopt::virtualstate::{VirtualStateInfo, VirtualStateInfoNode};
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
        for &(ref key, ref field, ss_idx) in &self.rooted_refs {
            let updated = majit_gc::shadow_stack::get(ss_idx);
            match field {
                ExportedGcRefField::InfoPtrInfoConstant => {
                    if let Some(info) = self.exported_infos.get_mut(key) {
                        *info = crate::optimizeopt::info::OpInfo::ptr(PtrInfo::Constant(updated));
                    }
                }
                ExportedGcRefField::VirtualStateConstantRef(i) => {
                    if let Some(entry) = self.virtual_state.state.get_mut(*i) {
                        *entry = VirtualStateInfoNode::new_rc(VirtualStateInfo::Constant(
                            Value::Ref(updated),
                        ));
                        virtual_state_dirty = true;
                    }
                }
                ExportedGcRefField::ShortBoxConstValue(source) => {
                    if let Some(value) = self.short_box_const_values.get_mut(source) {
                        *value = Value::Ref(updated);
                    }
                }
                ExportedGcRefField::PartialTraceInputArgInfoPtrInfoConstant(i) => {
                    if let Some(ia) = self.partial_trace_inputargs.get(*i) {
                        refresh_forwarded_ptrinfo_constant(&ia.forwarded, updated);
                    }
                }
                ExportedGcRefField::PartialTraceInputArgConstRef(i) => {
                    if let Some(ia) = self.partial_trace_inputargs.get(*i) {
                        refresh_forwarded_const_ref(&ia.forwarded, updated);
                    }
                }
                ExportedGcRefField::PartialTraceOpInfoPtrInfoConstant(i) => {
                    if let Some(op) = self.partial_trace_operations.get(*i) {
                        refresh_forwarded_ptrinfo_constant(&op.forwarded, updated);
                    }
                }
                ExportedGcRefField::PartialTraceOpConstRef(i) => {
                    if let Some(op) = self.partial_trace_operations.get(*i) {
                        refresh_forwarded_const_ref(&op.forwarded, updated);
                    }
                }
            }
        }
        if virtual_state_dirty {
            // Re-share slots that originally aliased: walk the snapshot
            // map and copy each group's first canonical Rc into every
            // peer slot, restoring the pre-GC `Rc::as_ptr` equivalences.
            let mut canonical_by_old: indexmap::IndexMap<usize, Rc<VirtualStateInfoNode>> =
                indexmap::IndexMap::new();
            for (slot_idx, &old_ptr) in original_ptrs.iter().enumerate() {
                let entry = canonical_by_old.entry_or_insert_with(old_ptr, || {
                    Rc::clone(&self.virtual_state.state[slot_idx])
                });
                if !Rc::ptr_eq(entry, &self.virtual_state.state[slot_idx]) {
                    self.virtual_state.state[slot_idx] = Rc::clone(entry);
                }
            }
            // Rc identities have shifted, so position cells must be
            // re-derived against the canonical state graph.
            self.virtual_state.enum_top_level();
        }
        let rooted_const_ptr_slots = self.rooted_const_ptr_slots.clone();
        let mut root_idx = 0usize;
        self.walk_const_ptr_refs_mut(&mut |slot| {
            if let Some(&ss_idx) = rooted_const_ptr_slots.get(root_idx) {
                *slot = majit_gc::shadow_stack::get(ss_idx);
            }
            root_idx = root_idx.saturating_add(1);
        });
        debug_assert_eq!(
            root_idx,
            rooted_const_ptr_slots.len(),
            "ExportedState ConstPtr root walk changed between root_all_gcrefs and refresh_from_gc"
        );
    }

    /// Release shadow stack roots.
    fn release_roots(&mut self) {
        if !self.rooted_refs.is_empty() || !self.rooted_const_ptr_slots.is_empty() {
            majit_gc::shadow_stack::pop_to(self.shadow_stack_base);
            self.rooted_refs.clear();
            self.rooted_const_ptr_slots.clear();
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
            label_source_positions: self.label_source_positions.clone(),
            next_iteration_args: self.next_iteration_args.clone(),
            end_arg_types: self.end_arg_types.clone(),
            virtual_state: self.virtual_state.clone(),
            exported_infos: self.exported_infos.clone(),
            exported_short_boxes: self.exported_short_boxes.clone(),
            short_boxes: self.short_boxes.clone(),
            short_box_const_values: self.short_box_const_values.clone(),
            short_preamble: self.short_preamble.clone(),
            renamed_inputargs: self.renamed_inputargs.clone(),
            short_inputargs: self.short_inputargs.clone(),
            short_inputarg_refs: self.short_inputarg_refs.clone(),
            runtime_boxes: self.runtime_boxes.clone(),
            patchguardop: self.patchguardop.clone(),
            phase1_emit_high_water: self.phase1_emit_high_water,
            partial_trace_inputargs: self.partial_trace_inputargs.clone(),
            partial_trace_operations: self.partial_trace_operations.clone(),
            short_box_producer_roots: self.short_box_producer_roots.clone(),
            rooted_refs: Vec::new(),
            rooted_const_ptr_slots: Vec::new(),
            shadow_stack_base: majit_gc::shadow_stack::depth(),
        }
    }
}

impl Drop for ExportedState {
    fn drop(&mut self) {
        self.release_roots();
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
    /// Quasi-immutable dependencies discovered during optimization
    /// (`optimizer.py:243` + `heap.py:807-808`). Vec-backed set with
    /// linear-scan dedup.
    pub quasi_immutable_deps: Vec<(u64, u32)>,
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
    /// unroll.py:264-272 disable_retracing_if_max_retrace_guards.
    ///
    /// When the peeled body contains more guards than `max_retrace_guards`,
    /// set `retraced_count = u32::MAX` on the targeting JitCellToken so that
    /// future bridges jump to the preamble instead of requesting a retrace.
    pub fn disable_retracing_if_max_retrace_guards<T: AsRef<Op>>(
        ops: &[T],
        retraced_count: &mut u32,
        max_retrace_guards: u32,
    ) {
        let guard_count = ops
            .iter()
            .filter(|op| op.as_ref().opcode.is_guard())
            .count();
        if guard_count > max_retrace_guards as usize {
            *retraced_count = u32::MAX;
        }
    }

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
        exported_int_bounds: Option<
            &indexmap::IndexMap<majit_ir::operand::Operand, crate::optimizeopt::intutils::IntBound>,
        >,
    ) -> ExportedState {
        // unroll.py:454: end_args = [force_at_the_end_of_preamble(a) ...]
        let end_args: Vec<OpRef> = ctx.preamble_end_args.clone().unwrap_or_else(|| {
            original_label_args
                .iter()
                .map(|&a| ctx.get_replacement_opref(a))
                .collect()
        });
        // unroll.py:457 `virtual_state = self.get_virtual_state(end_args)`
        // — VS captured AFTER `force_box_for_end_of_preamble` and AFTER
        // `flush()`. The caller (`Optimizer::optimize_with_constants_and_inputs_at`)
        // already ran both passes before invoking us, so `end_args` is in
        // the same post-force, post-flush state RPython feeds in.
        let virtual_state = crate::optimizeopt::virtualstate::export_state(&end_args, ctx);
        // unroll.py:459-461: infos = {}; for arg in end_args: _expand_info(arg, infos)
        let mut infos: indexmap::IndexMap<Operand, crate::optimizeopt::info::OpInfo> =
            indexmap::IndexMap::new();
        // Resolve the ONE canonical box per end_arg up front: it is the
        // exported_infos key AND (unroll.py:467 next_iteration_args = end_args)
        // the carried import key, so they are the identical Rc and import_state's
        // lookup is an identity hit for const / inputarg / resop alike. Computing
        // it once keeps a single canonical Const cell per arg (Operand::Const
        // compares by cell identity, so re-resolving is tolerable but one cell
        // mirrors RPython's single Const box object).
        let end_arg_boxes: Vec<Operand> = end_args
            .iter()
            .map(|&a| match ctx.get_box_replacement_operand_opt(a) {
                Some(o) => o,
                // The None arm fires only for an unregistered ResOp position
                // (Const / InputArg always resolve); #157 drained those fires
                // to zero. materialize_operand_at mints+registers a canonical
                // synthetic so this key is identity-stable and a later resolve
                // hits the same host — preserving the exported_infos /
                // next_iteration_args identity carry.
                None => ctx.materialize_operand_at(a),
            })
            .collect();
        for (arg, arg_box) in end_args.iter().zip(end_arg_boxes.iter()) {
            self.expand_info(*arg, arg_box, ctx, exported_int_bounds, &mut infos);
        }
        // unroll.py:462-463 `label_args, virtuals =
        //   virtual_state.make_inputargs_and_virtuals(end_args, self.optimizer)`.
        let (label_args, virtuals) = virtual_state
            .make_inputargs_and_virtuals(&end_args, optimizer, ctx, false)
            .expect("export_state make_inputargs_and_virtuals failed");
        let label_source_positions = end_args
            .iter()
            .enumerate()
            .filter_map(|(index, arg)| (!arg.is_constant()).then_some(index))
            .collect::<Vec<_>>();
        if crate::callee_rca_enabled() {
            eprintln!(
                "[callee-rca][export-state] original_label_args={:?} end_args={:?} \
                 label_source_positions={:?} label_args={:?} virtuals={:?} \
                 renamed_inputargs={:?} num_boxes={} entries={}",
                original_label_args,
                end_args,
                label_source_positions,
                label_args,
                virtuals,
                renamed_inputargs,
                virtual_state.num_boxes(),
                virtual_state.num_entries(),
            );
            for line in callee_rca_virtual_state_summary(&virtual_state) {
                eprintln!("[callee-rca][export-vs] {line}");
            }
        }
        // unroll.py:464-465: for arg in label_args: _expand_info(arg, infos)
        for &arg in &label_args {
            let arg_box = match ctx.get_box_replacement_operand_opt(arg) {
                Some(o) => o,
                // Same canonical-key fallback as end_arg_boxes above: an
                // unregistered ResOp position (drained to zero by #157) mints a
                // canonical registered synthetic, keeping the exported_infos
                // key identity-stable.
                None => ctx.materialize_operand_at(arg),
            };
            self.expand_info(arg, &arg_box, ctx, exported_int_bounds, &mut infos);
        }
        let mut short_args = label_args.to_vec();
        short_args.extend(virtuals);
        // unroll.py:480 `short_inputargs = sb.create_short_inputargs(
        // label_args + virtuals)` — read the ShortBoxes-derived list off the
        // ctx channel. The preview pass computed it from the same
        // `label_args + virtuals` (measured identical across the corpus);
        // paths that never ran the preview (test ExportedState setups) fall
        // back to the local recompute.
        let (short_inputargs, short_inputarg_refs): (Vec<OpRef>, Vec<majit_ir::InputArgRc>) =
            if ctx.exported_short_inputargs.is_empty() {
                // No preview pass ran (test ExportedState setups): mint the fresh
                // renamed InputArg positions directly, mirroring `add_short_input_arg`
                // (shortpreamble.py:257 `OpHelpers.inputarg_from_tp(box.type)`).
                // Each is DISTINCT from its `short_args[i]` original so the
                // rename is a real rename, not an identity no-op. Build the
                // rooted `InputArgRc` pool alongside, matching the preview pass.
                let mut inputargs = Vec::with_capacity(short_args.len());
                let mut refs = Vec::with_capacity(short_args.len());
                for &a in &short_args {
                    let ty = a.ty().unwrap_or_else(|| {
                        panic!("short preamble inputarg {a:?} has no value type")
                    });
                    let pos = ctx.alloc_op_position_typed(ty).raw();
                    let ia = majit_ir::InputArg::from_type_rc(ty, pos);
                    inputargs.push(ia.opref());
                    refs.push(ia);
                }
                (inputargs, refs)
            } else {
                debug_assert_eq!(
                    ctx.exported_short_inputargs.len(),
                    short_args.len(),
                    "preview-pass create_short_inputargs length diverged from the \
                     export-site label_args + virtuals recompute"
                );
                debug_assert_eq!(
                    ctx.exported_short_inputarg_refs.len(),
                    ctx.exported_short_inputargs.len(),
                    "exported_short_inputarg_refs must stay index-aligned with \
                     exported_short_inputargs"
                );
                (
                    ctx.exported_short_inputargs.clone(),
                    ctx.exported_short_inputarg_refs.clone(),
                )
            };
        let exported_short_boxes = ctx.exported_short_boxes.clone();
        // #173 producer-rooting: capture each exported short-box's Phase-1
        // producer Op (`res.bound_op()`) while the Phase-1 ctx still owns the
        // strong `OpRc`, so `res`'s Weak still upgrades after the peel boundary
        // drops that ctx. InputArg-kind res is `bound_inputarg` (None here),
        // rooted via `short_inputarg_refs` instead.
        let short_box_producer_roots: Vec<majit_ir::OpRc> = exported_short_boxes
            .iter()
            .filter_map(|e| e.res.bound_op())
            .collect();
        // unroll.py:466-473 line-by-line:
        //
        //   sb = ShortBoxes()
        //   short_boxes = sb.create_short_boxes(self.optimizer,
        //       renamed_inputargs, label_args + virtuals)
        //   short_inputargs = sb.create_short_inputargs(label_args + virtuals)
        //   for produced_op in short_boxes:
        //       op = produced_op.short_op.res
        //       if not isinstance(op, Const):
        //           self._expand_info(op, infos)
        //
        // RPython expands info from the post-`create_short_boxes` list (the
        // same `short_boxes` that survives into `ExportedState.short_boxes`).
        // pyre's analog is `produced_short_boxes_from_exported_boxes` (which
        // applies the GuardOverflow filter); iterating the raw
        // `exported_short_boxes` here would expand info for entries (e.g.
        // standalone `GuardOverflow`) that PyPy never carries into
        // `short_boxes`, polluting the dict.
        let short_boxes_for_info =
            crate::optimizeopt::shortpreamble::produced_short_boxes_from_exported_boxes(
                &exported_short_boxes,
            );
        // unroll.py:481-484:
        //     for produced_op in short_boxes:
        //         op = produced_op.short_op.res
        //         if not isinstance(op, Const):
        //             self._expand_info(op, infos)
        // Key by the short_op result BOX (`produced_op.res`). Every
        // `produced_short_boxes_from_exported_boxes` invocation clones `res`
        // from the same `exported_short_boxes[k].res` (shortpreamble.rs:3113),
        // so the consumer-side
        // `initialize_imported_short_preamble_builder_from_short_boxes` lookup
        // re-derives the identical Rc and the box-identity lookup hits.
        for (_, produced_op) in &short_boxes_for_info {
            let op = produced_op.res.to_opref();
            if !op.is_constant() {
                // `expand_info` keys the operand-keyed `exported_infos`
                // (#158) on the producer-bound `res`, which is
                // `Rc::ptr_eq`-stable on the producer, so the import lookup
                // still hits.
                self.expand_info(op, &produced_op.res, ctx, exported_int_bounds, &mut infos);
            }
        }

        // RPython unroll.py:467: next_iteration_args = end_args (post-force).
        // Aliased boxes (same resolved OpRef) are handled by export_state's
        // create_state cache + make_inputargs' position_in_notvirtuals dedup.
        // RPython parity: store next_iteration_args in their post-forwarding
        // form so the cache key invariant established by `export_single_value`
        // (which uses `get_box_replacement` as the cache key, virtualstate.rs:2213)
        // survives the phase boundary. `import_state` later runs in a fresh
        // Phase 2 ctx that does NOT inherit Phase 1's forwarding map; if we
        // stored raw `end_args[i]` here, two top-level entries that shared
        // the same `VirtualStateInfoNode` Rc (because their post-resolution
        // targets coincided in Phase 1) would `make_equal_to(source(i), raw_i)`
        // to *different* raw OpRefs, and `make_inputargs` in Phase 2 would
        // see divergent forwarding for the supposedly-shared state slot.
        // RPython sidesteps this naturally because `box._forwarded` persists
        // across phases. virtualstate.py make_inputargs assumes shared state
        // ⇒ same forwarded target.
        // unroll.py:458/467 `end_args = [get_box_replacement(arg) ...]`;
        // `next_iteration_args = end_args`. Carry the SAME canonical box Rcs
        // already resolved as the exported_infos keys (`end_arg_boxes`), so the
        // import lookup is a ptr_eq hit. Each box's `.to_opref()` yields the
        // resolved (post-forwarding) position, so consumers that read `.to_opref()`
        // see the same value the prior `get_replacement_opref` form produced.
        let resolved_next_iteration_args: Vec<Operand> = end_arg_boxes;
        // Phase B B1: `produced_short_boxes` is derived from
        // `exported_short_boxes` lazily at the consumer site
        // (`build_short_preamble_from_produced_boxes` in `import_state`)
        // via `produced_short_boxes_from_exported_boxes`
        // (`shortpreamble.rs:2100`). Storing both shapes on
        // `ExportedState` would risk drift across mutations.
        let mut state = ExportedState::new(
            label_args.clone(),
            label_source_positions,
            resolved_next_iteration_args,
            virtual_state,
            infos,
            exported_short_boxes,
            renamed_inputargs.to_vec(),
            short_inputargs,
            short_inputarg_refs,
        );
        // #173: install the Phase-1 producer roots captured above. Kept off the
        // positional `new` ctor (callers write fields post-construct).
        state.short_box_producer_roots = short_box_producer_roots;
        // `OptContext::next_pos` is the strict upper bound on raw OpRefs
        // Phase 1 allocated, including intermediates folded / forwarded
        // away before any structure-stored field could observe them.
        // `reserve_pos_typed` skips `materialize_operand_at` on the zero-inputarg /
        // retrace baselines (`optimizeopt/mod.rs:2026`), so capturing
        // `ctx.next_pos` at export is the only reliable floor for
        // `opref_high_water()` to feed retrace's `start_fresh`.
        state.phase1_emit_high_water = ctx.next_pos;
        // `partial_trace.inputargs` / `partial_trace.operations`
        // (compile.py:362) identity carriage: snapshot the InputArgRc /
        // OpRc lists the preamble pass mutated `_forwarded` on. A later
        // `compile_retrace` reads `Op.forwarded` / `InputArg.forwarded`
        // directly off the same objects (resoperation.py:233-242 / :700),
        // the PyPy-orthodox identity carry.
        //
        // Walk `ctx.inputargs` (the canonical inputarg-order OpRef list)
        // and pick up the `InputArgRc` for each inputarg's raw OpRef
        // position. `inputarg_refs[idx]` is the canonical strong-owner,
        // populated by either `with_inputarg_types`,
        // `ensure_inputarg_bindings`, or retrace prefix import; reading it
        // directly preserves any `_forwarded` state the Phase 1 passes wrote
        // on the original `InputArg` object (resoperation.py:700 `_forwarded`
        // host). Read it only when it actually matches the inputarg's type
        // and index (the `ensure_inputarg_bindings` resize fills gaps with
        // `new_int(0)` placeholders that would otherwise leak in here);
        // last-resort fresh allocation when it carries no matching Rc (test
        // fixtures / type-mismatch edge cases).
        state.partial_trace_inputargs = ctx
            .inputargs
            .iter()
            .map(|ia_opref| {
                let idx = ia_opref.raw() as usize;
                let want_ty = ia_opref.ty().unwrap_or(majit_ir::Type::Void);
                if let Some(rc) = ctx.inputarg_refs.get(idx).cloned() {
                    if rc.tp == want_ty && rc.index == idx as u32 {
                        return rc;
                    }
                }
                // `inputarg_refs[idx]` is the canonical `InputArgRc` for this
                // OpRef once `ensure_inputarg_bindings` / `materialize_operand_at`'s
                // InputArg placeholder arm have run (both write the canonical
                // `InputArgRc` matching the OpRef's type). Fall through to a
                // fresh `InputArg` allocation for the type-mismatch edge case.
                //
                // Reaching here means the slot was either absent or
                // type-mismatched. In production every canonical inputarg
                // is seeded by `ensure_inputarg_bindings`, so the slot must
                // be *present* (and we only fall through on a genuine type
                // mismatch); an absent slot would mean a fresh allocation
                // silently drops the Phase-1 `_forwarded` state. Trip a
                // future binding regression into a test failure here rather
                // than letting it lose state silently.
                debug_assert!(
                    ctx.inputarg_refs.get(idx).is_some(),
                    "partial_trace_inputargs: inputarg slot {idx} unpopulated at \
                     unroll close; ensure_inputarg_bindings must seed every canonical \
                     inputarg so this fresh allocation is only a type-mismatch repair",
                );
                std::rc::Rc::new(majit_ir::InputArg::from_type(want_ty, idx as u32))
            })
            .collect();
        state.partial_trace_operations = optimizer.phase1_emit_ops.clone();
        // TODO: snapshot producer-side const values for
        // any const-namespace OpRef referenced by `short_boxes` op args.
        // Phase B.2 `ProducedShortOp::produce_op` reads raw OpRefs (not the
        // legacy `ExportedShortArg::Const { source, value }` enum), so we
        // capture the const value here for the consumer's
        // `classify_short_arg` to find. Bridges and unit tests run with
        // a consumer ctx that may not have the producer's const slot
        // populated; without this snapshot, the import path silently
        // skips the short op.
        for (_, produced) in &state.short_boxes {
            for arg in produced.preamble_op.getarglist().iter() {
                if !arg.is_constant() {
                    continue;
                }
                let arg = arg.to_opref();
                if state.short_box_const_values.contains_key(&arg) {
                    continue;
                }
                if let Some(value) = ctx
                    .get_box_replacement_operand_opt(arg)
                    .and_then(|cb| cb.const_value())
                {
                    state.short_box_const_values.insert(arg, value);
                }
            }
        }
        state
    }

    /// unroll.py:432-443: _expand_info
    fn expand_info(
        &self,
        arg: OpRef,
        arg_box: &Operand,
        ctx: &OptContext,
        exported_int_bounds: Option<
            &indexmap::IndexMap<majit_ir::operand::Operand, crate::optimizeopt::intutils::IntBound>,
        >,
        infos: &mut indexmap::IndexMap<Operand, crate::optimizeopt::info::OpInfo>,
    ) {
        // unroll.py:438-443 `_expand_info`:
        //     if arg in infos:
        //         return
        //     if info:
        //         infos[arg] = info
        //         if info.is_virtual():
        //             self._expand_infos_from_virtual(info, infos)
        //
        // Keyed by `arg_box`, the ONE canonical Phase-1 box the caller resolved
        // for this arg (shared verbatim with the `next_iteration_args` carry so
        // the import lookup is a ptr_eq hit). The dual OpRef-key insert is gone:
        // a ptr-stable operand key tracks its position through the shared
        // set_position Cell across compaction, so there is no resolved-vs-original
        // OpRef drift left to bridge.
        if infos.contains_key(arg_box) {
            return;
        }
        let resolved = ctx.get_replacement_opref(arg);
        // RPython stores the entry only when `info` is truthy — a falsy
        // `info` (None) simply skips the insert, so downstream
        // `setinfo_from_preamble_list` at unroll.py:45-51 sees "no entry"
        // and drops any inherited forwarded via `item.set_forwarded(None)`.
        let Some(info) = self.collect_exported_info(resolved, ctx, exported_int_bounds) else {
            return;
        };
        // `arg_box` is the canonical Phase-1 box, which can be position-only
        // (a virtual field box), so read its PtrInfo directly off the box —
        // exactly what `peek_ptr_info` does (`get_box_replacement(false)
        // .ptr_info()`, `self`-independent) — without the `from_opref`
        // position-only panic.
        let arg_pi = arg_box
            .get_box_replacement(false)
            .ptr_info()
            .map(|p| p.clone());
        let has_fields = matches!(
            arg_pi,
            Some(pi) if pi.is_virtual() || !pi.all_items().is_empty()
        );
        infos.insert(arg_box.clone(), info);
        if has_fields {
            self.expand_infos_from_virtual(resolved, ctx, exported_int_bounds, infos);
        }
    }

    /// unroll.py:445-450: _expand_infos_from_virtual
    fn expand_infos_from_virtual(
        &self,
        opref: OpRef,
        ctx: &OptContext,
        exported_int_bounds: Option<
            &indexmap::IndexMap<majit_ir::operand::Operand, crate::optimizeopt::intutils::IntBound>,
        >,
        infos: &mut indexmap::IndexMap<Operand, crate::optimizeopt::info::OpInfo>,
    ) {
        let opref_box = ctx.get_box_replacement_operand_opt(opref);
        // unroll.py:445-450 `_expand_infos_from_virtual`:
        //     items = info.all_items()
        //     for item in items:
        //         if item is None: continue
        //         self._expand_info(item, infos)
        // Key each field by its raw `all_items()` box, NOT a re-resolved OpRef.
        // `peek_ptr_info` returns the SAME `Rc<RefCell<PtrInfo>>` cell whose
        // handle `collect_exported_info` stores as the `exported_infos` value
        // (unroll.rs:4226 `ptr_info_handle`), so the import-side reader
        // (`setinfo_from_preamble_list`) walks the same `v.fields` and the
        // box-identity (`Rc::ptr_eq`) lookup hits.
        let Some(info) = opref_box
            .as_ref()
            .and_then(|b| b.get_box_replacement(false).ptr_info().map(|p| p.clone()))
        else {
            return;
        };
        for (_, entry) in info.all_items() {
            let field_box = entry.as_seen_operand();
            if field_box.to_opref().is_none() {
                continue;
            }
            self.expand_info(
                field_box.to_opref(),
                &field_box,
                ctx,
                exported_int_bounds,
                infos,
            );
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
        let mut target_token = TargetToken::new_loop(token_id);
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
    /// `runtime_boxes`: the concrete runtime values at the jump point
    /// (unroll.py:153/166/207/222), used by generate_guards to emit
    /// GUARD_VALUE when the runtime value matches a known constant. Always a
    /// position-aligned list — both the loop (`state.runtime_boxes`) and the
    /// bridge (`optimize_bridge`'s `runtime_boxes`) paths supply it.
    pub fn jump_to_existing_trace(
        &self,
        jump_args: &[OpRef],
        current_label_args: Option<&[OpRef]>,
        target_tokens: &mut [TargetToken],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
        runtime_boxes: &[OpRef],
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
        runtime_boxes: &[OpRef],
        pre_vs: Option<crate::optimizeopt::virtualstate::VirtualState>,
    ) -> Option<crate::optimizeopt::virtualstate::VirtualState> {
        // optimizer.py:317 `with self.optimizer.cant_replace_guards():`
        // line-by-line — save current `can_replace_guards`, set False
        // for the guarded section, restore on exit. Nested scopes
        // preserve the outer False via the saved token. An InvalidLoop is
        // recorded as a deferred signal on `ctx` (no unwinding), so a plain
        // call + restore preserves the "restore on exit" contract.
        let guard = optimizer.cant_replace_guards();
        let result = self.jump_to_existing_trace_impl(
            jump_args,
            current_label_args,
            target_tokens,
            optimizer,
            ctx,
            force_boxes,
            runtime_boxes,
            pre_vs,
        );
        optimizer.restore_can_replace_guards(guard);
        result
    }

    fn jump_to_existing_trace_impl(
        &self,
        jump_args: &[OpRef],
        current_label_args: Option<&[OpRef]>,
        target_tokens: &mut [TargetToken],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
        runtime_boxes: &[OpRef],
        pre_vs: Option<crate::optimizeopt::virtualstate::VirtualState>,
    ) -> Option<crate::optimizeopt::virtualstate::VirtualState> {
        let mut virtual_state = pre_vs
            .unwrap_or_else(|| crate::optimizeopt::virtualstate::export_state(jump_args, ctx));
        let mut args: Vec<OpRef> = jump_args
            .iter()
            .map(|&a| ctx.get_replacement_opref(a))
            .collect();

        for (tt_idx, target_token) in target_tokens.iter_mut().enumerate() {
            if crate::debug::have_debug_prints() {
                crate::debug::log_one(
                    "jit-tracing",
                    &format!("jump_to_existing trying target_token #{tt_idx}"),
                );
            }
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
                ctx,
                force_boxes,
            ) {
                Ok(guards) => guards,
                Err(()) => {
                    if crate::log_jtet_enabled() {
                        eprintln!(
                            "[jit][jte] target_token #{tt_idx} generate_guards failed (force_boxes={force_boxes})",
                        );
                    }
                    continue;
                }
            };
            // unroll.py:333-338 parity: read patchguardop.rd_resume_position
            // and stamp it onto every extra guard. RPython has no fallback —
            // `optimize_GUARD_FUTURE_CONDITION` (rewrite.py / simplify.py) runs
            // unconditionally on the GUARD_FUTURE_CONDITION emitted at
            // `reached_loop_header` (pyjitpl.py:2969), so by the time
            // `_jump_to_existing_trace` runs `self.optimizer.patchguardop` is
            // always populated. pyre mirrors this: `close_loop_args_at`
            // (trace_opcode.rs:1863) emits the same GFC, and Phase 1 captures
            // it into `patchguardop` (rewrite.rs:2737), which Phase 2 inherits
            // (unroll.rs:513). Skip the read when `extra_guards` is empty —
            // RPython's `for guard in extra_guards.extra_guards` loop body is
            // also skipped in that case.
            for guard_req in &extra_guards {
                let emitted = guard_req.to_ops(&args, ctx);
                if emitted.is_empty() {
                    continue;
                }
                // unroll.py:333 reads `patchguardop` once per loop iteration.
                // Pull it eagerly so the stamp branch below stays cheap and
                // the invariant is asserted before any op streams out.
                let patch = ctx.patchguardop.as_ref().unwrap_or_else(|| {
                    panic!(
                        "unroll.py:333 invariant: patchguardop must be set \
                         when extra_guards is non-empty (target_token #{}, \
                         {} extra guard(s), force_boxes={})",
                        tt_idx,
                        extra_guards.len(),
                        force_boxes,
                    )
                });
                let rd_resume_position = patch.rd_resume_position.get();
                for mut guard_op in emitted {
                    if crate::log_jtet_enabled() {
                        let arg_values: Vec<_> = guard_op
                            .getarglist()
                            .iter()
                            .map(|arg| {
                                let arg = arg.to_opref();
                                (
                                    arg,
                                    ctx.get_box_replacement_operand_opt(arg)
                                        .and_then(|cb| cb.const_value()),
                                )
                            })
                            .collect();
                        eprintln!(
                            "[jit][jte] target_token #{tt_idx} emit guard {:?} from {:?} args={:?}",
                            guard_op.opcode, guard_req, arg_values
                        );
                    }
                    // unroll.py:335-337 line-by-line:
                    //
                    //     if isinstance(guard, GuardResOp):
                    //         guard.rd_resume_position = patchguardop.rd_resume_position
                    //         guard.setdescr(compile.ResumeAtPositionDescr())
                    //     self.optimizer.send_extra_operation(guard)
                    //
                    // intutils.py:1264 IntBound.make_guards interleaves
                    // INT_GE/INT_LE/INT_AND (non-GuardResOp) with their
                    // GUARD_TRUE/GUARD_VALUE pairs; only the latter inherit
                    // resume metadata. Mirror the type filter via `is_guard()`.
                    if guard_op.opcode.is_guard() {
                        guard_op.rd_resume_position.set(rd_resume_position);
                        guard_op.setdescr(crate::optimizeopt::make_resume_at_position_descr());
                    }
                    // unroll.py:338 lets send_extra_operation raise InvalidLoop
                    // (only VirtualStatesCantMatch is caught around it). This
                    // function returns Option (flag convention), and
                    // propagate consumed the flag into the Err, so re-defer it
                    // for the caller's take_invalid_loop barrier and abort.
                    if let Err(e) = optimizer.send_extra_operation(&guard_op, ctx) {
                        ctx.signal_invalid_loop(e.0);
                        return None;
                    }
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
                    if crate::log_jtet_enabled() {
                        eprintln!(
                            "[jit][jte] target_token #{tt_idx} make_inputargs failed (force_boxes={force_boxes})",
                        );
                    }
                    if force_boxes {
                        args = jump_args
                            .iter()
                            .map(|&a| ctx.get_replacement_opref(a))
                            .collect();
                        virtual_state = crate::optimizeopt::virtualstate::export_state(&args, ctx);
                    }
                    continue;
                }
            };
            // unroll.py:354 `short_jump_args = args + virtuals`: the short
            // preamble's label (`short[0].getarglist()`, unroll.py:374) carries
            // one inputarg per `args` entry AND one per `virtuals` entry, so
            // inline_short_preamble's `len(short_inputargs) == len(jump_args)`
            // (unroll.py:393) holds.
            //
            // Pyre diverges for a virtualizable-frame loop whose locals are
            // carried UNBOXED: the reduced loop-body target token
            // (LoopTargetDescr(N)) finalizes its short-preamble inputargs to the
            // non-virtual loop label only (`ShortBoxes::with_label_args`,
            // shortpreamble.rs:558) and reconstructs the boxed-int forms from
            // those args via its own ops — it does NOT expect the virtual-box
            // slots as inputargs. Appending `virtuals` then makes
            // `short_jump_args` longer than the target's `short_inputargs` and
            // trips the contract (the `except`-handler bridge: target_args=4,
            // virtuals=2 [boxed s/i], but sp.inputargs=4 → 6 != 4 InvalidLoop).
            //
            // Match the target's actual short-inputarg arity: append `virtuals`
            // only when the target short preamble has slots for them. The
            // loop-self-build path (sp.inputargs already includes the virtual
            // slots) is unchanged; only the reduced-target bridge retarget that
            // would otherwise InvalidLoop is affected.
            let mut short_jump_args = target_args.clone();
            let target_expects_virtuals = target_token
                .short_preamble
                .as_ref()
                .map_or(true, |sp| sp.inputargs.len() != short_jump_args.len());
            if target_expects_virtuals {
                short_jump_args.extend(virtuals);
            }

            // Ensure jump_args carry PtrInfo from Phase 2 body.
            // RPython Box identity preserves info across forwarding.
            // In majit, forwarding target may lack PtrInfo — propagate
            // from the original label arg (before forwarding).
            if let Some(label) = current_label_args {
                for (i, &jump_arg) in short_jump_args.iter().enumerate() {
                    let resolved_has_info = ctx
                        .get_box_replacement_operand_opt(jump_arg)
                        .as_ref()
                        .map_or(false, |b| ctx.has_ptr_info(b));
                    if !resolved_has_info {
                        // Try label arg at same index
                        if let Some(&label_arg) = label.get(i) {
                            let label_box = ctx.get_box_replacement_operand_opt(label_arg);
                            if let Some(info) =
                                label_box.as_ref().and_then(|b| ctx.peek_ptr_info(b))
                            {
                                ctx.ensure_ptr_info_preserve_forwarding(jump_arg, info);
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
                        if !builder.setup(&sp, label_args, ctx) {
                            target_token.short_preamble_producer = Some(builder);
                            // Drop the peeled trace and let the caller fall back
                            // to jump_to_preamble. Recorded as a deferred
                            // InvalidLoop signal (checked right after this
                            // returns) so no unwinding is needed.
                            ctx.signal_invalid_loop("short preamble has unresolvable Phase 1 args");
                            return None;
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
                            // history.py:227/268/314 — `Const{Int,Float,Ptr}.value`
                            // rides inline on the OpRef. Production no longer
                            // seeds `ctx.const_pool`
                            // (`merge_backend_constants_from_ctx` asserts the
                            // pool is empty at export), so the cross-compile
                            // `loop_constants` snapshot is no longer built:
                            // short-preamble ops embed the Const value
                            // directly in `op.args`, mirroring RPython's
                            // `shortpreamble.py` which has no parallel side
                            // table.
                            target_token.short_preamble =
                                Some(builder.build_short_preamble_struct());
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
                    if crate::debug::have_debug_prints() {
                        crate::debug::log_one(
                            "jit-tracing",
                            &format!("jte-isp done, extra_len={}", extra.len()),
                        );
                    }
                }
            }

            // A short-preamble replay that hit an unresolvable Phase 1 arg or
            // a structurally incompatible mapping recorded a deferred
            // InvalidLoop signal above; abandon this jump attempt so the caller
            // falls back to jump_to_preamble.
            if ctx.has_pending_invalid_loop() {
                return None;
            }

            // unroll.py:357-359: emit JUMP to target
            let mut jump_args = target_args;
            jump_args.extend(extra);
            let mut jump_args_box_operand: Vec<majit_ir::operand::Operand> =
                Vec::with_capacity(jump_args.len());
            for a in &jump_args {
                jump_args_box_operand.push(ctx.materialize_operand_at(*a));
            }
            let mut jump = Op::new(OpCode::Jump, &jump_args_box_operand);
            jump.setdescr(target_token.as_jump_target_descr());
            // unroll.py:357 lets send_extra_operation raise InvalidLoop. This
            // function returns Option (flag convention); propagate consumed
            // the flag into the Err, so re-defer it for the caller's
            // take_invalid_loop barrier (the None return below then reads as an
            // aborted jump rather than a successful one).
            if let Err(e) = optimizer.send_extra_operation(&jump, ctx) {
                ctx.signal_invalid_loop(e.0);
            }
            return None; // successfully jumped (or aborted via deferred InvalidLoop)
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
        // history.py:227/268/314 — `Const{Int,Float,Ptr}.value` is inline on
        // the OpRef. All production short-preamble capture sites early-return
        // on Const OpRefs (shortpreamble.rs:1897 `capture_const`), so
        // `short_preamble.constants` is empty along every production export
        // and the bridge has no const-pool entries to replay through
        // `ctx.const_pool`. The export-side invariant at
        // `optimizer::merge_backend_constants_from_ctx` asserts the same
        // pool-empty contract; mirror it at the producer entry so any
        // re-introduction of const-pool seeding fails loudly here
        // rather than silently leaking into a backend that no longer
        // consumes `ctx.const_pool`.
        debug_assert!(
            short_preamble.constants.is_empty(),
            "inline_short_preamble: short_preamble.constants must be empty in production — \
             history.py:227/268/314 inline-Const is the single source of truth; \
             a non-empty entry indicates a stale legacy producer that would never reach \
             the backend (merge_backend_constants_from_ctx no longer exports ctx.const_pool)"
        );

        let mut mapping: indexmap::IndexMap<OpRef, OpRef> = indexmap::IndexMap::new();

        // unroll.py:393 `assert len(short_inputargs) == len(jump_args)` —
        // the mapping below is positional, so a length mismatch misaligns
        // every seeded pair (a recurring red maps to the wrong slot and the
        // back edge carries the loop-entry value as if it were invariant).
        // RPython guarantees equality by construction; reaching here with a
        // mismatch means the exported short inputargs cover a different
        // label shape than the target's make_inputargs output. Fall back to
        // jump_to_preamble via InvalidLoop instead of seeding a misaligned
        // mapping.
        if short_preamble.inputargs.len() != jump_args.len() {
            if crate::optimizeopt::majit_log_enabled() {
                eprintln!(
                    "[jit] inline_short_preamble: short_inputargs len {} != jump_args len {} — InvalidLoop",
                    short_preamble.inputargs.len(),
                    jump_args.len()
                );
            }
            ctx.signal_invalid_loop(
                "inline_short_preamble: short_inputargs/jump_args arity mismatch",
            );
            return Vec::new();
        }
        for (i, short_inputarg) in short_preamble.inputargs.iter().enumerate() {
            let short_inputarg = *short_inputarg;
            if let Some(&jump_arg) = jump_args.get(i) {
                mapping.insert(short_inputarg, jump_arg);
                // RPython: jump_arg Box inherits info via identity.
                // In majit, propagate PtrInfo from short_inputarg (which
                // has info from Phase 1 export) to the resolved jump_arg.
                // shortpreamble.py:414-425 parity: propagate PtrInfo from
                // Phase 1 export to jump_args so guards are redundant.
                let resolved_has_info = ctx
                    .get_box_replacement_operand_opt(jump_arg)
                    .as_ref()
                    .map_or(false, |b| ctx.has_ptr_info(b));
                if !resolved_has_info {
                    let jump_box = ctx.get_box_replacement_operand_opt(jump_arg);
                    let short_box = ctx.get_box_replacement_operand_opt(short_inputarg);
                    let info = jump_box
                        .as_ref()
                        .and_then(|b| ctx.peek_ptr_info(b))
                        .or_else(|| short_box.as_ref().and_then(|b| ctx.peek_ptr_info(b)))
                        .or_else(|| {
                            short_preamble
                                .inputarg_infos
                                .get(i)
                                .and_then(|opt| opt.clone())
                        });
                    if let Some(info) = info {
                        ctx.ensure_ptr_info_preserve_forwarding(jump_arg, info);
                    }
                }
            }
        }

        // Map the second short-inputarg domain that the `inputargs → jump_args`
        // seeding above does NOT cover. pyre keeps disjoint Phase-1/Phase-2 box
        // namespaces (the export serializes boxes to integer OpRef positions), so
        // a short preamble has two inputarg domains — the ORIGINAL Phase-1 label
        // boxes and the RENAMED short_inputargs (shortpreamble.py:256-259) — and
        // `short_preamble.inputargs` carries only one of them. The two producers
        // assign OPPOSITE domains:
        //   - the import seeding (this file, the `slot_to_original` block above):
        //     inputargs = renamed Label, phase1_inputargs = originals;
        //   - build_short_preamble_struct (shortpreamble.rs `if inputargs !=
        //     &short_inputargs`): inputargs = original label_args, phase1_inputargs
        //     = renamed short_inputargs.
        // In the second (Extended/active-builder) case the short ops reference the
        // RENAMED boxes — produce_arg embedded them at export (#217) — which are
        // NOT in `inputargs` (= originals there), so THIS leg is the sole mapper
        // that resolves them to jump_args. Measured LOAD-BEARING: 176 consumptions
        // across the check.py corpus on both backends, all GuardNonnullClass /
        // GetfieldGcPure over ref inputargs (the redundant-guard / pure-getfield
        // elimination on loop-carried references). Dropping this leg routes those
        // args to the None → InvalidLoop arm below — correct, but it loses the loop
        // inlining for those traces. (The originals-domain seeding from the import
        // path is, by contrast, empirically inert: 0 of the 176 consumptions were
        // original boxes, because post-#217 no short op references an original.)
        // Upstream needs no analog: its single Box namespace is stable across the
        // boundary, so the renamed inputarg IS the object the short ops reference
        // and unroll.py:393-396 seeds only short_inputargs → jump_args.
        // CONVERGENCE (issue #217 step 5 "known blocker"): make
        // build_short_preamble_struct build the Label from the RENAMED
        // short_inputargs (matching the import-seeding convention, #217, and
        // upstream); then `inputargs` covers the renamed short-op args directly and
        // this leg plus the phase1_inputargs field can be removed.
        if let Some(ref phase1) = short_preamble.phase1_inputargs {
            for (i, phase1_inputarg) in phase1.iter().enumerate() {
                let phase1_inputarg = *phase1_inputarg;
                if let Some(&jump_arg) = jump_args.get(i) {
                    if !mapping.contains_key(&phase1_inputarg) {
                        mapping.insert(phase1_inputarg, jump_arg);
                    }
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
                if crate::majit_log_enabled() {
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
                for i in 0..new_op.num_args() {
                    let arg = new_op.arg(i);
                    // unroll.py:367: isinstance(box, Const) — true Const objects
                    // only (ConstInt/ConstPtr/ConstFloat). make_constant'd values
                    // are NOT Const objects — they are regular boxes with forwarded
                    // set to a Const, so they must go through the mapping.
                    if arg.is_constant()
                        || short_preamble
                            .constants
                            .iter()
                            .any(|(k, _)| *k == arg.to_opref().raw())
                    {
                        continue;
                    }
                    // unroll.py:404: _map_args — non-Const must be in mapping.
                    // RPython: mapping is complete (seeded from short_inputargs →
                    // jump_args, extended by mapping[sop] = op). Missing keys
                    // indicate a structural mismatch (e.g., cross-loop bridge
                    // with incompatible short preamble). Raise InvalidLoop.
                    match mapping.get(&arg.to_opref()) {
                        // unroll.py:367: mapping values are the replayed op
                        // objects; bind to the registered producer (memoized
                        // on box_cache) rather than minting an unbound box.
                        Some(&mapped) => new_op.setarg(i, ctx.materialize_operand_at(mapped)),
                        None => {
                            // RPython: _map_args raises KeyError for unmapped
                            // args. This is equivalent to InvalidLoop — the
                            // short preamble is structurally incompatible.
                            // Recorded as a deferred InvalidLoop signal that
                            // jump_to_existing_trace's caller observes, falling
                            // back to jump_to_preamble (unroll.py:154-158,
                            // 209-211).
                            if crate::optimizeopt::majit_log_enabled() {
                                eprintln!(
                                    "[jit] inline_short_preamble: unmapped arg {:?} in {:?} — InvalidLoop",
                                    arg, new_op.opcode
                                );
                            }
                            ctx.signal_invalid_loop(
                                "inline_short_preamble: unmapped arg in short preamble",
                            );
                            return Vec::new();
                        }
                    }
                }
                // unroll.py:405-414: unified guard/non-guard handling.
                // RPython: both guards and non-guards follow the same path:
                //   copy_and_change → mapping[sop] = op → send_extra_operation(op)
                if new_op.opcode.is_guard() {
                    // unroll.py:406-409: copy_and_change with ResumeAtPositionDescr.
                    // The fresh ResumeAtPositionDescr has an empty RdPayload, so
                    // the new guard reads None for every rd_* until
                    // store_final_boxes_in_guard repopulates them from the live
                    // snapshot.
                    new_op.setdescr(crate::optimizeopt::make_resume_at_position_descr());
                    new_op.clearfailargs();
                    new_op.clear_fail_arg_types();
                    // unroll.py:409: op.rd_resume_position = patchguardop.rd_resume_position
                    // RPython: patchguardop is always set (from GUARD_FUTURE_CONDITION).
                    if let Some(ref patch) = ctx.patchguardop {
                        new_op
                            .rd_resume_position
                            .set(patch.rd_resume_position.get());
                    }
                    // history.py:227/268/314 — Const values ride inline
                    // on the OpRef (ConstInt/ConstFloat/
                    // ConstPtr). No pool replay needed.
                    debug_assert!(short_preamble.constants.is_empty());
                } else if let Some(fail_args) = new_op.fail_args_mut() {
                    for arg in fail_args.iter_mut() {
                        if let Some(&mapped) = mapping.get(&arg.to_opref()) {
                            // Measured dead (PYRE_REMAP_PROBE 2026-06-11: 0
                            // fires across check.py corpus + lib tests) —
                            // only guards carry fail_args and guards take the
                            // clearfailargs arm above, matching unroll.py:
                            // 405-414 which has no fail_args handling on the
                            // non-guard path. Rewrite kept as a release
                            // safety net.
                            debug_assert!(
                                false,
                                "non-guard short-preamble op carried fail_args: {:?}",
                                new_op.opcode
                            );
                            *arg = ctx.materialize_operand_at(mapped);
                        }
                    }
                }
                let new_ref = ctx.alloc_op_position_typed(new_op.result_type());
                new_op.pos.set(new_ref);
                // unroll.py:412-414: mapping[sop] = op; i += 1; send_extra_operation(op)
                // RPython sets mapping BEFORE send_extra_operation.
                mapping.insert(sp_op.pos.get(), new_ref);
                replay_index += 1;
                // unroll.py:414 lets send_extra_operation raise InvalidLoop.
                // This function returns Vec (flag convention, as the arity /
                // unmapped-arg signals above); propagate consumed the flag into
                // the Err, so re-defer it for the caller's take_invalid_loop
                // barrier and bail to jump_to_preamble.
                if let Err(e) = optimizer.send_extra_operation(&new_op, ctx) {
                    ctx.signal_invalid_loop(e.0);
                    return Vec::new();
                }
            }

            // unroll.py:417-423: force all except virtuals.
            loop {
                let short_jump_args = current_short_jump_args(short_preamble, ctx);
                let num_short_jump_args = short_jump_args.len();
                // unroll.py:364 `_map_args(mapping, args)`: Const passes
                // through unchanged, non-Const requires mapping.
                let mapped_jump_args: Vec<OpRef> = short_jump_args
                    .iter()
                    .map(|jump_arg| {
                        let mapped = if jump_arg.is_constant() {
                            *jump_arg
                        } else {
                            *mapping.get(jump_arg).expect("mapping missing jump_arg")
                        };
                        ctx.get_replacement_opref(mapped)
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
            // flush may raise InvalidLoop via send_extra_operation; this
            // function returns Vec (flag convention), so re-defer the
            // consumed flag for the caller's take_invalid_loop barrier and
            // bail to jump_to_preamble.
            if let Err(e) = optimizer.flush(ctx) {
                ctx.signal_invalid_loop(e.0);
                return Vec::new();
            }
            // unroll.py:426: done unless "short" has grown again
            if replay_index == current_short_len(short_preamble, ctx) {
                break;
            }
        }

        // RPython: get_box_replacement follows forwarding after mapping
        current_short_jump_args(short_preamble, ctx)
            .iter()
            .map(|&jump_arg| {
                let mapped = mapping.get(&jump_arg).copied().unwrap_or(jump_arg);
                ctx.get_replacement_opref(mapped)
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
    /// `initialize_imported_short_preamble_builder_from_exported_ops`
    /// (a serialized form of the `ShortPreambleBuilder` constructor).
    /// Both consume the same `short_args = label_args + virtuals` slot
    /// space that `export_state` used to build `ExportedShortOp` (see
    /// `collect_exported_short_ops`); the slot indices are the
    /// majit-only stand-in for RPython's Box-keyed `produced_short_boxes`
    /// dict.
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
        for (i, carried) in exported_state.next_iteration_args.iter().enumerate() {
            // `carried` is the literal Phase-1 box (the same Rc used as the
            // exported_infos key); `.to_opref()` is its resolved position, used
            // only for the forwarding plumbing below.
            let target = carried.to_opref();
            // source = targetargs[i]
            let source = targetargs[i];
            // assert source is not target — see commit log for the
            // disjoint-namespace invariant from Step 2 Commit D2 that
            // makes this hold by construction in production callers.
            debug_assert!(source != target, "import_state: source is target");
            // source.set_forwarded(target)
            // `source` is `targetargs[i]`, produced by the caller's
            // cross-slot resolution as either a materialized inputarg or a
            // `reserve_virtual_box`-minted alias — both resolve without
            // minting here.
            let b_source = ctx
                .get_box_replacement_operand_opt(source)
                .expect("import_state source must have a materialized operand slot");
            // `target` is a Phase-1 next-iteration ref whose producer may not
            // be carried into this rebuilt context; materialize its canonical
            // host instead of fabricating a position-only box (`make_equal_to`
            // would re-materialize the unbound target internally anyway —
            // resolve-or-materialize here keeps the chain target canonical
            // from the start).
            let b_target = match ctx.get_box_replacement_operand_opt(target) {
                Some(o) => o,
                None => ctx.materialize_operand_at(target),
            };
            ctx.make_equal_to(&b_source, &b_target);
            if crate::debug::have_debug_prints() {
                crate::debug::log_one(
                    "jit-optimizer",
                    &format!("import_state_map[{i}]: source={source:?} target={target:?}"),
                );
            }
            // info = exported_state.exported_infos.get(target, None)
            // Look up by the carried Phase-1 box (the same Rc the exporter used
            // as the key) — a ptr_eq hit, with no Phase-2 re-resolution that would
            // mint a fresh (non-ptr_eq) Const/InputArg box.
            if let Some(info) = exported_state.exported_infos.get(carried) {
                //     self.optimizer.setinfo_from_preamble(source, info,
                //                                     exported_state.exported_infos)
                self.setinfo_from_preamble(source, info, &exported_state.exported_infos, ctx);
            }
        }
        // label_args = exported_state.virtual_state.make_inputargs(
        //     targetargs, self.optimizer)
        let label_args = match exported_state
            .virtual_state
            .make_inputargs(targetargs, optimizer, ctx, false)
        {
            Ok(args) => args,
            // unroll.py:483 `raise InvalidLoop`: the imported virtual state is
            // incompatible. Recorded as a deferred signal (checked by the
            // caller) so the loop is abandoned without unwinding.
            Err(()) => {
                ctx.signal_invalid_loop("Cannot import state, virtual states don't match");
                return Vec::new();
            }
        };
        if crate::callee_rca_enabled() {
            let next_iteration_args: Vec<_> = exported_state
                .next_iteration_args
                .iter()
                .map(|arg| arg.to_opref())
                .collect();
            eprintln!(
                "[callee-rca][import-state] targetargs={:?} next_iteration_args={:?} \
                 label_args={:?} num_boxes={} entries={}",
                targetargs,
                next_iteration_args,
                label_args,
                exported_state.virtual_state.num_boxes(),
                exported_state.virtual_state.num_entries(),
            );
            for line in callee_rca_virtual_state_summary(&exported_state.virtual_state) {
                eprintln!("[callee-rca][import-vs] {line}");
            }
        }
        // The short-preamble replay part of unroll.py:496-502 is performed
        // by import_short_preamble_state after majit's split-out
        // Optimizer::install_imported_virtuals has completed. RPython's
        // virtual_state.make_inputargs installs virtual PtrInfo before
        // produced_op.produce_op; doing it earlier here replays short ops
        // against incomplete virtual state.
        // return label_args
        label_args
    }

    pub fn import_short_preamble_state(
        &self,
        targetargs: &[OpRef],
        label_args: &[OpRef],
        exported_state: &ExportedState,
        ctx: &mut OptContext,
    ) {
        // self.short_preamble_producer = ShortPreambleBuilder(
        //     label_args, exported_state.short_boxes,
        //     exported_state.short_inputargs, exported_state.exported_infos,
        //     self.optimizer)
        //
        // majit's `ShortPreambleBuilder` constructor is the
        // `initialize_imported_short_preamble_builder_from_exported_ops`
        // call below. The majit serialization keys short ops by slot index
        // over the combined `label_args + virtuals` slot space (the same
        // space `collect_exported_short_ops` built against), so we compute
        // the virtuals tail inline before calling the initializer.
        let virtuals: Vec<OpRef> = exported_state
            .virtual_state
            .state
            .iter()
            .enumerate()
            .filter(|(_, info)| info.is_virtual())
            .filter_map(|(i, _)| targetargs.get(i).copied())
            .collect();
        let mut short_args = label_args.to_vec();
        short_args.extend(virtuals);
        // unroll.py:486-489 ShortPreambleBuilder constructor parity.
        //
        // RPython already has all Box identities before this point, so the
        // ShortPreambleBuilder can be constructed before the produce_op loop.
        // Majit precomputes the equivalent Phase-2 result OpRefs first, then
        // shares that map with both the builder constructor and produce_op.
        // history.py:220 Box `.type` is fixed at construction.  RPython
        // assigns a Box of the correct kind from `produced_op.short_op.res`
        // before `produce_op` runs; majit must allocate the result OpRef
        // with the typed allocator so `opref_type` resolution doesn't
        // depend on a downstream type-registration patch.
        let mut result_map: indexmap::IndexMap<OpRef, OpRef> = indexmap::IndexMap::new();
        for (source, produced) in &exported_state.short_boxes {
            let result_type = produced.preamble_op.result_type();
            let result = match produced.kind {
                crate::optimizeopt::shortpreamble::PreambleOpKind::Pure
                | crate::optimizeopt::shortpreamble::PreambleOpKind::LoopInvariant => {
                    // The short-box result that coincides with a label/virtual
                    // slot maps to that slot's body OpRef. For these non-InputArg
                    // kinds the export records `label_arg_idx` as
                    // `lookup_label_arg(canonical_result)` (optimizer.rs is
                    // kind-aware: InputArg keeps the stamped original slot, every
                    // other kind takes the FORWARDED-result lookup). `source` ==
                    // `canonical_result` == `preamble_op.pos`, so this slot is the
                    // position of the FORWARDED `source` within the original
                    // `label_args + virtuals` — a Pure/LoopInvariant result proven
                    // equal to a label arg it did not originally occupy still
                    // reuses that slot's `short_args[slot]`. (The renamed
                    // `short_inputargs[slot]` is a distinct box and would never
                    // equal `source` anyway.)
                    if let Some(slot) = produced.label_arg_idx {
                        short_args.get(slot).copied()
                    } else {
                        Some(ctx.alloc_op_position_typed(result_type))
                    }
                }
                crate::optimizeopt::shortpreamble::PreambleOpKind::Heap => {
                    match produced.preamble_op.opcode {
                        OpCode::GetfieldGcI
                        | OpCode::GetfieldGcR
                        | OpCode::GetfieldGcF
                        | OpCode::GetarrayitemGcI
                        | OpCode::GetarrayitemGcR
                        | OpCode::GetarrayitemGcF => Some(ctx.alloc_op_position_typed(result_type)),
                        _ => None,
                    }
                }
                crate::optimizeopt::shortpreamble::PreambleOpKind::InputArg
                | crate::optimizeopt::shortpreamble::PreambleOpKind::Guard => None,
            };
            if let Some(result) = result {
                // shortpreamble.py:327: `self.res` is a Box object that
                // exists from import time. The freshly allocated
                // body-visible result slots (the `alloc_op_position_typed`
                // arms above) have no producer yet; mint their canonical
                // `SameAs*` stand-in here so a later `get_box_replacement`
                // resolves them instead of fabricating a position-only box.
                // Slot-mapped results (`short_args[slot]`) are already bound
                // inputargs and resolve without minting.
                if ctx.get_box_replacement_operand_opt(result).is_none() {
                    ctx.mint_box_at(result);
                }
                result_map.insert(*source, result);
            }
        }
        let mut imported_constants: indexmap::IndexMap<OpRef, OpRef> = indexmap::IndexMap::new();
        let from_short_boxes = ctx.initialize_imported_short_preamble_builder_from_short_boxes(
            &short_args,
            &exported_state.short_inputargs,
            &exported_state.short_boxes,
            &exported_state.short_box_const_values,
            &result_map,
            &mut imported_constants,
            &exported_state.exported_infos,
        );
        debug_assert!(
            from_short_boxes,
            "initialize_imported_short_preamble_builder_from_short_boxes returned false: \
             the short-preamble import path failed to handle some entry. \
             This signals an unresolvable arg classification (Slot/Const/Produced)."
        );
        // unroll.py:501-502 line-by-line port:
        //
        //   for produced_op in exported_state.short_boxes:
        //       produced_op.produce_op(self, exported_state.exported_infos)
        //
        // `produced_results` accumulates `source pos -> replay identity` so
        // successor entries can resolve `Produced` arg classifications. After
        // Path B (B.6.7) the producer-side `produce_*` methods return
        // `Some(source)` (Phase 1 OpRef = `self.res`); for invented Pure the
        // value is the body-visible OpRef per `replay_pos` in
        // `initialize_imported_short_preamble_builder_from_short_boxes`.
        let mut produced_results: indexmap::IndexMap<OpRef, OpRef> = indexmap::IndexMap::new();
        for (_, produced) in &exported_state.short_boxes {
            let produced_result = produced.produce_op(
                ctx,
                &exported_state.exported_infos,
                &exported_state.short_inputargs,
                &short_args,
                &result_map,
                &mut produced_results,
                &mut imported_constants,
                &exported_state.short_box_const_values,
            );
            debug_assert!(
                produced_result.is_some()
                    || matches!(
                        produced.kind,
                        crate::optimizeopt::shortpreamble::PreambleOpKind::InputArg
                            | crate::optimizeopt::shortpreamble::PreambleOpKind::Guard
                    )
                    || !result_map.contains_key(&produced.preamble_op.pos.get()),
                "ProducedShortOp::produce_op failed for source {:?} kind {:?}",
                produced.preamble_op.pos.get(),
                produced.kind
            );
        }
    }

    /// unroll.py:432-443 `_expand_info` + `unroll.py:548`
    /// `ExportedState.exported_infos` entry producer. Returns the single
    /// `OpInfo` variant RPython's dict would hold for this box, or `None`
    /// for the RPython `if info:` falsy fall-through at unroll.py:440 — the
    /// caller must treat `None` as "no entry in the dict" so downstream
    /// `setinfo_from_preamble_list` takes the `item.set_forwarded(None)`
    /// branch at unroll.py:49.
    ///
    ///   - Ref-typed live box with PtrInfo → `Some(OpInfo::Ptr(...))`.
    ///   - Constant OpRef (value-carrying) → `Some(OpInfo::Ptr(PtrInfo::Constant))`
    ///     for Refs, `Some(OpInfo::FloatConstInfo(FloatConstInfo))` for floats,
    ///     `Some(OpInfo::IntBound(IntBound::from_constant(v)))` for ints
    ///     (mirroring RPython's `ConstPtrInfo` / `FloatConstInfo` /
    ///     `IntBound` dispatch).
    ///   - Int-typed box with a preamble-exported `IntBound` →
    ///     `Some(OpInfo::IntBound(...))`.
    ///   - No info → `None`. `OpInfo::Unknown` must not appear in
    ///     `ExportedState.exported_infos`.
    fn collect_exported_info(
        &self,
        opref: OpRef,
        ctx: &OptContext,
        exported_int_bounds: Option<
            &indexmap::IndexMap<majit_ir::operand::Operand, crate::optimizeopt::intutils::IntBound>,
        >,
    ) -> Option<crate::optimizeopt::info::OpInfo> {
        use crate::optimizeopt::info::{FloatConstInfo, OpInfo, PtrInfo};
        let resolved = ctx.get_replacement_opref(opref);
        // unroll.py:432-443 `_expand_info` calls `self.optimizer.getinfo(arg)`
        // which itself runs `get_box_replacement` first, so a non-constant
        // OpRef forwarded to a Const surfaces the corresponding constant
        // info class (ConstPtrInfo / FloatConstInfo / IntBound from_constant).
        let synthesize_const_info = |value: Value| -> Option<OpInfo> {
            match value {
                // ConstPtrInfo parity: RPython stores Ref constants as
                // ConstPtrInfo (a `PtrInfo` subclass). `setinfo_from_preamble`
                // at unroll.py:65-68 dispatches through `is_constant()`.
                Value::Ref(gcref) => Some(OpInfo::ptr(PtrInfo::Constant(gcref))),
                // FloatConstInfo parity: unroll.py:97-98 handles
                // `isinstance(preamble_info, info.FloatConstInfo)` with
                // `op.set_forwarded(preamble_info._const)`.
                Value::Float(f) => Some(OpInfo::FloatConstInfo(FloatConstInfo::new(f))),
                // Int constants: RPython uses IntBound with lower==upper.
                Value::Int(v) => Some(OpInfo::int_bound(
                    crate::optimizeopt::intutils::IntBound::from_constant(v),
                )),
                Value::Void => None,
            }
        };
        if resolved.is_constant() {
            if let Some(value) = ctx
                .get_box_replacement_operand_opt(resolved)
                .and_then(|cb| cb.const_value())
            {
                return synthesize_const_info(value);
            }
        }
        // make_constant mirrors optimizer.py:432 as `Forwarded::Const(constval)`.
        // The walker has advanced to the constbox terminal — surface RPython's
        // ConstPtrInfo / FloatConstInfo / IntBound dispatch via const_value().
        let resolved_box = ctx.get_box_replacement_operand_opt(opref);
        if let Some(b) = resolved_box.as_ref() {
            if b.is_constant() {
                if let Some(value) = b.const_value() {
                    return synthesize_const_info(value);
                }
            }
        }
        // unroll.py:432-443 _expand_info uses self.optimizer.getinfo(arg) which
        // dispatches by op.type ('r' → getptrinfo, 'i' → getintbound). The Rust
        // port stores int bounds in a separate table populated earlier by
        // `OptIntBounds::export_arg_int_bounds`, which already filters by
        // `opref_type(resolved) == Some(Int)`. We rely on that filter so the
        // lookup here cannot pull a bound for a ref/float box.
        if let Some(handle) = resolved_box.as_ref().and_then(|b| b.ptr_info_handle()) {
            // RPython object identity: re-export the same Rc handle so
            // downstream `setinfo_from_preamble` sees the live cell, not
            // a snapshot. Matches PyPy `_forwarded` reference passing.
            return Some(OpInfo::Ptr(handle));
        }
        // unroll.py:443-454 `_expand_info` calls `self.optimizer.getinfo(arg)`
        // for EVERY exported value with no jump-arg restriction, so an int-typed
        // value's bound is read straight off its `_forwarded` slot
        // (`getintbound`). Read the live bound from the still-alive Phase-1
        // optimizer context; this recovers the `[0, mask]` bound of a masked
        // short-preamble / loop-invariant pure result that is not a jump arg and
        // therefore never entered the `exported_int_bounds` side table. Mirror
        // `export_arg_int_bounds` (intbounds.rs): skip Const/non-Int values and
        // unbounded bounds.
        if let Some(box_op) = resolved_box.as_ref() {
            if !resolved.is_constant()
                && matches!(ctx.opref_type(resolved), Some(majit_ir::Type::Int))
            {
                if let Some(bound) = ctx.peek_intbound_box(box_op) {
                    if !bound.is_unbounded() {
                        return Some(OpInfo::int_bound(bound));
                    }
                }
            }
        }
        // Fallback: read from `exported_int_bounds`
        // side table.  RPython's `IntBound` flows through
        // `OptInfo.IntBound` on the Box itself
        // (`optimizeopt/info.py:580 IntBoundInfo`), so successive peeling
        // iterations see the bound without an explicit hand-off.
        // pyre's flat-OpRef `OptContext` is rebuilt per round so the
        // preamble's bound must be exported by
        // `intbounds.rs::export_arg_int_bounds` and re-imported here.
        // Convergence: extend `setinfo_from_preamble_item` (`mod.rs`)
        // to attach `OpInfo::IntBound` alongside `OpInfo::Ptr` so this
        // branch becomes redundant and the side-table parameter
        // disappears.
        if let Some(bound) = exported_int_bounds.and_then(|bounds| {
            // Same Phase-1 ctx as the export producer, so the canonical box for
            // `opref` is the memoized `Rc` the bound was keyed under (ptr_eq).
            ctx.get_box_replacement_operand_opt(opref)
                .and_then(|o| bounds.get(&o).cloned())
        }) {
            return Some(OpInfo::int_bound(bound));
        }
        None
    }

    /// unroll.py:53-98 `setinfo_from_preamble(op, preamble_info, exported_infos)`.
    ///
    /// Thin forwarding wrapper onto `OptContext::setinfo_from_preamble_item`
    /// (mod.rs), which is the shared RPython-literal dispatcher used both by
    /// this top-level import and by the recursive virtual-field walker at
    /// `setinfo_from_preamble_list`. Keeping a single implementation prevents
    /// the two call sites from drifting: any shortcut in one must also apply
    /// to the other or the early-return semantics at unroll.py:54-58 are
    /// bypassed for part of the import path.
    fn setinfo_from_preamble(
        &self,
        opref: OpRef,
        info: &crate::optimizeopt::info::OpInfo,
        exported_infos: &indexmap::IndexMap<
            majit_ir::operand::Operand,
            crate::optimizeopt::info::OpInfo,
        >,
        ctx: &mut OptContext,
    ) {
        ctx.setinfo_from_preamble_item(opref, info, exported_infos);
    }
}

/// unroll.py: export_state — module-level entry point.
pub(crate) fn export_state(
    jump_args: &[OpRef],
    renamed_inputargs: &[OpRef],
    optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
    ctx: &mut OptContext,
    exported_int_bounds: Option<
        &indexmap::IndexMap<majit_ir::operand::Operand, crate::optimizeopt::intutils::IntBound>,
    >,
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

pub(crate) fn import_short_preamble_state(
    targetargs: &[OpRef],
    label_args: &[OpRef],
    exported_state: &ExportedState,
    ctx: &mut OptContext,
) {
    OptUnroll::new().import_short_preamble_state(targetargs, label_args, exported_state, ctx)
}

/// RPython unroll.py:479-504 `import_state(targetargs, exported_state)`
/// canonical end-to-end orchestrator.  Bundles the three majit-side
/// sub-steps that mirror the RPython single-function flow:
///
/// 1. `OptUnroll::import_state` — `unroll.py:483-494` forwarding +
///    `virtual_state.make_inputargs` to compute label_args.
/// 2. `Optimizer::install_imported_virtuals` — majit-only counterpart
///    to RPython's inline virtualizable PtrInfo installation that
///    happens during `make_inputargs` itself.  Split out for borrow
///    reasons (`Optimizer` access to `imported_virtuals` / `imported_loop_state`).
/// 3. `OptUnroll::import_short_preamble_state` — `unroll.py:496-502`
///    `ShortPreambleBuilder` construction + `produce_op` loop.
///
/// Production callers should use this end-to-end form.  Test callers
/// that need to inspect intermediate state can still invoke the
/// individual functions above.
pub(crate) fn import_state_full(
    targetargs: &[OpRef],
    exported_state: &ExportedState,
    optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
    ctx: &mut OptContext,
) -> Vec<OpRef> {
    let label_args = OptUnroll::new().import_state(targetargs, exported_state, optimizer, ctx);
    // import_state records a deferred InvalidLoop signal on `ctx` when the
    // virtual states don't match; stop here (the caller observes the signal and
    // abandons the loop) rather than replaying short preamble ops against an
    // incomplete import.
    if ctx.has_pending_invalid_loop() {
        return label_args;
    }
    optimizer.imported_label_args = Some(label_args.clone());
    if !optimizer.imported_virtuals.is_empty() {
        optimizer.install_imported_virtuals(ctx);
    }
    OptUnroll::new().import_short_preamble_state(targetargs, &label_args, exported_state, ctx);
    label_args
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
    use crate::optimizeopt::virtualstate::{VirtualStateInfo, VirtualStateInfoNode};

    /// virtualstate.py:158-165 AbstractVirtualStructStateInfo.generate_guards
    /// parity: walk `fielddescrs` in parent-local slot order, looking up
    /// the matching field state via descr.get_index() (= field_idx in pyre).
    fn ordered_fields(
        field_descrs: &[majit_ir::DescrRef],
        fields: &[(u32, std::rc::Rc<VirtualStateInfoNode>)],
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
                    .map(|(_, field_value)| (field_descr.clone(), field_value.info.clone()))
            })
            .collect()
    }

    let mut result = Vec::new();
    for (idx, info) in state.virtual_state.state.iter().enumerate() {
        match &info.info {
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
#[cfg(test)]
fn assemble_peeled_trace(
    p1_ops: &[Op],
    p2_ops: &[Op],
    label_args: &[OpRef],
    start_label_args: &[OpRef],
    extra_label_args: &[OpRef],
    body_num_inputs: usize,
    jump_to_self: bool,
    imported_short_aliases: &[crate::optimizeopt::ImportedShortAlias],
    constants: &majit_ir::ConstMap<majit_ir::Value>,
    start_label_descr: Option<DescrRef>,
    loop_label_descr: Option<DescrRef>,
) -> Vec<majit_ir::OpRc> {
    let mut ctx = assemble_test_context(p1_ops, p2_ops, body_num_inputs);
    let p1_ops_rc: Vec<majit_ir::OpRc> = p1_ops
        .iter()
        .map(|op| std::rc::Rc::new(op.clone()))
        .collect();
    let p2_ops_rc: Vec<majit_ir::OpRc> = p2_ops
        .iter()
        .map(|op| std::rc::Rc::new(op.clone()))
        .collect();
    assemble_peeled_trace_with_jump_args(
        &p1_ops_rc,
        &p2_ops_rc,
        label_args,
        start_label_args,
        extra_label_args,
        extra_label_args,
        body_num_inputs,
        0, // inputarg_base — tests/simple cases use shared namespace
        jump_to_self,
        imported_short_aliases,
        constants,
        start_label_descr,
        loop_label_descr,
        &[], // no p1_end_args for simple assembly
        &mut ctx,
    )
}

#[cfg(test)]
fn assemble_test_context(p1_ops: &[Op], p2_ops: &[Op], body_num_inputs: usize) -> OptContext {
    // Test helper — caller provides only the body inputarg count, so seed
    // every slot as Ref (matches the common loop-body shape; fixtures that
    // need typed inputargs construct ctx directly).
    //
    // Type lookup now resolves through `op.pos.ty()` (variant
    // tag, post-Slice-P5/P6) at priority 0 of `opref_type` and
    // `op.type_` at priority 2 — no longer through a `value_types`
    // side table.
    let types = vec![Type::Ref; body_num_inputs];
    OptContext::with_inputarg_types(p1_ops.len() + p2_ops.len(), &types)
}

/// shortpreamble.py:436-439 alias-side `extra_same_as` emission.
///
/// Compound alternates (`invented_name=true`) record their SameAs source in
/// `imported_short_aliases`. Emit one `SameAs(alias.same_as_source)` op at
/// `pos=alias.result` per entry so the body's reference to the alias result
/// has a defining op.
fn emit_alias_same_as_for_imports(
    result: &mut Vec<majit_ir::OpRc>,
    imported_short_aliases: &[crate::optimizeopt::ImportedShortAlias],
) {
    for alias in imported_short_aliases {
        let mut op = Op::new(alias.same_as_opcode, &[alias.same_as_source.clone()]);
        op.pos.set(alias.result);
        result.push(std::rc::Rc::new(op));
    }
}

fn assemble_peeled_trace_with_jump_args(
    p1_ops: &[majit_ir::OpRc],
    p2_ops: &[majit_ir::OpRc],
    label_args: &[OpRef],
    start_label_args: &[OpRef],
    extra_label_args: &[OpRef],
    extra_jump_args: &[OpRef],
    body_num_inputs: usize,
    inputarg_base: u32,
    jump_to_self: bool,
    imported_short_aliases: &[crate::optimizeopt::ImportedShortAlias],
    constants: &majit_ir::ConstMap<majit_ir::Value>,
    start_label_descr: Option<DescrRef>,
    loop_label_descr: Option<DescrRef>,
    _p1_end_args: &[OpRef],
    ctx: &mut crate::optimizeopt::OptContext,
) -> Vec<majit_ir::OpRc> {
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
    // The only entries that legitimately drop out here are literal Const
    // boxes. `used_boxes` / `short_preamble_jump` are RPython Box lists;
    // a stale backend-constants entry must not make us drop a live runtime
    // Box from the LABEL/JUMP contract (e.g. an imported short-preamble
    // HeapField result that still has optimizer constant knowledge attached).
    for (idx, &label_arg) in extra_label_args.iter().enumerate() {
        let jump_arg = extra_jump_args.get(idx).copied().unwrap_or(label_arg);
        if label_arg.is_constant() {
            continue;
        }
        filtered_extra_label_args.push(label_arg);
        filtered_extra_jump_args.push(jump_arg);
    }
    let next_free_pos = |mut next: u32| {
        next = next.max(inputarg_base + body_num_inputs as u32);
        while constants.contains_key(&next) {
            next += 1;
        }
        next
    };

    if let Some(start_label_descr) = start_label_descr {
        let mut start_label_args_box_operand: Vec<majit_ir::operand::Operand> =
            Vec::with_capacity(start_label_args.len());
        for a in start_label_args {
            start_label_args_box_operand.push(ctx.materialize_operand_at(*a));
        }
        let mut start_label = Op::new(OpCode::Label, &start_label_args_box_operand);
        start_label.pos.set(OpRef::NONE);
        start_label.setdescr(start_label_descr);
        result.push(std::rc::Rc::new(start_label));
    }

    // Preamble: everything except Jump
    for op in p1_ops {
        if op.opcode == OpCode::Jump {
            break;
        }
        result.push(std::rc::Rc::new((**op).clone()));
    }

    // max_pos must account for ALL p1_ops positions, including SameAs
    // ops AFTER the Jump that weren't copied into result. These positions
    // are referenced by the body Label args and must not be reused.
    let result_max = result
        .iter()
        .map(|op| op.pos.get().raw())
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(inputarg_base + body_num_inputs as u32);
    let p1_all_max = p1_ops
        .iter()
        .map(|op| op.pos.get().raw())
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(0);
    let mut max_pos = result_max.max(p1_all_max);
    max_pos = next_free_pos(max_pos.saturating_add(1));
    emit_alias_same_as_for_imports(&mut result, imported_short_aliases);

    // Label position
    let label_pos = next_free_pos(max_pos);
    let mut full_label_args: Vec<OpRef> = label_args
        .iter()
        .copied()
        .filter(|arg| !is_trace_constant_ref(*arg, constants))
        .collect();

    // Collect preamble-defined OpRefs BEFORE adding extra label args,
    // so we can filter out virtual remnants (removed New ops).
    //
    // resoperation.py:719/727/739 InputArg{Int,Ref,Float}: preamble-defined
    // inputargs carry `box.type` intrinsically (history.py:220). Mint
    // typed variants from `inputarg_types` so this set's OpRefs match the
    // typed mints used at trace start / Phase 2 import under variant-aware
    // Eq.
    let preamble_defs: indexmap::IndexSet<OpRef> = {
        let mut s: indexmap::IndexSet<OpRef> = (0..body_num_inputs)
            .map(|i| {
                let pos = inputarg_base + i as u32;
                // history.py:220 box.type / resoperation.py:719/727/739
                // InputArgInt/Float/Ref invariant: every label / inputarg
                // carries a value-box class. RPython has no InputArgVoid;
                // a missing type at this site is a structural bookkeeping
                // bug. `inputarg_type_at_strict` panics on miss with the
                // RPython-citation message rather than recovering into
                // VoidOp / a guessed default.
                let tp = ctx.inputarg_type_at_strict(i);
                OpRef::input_arg_typed(pos, tp)
            })
            .collect();
        for op in &result {
            if !op.pos.get().is_none()
                && op.opcode != OpCode::Jump
                && op.result_type() != Type::Void
            {
                s.insert(op.pos.get());
            }
        }
        s
    };

    // Append non-constant extra label args. The caller (unroll pass)
    // determines which extra values the loop header needs; virtual
    // remnants have already been filtered by the caller.
    //
    // Advance max_pos past every position already in use before allocating
    // fresh SameAs positions. In RPython, Box identity prevents collisions;
    // in the flat OpRef model, the assembly-allocated SameAs position must
    // not overlap base label_args, filtered_extra_label_args, or p2_ops
    // positions/args/fail_args — each of those may be referenced by the
    // body. A single colliding OpRef aliases two distinct values and
    // corrupts the loop.
    for &la in &full_label_args {
        if is_trace_runtime_ref(la, constants) {
            max_pos = max_pos.max(la.raw().saturating_add(1));
        }
    }
    for &la in &filtered_extra_label_args {
        if is_trace_runtime_ref(la, constants) {
            max_pos = max_pos.max(la.raw().saturating_add(1));
        }
    }
    for op in p2_ops.iter() {
        if is_trace_runtime_ref(op.pos.get(), constants) {
            max_pos = max_pos.max(op.pos.get().raw().saturating_add(1));
        }
        for arg in op.getarglist().iter() {
            if is_trace_runtime_ref(arg.to_opref(), constants) {
                max_pos = max_pos.max(arg.to_opref().raw().saturating_add(1));
            }
        }
        if let Some(fa) = op.getfailargs() {
            for arg in fa.iter() {
                if is_trace_runtime_ref(arg.to_opref(), constants) {
                    max_pos = max_pos.max(arg.to_opref().raw().saturating_add(1));
                }
            }
        }
    }
    let extra_label_start_idx = full_label_args.len();
    full_label_args.extend(filtered_extra_label_args.iter().copied());

    // RPython compile.py parity: after the loop label, only the loop-header
    // contract is live. When `splice_redirected_tail` glues a redirected
    // tail onto the body, the spliced output may mention OpRefs whose
    // defining op was removed (the section between the splice point and
    // the original Jump). Carry such "body use-before-def" references
    // through the label so the assembled body doesn't contain dangling
    // references. With Phase 2's disjoint OpRef namespace, body op args
    // are pre-resolved through ctx.get_box_replacement, so the carried
    // OpRef can be appended directly to full_label_args — the JUMP's
    // mapped_base_args path picks up the corresponding fresh value on the
    // next iteration. The filter only needs to skip filtered_extra_jump_args.
    let mut carried_source_slots: indexmap::IndexSet<OpRef> = indexmap::IndexSet::new();
    carried_source_slots.extend(filtered_extra_jump_args.iter().copied());
    // `label_set` tracks which OpRefs are already carried by the label so
    // that the body-use-before-def pass doesn't add the same OpRef twice
    // (which would inflate label arity without adding new information).
    // RPython's Box identity makes this a correctness filter: two
    // references to the same Box collapse into one live-in slot. This
    // is NOT the Issue 1 dedup — which drops distinct Boxes that happen
    // to share an OpRef — it is RPython parity: the same Box appears
    // once in the label arglist.
    let mut label_set: indexmap::IndexSet<OpRef> = full_label_args.iter().copied().collect();
    let mut fallthrough_aliases = Vec::new();
    {
        let mut seen_body_defs = indexmap::IndexSet::new();
        for op in p2_ops {
            // compile.py assembles the loop LABEL from `label_op` plus
            // short-preamble `used_boxes`; the terminal JUMP's target-local
            // payload is not part of that contract. In particular,
            // jump_to_preamble retargets the body JUMP to the preamble start
            // label, so carrying its extra args into the loop LABEL mutates
            // the body arity in a way upstream never does.
            if op.opcode == OpCode::Jump {
                continue;
            }
            let op_args = op.getarglist_copy();
            let all_refs = op_args
                .iter()
                .map(|a| a.to_opref())
                .chain(op.getfailargs().into_iter().flatten().map(|b| b.to_opref()));
            for arg in all_refs {
                if !is_trace_runtime_ref(arg, constants) {
                    continue; // skip NONE and constants
                }
                if label_set.contains(&arg)
                    || carried_source_slots.contains(&arg)
                    || seen_body_defs.contains(&arg)
                {
                    continue;
                }
                // RPython Box identity parity: Phase 2 may forward a
                // preamble-defined Box to a fresh body-visible Box. The
                // Label carries the forwarded Box, but first fall-through
                // only has the preamble source; pyre's flat OpRef model
                // needs an explicit SameAs bridge before the Label.
                if let Some(source) = preamble_defs
                    .iter()
                    .copied()
                    .find(|&source| source != arg && ctx.get_replacement_opref(source) == arg)
                {
                    let tp = ctx
                        .opref_type(arg)
                        .or_else(|| ctx.opref_type(source))
                        .or_else(|| arg.ty())
                        .unwrap_or_else(|| {
                            panic!(
                                "assemble_peeled_trace_with_jump_args: cannot type \
                                 fallthrough SameAs alias arg={:?} source={:?}; \
                                 ctx.opref_type(arg), ctx.opref_type(source), and \
                                 arg.ty() all returned None",
                                arg, source
                            )
                        });
                    if tp != Type::Void {
                        let arg_source = ctx.materialize_operand_at(source);
                        let mut same_as = Op::new(OpCode::same_as_for_type(tp), &[arg_source]);
                        same_as.pos.set(arg);
                        fallthrough_aliases.push(same_as);
                    }
                }
                full_label_args.push(arg);
                label_set.insert(arg);
            }
            if op.result_type() != Type::Void && !op.pos.get().is_none() {
                seen_body_defs.insert(op.pos.get());
            }
        }
    }

    let mut full_label_args_box_operand: Vec<majit_ir::operand::Operand> =
        Vec::with_capacity(full_label_args.len());
    for a in &full_label_args {
        full_label_args_box_operand.push(ctx.materialize_operand_at(*a));
    }
    let mut label_op = Op::new(OpCode::Label, &full_label_args_box_operand);
    // resoperation.py:260 AbstractResOp.type = 'v' default — Label has no
    // result Box, so its OpRef position carries the Void tag rather than
    // a stray Int tag. `op_index` filters Void ops so this OpRef never
    // shadows a real Box-bearing op at the same raw position, and
    // variant-aware Hash matches when downstream consumers compare
    // label_op.pos against Op-keyed maps.
    label_op
        .pos
        .set(OpRef::op_typed(label_pos, label_op.result_type()));
    if let Some(d) = loop_label_descr {
        label_op.setdescr(d);
    }
    result.extend(fallthrough_aliases.into_iter().map(std::rc::Rc::new));
    result.push(std::rc::Rc::new(label_op));

    // Body: 2-pass remap (inputarg refs -> label args, body results -> fresh boxes)
    let max_label_arg_pos = full_label_args
        .iter()
        .map(|a| a.raw())
        .max()
        .unwrap_or(label_pos);
    let max_emitted_pos = result
        .iter()
        .map(|op| op.pos.get().raw())
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(label_pos);
    // Fresh body positions must be higher than ALL existing positions:
    // already-emitted preamble/imported-short ops, label, label args, AND
    // Phase 2 op positions (which may be higher than label positions due
    // to Phase 2 remap or redirect ops).
    //
    // RPython's distinct Box identities prevent an imported short-preamble
    // replay result from colliding with a freshly emitted body result. In
    // majit's flat OpRef model we must keep the body allocator above every
    // position already emitted into `result`.
    let max_p2_pos = p2_ops
        .iter()
        .map(|op| op.pos.get().raw())
        .filter(|&p| p != u32::MAX)
        .max()
        .unwrap_or(0);
    let mut next_body_pos = next_free_pos(
        max_emitted_pos
            .max(label_pos)
            .max(max_label_arg_pos)
            .max(max_p2_pos)
            .saturating_add(1),
    );
    // Keyed lookup only (`get`/`insert`, never iterated — codegen order comes
    // from the `p2_ops` walk below), so a hash map keeps each per-op remap O(1)
    // instead of the IndexMap linear `get_index_of` scan on long peeled traces.
    let mut body_result_remap: std::collections::HashMap<OpRef, OpRef> =
        std::collections::HashMap::new();
    let visible_before_label: indexmap::IndexSet<OpRef> = full_label_args
        .iter()
        .copied()
        .chain(preamble_defs.iter().copied())
        .collect();

    // shortpreamble.py:436-439 + unroll.py:497-504 parity: when
    // `sp.used_boxes[i] != sp.jump_args[i]` (Case A/B end-of-preamble
    // SameAs alias splits identity — preamble JUMP carries `jump_source`,
    // LABEL takes `source_slot`), register the forwarding chain so body
    // ops referencing `jump_source` resolve to `source_slot` via
    // `ctx.get_box_replacement` (`optimizer.py:266 set_forwarded`).
    // RPython's Box identity makes this implicit — the alias's Box is
    // the same Python object that body ops already hold. Pyre's flat
    // OpRef model needs an explicit forwarding registration here.
    let mut assembly_alias_remap: std::collections::HashMap<OpRef, OpRef> =
        std::collections::HashMap::new();
    // Keep the assembly-only alias map separate from the general `_forwarded`
    // walk. PyPy has object identity for these short-preamble boxes; pyre needs
    // the explicit jump_source -> label_arg substitution, but must not follow
    // later postprocess Const forwarding on unrelated emitted guard args.
    for (i, &source_slot) in filtered_extra_label_args.iter().enumerate() {
        if source_slot.is_none() {
            continue;
        }
        let Some(&extended_label_arg) = full_label_args.get(extra_label_start_idx + i) else {
            continue;
        };
        debug_assert_eq!(
            source_slot, extended_label_arg,
            "full_label_args at extra_label_start_idx + i must equal \
             filtered_extra_label_args[i] (line 4338 extend invariant)"
        );
        if let Some(&jump_source) = filtered_extra_jump_args.get(i) {
            if !jump_source.is_none() && jump_source != source_slot {
                let b_js = ctx.materialize_operand_at(jump_source);
                // Chain target: resolve-or-materialize the canonical host
                // (make_equal_to materializes an unbound target internally;
                // doing it here keeps the target off the position-only
                // fabrication path).
                let b_ela = match ctx.get_box_replacement_operand_opt(extended_label_arg) {
                    Some(b) => b,
                    None => ctx.materialize_operand_at(extended_label_arg),
                };
                ctx.make_equal_to(&b_js, &b_ela);
                assembly_alias_remap.insert(jump_source, extended_label_arg);
            }
        }
    }
    for op in p2_ops.iter() {
        // Only map non-Void ops that actually produce a result.
        // Void ops (SetfieldGc, guards, Jump) don't define values at
        // their position — mapping them creates phantom OpRefs.
        if op.pos.get().raw() != u32::MAX && op.result_type() != Type::Void {
            // history.py:220 box.type / resoperation.py:567/589/615 IntOp /
            // RefOp / FloatOp.type — the fresh result Box inherits the
            // producing op's type tag so downstream readers (`opref_type`
            // typed-first arm + variant-aware HashMap/HashSet lookups)
            // see the correct `box.type` instead of a default-int guess.
            let fresh = OpRef::op_typed(next_body_pos, op.result_type());
            next_body_pos = next_free_pos(next_body_pos.saturating_add(1));
            body_result_remap.insert(op.pos.get(), fresh);
        }
    }

    let mut seen_body_defs = indexmap::IndexSet::new();
    let mut current_inner_label_index: Option<usize> = None;
    let mut defs_since_inner_label: indexmap::IndexSet<OpRef> = indexmap::IndexSet::new();
    // Combined-trace position → emitted clone, so remap hits bind to the
    // body clone producer instead of re-minting a position-only box (SSA:
    // a remapped arg's target clone was pushed in an earlier iteration).
    let mut emitted_at: std::collections::HashMap<OpRef, majit_ir::OpRc> =
        std::collections::HashMap::new();
    for (op_idx, op) in p2_ops.iter().enumerate() {
        let mut new_op = (**op).clone();
        let mut original_args: Vec<OpRef> =
            op.getarglist_copy().iter().map(|a| a.to_opref()).collect();
        if let Some(&mapped_pos) = body_result_remap.get(&op.pos.get()) {
            new_op.pos.set(mapped_pos);
        }
        // Body op args were already resolved at emit time by
        // optimizer.py:614-625 / Optimizer::emit_operation. Do not walk
        // forwarding chains again here: postprocess_GUARD_TRUE/FALSE may have
        // installed Const forwarding after the guard was emitted, and PyPy keeps
        // the guard's original runtime argument.
        let remap_body_arg = |arg: OpRef,
                              assembly_alias_remap: &std::collections::HashMap<OpRef, OpRef>,
                              body_result_remap: &std::collections::HashMap<OpRef, OpRef>,
                              seen_body_defs: &indexmap::IndexSet<OpRef>,
                              visible_before_label: &indexmap::IndexSet<OpRef>|
         -> OpRef {
            if let Some(&mapped) = assembly_alias_remap.get(&arg) {
                return mapped;
            }
            if let Some(&mapped) = body_result_remap.get(&arg) {
                if seen_body_defs.contains(&arg) || !visible_before_label.contains(&arg) {
                    return mapped;
                }
            }
            arg
        };
        // optimizer.py:651-652 force_box loop pattern:
        //   for i in range(op.numargs()): op.setarg(i, ...)
        // Only remap hits are rewritten; a miss keeps the existing operand
        // (a bound operand stays live-tracking instead of degrading to a
        // position-only re-mint of the same position).
        for i in 0..new_op.num_args() {
            let arg = new_op.arg(i).to_opref();
            let mapped = remap_body_arg(
                arg,
                &assembly_alias_remap,
                &body_result_remap,
                &seen_body_defs,
                &visible_before_label,
            );
            if mapped != arg {
                // A remap hit whose target clone was already pushed binds to
                // that producer; otherwise route through the canonical
                // "box always exists" materializer (parity with the start_label
                // / jump_source / extended_label_arg sites above) so the arg
                // carries a bound `Operand::Op`/`InputArg` instead of a
                // position-only box. `materialize_operand_at(mapped).to_opref() ==
                // mapped`, so the rewritten arg is OpRef-identical.
                let boxed = match emitted_at.get(&mapped) {
                    Some(rc) => majit_ir::operand::Operand::from_bound_op(rc),
                    None => ctx.materialize_operand_at(mapped),
                };
                new_op.setarg(i, boxed);
            }
        }
        if new_op.opcode == OpCode::Label {
            let mut seen_after_label_defs = indexmap::IndexSet::new();
            let mut extra_inner_sources = Vec::new();
            let mut extra_inner_set = indexmap::IndexSet::new();
            let label_arg_set: indexmap::IndexSet<OpRef> = original_args
                .iter()
                .copied()
                .filter(|arg| !arg.is_none())
                .collect();
            for later_op in p2_ops.iter().skip(op_idx + 1) {
                for arg in later_op.getarglist().iter().map(|a| a.to_opref()).chain(
                    later_op
                        .getfailargs()
                        .into_iter()
                        .flatten()
                        .map(|b| b.to_opref()),
                ) {
                    // unroll.py:364 `_map_args` passes Const through unchanged
                    // — inline-Const args (history.py:227/268/314) carry their
                    // value on the OpRef itself, so they are never label-args.
                    // `arg.is_constant()` short-circuits before `.raw()` (which
                    // panics on inline-Const variants).
                    if arg.is_none()
                        || arg.is_constant()
                        || constants.contains_key(&arg.raw())
                        || label_arg_set.contains(&arg)
                        || extra_inner_set.contains(&arg)
                        || seen_after_label_defs.contains(&arg)
                    {
                        continue;
                    }
                    // optimizer.py:614-625 freezes op args at emit time;
                    // walking ctx.get_box_replacement here would follow Const
                    // forwarding that postprocess installed AFTER the body op
                    // was emitted (corrupting already-emitted trace). The
                    // short-preamble alias (jump_source -> extended_label_arg)
                    // is the only assembly-time substitution we want; consult
                    // the local map directly.
                    let resolved_arg = assembly_alias_remap.get(&arg).copied().unwrap_or(arg);
                    let available_before_label = visible_before_label.contains(&arg)
                        || visible_before_label.contains(&resolved_arg)
                        || seen_body_defs.contains(&arg);
                    if available_before_label {
                        extra_inner_set.insert(arg);
                        extra_inner_sources.push(arg);
                    }
                }
                if later_op.result_type() != Type::Void && !later_op.pos.get().is_none() {
                    seen_after_label_defs.insert(later_op.pos.get());
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
            // unroll.py:301 label_op.initarglist(label_op.getarglist() +
            //                                    sb.used_boxes)
            let mut extended_args: smallvec::SmallVec<[OpRef; 3]> = new_op
                .getarglist_copy()
                .iter()
                .map(|a| a.to_opref())
                .collect();
            for &source_arg in &extra_inner_sources {
                // optimizer.py:614-625 freeze: do not follow ctx forwarding
                // chains here; postprocess Const forwarding on body ops would
                // otherwise leak into the extended Label's arg list, turning
                // a runtime inputarg slot into a Const reference.
                let resolved = assembly_alias_remap
                    .get(&source_arg)
                    .copied()
                    .unwrap_or(source_arg);
                let mapped_arg = remap_body_arg(
                    resolved,
                    &assembly_alias_remap,
                    &body_result_remap,
                    &seen_body_defs,
                    &visible_before_label,
                );
                extended_args.push(mapped_arg);
                original_args.push(source_arg);
            }
            let mut extended_args_box: smallvec::SmallVec<[majit_ir::operand::Operand; 3]> =
                smallvec::SmallVec::with_capacity(extended_args.len());
            for a in &extended_args {
                extended_args_box.push(ctx.materialize_operand_at(*a));
            }
            new_op.initarglist(extended_args_box);
        }
        if new_op.opcode == OpCode::Jump {
            // unroll.py:238-242 jump_to_preamble sends the live JUMP after
            // force_box / send_extra_operation rewrites. Body args reference
            // the trace inputarg slots OpRef(0)..OpRef(start_label_args.len());
            // remap those positional refs to start_label_args[i].
            let mapped_base_args: Vec<OpRef> = new_op
                .getarglist()
                .iter()
                .map(|arg| {
                    let arg = arg.to_opref();
                    if jump_to_self {
                        return arg;
                    }
                    // unroll.py:364 `_map_args` passes Const through unchanged.
                    // Inline-Const variants (history.py:227/268/314) panic on
                    // `.raw()`; the inline payload IS the value and has no
                    // positional remap target.
                    if arg.is_constant() {
                        return arg;
                    }
                    let i = arg.raw() as usize;
                    start_label_args.get(i).copied().unwrap_or(arg)
                })
                .collect();
            let target_label_args: Vec<OpRef> = current_inner_label_index
                .and_then(|label_idx| {
                    result
                        .get(label_idx)
                        .map(|op| op.getarglist().iter().map(|a| a.to_opref()).collect())
                })
                .unwrap_or_else(|| full_label_args.clone());
            let target_base_len = if current_inner_label_index.is_some() {
                original_args.len()
            } else {
                label_args.len()
            };
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] assemble_jump: inner_label={:?} original_args={:?} mapped_base_args={:?} label_args={:?} filtered_extra_jump_args={:?}",
                    current_inner_label_index,
                    original_args,
                    mapped_base_args,
                    label_args,
                    filtered_extra_jump_args,
                );
            }
            let mut jump_args = mapped_base_args;
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
                    // Extra args are assembly-allocated label positions;
                    // body-scoped remaps never contain them. Probe across 14
                    // benchmarks × 2 backends showed the prior lookup chain
                    // returned identity in 100% of fires. Pass through.
                    jump_args.push(extra_arg);
                }
            }
            // unroll.py-style bulk replace: jump arity is finalized here.
            let mut jump_args_box: smallvec::SmallVec<[majit_ir::operand::Operand; 3]> =
                smallvec::SmallVec::with_capacity(jump_args.len());
            for a in &jump_args {
                jump_args_box.push(ctx.materialize_operand_at(*a));
            }
            new_op.initarglist(jump_args_box);
        }
        // RPython resume.py parity: fail_args capture guard-point state
        // (the snapshot), not the body's final state. body_result_remap
        // applies only to values body-defined AND not visible before the
        // label, so snapshot refs to label args stay intact.
        if let Some(fa) = new_op.fail_args_mut() {
            for a in fa.iter_mut() {
                if let Some(&mapped) = body_result_remap.get(&a.to_opref()) {
                    if seen_body_defs.contains(&a.to_opref())
                        && !visible_before_label.contains(&a.to_opref())
                    {
                        *a = match emitted_at.get(&mapped) {
                            Some(rc) => majit_ir::operand::Operand::from_bound_op(rc),
                            None => majit_ir::operand::Operand::from_opref(mapped),
                        };
                    }
                }
            }
        }
        if let Some(label_idx) = current_inner_label_index {
            let mut extra_live_args = Vec::new();
            let label_args: smallvec::SmallVec<[OpRef; 3]> = result[label_idx]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect();
            for arg in new_op.getarglist().iter().map(|a| a.to_opref()).chain(
                new_op
                    .getfailargs()
                    .into_iter()
                    .flatten()
                    .map(|b| b.to_opref()),
            ) {
                // unroll.py:364 `_map_args` passes Const through; inline-Const
                // (history.py:227/268/314) carries its value on the OpRef and
                // is never an inner-label-extension candidate. Short-circuit
                // before `.raw()` (panics on inline-Const variants).
                if arg.is_none()
                    || arg.is_constant()
                    || constants.contains_key(&arg.raw())
                    || defs_since_inner_label.contains(&arg)
                    || label_args.contains(&arg)
                    || extra_live_args.contains(&arg)
                {
                    continue;
                }
                extra_live_args.push(arg);
            }
            if !extra_live_args.is_empty() {
                let existing: indexmap::IndexSet<OpRef> = result[label_idx]
                    .getarglist()
                    .iter()
                    .map(|a| a.to_opref())
                    .collect();
                let mut new_args: smallvec::SmallVec<[majit_ir::operand::Operand; 3]> =
                    (0..result[label_idx].num_args())
                        .map(|i| result[label_idx].arg(i))
                        .collect();
                new_args.extend(
                    extra_live_args
                        .into_iter()
                        .filter(|arg| !existing.contains(arg))
                        .map(|arg| ctx.materialize_operand_at(arg)),
                );
                result[label_idx].initarglist(new_args);
            }
        }
        // RPython parity: each guard in the assembled trace owns a
        // distinct ResumeGuardDescr with a globally unique fail_index.
        // optimizeopt::store_final_boxes_in_guard (mod.rs:3392-3404,
        // commit 43c64ee0bb) installs a fresh ResumeGuardDescr on every
        // optimizer-routed guard, and Optimizer::_copy_resume_data_from /
        // OptContext::emit_guard_operation share-branch (commit
        // 329297b38a) call Descr::clone_descr to allocate a fresh
        // fail_index for the sharing-path guard. Both phases of the
        // unroll optimizer therefore emit body guards with unique
        // descrs already; no post-process re-stamping is needed.
        let new_rc = std::rc::Rc::new(new_op);
        if new_rc.result_type() != Type::Void && !new_rc.pos.get().is_none() {
            emitted_at.insert(new_rc.pos.get(), new_rc.clone());
        }
        result.push(new_rc);
        if op.opcode == OpCode::Label {
            current_inner_label_index = Some(result.len() - 1);
            defs_since_inner_label.clear();
        }
        if op.result_type() != Type::Void && !op.pos.get().is_none() {
            seen_body_defs.insert(op.pos.get());
            defs_since_inner_label.insert(result.last().unwrap().pos.get());
        }
    }

    if crate::callee_rca_enabled() {
        let labels: Vec<_> = result
            .iter()
            .enumerate()
            .filter(|(_, op)| op.opcode == OpCode::Label)
            .map(|(idx, op)| {
                (
                    idx,
                    op.pos.get(),
                    op.getarglist()
                        .iter()
                        .map(|a| a.to_opref())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let jumps: Vec<_> = result
            .iter()
            .enumerate()
            .filter(|(_, op)| op.opcode == OpCode::Jump)
            .map(|(idx, op)| {
                (
                    idx,
                    op.pos.get(),
                    op.getarglist()
                        .iter()
                        .map(|a| a.to_opref())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        eprintln!(
            "[callee-rca][assembled-loop] labels={:?} jumps={:?}",
            labels, jumps
        );
    }

    result
}

fn splice_redirected_tail(
    body_ops: &[majit_ir::OpRc],
    redirected_tail_ops: &[majit_ir::OpRc],
) -> Vec<majit_ir::OpRc> {
    let mut result = Vec::with_capacity(body_ops.len() + redirected_tail_ops.len());
    let split_idx = body_ops
        .iter()
        .rposition(|op| op.opcode == OpCode::Jump)
        .unwrap_or(body_ops.len());
    result.extend_from_slice(&body_ops[..split_idx]);
    result.extend_from_slice(redirected_tail_ops);
    result
}

fn replace_terminal_jump(body_ops: &[majit_ir::OpRc], jump_op: Op) -> Vec<majit_ir::OpRc> {
    let mut result = Vec::with_capacity(body_ops.len() + 1);
    let split_idx = body_ops
        .iter()
        .rposition(|op| op.opcode == OpCode::Jump)
        .unwrap_or(body_ops.len());
    result.extend_from_slice(&body_ops[..split_idx]);
    result.push(std::rc::Rc::new(jump_op));
    result
}

fn reshape_jump_args_for_preamble(jump_args: &mut Vec<OpRef>, preamble_args: &[OpRef]) {
    if jump_args.len() > preamble_args.len() {
        jump_args.truncate(preamble_args.len());
    }
    while jump_args.len() < preamble_args.len() {
        jump_args.push(preamble_args[jump_args.len()]);
    }
}

fn pick_virtual_state(
    my_vs: &crate::optimizeopt::virtualstate::VirtualState,
    target_states: &[crate::optimizeopt::virtualstate::VirtualState],
    ctx: &mut OptContext,
) -> Option<usize> {
    for (i, target_vs) in target_states.iter().enumerate() {
        if target_vs.generalization_of(my_vs, ctx) {
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
    /// Emit the peeled iteration, the loop Label, and the body iteration
    /// in that order, returning the `original_pos -> body_pos` map the
    /// caller uses to rewrite the back-edge Jump's args.
    ///
    /// Positions are allocated through `reserve_pos` (which honors the
    /// `inputarg_base + num_inputs` floor) instead of being computed
    /// arithmetically from `new_operations.len()`. The arithmetic form
    /// is fine for production traces (inputargs occupy a disjoint low
    /// range) but collides with the 1024-slot `trace_inputargs`
    /// stubs that the unit-test harnesses seed. Using `reserve_pos`
    /// closes that collision and lets the Box.type invariant enforce
    /// itself uniformly at `emit()` / `emit_extra()` /
    /// `propagate_from_pass_range`.
    fn peel_iteration(
        &self,
        jump_op: &Op,
        ctx: &mut OptContext,
    ) -> indexmap::IndexMap<OpRef, OpRef> {
        // First pass: reserve peeled-iteration positions, tagged with each
        // source op's result type ( `OpRef.ty()`
        // matches RPython's `box.type` at allocation time).
        let peeled_positions: Vec<OpRef> = self
            .buffer
            .iter()
            .map(|op| ctx.reserve_pos_typed(op.result_type()))
            .collect();
        let mut ref_map: indexmap::IndexMap<OpRef, OpRef> = indexmap::IndexMap::new();
        for (op, &new_pos) in self.buffer.iter().zip(peeled_positions.iter()) {
            ref_map.insert(op.pos.get(), new_pos);
        }

        // Emit peeled iteration with remapped refs.
        for (op, &new_pos) in self.buffer.iter().zip(peeled_positions.iter()) {
            let mut peeled = op.clone();
            peeled.pos.set(new_pos);
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..peeled.num_args() {
                let arg = peeled.arg(i);
                if let Some(&new_ref) = ref_map.get(&arg.to_opref()) {
                    peeled.setarg(i, remapped_producer_operand(ctx, new_ref));
                }
                // Args referencing ops outside the buffer (e.g., input args)
                // are kept as-is.
            }
            if let Some(fa) = peeled.fail_args_mut() {
                for arg in fa.iter_mut() {
                    if let Some(&new_ref) = ref_map.get(&arg.to_opref()) {
                        *arg = remapped_producer_operand(ctx, new_ref);
                    }
                }
            }
            if peeled.opcode.is_guard() {
                clone_guard_snapshot_remapped(ctx, &mut peeled, &ref_map);
                // opencoder.py:391-401 parity: trace iteration strips guard
                // descrs, keeping only rd_resume_position. optimizer.py then
                // invents a fresh opcode-appropriate descr at emission time.
                peeled.cleardescr();
            }
            ctx.emit(peeled);
        }

        // Emit Label between peeled and original body.
        // The Label's args match the Jump's args, forming the loop header.
        let label_pos = ctx.reserve_pos_typed(OpCode::Label.result_type());
        let jump_label_args: Vec<majit_ir::operand::Operand> =
            (0..jump_op.num_args()).map(|i| jump_op.arg(i)).collect();
        let mut label_op = Op::new(OpCode::Label, &jump_label_args);
        label_op.pos.set(label_pos);
        ctx.emit(label_op);

        // Reserve body positions, tagged per source op ( ).
        let body_positions: Vec<OpRef> = self
            .buffer
            .iter()
            .map(|op| ctx.reserve_pos_typed(op.result_type()))
            .collect();
        let mut orig_ref_map: indexmap::IndexMap<OpRef, OpRef> = indexmap::IndexMap::new();
        for (op, &new_pos) in self.buffer.iter().zip(body_positions.iter()) {
            orig_ref_map.insert(op.pos.get(), new_pos);
        }

        // Emit original body ops with remapped positions and refs.
        for (op, &new_pos) in self.buffer.iter().zip(body_positions.iter()) {
            let mut body_op = op.clone();
            body_op.pos.set(new_pos);
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..body_op.num_args() {
                let arg = body_op.arg(i);
                if let Some(&new_ref) = orig_ref_map.get(&arg.to_opref()) {
                    body_op.setarg(i, remapped_producer_operand(ctx, new_ref));
                }
            }
            if let Some(fa) = body_op.fail_args_mut() {
                for arg in fa.iter_mut() {
                    if let Some(&new_ref) = orig_ref_map.get(&arg.to_opref()) {
                        *arg = remapped_producer_operand(ctx, new_ref);
                    }
                }
            }
            if body_op.opcode.is_guard() {
                clone_guard_snapshot_remapped(ctx, &mut body_op, &orig_ref_map);
                // Same opencoder.py guard-descr stripping as the peeled copy.
                // ResumeAtPositionDescr is inline_short_preamble-only
                // (unroll.py:406-409), not normal peeled body guard state.
                body_op.cleardescr();
            }
            ctx.emit(body_op);
        }
        orig_ref_map
    }
}

/// Resolve a remapped peel position to its canonical producer operand,
/// mirroring the import-path binding (`get_box_replacement_operand_opt`, else
/// `materialize_operand_at` — unroll.rs:2997-3024 / S10). The peeled / body
/// producer emitted at `new_ref` precedes its consumers' arg writes (SSA
/// def-before-use), so the read resolves to the bound `Op`; the
/// `materialize_operand_at` arm mints a registered stand-in that the producer's
/// later `emit` catches up (mod.rs forward-reference path), never a
/// position-only operand.
fn remapped_producer_operand(ctx: &mut OptContext, new_ref: OpRef) -> Operand {
    match ctx.get_box_replacement_operand_opt(new_ref) {
        Some(o) => o,
        None => ctx.materialize_operand_at(new_ref),
    }
}

fn fresh_snapshot_key(ctx: &OptContext) -> i32 {
    next_snapshot_pos(&ctx.snapshot_boxes)
}

fn remap_snapshot_boxes(
    boxes: &[SnapshotBox],
    ref_map: &indexmap::IndexMap<OpRef, OpRef>,
) -> Vec<SnapshotBox> {
    boxes
        .iter()
        .map(|boxref| boxref.map_opref(|opref| ref_map.get(&opref).copied().unwrap_or(opref)))
        .collect()
}

fn clone_guard_snapshot_remapped(
    ctx: &mut OptContext,
    guard: &mut Op,
    ref_map: &indexmap::IndexMap<OpRef, OpRef>,
) {
    let old_pos = guard.rd_resume_position.get();
    if old_pos < 0 {
        return;
    }
    let Some(snapshot_boxes) = snapshot_get(&ctx.snapshot_boxes, old_pos).cloned() else {
        return;
    };

    // RPython TraceIterator decodes snapshot box references against the
    // current iteration's box cache.  Majit keeps snapshots in side tables,
    // so cloned guards need a cloned snapshot table entry with the same
    // raw-box -> cloned-box mapping used for op args.
    let new_pos = fresh_snapshot_key(ctx);
    snapshot_insert(
        &mut ctx.snapshot_boxes,
        new_pos,
        remap_snapshot_boxes(&snapshot_boxes, ref_map),
    );
    if let Some(vable_boxes) = snapshot_get(&ctx.snapshot_vable_boxes, old_pos).cloned() {
        snapshot_insert(
            &mut ctx.snapshot_vable_boxes,
            new_pos,
            remap_snapshot_boxes(&vable_boxes, ref_map),
        );
    }
    if let Some(vref_boxes) = snapshot_get(&ctx.snapshot_vref_boxes, old_pos).cloned() {
        snapshot_insert(
            &mut ctx.snapshot_vref_boxes,
            new_pos,
            remap_snapshot_boxes(&vref_boxes, ref_map),
        );
    }
    if let Some(frame_pcs) = snapshot_get(&ctx.snapshot_frame_pcs, old_pos).cloned() {
        snapshot_insert(&mut ctx.snapshot_frame_pcs, new_pos, frame_pcs);
    }
    if let Some(frame_sizes) = snapshot_get(&ctx.snapshot_frame_sizes, old_pos).cloned() {
        snapshot_insert(&mut ctx.snapshot_frame_sizes, new_pos, frame_sizes);
    }
    guard.rd_resume_position.set(new_pos);
}

impl Default for OptUnroll {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptUnroll {
    fn propagate_forward(
        &mut self,
        op: &Op,
        _op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // Only peel once per trace, and only for Jump (back-edge).
        if op.opcode == OpCode::Jump && !self.seen_jump {
            self.seen_jump = true;

            if self.buffer.is_empty() {
                // Empty loop body, nothing to peel.
                return OptimizationResult::Emit(op.clone());
            }

            // Perform the peeling: emit peeled body + Label + original body.
            // The returned map forwards original `buffer` positions to
            // their freshly-allocated body positions so we can rewrite
            // the back-edge Jump's args without re-deriving the layout.
            let orig_ref_map = self.peel_iteration(op, ctx);

            // Emit the final Jump with remapped args pointing to the
            // body iteration's ops.
            let mut jump = op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..jump.num_args() {
                let arg = jump.arg(i);
                if let Some(&new_ref) = orig_ref_map.get(&arg.to_opref()) {
                    jump.setarg(i, remapped_producer_operand(ctx, new_ref));
                }
            }
            // Reserve the Jump's own position so it lands above any
            // inputarg range and above every body op allocated by
            // peel_iteration. Jump is Void-typed.
            jump.pos.set(ctx.reserve_pos_typed(jump.result_type()));
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
    use crate::history::test_support::{rooted_inputarg_operand, rooted_resop_operand};
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::GcRef;
    use majit_ir::operand::Operand;

    /// Assign sequential positions to ops starting from `base`.
    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos
                .set(OpRef::op_typed(base + i as u32, op.opcode.result_type()));
        }
    }

    #[test]
    fn test_exported_state_high_water_covers_retrace_namespace() {
        let exported = ExportedState::new(
            vec![OpRef::int_op(52)],
            vec![0],
            vec![
                rooted_resop_operand(Type::Int, 109),
                Operand::from_opref(OpRef::const_int(3)),
            ],
            crate::optimizeopt::virtualstate::VirtualState::new(Vec::new()),
            indexmap::IndexMap::new(),
            Vec::new(),
            vec![OpRef::int_op(14)],
            vec![OpRef::int_op(23)],
            // short_inputarg_refs: this fixture stores plain OpRef positions,
            // so no parallel InputArgRc pool is needed here.
            Vec::new(),
        );

        assert_eq!(exported.opref_high_water(), 110);
    }

    fn run_unroll_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptUnroll::new()));
        // See `run_heap_opt` in heap.rs for the 1024-slot Ref seed
        // rationale — the preamble exporter needs an intrinsic type
        // for every renamed inputarg, which production derives from
        // the recorder's trace_inputargs.
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&vec![majit_ir::Type::Ref; 1024]);
        // Production trace iteration gets rd_resume_position from
        // opencoder.py:399-401.  Direct unit-test guards must seed the
        // corresponding snapshot explicitly before store_final_boxes_in_guard.
        let (ops, snapshots) = super::super::seed_guard_snapshots_with(ops, |guard| {
            // Direct unroll tests use guard brackets as the explicit active
            // snapshot so remapping can verify the TraceIterator cache
            // semantics that RPython gets from opencoder.py.
            guard
                .getfailargs()
                .map(|fail_args| fail_args.iter().map(|a| a.to_opref()).collect())
                .unwrap_or_default()
        });
        opt.snapshot_boxes = snapshots;
        opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::new(), 1024)
    }

    // ── Basic peeling ─────────────────────────────────────────────────

    #[test]
    fn test_no_jump_no_unroll() {
        // Without a Jump back-edge, the pass just buffers and nothing is emitted.
        // (In practice, traces always end with Jump or Finish.)
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(OpCode::Finish, &[rooted_resop_operand(Type::Int, 0)]),
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
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 0),
                        rooted_resop_operand(Type::Int, 1),
                    ],
                );
                op.pos.set(OpRef::int_op(2));
                op
            },
            Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 2),
                    rooted_resop_operand(Type::Int, 50),
                ],
            ),
        ];
        let preamble_target = TargetToken::new_preamble(7);

        let body_ops: Vec<majit_ir::OpRc> = body_ops.into_iter().map(std::rc::Rc::new).collect();
        let result = UnrollOptimizer::jump_to_preamble(&body_ops, &preamble_target);
        assert_eq!(result[1].opcode, OpCode::Jump);
        assert_eq!(
            result[1]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(0), OpRef::int_op(2), OpRef::int_op(50)]
        );
        assert_eq!(
            result[1].getdescr().map(|descr| descr.repr()),
            Some("LoopTargetDescr(start:7)".to_string())
        );
    }

    #[test]
    fn test_reshape_jump_args_for_preamble_pads_missing_invariants() {
        let mut jump_args = vec![OpRef::int_op(0)];
        reshape_jump_args_for_preamble(&mut jump_args, &[OpRef::int_op(0), OpRef::int_op(1)]);
        assert_eq!(jump_args.as_slice(), &[OpRef::int_op(0), OpRef::int_op(1)]);
    }

    #[test]
    fn test_reshape_jump_args_for_preamble_truncates_extra_slots() {
        let mut jump_args = vec![OpRef::int_op(0), OpRef::int_op(1), OpRef::int_op(2)];
        reshape_jump_args_for_preamble(&mut jump_args, &[OpRef::int_op(10), OpRef::int_op(11)]);
        assert_eq!(jump_args.as_slice(), &[OpRef::int_op(0), OpRef::int_op(1)]);
    }

    #[test]
    fn test_replace_terminal_jump_appends_when_body_prefix_has_no_jump() {
        let body_ops = vec![{
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            );
            op.pos.set(OpRef::int_op(2));
            op
        }];
        let mut jump = Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 2)]);
        jump.setdescr(TargetToken::new_preamble(7).as_jump_target_descr());

        let body_ops: Vec<majit_ir::OpRc> = body_ops.into_iter().map(std::rc::Rc::new).collect();
        let result = replace_terminal_jump(&body_ops, jump);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::Jump);
        assert_eq!(
            result[1]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(2)]
        );
        assert_eq!(
            result[1].getdescr().map(|descr| descr.repr()),
            Some("LoopTargetDescr(start:7)".to_string())
        );
    }

    #[test]
    fn test_ensure_preamble_target_token_inserts_start_descr_first() {
        let mut unroll = UnrollOptimizer::new();
        let mut regular = TargetToken::new_loop(3);
        regular.virtual_state = Some(crate::optimizeopt::virtualstate::VirtualState::new(vec![
            crate::optimizeopt::virtualstate::VirtualStateInfo::Unknown(majit_ir::Type::Int),
        ]));
        unroll.target_tokens.push(regular);

        unroll.ensure_preamble_target_token();

        assert_eq!(unroll.target_tokens.len(), 2);
        assert!(unroll.target_tokens[0].is_preamble_target);
        assert!(unroll.target_tokens[0].virtual_state.is_none());
        assert_eq!(unroll.target_tokens[1].token_id, 3);
    }

    #[test]
    fn test_pick_virtual_state_uses_target_generalization_direction() {
        // NonNull is Ref-typed; target Unknown must also be Ref to match.
        // RPython: NotVirtualStateInfoPtr(LEVEL_UNKNOWN) generalizes
        // NotVirtualStateInfoPtr(LEVEL_NONNULL).
        let my_vs = crate::optimizeopt::virtualstate::VirtualState::new(vec![
            crate::optimizeopt::virtualstate::VirtualStateInfo::NonNull,
        ]);
        let target_states = vec![
            crate::optimizeopt::virtualstate::VirtualState::new(vec![
                crate::optimizeopt::virtualstate::VirtualStateInfo::Unknown(majit_ir::Type::Ref),
            ]),
            crate::optimizeopt::virtualstate::VirtualState::new(vec![
                crate::optimizeopt::virtualstate::VirtualStateInfo::KnownClass {
                    class_ptr: 0x1234,
                },
            ]),
        ];

        let mut ctx = OptContext::new(128);
        assert_eq!(
            pick_virtual_state(&my_vs, &target_states, &mut ctx),
            Some(0)
        );
    }

    #[test]
    fn test_exported_state_walks_inline_const_ptr_slots() {
        use crate::optimizeopt::info::{OpInfo, PtrInfo};
        use crate::optimizeopt::shortpreamble::{
            PreambleOp, PreambleOpKind, ProducedShortOp, ShortPreamble, ShortPreambleOp,
        };
        use crate::optimizeopt::virtualstate::{VirtualState, VirtualStateInfo};

        let old = GcRef(0x1111_0000);
        let new = GcRef(0x2222_0000);
        let old_ref = OpRef::const_ptr(old);
        let new_ref = OpRef::const_ptr(new);
        let mut exported_infos = indexmap::IndexMap::new();
        exported_infos.insert(
            Operand::from_opref(old_ref),
            OpInfo::ptr(PtrInfo::Constant(old)),
        );
        let mut short_box_const_values = indexmap::IndexMap::new();
        short_box_const_values.insert(old_ref, Value::Ref(old));
        let mut constants = majit_ir::ConstMap::new();
        constants.insert(0, majit_ir::Const::Ref(old));

        let mut state = ExportedState::new(
            vec![old_ref],
            Vec::new(),
            vec![Operand::from_opref(old_ref)],
            VirtualState::new(vec![VirtualStateInfo::Constant(Value::Ref(old))]),
            exported_infos,
            vec![PreambleOp {
                op: std::rc::Rc::new(Op::new(OpCode::SameAsR, &[Operand::from_opref(old_ref)])),
                res: Operand::bound_from_opref(old_ref),
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: Some(Operand::from_opref(old_ref)),
            }],
            vec![old_ref],
            vec![old_ref],
            // short_inputarg_refs: `old_ref` is a ConstPtr inline position.
            Vec::new(),
        );
        state.short_boxes.push((
            old_ref,
            ProducedShortOp {
                kind: PreambleOpKind::Pure,
                res: Operand::from_opref(old_ref),
                preamble_op: std::rc::Rc::new(Op::new(
                    OpCode::SameAsR,
                    &[Operand::from_opref(old_ref)],
                )),
                invented_name: false,
                same_as_source: Some(Operand::from_opref(old_ref)),
                label_arg_idx: None,
            },
        ));
        state.short_box_const_values = short_box_const_values;
        state.short_preamble = Some(ShortPreamble {
            ops: vec![ShortPreambleOp {
                op: Op::new(OpCode::GuardNonnull, &[Operand::from_opref(old_ref)]),
                arg_mapping: Vec::new(),
                fail_arg_mapping: Vec::new(),
            }],
            inputargs: vec![old_ref],
            used_boxes: vec![old_ref],
            jump_args: vec![old_ref],
            exported_state: Some(VirtualState::new(vec![VirtualStateInfo::KnownClass {
                class_ptr: old.as_usize() as i64,
            }])),
            constants,
            inputarg_infos: vec![Some(PtrInfo::Constant(old))],
            phase1_inputargs: Some(vec![old_ref]),
        });
        state.runtime_boxes.push(old_ref);
        state.patchguardop = Some(Op::new(
            OpCode::GuardNonnull,
            &[Operand::from_opref(old_ref)],
        ));

        state.walk_const_ptr_refs_mut(&mut |slot| {
            if *slot == old {
                *slot = new;
            }
        });

        assert_eq!(state.end_args[0], new_ref);
        assert_eq!(state.next_iteration_args[0].to_opref(), new_ref);
        assert_eq!(state.renamed_inputargs[0], new_ref);
        assert_eq!(state.short_inputargs[0], new_ref);
        assert_eq!(state.runtime_boxes[0], new_ref);
        assert!(state.exported_infos.keys().any(|k| k.to_opref() == new_ref));
        assert_eq!(state.exported_short_boxes[0].op.arg(0).to_opref(), new_ref);
        assert_eq!(
            state.exported_short_boxes[0]
                .same_as_source
                .as_ref()
                .map(|b| b.to_opref()),
            Some(new_ref)
        );
        let produced = state
            .short_boxes
            .iter()
            .find_map(|(key, produced)| (*key == new_ref).then_some(produced))
            .expect("short_boxes key must be forwarded");
        assert_eq!(produced.preamble_op.arg(0).to_opref(), new_ref);
        assert_eq!(
            produced.same_as_source.as_ref().map(|b| b.to_opref()),
            Some(new_ref)
        );
        assert_eq!(
            state.short_box_const_values.get(&new_ref),
            Some(&Value::Ref(new))
        );
        assert_eq!(
            state.patchguardop.as_ref().map(|op| op.arg(0).to_opref()),
            Some(new_ref)
        );
        match &state.virtual_state.state[0].info {
            VirtualStateInfo::Constant(Value::Ref(gcref)) => assert_eq!(*gcref, new),
            other => panic!("unexpected virtual state after walk: {other:?}"),
        }
        let short = state.short_preamble.as_ref().unwrap();
        assert_eq!(short.ops[0].op.arg(0).to_opref(), new_ref);
        assert_eq!(short.inputargs[0], new_ref);
        assert_eq!(short.used_boxes[0], new_ref);
        assert_eq!(short.jump_args[0], new_ref);
        assert_eq!(short.phase1_inputargs.as_ref().unwrap()[0], new_ref);
        assert_eq!(short.constants.get(&0), Some(&majit_ir::Const::Ref(new)));
    }

    #[test]
    fn test_simple_loop_peeled() {
        // A simple loop: one add op, then Jump.
        // Expected output: peeled_add, Label, original_add, Jump
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
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
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(
                OpCode::IntSub,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 2 peeled + Label + 2 original + Jump = 6
        assert_eq!(result.len(), 6);

        // All positions should be unique.
        let positions: Vec<OpRef> = result.iter().map(|op| op.pos.get()).collect();
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
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(
                OpCode::IntMul,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ), // references op0
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
        assert_eq!(peeled_mul.arg(0).to_opref(), peeled_add.pos.get());
        // Second arg (input ref) should be unchanged.
        assert_eq!(peeled_mul.arg(1).to_opref(), OpRef::int_op(101));

        // Original body:
        let body_add = &result[3];
        let body_mul = &result[4];
        assert_eq!(body_add.opcode, OpCode::IntAdd);
        assert_eq!(body_mul.opcode, OpCode::IntMul);
        // body_mul should reference body_add's position.
        assert_eq!(body_mul.arg(0).to_opref(), body_add.pos.get());
        assert_eq!(body_mul.arg(1).to_opref(), OpRef::int_op(101));
    }

    #[test]
    fn test_external_refs_preserved() {
        // Refs to ops outside the buffer (input arguments) should not be remapped.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // Peeled add should still reference v100 and v101.
        assert_eq!(result[0].arg(0).to_opref(), OpRef::int_op(100));
        assert_eq!(result[0].arg(1).to_opref(), OpRef::int_op(101));

        // Body add should also reference v100 and v101.
        assert_eq!(result[2].arg(0).to_opref(), OpRef::int_op(100));
        assert_eq!(result[2].arg(1).to_opref(), OpRef::int_op(101));
    }

    // ── Guard preservation ────────────────────────────────────────────

    #[test]
    fn test_guards_duplicated_in_peel() {
        // Guards in the preamble serve as type checks.
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 100)]),
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
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
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            {
                let mut guard = Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 100)]);
                guard.setfailargs(vec![rooted_resop_operand(Type::Int, 0)].into()); // refs op0
                guard
            },
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // Check peeled guard's fail_args.
        let peeled_guard = &result[1];
        assert_eq!(peeled_guard.opcode, OpCode::GuardTrue);
        let peeled_add_pos = result[0].pos.get();
        assert_eq!(
            peeled_guard.getfailargs().unwrap()[0].to_opref(),
            peeled_add_pos,
            "peeled guard's fail_args should reference peeled add"
        );

        // Check body guard's fail_args.
        let body_guard = &result[4]; // after Label (idx 3) and body_add (idx 3)
        assert_eq!(body_guard.opcode, OpCode::GuardTrue);
        let body_add_pos = result[3].pos.get();
        assert_eq!(
            body_guard.getfailargs().unwrap()[0].to_opref(),
            body_add_pos,
            "body guard's fail_args should reference body add"
        );
    }

    // ── Jump args remapping ───────────────────────────────────────────

    #[test]
    fn test_jump_args_remapped_to_body() {
        // Jump args should reference the body's ops, not the original positions.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 0)]), // carries v0 (the add result)
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // peeled_add, Label, body_add, Jump
        assert_eq!(result.len(), 4);

        let jump = result.last().unwrap();
        assert_eq!(jump.opcode, OpCode::Jump);

        let body_add_pos = result[2].pos.get();
        assert_eq!(
            jump.arg(0).to_opref(),
            body_add_pos,
            "Jump arg should reference body add, not original"
        );
    }

    #[test]
    fn test_label_args_match_jump_args() {
        // The Label should carry the same args as the Jump.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 100),
                ],
            ),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        let label = result.iter().find(|o| o.opcode == OpCode::Label).unwrap();
        let jump_args = ops.last().unwrap().getarglist_copy();
        assert_eq!(
            label
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            jump_args.iter().map(|a| a.to_opref()).collect::<Vec<_>>(),
            "Label args should match original Jump args"
        );
    }

    // ── Multiple ops in loop body ─────────────────────────────────────

    #[test]
    fn test_multi_op_loop() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(
                OpCode::IntSub,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(
                OpCode::IntMul,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            ),
            Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 2)]),
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 2)]),
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
        pass.buffer.push(Op::new(
            OpCode::IntAdd,
            &[rooted_resop_operand(Type::Int, 0)],
        ));
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
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 0)]),
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 0)]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptUnroll::new()));
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&vec![majit_ir::Type::Ref; 1024]);
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::new(), 1024);

        // Expect: peeled_add, peeled_guard, Label, body_add, body_guard, Jump = 6
        assert_eq!(result.len(), 6);

        // All ops should have valid (non-NONE) positions.
        for op in &result {
            assert!(
                !op.pos.get().is_none(),
                "op {:?} should have a valid pos",
                op.opcode
            );
        }
    }

    #[test]
    fn test_unroll_mints_fresh_guard_descrs() {
        // RPython parity: normal trace guards do not keep their encoded descr.
        //
        // opencoder.py:391 strips guard descrs to None during trace iteration;
        // optimizer.py:724-729 invents a fresh opcode-specific descr at guard
        // emission. ResumeAtPositionDescr is only for inline_short_preamble().
        let descr = crate::compile::make_resume_guard_descr_typed(Vec::new());
        let original_index = descr.index();
        let mut ops = vec![
            Op::with_descr(
                OpCode::GuardTrue,
                &[rooted_resop_operand(Type::Int, 100)],
                descr.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        let guards: Vec<&Op> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardTrue)
            .collect();
        assert_eq!(guards.len(), 2);
        for guard in &guards {
            assert!(guard.has_descr(), "guard should have a descriptor");
            assert!(
                guard.getdescr().unwrap().is_resume_guard(),
                "optimizer.py:723: descr must be a ResumeGuardDescr subtype"
            );
        }
        let peel_index = guards[0].getdescr().unwrap().index();
        let body_index = guards[1].getdescr().unwrap().index();
        assert_ne!(peel_index, original_index);
        assert_ne!(body_index, original_index);
        assert_ne!(
            peel_index, body_index,
            "peeled and body guards must own distinct freshly invented descrs"
        );
        assert!(!guards[0].getdescr().unwrap().is_resume_at_position());
        assert!(!guards[1].getdescr().unwrap().is_resume_at_position());
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
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 100),
                    rooted_resop_operand(Type::Int, 101),
                ],
            ),
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 100),
                ],
            ),
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 1),
                    rooted_resop_operand(Type::Int, 0),
                ],
            ),
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 2)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 3 peeled + Label + 3 body + Jump = 8
        assert_eq!(result.len(), 8);

        // Peeled iteration refs:
        let p0 = result[0].pos.get();
        let p1 = result[1].pos.get();
        let _p2 = result[2].pos.get();
        assert_eq!(
            result[1].arg(0).to_opref(),
            p0,
            "peeled v1 should ref peeled v0"
        );
        assert_eq!(
            result[2].arg(0).to_opref(),
            p1,
            "peeled v2 should ref peeled v1"
        );
        assert_eq!(
            result[2].arg(1).to_opref(),
            p0,
            "peeled v2 should ref peeled v0"
        );

        // Body refs:
        let b0 = result[4].pos.get();
        let b1 = result[5].pos.get();
        let b2 = result[6].pos.get();
        assert_eq!(
            result[5].arg(0).to_opref(),
            b0,
            "body v1 should ref body v0"
        );
        assert_eq!(
            result[6].arg(0).to_opref(),
            b1,
            "body v2 should ref body v1"
        );
        assert_eq!(
            result[6].arg(1).to_opref(),
            b0,
            "body v2 should ref body v0"
        );

        // Jump should reference body v2.
        let jump = &result[7];
        assert_eq!(jump.arg(0).to_opref(), b2, "Jump should ref body v2");
    }

    #[test]
    fn test_unroll_optimizer_optimize_trace() {
        let mut unroll_opt = UnrollOptimizer::new();
        // IntAdd operates on Int-typed inputs — seed the inner phase1/2
        // optimizers' trace_inputargs via UnrollOptimizer so the
        // intbounds pass sees Int on the two inputargs.
        unroll_opt.trace_inputargs =
            majit_ir::OpRef::inputarg_refs(&[majit_ir::Type::Int, majit_ir::Type::Int]);
        // Use optimize_trace_with_constants_and_inputs to properly set
        // num_inputs so input args don't collide with op positions. Args
        // address inputarg slots via `InputArg*` OpRef variants so the
        // operand shape (`with_inputarg_types` plants the inputarg operands)
        // and the orthodox `_forwarded` mirror agree on the namespace.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_inputarg_operand(Type::Int, 0),
                    rooted_inputarg_operand(Type::Int, 1),
                ],
            ),
            Op::new(OpCode::Jump, &[rooted_inputarg_operand(Type::Int, 0)]),
        ];
        assign_positions(&mut ops, 2);
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::new();
        let (result, _) =
            unroll_opt.optimize_trace_with_constants_and_inputs(&ops, &mut constants, 2);
        // The optimizer processes the trace; result should not be empty
        assert!(!result.is_empty(), "optimize_trace should produce output");
    }

    #[test]
    fn test_unroll_optimizer_count_guards() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 0)]),
            Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            ),
            Op::new(OpCode::GuardNonnull, &[rooted_resop_operand(Type::Int, 0)]),
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 0)]),
        ];
        assert_eq!(UnrollOptimizer::count_guards(&ops), 2);
    }

    #[test]
    fn test_exported_state_reimports_widened_intbounds() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        use crate::optimizeopt::intutils::IntBound;

        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(4, 0);
        let mut exported_bounds: indexmap::IndexMap<majit_ir::operand::Operand, IntBound> =
            indexmap::IndexMap::new();
        // Bind the export-input position at its source (a forced end-arg is a
        // bound box in production); virtualstate.py:711-720 create_state
        // receives real AbstractValues.
        ctx.materialize_operand_at(OpRef::int_op(21));
        // Key by the canonical box identity the consumer resolves `int_op(21)`
        // to, so the operand-keyed lookup hits by `Rc::ptr_eq`.
        let box21 = ctx
            .get_box_replacement_operand_opt(OpRef::int_op(21))
            .expect("int_op(21) bound to a box");
        exported_bounds.insert(box21, IntBound::bounded(10, 20));

        let exported = export_state(
            &[OpRef::int_op(21)],
            &[],
            &mut optimizer,
            &mut ctx,
            Some(&exported_bounds),
        );

        assert_eq!(
            match exported
                .exported_infos
                .iter()
                .find(|(k, _)| k.to_opref() == OpRef::int_op(21))
                .map(|(_, v)| v)
                .unwrap()
            {
                crate::optimizeopt::info::OpInfo::IntBound(b) => {
                    let b = b.borrow();
                    Some((b.lower, b.upper))
                }
                _ => None,
            },
            Some((10, 20))
        );

        let mut ctx2 = crate::optimizeopt::OptContext::with_inputarg_types(4, &[Type::Int]);
        let label_args = import_state(
            &[OpRef::input_arg_int(0)],
            &exported,
            &mut optimizer,
            &mut ctx2,
        );
        assert_eq!(label_args, vec![OpRef::int_op(21)]);
        // unroll.py:93-96: IntBound IS imported with widen() and stored
        // directly on the box's _forwarded slot via setintbound.
        // widen() relaxes bounds: lower < MININT/2 → MININT, upper > MAXINT/2 → MAXINT.
        // For [10, 20], both are within MININT/2..MAXINT/2 so widen() preserves them.
        let imported_bound = {
            let __mb = ctx2.materialize_operand_at(OpRef::int_op(21));
            ctx2.getintbound_handle(&__mb).borrow().clone()
        };
        assert_eq!((imported_bound.lower, imported_bound.upper), (10, 20));
    }

    #[test]
    fn test_exported_state_reads_live_bound_for_non_jumparg() {
        // unroll.py:443-454 `_expand_info` calls `self.optimizer.getinfo(arg)`
        // for EVERY exported value with no jump-arg restriction. A masked pure
        // result (`IntAnd(x, mask)`) that is a short-preamble / loop-invariant
        // value is not a jump arg, so its `[0, mask]` IntBound never enters the
        // `exported_int_bounds` side table (populated only from
        // `jump.getarglist()`). The live bound must still be exported by reading
        // it directly from the Phase-1 optimizer context.
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        use crate::optimizeopt::intutils::IntBound;
        const MASK: i64 = 0xFFFF_FFFF;

        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(4, 0);
        // Bind the export-input position at its source (a forced end-arg is a
        // bound box in production).
        ctx.materialize_operand_at(OpRef::int_op(21));
        let box21 = ctx
            .get_box_replacement_operand_opt(OpRef::int_op(21))
            .expect("int_op(21) bound to a box");
        // The bound lives on the box's `_forwarded` slot (as it does after
        // `IntAnd` narrows it), but it is NOT registered in the side table —
        // exactly the state of a non-jump-arg short-box result.
        ctx.setintbound(&box21, &IntBound::bounded(0, MASK));

        // `None` side table → current code has no way to recover the bound and
        // drops it; the live-bound read must surface it.
        let exported = export_state(&[OpRef::int_op(21)], &[], &mut optimizer, &mut ctx, None);

        assert_eq!(
            match exported
                .exported_infos
                .iter()
                .find(|(k, _)| k.to_opref() == OpRef::int_op(21))
                .map(|(_, v)| v)
            {
                Some(crate::optimizeopt::info::OpInfo::IntBound(b)) => {
                    let b = b.borrow();
                    Some((b.lower, b.upper))
                }
                _ => None,
            },
            Some((0, MASK)),
            "live [0, MASK] bound on a non-jump-arg short-box result must be exported"
        );
    }

    #[test]
    fn test_short_box_dependency_preserves_mask_bound_for_loop_close() {
        // RPython keeps the replay op's `_forwarded` info separate from the
        // body Box's optimizer info. A preamble-computed masked pure value can
        // therefore be used by another short-box replay op without losing the
        // live `[0, MASK]` IntBound needed by virtualstate.py:491-492 to avoid
        // loop-close re-establishment guards.
        use crate::optimizeopt::intutils::IntBound;
        use crate::optimizeopt::shortpreamble::{
            PreambleOpKind, ProducedShortOp, ShortPreambleBuilder,
        };
        use crate::optimizeopt::virtualstate::{VirtualState, VirtualStateInfo};
        const MASK: i64 = 0xFFFF_FFFF;

        let masked = OpRef::int_op(58);
        let dependent = OpRef::int_op(59);
        let mut ctx =
            crate::optimizeopt::OptContext::with_inputarg_types(128, &vec![Type::Int; 128]);

        let mask_op = {
            let mut op = Op::new(
                OpCode::IntAnd,
                &[
                    rooted_inputarg_operand(Type::Int, 8),
                    Operand::from_opref(OpRef::const_int(MASK)),
                ],
            );
            op.pos.set(masked);
            std::rc::Rc::new(op)
        };
        let dep_op = {
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    Operand::from_bound_op(&mask_op),
                    Operand::from_opref(OpRef::const_int(2160)),
                ],
            );
            op.pos.set(dependent);
            std::rc::Rc::new(op)
        };
        let mask_box = ctx.materialize_operand_at(masked);
        let dep_box = ctx.materialize_operand_at(dependent);
        ctx.setintbound(&mask_box, &IntBound::bounded(0, MASK));
        ctx.imported_short_preamble_builder = Some(ShortPreambleBuilder::new(
            &[masked, dependent],
            &[
                (
                    mask_box.clone(),
                    ProducedShortOp {
                        kind: PreambleOpKind::Pure,
                        res: mask_box.clone(),
                        preamble_op: mask_op.clone(),
                        invented_name: false,
                        same_as_source: None,
                        label_arg_idx: Some(0),
                    },
                ),
                (
                    dep_box.clone(),
                    ProducedShortOp {
                        kind: PreambleOpKind::Pure,
                        res: dep_box.clone(),
                        preamble_op: dep_op.clone(),
                        invented_name: false,
                        same_as_source: None,
                        label_arg_idx: Some(1),
                    },
                ),
            ],
            &[OpRef::input_arg_int(8)],
        ));

        let mask_pop = crate::optimizeopt::info::PreambleOp {
            op: mask_box.clone(),
            invented_name: false,
            preamble_op: mask_op.clone(),
            same_as_source: None,
        };
        let dep_pop = crate::optimizeopt::info::PreambleOp {
            op: dep_box,
            invented_name: false,
            preamble_op: dep_op,
            same_as_source: None,
        };
        assert_eq!(ctx.force_op_from_preamble_op(&mask_pop), masked);
        assert_eq!(ctx.force_op_from_preamble_op(&dep_pop), dependent);

        let target_vs = VirtualState::new(vec![VirtualStateInfo::IntBounded(IntBound::bounded(
            0, MASK,
        ))]);
        let incoming_vs = crate::optimizeopt::virtualstate::export_state(&[masked], &ctx);
        let runtime_box = ctx.make_constant_int(0);
        let guard_reqs = target_vs
            .generate_guards(&incoming_vs, &[masked], &[runtime_box], &mut ctx, false)
            .expect("masked short-box state should match target loop state");
        let mut emitted = Vec::new();
        for req in &guard_reqs {
            emitted.extend(req.to_ops(&[masked], &mut ctx));
        }

        assert!(
            emitted
                .iter()
                .all(|op| !matches!(op.opcode, OpCode::IntGe | OpCode::IntLe | OpCode::GuardTrue)),
            "loop close must not emit IntBound re-establishment guards for masked short-box arg: {:?}",
            emitted.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_export_state_uses_forced_end_args_snapshot() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(4, &[Type::Ref]);
        // Forced end-of-preamble args are bound boxes in production
        // (force_box_for_end_of_preamble); bind the fixture's export-input
        // position at its source so every value reaching export_single_value
        // has a canonical box (virtualstate.py:711-720 create_state receives
        // real AbstractValues, never bare positions).
        ctx.materialize_operand_at(OpRef::int_op(21));
        ctx.preamble_end_args = Some(vec![OpRef::int_op(21)]);

        let exported = export_state(&[OpRef::int_op(0)], &[], &mut optimizer, &mut ctx, None);

        assert_eq!(exported.end_args.clone(), vec![OpRef::int_op(21)]);
    }

    #[test]
    fn test_exported_state_reimports_short_heap_field_facts() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(4, 0);
        // optimizer.py:478 ensure_ptr_info_arg0 parity: every FieldDescr
        // must resolve its parent SizeDescr so that the routing between
        // StructPtrInfo and InstancePtrInfo can ask `is_object()`.
        // SimpleFieldDescr stores the parent as Weak<DescrRef>, so the
        // test must keep the parent Arc alive until import_state has run.
        let parent = majit_ir::descr::make_size_descr(16);
        let field_descr = std::sync::Arc::new(
            majit_ir::descr::SimpleFieldDescr::new(0, 0, 8, majit_ir::Type::Int, false)
                .with_signed(true)
                .with_parent_descr(parent.clone(), 0),
        ) as majit_ir::DescrRef;
        // shortpreamble.py:257/285: the export-time rename mints a fresh
        // renamed short_inputarg per label/virtual slot, and exported short
        // ops carry the renamed box in their args (not the original label
        // arg). Seed the two slot boxes and use slot 0 — the GETFIELD
        // receiver, whose original is int_op(10).
        // Bound short_inputarg boxes rooted by an index-aligned `InputArgRc`
        // pool, mirroring production
        // (optimizer.rs:2937/2942 set `exported_short_inputargs` and
        // `exported_short_inputarg_refs` in lockstep); the context channel
        // stores their positions.
        let (si0, ia0) = crate::history::test_support::bound_inputarg_operand(
            Type::Int,
            ctx.alloc_op_position_typed(Type::Int).raw(),
        );
        let (si1, ia1) = crate::history::test_support::bound_inputarg_operand(
            Type::Int,
            ctx.alloc_op_position_typed(Type::Int).raw(),
        );
        ctx.exported_short_inputargs = vec![si0.to_opref(), si1.to_opref()];
        ctx.exported_short_inputarg_refs = vec![ia0, ia1];
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op =
                        Op::with_descr(OpCode::GetfieldGcI, &[si0.clone()], field_descr.clone());
                    op.pos.set(OpRef::int_op(11));
                    std::rc::Rc::new(op)
                },
                res: rooted_resop_operand(Type::Int, 11),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Heap,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            });
        // Bind export-input positions at their source: the GETFIELD receiver
        // and result are bound boxes in production (label arg / ProducedShortOp.res
        // = materialize_operand_at, shortpreamble.rs:436). virtualstate.py:711-720
        // create_state receives real AbstractValues, never bare positions.
        ctx.materialize_operand_at(OpRef::int_op(10));
        ctx.materialize_operand_at(OpRef::int_op(11));

        let exported = export_state(
            &[OpRef::int_op(10), OpRef::int_op(11)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        // The fixture references both slots via `int_op(N)` (Int variant
        // tag), so targetargs must agree under variant-aware OpRef Eq —
        // a Ref-typed targetarg would trip the Box.type cross-type
        // forward check in `make_equal_to`. Heap-field-cache mechanics are
        // exercised independent of base type.
        let mut ctx2 =
            crate::optimizeopt::OptContext::with_inputarg_types(4, &[Type::Int, Type::Int]);
        let targetargs = [OpRef::input_arg_int(0), OpRef::input_arg_int(1)];
        let label_args = import_state(&targetargs, &exported, &mut optimizer, &mut ctx2);
        import_short_preamble_state(&targetargs, &label_args, &exported, &mut ctx2);
        assert_eq!(label_args, vec![OpRef::int_op(10), OpRef::int_op(11)]);
        // RPython PreambleOp parity: PreambleOp stored in PtrInfo._fields.
        // No imported_short_fields for heap fields — PtrInfo is the single
        // source of truth, matching RPython's HeapOp.produce_op → opinfo.setfield.
        let obj_box = ctx2
            .get_box_replacement_operand_opt(OpRef::int_op(10))
            .unwrap();
        let pop = ctx2
            .with_ptr_info_mut(&obj_box, |info| info.take_preamble_field(0))
            .flatten();
        assert!(pop.is_some(), "PreambleOp must be in PtrInfo._fields");
        let pop = pop.unwrap();
        assert_eq!(pop.op.to_opref(), OpRef::int_op(11)); // Phase 1 source — pop.op
        // forwards via make_equal_to to the body-visible OpRef.
        drop(parent);
    }

    #[test]
    fn test_import_state_reimports_short_ref_constant_identity() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(8, 0);
        let ptr = GcRef(0x1234_5678);
        let field_descr = majit_ir::descr::make_field_descr_full(88, 0, 8, Type::Int, false);
        // ConstPtr.value inline (history.py:314): the producer seeds the
        // inline variant that carries the pointer directly; the consumer
        // reads it back without any pool lookup.
        let ptr_box = ctx.materialize_operand_at(OpRef::const_ptr(ptr));
        ctx.seed_constant(&ptr_box, Value::Ref(ptr));
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::with_descr(
                        OpCode::GetfieldGcPureI,
                        &[Operand::from_opref(OpRef::const_ptr(ptr))],
                        field_descr.clone(),
                    );
                    op.pos.set(OpRef::int_op(11));
                    std::rc::Rc::new(op)
                },
                res: rooted_resop_operand(Type::Int, 11),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            });
        // Bind export-input positions at their source (label arg /
        // ProducedShortOp.res = materialize_operand_at, shortpreamble.rs:436);
        // virtualstate.py:711-720 create_state receives real AbstractValues.
        ctx.materialize_operand_at(OpRef::int_op(12));
        ctx.materialize_operand_at(OpRef::int_op(11));

        let exported = export_state(
            &[OpRef::int_op(12), OpRef::int_op(11)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        // Both label args resolve to Int op slots (12 / 11 = GetfieldGcPureI base / result).
        let mut ctx2 =
            crate::optimizeopt::OptContext::with_inputarg_types(8, &[Type::Int, Type::Int]);
        let targetargs = [OpRef::input_arg_int(0), OpRef::input_arg_int(1)];
        let label_args = import_state(&targetargs, &exported, &mut optimizer, &mut ctx2);
        import_short_preamble_state(&targetargs, &label_args, &exported, &mut ctx2);
        assert_eq!(label_args, vec![OpRef::int_op(12), OpRef::int_op(11)]);
        // history.py:314 ConstPtr.value inline: the imported constant
        // lands at `OpRef::ConstPtr(ptr)`, carrying the pointer
        // directly.
        let fresh_const = OpRef::const_ptr(ptr);
        assert_eq!(ctx2.get_constant(fresh_const), Some(Value::Ref(ptr)));
        let expected = crate::optimizeopt::ImportedShortPureOp::new(
            &mut ctx2,
            OpCode::GetfieldGcPureI,
            Some(field_descr.clone()),
            vec![crate::optimizeopt::ImportedShortPureArg::Const(
                Value::Ref(ptr),
                fresh_const,
            )],
            OpRef::int_op(11),
            OpRef::int_op(11),
            false,
            None,
        );
        assert_eq!(ctx2.imported_short_pure_ops, vec![expected]);
    }

    #[test]
    fn test_import_short_loopinvariant_uses_producer_const_snapshot() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(8, 0);
        // ConstInt.value inline (history.py:227): the producer seeds the
        // inline variant carrying the func address directly; the consumer
        // reads it back without any pool lookup.
        let func_ptr = 0xCAFE;
        let func = OpRef::const_int(func_ptr);
        let func_box = ctx.materialize_operand_at(func);
        ctx.seed_constant(&func_box, Value::Int(func_ptr));
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::CallLoopinvariantI, &[Operand::from_opref(func)]);
                    op.pos.set(OpRef::int_op(11));
                    std::rc::Rc::new(op)
                },
                res: rooted_resop_operand(Type::Int, 11),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::LoopInvariant,
                label_arg_idx: Some(1),
                invented_name: false,
                same_as_source: None,
            });

        // Bind export-input positions at their source (label arg /
        // ProducedShortOp.res = materialize_operand_at, shortpreamble.rs:436);
        // virtualstate.py:711-720 create_state receives real AbstractValues.
        ctx.materialize_operand_at(OpRef::int_op(10));
        ctx.materialize_operand_at(OpRef::int_op(11));

        let exported = export_state(
            &[OpRef::int_op(10), OpRef::int_op(11)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        // CallLoopinvariantI produces an Int at int_op(11); slot 10 is the
        // Int func-address constant base, so both label args are Int.
        let mut ctx2 =
            crate::optimizeopt::OptContext::with_inputarg_types(8, &[Type::Int, Type::Int]);
        let targetargs = [OpRef::input_arg_int(0), OpRef::input_arg_int(1)];
        let label_args = import_state(&targetargs, &exported, &mut optimizer, &mut ctx2);
        import_short_preamble_state(&targetargs, &label_args, &exported, &mut ctx2);

        assert_eq!(
            ctx2.imported_loop_invariant_results
                .iter()
                .find(|(k, _)| *k == func_ptr)
                .map(|(_, v)| *v),
            Some(OpRef::int_op(11))
        );
        // history.py:227 ConstInt.value inline — the imported func
        // address lands at `OpRef::ConstInt(func_ptr)`, which
        // carries the value directly.
        assert_eq!(
            ctx2.get_constant(OpRef::const_int(func_ptr)),
            Some(Value::Int(func_ptr))
        );
    }

    #[test]
    fn test_import_short_loopinvariant_result_uses_short_inputarg_slot() {
        // history.py:227 — the CallLoopinvariant func address is an inline
        // `ConstInt` arg; the value rides on the OpRef and the import path
        // keys `imported_loop_invariant_results` by that value.
        let func_ptr = 0xBEEF;
        let func = OpRef::const_int(func_ptr);
        let source = OpRef::int_op(11);
        let source_box = rooted_resop_operand(Type::Int, 11);
        let phase2_result = OpRef::int_op(3);
        let exported = ExportedState::new(
            vec![source],
            vec![0],
            vec![source_box.clone()],
            crate::optimizeopt::virtualstate::VirtualState::new(Vec::new()),
            indexmap::IndexMap::new(),
            vec![crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::CallLoopinvariantI, &[Operand::from_opref(func)]);
                    op.pos.set(source);
                    std::rc::Rc::new(op)
                },
                res: source_box.clone(),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::LoopInvariant,
                label_arg_idx: Some(0),
                invented_name: false,
                same_as_source: None,
            }],
            Vec::new(),
            vec![source_box.to_opref()],
            // short_inputarg_refs: this fixture stores plain OpRef positions.
            Vec::new(),
        );
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(
            8,
            &[Type::Int, Type::Int, Type::Int, Type::Int],
        );
        let func_box = ctx.materialize_operand_at(func);
        ctx.seed_constant(&func_box, Value::Int(func_ptr));

        import_short_preamble_state(&[OpRef::int_op(0)], &[phase2_result], &exported, &mut ctx);

        // Path B (B.6.7-loopinv): imported_loop_invariant_results stores the
        // Phase 1 source directly (RPython `shortpreamble.py:120 op = self.res`).
        let _ = phase2_result;
        assert_eq!(
            ctx.imported_loop_invariant_results
                .iter()
                .find(|(k, _)| *k == func_ptr)
                .map(|(_, v)| *v),
            Some(source)
        );
    }

    /// B.6.3 invariant: `force_op_from_preamble_op` follows RPython
    /// `unroll.py:26-39` line-by-line. It calls `use_box` (updates
    /// `self.short`) and seeds `potential_extra_ops`. It does NOT touch
    /// `used_boxes` / `short_preamble_jump`. Those grow only via the
    /// orthodox `force_box` → `potential_extra_ops.pop` →
    /// `add_preamble_op` (`shortpreamble.py:432-440`) path.
    #[test]
    fn test_force_op_from_preamble_orthodox_does_not_record_used_boxes() {
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(
            10,
            &[Type::Ref, Type::Ref, Type::Ref],
        );
        ctx.initialize_imported_short_preamble_builder(
            &[OpRef::int_op(0), OpRef::int_op(1), OpRef::int_op(2)],
            &[OpRef::int_op(10), OpRef::int_op(11), OpRef::int_op(12)],
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(
                        OpCode::IntAdd,
                        &[
                            rooted_resop_operand(Type::Int, 0),
                            rooted_resop_operand(Type::Int, 1),
                        ],
                    );
                    op.pos.set(OpRef::int_op(20));
                    std::rc::Rc::new(op)
                },
                res: rooted_resop_operand(Type::Int, 20),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            }],
        );

        let src20 = ctx.materialize_operand_at(OpRef::int_op(20));
        let produced = ctx
            .imported_short_preamble_builder
            .as_ref()
            .unwrap()
            .produced_short_op(&src20)
            .unwrap();
        let pop = crate::optimizeopt::info::PreambleOp {
            op: rooted_resop_operand(Type::Int, 20),
            invented_name: produced.invented_name,
            same_as_source: produced.same_as_source.clone(),
            preamble_op: produced.preamble_op,
        };
        let forced = ctx.force_op_from_preamble_op(&pop);
        assert_eq!(forced, OpRef::int_op(20));

        // RPython `unroll.py:32` `use_box` populates `self.short`.
        let sp = ctx.build_imported_short_preamble().unwrap();
        assert_eq!(sp.ops.len(), 1);
        // RPython `unroll.py:34-37` seeds `potential_extra_ops` so a later
        // `force_box` will run `add_preamble_op` (shortpreamble.py:432-440).
        assert!(
            ctx.has_potential_extra_op(OpRef::int_op(20)),
            "force_op_from_preamble_op must seed potential_extra_ops"
        );
        // RPython parity: `force_op_from_preamble` does NOT call
        // `add_preamble_op`. `used_boxes` / `short_preamble_jump` /
        // `extra_same_as` stay empty until `force_box` pops the entry.
        assert!(sp.used_boxes.is_empty());
        assert!(sp.jump_args.is_empty());

        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let _ = optimizer.force_box(OpRef::int_op(20), &mut ctx);

        let sp = ctx.build_imported_short_preamble().unwrap();
        // After force_box: orthodox `add_preamble_op` (shortpreamble.py:432-440)
        // populated all three lists in lock-step.
        assert_eq!(sp.used_boxes.clone(), vec![OpRef::int_op(20)]);
        assert_eq!(sp.jump_args.clone(), vec![OpRef::int_op(20)]);
        assert!(
            !ctx.has_potential_extra_op(OpRef::int_op(20)),
            "force_box must consume the potential_extra_ops entry"
        );
    }

    /// B.6.3 invariant (Heap variant): same orthodox boundary as the Pure
    /// case, with the producer's `make_equal_to(source, value)` forwarding
    /// installed (`shortpreamble.rs::produce_heap_field`).
    /// `force_op_from_preamble_op` returns `preamble_source` (RPython
    /// `unroll.py:38 return preamble_op.op` ≡ `self.res`); the producer's
    /// `get_box_replacement(source)` lookup is consumed inside `force_box`
    /// (`shortpreamble.py:436 op = preamble_op.op.get_box_replacement()`)
    /// when it runs `add_preamble_op`.
    #[test]
    fn test_force_op_from_preamble_returns_source_for_heap_variant() {
        let mut ctx = crate::optimizeopt::OptContext::with_inputarg_types(
            32,
            &[Type::Ref, Type::Ref, Type::Ref, Type::Ref],
        );
        ctx.initialize_imported_short_preamble_builder(
            &[
                OpRef::ref_op(0),
                OpRef::ref_op(1),
                OpRef::ref_op(2),
                OpRef::ref_op(3),
            ],
            &[
                OpRef::ref_op(10),
                OpRef::ref_op(11),
                OpRef::ref_op(12),
                OpRef::ref_op(13),
            ],
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::with_descr(
                        OpCode::GetfieldGcR,
                        &[rooted_resop_operand(Type::Ref, 3)],
                        majit_ir::descr::make_field_descr_full(56, 0, 8, Type::Ref, false),
                    );
                    op.pos.set(OpRef::ref_op(19));
                    std::rc::Rc::new(op)
                },
                res: rooted_resop_operand(Type::Ref, 19),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Heap,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            }],
        );
        let src19 = ctx.materialize_operand_at(OpRef::ref_op(19));
        let produced = ctx
            .imported_short_preamble_builder
            .as_ref()
            .unwrap()
            .produced_short_op(&src19)
            .unwrap();
        // Path B (B.6.7-heap-field): produce_heap_field no longer installs
        // make_equal_to, but the test still walks the get_box_replacement
        // chain inside force_box's add_preamble_op, so install a manual
        // forwarding to the body-visible OpRef to exercise that path.
        let b_src = ctx.materialize_operand_at(OpRef::ref_op(19));
        let b_tgt = ctx
            .get_box_replacement_operand_opt(OpRef::ref_op(14))
            .unwrap_or_else(|| ctx.materialize_operand_at(OpRef::ref_op(14)));
        ctx.make_equal_to(&b_src, &b_tgt);
        let pop = crate::optimizeopt::info::PreambleOp {
            op: b_src.clone(),
            invented_name: produced.invented_name,
            same_as_source: produced.same_as_source.clone(),
            preamble_op: produced.preamble_op,
        };
        let forced = ctx.force_op_from_preamble_op(&pop);
        // RPython `unroll.py:38 return preamble_op.op` ≡ self.res.
        // pyre's Phase 1 source IS self.res for the imported short box.
        assert_eq!(forced, OpRef::ref_op(19));

        let sp = ctx.build_imported_short_preamble().unwrap();
        assert_eq!(sp.ops.len(), 1);
        // RPython parity: orthodox boundary — used_boxes / jump_args stay
        // empty until force_box runs add_preamble_op.
        assert!(sp.used_boxes.is_empty());
        assert!(sp.jump_args.is_empty());
        // `force_op_from_preamble_op` keys potential_extra_ops by the
        // body-visible box `get_box_replacement(preamble_source)`
        // (unroll.py:35-37 `op = get_box_replacement(op)`).  This heap
        // variant forwarded source 19 -> body-visible 14 (produce_heap_field
        // installs the `make_equal_to` upstream heap.py omits), so the entry
        // lands on 14, not on the source 19.
        assert!(
            ctx.has_potential_extra_op(OpRef::ref_op(14)),
            "force_op_from_preamble_op must seed potential_extra_ops by the body-visible box"
        );
        assert!(
            !ctx.has_potential_extra_op(OpRef::ref_op(19)),
            "the forwarded source box must not carry the potential_extra_ops entry"
        );

        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let _ = optimizer.force_box(forced, &mut ctx);

        let sp = ctx.build_imported_short_preamble().unwrap();
        // shortpreamble.py:436 `op = preamble_op.op.get_box_replacement()`.
        // pop.op=19 forwards to body-visible 14 via the producer's make_equal_to,
        // so used_boxes carries the resolved body-visible OpRef while
        // jump_args carries the unresolved Phase 1 source.
        assert_eq!(sp.used_boxes.clone(), vec![OpRef::ref_op(14)]);
        assert_eq!(sp.jump_args.clone(), vec![OpRef::ref_op(19)]);
        // force_box resolves the body arg forward (19 -> 14) and pops the
        // entry at the body-visible key 14.
        assert!(
            !ctx.has_potential_extra_op(OpRef::ref_op(14)),
            "force_box must consume the potential_extra_ops entry"
        );
    }

    #[test]
    fn test_exported_state_reimports_invented_short_alias_metadata() {
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let mut ctx = crate::optimizeopt::OptContext::with_num_inputs(6, 0);
        // shortpreamble.py:257/285: exported short ops carry the renamed
        // short_inputargs in their args. Seed three slot boxes (label args
        // 12/13/14) and rename the IntAdd operands to slots 0/1. The
        // `same_as_source` alias is a ProducedShortOp field, not an op arg,
        // so it keeps its original (int_op(14)).
        // Bound short_inputarg boxes rooted by an index-aligned `InputArgRc`
        // pool, matching production (optimizer.rs:2937/2942 set
        // `exported_short_inputargs` / `exported_short_inputarg_refs` in
        // lockstep); the context channel stores their positions.
        let (si0, ia0) = crate::history::test_support::bound_inputarg_operand(
            Type::Int,
            ctx.alloc_op_position_typed(Type::Int).raw(),
        );
        let (si1, ia1) = crate::history::test_support::bound_inputarg_operand(
            Type::Int,
            ctx.alloc_op_position_typed(Type::Int).raw(),
        );
        let (si2, ia2) = crate::history::test_support::bound_inputarg_operand(
            Type::Int,
            ctx.alloc_op_position_typed(Type::Int).raw(),
        );
        ctx.exported_short_inputargs = vec![si0.to_opref(), si1.to_opref(), si2.to_opref()];
        ctx.exported_short_inputarg_refs = vec![ia0, ia1, ia2];
        ctx.exported_short_boxes
            .push(crate::optimizeopt::shortpreamble::PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntAdd, &[si0.clone(), si1.clone()]);
                    op.pos.set(OpRef::int_op(30));
                    std::rc::Rc::new(op)
                },
                res: rooted_resop_operand(Type::Int, 30),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: true,
                same_as_source: Some(rooted_resop_operand(Type::Int, 14)),
            });
        // Bind export-input positions at their source (IntAdd operands /
        // same-as alias are bound boxes in production); virtualstate.py:711-720
        // create_state receives real AbstractValues, never bare positions.
        ctx.materialize_operand_at(OpRef::int_op(12));
        ctx.materialize_operand_at(OpRef::int_op(13));
        ctx.materialize_operand_at(OpRef::int_op(14));

        let exported = export_state(
            &[OpRef::int_op(12), OpRef::int_op(13), OpRef::int_op(14)],
            &[],
            &mut optimizer,
            &mut ctx,
            None,
        );
        // IntAdd producer at int_op(30) consumes Int op slots, so the
        // imported targetargs must be Int-typed too — generous pool keeps
        // typed `OpRef::input_arg_int(K)` handles for any K the test walks.
        let mut ctx2 =
            crate::optimizeopt::OptContext::with_inputarg_types(6, &vec![Type::Int; 1024]);
        let targetargs = [
            OpRef::input_arg_int(0),
            OpRef::input_arg_int(1),
            OpRef::input_arg_int(2),
        ];
        let label_args = import_state(&targetargs, &exported, &mut optimizer, &mut ctx2);
        import_short_preamble_state(&targetargs, &label_args, &exported, &mut ctx2);
        let imported_result = ctx2.imported_short_pure_ops[0].result;
        assert_ne!(imported_result, OpRef::int_op(30));
        let pop = ctx2.imported_short_pure_ops[0].pop.clone();
        let forced = ctx2.force_op_from_preamble_op(&pop);
        // force_op_from_preamble may return the imported position (not necessarily 30)
        let _ = forced;
        assert_eq!(ctx2.imported_short_pure_ops.len(), 1);
        // RPython parity: extra_same_as is populated lazily by add_preamble_op
        // (called from optimizer.force_box's potential_extra_ops.pop path).
        // shortpreamble.py:432-440 record the SameAs Box at use-box time, not
        // at produce_op time — verify the lazy population fires.
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let _ = optimizer.force_box(forced, &mut ctx2);
        let aliases = ctx2.used_imported_short_aliases();
        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0].same_as_source.to_opref(), OpRef::int_op(14));
        assert_eq!(aliases[0].same_as_opcode, OpCode::SameAsI);
    }

    #[test]
    fn test_assemble_peeled_trace_emits_extra_same_as_before_label() {
        let p1_ops = vec![{
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            );
            op.pos.set(OpRef::int_op(3));
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntMul,
                    &[
                        rooted_resop_operand(Type::Int, 50),
                        rooted_resop_operand(Type::Int, 0),
                    ],
                );
                op.pos.set(OpRef::int_op(1));
                op
            },
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 50)]),
        ];

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[],
            1,
            true,
            &[crate::optimizeopt::ImportedShortAlias {
                result: OpRef::int_op(50),
                same_as_source: rooted_resop_operand(Type::Int, 10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &majit_ir::ConstMap::new(),
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::IntAdd);
        assert_eq!(combined[1].opcode, OpCode::SameAsI);
        assert_eq!(
            combined[1]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10)]
        );
        assert_eq!(combined[2].opcode, OpCode::Label);
        assert_eq!(combined[3].opcode, OpCode::IntMul);
        assert_eq!(combined[3].arg(0).to_opref(), combined[1].pos.get());
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].arg(0).to_opref(), combined[1].pos.get());
    }

    #[test]
    fn test_assemble_peeled_trace_preserves_visible_label_arg_until_body_redef() {
        let p1_ops = vec![{
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            );
            op.pos.set(OpRef::int_op(11));
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntGe,
                    &[
                        rooted_resop_operand(Type::Int, 11),
                        Operand::from_opref(OpRef::const_int(2)),
                    ],
                );
                op.pos.set(OpRef::int_op(4));
                op
            },
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 11),
                        Operand::from_opref(OpRef::const_int(1)),
                    ],
                );
                op.pos.set(OpRef::int_op(11));
                op
            },
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 11)]),
        ];

        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::new();

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef::int_op(11)],
            &[OpRef::int_op(0)],
            &[],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(
            combined[1]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(11)]
        );
        assert_eq!(combined[2].opcode, OpCode::IntGe);
        assert_eq!(combined[2].arg(0).to_opref(), OpRef::int_op(11));
        assert_eq!(combined[3].opcode, OpCode::IntAdd);
        assert_eq!(combined[3].arg(0).to_opref(), OpRef::int_op(11));
        assert_ne!(combined[3].pos.get(), OpRef::int_op(11));
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].arg(0).to_opref(), combined[3].pos.get());
    }

    #[test]
    fn test_assemble_peeled_trace_preserves_visible_preamble_box_over_body_collision() {
        let p1_ops = vec![{
            let mut op = Op::new(OpCode::GetfieldGcR, &[rooted_resop_operand(Type::Int, 3)]);
            op.pos.set(OpRef::int_op(19));
            op.setdescr(majit_ir::descr::make_field_descr_full(
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
                let mut op = Op::new(
                    OpCode::SetfieldGc,
                    &[
                        rooted_resop_operand(Type::Int, 25),
                        rooted_resop_operand(Type::Int, 19),
                    ],
                );
                op.setdescr(majit_ir::descr::make_field_descr_full(
                    57,
                    8,
                    8,
                    Type::Ref,
                    false,
                ));
                op
            },
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 0),
                        Operand::from_opref(OpRef::const_int(1)),
                    ],
                );
                op.pos.set(OpRef::int_op(19));
                op
            },
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 19)]),
        ];

        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::new();

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef::int_op(0)],
            &[OpRef::int_op(0)],
            &[],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(combined[2].opcode, OpCode::SetfieldGc);
        assert_eq!(combined[2].arg(1).to_opref(), OpRef::int_op(19));
        assert_eq!(combined[3].opcode, OpCode::IntAdd);
        assert_ne!(combined[3].pos.get(), OpRef::int_op(19));
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(combined[4].arg(0).to_opref(), combined[3].pos.get());
    }

    #[test]
    fn test_assemble_peeled_trace_extends_label_with_used_boxes() {
        // Production caller passes body ops whose args have already been
        // resolved through ctx.get_box_replacement (optimizer.rs:2236-2262
        // forwarding-resolve pass). We mirror that here by writing the
        // resolved label_arg (OpRef::int_op(10)) directly into the body op rather
        // than the raw inputarg position OpRef::int_op(0).
        let p1_ops = vec![{
            let mut op = Op::new(
                OpCode::IntAdd,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                ],
            );
            op.pos.set(OpRef::int_op(3));
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntMul,
                    &[
                        rooted_resop_operand(Type::Int, 50),
                        rooted_resop_operand(Type::Int, 10),
                    ],
                );
                op.pos.set(OpRef::int_op(1));
                op
            },
            Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Int, 10),
                    rooted_resop_operand(Type::Int, 50),
                ],
            ),
        ];

        let combined = assemble_peeled_trace(
            &p1_ops,
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[OpRef::int_op(50)],
            1,
            true,
            &[crate::optimizeopt::ImportedShortAlias {
                result: OpRef::int_op(50),
                same_as_source: rooted_resop_operand(Type::Int, 10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &majit_ir::ConstMap::new(),
            None,
            None,
        );

        assert_eq!(combined[2].opcode, OpCode::Label);
        assert_eq!(
            combined[2]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10), combined[1].pos.get()]
        );
        assert_eq!(combined[4].opcode, OpCode::Jump);
        assert_eq!(
            combined[4]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10), combined[1].pos.get()]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_keeps_used_box_with_inline_constant_operand() {
        // assemble_peeled_trace must preserve an inline-Const operand
        // (history.py:227 `ConstInt.value`) untouched as the GuardValue
        // immediate, and the box-namespace entries in the constants
        // snapshot must round-trip unchanged.
        let p1_ops = vec![{
            let mut op = Op::new(OpCode::SameAsI, &[rooted_resop_operand(Type::Int, 37)]);
            op.pos.set(OpRef::void_op(857));
            op
        }];
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::GuardValue,
                    &[
                        rooted_resop_operand(Type::Void, 857),
                        Operand::from_opref(OpRef::const_int(2)),
                    ],
                );
                op.setfailargs(vec![rooted_resop_operand(Type::Void, 857)].into());
                op
            },
            Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Int, 10),
                    rooted_resop_operand(Type::Int, 853),
                    rooted_resop_operand(Type::Void, 857),
                    rooted_resop_operand(Type::Int, 850),
                ],
            ),
        ];
        let constants =
            majit_ir::ConstMap::from([(OpRef::void_op(857).raw(), majit_ir::Value::Int(2))]);

        let mut ctx = assemble_test_context(&p1_ops, &p2_ops, 1);
        let p1_ops_rc: Vec<majit_ir::OpRc> = p1_ops
            .iter()
            .map(|op| std::rc::Rc::new(op.clone()))
            .collect();
        let p2_ops_rc: Vec<majit_ir::OpRc> = p2_ops
            .iter()
            .map(|op| std::rc::Rc::new(op.clone()))
            .collect();
        let combined = assemble_peeled_trace_with_jump_args(
            &p1_ops_rc,
            &p2_ops_rc,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[OpRef::int_op(22), OpRef::void_op(857), OpRef::int_op(150)],
            &[OpRef::int_op(853), OpRef::void_op(857), OpRef::int_op(850)],
            1,
            0,
            true,
            &[],
            &constants,
            None,
            None,
            &[],
            &mut ctx,
        );

        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(
            combined[1]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[
                OpRef::int_op(10),
                OpRef::int_op(22),
                OpRef::void_op(857),
                OpRef::int_op(150)
            ]
        );
        assert_eq!(combined[2].opcode, OpCode::GuardValue);
        assert_eq!(
            combined[2]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::void_op(857), OpRef::const_int(2)]
        );
        assert_eq!(combined[3].opcode, OpCode::Jump);
        assert_eq!(
            combined[3]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[
                OpRef::int_op(10),
                OpRef::int_op(22),
                OpRef::void_op(857),
                OpRef::int_op(150)
            ]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_remaps_extra_label_source_slots() {
        // Production caller pre-resolves body args through
        // ctx.get_box_replacement; OpRef::int_op(0) (Phase 2 inputarg slot) is
        // pre-replaced with OpRef::int_op(10) (the corresponding label_arg).
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::GetfieldGcPureI,
                    &[rooted_resop_operand(Type::Int, 50)],
                );
                op.pos.set(OpRef::int_op(1));
                op.setdescr(majit_ir::make_field_descr(
                    0,
                    8,
                    majit_ir::Type::Int,
                    majit_ir::ArrayFlag::Signed,
                ));
                op
            },
            Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Int, 10),
                    rooted_resop_operand(Type::Int, 50),
                ],
            ),
        ];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[OpRef::int_op(50)],
            1,
            true,
            &[crate::optimizeopt::ImportedShortAlias {
                result: OpRef::int_op(50),
                same_as_source: rooted_resop_operand(Type::Int, 10),
                same_as_opcode: OpCode::SameAsI,
            }],
            &majit_ir::ConstMap::new(),
            None,
            None,
        );

        let label_idx = combined
            .iter()
            .position(|op| op.opcode == OpCode::Label)
            .expect("label");
        let label = &combined[label_idx];
        let extra_label_arg = label.arg(1);
        assert_eq!(
            label
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10), extra_label_arg.to_opref()]
        );
        let body_getfield = &combined[label_idx + 1];
        assert_eq!(body_getfield.opcode, OpCode::GetfieldGcPureI);
        assert_eq!(
            body_getfield
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![extra_label_arg.to_opref()]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_carries_body_value_used_before_local_def() {
        // Production caller pre-resolves body args through
        // ctx.get_box_replacement (optimizer.rs:2236-2262), so
        // OpRef::int_op(0) (Phase 2 inputarg slot) would already be replaced
        // with OpRef::int_op(10) (the label_arg for that slot) by the time
        // it reaches the assembler. We mirror that here.
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Int, 64)]);
                op.setfailargs(vec![rooted_resop_operand(Type::Int, 64)].into());
                op
            },
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 10),
                        Operand::from_opref(OpRef::const_int(1)),
                    ],
                );
                op.pos.set(OpRef::int_op(64));
                op
            },
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 64)]),
        ];
        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::new();

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(
            combined[0]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10), OpRef::int_op(64)]
        );
        assert_eq!(combined[1].opcode, OpCode::GuardTrue);
        assert_eq!(
            combined[1]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(64)]
        );
        assert_eq!(
            combined[1]
                .getfailargs()
                .expect("guard fail args")
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(64)]
        );
        assert_eq!(combined[2].opcode, OpCode::IntAdd);
        assert_ne!(combined[2].pos.get(), OpRef::int_op(64));
    }

    #[test]
    fn test_assemble_peeled_trace_does_not_route_jump_by_preamble_descr() {
        let start_descr = TargetToken::new_preamble(0).as_jump_target_descr();
        let p2_ops = vec![{
            let mut jump = Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Int, 0),
                    rooted_resop_operand(Type::Int, 1),
                    rooted_resop_operand(Type::Int, 2),
                    rooted_resop_operand(Type::Int, 3),
                    rooted_resop_operand(Type::Int, 4),
                ],
            );
            jump.setdescr(start_descr.clone());
            jump
        }];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[
                OpRef::int_op(10),
                OpRef::int_op(11),
                OpRef::int_op(12),
                OpRef::int_op(13),
                OpRef::int_op(14),
            ],
            &[OpRef::int_op(100), OpRef::int_op(101), OpRef::int_op(102)],
            &[OpRef::int_op(13), OpRef::int_op(14)],
            5,
            false,
            &[],
            &majit_ir::ConstMap::new(),
            Some(start_descr),
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(combined[1].opcode, OpCode::Label);
        assert_eq!(combined[2].opcode, OpCode::Jump);
        // RPython parity: unroll.py:238-242 `jump_to_preamble` retargets the
        // Jump before compile.py assembles the trace, and this assembly helper
        // must not derive a different arg contract by inspecting that descr.
        // The caller is responsible for constructing the preamble-shaped Jump.
        assert_eq!(
            combined[2]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[
                OpRef::int_op(100),
                OpRef::int_op(101),
                OpRef::int_op(102),
                OpRef::int_op(3),
                OpRef::int_op(4)
            ]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_preserves_preamble_jump_arity_when_body_label_is_shorter() {
        let start_descr = TargetToken::new_preamble(0).as_jump_target_descr();
        let loop_descr = TargetToken::new_loop(1).as_jump_target_descr();
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 0),
                        rooted_resop_operand(Type::Int, 1),
                    ],
                );
                op.pos.set(OpRef::int_op(2));
                op
            },
            {
                let mut jump = Op::new(
                    OpCode::Jump,
                    &[
                        rooted_resop_operand(Type::Int, 0),
                        rooted_resop_operand(Type::Int, 1),
                    ],
                );
                jump.setdescr(start_descr.clone());
                jump
            },
        ];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(0)],
            &[OpRef::int_op(100), OpRef::int_op(101)],
            &[],
            2,
            false,
            &[],
            &majit_ir::ConstMap::new(),
            Some(start_descr),
            Some(loop_descr),
        );

        let jump = combined.last().expect("assembled jump");
        assert_eq!(jump.opcode, OpCode::Jump);
        assert_eq!(
            jump.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(100), OpRef::int_op(101)]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_does_not_extend_body_label_with_preamble_jump_only_args() {
        let start_descr = TargetToken::new_preamble(0).as_jump_target_descr();
        let loop_descr = TargetToken::new_loop(1).as_jump_target_descr();
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 10),
                        Operand::from_opref(OpRef::const_int(1)),
                    ],
                );
                op.pos.set(OpRef::int_op(20));
                op
            },
            {
                let mut jump = Op::new(
                    OpCode::Jump,
                    &[
                        rooted_resop_operand(Type::Int, 10),
                        rooted_resop_operand(Type::Int, 50),
                        rooted_resop_operand(Type::Int, 60),
                    ],
                );
                jump.setdescr(start_descr.clone());
                jump
            },
        ];
        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::new();

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(100), OpRef::int_op(101), OpRef::int_op(102)],
            &[],
            1,
            false,
            &[],
            &constants,
            Some(start_descr),
            Some(loop_descr),
        );

        let body_label = combined
            .iter()
            .find(|op| {
                op.opcode == OpCode::Label
                    && op.getdescr().map(|descr| descr.repr())
                        == Some("LoopTargetDescr(1)".to_string())
            })
            .expect("body label");
        assert_eq!(
            body_label
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10)]
        );

        let jump = combined.last().expect("assembled jump");
        assert_eq!(jump.opcode, OpCode::Jump);
        assert_eq!(
            jump.getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10), OpRef::int_op(50), OpRef::int_op(60)]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_skips_constant_slots_for_new_body_positions() {
        let p2_ops = vec![
            {
                let mut op = Op::new(OpCode::New, &[]);
                op.pos.set(OpRef::ref_op(1));
                op
            },
            Op::new(
                OpCode::SetfieldGc,
                &[
                    rooted_resop_operand(Type::Ref, 1),
                    rooted_resop_operand(Type::Int, 0),
                ],
            ),
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 0)]),
        ];
        let constants = majit_ir::ConstMap::from([
            (2_u32, majit_ir::Value::Int(606)),
            (4_u32, majit_ir::Value::Int(611)),
        ]);

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(combined[1].opcode, OpCode::New);
        assert_ne!(combined[1].pos.get(), OpRef::int_op(2));
        assert_ne!(combined[1].pos.get(), OpRef::int_op(4));
        assert_eq!(combined[2].opcode, OpCode::SetfieldGc);
        assert_eq!(combined[2].arg(0).to_opref(), combined[1].pos.get());
    }

    #[test]
    fn test_assemble_peeled_trace_skips_constant_extra_label_args() {
        // Production caller pre-resolves the Jump's body inputarg ref
        // (OpRef::int_op(0)) to label_args[0] = OpRef::int_op(10) before it reaches the
        // assembler.
        //
        // `extra_label_args` holds one literal Const entry and one Box.
        // The filter keeps Boxes (they still back runtime slots) and drops
        // literal Const entries (their value is emitted inline at use
        // sites). Mirrors `assemble_peeled_trace_with_jump_args`'s
        // `label_arg.is_constant()` predicate.
        let const_extra = OpRef::const_int(606);
        let p2_ops = vec![Op::new(
            OpCode::Jump,
            &[rooted_resop_operand(Type::Int, 10)],
        )];
        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::new();

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(10)],
            &[OpRef::int_op(0)],
            &[const_extra, OpRef::int_op(8)],
            1,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(
            combined[0]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(10), OpRef::int_op(8)]
        );
    }

    #[test]
    fn test_assemble_peeled_trace_passes_through_resolved_body_inputs() {
        // Production caller pre-resolves Phase 2 body op args through
        // ctx.get_box_replacement (optimizer.rs:2236-2262), so by the time
        // they reach assemble_peeled_trace they already match label_args.
        // The assembler is therefore a passthrough for inputarg references
        // — no source_slot input_remap needed. This test verifies that
        // pre-resolved body args survive intact.
        let constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::new();
        let p2_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 200),
                        Operand::from_opref(OpRef::const_int(1)),
                    ],
                );
                op.pos.set(OpRef::int_op(20));
                op
            },
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 200)]),
        ];

        let combined = assemble_peeled_trace(
            &[],
            &p2_ops,
            &[OpRef::int_op(200), OpRef::int_op(300)],
            &[OpRef::int_op(0)],
            &[],
            6,
            true,
            &[],
            &constants,
            None,
            None,
        );

        assert_eq!(combined[0].opcode, OpCode::Label);
        assert_eq!(
            combined[0]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(200), OpRef::int_op(300)]
        );
        assert_eq!(combined[1].opcode, OpCode::IntAdd);
        assert_eq!(combined[1].arg(0).to_opref(), OpRef::int_op(200));
        assert_eq!(combined[2].opcode, OpCode::Jump);
        assert_eq!(
            combined[2]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(200), OpRef::int_op(300)]
        );
    }

    #[test]
    fn test_splice_redirected_tail_replaces_terminal_jump() {
        let body_ops = vec![
            {
                let mut op = Op::new(
                    OpCode::IntAdd,
                    &[
                        rooted_resop_operand(Type::Int, 0),
                        rooted_resop_operand(Type::Int, 1),
                    ],
                );
                op.pos.set(OpRef::void_op(3));
                op
            },
            Op::new(OpCode::Jump, &[rooted_resop_operand(Type::Int, 0)]),
        ];
        let redirected_tail = vec![
            {
                let mut op = Op::new(OpCode::GuardTrue, &[rooted_resop_operand(Type::Void, 3)]);
                op.setfailargs(vec![rooted_resop_operand(Type::Void, 3)].into());
                op
            },
            Op::new(
                OpCode::Jump,
                &[
                    rooted_resop_operand(Type::Void, 3),
                    rooted_resop_operand(Type::Int, 4),
                ],
            ),
        ];

        let body_ops: Vec<majit_ir::OpRc> = body_ops.into_iter().map(std::rc::Rc::new).collect();
        let redirected_tail: Vec<majit_ir::OpRc> =
            redirected_tail.into_iter().map(std::rc::Rc::new).collect();
        let spliced = splice_redirected_tail(&body_ops, &redirected_tail);
        assert_eq!(spliced.len(), 3);
        assert_eq!(spliced[0].opcode, OpCode::IntAdd);
        assert_eq!(spliced[1].opcode, OpCode::GuardTrue);
        assert_eq!(spliced[2].opcode, OpCode::Jump);
        assert_eq!(
            spliced[2]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::void_op(3), OpRef::int_op(4)]
        );
    }

    #[test]
    fn test_closing_loop_contract_arity_uses_actual_jump_contract() {
        let ops = vec![Op::new(
            OpCode::Jump,
            &[
                rooted_resop_operand(Type::Int, 0),
                rooted_resop_operand(Type::Int, 1),
                rooted_resop_operand(Type::Int, 2),
            ],
        )];

        assert_eq!(closing_loop_contract_arity(&ops, 5), 3);
    }
}
