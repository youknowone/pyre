/// Vectorization optimization pass.
///
/// Translated from rpython/jit/metainterp/optimizeopt/vector.py.
/// This is the core of the vec optimization — it combines dependency.py
/// and schedule.py to rewrite a loop in vectorized form.
///
/// # RPython parity notes
///
/// - In RPython, `VectorizingOptimizer` extends `Optimizer` (the main
///   optimization dispatcher). In Rust, `VectorizingOptimizer` implements
///   the `Optimization` trait (a sub-pass in the `Optimizer` pipeline).
///   This is a TODO: the Rust optimization pipeline
///   uses a trait-object chain rather than inheritance. The standalone
///   `optimize_vector()` function provides the RPython-shaped entry point.
///
/// - `CostModel`, `GenericCostModel`, `PackSet`, `isomorphic` live in
///   `schedule.rs` in Rust but in `vector.py` in RPython. This is a
///   TODO driven by the Rust module split done before
///   parity enforcement. They are re-exported here.
use majit_ir::vec_set::VecSet;

use majit_ir::operand::Operand;
use majit_ir::{Op, OpCode, OpRc, OpRef};

use crate::r#box::BoxRef;
use crate::optimizeopt::dependency::DependencyGraph;
use crate::optimizeopt::renamer::Renamer;
use crate::optimizeopt::vec_assoc::VecAssoc;
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

// Re-exports: these types live in schedule.rs but are defined in vector.py
// in RPython. Re-exporting preserves the public API surface.
pub use crate::optimizeopt::dependency::{Node, schedule_operations};
pub use crate::optimizeopt::schedule::{
    AccumEntry, AccumPack, CostModel, GenericCostModel, GuardAnalysis, NotAProfitableLoop,
    NotAVectorizeableLoop, Pack, PackSet, VecScheduleState, VectorizeError,
    are_adjacent_memory_refs, isomorphic, turn_into_vector, unpack_from_vector,
};

// ── vector.py:35-40: copy_resop ────────────────────────────────────────

/// vector.py:35-40: copy_resop — clone an op, preserving VectorizationInfo.
///
/// In RPython, `get_forwarded()` returns VectorizationInfo if set on the
/// op, and copy_resop re-attaches it to the clone. In Rust, `Op::clone()`
/// already preserves the `vecinfo` field, so this is a trivial clone.
///
/// Returns an owned `Op`: the caller renames/retargets it, then wraps it
/// into the canonical producer `OpRc` as it enters the buffer, so later
/// ops in the same buffer bind their args to that exact `Rc`
/// (`Operand::from_bound_op`) instead of a position-only `from_opref`.
fn copy_resop(op: &Op) -> Op {
    op.clone()
}

// ── vector.py:42-120: VectorLoop ───────────────────────────────────────

/// vector.py:42-120: VectorLoop — wraps a loop body (Label..operations..Jump)
/// for vectorization analysis and transformation.
#[derive(Clone, Debug)]
pub struct VectorLoop {
    /// vector.py:44: self.label = label
    pub label: Op,
    /// vector.py:45: self.inputargs = label.getarglist_copy()
    pub inputargs: Vec<OpRef>,
    /// vector.py:46: self.prefix = []
    pub prefix: Vec<OpRc>,
    /// vector.py:47: self.prefix_label = None
    pub prefix_label: Option<Op>,
    /// vector.py:49: self.operations = oplist
    ///
    /// `Vec<OpRc>` (not `Vec<Op>`): each element is the canonical producer
    /// box for its value, so a later op in the body binds its arg directly
    /// to the producer `Rc` (`Operand::from_bound_op`) instead of minting a
    /// position-only `Operand::Box` — parity with vector.py's box→box
    /// renamer which carries box objects, not integer positions.
    pub operations: Vec<OpRc>,
    /// vector.py:50: self.jump = jump
    pub jump: Op,
    /// vector.py:52: self.align_operations = []
    pub align_operations: Vec<OpRc>,
}

impl VectorLoop {
    /// vector.py:43-52: __init__(self, label, oplist, jump)
    ///
    /// Each body op is wrapped into an `OpRc` on entry so it becomes the
    /// canonical producer box for its value; the buffer then carries
    /// producer identity (see `operations` field doc).
    pub fn new(label: Op, operations: Vec<Op>, jump: Op) -> Self {
        Self::new_rc(
            label,
            operations.into_iter().map(std::rc::Rc::new).collect(),
            jump,
        )
    }

    /// `new` variant taking already-`OpRc`-wrapped operations, so the
    /// caller's producer boxes flow into the buffer without a re-clone.
    pub fn new_rc(label: Op, operations: Vec<OpRc>, jump: Op) -> Self {
        debug_assert_eq!(label.opcode, OpCode::Label);
        debug_assert!(
            jump.opcode == OpCode::Jump,
            "expected Jump, got {:?}",
            jump.opcode
        );
        let inputargs = label.getarglist().iter().map(|a| a.to_opref()).collect();
        VectorLoop {
            label,
            inputargs,
            prefix: Vec::new(),
            prefix_label: None,
            operations,
            jump,
            align_operations: Vec::new(),
        }
    }

    /// Convenience constructor: extract VectorLoop from a trace by finding
    /// Label..Jump. Not in RPython — the caller splits the trace before
    /// constructing VectorLoop. Kept for backward compatibility with tests.
    pub fn from_trace(ops: &[Op]) -> Option<Self> {
        let label_pos = ops.iter().position(|op| op.opcode == OpCode::Label)?;
        let jump_pos = ops.iter().rposition(|op| op.opcode == OpCode::Jump)?;
        if jump_pos <= label_pos {
            return None;
        }
        Some(VectorLoop::new(
            ops[label_pos].clone(),
            ops[label_pos + 1..jump_pos].to_vec(),
            ops[jump_pos].clone(),
        ))
    }

    /// vector.py:54-56: setup_vectorization — attach a `VectorizationInfo`
    /// to every loop operation (`for op in self.operations:
    /// op.set_forwarded(VectorizationInfo(op))`). PyPy stores it on
    /// `op._forwarded`; pyre's flat-OpRef operands keep the per-op vecinfo
    /// in the scheduler's pos-keyed store, so `state` carries it. INT_SIGNEXT
    /// reads arg1's constant through `constant_of` (resoperation.py:181).
    pub fn setup_vectorization(
        &self,
        state: &mut VecScheduleState,
        constant_of: &dyn Fn(OpRef) -> Option<i64>,
    ) {
        for op in &self.operations {
            state.set_op_forwarded_vecinfo(op, constant_of);
        }
    }

    /// vector.py:58-60: teardown_vectorization — drop every loop op's
    /// `VectorizationInfo` (`for op in self.operations:
    /// op.set_forwarded(None)`), clearing the scheduler's forwarded store.
    pub fn teardown_vectorization(&self, state: &mut VecScheduleState) {
        for op in &self.operations {
            state.clear_op_forwarded_vecinfo(op.pos.get());
        }
    }

    /// vector.py:62-92: finaloplist — assemble the complete operation list
    /// for compilation.
    ///
    /// `jitcell_token`: when supplied, allocate fresh `TargetToken`s for
    ///   `self.label` (`reset_label_token` true) and `self.prefix_label` /
    ///   `self.jump`, mirroring `vector.py:62-79`. When `None`, the descr
    ///   block is skipped — equivalent to RPython's `if jitcell_token:` guard.
    /// `reset_label_token`: vector.py:64 — choose between minting a new
    ///   `TargetToken` for the label (true) or pulling the existing jump
    ///   descr (false).
    /// `label`: include `self.label` at the front. When false, follow
    ///   `vector.py:87-90` and clear the vectorize-time scratch
    ///   (`set_forwarded(None)` upstream) from every emitted prefix op plus
    ///   the jump. That scratch lives in the scheduler's pos-keyed forwarded
    ///   store (`state`), not on the permanent `Op.vecinfo`, so the
    ///   equivalent is `state.clear_op_forwarded_vecinfo(pos)` — clearing
    ///   `Op.vecinfo` here would instead wipe VecOperationNew metadata.
    pub fn finaloplist(
        &self,
        jitcell_token: Option<&std::sync::Arc<majit_backend::JitCellToken>>,
        reset_label_token: bool,
        label: bool,
        state: &mut VecScheduleState,
    ) -> Vec<Op> {
        // vector.py:63-79: descr wiring against the owning JitCellToken.
        if let Some(jcell) = jitcell_token {
            // vector.py:64-72
            if reset_label_token {
                let token =
                    std::sync::Arc::new(crate::optimizeopt::unroll::TargetToken::new_loop(0));
                let descr = token.as_jump_target_descr();
                jcell.target_tokens.lock().push(descr.clone());
                self.label.setdescr(descr);
            }
            // else: vector.py:70-72 grabs token = self.jump.getdescr(); the
            // result is only used below to re-setdescr the jump when there's
            // no prefix_label, which is already what's there. Skip.

            // vector.py:73-77: prefix_label gets its own TargetToken, and
            // the jump is rebound to point at it.
            if let Some(ref prefix_label) = self.prefix_label {
                let pre_token =
                    std::sync::Arc::new(crate::optimizeopt::unroll::TargetToken::new_loop(0));
                let pre_descr = pre_token.as_jump_target_descr();
                jcell.target_tokens.lock().push(pre_descr.clone());
                prefix_label.setdescr(pre_descr.clone());
                self.jump.setdescr(pre_descr);
            } else if reset_label_token {
                // vector.py:78-79: with no prefix_label, re-bind jump to the
                // label's freshly-minted token.
                if let Some(label_descr) = self.label.getdescr() {
                    self.jump.setdescr(label_descr);
                }
            }
        }

        // vector.py:80-84
        // The producer buffers carry `OpRc`; the final emission boundary
        // hands back owned `Op` (the ops leave the vectorizer's producer
        // world here). Each op's args still bind to their producers — those
        // are `Operand::Op(Rc)` inside the cloned `args` (see `Op::clone`).
        let mut oplist: Vec<Op> = Vec::new();
        if let Some(ref prefix_label) = self.prefix_label {
            oplist.extend(self.prefix.iter().map(|op| (**op).clone()));
            oplist.push(prefix_label.clone());
        } else if !self.prefix.is_empty() {
            oplist.extend(self.prefix.iter().map(|op| (**op).clone()));
        }
        // vector.py:85-86
        if label {
            oplist.insert(0, self.label.clone());
        }
        // vector.py:87-90: when not emitting the label op (i.e. the prefix
        // is the *only* thing being compiled this round, e.g. a bridge),
        // strip vectorization scratch so nothing leaks into the next pass.
        // The scratch is the pos-keyed forwarded store, not the permanent
        // `Op.vecinfo`; mirror `teardown_vectorization`.
        if !label {
            for op in &oplist {
                state.clear_op_forwarded_vecinfo(op.pos.get());
            }
            state.clear_op_forwarded_vecinfo(self.jump.pos.get());
        }
        // vector.py:91
        oplist.extend(self.operations.iter().map(|op| (**op).clone()));
        oplist.push(self.jump.clone());
        oplist
    }

    /// vector.py:94-120: clone — deep-clone the loop with renaming.
    pub fn clone_loop(&self) -> Self {
        let mut renamer = Renamer::new();
        let mut prefix: Vec<OpRc> = Vec::new();
        for op in &self.prefix {
            let mut newop = (**op).clone();
            renamer.rename(&mut newop);
            if newop.opcode.result_type() != majit_ir::Type::Void {
                renamer.start_renaming(op.pos.get(), newop.pos.get());
            }
            prefix.push(std::rc::Rc::new(newop));
        }
        let prefix_label = self.prefix_label.as_ref().map(|pl| {
            let mut newpl = pl.clone();
            renamer.rename(&mut newpl);
            newpl
        });
        let mut operations: Vec<OpRc> = Vec::new();
        for op in &self.operations {
            let mut newop = (**op).clone();
            renamer.rename(&mut newop);
            if newop.opcode.result_type() != majit_ir::Type::Void {
                renamer.start_renaming(op.pos.get(), newop.pos.get());
            }
            operations.push(std::rc::Rc::new(newop));
        }
        let mut jump = self.jump.clone();
        renamer.rename(&mut jump);
        let mut loop_ = VectorLoop::new_rc(self.label.clone(), operations, jump);
        loop_.prefix = prefix;
        loop_.prefix_label = prefix_label;
        loop_
    }

    /// Number of ops in the loop body (excluding Label and Jump).
    pub fn body_len(&self) -> usize {
        self.operations.len()
    }

    /// Materialize an owned `Vec<Op>` view of the body for the read-only
    /// `&[Op]` scanners (`DependencyGraph::build`, `LoopVersionInfo::snapshot`,
    /// `GuardStrengthenOpt::propagate_all_forward`). These passes only inspect
    /// the ops (structure / vecinfo) and never bind args to the scanned ops'
    /// identity, so a deep clone at the call boundary is faithful; the canonical
    /// producer `OpRc`s stay in `self.operations`.
    fn operations_as_ops(&self) -> Vec<Op> {
        self.operations.iter().map(|op| (**op).clone()).collect()
    }
}

// ── vector.py:122-173: optimize_vector ─────────────────────────────────

/// vector.py:122-173: optimize_vector — top-level entry point.
///
/// Creates a VectorizingOptimizer, runs vectorization on the loop, and
/// returns the rewritten op list. The loop is modified in place.
///
/// TODO: In RPython this receives `metainterp_sd`,
/// `jitdriver_sd`, `warmstate`, `loop_info`, `loop_ops`, `jitcell_token`.
/// In Rust we receive a `VectorLoop` directly and pass cost/size params.
/// The `metainterp_sd`-dependent parts (profiler counting, logger) are
/// handled via the `Optimization` trait impl when used in the pipeline.
pub fn optimize_vector(
    loop_: &mut VectorLoop,
    cost_threshold: i32,
    vec_size: usize,
    info: &mut crate::optimizeopt::version::LoopVersionInfo,
    user_code: bool,
) -> Result<(Vec<Op>, crate::optimizeopt::vec_assoc::VecAssoc<OpRef, i64>), VectorizeError> {
    // vector.py:126-128
    if loop_.operations.is_empty() {
        return Err(VectorizeError::NotVectorizeable);
    }

    // vector.py:134 `version = info.snapshot(loop)` — register the
    // pre-vectorize loop as the single tracked version (GuardStrengthenOpt
    // asserts versions.len() == 1) and keep an untouched clone so that *any*
    // downstream failure (NotAVectorizeableLoop / NotAProfitableLoop /
    // panic-equivalent) restores the caller-visible VectorLoop to its
    // pre-vectorize shape. The clone is only used on the error path; on
    // success we hand back the vectorized ops directly.
    let label_args: Vec<OpRef> = loop_
        .label
        .getarglist()
        .iter()
        .map(|a| a.to_opref())
        .collect();
    info.snapshot(&loop_.operations_as_ops(), &label_args);
    let version = loop_.clone_loop();

    let result = (|| {
        // vector.py:142-143. `run_optimization` owns the scheduler state, so
        // it calls vector.py:135 `loop.setup_vectorization()` (and the
        // vector.py:172 `teardown_vectorization()`) against that state
        // internally, stamping each op's VectorizationInfo into the
        // `_forwarded` equivalent that `forwarded_vecinfo` reads.
        let mut opt = VectorizingOptimizer::new_with_params(cost_threshold, vec_size);
        opt.run_optimization(loop_, info, user_code)
    })();

    if result.is_err() {
        // vector.py:155 / :160: `return loop_info, version.loop.finaloplist()`.
        // Restore the pre-vectorize ops into loop_ so the caller can resume
        // from a clean state if it wants to inspect the loop further.
        *loop_ = version;
    }

    result
}

// ── compile.py:302-308: vectorization post-pass entry ──────────────────

/// compile.py:302-308 — apply the SIMD vectorizer to an optimizer-assembled
/// loop and return the rewritten op list.
///
/// `optimized_ops` is the flat loop the unroll optimizer produced:
/// `[prefix…, loop_label, body…, jump]`. Split off the loop part at the
/// final LABEL (compile.py:322 `loop_info.label_op`), run `optimize_vector`
/// on `[label] + body + jump` (compile.py:305), and re-assemble
/// `prefix + extra_before_label + [label] + loop_ops` (compile.py:327-328).
///
/// Bails to `optimized_ops` unchanged when the loop cannot or should not be
/// vectorized (NotAVectorizeableLoop / NotAProfitableLoop) — the common
/// case, matching optimize_vector's `return loop_info, loop_ops`.
///
/// No re-numbering is needed: the vectorizer assigns every op it creates
/// (unroll copies via `base_offset = max(body pos) + 1`, packed VEC ops via
/// `VecScheduleState::next_pos = max(body pos) + 1`) a position strictly
/// greater than any prefix position, because the loop body is the tail of
/// `optimized_ops`. Retained scalar body ops keep their original positions,
/// which are likewise above the prefix. The gso index constants are inline
/// `OpRef::const_int` (guard.rs:614) carrying their value on the OpRef, so
/// nothing needs registering in the constant pool.
pub fn apply_loop_vectorization(
    optimized_ops: Vec<Op>,
    vec_size: usize,
    cost_threshold: i32,
    user_code: bool,
) -> Vec<Op> {
    // compile.py:322 — the loop header the closing JUMP targets is the last
    // LABEL in the assembled trace.
    let Some(label_idx) = optimized_ops
        .iter()
        .rposition(|op| op.opcode == OpCode::Label)
    else {
        return optimized_ops;
    };
    // vector.py:147 `assert rop.is_final(loop_ops[e])` — the loop must close
    // with a JUMP for the vectorizer to model it.
    if optimized_ops
        .last()
        .map(|op| op.opcode != OpCode::Jump)
        .unwrap_or(true)
    {
        return optimized_ops;
    }
    // vector.py:146 `assert e > 0` — the body between label and jump must be
    // non-empty.
    let jump_idx = optimized_ops.len() - 1;
    if jump_idx <= label_idx + 1 {
        return optimized_ops;
    }

    let prefix: Vec<Op> = optimized_ops[..label_idx].to_vec();
    let label = optimized_ops[label_idx].clone();
    let body: Vec<Op> = optimized_ops[label_idx + 1..jump_idx].to_vec();
    let jump = optimized_ops[jump_idx].clone();

    let mut vloop = VectorLoop::new(label, body, jump);
    let mut info = crate::optimizeopt::version::LoopVersionInfo::new();

    match optimize_vector(&mut vloop, cost_threshold, vec_size, &mut info, user_code) {
        // NotAVectorizeableLoop / NotAProfitableLoop — keep the scalar loop.
        Err(_) => optimized_ops,
        Ok((loop_ops, _gso_consts)) => {
            // compile.py:327-328: `… + extra_before_label + [label_op] +
            // loop_ops`. `optimized_ops[..label_idx]` already carries the
            // preamble + extra_same_as; `vloop.align_operations`
            // (vector.py:267) is `extra_before_label`; `vloop.label` is
            // `loop_info.label_op` (vector.py:153 `info.label_op = loop.label`).
            // `loop_ops` came from `finaloplist(label=false)` so it is
            // `body + jump` without the label.
            let mut assembled = Vec::with_capacity(
                prefix.len() + vloop.align_operations.len() + 1 + loop_ops.len(),
            );
            assembled.extend(prefix);
            assembled.extend(vloop.align_operations.iter().map(|op| (**op).clone()));
            assembled.push(vloop.label.clone());
            assembled.extend(loop_ops);
            assembled
        }
    }
}

// ── vector.py:175-205: user_loop_bail_fast_path ────────────────────────

/// vector.py:175-205: user_loop_bail_fast_path — quick pre-check.
///
/// Returns `true` if the loop should be SKIPPED (bailed on) for
/// vectorization. In RPython, `user_code and user_loop_bail_fast_path()`
/// is checked before entering the optimizer.
pub fn user_loop_bail_fast_path(loop_: &VectorLoop) -> bool {
    let mut _resop_count = 0;
    let mut _vector_instr = 0;
    let mut _guard_count = 0;
    // vector.py:183: at_least_one_array_access = True  (RPython bug — always True,
    // because line 194 only ever re-assigns True.  Match upstream literal.)
    let mut at_least_one_array_access = true;

    for op in &loop_.operations {
        // vector.py:185-186: skip jit debug ops
        if op.opcode.is_jit_debug() {
            continue;
        }
        // vector.py:188-189: count vectorizable non-guard ops
        if op.opcode.to_vector().is_some() && !op.opcode.is_guard() {
            _vector_instr += 1;
        }
        _resop_count += 1;
        // vector.py:193-194: is_primitive_array_access
        if op.opcode.is_getarrayitem()
            || op.opcode.is_setarrayitem()
            || matches!(
                op.opcode,
                OpCode::RawLoadI | OpCode::RawLoadF | OpCode::RawStore
            )
        {
            at_least_one_array_access = true;
        }
        // vector.py:196-197: bail on calls
        if op.opcode.is_call() || op.opcode.is_call_assembler() {
            return true;
        }
        // vector.py:199-200
        if op.opcode.is_guard() {
            _guard_count += 1;
        }
    }
    // vector.py:202-203
    if !at_least_one_array_access {
        return true;
    }
    false
}

// ── vector.py:207-600: VectorizingOptimizer ────────────────────────────

/// vector.py:207-600: VectorizingOptimizer — the vectorization optimizer.
///
/// In RPython, this extends `Optimizer` and is the top-level optimizer for
/// the vector pass. In Rust, it implements `Optimization` and is used as
/// a sub-pass in the `Optimizer` pipeline (TODO).
///
/// The RPython-shaped methods (`run_optimization`, `unroll_loop_iterations`,
/// etc.) are provided alongside the `Optimization` trait impl.
pub struct VectorizingOptimizer {
    /// vector.py:215: self.packset = None
    packset: Option<PackSet>,
    /// vector.py:216: self.unroll_count = 0
    pub unroll_count: usize,
    /// vector.py:217: self.smallest_type_bytes = 0
    smallest_type_bytes: usize,
    /// vector.py:218: self.orig_label_args = None
    orig_label_args: Option<Vec<OpRef>>,
    /// vector.py:213: self.cost_threshold = cost_threshold
    cost_threshold: i32,
    /// vector.py:214: self.vector_ext.vec_size()
    vec_size: usize,
    /// vector.py:244: self.vector_ext.should_align_unroll
    /// True on x86 SSE (default); a future backend abstraction can set
    /// false for platforms where alignment unrolling is not beneficial.
    should_align_unroll: bool,

    // ── Rust Optimization trait fields (TODO) ──
    // These support the sub-pass integration path where VectorizingOptimizer
    // is used inside an Optimizer pipeline via the Optimization trait.
    /// Buffered loop body ops (populated by propagate_forward).
    body_ops: Vec<Op>,
    /// Whether we're inside a Label..Jump loop body.
    in_loop: bool,
    /// schedule.py:669: label inputargs — populated on Label entry.
    label_args: Vec<OpRef>,
    /// The loop LABEL op, held (not emitted) from the Label entry until the
    /// Jump so the streaming path can build a VectorLoop and let post_schedule/
    /// finaloplist place the prefix BEFORE the loop entry. majit-only: RPython
    /// has a single VectorLoop entry (vector.py) and never emits the label
    /// eagerly into a streaming op list.
    pending_label: Option<Op>,
    /// Deferred profiler counter: OPT_VECTORIZE_TRY.
    pub(crate) opt_vectorize_try_emitted: usize,
    /// Deferred profiler counter: OPT_VECTORIZED.
    pub(crate) opt_vectorized_emitted: usize,
}

impl VectorizingOptimizer {
    /// vector.py:210-218: __init__ — default constructor for sub-pass use.
    pub fn new() -> Self {
        VectorizingOptimizer {
            packset: None,
            unroll_count: 0,
            smallest_type_bytes: 0,
            orig_label_args: None,
            cost_threshold: 0,
            vec_size: 16, // SSE default
            should_align_unroll: true,
            body_ops: Vec::new(),
            in_loop: false,
            label_args: Vec::new(),
            pending_label: None,
            opt_vectorize_try_emitted: 0,
            opt_vectorized_emitted: 0,
        }
    }

    /// Constructor with explicit parameters (for standalone optimize_vector).
    pub fn new_with_params(cost_threshold: i32, vec_size: usize) -> Self {
        let mut opt = Self::new();
        opt.cost_threshold = cost_threshold;
        opt.vec_size = vec_size;
        opt
    }

    // ── vector.py:220-271: run_optimization ────────────────────────────

    /// vector.py:220-271: run_optimization — the main vectorization pipeline.
    ///
    /// 1. Find smallest type → determine unroll count
    /// 2. Analyse index calculations → reorder for guard hoisting
    /// 3. Unroll the loop
    /// 4. Build dependency graph → find adjacent memory refs
    /// 5. Extend and combine packset
    /// 6. Schedule with cost model
    /// 7. Guard strengthening
    /// 8. Re-schedule for cleanup
    pub fn run_optimization(
        &mut self,
        loop_: &mut VectorLoop,
        info: &mut crate::optimizeopt::version::LoopVersionInfo,
        user_code: bool,
    ) -> Result<(Vec<Op>, crate::optimizeopt::vec_assoc::VecAssoc<OpRef, i64>), VectorizeError>
    {
        // vector.py:221
        self.orig_label_args = Some(
            loop_
                .label
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect(),
        );

        // vector.py:222
        self.linear_find_smallest_type(loop_);
        let byte_count = self.smallest_type_bytes;

        // vector.py:224
        let vsize = self.vec_size;

        // vector.py:227-235: bail checks
        if vsize == 0 {
            return Err(VectorizeError::NotVectorizeable);
        }
        if byte_count == 0 {
            return Err(VectorizeError::NotVectorizeable);
        }
        if loop_.label.opcode != OpCode::Label {
            return Err(VectorizeError::NotVectorizeable);
        }

        // vector.py:237-240: analyse_index_calculations → reorder.
        // resoperation.py:181 reads `op.getarg(1).value` off the inline
        // ConstInt; the standalone pass has no Optimizer context, so an
        // inline `OpRef::ConstInt` is the only — and the faithful — const
        // source for INT_SIGNEXT bytesize and adjacent-ref index detection.
        let constant_of = |opref: OpRef| -> Option<i64> { opref.as_const_int() };
        if let Some(graph) = self.analyse_index_calculations(loop_, &constant_of) {
            let schedule = schedule_operations(&graph);
            if schedule.len() == loop_.operations.len() {
                // Reorder by cheaply cloning the `OpRc`s — identity preserved
                // (same producer boxes, new order), no op deep-clone.
                let scheduled: Vec<OpRc> = schedule
                    .iter()
                    .map(|&i| loop_.operations[i].clone())
                    .collect();
                loop_.operations = scheduled;
            }
        }

        // vector.py:243-247: unroll.
        // RPython: `align_unroll = self.unroll_count == 1 and
        //           self.vector_ext.should_align_unroll`
        // should_align_unroll is a backend flag (True on x86 SSE, False on
        // some other backends). We default to true since only x86_64 is
        // supported; the flag is stored on self so a future backend
        // abstraction can override it.
        self.unroll_count = Self::get_unroll_count(byte_count, vsize);
        let align_unroll = self.unroll_count == 1 && self.should_align_unroll;
        loop_.unroll_loop_iterations(self.unroll_count, align_unroll);

        // vector.py:250-253: vectorize — build graph, find adjacent memory refs
        let graph = DependencyGraph::build(&loop_.operations_as_ops(), &constant_of);
        // VecScheduleState is created before find_adjacent_memory_refs/
        // extend_packset because PackSet::can_be_packed now consults it via
        // isomorphic (vector.py: packset.can_be_packed reaches
        // state for accumulation/invariant lookups; pre-state form was a
        // pre-rebase fork).
        let start_pos = loop_
            .operations
            .iter()
            .map(|op| op.pos.get().raw())
            .max()
            .unwrap_or(0)
            + 1;
        let mut sched_state = VecScheduleState::new(start_pos);
        // vector.py:135 loop.setup_vectorization()
        loop_.setup_vectorization(&mut sched_state, &constant_of);
        // vector.py:606-609 CostModel.__init__: savings = 0, threshold stored
        // separately. Initializing savings = self.cost_threshold inverted the
        // gate — a positive threshold made profitable() trivially true.
        let costmodel = CostModel {
            min_pack_size: 2,
            pack_cost: 2,
            scalar_save: 1,
            savings: 0,
        };
        sched_state.costmodel = costmodel;

        self.find_adjacent_memory_refs(&graph, loop_, &mut sched_state);

        // vector.py:253-254: extend and combine — combine_packset raises
        // NotAVectorizeableLoop on an empty packset (vector.py:468-470).
        self.extend_packset(&graph, &mut sched_state);
        self.combine_packset()?;

        // vector.py:254-258: schedule with cost model
        let packset = self.packset.take().unwrap_or_default();

        // Populate inputargs/seen from label args
        for arg in loop_.label.getarglist().iter() {
            sched_state.inputargs.insert(arg.to_opref(), ());
        }
        let mut seen: VecSet<OpRef> = loop_
            .label
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect();

        // accumulate_prepare for accumulation packs
        for pack in &packset.packs {
            if !pack.is_accumulating {
                continue;
            }
            let first_op = &loop_.operations[pack.members[0]];
            if first_op.opcode.is_guard() {
                continue;
            }
            let pos = pack.position.max(0) as usize;
            let seed = if pos < first_op.num_args() {
                first_op.arg(pos).to_opref()
            } else {
                OpRef::NONE
            };
            let operator = pack.operator.unwrap_or('+');
            for &member_idx in &pack.members {
                let op = &loop_.operations[member_idx];
                if op.opcode.is_guard() {
                    continue;
                }
                sched_state.accumulation.insert(
                    op.pos.get(),
                    AccumEntry {
                        seed,
                        operator,
                        accum_opcode: pack.scalar_opcode,
                    },
                );
            }
            let is_float = first_op.opcode.result_type() == majit_ir::Type::Float;
            if is_float {
                return Err(VectorizeError::NotVectorizeable);
            }
            let datatype = 'i';
            // schedule.py:838-840: bytesize = pack.getbytesize() — read the
            // seed's forwarded VectorizationInfo from the same cache that
            // VectorizationInfo(op) populated, not a separate inline slot.
            let bytesize: i32 = sched_state
                .forwarded_vecinfo_for_ref(seed, &loop_.operations)
                .getbytesize() as i32;
            let vec_reg_size: i32 = self.vec_size as i32;
            let count = (vec_reg_size / bytesize) as usize;
            let signed = true;

            let vec_create =
                sched_state.create_vec_op(OpCode::VecI, &[], datatype, bytesize, signed, count);
            let zero_vec = vec_create.pos.get();
            sched_state
                .invariant_oplist
                .push(std::rc::Rc::new(vec_create));

            let xor_op = sched_state.create_vec_op(
                OpCode::VecIntXor,
                &[zero_vec, zero_vec],
                datatype,
                bytesize,
                signed,
                count,
            );
            let zeroed_vec = xor_op.pos.get();
            sched_state.invariant_oplist.push(std::rc::Rc::new(xor_op));

            // VEC_PACK_I args are [vector, scalar, index, count]; index/count
            // are inline ConstInt (history.py:227), not pool indices.
            let zero_const = OpRef::const_int(0);
            let one_const = OpRef::const_int(1);
            let pack_op = sched_state.create_vec_op(
                OpCode::VecPackI,
                &[zeroed_vec, seed, zero_const, one_const],
                datatype,
                bytesize,
                signed,
                count,
            );
            let seed_vec = pack_op.pos.get();
            sched_state.invariant_oplist.push(std::rc::Rc::new(pack_op));

            sched_state.accumulation.insert(
                seed,
                AccumEntry {
                    seed,
                    operator,
                    accum_opcode: pack.scalar_opcode,
                },
            );
            sched_state.setvector_of_box(seed, 0, seed_vec);
            sched_state.renamer.start_renaming(seed, seed_vec);
        }

        // Build node→pack mapping
        let mut node_to_pack: crate::optimizeopt::vec_assoc::VecAssoc<usize, usize> =
            crate::optimizeopt::vec_assoc::VecAssoc::new();
        for (pi, group) in packset.packs.iter().enumerate() {
            for &idx in &group.members {
                node_to_pack.insert(idx, pi);
            }
        }

        let mut pack_emitted = vec![false; packset.packs.len()];
        let mut pack_visited_count = vec![0usize; packset.packs.len()];

        let scheduled_order = schedule_operations(&graph);
        for &node_idx in &scheduled_order {
            if let Some(&pack_idx) = node_to_pack.get(&node_idx) {
                pack_visited_count[pack_idx] += 1;
                let pack = &packset.packs[pack_idx];
                let all_ready = pack_visited_count[pack_idx] == pack.members.len();

                if all_ready && !pack_emitted[pack_idx] {
                    pack_emitted[pack_idx] = true;
                    for &member_idx in &pack.members {
                        let mut member_op = (*loop_.operations[member_idx]).clone();
                        pre_emit_guard_accum(&sched_state, &mut member_op);
                        sched_state.renamer.rename(&mut member_op);
                        // Bind the renamed args (e.g. accumulation-renamed vec
                        // ops) to their producers already in `oplist`.
                        sched_state.rebind_op_args(&member_op);
                        // schedule.py:677-680: packed members are emitted via
                        // mark_emitted(node, unpack=False) — renamed but NOT
                        // recorded in `seen`. They live only in box_to_vbox
                        // (turn_into_vector → setvector_of_box) so a later
                        // ensure_args_unpacked materializes a VecUnpack when the
                        // result is used as a scalar (e.g. carried by the jump).
                        // The renamed op becomes the new canonical producer box
                        // at this slot, so later consumers bind to this `Rc`.
                        loop_.operations[member_idx] = std::rc::Rc::new(member_op);
                    }
                    turn_into_vector(&mut sched_state, pack, &loop_.operations);
                }
            } else {
                let mut scalar_op = (*loop_.operations[node_idx]).clone();
                pre_emit_guard_accum(&sched_state, &mut scalar_op);
                sched_state.renamer.rename(&mut scalar_op);
                ensure_args_unpacked(&mut sched_state, &mut scalar_op, &mut seen);
                // Bind the renamed / unpacked args to their producer boxes in
                // the already-emitted oplist (no position-only mint).
                sched_state.rebind_op_args(&scalar_op);
                seen.insert(scalar_op.pos.get());
                sched_state.append_to_oplist(scalar_op);
            }
        }

        // vector.py:515-520 schedule(): `walk_and_emit` then, only when the
        // cost model is profitable, `post_schedule()`. An unprofitable loop
        // returns *before* post_schedule, so loop_ is never mutated by it.
        // vector.py:256-258 then raises NotAProfitableLoop on the same check;
        // run_optimization collapses both into this single early Err.
        if !sched_state.costmodel.profitable() {
            return Err(VectorizeError::NotProfitable);
        }

        // schedule.py:762-779: VecScheduleState.post_schedule — moves
        // invariant_oplist into loop.prefix and routes invariant_vector_vars
        // through prefix_label/jump.
        sched_state.post_schedule(loop_, &mut seen);

        // vector.py:259-260: gso = GuardStrengthenOpt(graph.index_vars);
        //                    gso.propagate_all_forward(info, loop, user_code).
        // Strengthen and de-duplicate the guards in the scheduled body.
        // `graph` is the vectorize-phase dependency graph (vector.py:250); its
        // index_vars drive index-guard strength reduction. `info` carries the
        // single snapshot version (versions.len() == 1, asserted by
        // propagate_all_forward). The returned const_values map IndexVar-
        // materialized constant OpRefs to their i64 values; the caller must
        // register them in the trace constant pool.
        let mut gso = crate::optimizeopt::guard::GuardStrengthenOpt::new(graph.index_vars.clone());
        let gso_label_args: Vec<OpRef> = loop_
            .label
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect();
        let (strengthened, gso_consts) =
            gso.propagate_all_forward(&loop_.operations_as_ops(), info, &gso_label_args, user_code);
        // The guard-strengthened body is a fresh op list; wrap each into the
        // canonical producer `OpRc` as it re-enters the buffer.
        loop_.operations = strengthened.into_iter().map(std::rc::Rc::new).collect();

        // vector.py:262-265: re-schedule the trace to drop pure operations left
        // dead by guard strengthening (graph = DependencyGraph(loop);
        // state = SchedulerState(cpu, graph); state.schedule()). TODO: the base
        // SchedulerState walk_and_emit is not yet ported; the cleanup reschedule
        // is deferred. The body is still correct without it, only less optimal.

        // vector.py:267-269: extra_before_label = loop.align_operations;
        // for op in loop.align_operations: op.set_forwarded(None).
        // We hand the align_operations back through `loop_.align_operations`
        // (already populated by `unroll_loop_iterations` on the align arm);
        // clearing the pos-keyed forwarded scratch matches the upstream
        // `set_forwarded(None)` reset so post-vectorize passes don't see
        // stale VectorizationInfo; the permanent `Op.vecinfo` is preserved.
        for op in &loop_.align_operations {
            sched_state.clear_op_forwarded_vecinfo(op.pos.get());
        }

        // vector.py:172 `finally: loop.teardown_vectorization()`. The
        // earlier `?`/`return Err` exits drop `sched_state` instead, which
        // discards the same pos-keyed forwarded store.
        loop_.teardown_vectorization(&mut sched_state);

        // vector.py:271: return loop.finaloplist(jitcell_token, reset_label_token=False).
        // post_schedule already set loop_.operations / prefix / prefix_label / jump,
        // so finaloplist concatenates [prefix][prefix_label] operations [jump].
        // TODO: thread jitcell_token through when optimize_vector is wired to the
        // compiler; None here skips the descr/token wiring (faithful for the
        // currently-disconnected compile path). `label=false` matches RPython's
        // default (the vector.py:271 call omits the `label` argument).
        let ops = loop_.finaloplist(None, false, false, &mut sched_state);
        Ok((ops, gso_consts))
    }

    // ── vector.py:273-344: unroll_loop_iterations ──────────────────────

    /// vector.py:359-367: linear_find_smallest_type — scan ops for the
    /// smallest array element byte size to determine SIMD width.
    pub fn linear_find_smallest_type(&mut self, loop_: &VectorLoop) {
        for op in &loop_.operations {
            if op.opcode.is_getarrayitem()
                || op.opcode.is_setarrayitem()
                || matches!(
                    op.opcode,
                    OpCode::RawLoadI | OpCode::RawLoadF | OpCode::RawStore
                )
            {
                if let Some(descr) = op.getdescr() {
                    if let Some(ad) = descr.as_array_descr() {
                        let item_size = ad.item_size();
                        if self.smallest_type_bytes == 0 || item_size < self.smallest_type_bytes {
                            self.smallest_type_bytes = item_size;
                        }
                    }
                }
            }
        }
    }

    /// vector.py:369-376: get_unroll_count — compute how many times to
    /// unroll based on SIMD register width and smallest type.
    pub fn get_unroll_count(smallest_type_bytes: usize, simd_reg_bytes: usize) -> usize {
        if smallest_type_bytes == 0 {
            return 0;
        }
        let count = simd_reg_bytes / smallest_type_bytes;
        count.saturating_sub(1) // already unrolled once
    }

    // ── vector.py:346-357: copy_guard_descr ────────────────────────────

    /// vector.py:346-357: copy_guard_descr — clone guard descriptor and
    /// rename fail args during unrolling.
    fn copy_guard_descr(renamer: &Renamer, copied_op: &mut Op) {
        // vector.py:349-350: descr.clone() — already cloned by copy_resop
        // vector.py:351: failargs = renamer.rename_failargs(copied_op, clone=True)
        if let Some(fail_args) = copied_op.getfailargs() {
            let fail_args_oprefs: Vec<OpRef> = fail_args.iter().map(|a| a.to_opref()).collect();
            let renamed = renamer.rename_failargs(&fail_args_oprefs);
            copied_op.setfailargs(renamed.iter().map(|r| BoxRef::from_opref(*r)).collect());
        }
    }

    // ── vector.py:378-402: find_adjacent_memory_refs ───────────────────

    /// vector.py:378-402: find_adjacent_memory_refs — seed the packset
    /// with pairs of adjacent memory accesses.
    fn find_adjacent_memory_refs(
        &mut self,
        graph: &DependencyGraph,
        _loop: &VectorLoop,
        state: &mut VecScheduleState,
    ) {
        let _vec_size = self.vec_size;
        let mut packset = PackSet::new();

        // vector.py:391-402: for each pair of memory refs, check adjacency
        let memory_refs: Vec<(usize, &crate::optimizeopt::dependency::MemoryRef)> = graph
            .memory_refs
            .iter()
            .map(|(&idx, mref)| (idx, mref))
            .collect();

        for &(node_a, memref_a) in &memory_refs {
            for &(node_b, memref_b) in &memory_refs {
                if node_a == node_b {
                    continue;
                }
                // vector.py:399: memref_a.is_adjacent_after(memref_b)
                if memref_a.is_adjacent_after(memref_b) {
                    // vector.py:400-401: packset.can_be_packed(node_a, node_b, None, False)
                    match packset.can_be_packed(state, node_a, node_b, None, false, graph) {
                        Ok(Some(pair)) => packset.add_pack(pair),
                        _ => {}
                    }
                }
            }
        }

        self.packset = Some(packset);
    }

    // ── vector.py:404-458: extend_packset / follow_def_uses / follow_use_defs

    /// vector.py:404-425: extend_packset — follow dependency chains to find
    /// more candidates to put into pairs.
    pub fn extend_packset(&mut self, graph: &DependencyGraph, state: &mut VecScheduleState) {
        let packset = match self.packset.as_mut() {
            Some(ps) => ps,
            None => return,
        };
        let mut pack_count = packset.num_packs();
        loop {
            // vector.py:411-415: follow_def_uses for each 2-pack
            let num_packs = packset.packs.len();
            for i in 0..num_packs {
                if packset.packs[i].members.len() == 2 {
                    let pack_snap = packset.packs[i].clone();
                    Self::follow_def_uses(packset, &pack_snap, graph, state);
                }
            }
            if pack_count == packset.num_packs() {
                // vector.py:417-423: no new packs from def-uses, try use-defs
                pack_count = packset.num_packs();
                let num_packs = packset.packs.len();
                for i in 0..num_packs {
                    if packset.packs[i].members.len() == 2 {
                        let pack_snap = packset.packs[i].clone();
                        Self::follow_use_defs(packset, &pack_snap, graph, state);
                    }
                }
                if pack_count == packset.num_packs() {
                    break;
                }
            }
            pack_count = packset.num_packs();
        }
    }

    /// vector.py:427-442: follow_use_defs — for a 2-pack, check if
    /// dependencies of leftmost/rightmost can form new pairs.
    fn follow_use_defs(
        packset: &mut PackSet,
        pack: &Pack,
        graph: &DependencyGraph,
        state: &mut VecScheduleState,
    ) {
        debug_assert!(pack.members.len() == 2);
        let left_idx = pack.members[0];
        let right_idx = *pack.members.last().unwrap();

        // vector.py:429-430: for ldep in pack.leftmost(True).depends()
        let l_deps: Vec<usize> = graph.nodes[left_idx].deps.clone();
        let r_deps: Vec<usize> = graph.nodes[right_idx].deps.clone();

        for &l_dep in &l_deps {
            for &r_dep in &r_deps {
                // vector.py:434-437: left = lnode.getoperation();
                // args = pack.leftmost().getarglist(); if left not in args: continue
                let dep_opref = graph.nodes[l_dep].op.pos.get();
                let left_args = graph.nodes[left_idx].op.getarglist();
                if !left_args.iter().any(|a| a.to_opref() == dep_opref) {
                    continue;
                }
                let l_op = &graph.nodes[l_dep].op;
                let r_op = &graph.nodes[r_dep].op;
                // vector.py:438-439: isomorphic and lnode.is_before(rnode)
                if isomorphic(state, l_op, r_op) && l_dep < r_dep {
                    match packset.can_be_packed(state, l_dep, r_dep, Some(pack), false, graph) {
                        Ok(Some(pair)) => packset.add_pack(pair),
                        Err(_) => return,
                        _ => {}
                    }
                }
            }
        }
    }

    /// vector.py:444-458: follow_def_uses — for a 2-pack, check if users
    /// of leftmost/rightmost can form new pairs via can_be_packed.
    fn follow_def_uses(
        packset: &mut PackSet,
        pack: &Pack,
        graph: &DependencyGraph,
        state: &mut VecScheduleState,
    ) {
        debug_assert!(pack.members.len() == 2);
        let left_idx = pack.members[0];
        let right_idx = *pack.members.last().unwrap();
        let left_opref = graph.nodes[left_idx].op.pos.get();

        // vector.py:446-447: for ldep in pack.leftmost(node=True).provides()
        let l_users: Vec<usize> = graph.nodes[left_idx].users.clone();
        let r_users: Vec<usize> = graph.nodes[right_idx].users.clone();

        for &l_user in &l_users {
            for &r_user in &r_users {
                // vector.py:451-453: left = pack.leftmost()
                // args = lnode.getoperation().getarglist()
                // if left not in args: continue
                if !graph.nodes[l_user]
                    .op
                    .getarglist()
                    .iter()
                    .any(|a| a.to_opref() == left_opref)
                {
                    continue;
                }
                let l_op = &graph.nodes[l_user].op;
                let r_op = &graph.nodes[r_user].op;
                // vector.py:454-455: isomorphic and lnode.is_before(rnode)
                if isomorphic(state, l_op, r_op) && l_user < r_user {
                    match packset.can_be_packed(state, l_user, r_user, Some(pack), true, graph) {
                        Ok(Some(pair)) => packset.add_pack(pair),
                        Err(_) => return,
                        _ => {}
                    }
                }
            }
        }
    }

    // ── vector.py:460-494: combine_packset ─────────────────────────────

    /// vector.py:460-496: combine_packset — merge adjacent 2-packs into
    /// larger packs, then split overloaded packs.
    pub fn combine_packset(&mut self) -> Result<(), NotAVectorizeableLoop> {
        let packset = match self.packset.as_mut() {
            Some(ps) => ps,
            None => return Err(NotAVectorizeableLoop),
        };

        // vector.py:468-470: empty packset → raise NotAVectorizeableLoop
        if packset.packs.is_empty() {
            return Err(NotAVectorizeableLoop);
        }

        // vector.py:474-494: iterative merge
        loop {
            let len_before = packset.packs.len();
            packset.try_merge_packs();
            if packset.packs.len() == len_before {
                break;
            }
        }

        // vector.py:496: self.packset.split_overloaded_packs(self.cpu.vector_ext)
        // TODO: split_overloaded_packs not yet ported.
        // RPython splits packs that exceed the vector register size and
        // removes packs that are too small (< FULL load). This requires
        // pack_load() and Pack.FULL which depend on vectorization info
        // infrastructure not yet available in Rust.
        Ok(())
    }

    // ── vector.py:515-521: schedule ────────────────────────────────────

    /// vector.py:515-521: schedule — run the scheduler on the given state.
    fn schedule_state(_state: &mut VecScheduleState, _graph: &DependencyGraph) {
        // vector.py:516: state.prepare() — handled by caller
        // vector.py:517-518: scheduler.walk_and_emit(state) — scheduling
        //   is done inline in run_optimization via schedule_operations
        // vector.py:520: state.post_schedule() — handled by caller
    }

    // ── vector.py:523-583: analyse_index_calculations ──────────────────

    /// vector.py:523-583: analyse_index_calculations — move guarding
    /// instructions (and their dependencies) to the loop header.
    ///
    /// This ensures guards fail "early" and relax dependencies, which is
    /// a prerequisite for vectorization.
    ///
    /// TODO: The full RPython implementation requires:
    /// - DependencyGraph.imaginary_node() — synthetic graph nodes
    /// - Node.iterate_paths() — path enumeration with blacklist
    /// - Path.is_always_pure() — purity analysis along paths
    /// - Node.remove_edge_to() / edge_to() — graph mutation
    /// These dependency.rs primitives are not yet ported. Until they are,
    /// return None unconditionally — the earlier "zero-dep guard" heuristic
    /// did not actually rewire the graph the way RPython does, and feeding
    /// the unmodified graph back to the caller as a reschedule basis was a
    /// silent divergence. mark_guard is similarly stubbed.
    fn analyse_index_calculations(
        &self,
        _loop_: &VectorLoop,
        _constant_of: &dyn Fn(OpRef) -> Option<i64>,
    ) -> Option<DependencyGraph> {
        None
    }

    // ── vector.py:585-599: mark_guard ──────────────────────────────────

    /// vector.py:585-599: mark_guard — marks a guard as an early exit
    /// by attaching a CompileLoopVersionDescr and setting failargs to
    /// the label's input args.
    ///
    /// TODO: CompileLoopVersionDescr is not yet ported
    /// to Rust. When it is, this should create the descr and attach it.
    fn mark_guard(&self, _guard_idx: usize, _loop_: &VectorLoop) {
        // vector.py:588-594: create CompileLoopVersionDescr, copy attrs
        // vector.py:595-599: set failargs to label.getarglist_copy()
        // Requires CompileLoopVersionDescr from compile.rs — not yet ported.
    }

    // ── Optimization trait helper: try_vectorize ───────────────────────

    /// Attempt to vectorize the buffered loop body (Optimization trait path).
    ///
    /// This is the sub-pass equivalent of run_optimization, used when
    /// VectorizingOptimizer is embedded in an Optimizer pipeline.
    fn try_vectorize(&mut self, ctx: &mut OptContext, loop_: &mut VectorLoop) -> Option<Vec<Op>> {
        if loop_.operations.len() < 4 {
            return None;
        }

        let constant_of = |opref: OpRef| -> Option<i64> {
            ctx.get_box_replacement_box(opref)
                .and_then(|cb| cb.const_int())
        };

        // Seed fresh vector OpRefs one past the highest live position, mirroring
        // run_optimization (`max(op.pos.raw()) + 1`). A count-based start
        // (`new_operations.len() + operations.len()`) is not guaranteed to exceed
        // existing OpRef::raw() values — and since the LABEL is now held back (no
        // longer in ctx.new_operations), the sum can land on a live body position,
        // so create_vec_op would reuse an already-live OpRef and corrupt SSA.
        // (ctx.new_operations holds Rc<Op>, loop_ fields hold Op; collapse each
        // source to its OpRef position so the chain is uniform.)
        let start_pos = ctx
            .new_operations
            .iter()
            .map(|op| op.pos.get())
            .chain(std::iter::once(loop_.label.pos.get()))
            .chain(loop_.operations.iter().map(|op| op.pos.get()))
            .chain(std::iter::once(loop_.jump.pos.get()))
            .filter(|pos| !pos.is_none())
            .map(|pos| pos.raw())
            .max()
            .unwrap_or(0)
            + 1;
        let mut sched_state = VecScheduleState::new(start_pos);
        // vector.py:135 loop.setup_vectorization() — stamps
        // VectorizationInfo on each op. INT_SIGNEXT reads arg1's const
        // value for bytesize (resoperation.py:181); the constant_of
        // resolver feeds that through so the inline vecinfo slot matches
        // PyPy's VectorizationInfo(op) constructor.
        loop_.setup_vectorization(&mut sched_state, &constant_of);

        // First schedule operations for ILP before packing.
        let dep_graph = DependencyGraph::build(&loop_.operations_as_ops(), &constant_of);
        let schedule = schedule_operations(&dep_graph);
        if schedule.len() == loop_.operations.len() {
            // Reorder by cheaply cloning the `OpRc`s (identity preserved).
            let scheduled: Vec<OpRc> = schedule
                .iter()
                .map(|&i| loop_.operations[i].clone())
                .collect();
            loop_.operations = scheduled;
            loop_.setup_vectorization(&mut sched_state, &constant_of);
        }

        // Then rebuild the dependency graph and find packs.
        let dep_graph = DependencyGraph::build(&loop_.operations_as_ops(), &constant_of);
        let seed_packs = dep_graph.find_packable_groups();
        if seed_packs.is_empty() {
            return None;
        }
        let mut pack_set = PackSet::new();
        for pack in seed_packs {
            pack_set.add_pack(pack);
        }
        Self::extend_packset_static(&mut pack_set, &dep_graph, &mut sched_state);
        Self::combine_packset_static(&mut pack_set);
        let profitable = pack_set.packs;
        if profitable.is_empty() {
            return None;
        }

        // schedule.py:666-670: prepare() — populate inputargs and seen
        for &arg in &self.label_args {
            sched_state.inputargs.insert(arg, ());
        }
        let mut seen: VecSet<OpRef> = self.label_args.iter().copied().collect();

        // accumulate_prepare
        for pack in &profitable {
            if !pack.is_accumulating {
                continue;
            }
            let first_op = &loop_.operations[pack.members[0]];
            if first_op.opcode.is_guard() {
                continue;
            }
            let pos = pack.position.max(0) as usize;
            let seed = if pos < first_op.num_args() {
                first_op.arg(pos).to_opref()
            } else {
                OpRef::NONE
            };
            let operator = pack.operator.unwrap_or('+');
            for &member_idx in &pack.members {
                let op = &loop_.operations[member_idx];
                if op.opcode.is_guard() {
                    continue;
                }
                sched_state.accumulation.insert(
                    op.pos.get(),
                    AccumEntry {
                        seed,
                        operator,
                        accum_opcode: pack.scalar_opcode,
                    },
                );
            }
            let is_float = first_op.opcode.result_type() == majit_ir::Type::Float;
            if is_float {
                return None;
            }
            let datatype = 'i';
            // schedule.py:838-840: bytesize = pack.getbytesize() — read the
            // seed's forwarded VectorizationInfo from the same cache that
            // VectorizationInfo(op) populated, not a separate inline slot.
            let bytesize: i32 = sched_state
                .forwarded_vecinfo_for_ref(seed, &loop_.operations)
                .getbytesize() as i32;
            let vec_reg_size: i32 = self.vec_size as i32;
            let count = (vec_reg_size / bytesize) as usize;
            let signed = true;

            let vec_create =
                sched_state.create_vec_op(OpCode::VecI, &[], datatype, bytesize, signed, count);
            let zero_vec = vec_create.pos.get();
            sched_state
                .invariant_oplist
                .push(std::rc::Rc::new(vec_create));

            let xor_op = sched_state.create_vec_op(
                OpCode::VecIntXor,
                &[zero_vec, zero_vec],
                datatype,
                bytesize,
                signed,
                count,
            );
            let zeroed_vec = xor_op.pos.get();
            sched_state.invariant_oplist.push(std::rc::Rc::new(xor_op));

            // vector.py:866-869: pack the seed scalar into position 0
            let zero_const = OpRef::const_int(0);
            let one_const = OpRef::const_int(1);
            let pack_op = sched_state.create_vec_op(
                OpCode::VecPackI,
                &[zeroed_vec, seed, zero_const, one_const],
                datatype,
                bytesize,
                signed,
                count,
            );
            let seed_vec = pack_op.pos.get();
            sched_state.invariant_oplist.push(std::rc::Rc::new(pack_op));

            sched_state.accumulation.insert(
                seed,
                AccumEntry {
                    seed,
                    operator,
                    accum_opcode: pack.scalar_opcode,
                },
            );
            sched_state.setvector_of_box(seed, 0, seed_vec);
            sched_state.renamer.start_renaming(seed, seed_vec);
        }

        // Build node→pack mapping
        let mut node_to_pack: crate::optimizeopt::vec_assoc::VecAssoc<usize, usize> =
            crate::optimizeopt::vec_assoc::VecAssoc::new();
        for (pi, group) in profitable.iter().enumerate() {
            for &idx in &group.members {
                node_to_pack.insert(idx, pi);
            }
        }

        let mut pack_emitted = vec![false; profitable.len()];
        let mut pack_visited_count = vec![0usize; profitable.len()];

        let scheduled_order = schedule_operations(&dep_graph);
        for &node_idx in &scheduled_order {
            if let Some(&pack_idx) = node_to_pack.get(&node_idx) {
                pack_visited_count[pack_idx] += 1;
                let pack = &profitable[pack_idx];
                let all_ready = pack_visited_count[pack_idx] == pack.members.len();

                if all_ready && !pack_emitted[pack_idx] {
                    pack_emitted[pack_idx] = true;
                    for &member_idx in &pack.members {
                        let mut member_op = (*loop_.operations[member_idx]).clone();
                        pre_emit_guard_accum(&sched_state, &mut member_op);
                        sched_state.renamer.rename(&mut member_op);
                        // Bind the renamed args (e.g. accumulation-renamed vec
                        // ops) to their producers already in `oplist`.
                        sched_state.rebind_op_args(&member_op);
                        // schedule.py:677-680: packed members are emitted via
                        // mark_emitted(node, unpack=False) — renamed but NOT
                        // recorded in `seen`. They live only in box_to_vbox
                        // (turn_into_vector → setvector_of_box) so a later
                        // ensure_args_unpacked materializes a VecUnpack when the
                        // result is used as a scalar (e.g. carried by the jump).
                        // The renamed op becomes the new canonical producer box
                        // at this slot, so later consumers bind to this `Rc`.
                        loop_.operations[member_idx] = std::rc::Rc::new(member_op);
                    }
                    turn_into_vector(&mut sched_state, pack, &loop_.operations);
                }
            } else {
                let mut scalar_op = (*loop_.operations[node_idx]).clone();
                pre_emit_guard_accum(&sched_state, &mut scalar_op);
                sched_state.renamer.rename(&mut scalar_op);
                ensure_args_unpacked(&mut sched_state, &mut scalar_op, &mut seen);
                // Bind the renamed / unpacked args to their producer boxes in
                // the already-emitted oplist (no position-only mint).
                sched_state.rebind_op_args(&scalar_op);
                seen.insert(scalar_op.pos.get());
                sched_state.append_to_oplist(scalar_op);
            }
        }

        // vector.py:515-520 schedule(): post_schedule runs only when the cost
        // model is profitable; an unprofitable loop returns before post_schedule
        // mutates loop_ (matches the run_optimization path and PyPy).
        if !sched_state.costmodel.profitable() {
            return None;
        }

        // schedule.py:762-779: VecScheduleState.post_schedule. Moves
        // invariant_oplist into loop_.prefix and routes invariant_vector_vars
        // through prefix_label/jump renaming. Reachable in the streaming path
        // because propagate_forward holds the LABEL (self.pending_label) until
        // the JUMP, builds this VectorLoop, and emits the finaloplist result —
        // so prefix ops land BEFORE the loop entry, not inside the body.
        sched_state.post_schedule(loop_, &mut seen);

        // Emit the original loop label only when post_schedule did NOT mint a
        // prefix_label (which replaces the label as the vectorized loop entry):
        //   - no invariants → prefix_label None → label=true  → [label] body [jump]
        //   - invariants    → prefix_label Some → label=false → prefix [prefix_label] body [jump]
        // jitcell_token=None: copy_and_change preserves descr, so prefix_label
        // inherits the label's loop token and the rewritten jump inherits the
        // jump's token — for a loop these are the same token, so the jump
        // correctly targets prefix_label. TODO: thread a JitCellToken when the
        // compile path is un-gated so finaloplist mints fresh prefix-label
        // tokens (vector.rs:156-185).
        // vector.py:172 `finally: loop.teardown_vectorization()`. The earlier
        // `return None` exits drop `sched_state` instead, discarding the same
        // pos-keyed forwarded store.
        loop_.teardown_vectorization(&mut sched_state);

        let include_label = loop_.prefix_label.is_none();
        Some(loop_.finaloplist(None, false, include_label, &mut sched_state))
    }

    // ── Static variants for extend/combine (used by try_vectorize) ─────

    fn extend_packset_static(
        pack_set: &mut PackSet,
        graph: &DependencyGraph,
        state: &mut VecScheduleState,
    ) {
        let mut pack_count = pack_set.num_packs();
        loop {
            let num_packs = pack_set.packs.len();
            for i in 0..num_packs {
                if pack_set.packs[i].members.len() == 2 {
                    let pack_snap = pack_set.packs[i].clone();
                    Self::follow_def_uses(pack_set, &pack_snap, graph, state);
                }
            }
            if pack_count == pack_set.num_packs() {
                pack_count = pack_set.num_packs();
                let num_packs = pack_set.packs.len();
                for i in 0..num_packs {
                    if pack_set.packs[i].members.len() == 2 {
                        let pack_snap = pack_set.packs[i].clone();
                        Self::follow_use_defs(pack_set, &pack_snap, graph, state);
                    }
                }
                if pack_count == pack_set.num_packs() {
                    break;
                }
            }
            pack_count = pack_set.num_packs();
        }
    }

    fn combine_packset_static(pack_set: &mut PackSet) {
        if pack_set.packs.is_empty() {
            return;
        }
        loop {
            let len_before = pack_set.packs.len();
            pack_set.try_merge_packs();
            if pack_set.packs.len() == len_before {
                break;
            }
        }
    }
}

// ── VectorLoop: unroll_loop_iterations ─────────────────────────────────

impl VectorLoop {
    /// vector.py:273-344: unroll_loop_iterations — unroll the loop body
    /// `count` times with proper renaming.
    ///
    /// `align_unroll_once` (vector.py:273) requests one extra alignment
    /// unroll. When set: `count` is bumped by one before the unroll runs;
    /// after the first iteration a fresh `LABEL` is materialised and
    /// installed as `self.label`; the *original* body is moved to
    /// `self.align_operations` (consumed before the unrolled loop) while
    /// `self.operations` is replaced by the unrolled sequence. When unset,
    /// `self.operations` becomes `original + unrolled` (the default body
    /// shape).
    pub fn unroll_loop_iterations(&mut self, count: usize, align_unroll_once: bool) {
        if count == 0 {
            return;
        }
        // vector.py:284 — bump count once for the alignment pass.
        let unroll_count = if align_unroll_once { count + 1 } else { count };
        let original_body = self.operations.clone();
        let label_args = self.label.getarglist_copy();
        let jump_args = self.jump.getarglist_copy();

        // vector.py:281-283: prohibited opcodes — not duplicated during unroll
        let prohibit = [
            OpCode::GuardFutureCondition,
            OpCode::GuardNotInvalidated,
            OpCode::DebugMergePoint,
        ];

        let mut renamer = Renamer::new();
        let mut unrolled: Vec<OpRc> = Vec::new();
        // vector.py's renamer maps box→box; pyre's args are integer positions,
        // so this side map recovers the producer box for a renamed position:
        // each copied op's NEW position → the `OpRc` actually pushed to
        // `unrolled`. SSA guarantees a producer is pushed before its
        // consumers, so a renamed arg resolves to the canonical producer box
        // (`Operand::from_bound_op`, no mint). A miss (label/inputarg/outer
        // position with no producer in this buffer) stays `from_opref`.
        let mut produced: VecAssoc<OpRef, OpRc> = VecAssoc::new();
        // Recover the producer box for a renamed position. A hit in `produced`
        // (a copied op pushed to `unrolled`) or `original_body` (a
        // first-iteration loop-carried producer) binds to that exact `OpRc`
        // (`Operand::from_bound_op`, no mint). A miss is an inputarg / outer
        // position with no producer in this buffer; it binds to a
        // renamer-rooted producer box carrying the same `pos`
        // (`Renamer::bound_box`), never a position-only `Operand::Box`.
        // A nested `fn` (not a closure) so it can take `&mut renamer` without
        // capturing it, leaving `renamer.rename_box` / `start_renaming` free.
        fn bind_unroll(
            produced: &VecAssoc<OpRef, OpRc>,
            original_body: &[OpRc],
            renamer: &mut Renamer,
            renamed: OpRef,
        ) -> BoxRef {
            if let Some(rc) = produced.get(&renamed) {
                return BoxRef::from_bound_op(rc);
            }
            if !renamed.is_constant() && !renamed.is_none() {
                if let Some(rc) = original_body.iter().find(|op| op.pos.get() == renamed) {
                    return BoxRef::from_bound_op(rc);
                }
            }
            renamer.bound_box(renamed)
        }
        // vector.py:292 `new_label = loop.label` — the label install-target
        // is the existing label by default; the align-unroll arm overwrites
        // it with a freshly minted LABEL after the first body copy.
        let mut new_label = self.label.clone();

        let base_offset = original_body
            .iter()
            .map(|op| op.pos.get().raw())
            .max()
            .unwrap_or(0)
            + 1;

        for u in 0..unroll_count {
            // vector.py:296-301: fill rename map: label args → jump args
            for i in 0..label_args.len().min(jump_args.len()) {
                let la = label_args[i].to_opref();
                let ja = renamer.rename_box(jump_args[i].to_opref());
                if la != ja {
                    renamer.start_renaming(la, ja);
                }
            }

            let offset = base_offset + (u as u32) * (original_body.len() as u32);

            // vector.py:303-322: copy and rename each op
            for op in &original_body {
                if prohibit.contains(&op.opcode) {
                    continue;
                }
                let mut copied_op = copy_resop(op);

                // vector.py:307-310: new result box → rename mapping
                let new_pos = op.pos.get().with_raw(op.pos.get().raw() + offset);
                if !op.pos.get().is_none() {
                    renamer.start_renaming(op.pos.get(), new_pos);
                }
                copied_op.pos.set(new_pos);

                // vector.py:312-315: rename args
                for i in 0..copied_op.num_args() {
                    let renamed = renamer.rename_box(copied_op.arg(i).to_opref());
                    copied_op.setarg(
                        i,
                        Operand::from_boxref(&bind_unroll(
                            &produced,
                            &original_body,
                            &mut renamer,
                            renamed,
                        )),
                    );
                }

                // vector.py:319-320: rename guard fail args
                if copied_op.opcode.is_guard() {
                    VectorizingOptimizer::copy_guard_descr(&renamer, &mut copied_op);
                }

                // The copied op becomes the canonical producer box for its
                // (renamed) result position; register it before pushing so
                // later ops in this body bind to it.
                let rc: OpRc = std::rc::Rc::new(copied_op);
                if !new_pos.is_none() {
                    produced.insert(new_pos, std::rc::Rc::clone(&rc));
                }
                unrolled.push(rc);
            }

            // vector.py:324-328 — after the first iteration of an align
            // unroll, mint a fresh LABEL using the same descr and arglist
            // as the original label, then run the renamer over it so its
            // args track the rename state at this point.
            if align_unroll_once && u == 0 {
                let label_args_ops: Vec<Operand> =
                    label_args.iter().map(Operand::from_boxref).collect();
                let mut minted = Op::new(OpCode::Label, &label_args_ops);
                if let Some(descr) = self.label.getdescr() {
                    minted.setdescr(descr);
                }
                for i in 0..minted.num_args() {
                    let renamed = renamer.rename_box(minted.arg(i).to_opref());
                    minted.setarg(
                        i,
                        Operand::from_boxref(&bind_unroll(
                            &produced,
                            &original_body,
                            &mut renamer,
                            renamed,
                        )),
                    );
                }
                new_label = minted;
            }
        }

        // vector.py:334-337: update jump args with final renaming
        for i in 0..self.jump.num_args() {
            let renamed = renamer.rename_box(self.jump.arg(i).to_opref());
            self.jump.setarg(
                i,
                Operand::from_boxref(&bind_unroll(
                    &produced,
                    &original_body,
                    &mut renamer,
                    renamed,
                )),
            );
        }

        // vector.py:339-344
        self.label = new_label;
        if align_unroll_once {
            self.align_operations = original_body;
            self.operations = unrolled;
        } else {
            self.operations.extend(unrolled);
        }
    }
}

// ── schedule.py helpers used by the vectorizer ─────────────────────────
// These functions are from schedule.py in RPython, not vector.py.
// They are placed here because they are called from the vectorizer's
// scheduling logic in try_vectorize / run_optimization.

/// schedule.py:638-658: pre_emit_guard_accum — guard accumulation stitching.
/// For guard ops, scan failargs for accumulation variables. When found:
///   - attach AccumInfo to the guard descriptor (schedule.py:654-655)
///   - replace the failarg with the renamed seed (schedule.py:656-657)
fn pre_emit_guard_accum(state: &VecScheduleState, op: &mut Op) {
    if !op.opcode.is_guard() {
        return;
    }
    if let Some(fa) = op.getfailargs() {
        let mut new_fa = fa.clone();
        for (fi, arg) in new_fa.iter_mut().enumerate() {
            if arg.is_none() {
                continue;
            }
            if let Some(entry) = state.accumulation.get(&arg.to_opref()) {
                let location = state
                    .getvector_of_box(arg.to_opref())
                    .map(|(_, vec_ref)| vec_ref)
                    .unwrap_or(arg.to_opref());
                if let Some(descr) = op.getdescr() {
                    if let Some(fail_descr) = descr.as_fail_descr() {
                        fail_descr.attach_vector_info(majit_ir::AccumInfo {
                            prev: None,
                            failargs_pos: fi,
                            variable: arg.to_opref(),
                            location,
                            accum_operation: entry.operator,
                            scalar: OpRef::NONE,
                        });
                    }
                }
                *arg = BoxRef::from_opref(entry.seed);
            }
        }
        op.setfailargs(new_fa);
    }
}

/// schedule.py:697-736: ensure_args_unpacked — unpack vector-boxed args
/// for a scalar op, respecting seen/invariant/accumulation state.
// TODO(parity): schedule.py:697 ensure_args_unpacked is a method on
// VecScheduleState. Kept as a free `pub(crate)` fn so both the inline
// scheduling loops here and VecScheduleState::post_schedule (schedule.rs)
// can call it; promote to a method when the call sites are unified.
pub(crate) fn ensure_args_unpacked(
    state: &mut VecScheduleState,
    op: &mut Op,
    seen: &mut VecSet<OpRef>,
) {
    // schedule.py:702-706: unpack immediate-use args
    for j in 0..op.num_args() {
        let arg = op.arg(j).to_opref();
        if arg.is_constant() || seen.contains(&arg) {
            continue;
        }
        if let Some((pos, vec_ref)) = state.getvector_of_box(arg) {
            if state.invariant_vector_vars.contains(&vec_ref) {
                continue;
            }
            if state.accumulation.contains_key(&arg) {
                continue;
            }
            let unpacked = unpack_from_vector(state, vec_ref, pos, 1);
            state.renamer.start_renaming(arg, unpacked);
            seen.insert(unpacked);
            // The VecUnpack producer was just appended to `oplist`; bind the
            // arg to it (no position-only mint).
            op.setarg(j, Operand::from_boxref(&state.bound_arg_boxref(unpacked)));
        }
    }
    // schedule.py:708-716: unpack guard failargs
    if op.opcode.is_guard() {
        if let Some(mut fail_args) = op.getfailargs() {
            for arg in fail_args.iter_mut() {
                if arg.is_constant() || seen.contains(&arg.to_opref()) {
                    continue;
                }
                if let Some((pos, vec_ref)) = state.getvector_of_box(arg.to_opref()) {
                    if state.accumulation.contains_key(&arg.to_opref()) {
                        continue;
                    }
                    let unpacked = unpack_from_vector(state, vec_ref, pos, 1);
                    state.renamer.start_renaming(arg.to_opref(), unpacked);
                    seen.insert(unpacked);
                    *arg = state.bound_arg_boxref(unpacked);
                }
            }
            op.setfailargs(fail_args);
        }
    }
}

// ── Optimization trait impl (TODO) ──────────────────
// In RPython, VectorizingOptimizer extends Optimizer and is called via
// optimize_vector(). In Rust, it participates in the Optimizer pipeline
// as an Optimization sub-pass. This impl bridges the two worlds.

impl Default for VectorizingOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for VectorizingOptimizer {
    fn propagate_forward(
        &mut self,
        op: &Op,
        _op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        match op.opcode {
            OpCode::Label => {
                self.in_loop = true;
                self.label_args = op.getarglist().iter().map(|a| a.to_opref()).collect();
                // Hold the LABEL instead of emitting it: post_schedule may need to
                // place prefix ops (and a prefix_label) BEFORE the loop entry, which
                // is impossible once the label is in the stream. The held label is
                // re-emitted (or replaced by a prefix_label) at the JUMP via
                // finaloplist. majit-only: RPython never streams the label.
                self.pending_label = Some(op.clone());
                OptimizationResult::Remove
            }
            OpCode::Jump if self.in_loop => {
                // vector.py:139
                self.opt_vectorize_try_emitted = self.opt_vectorize_try_emitted.saturating_add(1);
                self.in_loop = false;
                let body = std::mem::take(&mut self.body_ops);
                let Some(label) = self.pending_label.take() else {
                    // Defensive: a Jump while in_loop should always follow a held
                    // Label. If not, emit the body verbatim and pass the jump.
                    for body_op in &body {
                        ctx.emit(body_op.clone());
                    }
                    return OptimizationResult::Emit(op.clone());
                };
                // Pristine copies for the non-vectorized restore path. Mirrors
                // optimize_vector's `version = info.snapshot(loop)` + restore on
                // bail (vector.py:134,158,163). NOT clone_loop() — that renames
                // boxes; the snapshot must keep the original op identities so
                // post-loop references stay valid.
                let orig_label = label.clone();
                let orig_body = body.clone();
                let mut loop_ = VectorLoop::new(label, body, op.clone());
                if let Some(vectorized) = self.try_vectorize(ctx, &mut loop_) {
                    // vector.py:146
                    self.opt_vectorized_emitted = self.opt_vectorized_emitted.saturating_add(1);
                    // `vectorized` is the full finaloplist — it already includes
                    // the label (or prefix_label) and the (possibly rewritten)
                    // jump, so do NOT also emit the original label/jump.
                    for vop in vectorized {
                        ctx.emit(vop);
                    }
                    OptimizationResult::Remove
                } else {
                    // Not vectorized / unprofitable: restore the original loop —
                    // label, body, jump — exactly as it arrived.
                    ctx.emit(orig_label);
                    for body_op in &orig_body {
                        ctx.emit(body_op.clone());
                    }
                    OptimizationResult::Emit(op.clone())
                }
            }
            _ => {
                if self.in_loop {
                    // A non-Jump final op (e.g. Finish) ends the region without a
                    // Jump, so this Label..terminator span is not a vectorizable
                    // loop. Flush the held LABEL + buffered body verbatim and emit
                    // the terminator; otherwise they would sit buffered with no
                    // Jump to flush them and get wiped on the next setup(), dropping
                    // the trace tail. (A Jump while in_loop is handled by the arm
                    // above, so `is_final` here means a non-Jump terminator.)
                    if op.opcode.is_final() {
                        self.in_loop = false;
                        if let Some(label) = self.pending_label.take() {
                            ctx.emit(label);
                        }
                        for body_op in std::mem::take(&mut self.body_ops) {
                            ctx.emit(body_op);
                        }
                        OptimizationResult::Emit(op.clone())
                    } else {
                        self.body_ops.push(op.clone());
                        OptimizationResult::Remove
                    }
                } else {
                    OptimizationResult::PassOn
                }
            }
        }
    }

    fn setup(&mut self) {
        self.body_ops.clear();
        self.in_loop = false;
        self.pending_label = None;
        self.packset = None;
        self.unroll_count = 0;
        self.smallest_type_bytes = 0;
        self.orig_label_args = None;
    }

    fn name(&self) -> &'static str {
        "vectorize_simd"
    }

    fn drain_profiler_counters(&mut self, profiler: &crate::jitprof::JitProfiler) {
        profiler.count(
            crate::pyjitpl::counters::OPT_VECTORIZE_TRY,
            self.opt_vectorize_try_emitted,
        );
        profiler.count(
            crate::pyjitpl::counters::OPT_VECTORIZED,
            self.opt_vectorized_emitted,
        );
        self.opt_vectorize_try_emitted = 0;
        self.opt_vectorized_emitted = 0;
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{Op, OpCode, OpRef, Type};

    /// oparser-faithful op-arg / fail-arg box for the position-keyed VectorLoop
    /// fixtures (`rpython/jit/tool/oparser.py`): an `OpRef` that names a producer
    /// position becomes a rooted bound `BoxRef` (`Operand::Op` / `Operand::InputArg`)
    /// whose `to_opref()` is byte-identical to the original `OpRef`, so the
    /// `assign_positions` / `to_opref`-keyed assertions are unchanged; constants and
    /// `None` shed to `Operand::Const` / none as before. Replaces the position-only
    /// `BoxRef::from_opref` that minted `Operand::Box` at `Op::new`.
    fn bx(r: OpRef) -> Operand {
        use crate::r#box::test_support::{rooted_inputarg_box, rooted_resop_box};
        Operand::from_boxref(&match r {
            OpRef::InputArgInt(n) => rooted_inputarg_box(Type::Int, n),
            OpRef::InputArgFloat(n) => rooted_inputarg_box(Type::Float, n),
            OpRef::InputArgRef(n) => rooted_inputarg_box(Type::Ref, n),
            OpRef::IntOp(n) => rooted_resop_box(Type::Int, n),
            OpRef::FloatOp(n) => rooted_resop_box(Type::Float, n),
            OpRef::RefOp(n) => rooted_resop_box(Type::Ref, n),
            OpRef::VoidOp(n) => rooted_resop_box(Type::Void, n),
            // Const* / None shed to Operand::Const / none — no Operand::Box mint.
            _ => BoxRef::from_opref(r),
        })
    }

    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos
                .set(OpRef::op_typed(base + i as u32, op.result_type()));
        }
    }

    // ── VectorLoop tests ──

    #[test]
    fn test_vector_loop_from_trace() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(OpCode::Label, &[bx(OpRef::input_arg_int(100))]),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntMul,
                &[bx(OpRef::int_op(2)), bx(OpRef::input_arg_int(102))],
            ),
            Op::new(OpCode::Jump, &[bx(OpRef::int_op(3))]),
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos.set(OpRef::op_typed(i as u32, op.result_type()));
        }
        let vloop = VectorLoop::from_trace(&ops).unwrap();
        assert_eq!(vloop.body_len(), 2); // IntAdd + IntMul
        assert_eq!(vloop.label.opcode, OpCode::Label);
        assert_eq!(vloop.jump.opcode, OpCode::Jump);
        assert_eq!(vloop.inputargs.len(), 1);
    }

    #[test]
    fn test_vector_loop_new() {
        let label = Op::new(OpCode::Label, &[bx(OpRef::input_arg_int(100))]);
        let ops = vec![Op::new(
            OpCode::IntAdd,
            &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
        )];
        let jump = Op::new(OpCode::Jump, &[bx(OpRef::int_op(0))]);
        let vloop = VectorLoop::new(label, ops, jump);
        assert_eq!(vloop.body_len(), 1);
        assert_eq!(vloop.inputargs, vec![OpRef::input_arg_int(100)]);
        assert!(vloop.prefix.is_empty());
        assert!(vloop.prefix_label.is_none());
        assert!(vloop.align_operations.is_empty());
    }

    #[test]
    fn test_vector_loop_finaloplist() {
        let label = Op::new(OpCode::Label, &[bx(OpRef::input_arg_int(100))]);
        let ops = vec![Op::new(
            OpCode::IntAdd,
            &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
        )];
        let jump = Op::new(OpCode::Jump, &[bx(OpRef::int_op(0))]);
        let vloop = VectorLoop::new(label, ops, jump);

        let mut state = VecScheduleState::new(0);
        let with_label = vloop.finaloplist(None, true, true, &mut state);
        assert_eq!(with_label.len(), 3); // Label + IntAdd + Jump
        assert_eq!(with_label[0].opcode, OpCode::Label);

        let without_label = vloop.finaloplist(None, true, false, &mut state);
        assert_eq!(without_label.len(), 2); // IntAdd + Jump
    }

    // ── post_schedule tests (schedule.py:762-779) ──

    /// schedule.py:762-779: invariant ops are routed into loop.prefix and the
    /// invariant vector var is appended to a fresh prefix_label and jump.
    #[test]
    fn test_post_schedule_routes_invariants_to_prefix() {
        use majit_ir::vec_set::VecSet;

        // loop: label(i0, i1) { i_add } jump(i0, i1)
        let label = Op::new(
            OpCode::Label,
            &[bx(OpRef::input_arg_int(0)), bx(OpRef::input_arg_int(1))],
        );
        let mut body = vec![Op::new(
            OpCode::IntAdd,
            &[bx(OpRef::input_arg_int(0)), bx(OpRef::input_arg_int(1))],
        )];
        assign_positions(&mut body, 10);
        let jump = Op::new(
            OpCode::Jump,
            &[bx(OpRef::input_arg_int(0)), bx(OpRef::input_arg_int(1))],
        );
        let mut vloop = VectorLoop::new(label, body.clone(), jump);

        let mut st = VecScheduleState::new(100);
        // Simulate accumulate_prepare (vector.rs run_optimization / try_vectorize):
        // three invariant ops — zero vector, xor-zero, pack seed into lane 0.
        let vc = st.create_vec_op(OpCode::VecI, &[], 'i', 8, true, 2);
        let vc_ref = vc.pos.get();
        st.invariant_oplist.push(std::rc::Rc::new(vc));
        let xor = st.create_vec_op(OpCode::VecIntXor, &[vc_ref, vc_ref], 'i', 8, true, 2);
        st.invariant_oplist.push(std::rc::Rc::new(xor));
        let pack = st.create_vec_op(
            OpCode::VecPackI,
            &[
                vc_ref,
                OpRef::input_arg_int(0),
                OpRef::const_int(0),
                OpRef::const_int(1),
            ],
            'i',
            8,
            true,
            2,
        );
        let seed_vec = pack.pos.get();
        st.invariant_oplist.push(std::rc::Rc::new(pack));
        // expand() (schedule.py:554-555) registers the splat vector here.
        st.invariant_vector_vars.insert(seed_vec);

        // The scheduled body lives in oplist; the base post_schedule
        // (schedule.py:116) moves it into loop_.operations.
        st.oplist = body.iter().cloned().map(std::rc::Rc::new).collect();

        let mut seen: VecSet<OpRef> = vloop
            .label
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect();
        st.post_schedule(&mut vloop, &mut seen);

        // schedule.py:766: prefix == the three invariant ops, in insertion order.
        assert_eq!(vloop.prefix.len(), 3);
        assert_eq!(vloop.prefix[0].opcode, OpCode::VecI);
        assert_eq!(vloop.prefix[1].opcode, OpCode::VecIntXor);
        assert_eq!(vloop.prefix[2].opcode, OpCode::VecPackI);
        assert!(st.invariant_oplist.is_empty()); // drained into prefix

        // schedule.py:773: prefix_label carries label args + the invariant var.
        let pl_args = vloop
            .prefix_label
            .as_ref()
            .expect("prefix_label must be set")
            .getarglist_copy();
        assert_eq!(vloop.prefix_label.as_ref().unwrap().opcode, OpCode::Label);
        assert_eq!(pl_args.len(), 3); // i0, i1, seed_vec
        assert_eq!(pl_args[2].to_opref(), seed_vec);

        // schedule.py:779: jump rebuilt with the extra invariant var.
        let j_args = vloop.jump.getarglist_copy();
        assert_eq!(vloop.jump.opcode, OpCode::Jump);
        assert_eq!(j_args.len(), 3);
        assert_eq!(j_args[2].to_opref(), seed_vec);

        // base post_schedule (schedule.py:116): operations came from oplist.
        assert_eq!(vloop.operations.len(), 1);
        assert_eq!(vloop.operations[0].opcode, OpCode::IntAdd);
    }

    /// schedule.py:767 false branch: with no invariants, post_schedule leaves
    /// prefix/prefix_label empty and the jump arglist unchanged.
    #[test]
    fn test_post_schedule_no_invariants_leaves_label_and_jump() {
        use majit_ir::vec_set::VecSet;

        let label = Op::new(
            OpCode::Label,
            &[bx(OpRef::input_arg_int(0)), bx(OpRef::input_arg_int(1))],
        );
        let body = vec![Op::new(
            OpCode::IntAdd,
            &[bx(OpRef::input_arg_int(0)), bx(OpRef::input_arg_int(1))],
        )];
        let jump = Op::new(
            OpCode::Jump,
            &[bx(OpRef::input_arg_int(0)), bx(OpRef::input_arg_int(1))],
        );
        let mut vloop = VectorLoop::new(label, body.clone(), jump);

        let mut st = VecScheduleState::new(100);
        st.oplist = body.iter().cloned().map(std::rc::Rc::new).collect();
        let mut seen: VecSet<OpRef> = vloop
            .label
            .getarglist()
            .iter()
            .map(|a| a.to_opref())
            .collect();
        st.post_schedule(&mut vloop, &mut seen);

        assert!(vloop.prefix.is_empty());
        assert!(vloop.prefix_label.is_none());
        assert_eq!(vloop.jump.getarglist_copy().len(), 2);
        assert_eq!(vloop.operations.len(), 1);
    }

    /// `run_optimization`'s standalone resolver has no Optimizer context, so
    /// an `INT_SIGNEXT(x, ConstInt(n))` must take arg1's bytesize from the
    /// inline `OpRef::ConstInt` (resoperation.py:181 reads `arg1.value`).
    /// Regression: the resolver previously returned `None` for every opref,
    /// so `int_signext_vecinfo`'s fail-fast `.expect()` panicked on a valid
    /// INT_SIGNEXT.
    #[test]
    fn int_signext_setup_resolves_inline_const_in_standalone_pass() {
        let label = Op::new(OpCode::Label, &[bx(OpRef::input_arg_int(0))]);
        let signext = Op::new(
            OpCode::IntSignext,
            &[bx(OpRef::input_arg_int(0)), bx(OpRef::const_int(4))],
        );
        let jump = Op::new(OpCode::Jump, &[bx(OpRef::input_arg_int(0))]);
        let mut body = vec![signext];
        assign_positions(&mut body, 10);
        let vloop = VectorLoop::new(label, body, jump);

        let mut st = VecScheduleState::new(100);
        // The standalone run_optimization resolver: inline consts only.
        let constant_of = |opref: OpRef| -> Option<i64> { opref.as_const_int() };
        vloop.setup_vectorization(&mut st, &constant_of);

        let info = st.forwarded_vecinfo(&vloop.operations[0]);
        assert_eq!(info.datatype, 'i');
        assert_eq!(info.bytesize, 4);
        assert!(info.signed);
    }

    /// Streaming refactor: a 4-op loop runs through the VectorizingOptimizer
    /// pass (which now holds the LABEL until the JUMP, builds a VectorLoop, and
    /// emits finaloplist). The result must be a single coherent loop — exactly
    /// one Label and one Jump — whether or not vectorization fires.
    #[test]
    fn test_vectorize_pass_four_ops_single_loop() {
        use crate::optimizeopt::optimizer::Optimizer;

        let mut ops = vec![
            Op::new(
                OpCode::Label,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(102)), bx(OpRef::input_arg_int(103))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(104)), bx(OpRef::input_arg_int(105))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(106)), bx(OpRef::input_arg_int(107))],
            ),
            Op::new(OpCode::Jump, &[bx(OpRef::input_arg_int(100))]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(VectorizingOptimizer::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        let labels = result
            .iter()
            .filter(|op| op.opcode == OpCode::Label)
            .count();
        let jumps = result.iter().filter(|op| op.opcode == OpCode::Jump).count();
        assert_eq!(labels, 1, "exactly one loop entry label");
        assert_eq!(jumps, 1, "exactly one jump");
        // The (possibly vectorized) body sits between the label and the jump.
        let label_pos = result.iter().position(|op| op.opcode == OpCode::Label);
        let jump_pos = result.iter().rposition(|op| op.opcode == OpCode::Jump);
        assert!(matches!((label_pos, jump_pos), (Some(l), Some(j)) if l < j));
    }

    /// schedule.py:765 + 718-732: a packed scalar result that is loop-carried by
    /// the `Jump` must be unpacked. `turn_into_vector` records the packed member
    /// only in `box_to_vbox`; PyPy's pack branch (`mark_emitted(node,
    /// unpack=False)`) never adds it to `seen`. post_schedule's
    /// `ensure_args_unpacked(jump)` therefore finds it absent from `seen`, emits
    /// a `VecUnpack`, and rewrites the Jump arg.
    ///
    /// Regression guard for the scheduling-loop `seen.insert(member)` divergence:
    /// the second half asserts that *if* the member is in `seen`, the unpack is
    /// (wrongly) skipped — so seeding `seen` with packed members is exactly what
    /// would leave the Jump referencing the folded scalar.
    #[test]
    fn test_post_schedule_unpacks_packed_member_carried_to_jump() {
        use majit_ir::vec_set::VecSet;

        fn run(member_in_seen: bool) -> (OpRef, bool) {
            let member_ref = OpRef::int_op(7); // scalar result folded into a pack
            let label = Op::new(OpCode::Label, &[bx(OpRef::input_arg_int(0))]);
            let jump = Op::new(OpCode::Jump, &[bx(member_ref)]);
            let mut vloop = VectorLoop::new(label, Vec::new(), jump);

            let mut st = VecScheduleState::new(100);
            // turn_into_vector emits the vector op into the oplist and maps each
            // packed scalar to a lane via setvector_of_box.
            let vecop = st.create_vec_op(OpCode::VecIntAdd, &[], 'i', 8, true, 2);
            let vec_ref = vecop.pos.get();
            st.oplist.push(std::rc::Rc::new(vecop));
            st.setvector_of_box(member_ref, 0, vec_ref);

            // seen seeded as the scheduling loop leaves it: always the label
            // args; the packed member only when reproducing the buggy path.
            let mut seen: VecSet<OpRef> = vloop
                .label
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect();
            if member_in_seen {
                seen.insert(member_ref);
            }
            st.post_schedule(&mut vloop, &mut seen);

            let jump_arg0 = vloop.jump.getarglist_copy()[0].to_opref();
            let has_unpack = vloop
                .operations
                .iter()
                .any(|op| matches!(op.opcode, OpCode::VecUnpackI | OpCode::VecUnpackF));
            (jump_arg0, has_unpack)
        }

        // Correct path (packed member absent from seen): the Jump arg is
        // unpacked away from the stale scalar and a VecUnpack is materialized.
        let (jump_arg0, has_unpack) = run(false);
        assert_ne!(
            jump_arg0,
            OpRef::int_op(7),
            "Jump must not reference the scalar folded into the vector pack"
        );
        assert!(
            has_unpack,
            "ensure_args_unpacked must emit a VecUnpack for the packed member"
        );

        // Buggy path (packed member present in seen): the unpack is skipped and
        // the Jump still references the folded scalar — the exact failure the
        // pack-branch `seen.insert(member)` divergence produced.
        let (stale_arg0, no_unpack) = run(true);
        assert_eq!(
            stale_arg0,
            OpRef::int_op(7),
            "seeding seen with the packed member suppresses the unpack"
        );
        assert!(
            !no_unpack,
            "no VecUnpack is emitted when the packed member is wrongly in seen"
        );
    }

    /// vector.py:134/160: `optimize_vector` registers exactly one LoopVersion
    /// snapshot before running the pipeline — so GuardStrengthenOpt's
    /// `versions.len() == 1` assert holds when the gso step is reached — and
    /// restores the caller's loop on a bail. A scalar (non-array) loop bails at
    /// the `byte_count == 0` gate; the snapshot still runs and loop_ is left
    /// at its pre-vectorize shape.
    #[test]
    fn test_optimize_vector_snapshots_single_version_and_restores_on_bail() {
        let label = Op::new(OpCode::Label, &[bx(OpRef::input_arg_int(0))]);
        let body = vec![Op::new(
            OpCode::IntAdd,
            &[bx(OpRef::input_arg_int(0)), bx(OpRef::input_arg_int(1))],
        )];
        let jump = Op::new(OpCode::Jump, &[bx(OpRef::int_op(0))]);
        let mut vloop = VectorLoop::new(label, body, jump);
        let before_len = vloop.operations.len();

        let mut info = crate::optimizeopt::version::LoopVersionInfo::new();
        let result = optimize_vector(&mut vloop, 0, 16, &mut info, false);

        // A scalar loop has no array access → byte_count == 0 → bail.
        assert!(result.is_err(), "scalar loop must bail (not vectorizeable)");
        // vector.py:134 snapshot ran before the bail: exactly one tracked
        // version, which is what makes the gso assert reachable.
        assert_eq!(
            info.versions.len(),
            1,
            "optimize_vector must snapshot exactly one LoopVersion"
        );
        assert_eq!(
            info.versions[0].ops.len(),
            before_len,
            "snapshot must capture the pre-vectorize body"
        );
        // vector.py:160: loop_ restored to its pre-vectorize shape on bail.
        assert_eq!(
            vloop.operations.len(),
            before_len,
            "loop_ must be restored on bail"
        );
    }

    /// End-to-end SIMD fixture: a loop with two adjacent 8-byte raw loads that
    /// the vectorizer packs, schedules profitably, and carries through
    /// `post_schedule` into GuardStrengthenOpt. First fixture that drives the
    /// standalone `optimize_vector` pipeline all the way to the gso step
    /// (vector.py:259), exercising the increment-1 wiring end to end.
    #[test]
    fn test_optimize_vector_packs_adjacent_loads_through_gso() {
        use majit_ir::{Type, make_array_descr};

        let i = OpRef::input_arg_int(0); // index base
        let base1 = OpRef::input_arg_int(1); // src1 pointer
        let base2 = OpRef::input_arg_int(2); // src2 pointer
        let descr = make_array_descr(0, 8, Type::Int); // 8-byte int array

        // dst[i] = src1[i] + src2[i], pre-unrolled to elements i and i+8 so two
        // adjacent loads per array pack and the two sums pair:
        //  0: Label [i, base1, base2]
        //  1: a0 = RawLoadI [base1, i]        (mref var=i, const=0)
        //  2: i2 = IntAdd   [i, ConstInt(8)]
        //  3: a1 = RawLoadI [base1, i2]       (adjacent to a0)
        //  4: b0 = RawLoadI [base2, i]
        //  5: b1 = RawLoadI [base2, i2]       (adjacent to b0)
        //  6: s0 = IntAdd   [a0, b0]
        //  7: s1 = IntAdd   [a1, b1]          (pairs with s0 via follow_def_uses)
        //  8: Jump [i, s0, s1]                (carry sums so they live)
        let mut all = vec![
            Op::new(OpCode::Label, &[bx(i), bx(base1), bx(base2)]),
            Op::with_descr(OpCode::RawLoadI, &[bx(base1), bx(i)], descr.clone()),
            Op::new(OpCode::IntAdd, &[bx(i), bx(OpRef::const_int(8))]),
            Op::with_descr(
                OpCode::RawLoadI,
                &[bx(base1), bx(OpRef::int_op(2))],
                descr.clone(),
            ),
            Op::with_descr(OpCode::RawLoadI, &[bx(base2), bx(i)], descr.clone()),
            Op::with_descr(
                OpCode::RawLoadI,
                &[bx(base2), bx(OpRef::int_op(2))],
                descr.clone(),
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::int_op(1)), bx(OpRef::int_op(4))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::int_op(3)), bx(OpRef::int_op(5))],
            ),
            Op::new(
                OpCode::Jump,
                &[bx(i), bx(OpRef::int_op(6)), bx(OpRef::int_op(7))],
            ),
        ];
        assign_positions(&mut all, 0);

        let label = all[0].clone();
        let jump = all[all.len() - 1].clone();
        let body: Vec<Op> = all[1..all.len() - 1].to_vec();
        let mut vloop = VectorLoop::new(label, body, jump);

        let mut info = crate::optimizeopt::version::LoopVersionInfo::new();
        // vec_size 16 (SSE), cost_threshold 0.
        let result = optimize_vector(&mut vloop, 0, 16, &mut info, false);

        // Reaching Ok proves the whole pipeline ran past the profitability gate
        // and through GuardStrengthenOpt: gso runs unconditionally between
        // post_schedule and finaloplist (vector.py:259-271), so an Ok return
        // means gso.propagate_all_forward was invoked AND its
        // `versions.len() == 1` assert held (otherwise it would panic, not Err).
        let (ops, gso_consts) = result.expect("adjacent-load loop must vectorize");

        // The pre-vectorize loop is the single tracked version (gso precondition).
        assert_eq!(info.versions.len(), 1, "exactly one snapshot version");

        // Real vectorization happened: the two adjacent loads became packed
        // VEC_LOAD ops and the paired sums a VEC_INT_ADD — which only exists
        // now that `to_vector()` maps the memory loads (resoperation.py:1746).
        assert!(
            ops.iter().any(|op| op.opcode == OpCode::VecLoadI),
            "adjacent loads must pack into VecLoadI"
        );
        assert!(
            ops.iter().any(|op| op.opcode == OpCode::VecIntAdd),
            "paired sums must pack into VecIntAdd"
        );
        // Loop structure is preserved end to end.
        assert!(ops.iter().any(|op| op.opcode == OpCode::Label));
        assert!(ops.iter().any(|op| op.opcode == OpCode::Jump));
        // gso materialized the index-var constant it strength-reduced and the
        // wiring surfaced it for the caller to register in the constant pool.
        assert!(
            !gso_consts.is_empty(),
            "gso must surface its materialized index constants"
        );
    }

    /// compile.py:302-308 hook: a loop that cannot vectorize (no array
    /// access → byte_count == 0) is returned unchanged. This is the common
    /// path on every real loop until adjacent memory refs appear, and the
    /// invariant that matters for production safety: a bail must not perturb
    /// the optimizer's assembled trace.
    #[test]
    fn test_apply_loop_vectorization_bails_keeps_scalar_loop() {
        // [start_label, prefix_op, loop_label, body_add, jump].
        let i = OpRef::input_arg_int(0);
        let j = OpRef::input_arg_int(1);
        let mut assembled = vec![
            Op::new(OpCode::Label, &[bx(i), bx(j)]),
            Op::new(OpCode::IntAdd, &[bx(i), bx(OpRef::const_int(1))]),
            Op::new(OpCode::Label, &[bx(i), bx(j)]),
            Op::new(OpCode::IntAdd, &[bx(i), bx(j)]),
            Op::new(OpCode::Jump, &[bx(OpRef::int_op(3))]),
        ];
        assign_positions(&mut assembled, 0);

        let before: Vec<(OpCode, u32)> = assembled
            .iter()
            .map(|op| (op.opcode, op.pos.get().raw()))
            .collect();
        let out = apply_loop_vectorization(assembled, 16, 0, false);
        let after: Vec<(OpCode, u32)> = out
            .iter()
            .map(|op| (op.opcode, op.pos.get().raw()))
            .collect();

        // No array access → NotAVectorizeableLoop → trace returned verbatim.
        assert_eq!(before, after, "non-vectorizable loop must pass through");
        assert!(!out.iter().any(|op| op.opcode == OpCode::VecLoadI));
    }

    /// compile.py:302-308 hook end to end: feed `apply_loop_vectorization` an
    /// optimizer-assembled loop `[start_label, prefix_op, loop_label, body…,
    /// jump]` whose body holds two adjacent array loads. The helper must
    /// split at the loop LABEL, vectorize the loop part, and re-assemble
    /// `prefix + [label] + vectorized` (compile.py:327-328) with the prefix
    /// untouched and the new VEC ops in a position namespace disjoint from
    /// the prefix.
    #[test]
    fn test_apply_loop_vectorization_splices_vectorized_loop() {
        use majit_ir::{Type, make_array_descr};

        let i = OpRef::input_arg_int(0);
        let base1 = OpRef::input_arg_int(1);
        let base2 = OpRef::input_arg_int(2);
        let descr = make_array_descr(0, 8, Type::Int);

        //  0: start_label LABEL [i, base1, base2]
        //  1: prefix_op  IntAdd [i, ConstInt(1)]     (preamble, pos 1)
        //  2: loop_label LABEL [i, base1, base2]      (last LABEL)
        //  3: a0 = RawLoadI [base1, i]
        //  4: i2 = IntAdd   [i, ConstInt(8)]
        //  5: a1 = RawLoadI [base1, int_op(4)]
        //  6: b0 = RawLoadI [base2, i]
        //  7: b1 = RawLoadI [base2, int_op(4)]
        //  8: s0 = IntAdd   [int_op(3), int_op(6)]
        //  9: s1 = IntAdd   [int_op(5), int_op(7)]
        // 10: jump JUMP [i, int_op(8), int_op(9)]
        let mut assembled = vec![
            Op::new(OpCode::Label, &[bx(i), bx(base1), bx(base2)]),
            Op::new(OpCode::IntAdd, &[bx(i), bx(OpRef::const_int(1))]),
            Op::new(OpCode::Label, &[bx(i), bx(base1), bx(base2)]),
            Op::with_descr(OpCode::RawLoadI, &[bx(base1), bx(i)], descr.clone()),
            Op::new(OpCode::IntAdd, &[bx(i), bx(OpRef::const_int(8))]),
            Op::with_descr(
                OpCode::RawLoadI,
                &[bx(base1), bx(OpRef::int_op(4))],
                descr.clone(),
            ),
            Op::with_descr(OpCode::RawLoadI, &[bx(base2), bx(i)], descr.clone()),
            Op::with_descr(
                OpCode::RawLoadI,
                &[bx(base2), bx(OpRef::int_op(4))],
                descr.clone(),
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::int_op(3)), bx(OpRef::int_op(6))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::int_op(5)), bx(OpRef::int_op(7))],
            ),
            Op::new(
                OpCode::Jump,
                &[bx(i), bx(OpRef::int_op(8)), bx(OpRef::int_op(9))],
            ),
        ];
        assign_positions(&mut assembled, 0);

        let out = apply_loop_vectorization(assembled, 16, 0, false);

        // Prefix is preserved verbatim: start_label then the pos-1 IntAdd.
        assert_eq!(out[0].opcode, OpCode::Label);
        assert_eq!(out[1].opcode, OpCode::IntAdd);
        assert_eq!(out[1].pos.get().raw(), 1, "prefix op keeps its position");
        // The loop part vectorized.
        assert!(
            out.iter().any(|op| op.opcode == OpCode::VecLoadI),
            "adjacent loads must pack into VecLoadI"
        );
        assert!(
            out.iter().any(|op| op.opcode == OpCode::VecIntAdd),
            "paired sums must pack into VecIntAdd"
        );
        // The loop LABEL survives and the trace still closes with a JUMP.
        assert!(out.iter().any(|op| op.opcode == OpCode::Label));
        assert_eq!(out.last().unwrap().opcode, OpCode::Jump);
        // Loop entry contract preserved: the loop LABEL and the closing JUMP
        // keep matching arity (3 args each).
        let loop_label = out
            .iter()
            .rev()
            .find(|op| op.opcode == OpCode::Label)
            .unwrap();
        assert_eq!(loop_label.num_args(), out.last().unwrap().num_args());
        // Position disjointness: every VEC op the vectorizer created lands
        // above the prefix's positions {0, 1}, so it cannot alias a prefix
        // value when the backend reads positions as SSA numbers.
        for op in &out {
            if matches!(op.opcode, OpCode::VecLoadI | OpCode::VecIntAdd) {
                assert!(
                    op.pos.get().raw() > 1,
                    "vectorized op position {} must clear the prefix",
                    op.pos.get().raw()
                );
            }
        }
    }

    /// Streaming refactor: a non-loop `Label .. Finish` trace (no trailing Jump)
    /// must survive. The held label + buffered body are flushed verbatim on the
    /// non-Jump terminator instead of being dropped on the next setup().
    #[test]
    fn test_vectorize_pass_label_finish_preserved() {
        use crate::optimizeopt::optimizer::Optimizer;

        let mut ops = vec![
            Op::new(OpCode::Label, &[bx(OpRef::input_arg_int(100))]),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(OpCode::Finish, &[bx(OpRef::int_op(1))]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(VectorizingOptimizer::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert!(
            result.iter().any(|op| op.opcode == OpCode::Label),
            "held label must be flushed, not dropped"
        );
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntAdd),
            "buffered body must be flushed"
        );
        assert!(
            result.iter().any(|op| op.opcode == OpCode::Finish),
            "non-Jump terminator must be emitted"
        );
    }

    #[test]
    fn test_user_loop_bail_fast_path_no_array() {
        let label = Op::new(OpCode::Label, &[bx(OpRef::input_arg_int(100))]);
        let ops = vec![Op::new(
            OpCode::IntAdd,
            &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
        )];
        let jump = Op::new(OpCode::Jump, &[bx(OpRef::int_op(0))]);
        let vloop = VectorLoop::new(label, ops, jump);
        // vector.py:183 initializes at_least_one_array_access = True and only
        // re-assigns True, so the "no array access" branch is unreachable.
        // Match upstream literal: no array access does NOT bail.
        assert!(!user_loop_bail_fast_path(&vloop));
    }

    // ── Dependency graph tests ──

    #[test]
    fn test_dep_graph_basic() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntMul,
                &[bx(OpRef::int_op(0)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntSub,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        assert_eq!(graph.nodes.len(), 3);
        assert!(graph.nodes[1].deps.contains(&0));
        assert!(!graph.nodes[2].deps.contains(&0));
    }

    #[test]
    fn test_dep_graph_no_self_dep() {
        let mut ops = vec![Op::new(
            OpCode::IntAdd,
            &[bx(OpRef::int_op(0)), bx(OpRef::input_arg_int(101))],
        )];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        assert!(graph.nodes[0].deps.is_empty());
    }

    // ── Pack group tests ──

    #[test]
    fn test_find_packable_groups() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(102)), bx(OpRef::input_arg_int(103))],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let groups = graph.find_packable_groups();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].scalar_opcode, OpCode::IntAdd);
        assert_eq!(groups[0].vector_opcode, OpCode::VecIntAdd);
        assert_eq!(groups[0].members.len(), 2);
    }

    #[test]
    fn test_dependent_ops_not_packed() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::int_op(0)), bx(OpRef::input_arg_int(101))],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let groups = graph.find_packable_groups();
        assert!(groups.is_empty());
    }

    #[test]
    fn test_different_opcodes_not_packed() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntSub,
                &[bx(OpRef::input_arg_int(102)), bx(OpRef::input_arg_int(103))],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let groups = graph.find_packable_groups();
        for g in &groups {
            assert!(g.members.len() >= 2);
        }
    }

    #[test]
    fn test_three_independent_ops() {
        let mut ops = vec![
            Op::new(
                OpCode::IntMul,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntMul,
                &[bx(OpRef::input_arg_int(102)), bx(OpRef::input_arg_int(103))],
            ),
            Op::new(
                OpCode::IntMul,
                &[bx(OpRef::input_arg_int(104)), bx(OpRef::input_arg_int(105))],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let groups = graph.find_packable_groups();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members.len(), 3);
    }

    // ── Cost model tests ──

    #[test]
    fn test_cost_model_profitable() {
        let cm = CostModel::new();
        let group = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2, 3],
            is_accumulating: false,
            position: -1,
            operator: None,
        };
        assert!(!cm.is_profitable(&group));

        let group5 = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2, 3, 4],
            is_accumulating: false,
            position: -1,
            operator: None,
        };
        assert!(!cm.is_profitable(&group5));
    }

    #[test]
    fn test_cost_model_too_small() {
        let cm = CostModel::new();
        let group = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0],
            is_accumulating: false,
            position: -1,
            operator: None,
        };
        assert!(!cm.is_profitable(&group));
    }

    #[test]
    fn test_cost_model_custom_params() {
        let cm = CostModel {
            min_pack_size: 2,
            pack_cost: 1,
            scalar_save: 2,
            savings: 0,
        };
        let group = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1],
            is_accumulating: false,
            position: -1,
            operator: None,
        };
        assert!(!cm.is_profitable(&group));

        let group3 = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2],
            is_accumulating: false,
            position: -1,
            operator: None,
        };
        assert!(cm.is_profitable(&group3));
    }

    // ── Memory access detection ──

    #[test]
    fn test_is_memory_access() {
        assert!(OpCode::GetfieldGcI.is_memory_access());
        assert!(OpCode::SetarrayitemGc.is_memory_access());
        assert!(OpCode::RawLoadI.is_memory_access());
        assert!(!OpCode::IntAdd.is_memory_access());
        assert!(!OpCode::GuardTrue.is_memory_access());
    }

    // ── VectorizingOptimizer pass tests ──

    #[test]
    fn test_vectorize_pass_no_loop() {
        use crate::optimizeopt::optimizer::Optimizer;

        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(OpCode::Finish, &[bx(OpRef::int_op(0))]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(VectorizingOptimizer::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_vectorize_pass_preserves_structure() {
        use crate::optimizeopt::optimizer::Optimizer;

        let mut ops = vec![
            Op::new(
                OpCode::Label,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntSub,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(OpCode::Jump, &[bx(OpRef::int_op(1)), bx(OpRef::int_op(2))]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(VectorizingOptimizer::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);
        assert!(result.iter().any(|op| op.opcode == OpCode::Label));
        assert!(result.iter().any(|op| op.opcode == OpCode::Jump));
    }

    // ── Scheduler tests ──

    #[test]
    fn test_schedule_respects_dependencies() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntMul,
                &[bx(OpRef::int_op(0)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntSub,
                &[bx(OpRef::int_op(1)), bx(OpRef::input_arg_int(101))],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 3);
        let pos_a = sched.iter().position(|&x| x == 0).unwrap();
        let pos_b = sched.iter().position(|&x| x == 1).unwrap();
        let pos_c = sched.iter().position(|&x| x == 2).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_schedule_maximizes_parallelism() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntSub,
                &[bx(OpRef::input_arg_int(102)), bx(OpRef::input_arg_int(103))],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 2);
        assert!(sched.contains(&0));
        assert!(sched.contains(&1));
    }

    #[test]
    fn test_schedule_prioritizes_critical_path() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntMul,
                &[bx(OpRef::int_op(0)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntSub,
                &[bx(OpRef::int_op(1)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(102)), bx(OpRef::input_arg_int(103))],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 4);
        let pos_a = sched.iter().position(|&x| x == 0).unwrap();
        let pos_d = sched.iter().position(|&x| x == 3).unwrap();
        assert!(pos_a < pos_d, "A (height 3) should precede D (height 1)");
    }

    #[test]
    fn test_schedule_diamond() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntMul,
                &[bx(OpRef::int_op(0)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntSub,
                &[bx(OpRef::int_op(0)), bx(OpRef::input_arg_int(102))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::int_op(1)), bx(OpRef::int_op(2))],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 4);
        let pos_a = sched.iter().position(|&x| x == 0).unwrap();
        let pos_b = sched.iter().position(|&x| x == 1).unwrap();
        let pos_c = sched.iter().position(|&x| x == 2).unwrap();
        let pos_d = sched.iter().position(|&x| x == 3).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn test_schedule_empty_graph() {
        let graph = DependencyGraph {
            nodes: Vec::new(),
            memory_refs: Default::default(),
            index_vars: Default::default(),
            guards: Vec::new(),
            invariant_vars: Default::default(),
        };
        let sched = schedule_operations(&graph);
        assert!(sched.is_empty());
    }

    #[test]
    fn test_pack_set_merge() {
        let mut ps = PackSet::new();
        ps.add_pack(Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1],
            is_accumulating: false,
            position: -1,
            operator: None,
        });
        ps.add_pack(Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![1, 2],
            is_accumulating: false,
            position: -1,
            operator: None,
        });
        assert_eq!(ps.num_packs(), 2);
        assert_eq!(ps.total_ops(), 4);

        ps.try_merge_packs();
        assert_eq!(ps.num_packs(), 1);
        assert_eq!(ps.total_ops(), 3);
    }

    #[test]
    fn test_pack_set_no_merge_disjoint() {
        let mut ps = PackSet::new();
        ps.add_pack(Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1],
            is_accumulating: false,
            position: -1,
            operator: None,
        });
        ps.add_pack(Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![2, 3],
            is_accumulating: false,
            position: -1,
            operator: None,
        });
        ps.try_merge_packs();
        assert_eq!(ps.num_packs(), 2);
    }

    // ── isomorphic + can_be_packed + accumulates_pair tests ──

    #[test]
    fn test_isomorphic_same_opcode() {
        let mut state = VecScheduleState::new(0);
        let a = Op::new(
            OpCode::IntAdd,
            &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
        );
        let b = Op::new(
            OpCode::IntAdd,
            &[bx(OpRef::input_arg_int(102)), bx(OpRef::input_arg_int(103))],
        );
        assert!(isomorphic(&mut state, &a, &b));
    }

    #[test]
    fn test_isomorphic_different_opcode() {
        let mut state = VecScheduleState::new(0);
        let a = Op::new(
            OpCode::IntAdd,
            &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
        );
        let b = Op::new(
            OpCode::IntSub,
            &[bx(OpRef::input_arg_int(102)), bx(OpRef::input_arg_int(103))],
        );
        assert!(!isomorphic(&mut state, &a, &b));
    }

    #[test]
    fn test_can_be_packed_independent_seed() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(102)), bx(OpRef::input_arg_int(103))],
            ),
        ];
        assign_positions(&mut ops, 0);
        let graph = DependencyGraph::build(&ops, &|_| None);
        let ps = PackSet::new();

        let result = ps.can_be_packed(&mut VecScheduleState::new(0), 0, 1, None, false, &graph);
        assert!(result.is_ok());
        let pack = result.unwrap();
        assert!(pack.is_some());
        let pack = pack.unwrap();
        assert_eq!(pack.members, vec![0, 1]);
        assert!(!pack.is_accumulating);
    }

    #[test]
    fn test_can_be_packed_dependent_no_origin() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::int_op(0)), bx(OpRef::input_arg_int(101))],
            ),
        ];
        assign_positions(&mut ops, 0);
        let graph = DependencyGraph::build(&ops, &|_| None);
        let ps = PackSet::new();

        let result = ps
            .can_be_packed(&mut VecScheduleState::new(0), 0, 1, None, false, &graph)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_can_be_packed_accumulation() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(102)), bx(OpRef::input_arg_int(103))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(200)), bx(OpRef::int_op(0))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::int_op(2)), bx(OpRef::int_op(1))],
            ),
        ];
        assign_positions(&mut ops, 0);
        let graph = DependencyGraph::build(&ops, &|_| None);

        let origin = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1],
            is_accumulating: false,
            position: -1,
            operator: None,
        };

        let ps = PackSet::new();
        let result = ps.can_be_packed(
            &mut VecScheduleState::new(0),
            2,
            3,
            Some(&origin),
            true,
            &graph,
        );
        assert!(result.is_ok());
        let pack = result.unwrap();
        if let Some(p) = pack {
            assert!(p.is_accumulating);
            assert_eq!(p.operator, Some('+'));
            assert_eq!(p.position, 0);
        }
    }

    #[test]
    fn test_can_be_packed_blocks_already_packed() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(102)), bx(OpRef::input_arg_int(103))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(104)), bx(OpRef::input_arg_int(105))],
            ),
        ];
        assign_positions(&mut ops, 0);
        let graph = DependencyGraph::build(&ops, &|_| None);

        let mut ps = PackSet::new();
        ps.add_pack(Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1],
            is_accumulating: false,
            position: -1,
            operator: None,
        });
        let result = ps
            .can_be_packed(&mut VecScheduleState::new(0), 0, 2, None, false, &graph)
            .unwrap();
        assert!(result.is_none());
        let result = ps
            .can_be_packed(&mut VecScheduleState::new(0), 2, 1, None, false, &graph)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_generic_cost_model() {
        let model = GenericCostModel::new();
        assert!(model.op_cost(OpCode::GetarrayitemGcI) > model.op_cost(OpCode::IntAdd));
        assert!(model.op_cost(OpCode::FloatTrueDiv) >= model.op_cost(OpCode::FloatAdd));
    }

    #[test]
    fn test_guard_analysis_hoistable() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[bx(OpRef::input_arg_int(100))]),
            Op::new(
                OpCode::IntAdd,
                &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
            ),
            Op::new(OpCode::GuardTrue, &[bx(OpRef::int_op(1))]),
        ];
        let mut positioned = ops;
        for (i, op) in positioned.iter_mut().enumerate() {
            op.pos.set(OpRef::op_typed(i as u32, op.result_type()));
        }
        let analysis = GuardAnalysis::analyze(&positioned);
        assert_eq!(analysis.hoistable.len(), 1);
        assert_eq!(analysis.hoistable[0], 0);
        assert_eq!(analysis.body_guards.len(), 1);
        assert_eq!(analysis.body_guards[0], 2);
    }

    #[test]
    fn drain_profiler_counters_folds_opt_vectorize_try_and_opt_vectorized_into_profiler() {
        use crate::pyjitpl::counters;
        let mut vopt = VectorizingOptimizer::new();
        vopt.opt_vectorize_try_emitted = 4;
        vopt.opt_vectorized_emitted = 1;
        let prof = crate::jitprof::JitProfiler::default();
        Optimization::drain_profiler_counters(&mut vopt, &prof);
        assert_eq!(prof.get_counter(counters::OPT_VECTORIZE_TRY), Some(4));
        assert_eq!(prof.get_counter(counters::OPT_VECTORIZED), Some(1));
        assert_eq!(vopt.opt_vectorize_try_emitted, 0);
        assert_eq!(vopt.opt_vectorized_emitted, 0);
        Optimization::drain_profiler_counters(&mut vopt, &prof);
        assert_eq!(prof.get_counter(counters::OPT_VECTORIZE_TRY), Some(4));
        assert_eq!(prof.get_counter(counters::OPT_VECTORIZED), Some(1));
    }

    // ── Unroll tests ──

    #[test]
    fn test_unroll_loop_iterations() {
        let mut label = Op::new(
            OpCode::Label,
            &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
        );
        label.pos.set(OpRef::op_typed(0, majit_ir::Type::Void));
        let mut body_op = Op::new(
            OpCode::IntAdd,
            &[bx(OpRef::input_arg_int(100)), bx(OpRef::input_arg_int(101))],
        );
        body_op.pos.set(OpRef::int_op(1));
        let mut jump = Op::new(
            OpCode::Jump,
            &[bx(OpRef::int_op(1)), bx(OpRef::input_arg_int(101))],
        );
        jump.pos.set(OpRef::op_typed(2, majit_ir::Type::Void));

        let mut vloop = VectorLoop::new(label, vec![body_op], jump);
        assert_eq!(vloop.body_len(), 1);

        vloop.unroll_loop_iterations(2, false);
        // Original 1 + 2 unrolled copies = 3
        assert_eq!(vloop.body_len(), 3);
    }

    #[test]
    fn test_get_unroll_count() {
        // 16 byte SIMD register, 4 byte elements → 3 additional unrolls
        assert_eq!(VectorizingOptimizer::get_unroll_count(4, 16), 3);
        // 16 byte SIMD register, 8 byte elements → 1 additional unroll
        assert_eq!(VectorizingOptimizer::get_unroll_count(8, 16), 1);
        // 0 byte smallest → 0
        assert_eq!(VectorizingOptimizer::get_unroll_count(0, 16), 0);
    }
}
