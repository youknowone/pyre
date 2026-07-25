//! Force-before-writeback (FBW) walker state: the mutable side channels
//! the tracer consults while deciding whether an opcode can be folded,
//! inlined, or must fall back to a residual call.
//!
//! **Parity:** pyre-specific — the FBW (full-body-walk) live tracer has no
//! `rpython/jit/metainterp/` file counterpart. PyPy's parity-faithful
//! `MIFrame` interpret loop lives in `majit-metainterp/pyjitpl.rs` and is
//! retired as the production path; this is walker-local state with no
//! upstream analogue.
//!
//! Relocated verbatim from `jitcode_dispatch/mod.rs`. Groups the
//! `PYRE_FBW_*` feature gates, the store / append / for-iter journals and
//! their rollback machinery, the executed-effect and residual counters,
//! the finish-payload channel, and the abort-resume carriers. These are
//! thread-local / walker-scoped helpers with no opcode dispatch of their
//! own; the dispatch arms in `mod.rs` call into them.

use super::*;

/// Maximum inline depth the multiframe guard-snapshot path
/// (`walker_capture_multi_frame_inline_snapshot`) unrolls a straight-line
/// value-returning callee CHAIN before folding to the `CALL_ASSEMBLER` tail.
/// Default 7: a deep value-returning chain (`b→c→…→h`) stays in compiled code
/// and each extra inlined level removes a residual call — measured ~2.0–2.3×
/// on the `depthN_inline_chain` fixtures with no regression elsewhere.
///
/// Deep unrolling is a loss for exactly two callee shapes, each capped
/// separately so the chain depth can rise freely:
///   - a self-recursive callee (tree recursion `fib`, whose `n < 2` base-case
///     guard fails per call) — its unrolled copy guard-fails and deopts to the
///     blackhole on essentially every recursive call (two orders of magnitude
///     more blackhole resumes, ~20–30× slower).  Bounded by the distinct
///     `max_unroll_recursion` limit (`fbw_max_rec_unroll_depth`, kept at 1).
///   - a callee that raises inline below an intermediate frame — its unwind
///     needs the cross-frame bridge (gh#343 / gh#467) the drain cannot yet
///     build.  Capped to the top inline level by `callee_body_contains_raise`.
/// This mirrors `max_unroll_recursion` folding a recursive call straight to
/// `CALL_ASSEMBLER` (`_opimpl_recursive_call` → `do_recursive_call`,
/// `pyjitpl.py`) past the bound rather than continuing to unroll the call tree.
/// (The depth-≥2 blackhole-resume crash that previously blocked this path was a
/// GC-rooting gap in the nested `run()` chain, fixed by rooting the whole
/// pending `nextblackholeinterp` chain across `run()`; the bound is now a
/// performance, not a soundness, question.)
pub(crate) fn fbw_max_multiframe_depth() -> usize {
    static DEPTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("PYRE_FBW_MULTIFRAME_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(7)
            .clamp(1, 7)
    })
}

/// `max_unroll_recursion`: how many levels a self-recursive callee unrolls
/// before folding the deepest self-call to `CALL_ASSEMBLER`
/// (`_opimpl_recursive_call` → `do_recursive_call`, `pyjitpl.py`).  This is a
/// bound distinct from the straight-line inline depth
/// (`fbw_max_multiframe_depth`): a value-returning callee CHAIN inlines deeply
/// with a clean win, but unrolling a self-recursive callee past one level is a
/// loss — the extra copy guard-fails / cannot materialize a residual self-call
/// argument on the exception-unwind path and deopts (measured: a depth-2 unroll
/// of `recur(depth-1, acc+check(...))` drops a bridge and runs ~13% slower).
/// Kept at 1 so that raising the chain-inline depth never deepens recursion
/// unrolling.  `PYRE_FBW_REC_UNROLL_DEPTH` overrides for A/B.
pub(crate) fn fbw_max_rec_unroll_depth() -> usize {
    static DEPTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("PYRE_FBW_REC_UNROLL_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 7)
    })
}

/// Recursion depth of `w_code` on the walk's framestack.
pub(crate) fn fbw_inline_recursion_count<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    w_code: usize,
) -> usize {
    ctx.session
        .borrow()
        .framestack
        .iter()
        .filter(|frame| frame.w_code == w_code)
        .count()
}

/// The innermost inline level's strict-fold frame register (`u16::MAX` when
/// inactive / no inline level).
pub(crate) fn fbw_strict_fold_frame_reg<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>) -> u16 {
    ctx.callee_shadow
        .as_ref()
        .map_or(u16::MAX, |shadow| shadow.fold_frame_reg)
}

/// `PYRE_FBW_VABLE_SCALAR_CA` (default OFF) — sub-mode of the loop-callee
/// CALL_ASSEMBLER passes the callee's loop-carried locals as scalar
/// CALL_ASSEMBLER args plus a `VableExpansion` (`arg_overrides` mapping each
/// scalar to a callee jitframe slot), so the optimizer can elide the per-call
/// frame-array build (`NewArrayClear` + per-element `SetarrayitemGc`) instead
/// of forcing the virtual frame. Mirrors `direct_assembler_call`
/// (`pyjitpl.py`, raw red boxes) + `handle_call_assembler`
/// (`rewrite.py`, GC_STORE scalars into the callee jitframe). Default OFF
/// until the callee scalar contract + optimizer array-elision land and the
/// path is verified fib-safe on both backends.
pub(crate) fn fbw_vable_scalar_ca_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_VABLE_SCALAR_CA") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => false,
    })
}

/// `PYRE_FBW_CALLEE_VSTACK` (default OFF) — maintain a callee-local
/// operand-stack mirror while walking an inline sub-call.  The callee enters
/// with an empty operand stack; subsequent boundaries must use the active
/// callee jitcode metadata rather than the outer full-body tables.
pub(crate) fn fbw_callee_vstack_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_CALLEE_VSTACK") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => false,
    })
}

thread_local! {
    /// Finish payload stashed by a top-level `*_return` arm, read back by
    /// [`crate::trace::full_body_walk_trace`] to build a
    /// `TraceAction::Finish` for a loop-free (Finish-terminated) portal.
    ///
    /// `(finish_value, finish_arg_type)` — the re-boxed return value and
    /// its `Type::Ref` portal-exit type.
    /// Reset at the start of every walk (`fbw_finish_payload_reset`) so a
    /// stale payload from a prior aborted walk cannot leak into this one.
    static FBW_FINISH_PAYLOAD: std::cell::Cell<Option<(OpRef, Type)>> =
        const { std::cell::Cell::new(None) };

    /// Discriminates the `FBW_FINISH_PAYLOAD` disposition: `true` when the
    /// payload is a top-level uncaught raise (`fbw_terminate_with_raise`),
    /// so [`crate::trace::full_body_walk_trace`] builds a
    /// `TraceAction::Finish { exit_with_exception: true }`
    /// (`compile_exit_frame_with_exception`) rather than a value-return
    /// FINISH.  A dedicated flag rather than the `FBW_FINISH_CONCRETE::Raise`
    /// marker because the latter is null-guarded for GC-rooting and so is
    /// absent when the raised exception has no concrete Ref.  Reset with the
    /// payload at the start of every walk.
    static FBW_FINISH_IS_EXCEPTION: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };

    /// The terminal disposition a top-level walk produced, set for a
    /// loop-free portal exit (`DispatchOutcome::Terminate`).  Unlike
    /// `FBW_FINISH_PAYLOAD` (the symbolic re-boxed `OpRef` the compile
    /// consumer records into the trace), this holds the value the walk
    /// *concretely* computed.
    ///
    /// A function trace that fully unrolls to `done_with_this_frame`
    /// executed every residual call concretely (consuming side-effecting
    /// callees like a tokenizer's `get`), so re-running the freshly
    /// compiled trace for the SAME invocation (`ContinueRunningNormally`)
    /// would re-read the already-mutated heap and deopt.  The portal
    /// instead returns this captured value directly (no replay); the
    /// compiled trace serves only subsequent invocations.  See the
    /// consume site in `eval.rs` (`maybe_compile_and_run` portal exit).
    ///
    /// `ConcreteValue::Ref` payloads hold a nursery-resident object across
    /// the post-walk compile (which allocates), so the slot is GC-rooted
    /// via [`fbw_finish_concrete_root_walker`].  `None` for ungated /
    /// loop-closing / float (no concrete float shadow bank) walks → the
    /// portal degrades to the legacy `ContinueRunningNormally` replay.
    static FBW_FINISH_CONCRETE: std::cell::Cell<Option<FinishConcrete>> =
        const { std::cell::Cell::new(None) };

    /// Armed by the bridge tracer (`call_jit::trace_and_compile_from_bridge`)
    /// before a single-frame, direct-return-capable guard-failure walk.  When
    /// set, the `run_perfn_walk` epilogue lets a bridge `Terminate` walk keep
    /// the no-replay shortcut — commit the store journal and keep the
    /// finish-concrete stash — so the caller hands the captured result forward
    /// as `DoneWithThisFrame` instead of rewinding to the guard pc and
    /// re-interpreting the region (which would double every eagerly executed
    /// residual side effect, #177).  Only the bridge tracer sets it, and only
    /// when the resume is single-frame; the general guard path consumes the
    /// kept stash as a terminal `BridgeResolution`, and the CALL_ASSEMBLER
    /// callback hands it to its back-to-back blackhole hook, so a committed
    /// journal never strands into a guard-state re-run.  Cleared after
    /// every bridge walk.
    static FBW_BRIDGE_NOREPLAY_ARMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Set when this walk concretely executed a residual call that is not
    /// provably side-effect-free. Such a residual may have committed a heap
    /// effect outside the FBW journals; later exit handling must not replay it.
    static FBW_EXECUTED_NONPURE_RESIDUAL: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };

    /// Set when this walk concretely executed a non-provably-pure residual that
    /// is NOT the self-recursive `CALL_ASSEMBLER` fold target — a foreign body
    /// write (`events.append(n)`).  A self-recursive fold ahead of which such a
    /// residual ran declines, since folding would leave the walk uncommittable
    /// and the interpreter would replay the executed mutation.
    static FBW_EXECUTED_BODY_RESIDUAL: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Arm/disarm the bridge `Terminate` no-replay shortcut for the next walk
/// (see [`FBW_BRIDGE_NOREPLAY_ARMED`]).  The bridge tracer sets it before
/// the walk and clears it after.
pub fn fbw_bridge_noreplay_arm(armed: bool) {
    FBW_BRIDGE_NOREPLAY_ARMED.with(|c| c.set(armed));
}

/// Whether the bridge `Terminate` no-replay shortcut is armed for the
/// current walk (read by the `run_perfn_walk` epilogue predicate).
pub(crate) fn fbw_bridge_noreplay_armed() -> bool {
    FBW_BRIDGE_NOREPLAY_ARMED.with(|c| c.get())
}

/// Record that the current walk concretely executed a residual which could
/// have committed non-journaled heap state.
pub(crate) fn fbw_mark_executed_nonpure_residual() {
    FBW_EXECUTED_NONPURE_RESIDUAL.with(|c| c.set(true));
}

/// Whether the current walk has concretely executed a non-provably-pure
/// residual.
pub(crate) fn fbw_executed_nonpure_residual() -> bool {
    FBW_EXECUTED_NONPURE_RESIDUAL.with(|c| c.get())
}

/// Clear the executed-residual latch at a walk boundary.
pub(crate) fn fbw_executed_nonpure_residual_reset() {
    FBW_EXECUTED_NONPURE_RESIDUAL.with(|c| c.set(false));
}

/// Record a foreign (non self-recursive) non-pure residual concrete execution.
pub(crate) fn fbw_mark_executed_body_residual() {
    FBW_EXECUTED_BODY_RESIDUAL.with(|c| c.set(true));
}

/// Whether a foreign non-pure residual has concretely executed this walk.
pub(crate) fn fbw_executed_body_residual() -> bool {
    FBW_EXECUTED_BODY_RESIDUAL.with(|c| c.get())
}

/// Clear the foreign-body-residual latch at a walk boundary.
pub(crate) fn fbw_executed_body_residual_reset() {
    FBW_EXECUTED_BODY_RESIDUAL.with(|c| c.set(false));
}

/// Whether `PYRE_FBW_DEBUG_ABORT` is set.  When on, `full_body_walk_trace`
/// prints the structured reason (the `DispatchError` variant or the
/// non-loop-closing `DispatchOutcome`) for every walk that maps to
/// `TraceAction::Abort` / `AbortPermanent`.  The metainterp's own
/// "abort trace at key={} (permanent={})" log (`pyjitpl.rs`) only
/// reports the key and permanence; the walker-side reason is otherwise
/// swallowed.  Default OFF → no output, zero production effect.
pub fn fbw_debug_abort_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_FBW_DEBUG_ABORT").is_some())
}

/// Clear any stashed Finish payload before a walk begins (mirrors
/// [`bool_box_truth_reset`]).  Also clears the concrete-return cell so a
/// stale value from a prior aborted walk cannot leak into this one.
pub(crate) fn fbw_finish_payload_reset() {
    FBW_FINISH_PAYLOAD.with(|c| c.set(None));
    FBW_FINISH_IS_EXCEPTION.with(|c| c.set(false));
    FBW_FINISH_CONCRETE.with(|c| c.set(None));
}

/// Consume the Finish payload stashed by a top-level `*_return` arm.
pub(crate) fn fbw_finish_payload_take() -> Option<(OpRef, Type)> {
    FBW_FINISH_PAYLOAD.with(|c| c.take())
}

/// Stash the concrete return value of a top-level value-returning
/// `*_return` arm (see [`FBW_FINISH_CONCRETE`]).
pub(crate) fn fbw_finish_concrete_set(value: ConcreteValue) {
    FBW_FINISH_CONCRETE.with(|c| c.set(Some(FinishConcrete::Return(value))));
}

/// Stash the concrete exception object of a top-level uncaught raise.
pub(crate) fn fbw_finish_raise_set(value: ConcreteValue) {
    FBW_FINISH_CONCRETE.with(|c| c.set(Some(FinishConcrete::Raise(value))));
}

/// Peek at the stashed terminal disposition without consuming it (the
/// `run_perfn_walk` epilogue uses this to decide whether to commit the
/// store journal and keep the no-replay shortcut; the CALL_ASSEMBLER
/// bridge callback uses it to leave a kept stash in its rooted cell for
/// the back-to-back blackhole hook).
pub fn fbw_finish_concrete_peek() -> Option<FinishConcrete> {
    FBW_FINISH_CONCRETE.with(|c| c.get())
}

/// Clear the stashed terminal disposition.  The `run_perfn_walk`
/// epilogue calls this when the no-replay shortcut is declined (not a
/// `Terminate` walk, or an unjournaled effect only the replay applies) so
/// the portal degrades to `ContinueRunningNormally`; the CALL_ASSEMBLER
/// blackhole hook calls it so a kept stash that cannot be consumed does
/// not leak into a later portal take.
pub fn fbw_finish_concrete_reset() {
    FBW_FINISH_CONCRETE.with(|c| c.set(None));
}

/// Consume the stashed terminal disposition at the portal exit.
pub fn fbw_finish_concrete_take() -> Option<FinishConcrete> {
    FBW_FINISH_CONCRETE.with(|c| c.take())
}

/// `framework.py root_walker.walk_roots` parity for the concrete terminal
/// value: a `Ref` payload holds a nursery-resident object across the
/// post-walk compile (which allocates and may trigger a minor collection
/// that moves nursery objects), so the slot is forwarded as a root.
/// Registered once via `register_extra_root_walker` at JIT init, mirroring
/// [`fbw_store_journal_root_walker`].
pub fn fbw_finish_concrete_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let data = capture_fbw_finish_concrete_root_area();
    unsafe { fbw_finish_concrete_root_walker_area(data, visitor) };
}

pub fn capture_fbw_finish_concrete_root_area() -> *const () {
    FBW_FINISH_CONCRETE.with(|value| value as *const _ as *const ())
}

/// Record that `op` is a walker-built inline exception (B3 construct fold).
pub(crate) fn fbw_built_exc_insert(op: OpRef) {
    FBW_BUILT_EXC.with(|s| {
        s.borrow_mut().insert(op);
    });
}

/// Consume (remove) `op` from the walker-built-exception set.  Returns
/// `true` if it was present — i.e. the raised value was built inline by
/// [`try_walker_trace_exception_new`].  Removed (not just read) so a
/// second raise of the same object (whose `w_context` is now stamped)
/// takes the residual path, matching the trait's
/// `trace_built_exc.remove(&exc_val.opref)`.
pub(crate) fn fbw_built_exc_take(op: OpRef) -> bool {
    FBW_BUILT_EXC.with(|s| s.borrow_mut().remove(&op))
}

/// Clear the store journal and residual-call census before a walk
/// begins (mirrors [`bool_box_truth_reset`]).
pub(crate) fn fbw_store_journal_reset() {
    FBW_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_APPEND_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_APPEND_PROMOTE_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_CELL_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_SYS_EXC_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_UNJOURNALED_VALUE_UNAVAILABLE.with(|c| c.set(false));
    FBW_UNJOURNALED_SYMBOLIC.with(|c| c.set(false));
    FBW_EXECUTED_RESIDUAL_VOID.with(|c| c.set(0));
    FBW_EXECUTED_RESIDUAL_MAYFORCE.with(|c| c.set(0));
    FBW_EXECUTED_RESIDUAL_PLAIN.with(|c| c.set(0));
    // gh#467: reset the executed-effect odometer and any stale
    // forward-flush carrier a prior aborted walk latched.
    FBW_EXECUTED_EFFECT_COUNT.with(|c| c.set(0));
    FBW_STRUCTURAL_ABORT_OPCODE_EFFECTS.with(|c| c.set(None));
    FBW_ABORT_CALL_RESUME.with(|c| *c.borrow_mut() = None);
    // #57 Option C: drop any in-flight FOR_ITER items a prior aborted walk
    // left undelivered (its live frame already consumed the delivery), so a
    // stale item cannot be re-delivered by this walk's abort.  This also
    // clears the per-entry body-effect signal so a prior walk's committed
    // mutation cannot block this walk's delivery.
    FBW_FORITER_INFLIGHT.with(|c| c.borrow_mut().clear());
    // B3: drop any inline-built-exception OpRef keys a
    // prior aborted walk recorded, so they cannot match a same-numbered
    // OpRef minted by this walk's recorder.
    FBW_BUILT_EXC.with(|s| s.borrow_mut().clear());
    // B3: drop any unbalanced PUSH_EXC_INFO prev saves a
    // prior aborted walk left (an exception that propagated out without its
    // POP_EXCEPT restore), so a stale saved-prev cannot be popped by an
    // unrelated POP_EXCEPT in this walk.
    FBW_EXC_PREV.with(|s| s.borrow_mut().clear());
    FBW_EXC_PENDING_PUSH_SET.with(|c| c.set(false));
}

/// Record the element a walked eager list store displaces, for rollback
/// when the walk does not commit its end state.
pub(crate) fn fbw_store_journal_push(
    list: pyre_object::PyObjectRef,
    key: pyre_object::PyObjectRef,
    displaced: pyre_object::PyObjectRef,
) {
    FBW_STORE_JOURNAL.with(|j| j.borrow_mut().push([list, key, displaced]));
    // gh#467: a journaled store still mutated the heap this iteration; the
    // forward-flush gate counts it so a callee sub-walk that appends/setitems
    // cannot be committed-then-re-executed (a double).
    fbw_bump_executed_effect();
}

/// Record the live length a walked eager list append grew past, for the
/// length rewind when the walk does not commit its end state.  `list` must
/// be an Integer-strategy list whose backing array had spare capacity (the
/// append's gate), so the rewind is allocation-free.
// Consumed by the #171 `list.append` orthodox descent
// (`try_walker_orthodox_list_append`).
pub(crate) fn fbw_append_journal_push(list: pyre_object::PyObjectRef, length_before: usize) {
    FBW_APPEND_JOURNAL.with(|j| j.borrow_mut().push((list, length_before)));
    // gh#467: see `fbw_store_journal_push`.
    fbw_bump_executed_effect();
}

/// Record an Empty-list first append whose eager execution promoted the list
/// to a typed strategy, for strategy restore when the walk does not commit.
#[allow(dead_code)]
pub(crate) fn fbw_append_promote_journal_push(list: pyre_object::PyObjectRef) {
    FBW_APPEND_PROMOTE_JOURNAL.with(|j| j.borrow_mut().push(list));
}

/// Undo the most recent Empty-to-typed list promotion when its speculative
/// append fold is locally declined.
pub(crate) fn fbw_append_promote_journal_rollback_last(list: pyre_object::PyObjectRef) {
    FBW_APPEND_PROMOTE_JOURNAL.with(|j| {
        let popped = j.borrow_mut().pop();
        assert_eq!(popped, Some(list));
    });
    unsafe { pyre_object::listobject::w_list_clear(list) };
}

/// Record the `intvalue` a walked eager `IntMutableCell` store displaces,
/// for the in-place restore when the walk does not commit its end state.
// Consumed by the StoreName/StoreGlobal cell fold
// (`emit_namespace_cell_store_fold`).
pub(crate) fn fbw_cell_store_journal_push(cell: pyre_object::PyObjectRef, intvalue_before: i64) {
    if fbw_debug_abort_enabled() {
        eprintln!(
            "[fbw-cell-journal] push cell=0x{:x} before={intvalue_before}",
            cell as usize
        );
    }
    FBW_CELL_STORE_JOURNAL.with(|j| j.borrow_mut().push((cell, intvalue_before)));
    // gh#467: see `fbw_store_journal_push`.
    fbw_bump_executed_effect();
}

/// Record the `sys_exc_value` a walked eager `set_current_exception`
/// displaces, for the in-place restore when the walk does not commit its
/// end state.  Pushed by [`try_walker_lower_exc_info_residual`] before it
/// applies the concrete store.
pub(crate) fn fbw_sys_exc_journal_push(displaced: pyre_object::PyObjectRef) {
    FBW_SYS_EXC_JOURNAL.with(|j| j.borrow_mut().push(displaced));
}

/// Commit-path epilogue: the walk's eager stores and appends stand; drop
/// the undo logs.
pub(crate) fn fbw_store_journal_commit() {
    FBW_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_APPEND_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_APPEND_PROMOTE_JOURNAL.with(|j| j.borrow_mut().clear());
    FBW_CELL_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
    // A committed walk keeps its eager `sys_exc_value` store (the compiled
    // trace or the adopted end state carries the same exception state), so
    // drop the undo log without re-applying it.
    FBW_SYS_EXC_JOURNAL.with(|j| j.borrow_mut().clear());
    // #57 Option C: a committed walk's end-flush adopts the advanced
    // iterator + the body that consumed it (counted once), so the in-flight
    // items must NOT also be delivered — drop the stash (and with it the
    // per-entry body-effect signals).
    FBW_FORITER_INFLIGHT.with(|c| c.borrow_mut().clear());
}

/// Record a bridge/retrace recording walk's range-iterator cursor before its
/// eager advance, so the abort path can restore it ([`FBW_BRIDGE_ITER_JOURNAL`]).
/// Called from the range FOR_ITER specialization ONLY while `is_bridge_trace`.
pub(crate) fn fbw_bridge_iter_journal_push(
    iter: pyre_object::PyObjectRef,
    pre_current: i64,
    pre_remaining: i64,
) {
    FBW_BRIDGE_ITER_JOURNAL.with(|j| j.borrow_mut().push((iter, pre_current, pre_remaining)));
}

/// Non-commit epilogue for a bridge/retrace recording walk: restore each
/// range iterator to the cursor it held before the walk advanced it, in
/// reverse push order.  The interpreter resume then re-consumes the item the
/// aborted recording had taken, so the iteration is executed exactly once.
pub(crate) fn fbw_bridge_iter_journal_rollback() {
    FBW_BRIDGE_ITER_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some((iter, pre_current, pre_remaining)) = entries.pop() {
            unsafe {
                pyre_object::functional::w_range_iter_set_cursor(iter, pre_current, pre_remaining);
            }
        }
    });
}

/// Commit epilogue: a committed bridge recording keeps its advanced cursor
/// (the compiled bridge adopts it as the authoritative continuation), so drop
/// the undo log without restoring.
pub(crate) fn fbw_bridge_iter_journal_clear() {
    FBW_BRIDGE_ITER_JOURNAL.with(|j| j.borrow_mut().clear());
}

/// Record the in-flight FOR_ITER continuation (#57 Option C): the consumed
/// item the `for_iter_next` residual produced and its FOR_ITER body coordinate.
/// Called from the residual
/// executor's success arm when the helper is [`PyreHelperKind::ForIterNext`]
/// and it produced a non-null item (a null item is the exhaustion arm — no
/// body runs, nothing to deliver).  The stack mirrors loop nesting: a consume
/// of a DIFFERENT (deeper) FOR_ITER pushes a new entry on top of the loops
/// that enclose it, while a re-consume of a FOR_ITER ALREADY on the stack is
/// that loop advancing to its next iteration — every entry above it belongs
/// to nested loops that have run to completion inside the prior body, so they
/// are popped, and the loop's own entry is replaced (a fresh body-effect
/// window).  The outer loop's in-flight item is thus no longer destroyed by an
/// inner consume, and a completed inner loop leaves no stale entry.
pub(crate) fn fbw_foriter_inflight_capture(
    item: pyre_object::PyObjectRef,
    body: InflightForiterBody,
) {
    FBW_FORITER_INFLIGHT.with(|c| {
        let mut stack = c.borrow_mut();
        // The "body effect since consume" window restarts at each consume:
        // only effects committed after THIS consume can double on a re-run of
        // THIS iteration's body (Finding #1).  A fresh entry starts clear.
        let entry = InflightForiter {
            item,
            body,
            body_effect_since_consume: false,
            body_completed: false,
        };
        let Some(body_pc) = inflight_foriter_body_pc(body) else {
            // An unresolvable native coordinate cannot identify an existing
            // loop. Keep this item as a distinct entry; later consumers also
            // refuse it conservatively instead of guessing a Python pc.
            stack.push(entry);
            return;
        };
        match stack
            .iter()
            .position(|e| inflight_foriter_body_pc(e.body) == Some(body_pc))
        {
            Some(at) => {
                stack.truncate(at + 1);
                stack[at] = entry;
            }
            None => stack.push(entry),
        }
    });
}

/// Whether an in-flight FOR_ITER item is currently captured (a consume ran
/// this iteration and no commit/abort has cleared it yet).  Sampled by the
/// residual executor to decide whether a non-elidable concrete mutation
/// counts as a body effect committed after the consume (Finding #1).
pub(crate) fn fbw_foriter_inflight_active() -> bool {
    FBW_FORITER_INFLIGHT.with(|c| !c.borrow().is_empty())
}

/// Mark the in-flight entry for `body` body-completed: a NEW
/// `for_iter_next` attempt is being dispatched for the same FOR_ITER, so the
/// prior consumed item's body has run to completion (the walk is back at the
/// header).  Called from the residual dispatch BEFORE the call executes so an
/// attempt that aborts mid-way (a kept-stack guard on the exhaustion arm)
/// still leaves the completion recorded; a successful attempt replaces the
/// entry with a fresh one anyway ([`fbw_foriter_inflight_capture`]).
pub(crate) fn fbw_foriter_inflight_mark_attempt(body: InflightForiterBody) {
    FBW_FORITER_INFLIGHT.with(|c| {
        let Some(body_pc) = inflight_foriter_body_pc(body) else {
            return;
        };
        if let Some(entry) = c
            .borrow_mut()
            .iter_mut()
            .find(|e| inflight_foriter_body_pc(e.body) == Some(body_pc))
        {
            entry.body_completed = true;
        }
    });
}

/// Flag that a non-elidable concrete residual committed an irreversible heap
/// mutation after the in-flight FOR_ITER consume (Finding #1, R1).  A mutation
/// committed while several FOR_ITER items are in flight is "after" every one
/// of them — re-running ANY of their bodies on delivery re-applies it — so
/// mark every active entry.
pub(crate) fn fbw_mark_foriter_body_effect_since_consume() {
    FBW_FORITER_INFLIGHT.with(|c| {
        for entry in c.borrow_mut().iter_mut() {
            entry.body_effect_since_consume = true;
        }
    });
}

/// Drop every in-flight FOR_ITER entry (#32 S2): a committed branch-flush has
/// adopted the walk's end state and owns the iteration count, so no item may be
/// delivered afterward.
pub fn fbw_foriter_inflight_clear() {
    FBW_FORITER_INFLIGHT.with(|c| c.borrow_mut().clear());
}

/// #32 S2 deliver selector for the branch-flush leg.  Returns
/// `Some((item, body_pc))` to push at the body ONLY when `resume_py_pc` is the
/// header of a FOR_ITER whose consumed item is in flight (`body_pc ==
/// resume_py_pc + 1`, and the opcode there really is a FOR_ITER) — Shape A, the
/// abort parked on the FOR_ITER before its body ran, so the item is not yet on
/// the flushed header stack and must be delivered.  Returns `None` when the
/// resume pc is not such a header, or when the matching entry carries a
/// body-effect signal (the R1 never-double guard: re-running the body would
/// re-apply an irreversible mutation).  Read-only — the caller drops the stash
/// via [`fbw_foriter_inflight_clear`] only after the flush commits, so a
/// declined flush leaves the in-flight items intact for the legacy deliver.
pub fn fbw_foriter_inflight_take_for_resume(
    frame: usize,
    resume_py_pc: usize,
) -> Option<(pyre_object::PyObjectRef, usize)> {
    let body_pc = resume_py_pc + 1;
    if !foriter_header_at(frame, resume_py_pc) {
        return None;
    }
    FBW_FORITER_INFLIGHT.with(|c| {
        let stack = c.borrow();
        let at = stack
            .iter()
            .position(|e| inflight_foriter_body_pc(e.body) == Some(body_pc))?;
        // R1 never-double guard (cross-checks #33): an irreversible body effect
        // committed since this consume means re-running the body on delivery
        // would double it — refuse delivery.  A body-COMPLETED entry (the walk
        // re-reached the consume, so this item's body already ran) must never
        // be delivered either — that is the header-flush-without-delivery
        // shape ([`fbw_foriter_inflight_completed_at_resume`]).  Also refuse if
        // either journal is non-empty or an unjournaled effect stands (same
        // signals as `fbw_foriter_inflight_take`).
        if stack[at].body_effect_since_consume
            || stack[at].body_completed
            || fbw_store_journal_len() != 0
            || FBW_APPEND_JOURNAL.with(|j| j.borrow().len()) != 0
            || fbw_has_unjournaled_effect()
        {
            return None;
        }
        Some((stack[at].item, body_pc))
    })
}

/// #493 selector for the header-flush-without-delivery shape: `resume_py_pc`
/// is a FOR_ITER header whose in-flight entry is body-COMPLETED — the abort
/// fired during the NEXT consume attempt (a kept-stack guard on the FOR_ITER
/// arms after the `for_iter_next` residual), so the consumed item's body
/// already ran during the walk.  The walk end state at the header is then the
/// complete post-body state: the flush adopts it WITHOUT delivering the item
/// and the interpreter re-attempts the consume against the advanced iterator.
/// Refuses when an effect committed since the consume (re-attempting the
/// consume could re-apply the failed attempt's effect) — same signals as the
/// delivery selector above.
pub fn fbw_foriter_inflight_completed_at_resume(frame: usize, resume_py_pc: usize) -> bool {
    let body_pc = resume_py_pc + 1;
    if !foriter_header_at(frame, resume_py_pc) {
        return false;
    }
    FBW_FORITER_INFLIGHT.with(|c| {
        let stack = c.borrow();
        let Some(at) = stack
            .iter()
            .position(|e| inflight_foriter_body_pc(e.body) == Some(body_pc))
        else {
            return false;
        };
        stack[at].body_completed
            && !stack[at].body_effect_since_consume
            && fbw_store_journal_len() == 0
            && FBW_APPEND_JOURNAL.with(|j| j.borrow().len()) == 0
            && !fbw_has_unjournaled_effect()
    })
}

/// Whether a body effect committed since the most-recent in-flight FOR_ITER
/// consume (Finding #1, R1) — the top entry, the one [`fbw_foriter_inflight_take`]
/// delivers.
pub(crate) fn fbw_foriter_body_effect_since_consume() -> bool {
    FBW_FORITER_INFLIGHT
        .with(|c| c.borrow().last().map(|e| e.body_effect_since_consume))
        .unwrap_or(false)
}

/// Whether ANY of the three R1 body-effect signals is currently present:
/// the body-effect-since-consume flag, either journal non-empty, or the
/// unjournaled-effect flag.  These are the exact signals
/// [`fbw_foriter_inflight_take`] consults to REFUSE delivery, and `take`
/// leaves them untouched.  Exposed for the deliver-path loud-failure
/// debug-assert (#57 Finding #3): a successful take (delivery) while any
/// signal stands would be a silent double, so the deliver site asserts this
/// is `false` in debug builds.
pub fn fbw_foriter_any_body_effect_signal() -> bool {
    fbw_foriter_body_effect_since_consume()
        || fbw_store_journal_len() != 0
        || FBW_APPEND_JOURNAL.with(|j| j.borrow().len()) != 0
        || FBW_CELL_STORE_JOURNAL.with(|j| j.borrow().len()) != 0
        || fbw_has_unjournaled_effect()
}

/// Take the in-flight FOR_ITER continuation for delivery on a trace abort
/// (#57 Option C).  Returns `(consumed_item, body_pc)` and clears the stash
/// so it is delivered at most once.
///
/// R1 (double-apply guard): delivery resumes the live frame at the FOR_ITER
/// body, so any body op that ALREADY ran concretely during the aborted walk
/// would be re-applied.  C may DELIVER only when it can PROVE no body effect
/// committed for the in-flight iteration — then re-running the body cannot
/// double.  Three signals together cover every committed body effect:
///
/// * `fbw_foriter_body_effect_since_consume()` — a non-elidable concrete
///   residual mutated the heap OUTSIDE the journals after the consume (a dict
///   `store_subscr_fn`, an unmodeled container method).  Irreversible: the
///   mutation already stands on the live heap, so a body re-run would double
///   it (Finding #1).
/// * either journal non-empty (`FBW_STORE_JOURNAL` list setitem /
///   `FBW_APPEND_JOURNAL` list append).  On the production abort path
///   `fbw_store_journal_rollback` empties these BEFORE this take, so this is
///   normally false here; the check is a belt-and-suspenders refusal in case
///   a future caller takes before the rollback.
/// * `fbw_has_unjournaled_effect()` — a void/symbolic residual only the
///   legacy replay applies, which the rollback cannot undo.
///
/// Any signal set → refuse delivery (drop the stash → the legacy bypass keeps
/// the prior drop-on-abort behaviour for that shape, never a double).
/// `for_mutate` aborts BEFORE the append's effect, so all three signals are
/// clear at the abort point — the clean continuation case.
pub fn fbw_foriter_inflight_take() -> Option<(pyre_object::PyObjectRef, usize)> {
    // Take the MOST-RECENT (top) entry and drop the rest, matching the
    // single-slot behaviour: one take delivers the innermost in-flight item
    // and leaves nothing for a subsequent deliver call.  (S2 will instead
    // deliver every entry at its true frame slot.)
    let stash = FBW_FORITER_INFLIGHT.with(|c| {
        let mut stack = c.borrow_mut();
        let top = stack.pop();
        stack.clear();
        top
    });
    let stash = stash?;
    let body_effect = stash.body_effect_since_consume;
    let Some(body_pc) = inflight_foriter_body_pc(stash.body) else {
        return None;
    };
    let store_len = fbw_store_journal_len();
    let append_len = FBW_APPEND_JOURNAL.with(|j| j.borrow().len());
    let cell_store_len = FBW_CELL_STORE_JOURNAL.with(|j| j.borrow().len());
    let unjournaled = fbw_has_unjournaled_effect();
    if body_effect || store_len != 0 || append_len != 0 || cell_store_len != 0 || unjournaled {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-foriter] deliver REFUSED (body effect committed since consume) body_pc={} \
                 body_effect={body_effect} store_journal_len={store_len} \
                 append_journal_len={append_len} unjournaled={unjournaled} \
                 — keeping legacy drop-on-abort to avoid a double-apply (R1)",
                body_pc
            );
        }
        return None;
    }
    if fbw_debug_abort_enabled() {
        eprintln!(
            "[fbw-foriter] deliver item=0x{:x} body_pc={} store_journal_len={store_len} \
             unjournaled={unjournaled}",
            stash.item as usize, body_pc,
        );
    }
    Some((stash.item, body_pc))
}

/// Non-commit epilogue: restore each displaced element in reverse push
/// order so the legacy replay re-executes against the pre-walk heap.
/// `w_list_setitem` allocates nothing on the restore (the displaced value
/// is already boxed and strategy-matching), so entries cannot move
/// mid-rollback.
///
/// Stores are restored BEFORE appends are rewound: a store's key was
/// in-bounds at store time and stays in-bounds at the walk's final
/// (max) length, so every restore lands while the list is still grown;
/// shrinking first could push a restore index past the length and drop it.
pub(crate) fn fbw_store_journal_rollback() {
    FBW_STORE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some([list, key, displaced]) = entries.pop() {
            let restored = unsafe {
                let index = pyre_object::w_int_get_value(key);
                pyre_object::w_list_setitem(list, index, displaced)
            };
            if !restored {
                // Only reachable when another eagerly executed residual
                // shrank the list after the store — a shape the replay
                // already cannot undo (the residual re-runs).  Surface it
                // under the debug gate instead of corrupting silently.
                if fbw_debug_abort_enabled() {
                    eprintln!("[fbw-store-journal] rollback failed (index out of bounds)");
                }
            }
        }
    });
    // Rewind each eager append's length in reverse push order
    // (allocation-free length set; the journal records only spare-capacity
    // appends, so there is no realloc to undo and the strategy at rollback
    // equals the strategy at push). Dispatch the rewind to the strategy's
    // length field: Object rewinds the `W_ListObject.length` header,
    // Integer/Float the `int_items`/`float_items` length.
    FBW_APPEND_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some((list, length_before)) = entries.pop() {
            unsafe {
                let list_ref = &mut *(list as *mut pyre_object::listobject::W_ListObject);
                match list_ref.strategy {
                    pyre_object::listobject::ListStrategy::Object => {
                        // The appended element is a GC ptr and the items block is
                        // scanned over [0..capacity], so null the vacated slot
                        // before shrinking (ll_pop_default: ll_setitem_fast(index,
                        // ll_null_item) then _ll_resize_le) — otherwise the slot at
                        // `length_before` holds a stale ref past the logical length.
                        pyre_object::listobject::ll_list_obj_setitem_fast(
                            list_ref,
                            length_before,
                            pyre_object::pyobject::PY_NULL,
                        );
                        pyre_object::listobject::ll_list_obj_set_len(list_ref, length_before);
                    }
                    pyre_object::listobject::ListStrategy::Integer => {
                        pyre_object::listobject::ll_list_int_set_len(list_ref, length_before);
                    }
                    // Float items are non-ptr f64 scalars (no stale GC ref to
                    // clear, unlike the Object slot), so rewinding the length
                    // field suffices.
                    pyre_object::listobject::ListStrategy::Float => {
                        pyre_object::listobject::ll_list_float_set_len(list_ref, length_before);
                    }
                    // Empty never enters the append journal (no spare-capacity
                    // fold path records it); nothing to rewind.
                    pyre_object::listobject::ListStrategy::Empty => {}
                }
            }
        }
    });
    // The length rewind above already shrank the list back to length 0.
    // `w_list_clear` additionally restores the Empty strategy and drops the
    // typed backing block, completing the undo of the Empty-to-typed switch
    // that the length journal alone cannot undo.
    FBW_APPEND_PROMOTE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some(list) = entries.pop() {
            unsafe {
                pyre_object::listobject::w_list_clear(list);
            }
        }
    });
    // Restore each eagerly stored `IntMutableCell`'s prior `intvalue` in
    // reverse push order (raw i64 write; allocation-free, cells immovable).
    FBW_CELL_STORE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some((cell, intvalue_before)) = entries.pop() {
            unsafe {
                if fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-cell-journal] rollback cell=0x{:x} {} -> {intvalue_before}",
                        cell as usize,
                        (*(cell as *const pyre_object::celldict::IntMutableCell)).intvalue
                    );
                }
                (*(cell as *mut pyre_object::celldict::IntMutableCell)).intvalue = intvalue_before;
            }
        }
    });
    // Restore `sys_exc_value` to its pre-walk value.  Replaying in reverse
    // push order makes the LAST write the value read at walk entry (the
    // first eager store's displaced prior), so an aborted walk leaves the
    // live per-thread EC exactly as the legacy replay-from-start expects —
    // in particular an exception that propagated OUT of an except-handler
    // (walk aborted before its POP_EXCEPT restore) no longer leaks the
    // caught exception into the next frame's `__context__`.
    FBW_SYS_EXC_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some(displaced) = entries.pop() {
            pyre_interpreter::eval::set_current_exception(displaced);
        }
    });
}

/// Current journal length (commit-point diagnostics).
pub(crate) fn fbw_store_journal_len() -> usize {
    FBW_STORE_JOURNAL.with(|j| j.borrow().len())
}

/// Mark the walk as carrying a recorded-but-unexecuted side effect only
/// the legacy replay applies.
pub(crate) fn fbw_mark_unjournaled_effect(cause: ResidualDecline) {
    match cause {
        ResidualDecline::ValueUnavailable => {
            FBW_UNJOURNALED_VALUE_UNAVAILABLE.with(|c| c.set(true));
        }
        ResidualDecline::Symbolic => FBW_UNJOURNALED_SYMBOLIC.with(|c| c.set(true)),
    }
}

/// gh#467 executed-effect odometer read (see [`FBW_EXECUTED_EFFECT_COUNT`]).
pub(crate) fn fbw_executed_effect_count() -> usize {
    FBW_EXECUTED_EFFECT_COUNT.with(|c| c.get())
}

pub(crate) fn fbw_structural_abort_opcode_is_effect_free(pc: usize) -> bool {
    FBW_STRUCTURAL_ABORT_OPCODE_EFFECTS.with(|c| c.get() == Some((pc, 0)))
}

/// gh#467 bump the executed-effect odometer (see [`FBW_EXECUTED_EFFECT_COUNT`]).
pub(crate) fn fbw_bump_executed_effect() {
    FBW_EXECUTED_EFFECT_COUNT.with(|c| c.set(c.get() + 1));
}

/// gh#467 latch the inline-abort forward-flush carrier (see [`FBW_ABORT_CALL_RESUME`]).
pub(crate) fn fbw_set_abort_call_resume(
    outer_jitcode_index: u32,
    call_jitcode_pc: usize,
    stack: Vec<pyre_object::PyObjectRef>,
) {
    FBW_ABORT_CALL_RESUME.with(|c| {
        *c.borrow_mut() = Some(InlineAbortCarrier::Entry {
            outer_jitcode_index,
            call_jitcode_pc,
            call_stack: stack,
        })
    });
}

pub(crate) fn fbw_set_midbody_abort_resume(payload: MidBodyPayload) {
    FBW_ABORT_CALL_RESUME.with(|c| *c.borrow_mut() = Some(InlineAbortCarrier::MidBody(payload)));
}

pub(crate) fn fbw_abort_carrier_clone() -> Option<InlineAbortCarrier> {
    FBW_ABORT_CALL_RESUME.with(|c| c.borrow().clone())
}

pub(crate) fn fbw_abort_carrier_set_return(value: pyre_object::PyObjectRef) {
    FBW_ABORT_CALL_RESUME.with(|c| {
        if let Some(InlineAbortCarrier::MidBody(payload)) = c.borrow_mut().as_mut() {
            payload.return_value = value;
        }
    });
}

pub(crate) fn fbw_abort_carrier_clear() {
    FBW_ABORT_CALL_RESUME.with(|c| *c.borrow_mut() = None);
}

/// A declined residual call (`try_execute_residual_call_via_executor`
/// returned `None`) reached during a multiframe-inlined callee sub-walk
/// (the framestack is non-empty) cannot fall back to the walk-end
/// legacy replay.  The replay re-enters the freshly compiled loop from the
/// recorded entry state while sibling concretely-executed heap mutations of
/// the SAME iteration (the enclosing loop's `i = i + 1`) have already
/// applied, so the first compiled iteration is half-applied — one
/// iteration's contribution silently dropped (#68 depth-2 multiframe:
/// `s = s + outer(i)` lands short by exactly `outer(N+1)` because the
/// nested callee's residual never ran). Decline the enclosing trace to
/// interpretation instead. At top level (no active inline)
/// the unjournaled-effect / legacy-replay path is sound, so only abort when
/// nested.
thread_local! {
    /// Marks the self-recursive `CALL_ASSEMBLER` fold's concrete-stamp
    /// executor call. RPython `do_residual_call` executes the recorded
    /// residual at any framestack depth (`pyjitpl.py`), while
    /// pyre's nested-residual decline below is a local protection for
    /// FOREIGN unjournaled residuals.
    pub(crate) static SELFREC_CA_FOLD_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The bounded `str(exc)` / `repr(exc)` descriptor inline may retain an
    /// interior residual such as `repr(self.args)`. The caller's original
    /// iteration already supplied the concrete result, while the compiled
    /// trace executes that residual once on later iterations, so the generic
    /// nested-replay decline does not apply to this resolved descriptor path.
    pub(crate) static EXCEPTION_STRING_INLINE_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Code keys of the callees a FOR_ITER body admitted under
    /// [`CalleeReplaySafety::DeferredCall`], outermost first.  Non-empty for
    /// the lifetime of such a sub-walk ([`ForiterDeferredInlineGuard`]), which
    /// is what arms the deferred-call arm of
    /// [`fbw_abort_nested_unjournaled_residual`].
    static FBW_FORITER_DEFERRED_INLINE: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// Callee code keys whose deferred body reached a CALL residual the lever
    /// could not inline.  The gate declines them up front from then on, so the
    /// backstop abort costs one attempt per callee instead of storming.
    static FBW_FORITER_DEFERRED_DENY: std::cell::RefCell<std::collections::HashSet<usize>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

/// Marks the sub-walk of a callee admitted into a FOR_ITER body under
/// [`CalleeReplaySafety::DeferredCall`] for its whole lifetime, so a nested
/// residual the lever could not inline can recognise the admission it breaks
/// (and the callee to deny) rather than executing.
pub(crate) struct ForiterDeferredInlineGuard(bool);

impl ForiterDeferredInlineGuard {
    pub(crate) fn enter(callee_code_key: usize, deferred: bool) -> Self {
        if deferred {
            FBW_FORITER_DEFERRED_INLINE.with(|c| c.borrow_mut().push(callee_code_key));
        }
        ForiterDeferredInlineGuard(deferred)
    }
}

impl Drop for ForiterDeferredInlineGuard {
    fn drop(&mut self) {
        if self.0 {
            FBW_FORITER_DEFERRED_INLINE.with(|c| {
                c.borrow_mut().pop();
            });
        }
    }
}

/// The outermost callee the active sub-walk was admitted for under
/// [`CalleeReplaySafety::DeferredCall`], or `None` outside such a sub-walk.
/// Declining that callee suppresses the whole nest: a body that calls another
/// is itself `DeferredCall`, so no admitted caller sits above it.
fn fbw_foriter_deferred_inline_outermost() -> Option<usize> {
    FBW_FORITER_DEFERRED_INLINE.with(|c| c.borrow().first().copied())
}

pub(crate) fn fbw_foriter_deferred_call_denied(callee_code_key: usize) -> bool {
    FBW_FORITER_DEFERRED_DENY.with(|c| c.borrow().contains(&callee_code_key))
}

fn fbw_foriter_deny_deferred_call(callee_code_key: usize) {
    FBW_FORITER_DEFERRED_DENY.with(|c| {
        c.borrow_mut().insert(callee_code_key);
    });
}

/// Whether the active inline sub-walk is one of the hazard classes the blanket
/// nested-residual decline was masking, as opposed to a straight-line mutating
/// callee (the #73 depth-≥2 payoff, which inlines).  Two classes decline:
///
/// * **Loop-bearing** — a framestack callee whose `CodeObject` has a
///   `FOR_ITER`.  Its side-effecting `for` consume runs concretely in the
///   sub-walk, and a later kept-stack guard abort can REFUSE the Option-C item
///   delivery (a `for..break` frame parked past the loop header,
///   eval.rs:5445), so the re-run re-executes the consume and double-advances
///   the iterator (the two `foriter_exempt_*` witnesses).
/// * **Self-recursive** — the callee calls itself.  A hot self-recursion
///   forms a `CALL_ASSEMBLER` bridge whose moving-nursery callee frame cannot
///   survive the residual trampoline retaining a pre-call frame pointer; on
///   the wasm always-portal path the inlined body also type-confuses the
///   optimizer (`setintbound: got Ref`, the `wasm_ca_trampoline_decline`
///   witness).  Detected both dynamically (the same `w_code` already nested in
///   the framestack — mutual/deep recursion) and statically
///   (`code_is_self_recursive`), since the recursive call residualizes to a
///   `CALL_ASSEMBLER` rather than nesting the framestack, so it is already a
///   hazard at inline depth 1.
///
/// The `w_code` field is the `jitcode_for` code key, resolved to the raw
/// `CodeObject` via the jitcode index (the `current`-frame pattern,
/// mod.rs:4664).
fn fbw_inline_callee_hazardous<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>) -> bool {
    let session = ctx.session.borrow();
    let mut seen: Vec<usize> = Vec::with_capacity(session.framestack.len());
    for frame in session.framestack.iter() {
        if seen.contains(&frame.w_code) {
            return true;
        }
        seen.push(frame.w_code);
        if let Some(idx) = crate::state::ensure_jitcode_index(frame.w_code as *const ()) {
            if let Some(raw_code) = crate::state::raw_code_for_jitcode_index(idx) {
                let code = unsafe { raw_code.as_ref() };
                if let Some(code) = code {
                    if pyre_interpreter::code_has_for_iter(code)
                        || pyre_interpreter::code_is_self_recursive(code)
                    {
                        return true;
                    }
                }
            }
        }
    }
    false
}

pub(crate) fn fbw_abort_nested_unjournaled_residual<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    pc: usize,
) -> Result<(), DispatchError> {
    // RPython `do_residual_call` runs the residual executor at any framestack
    // depth (`pyjitpl.py`). Exempt only the self-recursive
    // `CALL_ASSEMBLER` fold's concrete-stamp executor from this pyre-local
    // nested-decline guard, which is for FOREIGN unjournaled residuals.
    let in_selfrec_fold = SELFREC_CA_FOLD_ACTIVE.with(|c| c.get());
    let in_exception_string_inline = EXCEPTION_STRING_INLINE_ACTIVE.with(|c| c.get());
    // A FOR_ITER-body inline admitted under `CalleeReplaySafety::DeferredCall`
    // stands on the promise that the sub-walk commits nothing: the static scan
    // cleared every direct heap write, leaving only Python-level CALL residuals
    // whose callee the lever resolves here.  One that did not inline breaks the
    // promise, so abort BEFORE it executes — every op the sub-walk has run so
    // far is write-free, so the resume re-runs the body benignly.  Denying the
    // admitted callee makes the next attempt decline it statically, so this
    // costs one abort per callee rather than an abort per trace attempt.
    let foriter_deferred_inline = fbw_foriter_deferred_inline_outermost();
    // Narrowed decline: the general depth-≥2 nested
    // residual inline is sound now that the portal-runner ABI is correct — a
    // straight-line mutating callee inlines bit-exact.  Only two callee shapes
    // still miscompile, both masked by the old blanket decline and captured by
    // [`fbw_inline_callee_hazardous`]: a LOOP-BEARING callee (the FOR_ITER
    // Option-C refused-delivery double-advance, the `foriter_exempt_*`
    // witnesses) and a SELF-RECURSIVE callee (the hot `CALL_ASSEMBLER`
    // recursion-bridge / wasm always-portal `setintbound` type-confusion, the
    // `wasm_ca_trampoline_decline` witness).  Both are properties of the
    // framestack knowable at the residual decline point, so the whole trace
    // aborts before the hazardous body is committed.  Every other nested
    // residual inlines.  The hazard scan is last so the cheap checks
    // short-circuit it.
    if !in_selfrec_fold
        && !in_exception_string_inline
        && !ctx.session.borrow().framestack.is_empty()
        && (foriter_deferred_inline.is_some() || fbw_inline_callee_hazardous(ctx))
    {
        if let Some(callee_code_key) = foriter_deferred_inline {
            fbw_foriter_deny_deferred_call(callee_code_key);
        }
        let (outer_resume, stack_overrides) = {
            let session = ctx.session.borrow();
            match session.framestack.first().and_then(|f| f.parent.as_ref()) {
                Some(frame) => (
                    frame
                        .call_jitcode_pc
                        .map(|jit_pc| (frame.jitcode_index, jit_pc)),
                    frame.call_stack_overrides.clone(),
                ),
                None => (None, Vec::new()),
            }
        };
        FBW_ABORT_OUTER_RESUME.with(|c| c.set(outer_resume));
        FBW_ABORT_OUTER_STACK_OVERRIDES.with(|c| {
            *c.borrow_mut() = stack_overrides;
        });
        return Err(DispatchError::callee_inline_unsupported(pc));
    }
    Ok(())
}

/// Take the outer-caller CALL JitCode coordinate stashed by
/// [`fbw_abort_nested_unjournaled_residual`].  The stack overrides stay in
/// `FBW_ABORT_OUTER_STACK_OVERRIDES` (rooted by the #447 area walker,
/// `abort_overrides`) until [`fbw_abort_outer_stack_overrides_clear`]; the
/// flush reads them in place from the rooted cell so a minor collection while
/// boxing Int/Float locals forwards the very refs it writes.
pub(crate) fn fbw_abort_outer_resume_take() -> Option<(u32, usize)> {
    FBW_ABORT_OUTER_RESUME.with(|c| c.replace(None))
}

/// Run `f` with the rooted outer-frame stack overrides borrowed in place.
/// A GC during `f` forwards the cell's ref slots via the area walker's
/// `as_ptr` access, so the borrowed slice observes the forwarded values.
pub(crate) fn fbw_abort_outer_stack_overrides_with<R>(
    f: impl FnOnce(&[(usize, pyre_object::PyObjectRef)]) -> R,
) -> R {
    FBW_ABORT_OUTER_STACK_OVERRIDES.with(|c| f(&c.borrow()))
}

/// Clear the outer-frame stack overrides after the flush consumed them.
pub(crate) fn fbw_abort_outer_stack_overrides_clear() {
    FBW_ABORT_OUTER_STACK_OVERRIDES.with(|c| c.borrow_mut().clear());
}

/// Clear the nested inline abort resume latch at a walk boundary.
pub(crate) fn fbw_abort_outer_resume_reset() {
    FBW_ABORT_OUTER_RESUME.with(|c| c.set(None));
    FBW_ABORT_OUTER_STACK_OVERRIDES.with(|c| c.borrow_mut().clear());
}

/// Whether the walk recorded an effect outside the journal's reach.
pub(crate) fn fbw_has_unjournaled_effect() -> bool {
    let (value_unavailable, symbolic) = fbw_unjournaled_kinds();
    value_unavailable || symbolic
}

pub(crate) fn fbw_unjournaled_kinds() -> (bool, bool) {
    (
        FBW_UNJOURNALED_VALUE_UNAVAILABLE.with(|c| c.get()),
        FBW_UNJOURNALED_SYMBOLIC.with(|c| c.get()),
    )
}

pub(crate) fn fbw_count_executed_residual(is_void: bool, is_may_force: bool) {
    let counter = if is_void {
        &FBW_EXECUTED_RESIDUAL_VOID
    } else if is_may_force {
        &FBW_EXECUTED_RESIDUAL_MAYFORCE
    } else {
        &FBW_EXECUTED_RESIDUAL_PLAIN
    };
    counter.with(|c| c.set(c.get().wrapping_add(1)));
}

pub(crate) fn fbw_executed_residual_counts() -> (u32, u32, u32) {
    (
        FBW_EXECUTED_RESIDUAL_VOID.with(|c| c.get()),
        FBW_EXECUTED_RESIDUAL_MAYFORCE.with(|c| c.get()),
        FBW_EXECUTED_RESIDUAL_PLAIN.with(|c| c.get()),
    )
}

/// `framework.py root_walker.walk_roots` parity for the store and append
/// journals: the entries hold nursery-resident refs across the rest of the
/// walk (residual calls allocate, and a minor collection moves nursery
/// objects), so every ref slot is forwarded as a root.  Registered once via
/// `majit_gc::shadow_stack::register_extra_root_walker` at JIT init.
pub fn fbw_store_journal_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let data = capture_fbw_store_journal_root_area();
    unsafe { fbw_store_journal_root_walker_area(data, visitor) };
}

pub fn capture_fbw_store_journal_root_area() -> *const () {
    FBW_STORE_JOURNAL_ROOT_AREA.with(|area| area as *const _ as *const ())
}

/// FBW-native port of [`crate::state::ensure_boxed_for_ca`] that operates
/// purely on the [`TraceCtx`] (no borrowed `MIFrame`).  A portal-exit
/// FINISH must carry `Type::Ref` (`pyjitpl.py` REF result_type);
/// if the optimizer left the return value unboxed as Int/Float, re-box it
/// (`wrapint` / `wrapfloat` = `NewWithVtable` + `SetfieldGc`).  `value_type`
/// here is `ctx.get_opref_type(value).unwrap_or(Type::Ref)`, the exact body
/// of `MIFrame::value_type` minus the borrow.
pub(crate) fn fbw_ensure_boxed_for_ca<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    value: OpRef,
) -> Result<OpRef, DispatchError> {
    let ty = if value.is_none() {
        Type::Ref
    } else {
        ctx.trace_ctx.get_opref_type(value).unwrap_or(Type::Ref)
    };
    let _ = op_pc;
    let boxed = match ty {
        Type::Int => crate::state::wrapint(ctx.trace_ctx, value),
        Type::Float => crate::state::wrapfloat(ctx.trace_ctx, value),
        Type::Ref | Type::Void => value,
    };
    Ok(boxed)
}

/// FBW-native port of `MIFrame::store_token_in_vable` (`pyjitpl.py`).
/// Records `FORCE_TOKEN` + `SETFIELD_GC(vbox, token, vable_token_descr)`
/// via `store_token_in_vable_setfield` and, when that fires, the
/// `GUARD_NOT_FORCED_2` with resumedata captured through the walker's own
/// single-frame snapshot machinery (`walker_capture_snapshot_for_last_guard`)
/// — the same resume coordinate (`entry_py_pc` / `outer_active_boxes`) every
/// other FBW guard uses, since pyre's blackhole can only re-enter the outer
/// Python opcode boundary.  No-op when there is no standard virtualizable.
pub(crate) fn fbw_store_token_in_vable<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<(), DispatchError> {
    if ctx.trace_ctx.store_token_in_vable_setfield() {
        ctx.trace_ctx.record_guard(OpCode::GuardNotForced2, &[], 0);
        walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    }
    Ok(())
}

/// Shared top-level finish path for the three value-returning arms
/// (`ref_return` / `int_return` / `float_return`).  Re-boxes `result` to
/// `Type::Ref`, records the vable store-back + `GUARD_NOT_FORCED_2`, and
/// stashes the finish payload for `full_body_walk_trace`.  Deliberately
/// does NOT record the `FINISH` op: under the gate the compile consumer
/// (`finish_and_compile` -> `recorder.finish`, mod.rs) records it from
/// `finish_args`, so recording it here too would double it.
pub(crate) fn fbw_terminate_with_finish<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    result: OpRef,
    op_pc: usize,
) -> Result<(), DispatchError> {
    let finish_value = fbw_ensure_boxed_for_ca(ctx, op_pc, result)?;
    fbw_store_token_in_vable(ctx, op_pc)?;
    FBW_FINISH_PAYLOAD.with(|c| c.set(Some((finish_value, Type::Ref))));
    Ok(())
}

/// Void variant of [`fbw_terminate_with_finish`] for the top-level
/// `void_return/` portal exit (`compile_done_with_this_frame`'s VOID
/// branch, pyjitpl.py).  Records the vable store-back +
/// `GUARD_NOT_FORCED_2`, then stashes a `Type::Void`-marked payload so
/// [`crate::trace::full_body_walk_trace`] builds a `TraceAction::Finish`
/// with no args (`done_with_this_frame_descr_from_types(&[])` resolves the
/// void descr).  Like the value path it does NOT record the `FINISH` op —
/// the compile consumer records it from the empty `finish_args`.
pub(crate) fn fbw_terminate_void_with_finish<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<(), DispatchError> {
    fbw_store_token_in_vable(ctx, op_pc)?;
    FBW_FINISH_PAYLOAD.with(|c| c.set(Some((OpRef::NONE, Type::Void))));
    Ok(())
}

/// Exception variant of [`fbw_terminate_with_finish`] for the top-level
/// uncaught raise (`compile_exit_frame_with_exception`, pyjitpl.py).
/// Stashes the exception box (`exc`, already a `Type::Ref`) as an
/// `is_exception` payload and, when the raised exception has a concrete Ref,
/// the concrete disposition for the GC root walker / no-replay portal.  Like
/// the value path it does NOT record the `FINISH` op — [`crate::trace::
/// full_body_walk_trace`]'s Terminate arm builds
/// `TraceAction::Finish { exit_with_exception: true }` and the compile
/// consumer records it once against `exit_frame_with_exception_descr`.
pub(crate) fn fbw_terminate_with_raise(exc: OpRef, exc_concrete: ConcreteValue) {
    FBW_FINISH_PAYLOAD.with(|c| c.set(Some((exc, Type::Ref))));
    FBW_FINISH_IS_EXCEPTION.with(|c| c.set(true));
    if let ConcreteValue::Ref(p) = exc_concrete {
        if !p.is_null() {
            fbw_finish_raise_set(exc_concrete);
        }
    }
}

/// Whether the stashed `FBW_FINISH_PAYLOAD` is a top-level uncaught raise
/// (see [`fbw_terminate_with_raise`]).  Read by the Terminate arm before
/// taking the payload; reset with the payload at the start of every walk.
pub(crate) fn fbw_finish_is_exception() -> bool {
    FBW_FINISH_IS_EXCEPTION.with(|c| c.get())
}

/// Map an `abort_permanent` marker's jitcode pc back to the Python opcode
/// the interpreter must resume at.  `emit_abort_permanent` (codewriter)
/// anchors the graph marker at `py_pc` and additionally stores
/// `last_instr = py_pc - 1` for portal frames; the full-body walk reads the
/// marker coordinate here to flush the abort-point frame instead of replaying
/// the walked region.  Returns None when the sym's jitcode / `code_ptr` is
/// unavailable (no resume coordinate derivable → legacy replay).
pub(crate) fn fbw_abort_resume_py_pc<Sym: WalkSym>(
    sym: &Sym,
    abort_jit_pc: usize,
) -> Option<usize> {
    if sym.jitcode().is_null() {
        return None;
    }
    // SAFETY: read-only access to the sym's immutable jitcode layout, live
    // for the walk that produced `abort_jit_pc`.
    let jc = unsafe { &*sym.jitcode() };
    if jc.payload.code_ptr.is_null() {
        return None;
    }
    Some(python_pc_for_jitcode_pc(&jc.payload.metadata, abort_jit_pc) as usize)
}

/// Every pc in `body_code` that some op can branch to: the `goto` family and
/// `catch_exception` carry their target as the label operand, and
/// `int_*_jump_if_ovf` carries an overflow target ahead of its operands.
///
/// `None` when an op carries a label this decode cannot locate — a var-list
/// or a pyre payload ahead of the `L` — since a missed target would let a
/// freshness claim survive a join it does not hold across.
fn body_branch_targets(body_code: &[u8]) -> Option<std::collections::HashSet<usize>> {
    let mut targets = std::collections::HashSet::new();
    let mut pc = 0usize;
    while pc < body_code.len() {
        let d = crate::jitcode_runtime::decode_op_at(body_code, pc)?;
        if d.argcodes.contains('L') {
            // Operand widths follow `decode_op_at`; only the fixed-width forms
            // can precede the label, so anything else gives up.
            let mut cursor = d.pc + 1;
            let mut target = None;
            for operand in d.argcodes.chars() {
                match operand {
                    'L' => {
                        target = Some(u16::from_le_bytes([
                            *body_code.get(cursor)?,
                            *body_code.get(cursor + 1)?,
                        ]) as usize);
                        break;
                    }
                    'i' | 'c' | 'r' | 'f' => cursor += 1,
                    'd' | 'j' => cursor += 2,
                    _ => break,
                }
            }
            targets.insert(target?);
        }
        pc = d.next_pc;
    }
    Some(targets)
}

/// Replay safety of one inline candidate's body inside a FOR_ITER body, as
/// judged by [`fbw_callee_body_replay_safety`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CalleeReplaySafety {
    /// No op in the body can commit a live-heap effect.
    Clean,
    /// Clean apart from Python-level CALL residuals, whose callee is resolved
    /// only at walk time.
    DeferredCall,
    /// Carries a live-heap effect a replay would double.
    Dirty,
}

/// Whether an inline callee can be replayed from its caller's CALL boundary
/// without duplicating a live-heap effect.  The inline sub-walk's deopt
/// snapshot does not yet carry its own callee frame, so this is deliberately
/// stricter than ordinary inlining: every live-heap write declines up front.
///
/// A Python-level CALL residual is the one shape this static scan cannot
/// settle: its callee is a runtime value, so whether the sub-walk inlines it
/// (leaving nothing to replay) or executes it (which may write) is known only
/// at the call.  Those bodies report [`CalleeReplaySafety::DeferredCall`] and
/// the lever decides at the call — see
/// [`fbw_abort_nested_unjournaled_residual`], which aborts before executing a
/// residual that did not inline.  Every other unproven residual is `Dirty`.
///
/// A `new_with_vtable/d>r` or `new_array*` result is fresh within this body.
/// A `setfield_gc` initialization write into one is benign only when the
/// target field is immutable (`wrapint` is the important instance,
/// `W_IntObject.intval`); a `setarrayitem_gc` into a fresh array is benign
/// outright, since replay writes the replay's own array (`BUILD_TUPLE` /
/// `BUILD_LIST` fill their backing block this way).  Freshness may pass
/// through `ref_copy`, but every other Ref-producing instruction clears it,
/// so a later store cannot accidentally be classified as an initialization of
/// an earlier allocation, and every branch target drops the whole set — a
/// register reaching a join can hold whichever allocation the taken path put
/// there, which this straight-line scan cannot name.
pub(crate) fn fbw_callee_body_replay_safety(
    body_code: &[u8],
    args_all_numeric: bool,
    num_regs_i: usize,
    constants_i: &[i64],
    callee_descr_refs: &[DescrRef],
) -> CalleeReplaySafety {
    let Some(branch_targets) = body_branch_targets(body_code) else {
        return CalleeReplaySafety::Dirty;
    };
    let mut fresh_ref_regs = [false; u8::MAX as usize + 1];
    let mut deferred_call = false;
    let mut pc = 0usize;
    while pc < body_code.len() {
        if branch_targets.contains(&pc) {
            fresh_ref_regs = [false; u8::MAX as usize + 1];
        }
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return CalleeReplaySafety::Dirty;
        };

        if d.opname.starts_with("residual_call") {
            let Some(descr_index) = residual_call_descr_index_in_body(body_code, &d) else {
                return CalleeReplaySafety::Dirty;
            };
            let Some(call_descr) = callee_descr_refs
                .get(descr_index)
                .and_then(|descr| descr.as_call_descr())
            else {
                return CalleeReplaySafety::Dirty;
            };
            let ei = call_descr.get_extra_info();
            // `ForIterNext` is deliberately not accepted here: it advances the
            // shared heap iterator irreversibly (no journal undo), so replaying
            // a callee that contains it from the caller's CALL boundary would
            // double-consume.  A FOR_ITER-bearing body is declined anyway — its
            // mandatory `GET_ITER` (`MayForce`) predecessor fails this scan
            // first — so this only removes a latent landmine, not live inlines.
            // `load_const` / `load_global` / `box_int` are tagged `CanRaise`
            // only to keep the `_OS_CANRAISE` invariant (effectinfo.rs); each
            // is a read or a fresh allocation, so re-running one commits
            // nothing to the live heap.  The BUILD_TUPLE / BUILD_LIST array
            // consumers are the same shape one level up: they read a
            // freshly-built backing array and return a brand-new container.
            let replay_safe_read = matches!(
                ei.pyre_helper,
                majit_ir::PyreHelperKind::LoadConst
                    | majit_ir::PyreHelperKind::LoadGlobal
                    | majit_ir::PyreHelperKind::BoxInt
                    | majit_ir::PyreHelperKind::NewtupleFromArray
                    | majit_ir::PyreHelperKind::NewlistFromArray
            );
            let provably_side_effect_free = replay_safe_read
                || ei.check_is_elidable()
                || ei.extraeffect == majit_ir::ExtraEffect::LoopInvariant;
            if !provably_side_effect_free
                && !residual_call_is_specialized_plain_int_add(
                    body_code,
                    args_all_numeric,
                    &d,
                    num_regs_i,
                    constants_i,
                    callee_descr_refs,
                )
            {
                // A Python-level CALL is the one shape this scan cannot
                // settle: the inline lever binds its callee only at the call,
                // so whether it leaves a residual behind — and what that
                // residual writes — is not a property of this body.  Defer it;
                // the backstop aborts before executing one that did not
                // inline.
                if matches!(
                    ei.pyre_helper,
                    majit_ir::PyreHelperKind::CallFn
                        | majit_ir::PyreHelperKind::CallKw
                        | majit_ir::PyreHelperKind::CallFunctionEx
                ) {
                    deferred_call = true;
                } else {
                    return CalleeReplaySafety::Dirty;
                }
            }
        } else if d.opname.starts_with("setfield_gc") {
            // Canonical setfield shapes are `r<value>d`: the target ref is
            // operand 0 and the field descr is operand 2.
            let Some(&target_reg) = body_code.get(d.pc + 1) else {
                return CalleeReplaySafety::Dirty;
            };
            let descr_index = decode_descr_index(body_code, &d, 2);
            let immutable_field = callee_descr_refs
                .get(descr_index)
                .and_then(|descr| descr.as_field_descr())
                .is_some_and(|field| field.is_immutable());
            if !fresh_ref_regs[target_reg as usize] || !immutable_field {
                return CalleeReplaySafety::Dirty;
            }
        } else if d.opname.starts_with("setarrayitem_gc") {
            // The dual of the `setfield_gc` rule: a store into an array this
            // body just allocated is an initialization, not a live-heap write,
            // so replaying it writes the replay's own fresh array.  The
            // canonical shapes put the array register in operand 0 (`r…`); the
            // `iiid` raw-address form carries no array register to prove fresh.
            let target_fresh = d.argcodes.starts_with('r')
                && body_code
                    .get(d.pc + 1)
                    .is_some_and(|reg| fresh_ref_regs[*reg as usize]);
            if !target_fresh {
                return CalleeReplaySafety::Dirty;
            }
        } else if d.opname.starts_with("setinteriorfield_gc")
            || d.opname.starts_with("raw_store")
            || d.opname.starts_with("cond_call")
            || d.opname.starts_with("call_assembler")
            || d.opname.starts_with("inline_call")
        {
            // Interior/raw stores and non-residual call forms cannot be proven
            // replay-safe from this single callee body.
            return CalleeReplaySafety::Dirty;
        }

        // The result byte is always the final operand for `>r` forms.
        if d.argcodes.ends_with(">r") {
            let Some(&dst) = body_code.get(d.next_pc.saturating_sub(1)) else {
                return CalleeReplaySafety::Dirty;
            };
            fresh_ref_regs[dst as usize] = d.key == "new_with_vtable/d>r"
                || d.opname.starts_with("new_array")
                || (d.key == "ref_copy/r>r"
                    && body_code
                        .get(d.pc + 1)
                        .is_some_and(|src| fresh_ref_regs[*src as usize]));
        }
        pc = d.next_pc;
    }
    if deferred_call {
        CalleeReplaySafety::DeferredCall
    } else {
        CalleeReplaySafety::Clean
    }
}

pub(crate) fn fbw_callee_body_has_binary_op_residual(
    body_code: &[u8],
    callee_descr_refs: &[DescrRef],
) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(op) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return false;
        };
        if op.opname.starts_with("residual_call")
            && residual_call_descr_index_in_body(body_code, &op)
                .and_then(|index| callee_descr_refs.get(index))
                .and_then(|descr| descr.as_call_descr())
                .is_some_and(|descr| {
                    descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::BinaryOp
                })
        {
            return true;
        }
        pc = op.next_pc;
    }
    false
}
