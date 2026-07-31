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
/// A callee that raises inline below an intermediate frame is capped separately,
/// to the top inline level by `callee_body_contains_raise`: its unwind needs the
/// cross-frame bridge (gh#343 / gh#467) the drain cannot yet build.
///
/// A self-recursive callee is bounded instead by
/// [`fbw_inline_recursion_count`] against `max_unroll_recursion`, mirroring
/// `opimpl_recursive_call` (`pyjitpl.py:1390-1416`) folding the recursive call
/// straight to `CALL_ASSEMBLER` past the bound rather than continuing to unroll
/// the call tree.
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

/// Recursion depth of `w_code` on the walk's framestack —
/// `opimpl_recursive_call` (`pyjitpl.py:1390-1402`) counting the portal frames
/// already on `MetaInterp.framestack` before comparing against
/// `max_unroll_recursion`.
///
/// Upstream compares the whole greenkey element-wise (`pyjitpl.py:1396-1401`
/// `gk[i].same_constant(greenboxes[i])`) over `(next_instr, is_being_profiled,
/// bytecode)` (`interp_jit.py:34`); matching `w_code` alone is that comparison,
/// because every entry this scan can see carries `next_instr == 0`:
///
/// * `MIFrame.setup` (`pyjitpl.py:74-80`) assigns `greenkey` once and never
///   updates it, so a frame's greenkey stays its ENTRY greens however far its
///   pc has since advanced;
/// * every [`InlineFrame`] is pushed by `InlineFrameGuard::enter` at a callee
///   entry, and the bridge-reconstructed frames stamp the same `(w_code, 0)`
///   identity (`state.rs` `reconstruct_inline_recipe`).
///
/// The root frame is deliberately absent from this framestack, which is also
/// what upstream's greenkey compare achieves: the root's greenkey holds the
/// merge-point pc, so it fails `same_constant` against a call's `next_instr ==
/// 0` greens and is not counted either.
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

    /// `(frame, last_instr_before)` for the concrete half of
    /// [`fbw_publish_exit_last_instr`], so a walk that does not commit can put
    /// the field back.
    ///
    /// The publish fires when the walk reaches the exit; whether that exit is
    /// kept is decided afterwards, in the walk-end epilogue.  A declined walk
    /// returns to a replay that resumes the frame from the pre-walk state, and
    /// `last_instr` is exactly what that resume reads
    /// (`PyFrame::next_instr` = `last_instr + 1`), so an exit coordinate left
    /// behind would restart the frame PAST its own return or raise.  The
    /// recorded `setfield_vable_i` half needs no undo — it only reaches a frame
    /// on a compiled run.
    ///
    /// Only the first publish of a walk is recorded, so the restore targets the
    /// value the frame carried when the walk began rather than an intermediate
    /// one.  Cleared with the store journal at the start of every walk.
    static FBW_EXIT_LAST_INSTR_UNDO: std::cell::Cell<Option<(usize, isize)>> =
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
    FBW_TRACEBACK_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
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
    FBW_EXIT_LAST_INSTR_UNDO.with(|c| c.set(None));
}

/// Put `last_instr` back for a walk that did not commit its end state, so the
/// replay resumes where the frame stood before the walk.  Runs beside
/// [`fbw_store_journal_rollback`] on every non-committed exit; the commit side
/// just drops the undo ([`fbw_exit_last_instr_commit`]).
pub(crate) fn fbw_exit_last_instr_rollback() {
    let Some((frame, before)) = FBW_EXIT_LAST_INSTR_UNDO.with(|c| c.take()) else {
        return;
    };
    // SAFETY: the frame the publish wrote is the walk's live recording frame,
    // which outlives the walk, and `frame_layout` pins `last_instr` to this
    // offset with a compile-time assertion against the interpreter's constant.
    unsafe {
        *((frame + crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET) as *mut isize) = before;
    }
}

/// Drop the undo: the walk's end state is kept, so the published exit
/// coordinate is the one the frame should carry.
pub(crate) fn fbw_exit_last_instr_commit() {
    FBW_EXIT_LAST_INSTR_UNDO.with(|c| c.set(None));
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

/// Read an exception's concrete traceback head before a bridge-entry recording
/// helper runs.  `None` means the concrete is not an exception and therefore
/// cannot receive a host-side attach.
pub(crate) fn fbw_traceback_journal_head(
    exception: pyre_object::PyObjectRef,
) -> Option<pyre_object::PyObjectRef> {
    if exception.is_null() || unsafe { !pyre_object::is_exception(exception) } {
        return None;
    }
    Some(unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exception) })
}

/// Journal the node a bridge-entry recording helper concretely prepended.
/// An unchanged head means the helper declined the host attach.
pub(crate) fn fbw_traceback_journal_push_if_attached(
    exception: pyre_object::PyObjectRef,
    previous_head: Option<pyre_object::PyObjectRef>,
) {
    let Some(previous_head) = previous_head else {
        return;
    };
    let node = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exception) };
    if node == previous_head {
        return;
    }
    debug_assert!(!node.is_null());
    debug_assert!(unsafe { pyre_interpreter::pytraceback::is_pytraceback(node) });
    debug_assert_eq!(
        unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_w_next(node) },
        previous_head
    );
    // A carrier sub-walk that keeps its journals across the root continuation
    // attaches once per entered handler, so the log grows rather than asserting
    // a single attach; the rollback unwinds the entries in reverse push order.
    FBW_TRACEBACK_STORE_JOURNAL.with(|j| j.borrow_mut().push((exception, node)));
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
    // The compiled trace owns the recorded attach and the authoritative walk
    // already applied this iteration's concrete node, so keep it in place.
    FBW_TRACEBACK_STORE_JOURNAL.with(|j| j.borrow_mut().clear());
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

/// Resolved `body_pc` of the most-recent (top) in-flight FOR_ITER entry — the
/// FOR_ITER continue-arm fallthrough, i.e. the pc of the loop-variable store
/// that binds the just-consumed item.  `None` when no item is in flight or its
/// coordinate is an unresolvable native pc.
pub(crate) fn fbw_foriter_inflight_top_body_pc() -> Option<usize> {
    FBW_FORITER_INFLIGHT.with(|c| {
        c.borrow()
            .last()
            .and_then(|e| inflight_foriter_body_pc(e.body))
    })
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
    // Splice every journaled bridge-entry node back out of its exception's
    // chain.  The node is not necessarily still the head: concrete execution
    // after the attach can prepend further nodes to the same exception (a
    // handler that re-raises, or another raising callee), and dropping the undo
    // in that case would leave the speculative node behind for the replay to
    // record the catching frame on top of — a duplicated frame in the reported
    // traceback.  Newer heads are preserved; only the journaled link is
    // removed, in reverse push order so the chain collapses to its pre-walk
    // shape.
    FBW_TRACEBACK_STORE_JOURNAL.with(|j| {
        let mut entries = j.borrow_mut();
        while let Some((exception, node)) = entries.pop() {
            let next = unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_w_next(node) };
            let head =
                unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exception) };
            if head == node {
                unsafe {
                    pyre_object::interp_exceptions::w_exception_set_traceback(exception, next);
                }
                continue;
            }
            let mut prev = head;
            while !prev.is_null() && unsafe { pyre_interpreter::pytraceback::is_pytraceback(prev) }
            {
                let curr = unsafe { pyre_interpreter::pytraceback::w_pytraceback_get_w_next(prev) };
                if curr == node {
                    // The removed node sat between `prev` and `next`, so the
                    // shortened chain cannot reach `prev` again and the
                    // setter's loop check always passes.
                    let _ = unsafe {
                        pyre_interpreter::pytraceback::w_pytraceback_set_w_next(prev, next)
                    };
                    break;
                }
                prev = curr;
            }
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

/// gh#467 latch the inline-abort forward-flush carrier (see
/// [`FBW_ABORT_CALL_RESUME`]).
///
/// Does not displace an already-latched `MidBody` carrier — it is stored inside
/// it as [`MidBodyPayload::entry_fallback`] instead.  Rebuilding the callee at
/// its own pc and resuming the caller past its call is what upstream does
/// (`blackhole.py:1799-1821`, `:1653-1662`); rewinding the caller TO the call
/// has no upstream counterpart, so it stands in only for a callee the rebuild
/// could not describe or could not flush.  Both sites are `is_top_inline` on an
/// aborting sub-walk, which ends the walk, so at most one of each is latched
/// per walk, and both read the same outer CALL coordinate.
pub(crate) fn fbw_set_abort_call_resume(
    outer_jitcode_index: u32,
    call_jitcode_pc: usize,
    stack: Vec<pyre_object::PyObjectRef>,
) {
    FBW_ABORT_CALL_RESUME.with(|c| {
        let mut slot = c.borrow_mut();
        if let Some(InlineAbortCarrier::MidBody(payload)) = slot.as_mut() {
            if payload.outer_jitcode_index == outer_jitcode_index
                && payload.call_jitcode_pc == call_jitcode_pc
            {
                payload.entry_fallback = Some(crate::jitcode_dispatch::EntryFallback {
                    call_stack: stack,
                    entry_executed_effects: fbw_executed_effect_count(),
                });
            }
            return;
        }
        *slot = Some(InlineAbortCarrier::Entry {
            outer_jitcode_index,
            call_jitcode_pc,
            call_stack: stack,
            // The latch is only set at the CALL, and only under the caller's
            // zero-delta gate, so the odometer read here IS the count at the
            // pc this carrier resumes at.
            entry_executed_effects: fbw_executed_effect_count(),
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
    /// Code keys of the callees [`fbw_inline_callee_hazardous`] named when the
    /// hazard arm of [`fbw_abort_nested_unjournaled_residual`] fired.  The
    /// inline callsite declines them from then on, so the call residualizes
    /// and the enclosing trace never re-enters the identical abort.
    ///
    /// This is `disable_noninlinable_function` (`warmstate.py:331`, which sets
    /// `JC_DONT_TRACE_HERE` = "do not inline calls to this function"): upstream
    /// answers an abort attributable to one inlined callee by denying THAT
    /// callee and retracing the enclosing loop, not by penalising the loop
    /// (`pyjitpl.py:2818-2828`).  Like the upstream flag the set has no removal
    /// path — the hazard is a static property of the callee's `CodeObject`.
    ///
    /// Keying on the `CodeObject` alone is the full key for this decision, not
    /// a truncation of one.  The flag is consumed by `can_inline_callable`
    /// (`warmstate.py:669-677`), whose only caller is `_opimpl_recursive_call`
    /// (`pyjitpl.py:1376-1382`) — it passes the CALLEE's green args, and a
    /// callee reached through a CALL is always entered at its own entry, so the
    /// `next_instr` component is constant and `pycode` carries the whole
    /// decision.  The same holds here: the deny is recorded and queried for an
    /// inline frame pushed at a CALL boundary (`inline_call.rs`), never for a
    /// mid-body resume.
    ///
    /// Per-thread for the same reason as [`crate::trace::fbw_declined`]'s
    /// `FBW_DECLINED_KEYS` and `RANGE_FORITER_DEMOTED`: pyre's walk state is
    /// per-thread, and an inline hazard observed while tracing is a property of
    /// the tracing thread's framestack.  Sharing one memo while its siblings
    /// stay per-thread would be the inconsistency.
    ///
    /// NOT yet ported: `warmstate.py:485-495` also treats the flag as "please
    /// trace from here as soon as possible" — a denied cell that never had a
    /// procedure token reaches `bound_reached` immediately, so the callee gets
    /// its own trace instead of staying a plain residual forever.  Since
    /// residual calls re-enter the JIT the callee does reach its own threshold,
    /// just on the ordinary counter rather than at once.
    ///
    /// A raw address is a sound permanent key only because `w_code_new`
    /// (`pycode.rs`) allocates every `PyCode` with `Box::into_raw` and nothing
    /// frees it: the address is unique for the process and never moves.
    /// Upstream can key on the object because a `JitCell` holds its greens and
    /// `should_remove_jitcell` (`warmstate.py:212`) prunes dead ones; this set
    /// has neither, so it relies on that immortality.  `eval.rs`'s `PyCode`
    /// registration names the change that would end it — switching `w_code_new`
    /// to `try_gc_alloc_stable`.  At that point a reclaimed address can be
    /// handed to a later code object and this set must gain a removal path or
    /// a key that outlives the allocation.
    static FBW_HAZARDOUS_INLINE_DENY: std::cell::RefCell<std::collections::HashSet<usize>> =
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

pub(crate) fn fbw_hazardous_inline_denied(callee_code_key: usize) -> bool {
    FBW_HAZARDOUS_INLINE_DENY.with(|c| c.borrow().contains(&callee_code_key))
}

fn fbw_deny_hazardous_inline(callee_code_key: usize) {
    FBW_HAZARDOUS_INLINE_DENY.with(|c| {
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
///
/// Returns the code key of the offending callee, which is the entity the
/// decline is a property of and therefore the one to deny — the same
/// attribution `find_biggest_function` (`pyjitpl.py:3538`) performs before
/// `disable_noninlinable_function`.  Declining it at its own callsite makes
/// the next attempt residualize that call, so the surviving nest is
/// hazard-free and the enclosing loop can compile.
fn fbw_inline_callee_hazardous<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>) -> Option<usize> {
    let session = ctx.session.borrow();
    let mut seen: Vec<usize> = Vec::with_capacity(session.framestack.len());
    for frame in session.framestack.iter() {
        if seen.contains(&frame.w_code) {
            return Some(frame.w_code);
        }
        seen.push(frame.w_code);
        if let Some(idx) = crate::state::ensure_jitcode_index(frame.w_code as *const ()) {
            if let Some(raw_code) = crate::state::raw_code_for_jitcode_index(idx) {
                let code = unsafe { raw_code.as_ref() };
                if let Some(code) = code {
                    if pyre_interpreter::code_has_for_iter(code)
                        || pyre_interpreter::code_is_self_recursive(code)
                    {
                        return Some(frame.w_code);
                    }
                }
            }
        }
    }
    None
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
    // A carrier-resume sub-walk starts at the failed guard; it does not replay
    // an enclosing CALL. RPython resumes residual calls at every rebuilt
    // framestack depth, so this forward-capture hazard excludes the carrier.
    let nested = !ctx.fbw_mode.carrier_resume
        && !in_selfrec_fold
        && !in_exception_string_inline
        && !ctx.session.borrow().framestack.is_empty();
    let hazardous_callee = if nested && foriter_deferred_inline.is_none() {
        fbw_inline_callee_hazardous(ctx)
    } else {
        None
    };
    if nested && (foriter_deferred_inline.is_some() || hazardous_callee.is_some()) {
        if let Some(callee_code_key) = foriter_deferred_inline {
            fbw_foriter_deny_deferred_call(callee_code_key);
        }
        // Deny the named callee so the enclosing loop's next attempt
        // residualizes that call instead of re-entering this abort.  Without
        // it the decline is a property of the framestack, which the next
        // attempt rebuilds identically: the abort recurs byte-for-byte until
        // the enclosing location is retired, so the loop never compiles at
        // all.  Upstream answers the same situation by denying the callee and
        // letting the enclosing loop retrace (`pyjitpl.py:2818-2828`).
        if let Some(callee_code_key) = hazardous_callee {
            fbw_deny_hazardous_inline(callee_code_key);
        }
        // The flush this latch feeds resumes the OUTERMOST caller at the CALL
        // that entered the inline region, re-executing that call from scratch,
        // while the walk's store journal is committed — a
        // `WalkEndResume::Rewind` leg.  So it is sound only while the inline
        // region has executed nothing irreversible: an executed-effect delta
        // means the call would apply its effects a second time on top of the
        // committed ones.  Same zero-delta gate the entry carrier applies at
        // its own CALL (`try_walker_inline_user_call`) and the contract
        // `FBW_EXECUTED_EFFECT_COUNT` documents; declining here leaves the
        // legacy path, whose journal rollback makes the replay exactly-once.
        // The snapshot travels with the latch so `commit_walk_end` re-checks
        // it at the commit point, not just here.
        let (outer_resume, stack_overrides) = {
            let session = ctx.session.borrow();
            let outermost = session
                .framestack
                .first()
                .filter(|f| fbw_executed_effect_count() == f.entry_executed_effects);
            match outermost.and_then(|f| f.parent.as_ref()) {
                Some(frame) => (
                    frame.call_jitcode_pc.map(|jit_pc| {
                        (
                            frame.jitcode_index,
                            jit_pc,
                            crate::jitcode_dispatch::fbw_executed_effect_count(),
                        )
                    }),
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
///
/// The third element is the executed-effect odometer at the outer CALL this
/// resumes at — `WalkEndResume::Rewind`'s `effects_at_resume_point`.
pub(crate) fn fbw_abort_outer_resume_take() -> Option<(u32, usize, usize)> {
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
/// `Type::Ref`, publishes the return coordinate into `last_instr`, stores the
/// virtualizable back into the frame, and stashes the finish payload for
/// `full_body_walk_trace`.  The store-back is what leaves `vable_token` clear,
/// so the `GUARD_NOT_FORCED_2` arming below it declines.  Deliberately
/// does NOT record the `FINISH` op: under the gate the compile consumer
/// (`finish_and_compile` -> `recorder.finish`, mod.rs) records it from
/// `finish_args`, so recording it here too would double it.
pub(crate) fn fbw_terminate_with_finish<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    result: OpRef,
    op_pc: usize,
) -> Result<(), DispatchError> {
    let finish_value = fbw_ensure_boxed_for_ca(ctx, op_pc, result)?;
    fbw_publish_exit_last_instr(ctx, op_pc);
    fbw_force_virtualizable_before_return(ctx);
    fbw_store_token_in_vable(ctx, op_pc)?;
    FBW_FINISH_PAYLOAD.with(|c| c.set(Some((finish_value, Type::Ref))));
    Ok(())
}

/// `jit.hint(frame, force_virtualizable=True)` on the way out of the portal
/// (`opimpl_hint_force_virtualizable` → `gen_store_back_in_vable`, pyjitpl.py).
///
/// `doc/jit/virtualizable.rst` names this as the remedy for exactly this shape:
/// "If you have something equivalent of a Python generator, where the
/// virtualizable survives for longer, you want to force it before returning.
/// It's better to do it that way than by an external call some time later."
///
/// Upstream applies it to ONE exit — `interp_jit.py` `PyFrame.dispatch` reads
/// `except Yield: … jit.hint(self, force_virtualizable=True)` against a bare
/// `except Return: return self.popvalue()`.  A generator frame is the only one
/// that outlives its dispatch there; every other frame is answered lazily,
/// through the marker `store_token_in_vable` leaves behind and the deadframe it
/// names.  So this fires on an exit upstream leaves alone, and the ordinary
/// return gives up the `FORCE_TOKEN`/`GUARD_NOT_FORCED_2` protocol for an
/// unconditional store-back.
///
/// A frame the function-entry portal compiled can outlive its trace the same
/// way a generator's does — a traceback it hands out keeps it alive — and the
/// lazy route is not available to narrow this back down.  Two things have to
/// land before it is.  The dynasm backend frees the jitframe chain before
/// `execute_token` returns, so the marker would name freed memory rather than
/// a retained deadframe.  And arming `jf_force_descr` for a standalone
/// trailing `GUARD_NOT_FORCED_2` is uneven: cranelift does it, dynasm folds
/// the opcode into the no-args guard bucket and so does not, and wasm cannot
/// yet — its `FORCE_TOKEN` is a zero sentinel, leaving no frame identity for a
/// token to name.  Upstream arms it on every backend, from
/// `consider_guard_not_forced_2` (x86/regalloc.py), so until dynasm and wasm
/// follow, the armed-token test answers false for a portal exit even once the
/// chain is retained.  Narrowing the force to the frames that actually escape
/// needs both; the escape is a runtime property, which is what the token
/// protocol answers.
///
/// Storing back here is what makes the token store unnecessary rather than
/// merely redundant: `gen_store_back_in_vable` sets `forced_virtualizable`, and
/// `store_token_in_vable` returns early on that (pyjitpl.py) — the two are
/// alternatives, not a sequence — and its final store zeroes the token slot.
fn fbw_force_virtualizable_before_return<Sym: WalkSym>(ctx: &mut WalkContext<'_, '_, Sym>) {
    let Some(vbox) = ctx.trace_ctx.standard_virtualizable_box() else {
        return;
    };
    ctx.trace_ctx.gen_store_back_in_vable(vbox);
}

/// Void variant of [`fbw_terminate_with_finish`] for the top-level
/// `void_return/` portal exit (`compile_done_with_this_frame`'s VOID
/// branch, pyjitpl.py).  Publishes the return coordinate and stores the
/// virtualizable back the same way, then stashes a `Type::Void`-marked payload so
/// [`crate::trace::full_body_walk_trace`] builds a `TraceAction::Finish`
/// with no args (`done_with_this_frame_descr_from_types(&[])` resolves the
/// void descr).  Like the value path it does NOT record the `FINISH` op —
/// the compile consumer records it from the empty `finish_args`.
pub(crate) fn fbw_terminate_void_with_finish<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<(), DispatchError> {
    fbw_publish_exit_last_instr(ctx, op_pc);
    fbw_force_virtualizable_before_return(ctx);
    fbw_store_token_in_vable(ctx, op_pc)?;
    FBW_FINISH_PAYLOAD.with(|c| c.set(Some((OpRef::NONE, Type::Void))));
    Ok(())
}

/// Publish the exiting instruction's Python coordinate into the standard
/// virtualizable's `last_instr` slot before the top-level frame exits.
///
/// Compiled code never runs the per-opcode interpreter store, so a frame
/// entered through the function-entry portal still carries the `-1`
/// initialization sentinel and a loop-entry frame carries the loop header.
/// `offset2lineno` answers `-1` with the code object's first line — the `def`
/// line — so whatever reads the field afterwards reports a line the frame was
/// not executing.
///
/// Both frame exits publish, because both leave a reader behind.  On the
/// uncaught raise, `handle_exception` (pyre-interpreter) stamps the traceback
/// node from `frame.last_instr` and keys the exception-table lookup on the
/// same field.  On the normal return, a traceback the frame handed out
/// outlives it, and `tb_frame.f_lineno` resolves through `offset2lineno` on
/// this field — the return is the last coordinate the frame ever reached.
///
/// Upstream never faces either shape: the portal is entered only from a
/// backward jump (`can_enter_jit`, `interp_jit.py`), so a loop-free function
/// is never compiled as one, and the frame's `dispatch` loop runs inside the
/// traced portal where every opcode writes `last_instr` (`pyopcode.py`).
/// pyre's function-entry portal reaches the field from the interpreter after
/// the trace has finished, so the coordinate has to be published before it
/// does.
///
/// The store is the static-field shape `gen_store_back_in_vable` emits for
/// this slot, so it reaches the frame on a compiled run; the shadow mirror
/// keeps the walker's own virtualizable view in step with it.
pub(crate) fn fbw_publish_exit_last_instr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode_position: usize,
) {
    let (recording_frame_ptr, jitcode_index) = {
        let session = ctx.session.borrow();
        (session.recording_frame_ptr, session.recording_jitcode_index)
    };
    let Some(py_pc) =
        crate::state::python_pc_for_jitcode_pc_public(jitcode_index, opcode_position as i32)
    else {
        return;
    };
    let Some(vbox) = ctx.trace_ctx.standard_virtualizable_box() else {
        return;
    };
    let Some(info) = ctx.trace_ctx.virtualizable_info().cloned() else {
        return;
    };
    let Some(field_index) = info.static_field_index_by_name("last_instr") else {
        return;
    };
    let value = ctx.trace_ctx.const_int(i64::from(py_pc));
    ctx.trace_ctx
        .vable_setfield_descr(vbox, value, info.static_field_descr(field_index));
    crate::trace_opcode::mirror_vable_static_to_boxes(
        ctx.trace_ctx,
        "last_instr",
        value,
        Value::Int(i64::from(py_pc)),
    );
    // The recorded store has to have a concrete counterpart.  Upstream's
    // tracing IS the interpreter, so its per-opcode `last_instr` write
    // (`pyopcode.py`) lands in the real frame on the very iteration the trace
    // is recorded from; the walker only records ops, so without this the
    // recording iteration is the one iteration that still reports the stale
    // sentinel — a single wrong answer in the middle of a survey.
    // `recording_frame_ptr` is the LIVE frame, not `virtualizable_heap_ptr`'s
    // trace-stepping snapshot: the snapshot's storage is released when tracing
    // ends, so a store there reaches nothing the interpreter goes on to read.
    // Unlike the recorded store, this one lands whether or not the walk goes on
    // to commit, so it is journaled: a declined walk resumes the frame from its
    // pre-walk state and reads this very field to find the next instruction.
    if recording_frame_ptr != 0 {
        // SAFETY: the recording frame is the live `PyFrame` this walk steps,
        // and `frame_layout` pins `last_instr` to this offset with a
        // compile-time assertion against the interpreter's own constant.
        let slot =
            (recording_frame_ptr + crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET) as *mut isize;
        FBW_EXIT_LAST_INSTR_UNDO.with(|c| {
            if c.get().is_none() {
                c.set(Some((recording_frame_ptr, unsafe { *slot })));
            }
        });
        unsafe {
            *slot = py_pc as isize;
        }
    }
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
/// `last_instr = py_pc - 1` into the frame red; the full-body walk reads the
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
/// How many of a callee frame's `localsplus` slots
/// [`fbw_callee_body_replay_safety`] tracks.  A body reaching past this keeps
/// working; its higher slots simply prove nothing.
const BODY_TRACKED_FRAME_SLOTS: usize = 256;

/// Resolve a jitcode `i` operand to the immutable constant it names.  `None`
/// when it indexes a live int register instead, whose value this static scan
/// cannot know.
fn body_int_operand_constant(ireg: u8, num_regs_i: usize, constants_i: &[i64]) -> Option<i64> {
    (ireg as usize)
        .checked_sub(num_regs_i)
        .and_then(|index| constants_i.get(index))
        .copied()
}

/// Which of the tracked frame slots this body can store into.
///
/// A slot the body never writes still holds exactly the argument the
/// exact-positional entry convention bound from the caller, on every path
/// through the body — so unlike a register, its provenance is not something a
/// join can invalidate.  `def leaf(i): if i % 3 == 1: ... ; return i % 5` needs
/// this: the second `i` is read after a branch target, and without it the slot
/// reset drops the caller's exact-int proof and the second `BINARY_OP`
/// residualizes the whole callee.
///
/// A store this scan cannot resolve to a slot number could name any of them, so
/// it gives the whole window up.  Same for an undecodable body — the main scan
/// answers `Dirty` on it anyway.
fn body_stored_frame_slots(
    body_code: &[u8],
    num_regs_i: usize,
    constants_i: &[i64],
) -> [bool; BODY_TRACKED_FRAME_SLOTS] {
    let mut written = [false; BODY_TRACKED_FRAME_SLOTS];
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return [true; BODY_TRACKED_FRAME_SLOTS];
        };
        if d.key.starts_with("setarrayitem_vable_r") {
            // `ri…`: the frame is operand 0 and the slot operand 1.  The frame
            // register is deliberately not matched against the scan's
            // `vable_reg` here — a store this pass cannot attribute is treated
            // as reaching every slot, which is the conservative direction.
            let slot = d
                .argcodes
                .starts_with("ri")
                .then(|| body_code.get(d.pc + 2))
                .flatten()
                .and_then(|ireg| body_int_operand_constant(*ireg, num_regs_i, constants_i))
                .and_then(|slot| usize::try_from(slot).ok());
            match slot {
                Some(slot) if slot < BODY_TRACKED_FRAME_SLOTS => written[slot] = true,
                // Past the tracked window: cannot alias a slot inside it.
                Some(_) => {}
                None => return [true; BODY_TRACKED_FRAME_SLOTS],
            }
        }
        pc = d.next_pc;
    }
    written
}

fn body_branch_targets(body_code: &[u8]) -> Option<std::collections::HashSet<usize>> {
    let mut targets = std::collections::HashSet::new();
    let mut pc = 0usize;
    while pc < body_code.len() {
        let d = crate::jitcode_runtime::decode_op_at(body_code, pc)?;
        // `switch/id` carries its case targets in the descr, not in an operand,
        // so this decode cannot name the joins it creates.  Give up on the
        // whole body rather than return a target set that is missing them.
        if d.opname.starts_with("switch") {
            return None;
        }
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
///
/// Exact numeric provenance for one positional parameter of an inline callee.
/// The two facts stay separate because bitwise specialization accepts only
/// exact ints, while add/subtract/multiply also accept exact floats.
#[derive(Clone, Copy, Default)]
pub(crate) struct ExactNumericArg {
    pub(crate) numeric: bool,
    pub(crate) plain_int: bool,
}

/// The `BINARY_OP` exemption needs the mirror-image fact — which values came
/// FROM exact-numeric caller arguments — so parameter provenance is tracked
/// slot-by-slot alongside freshness.  This matters for method-form calls:
/// `self` is commonly nonnumeric while a later argument is an exact int.
/// Name the body op that made [`fbw_callee_body_replay_safety`] answer
/// [`CalleeReplaySafety::Dirty`], under `PYRE_FBW_INLINE_DIAG`.  Without it the
/// declining instruction is invisible: the caller only sees `safety=Dirty` on
/// the `[inline-foriter-gate]` line and every one of the return sites below
/// looks alike.
macro_rules! replay_dirty {
    ($why:expr, $pc:expr, $opname:expr) => {{
        if std::env::var_os("PYRE_FBW_INLINE_DIAG").is_some() {
            eprintln!("[replay-dirty] pc={} op={} why={}", $pc, $opname, $why);
        }
        return CalleeReplaySafety::Dirty;
    }};
}

pub(crate) fn fbw_callee_body_replay_safety(
    body_code: &[u8],
    exact_numeric_args: &[ExactNumericArg],
    num_regs_i: usize,
    constants_i: &[i64],
    num_regs_r: usize,
    constants_r: &[i64],
    callee_descr_refs: &[DescrRef],
) -> CalleeReplaySafety {
    let Some(branch_targets) = body_branch_targets(body_code) else {
        replay_dirty!("BranchTargetsUndecodable", 0, "-");
    };
    let mut fresh_ref_regs = [false; u8::MAX as usize + 1];
    // Ref registers holding the `bool` an accepted `COMPARE_OP` produced.  A
    // truth residual over one of them is a read, not a live-heap write.
    let mut bool_ref_regs = [false; u8::MAX as usize + 1];
    // The dual of freshness, for the opposite question: which ref registers hold
    // a value whose Python-visible class is provably an IMMUTABLE BUILTIN.  That
    // is what the `BINARY_OP` exemption below actually needs — such an operand
    // can neither dispatch to a user `__add__` nor be mutated in place, so the
    // op commits nothing a replay could double.  Three sources close over it:
    //
    // - an incoming argument, which the caller checked is an exact int or float.
    //   A body does not receive parameters in registers;
    //   it reads them out of its own frame, whose `localsplus` slots
    //   `[0, nparams)` the exact-positional entry convention binds from the
    //   passed args (`callee_args.len() == nparams`, closure-free).  So the
    //   frame slots are tracked too, and `LOAD_FAST` / `STORE_FAST` propagate
    //   between the two through `getarrayitem_vable_r` / `setarrayitem_vable_r`.
    // - a `LoadConst` or `BoxInt` result.  A code constant is compiler-produced
    //   — int, float, complex, str, bytes, tuple, frozenset, None, code — never
    //   an instance of a user class.  `LoadGlobal` is pointedly NOT here: a
    //   module global is exactly where a numeric subclass with a side-effecting
    //   dunder reaches an operand, which is the hole this set closes.
    // - an accepted `BINARY_OP` result, since a builtin op over immutable
    //   builtin operands returns one.
    //
    // Registers at and above `num_regs_r` are the callee's immutable ref
    // constant pool rather than a register file, so a literal operand — the
    // `2.0` in `i * 2.0` — arrives as one of those.  Their objects are
    // reachable right here, so they are tested outright with the same
    // exactness the specialization itself demands, instead of argued about.
    //
    // Everything the body computes drops at a branch target for the reason
    // freshness does: a slot or register reaching a join holds whatever the
    // taken path put there, which this straight-line scan cannot name.  Two
    // things are re-seeded across that reset instead of dropped, because
    // neither is something a path could have changed: the constant pool, being
    // immutable, and any frame slot the body never stores into, which still
    // holds the caller's argument.
    let mut seed_numeric_ref_regs = [false; u8::MAX as usize + 1];
    let mut seed_plain_int_ref_regs = [false; u8::MAX as usize + 1];
    for (index, &raw) in constants_r.iter().enumerate() {
        let Some(reg) = num_regs_r
            .checked_add(index)
            .filter(|r| *r < seed_numeric_ref_regs.len())
        else {
            break;
        };
        let obj = raw as usize as pyre_object::PyObjectRef;
        if !obj.is_null() {
            let exact_int = unsafe { pyre_object::is_plain_int1(obj) };
            seed_plain_int_ref_regs[reg] = exact_int;
            seed_numeric_ref_regs[reg] =
                exact_int || unsafe { pyre_object::is_plain_float_strict(obj) };
        }
    }
    let mut numeric_ref_regs = seed_numeric_ref_regs;
    let mut plain_int_ref_regs = seed_plain_int_ref_regs;
    // A never-stored parameter slot keeps the caller's proof across a join, so
    // it is re-seeded rather than dropped — see [`body_stored_frame_slots`].
    let stored_slots = body_stored_frame_slots(body_code, num_regs_i, constants_i);
    let mut seed_numeric_slots = [false; BODY_TRACKED_FRAME_SLOTS];
    let mut seed_plain_int_slots = [false; BODY_TRACKED_FRAME_SLOTS];
    let mut numeric_slots = [false; BODY_TRACKED_FRAME_SLOTS];
    let mut plain_int_slots = [false; BODY_TRACKED_FRAME_SLOTS];
    for (slot, exact) in exact_numeric_args
        .iter()
        .take(BODY_TRACKED_FRAME_SLOTS)
        .enumerate()
    {
        numeric_slots[slot] = exact.numeric;
        plain_int_slots[slot] = exact.plain_int;
        if !stored_slots[slot] {
            seed_numeric_slots[slot] = exact.numeric;
            seed_plain_int_slots[slot] = exact.plain_int;
        }
    }
    // The frame register every vable op in this body has used so far.  A second
    // one would mean the slot bookkeeping above is tracking two different
    // frames, so it stops claiming anything.
    let mut vable_reg: Option<u8> = None;
    let mut deferred_call = false;
    let mut pc = 0usize;
    while pc < body_code.len() {
        if branch_targets.contains(&pc) {
            fresh_ref_regs = [false; u8::MAX as usize + 1];
            bool_ref_regs = [false; u8::MAX as usize + 1];
            numeric_ref_regs = seed_numeric_ref_regs;
            plain_int_ref_regs = seed_plain_int_ref_regs;
            numeric_slots = seed_numeric_slots;
            plain_int_slots = seed_plain_int_slots;
        }
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            replay_dirty!("DecodeOpFailed", pc, "-");
        };
        // Set by the arms below when this op's `>r` result is itself an
        // immutable builtin.
        let mut dst_exact_numeric = false;
        let mut dst_exact_plain_int = false;
        let mut dst_exact_bool = false;

        // The ref-slot accessors name the frame in operand 0 and the slot in
        // operand 1, both one byte wide.  The `_i` / `_f` variants address a
        // different vable array and so cannot alias a ref slot; anything whose
        // operands do not start `ri` is not this shape at all.
        let ref_slot_access = d.opname.starts_with("getarrayitem_vable_r")
            || d.key.starts_with("setarrayitem_vable_r");
        let vable_slot = if ref_slot_access && d.argcodes.starts_with("ri") {
            body_code
                .get(d.pc + 1)
                .filter(|frame| *vable_reg.get_or_insert(**frame) == **frame)
                .and(body_code.get(d.pc + 2))
                .and_then(|ireg| body_int_operand_constant(*ireg, num_regs_i, constants_i))
                .and_then(|slot| usize::try_from(slot).ok())
        } else {
            None
        };
        if d.key.starts_with("setarrayitem_vable_r") {
            // `rir…`: the stored register is operand 2.  A slot this scan cannot
            // resolve could name any of them, so it drops the lot.
            match vable_slot {
                Some(slot) if slot < BODY_TRACKED_FRAME_SLOTS => {
                    let src = d
                        .argcodes
                        .starts_with("rir")
                        .then(|| body_code.get(d.pc + 3))
                        .flatten()
                        .copied();
                    numeric_slots[slot] = src.is_some_and(|src| numeric_ref_regs[src as usize]);
                    plain_int_slots[slot] = src.is_some_and(|src| plain_int_ref_regs[src as usize]);
                }
                // Past the tracked window: cannot alias a slot inside it.
                Some(_) => {}
                None => {
                    numeric_slots = [false; BODY_TRACKED_FRAME_SLOTS];
                    plain_int_slots = [false; BODY_TRACKED_FRAME_SLOTS];
                }
            }
        }

        if d.opname.starts_with("residual_call") {
            let Some(descr_index) = residual_call_descr_index_in_body(body_code, &d) else {
                replay_dirty!("ResidualCallDescrIndexMissing", d.pc, d.opname);
            };
            let Some(call_descr) = callee_descr_refs
                .get(descr_index)
                .and_then(|descr| descr.as_call_descr())
            else {
                replay_dirty!("ResidualCallDescrNotACall", d.pc, d.opname);
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
            // `get_current_exception` is the PUSH_EXC_INFO `prev` save, which
            // `try_walker_lower_exc_info_residual` lowers to a bare
            // `GETFIELD_GC_R(ec, sys_exc_value)`: a field read, so a replay
            // reads the same value again.  Its writing twin
            // `SetCurrentException` is not here — it is journalled, and so
            // reaches the `deferred_call` arm below instead.
            let replay_safe_read = matches!(
                ei.pyre_helper,
                majit_ir::PyreHelperKind::LoadConst
                    | majit_ir::PyreHelperKind::LoadGlobal
                    | majit_ir::PyreHelperKind::BoxInt
                    | majit_ir::PyreHelperKind::NewtupleFromArray
                    | majit_ir::PyreHelperKind::NewlistFromArray
                    | majit_ir::PyreHelperKind::GetCurrentException
            );
            // `box_int` is the only generic replay-safe helper here whose
            // result is necessarily numeric.  `load_const` may return a str,
            // tuple, or another nonnumeric immutable value, while the typed
            // jitcode constant pool was classified exactly above.
            let dst_boxed_int = ei.pyre_helper == majit_ir::PyreHelperKind::BoxInt;
            let provably_side_effect_free = replay_safe_read
                || ei.check_is_elidable()
                || ei.extraeffect == majit_ir::ExtraEffect::LoopInvariant;
            let accepted_numeric_op = if provably_side_effect_free {
                None
            } else {
                residual_call_specialized_plain_numeric_binop(
                    body_code,
                    &numeric_ref_regs,
                    &plain_int_ref_regs,
                    &d,
                    num_regs_i,
                    constants_i,
                    callee_descr_refs,
                )
            };
            // A `COMPARE_OP` over the same proven operands is accepted for the
            // same reason, but its result is a `bool`, not an operand the
            // numeric provenance below may chain on.
            let accepted_binop = matches!(
                accepted_numeric_op,
                Some(SpecializedBinop::Numeric | SpecializedBinop::PlainInt)
            );
            // `CHECK_EXC_MATCH` shares the `COMPARE_OP` shape but reads only
            // types, so it needs no operand proof at all.
            let accepted_exc_match = !provably_side_effect_free
                && crate::jitcode_dispatch::residual_call::residual_call_is_exception_match(
                    body_code,
                    &d,
                    num_regs_i,
                    constants_i,
                    callee_descr_refs,
                );
            dst_exact_bool =
                accepted_numeric_op == Some(SpecializedBinop::Compare) || accepted_exc_match;
            let accepted_truth = !provably_side_effect_free
                && crate::jitcode_dispatch::residual_call::residual_call_is_proven_truth(
                    body_code,
                    &numeric_ref_regs,
                    &bool_ref_regs,
                    &d,
                    callee_descr_refs,
                );
            // An accepted arithmetic op over exact numeric operands returns an
            // exact numeric.  Bitwise ops require and return exact ints.
            dst_exact_numeric = dst_boxed_int || accepted_binop;
            dst_exact_plain_int =
                dst_boxed_int || accepted_numeric_op == Some(SpecializedBinop::PlainInt);
            if !provably_side_effect_free
                && accepted_numeric_op.is_none()
                && !accepted_truth
                && !accepted_exc_match
            {
                // A Python-level CALL is the one shape this scan cannot
                // settle: the inline lever binds its callee only at the call,
                // so whether it leaves a residual behind — and what that
                // residual writes — is not a property of this body.  Defer it;
                // the backstop aborts before executing one that did not
                // inline.
                // `RAISE_VARARGS` lowers to the same shape: its
                // `normalize_raise_varargs_fn` residual instantiates a raised
                // CLASS and normalizes an optional `from` cause, so what it
                // touches is a runtime value exactly like a CALL's callee.  For
                // the shape a loop actually repeats — an exception the walk
                // built itself, no `from` cause — the three walker-native folds
                // (`try_walker_trace_exception_new`,
                // `try_walker_trace_raise_builtin`,
                // `try_walker_trace_raise_bare_class`) erase the residual before
                // the backstop is reached, and each writes only into the object
                // it just allocated.  Anything else reaches
                // `fbw_abort_nested_unjournaled_residual` and aborts before the
                // helper runs, since the decline there covers every residual
                // that is not elidable / loop-invariant / `ForIterNext`.
                // `set_current_exception` is the same shape once more, and it
                // is the one every `try`/`except` body carries (the
                // PUSH_EXC_INFO store and the POP_EXCEPT restore).  Its fold
                // [`try_walker_lower_exc_info_residual`] journals the displaced
                // `sys_exc_value` through [`fbw_sys_exc_journal_push`] BEFORE
                // applying the concrete store, and
                // [`fbw_store_journal_rollback`] replays the journal in reverse
                // on a non-committed exit — so a folded store is undone for the
                // replay, and an unfolded one never runs.
                // `binary_op` / `compare_op` over operands this scan could not
                // prove exact-numeric is the same shape one more time: which
                // `__add__` / `__lt__` runs is a property of the operand's
                // runtime class, not of this body.  The proven-operand case was
                // already accepted above; what is left here is exactly the
                // operand whose provenance the scan lost — most commonly a
                // `LOAD_ATTR` result, since that arm is itself deferred and
                // clears numeric provenance.  Deferring instead of declining is
                // what lets `self.v + i` inline: at trace time the attribute
                // read folds to a mapdict slot with a concrete int shadow, so
                // the walker's numeric specialization erases the residual before
                // the backstop is reached.  An operand pair that stays opaque
                // leaves the residual standing, and it reaches
                // `fbw_abort_nested_unjournaled_residual` like any other — the
                // helper never runs.
                if matches!(
                    ei.pyre_helper,
                    majit_ir::PyreHelperKind::CallFn
                        | majit_ir::PyreHelperKind::CallKw
                        | majit_ir::PyreHelperKind::CallFunctionEx
                        | majit_ir::PyreHelperKind::RaiseVarargs
                        | majit_ir::PyreHelperKind::SetCurrentException
                        | majit_ir::PyreHelperKind::LoadAttr
                        | majit_ir::PyreHelperKind::BinaryOp
                        | majit_ir::PyreHelperKind::CompareOp
                ) {
                    deferred_call = true;
                    // The callee this resolves to is a runtime value, so what it
                    // can reach through this frame is unknown here.
                    numeric_slots = [false; BODY_TRACKED_FRAME_SLOTS];
                    plain_int_slots = [false; BODY_TRACKED_FRAME_SLOTS];
                } else {
                    replay_dirty!(
                        format!("ResidualCallWritesLiveHeap/{:?}", ei.pyre_helper),
                        d.pc,
                        d.opname
                    );
                }
            }
        } else if d.opname.starts_with("setfield_gc") {
            // Only the `r<value>d` shapes name their target in a ref register
            // (operand 0), with the field descr at operand 2.  The `i…` forms
            // (`setfield_gc_i/iid`, `setfield_gc_r/ird`, `setfield_gc_v/iid`,
            // `setfield_gc_v/ird`) address the target by raw int instead, so
            // there is no ref register whose freshness could be proven — and
            // reading operand 0 as one would index the freshness set with an
            // int register number.
            if !d.argcodes.starts_with('r') {
                replay_dirty!("SetfieldGcTargetNotRefReg", d.pc, d.opname);
            }
            let Some(&target_reg) = body_code.get(d.pc + 1) else {
                replay_dirty!("SetfieldGcTargetRegMissing", d.pc, d.opname);
            };
            let descr_index = decode_descr_index(body_code, &d, 2);
            let immutable_field = callee_descr_refs
                .get(descr_index)
                .and_then(|descr| descr.as_field_descr())
                .is_some_and(|field| field.is_immutable());
            if !fresh_ref_regs[target_reg as usize] || !immutable_field {
                replay_dirty!("SetfieldGcTargetNotFreshOrMutable", d.pc, d.opname);
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
                replay_dirty!("SetarrayitemGcTargetNotFresh", d.pc, d.opname);
            }
        } else if d.opname.starts_with("setinteriorfield_gc")
            || d.opname.starts_with("raw_store")
            || d.opname.starts_with("cond_call")
            || d.opname.starts_with("call_assembler")
            || d.opname.starts_with("inline_call")
        {
            // Interior/raw stores and non-residual call forms cannot be proven
            // replay-safe from this single callee body.
            replay_dirty!("UnprovableStoreOrCallForm", d.pc, d.opname);
        }

        // The result byte is always the final operand for `>r` forms.
        if d.argcodes.ends_with(">r") {
            let Some(&dst) = body_code.get(d.next_pc.saturating_sub(1)) else {
                replay_dirty!("ResultRegisterByteMissing", d.pc, d.opname);
            };
            fresh_ref_regs[dst as usize] = d.key == "new_with_vtable/d>r"
                || d.opname.starts_with("new_array")
                || (d.key == "ref_copy/r>r"
                    && body_code
                        .get(d.pc + 1)
                        .is_some_and(|src| fresh_ref_regs[*src as usize]));
            // A proven value enters a register by being loaded out of a proven
            // frame slot, from the residual arm above, or through a verbatim
            // copy.  Every other producer overwrites the register with an
            // unproven value.
            numeric_ref_regs[dst as usize] = dst_exact_numeric
                || vable_slot.is_some_and(|slot| {
                    d.opname.starts_with("getarrayitem_vable_r")
                        && numeric_slots.get(slot).copied().unwrap_or(false)
                })
                || (d.key == "ref_copy/r>r"
                    && body_code
                        .get(d.pc + 1)
                        .is_some_and(|src| numeric_ref_regs[*src as usize]));
            bool_ref_regs[dst as usize] = dst_exact_bool
                || (d.key == "ref_copy/r>r"
                    && body_code
                        .get(d.pc + 1)
                        .is_some_and(|src| bool_ref_regs[*src as usize]));
            plain_int_ref_regs[dst as usize] = dst_exact_plain_int
                || vable_slot.is_some_and(|slot| {
                    d.opname.starts_with("getarrayitem_vable_r")
                        && plain_int_slots.get(slot).copied().unwrap_or(false)
                })
                || (d.key == "ref_copy/r>r"
                    && body_code
                        .get(d.pc + 1)
                        .is_some_and(|src| plain_int_ref_regs[*src as usize]));
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
