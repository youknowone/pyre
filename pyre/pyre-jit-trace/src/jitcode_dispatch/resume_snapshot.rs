//! Guard resume-data snapshot capture.
//!
//! **Parity:** trace-side counterpart of `resume.py`'s `capture_resumedata`
//! (implementation mirror `majit-metainterp/resume.rs`).
//!
//! Builds the resume-data snapshot attached to the last recorded guard
//! (the `capture_resumedata` analogue): the single-frame path
//! (`walker_capture_snapshot_for_last_guard*`) and the multi-frame inline
//! path (`walker_capture_multi_frame_inline_snapshot`) plus the
//! inline-caller-frame computation helpers they rely on.

use super::*;

/// Exact `jtransform.py handle_residual_call` trailing `-live-` marker.
///
/// The codewriter emits this immediately after every may-force/can-raise
/// residual.  Runtime per-Code metadata normally carries the same coordinate;
/// a terminal opcode immediately after the call (`RAISE_VARARGS`, for example)
/// may have no first-insn pivot, so retain the bytecode-local identity as the
/// authoritative fallback rather than rewinding to the call.
fn trailing_live_marker(payload: &crate::PyJitCode, call_pc: usize) -> Option<usize> {
    let call = crate::jitcode_runtime::decode_op_at(payload.jitcode.code.as_slice(), call_pc)?;
    let marker =
        crate::jitcode_runtime::decode_op_at(payload.jitcode.code.as_slice(), call.next_pc)?;
    (marker.opname == "live").then_some(marker.pc)
}

/// Read the Python PC paired with a native resume word.  The codewriter builds
/// this table from Python-PC keys, including the same forward trivia skip as
/// `backxlat_py_pc`; capture deliberately never projects the word backwards.
fn forward_snapshot_py_pc(jitcode_index: u32, pc: u32) -> Result<u32, DispatchError> {
    if pc == majit_ir::resumedata::NO_JITCODE_PC as u32 {
        return Ok(u32::MAX);
    }
    crate::state::pyjitcode_for_jitcode_index(jitcode_index as i32)
        .and_then(|payload| payload.resume_position_for_jitcode_pc(pc as usize))
        .map(|(_, py_pc)| py_pc)
        .ok_or(DispatchError::GuardResumeCoordinateUnavailable { pc: pc as usize })
}

/// `generate_guard` (`pyjitpl.py`) keys `after_residual_call`
/// on the guard opcode itself: `GUARD_EXCEPTION` / `GUARD_NO_EXCEPTION` /
/// `GUARD_NOT_FORCED` / `GUARD_ALWAYS_FAILS` resume *after* the residual
/// call; every other guard resumes at its own opcode.  The call already
/// executed in compiled code and consumed its Python stack operands;
/// resuming at the call's own opcode would re-execute it from a
/// coordinate whose stack no longer holds those operands,
/// dropping/duplicating the side effect (e.g. an in-place `a[i] = a[j]`
/// swap losing one store at a guard-failure transition).  #124/#281.
pub(crate) fn walker_capture_snapshot_for_last_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
) -> Result<(), DispatchError> {
    walker_capture_snapshot_for_last_guard_scoped(ctx, op_pc, GuardCaptureScope::default())
}

pub(crate) fn walker_capture_snapshot_for_last_guard_scoped<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    scope: GuardCaptureScope<'_>,
) -> Result<(), DispatchError> {
    let after_residual_call = matches!(
        ctx.trace_ctx.last_guard_opcode(),
        Some(
            OpCode::GuardException
                | OpCode::GuardNoException
                | OpCode::GuardNotForced
                | OpCode::GuardAlwaysFails
        )
    );
    walker_capture_snapshot_for_last_guard_impl(ctx, op_pc, after_residual_call, scope)
}

/// Attach a resume snapshot to a `_nonstandard_virtualizable` PTR_EQ
/// promote guard if a `TraceCtx::vable_*` call emitted one.
///
/// `TraceCtx::vable_getfield_*` / `vable_setfield` /
/// `vable_get|setarrayitem_*` run the `_nonstandard_virtualizable`
/// check (`trace_ctx.rs _nonstandard_virtualizable`); for a frame that
/// is not the standard virtualizable it records a `PTR_EQ` + a
/// `promote_int` `GuardValue` *internally*, without a resume snapshot.
/// The walker never sees that guard at its own emit sites, so it would
/// reach `store_final_boxes_in_guard` with `rd_resume_position == -1`
/// and panic.  A callee frame is non-standard whether reached by an
/// inline sub-walk or compiled as its own Finish portal (via
/// call_user_function_with_eval); the production main frame IS the
/// standard virtualizable, so the check short-circuits at Step 1 and
/// emits nothing — gate on a full-body walk being active so this is a
/// no-op (one cheap `num_guards` read) outside it.  The non-standard
/// fact is cached per box
/// after the first access, so at most one such guard is emitted per
/// inlined frame and `walker_capture_snapshot_for_last_guard`'s
/// "last guard" target is unambiguous.
pub(crate) fn walker_capture_inline_nonstandard_vable_guard<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    guards_before: usize,
) -> Result<(), DispatchError> {
    // The non-standard virtualizable's internal promote GuardValue is
    // emitted only inside a full-body walk against a callee frame that is
    // not the production standard virtualizable: either an inline sub-walk's
    // callee heap frame, or a callee compiled as its own Finish portal
    // (reached via call_user_function_with_eval). Outside a full-body walk
    // the frame is always the standard virtualizable and emits no such
    // guard.
    if ctx.fbw_mode.snapshot_sym.is_null() {
        return Ok(());
    }
    if ctx.trace_ctx.num_guards() <= guards_before {
        // Standard virtualizable (the production main frame): the
        // `_nonstandard_virtualizable` check short-circuited and emitted no
        // guard, so there is nothing to capture.
        return Ok(());
    }
    if !ctx.fbw_mode.inline_subwalk {
        // A callee compiled as its own Finish portal hit the non-standard
        // virtualizable path. Its internal promote GuardValue + force
        // store-back are not yet wired with a resume snapshot / FieldDescr
        // for the own-portal compile (only the inline sub-walk path below
        // is), so the optimizer's `store_final_boxes_in_guard` /
        // `optimize_setfield_gc` would trip. Abort to the interpreter rather
        // than compile a trace that cannot be finalized.
        return Err(DispatchError::NonStandardVableFinishPortalUnsupported { pc: op_pc });
    }
    // Same buildability precondition as `walker_capture_snapshot_for_last_guard`:
    // every virtualizable box must carry `OpRef::ty()` or the snapshot
    // encoder panics — abort to interpretation instead.
    if !ctx.trace_ctx.vable_snapshot_buildable() {
        return Err(DispatchError::GuardSnapshotVableUntyped { pc: op_pc });
    }
    // #73: give this promote guard the full multi-frame resume every RPython
    // guard gets — `_nonstandard_virtualizable` (pyjitpl.py:1120) →
    // `implement_guard_value(eqbox, pc)` (1916) →
    // `generate_guard(GUARD_VALUE, resumepc=orgpc)` (2582) →
    // `capture_resumedata(resumepc)` (2610) walking the WHOLE MIFrame chain
    // (opencoder.py:819).  When the paused-caller chain covers the full inline
    // depth (the same gate as the standard multi-frame guard path,
    // `walker_capture_snapshot_for_last_guard_impl`), publish the callee's OWN
    // coordinate plus each paused caller instead of the single-frame sentinel
    // collapse below — whose carried word is `NO_JITCODE_PC`, which
    // `resolve_resume_pc_with_jitcode_pc` rejects, so the collapse
    // unconditionally aborts (`GuardResumeCoordinateUnavailable`) every inline
    // sub-walk emit of this guard.  Stamp the last *guard* op, not the last op:
    // `emit_force_virtualizable` records GETFIELD_GC / PTR_NE / COND_CALL after
    // the promote.  A chain that is not full, or a callee/caller frame the
    // publisher cannot build, falls through to (or aborts the same as) the
    // sentinel below — never a wrong resume.
    let (n_parents, n_callees, parent_frames) = {
        let session = ctx.session.borrow();
        (
            session
                .framestack
                .iter()
                .filter(|frame| frame.parent.is_some())
                .count(),
            session.framestack.len(),
            session
                .framestack
                .iter()
                .filter_map(|frame| frame.parent.clone())
                .collect::<Vec<_>>(),
        )
    };
    if n_parents > 0 && n_parents == n_callees {
        return walker_capture_multi_frame_inline_snapshot(
            ctx,
            op_pc,
            false,
            parent_frames,
            GuardCaptureScope::default(),
            true,
        );
    }
    // The guard is not the last recorded op: `emit_force_virtualizable`
    // records GETFIELD_GC / PTR_NE / COND_CALL after the promote, so stamp
    // the last *guard* op (`..._for_last_guard_op_...`).  Resume at the
    // caller's CALL boundary (`outer_active_boxes` / `entry_py_pc`),
    // re-executing the whole call on deopt — sound for the side-effect-free
    // leaves this path inlines (the non-standard identity guard is itself
    // deterministic and never fails at runtime).
    // The non-standard identity guard has no representable JitCode resume
    // coordinate: its carried word is the sentinel, so decline rather than
    // publishing the caller's Python pc as a JitCode offset.
    let nsvable_word = majit_ir::resumedata::NO_JITCODE_PC;
    let Some(nsvable_pc_word) =
        crate::state::pyjitcode_for_jitcode_index(ctx.outer_jitcode_index as i32)
            .and_then(|payload| {
                let resolved = payload
                    .resolve_resume_pc_with_jitcode_pc(nsvable_word, crate::state::op_live());
                resolved
            })
            .map(|offset| offset as u32)
    else {
        return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc });
    };
    let (vable_boxes, vref_boxes) = ctx.trace_ctx.build_snapshot_vable_vref_boxes();
    let nsvable_py_pc = forward_snapshot_py_pc(ctx.outer_jitcode_index, nsvable_pc_word)?;
    ctx.trace_ctx
        .capture_snapshot_for_last_guard_op_with_vable_vref(
            &ctx.outer_active_boxes,
            ctx.outer_jitcode_index,
            nsvable_pc_word,
            nsvable_py_pc,
            &vable_boxes,
            &vref_boxes,
        );
    Ok(())
}

pub(crate) fn walker_capture_snapshot_for_last_guard_impl<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    after_residual_call: bool,
    scope: GuardCaptureScope<'_>,
) -> Result<(), DispatchError> {
    // A guard whose resume snapshot cannot be built must abort the trace,
    // not panic.  `build_vable_snapshot_boxes` requires every virtualizable
    // box (including the identity at `[-1]`) to carry `OpRef::ty()`; a
    // deeper inlined / recursive frame can leave the identity untyped.
    // Surface a typed abort rather than tripping the
    // invariant panic (the multi-frame vable snapshot is unported).
    if !ctx.trace_ctx.vable_snapshot_buildable() {
        return Err(DispatchError::GuardSnapshotVableUntyped { pc: op_pc });
    }
    // Snapshot semantics for walker-emitted guards
    // (`pyjitpl.py generate_guard` + `capture_resumedata`):
    //
    // RPython treats helper jitcodes (pop_value, nlocals, etc.) as
    // separate `MIFrame`s on `metainterp.framestack`, capturing one
    // snapshot frame per `MIFrame` plus a vable_array / vref_array
    // prefix on the top frame (`opencoder.py create_top_snapshot`).
    // At resume, RPython's blackhole interpreter re-enters each frame's
    // jitcode and replays from the saved pc.
    //
    // Pyre's blackhole interpreter only knows how to run *pyjitcode*
    // bytecode (Python bytecode), not helper jitcodes — pyre's
    // per-opcode arm jitcodes and sub-jitcode helpers are walker-only
    // structures with no blackhole entry point.  The structural
    // consequence: any walker-emitted guard, regardless of how deep
    // the sub-walk nesting is, must resume to the *outer* Python
    // opcode boundary (`sym.jitcode` at `entry_py_pc`) — that is the
    // only resume point pyre's blackhole can re-enter.  The
    // framestack-collapse is a deliberate adaptation, not a parity
    // miss; the walker context carries the outer Python frame only.
    // Inline-traced Python frames from the retired inline-frame builder are
    // not reachable from this entry point because the production
    // walker allow-list does not yet enable opcodes that drive inline
    // tracing — when that expands, `WalkContext` must grow a parent-
    // Python-frame chain (analogous to `MIFrame.parent_frames`) and
    // this helper switches to
    // `capture_snapshot_for_last_guard_multi_frame_with_vable_vref`.
    //
    // The snapshot is therefore a single Python frame at the outer
    // pyjitcode coordinates.  `ctx.outer_jitcode_index` +
    // `ctx.entry_py_pc` track those coordinates; `outer_active_boxes`
    // carries the `PyFrame` state at the Python opcode boundary
    // (snapshotted once at walk entry — the retired per-opcode arm
    // entry, or the inline CALL-site capture — from
    // `sym.registers_r ∪ sym.registers_i.opref ∪ sym.registers_f.opref`
    // via `collect_outer_active_boxes` / `frame_liveness_reg_indices_
    // by_bank_at`).
    //
    // `opencoder.py create_top_snapshot` writes vable_array +
    // vref_array on the top snapshot.  The walker-emitted guard IS
    // a top snapshot for pyre (helper frames don't resume), so feed
    // the trace-time vable/vref shadow through.  Empty when no
    // virtualizable / virtualref is live, matching the upstream
    // 0-length-array shape.  The build is deferred until after the
    // resume py_pc is resolved and the `last_instr` vable scalar is
    // published, so the snapshot carries this guard's coordinate (the
    // walker never crosses `set_orgpc`, so the scalar is otherwise
    // stale at the loop-header pc — see the publish below).

    // Full-body walk (Phase 7): the walk processes the outer
    // `sym.jitcode` directly, so `op_pc` is a real resume coordinate.
    // Map it back to the containing Python opcode (the blackhole resumes
    // and re-executes that opcode — `orgpc` parity) and read liveness
    // from the live walk register banks at that pc, instead of the
    // static entry-time coordinate the per-opcode arm path uses.
    let full_body_sym = ctx.fbw_mode.snapshot_sym;
    // Inline sub-walk: the guard's `op_pc` is a *callee* coordinate that
    // does not exist in the outer (`full_body_sym`) jitcode's py_pc→jitcode
    // tables.
    // Skip the full-body mapping and fall through to the caller-boundary
    // capture below, which resumes at the CALL site (re-execute the
    // call on deopt — see `fbw_mode.inline_subwalk`).
    let inline_subwalk = ctx.fbw_mode.inline_subwalk;
    // #68 multi-frame inline guard: a guard emitted inside an inlined callee
    // sub-walk with paused caller frames on the walk framestack resumes
    // BOTH the callee (at its own pc) and the caller(s) (at the CALL return
    // point), instead of collapsing to the caller boundary (re-execute).  Only
    // the forward-branch inline path
    // populates the chain; straight-line callees keep the empty chain + the
    // single-frame collapse below.
    if inline_subwalk {
        // Fire the multi-frame snapshot only when the paused-caller chain
        // covers the FULL current inline depth: framestack levels with parents
        // must have one entry per active inlined callee. A nested
        // straight-line callee inlined under a multiframe ancestor (e.g.
        // `add3` inside a multiframe `mix`) pushes NO parent frame, so its own
        // guards see a SHORTER chain than the callee depth — fall through to
        // the single-frame collapse (the strict callee's resume-at-CALL
        // behavior) rather than emit a chain that skips the intermediate frame.
        let (n_parents, n_callees, parent_frames) = {
            let session = ctx.session.borrow();
            (
                session
                    .framestack
                    .iter()
                    .filter(|frame| frame.parent.is_some())
                    .count(),
                session.framestack.len(),
                session
                    .framestack
                    .iter()
                    .filter_map(|frame| frame.parent.clone())
                    .collect::<Vec<_>>(),
            )
        };
        if n_parents > 0 && n_parents == n_callees {
            // A STRICT straight-line callee (gh#420) whose own frame is not
            // MF-snapshot-able (a kept operand-stack temp the sub-walk does not
            // mirror) propagates the `Unsupported` error the same as the branch
            // path: the enclosing inline declines to a residual call rather than
            // resuming through the single-frame collapse, whose caller-boundary
            // re-execute both mis-sizes the resumed frame (a decode/`LOAD_FAST`
            // out-of-bounds) and re-applies the callee's committed side effect.
            return walker_capture_multi_frame_inline_snapshot(
                ctx,
                op_pc,
                after_residual_call,
                parent_frames,
                scope,
                false,
            );
        }
    }
    if !inline_subwalk && !full_body_sym.is_null() {
        // SAFETY: the pointer is set only for the lifetime of the
        // full-body `dispatch_via_miframe` (the guard restores it on
        // exit); the `PyreSym` outlives the walk.  Read-only access to
        // immutable layout fields (jitcode / color maps / frame / ec) —
        // the walk's mutable register file lives in `ctx.registers_*`
        // (fresh `top_regs`), not in `sym.registers_*`.
        let sym = unsafe { &*full_body_sym };
        if !sym.jitcode().is_null() {
            // Set when an after-residual-call guard's residual call sits inside
            // a try-block (its per-CodeObject jitcode emitted a post-call
            // catch); the JitCode resume coordinate then selects that catch.
            let mut marker_call_jit_pc: Option<usize> = None;
            let (py_pc, jitcode_index, num_instrs) = unsafe {
                let jc = &*sym.jitcode();
                let mut py = python_pc_for_jitcode_pc(&jc.payload.metadata, op_pc);
                // The jitcode-pc→py-pc inversion can land on a Python trivia
                // instruction's jitcode region (e.g. a branch target
                // whose block lowers `NOT_TAKEN`).  A resume coordinate
                // must be a real opcode: the interpreter resumes branches
                // at `semantic_fallthrough_pc` / `jump_target_forward`,
                // both of which forward-skip trivia.  Advance to the same
                // real opcode so the resume reader's BACKWARD trivia
                // backtrack (call_jit.rs) is a no-op — otherwise a
                // `NOT_TAKEN` py_pc backtracks to the preceding branch
                // opcode, whose block-entry liveness differs from the
                // target's and desyncs the snapshot box-count.
                if !jc.payload.code_ptr.is_null() {
                    let code = &*jc.payload.code_ptr;
                    py = skip_python_trivia_forward(code, py as usize) as u32;
                    // Entry-resume capture: the carried coordinate is the
                    // authority; take its forward-py twin (the same pairing
                    // the decoder resolves) instead of the inversion above,
                    // so the liveness/entry windows key exactly where the
                    // decode will.
                    if let Some(carried) = scope.carried_resume_jit_pc {
                        if let Some(fwd) = jc.payload.forward_py_pc_for_jitcode_pc(carried) {
                            py = fwd;
                        }
                    }
                    // after_residual_call=True (`pyjitpl.py`): the
                    // may-force call already executed in compiled code and
                    // consumed its Python stack operands. Resume at the NEXT
                    // executable opcode so the blackhole continues past the call
                    // (re-executing from the call's coordinate would drop/dup the
                    // side effect, e.g. an in-place list swap store). The
                    // fallthrough resume routes a raise through the next opcode's
                    // own `catch_exception` (still inside the same try-block) and
                    // the bridge-decline path, which handles every sequential
                    // residual call whose NEXT opcode shares the try.
                    //
                    // A residual whose CALL pc is itself directly covered by an
                    // enclosing exception-table handler needs its OWN catch to
                    // receive the raise: its fallthrough may leave the covered
                    // region (e.g. FOR_ITER-next's fallthrough is the continue-arm
                    // body, reached only on a NON-null item, which carries no
                    // catch for the call's OWN raise). When the guard capture
                    // requested catch-resume
                    // (`GuardCaptureScope::residual_call_catch_resume`) and the
                    // call's CALL pc is covered by the code's exception table,
                    // carry the CALL jitcode offset so the blackhole resumes at
                    // the call's OWN catch and routes the raise to the enclosing
                    // handler instead of escaping the frame.
                    if after_residual_call {
                        let call_py_pc = py;
                        let exact_after_marker = jc
                            .payload
                            .after_residual_marker_for_jitcode_pc(op_pc)
                            .or_else(|| jc.payload.after_residual_call_resume_for_jitcode_pc(op_pc))
                            .or_else(|| trailing_live_marker(&jc.payload, op_pc));
                        py = exact_after_marker
                            .and_then(|marker| jc.payload.forward_py_pc_for_jitcode_pc(marker))
                            .unwrap_or_else(|| {
                                crate::pyjitpl::semantic_fallthrough_pc(code, py as usize) as u32
                            });
                        let call_pc_has_catch = pyre_interpreter::pycode::lookup_exceptiontable(
                            &code.exceptiontable,
                            call_py_pc * 2,
                        )
                        .is_some();
                        if scope.residual_call_catch_resume
                            && call_pc_has_catch
                            && jc
                                .payload
                                .after_residual_call_resume_for_jitcode_pc(op_pc)
                                .is_some()
                        {
                            marker_call_jit_pc = Some(op_pc);
                        }
                    }
                }
                (
                    py,
                    jc.index as u32,
                    jc.payload.metadata.n_py_instrs as usize,
                )
            };
            // #67/#124: synthetic loop-close guard pc overshoots past the last
            // Python opcode → resume at the trace's entry py (loop header).
            let loop_close_overshoot = py_pc as usize >= num_instrs;
            let py_pc = if loop_close_overshoot {
                ctx.entry_py_pc()
            } else {
                py_pc
            };
            // The three resume-depth consumers below (vstack reconcile,
            // valuestackdepth publication, and vable stack sync) share one
            // JitCode-PC twin. Plain guards key their trivia twin at `op_pc`;
            // after-residual guards key the same family at their post-call
            // marker. A loop-close overshoot falls back to entry_py_pc and has
            // no corresponding op_pc-derived twin, so it retains the legacy
            // Python-PC lookup below.
            let resume_depth_twin: Option<u16> = unsafe {
                let jc = &*sym.jitcode();
                if loop_close_overshoot {
                    None
                } else if scope.carried_resume_jit_pc.is_some() {
                    // Entry-resume capture: the depth twins are keyed for
                    // op-START coordinates; keying them with an
                    // already-advanced resume coordinate reads the depth of
                    // the coordinate's KEY opcode (the call, with its
                    // operands still on the stack) — an over-count whose
                    // published `valuestackdepth` makes the resume read
                    // garbage slots as Refs.  Fall back to the raw static
                    // liveness at the resume py (the forward twin above),
                    // exactly what the decode derives.
                    None
                } else if after_residual_call {
                    jc.payload
                        .after_residual_marker_for_jitcode_pc(op_pc)
                        .or_else(|| jc.payload.after_residual_call_resume_for_jitcode_pc(op_pc))
                        .or_else(|| trailing_live_marker(&jc.payload, op_pc))
                        .and_then(|marker| jc.payload.depth_trivia_for_jitcode_pc(marker))
                } else {
                    jc.payload.depth_trivia_for_jitcode_pc(op_pc)
                }
            };
            // `capture_resumedata(after_residual_call=True)` snapshots the
            // trailing `-live-`, after the residual result has replaced the
            // Python opcode's consumed operands (pyjitpl.py,
            // opencoder.py).  The walk-level stack mirror normally
            // applies that replacement only when `step_vstack_mirror` reaches
            // the next Python opcode.  A guard emitted by the residual itself
            // captures before that step, so advance the mirror here to the
            // same post-call boundary before it supplies the vable snapshot.
            // This is the same transition the following walk step would make;
            // it merely makes the guard's resume image observe it at the
            // required point.
            if after_residual_call && ctx.vstack_valid {
                let jc = unsafe { &*sym.jitcode() };
                let code_ptr = jc.payload.code_ptr;
                if !code_ptr.is_null() {
                    let resume_depth = match resume_depth_twin {
                        Some(depth) => depth as usize,
                        None => crate::liveness::liveness_for(code_ptr)
                            .depth_at_py_pc()
                            .get(py_pc as usize)
                            .copied()
                            .unwrap_or(0) as usize,
                    };
                    let code = unsafe { &*code_ptr };
                    reconcile_vstack_at_boundary(ctx, code, py_pc, resume_depth);
                }
            }
            // Publish `last_instr = py_pc - 1` to the vable static shadow
            // before snapshotting.  The walker walks JitCode and never
            // crosses `set_orgpc`, so the `last_instr` scalar in
            // `virtualizable_boxes` keeps whatever the trace seed or the
            // previous `close_loop_args_at` override wrote (the loop-header
            // pc).  The blackhole / vable-sync resume reads this scalar into
            // `frame.last_instr`; a stale loop-header value makes a mid-body
            // guard resume at the loop header instead of the guard's own
            // opcode, re-running loop iterations and corrupting the result.
            // Mirror of `MIFrame::publish_last_instr_to_vable`.
            if sym.owns_virtualizable_shadow() {
                let last_instr_value = py_pc as i64 - 1;
                let last_instr_op = ctx.trace_ctx.const_int(last_instr_value);
                crate::trace_opcode::mirror_vable_static_to_boxes(
                    ctx.trace_ctx,
                    "last_instr",
                    last_instr_op,
                    Value::Int(last_instr_value),
                );
                // Publish `valuestackdepth` for THIS guard's resume
                // coordinate the same way `last_instr` is published above.
                // `sym.valuestackdepth` is NOT usable: the walker never
                // crosses `set_orgpc`, so that scalar keeps the loop-entry
                // inputarg the trace was seeded with (a loop-invariant the
                // optimizer folds to a constant = the loop-header depth).  A
                // guard that resumes at a different depth — the `while`
                // condition branch, or a may-force call whose
                // `semantic_fallthrough_pc` lands on an opcode that keeps an
                // operand-stack temp (`#124`) — then carries the wrong depth
                // into its snapshot.  The resume reader writes this scalar
                // into `frame.valuestackdepth`; an under-count hides the kept
                // operand (the interpreter resumes thinking the stack is
                // shallower than it is), and `setup_bridge_sym` derives
                // `stack_only = valuestackdepth - nlocals` from it
                // (`state.rs` Part 1), so a wrong count desyncs the bridge's
                // operand-stack slots.  Compute the depth at the resume py_pc
                // the SAME way the encoder (`collect_outer_active_boxes`)
                // derives `valid_stack_only` — `nlocals +
                // depth_at_py_pc[py_pc]` — so the published scalar stays
                // symmetric with the active-box layout.  Fall back to
                // `sym.valuestackdepth` only when liveness is unavailable.
                let vsd_value = unsafe {
                    let jc = &*sym.jitcode();
                    if jc.payload.code_ptr.is_null() {
                        sym.valuestackdepth() as i64
                    } else {
                        match resume_depth_twin {
                            Some(d) => (sym.nlocals() + d as usize) as i64,
                            None => crate::liveness::liveness_for(jc.payload.code_ptr)
                                .depth_at_py_pc()
                                .get(py_pc as usize)
                                .copied()
                                .map(|d| (sym.nlocals() + d as usize) as i64)
                                .unwrap_or(sym.valuestackdepth() as i64),
                        }
                    }
                };
                let vsd_op = ctx.trace_ctx.const_int(vsd_value);
                crate::trace_opcode::mirror_vable_static_to_boxes(
                    ctx.trace_ctx,
                    "valuestackdepth",
                    vsd_op,
                    Value::Int(vsd_value),
                );
            }
            // Sync the live operand-stack registers into the vable shadow for
            // THIS snapshot only.  The walker keeps the operand stack in the
            // walk register file (`ctx.registers_r`); the vable shadow array
            // syncs the operand stack only at merge points
            // (`close_loop_args_at`) and is otherwise NULL for the slots a
            // mid-opcode guard resumes before (the two operands a
            // value-guarded `BINARY_OP` / `BINARY_SUBSCR` pops).  The
            // blackhole rebuilds those from the frame snapshot
            // (`collect_outer_active_boxes`), but the bridge re-trace reads
            // them back from the resumed vable shadow — without this overlay it
            // constant-folds NULL operands into the residual call
            // (`CallR(binop, null, null, ...)`) and the compiled bridge then
            // dereferences a null operand.  Overlay the live operands, build
            // the snapshot, then restore the shadow so this transient write
            // never leaks into a later op or merge-point sync.
            // Whether this is a kept-stack branch guard. Its resume/liveness
            // window is keyed by the guard's own jitcode pc
            // (`scope.branch_guard_jitcode_pc`, resolved to `entry_jitcode_pc`
            // below), so only its presence is needed here.
            let has_branch_guard = scope.branch_guard_jitcode_pc.is_some();
            let stack_sync: Vec<(usize, OpRef)> = if sym.owns_virtualizable_shadow() {
                let depth = unsafe {
                    let jc = &*sym.jitcode();
                    if jc.payload.code_ptr.is_null() {
                        0usize
                    } else {
                        match resume_depth_twin {
                            Some(depth) => depth as usize,
                            None => crate::liveness::liveness_for(jc.payload.code_ptr)
                                .depth_at_py_pc()
                                .get(py_pc as usize)
                                .copied()
                                .unwrap_or(0) as usize,
                        }
                    }
                };
                let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                let nlocals = sym.nlocals();
                // #73: the walk-level operand-stack box mirror
                // (`ctx.vstack_boxes[s]`, maintained per-op by
                // `step_vstack_mirror`) is the analog of PyPy
                // `MIFrame.registers_r` and the sole kept-stack source at a
                // branch guard.  The retired flat `stack_slot_color_map` read
                // (`registers_r[color]` plus the #420 edge-move recovery) was
                // unreliable in loop traces: loop-carried slots are renamed to
                // inputarg colors and chordal coloring reuses colors, so a
                // static-color read returned a stale / reused box (the #424
                // merge-color staleness).  Dropping it was proven redundant —
                // force-off byte-identical on 37/169 corpus benches on both
                // backends — so a slot the mirror does not cover (invalid
                // mirror, slot beyond the mirror, or an Int-bank temp the
                // Ref-only mirror leaves NONE) is simply omitted; resume
                // re-materializes it rather than reading the flat color.
                (0..depth)
                    .filter_map(|s| {
                        let v = if ctx.vstack_valid {
                            ctx.vstack_boxes.get(s).copied().unwrap_or(OpRef::NONE)
                        } else {
                            OpRef::NONE
                        };
                        if v != OpRef::NONE && !opref_is_null_const_ptr(v) {
                            Some((nvs + nlocals + s, v))
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };
            // Recover a kept operand-stack slot the walk mirror lost across a
            // not-taken branch merge.  At a branch guard whose not-taken arm
            // keeps a CALL result deep on the operand stack
            // (`t=(g(i),h(i),(g(i) if p or q else h(i)))`), the walk took the
            // OTHER arm, so `ctx.vstack_boxes[s]` is a `PY_NULL` hole for the
            // deep slot and the `stack_sync` overlay above omits it — the
            // resumed vable array then reads that slot NULL and the interpreted
            // body dereferences it (SIGSEGV).  The trampoline edge-move recovery
            // (`resolved_recovered`) is empty for this shape, so the mirror is
            // the only kept-stack source and it has a hole.
            // Fill the hole from the guard-PC register file: the per-PC
            // `pcdep_color_slots` map at the guard's own jitcode pc names the
            // Ref-bank color that holds operand-stack slot `nlocals + s` at THIS
            // guard PC (the same authoritative twin `collect_outer_active_boxes` reads),
            // and `ctx.registers_r[color]` holds the live guard-state box.  This
            // is the guard-PC color read (as `resolved_recovered` does for
            // `registers_r[src]`), NOT the retired stale merge-color read.
            // This ports `get_list_of_active_boxes`
            // (rpython/jit/metainterp/pyjitpl.py), which captures guard
            // resume boxes from `registers_r[index]` via the per-PC `-live-`
            // set.
            // Capture-only: writes the transient snapshot overlay, never the live
            // shadow (the trace_opcode.rs bridge-NULL constraint holds).
            let mut stack_sync: Vec<(usize, OpRef)> = if has_branch_guard
                && sym.owns_virtualizable_shadow()
                && !sym.jitcode().is_null()
            {
                // The pcdep twin here must be the plain predecessor-keyed flavor
                // (no Python-trivia skip), keyed by the guard's own jitcode pc.
                let gjc = scope.branch_guard_jitcode_pc.unwrap();
                let nlocals = sym.nlocals();
                let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                let depth = unsafe {
                    let jc = &*sym.jitcode();
                    jc.payload.depth_for_jitcode_pc_pred(gjc).unwrap_or(0) as usize
                };
                let pcdep: Vec<(u8, u16, u16)> = unsafe {
                    let jc = &*sym.jitcode();
                    jc.payload.pcdep_for_jitcode_pc(gjc).unwrap_or_default()
                };
                let mut covered: std::collections::HashSet<usize> =
                    stack_sync.iter().map(|&(idx, _)| idx).collect();
                let mut augmented = stack_sync;
                for s in 0..depth {
                    let vidx = nvs + nlocals + s;
                    if covered.contains(&vidx) {
                        continue;
                    }
                    // Guard-PC Ref color that owns operand-stack slot
                    // `nlocals + s` (`get_list_of_active_boxes` `if length_r:`
                    // section, `pyjitpl.py`).
                    if let Some(color) =
                        crate::state::semantic_slot_color_for_ref_slot(&pcdep, nlocals + s)
                    {
                        if let Some(&box_op) = ctx.registers_r.get(color) {
                            // Only a genuine Ref box may fill an operand-stack
                            // slot: the vable array is uniformly Ref-typed, and
                            // `build_vable_snapshot_boxes` reads each entry's
                            // `OpRef::ty()`.  An unboxed-int kept temp (a `Ref`-
                            // bank color holding an `IntAdd` result, e.g.
                            // condexpr's `a+i`) is Int-typed and would decode as
                            // `Box type Int != expected Ref`. Boxing it after
                            // the guard is appended would make the guard
                            // snapshot reference a box defined after the guard,
                            // so the box was never computed on guard failure.
                            // RPython materializes failargs before appending the
                            // guard (`optimizer.py`).
                            //
                            // The hole stands because pyre elides the int box at
                            // the tracer layer, losing the descr/known_class/
                            // field structure deopt needs. RPython elides at
                            // the optimizer layer (`virtualize.py`),
                            // keeps `InstancePtrInfo`, and serializes a
                            // TAGVIRTUAL recipe (`resume.py`)
                            // materialized lazily only on guard failure
                            // (`resume.py`). The orthodox fix is
                            // push-time boxing plus optimizer virtualization,
                            // beyond this capture hook.
                            if box_op != OpRef::NONE
                                && !opref_is_null_const_ptr(box_op)
                                && box_op.ty() == Some(majit_ir::Type::Ref)
                            {
                                augmented.push((vidx, box_op));
                                covered.insert(vidx);
                            }
                        }
                    }
                }
                augmented
            } else {
                stack_sync
            };
            // A branch guard (`scope.branch_guard_jitcode_pc`, consumed by the
            // stack overlay above for the #124 kept-stack source recovery) keys
            // the encoder liveness window on the guard's own jitcode pc — the
            // resume `py_pc` is a not-taken merge point whose live colors the
            // walk has not written.
            // #124 Approach B (M2): carry the guard's raw JitCode byte offset
            // as the resume coordinate ONLY for branch guards — they supply
            // their own pc in `GuardCaptureScope::branch_guard_jitcode_pc`, the
            // kept-stack-across-branch precision `setposition(jitcode,
            // miframe.pc)` preserves and the lossy `py_pc → jitcode`
            // resume-translation collapses.
            //
            // Every other guard (guard_value / guard_class / guard_no_exception,
            // the `after_residual_call` family) resumes at a `py_pc` whose
            // operand stack is in a deterministic state with no kept temp.
            // Carrying `op_pc` for those broke encoder ↔ decoder
            // symmetry: `collect_outer_active_boxes` resolves the reg banks at
            // the carried coordinate but `live_locals` / `stack_color_map` at
            // the entry coordinate, and for a non-branch guard
            // `op_pc != marker`
            // — the two windows diverge and the decoded box layout mismatches.
            let guard_jitcode_pc: i32 = if let Some(carried) = scope.carried_resume_jit_pc {
                // Bridge-entry flavor guard: re-carry the source guard's own
                // resume word verbatim (see `GuardCaptureScope`).  It is a
                // decodable `-live-` startpoint by construction — the runtime
                // resolved it to reach this walk.
                carried as i32
            } else if let Some(guard_jc_pc) = scope.branch_guard_jitcode_pc {
                // The kept-stack branch guard's own `op.pc` (walker
                // `MIFrame.pc`) — the ONE carried word not sourced from the
                // resume-translation.
                guard_jc_pc as i32
            } else if !after_residual_call
                && matches!(
                    ctx.trace_ctx.last_guard_opcode(),
                    Some(
                        OpCode::GuardValue
                            | OpCode::GuardClass
                            | OpCode::GuardTrue
                            | OpCode::GuardFalse
                    )
                )
            {
                // #366: carry the `-live-` marker offset, NOT the raw guard
                // `op_pc`. Reached only on the
                // default-scope path (`scope.branch_guard_jitcode_pc.is_none()`): the
                // specialization guards (`GuardValue`/`GuardClass`) and the
                // depth-0 branch guards (`GuardTrue`/`GuardFalse`, kept-stack
                // depth>0 branches take the first arm carrying `op.pc`).  For
                // every guard here the codewrite-time twin resolves the
                // `-live-` marker, keeping the
                // encoder reg-bank window and decoder liveness symmetric.  It is
                // a valid startpoint (`can_decode_live_vars` holds).
                // The `after_residual_call` family is excluded — it routes
                // through the separate post-call catch-marker twin, so the
                // ordinary marker would name a different resume point.
                // Every guard-capture point is emitted after a `-live-` marker;
                // its populated codewrite-time twin is therefore total here.
                let marker = unsafe {
                    (&*sym.jitcode())
                        .payload
                        .resume_marker_for_jitcode_pc(op_pc)
                };
                // A specialization guard (`GuardValue`/`GuardClass`) sources its
                // resume coordinate from the walk cursor's per-op `-live-`
                // BEFORE anchor (`ctx.live_before_jit_pc`, `pyjitpl.py`)
                // directly, dropping the py_pc-keyed block-head lookup.
                // Requires a stepped `-live-` and a resolvable
                // block-head marker (else keep the baseline, byte-identical).
                // The `PYRE_M73_PEROP_AUDIT` census certifies both offsets
                // decode identically for every consumer (banks + bridge maps +
                // const refill) where the anchor coincides, and flags the
                // divergent minority for the check.py output-equality gate.
                // A depth-0 branch guard (`GuardTrue`/`GuardFalse`) carries the walk's
                // arm-independent `-live-` BEFORE anchor (`ctx.live_before_jit_pc`,
                // `orgpc`) TAGGED into the negative space of the word plus the
                // guard flavor, instead of the py_pc-keyed block-head `marker`.
                // Carried ONLY when the tagged word's decode-side expansion
                // (`expand_branch_carried`, the same fn every consumer runs)
                // reproduces today's `marker` (self-cert) — so decode expands it
                // back to exactly `marker` and the encode is byte-identical by
                // construction. Requires a stepped `-live-` anchor in i16-tag
                // range (`BRANCH_ORGPC_MAX`) and a resolvable `marker`; otherwise
                // fall through to the perop / marker paths unchanged.
                let flavor_true =
                    matches!(ctx.trace_ctx.last_guard_opcode(), Some(OpCode::GuardTrue));
                let is_branch = matches!(
                    ctx.trace_ctx.last_guard_opcode(),
                    Some(OpCode::GuardTrue | OpCode::GuardFalse)
                );
                if is_branch
                    && ctx.live_before_jit_pc != usize::MAX
                    && ctx.live_before_jit_pc <= majit_ir::resumedata::BRANCH_ORGPC_MAX
                    && marker.is_some()
                    && {
                        // Self-certify byte-identity: the tagged word must expand
                        // back to today's carried `marker`.
                        let tagged = majit_ir::resumedata::encode_branch_orgpc(
                            ctx.live_before_jit_pc,
                            flavor_true,
                        );
                        let jc = unsafe { &*sym.jitcode() };
                        expand_branch_carried(&jc.payload, tagged)
                            == marker
                                .map(|m| m as i32)
                                .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC)
                    }
                {
                    majit_ir::resumedata::encode_branch_orgpc(ctx.live_before_jit_pc, flavor_true)
                } else if ctx.live_before_jit_pc != usize::MAX
                    && matches!(
                        ctx.trace_ctx.last_guard_opcode(),
                        Some(OpCode::GuardValue | OpCode::GuardClass)
                    )
                    && unsafe {
                        (&*sym.jitcode())
                            .payload
                            .jitcode
                            .can_decode_live_vars(ctx.live_before_jit_pc, crate::state::op_live())
                    }
                {
                    ctx.live_before_jit_pc as i32
                } else {
                    match marker {
                        Some(jp) => jp as i32,
                        None => majit_ir::resumedata::NO_JITCODE_PC,
                    }
                }
            } else if after_residual_call {
                // #366: extend the direct-pc carry to the after-residual-call
                // guard family (`GuardException`/`GuardNoException`/
                // `GuardNotForced`/`GuardAlwaysFails` — exactly what the
                // `after_residual_call` bool tracks, `walker_capture_snapshot_for_last_guard`).
                // These resume AFTER the residual call at the marker the
                // decoder already resolves for this guard, so carrying it
                // directly makes decode consult the carried word instead of a
                // fallback `py_pc → jitcode` resume translation.
                //
                // Which marker the decoder resolves depends on the sub-case
                // captured at the CALL pc:
                //   * `Some(call_jit_pc)` — the residual call is in a try-block
                //     (FOR_ITER-next catch resume): carry that same post-call
                //     catch `-live-` offset.
                //   * `None` — plain sequential residual call resuming at the
                //     next opcode's start marker: decode routes the plain word
                //     through the resume-translation, so carry
                //     the ordinary post-call marker.
                //     `py_pc == liveness_py_pc` here: the residual-call path
                //     never supplies `GuardCaptureScope::branch_guard_jitcode_pc`
                //     (only kept-stack branch guards do), so this is not a
                //     branch guard.
                // Both offsets are the SAME physical post-call `-live-` insn by
                // codewriter construction (the plain fallthrough re-key and the
                // catch-predecessor anchor resolve to one marker), so the
                // carried value is identical to the decoder's fallback by
                // construction — encoder reg-bank window and decoder liveness
                // stay symmetric, and both pass `can_decode_live_vars` (a
                // `-live-` marker). Fall back to the sentinel on a map miss.
                let marker = unsafe {
                    let jc = &*sym.jitcode();
                    match marker_call_jit_pc {
                        Some(call_jit_pc) => jc
                            .payload
                            .after_residual_call_resume_for_jitcode_pc(call_jit_pc),
                        None => {
                            // Every plain post-call guard capture is emitted
                            // after its fallthrough `-live-` marker, so this
                            // populated twin is total at the capture point.
                            jc.payload
                                .after_residual_marker_for_jitcode_pc(op_pc)
                                .or_else(|| {
                                    jc.payload.after_residual_call_resume_for_jitcode_pc(op_pc)
                                })
                                .or_else(|| trailing_live_marker(&jc.payload, op_pc))
                        }
                    }
                };
                match marker {
                    Some(jp) => jp as i32,
                    None => majit_ir::resumedata::NO_JITCODE_PC,
                }
            } else {
                // The remaining nonbranch guards (outside the
                // arm-2 allow-list, not after-residual) carry the same
                // block-head marker as arm-2, sourced from the jitcode-keyed
                // twin at the guard's own `op_pc`.
                let twin = unsafe {
                    (&*sym.jitcode())
                        .payload
                        .resume_marker_for_jitcode_pc(op_pc)
                };
                match twin {
                    Some(jp) => jp as i32,
                    None => majit_ir::resumedata::NO_JITCODE_PC,
                }
            };
            // The liveness-window py coordinate. A branch guard keys its
            // liveness window on the guard's own jitcode pc (via
            // `entry_jitcode_pc` below); every other `liveness_py_pc` consumer
            // sits on a non-branch-guard path — the `after_residual_call`
            // bridge-semantic-maps call below and the fallthrough
            // `first_floor_boundary_for_py` entry lookup, both reached only
            // when `has_branch_guard` is false — so this is always the merge
            // `py_pc`.
            let liveness_py_pc = py_pc;
            let payload = unsafe { &(&*sym.jitcode()).payload };
            let resolved = payload
                .resolve_resume_pc_with_jitcode_pc(guard_jitcode_pc, crate::state::op_live());
            let Some(resolved_offset) = resolved else {
                return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc });
            };
            // A residual result is installed in the active register bank before
            // the exception guard captures resume data
            // (`rpython/jit/metainterp/pyjitpl.py`).  Project every
            // live Ref register of the owning frame through the resume
            // position's color-to-slot map so the separately encoded
            // virtualizable array read by `virtualizable.py` carries the
            // same locals and operand-stack values as the frame section.  Inner
            // frames have no ownership of this shadow and are handled by the
            // multi-frame path above.
            if after_residual_call && sym.owns_virtualizable_shadow() {
                let maps = crate::state::bridge_semantic_maps_at_with_jitcode_pc(
                    jitcode_index as i32,
                    liveness_py_pc as i32,
                    guard_jitcode_pc,
                );
                let banks = crate::state::frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
                    jitcode_index as i32,
                    guard_jitcode_pc,
                );
                let semantic_limit = sym.nlocals() + maps.stack_depth_at_pc;
                let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                for &(bank, color, slot) in &maps.pcdep_entries {
                    if bank != 1
                        || slot as usize >= semantic_limit
                        || !banks.ref_.contains(&(color as u32))
                    {
                        continue;
                    }
                    let Some(&value) = ctx.registers_r.get(color as usize) else {
                        continue;
                    };
                    if value == OpRef::NONE || value.ty() != Some(majit_ir::Type::Ref) {
                        continue;
                    }
                    let vable_index = nvs + slot as usize;
                    if let Some((_, current)) = stack_sync
                        .iter_mut()
                        .find(|(index, _)| *index == vable_index)
                    {
                        *current = value;
                    } else {
                        stack_sync.push((vable_index, value));
                    }
                }
            }
            let saved_shadow: Vec<(usize, Option<OpRef>)> = stack_sync
                .iter()
                .map(|&(idx, _)| (idx, ctx.trace_ctx.virtualizable_box_at(idx)))
                .collect();
            for &(idx, value) in &stack_sync {
                ctx.trace_ctx.set_virtualizable_box_at(idx, value);
            }
            let (vable_boxes, vref_boxes) = ctx.trace_ctx.build_snapshot_vable_vref_boxes();
            for (idx, old) in saved_shadow {
                if let Some(old) = old {
                    ctx.trace_ctx.set_virtualizable_box_at(idx, old);
                }
            }
            let (entry_jitcode_pc, entry_twin, entry_caller) = if has_branch_guard {
                let entry_jitcode_pc = unsafe {
                    let metadata = &(&*sym.jitcode()).payload.metadata;
                    (guard_jitcode_pc >= 0)
                        .then(|| {
                            crate::pyjitcode::floor_segment_for_jitcode_pc(
                                &metadata.py_floor_by_jit_pc,
                                guard_jitcode_pc as usize,
                            )
                        })
                        .flatten()
                        .map_or(majit_ir::resumedata::NO_JITCODE_PC, |(pc, _)| pc as i32)
                };
                (
                    entry_jitcode_pc,
                    OuterActiveBoxesEntryTwin::Plain,
                    "guard_snapshot_guard_pc",
                )
            } else {
                let entry = unsafe {
                    first_floor_boundary_for_py(&(&*sym.jitcode()).payload.metadata, liveness_py_pc)
                        .map(|(pc, _)| pc)
                };
                match entry {
                    Some(pc) => (
                        pc as i32,
                        OuterActiveBoxesEntryTwin::Plain,
                        "guard_snapshot_fallthrough",
                    ),
                    None => (
                        resolved_offset as i32,
                        OuterActiveBoxesEntryTwin::Trivia,
                        "guard_snapshot_fallthrough",
                    ),
                }
            };
            let active = collect_outer_active_boxes(
                sym,
                ctx.trace_ctx,
                ctx.registers_i,
                ctx.registers_r,
                ctx.registers_f,
                jitcode_index,
                has_branch_guard,
                guard_jitcode_pc,
                entry_jitcode_pc,
                entry_twin,
                entry_caller,
                ctx.vstack_valid.then_some(ctx.vstack_boxes.as_slice()),
                scope.branch_guard_kept_recovered,
            );
            let pc_word = resolved_offset as u32;
            let forward_py_pc = forward_snapshot_py_pc(jitcode_index, pc_word)?;
            ctx.trace_ctx
                .capture_snapshot_for_last_guard_with_vable_vref(
                    &active,
                    jitcode_index,
                    pc_word,
                    forward_py_pc,
                    &vable_boxes,
                    &vref_boxes,
                );
            return Ok(());
        }
    }

    // Per-opcode arm path: `op_pc` is arm-local, so its snapshot must use
    // the carried static outer JitCode coordinate; an absent coordinate
    // declines instead of reconstructing one from the Python pc.
    let (vable_boxes, vref_boxes) = ctx.trace_ctx.build_snapshot_vable_vref_boxes();
    let arm_word = ctx
        .outer_resume_marker_jit_pc
        .map(|m| m as i32)
        .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC);
    let Some(arm_pc_word) =
        crate::state::pyjitcode_for_jitcode_index(ctx.outer_jitcode_index as i32)
            .and_then(|payload| {
                let resolved =
                    payload.resolve_resume_pc_with_jitcode_pc(arm_word, crate::state::op_live());
                resolved
            })
            .map(|offset| offset as u32)
    else {
        return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc });
    };
    let arm_py_pc = forward_snapshot_py_pc(ctx.outer_jitcode_index, arm_pc_word)?;
    ctx.trace_ctx
        .capture_snapshot_for_last_guard_with_vable_vref(
            &ctx.outer_active_boxes,
            ctx.outer_jitcode_index,
            arm_pc_word,
            arm_py_pc,
            &vable_boxes,
            &vref_boxes,
        );
    Ok(())
}

/// Build the paused caller frame for a multi-frame inline snapshot (#68),
/// computed at the inline CALL site where the caller's live register banks
/// (`ctx.registers_*`) are still in scope.  `call_jit_pc` is the CALL op's
/// jitcode pc in the caller.  Returns a named decline reason
/// when the caller frame is not snapshot-able for this first slice: missing
/// liveness / resume tables, a CLOSURE callee at a CALL inside a try-block
/// whose `except` handler does NOT rejoin a loop (it reads caller cells the
/// handler cleanup clears, see
/// [`decline_inline_caller_frame_for_catch_marker`]), or no result on the
/// operand stack at the return point.
///
/// The caller resumes at the CALL's return point (fallthrough) with the
/// not-yet-produced call-result slot nulled — `get_list_of_active_boxes(
/// in_a_call=true)` parity (`trace_opcode.rs`). Reuses
/// [`collect_outer_active_boxes`] (the caller owns the portal virtualizable)
/// after temporarily nulling the result slot's register.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InlineCallerFrameDecline {
    TryBlockCatchMarker,
    Unavailable,
}

pub(crate) fn decline_inline_caller_frame_for_catch_marker(
    after_residual_call_resume: Option<usize>,
    caller_jitcode: &[u8],
    callee_has_freevars: bool,
) -> Result<(), InlineCallerFrameDecline> {
    let Some(resume_pos) = after_residual_call_resume else {
        // Not a try-block CALL — nothing to route on a raise.
        return Ok(());
    };
    // A closure callee reads its captured values out of the cells the caller
    // frame owns, and inside a try-block one of those cells is the `except E as
    // e` binding, which the handler's implicit cleanup stores `None` into and
    // then clears.  Inlining such a callee here reads that cell as `None`
    // (`bench/synth/exception_as_cell_cleanup`).  A callee with no free
    // variables cannot reach a caller cell at all, so the decline below is not
    // needed for it.
    if !callee_has_freevars {
        return Ok(());
    }
    // The CALL is inside a try-block.  Inline it when its exception handler
    // rejoins a loop (the exc-edge-bridgeable shape): the paused caller frame
    // resumes at the CALL fallthrough on the no-raise path, and on a raise the
    // caller's `lastblock` (a static box in its virtualizable image) unwinds to
    // the catch handler in the blackhole — bit-exact — while a hot raise bridges
    // into the enclosing loop via the carrier-boundary delivery
    // (`drive_bridge_carrier_walk`'s `finishframe_exception`).
    //
    // The rejoin test is a TRACEBACK constraint here, not a bridgeability one:
    // the exc-edge router itself no longer needs it (`bridge_subwalk.rs` routes
    // on the catch alone, as `pyjitpl.py:2530-2546` does).  Inlining a caller
    // whose handler returns out of the frame instead makes
    // `exception_traceback_frame_lineno` report the raising frame twice, once at
    // the wrong lineno — the catching frame's traceback node is dropped when the
    // inlined callee's raise is delivered.  Keep the decline until that is
    // fixed; both backends reproduce it.
    //
    // `resume_pos` is the CALL's post-call `live/`; the `catch_exception/L` for
    // the enclosing try sits right after it (`finishframe_exception` lookahead),
    // so read the handler target forward from there.
    let rejoins = crate::jitcode_dispatch::try_catch_exception_at(caller_jitcode, resume_pos)
        .is_some_and(|catch_target| {
            crate::jitcode_dispatch::exc_handler_rejoins_loop(caller_jitcode, catch_target)
        });
    if rejoins {
        return Ok(());
    }
    Err(InlineCallerFrameDecline::TryBlockCatchMarker)
}

pub(crate) fn concrete_ref_for_color<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    color: usize,
) -> Option<pyre_object::PyObjectRef> {
    if let Some(ConcreteValue::Ref(ptr)) = ctx.concrete_registers_r.get(color) {
        if !ptr.is_null() {
            return Some(*ptr);
        }
    }
    let opref = ctx.registers_r.get(color).copied()?;
    match ctx.trace_ctx.concrete_of_opref(opref) {
        Some(Value::Ref(r)) if !r.is_null() => Some(r.as_usize() as pyre_object::PyObjectRef),
        _ => None,
    }
}

pub(crate) fn concrete_ref_for_opref<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    opref: OpRef,
) -> Option<pyre_object::PyObjectRef> {
    match ctx.trace_ctx.concrete_of_opref(opref) {
        Some(Value::Ref(r)) => Some(r.as_usize() as pyre_object::PyObjectRef),
        _ => None,
    }
}

pub(crate) fn collect_call_stack_overrides<Sym: WalkSym>(
    caller_sym: &Sym,
    ctx: &WalkContext<'_, '_, Sym>,
    call_jitcode_pc: usize,
) -> Vec<(usize, pyre_object::PyObjectRef)> {
    if caller_sym.jitcode().is_null() {
        return Vec::new();
    }
    let (nlocals, depth, pcdep_entries) = unsafe {
        let jc = &*caller_sym.jitcode();
        if jc.payload.code_ptr.is_null() {
            return Vec::new();
        }
        // The caller CALL key is a plain JitCode→Python inversion, so use the
        // certified predecessor twins rather than the marker/trivia flavor
        // reserved for the CALL fallthrough below.
        let depth = jc
            .payload
            .depth_for_jitcode_pc_pred(call_jitcode_pc)
            .unwrap_or(0) as usize;
        let pcdep = jc
            .payload
            .pcdep_for_jitcode_pc(call_jitcode_pc)
            .unwrap_or_default();
        (caller_sym.nlocals(), depth, pcdep)
    };
    let stack_end = nlocals + depth;
    if depth == 0 {
        return Vec::new();
    }
    let mut overrides = Vec::new();
    if ctx.vstack_valid && ctx.vstack_depth == depth && ctx.vstack_boxes.len() >= depth {
        for d in 0..depth {
            let slot = nlocals + d;
            if let Some(value) = concrete_ref_for_opref(ctx, ctx.vstack_boxes[d]) {
                overrides.push((slot, value));
            }
        }
    }
    if pcdep_entries.is_empty() {
        for slot in nlocals..stack_end {
            if overrides.iter().any(|&(present, _)| present == slot) {
                continue;
            }
            if let Some(value) = concrete_ref_for_color(ctx, slot) {
                overrides.push((slot, value));
            }
        }
    } else {
        for &(bank, color, slot) in &pcdep_entries {
            let slot = slot as usize;
            if bank == 1 && slot >= nlocals && slot < stack_end {
                if overrides.iter().any(|&(present, _)| present == slot) {
                    continue;
                }
                if let Some(value) = concrete_ref_for_color(ctx, color as usize) {
                    overrides.push((slot, value));
                }
            }
        }
    }
    // Use the virtualizable shadow only after the live register/vstack
    // sources, and only for slots those sources left unresolved.  A mid-opcode
    // shadow stack slot can be NULL because it was not mirrored, while the
    // corresponding live color still holds the value — so a NULL reaching this
    // fallback is an UNRESOLVED slot, not a proven null (see the per-slot note
    // below), and is dropped rather than committed.
    let base = ctx
        .trace_ctx
        .virtualizable_info()
        .map(|info| info.num_static_extra_boxes)
        .unwrap_or(0);
    for slot in nlocals..stack_end {
        if overrides.iter().any(|&(present, _)| present == slot) {
            continue;
        }
        if let Some((_opref, Value::Ref(value))) = ctx.trace_ctx.virtualizable_entry_at(base + slot)
        {
            // Only a positively-resolved (non-null) shadow value is a faithful
            // stack slot.  The live vstack/color sources above emit a genuine
            // null-or-self sentinel only where they hold a box for the slot at
            // all — the `PUSH_NULL` position has none, which is what the
            // `call_null_or_self_slot` pass below exists to cover.  A slot that
            // reaches this shadow fallback with a NULL Ref is one the walk could
            // not resolve — e.g. an
            // unmaterialized `LOAD_CONST` operand whose concrete value was
            // never mirrored — not a real null.  Leaving it ABSENT makes the
            // outer-call flush validation decline, so the legacy replay
            // rebuilds the frame from its start state instead of resuming the
            // interpreter over a NULL where a live object belongs.
            if value.as_usize() != 0 {
                overrides.push((slot, value.as_usize() as pyre_object::PyObjectRef));
            }
        }
    }
    // The `null_or_self` operand a `PUSH_NULL` leaves under a `CALL` is the one
    // stack slot NO source above can speak for: nothing pushes a value there,
    // so the walk holds no box and no live color for it, and the shadow's NULL
    // is the same NULL an unmirrored slot reads back as. It therefore stays
    // absent and declines the outer-call flush — on a slot whose correct value
    // is exactly that null. The CALL's own operand layout names it without
    // guessing: `[callable, null_or_self, arg0 .. arg_{argc-1}]` ends at
    // `stack_end`, so the sentinel sits `argc + 1` below it, right under the
    // arguments and right above the callable.
    if let Some(slot) = call_null_or_self_slot(caller_sym, call_jitcode_pc, stack_end)
        && slot >= nlocals
        && !overrides.iter().any(|&(present, _)| present == slot)
    {
        overrides.push((slot, std::ptr::null_mut::<u8>() as pyre_object::PyObjectRef));
    }
    overrides
}

/// Absolute frame slot holding the `null_or_self` operand of the `CALL` whose
/// JitCode coordinate is `call_jitcode_pc`, for a caller whose operand stack
/// ends at `stack_end`. `None` when that coordinate does not invert to a plain
/// `CALL` — every other resume shape keeps the conservative decline.
fn call_null_or_self_slot<Sym: WalkSym>(
    caller_sym: &Sym,
    call_jitcode_pc: usize,
    stack_end: usize,
) -> Option<usize> {
    let jc = unsafe { caller_sym.jitcode().as_ref()? };
    let code = unsafe { (jc.payload.code_ptr as *const pyre_interpreter::CodeObject).as_ref()? };
    let py_pc =
        crate::jitcode_dispatch::python_pc_for_jitcode_pc(&jc.payload.metadata, call_jitcode_pc)
            as usize;
    let (pyre_interpreter::Instruction::Call { argc }, op_arg) =
        pyre_interpreter::decode_instruction_at(code, py_pc)?
    else {
        return None;
    };
    stack_end.checked_sub(argc.get(op_arg) as usize + 1)
}

fn capture_inline_parent_blackhole<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    jitcode_index: u32,
    call_jit_pc: usize,
) -> Option<InlineParentBlackhole> {
    let pjc = crate::state::pyjitcode_for_jitcode_index(jitcode_index as i32)?;
    let call = decode_op_at(&pjc.jitcode.code, call_jit_pc)?;
    let result_bank = call.argcodes.chars().last()?;
    let result_color = (result_bank != 'v')
        .then(|| pjc.jitcode.code.get(call.next_pc.checked_sub(1)?).copied())
        .flatten()
        .map(usize::from);
    let live = crate::state::frame_liveness_reg_indices_by_bank_at(
        jitcode_index as i32,
        i32::try_from(call.next_pc).ok()?,
    );
    let mut int_values = Vec::with_capacity(live.int.len());
    for &color in &live.int {
        let color = color as usize;
        if result_bank == 'i' && result_color == Some(color) {
            continue;
        }
        let ConcreteValue::Int(value) = ctx.concrete_registers_i.get(color).copied()? else {
            return None;
        };
        int_values.push((color, value));
    }

    let mut ref_values = Vec::with_capacity(live.ref_.len());
    for &color in &live.ref_ {
        let color = color as usize;
        if result_bank == 'r' && result_color == Some(color) {
            continue;
        }
        // `concrete_registers_r` is the color-indexed shadow of `registers_r`:
        // it is allocated to `num_regs_r + constants_r.len()` and every walker
        // handler writes `concrete_registers_r[dst]` in lock-step with
        // `registers_r[dst]`.  `semantic_ref_slot_for_reg_color` maps a color
        // to a `locals_cells_stack_w` slot, which is what
        // `restore_guard_failure_values` needs to call `set_local_at` /
        // `set_stack_at` on a concrete PyFrame — a different index space.
        // Reading the shadow through it stamped whatever register happened to
        // live at the slot's number.
        let ConcreteValue::Ref(value) = ctx.concrete_registers_r.get(color).copied()? else {
            return None;
        };
        ref_values.push((color, value));
    }

    let mut float_values = Vec::with_capacity(live.float.len());
    for &color in &live.float {
        let color = color as usize;
        if result_bank == 'f' && result_color == Some(color) {
            continue;
        }
        let opref = ctx.registers_f.get(color).copied()?;
        if opref == OpRef::NONE {
            return None;
        }
        float_values.push((color, opref));
    }

    Some(InlineParentBlackhole {
        resume_pc: call.next_pc,
        int_values,
        ref_values,
        float_values,
    })
}

/// Return the exact JitCode coordinate at which a paused caller continues
/// after an inlined residual call.
///
/// RPython's `MIFrame.pc` remains immediately after the call instruction
/// while the callee runs.  Preserve that shape here: blackhole return setup
/// walks backward from this coordinate to recover the call's destination
/// register.  Advancing to a later semantic-fallthrough marker skips the
/// intervening `live` / virtualizable synchronization instructions and makes
/// that backward walk read an unrelated operand byte.
pub(crate) fn inline_call_return_marker(
    pjc: &crate::pyjitcode::PyJitCode,
    call_jit_pc: usize,
) -> Option<usize> {
    let marker =
        crate::jitcode_runtime::decode_op_at(pjc.jitcode.code.as_slice(), call_jit_pc)?.next_pc;
    pjc.jitcode
        .can_decode_live_vars(marker, crate::state::op_live())
        .then_some(marker)
}

pub(crate) fn compute_inline_caller_frame<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_jit_pc: usize,
    callee_has_freevars: bool,
) -> Result<InlineParentFrame, InlineCallerFrameDecline> {
    // #68 nested multiframe: when an inlined callee is already active
    // (the framestack has a top level), the immediate caller of THIS
    // call is that intermediate callee (a sym-less sub-jitcode), not the
    // top-level `fbw_mode.snapshot_sym`.  Compute its paused frame from that
    // jitcode (index / liveness / resume tables via its `PyJitCode`), reading the
    // boxes from the caller's live register banks (`ctx.registers_*`, which ARE
    // the intermediate callee's banks here) via the sym-less
    // `collect_callee_active_boxes`.  The stack is empty for a top-level
    // caller, falling through to the `fbw_mode.snapshot_sym` path below.
    let caller_code = ctx
        .session
        .borrow()
        .framestack
        .last()
        .map(|frame| frame.w_code);
    if let Some(caller_code) = caller_code {
        return compute_nested_inline_caller_frame(
            ctx,
            call_jit_pc,
            caller_code,
            callee_has_freevars,
        );
    }
    let caller_sym_ptr = ctx.fbw_mode.snapshot_sym;
    if caller_sym_ptr.is_null() {
        return Err(InlineCallerFrameDecline::Unavailable);
    }
    // SAFETY: set for the lifetime of the top-level `dispatch_via_miframe`;
    // read-only access to immutable layout fields.
    let caller_sym = unsafe { &*caller_sym_ptr };
    if caller_sym.jitcode().is_null() {
        return Err(InlineCallerFrameDecline::Unavailable);
    }
    let (jitcode_index, fallthrough_py_pc, resume_marker_jit_pc, code_ptr) = unsafe {
        let jc = &*caller_sym.jitcode();
        if jc.payload.code_ptr.is_null() || !jc.payload.is_populated() {
            return Err(InlineCallerFrameDecline::Unavailable);
        }
        let call_py = python_pc_for_jitcode_pc(&jc.payload.metadata, call_jit_pc) as usize;
        // A CALL inside a try-block: inline it only when its exception handler
        // rejoins the loop (the exc-edge-bridgeable shape); otherwise decline so
        // the residual path handles the raise via its after-residual catch
        // marker without a per-iteration deopt.
        decline_inline_caller_frame_for_catch_marker(
            jc.payload
                .after_residual_call_resume_for_jitcode_pc(call_jit_pc),
            jc.payload.jitcode.code.as_slice(),
            callee_has_freevars,
        )?;
        let code = &*jc.payload.code_ptr;
        let fallthrough = crate::pyjitpl::semantic_fallthrough_pc(code, call_py) as u32;
        // #73 Slice 4 (twin-first): certify the forward `after_residual_fallthrough`
        // twin reproduces the inverted-then-fallthrough coordinate before any
        // consumer cuts over to it.
        if pcmap_afterresidual_audit_enabled()
            && jc.payload.after_residual_fallthrough_py_pc_populated()
        {
            assert_eq!(
                jc.payload
                    .after_residual_fallthrough_py_pc_for_jitcode_pc(call_jit_pc),
                Some(fallthrough),
                "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: inline-caller fallthrough-py twin diverged at jit_pc {call_jit_pc} (call_py {call_py})"
            );
        }
        (
            jc.index as u32,
            fallthrough,
            inline_call_return_marker(&jc.payload, call_jit_pc),
            jc.payload.code_ptr,
        )
    };
    // The call result is the top operand-stack slot at the return point.
    let depth = match resume_marker_jit_pc {
        Some(marker) => unsafe { &(*caller_sym.jitcode()).payload }
            .depth_trivia_for_jitcode_pc(marker)
            .unwrap_or(0) as usize,
        // A missing marker has no fallthrough-native key. Preserve the
        // existing Python-keyed path rather than declining a formerly valid
        // caller frame.
        None => {
            let payload = unsafe { &(*caller_sym.jitcode()).payload };
            let raw = || unsafe {
                crate::liveness::liveness_for(code_ptr)
                    .depth_at_py_pc()
                    .get(fallthrough_py_pc as usize)
                    .copied()
                    .unwrap_or(0) as usize
            };
            if payload.depth_after_residual_populated() {
                let depth = payload
                    .depth_after_residual_for_jitcode_pc(call_jit_pc)
                    .unwrap_or(0) as usize;
                if pcmap_afterresidual_audit_enabled() {
                    assert_eq!(
                        depth,
                        raw(),
                        "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: inline-caller after-residual depth twin diverged at jit_pc {call_jit_pc} (py {fallthrough_py_pc})"
                    );
                }
                depth
            } else {
                raw()
            }
        }
    };
    if depth == 0 {
        return Err(InlineCallerFrameDecline::Unavailable);
    }
    let call_stack_overrides = collect_call_stack_overrides(caller_sym, ctx, call_jit_pc);
    // #73: the result slot's color comes from the codewriter-precomputed
    // `result_color_at_pc` (top-of-stack color at the return pc), not the flat
    // `stack_slot_color_map` — the result is not a live Variable here, so it
    // carries no pcdep entry.
    let result_color = {
        let payload = unsafe { &(*caller_sym.jitcode()).payload };
        // The trivia twin resolves the return marker only while that marker
        // doubles as the fallthrough PC's own resume marker (a pc_map
        // block-head).  A CALL whose fallthrough carries its own per-PC
        // marker leaves the trailing marker un-keyed there — the
        // after-residual twin keys the same fallthrough coordinate by the
        // CALL's byte offset and stays exact in both shapes.
        resume_marker_jit_pc
            .and_then(|marker| payload.result_color_trivia_for_jitcode_pc(marker))
            .filter(|&color| color != u16::MAX)
            .or_else(|| {
                payload
                    .result_color_after_residual_for_jitcode_pc(call_jit_pc)
                    .filter(|&color| color != u16::MAX)
            })
            .map(|color| color as usize)
    }
    .ok_or(InlineCallerFrameDecline::Unavailable)?;
    // Null the not-yet-produced result slot, build the box list, then restore
    // the caller's register (the inlined callee, not the walk, produces the
    // result; the inner frame supplies it on resume).
    let null_ref = ctx.trace_ctx.const_ref(pyre_object::PY_NULL as i64);
    let saved = ctx.registers_r.get(result_color).copied();
    if result_color < ctx.registers_r.len() {
        ctx.registers_r[result_color] = null_ref;
    }
    // Keep the caller at the immediate post-call `-live-`, matching the
    // paused `MIFrame.pc` used by RPython and the blackhole return ABI.
    let caller_liveness_word = match resume_marker_jit_pc {
        Some(m) => m as i32,
        None => majit_ir::resumedata::NO_JITCODE_PC,
    };
    let boxes = collect_outer_active_boxes(
        caller_sym,
        ctx.trace_ctx,
        ctx.registers_i,
        ctx.registers_r,
        ctx.registers_f,
        jitcode_index,
        false,
        caller_liveness_word,
        caller_liveness_word,
        OuterActiveBoxesEntryTwin::Trivia,
        "inline_caller",
        None,
        &[],
    );
    if let (Some(saved), true) = (saved, result_color < ctx.registers_r.len()) {
        ctx.registers_r[result_color] = saved;
    }
    Ok(InlineParentFrame {
        jitcode_index,
        call_jitcode_pc: Some(call_jit_pc),
        call_stack_overrides,
        blackhole: capture_inline_parent_blackhole(ctx, jitcode_index, call_jit_pc),
        resume_coord: ParentResumeCoord::CallFallthrough(call_jit_pc),
        resume_marker_jit_pc,
        boxes,
    })
}

/// Paused-caller-frame computation for a NESTED multiframe inline (#68): the
/// immediate caller is an intermediate inlined callee (`caller_code`), a
/// sym-less sub-jitcode whose live boxes are in `ctx.registers_*` (this `ctx`
/// IS that callee's sub-walk).  Mirror of the top-level
/// [`compute_inline_caller_frame`] body but keyed on `caller_code`'s own
/// `PyJitCode` instead of `fbw_mode.snapshot_sym`, and reading boxes via the
/// sym-less [`collect_callee_active_boxes`] (no portal-vable shadow to fold).
pub(crate) fn compute_nested_inline_caller_frame<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_jit_pc: usize,
    caller_code: usize,
    callee_has_freevars: bool,
) -> Result<InlineParentFrame, InlineCallerFrameDecline> {
    let jitcode_index = crate::state::ensure_jitcode_index(caller_code as *const ())
        .ok_or(InlineCallerFrameDecline::Unavailable)? as u32;
    let pjc = crate::state::pyjitcode_for_jitcode_index(jitcode_index as i32)
        .ok_or(InlineCallerFrameDecline::Unavailable)?;
    if !pjc.is_populated() || pjc.code_ptr.is_null() {
        return Err(InlineCallerFrameDecline::Unavailable);
    }
    let resume_marker_jit_pc = inline_call_return_marker(&pjc, call_jit_pc);
    let after_residual_call_resume = pjc.after_residual_call_resume_for_jitcode_pc(call_jit_pc);
    // A CALL inside a try-block at inline depth ≥2: the rejoin-loop lift is
    // scoped to the top-level caller (`compute_inline_caller_frame`) for now, so
    // pass an empty jitcode here — a nested try-block CALL keeps declining
    // (its catch-marker resume is not yet routed through the deeper snapshot).
    decline_inline_caller_frame_for_catch_marker(
        after_residual_call_resume,
        &[],
        callee_has_freevars,
    )?;
    let legacy_fallthrough_py_pc = || unsafe {
        let call_py = python_pc_for_jitcode_pc(&pjc.metadata, call_jit_pc) as usize;
        let code = &*pjc.code_ptr;
        crate::pyjitpl::semantic_fallthrough_pc(code, call_py) as u32
    };
    // The call result is the top operand-stack slot at the return point.
    let depth = match resume_marker_jit_pc {
        Some(marker) => pjc.depth_trivia_for_jitcode_pc(marker).unwrap_or(0) as usize,
        None => {
            let raw = || {
                let fallthrough_py_pc = legacy_fallthrough_py_pc();
                unsafe {
                    crate::liveness::liveness_for(pjc.code_ptr)
                        .depth_at_py_pc()
                        .get(fallthrough_py_pc as usize)
                        .copied()
                        .unwrap_or(0) as usize
                }
            };
            if pjc.depth_after_residual_populated() {
                let depth = pjc
                    .depth_after_residual_for_jitcode_pc(call_jit_pc)
                    .unwrap_or(0) as usize;
                if pcmap_afterresidual_audit_enabled() {
                    assert_eq!(
                        depth,
                        raw(),
                        "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: nested inline-caller after-residual depth twin diverged at jit_pc {call_jit_pc}"
                    );
                }
                depth
            } else {
                raw()
            }
        }
    };
    if depth == 0 {
        return Err(InlineCallerFrameDecline::Unavailable);
    }
    // #73: result slot color from the precomputed `result_color_at_pc`, not
    // the flat `stack_slot_color_map` (see `compute_inline_caller_frame`).
    let result_color = match resume_marker_jit_pc {
        Some(marker) => pjc
            .result_color_trivia_for_jitcode_pc(marker)
            .filter(|&color| color != u16::MAX)
            .map(|color| color as usize),
        None => pjc
            .result_color_after_residual_for_jitcode_pc(call_jit_pc)
            .filter(|&color| color != u16::MAX)
            .map(|color| color as usize),
    }
    .ok_or(InlineCallerFrameDecline::Unavailable)?;
    // Null the not-yet-produced result slot, build the box list, then restore
    // the caller's register (the inlined callee, not the walk, produces the
    // result; the inner frame supplies it on resume) — same as the top-level
    // `in_a_call=true` shape.
    let null_ref = ctx.trace_ctx.const_ref(pyre_object::PY_NULL as i64);
    let saved = ctx.registers_r.get(result_color).copied();
    if result_color < ctx.registers_r.len() {
        ctx.registers_r[result_color] = null_ref;
    }
    // Keep the caller at the immediate post-call `-live-`, matching the
    // paused `MIFrame.pc` used by RPython and the blackhole return ABI.
    // Without a marker there is no coordinate to encode against, so the
    // sentinel declines the caller frame.
    let caller_liveness_word = match resume_marker_jit_pc {
        Some(m) => m as i32,
        None => majit_ir::resumedata::NO_JITCODE_PC,
    };
    let boxes = collect_callee_active_boxes(
        ctx.registers_i,
        ctx.registers_r,
        ctx.registers_f,
        jitcode_index,
        call_jit_pc,
        caller_liveness_word,
    );
    if let (Some(saved), true) = (saved, result_color < ctx.registers_r.len()) {
        ctx.registers_r[result_color] = saved;
    }
    let boxes = match boxes {
        Ok(b) => b,
        Err(_) => return Err(InlineCallerFrameDecline::Unavailable),
    };
    // #343 depth-2 operand flush.  `emit_new_pyframe_inline_with_params` seeds
    // this inlined-callee frame's `locals_cells_stack_w` virtual array with LOCALS
    // only; its live operand-stack slots live in the register file and are never
    // written back into the frame array.  When a guard fires inside a NESTED
    // inline (this frame is the paused MIDDLE caller), the multi-frame snapshot
    // captures this frame virtual and numbers its array — an unwritten operand
    // slot numbers to `NULLREF`, which `reconstruct_inline_recipe`
    // (state.rs, semantic-slot-ordered `arr[k]` read) decodes to a
    // concrete-NULL Ref box, aborting the reconstructed middle at its next
    // `CALL_MAY_FORCE` (`MayForceNullRefArgUnsupported`).  Materialize the live
    // operand boxes into the frame array now — at the point we pause this frame to
    // enter the nested inline — mirroring the locals seed at `helpers.rs`
    // (`virtualizable.py` write_boxes writes the frame's field boxes into
    // its array when it is forced).  Only NON-pending operand slots are written;
    // the top slot is the not-yet-produced nested-call result, delivered on resume
    // (`pending_result_abs_slot`), so the decoder skips it and so do we.  Emitting
    // here (before the nested callee's body walk) — not at capture — places the
    // stores ahead of every in-callee guard, so `_number_virtuals` reads the
    // populated array when it numbers those guards' resume data.
    if let Some(marker) = resume_marker_jit_pc {
        if caller_liveness_word != majit_ir::resumedata::NO_JITCODE_PC && depth > 1 {
            let (frame_reg, _) = crate::state::portal_red_regs_at(jitcode_index as i32);
            let frame_red = (frame_reg != u16::MAX)
                .then(|| ctx.registers_r.get(frame_reg as usize).copied())
                .flatten()
                .filter(|&r| r != OpRef::NONE);
            let locals_idx = crate::descr::pyframe_locals_cells_stack_descr().index();
            if let Some(locals_array) =
                frame_red.and_then(|fr| ctx.trace_ctx.heapcache_getfield_cached(fr, locals_idx))
            {
                // nlocals from the paused caller's own code (varnames count), the
                // same base the decoder uses (`code_ref.varnames.len()`).
                let nlocals = unsafe { &*pjc.code_ptr }.varnames.len();
                // `depth` is the operand-stack depth at the CALL return point,
                // top slot = the pending nested-call result.
                let pending_top_slot = nlocals + depth - 1;
                let maps = crate::state::bridge_semantic_maps_at_with_jitcode_pc(
                    jitcode_index as i32,
                    marker as i32,
                    caller_liveness_word,
                );
                let array_descr = crate::state::pyobject_gcarray_descr();
                // Heapcache array-element key = the virtualizable info's array
                // item descr, matching the seed store at `helpers.rs`.
                let item_descr_index = ctx
                    .trace_ctx
                    .virtualizable_info()
                    .map(|info| info.array_item_descr(0).index())
                    .unwrap_or_else(|| array_descr.index());
                // Live Ref colors at the CALL return coordinate, same enumeration
                // as `collect_callee_active_boxes`.
                let banks = crate::state::frame_liveness_reg_indices_by_bank_from_pc(
                    jitcode_index as i32,
                    caller_liveness_word,
                );
                for &color in &banks.ref_ {
                    let Some(slot) = crate::state::semantic_ref_slot_for_reg_color(
                        nlocals,
                        depth,
                        &maps.pcdep_entries,
                        color as usize,
                    ) else {
                        continue;
                    };
                    // Locals are already seeded; the pending result slot is
                    // delivered on resume — only sibling operands need the flush.
                    if slot < nlocals || slot == pending_top_slot {
                        continue;
                    }
                    let Some(&operand) = ctx.registers_r.get(color as usize) else {
                        continue;
                    };
                    if operand == OpRef::NONE {
                        continue;
                    }
                    let idx = ctx.trace_ctx.const_int(slot as i64);
                    // pyjitpl.py:3489-3521 `gen_store_back_in_vable`
                    // escape flush uses the standard recorded store path.
                    ctx.trace_ctx
                        .profiler()
                        .count_ops(OpCode::SetarrayitemGc, majit_metainterp::counters::OPS);
                    ctx.trace_ctx.profiler().count_ops(
                        OpCode::SetarrayitemGc,
                        majit_metainterp::counters::RECORDED_OPS,
                    );
                    ctx.trace_ctx.record_op_with_descr(
                        OpCode::SetarrayitemGc,
                        &[locals_array, idx, operand],
                        array_descr.clone(),
                    );
                    ctx.trace_ctx.heapcache_setarrayitem(
                        locals_array,
                        idx,
                        item_descr_index,
                        operand,
                    );
                }
            }
        }
    }
    Ok(InlineParentFrame {
        jitcode_index,
        call_jitcode_pc: Some(call_jit_pc),
        call_stack_overrides: Vec::new(),
        blackhole: capture_inline_parent_blackhole(ctx, jitcode_index, call_jit_pc),
        resume_coord: ParentResumeCoord::CallFallthrough(call_jit_pc),
        resume_marker_jit_pc,
        boxes,
    })
}

/// Capture the current inlined Python frame at a translated helper call's
/// entry boundary. The helper has no blackhole frame, so a guard inside it
/// must rebuild this frame at the CALL and re-execute the helper, while the
/// already-paused outer frames remain below it.
pub(crate) fn compute_inline_helper_call_entry_frame<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_jit_pc: usize,
) -> Result<InlineParentFrame, InlineCallerFrameDecline> {
    let caller_code = ctx
        .session
        .borrow()
        .framestack
        .last()
        .map(|frame| frame.w_code)
        .ok_or(InlineCallerFrameDecline::Unavailable)?;
    let jitcode_index = crate::state::ensure_jitcode_index(caller_code as *const ())
        .ok_or(InlineCallerFrameDecline::Unavailable)? as u32;
    let pjc = crate::state::pyjitcode_for_jitcode_index(jitcode_index as i32)
        .ok_or(InlineCallerFrameDecline::Unavailable)?;
    if !pjc.is_populated() || pjc.code_ptr.is_null() {
        return Err(InlineCallerFrameDecline::Unavailable);
    }
    let resume_marker_jit_pc = pjc
        .resume_marker_for_jitcode_pc(call_jit_pc)
        .ok_or(InlineCallerFrameDecline::Unavailable)?;
    let boxes = collect_callee_active_boxes(
        ctx.registers_i,
        ctx.registers_r,
        ctx.registers_f,
        jitcode_index,
        call_jit_pc,
        resume_marker_jit_pc as i32,
    )
    .map_err(|_| InlineCallerFrameDecline::Unavailable)?;
    Ok(InlineParentFrame {
        jitcode_index,
        call_jitcode_pc: None,
        call_stack_overrides: Vec::new(),
        blackhole: None,
        resume_coord: ParentResumeCoord::Backxlat(resume_marker_jit_pc),
        resume_marker_jit_pc: Some(resume_marker_jit_pc),
        boxes,
    })
}

fn publish_outermost_parent_vable_scalars<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    parent_frames: &[InlineParentFrame],
    op_pc: usize,
) -> Result<(), DispatchError> {
    let Some(outer) = parent_frames.first() else {
        return Err(DispatchError::callee_inline_unsupported(op_pc));
    };
    let caller_sym_ptr = ctx.fbw_mode.snapshot_sym;
    if caller_sym_ptr.is_null() {
        return Ok(());
    }
    let caller_sym = unsafe { &*caller_sym_ptr };
    if !caller_sym.owns_virtualizable_shadow() || caller_sym.jitcode().is_null() {
        return Ok(());
    }

    let resume_py_pc = resolve_parent_resume_py_pc(outer)
        .ok_or(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc })?;
    let last_instr_value = resume_py_pc as i64 - 1;
    let last_instr_op = ctx.trace_ctx.const_int(last_instr_value);
    crate::trace_opcode::mirror_vable_static_to_boxes(
        ctx.trace_ctx,
        "last_instr",
        last_instr_op,
        Value::Int(last_instr_value),
    );
    let vsd_value = unsafe {
        let jc = &*caller_sym.jitcode();
        if jc.payload.code_ptr.is_null() {
            caller_sym.valuestackdepth() as i64
        } else {
            let raw = || {
                crate::liveness::liveness_for(jc.payload.code_ptr)
                    .depth_at_py_pc()
                    .get(resume_py_pc as usize)
                    .copied()
            };
            let twin: Option<Option<u16>> = if outer.jitcode_index != jc.index as u32 {
                None
            } else {
                match outer.resume_coord {
                    ParentResumeCoord::Backxlat(jitcode_pc) => jc
                        .payload
                        .depth_trivia_populated()
                        .then(|| jc.payload.depth_trivia_for_jitcode_pc(jitcode_pc)),
                    ParentResumeCoord::CallFallthrough(call_jit_pc) => jc
                        .payload
                        .depth_after_residual_populated()
                        .then(|| jc.payload.depth_after_residual_for_jitcode_pc(call_jit_pc)),
                }
            };
            let depth = match twin {
                Some(d) => {
                    if pcmap_afterresidual_audit_enabled() {
                        assert_eq!(
                            d,
                            raw(),
                            "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: multiframe vsd depth twin diverged (py {resume_py_pc})"
                        );
                    }
                    d
                }
                None => raw(),
            };
            match depth {
                Some(d) => (caller_sym.nlocals() + d as usize) as i64,
                None => caller_sym.valuestackdepth() as i64,
            }
        }
    };
    let vsd_op = ctx.trace_ctx.const_int(vsd_value);
    crate::trace_opcode::mirror_vable_static_to_boxes(
        ctx.trace_ctx,
        "valuestackdepth",
        vsd_op,
        Value::Int(vsd_value),
    );
    Ok(())
}

fn walker_capture_transparent_helper_snapshot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    parent_frames: Vec<InlineParentFrame>,
    stamp_last_guard_op: bool,
) -> Result<(), DispatchError> {
    if parent_frames.is_empty() || !ctx.trace_ctx.vable_snapshot_buildable() {
        return Err(DispatchError::callee_inline_unsupported(op_pc));
    }
    publish_outermost_parent_vable_scalars(ctx, &parent_frames, op_pc)?;
    let (vable_boxes, vref_boxes) = ctx.trace_ctx.build_snapshot_vable_vref_boxes();
    let mut frames: Vec<(u32, u32, u32, &[OpRef])> = Vec::with_capacity(parent_frames.len());
    for frame in &parent_frames {
        let word = frame
            .resume_marker_jit_pc
            .map(|marker| marker as i32)
            .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC);
        let pc_word = crate::state::pyjitcode_for_jitcode_index(frame.jitcode_index as i32)
            .and_then(|payload| {
                payload.resolve_resume_pc_with_jitcode_pc(word, crate::state::op_live())
            })
            .map(|offset| offset as u32)
            .ok_or(DispatchError::GuardResumeCoordinateUnavailable { pc: op_pc })?;
        let py_pc = forward_snapshot_py_pc(frame.jitcode_index, pc_word)?;
        frames.push((frame.jitcode_index, pc_word, py_pc, frame.boxes.as_slice()));
    }
    if stamp_last_guard_op {
        ctx.trace_ctx
            .capture_snapshot_for_last_guard_op_multi_frame_with_vable_vref(
                &frames,
                &vable_boxes,
                &vref_boxes,
            );
    } else {
        ctx.trace_ctx
            .capture_snapshot_for_last_guard_multi_frame_with_vable_vref(
                &frames,
                &vable_boxes,
                &vref_boxes,
            );
    }
    Ok(())
}

/// Emit a multi-frame inline guard snapshot (#68): the inlined callee's OWN
/// (top/innermost) frame built from the live sub-walk register banks, plus the
/// pre-computed paused caller frame(s) on the walk framestack. Frame
/// order is OUTERMOST-FIRST (`recorder.rs` / `build_resumed_frames`
/// `eval.rs`): the parent chain followed by the callee top frame.  The
/// stale doc on `capture_snapshot_for_last_guard_multi_frame_with_vable_vref`
/// claiming `frames[0]=top` is wrong — the function writes frames verbatim.
pub(crate) fn walker_capture_multi_frame_inline_snapshot<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    callee_op_pc: usize,
    after_residual_call: bool,
    parent_frames: Vec<InlineParentFrame>,
    scope: GuardCaptureScope<'_>,
    // The `_nonstandard_virtualizable` promote guard is not the last recorded op
    // (`emit_force_virtualizable` records GETFIELD_GC / PTR_NE / COND_CALL after
    // it), so its caller stamps the resume position on the last *guard* op.
    // Straight-line / branch inline guards ARE the last op and pass `false`.
    stamp_last_guard_op: bool,
) -> Result<(), DispatchError> {
    if ctx.fbw_mode.transparent_helper_subwalk {
        return walker_capture_transparent_helper_snapshot(
            ctx,
            callee_op_pc,
            parent_frames,
            stamp_last_guard_op,
        );
    }
    if !ctx.trace_ctx.vable_snapshot_buildable() {
        return Err(DispatchError::GuardSnapshotVableUntyped { pc: callee_op_pc });
    }
    // Callee (top/innermost) frame: map the guard's jitcode pc to the callee's
    // own Python pc, read liveness from the live sub-walk register banks.
    let callee_w_code = ctx
        .session
        .borrow()
        .framestack
        .last()
        .map(|frame| frame.w_code);
    let Some(callee_w_code) = callee_w_code else {
        return Err(DispatchError::callee_inline_unsupported(callee_op_pc));
    };
    let Some(callee_jitcode_index) = crate::state::ensure_jitcode_index(callee_w_code as *const ())
    else {
        return Err(DispatchError::callee_inline_unsupported(callee_op_pc));
    };
    let Some(callee_pjc) = crate::state::pyjitcode_for_jitcode_index(callee_jitcode_index) else {
        return Err(DispatchError::callee_inline_unsupported(callee_op_pc));
    };
    if !callee_pjc.is_populated() || callee_pjc.code_ptr.is_null() {
        return Err(DispatchError::callee_inline_unsupported(callee_op_pc));
    }
    // Mirror of the single-frame path: the callee (top) frame carries a branch
    // guard's supplied pc (`GuardCaptureScope::branch_guard_jitcode_pc`)
    // unchanged. For other guards, the callee payload supplies the
    // resume-marker twin:
    // after-residual guards use the fallthrough twin and retain the sentinel
    // on a miss, while plain guards retain the raw `callee_op_pc` on a miss.
    // Computed before box collection so encoder liveness and decoder resume
    // use the same coordinate.
    let callee_jitcode_pc: i32 = match scope.branch_guard_jitcode_pc {
        Some(g) => g as i32,
        None if after_residual_call => callee_pjc
            .after_residual_marker_for_jitcode_pc(callee_op_pc)
            .or_else(|| {
                // Fallback only, for the same reason as the single-frame path:
                // the sticky cursor names this frame's op only while no other
                // residual has passed, so it may not displace the twin.
                (ctx.live_after_jit_pc != usize::MAX
                    && callee_pjc
                        .jitcode
                        .can_decode_live_vars(ctx.live_after_jit_pc, crate::state::op_live()))
                .then_some(ctx.live_after_jit_pc)
            })
            .map(|m| m as i32)
            .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC),
        // No `-live-` BEFORE anchor arm here, unlike the single-frame path.
        // The marker below is `callee_op_pc`'s codewrite-time twin: the
        // coordinate whose liveness window the boxes this frame is about to
        // collect were emitted against. `ctx.live_before_jit_pc` is merely
        // another `-live-` byte the walk stepped over, and passing
        // `can_decode_live_vars` only says its window decodes, not that it is
        // the one those boxes belong to. Carrying it makes
        // `collect_callee_active_boxes` size the section from one coordinate
        // while reading the values out of the sub-walk registers at another.
        // The single-frame path can substitute the anchor because it re-reads
        // the owning frame's vable shadow at the carried coordinate; the callee
        // sub-walk owns no shadow to re-read.
        None => callee_pjc
            .resume_marker_for_jitcode_pc(callee_op_pc)
            .map(|m| m as i32)
            .unwrap_or(callee_op_pc as i32),
    };
    let mf_diag = std::env::var_os("PYRE_FBW_MF_DIAG").is_some();
    let recipe_resultcolor_audit = pcmap_recipe_resultcolor_audit_enabled();
    if mf_diag || recipe_resultcolor_audit {
        let callee_py_pc = unsafe {
            let code = &*callee_pjc.code_ptr;
            let mut py = python_pc_for_jitcode_pc(&callee_pjc.metadata, callee_op_pc);
            py = skip_python_trivia_forward(code, py as usize) as u32;
            if after_residual_call {
                py = crate::pyjitpl::semantic_fallthrough_pc(code, py as usize) as u32;
            }
            py
        };
        if recipe_resultcolor_audit {
            // The snapshot selects this native resume coordinate below. For
            // non-branch captures, its marker construction already applies
            // the same trivia (and, after a residual call, semantic
            // fallthrough) transform as the diagnostic inversion above.
            let (site, native_marker) = match scope.branch_guard_jitcode_pc {
                Some(_) => ("mf_callee_inversion_branch_external", None),
                None if after_residual_call => (
                    "mf_callee_inversion_after_residual",
                    callee_pjc.after_residual_marker_for_jitcode_pc(callee_op_pc),
                ),
                None => (
                    "mf_callee_inversion_plain",
                    callee_pjc.resume_marker_for_jitcode_pc(callee_op_pc),
                ),
            };
            pcmap_recipe_resultcolor_audit_probe(site, "fire");
            let verdict = match (scope.branch_guard_jitcode_pc, native_marker) {
                (Some(_), _) => "branch_external",
                (None, Some(marker)) => {
                    let native_py = python_pc_for_jitcode_pc(&callee_pjc.metadata, marker) as usize;
                    if native_py == callee_py_pc as usize {
                        "eq"
                    } else {
                        "di"
                    }
                }
                (None, None) => "native_miss",
            };
            pcmap_recipe_resultcolor_audit_probe(site, verdict);
        }
        if mf_diag {
            let pcdep = callee_pjc
                .pcdep_for_jitcode_pc(callee_op_pc)
                .unwrap_or_default();
            let depth = callee_pjc
                .depth_for_jitcode_pc_pred(callee_op_pc)
                .unwrap_or(0);
            // Read the coordinate the box collection below actually uses, not
            // the raw `callee_op_pc`; and read it through the
            // decline-reporting form, since the lossy query maps an
            // unresolvable coordinate and a genuinely empty window onto the
            // same all-empty result.
            let resolved = crate::state::try_frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
                callee_jitcode_index as i32,
                callee_jitcode_pc,
            );
            let window = if resolved.is_some() {
                "resolved"
            } else {
                "UNRESOLVED"
            };
            let banks = resolved.unwrap_or_default();
            eprintln!(
                "[fbw-mf-diag] callee jc={callee_jitcode_index} op_pc={callee_op_pc} \
             py_pc={callee_py_pc} after_residual={after_residual_call} depth={depth} \
             pcdep_color_slots={pcdep:?}"
            );
            // The color→slot inversion must be read at the same coordinate as
            // the banks above; `pcdep` at the raw `callee_op_pc` can name a
            // different set, which is what makes a live color look unmapped.
            let coord_pcdep = usize::try_from(callee_jitcode_pc)
                .ok()
                .and_then(|coord| callee_pjc.pcdep_for_jitcode_pc(coord))
                .unwrap_or_default();
            eprintln!(
                "[fbw-mf-diag]   live banks: i={:?} r={:?} f={:?} coord={callee_jitcode_pc} \
                 window={window} coord_pcdep={coord_pcdep:?}",
                banks.int, banks.ref_, banks.float,
            );
            for &c in &banks.ref_ {
                eprintln!(
                    "[fbw-mf-diag]   regs_r[{c}] = {:?}",
                    ctx.registers_r.get(c as usize)
                );
            }
            for &c in &banks.int {
                eprintln!(
                    "[fbw-mf-diag]   regs_i[{c}] = {:?}",
                    ctx.registers_i.get(c as usize)
                );
            }
            unsafe {
                let code = &*callee_pjc.code_ptr;
                let lo = callee_py_pc.saturating_sub(3);
                for py in lo..callee_py_pc + 5 {
                    if let Some((instr, arg)) =
                        pyre_interpreter::decode_instruction_at(code, py as usize)
                    {
                        let mark = if py == callee_py_pc {
                            " <== resume"
                        } else {
                            ""
                        };
                        eprintln!("[fbw-mf-diag]   py{py}: {instr:?} arg={arg:?}{mark}");
                    }
                }
            }
        }
    }
    // `MIFrame.registers_r` is the live color bank, but a reconstructed
    // carrier frame also owns the semantic locals/stack image from which that
    // bank was rebuilt.  A post-call liveness marker can conservatively carry
    // a color whose current bank value is NONE (adjacent `-live-` union); map
    // that color back to this callee's OWN frame slot and source the box from
    // its frame-local shadow.  This is the inline-frame counterpart of
    // `collect_outer_active_boxes`' virtualizable-slot recovery, and preserves
    // the one-red-frame-per-MIFrame shape instead of falling back to the root.
    let mut recovered_regs_r = ctx.registers_r.to_vec();
    if recovered_regs_r.iter().any(|value| value.is_none()) && !callee_pjc.code_ptr.is_null() {
        let code = unsafe { &*callee_pjc.code_ptr };
        let (stack_base, _) = crate::state::callee_layout_for_call_assembler(code);
        let maps =
            crate::state::bridge_semantic_maps_from_pc(callee_jitcode_index, callee_jitcode_pc);
        if let Some(shadow) = ctx.callee_shadow.as_ref() {
            for (color, value) in recovered_regs_r.iter_mut().enumerate() {
                if !value.is_none() {
                    continue;
                }
                let Some(slot) = crate::state::semantic_ref_slot_for_reg_color(
                    stack_base,
                    maps.stack_depth_at_pc,
                    &maps.pcdep_entries,
                    color,
                ) else {
                    continue;
                };
                if let Some(&slot_value) = shadow.opref.get(&(slot as i64)) {
                    *value = slot_value;
                }
            }
        }
        // A dense fallthrough marker can name the NEXT Python opcode's result
        // color before that opcode has executed.  RPython's sparse stream has
        // no opcode-entry marker at this seam; the translated Ref bank slot is
        // still its empty/null value and the next opcode overwrites it before
        // any read.  Admit exactly that metadata-proven result color as NULL;
        // every other missing live color still declines below.
        if after_residual_call {
            let marker_result = usize::try_from(callee_jitcode_pc)
                .ok()
                .and_then(|coord| callee_pjc.result_color_trivia_for_jitcode_pc(coord))
                .filter(|&color| color != u16::MAX);
            let null_ref = ctx.trace_ctx.const_ref(pyre_object::PY_NULL as i64);
            seed_missing_fallthrough_result(&mut recovered_regs_r, marker_result, null_ref);
        }
    }
    let callee_boxes = collect_callee_active_boxes(
        ctx.registers_i,
        &recovered_regs_r,
        ctx.registers_f,
        callee_jitcode_index as u32,
        callee_op_pc,
        callee_jitcode_pc,
    )?;

    // Publish the OUTERMOST caller's vable scalars for its resume coordinate so
    // the resume reader restores the caller's `PyFrame` at the CALL return
    // point rather than the stale loop-header seed the walker never crosses
    // `set_orgpc` to update (mirror of the single-frame path above, 6366-6426).
    publish_outermost_parent_vable_scalars(ctx, &parent_frames, callee_op_pc)?;

    let (vable_boxes, vref_boxes) = ctx.trace_ctx.build_snapshot_vable_vref_boxes();

    // Frame tuples, OUTERMOST-FIRST: the paused caller chain, then the callee
    // top frame last (innermost).
    let mut frames: Vec<(u32, u32, u32, &[OpRef])> = Vec::with_capacity(parent_frames.len() + 1);
    for pf in &parent_frames {
        let pf_word = pf
            .resume_marker_jit_pc
            .map(|m| m as i32)
            .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC);
        let Some(pf_pc_word) = crate::state::pyjitcode_for_jitcode_index(pf.jitcode_index as i32)
            .and_then(|payload| {
                let resolved =
                    payload.resolve_resume_pc_with_jitcode_pc(pf_word, crate::state::op_live());
                resolved
            })
            .map(|offset| offset as u32)
        else {
            return Err(DispatchError::GuardResumeCoordinateUnavailable { pc: callee_op_pc });
        };
        let pf_py_pc = forward_snapshot_py_pc(pf.jitcode_index, pf_pc_word)?;
        frames.push((pf.jitcode_index, pf_pc_word, pf_py_pc, pf.boxes.as_slice()));
    }
    let callee_py_pc =
        forward_snapshot_py_pc(callee_jitcode_index as u32, callee_jitcode_pc as u32)?;
    frames.push((
        callee_jitcode_index as u32,
        callee_jitcode_pc as u32,
        callee_py_pc,
        callee_boxes.as_slice(),
    ));

    if stamp_last_guard_op {
        ctx.trace_ctx
            .capture_snapshot_for_last_guard_op_multi_frame_with_vable_vref(
                &frames,
                &vable_boxes,
                &vref_boxes,
            );
    } else {
        ctx.trace_ctx
            .capture_snapshot_for_last_guard_multi_frame_with_vable_vref(
                &frames,
                &vable_boxes,
                &vref_boxes,
            );
    }
    Ok(())
}

/// Seed only the not-yet-defined result color carried by pyre's dense
/// fallthrough marker.  Returns whether a hole was filled.
pub(crate) fn seed_missing_fallthrough_result(
    registers_r: &mut [OpRef],
    marker_result: Option<u16>,
    null_ref: OpRef,
) -> bool {
    let Some(dst) = marker_result
        .filter(|&color| color != u16::MAX)
        .map(usize::from)
    else {
        return false;
    };
    let Some(slot) = registers_r.get_mut(dst) else {
        return false;
    };
    if !slot.is_none() {
        return false;
    }
    *slot = null_ref;
    true
}
