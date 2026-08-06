//! Residual-call dispatch: the tracer path for a callee that cannot be
//! folded or inlined and must be recorded as a residual `CALL_*` operation.
//!
//! **Parity:** trace-side counterpart of `pyjitpl.py`'s
//! `opimpl_residual_call_*`; the executor fast paths call into
//! `majit-metainterp/executor.rs` (`executor.py`). PyPy keeps these opimpls
//! inside `pyjitpl.py`'s `MIFrame`; the split into this file is pyre-local
//! navigability, not a PyPy file boundary.
//!
//! Relocated verbatim from `jitcode_dispatch/mod.rs`. Covers the per-shape
//! dispatchers (`dispatch_residual_call_{iRd,iIRd,iIRFd}_kind`), the
//! executor fast paths (`try_fold_pure_call_via_executor`,
//! `try_execute_residual_call_via_executor`), opcode selection and arg
//! binding, the pre-call vable/vref sync, result writeback, and the
//! residual-call body classification helpers. The `residual_call_*` opname
//! arms themselves stay in `handle` (mod.rs) and call into these.

use super::*;

/// Which of [`flush_active_frame_escape`]'s two flushes committed the resume
/// pc.  They differ in exactly the way the walk-end commit contract cares
/// about, so the epilogue cannot classify the leg without being told.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum EscapeResumeKind {
    /// The latched mid-expression operand stack resolved.  The resume pc still
    /// re-runs the escaping opcode — both flushes take the same `py_pc` and the
    /// same `last_instr = py_pc - 1` — but this path is gated:
    /// [`escape_opcode_window_clean`] ran at the residual before the latch.
    Exact,
    /// The latch did not resolve and the flush fell back to the merge-point
    /// state.  Same resume pc, but NO gate ran on this path at any point.
    RerunsOpcode,
}

thread_local! {
    static ACTIVE_FRAME_ESCAPE: std::cell::Cell<Option<(usize, Option<usize>)>> =
        const { std::cell::Cell::new(None) };
    static ACTIVE_FRAME_ESCAPE_STACK: std::cell::RefCell<Option<Vec<OpRef>>> =
        const { std::cell::RefCell::new(None) };
    /// Step-0 attribution probe: classification of the most recent matched
    /// escape, read at the force site to attribute the token clear (and hence
    /// the abort) to the portal or the published-callee disjunct.
    static LAST_ESCAPE_WAS_CALLEE_ONLY: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static COMMITTED_FRAME_ESCAPE_PC: std::cell::Cell<Option<(usize, EscapeResumeKind)>> =
        const { std::cell::Cell::new(None) };
    /// Pre-flush frame state captured by [`flush_active_frame_escape`] so a
    /// post-call commit withdrawal can put the live frame back.  The legacy
    /// replay's correctness contract is "the live frame still holds pre-walk
    /// state"; a committed force-flush breaks it, so the replay leg restores
    /// this before re-entering.  Which leg runs is only known at walk end, so
    /// the withdrawal arms `ESCAPE_FLUSH_UNDO_PENDING` and the epilogue
    /// decides.
    static ESCAPE_FLUSH_UNDO: std::cell::RefCell<Option<EscapeFlushUndo>> =
        const { std::cell::RefCell::new(None) };
    /// Set when the force arm withdrew its commit: the pre-flush frame has to
    /// come back, but only on the legacy-replay leg (see
    /// `mark_escape_flush_undo_pending`).
    static ESCAPE_FLUSH_UNDO_PENDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Opcode-scoped purity window: `(py_pc, every_prior_residual_reentrant)`
    /// for the Python opcode currently being walked.  Re-executing a committed
    /// escape re-runs the WHOLE opcode, so the latch gate must know whether any
    /// EARLIER residual of the same opcode could have done something — not just
    /// the escaping one (rich-compare fallback chains run two residuals in one
    /// opcode).
    ///
    /// The verdict comes from each residual's DECLARED effect class
    /// ([`escape_opcode_window_note`], `EffectInfo::check_is_elidable` or
    /// `LoopInvariant`), not from counting what happened.  That is the axis
    /// upstream decides on: the licence to place a resume pc behind an executed
    /// residual is the codewriter-time `EF_ELIDABLE_CANNOT_RAISE` registration
    /// (`jtransform.py:620-630`), and upstream's only op counters are
    /// profiling-only and gate nothing (`jitprof.py:43-44`, `count_ops` is
    /// `pass`).
    ///
    /// Reset at walk start.  Residuals are the only executor that can put
    /// something irreversible between two points of ONE opcode: the eager
    /// journaled folds (`fbw_store_journal_push` / `fbw_append_journal_push` /
    /// `fbw_cell_store_journal_push`) each terminate their own opcode
    /// (STORE_SUBSCR, the `append` CALL, STORE_NAME/STORE_GLOBAL), so no
    /// residual of the same opcode instance can follow one.
    static ESCAPE_OPCODE_WINDOW: std::cell::Cell<Option<(usize, bool)>> =
        const { std::cell::Cell::new(None) };
    /// C3 S1 force-time image of the single live tracing frame.  The
    /// color-indexed concrete banks cease to exist when dispatch unwinds, so
    /// the MIFrame must be assembled beside `tracing_after_residual_call` and
    /// carried to the walk-end VableEscaped leg.
    static FBW_SINGLE_FRAME_BLACKHOLE: std::cell::RefCell<Option<LatchedSingleFrameBlackhole>> =
        const { std::cell::RefCell::new(None) };
    /// Force-time image of the paused caller chain plus the live innermost
    /// tracing frame.  Like the single-frame latch, this outlives the walk
    /// contexts whose concrete banks supplied it.
    static FBW_MULTI_FRAME_BLACKHOLE: std::cell::RefCell<Option<LatchedMultiFrameBlackhole>> =
        const { std::cell::RefCell::new(None) };
}

/// The concrete red frame owned by the current inlined MIFrame.
///
/// RPython carries this identity directly on every `MIFrame`; pyre's walker
/// brackets the corresponding per-thread execution state with
/// [`InlineConcreteFrameGuard`].  Vable writes use this accessor to keep that
/// frame's own heap image coherent for a later multi-frame blackhole handoff.
///
/// Read back out of the guard's root rather than from a raw copy: the value
/// is compared for identity against the concrete bank's own `Value::Ref`,
/// which the collector forwards, so a relocated frame would make the two
/// disagree and silently drop the mirror write.
pub(crate) fn current_inline_concrete_frame() -> usize {
    INLINE_CONCRETE_FRAME.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(0, |owner_root| owner_root.get().0)
    })
}

pub(crate) struct EscapeFlushUndo {
    frame: usize,
    last_instr: isize,
    valuestackdepth: usize,
    pub(crate) slots: Vec<pyre_object::PyObjectRef>,
}

/// The operand stack an abort image publishes, resolved from the walker's
/// OpRef mirror while the walk's concrete side tables are still live.
///
/// The detached tracing snapshot cannot supply it.  A root walk keeps its
/// virtualizable purely symbolic — `setarrayitem_vable_via_metainterp` mirrors
/// into a concrete frame only through `current_inline_vable_target`, i.e. only
/// for the frame an INLINE level owns — so the root snapshot's
/// `locals_cells_stack_w` still holds the stack from before the walk ran.  The
/// locals half has no such gap: `write_back_outer_locals` publishes those from
/// the virtualizable shadow.
pub(crate) struct MirrorStackImage {
    /// Python pc of the opcode the walk stopped inside.  Pairs with the slots:
    /// the mirror reflects the depth ON ENTRY to this opcode, which is the
    /// `last_instr = py_pc - 1` coordinate the codewriter stores for it.
    pub(crate) py_pc: usize,
    pub(crate) slots: Vec<pyre_object::PyObjectRef>,
}

pub(crate) struct LatchedSingleFrameBlackhole {
    pub(crate) miframe: majit_metainterp::MIFrame,
    pub(crate) last_exc_value: i64,
    pub(crate) raising_exception: bool,
    /// The walker's resolved operand-stack image, captured by whichever latch
    /// stops the walk.  The adopter uses it for legs that stop INSIDE an opcode
    /// (`WalkAbort` and `VableEscape`) and uses the frame's own snapshot array
    /// for `ABORT_TOO_LONG`, whose post-step coordinate is an opcode boundary
    /// the array does describe.
    ///
    /// `None` means the walk had no resolvable mirror, and the adopt then
    /// declines rather than publishing a partial stack.
    pub(crate) mirror_stack: Option<MirrorStackImage>,
}

pub(crate) struct LatchedMultiFrameBlackhole {
    pub(crate) framestack: majit_metainterp::MIFrameStack,
    pub(crate) last_exc_value: i64,
    pub(crate) raising_exception: bool,
    pub(crate) mirror_stack: Option<MirrorStackImage>,
    /// `ABORT_TOO_LONG` stops at an arbitrary post-step coordinate, so frame
    /// 0's active operand stack must cross from the detached tracing snapshot
    /// to the live red frame before the blackhole runs.  The vable-force path
    /// stops at a call resume marker and retains its existing handoff.
    pub(crate) publish_root_stack: bool,
}

pub(crate) fn single_frame_blackhole_cell_ptr()
-> *const std::cell::RefCell<Option<LatchedSingleFrameBlackhole>> {
    FBW_SINGLE_FRAME_BLACKHOLE.with(|cell| cell as *const _)
}

pub(crate) fn take_single_frame_blackhole() -> Option<LatchedSingleFrameBlackhole> {
    FBW_SINGLE_FRAME_BLACKHOLE.with(|slot| slot.borrow_mut().take())
}

pub(crate) fn multi_frame_blackhole_cell_ptr()
-> *const std::cell::RefCell<Option<LatchedMultiFrameBlackhole>> {
    FBW_MULTI_FRAME_BLACKHOLE.with(|cell| cell as *const _)
}

pub(crate) fn take_multi_frame_blackhole() -> Option<LatchedMultiFrameBlackhole> {
    FBW_MULTI_FRAME_BLACKHOLE.with(|slot| slot.borrow_mut().take())
}

/// True when an abort-recovery image is already staged.
///
/// The INNERMOST abort owns the handoff: its [`WalkContext`] is the one holding
/// the concrete banks of the frame the walk actually stopped in, and
/// `build_multi_frame_miframe` already appends every paused caller from
/// [`WalkSession::framestack`].  As the error propagates back out through the
/// enclosing `walk()` frames, each would otherwise re-latch and overwrite that
/// image with one rooted at the caller — resuming the blackhole above the frame
/// whose opcode was left half-executed.
pub(crate) fn abort_blackhole_latched() -> bool {
    FBW_SINGLE_FRAME_BLACKHOLE.with(|slot| slot.borrow().is_some())
        || FBW_MULTI_FRAME_BLACKHOLE.with(|slot| slot.borrow().is_some())
}

pub(crate) fn reset_single_frame_blackhole() {
    FBW_SINGLE_FRAME_BLACKHOLE.with(|slot| {
        *slot.borrow_mut() = None;
    });
    FBW_MULTI_FRAME_BLACKHOLE.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

/// Name each decline under `PYRE_FBW_DEBUG_ABORT`, the way `build_multi_frame_
/// miframe`'s `s2dbg!` and `try_adopt_multi_frame_blackhole`'s `mfdbg!` name
/// theirs: an unlatched abort and a latch that was never reached both end in the
/// same replay, and the two want different fixes.
macro_rules! latchdbg {
    ($($a:tt)*) => {
        if fbw_debug_abort_enabled() {
            eprintln!("[latch-decline] {}", format!($($a)*));
        }
    };
}

/// Snapshot the live meta-interpreter framestack for a `SwitchToBlackhole`
/// that stops the walk at an arbitrary coordinate.
///
/// Two triggers reach it, and both need the same image:
///
/// - `ABORT_TOO_LONG`.  RPython calls `blackhole_if_trace_too_long()`
///   immediately after `MIFrame.run_one_step()` and
///   `convert_and_run_from_pyjitpl` copies every live MIFrame at its
///   already-advanced `pc` (`pyjitpl.py:2863-2866`, `blackhole.py:1799-1821`).
///   `resume_pc` is `walk()`'s post-step `next_pc`.
/// - A bridge carrier sub-walk that stopped on a walker capability gap.  That
///   sub-walk IS the reconstructed callee's one real execution (see
///   `drive_bridge_frame_subwalk`'s `is_authoritative_executor` contract), so
///   the drain may not discard it and let the guard resume from `rd_numb` —
///   that re-runs every residual it already ran.  `resume_pc` is the
///   unexecuted instruction the walk stopped at
///   ([`DispatchError::stop_pc`]), which is the same "arbitrary coordinate"
///   shape.  Upstream never rewinds an aborted bridge either:
///   `_handle_guard_failure` ends `assert False, "should always raise"`
///   (`pyjitpl.py:2956`) and the conversion continues from the frames
///   `interpret()` reached.
///
/// Either way the current [`WalkContext`] plus [`WalkSession::framestack`] own
/// the concrete banks for the live frame and all paused callers.
///
/// Return `false` without publishing a partial image when any live value is
/// unresolved.  A zero-effect walk may then use the legacy entry replay;
/// an effectful walk must keep recording until a complete image can be built,
/// because replaying it would apply the effect twice.
/// Resolve `WalkContext::vstack_boxes` — the walker's counterpart of
/// `MIFrame.registers_r` snapshotted by `get_list_of_active_boxes`
/// (`pyjitpl.py`) — into the concrete operand stack an abort image publishes.
///
/// Same resolution [`flush_escape_state_with_latched_stack`] performs for the
/// escape flush, and for the same reason: the walk-abort latch and the
/// vable-escape latch both stop inside a Python opcode, where no other source
/// describes the operand stack.  The tracing snapshot's array is not it — a
/// root walk mirrors nothing into that frame (see [`MirrorStackImage`]).
///
/// A recorded concrete NULL is a REAL operand — it is `PUSH_NULL`'s
/// `self_or_null` sentinel — so only `GcRef::NO_CONCRETE`, which is what
/// "unavailable" looks like, rejects a slot.  Any unresolved slot declines the
/// whole image, leaving the abort on the legacy entry replay.
fn capture_vstack_mirror_image<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    origin: &'static str,
) -> Option<MirrorStackImage> {
    if !ctx.vstack_valid {
        latchdbg!("origin={origin} mirror-invalid");
        return None;
    }
    let mut slots = Vec::with_capacity(ctx.vstack_boxes.len());
    for &opref in ctx.vstack_boxes.iter() {
        match ctx.trace_ctx.concrete_of_opref(opref) {
            Some(majit_ir::Value::Ref(value)) if value != majit_ir::GcRef::NO_CONCRETE => {
                slots.push(value.as_usize() as pyre_object::PyObjectRef);
            }
            other => {
                latchdbg!(
                    "origin={origin} mirror-slot {}/{} unresolved opref={opref:?} concrete={other:?}",
                    slots.len(),
                    ctx.vstack_boxes.len(),
                );
                return None;
            }
        }
    }
    Some(MirrorStackImage {
        py_pc: ctx.vstack_cur_pypc as usize,
        slots,
    })
}

pub(crate) fn latch_abort_blackhole<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    resume_pc: usize,
    origin: &'static str,
) -> bool {
    if !ctx.is_authoritative_executor {
        latchdbg!("origin={origin} not-authoritative");
        return false;
    }
    let last_exc_value = match ctx.last_exc_value_concrete {
        ConcreteValue::Ref(value) => value as i64,
        _ => 0,
    };

    if ctx.session.borrow().framestack.is_empty() && !ctx.fbw_mode.inline_subwalk {
        let Some((jitcode, cf_addr, live_root_addr)) = (unsafe {
            if ctx.fbw_mode.snapshot_sym.is_null() {
                None
            } else {
                let sym = &*ctx.fbw_mode.snapshot_sym;
                let jitcode = sym.jitcode();
                let forwarded_live_root = match ctx.trace_ctx.lookup_opref_concrete(sym.frame()) {
                    Some(majit_ir::Value::Ref(value)) if value.0 != 0 => value.0,
                    _ => sym.live_vable_frame_addr(),
                };
                (!jitcode.is_null()).then(|| {
                    (
                        (&(*jitcode).payload).jitcode.clone(),
                        sym.tracing_vable_frame_addr(),
                        forwarded_live_root,
                    )
                })
            }
        }) else {
            latchdbg!("origin={origin} no-snapshot-sym-jitcode");
            return false;
        };
        let Some(miframe) = build_trace_too_long_single_frame_miframe(ctx, jitcode, resume_pc)
        else {
            latchdbg!("origin={origin} sf-build-miframe");
            return false;
        };
        // `walk()` has already executed this step, so returning TraceTooLong
        // is safe only when every pre-drive adopter gate is known to pass.
        // RPython's `run_blackhole_interp_to_cancel_tracing()` cannot return
        // to entry replay. Keep the same boundary: incomplete images merely
        // keep recording until a later step supplies a complete handoff.
        let Some(jitcode_index) = i32::try_from(miframe.jitcode.index()).ok() else {
            latchdbg!("origin={origin} sf-jitcode-index");
            return false;
        };
        if ctx.trace_ctx.virtualizable_info().is_none()
            || crate::state::concrete_nlocals(cf_addr).is_none()
        {
            latchdbg!("origin={origin} sf-no-vinfo-or-nlocals");
            return false;
        }
        let root_addr = if live_root_addr != 0 {
            live_root_addr
        } else {
            cf_addr
        };
        let (frame_reg, _) = crate::state::portal_red_regs_at(jitcode_index);
        let vable_frame = miframe
            .ref_values
            .get(frame_reg as usize)
            .copied()
            .flatten()
            .unwrap_or(0) as usize;
        if vable_frame == 0
            || vable_frame != root_addr
            || crate::state::capture_frame_locals(vable_frame).is_none()
            || !crate::state::can_write_back_outer_locals(ctx.trace_ctx, vable_frame)
            || !crate::state::can_publish_frame_stack(cf_addr, vable_frame)
        {
            latchdbg!("origin={origin} sf-vable-frame-mismatch");
            return false;
        }
        // Keep the per-frame red identity seeded by `frame_box`.  The
        // authoritative walk executed against `snapshot_for_tracing`, but the
        // adopter validates this identity and publishes that snapshot's locals
        // to the matching live frame before driving the blackhole. Replacing
        // the register with the snapshot address would collapse those two
        // identities and fail the live-root check.
        // `ctx` IS the root walk on this arm (`framestack.is_empty() &&
        // !inline_subwalk`), so its mirror describes the very frame whose
        // operand stack the adopter publishes.  `ABORT_TOO_LONG` stops at an
        // opcode boundary and keeps the snapshot-array source; a capability-gap
        // abort stops mid-opcode and needs this.
        let mirror_stack = capture_vstack_mirror_image(ctx, origin);
        FBW_SINGLE_FRAME_BLACKHOLE.with(|slot| {
            *slot.borrow_mut() = Some(LatchedSingleFrameBlackhole {
                miframe,
                last_exc_value,
                raising_exception: false,
                mirror_stack,
            });
        });
        true
    } else if ctx.fbw_mode.inline_subwalk {
        let Some(framestack) =
            build_multi_frame_miframe(ctx, resume_pc, InnermostMiframeBuild::TraceTooLong, origin)
        else {
            latchdbg!("origin={origin} mf-build-miframe");
            return false;
        };
        if !multi_frame_blackhole_preflight(ctx, &framestack, origin) {
            latchdbg!("origin={origin} mf-preflight");
            return false;
        }
        FBW_MULTI_FRAME_BLACKHOLE.with(|slot| {
            *slot.borrow_mut() = Some(LatchedMultiFrameBlackhole {
                framestack,
                last_exc_value,
                raising_exception: false,
                publish_root_stack: true,
                // `ctx` is the INNERMOST callee here, so its mirror describes
                // that callee, not frame 0 whose stack the adopter publishes.
                // Frame 0's own stack is recorded per level in
                // `InlineParentFrame::call_stack_overrides`, but sparsely (only
                // the slots the caller CALL touched), so it cannot stand in for
                // a complete image.  Left `None`: the `WalkAbort` adopt then
                // declines and the abort keeps the legacy entry replay, which
                // is exactly the pre-leg behaviour.
                mirror_stack: None,
            });
        });
        true
    } else {
        latchdbg!(
            "origin={origin} no-arm framestack_empty={} inline_subwalk={}",
            ctx.session.borrow().framestack.is_empty(),
            ctx.fbw_mode.inline_subwalk
        );
        false
    }
}

/// Read-only counterpart of every adopter gate that can reject a latched
/// multi-frame image.  `ABORT_TOO_LONG` runs after the opcode's effects, so it
/// may publish the image only when the later handoff cannot fall back to entry
/// replay.  RPython needs no split preflight: its per-frame red virtualizable
/// is already the live MIFrame state copied by
/// `convert_and_run_from_pyjitpl`.
fn multi_frame_blackhole_preflight<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    framestack: &majit_metainterp::MIFrameStack,
    origin: &'static str,
) -> bool {
    if ctx.trace_ctx.virtualizable_info().is_none() || ctx.fbw_mode.snapshot_sym.is_null() {
        latchdbg!("origin={origin} pf-no-vinfo-or-sym");
        return false;
    }
    let sym = unsafe { &*ctx.fbw_mode.snapshot_sym };
    let snapshot = sym.tracing_vable_frame_addr();
    let live_root = match ctx.trace_ctx.lookup_opref_concrete(sym.frame()) {
        Some(majit_ir::Value::Ref(value)) if value.0 != 0 => value.0,
        _ => sym.live_vable_frame_addr(),
    };
    let root = if live_root != 0 { live_root } else { snapshot };
    // Named by the first capability that refuses, and reported only then.
    // Emitted ahead of the test, the line announced every passing preflight
    // under the decline tag as well, so a census could not tell a rejected root
    // from an accepted one.
    let refused = if crate::state::concrete_nlocals(snapshot).is_none() {
        Some("nlocals")
    } else if crate::state::capture_frame_locals(root).is_none() {
        Some("locals")
    } else if !crate::state::can_write_back_outer_locals(ctx.trace_ctx, root) {
        Some("writeback")
    } else if !crate::state::can_publish_frame_stack(snapshot, root) {
        Some("publish")
    } else {
        None
    };
    if let Some(refused) = refused {
        latchdbg!(
            "origin={origin} pf-root-caps snapshot={snapshot:#x} root={root:#x} refused={refused}"
        );
        return false;
    }

    let mut seen = Vec::with_capacity(framestack.frames.len());
    for (index, frame) in framestack.frames.iter().enumerate() {
        let Ok(jitcode_index) = i32::try_from(frame.jitcode.index()) else {
            latchdbg!("origin={origin} pf-jitcode-index");
            return false;
        };
        let frame_reg = crate::state::portal_red_regs_at(jitcode_index).0;
        if frame_reg == u16::MAX {
            latchdbg!("origin={origin} pf-frame-reg-none");
            return false;
        }
        let Some(frame_ptr) = frame.ref_values.get(frame_reg as usize).copied().flatten() else {
            latchdbg!(
                "origin={origin} pf-frame-ptr-unset index={index}/{} jitcode={} frame_reg={frame_reg}",
                framestack.frames.len(),
                frame.jitcode.name()
            );
            return false;
        };
        let frame_ptr = frame_ptr as usize;
        let Some(stack_base) = crate::state::concrete_nlocals(frame_ptr) else {
            latchdbg!("origin={origin} pf-nlocals");
            return false;
        };
        let Some(stack_depth) = crate::state::concrete_stack_depth(frame_ptr) else {
            latchdbg!("origin={origin} pf-stack-depth");
            return false;
        };
        let Some(array_len) = crate::state::concrete_frame_array_len(frame_ptr) else {
            latchdbg!("origin={origin} pf-array-len");
            return false;
        };
        if stack_depth < stack_base
            || stack_depth > array_len
            || (index == 0 && frame_ptr != root)
            || (index > 0 && frame_ptr == root)
            || seen.contains(&frame_ptr)
        {
            latchdbg!("origin={origin} pf-shape");
            return false;
        }
        seen.push(frame_ptr);
    }
    true
}

fn build_single_frame_miframe<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    jitcode: std::sync::Arc<majit_metainterp::jitcode::JitCode>,
    resume_pc: usize,
    lastop_result: Option<(char, usize, i64)>,
) -> Option<majit_metainterp::MIFrame> {
    // An MIFrame is only representable at a `-live-` anchored startpoint: the
    // blackhole reads its registers straight out of the copied banks, so a
    // coordinate whose live set cannot be decoded must decline here rather
    // than yield a frame with every register unset.  All callers pass a
    // post-call `next_pc` today, which always resolves; the precondition is
    // what keeps that true as the pc domain widens.
    let live = crate::state::try_frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
        i32::try_from(jitcode.index()).ok()?,
        i32::try_from(resume_pc).ok()?,
    )?;
    let mut miframe = majit_metainterp::MIFrame::new(jitcode.clone(), resume_pc);

    // pyjitpl.py make_result_of_lastop: the residual returned before its
    // ordinary dispatcher writeback, while `resume_pc` already points past
    // the call.  Stamp the just-produced value first: its destination is
    // normally live at `resume_pc`, but the concrete shadow still contains
    // Null/old data and must not make the all-live-color pass decline.
    if let Some((bank, color, value)) = lastop_result {
        match bank {
            'i' => *miframe.int_values.get_mut(color)? = Some(value),
            'r' => *miframe.ref_values.get_mut(color)? = Some(value),
            'f' => *miframe.float_values.get_mut(color)? = Some(value),
            'v' => {}
            _ => return None,
        }
    }

    // An `OpRef::None` register is an ABSENT BOX, not an unresolved one: the
    // walk never defined this color.  `blackhole.py:1711-1730
    // _copy_data_from_miframe` copies a register only `if box`, so upstream
    // leaves the blackhole's slot exactly as unset as the MIFrame's, and the
    // drive cannot read it — every path that reaches this coordinate with the
    // color live also defines it, or the jitcode would read an undefined
    // register.  It arrives undefined here because a `-live-` set is the union
    // over the paths INTO its coordinate, and the walk took one of them.
    // Declining refused the whole image over a register nothing will read; the
    // shape it kept on the replay path is a `CALL_FUNCTION_EX` whose escape
    // fell back to a legacy replay of an already-executed frame.
    // A color that HAS a box but no recoverable concrete still declines: there
    // the drive can read it and the walk lost the value.
    for &color in &live.int {
        let color = color as usize;
        if miframe.int_values.get(color).copied().flatten().is_some() {
            continue;
        }
        let ConcreteValue::Int(value) = ctx.concrete_registers_i.get(color).copied()? else {
            if ctx.registers_i.get(color).copied()?.is_none() {
                continue;
            }
            return None;
        };
        *miframe.int_values.get_mut(color)? = Some(value);
    }
    for &color in &live.ref_ {
        let color = color as usize;
        if miframe.ref_values.get(color).copied().flatten().is_some() {
            continue;
        }
        let value = match ctx.concrete_registers_r.get(color).copied()? {
            ConcreteValue::Ref(value) => value as i64,
            // ConcreteValue::Null is the walker's "unknown" sentinel, not a
            // proven Python null.  The register shadow never held this color's
            // concrete — recover it from the recorded box's producer (a getfield
            // chain rooted at a seeded input arg an inlined sub-walk read but
            // never concretized in this frame's shadow); decline only when even
            // that cannot resolve it.
            _ => {
                let opref = ctx.registers_r.get(color).copied()?;
                if opref.is_none() {
                    continue;
                }
                match ctx.trace_ctx.recover_ref_value(opref, 8) {
                    Some(majit_ir::Value::Ref(gc)) => gc.0 as i64,
                    _ => return None,
                }
            }
        };
        *miframe.ref_values.get_mut(color)? = Some(value);
    }
    for &color in &live.float {
        let color = color as usize;
        if miframe.float_values.get(color).copied().flatten().is_some() {
            continue;
        }
        let opref = ctx.registers_f.get(color).copied()?;
        if opref.is_none() {
            continue;
        }
        let Some(majit_ir::Value::Float(value)) = ctx.trace_ctx.concrete_of_opref(opref) else {
            return None;
        };
        *miframe.float_values.get_mut(color)? = Some(value.to_bits() as i64);
    }

    // Seed every REMAINING color the walk has a concrete value for, not just the
    // ones live at `resume_pc`.
    //
    // `_copy_data_from_miframe` (blackhole.py:1711-1730) copies
    // `range(num_regs_i/r/f())` — the WHOLE bank, filtering only on "the MIFrame
    // has a box here", never on liveness.  Seeding a liveness-selected subset is
    // the deviation, and it is unsound the moment the drive leaves the straight
    // line: the blackhole runs on to a `jit_merge_point`, whose own live set is
    // generally LARGER (a loop-carried value defined in the prologue and not
    // rewritten in the body is dead at a mid-body pc but live at the header).
    // Those colors then read back NULL, and a NULL written into a live
    // operand-stack slot faults the interpreter.
    //
    // Measured on the `getframe_root_loop_force_*` fixtures: live at the build
    // pc was ref [0, 1, 6] while the merge wanted [0, 1, 2, 3, 4], and 2/3/4 all
    // had concrete values in the walk's own bank the whole time.
    //
    // Seeding cannot introduce a stale value: a color the drive rewrites is
    // overwritten before any read, and a color it does not rewrite still holds
    // what the walk observed at the force point, which is what a merge reached
    // without an intervening definition expects.  Colors with no concrete value
    // stay unset, exactly as an absent upstream box does.
    for (color, slot) in miframe.int_values.iter_mut().enumerate() {
        if slot.is_none()
            && let Some(ConcreteValue::Int(value)) = ctx.concrete_registers_i.get(color).copied()
        {
            *slot = Some(value);
        }
    }
    for (color, slot) in miframe.ref_values.iter_mut().enumerate() {
        // `ConcreteValue::Null` is the walker's "unknown" sentinel, not a proven
        // Python null, so it seeds nothing.
        if slot.is_none()
            && let Some(ConcreteValue::Ref(value)) = ctx.concrete_registers_r.get(color).copied()
        {
            *slot = Some(value as i64);
        }
    }
    // `num_regs_f()` is the third bank `_copy_data_from_miframe` walks, so the
    // rationale above covers floats too.  There is no float concrete shadow —
    // resolve the recorded box, and leave the color unset when it has none.
    for color in 0..miframe.float_values.len() {
        if miframe.float_values[color].is_some() {
            continue;
        }
        let Some(&opref) = ctx.registers_f.get(color) else {
            continue;
        };
        if let Some(majit_ir::Value::Float(value)) = ctx.trace_ctx.concrete_of_opref(opref) {
            miframe.float_values[color] = Some(value.to_bits() as i64);
        }
    }

    Some(miframe)
}

/// Build the `ABORT_TOO_LONG` image at an arbitrary post-step JitCode pc.
///
/// Unlike the residual-call handoff above, this coordinate need not be a
/// `-live-` marker: RPython copies the complete MIFrame banks after every
/// `run_one_step`. Requiring marker liveness here would make an effectful
/// straight-line tail unbounded again whenever the limit lands between
/// markers.
pub(super) fn build_trace_too_long_single_frame_miframe<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    jitcode: std::sync::Arc<majit_metainterp::jitcode::JitCode>,
    resume_pc: usize,
) -> Option<majit_metainterp::MIFrame> {
    let mut miframe = majit_metainterp::MIFrame::new(jitcode, resume_pc);
    fill_trace_too_long_register_banks(ctx, &mut miframe).then_some(miframe)
}

/// Fill every currently-known register color, matching
/// `blackhole.py:_copy_data_from_miframe`.
///
/// The vable-escape latch above only needs the resume marker's live set: it
/// resumes immediately after one forcing residual whose result is supplied
/// separately. `ABORT_TOO_LONG` is different — the blackhole can follow
/// arbitrary control flow from a post-step pc before reaching a terminal, so
/// colors outside that marker-local live set may be read later. RPython copies
/// all three complete MIFrame banks; do the same for this abort instead of
/// relying on the narrower resume-liveness cache.
///
/// `_copy_data_from_miframe` (`blackhole.py:1713-1730`) guards every bank entry
/// with `if box is not None` and has no failing path — but there, a `None`
/// register is a **dead** one the jitcode's liveness already cleared, and every
/// register that survives holds a box with a value.  "Live, but the walk knows
/// no value" has no upstream counterpart, and skipping such a color would leave
/// its pre-sized blackhole register at zero for a resume that can follow
/// arbitrary control flow to an instruction that reads it.  So `OpRef::NONE`
/// (dead, upstream's `box is None`) is skipped, and a live color with no
/// concrete refuses the whole frame.
///
/// One live Ref color carries a value neither concrete lookup can express: the
/// dead-`box_bool` marker holds a comparison's raw truth Int in a Ref register.
/// It is unfillable by construction rather than unknown, so the Ref bank
/// reconstructs the singleton instead of refusing — see there.
fn fill_trace_too_long_register_banks<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    miframe: &mut majit_metainterp::MIFrame,
) -> bool {
    // Name the declining bank and color under `PYRE_FBW_DEBUG_ABORT`, the way
    // `build_multi_frame_miframe` names its own: "innermost declined" alone
    // does not say which register had no concrete.
    macro_rules! s2dbg {
        ($($a:tt)*) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[s2-fill-decline] {}", format!($($a)*));
            }
        };
    }
    for color in 0..miframe.int_values.len() {
        if let Some(value) = ctx
            .concrete_registers_i
            .get(color)
            .copied()
            .and_then(|value| match value {
                ConcreteValue::Int(value) => Some(value),
                ConcreteValue::Bool(value) => Some(i64::from(value)),
                _ => None,
            })
        {
            miframe.int_values[color] = Some(value);
            continue;
        }
        if miframe.int_values[color].is_none() {
            let Some(opref) = ctx.registers_i.get(color).copied() else {
                continue;
            };
            if opref == OpRef::NONE {
                continue;
            }
            let Some(majit_ir::Value::Int(value)) = ctx.trace_ctx.concrete_of_opref(opref) else {
                s2dbg!("int color={color} opref={opref:?} has no concrete");
                return false;
            };
            miframe.int_values[color] = Some(value);
        }
    }

    for color in 0..miframe.ref_values.len() {
        let opref = ctx.registers_r.get(color).copied();
        let forwarded = opref
            .filter(|&value| value != OpRef::NONE)
            .and_then(|value| {
                match ctx
                    .trace_ctx
                    .lookup_opref_concrete(value)
                    .or_else(|| ctx.trace_ctx.recover_ref_value(value, 8))
                {
                    Some(majit_ir::Value::Ref(value)) => Some(value.0 as i64),
                    _ => None,
                }
            });
        let from_shadow =
            ctx.concrete_registers_r
                .get(color)
                .copied()
                .and_then(|value| match value {
                    ConcreteValue::Ref(value) => Some(value as i64),
                    _ => None,
                });
        // The dead-`box_bool` marker writes a comparison's RAW TRUTH Int
        // straight into the Ref register rather than boxing it, on
        // `compare_box_provably_dead`'s proof that no GUARD RESUME snapshot can
        // observe the slot.  This bank copy is not a guard resume: the
        // blackhole runs forward from a post-step pc, and the very next opcode
        // is the `is_true` that reads exactly this color.  Neither arm above
        // can represent an Int in a Ref bank, so the color would be skipped and
        // the blackhole would read NULL — `is_true(NULL)` is false, which sends
        // a `while` straight to its exit arm and drops every remaining
        // iteration.  The truth value is known, so materialize the immortal
        // singleton the box would have produced and keep the image exact.
        //
        // The marker is recognisable because its sites record the truth against
        // ITSELF; a genuinely boxed bool records a distinct `boxed` key and its
        // concrete is a Ref, so it is already served by `forwarded`.
        let from_bool_marker = opref
            .filter(|&value| value != OpRef::NONE)
            .filter(|&value| bool_box_truth_lookup(value) == Some(value))
            .and_then(|value| match ctx.trace_ctx.lookup_opref_concrete(value) {
                Some(majit_ir::Value::Int(truth)) => {
                    Some(pyre_object::w_bool_from(truth != 0) as i64)
                }
                _ => None,
            });
        if let Some(value) = forwarded.or(from_shadow).or(from_bool_marker) {
            miframe.ref_values[color] = Some(value);
        } else if opref.is_some_and(|value| value != OpRef::NONE) {
            s2dbg!("ref color={color} opref={opref:?} has no concrete");
            return false;
        }
    }

    for color in 0..miframe.float_values.len() {
        if miframe.float_values[color].is_some() {
            continue;
        }
        let Some(opref) = ctx.registers_f.get(color).copied() else {
            continue;
        };
        if opref == OpRef::NONE {
            continue;
        }
        let Some(majit_ir::Value::Float(value)) = ctx.trace_ctx.concrete_of_opref(opref) else {
            s2dbg!("float color={color} opref={opref:?} has no concrete");
            return false;
        };
        miframe.float_values[color] = Some(value.to_bits() as i64);
    }
    true
}

#[derive(Clone, Copy)]
enum InnermostMiframeBuild {
    LiveMarker(Option<(char, usize, i64)>),
    TraceTooLong,
}

fn build_multi_frame_miframe<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    resume_pc: usize,
    innermost_build: InnermostMiframeBuild,
    origin: &'static str,
) -> Option<majit_metainterp::MIFrameStack> {
    macro_rules! s2dbg {
        ($($a:tt)*) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[s2-build-decline] {}", format!($($a)*));
            }
        };
    }
    // The build's one success report.  It has to carry its own tag: a census
    // groups these lines by their `[…-decline]` prefix, so a completed build
    // announced under that prefix is counted as a refusal.
    macro_rules! s2built {
        ($($a:tt)*) => {
            if fbw_debug_abort_enabled() {
                eprintln!("[s2-build-ok] {}", format!($($a)*));
            }
        };
    }
    let session = ctx.session.borrow();
    if session.framestack.is_empty() {
        s2dbg!(
            "origin={origin} framestack empty depth={} transparent_helper_subwalk={}",
            session.framestack.len(),
            ctx.fbw_mode.transparent_helper_subwalk
        );
        return None;
    }
    let mut frames = majit_metainterp::MIFrameStack::empty();

    for (index, inline) in session.framestack.iter().enumerate() {
        let Some(parent) = inline.parent.as_ref() else {
            s2dbg!("origin={origin} frame {index}: no parent");
            return None;
        };
        let Some(concrete) = parent.blackhole.as_ref() else {
            s2dbg!("origin={origin} frame {index}: parent.blackhole None (capture missing)");
            return None;
        };
        let Some(pjc) = crate::state::pyjitcode_for_jitcode_index(parent.jitcode_index as i32)
        else {
            s2dbg!(
                "origin={origin} frame {index}: no pyjitcode for parent jitcode_index={}",
                parent.jitcode_index
            );
            return None;
        };
        let mut miframe = majit_metainterp::MIFrame::new(pjc.jitcode.clone(), concrete.resume_pc);
        for &(color, value) in &concrete.int_values {
            let bank_len = miframe.int_values.len();
            let Some(slot) = miframe.int_values.get_mut(color) else {
                s2dbg!(
                    "origin={origin} frame {index}: int color {color} out of range (len {bank_len})"
                );
                return None;
            };
            *slot = Some(value);
        }
        for &(color, value) in &concrete.ref_values {
            let bank_len = miframe.ref_values.len();
            let Some(slot) = miframe.ref_values.get_mut(color) else {
                s2dbg!(
                    "origin={origin} frame {index}: ref color {color} out of range (len {bank_len})"
                );
                return None;
            };
            *slot = Some(value as i64);
        }
        for &(color, opref) in &concrete.float_values {
            let Some(majit_ir::Value::Float(value)) = ctx.trace_ctx.concrete_of_opref(opref) else {
                s2dbg!("origin={origin} frame {index}: float opref {opref:?} not a stamped Float");
                return None;
            };
            let bank_len = miframe.float_values.len();
            let Some(slot) = miframe.float_values.get_mut(color) else {
                s2dbg!(
                    "origin={origin} frame {index}: float color {color} out of range (len {bank_len})"
                );
                return None;
            };
            *slot = Some(value.to_bits() as i64);
        }
        frames.push(miframe);
    }

    // The innermost frame is the callee `ctx` is actively sub-walking. In a
    // sub-walk its identity lives in `inline_callee_consts` — the copied
    // `snapshot_sym` still points at the outer portal, so resolving off it
    // pins the OUTER jitcode while `resume_pc` and the concrete banks are the
    // callee's, landing the resume mid-op in the wrong code. Resolve the
    // callee's own jitcode so all three share its coordinate space; fall back
    // to `snapshot_sym` for a top-level (non-sub-walk) abort.
    let innermost_jitcode = if let Some(consts) = ctx.inline_callee_consts {
        let Some(jc) = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index) else {
            s2dbg!(
                "origin={origin} innermost: no pyjitcode for callee jitcode_index={}",
                consts.jitcode_index
            );
            return None;
        };
        jc.jitcode.clone()
    } else {
        unsafe {
            let sym = &*ctx.fbw_mode.snapshot_sym;
            if sym.jitcode().is_null() {
                s2dbg!("origin={origin} innermost snapshot_sym jitcode null");
                return None;
            }
            (&(*sym.jitcode()).payload).jitcode.clone()
        }
    };
    let innermost = match innermost_build {
        InnermostMiframeBuild::LiveMarker(lastop_result) => {
            build_single_frame_miframe(ctx, innermost_jitcode, resume_pc, lastop_result)
        }
        InnermostMiframeBuild::TraceTooLong => {
            build_trace_too_long_single_frame_miframe(ctx, innermost_jitcode, resume_pc)
        }
    };
    let Some(innermost) = innermost else {
        s2dbg!("origin={origin} innermost build_single_frame_miframe declined");
        return None;
    };
    frames.push(innermost);
    s2built!(
        "origin={origin} BUILT multi-frame depth={}",
        frames.frames.len()
    );
    Some(frames)
}

/// TLS cell pointer for the store-journal root area: the undo stays armed
/// from force time until the abort epilogue consumes it, and its slots can be
/// the ONLY reference to pre-walk locals the flush displaced — the area
/// walker forwards them across any minor collection on that whole window
/// (the resume-ref-roots stack cannot cover it: the residual's post-call
/// `pop_resume_ref_roots_to` truncates past a force-time push).
pub(crate) fn escape_flush_undo_cell_ptr() -> *const std::cell::RefCell<Option<EscapeFlushUndo>> {
    ESCAPE_FLUSH_UNDO.with(|cell| cell as *const _)
}

thread_local! {
    /// The concrete `PyFrame` the innermost inline sub-walk is executing, or
    /// `None` outside one.  Set by [`InlineConcreteFrameGuard`], and held as
    /// a translated-livevar root rather than a raw pointer, the way
    /// [`ResidualFrameChainGuard`] holds the `topframeref` it displaces.
    static INLINE_CONCRETE_FRAME: std::cell::RefCell<
        Option<majit_gc::shadow_stack::OwnerRootGuard>,
    > = const { std::cell::RefCell::new(None) };

    /// The concrete frame currently published on the interpreter chain by a
    /// live [`ResidualFrameChainGuard`], or null.  Distinct from
    /// [`INLINE_CONCRETE_FRAME`], which stays set across the whole sub-walk.
    static PUBLISHED_INLINE_FRAME: std::cell::Cell<*mut pyre_interpreter::PyFrame> =
        const { std::cell::Cell::new(std::ptr::null_mut()) };

    /// The frame a live [`LiveLastInstrGuard`] published the executing pc onto,
    /// with the resume coordinate it displaced.
    static PUBLISHED_LAST_INSTR: std::cell::Cell<Option<(usize, isize)>> =
        const { std::cell::Cell::new(None) };
}

/// `_opimpl_setfield_vable`'s heap half for the recording walk.  While a
/// residual runs concretely the live frame must name the EXECUTING opcode, so
/// that a frame reader inside the callee reports the line the opcode is on —
/// what `eval.rs` `frame.last_instr = pc` gives an interpreted opcode.  The
/// walk dispatches opcodes itself instead of through `execute_opcode_step`, so
/// nothing else advances the field and the frame still carries the pc of the
/// last resume point; `_warnings::setup_context` then keys its registry on
/// that line and re-issues a warning the interpreted run already deduplicated.
///
/// Restored on the way out, because `last_instr` doubles as the resume
/// coordinate: `flush_walk_end_state_to_frame_inner` writes `resume_py_pc - 1`
/// and the escape guard's resume pc is this same pc, so the two meanings
/// differ by exactly one and may only coexist for the residual's duration.
/// Leaving the executing value behind makes an abort replay one opcode late.
struct LiveLastInstrGuard {
    frame: *mut pyre_interpreter::PyFrame,
    saved: isize,
    prev: Option<(usize, isize)>,
}

impl LiveLastInstrGuard {
    /// Publishes onto the frame `py_pc` indexes.  Inside an inline sub-walk
    /// that is the callee's concrete frame, NOT the walk's virtualizable: a
    /// sub-walk's pc is in the callee's code (the same reason `escape_stack`
    /// declines to latch a sub-walk's mirror), and writing it onto the outer
    /// frame strands that frame's replay on a stack depth it never had.
    fn enter(live_frame: usize, py_pc: u32, inline_py_pc: Option<u32>) -> Option<Self> {
        let inline = current_inline_concrete_frame() as *mut pyre_interpreter::PyFrame;
        // The frame and the pc have to name the same code.  Retargeting the
        // frame to the callee while keeping the outer walk's `vstack_cur_pypc`
        // publishes a pc from the CALLER's code onto the callee's frame — and
        // for a sub-walk that mirror is never advanced (`vstack_valid` is
        // false), so the published value is a constant 0 and a frame reader
        // inside the callee reports the function's first line.
        let (frame, py_pc) = match (inline.is_null(), inline_py_pc) {
            (false, Some(callee_pc)) => (inline, callee_pc),
            (false, None) => (inline, py_pc),
            (true, _) => (live_frame as *mut pyre_interpreter::PyFrame, py_pc),
        };
        Self::enter_frame(frame, py_pc)
    }

    fn enter_frame(frame: *mut pyre_interpreter::PyFrame, py_pc: u32) -> Option<Self> {
        if frame.is_null() {
            return None;
        }
        let saved = unsafe { (*frame).last_instr };
        unsafe { (*frame).last_instr = py_pc as isize };
        // Save/restore rather than set/clear: a residual can run user code that
        // records a nested walk whose own residual enters a second guard, and
        // clearing on the inner drop would leave the still-live outer
        // publication invisible — `capture_escape_flush_undo` would then snapshot
        // the executing pc as if it were the outer frame's resume coordinate.
        // Same discipline as [`InlineConcreteFrameGuard`] and
        // [`ResidualFrameChainGuard`].
        let prev = PUBLISHED_LAST_INSTR.with(|slot| slot.replace(Some((frame as usize, saved))));
        Some(Self { frame, saved, prev })
    }
}

impl Drop for LiveLastInstrGuard {
    fn drop(&mut self) {
        PUBLISHED_LAST_INSTR.with(|slot| slot.set(self.prev));
        // A flush that committed onto this frame wrote the resume coordinate
        // itself and is authoritative.  Its undo capture holds the value this
        // guard displaced (see [`capture_escape_flush_undo`]), so a later
        // commit withdrawal still restores the resume pc rather than the
        // executing one.
        let flushed = ESCAPE_FLUSH_UNDO.with(|slot| {
            slot.borrow().as_ref().map(|undo| undo.frame) == Some(self.frame as usize)
        });
        if !flushed {
            unsafe { (*self.frame).last_instr = self.saved };
        }
    }
}

/// Names the frame an inline sub-walk executes concretely for the duration of
/// that sub-walk.  Publishing it on the interpreter chain is left to
/// [`ResidualFrameChainGuard`], which brackets only the residual calls.
pub(crate) struct InlineConcreteFrameGuard(Option<majit_gc::shadow_stack::OwnerRootGuard>);

impl InlineConcreteFrameGuard {
    pub(crate) fn enter(frame: *mut pyre_interpreter::PyFrame) -> Self {
        // Set unconditionally, the empty case included: a nested sub-walk
        // whose seed block bailed has no frame of its own, and inheriting the
        // enclosing callee's frame would publish it for the inner callee's
        // residuals — resolving one level too shallow, the error this guard
        // exists to stop.  An empty slot makes `ResidualFrameChainGuard::enter`
        // publish nothing.
        //
        // The displaced root stays in this guard, so the enclosing level's
        // frame keeps its own root for the whole nested sub-walk.
        let entered = (!frame.is_null())
            .then(|| majit_gc::shadow_stack::OwnerRootGuard::new(majit_ir::GcRef(frame as usize)));
        Self(INLINE_CONCRETE_FRAME.with(|slot| slot.replace(entered)))
    }
}

impl Drop for InlineConcreteFrameGuard {
    fn drop(&mut self) {
        INLINE_CONCRETE_FRAME.with(|slot| *slot.borrow_mut() = self.0.take());
    }
}

/// `executioncontext.py enter` / `leave` around one concretely executed
/// residual call of an inline sub-walk.
///
/// Inlining a call elides the callee's real call sequence, so nothing
/// publishes the callee frame.  PyPy can leave the chain alone — its
/// metainterp never runs the callee for real while tracing — but the walker
/// does, and a residual that reads the chain (`sys._getframe`, a traceback)
/// would otherwise observe the caller as the running frame and resolve one
/// level too shallow.  Scoped to the call itself so the walk's own frame
/// bookkeeping outside it is untouched.
struct ResidualFrameChainGuard {
    ec: *mut pyre_interpreter::PyExecutionContext,
    frame: *mut pyre_interpreter::PyFrame,
    /// Shadow-stack index of the caller `topframeref` this guard displaced,
    /// held there rather than in the struct because the residual runs
    /// arbitrary user code.  Frames themselves never move — `FrameBox::new`
    /// allocates old-gen — but once the tracer stores a `JitVirtualRef` in the
    /// chain the displaced value is a nursery object, and a minor collection
    /// inside the residual would leave `Drop` writing back a pre-move pointer.
    /// Rooting lets the collector forward it in place, as `CurrentFrameGuard`
    /// already does for the same field.
    saved_root: usize,
    previous_published: *mut pyre_interpreter::PyFrame,
    /// Whether this guard performed the chain write, so `Drop` restores only
    /// what it changed.  False when the chain already named `frame`.
    entered: bool,
}

impl ResidualFrameChainGuard {
    fn enter() -> Option<Self> {
        let frame = current_inline_concrete_frame() as *mut pyre_interpreter::PyFrame;
        if frame.is_null() {
            return None;
        }
        let ec = unsafe { (*frame).execution_context } as *mut pyre_interpreter::PyExecutionContext;
        if ec.is_null() {
            return None;
        }
        let saved_topframeref = unsafe { (*ec).topframeref };
        // Re-entering the same frame would make it its own caller.  `topframeref`
        // holds a `jit.virtual_ref`, so a vref that NAMES this frame is the same
        // re-entry as the bare pointer; resolve the referent without forcing,
        // because forcing clears `TOKEN_TRACING_RESCALL` and
        // `tracing_after_residual_call` reads that as a callee escape.
        let entered = !std::ptr::eq(
            pyre_interpreter::executioncontext::vref_referent(saved_topframeref),
            frame,
        );
        if entered {
            // Same barrier obligation as the inline-call push: `f_backref` is
            // a traced `Type::Ref` field, `frame` can be old-generation, and
            // `saved_topframeref` can name a young frame.
            pyre_object::gc_hook::try_gc_write_barrier(frame as *mut u8);
            majit_gc::bh_probe_note_store(
                frame as usize,
                crate::frame_layout::PYFRAME_F_BACKREF_OFFSET,
                2,
            );
            unsafe {
                (*frame).f_backref = saved_topframeref;
                (*ec).topframeref = frame;
            }
        }
        let saved_root = majit_gc::shadow_stack::push(majit_ir::GcRef(saved_topframeref as usize));
        // Published whether or not this guard wrote the chain: `frame` is the
        // one a force inside the residual must redirect its escape onto
        // (`flush_active_frame_escape`), and the chain already naming it makes
        // that more true, not less.
        let previous_published = PUBLISHED_INLINE_FRAME.with(|slot| slot.replace(frame));
        Some(Self {
            ec,
            frame,
            saved_root,
            previous_published,
            entered,
        })
    }
}

impl Drop for ResidualFrameChainGuard {
    fn drop(&mut self) {
        unsafe {
            // `executioncontext.py leave`: move the raw caller vref back without
            // forcing it, then, when the frame escaped, force the caller and
            // mark it escaped too.  A frame handed to application code keeps a
            // reference to its caller, so the caller must stay materialised;
            // dropping that propagation would leave the escape recorded only on
            // a frame the walk owns privately.
            // Read the root back before popping: a collection during the
            // residual forwards it in place.
            let saved_topframeref =
                majit_gc::shadow_stack::get(self.saved_root).0 as *mut pyre_interpreter::PyFrame;
            majit_gc::shadow_stack::pop_to(self.saved_root);
            PUBLISHED_INLINE_FRAME.with(|slot| slot.set(self.previous_published));
            if self.entered {
                (*self.ec).topframeref = saved_topframeref;
            }
            if (*self.frame).escaped() {
                let f_back = (*self.frame).get_f_back();
                if !f_back.is_null() {
                    (*f_back).mark_as_escaped();
                }
            }
        }
    }
}

struct ActiveFrameEscapeGuard {
    prev: Option<(usize, Option<usize>)>,
    prev_stack: Option<Vec<OpRef>>,
}

impl ActiveFrameEscapeGuard {
    fn enter(frame: usize, py_pc: Option<usize>, stack: Option<Vec<OpRef>>) -> Self {
        let current = (frame != 0).then_some((frame, py_pc));
        COMMITTED_FRAME_ESCAPE_PC.with(|slot| slot.set(None));
        let latched = if current.is_some() { stack } else { None };
        let prev_stack = ACTIVE_FRAME_ESCAPE_STACK
            .with(|slot| std::mem::replace(&mut *slot.borrow_mut(), latched));
        Self {
            prev: ACTIVE_FRAME_ESCAPE.with(|slot| slot.replace(current)),
            prev_stack,
        }
    }
}

impl Drop for ActiveFrameEscapeGuard {
    fn drop(&mut self) {
        ACTIVE_FRAME_ESCAPE.with(|slot| slot.set(self.prev));
        ACTIVE_FRAME_ESCAPE_STACK.with(|slot| *slot.borrow_mut() = self.prev_stack.take());
    }
}

/// Returns whether `frame` is the one recorded as escaping this residual call
/// (`expected == frame`) — i.e. the traced virtualizable itself was handed to
/// Python, so its tracing token must be forced.  The concrete frame an inline
/// sub-walk published counts as well: it runs under that virtualizable, so
/// handing it out escapes the virtualizable too. Any other frame a residual
/// callee inspects returns false. Independently, the walk-end resume pc is
/// committed only when the state flush succeeds (a merge point with cached
/// depth); a directly matched frame that cannot flush still escaped and must
/// be forced, so for it the two signals are decoupled.
/// Step-0 attribution probe: consume the classification recorded by the most
/// recent matched escape and tally which disjunct owns this token clear.
pub fn attribute_last_escape_force() {
    if let Some(callee_only) = LAST_ESCAPE_WAS_CALLEE_ONLY.with(|c| c.take()) {
        use crate::trace::fbw_diag;
        fbw_diag::bump(if callee_only {
            fbw_diag::ESCAPE_FORCE_BY_CALLEE_ONLY
        } else {
            fbw_diag::ESCAPE_FORCE_BY_PORTAL
        });
    }
}

pub fn flush_active_frame_escape(ctx: &TraceCtx, frame: *mut pyre_interpreter::PyFrame) -> bool {
    // `executioncontext.py:104-106 leave` — a frame handed to application code
    // keeps a reference to its caller, so escaping the concrete frame an inline
    // sub-walk published escapes the traced virtualizable it runs under.  The
    // flush stays keyed on that virtualizable, whose resume pc this residual
    // already latched, so the walk resumes forward rather than replaying from
    // entry.
    let escaped_published = PUBLISHED_INLINE_FRAME.with(|slot| {
        let published = slot.get();
        let matched = !published.is_null() && std::ptr::eq(published, frame);
        if matched {
            let f_back = unsafe { (*published).get_f_back() };
            if !f_back.is_null() {
                unsafe { (*f_back).mark_as_escaped() };
            }
        }
        matched
    });
    ACTIVE_FRAME_ESCAPE.with(|slot| {
        if let Some((expected, portal_py_pc)) = slot.get()
            && {
                let escaped_portal = expected == frame as usize;
                if escaped_portal || escaped_published {
                    use crate::trace::fbw_diag;
                    match (escaped_portal, escaped_published) {
                        (true, false) => fbw_diag::bump(fbw_diag::ESCAPE_PORTAL_ONLY),
                        (false, true) => fbw_diag::bump(fbw_diag::ESCAPE_PUBLISHED_CALLEE_ONLY),
                        (true, true) => {
                            fbw_diag::bump(fbw_diag::ESCAPE_PORTAL_AND_PUBLISHED_CALLEE)
                        }
                        (false, false) => {}
                    }
                    LAST_ESCAPE_WAS_CALLEE_ONLY
                        .with(|c| c.set(Some(!escaped_portal && escaped_published)));
                    true
                } else {
                    false
                }
            }
        {
            // Force #2+ within this residual (`enter` resets the committed pc
            // per residual): the live frame is already heap-authoritative
            // from the first flush, and the callee may have legitimately
            // mutated it since (an `f_locals` write-through) — re-flushing
            // would overwrite that mutation with the same walk-end values.
            // The token force is a no-op then too (`force_now` on
            // TOKEN_NONE, virtualizable.py:248-260).
            if COMMITTED_FRAME_ESCAPE_PC
                .with(|committed| committed.get())
                .is_some()
            {
                return true;
            }
            capture_escape_flush_undo(expected);
            let (latched, flushed) = if let Some(py_pc) = portal_py_pc {
                let latched = flush_escape_state_with_latched_stack(ctx, expected, py_pc);
                let flushed =
                    latched || crate::state::flush_walk_end_state_to_frame(ctx, expected, py_pc);
                (latched, flushed)
            } else {
                (false, false)
            };
            // Reachability probe: the latched-stack path is gated on
            // `escape_opcode_window_clean`, the plain fallback is not, and this
            // resume pc re-runs the whole escaping opcode.
            if let Some(py_pc) = portal_py_pc
                && !latched
                && flushed
            {
                use crate::trace::fbw_diag;
                fbw_diag::bump(fbw_diag::ESCAPE_PLAIN_FALLBACK);
                if !escape_opcode_window_clean(py_pc) {
                    fbw_diag::bump(fbw_diag::ESCAPE_PLAIN_FALLBACK_UNCLEAN);
                }
            }
            if flushed {
                let kind = if latched {
                    EscapeResumeKind::Exact
                } else {
                    EscapeResumeKind::RerunsOpcode
                };
                if let Some(py_pc) = portal_py_pc {
                    COMMITTED_FRAME_ESCAPE_PC.with(|committed| committed.set(Some((py_pc, kind))));
                }
            } else if !crate::state::flush_locals_region_to_frame(ctx, expected) {
                // All-or-nothing decline: nothing was written, nothing to undo.
                discard_escape_flush_undo();
            }
            // A declined full flush still escaped the virtualizable, so the
            // locals region is written anyway (`virtualizable.py:101-138
            // write_boxes` has no decline) — otherwise the callee reads an
            // array of nulls.  That write claims no resume pc, and the undo
            // stays armed so the legacy replay re-enters the pre-flush frame.
            //
            // Upstream reports the escape from the vable token state alone,
            // independent of any resume-image write.  See
            // `virtualizable.py:231-255` (`tracing_after_residual_call` /
            // `force_now`), `virtualizable.py:311-330` (token states),
            // `virtualref.py:157-167` (vref'd inlined callee), and
            // `pyjitpl.py:3373-3390` (unconditional ABORT_ESCAPE).  Runtime
            // forcing also resets the token before writing fields
            // (`resume.py:1405-1408`).  Therefore a matched guard reports the
            // escape even when the flush declined.
            return true;
        }
        false
    })
}

/// Snapshot the frame region a successful escape flush overwrites (the whole
/// `locals_cells_stack_w` array plus `last_instr`/`valuestackdepth`).  GC:
/// the store-journal root area walks the armed capture's slots in place (see
/// [`escape_flush_undo_cell_ptr`]).  A second force in the same residual
/// keeps the OLDEST capture (the true pre-flush state); staleness across walk
/// attempts is impossible — the walk-start reset discards any leftover.
fn capture_escape_flush_undo(frame: usize) {
    ESCAPE_FLUSH_UNDO.with(|slot| {
        let mut slot = slot.borrow_mut();
        match slot.as_ref() {
            // Keep the oldest capture for the SAME frame (true pre-flush state).
            Some(existing) if existing.frame == frame => return,
            // A capture for a different frame is a stale leftover from an
            // unwound path; its frame may be dead — drop without restoring.
            Some(_) => *slot = None,
            None => {}
        }
        let pf = unsafe { &*(frame as *const pyre_interpreter::PyFrame) };
        // With a [`LiveLastInstrGuard`] live on this frame the field holds the
        // EXECUTING pc, not the resume coordinate.  The undo exists to hand a
        // legacy replay a pristine pre-flush frame, and a replay needs the
        // resume pc — so capture the value the guard displaced.
        let last_instr = PUBLISHED_LAST_INSTR
            .with(|slot| slot.get())
            .filter(|(published, _)| *published == frame)
            .map_or(pf.last_instr, |(_, saved)| saved);
        *slot = Some(EscapeFlushUndo {
            frame,
            last_instr,
            valuestackdepth: pf.valuestackdepth,
            slots: pf.locals_w().as_slice().to_vec(),
        });
    });
}

/// Put the pre-flush frame state back so the legacy replay re-enters a
/// pristine frame.  Called from the unforced / rootless continuation (where
/// the walk goes on and must not see the moved frame) and, for a withdrawn
/// commit, from the walk-end epilogue once no resume-PAST continuation has
/// claimed the flushed frame.
pub(crate) fn restore_escape_flush_undo() {
    ESCAPE_FLUSH_UNDO.with(|slot| {
        let Some(undo) = slot.borrow_mut().take() else {
            return;
        };
        unsafe {
            let pf = &mut *(undo.frame as *mut pyre_interpreter::PyFrame);
            let dst = pf.locals_w_mut();
            let n = undo.slots.len().min(dst.as_slice().len());
            for (i, &v) in undo.slots.iter().take(n).enumerate() {
                dst[i] = v;
            }
            pf.last_instr = undo.last_instr;
            pf.valuestackdepth = undo.valuestackdepth;
        }
    });
}

pub(crate) fn discard_escape_flush_undo() {
    ESCAPE_FLUSH_UNDO.with(|slot| {
        *slot.borrow_mut() = None;
    });
    ESCAPE_FLUSH_UNDO_PENDING.with(|slot| slot.set(false));
}

/// Note that the forced residual's commit was withdrawn, so the pre-flush
/// frame has to come back IF the walk ends up on the legacy replay.  The
/// restore itself waits for [`take_escape_flush_undo_pending`] at walk end:
/// the resume-PAST continuation keeps the flushed frame (upstream's
/// `virtualizable.py:101-138 write_boxes` has no undo once the vable is
/// forced), and only the replay-from-entry leg needs the pre-walk state back.
fn mark_escape_flush_undo_pending() {
    ESCAPE_FLUSH_UNDO_PENDING.with(|slot| slot.set(true));
}

/// Consume the deferred-restore request armed by the force arm.
pub(crate) fn take_escape_flush_undo_pending() -> bool {
    ESCAPE_FLUSH_UNDO_PENDING.with(|slot| slot.replace(false))
}

/// Opcode-scoped effect window check (see [`ESCAPE_OPCODE_WINDOW`]): true iff
/// every earlier residual of the CURRENT Python opcode is declared re-runnable.
/// A pure query — the window is written only by [`escape_opcode_window_note`],
/// so the residual asking the question never disqualifies itself.
///
/// Keyed on `py_pc` alone: an opcode revisited across walked inner-loop
/// iterations still sees the FIRST visit's verdict, so every revisit after any
/// non-re-runnable residual declines the latch — conservative (decline →
/// legacy).  An unexplained latch-decline spike on nested-loop shapes is this;
/// the refinement would reset the window on back-edge re-entry.
fn escape_opcode_window_clean(py_pc: usize) -> bool {
    ESCAPE_OPCODE_WINDOW.with(|slot| match slot.get() {
        Some((pc, clean)) if pc == py_pc => clean,
        _ => true,
    })
}

/// Declare this residual's re-runnability into the opcode window (see
/// [`ESCAPE_OPCODE_WINDOW`]).  Called for every residual that reaches
/// execution, AFTER the latch gate has read the window.
///
/// `reentrant` is the declared effect class, `EF_ELIDABLE_*`
/// (`check_is_elidable`) or `EF_LOOPINVARIANT`.  It is deliberately STRICTER
/// than `provably_side_effect_free`, which additionally exempts
/// [`majit_ir::PyreHelperKind::ForIterNext`]: that exemption answers a
/// different question (the consume is the SOURCE of an in-flight item, not a
/// body effect for it), and a user-defined `__next__` runs user bytecode that
/// re-executing the opcode would re-run.
fn escape_opcode_window_note(py_pc: usize, reentrant: bool) {
    ESCAPE_OPCODE_WINDOW.with(|slot| {
        let clean = match slot.get() {
            Some((pc, clean)) if pc == py_pc => clean,
            _ => true,
        };
        slot.set(Some((py_pc, clean && reentrant)));
    });
}

/// Reset the opcode window at walk start so a prior trace's sample cannot
/// alias a same-pc opcode of the new walk.
pub(crate) fn escape_opcode_window_reset() {
    ESCAPE_OPCODE_WINDOW.with(|slot| slot.set(None));
}

/// Resolve the latched operand-stack mirror (`ActiveFrameEscapeGuard::enter`)
/// to concrete refs and flush the walk-end state with the mid-expression
/// stack the vable shadow cannot provide (its stack region is only valid at
/// merge points).  The `_run_forever` continue-forward analog
/// (`blackhole.py:1752`) for the escape abort: the resume state is the exact
/// abort-point frame, so the walk's applied effects stand and nothing
/// replays.  Returns false (no frame mutation) when no stack was latched or
/// any slot lacks a concrete non-null Ref — the caller then falls back to the
/// plain merge-point flush.
fn flush_escape_state_with_latched_stack(ctx: &TraceCtx, frame: usize, py_pc: usize) -> bool {
    ACTIVE_FRAME_ESCAPE_STACK.with(|slot| {
        let latched = slot.borrow();
        let Some(oprefs) = latched.as_ref() else {
            return false;
        };
        flush_with_latched_stack(ctx, frame, py_pc, oprefs)
    })
}

/// Resolve a latched operand-stack OpRef mirror to concrete refs and flush the
/// walk-end state with it, keeping the resolved refs rooted across the call.
///
/// Shared by the escape flush above and the `ABORT_FORCE_QUASIIMMUT` leg
/// ([`flush_qmut_abort_state`]): both resume the interpreter
/// mid-expression, where the vable shadow's stack region reads NULL and only the
/// walk's own mirror can say what the operands are.
fn flush_with_latched_stack(ctx: &TraceCtx, frame: usize, py_pc: usize, oprefs: &[OpRef]) -> bool {
    {
        let mut stack = Vec::with_capacity(oprefs.len());
        for &opref in oprefs.iter() {
            match ctx.concrete_of_opref(opref) {
                Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE => {
                    // A recorded concrete NULL is a real operand, not an absent
                    // one — `NO_CONCRETE` is what "unavailable" looks like.  It
                    // is the `self_or_null` sentinel `PUSH_NULL` puts under a
                    // callable, so refusing it made every `CALL` /
                    // `CALL_FUNCTION_EX` escape fall through to the plain flush,
                    // which then declines on that same NULL slot and leaves the
                    // portal replaying the frame from its entry.
                    stack.push(r.as_usize() as pyre_object::PyObjectRef);
                }
                other => {
                    if fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-latched-flush] DECLINE at py_pc={py_pc}: slot {} of {} \
                             opref={opref:?} concrete={other:?}",
                            stack.len(),
                            oprefs.len(),
                        );
                    }
                    return false;
                }
            }
        }
        // Why this latch exists, checkable at runtime.  Measured over
        // pyre/bench/synth, the only slot that ever disagrees with the vable
        // shadow is the in-progress opcode's TOS, and it holds a compile-time
        // NULL *constant* rather than a stale or absent value: exactly what
        // `popvalue_maybe_none` writes (`pyframe.py:411-417` →
        // `setarrayitem_vable_r(locals_cells_stack_w, depth, ConstPtr.NULL)`
        // via `jtransform.py:1898`).  The opcode had already popped the slot
        // before its residual forced.
        //
        // So the array is correct and upstream agrees — RPython's `popvalue`
        // NULLs the slot the same way.  What this latch holds is the operand
        // the in-flight opcode already consumed, which is what upstream keeps
        // in `MIFrame.registers_r` and what `convert_and_run_from_pyjitpl`
        // resumes a blackhole from.  Emitting more vable stores does not
        // remove the need for it: `LOAD_ATTR` already emits the push mirror
        // via `emit_pushvalue_ref!` and its slot still reads NULL, because the
        // pop follows the push.
        if fbw_debug_abort_enabled() {
            let base = ctx
                .virtualizable_info()
                .map_or(usize::MAX, |info| info.num_static_extra_boxes);
            let nlocals = crate::state::concrete_nlocals(frame).unwrap_or(usize::MAX);
            for (rel, &obj) in stack.iter().enumerate() {
                let entry =
                    ctx.virtualizable_entry_at(base.saturating_add(nlocals).saturating_add(rel));
                let shadow = entry.map(|(_opref, value)| value);
                let agrees =
                    matches!(shadow, Some(majit_ir::Value::Ref(r)) if r.as_usize() == obj as usize);
                if !agrees {
                    // Report the OpRef too: a live box with a NULL value means
                    // the symbolic write landed and only the concrete mirror is
                    // absent, which is a different defect from no write at all.
                    eprintln!(
                        "[r6-latch] slot {rel}/{} latched=0x{:x} shadow={shadow:?} box={:?} (DISAGREES)",
                        stack.len(),
                        obj as usize,
                        entry.map(|(opref, _)| opref),
                    );
                }
            }
        }
        // The flush's Int/Float local boxing can trigger a minor collection;
        // register the resolved refs as resume roots so they are forwarded in
        // place across it (the same discipline as the vable root above).
        let root_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
        unsafe {
            majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_raw_parts_mut(
                stack.as_mut_ptr() as *mut i64,
                stack.len(),
            ));
        }
        let committed =
            crate::state::flush_walk_end_state_to_frame_with_full_stack(ctx, frame, py_pc, &stack);
        majit_gc::shadow_stack::pop_resume_ref_roots_to(root_depth);
        committed
    }
}

/// The `ABORT_FORCE_QUASIIMMUT` leg's flush: resume the interpreter at the
/// opcode carrying the forcing write, with the operand stack the walk mirrored
/// on entry to it ([`fbw_qmut_abort_stack_take`]).
pub(crate) fn flush_qmut_abort_state(
    ctx: &TraceCtx,
    frame: usize,
    py_pc: usize,
    oprefs: &[OpRef],
) -> bool {
    flush_with_latched_stack(ctx, frame, py_pc, oprefs)
}

/// Take the committed escape resume pc and which flush produced it (the
/// walk-end commit contract turns on the difference — see [`EscapeResumeKind`]).
pub(crate) fn take_committed_frame_escape_pc() -> Option<(usize, EscapeResumeKind)> {
    COMMITTED_FRAME_ESCAPE_PC.with(|slot| slot.take())
}

fn committed_frame_escape_pc() -> Option<(usize, EscapeResumeKind)> {
    COMMITTED_FRAME_ESCAPE_PC.with(|slot| slot.get())
}

/// Withdraw a committed escape resume pc: the residual that escaped also ran
/// user bytecode, so resuming AT its opcode would re-execute that user body —
/// the legacy replay (whose journals roll back) is the never-double fallback.
fn cancel_committed_frame_escape_pc() {
    COMMITTED_FRAME_ESCAPE_PC.with(|slot| slot.set(None));
}

/// Reject a residual_call whose `allboxes` (funcbox + permuted args)
/// contains an `OpRef::NONE`.  RPython's `do_residual_call` resolves
/// each argbox through `env[box]`, where an unbound box is a `KeyError`;
/// recording the op anyway lets `OpRef::NONE` reach the backend's
/// `resolve_opref`, which aborts the process.  Returning
/// `ResidualCallArgUnbound` instead lets the outer walker fall back to a
/// trace abort — the same graceful outcome a pre-seam inline arm reached
/// when its payload read surfaced `GotoIfNotValueNotConcrete`.
pub(crate) fn ensure_residual_call_args_bound(
    allboxes: &[OpRef],
    pc: usize,
) -> Result<(), DispatchError> {
    if let Some(arg_index) = allboxes.iter().position(|b| b.is_none()) {
        return Err(DispatchError::ResidualCallArgUnbound { pc, arg_index });
    }
    Ok(())
}

/// EffectInfo-driven opcode selector shared by `dispatch_residual_call_*`
/// dispatchers. Mirrors `pyjitpl.py do_residual_call`'s
/// precedence:
///   1. **forces branch** (`pyjitpl.py`): outer check on
///      `assembler_call or check_forces_virtual_or_virtualizable()`
///      records `CALL_MAY_FORCE_*` at step 2 and unconditionally fires
///      `GUARD_NOT_FORCED` (`:2079`).  The release-gil sub-case
///      (`pyjitpl.py if effectinfo.is_call_release_gil()`) is
///      handled by [`direct_call_release_gil`] **before** this
///      selector is called — the dispatcher early-returns on
///      `ei.is_call_release_gil()` so this function only ever sees
///      EI values where the sub-case is not active.
///   2. `EF_LOOPINVARIANT` (`:2087-2110`): `CALL_LOOPINVARIANT_*`.
///   3. `check_is_elidable()` (`:2112-2126`): `CALL_PURE_*`.
///   4. default (`:2126`): plain `CALL_*`.
///
/// Returns the `Call*` opcode for the call itself, whether
/// `handle_possible_exception` should emit `GUARD_NO_EXCEPTION`
/// (`check_can_raise(False)`), and whether the unconditional
/// `GUARD_NOT_FORCED` from the forces branch (`pyjitpl.py`)
/// should fire.
///
/// Non-elidable concrete-execute inventory.
///
/// PyPy `do_residual_call` (pyjitpl.py) concrete-executes
/// the helper at trace-record time across the **forces** /
/// **loopinvariant** (cache miss) / **elidable** / **default**
/// branches via `executor.execute_varargs(opnum, argboxes, descr,
/// exc=can_raise, pure=is_elidable)`.  Two narrower branches sit
/// outside this uniform call:
///   * `OS_NOT_IN_TRACE` short-circuits through
///     `do_not_in_trace_call` (`pyjitpl.py`) and never
///     reaches `executor.execute_varargs`.
///   * `is_call_release_gil()` runs the helper through
///     `do_call_release_gil` (`pyjitpl.py`), invoking
///     `executor.execute_varargs` directly **before** the recorded
///     `CALL_RELEASE_GIL_*` op is emitted.
/// The recorded opcode kind selected above is for the IR trace
/// only; concrete execution either fired (or was intentionally
/// skipped via the two narrow branches above) before the trace op
/// hits the recorder.
///
/// Per-branch concrete-execute status (every non-pure residual call
/// concrete-executes during the walk, matching
/// PyPy `do_residual_call` which runs `executor.execute_varargs` for the
/// whole forces branch regardless of EI):
///
/// | EI branch | Selected op | Pyre walker concrete-execute |
/// |---|---|---|
/// | `is_call_release_gil()` | (early-routed to [`direct_call_release_gil`], records `CALL_RELEASE_GIL_*`) | executed as `CallMayForce*` on the **original** `allboxes` via [`try_execute_residual_call_via_executor`] |
/// | `check_forces_virtual_or_virtualizable()` | `CallMayForce*` + `GuardNotForced` | executed via [`try_execute_residual_call_via_executor`] (active vable bracketed by the token protocol) |
/// | `extraeffect == LoopInvariant` | `CallLoopinvariant*` | executed on cache miss; [`loopinvariant_lookup`] reuses the cached OpRef on hit (no execute, no record) |
/// | `check_is_elidable()` | `CallPure*` | executed + cached via [`try_fold_pure_call_via_executor`] (elidable_cannot_raise only — see its caveats) |
/// | default | `Call*` + (`GuardNoException` iff can_raise) | executed via [`try_execute_residual_call_via_executor`] |
///
/// All three dispatch entry points — [`dispatch_residual_call_iRd_kind`]
/// (`_opimpl_residual_call1`), [`dispatch_residual_call_iIRd_kind`]
/// (`_opimpl_residual_call2`), [`dispatch_residual_call_iIRFd_kind`]
/// (`_opimpl_residual_call3`) — call [`select_residual_call_opcode`],
/// `record_op_with_descr`, then [`try_fold_pure_call_via_executor`] (pure)
/// + [`try_execute_residual_call_via_executor`] (non-pure).  The executor
/// self-gates and degrades to recording-only when a precondition fails
/// (non-authoritative walk, non-const funcbox, unpatched symbolic fnaddr).
///
pub(crate) fn select_residual_call_opcode(
    ei: &majit_ir::EffectInfo,
    dst_bank: char,
    caller: &'static str,
) -> (OpCode, bool, bool) {
    // Release-gil sub-case is handled by `direct_call_release_gil`
    // before this selector runs.  Any `is_call_release_gil()` EI
    // reaching here is a dispatcher bug.
    debug_assert!(
        !ei.is_call_release_gil(),
        "{caller}: select_residual_call_opcode received an is_call_release_gil() EI; \
         dispatcher should have routed via direct_call_release_gil first"
    );
    let (call_op, pure_op, may_force_op, loopinvariant_op): (OpCode, OpCode, OpCode, OpCode) =
        match dst_bank {
            'r' => (
                OpCode::CallR,
                OpCode::CallPureR,
                OpCode::CallMayForceR,
                OpCode::CallLoopinvariantR,
            ),
            'i' => (
                OpCode::CallI,
                OpCode::CallPureI,
                OpCode::CallMayForceI,
                OpCode::CallLoopinvariantI,
            ),
            // `_irf_f/iIRFd>f` (`pyjitpl.py opimpl_residual_call_irf_f =
            // _opimpl_residual_call3`, `blackhole.py bhimpl_residual_call_irf_f`).
            // `resoperation.py Type::Float => CallF`. The `_r_f` /
            // `_ir_f` shapes do not exist upstream — the only float-result
            // residual_call variant routes through the `iIRFd` arglist.
            'f' => (
                OpCode::CallF,
                OpCode::CallPureF,
                OpCode::CallMayForceF,
                OpCode::CallLoopinvariantF,
            ),
            // `_*_v/iRd|iIRd|iIRFd` void variants (`pyjitpl.py
            // opimpl_residual_call_r_v = _opimpl_residual_call1`,
            // `:1351 opimpl_residual_call_ir_v = _opimpl_residual_call2`,
            // `:1355 opimpl_residual_call_irf_v = _opimpl_residual_call3`,
            // `blackhole.py bhimpl_residual_call_*_v`).
            // `resoperation.py Type::Void => CallN`. No dst writeback;
            // `write_residual_call_result_to_dst` no-ops on 'v'.
            'v' => (
                OpCode::CallN,
                OpCode::CallPureN,
                OpCode::CallMayForceN,
                OpCode::CallLoopinvariantN,
            ),
            _ => panic!("{caller}: unsupported dst_bank '{dst_bank}'"),
        };
    if ei.check_forces_virtual_or_virtualizable() {
        // pyjitpl.py forces-virtual-or-virtualizable branch
        // proper: CALL_MAY_FORCE_* + GUARD_NOT_FORCED.
        // `handle_possible_exception` also fires (forces always
        // satisfies check_can_raise).
        (may_force_op, ei.check_can_raise(false), true)
    } else if ei.extraeffect == majit_ir::ExtraEffect::LoopInvariant {
        // pyjitpl.py EF_LOOPINVARIANT branch: CALL_LOOPINVARIANT_*
        // via miframe_execute_varargs(..., exc=False). LoopInvariant
        // never raises (extraeffect=1 < CannotRaise=2 → check_can_raise=False).
        //
        // The `pyjitpl.py call_loopinvariant_known_result` lookup
        // and `pyjitpl.py call_loopinvariant_now_known` cache
        // update are wired at the dispatcher level via
        // [`loopinvariant_lookup`] and [`loopinvariant_now_known`]
        // around the `record_op_with_descr` call — they require the
        // dispatcher's `descr_index` and `arg0_int` so this opcode
        // selector cannot perform them on its own.
        (loopinvariant_op, ei.check_can_raise(false), false)
    } else if ei.check_is_elidable() {
        // pyjitpl.py elidable branch: CALL_PURE_*.
        (pure_op, ei.check_can_raise(false), false)
    } else {
        // pyjitpl.py default branch: CALL_*.
        (call_op, ei.check_can_raise(false), false)
    }
}

/// `pyjitpl.py _record_helper_pure` parity for the
/// walker layer: when a residual_call routes to `CallPure*` (elidable +
/// cannot-raise EI per [`select_residual_call_opcode`]) AND every
/// argument in `allboxes` has a known concrete value
/// (`TraceCtx::box_value` returns `Some`), execute the helper at trace
/// time via [`majit_metainterp::executor::execute_pure_call`] and stamp
/// `recorded` with the result.
///
/// PyPy `_opimpl_*` methods (e.g. `_opimpl_setitem`,
/// `_opimpl_setfield_*`) concrete-execute every `do_residual_call`
/// regardless of `check_is_elidable()` — the EI flag only selects the
/// recorded opcode kind (`CALL_PURE_*` vs `CALL_*`), not whether the
/// helper runs.  This function covers the elidable arm; the
/// non-elidable arms (`CallMayForce*`, `CallLoopinvariant*`, `Call*`)
/// are concrete-executed by [`try_execute_residual_call_via_executor`]
/// with raised exceptions surfaced through `BH_LAST_EXC_VALUE` so
/// `eval_loop_jit` can route them into the bytecode exception handler.
///
/// RPython upstream `_record_helper_pure` invokes
/// `executor.execute_varargs(opnum, argboxes, descr, exc=False, pure=True)`
/// which dispatches to `cpu.bh_call_*` and stores the result on
/// `result_box.value` (`pyjitpl.py`).  Pyre's walker observes the
/// same effect through the `set_opref_concrete` stamp — downstream walker
/// chain (sub-jitcode bodies that consume the call result via
/// `concrete_of_opref`) folds end-to-end instead of stalling at
/// `RefOp/IntOp(N)` unknown values.
///
/// **Caller contract**:
/// * `call_opcode` must be one of `CallPureI`/`CallPureR`/`CallPureF`/
///   `CallPureN` — the `select_residual_call_opcode` elidable arm
///   (`pyjitpl.py` proper, `dispatch.rs`).  Other call
///   shapes (`CallMayForce*`, `CallLoopinvariant*`, `Call*`) carry
///   `can_raise=true` or escape semantics that require the full
///   `execute_varargs` MetaInterp seam — they MUST NOT route here.
/// * `allboxes[0]` is the funcbox (per `build_allboxes` layout); the
///   remaining slots are user args in `descr.arg_types()` ABI order.
///
/// Best-effort: returns silently when any operand lacks a concrete
/// `box_value` (the walker has no way to read the runtime value), or
/// when the arity exceeds `MAX_HOST_CALL_ARITY` (16) — the trace still
/// has the recorded `CallPure*` op for the optimizer to consume later,
/// just without the per-record fold.
pub(crate) fn try_fold_pure_call_via_executor<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_opcode: OpCode,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    recorded: OpRef,
) {
    if !matches!(
        call_opcode,
        OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN
    ) {
        return;
    }
    // pyjitpl.py — `_record_helper_pure` only fires for
    // `EF_ELIDABLE_CANNOT_RAISE`. `select_residual_call_opcode` returns
    // `CallPure*` whenever `check_is_elidable()` is true (including
    // `EF_ELIDABLE_CAN_RAISE`), so re-check the can-raise predicate here
    // before dispatching through the `execute_pure_call` no-metainterp
    // carve-out.  A `EF_ELIDABLE_CAN_RAISE` callee would silently swallow
    // the exception via `BH_LAST_EXC_VALUE` with no metainterp to
    // transcribe it.
    let ei = call_descr.get_extra_info();
    if ei.check_can_raise(false) {
        return;
    }
    if allboxes.is_empty() {
        return;
    }
    // pyjitpl.py `_build_allboxes`: slot 0 is funcbox, slots
    // 1.. are user args in `descr.arg_types()` ABI order.  Walker's
    // [`build_allboxes`] preserves the same layout.
    //
    // pyjitpl.py invariant: `_record_helper_pure` requires
    // `funcbox` to be a Const so its `getint()` is the actual fn
    // pointer.  Non-constant funcboxes carry a stale-stamped Int
    // (from `cast_ptr_to_int` of a Ref-bank receiver, etc.) and
    // dereferencing as a code address yields SIGSEGV.  Skip the fold
    // when the funcbox is non-constant; the recorded `CallPure*` op
    // stays in the trace for the optimizer to consume later.
    if !allboxes[0].is_constant() {
        return;
    }
    let funcptr_val = ctx.trace_ctx.box_value(allboxes[0]);
    let func_ptr = match funcptr_val {
        Some(majit_ir::Value::Int(addr)) => addr,
        _ => return,
    };
    // Cap at MAX_HOST_CALL_ARITY (`call_int_function` / `call_void_function`
    // panic on excess arity).  `allboxes.len() - 1` is the arg count
    // (funcbox doesn't pass through).
    if allboxes.len() - 1 > majit_translate::codewriter::insns::MAX_HOST_CALL_ARITY {
        return;
    }
    let mut args = Vec::with_capacity(allboxes.len() - 1);
    for &arg in &allboxes[1..] {
        let v = match ctx.trace_ctx.box_value(arg) {
            Some(majit_ir::Value::Int(n)) => n,
            Some(majit_ir::Value::Ref(r)) => {
                // `usize::MAX` sentinel from `concrete_of_opref` means
                // "no concrete known" — never reach this path because
                // `box_value` returns `None` for un-stamped OpRefs, but
                // belt-and-suspenders against future plumbing.
                if r == majit_ir::GcRef::NO_CONCRETE {
                    return;
                }
                r.as_usize() as i64
            }
            Some(majit_ir::Value::Float(f)) => f.to_bits() as i64,
            Some(majit_ir::Value::Void) => 0,
            None => return,
        };
        args.push(v);
    }
    // Refuse to invoke the helper when any Ref argument is NULL.
    //
    // `pyjitpl.py:3586-3603 record_result_of_call_pure` folds on the weaker
    // test "every argbox is a Const", which admits `ConstPtr(NULL)`.  It can
    // afford to: upstream reaches that line only *after* the call has already
    // been executed for real, and its Const arguments come from boxes the
    // interpreter itself made constant.
    //
    // Pyre's walker folds from a different source of constants.  Its
    // getfield_gc_r handler propagates field reads (including pointer-valued
    // fields like `PyFrame.f_backref`) as concrete values whenever the parent
    // struct is concrete-known, so a top-level frame stamps
    // `Value::Ref(GcRef(0))` onto the recorded `GetfieldGcR` OpRef
    // (`set_opref_concrete`), which `box_value` then hands straight to this
    // fold.  Upstream never derives that NULL in the first place: a pointer
    // becomes nonnull-known there only from the traced program's own null test
    // (`pyjitpl.py:558-575 _establish_nullity`), so the box stays symbolic and
    // the call is never folded on a NULL it invented.  Executing
    // `helper(NULL)` here would dereference NULL and SEGV.
    //
    // So guard the executor entry against NULL Ref arguments and fall through
    // to recording the IR op as-is.  The downstream optimizer then sees the
    // call op and emits the necessary guards.
    for (i, &arg) in args.iter().enumerate() {
        if matches!(call_descr.arg_types().get(i), Some(majit_ir::Type::Ref)) && arg == 0 {
            return;
        }
    }
    let result_i64 = majit_metainterp::executor::execute_pure_call(call_descr, func_ptr, &args);
    // pyjitpl.py `result_box.value = result`: stamp the recorded
    // OpRef with the executed concrete so downstream
    // `concrete_of_opref` / `box_value` consumers see the folded value.
    let result_value = match call_descr.result_type() {
        majit_ir::Type::Int => majit_ir::Value::Int(result_i64),
        majit_ir::Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(result_i64 as usize)),
        majit_ir::Type::Float => majit_ir::Value::Float(f64::from_bits(result_i64 as u64)),
        // void callees discard the result upstream too (`bh_call_v` has
        // no return value); `CallPureN` is included in the matched set
        // only to mirror PyPy's `_record_helper_pure` handling of all
        // pure shapes — skip the stamp for void.
        majit_ir::Type::Void => return,
    };
    // Stamp only when the recorded result has a live slot in the active
    // recorder.  A deeper inlined / recursive frame's residual result may be
    // recorded in a context whose position is not allocated in the active
    // recorder; stamping it would violate the `*FrontendOp(pos, value)`
    // invariant.  Skipping leaves the result symbolic so the downstream branch
    // aborts the trace into the trait fallback instead of crashing.
    ctx.trace_ctx.try_set_opref_concrete(recorded, result_value);
}

/// Abort the walk when a result-bearing may-force CALL is recorded with a
/// concrete-NULL Ref argument — the specialized direct-call shape whose
/// baked `ptr(0x0)` (the `PUSH_NULL` self-slot) makes the runtime call pass
/// NULL where the callee entry expects its globals/closure, yielding a NULL
/// result (closures / locals-bound callees called in a loop). See
/// [`DispatchError::MayForceNullRefArgUnsupported`].
pub(crate) fn walker_abort_if_mayforce_null_ref_arg<Sym: WalkSym>(
    call_opcode: OpCode,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    ctx: &WalkContext<'_, '_, Sym>,
    pc: usize,
) -> Result<(), DispatchError> {
    if !matches!(
        call_opcode,
        OpCode::CallMayForceR | OpCode::CallMayForceI | OpCode::CallMayForceF
    ) {
        return Ok(());
    }
    // `allboxes[0]` is the funcbox; `allboxes[1 + i]` aligns with
    // `arg_types[i]` (see `build_allboxes`).  A Ref arg folded to the
    // NULL constant (`GcRef(0)`) is the broken self-slot; the sentinel
    // `GcRef(usize::MAX)` means "no concrete known" and is left alone.
    //
    // Exemption: `bh_call_fn_N(callable, null_or_self, args...)`'s
    // `null_or_self` (arg index 1) is a checked sentinel — `PY_NULL`
    // means "no receiver" and is never dereferenced (`bh_call_fn_impl`
    // prepends it as arg0 only when non-null), so a concrete-NULL there
    // is the normal plain-call shape, not the broken baked-NULL shape.
    let is_call_fn = call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::CallFn;
    // `RaiseVarargs` (`normalize_raise_varargs`) carries a trailing `cause`
    // Ref that is a checked `PY_NULL` sentinel for `raise X` without `from`
    // (never dereferenced when null); exempt it so the
    // FBW path can own the raise instead of declining to the trait.
    let is_raise_varargs =
        call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::RaiseVarargs;
    // `bh_call_function_ex_fn(callable, self_or_null, starargs, kwargs_or_null)`
    // — `self_or_null` (arg 1) and `kwargs_or_null` (arg 3) are checked
    // `PY_NULL` sentinels (never dereferenced when null), so a concrete-NULL
    // there is the normal `f(*args)` / no-`**` shape, not the broken baked-NULL.
    let is_call_function_ex =
        call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::CallFunctionEx;
    // `bh_call_kw_N(callable, null_or_self, kwnames, args...)` — `null_or_self`
    // (arg 1) is a checked `PY_NULL` sentinel (prepended as arg0 only when
    // non-null), so a concrete-NULL there is the normal plain-call shape.
    let is_call_kw = call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::CallKw;
    // `bh_store_deref_value_fn(cell, value)` — `value` (arg 1) is a checked
    // `PY_NULL` sentinel handed to `w_cell_set` unread; DELETE_DEREF lowers to
    // exactly that shape.  Kept in step with the same exemption in
    // `try_execute_residual_call_via_executor`.
    let is_store_deref =
        call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::StoreDeref;
    // `bh_load_global_fn(namespace_ptr, w_code, frame, namei)` — `namespace_ptr`
    // (arg index 0) is never dereferenced.  The helper discards it and resolves
    // the namespace from the executing frame or the callee's own promoted
    // `w_code`, because an inlined / chained callee's frame register aliases an
    // outer frame; the operand survives only as the cell-fold recogniser's hint.
    // A nested function's `LOAD_GLOBAL` folds it to the NULL constant, so
    // aborting on it stops the walk after an effectful residual has already run
    // concretely and the caller replays that region.
    let is_load_global =
        call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::LoadGlobal;
    for (i, &ty) in call_descr.arg_types().iter().enumerate() {
        if ty != majit_ir::Type::Ref {
            continue;
        }
        if is_call_fn && i == 1 {
            continue;
        }
        if is_load_global && i == 0 {
            continue;
        }
        if is_call_function_ex && (i == 1 || i == 3) {
            continue;
        }
        if is_call_kw && i == 1 {
            continue;
        }
        if is_raise_varargs && i + 1 == call_descr.arg_types().len() {
            continue;
        }
        if is_store_deref && i == 1 {
            continue;
        }
        if let Some(&b) = allboxes.get(1 + i) {
            if matches!(
                ctx.trace_ctx.box_value(b),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(0)))
            ) {
                // Phase-1 diagnostic (gh#343 depth-2): pinpoint which Ref arg
                // folded to concrete NULL and its provenance.  Gated on
                // `PYRE_P2_DIAG` (the depth-2 framestack-walk diag flag) and
                // computed only on the abort path, so the default trace path
                // pays nothing.
                if p2_diag_enabled() {
                    eprintln!(
                        "[p2-mayforce] NULL Ref arg: pc={pc} call_opcode={call_opcode:?} \
                         helper={:?} arg_index={i} nargs={} funcbox={:?}(={:?})",
                        call_descr.get_extra_info().pyre_helper,
                        call_descr.arg_types().len(),
                        allboxes.first(),
                        allboxes.first().and_then(|&f| ctx.trace_ctx.box_value(f)),
                    );
                    for (j, &aty) in call_descr.arg_types().iter().enumerate() {
                        let ab = allboxes.get(1 + j).copied();
                        eprintln!(
                            "[p2-mayforce]   arg[{j}] ty={aty:?} opref={ab:?} val={:?}",
                            ab.and_then(|b| ctx.trace_ctx.box_value(b)),
                        );
                    }
                }
                return Err(DispatchError::MayForceNullRefArgUnsupported { pc });
            }
        }
    }
    Ok(())
}

/// Diagnostic (`PYRE_FBW_DEBUG_ABORT`): dump the Python coordinate and per-arg
/// provenance behind a ValueUnavailable residual decline — the resume py_pc,
/// the decoded Python opcode, the declined arg's OpRef and whether it is a
/// constant, and its `box_value`.  Attributes an unj_val census walk to a
/// knowable-but-unpopulated value versus a genuinely-symbolic heap object
/// without re-instrumenting.
pub(crate) fn probe_resid_decline_ctx<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    why: &str,
    op_pc: usize,
    arg_index: usize,
    arg: OpRef,
    allboxes: &[OpRef],
) {
    let sym = ctx.fbw_mode.snapshot_sym;
    let (py_pc, opcode) = if !sym.is_null() {
        let s = unsafe { &*sym };
        if !s.jitcode().is_null() {
            let jc = unsafe { &*s.jitcode() };
            let pc = crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.payload.metadata, op_pc)
                as usize;
            let op = if !jc.payload.code_ptr.is_null() {
                pyre_interpreter::decode_instruction_at(unsafe { &*jc.payload.code_ptr }, pc)
                    .map(|(i, _)| format!("{i:?}"))
            } else {
                None
            };
            (Some(pc), op)
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };
    let box_v = ctx.trace_ctx.box_value(arg);
    let arg_id = if arg.is_constant() || arg.is_none() {
        "const".to_string()
    } else {
        format!("r{}", arg.raw())
    };
    // The declined arg's semantic register slot and its index-keyed concrete
    // shadow (`concrete_registers_r`): a `Null` shadow at a found slot with a
    // `None` box_value is the bridge-resume seed gap (neither store populated).
    let reg_slot = ctx.registers_r.iter().position(|&r| r == arg);
    let shadow = reg_slot.and_then(|s| ctx.concrete_registers_r.get(s));
    eprintln!(
        "[fbw-resid-decline] {why} op_pc={op_pc} py_pc={py_pc:?} py_op={opcode:?} \
         arg_index={arg_index} arg={arg_id} box_value={box_v:?} reg_slot={reg_slot:?} \
         reg_shadow={shadow:?} nargs={}",
        allboxes.len() - 1,
    );
}

/// Resolve a Ref register through the rebuilt carrier frame's semantic
/// operand-stack slot. RPython's MIFrame stores the Box at that slot; the flat
/// post-regalloc color is only an encoding detail and can name a freshly
/// recorded load whose runtime value was lost after heapcache invalidation.
fn carrier_stack_box_for_ref_arg<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    op_pc: usize,
    arg: OpRef,
) -> Option<OpRef> {
    if !ctx.fbw_mode.carrier_resume || !ctx.vstack_valid {
        return None;
    }
    let consts = ctx.inline_callee_consts?;
    let color = ctx.registers_r.iter().position(|&value| value == arg)?;
    let raw_code =
        unsafe { pyre_interpreter::w_code_get_ptr(consts.w_code as pyre_object::PyObjectRef) }
            as *const pyre_interpreter::CodeObject;
    if raw_code.is_null() {
        return None;
    }
    let nlocals = unsafe { (&(*raw_code).varnames).len() };
    let maps = crate::state::bridge_semantic_maps_from_pc(consts.jitcode_index, op_pc as i32);
    let semantic = crate::state::semantic_ref_slot_for_reg_color(
        nlocals,
        maps.stack_depth_at_pc,
        &maps.pcdep_entries,
        color,
    )?;
    let stack_slot = semantic.checked_sub(nlocals)?;
    ctx.vstack_boxes
        .get(stack_slot)
        .copied()
        .filter(|&value| value != OpRef::NONE)
}

/// Make a carrier-resumed `CallFn` consume the Box objects owned by its
/// reconstructed MIFrame, as `pyjitpl.py do_residual_call` does. Prefer the
/// semantic color-to-stack mapping; for a post-regalloc color whose mapping is
/// stale, use the CALL operand suffix only when the encoded box has lost its
/// concrete value and the frame box still has one. Keeping an already-concrete
/// encoded box preserves its SSA identity for the optimizer.
fn repair_carrier_call_ref_args<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    op_pc: usize,
    helper: majit_ir::PyreHelperKind,
    r_args: &mut [OpRef],
) {
    if !ctx.fbw_mode.carrier_resume || helper != majit_ir::PyreHelperKind::CallFn {
        return;
    }
    for arg in r_args.iter_mut() {
        if let Some(frame_box) = carrier_stack_box_for_ref_arg(ctx, op_pc, *arg) {
            *arg = frame_box;
        }
    }
    if !ctx.vstack_valid || ctx.vstack_boxes.len() < r_args.len() {
        return;
    }
    let start = ctx.vstack_boxes.len() - r_args.len();
    for (arg, &frame_box) in r_args.iter_mut().zip(&ctx.vstack_boxes[start..]) {
        if ctx.trace_ctx.concrete_of_opref(*arg).is_none()
            && frame_box != OpRef::NONE
            && ctx.trace_ctx.concrete_of_opref(frame_box).is_some()
        {
            *arg = frame_box;
        }
    }
}

/// Whether a residual call is a self-recursive call to the walk's own code —
/// the `CALL_ASSEMBLER` fold target running as a plain residual because the
/// fold declined (no compiled token yet, a non-concrete argument during a
/// bridge resume, etc.).  Mirrors the callee/self resolution in
/// `try_walker_call_assembler_self_recursive`.  Keeps the recursion itself out
/// of the foreign-body-residual latch so pure recursion (`fib`) still folds.
pub(crate) fn residual_callee_is_walk_self_recursive<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    allboxes: &[OpRef],
    helper: majit_ir::PyreHelperKind,
) -> bool {
    if helper != majit_ir::PyreHelperKind::CallFn {
        return false;
    }
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return false;
    }
    // A `bh_call_fn` residual is `[funcptr, callable, null_or_self, arg0, ...]`;
    // the Python callable is `allboxes[1]`.
    let Some(&callable_box) = allboxes.get(1) else {
        return false;
    };
    let Some(majit_ir::Value::Ref(callable_ref)) = ctx.trace_ctx.box_value(callable_box) else {
        return false;
    };
    if callable_ref == majit_ir::GcRef::NO_CONCRETE || callable_ref.as_usize() == 0 {
        return false;
    }
    let callable = callable_ref.as_usize() as pyre_object::PyObjectRef;
    unsafe {
        let Some((w_code, _nparams, _has_closure)) = resolve_inlinable_callee(callable) else {
            return false;
        };
        let sym = &*sym_ptr;
        if sym.jitcode().is_null() {
            return false;
        }
        let caller_code =
            pyre_interpreter::live_code_wrapper((*sym.jitcode()).raw_code() as *const ())
                as *const ();
        w_code as usize == caller_code as usize
    }
}

pub(crate) fn try_execute_residual_call_via_executor<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    call_opcode: OpCode,
    allboxes: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    recorded: OpRef,
    op_pc: usize,
    blackhole_result: Option<(usize, char, usize)>,
) -> Result<ResidualExecOutcome, DispatchError> {
    // `execute_varargs` clears the metainterp exception slot at residual-call
    // entry, before the helper can either run or leave the call recorded
    // symbolically. A handled exception from an earlier opcode must not survive
    // across the back edge and make a later linear `catch_exception/L` look like
    // it is handling a fresh raise.
    clear_walk_exception(ctx);

    // Orthodox sub-jitcode walk safety (#171 wall-5d): a residual call whose
    // funcbox is a `symbolic_fnaddr` placeholder — a 64-bit `DefaultHasher`
    // hash of an in-body helper's `CallPath`/`CallTarget`, minted when
    // `jit_trace_fnaddrs()` has no entry for it (e.g. the zero-arg
    // `SyntheticTransparentCtor "Tuple"` unit constructor inside
    // `w_list_append`) — must not be recorded while inlining a sub-jitcode
    // body.  The production fall-throughs below leave such a call symbolic when
    // folding declines, on the contract that it runs at runtime against live
    // state; but a sub-walk's recorded trace is committed and compiled, so the
    // backend bakes the hash as a code address and the trace branches straight
    // to it -> SIGSEGV.  Decline the whole descent so it aborts gracefully at
    // the first un-lowered helper.  Symbolic hashes carry the
    // `SYMBOLIC_FNADDR_BASE` high-16-bit tag no real funcptr can carry.
    if ctx.fbw_mode.inline_subwalk
        && allboxes.first().is_some_and(|b| b.is_constant())
        && let Some(majit_ir::Value::Int(addr)) = ctx.trace_ctx.box_value(allboxes[0])
        && majit_translate::codewriter::call::is_symbolic_fnaddr(addr)
    {
        return Err(DispatchError::OrthodoxSubWalkTraceUnsupported { pc: op_pc });
    }
    // Authoritative-executor gate: fire ONLY when the walk is the sole
    // concrete-execution leg (the production full-body walk and its
    // inline sub-walks; the per-opcode arm walk is retired).  Shadow /
    // diagnostic-probe runs leave the flag `false` so the call is
    // recorded symbolically without re-running its side effects.
    if !ctx.is_authoritative_executor {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    let plain_or_loopinvariant = matches!(
        call_opcode,
        OpCode::CallI
            | OpCode::CallR
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN
    );
    // `pyjitpl.py do_residual_call` forces branch: every
    // `CallMayForce*` is concrete-executed, with the active
    // virtualizable bracketed by the token protocol (set
    // TOKEN_TRACING_RESCALL before the call, probe-and-clear after —
    // see the doc bullet above).
    let is_may_force = matches!(
        call_opcode,
        OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN
    );
    if !plain_or_loopinvariant && !is_may_force {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    if allboxes.is_empty() {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    // Same funcbox-must-be-const invariant as `try_fold_pure_call_via_executor`:
    // a non-const funcbox carries a stale stamp and dereferencing it as a
    // code address SEGVs.  pyjitpl.py forces the funcbox through
    // the executor's `cpu.bh_call_*` `ConstInt.getint()` path which
    // implicitly requires constness too (residual_call descrs always
    // carry a fixed funcptr at translation time).
    if !allboxes[0].is_constant() {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    // The LOAD_CONST helper (oopspec `LoadConst`) has a dedicated fold in the
    // residual_call dispatchers: when the const index AND the code pointer
    // (`frame.pycode`) are both concrete, it materializes `co_consts[idx]`
    // directly and suppresses the residual.  When that fold declines — the
    // promoted `frame.pycode` is concrete for the portal frame but an inlined
    // callee sub-walk does not seed it — the residual is recorded so the
    // loop computes it at runtime from the live frame's real `pycode`.
    // Executing it concretely here would pass the unseeded (null/garbage)
    // code pointer to `bh_load_const_fn`, which dereferences it via
    // `w_code_get_ptr` and faults.  Leave it symbolic, mirroring the fold's
    // "falls through to the generic record" contract.
    if call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::LoadConst {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    let funcptr_val = ctx.trace_ctx.box_value(allboxes[0]);
    let func_ptr = match funcptr_val {
        Some(majit_ir::Value::Int(addr)) => addr,
        _ => return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic)),
    };
    // Safety gate — reject `symbolic_fnaddr_for_path`
    // placeholder values that escaped runtime patching.  Pyre's
    // codewriter mints a 64-bit hash of the helper's `CallPath` when
    // the build-time `pyre_interpreter::jit_trace_fnaddrs()` snapshot
    // has no entry for it (`symbolic_fnaddr_for_path` in
    // `majit-translate/src/codewriter/call.rs`).  `runtime_fnaddr_patch` rewrites
    // these to real runtime addresses only when the path appears in
    // both the build-time and runtime registries; helpers absent from
    // the runtime registry retain the hash and dereferencing it as a
    // code address SIGBUSes.  Hashes carry the `SYMBOLIC_FNADDR_BASE`
    // high-16-bit tag no real funcptr can carry on any target (a bit-47
    // range test would misclassify every real funcptr on aarch64 Linux,
    // whose 48-bit VA maps code at 0xaaab…/0xffff…).
    if majit_translate::codewriter::call::is_symbolic_fnaddr(func_ptr) {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    // A residual whose funcptr is a `PyFrame` operand-stack accessor
    // (`pop`/`push`/`peek`/`peek_at`) reads or mutates the live frame's
    // operand stack.  During a walk that stack is empty — the walk holds
    // operand values symbolically in its register banks, not on the real
    // frame (the portal lowers stack ops to vable array writes; these
    // accessors appear only inside inlined callee sub-jitcode bodies such as
    // `pop_value`).  Executing one here underflows `PyFrame::pop`'s
    // `valuestackdepth > stack_base()` assertion against the paused outer
    // frame.  Record it symbolically instead, mirroring the tracer's
    // never-mutate-the-traced-frame discipline; it runs at runtime against a
    // frame whose operand stack the compiled trace's preceding pushes have
    // populated.
    if pyre_interpreter::is_pyframe_operand_stack_accessor(func_ptr as usize) {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    // The `list_write_barrier` residual (the #171 object-append fold's Object
    // arm leaves it as a residual because it is `#[dont_look_inside]`) is pure
    // idempotent GC bookkeeping — re-running it on a body replay only re-adds
    // the list to the remembered set, never doubling user-visible state.  It
    // must still EXECUTE concretely below — the walk mutates the live heap, so
    // the store it guards has really happened — but it is not a
    // body effect: keep it out of the in-flight-FOR_ITER body-effect accounting
    // so an Object-strategy comprehension append (`[(i, i) for …]`, `[None …]`)
    // is not refuse-dropped.  RPython treats the write barrier the same way —
    // `COND_CALL_GC_WB` is never executed by pyjitpl
    // (`rpython/jit/metainterp/executor.py:446`), is neither can-raise nor a
    // call (`resoperation.py:1124-1125`), and is inserted only by the backend
    // GC rewrite pass after optimization (`backend/llsupport/rewrite.py:948`),
    // so it never participates in the metainterp's side-effect analysis.
    let is_idempotent_gc_barrier = pyre_interpreter::is_list_write_barrier(func_ptr as usize);
    if allboxes.len() - 1 > majit_translate::codewriter::insns::MAX_HOST_CALL_ARITY {
        return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
    }
    // A void residual (CALL_N family) is a side effect with no result box, so
    // `do_residual_call` executes it EAGERLY during the walk and resumes the
    // compiled loop at iteration N+1 (the commit-invariant note below) — a
    // deferred void store is lost (its symbolic op fires only for N+1+).  When a
    // void call's arg cannot be resolved to a concrete (`GcRef(usize::MAX)` =
    // "no concrete known", or `None` = unbound), eager execution is impossible
    // AND deferral drops the store, so neither path is correct: abort the trace
    // gracefully (interpreter fallback) rather than silently drop it.  This is
    // the off-by-one for a module-global loop that builds and stores a heap
    // object then reads it back (`g = [n]; ... g[0]`): the `STORE_NAME` is a
    // void `CallN` whose value is the still-virtual `BUILD_LIST` result, so the
    // iteration-N store never reaches the cell.  A value-returning call with a
    // non-concrete arg is safe to leave symbolic — the compiled loop computes
    // its result at runtime with no lost side effect.
    let is_void = matches!(
        call_opcode,
        OpCode::CallN | OpCode::CallMayForceN | OpCode::CallLoopinvariantN
    );
    let mut args = Vec::with_capacity(allboxes.len() - 1);
    for (arg_index, &arg) in allboxes[1..].iter().enumerate() {
        let v = match ctx.trace_ctx.box_value(arg) {
            Some(majit_ir::Value::Int(n)) => n,
            Some(majit_ir::Value::Ref(r)) => {
                if r == majit_ir::GcRef::NO_CONCRETE {
                    if is_void {
                        return Err(DispatchError::ResidualCallArgUnbound {
                            pc: op_pc,
                            arg_index,
                        });
                    }
                    if fbw_debug_abort_enabled() {
                        probe_resid_decline_ctx(
                            ctx,
                            "NO_CONCRETE",
                            op_pc,
                            arg_index,
                            arg,
                            allboxes,
                        );
                    }
                    return Ok(ResidualExecOutcome::Declined(
                        ResidualDecline::ValueUnavailable,
                    ));
                }
                r.as_usize() as i64
            }
            Some(majit_ir::Value::Float(f)) => f.to_bits() as i64,
            Some(majit_ir::Value::Void) => 0,
            None => {
                if is_void {
                    return Err(DispatchError::ResidualCallArgUnbound {
                        pc: op_pc,
                        arg_index,
                    });
                }
                if fbw_debug_abort_enabled() {
                    probe_resid_decline_ctx(ctx, "box_value=None", op_pc, arg_index, arg, allboxes);
                }
                return Ok(ResidualExecOutcome::Declined(
                    ResidualDecline::ValueUnavailable,
                ));
            }
        };
        args.push(v);
    }
    // NULL-Ref-arg refusal: same SEGV-avoidance contract as the pure
    // path (see `try_fold_pure_call_via_executor`'s NULL guard).  Pyre's
    // optimizer emits `guard_nonnull` after this walker fold, so a NULL
    // receiver dereferences before that guard exists; fall through to
    // recording the call op and let the optimizer's guard emission
    // handle it at compile time.
    // Exemption: `bh_call_fn_N(callable, null_or_self, args...)`'s
    // `null_or_self` (arg index 1) is a checked sentinel — `PY_NULL`
    // means "no receiver" and is never dereferenced (`bh_call_fn_impl`
    // prepends it as arg0 only when non-null), so a concrete-NULL there
    // is the normal plain-call shape.  These exemptions MUST match
    // `walker_abort_if_mayforce_null_ref_arg`'s — otherwise a normal
    // no-receiver keyword/star call is declined as symbolic
    // (left symbolic), which drops the recording iteration's call
    // exactly once (`g(i, d=4)` in a hot loop summed to n-1, callee
    // ran n-1 times).
    let is_call_fn = call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::CallFn;
    // `bh_call_kw_N(callable, null_or_self, kwnames, args...)` — `null_or_self`
    // (arg index 1) is the same checked `PY_NULL` sentinel.
    let is_call_kw = call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::CallKw;
    // `bh_call_function_ex_fn(callable, self_or_null, starargs, kwargs_or_null)`
    // — `self_or_null` (arg 1) and `kwargs_or_null` (arg 3) are checked `PY_NULL`
    // sentinels (never dereferenced when null), so a concrete-NULL there is the
    // normal `f(*args)` / no-`**` shape.
    let is_call_function_ex =
        call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::CallFunctionEx;
    // Same `RaiseVarargs` trailing-`cause` sentinel exemption as
    // `walker_abort_if_mayforce_null_ref_arg`.
    let is_raise_varargs =
        call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::RaiseVarargs;
    // `bh_store_deref_value_fn(cell, value)` — `value` (arg index 1) is a
    // checked `PY_NULL` sentinel: DELETE_DEREF lowers to
    // `store_deref_value(cell, none)` after its own bound check
    // (`codewriter.rs` `Instruction::DeleteDeref`), and the helper hands the
    // NULL straight to `w_cell_set` without ever dereferencing it.  Declining
    // it left the clear-the-cell half of every `del <cellvar>` and every
    // `except E as e` handler cleanup recorded but UNEXECUTED, which marks the
    // walk unjournaled — so the walk-end flush declines and the caller replays
    // the region on top of the residuals the walk already ran concretely.
    let is_store_deref =
        call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::StoreDeref;
    // Same `bh_load_global_fn` `namespace_ptr` exemption as
    // `walker_abort_if_mayforce_null_ref_arg`: arg index 0 is discarded by the
    // helper, so a concrete NULL there is the normal nested-function shape.
    let is_load_global =
        call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::LoadGlobal;
    for (i, &arg) in args.iter().enumerate() {
        if is_call_fn && i == 1 {
            continue;
        }
        if is_load_global && i == 0 {
            continue;
        }
        if is_call_kw && i == 1 {
            continue;
        }
        if is_call_function_ex && (i == 1 || i == 3) {
            continue;
        }
        if is_raise_varargs && i + 1 == args.len() {
            continue;
        }
        if is_store_deref && i == 1 {
            continue;
        }
        if matches!(call_descr.arg_types().get(i), Some(majit_ir::Type::Ref)) && arg == 0 {
            return Ok(ResidualExecOutcome::Declined(ResidualDecline::Symbolic));
        }
    }
    // #57 (Finding #1, in-place container mutation): an in-flight FOR_ITER
    // body's `acc += delta` is a bare `NB_INPLACE_*` `BinaryOp` residual (args
    // = [lhs, rhs, op_code]) that may mutate its receiver in place at the C
    // level — no Void result, no write tag, no user frame — so none of the
    // body-effect signals below see it.  A committed non-journaled in-place
    // mutation that an aborting walk delivers would be re-run (double); dropped,
    // it would lose the iteration's tail.  Two recoverable shapes are handled
    // here, decided BEFORE any vable/tracing-call state is set up so an early
    // decline strands nothing:
    //
    //  * `acc += [ints]` for two Integer-strategy lists — the extend keeps `acc`
    //    Integer-strategy, so `w_list_int_set_len` can rewind it.  Capture the
    //    pre-extend length; the success arm journals it so the abort rollback
    //    undoes the one extend and the deliver re-applies it exactly once.
    //  * an immutable receiver (`int`/`bool`/`float`/`tuple`/`str`/`bytes`) —
    //    `+=` yields a FRESH object and rebinds the journaled local, so a plain
    //    deliver re-run is exact with no journaling.
    //
    // Any OTHER *exact builtin* receiver — an object-/float-strategy list,
    // `bytearray`, `set`, `dict`, `array`, a mixed `int-list += non-ints` that
    // would change strategy, … — may commit a mutation the rollback cannot
    // rewind, so decline the walk here and let this loop run interpreted
    // (exact), like the gate refusing an unsupported body op.  A user instance
    // must not take that decline: its `__iadd__`/`__isub__`/… has not run yet,
    // while the in-flight `FOR_ITER` item has already been consumed.  The
    // permanent-abort path then drops that item (the conservative delivery
    // gate sees the loop's preceding `STORE_NAME`), silently skipping one
    // augmented-assignment iteration at the trace-entry boundary.  Let the
    // normal residual dispatch execute user special methods; its existing
    // user-frame effect accounting handles any later abort.
    let inplace_list_journal: Option<(pyre_object::PyObjectRef, usize)> =
        if call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::BinaryOp
            && args.len() >= 3
            && pyre_interpreter::runtime_ops::binary_op_tag_is_inplace(args[2])
            && fbw_foriter_inflight_active()
        {
            let lhs = args[0] as usize as pyre_object::PyObjectRef;
            let rhs = args[1] as usize as pyre_object::PyObjectRef;
            unsafe {
                if pyre_object::pyobject::is_exact_list(lhs)
                    && pyre_object::listobject::w_list_is_integer_strategy(lhs)
                    && pyre_object::pyobject::is_exact_list(rhs)
                    && pyre_object::listobject::w_list_is_integer_strategy(rhs)
                {
                    Some((lhs, pyre_object::w_list_len(lhs)))
                } else if pyre_object::pyobject::is_int_or_long(lhs)
                    || pyre_object::pyobject::is_bool(lhs)
                    || pyre_object::pyobject::is_float(lhs)
                    || pyre_object::pyobject::is_tuple(lhs)
                    || pyre_object::unicodeobject::is_str(lhs)
                    || pyre_object::bytesobject::is_bytes(lhs)
                {
                    None
                } else if pyre_object::pyobject::is_exact_builtin_instance(lhs) {
                    return Err(DispatchError::InplaceContainerMutationUnsupported { pc: op_pc });
                } else {
                    None
                }
            }
        } else {
            None
        };
    // A `jit_list_append(list, value)` residual (LIST_APPEND opcode, tagged
    // `ListAppendValue`) reaches this generic executor only when the orthodox
    // append fold DECLINED and fell through — most commonly at a realloc
    // boundary, where the backing block is full and the in-place fast-store
    // fold cannot lower the resize (`w_list_append`'s `Vec::push` else-leg is an
    // un-lowered allocating helper).  Executing it here mutates `list` in place
    // (resize + store); journal the pre-append length so an aborting walk's
    // rollback rewinds the one append and the deliver re-applies it exactly once
    // (the same `fbw_append_journal_push` contract the fold's own commit uses),
    // making the fall-through abort-safe instead of a silent double.
    let list_append_journal: Option<(pyre_object::PyObjectRef, usize)> =
        if call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::ListAppendValue {
            let list = args
                .first()
                .map(|&a| a as usize as pyre_object::PyObjectRef);
            list.filter(|&l| !l.is_null() && unsafe { pyre_object::pyobject::is_list(l) })
                .map(|l| (l, unsafe { pyre_object::w_list_len(l) }))
        } else {
            None
        };
    // `do_residual_call` (pyjitpl.py for CALL_MAY_FORCE_N /
    // CALL_LOOPINVARIANT_N / CALL_N) runs `executor.execute_varargs` for a void
    // call exactly like the value-returning shapes, applying the side effect
    // once during tracing and then resuming the compiled loop at the *next*
    // iteration (`raise_continue_running_normally`, pyjitpl.py, hands
    // back the end-of-iteration-N state so iteration N is never re-run).  Pyre
    // mirrors the second half via the walk-end commit
    // (`flush_walk_end_state_to_frame`, run_perfn_walk): a successful commit
    // adopts the end-of-walk frame so the compiled loop enters at iteration
    // N+1, leaving the eagerly-applied side effect counted once.  Executing
    // void calls here (rather than recording-only) keeps that invariant whole —
    // a deferred void store would be lost on commit (its symbolic op only fires
    // for N+1+), which is why deferral previously forced the no-commit legacy
    // replay.  The replay path that re-runs iteration N has the symmetric
    // hazard for already-executed value calls (e.g. `list.insert` returns the
    // None ref, so it is not a void call yet still mutates) — eager-everything +
    // commit is the single consistent rule, matching `do_residual_call`.
    //
    // The standard virtualizable box pointer for a MayForce residual — a
    // force inside the callee could escape the frame.  None for non-forces
    // opcodes and when no live vable exists (the jitdriver has no standard
    // virtualizable, or unit-test init disabled the heap pointer) — nothing
    // the callee could force.  The token is armed further below, past every
    // decline gate.
    let mut vable_obj_root = if is_may_force {
        ctx.trace_ctx
            .standard_virtualizable_box()
            .and_then(|_| ctx.trace_ctx.virtualizable_heap_ptr())
            .filter(|p| !p.is_null())
            .map(|p| Box::new(p as usize as i64))
    } else {
        None
    };
    // A Python-level callee (e.g. a recursive `fib`) re-enters the
    // interpreter (`eval_loop_jit` → `jit_merge_point`) while this walk still
    // holds the driver in the tracing state.  Suspend re-entrant trace
    // continuation for the duration of the concrete call so the callee runs as
    // plain interpretation instead of starting a nested trace that would share
    // and corrupt this walk's `TraceCtx` (flaky `libsystem_malloc` freelist
    // abort during deep recursion).  Plain C-helper callees never re-enter, so
    // the guard is a no-op for them.
    //
    // In RPython the tracing metainterp and the executing (blackhole /
    // compiled) interpreter are SEPARATE objects, so `do_residual_call`
    // never perturbs the tracer's `MetaInterp.vable_ptr` /
    // `virtualizable_boxes`.  Pyre shares one `TraceCtx` across the walk
    // and any re-entrant JIT activity the concrete call triggers: a
    // self-recursive `CALL_ASSEMBLER` callee that re-enters compiled code
    // and deopts runs `set_vable_ptr` for the nested frames, leaving
    // `virtualizable_heap_ptr` pointing at a nested callee frame whose
    // `vable_token` is still the live JIT FORCE_TOKEN.  The next
    // `tracing_before_residual_call` in this same walk would then assert on
    // the non-NONE token (virtualizable.rs).  Snapshot the standard
    // virtualizable pointer and restore it after the call so the walk's
    // subsequent vable token protocol / field reads see the frame being
    // traced, mirroring RPython's separate-state isolation.
    let saved_vable_heap_ptr = ctx.trace_ctx.virtualizable_heap_ptr();
    // #57 Option C (Finding #1, R1 double-apply guard): whether THIS residual
    // could commit an irreversible heap mutation the journals do not cover,
    // while an in-flight FOR_ITER item is already captured (a consume ran
    // earlier this iteration).  The journaled list ops (setitem / append) run
    // OUTSIDE this executor (`try_walker_store_subscr_specialization` /
    // `try_walker_orthodox_list_append`) and roll back on abort, so they are
    // not body-effect candidates here.
    //
    // The OLD allow-list (`StoreSubscr` / `CallFn` / `SetCurrentException`
    // only) MISSED the many statement-level mutators that reach this executor
    // concretely, succeed, and carry `PyreHelperKind::None` (`store_attr_fn` /
    // `delete_subscr_fn` / `delete_attr_fn` / `list_extend_fn` / `store_name_fn`
    // / `store_global` / `store_slice` …): a missed mutator is a silent double
    // on a body re-run (correctness-FATAL).  Track residuals that WRITE live
    // heap state outside the journals.
    //
    // The write discriminator is the residual's RESULT TYPE plus the
    // value-returning-mutator helper tags — NOT `extraeffect`, which cannot
    // separate a write from a read: `getattr_fn` (a pure `.append` bound-method
    // lookup) and `store_attr_fn` are BOTH `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
    // (the analyzer is stubbed, so `write_descrs_*` are empty for both).  A
    // `Void`-result residual produces no value, so it is executed solely for
    // its heap side effect → a write.  Every statement-level mutator above
    // lowers to a `residual_call_*_v` (Void); the benign reads (`getattr`,
    // `load_global`, `load_name`, `load_deref` …) all RETURN a value, so a
    // Void result is a clean write proxy that does not over-refuse a read.
    //
    // The few VALUE-returning writes the Void test cannot see are caught by
    // helper tag: `CallFn` (an opaque Python call returning its None ref may
    // mutate arbitrary state), `StoreSubscr` (a dict/object `o[k]=v` returning
    // the stored value), `SetCurrentException` (the TLS exc-slot write), and
    // `StoreDeref` (an in-place closure-cell write returning the slot value —
    // `nonlocal n; n += 1`).  Over-refusing only these (never a benign read)
    // keeps the journaled-append body (`for_mutate`) delivering, since its
    // `getattr`/append residuals return `Ref` and carry no write tag.
    //
    // Provably read-only/elidable residuals are exempt up front: `@jit.elidable`-
    // class (`check_is_elidable`: `EF_ELIDABLE_*`, the pure executor folds
    // these) or `EF_LOOPINVARIANT` (loop-hoisted).  The `for_iter_next` consume
    // itself ([`PyreHelperKind::ForIterNext`]) is excluded — it is the SOURCE of
    // the capture (it runs while the PRIOR iteration's item is still in flight),
    // not a body effect for that prior iteration.  Sampled BEFORE the call so
    // the success arm can flag an effect that committed AFTER the in-flight
    // consume.
    let ei = call_descr.get_extra_info();
    let helper = ei.pyre_helper;
    // The declared re-runnability class, the axis `EffectInfo` is decided on at
    // codewriter time (`jtransform.py:620-630`): re-executing the opcode that
    // contains this residual re-executes the residual, which is harmless only
    // for an elidable or loop-hoisted one.  Feeds [`ESCAPE_OPCODE_WINDOW`].
    let reentrant_residual =
        ei.check_is_elidable() || ei.extraeffect == majit_ir::ExtraEffect::LoopInvariant;
    let provably_side_effect_free =
        reentrant_residual || helper == majit_ir::PyreHelperKind::ForIterNext;
    let writes_live_heap = call_descr.result_type() == majit_ir::Type::Void
        || matches!(
            helper,
            majit_ir::PyreHelperKind::CallFn
                | majit_ir::PyreHelperKind::StoreSubscr
                | majit_ir::PyreHelperKind::SetCurrentException
                | majit_ir::PyreHelperKind::StoreDeref
        );
    // Inside an inline sub-walk, decline before any residual that is not
    // provably side-effect-free.  Ref-result getters/dunders/user `__next__`
    // can mutate live heap through user frames while `writes_live_heap` is
    // false, and rollback would miss that concrete mutation.  The helper no-ops
    // on an empty session framestack, so top-level depth-1 behavior is unchanged.
    if !provably_side_effect_free {
        fbw_abort_nested_unjournaled_residual(ctx, op_pc)?;
    }
    // `vinfo.tracing_before_residual_call(virtualizable)`
    // heap half: every decline gate has now passed, so the helper WILL
    // execute — set TOKEN_TRACING_RESCALL on the active virtualizable so a
    // force inside the callee is observable afterwards.  Armed AFTER the
    // inline-subwalk decline so a declined residual never strands the token:
    // tracing_before pairs with the tracing_after clear below only for
    // residuals that proceed, mirroring `do_residual_call` where
    // `vable_and_vrefs_before_residual_call` (pyjitpl.py) runs only past
    // the OS_NOT_IN_TRACE / force-virtual short-circuits. RPython mirror:
    // `pyjitpl.py`.
    // `vrefinfo.tracing_before_residual_call(vref)` for every live vref
    // (`pyjitpl.py:3341-3348`), which upstream runs first inside the same
    // `vable_and_vrefs_before_residual_call` the vable half below mirrors.
    // Stamps TOKEN_TRACING_RESCALL so the post-call check can tell "forced by
    // this callee" from "untouched".  Armed here, past every decline gate, for
    // the same reason the vable half is: a declined residual must not strand a
    // token.
    //
    // Gated on `is_may_force` — the walker's `check_forces_virtual_or_
    // virtualizable()` — because `do_residual_call` runs the whole preparation
    // block only for `assembler_call or effectinfo.check_forces_...`
    // (`pyjitpl.py:2007`).  A call that cannot force needs no stamp, and
    // stamping one would leave `tracing_after_residual_call` reading a token
    // nobody will clear.
    if is_may_force {
        ctx.trace_ctx.vrefs_before_residual_call();
    }
    let live_frame = if ctx.fbw_mode.snapshot_sym.is_null() {
        0
    } else {
        unsafe { (*ctx.fbw_mode.snapshot_sym).live_vable_frame_addr() }
    };
    // Resolved against the callee's OWN metadata, because `vstack_cur_pypc` is
    // the outer walk's mirror and a sub-walk never advances it.
    let inline_callee_pc = inline_callee_py_pc(ctx, op_pc);
    let vable_root_depth = if let Some(obj) = vable_obj_root.as_mut() {
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        let root_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
        // Publish the current Python pc so a force inside the callee reports
        // the executing line rather than the one the last resume point left
        // behind.
        //
        // Only while the pc indexes the walk's own virtualizable.  Inside an
        // inline sub-walk `vstack_cur_pypc` is in the CALLEE's code, and both
        // halves below name the outer frame, so publishing there stamps a
        // foreign pc onto it — `offset2lineno` then resolves it against the
        // outer code object and reports whatever line that byte happens to sit
        // on.  [`LiveLastInstrGuard`] makes the matching retarget for the
        // concrete store, publishing onto the callee's own frame instead.
        if current_inline_concrete_frame() == 0 {
            let last_instr = ctx.trace_ctx.const_int(ctx.vstack_cur_pypc as i64);
            crate::trace_opcode::mirror_vable_static_to_boxes(
                ctx.trace_ctx,
                "last_instr",
                last_instr,
                majit_ir::Value::Int(ctx.vstack_cur_pypc as i64),
            );
            // …and the heap half of the same `_opimpl_setfield_vable` shape
            // (`virtualizable_boxes[index] = valuebox; synchronize_virtualizable()`
            // — the mirror above is only the box half, and writes the trace's
            // shadow, never the frame).  The same constant feeds both so the
            // shadow and the heap cannot disagree.
            //
            // Upstream needs neither at a residual call: its frame readers are
            // traced in and read `last_instr` off the virtual frame.  pyre
            // residualizes them and reads the heap, and the frame-chain walk no
            // longer forces (`ExecutionContext::force_frame`), so without this
            // store nothing keeps the field current in compiled code: left at
            // the last resume point, `_warnings::setup_context` keys its
            // registry on the wrong line and re-issues a warning the
            // interpreted run already deduplicated.
            if let Some(vable_ref) = ctx.trace_ctx.standard_virtualizable_box()
                && let Some(idx) = info.static_field_index_by_name("last_instr")
            {
                // Record under the PARENT-STRUCT field descr, the resolution
                // `vable_setfield` applies through `vable_static_record_descr`;
                // `vable_setfield_descr` is a raw pass-through and does not.
                // The vinfo's own `static_field_descrs[i]` numbers the field by
                // the vinfo `[token, statics, arrays]` order, which diverges
                // from PyFrame's struct declaration order, so the store pairs
                // against the wrong slot.  `virtualizable.py:71` builds the
                // vinfo descrs with `cpu.fielddescrof(VTYPE, name)`, so
                // upstream's vinfo descr and the descr an ordinary
                // `setfield_gc` carries are one object and the question cannot
                // arise there.  Here it decides whether this publish and
                // `emit_traceback_node`'s store — same frame, same offset —
                // supersede each other or survive as two independent
                // locations flushed in an arbitrary order.
                let descr = info.static_field_struct_descr(idx);
                ctx.trace_ctx
                    .vable_setfield_descr(vable_ref, last_instr, descr);
            }
        }
        unsafe {
            majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_mut(&mut **obj));
            info.tracing_before_residual_call(**obj as usize as *mut u8);
        }
        Some(root_depth)
    } else {
        None
    };
    // The loop-variable binding store is the op at the in-flight FOR_ITER's
    // `body_pc` (the FOR_ITER continue-arm fallthrough), a STORE_NAME/
    // STORE_GLOBAL that writes the just-consumed item to the loop target (a
    // module/global-scope `for i in …`; a function-scope loop var is a
    // STORE_FAST frame local that never becomes a residual).  Re-delivery
    // re-runs the body from `body_pc`, re-storing the SAME re-delivered item to
    // the SAME name — an idempotent write, never an accumulating double.  Like
    // the `is_idempotent_gc_barrier` write barrier it still EXECUTES concretely
    // (the module dict must hold the binding for the walk's remaining reads) but
    // it is not a body effect: keep it out of the R1 in-flight-FOR_ITER
    // accounting so an escaping residual later in the same body does not
    // refuse-drop the whole iteration.
    // `vstack_cur_pypc` is the pc the walk is ABOUT TO ENTER
    // (`reconcile_vstack_at_boundary` sets it to `new_pypc` after reconciling
    // the PREVIOUS opcode), so at this residual it names the opcode being
    // walked.  The loop-var store is recognised by the FOR_ITER body's own
    // relation `body_pc + 1 == vstack_cur_pypc`, i.e. the walk has advanced one
    // opcode past the recorded body pc — not by a next-instr convention.
    let is_loop_var_binding_store = matches!(
        helper,
        majit_ir::PyreHelperKind::StoreName | majit_ir::PyreHelperKind::StoreGlobal
    ) && fbw_foriter_inflight_top_body_pc()
        .is_some_and(|body_pc| body_pc + 1 == ctx.vstack_cur_pypc as usize);
    let body_effect_candidate = !provably_side_effect_free
        && !is_idempotent_gc_barrier
        && !is_loop_var_binding_store
        && writes_live_heap
        && fbw_foriter_inflight_active();
    // #57 Option C (Finding #1, user-frame signal): the Void/helper-tag write
    // discriminator above cannot see a body effect committed through USER
    // PYTHON CODE by a value-returning (`Ref`), `PyreHelperKind::None`,
    // `MayForce` residual: `obj.prop` (a `@property` getter / `__getattr__` /
    // descriptor `__get__`), `a + b` / `a == b` (user `__add__` / `__eq__`),
    // `iter(obj)` (user `__iter__`), `str(obj)` / `f"{obj}"` (user `__str__` /
    // `__format__`), `import name`.  Each RETURNS a value (so the Void proxy
    // misses it) and carries no write tag, yet its getter/dunder/module body
    // may mutate live heap.  Those mutations all run a USER PYTHON FRAME (the
    // getter's bytecode); a pure builtin path (`seen.append`'s C-level
    // bound-method lookup, `int.__add__`) does NOT.  Snapshot the monotonic
    // frame eval-loop entry odometer before the call; if it advanced while an
    // in-flight FOR_ITER item is active, the residual ran user bytecode that
    // could have committed an irreversible body effect → flag it (the success
    // arm compares post-call).  Sampled only when an item is in flight and the
    // residual is not provably read-only (an elidable / loop-invariant fold or
    // the `for_iter_next` consume itself never counts).
    let user_frame_snapshot = (!provably_side_effect_free && fbw_foriter_inflight_active())
        .then(pyre_interpreter::call::frame_entry_count);
    // #493: a NEW consume attempt for a FOR_ITER whose prior item is still in
    // flight means that item's body ran to completion — mark the entry BEFORE
    // the call so an attempt that aborts mid-way (a kept-stack guard on the
    // exhaustion arm) still records the completion; a successful attempt
    // replaces the entry with a fresh one anyway.
    if helper == majit_ir::PyreHelperKind::ForIterNext {
        let body = fbw_foriter_body_from_op_pc(ctx.fbw_mode.snapshot_sym, op_pc)
            .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
        fbw_foriter_inflight_mark_attempt(body);
    }
    // gh#467: sample the user-frame odometer UNCONDITIONALLY (not only under an
    // in-flight FOR_ITER) so the concrete-heap-write gate can detect a callee
    // sub-walk mutation committed through a value-returning dunder body — the
    // same user-frame signal Finding #1 uses, generalized past FOR_ITER.
    let heap_write_odometer_before =
        (!provably_side_effect_free).then(pyre_interpreter::call::frame_entry_count);
    let exec_result = {
        let escape_frame = if is_may_force { live_frame } else { 0 };
        // Latch the operand-stack mirror for the escape flush: at force time
        // the walk-end flush needs the caller's mid-expression stack, which
        // the vable shadow cannot provide (`reconstructed_all_ref_call_stack`
        // resolves the same mirror for the inline-abort Entry carrier).
        //
        // A WRITING residual is latched too, but never to resume AT this
        // opcode: the withdrawal below cancels its commit and restores the
        // pre-flush frame unconditionally, so the interpreter never
        // re-executes it and never re-applies the write.  What the commit
        // leaves behind for it is a WITNESS — it proves every mirror slot
        // resolved to a concrete non-null Ref, i.e. the walk holds a complete
        // mid-expression stack image, which is exactly the precondition for
        // building the blackhole resume PAST the residual.  Forcing is NOT
        // limited to frame-introspection reads (`hook_access_field`,
        // rvirtualizable.py:49-53, forces on every redirected-field access,
        // reads AND writes), and every Python-visible frame MUTATOR (the
        // `f_lineno`/`f_trace` setters, `sys.settrace`, `_warnings.warn`,
        // which forces the caller frame for `__name__` and then mutates
        // `__warningregistry__` with no user frame entered) is a
        // Void-returning store or a CALL-shaped helper, so all of them land
        // on the writing side and take that withdrawal.
        //
        // A non-writing residual that entered no user frame keeps its commit
        // and the resume-AT-opcode semantics: what survives the gates there
        // is attribute/item READS of frame-family objects, which are
        // idempotently re-executable (re-execution reads the same flushed
        // values the first execution saw; a token re-force is a no-op).
        //
        // Why the licence is `!writes_live_heap` and not the effect class the
        // rewound residual declares: BOTH upstream-shaped static licences are
        // empty on this path.  `check_is_elidable()`/`EF_LOOPINVARIANT` is
        // disjoint from forcing by construction — `pyjitpl.py:2007` takes the
        // forcing arm iff `check_forces_virtual_or_virtualizable()`
        // (`extraeffect >= EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`, effectinfo.py:250)
        // and routes elidable/loop-invariant down the `else` arm at `:2084`.
        // An empty declared write set is unavailable too: every residual that
        // reaches this gate carries `EF_RANDOM_EFFECTS`, whose write-descr sets
        // are `None` — top, not bottom — by upstream's own assertion
        // (effectinfo.py:149-155).  So no `EffectInfo` predicate can license
        // this rewind, and the licence has to come from the shape gates above.
        // Measured over `pyre/bench/synth` (312 files, 115 forces): the only
        // shape that commits an `EscapeResumeKind::Exact` is a `LoadAttr`/`Ref`
        // frame-family read, 10 of them, from `getframe_stored_fback_walk` and
        // `getframe_force_cancel_journal`; 5 enter a user frame and have the
        // commit withdrawn below, 5 do not and are re-executed.  Every other
        // force either commits nothing or takes the merge-point fallback, whose
        // `RerunsOpcode` kind `commit_walk_end` refuses outright.
        //
        // A sub-walk's mirror describes the callee frame, not the escape
        // frame, so it is never latched (the nested unjournaled-residual
        // decline above already aborts before this).
        // Not latched on a bridge walk: its abort path bypasses the
        // run_perfn_walk epilogue that adopts (or restores) a committed
        // escape flush, so a commit there would strand a moved frame.
        let escape_py_pc = (!ctx.fbw_mode.inline_subwalk && ctx.vstack_valid)
            .then_some(ctx.vstack_cur_pypc as usize);
        let escape_stack = escape_py_pc
            .filter(|py_pc| {
                escape_frame != 0
                    && !ctx.trace_ctx.is_bridge_trace
                    && escape_opcode_window_clean(*py_pc)
            })
            .map(|_| ctx.vstack_boxes.clone());
        let _frame_escape = ActiveFrameEscapeGuard::enter(escape_frame, escape_py_pc, escape_stack);
        // `executioncontext.py:85 enter` for the inlined callee this residual
        // runs inside of.
        let _frame_chain = ResidualFrameChainGuard::enter();
        let live_py_pc = if ctx.fbw_mode.inline_subwalk {
            ctx.vstack_cur_pypc
        } else {
            live_py_pc_from_snapshot(ctx, op_pc).unwrap_or(ctx.vstack_cur_pypc)
        };
        let _callee_last_instr =
            LiveLastInstrGuard::enter(live_frame, live_py_pc, inline_callee_pc);
        let _caller_last_instr = ctx.fbw_mode.inline_caller_py_pc.map(|py_pc| {
            LiveLastInstrGuard::enter_frame(live_frame as *mut pyre_interpreter::PyFrame, py_pc)
        });
        let _suspend = majit_metainterp::TraceContinuationSuspendGuard::enter();
        majit_metainterp::executor::execute_residual_call(call_descr, func_ptr, &args)
    };
    // Declared only now, so this residual constrains the residuals that FOLLOW
    // it inside the same opcode and never itself: the gate above read the
    // window, and a force inside the callee reads it again from
    // `flush_active_frame_escape` while it is still the gate's view.
    escape_opcode_window_note(ctx.vstack_cur_pypc as usize, reentrant_residual);
    if !provably_side_effect_free && !is_idempotent_gc_barrier {
        fbw_mark_executed_nonpure_residual();
        // Count only a FOREIGN non-pure residual: a self-recursive call is the
        // fold target running because its fold declined, not a body side effect.
        if !residual_callee_is_walk_self_recursive(ctx, allboxes, helper) {
            fbw_mark_executed_body_residual();
        }
    }
    let restored_vable_heap_ptr = vable_obj_root
        .as_ref()
        .map(|obj| **obj as usize as *const u8)
        .or(saved_vable_heap_ptr)
        .unwrap_or(std::ptr::null());
    ctx.trace_ctx
        .set_virtualizable_heap_ptr(restored_vable_heap_ptr);
    // `pyjitpl.py:2049` step 3, "after this call, check the vrefs.  If any
    // have been forced by the call, then we record in the trace a
    // VIRTUAL_REF_FINISH---before we record any CALL".  Runs before the
    // virtualizable check below, matching the upstream order, and before the
    // CALL op is recorded further down.  A vref the callee handed to Python
    // stops being tracked and its box becomes ConstPtr(NULL), so the resume
    // snapshot no longer claims the frame is still virtual.  Paired with the
    // pre-call stamp, so it carries the same `is_may_force` gate.
    if is_may_force {
        ctx.trace_ctx.vrefs_after_residual_call();
    }
    // `vinfo.tracing_after_residual_call(virtualizable)`
    // heap half: a cleared token means the callee forced the virtualizable —
    // the frame escaped, the trace must abort (pyjitpl.py
    // `SwitchToBlackhole(Counters.ABORT_ESCAPE, raising_exception=True)`).
    // The interpreter resumes from the live frame, which the callee's force
    // path made heap-authoritative. Reload the walk shadow before aborting so
    // later cleanup cannot restore pre-force fields.  An intact token is
    // cleared back to TOKEN_NONE.
    if let Some(obj) = vable_obj_root.as_ref() {
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        let forced = unsafe { info.tracing_after_residual_call(**obj as usize as *mut u8) };
        if let Some(depth) = vable_root_depth {
            majit_gc::shadow_stack::pop_resume_ref_roots_to(depth);
        }
        if forced {
            disarm_folded_inline_callee_after_escape(ctx, op_pc)?;
            if fbw_debug_abort_enabled() {
                eprintln!(
                    "[force-shape] helper={helper:?} rtype={:?} writes_live={writes_live_heap} \
                     reentrant={reentrant_residual} commit={} entered_frame={} bh={} \
                     fs={} subwalk={} bridge={} wf={:?} wa={:?} wi={:?} rnd={} fn=0x{:x} \
                     pc={op_pc}",
                    call_descr.result_type(),
                    match committed_frame_escape_pc() {
                        None => "none",
                        Some((_, EscapeResumeKind::Exact)) => "exact",
                        Some((_, EscapeResumeKind::RerunsOpcode)) => "reruns",
                    },
                    heap_write_odometer_before
                        .is_some_and(|before| pyre_interpreter::call::frame_entry_count() != before),
                    blackhole_result.is_some(),
                    ctx.session.borrow().framestack.len(),
                    ctx.fbw_mode.inline_subwalk,
                    ctx.trace_ctx.is_bridge_trace,
                    ei._write_descrs_fields.as_ref().map(Vec::len),
                    ei._write_descrs_arrays.as_ref().map(Vec::len),
                    ei._write_descrs_interiorfields.as_ref().map(Vec::len),
                    ei.has_random_effects(),
                    func_ptr as usize,
                );
            }
            // The escaping residual either wrote live heap outside the
            // journals, or entered a user Python frame whose body may have
            // committed irreversible effects.  Either way a committed escape
            // resume would re-execute this opcode and so re-apply that write
            // / re-run that body.  Withdraw the commit AND restore the
            // pre-flush frame so the legacy replay re-enters pristine
            // pre-walk state (the flush moved the live frame mid-iteration;
            // replaying on top of it loses journal-rolled-back effects and
            // re-runs partial state).
            if writes_live_heap
                || heap_write_odometer_before
                    .is_some_and(|before| pyre_interpreter::call::frame_entry_count() != before)
            {
                cancel_committed_frame_escape_pc();
                // The restore is DEFERRED to the walk-end epilogue, which is
                // where the legacy replay is actually chosen.  Undoing it here
                // also undid `write_boxes`' locals materialization on the
                // resume-PAST path below — the blackhole then continued into a
                // frame whose fastlocals were back to their unwritten nulls
                // (`x = 1; a = sys._getframe(0); return x` returned NULL).
                // The capture stays armed; `run_perfn_walk` restores it only
                // when neither the force blackhole nor a committed escape pc
                // takes over the continuation.
                mark_escape_flush_undo_pending();
            }
            // C3 S1: the writes-live-heap force shape cannot safely resume AT
            // this opcode, but a top-level one-frame walk can resume PAST it
            // through the existing blackhole when every live color has a
            // concrete value.  Build here, before WalkContext's concrete banks
            // disappear, and consume only in run_perfn_walk's VableEscaped
            // epilogue.  Every condition is additive under an exact opt-in;
            // failure leaves the pre-existing escape/replay path untouched.
            let odometer_unchanged = heap_write_odometer_before
                .is_some_and(|before| pyre_interpreter::call::frame_entry_count() == before);
            if fbw_debug_abort_enabled() && ctx.fbw_mode.inline_subwalk {
                eprintln!(
                    "[s2-gate] inline_subwalk fs={} writes_live={} odo_unchanged={} \
                     committed_none={} not_bridge={} bh_result_some={} sym_nonnull={}",
                    ctx.session.borrow().framestack.len(),
                    writes_live_heap,
                    odometer_unchanged,
                    committed_frame_escape_pc().is_none(),
                    !ctx.trace_ctx.is_bridge_trace,
                    blackhole_result.is_some(),
                    !ctx.fbw_mode.snapshot_sym.is_null(),
                );
            }
            // `vable_after_residual_call` has exactly one continuation for an
            // escape: `load_fields_from_virtualizable()` then
            // `SwitchToBlackhole(ABORT_ESCAPE, raising_exception=True)`.  It
            // never rewinds — the residual has run, its result is in hand, and
            // the blackhole picks up PAST the call.  A top-level one-frame walk
            // reaches that continuation whenever every live color has a
            // concrete value, so build the resume-past image here, before
            // WalkContext's concrete banks disappear, and consume it in
            // run_perfn_walk's VableEscaped epilogue.  Both the latch
            // (non-bridge, empty framestack, not an inline sub-walk, resolvable
            // snapshot sym) and the adopt (`try_adopt_single_frame_blackhole` →
            // `apply_single_frame_blackhole_crn`, which validates every mapped
            // color and every live operand-stack slot before writing anything)
            // decline to the pre-existing escape/replay path on any unmet
            // condition, so this only ever replaces a replay that would have
            // produced the same state.
            //
            // Neither `writes_live_heap` nor the odometer gates it.  Both
            // describe hazards of RE-RUNNING the escaping opcode, which is what
            // the rewind latch and the legacy replay do; this leg resumes past
            // the opcode with the residual's result (or its raise) spliced in,
            // so the residual runs exactly once no matter what it wrote or
            // whether it entered a Python frame.  `writes_live_heap` is a static
            // helper-kind list that does not even contain `CallFunctionEx`,
            // whose callee is arbitrary user code; gating on it left exactly the
            // frames whose replay is unsound — a re-entry guard plus a
            // non-idempotent store ahead of the escaping call — on the replay
            // path.  Upstream has no counterpart to either gate: ABORT_ESCAPE
            // goes straight to `run_blackhole_interp_to_cancel_tracing`
            // (`pyjitpl.py:2949` → `blackhole.py convert_and_run_from_pyjitpl`),
            // which converts the framestack and runs FORWARD, never replays.
            //
            // An already-committed escape pc does not gate it either.  The
            // withdraw above cancels the commit but only MARKS the frame
            // restore pending, so the image is built over the forced frame it
            // describes; `run_perfn_walk`'s epilogue runs the restore later and
            // only when neither this image nor an escape pc claimed the
            // continuation.  Requiring `committed_frame_escape_pc().is_none()`
            // here instead left the shapes that DID commit — a re-entry guard
            // plus a non-idempotent store ahead of the escaping call — on the
            // replay path.
            if let Some((resume_pc, result_bank, result_color)) = blackhole_result
                && !ctx.fbw_mode.snapshot_sym.is_null()
            {
                let (lastop_result, last_exc_value, raising_exception) = match exec_result {
                    Ok(value) => (
                        (result_bank != 'v').then_some((result_bank, result_color, value)),
                        0,
                        false,
                    ),
                    Err(exc) => (None, exc, true),
                };
                if ctx.session.borrow().framestack.is_empty() && !ctx.fbw_mode.inline_subwalk {
                    let jitcode = unsafe {
                        let sym = &*ctx.fbw_mode.snapshot_sym;
                        (!sym.jitcode().is_null())
                            .then(|| (&(*sym.jitcode()).payload).jitcode.clone())
                    };
                    if let Some(miframe) = jitcode.and_then(|jitcode| {
                        build_single_frame_miframe(ctx, jitcode, resume_pc, lastop_result)
                    }) {
                        // `ctx` is the root walk on this arm
                        // (`framestack.is_empty() && !inline_subwalk`).  It
                        // stops inside the residual call, and the resumed
                        // blackhole can reach a `getarrayitem_vable_r` that
                        // reloads an operand from the virtualizable array, so
                        // publish the root stack from the walker's mirror.
                        let mirror_stack = capture_vstack_mirror_image(ctx, "escape-flush");
                        FBW_SINGLE_FRAME_BLACKHOLE.with(|slot| {
                            *slot.borrow_mut() = Some(LatchedSingleFrameBlackhole {
                                miframe,
                                last_exc_value,
                                raising_exception,
                                mirror_stack,
                            });
                        });
                    }
                // An inlined sub-walk adopts the multi-frame blackhole image.
                // The build side (`build_multi_frame_miframe`, the input-arg
                // `_resref` seed, and the getfield-chain `recover_ref_value`)
                // reconstructs the frame stack; the resume side
                // (`drive_multi_frame_blackhole` →
                // `convert_and_run_from_pyjitpl`) publishes each level as the
                // chain reaches it. The blast radius is exactly
                // `inline_subwalk` at a vable escape: this is an `if`/`else
                // if`, and the single-frame arm requires
                // `framestack.is_empty() && !inline_subwalk`.
                //
                // This path was once gated because the walker executes
                // residuals concretely while an inline push does not run the
                // interpreter's call sequence. `ec.topframeref` therefore
                // named the CALLER while an inlined callee body ran, so a
                // `sys._getframe` that was itself the escaping residual read
                // the wrong frame at walk time. Adopting committed that answer
                // where legacy escape/replay discarded it. A `sys._getframe`
                // executed later, inside the blackhole, was always correct.
                // `walker_ec_enter` / `walker_ec_leave` (the port of
                // `executioncontext.py:85-107`) publish the callee frame at
                // the inlined-call push, which closed the gap.
                // `synth/getframe_while_escaping_read_frame_identity` is the
                // regression guard.
                } else if writes_live_heap
                    && odometer_unchanged
                    && ctx.fbw_mode.inline_subwalk
                    && let Some(framestack) = build_multi_frame_miframe(
                        ctx,
                        resume_pc,
                        InnermostMiframeBuild::LiveMarker(lastop_result),
                        "escape-flush",
                    )
                {
                    FBW_MULTI_FRAME_BLACKHOLE.with(|slot| {
                        *slot.borrow_mut() = Some(LatchedMultiFrameBlackhole {
                            framestack,
                            last_exc_value,
                            raising_exception,
                            publish_root_stack: false,
                            mirror_stack: None,
                        });
                    });
                }
            }
            // On a kept commit the undo stays armed: the abort epilogue
            // consumes it — discard on adoption, restore when the flush is
            // not adopted.  On the cancel arm the restore is only MARKED
            // pending, so no restore has run by the time this refresh does:
            // the reload is `load_fields_from_virtualizable()`'s either way,
            // i.e. the FORCED frame.  A ladder leg that needs PRE-walk shadow
            // state after a cancelled commit has to re-read it after the
            // walk-end restore, not here.
            ctx.trace_ctx.refresh_virtualizable_shadow_from_heap();
            if fbw_debug_abort_enabled() {
                // `vable_after_residual_call`'s
                // `debug_print('vable escaped during a call in %s')`: name the
                // callee through `get_name_from_address`, falling back to the
                // raw address.  Which helper forced the virtualizable is the
                // only thing that distinguishes one ABORT_ESCAPE from another,
                // and the trace is gone by the time the abort surfaces.
                let addr = match ctx.trace_ctx.box_value(allboxes[0]) {
                    Some(majit_ir::Value::Int(a)) => a,
                    _ => 0,
                };
                match pyre_interpreter::jit_trace_fnaddrs()
                    .into_iter()
                    .find(|(_, a)| *a == addr)
                {
                    Some((name, _)) => eprintln!(
                        "[fbw-escape] pc={op_pc} vable escaped during a call in ConstClass({name})"
                    ),
                    None => {
                        eprintln!(
                            "[fbw-escape] pc={op_pc} vable escaped during a call in {addr:#x}"
                        )
                    }
                }
            }
            // `pyjitpl.py:3389-3392`: ABORT_ESCAPE is raised with
            // `raising_exception=True` "because we must still have the eventual
            // exception raised (this is normally done after the call to
            // `vable_after_residual_call()`)" — the post-call `execute_raised`
            // bookkeeping still owes its work even though the walk is over.
            // Returning here skips the `Err(bh_exc)` arm below, and with it the
            // half of that bookkeeping that has no upstream counterpart: the
            // shared `bh_*` helper published this raise into the backend
            // `_store_exception` cells as well as `BH_LAST_EXC_VALUE`
            // (`publish_residual_call_exception`), and those cells belong to
            // compiled / blackhole execution, which RPython tracing never
            // touches.  A survivor is read by the next compiled trace's
            // `must_save_exception` guard and delivered as that frame's own
            // raise — an exception surfacing out of a frame that raised
            // nothing.  `BH_LAST_EXC_VALUE` stays as `execute_residual_call`
            // left it (cleared on read): the escape's consumers take the
            // exception from `exec_result` directly, so restoring it would only
            // outlive the walk.
            if exec_result.is_err()
                && let Some(cb) = crate::callbacks::try_get()
            {
                (cb.drain_backend_jit_exc)();
            }
            crate::state::note_vable_escape_abort();
            return Err(DispatchError::VableEscapedDuringResidualCall { pc: op_pc });
        }
    }
    // A flush that ran without a forced abort (an unarmed token or a missing
    // vable root) must not leak the moved frame into the continuing walk.
    restore_escape_flush_undo();
    // #57 Option C (Finding #1, R1): a residual that is not provably
    // side-effect-free has now EXECUTED AFTER the in-flight FOR_ITER consume —
    // whether it returned a value (Ok) or raised (Err).  The store/append
    // journals roll their entries back on abort (so a body re-run re-applies
    // them once), but a mutation outside those journals (a dict
    // `store_subscr_fn`, an `obj.attr = …` `store_attr_fn`, a `list.extend`,
    // a `del o[k]`, a name/global/deref store …) cannot be undone — delivering
    // the in-flight item and re-running the body would double it.  Flag it so
    // `fbw_foriter_inflight_take` refuses delivery (the legacy drop-on-abort
    // fallback) instead of doubling.
    // The user-frame signal: the residual's concrete execution entered a user
    // Python frame (the odometer advanced), so a value-returning
    // getter/dunder/module body may have mutated live heap outside the
    // journals — a body effect the Void/helper-tag discriminator misses.
    // `for_mutate`'s `seen.append` resolves its bound method at the C level (no
    // user frame), so its snapshot is unchanged and it still DELIVERS.
    //
    // The marking runs on BOTH arms.  A getter that mutates and THEN raises
    // (`for_prop_raise_abort`: `Obj.hits += 1; raise`, caught locally, walk
    // continues, later abort) takes the Err arm but still committed the
    // irreversible effect and still bumped the eval-loop entry odometer; if it
    // marked only on Ok, `fbw_foriter_inflight_take` would see no signal and
    // DELIVER, re-running the getter and DOUBLING `Obj.hits`.  The
    // `for_iter_next` consume itself is exempt (`provably_side_effect_free`
    // leaves both `body_effect_candidate` false and `user_frame_snapshot`
    // None), so a raising `__next__` never self-flags.
    //
    // The odometer bumps at frame ENTRY, so `entered_user_frame` cannot tell a
    // user frame that raised AFTER mutating from one that raised BEFORE: a
    // getter that raises before committing anything is also refused here.  That
    // is a harmless conservative DROP (the legacy bypass still runs the
    // iteration once), never a double — refusing a non-mutating raise costs
    // nothing but the never-double guarantee.
    let entered_user_frame = user_frame_snapshot
        .is_some_and(|before| pyre_interpreter::call::frame_entry_count() != before);
    if body_effect_candidate || entered_user_frame {
        if fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-foriter] body effect committed since consume (helper={helper:?} \
                 extraeffect={:?} result_type={:?} write_discriminator={body_effect_candidate} \
                 entered_user_frame={entered_user_frame})",
                ei.extraeffect,
                call_descr.result_type(),
            );
        }
        fbw_mark_foriter_body_effect_since_consume();
    }
    // gh#467: bump the concrete-heap-write odometer for any residual that is not
    // provably side-effect-free and either writes live heap (a Void / mutator-
    // tagged store the store/append journals do not cover) or entered a user
    // Python frame (a value-returning getter/dunder body that may have mutated).
    // The inline abort-forward-flush gate snapshots this at the CALL and refuses
    // the forward flush if a callee sub-walk moved it — re-executing the CALL
    // would double the effect.
    if !provably_side_effect_free
        && !is_idempotent_gc_barrier
        && (writes_live_heap
            || heap_write_odometer_before
                .is_some_and(|before| pyre_interpreter::call::frame_entry_count() != before))
    {
        fbw_bump_executed_effect();
    }
    match exec_result {
        Ok(result_i64) => {
            fbw_count_executed_residual(is_void, is_may_force);
            // #57 (Finding #1): the in-place int-list extend committed; journal
            // its pre-extend length so an aborting walk's rollback rewinds it and
            // the deliver re-applies it exactly once.  `result_i64 == lhs`
            // confirms the in-place mutation (list `__iadd__`/`__imul__` return
            // self) rather than a fresh-object op that merely shared the slot.
            if let Some((lhs, len_before)) = inplace_list_journal {
                if result_i64 as usize == lhs as usize {
                    fbw_append_journal_push(lhs, len_before);
                }
            }
            // A folded-decline `jit_list_append` fall-through (realloc-boundary
            // append) mutated `list` in place; journal its pre-append length so
            // the abort rollback rewinds it and the deliver re-applies exactly
            // once.  The append always mutates its receiver (void `0` result),
            // so no in-place `result == lhs` re-check is needed.
            if let Some((list, len_before)) = list_append_journal {
                fbw_append_journal_push(list, len_before);
            }
            // pyjitpl.py `result_box.value = result` analogue — stamp
            // the recorded OpRef with the executed concrete so downstream
            // `concrete_of_opref` / `box_value` consumers see the folded
            // value. An executed void helper has nothing to stamp.
            // `pyjitpl.py:2049-2068` checks forced virtual refs before
            // recording the selected CALL operation.  The assembler-call
            // walker uses `OpRef::NONE` while concrete-executing so
            // `vrefs_after_residual_call` can emit VIRTUAL_REF_FINISH before
            // CALL_ASSEMBLER, then stamps the newly recorded call itself.
            if !recorded.is_none() {
                match call_descr.result_type() {
                    majit_ir::Type::Int => {
                        ctx.trace_ctx
                            .set_opref_concrete(recorded, majit_ir::Value::Int(result_i64));
                    }
                    majit_ir::Type::Ref => {
                        ctx.trace_ctx.set_opref_concrete(
                            recorded,
                            majit_ir::Value::Ref(majit_ir::GcRef(result_i64 as usize)),
                        );
                    }
                    majit_ir::Type::Float => {
                        ctx.trace_ctx.set_opref_concrete(
                            recorded,
                            majit_ir::Value::Float(f64::from_bits(result_i64 as u64)),
                        );
                    }
                    majit_ir::Type::Void => {}
                }
            }
            // #57 Option C (capture): this residual is the FOR_ITER advance
            // (`for_iter_next`) — it just advanced the real shared heap
            // iterator (an irreversible side effect with no journal undo).
            // Stash the consumed item + the FOR_ITER body pc (the continue
            // arm's `py_pc + 1` fallthrough) so an aborting walk can DELIVER
            // the in-flight iteration to the live frame instead of dropping
            // it.  A null result is the exhaustion arm (no item, no body
            // runs) — nothing to deliver, leave the stash empty.
            if call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::ForIterNext
                && result_i64 != 0
            {
                // The body pc is the FOR_ITER continue-arm fallthrough — the
                // Python bytecode pc of the FOR_ITER opcode plus one (matching
                // `opcode_for_iter`'s `next_instr() == opcode_pc + 1`).
                //
                // Finding #2: derive it from the residual op's OWN JitCode pc
                // (`op_pc`) mapped to its containing Python opcode, NOT the
                // walk-ENTRY coordinate (`entry_py_pc + 1`).  The entry
                // coordinate equals the FOR_ITER fallthrough only when FOR_ITER
                // is the loop-header / walk-entry opcode; a second/nested
                // FOR_ITER reached deeper in a traced body has its own
                // `op_pc`, so the entry coordinate would point at the WRONG
                // body and deliver to the wrong pc.  The fallback (no outer
                // full-body sym / metadata) keeps the entry coordinate, which
                // is correct for the loop-header FOR_ITER.
                let body = fbw_foriter_body_from_op_pc(ctx.fbw_mode.snapshot_sym, op_pc)
                    .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
                fbw_foriter_inflight_capture(result_i64 as usize as pyre_object::PyObjectRef, body);
                // #73/#267: the item lands on the operand-stack TOS through the
                // codewriter's `pin!` slot binding (FOR_ITER lowering), not a
                // `setarrayitem_vable_r` push, and the residual result is
                // stamped via `set_opref_concrete`, not `write_ref_reg` — so
                // neither mirror chokepoint sees the item and `vstack_last_ref`
                // still holds whatever inner box the ForIterNext produced.  Seed
                // it with the item OpRef so the FOR_ITER boundary
                // (`ResultToTos`) places the item, not a stale box, on the new
                // TOS.  This runs for every `ForIterNext` residual once it
                // returns, placing the item OpRef on the new TOS for the
                // FOR_ITER `ResultToTos` boundary.
                ctx.vstack_last_ref = recorded;
                if fbw_debug_abort_enabled() {
                    let item = result_i64 as usize as pyre_object::PyObjectRef;
                    let intval = if unsafe { pyre_object::pyobject::is_int(item) } {
                        Some(unsafe { pyre_object::w_int_get_value(item) })
                    } else {
                        None
                    };
                    eprintln!(
                        "[fbw-foriter] capture item=0x{:x} intval={intval:?} foriter_pc={} body={body:?} \
                         store_journal_len={} append_journal_len={} unjournaled={}",
                        result_i64 as usize,
                        ctx.entry_py_pc(),
                        fbw_store_journal_len(),
                        FBW_APPEND_JOURNAL.with(|j| j.borrow().len()),
                        fbw_has_unjournaled_effect(),
                    );
                }
            }
        }
        Err(bh_exc) => {
            fbw_count_executed_residual(is_void, is_may_force);
            // `metainterp.execute_raised(exception, constant=False)`
            // analogue — seed the standing exception
            // state so downstream walker chain (`reraise/`,
            // `last_exc_value/>r`, `handle_possible_exception` guard
            // emission) sees a non-null `last_exc_value` and routes
            // through the GuardException path.
            //
            // `execute_residual_call` cleared `BH_LAST_EXC_VALUE` on read;
            // restore it so the eval-loop walker-skip path
            // (`eval.rs`) can detect the pending exception and
            // route into the bytecode-interpreter's exception handler
            // via `PyError::from_exc_object` — matching RPython's
            // metainterp framestack scan after a raising residual call
            // (`handle_possible_exception` + `finishframe_exception`).
            ctx.last_exc_value = Some(ctx.trace_ctx.const_ref(bh_exc));
            ctx.last_exc_value_concrete =
                ConcreteValue::Ref(bh_exc as usize as pyre_object::PyObjectRef);
            // `execute_raised(..., constant=False)`:
            // a residual exception has not had its class proven by a guard yet.
            ctx.fbw_mode.class_of_last_exc_is_const = false;
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(bh_exc));
            // `execute_raised` records the raise into `last_exc_value`
            // (above) only.  The shared `bh_*` residual helper also
            // published into the backend `_store_exception` cells
            // (`publish_residual_call_exception`), but RPython tracing never
            // touches them — they belong to compiled / blackhole execution.
            // Drain them so an aborted trace's snapshot-side raise cannot
            // leak into the live frame's re-run, where compiled
            // `GUARD_NO_EXCEPTION` would read it as a spurious pending
            // exception (the standing exception lives in `last_exc_value`
            // for the walk's `reraise` / `catch_exception` consumers).
            if let Some(cb) = crate::callbacks::try_get() {
                (cb.drain_backend_jit_exc)();
            }
        }
    }
    Ok(ResidualExecOutcome::Executed(exec_result))
}

/// `pyjitpl.py MetaInterp.direct_call_release_gil` port.
/// Sub-case of the forces-virtual-or-virtualizable branch
/// (`pyjitpl.py` `if effectinfo.is_call_release_gil()`): when the
/// descr's `call_release_gil_target` is a non-NULL `(realfuncaddr,
/// saveerr)` pair, the recorded trace op is `CALL_RELEASE_GIL_*`
/// with a re-shaped arglist:
///
/// ```text
///     realfuncaddr, saveerr = effectinfo.call_release_gil_target
///     funcbox = ConstInt(adr2int(realfuncaddr))
///     savebox = ConstInt(saveerr)
///     opnum   = rop.call_release_gil_for_descr(calldescr)
///     return self.history.record_nospec(
///         opnum, [savebox, funcbox] + argboxes[1:], ..., calldescr)
/// ```
///
/// `argboxes[0]` (the original funcbox) is replaced by the descr's real
/// target address, with `savebox` (`saveerr`) prepended.  The pyre-jit-
/// trace `allboxes` from [`build_allboxes`] starts with `funcptr` at
/// index 0 and the user-side arguments from index 1 onwards, matching
/// upstream's `argboxes[0] = funcbox` convention, so the slice rebuild
/// is `[savebox, funcbox_real] + allboxes[1..]`.
///
/// Mirror of `majit-metainterp/src/pyjitpl.rs
/// direct_call_release_gil` for the pyre-jit-trace dispatcher layer.
/// The two-frame-layer parity (majit `do_residual_call` and
/// pyre-jit-trace `dispatch_residual_call_*`) both implement the same
/// `pyjitpl.py` sub-case independently because the layers
/// receive different argument shapes.  `descr` is consumed (move) into
/// `record_op_with_descr` so the caller must `clone()` it before
/// calling if it needs the original after this returns.
///
/// Also emits the two guards the outer forces branch demands
/// (`pyjitpl.py GUARD_NOT_FORCED` unconditionally,
/// `pyjitpl.py GUARD_NO_EXCEPTION` when
/// `check_can_raise(False)` is true) — keeping guard emission inside
/// this helper means the dispatcher early-returns after a single call.
///
/// **`'r'` bank not supported.**  RPython
/// `resoperation.py call_release_gil_for_descr` has no
/// `CALL_RELEASE_GIL_R` arm (commented out as `# no such thing`),
/// and `:1462 is_call_release_gil` excludes `CALL_RELEASE_GIL_R`
/// from the predicate.  This helper panics on `dst_bank == 'r'` —
/// the closest behaviour to upstream's missing branch is fail-fast,
/// since silently routing to a non-existent OpCode would record an
/// IR op the optimizer / backend cannot consume.  Generic codewriter
/// `emit_residual_call` sites do not manufacture release-gil EIs via
/// `effect_info_for_call_flavor`; release-gil support is limited to
/// explicit via-target lowering that resolves the real call target
/// before materializing the final calldescr.  The panic is defensive
/// against a future producer that introduces a `'r'`-result release-gil
/// callee without first wiring an upstream `CALL_RELEASE_GIL_R` opcode.
///
/// `'i'` / `'f'` / `'v'` are the three result kinds upstream's
/// `call_release_gil_for_descr` accepts (`resoperation.py`).
/// All three are decoded here so the **opcode selection** matches
/// upstream's three-way result-kind table even though only
/// `dispatch_residual_call_iRd_kind` / `_iIRd_kind` currently route
/// `'i'` and `'r'` (the latter rejected per the panic above).
///
/// **Float / Void coverage is opcode-only, not full reuse.**  A
/// future float / void residual-call dispatcher would still have
/// to extend its own callsite to (a) widen `dst_bank` validation,
/// (b) add the corresponding writeback path to
/// `registers_f` / no-writeback, and (c) thread Float-typed
/// `argbox_types` through `build_allboxes` for the `'f'` arg-list
/// case.  This helper produces the right `OpCode::CallReleaseGil*`
/// once those landed; it does not by itself complete the dispatcher.
/// `pyjitpl.py do_residual_call` parity:
///
/// ```python
/// if effectinfo.oopspecindex == effectinfo.OS_NOT_IN_TRACE:
///     return self.metainterp.do_not_in_trace_call(allboxes, descr)
/// ```
///
/// Upstream's `do_not_in_trace_call` (pyjitpl.py) executes the
/// callee concretely and raises `SwitchToBlackhole(ABORT_ESCAPE,
/// raising_exception=True)` if it raised, otherwise returns `None` so
/// no IR op is recorded.
///
/// The pyre trace-walker has no concrete-execution callback for
/// jitcode-walked residual_call bytecodes yet — concrete execution
/// happens in the metainterp layer (`pyjitpl.rs
/// do_not_in_trace_call`) which dispatches `BC_CALL_*` not
/// `BC_RESIDUAL_CALL_*`. Therefore an `OS_NOT_IN_TRACE` callee that
/// reached this dispatcher cannot be safely treated as a regular
/// residual call: upstream records no IR for the normal case, and
/// aborts to blackhole only when the concrete call raises. Until that
/// concrete callback is threaded into `WalkContext`, the walker reports
/// a typed error instead of inventing either outcome.
///
/// `effect_info_for_call_flavor` (`flatten.rs` audit table) never
/// sets `oopspecindex`, so this branch is unreachable from production
/// today. A future producer that begins populating `oopspecindex`
/// should replace this guard with a real `do_not_in_trace_call`
/// callback returning `Ok(None)` on normal completion and
/// `SwitchToBlackhole(ABORT_ESCAPE, raising_exception=True)` only on
/// raise.
#[inline]
pub(crate) fn do_not_in_trace_call_result(
    ei: &majit_ir::EffectInfo,
    pc: usize,
) -> Result<Option<DispatchOutcome>, DispatchError> {
    if ei.oopspecindex == OopSpecIndex::NotInTrace {
        return Err(DispatchError::NotInTraceRequiresConcreteExecution { pc });
    }
    Ok(None)
}

/// IR-recording portion of `pyjitpl.py
/// vable_and_vrefs_before_residual_call`.  Records
/// `FORCE_TOKEN + SETFIELD_GC(vable_token_descr)` whenever the
/// jitdriver has a standard virtualizable registered for the current
/// frame.  RPython structure:
///
/// ```text
/// def vable_and_vrefs_before_residual_call(self):
///     self.vrefs_before_residual_call()                # heap mutation
///     vinfo = self.jitdriver_sd.virtualizable_info
///     if vinfo is not None:
///         virtualizable_box = self.virtualizable_boxes[-1]
///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
///         vinfo.tracing_before_residual_call(virtualizable) # heap mutation
///         force_token = self.history.record0(rop.FORCE_TOKEN, ...)  # IR
///         self.history.record2(rop.SETFIELD_GC, ..., descr=...)     # IR
/// ```
///
/// In pyre, the IR-recording role and the runtime heap-mutation role
/// are split.  This helper carries the IR portion only; the heap
/// halves of the vable token protocol
/// (`vinfo.tracing_before_residual_call(virtualizable)` /
/// `vinfo.tracing_after_residual_call(virtualizable)`) live with
/// the walk that executes the callee:
/// [`try_execute_residual_call_via_executor`] brackets the concrete
/// `execute_residual_call` with both halves of
/// `vable_and_vrefs_before_residual_call` /
/// `vable_after_residual_call` and surfaces
/// [`DispatchError::VableEscapedDuringResidualCall`] on a detected
/// force (pyjitpl.py ABORT_ESCAPE parity).
///
/// This helper records ONLY the IR portion here and never
/// touches the token; the heap-half token protocol is bracketed by
/// [`try_execute_residual_call_via_executor`], keeping the
/// `*token_ptr == 0` assertion in `tracing_before_residual_call`
/// intact.
///
/// The vref halves are bracketed by that same executor:
/// `TraceCtx::vrefs_before_residual_call` stamps every live vref before the
/// call and `vrefs_after_residual_call` turns any the callee forced into a
/// `VIRTUAL_REF_FINISH` + ConstPtr(NULL) box before the CALL op is recorded.
/// They iterate over the vrefs an inlined call's `enter` pushed
/// (`inline_call.rs::walker_ec_enter`), so a residual that hands an inlined
/// callee's frame to Python stops the trace from claiming it is still virtual.
pub(crate) fn walker_vable_and_vrefs_before_residual_call(ctx: &mut TraceCtx) {
    // pyjitpl.py: vinfo = self.jitdriver_sd.virtualizable_info;
    //                       if vinfo is not None:
    let Some(vable_ref) = ctx.standard_virtualizable_box() else {
        return;
    };
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    // pyjitpl.py: force_token + SETFIELD_GC vable_token_descr
    let force_token = ctx.force_token();
    ctx.vable_setfield_descr(vable_ref, force_token, info.token_field_descr());
}

/// The python pc `jit_pc` names in the inline callee's OWN code, or `None`
/// when this walk is not inside an inline callee.  A callee `jit_pc` has no
/// meaning in the outer jitcode's py_pc tables, so every consumer that writes
/// a pc onto the callee's frame has to go through the callee's own metadata.
fn inline_callee_py_pc<Sym: WalkSym>(ctx: &WalkContext<'_, '_, Sym>, jit_pc: usize) -> Option<u32> {
    let consts = ctx.inline_callee_consts?;
    let pjc = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index)?;
    Some(crate::py_coord::containing_py_pc_for_jitcode_pc(
        &pjc.metadata,
        jit_pc,
    ))
}

fn live_py_pc_from_snapshot<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    jit_pc: usize,
) -> Option<u32> {
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return None;
    }
    let sym = unsafe { &*sym_ptr };
    if sym.jitcode().is_null() {
        return None;
    }
    let jc = unsafe { &*sym.jitcode() };
    Some(crate::py_coord::containing_py_pc_for_jitcode_pc(
        &jc.payload.metadata,
        jit_pc,
    ))
}

/// Convenience wrapper for [`walker_vable_and_vrefs_before_residual_call`].
/// Kept as a thin pass-through so the dispatcher call sites stay
/// readable; collapses to direct `walker_*` once the dispatchers
/// inline.
fn maybe_record_inline_callee_last_instr<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    jit_pc: usize,
) {
    let Some(consts) = ctx.inline_callee_consts else {
        return;
    };
    let Some(pjc) = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index) else {
        return;
    };
    let frame_reg = pjc.metadata.portal_frame_reg as usize;
    let Some(&callee_frame) = ctx.registers_r.get(frame_reg) else {
        return;
    };
    if callee_frame == OpRef::NONE {
        return;
    }

    let callee_py_pc = crate::py_coord::containing_py_pc_for_jitcode_pc(&pjc.metadata, jit_pc);
    let last_instr = ctx.trace_ctx.const_int(callee_py_pc as i64);
    let last_instr_descr = crate::descr::pyframe_next_instr_descr();
    let last_instr_idx = last_instr_descr.index();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[callee_frame, last_instr],
        last_instr_descr,
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(callee_frame, last_instr_idx, last_instr);
}

pub(crate) fn disarm_folded_inline_callee_after_escape<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
) -> Result<(), DispatchError> {
    let Some(consts) = ctx.inline_callee_consts else {
        return Ok(());
    };
    let Some(pjc) = crate::state::pyjitcode_for_jitcode_index(consts.jitcode_index) else {
        return Ok(());
    };
    let Some(shadow) = ctx.callee_shadow.as_ref() else {
        return Ok(());
    };
    if shadow.fold_frame_reg == u16::MAX || shadow.frame_box == OpRef::NONE {
        return Ok(());
    }
    let frame_reg = pjc.metadata.portal_frame_reg as usize;
    let Some(&callee_frame) = ctx.registers_r.get(frame_reg) else {
        return Ok(());
    };
    if callee_frame != shadow.frame_box {
        return Ok(());
    }
    let Some(info) = ctx.trace_ctx.virtualizable_info() else {
        return Ok(());
    };
    let Some(fdescr) = info.array_field_descrs().first().cloned() else {
        return Ok(());
    };
    let Some(adescr) = info.array_descrs.first().cloned() else {
        return Ok(());
    };
    let mut slots: Vec<(i64, OpRef, Value)> = shadow
        .opref
        .iter()
        .filter_map(|(&slot, &value)| {
            (slot >= 0).then(|| {
                let concrete = shadow
                    .concrete
                    .get(&slot)
                    .filter(|entry| entry.frame_reg == shadow.fold_frame_reg)
                    .map(|entry| entry.value)
                    .or_else(|| ctx.trace_ctx.concrete_of_opref(value))
                    .unwrap_or(Value::Void);
                (slot, value, concrete)
            })
        })
        .collect();
    slots.sort_by_key(|(slot, _, _)| *slot);

    for (slot, value, concrete) in slots {
        let index = ctx.trace_ctx.const_int(slot);
        let guards_before = ctx.trace_ctx.num_guards();
        ctx.trace_ctx.vable_setarrayitem_indexed(
            pc,
            callee_frame,
            index,
            slot,
            fdescr.clone(),
            adescr.clone(),
            value,
            concrete,
            false,
        );
        walker_capture_inline_nonstandard_vable_guard(ctx, pc, guards_before)?;
    }
    if let Some(shadow) = ctx.callee_shadow.as_mut()
        && shadow.frame_box == callee_frame
    {
        shadow.fold_frame_reg = u16::MAX;
    }
    Ok(())
}

pub(crate) fn maybe_walker_vable_and_vrefs_before_residual_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    jit_pc: usize,
) {
    maybe_record_inline_callee_last_instr(ctx, jit_pc);
    walker_vable_and_vrefs_before_residual_call(ctx.trace_ctx);
}

/// Write a residual_call's recorded result OpRef into the dst register
/// chosen by `dst_bank`. Centralizes the result writeback so the
/// dispatchers can perform it BEFORE recording the
/// `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION` guards, matching
/// `pyjitpl.py _opimpl_residual_call*` ordering: the result
/// must populate `registers_*[dst]` before
/// `handle_possible_exception()` captures the guard's `fail_args`,
/// otherwise a raising call surfaces NONE in the slot the resume
/// snapshot reads.
pub(crate) fn write_residual_call_result_to_dst<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    dst: usize,
    dst_bank: char,
    result: OpRef,
) -> Result<(), DispatchError> {
    // concrete_of_opref shadow write: route the shadow write through `concrete_of_opref`
    // so a CallPure* descr whose argboxes are all constant (do_residual_call
    // path that lands a constant result via the executor.execute_varargs
    // stamp) propagates concrete to the dst slot.  Falls back to Null when
    // the result has no recorded concrete (matches the pre-#75.F shape for
    // every non-elidable call).
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'r' => {
            write_ref_reg(ctx, pc, dst, result, concrete_for_shadow)?;
        }
        'i' => {
            write_int_reg(ctx, pc, dst, result, concrete_for_shadow)?;
        }
        'f' => {
            let len = ctx.registers_f.len();
            let slot = ctx
                .registers_f
                .get_mut(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            *slot = result;
        }
        // Void variants (`pyjitpl.py opimpl_residual_call_*_v`):
        // the operand layout has no `>X` dst byte and no register slot to
        // populate. The cached / recorded OpRef is dropped on the floor
        // upstream too (the `_call*` body discards the call result for
        // void).
        'v' => {}
        _ => unreachable!("dst_bank validated by caller"),
    }
    Ok(())
}

pub(crate) fn residual_call_helper_kind_in_body(
    body_code: &[u8],
    d: &DecodedOp,
    callee_descr_refs: &[DescrRef],
) -> Option<majit_ir::PyreHelperKind> {
    let descr_index = residual_call_descr_index_in_body(body_code, d)?;
    callee_descr_refs
        .get(descr_index)
        .and_then(|descr| descr.as_call_descr())
        .map(|cd| cd.get_extra_info().pyre_helper)
}

/// Return the per-function descriptor-pool index carried by a residual call
/// in a callee jitcode body.  The layouts mirror the residual dispatchers:
/// the descriptor follows the one or two variable-length argument lists.
pub(crate) fn residual_call_descr_index_in_body(body_code: &[u8], d: &DecodedOp) -> Option<usize> {
    let descr_offset = match d.key {
        "residual_call_r_r/iRd>r" | "residual_call_r_i/iRd>i" | "residual_call_r_v/iRd" => {
            let r_len_pc = d.pc + 2;
            let r_len = *body_code.get(r_len_pc)? as usize;
            1 + 1 + r_len
        }
        "residual_call_ir_r/iIRd>r" | "residual_call_ir_i/iIRd>i" | "residual_call_ir_v/iIRd" => {
            let i_len_pc = d.pc + 2;
            let i_width = 1 + *body_code.get(i_len_pc)? as usize;
            let r_len_pc = d.pc + 1 + 1 + i_width;
            let r_width = 1 + *body_code.get(r_len_pc)? as usize;
            1 + i_width + r_width
        }
        _ => return None,
    };
    Some(decode_descr_index(body_code, d, descr_offset))
}

/// Which specialization table arm a body `BINARY_OP` residual was admitted
/// under, hence what its result is proven to be.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SpecializedBinop {
    /// `Add` / `Subtract` / `Multiply` (+ in-place): exact-numeric operands,
    /// exact-numeric result.
    Numeric,
    /// `And` / `Or` / `Xor` (+ in-place): exact-plain-int operands,
    /// exact-plain-int result.
    PlainInt,
    /// One of the six ordinary comparisons over exact numeric operands:
    /// exact-bool result.
    Compare,
}

/// A `BINARY_OP` has the generic residual shape in a per-function jitcode, but
/// the walker replaces a statically tagged plain arithmetic op with a native
/// one before the generic residual executor (and its nested-residual decline)
/// is reached, when every incoming callee argument is an exact int or float.
/// Non-numeric operands stay an impure residual, so admitting them here would
/// trigger the nested-residual 6421 abort storm.
///
/// The accepted set is every tag a specialization table lowers with no runtime
/// decline path left, so nothing survives as a residual.  Both tables key the
/// in-place tag to the SAME arm as its plain form, so the two forms are
/// admitted together:
///
/// - `Add` / `Subtract` / `Multiply` (+ in-place) — `IntAddOvf` / `IntSubOvf` /
///   `IntMulOvf` in `try_walker_specialize_binary_op_int`, `FloatAdd` /
///   `FloatSub` / `FloatMul` in `try_walker_specialize_binary_op_float`.  In
///   both tables `needs_concrete_check` is false, so either argument width
///   lowers unconditionally.
/// - `And` / `Or` / `Xor` (+ in-place) — `IntAnd` / `IntOr` / `IntXor`, also
///   unconditional, but *int-only*: the float table falls through to
///   `_ => return Ok(None)` for them.  Hence the separate
///   [`SpecializedBinop::PlainInt`] arm, which additionally demands both
///   operands be proven plain ints.
/// - `FloorDivide` / `Remainder` (+ in-place) — `IntFloorDiv` / `IntMod`,
///   int-only for the same reason (neither has a `FLOAT_*` opcode).  These two
///   are the one accepted pair whose lowering *can* decline — on a zero divisor
///   or on `i64::MIN` by `-1` — but a surviving residual is still replay-safe
///   on its own merits: `int.__floordiv__` / `int.__mod__` read two immutable
///   boxes and either allocate a fresh result or raise `ZeroDivisionError`,
///   which commits nothing a replay would double.  The `plain_int` proof is
///   what rules out a user `__mod__`.  `i % k` in an `if` is the common shape
///   that would otherwise residualize the whole callee
///   (`bench/synth/gc_bug_bridge_flavor_traceback_names`).
///
/// Every other tag is excluded because its lowering can still decline and
/// leave a residual that is NOT replay-safe on its own:
///
/// - `TrueDivide` (+ in-place) — float-table only, and it declines a zero
///   divisor so the raising `descr_truediv` stays recorded.
/// - `Lshift` (+ in-place) — the int table declines it outright (the reused
///   trace would bake a count the x86 `SHL` masks mod 64, and the guarded form
///   breaks the cranelift bridge).
/// - `Rshift` (+ in-place) — declines a negative or `>= LONG_BIT` count rather
///   than baking intobject.py's fold-to-`0`/`-1`.
/// - `Power` (+ in-place) — the int table has no arm; the float table inlines
///   `_pow` but keeps a cold-path residual for nan/inf/negative-base operands.
/// - `Subscr`, `MatrixMultiply` (+ in-place) — no arm in either table.
///
/// The two provenance sets describe the actual operands of each binop.  This
/// admits `def f(self, x): return x + 1` when only `x` is numeric, while still
/// rejecting `self + x` and global numeric subclasses with user dunders.
/// `None` for every op outside the accepted set.
pub(crate) fn residual_call_specialized_plain_numeric_binop(
    body_code: &[u8],
    numeric_ref_regs: &[bool; u8::MAX as usize + 1],
    plain_int_ref_regs: &[bool; u8::MAX as usize + 1],
    d: &DecodedOp,
    num_regs_i: usize,
    constants_i: &[i64],
    callee_descr_refs: &[DescrRef],
) -> Option<SpecializedBinop> {
    let helper = residual_call_helper_kind_in_body(body_code, d, callee_descr_refs);
    if !matches!(
        d.key,
        "residual_call_ir_r/iIRd>r" | "residual_call_ir_i/iIRd>i" | "residual_call_ir_v/iIRd"
    ) || !matches!(
        helper,
        Some(majit_ir::PyreHelperKind::BinaryOp | majit_ir::PyreHelperKind::CompareOp)
    ) {
        return None;
    }
    // `iIR`: the R-list follows the I-list.  `walker_int_specialization_operands`
    // / `walker_float_specialization_operands` read exactly `r_args[0]` (lhs)
    // and `r_args[1]` (rhs) and decline any other arity, so demand the same
    // shape here and require both operands to be proven.
    let Some(&i_len) = body_code.get(d.pc + 2) else {
        return None;
    };
    let r_len_pc = d.pc + 1 + 1 + 1 + i_len as usize;
    if body_code.get(r_len_pc) != Some(&2) {
        return None;
    }
    let (Some(&lhs_reg), Some(&rhs_reg)) =
        (body_code.get(r_len_pc + 1), body_code.get(r_len_pc + 2))
    else {
        return None;
    };
    if !numeric_ref_regs[lhs_reg as usize] || !numeric_ref_regs[rhs_reg as usize] {
        return None;
    }
    // The first I-list item is the BINARY_OP tag.  It must be in the callee's
    // immutable constants window; a runtime tag could select an operation
    // outside the accepted set.
    if i_len == 0 {
        return None;
    }
    let Some(&tag_reg) = body_code.get(d.pc + 3) else {
        return None;
    };
    let Some(&tag) = (tag_reg as usize)
        .checked_sub(num_regs_i)
        .and_then(|constant_index| constants_i.get(constant_index))
    else {
        return None;
    };
    // Both compare tables map all six `ComparisonOperator`s unconditionally
    // (`IntLt/Le/Gt/Ge/Eq/Ne` in `try_walker_specialize_compare_op_int`,
    // `FloatLt/Le/Gt/Ge/Eq/Ne` in `try_walker_specialize_compare_op_float`), so
    // no tag leaves the residual in place and there is no int-only carve-out
    // like the bitwise binops need.  `CHECK_EXC_MATCH` reuses the `CompareOp`
    // shape with `ISINSTANCE_OP` (tag 10), which is not one of the six and so
    // stays excluded.
    if helper == Some(majit_ir::PyreHelperKind::CompareOp) {
        return pyre_interpreter::runtime_ops::compare_op_from_tag(tag)
            .is_some()
            .then_some(SpecializedBinop::Compare);
    }
    use pyre_interpreter::bytecode::BinaryOperator;
    match pyre_interpreter::runtime_ops::binary_op_from_tag(tag) {
        Some(
            BinaryOperator::Add
            | BinaryOperator::Subtract
            | BinaryOperator::Multiply
            | BinaryOperator::InplaceAdd
            | BinaryOperator::InplaceSubtract
            | BinaryOperator::InplaceMultiply,
        ) => Some(SpecializedBinop::Numeric),
        Some(
            BinaryOperator::And
            | BinaryOperator::Or
            | BinaryOperator::Xor
            | BinaryOperator::InplaceAnd
            | BinaryOperator::InplaceOr
            | BinaryOperator::InplaceXor
            | BinaryOperator::FloorDivide
            | BinaryOperator::Remainder
            | BinaryOperator::InplaceFloorDivide
            | BinaryOperator::InplaceRemainder,
        ) => (plain_int_ref_regs[lhs_reg as usize] && plain_int_ref_regs[rhs_reg as usize])
            .then_some(SpecializedBinop::PlainInt),
        _ => None,
    }
}

/// Is this body op the `CHECK_EXC_MATCH` residual — `compare_fn(exc,
/// match_type, ISINSTANCE_OP)`?
///
/// Unlike the arithmetic tags, this one needs no operand proof.  The helper
/// validates the match target and then walks the exception class MRO
/// (`validate_check_exc_match_class` + `check_exc_match_against` →
/// `exception_match`), reading `is_tuple` / `is_type` / the MRO array and
/// nothing else: it reaches no user code for any operand, mutates nothing, and
/// returns one of the two immortal `bool` singletons.  Its single failure mode
/// — `TypeError` for a target that is not an exception class — allocates a
/// fresh exception exactly the way the `CanRaise`-tagged members of the
/// `replay_safe_read` set can, so a replay commits nothing new.
///
/// `iIRd>r`: the tag is the first I-list entry, and it must live in the
/// callee's immutable constant window — a runtime tag could select one of the
/// six ordinary comparisons, which do dispatch to user `__eq__`.
pub(crate) fn residual_call_is_exception_match(
    body_code: &[u8],
    d: &DecodedOp,
    num_regs_i: usize,
    constants_i: &[i64],
    callee_descr_refs: &[DescrRef],
) -> bool {
    if d.key != "residual_call_ir_r/iIRd>r"
        || residual_call_helper_kind_in_body(body_code, d, callee_descr_refs)
            != Some(majit_ir::PyreHelperKind::CompareOp)
    {
        return false;
    }
    if body_code.get(d.pc + 2).is_none_or(|i_len| *i_len == 0) {
        return false;
    }
    let Some(&tag_reg) = body_code.get(d.pc + 3) else {
        return false;
    };
    (tag_reg as usize)
        .checked_sub(num_regs_i)
        .and_then(|constant_index| constants_i.get(constant_index))
        .is_some_and(|tag| *tag == pyre_interpreter::runtime_ops::ISINSTANCE_OP_TAG)
}

/// Is this body op a `TO_BOOL` / `POP_JUMP_IF_*` truth residual whose single
/// Ref operand is a proven immutable builtin — an exact numeric, or the `bool`
/// an accepted `COMPARE_OP` in the same body produced?
///
/// Such an operand's `__bool__` is `int`'s or `bool`'s, so the call reads a
/// field and returns an int: it commits nothing a replay could double.  That is
/// the same argument the `replay_safe_read` set is built on, and it holds
/// whether or not the walk-time folds
/// (`bool_box_truth_lookup`, `try_walker_specialize_truth_int`,
/// `try_walker_specialize_truth_bool`) erase the residual.
///
/// `iRd>i`: the funcbox int operand, then the R-list (length byte, then one
/// register per entry), then the descr.  Only the one-operand arity is the
/// truth shape.
pub(crate) fn residual_call_is_proven_truth(
    body_code: &[u8],
    numeric_ref_regs: &[bool; u8::MAX as usize + 1],
    bool_ref_regs: &[bool; u8::MAX as usize + 1],
    d: &DecodedOp,
    callee_descr_refs: &[DescrRef],
) -> bool {
    if d.key != "residual_call_r_i/iRd>i"
        || residual_call_helper_kind_in_body(body_code, d, callee_descr_refs)
            != Some(majit_ir::PyreHelperKind::Truth)
    {
        return false;
    }
    if body_code.get(d.pc + 2) != Some(&1) {
        return false;
    }
    let Some(&arg_reg) = body_code.get(d.pc + 3) else {
        return false;
    };
    numeric_ref_regs[arg_reg as usize] || bool_ref_regs[arg_reg as usize]
}
/// `pyjitpl.py:1094-1118 opimpl_jit_force_quasi_immutable` for the module
/// namespace's `version?` (`celldict.py:34 _immutable_fields_ = ["version?"]`),
/// asked ahead of an opaque STORE_NAME / STORE_GLOBAL / DELETE_NAME /
/// DELETE_GLOBAL residual.
///
/// ```text
///  mutatebox = self.execute_with_descr(rop.GETFIELD_GC_R, mutatefielddescr, box)
///  if mutatebox.nonnull():
///      do_force_quasi_immutable(cpu, box.getref_base(), mutatefielddescr)
///      raise SwitchToBlackhole(Counters.ABORT_FORCE_QUASIIMMUT)
///  self.metainterp.generate_guard(rop.GUARD_ISNULL, mutatebox, resumepc=orgpc)
/// ```
///
/// Upstream meets the rtyper's `jit_force_quasi_immutable` inside the traced
/// write (`rclass.py:715-718`) and abandons the attempt cheaply, which is why
/// PyPy reports thousands of `abort: force quasi-immut` on this program shape
/// and still compiles its loops. Pyre's write runs inside a frontend helper the
/// walker never looks into, so the walker asks the same question here instead of
/// meeting the operation; without it the trace completes carrying a `version`
/// constant that is already stale, and the optimizer's revalidation then
/// discards the loop *and* every interpreter entry bridge.
///
/// `is_installed()` is `mutatebox.nonnull()`; the bump predicate is
/// [`pyre_object::celldict::store_would_bump_version`], the side-effect-free
/// twin of `write_cell`, because only the write that replaces the stored
/// pointer reaches `mutated()` (`celldict.py:80-90`) — an in-place cell write
/// leaves `version` alone and a hot module-scope loop must keep its trace.
///
/// Deliberately NO `GUARD_ISNULL` arm on the not-installed path. Upstream's
/// guard licenses the traced *inline* setfield that bypasses the invalidation
/// function (`pyjitpl.py:1095-1102`); pyre's write stays inside the residual,
/// which runs `notify_version_watchers` itself at runtime, so there is no
/// bypass to license.
///
/// Fires BEFORE the residual executes, so the write is still entirely ahead of
/// the walk — that is what lets the abort resume the interpreter at this opcode
/// and have the write happen exactly once.  Everything the walk applied EARLIER
/// stays applied: [`DispatchError::ForceQuasiImmutable`] carries the abort to
/// the flush leg that commits the journal instead of replaying the region.
fn try_walker_force_quasi_immut_namespace_write<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    helper: majit_ir::PyreHelperKind,
    r_args: &[OpRef],
) -> Option<()> {
    use majit_ir::PyreHelperKind as K;
    let is_store = matches!(helper, K::StoreName | K::StoreGlobal);
    if !is_store && !matches!(helper, K::DeleteName | K::DeleteGlobal) {
        return None;
    }
    let (&frame_opref, &name_opref) = (r_args.first()?, r_args.get(1)?);
    let (
        Some(majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr))),
        Some(majit_ir::Value::Ref(majit_ir::GcRef(w_name_ptr))),
    ) = (
        ctx.trace_ctx.box_value(frame_opref),
        ctx.trace_ctx.box_value(name_opref),
    )
    else {
        return None;
    };
    if frame_ptr == 0 || w_name_ptr == 0 {
        return None;
    }
    let frame = unsafe { &*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    let w_globals = frame.get_w_globals();
    if w_globals.is_null() {
        return None;
    }
    // A `*_NAME` opcode writes `get_or_create_w_locals`; only a module frame —
    // where `w_locals` aliases `w_globals` — touches the namespace the folds
    // pin.  An absent `w_locals` is a fresh throwaway mapping, not globals, so
    // it declines rather than forcing for a write that never reaches here.
    if matches!(helper, K::StoreName | K::DeleteName) {
        let w_locals = frame.get_w_locals();
        if !std::ptr::eq(w_locals, w_globals) {
            return None;
        }
    }
    // A plain `W_DictObject` for globals (`exec(src, {})`,
    // `FunctionType(code, {})`) has no `version?` field at all, so
    // `hook_setfield` (rclass.py:714-718) emits no `jit_force_quasi_immutable`
    // for its write and there is nothing to abandon the trace for.
    let strategy =
        unsafe { pyre_object::dictmultiobject::w_module_dict_strategy_or_null(w_globals) };
    // `mutatebox.nonnull()` — nothing is watching, so there is nothing to
    // abandon the trace for. The write still runs its own invalidation.
    if !unsafe {
        pyre_object::dictmultiobject::module_dict_strategy_version_qmut_installed(strategy)
    } {
        return None;
    }
    let name = unsafe {
        pyre_object::unicodeobject::w_str_get_value(w_name_ptr as pyre_object::PyObjectRef)
    };
    let slot = crate::state::module_dict_cell_slot_direct(w_globals, name);
    let bumps = if is_store {
        let &value_opref = r_args.get(2)?;
        let Some(majit_ir::Value::Ref(majit_ir::GcRef(value_ptr))) =
            ctx.trace_ctx.box_value(value_opref)
        else {
            return None;
        };
        if value_ptr == 0 {
            return None;
        }
        let cell = slot.and_then(|s| crate::state::module_dict_cell_value_direct(w_globals, s));
        unsafe {
            pyre_object::celldict::store_would_bump_version(
                cell,
                value_ptr as pyre_object::PyObjectRef,
            )
        }
    } else {
        // `celldict.py:106-126 delitem` reaches `mutated()` only after a
        // successful removal — `delitem_str` returns early on a missing key.
        slot.is_some()
    };
    if !bumps {
        return None;
    }
    // pyjitpl.py:1113-1115: the tracer performs the invalidation itself and
    // then abandons the attempt. Idempotent (`quasiimmut.py:47-48`), so the
    // interpreter re-running the opcode forces nothing a second time.
    unsafe { pyre_object::dictmultiobject::module_dict_strategy_force_version_qmut(strategy) };
    // Offer the flush leg the operand stack this opcode began with.  Same
    // preconditions as the escape latch: a sub-walk's mirror describes the
    // CALLEE frame, and a bridge walk's abort path never reaches the epilogue
    // that would adopt the flush.
    if ctx.vstack_valid && !ctx.fbw_mode.inline_subwalk && !ctx.trace_ctx.is_bridge_trace {
        fbw_qmut_abort_stack_latch(ctx.vstack_cur_pypc as usize, ctx.vstack_boxes.clone());
    }
    Some(())
}

/// Opt-in until the `ForceQuasiImmutable` flush leg in `trace.rs` re-delivers
/// an in-flight FOR_ITER item on its decline paths. `obj.attr = v` inside a
/// loop is the most common statement in the corpus, so turning this on by
/// default would make that dropped-iteration defect reachable.
fn mapdict_qmut_force_enabled() -> bool {
    std::env::var_os("PYRE_QMUT_MAPDICT_FORCE").is_some()
}

fn walker_pin_plain_ever_mutated<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    plain: *const pyre_interpreter::objspace::std::mapdict::PlainAttribute,
) -> Result<(), DispatchError> {
    if plain.is_null() {
        return Ok(());
    }
    let owner = ctx.trace_ctx.const_ref(plain as i64);
    crate::state::record_quasiimmut_field(
        ctx.trace_ctx,
        owner,
        crate::descr::plain_attribute_ever_mutated_descr(),
    );
    walker_flush_guard_not_invalidated(ctx, op_pc)
}

/// Tracer-side pyjitpl.py:1105-1120
/// `opimpl_jit_force_quasi_immutable` for mapdict writes hidden in residuals.
/// The target predicate is side-effect-free; record comes before the installed
/// test because pyjitpl.py:1074-1085 `opimpl_record_quasiimmut_field` creates
/// the hidden instance when it is null. Recording also preserves
/// `AbstractAttribute.write`'s `if not attr.ever_mutated` read
/// (mapdict.py:72).
///
/// Deliberately no `GUARD_ISNULL` arm after the installed test. Upstream's
/// guard (pyjitpl.py:1117-1118) licenses a traced INLINE setfield that bypasses
/// invalidation; pyre's write stays inside the residual and performs its own
/// notify. This is sound while compiled traces emit no store to these four
/// fields: `jit_mapdict_boxed_write`, `jit_mapdict_unboxed_write_raw`, and
/// `jit_mapdict_unboxed_write_f` reach only `write_boxed_storage` /
/// `write_unboxed_storage_raw`, while the add-transition inline emits only map
/// and storage stores.
fn try_walker_force_quasi_immut_mapdict_write<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    helper: majit_ir::PyreHelperKind,
    i_args: &[OpRef],
    r_args: &[OpRef],
) -> Option<()> {
    use majit_ir::PyreHelperKind as K;
    let is_store = matches!(helper, K::StoreAttr);
    if !is_store && !matches!(helper, K::DeleteAttr) {
        return None;
    }
    // Force only where the abort can hand the flush leg an operand-stack mirror
    // it can adopt — the same predicate the latch below uses. Without it the
    // abort falls to the legacy replay-from-entry, which re-runs every residual
    // the walk already executed concretely; `pickle_terminal_raise_resume`
    // reaches that through a `self.x = v` inside an inlined callee and
    // desynchronises the unpickler's read position
    // (`end=ForceQuasiImmutable committed=false effects=4`).
    //
    // Declining is sound, not a hole: without the abort the trace keeps
    // recording and the optimizer's revalidation (heap.py:798-804
    // `is_still_valid_for`) discards any loop whose recorded `?` value moved,
    // so the only cost is a wasted trace attempt.
    if !ctx.vstack_valid || ctx.fbw_mode.inline_subwalk || ctx.trace_ctx.is_bridge_trace {
        return None;
    }

    let &obj_opref = r_args.first()?;
    let &code_opref = r_args.get(if is_store { 2 } else { 1 })?;
    let &name_opref = i_args.first()?;
    let (
        Some(majit_ir::Value::Ref(majit_ir::GcRef(w_obj_ptr))),
        Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
        Some(majit_ir::Value::Int(name_idx)),
    ) = (
        ctx.trace_ctx.box_value(obj_opref),
        ctx.trace_ctx.box_value(code_opref),
        ctx.trace_ctx.box_value(name_opref),
    )
    else {
        return None;
    };
    if w_obj_ptr == 0 || w_code_ptr == 0 {
        return None;
    }
    let code = unsafe {
        let code_ptr = pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef);
        if code_ptr.is_null() {
            return None;
        }
        &*(code_ptr as *const pyre_interpreter::CodeObject)
    };
    let name = pyre_interpreter::pyframe::load_name_from_code(code, name_idx as usize)?;
    let (attrkind, is_slot) = unsafe {
        pyre_interpreter::objspace::std::mapdict::classify_mapdict_write_attr(
            w_obj_ptr as pyre_object::PyObjectRef,
            name,
        )
    }?;
    if attrkind == pyre_interpreter::objspace::std::mapdict::INVALID {
        return None;
    }
    let attrname = rustpython_wtf8::Wtf8::new(if is_slot { "slot" } else { name });
    let target = if is_store {
        let &value_opref = r_args.get(1)?;
        let Some(majit_ir::Value::Ref(majit_ir::GcRef(w_value_ptr))) =
            ctx.trace_ctx.box_value(value_opref)
        else {
            return None;
        };
        if w_value_ptr == 0 {
            return None;
        }
        unsafe {
            pyre_interpreter::objspace::std::mapdict::setattr_would_force_quasi_immut(
                w_obj_ptr as pyre_object::PyObjectRef,
                attrname,
                attrkind,
                w_value_ptr as pyre_object::PyObjectRef,
            )
        }
    } else {
        unsafe {
            pyre_interpreter::objspace::std::mapdict::delattr_would_force_quasi_immut(
                w_obj_ptr as pyre_object::PyObjectRef,
                attrname,
                attrkind,
            )
        }
    }?;

    use pyre_interpreter::objspace::std::mapdict::MapdictQmutTarget as T;
    // `op_pc` is the walker's jit pc — the resume coordinate
    // `walker_capture_snapshot_for_last_guard` stamps on the drained
    // `GUARD_NOT_INVALIDATED`, the same one every other fold guard uses. It is
    // not the Python pc the operand-stack latch below carries.
    match target {
        T::PlainEverMutated(plain) => walker_pin_plain_ever_mutated(ctx, op_pc, plain).ok()?,
        T::TerminatorAllowUnboxing(term) => {
            crate::jitcode_dispatch::walker_pin_terminator_allow_unboxing(ctx, op_pc, term).ok()?
        }
        T::HolderTyp(holder) => {
            crate::jitcode_dispatch::walker_pin_holder_typ(ctx, op_pc, holder).ok()?
        }
        T::HolderAttr(holder) => {
            crate::jitcode_dispatch::walker_pin_holder_attr(ctx, op_pc, holder).ok()?
        }
    }
    let installed = unsafe {
        match target {
            T::PlainEverMutated(plain) => pyre_interpreter::objspace::std::mapdict::plain_attribute_ever_mutated_qmut_installed(plain),
            T::TerminatorAllowUnboxing(term) => pyre_interpreter::objspace::std::mapdict::terminator_allow_unboxing_qmut_installed(term),
            T::HolderTyp(holder) => pyre_interpreter::objspace::std::mapdict::holder_typ_qmut_installed(holder),
            T::HolderAttr(holder) => pyre_interpreter::objspace::std::mapdict::holder_attr_qmut_installed(holder),
        }
    };
    if !installed {
        return None;
    }
    unsafe {
        match target {
            T::PlainEverMutated(plain) => {
                pyre_interpreter::objspace::std::mapdict::plain_attribute_force_ever_mutated_qmut(
                    plain,
                )
            }
            T::TerminatorAllowUnboxing(term) => {
                pyre_interpreter::objspace::std::mapdict::terminator_force_allow_unboxing_qmut(term)
            }
            T::HolderTyp(holder) => {
                pyre_interpreter::objspace::std::mapdict::holder_force_typ_qmut(holder)
            }
            T::HolderAttr(holder) => {
                pyre_interpreter::objspace::std::mapdict::holder_force_attr_qmut(holder)
            }
        }
    }
    // Offer the flush leg the operand stack this opcode began with. Its three
    // preconditions were established at the top of this function, which is the
    // last point where declining to force is still free.
    fbw_qmut_abort_stack_latch(ctx.vstack_cur_pypc as usize, ctx.vstack_boxes.clone());
    Some(())
}

pub(crate) fn dispatch_residual_call_iRd_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // execute_varargs (pyjitpl.py) opens every residual call
    // with metainterp.clear_exception(), so a caught exception's
    // last_exc_value never survives past the next call — the
    // opimpl_catch_exception assert (pyjitpl.py) relies on it.
    // Clear at the arm entry so declined/folded paths uphold the same
    // invariant as the concrete-execution success arm.
    ctx.last_exc_value = None;
    ctx.last_exc_value_concrete = ConcreteValue::Null;
    let funcptr = read_int_reg(code, op, 0, ctx)?;
    let (mut r_args, arg_width) = read_ref_var_list(code, op, 1, ctx)?;
    // #62: env-gated recognition probe (no-op unless PYRE_DIAG_INLINE_RECOG
    // set; full-body-walk authoritative path only).  First slice of the
    // call-inlining feature — confirms callable->JitCode recognition before
    // sub-walk wiring.
    if ctx.is_authoritative_executor && std::env::var("PYRE_DIAG_INLINE_RECOG").is_ok() {
        let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
        diagnose_inline_recognition(&arg_concretes, op.pc);
    }
    let descr_offset = 1 + arg_width;
    let descr_index = decode_descr_index(code, op, descr_offset);
    let descr = read_descr(code, op, descr_offset, ctx)?;
    // RPython `do_residual_or_indirect_call` always receives a
    // CallDescr (pyjitpl.py). Codewriter emits only CallDescrs
    // for residual_call slots; surface a typed error if a test fixture
    // (or future deviation) routes a non-CallDescr here.
    let call_descr = descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index,
        })?;
    let descr_key = descr.index();
    // Void shape `_r_v/iRd` (`pyjitpl.py opimpl_residual_call_r_v =
    // _opimpl_residual_call1`) has no trailing `>X` dst byte. The
    // result OpRef is discarded by `write_residual_call_result_to_dst`'s
    // `'v'` arm, so `dst` is irrelevant on the void path; reading the
    // byte would walk past the operand list.
    let dst = if dst_bank == 'v' {
        0
    } else {
        code[op.pc + 1 + descr_offset + 2] as usize
    };

    let ei = call_descr.get_extra_info();
    repair_carrier_call_ref_args(ctx, op.pc, ei.pyre_helper, &mut r_args);
    // Residual-call entry mirrors `execute_varargs`: even when the walker
    // folds the call or leaves it recorded symbolically, stale handled
    // exceptions from earlier opcodes are not visible to the following
    // linear `catch_exception/L`.
    clear_walk_exception(ctx);

    // BuiltinCode.func is an indirect PBC target exactly like RPython's
    // gateway wrappers.  Enter its generated JitCode before considering the
    // user-function-only full-body walk below.
    if let Some(inlined) =
        try_walker_inline_builtin_call(ctx, op, code, 1, &r_args, ei.pyre_helper, dst_bank, dst)?
    {
        return Ok(inlined);
    }

    // #62 slice (3c): attempt full-body-walk inline of a user-function call
    // unconditionally. Eligible exact-positional closure-free
    // calls sub-walk the callee body in place of the residual; ineligible
    // calls (including every non-`call_fn` helper, gated on `pyre_helper`)
    // fall through with no IR emitted.
    if let Some(inlined) = try_walker_inline_user_call(
        ctx,
        op,
        code,
        1,
        funcptr,
        &r_args,
        call_descr,
        ei.pyre_helper,
        dst_bank,
        dst,
    )? {
        return Ok(inlined);
    }

    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
    {
        if let Some(inlined) = try_walker_inline_exception_string_override(
            ctx, op, code, funcptr, &r_args, call_descr, dst,
        )? {
            return Ok(inlined);
        }
        // hash(x) over a user instance: inline __hash__ plus normalization.
        if let Some(inlined) =
            try_walker_inline_hash_builtin(ctx, op, code, funcptr, &r_args, call_descr, dst)?
        {
            return Ok(inlined);
        }
        // A class is not a `Function`, so the user-call route above declined it.
        if let Some(inlined) =
            try_walker_inline_type_call(ctx, op, code, funcptr, &r_args, call_descr, dst_bank, dst)?
        {
            return Ok(inlined);
        }
    }

    // #62: a self-recursive call the inline path declined (e.g. the
    // branchy `fib`) gets a direct `CALL_ASSEMBLER` to its own loop token
    // instead of the heavyweight func-entry
    // residency residual. Independent of inline eligibility.
    if let Some(ca) = try_walker_call_assembler_self_recursive(
        ctx,
        op,
        code,
        funcptr,
        &r_args,
        call_descr,
        ei.pyre_helper,
        dst_bank,
        dst,
    )? {
        return Ok(ca);
    }

    // `_r_*` shape: argboxes = R-list only; argbox_types = [Ref; n].
    let argbox_types: Vec<Type> = vec![Type::Ref; r_args.len()];
    let mut allboxes = build_allboxes(funcptr, &r_args, &argbox_types, call_descr.arg_types());
    replace_movable_load_global_namespace_with_frame_globals(ctx, ei, &mut allboxes);
    if let Err(e) = ensure_residual_call_args_bound(&allboxes, op.pc) {
        if fbw_debug_abort_enabled() {
            let len_pc = op.pc + 1 + 1;
            let n = code[len_pc] as usize;
            let regs: Vec<u8> = code[len_pc + 1..len_pc + 1 + n].to_vec();
            let funcaddr = ctx.trace_ctx.box_value(funcptr).and_then(|v| match v {
                majit_ir::Value::Int(n) => Some(n as u64),
                _ => None,
            });
            eprintln!(
                "[fbw-unbound] pc={} regs={:?} r_args={:?} func={:?} pyre_helper={:?}",
                op.pc,
                regs,
                r_args,
                funcaddr.map(|a| format!("{a:#x}")),
                ei.pyre_helper,
            );
        }
        return Err(e);
    }

    // Optional diagnostic for iRd-shape residual calls.  The STORE_SUBSCR
    // specialization keys on a fn-pointer match against `bh_store_subscr_fn`
    // plus `r_args.len() == 3` with `dst_bank == 'v'`; logging raw addresses
    // here makes mismatches visible without affecting production when the
    // env var is unset.
    if crate::probe_subscr_enabled() {
        let funcptr_addr = ctx.trace_ctx.box_value(funcptr).and_then(|v| match v {
            majit_ir::Value::Int(n) => Some(n as u64),
            _ => None,
        });
        let arg_addrs: Vec<Option<u64>> = r_args
            .iter()
            .map(|&op| {
                ctx.trace_ctx.box_value(op).and_then(|v| match v {
                    majit_ir::Value::Ref(r) => Some(r.as_usize() as u64),
                    _ => None,
                })
            })
            .collect();
        eprintln!(
            "[PYRE_PROBE_SUBSCR] dispatch_residual_call_iRd_kind pc={} dst_bank={} r_args.len={} funcptr_addr={:?} arg_addrs={:?}",
            op.pc,
            dst_bank,
            r_args.len(),
            funcptr_addr.map(|a| format!("{:#x}", a)),
            arg_addrs
                .iter()
                .map(|o| o.map(|a| format!("{:#x}", a)))
                .collect::<Vec<_>>(),
        );
    }

    // STORE_SUBSCR strategy-aware specialization.  Fires when funcptr
    // matches the registered `store_subscr_fn` address, r_args carries the
    // 3-arg `[obj_reg, key_reg, value_reg]` shape codewriter emits
    // (`codewriter.rs build_store_subscr_fn_residual_call_r_v_insn`),
    // dst_bank is `'v'` (STORE_SUBSCR returns void), and all 3 concrete
    // shadow slots are populated.  On success, records the specialized
    // IR shape (guard_class + guard_strategy + setarrayitem-family) via
    // the trait-equivalent `generated_store_subscr_value` helper (now
    // generic over `WalkerFrameOps`, with `WalkContext` impl).
    //
    // Production dispatch supplies the expected address via
    // `WalkContext.store_subscr_fn_addr`; tests and diagnostics may use
    // `PYRE_WALKER_STORE_SUBSCR_FNADDR=<hex>`.  Without either address,
    // the gate decays to no-op and dispatcher falls through to the generic
    // residual-call path.
    let specialization =
        try_walker_store_subscr_specialization(ctx, code, op, funcptr, &r_args, dst_bank);
    // Drain the snapshot-capture failure the `WalkerFrameOps`
    // `generate_guard` impl latched (its `()` trait signature has no error
    // channel): a guard recorded without a resume snapshot must abort the
    // walk, whether the specialization completed or declined mid-way.
    if let Some(e) = ctx.pending_guard_snapshot_error.take() {
        return Err(e);
    }
    if let Some(outcome) = specialization {
        return Ok((outcome, op.next_pc));
    }

    // StoreName/StoreGlobal IntMutableCell in-place store fold: module-scope
    // dual of the LoadName/LoadGlobal cell fold.  Fires when the target slot
    // holds a stabilised immovable `IntMutableCell` and the store value is a
    // provably-plain-int box; emits `QUASIIMMUT_FIELD` + `setfield_gc_i(cell,
    // intvalue)`, eliding the boxing + residual dict setitem.  Carries the
    // LoadName fold's handler-free gate, for the reason recorded there.
    //
    // Two staleness bugs were fixed before enabling this unconditionally:
    // (1) the fold now eagerly applies the concrete
    // `cell.intvalue` write (journaled in [`FBW_CELL_STORE_JOURNAL`]) —
    // without it the walk's remaining concrete execution read the pre-store
    // global and the next LOAD fold's cache-hit sanity check tripped;
    // (2) `int_mutable_cell_value_descr` is a singleton `Arc` so the
    // optimizer's `cached_fields` (keyed by `descr_identity`) connects the
    // store's lazy `setfield_gc_i` to the LOAD's `getfield_gc_i` —
    // per-call fresh Arcs let `force_lazy_sets_for_guard` flush the store
    // BELOW an emitted load of the same cell (the nested module-loop
    // `i = i + 1; while i < n` read the pre-increment value and ran one
    // extra iteration).
    if ctx.is_authoritative_executor
        && dst_bank == 'v'
        && r_args.len() == 3
        && matches!(
            ei.pyre_helper,
            majit_ir::PyreHelperKind::StoreName | majit_ir::PyreHelperKind::StoreGlobal
        )
        && !jitcode_has_exception_handler(code)
    {
        if let (Some(&frame_opref), Some(&name_opref), Some(&value_opref)) =
            (r_args.first(), r_args.get(1), r_args.get(2))
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_name_ptr))),
            ) = (
                ctx.trace_ctx.box_value(frame_opref),
                ctx.trace_ctx.box_value(name_opref),
            ) {
                if try_walker_store_name_cell_fold(
                    ctx,
                    op.pc,
                    ei.pyre_helper,
                    frame_ptr,
                    w_name_ptr,
                    value_opref,
                )? {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    // pyjitpl.py:1094-1118 `opimpl_jit_force_quasi_immutable`, asked at the
    // residual boundary because pyre's namespace write lives inside one.
    //
    // Ordered AFTER the cell fold above (a successful in-place fold IS the
    // no-bump case) and BEFORE `try_execute_residual_call_via_executor`, so
    // nothing has been applied when the abort fires and the resume re-runs the
    // whole opcode cleanly. Executing first — the order `do_not_in_trace_call`
    // uses for its own contract — would double-apply the store or the delete.
    if ctx.is_authoritative_executor
        && dst_bank == 'v'
        && try_walker_force_quasi_immut_namespace_write(ctx, ei.pyre_helper, &r_args).is_some()
    {
        // Carry the reason to the `abort_trace` that follows, the way upstream
        // carries it on the `SwitchToBlackhole` instance (pyjitpl.py:2906-2910).
        // Without it the abort lands in the `Generic` catch-all and the
        // `abort: force quasi-immut` counter stays at 0.
        crate::state::note_force_quasi_immut_abort();
        return Err(DispatchError::ForceQuasiImmutable { pc: op.pc });
    }

    // pyjitpl.py OS_NOT_IN_TRACE guard — see helper docstring
    // for the convergence rationale.
    if let Some(outcome) = do_not_in_trace_call_result(ei, op.pc)? {
        return Ok((outcome, op.next_pc));
    }
    // pyjitpl.py OS_JIT_FORCE_VIRTUAL fail-loud — walker
    // can't reproduce `_do_jit_force_virtual` without a concrete
    // `vref_ptr` resolver; surface a typed error rather than silently
    // recording `CALL_MAY_FORCE_*`.
    do_jit_force_virtual_guard(ei, op.pc)?;

    // #62: `is_true(box_bool(t))` -> `t` fold.  A `POP_JUMP_IF_*` lowers to an
    // `is_true` residual (`residual_call_r_i`, Int result) whose sole Ref arg
    // is the boxed bool a preceding COMPARE specialization produced.  Folding
    // it to the raw truth Int elides the may-force unbox (and lets the dead box
    // + value-stack store DCE), matching the retired MIFrame path's
    // branch-on-raw-compare behaviour. bool->int is value-preserving so the fold is sound. The
    // lookup is read-only (it does not remove the entry); OpRef SSA-uniqueness
    // (`recorder.rs`) guarantees the box opref never re-binds within one walk,
    // so a stale mis-fold is impossible and physical removal is unnecessary.
    // Authoritative walks only: `BOOL_BOX_TRUTH` is reset at FBW walk
    // entry; a non-authoritative context consulting it could read a stale
    // OpRef key from an earlier walk's recorder.
    if ctx.is_authoritative_executor && dst_bank == 'i' && r_args.len() == 1 {
        if let Some(truth) = bool_box_truth_lookup(r_args[0]) {
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, truth)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        // #124: a TO_BOOL / POP_JUMP truth residual on a provably-int box
        // (e.g. the `(i % 7)` in `(i % 7) and (i + 3)`) folds to a pure
        // `int_is_true`, eliding the may-force call whose force/exc guards
        // mis-resume the kept short-circuit stack.
        if ei.pyre_helper == majit_ir::PyreHelperKind::Truth {
            if let Some(truth) = try_walker_specialize_truth_int(ctx, op.pc, r_args[0])? {
                write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, truth)?;
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
            // The boxed bool a residual `COMPARE_OP` leaves behind — the int
            // arm above guards `INT_TYPE` and declines it, so without this the
            // test on every `if a == b:` stays a second may-force call.
            if let Some(truth) = try_walker_specialize_truth_bool(ctx, op.pc, r_args[0])? {
                write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, truth)?;
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
        }
    }

    // #62: specialize STORE_SUBSCR `list[int] = value` (int / float storage,
    // in-bounds, type-matching) to the walker-native `setarrayitem_raw` form,
    // eliding the `CALL_MAY_FORCE` that would force the virtualizable every
    // iteration.  Falls through to the generic residual otherwise (SAFE).
    // Full-body walks only: the eager store rides `FBW_STORE_JOURNAL`,
    // whose commit/rollback epilogues run on FBW walk ends.
    if ctx.is_authoritative_executor
        && dst_bank == 'v'
        && ei.pyre_helper == majit_ir::PyreHelperKind::StoreSubscr
    {
        if try_walker_specialize_store_subscr(ctx, op.pc, &r_args)?.is_some() {
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        // #171 setslice inline: `target[const_slice] = source` for a
        // same-length, step-1, Integer↔Integer slice — fold the assignment
        // into per-element getarrayitem/setarrayitem on the int_items blocks so
        // a virtualizable BUILD_LIST source temp is consumed without forcing.
        // Declines to the opaque residual for any shape it cannot reproduce
        // faithfully (SAFE — always byte-correct).
        if try_walker_specialize_setslice(ctx, op.pc, &r_args)?.is_some() {
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        if ctx.trace_ctx.is_bridge_trace && fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-store-fallthrough] bridge STORE_SUBSCR fell to GENERIC residual at pc={} \
                 (specialization declined — unjournaled concrete store)",
                op.pc
            );
        }
    }

    // Range GET_ITER: virtualize exact machine-word `range` into the same
    // `W_IntRangeIterator` shape PyPy's inlined `descr_iter` would trace.
    if ctx.is_authoritative_executor && ei.pyre_helper == majit_ir::PyreHelperKind::GetIter {
        if let Some(iter_op) = try_walker_specialize_get_iter(ctx, op.pc, &r_args, dst, dst_bank)? {
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, iter_op)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }

    // Range FOR_ITER is a C-level iterator advance.  Re-emit its field
    // updates so the opaque ForIterNext residual cannot invalidate optheap;
    // other iterator families retain the residual and its Python semantics.
    // The specialization supplies the same Ref result that the residual would,
    // including NULL for exhaustion, so the codewriter's trailing
    // GuardNonnull remains the only loop-exit guard.
    if ctx.is_authoritative_executor && ei.pyre_helper == majit_ir::PyreHelperKind::ForIterNext {
        if let Some(item_op) =
            try_walker_specialize_for_iter_next(ctx, op.pc, &r_args, dst, dst_bank)?
        {
            write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, item_op)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
        if let Some(outcome) = try_walker_specialize_seqiter_getitem_next(
            ctx, op, code, funcptr, &r_args, call_descr, dst, dst_bank,
        )? {
            return Ok(outcome);
        }
    }

    // Emit MAKE_FUNCTION's `Function.__init__` as New + SetField so a `def` in a
    // loop body virtualizes away instead of allocating through the opaque
    // residual.  Falls through to the residual for every shape the emit cannot
    // reproduce constant-for-constant (SAFE — never declined).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::MakeFunction
        && try_walker_specialize_make_function(ctx, op.pc, &r_args, dst, dst_bank)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // #171: virtualize a BUILD_TUPLE (`newtuple_from_array`) of any width as
    // the canonical array-backed `W_TupleObject` shape, so a non-escaping tuple
    // folds away rather than allocating through the opaque residual, and every
    // consumer fold that reads `wrappeditems` applies to it.  Falls through to
    // the residual for any shape it cannot reproduce (SAFE — never declined).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::NewtupleFromArray
        && try_walker_specialize_newtuple_object(ctx, op.pc, &r_args, dst, dst_bank)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // #195 / #73: the arity-2 plain-int `spec_ii` shape (`new_with_vtable` +
    // `value0` / `value1`) as the fallback for a pair the canonical fold above
    // could not reproduce — it needs a const backing-array length, which the
    // element probing here does not.  Falls through to the opaque residual for
    // any other shape (SAFE — never declined).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::NewtupleFromArray
        && try_walker_specialize_newtuple(ctx, op.pc, &r_args, dst, dst_bank)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // #171: virtualize a non-escaping BUILD_LIST (`newlist_from_array`) by
    // decomposing it into the `opimpl_newlist` shape (`pyjitpl.py`) —
    // `new_with_vtable` + `new_array` + `setarrayitem_gc` + `setfield_gc` —
    // choosing the storage strategy from the concrete element shadows exactly
    // like `w_list_new` / `list_strategy_for`, so the traced object matches what
    // the blackhole rebuilds on deopt.  Falls through to the opaque residual
    // for any shape it cannot reproduce faithfully (empty list, non-const
    // array length, an element without a concrete Ref shadow) — SAFE, never
    // declined.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::NewlistFromArray
        && try_walker_specialize_newlist(ctx, op.pc, &r_args, dst, dst_bank)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // #171: specialize `lst.append(x)` so its array ops reach the trace,
    // replacing the opaque `bh_call_fn` residual (orthodox descent of the
    // real `w_list_append` body; see `try_walker_orthodox_list_append`).  The
    // call arrives as `CallFn` with `dst_bank == 'r'` (the None result is a
    // Ref, not void) and `r_args = [bound-method, PY_NULL, value]`.  Falls
    // through to the residual for any non-matching shape (SAFE).  The eager append rides
    // `FBW_APPEND_JOURNAL`, whose commit/rollback epilogues run on FBW walk
    // ends (same lifecycle as the STORE_SUBSCR store journal).
    //
    // Restrict to the top full-body frame: inside an inlined callee sub-walk
    // (`fbw_mode.inline_subwalk`) the fold's gating guards collapse
    // their resume to the caller's CALL boundary (`entry_py_pc` /
    // `outer_active_boxes`), which re-executes the whole caller iteration on a
    // guard failure — doubling any caller side effect sequenced before the
    // inlined call (e.g. a `STORE_ATTR` ahead of an inlined `push(lst, x)`).
    // An inlined append falls back to the generic residual, which resumes
    // *past* the call (after_residual_call) and so re-runs nothing extra.
    //
    // Both loop and function-entry (no-loop) traces are eligible.  A
    // no-loop helper compiled from entry (e.g. `def push(a, v): a.append(v)`
    // called in a hot loop) traces with `header_pc == 0`; its spare-capacity
    // guard's resume reconstructs the receiver from the call-site coordinate
    // published below (`collect_outer_active_boxes` at the CALL py_pc), which
    // preserves the receiver local across the fold's mid-statement guards in
    // both trace kinds.  The earlier loop-only restriction was a carryover
    // from the retired hand-rolled fold (#227), whose function-entry exit
    // layout dropped the receiver; the orthodox descent's resume coordinate
    // does not, verified by a two-list alternating-receiver append stress
    // (any wrong receiver box corrupts the cross-checked lists) and the
    // parity suite folding function-entry helper appends on both backends.
    // #171 ORTHODOX descent: descend the real `w_list_append` body,
    // recording its array ops native.
    // A recognition or body-sub-walk decline falls through to the generic
    // residual below. Gated to top full-body frames, not inside a sub-walk.
    if ctx.is_authoritative_executor
        && !ctx.fbw_mode.inline_subwalk
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_orthodox_list_append(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // #171 ORTHODOX descent for the LIST_APPEND opcode (comprehension append,
    // e.g. `[f(x) for x in xs]`).  The codewriter lowers LIST_APPEND to a void
    // `jit_list_append(list, value)` residual tagged `ListAppendValue` (the
    // list is the peeked receiver, the value the popped operand — no
    // bound-method callable), so it arrives with `dst_bank == 'v'`.  Fold it
    // through the same `w_list_append` descent as the CallFn method-call form.
    // Gated to top full-body frames, not inside a sub-walk (same
    // caller-side-effect doubling concern as the CallFn form).
    if ctx.is_authoritative_executor
        && !ctx.fbw_mode.inline_subwalk
        && dst_bank == 'v'
        && ei.pyre_helper == majit_ir::PyreHelperKind::ListAppendValue
    {
        // Fold to native array stores when the receiver has spare capacity, or
        // fall through to the generic `jit_list_append` residual below.  The
        // fold's `orthodox_list_append_commit` journals the append
        // (`fbw_append_journal_push`) so `fbw_store_journal_rollback` rewinds it
        // on abort; the generic executor is now equally abort-safe — its
        // `list_append_journal` records the same pre-append length before
        // running `jit_list_append`, so a later abort + interpreter replay
        // applies the append exactly once (no silent double).  The fold's
        // decline point (`try_walker_orthodox_list_append_opcode`) is
        // side-effect-free — it declines BEFORE emitting any IR — so the
        // fall-through starts from a clean trace position.  This lets a resize
        // side-exit's bridge (the fold declines at a full backing block) compile
        // a `jit_list_append` residual instead of aborting the whole bridge.
        if try_walker_orthodox_list_append_opcode(ctx, code, op, &r_args, dst)?.is_some() {
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }

    // `type(x)` is `space.type(w_obj)` upstream: promote `w_obj.__class__`
    // and return `w_obj.getclass(space)`.  Lower that directly instead of
    // residualizing the builtin type object's full `descr_call`.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_builtin_type(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // Exact `dict.get(identity_key)` follows the promoted strategy-entry
    // shape produced by tracing `DictStrategy.getitem`: pin the key-set
    // iterator state and read the resolved entry value live.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_builtin_dict_get(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // `len(x)` on an exact canonical list: inline the strategy-guarded
    // length read (guard_value callable + guard_class + exact w_class +
    // guard_value strategy + length getfield + wrapint) instead of the
    // opaque `bh_call_fn(len_builtin, NULL, x)` residual — the shape the
    // meta-tracer produces upstream (descroperation.py `_len` →
    // `W_ListObject.length()`).  Read-only like the SUBSCR fold, so no
    // sub-walk restriction; any non-matching shape falls through to the
    // generic residual (SAFE).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_builtin_len(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // An exact `range(...)` constructor call becomes a virtual W_Range whose
    // four wrapped-int fields can fold directly into GET_ITER virtualization.
    // Non-canonical callables and arguments fall through to the residual.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_builtin_range(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // Zero-argument `locals()` / `vars()` / `dir()` on the walk's own portal
    // frame: model `fast2locals`' fastlocals reads as `getarrayitem_vable_r`
    // plus a non-forcing dict-build chain — the shape the meta-tracer produces
    // upstream, where `pyframe.py:539 fast2locals` is `@jit.unroll_safe` and
    // therefore looked into.  This arm runs BEFORE
    // `try_execute_residual_call_via_executor` arms the vable token protocol,
    // which is the point: the opaque residual is what turns the locals-read
    // barrier into `VableEscapedDuringResidualCall`.  Any non-matching shape
    // falls through to the generic residual (SAFE).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_builtin_locals(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // `math.sqrt(x)` / `float(x)` on an exact numeric argument: inline the
    // domain-guarded pure `CALL_F(sqrt_nonneg_jit)` (ll_math.rs) resp. the
    // `CastIntToFloat` / identity conversion instead of the opaque
    // `bh_call_fn` residual, so the result `W_FloatObject` virtualizes.  Any
    // non-matching shape (rebound name, subclass, non-numeric arg, negative /
    // non-finite sqrt) falls through to the generic residual (SAFE).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_math_sqrt(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_math_log_trig(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_math_frexp(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_math_ldexp(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_math_isqrt(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_int_call(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_float_call(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // `divmod(a, b)` on two exact ints: inline the guarded
    // `OS_INT_PY_DIV` / `OS_INT_PY_MOD` pair into a virtual `Cls_ii`
    // specialised tuple (intobject.py `_divmod` → `newtuple2`) instead of the
    // opaque `bh_call_fn(divmod_builtin, NULL, a, b)` residual.  Pure like the
    // `len` fold, so no sub-walk restriction; any non-matching shape falls
    // through to the generic residual (SAFE).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_specialize_builtin_divmod(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // B3: a `raise Type(args)` of a canonical
    // builtin exception class arrives as two residuals — a `CallFn` that
    // constructs the exception, and a `RaiseVarargs`
    // (`normalize_raise_varargs_jit`) that publishes it.  The construct fold
    // (`try_walker_trace_exception_new`) emits the inline virtualizable
    // `NewWithVtable` + SetField shape instead of the opaque `bh_call_fn`
    // constructor call (mirrors the retired trait-side constructor fold); the
    // raise fold (`try_walker_trace_raise_builtin`) then skips the residual
    // publish for a freshly-built exception with no `from` cause, emitting
    // `__context__` as a `SetfieldGc` on the still-virtual exception.
    // Together they drop the two may-force calls (construct + normalize) so
    // the exception virtualizes and DCEs.  Any non-matching shape falls
    // through to the generic residual (SAFE).
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::CallFn
        && try_walker_trace_exception_new(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::RaiseVarargs
        && r_args.is_empty()
        && ctx.fbw_mode.current_exception_seed.is_some()
    {
        let seed = ctx.fbw_mode.current_exception_seed.unwrap();
        let concrete = ctx.fbw_mode.current_exception_seed_concrete;
        if !concrete.is_null() && unsafe { pyre_object::is_exception(concrete) } {
            // `RAISE_VARARGS 0` may use the normalizing nullary helper rather
            // than the raw current-exception helper.  A bridge seed is already
            // a live BaseException, so the helper's successful result is the
            // pending fieldbox itself; retaining that OpRef keeps the value
            // loop-variant in the bridge namespace.
            ctx.trace_ctx.set_opref_concrete(
                seed,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
            );
            write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', seed)?;
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::RaiseVarargs
        && try_walker_trace_raise_builtin(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    // A bare-class `raise Type` has no construct residual to fold; synthesize
    // the zero-argument instance at the raise itself when the operand is a
    // canonical builtin exception class.
    if ctx.is_authoritative_executor
        && dst_bank == 'r'
        && ei.pyre_helper == majit_ir::PyreHelperKind::RaiseVarargs
        && try_walker_trace_raise_bare_class(ctx, code, op, &r_args, dst)?.is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    // B3 piece 3: lower the PUSH_EXC_INFO / POP_EXCEPT
    // exc-info-stack residuals to GETFIELD_GC_R / SETFIELD_GC on the EC's
    // `sys_exc_value` slot. Recognised by the
    // codewriter-stamped `pyre_helper` tag (not a funcptr address — the
    // residual calls the cross-crate `cpu.{get,set}_current_exception_fn`
    // wrappers).  A balanced PUSH save + POP restore on the same descr-
    // identity field is dead-store-eliminated by the heap optimizer, so a
    // non-escaping exception (built + raised + caught in one trace) stays
    // virtual and DCEs — eliding the per-iteration `set_current_exception`
    // CALL that otherwise forces the exception to materialize.
    if ctx.is_authoritative_executor
        && matches!(
            ei.pyre_helper,
            majit_ir::PyreHelperKind::GetCurrentException
                | majit_ir::PyreHelperKind::SetCurrentException
        )
        && try_walker_lower_exc_info_residual(
            ctx,
            code,
            op,
            ei.pyre_helper,
            &r_args,
            dst_bank,
            dst,
        )?
        .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // pyjitpl.py forces-branch sub-case: when the descr's
    // `call_release_gil_target` is a non-NULL `(realfuncaddr, saveerr)`
    // pair, route through `direct_call_release_gil` which records
    // `CALL_RELEASE_GIL_*` with the upstream-shape arglist
    // `[savebox, funcbox] + argboxes[1:]` (pyjitpl.py).  All
    // other forces-branch paths (CALL_MAY_FORCE_*, the loopinvariant
    // sub-case below, the elidable branch, the default branch) come
    // out of `select_residual_call_opcode`.
    if ei.is_call_release_gil() {
        if let Some(outcome) = direct_call_release_gil(
            ctx,
            ei,
            &allboxes,
            descr.clone(),
            call_descr,
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iRd_kind",
        )? {
            return Ok((outcome, op.next_pc));
        }
    } else if let Some(cached) = loopinvariant_lookup(ctx, ei, descr_key, funcptr) {
        // pyjitpl.py EF_LOOPINVARIANT short-circuit. The
        // cached path emits no IR op and no guard, so result-before-
        // guard ordering is moot — write the dst eagerly.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, cached)?;
    } else {
        let (call_opcode, can_raise, emit_guard_not_forced) =
            select_residual_call_opcode(ei, dst_bank, "dispatch_residual_call_iRd_kind");
        // The abort gate is static (EI flags) and must run BEFORE the
        // concrete executor below: `do_residual_call` (pyjitpl.py)
        // executes the helper only on a path that keeps recording — it has
        // no execute-then-abandon shape.  Aborting after execution would
        // leave the helper's heap/exception effects standing while the
        // declined retrace re-runs the same bytecode, double-applying them.
        walker_abort_if_mayforce_null_ref_arg(call_opcode, &allboxes, call_descr, ctx, op.pc)?;

        // pyjitpl.py `vable_and_vrefs_before_residual_call` — fires
        // unconditionally on the forces branch.  Records FORCE_TOKEN +
        // SETFIELD_GC IR for the active virtualizable; the runtime heap
        // mutations on `vinfo.tracing_before_residual_call` and
        // `vrefinfo.tracing_before_residual_call` (`pyjitpl.py`)
        // are handled by the residual-call execution path — see
        // [`walker_vable_and_vrefs_before_residual_call`] for the IR-vs-heap
        // split rationale.
        if emit_guard_not_forced {
            maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
        }

        // pyjitpl.py:2669-2682 `execute_and_record_varargs`; may-force
        // calls use `history.record_nospec` and therefore count nothing.
        if matches!(
            call_opcode,
            OpCode::CallI
                | OpCode::CallR
                | OpCode::CallF
                | OpCode::CallN
                | OpCode::CallPureI
                | OpCode::CallPureR
                | OpCode::CallPureF
                | OpCode::CallPureN
                | OpCode::CallLoopinvariantI
                | OpCode::CallLoopinvariantR
                | OpCode::CallLoopinvariantF
                | OpCode::CallLoopinvariantN
        ) {
            ctx.trace_ctx
                .profiler()
                .count_ops(call_opcode, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(call_opcode, majit_metainterp::counters::RECORDED_OPS);
        }
        // Always record `list_write_barrier` on the Object strategy's in-place
        // append arm.  Dropping it in favour of the backend's
        // `COND_CALL_GC_WB_ARRAY` on the block's `setarrayitem` is unsound: a
        // guard-failure bridge that re-materializes the items block appends into
        // it without that array barrier ever firing, so an `old -> young` slot
        // store leaves the block off the remembered set.  A later minor frees
        // the still-referenced young element and the collector then reads a
        // freed (poison) header.  The list barrier remembers the enclosing
        // `W_ListObject`, whose trace reaches every slot, and keeps them alive.
        let recorded = ctx
            .trace_ctx
            .record_op_with_descr(call_opcode, &allboxes, descr.clone());

        // pyjitpl.py `_record_helper_pure` parity: for
        // `CallPure*` whose every argbox carries a known `box_value`,
        // execute the helper now and stamp `recorded` with the result so
        // downstream walker chain (sub-jitcode bodies that consume the
        // result via `concrete_of_opref`) folds end-to-end.  No-op when
        // any argbox is symbolic, when the EI can raise, or for non-pure
        // call opcodes.
        try_fold_pure_call_via_executor(ctx, call_opcode, &allboxes, call_descr, recorded);

        // pyjitpl.py `_opimpl_residual_call{1,2,3}` parity
        // for the non-elidable shapes.  PyPy
        // concrete-executes EVERY residual call regardless of EI — the
        // `exc` flag only selects the *guard* shape downstream
        // (`handle_possible_exception` → `GUARD_EXCEPTION` vs
        // `GUARD_NO_EXCEPTION`), not whether the helper runs.  Without
        // this, walker-recorded non-elidable helpers
        // (`store_subscr_fn`, `set_current_exception`, …) would skip
        // their heap mutation because `eval.rs`'s walker-skip
        // path bypasses `execute_opcode_step` → SIGBUS on the next read
        // of the un-mutated container.
        // PyPy-orthodox activation.  PyPy's
        // `_opimpl_residual_call*` concrete-executes EVERY residual
        // call regardless of EI; the `exc` flag only selects the
        // post-call guard shape (`GUARD_EXCEPTION` vs `GUARD_NO_EXCEPTION`)
        // in `handle_possible_exception`.  Pyre matches by always
        // invoking the executor — `try_execute_residual_call_via_executor`
        // self-gates on a fnaddr-sanity check (rejecting unpatched
        // `symbolic_fnaddr_for_path` hashes whose bits ≥ 47 are set)
        // so unregistered helpers degrade gracefully to recording-only
        // instead of SIGBUSing.
        let resid_exec = try_execute_residual_call_via_executor(
            ctx,
            call_opcode,
            &allboxes,
            call_descr,
            recorded,
            op.pc,
            Some((op.next_pc, dst_bank, dst)),
        )?;
        // A decline leaves the call recorded symbolically WITHOUT running
        // it — a side effect only the legacy replay applies, so the
        // walk-end no-replay commit must stay off for this trace (see
        // `fbw_has_unjournaled_effect`).  Pure/elidable calls never reach
        // this dispatcher (they fold via the pure-call executor).
        let resid_raised = match resid_exec {
            ResidualExecOutcome::Executed(result) => result.is_err(),
            ResidualExecOutcome::Declined(cause) => {
                fbw_abort_nested_unjournaled_residual(ctx, op.pc)?;
                fbw_mark_unjournaled_effect(cause);
                false
            }
        };
        debug_assert!(
            !resid_raised || can_raise,
            "dispatch_residual_call_iRd_kind: helper raised on a \
             `!can_raise` EI — EffectInfo claim/reality mismatch"
        );

        // `_record_helper_varargs` parity: every
        // recorded varargs op invalidates the heapcache via
        // `heapcache.invalidate_caches_varargs(opnum, descr,
        // argboxes)`.  Pyre's `record_op_with_descr` does NOT
        // auto-invalidate, so call it explicitly here.  Forces
        // branch (`select_residual_call_opcode` returned a
        // `CallMayForce*`) thus matches `pyjitpl.py` which uses
        // `opnum1 = CALL_MAY_FORCE_*`; non-forces branches
        // (`CallLoopinvariant*`/`CallPure*`/`Call*`) match the
        // `_record_helper_varargs` invocation that runs inside
        // upstream's `executor.execute_varargs(opnum, ...)`.
        ctx.trace_ctx
            .heapcache_invalidate_caches_varargs(call_opcode, Some(ei), &allboxes);
        // pyjitpl.py execute_varargs: `make_result_of_lastop(op)`
        // runs BEFORE `handle_possible_exception()` precisely "because we need
        // the box to show up in get_list_of_active_boxes()".  Write the dst
        // for every non-void result REGARDLESS of whether the helper raised,
        // so the GUARD_NOT_FORCED fail_args snapshot reads the recorded OpRef
        // in the slot the resume position points at — otherwise a raising call
        // surfaces NONE in fail_args for the `>X` slot.  On a raised call the
        // OpRef carries a Null concrete shadow (never read on the exception
        // path); only the *caching* of a raised result is skipped below,
        // matching upstream's `not last_exc_value` pure-cache gate.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
        // pyjitpl.py `metainterp.generate_guard(rop.GUARD_NOT_FORCED)`
        // — unconditionally on the forces-virtual-or-virtualizable branch.
        // The walker omits the `vable_after_residual_call(funcbox)`
        // short-circuit (`pyjitpl.py`): the residual-call execution
        // path's heap-token bracket already detects a vable escape and
        // surfaces `VableEscapedDuringResidualCall` before this guard is
        // emitted.
        if emit_guard_not_forced {
            // #73: maintain the `-live-` AFTER anchor.  A
            // residual-call guard reads its resume point at `self.pc` (the
            // `-live-` trailing the call, `pyjitpl.py`).  `op.next_pc` is
            // the first byte after the residual_call opcode, which the
            // `[funcptr, Call, -live-]` layout (jitcode.rs) makes the
            // trailing `-live-` byte.  Side-data only.
            ctx.live_after_jit_pc = op.next_pc;
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        // `metainterp.handle_possible_exception()` —
        // emits `GUARD_EXCEPTION(exc_type)` when the recording-time
        // helper raised (pinning the class for guard recovery), else
        // `GUARD_NO_EXCEPTION`.  The capture ports
        // `capture_resumedata(after_residual_call=True)`
        // (keyed on the guard opcode) so the
        // optimizer's `store_final_boxes_in_guard` finds a
        // `rd_resume_position` advanced *past* the call.
        if can_raise {
            if resid_raised {
                walker_record_guard_exception(ctx, op.pc);
                // `handle_possible_exception` routes
                // the raising branch through `finishframe_exception()`
                // immediately after emitting `GUARD_EXCEPTION`, so the
                // remaining bytes of the arm never run.  Surface the
                // outcome to `walk_loop` as `SubRaise`: at top-level it
                // emits the outer `FINISH(exc)` and Terminates the trace;
                // at sub-walk depth it propagates up to the caller's
                // `inline_call_*` handler.  Continuing past this point
                // would record dead arm IR (e.g. the arm's tail
                // `*_return`) onto an exception path and confuse the
                // optimizer's guard-fail snapshot.
                let exc = ctx
                    .last_exc_value
                    .expect("resid_raised implies last_exc_value seeded by the Err branch");
                let exc_concrete = ctx.last_exc_value_concrete;
                return Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc));
            } else {
                ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
                // Request that this residual call's no-exception-guard resume
                // route through the call's OWN post-call catch
                // (`GuardCaptureScope::residual_call_catch_resume`).  The
                // snapshot helper carries the call's jitcode offset only when
                // the CALL pc is actually covered by the code's exception table
                // (checked in `walker_capture_snapshot_for_last_guard_impl`);
                // an uncovered residual keeps the generic fallthrough resume.
                // See the scope field's doc.
                walker_capture_snapshot_for_last_guard_scoped(
                    ctx,
                    op.pc,
                    GuardCaptureScope {
                        residual_call_catch_resume: true,
                        ..GuardCaptureScope::default()
                    },
                )?;
            }
        }

        // pyjitpl.py `heapcache.call_loopinvariant_now_known`:
        // populate the cache so a subsequent matching call short-
        // circuits via the lookup above.  No-op for non-loopinvariant
        // EI per `loopinvariant_now_known`'s extraeffect check.
        //
        // Skip on `resid_raised`: caching a `recorded` OpRef with no
        // stamped concrete would propagate the un-stamped value into a
        // subsequent loop iteration's `loopinvariant_lookup` hit,
        // bypassing the actual helper call.
        if !resid_raised {
            loopinvariant_now_known(ctx, ei, descr_key, funcptr, recorded);
        }
    }

    Ok((DispatchOutcome::Continue, op.next_pc))
}

#[allow(non_snake_case)]
pub(crate) fn dispatch_residual_call_iIRd_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // execute_varargs (pyjitpl.py) clear_exception at every
    // residual-call entry; see dispatch_residual_call_iRd_kind.
    let saved_last_exc_value = ctx.last_exc_value;
    let saved_last_exc_value_concrete = ctx.last_exc_value_concrete;
    let preserve_last_exc_for_handler =
        saved_last_exc_value.is_some() && reads_last_exc_before_next_catch(code, op.next_pc);
    if !preserve_last_exc_for_handler {
        clear_walk_exception(ctx);
    }
    let funcptr = read_int_reg(code, op, 0, ctx)?;
    let (i_args, i_width) = read_int_var_list(code, op, 1, ctx)?;
    let (mut r_args, r_width) = read_ref_var_list(code, op, 1 + i_width, ctx)?;
    let descr_offset = 1 + i_width + r_width;
    let descr_index = decode_descr_index(code, op, descr_offset);
    let mut descr = read_descr(code, op, descr_offset, ctx)?;
    let original_call_descr =
        descr
            .as_call_descr()
            .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
                pc: op.pc,
                descr_index,
            })?;
    let descr_key = descr.index();
    // Copy the recognition tag out of the descr borrow now: the DELETE_ATTR
    // fold gate below runs after the StoreAttr arm's `descr = specialized_
    // descr` reassignment, which the live `original_call_descr` borrow would
    // otherwise forbid (E0506).
    let pyre_helper_kind = original_call_descr.get_extra_info().pyre_helper;
    repair_carrier_call_ref_args(ctx, op.pc, pyre_helper_kind, &mut r_args);
    // Void shape `_ir_v/iIRd` (`pyjitpl.py opimpl_residual_call_ir_v =
    // _opimpl_residual_call2`) has no `>X` dst byte; see
    // `dispatch_residual_call_iRd_kind` for the void operand-layout note.
    let dst = if dst_bank == 'v' {
        0
    } else {
        code[op.pc + 1 + descr_offset + 2] as usize
    };

    // Flat argboxes = i_args ++ r_args (`boxes2` argcode order).
    // Parallel argbox_types stamps each entry with its source bank so
    // `_build_allboxes`'s type-filter loops can permute correctly.
    let mut argboxes: Vec<OpRef> = Vec::with_capacity(i_args.len() + r_args.len());
    let mut argbox_types: Vec<Type> = Vec::with_capacity(i_args.len() + r_args.len());
    argboxes.extend_from_slice(&i_args);
    argbox_types.extend(std::iter::repeat(Type::Int).take(i_args.len()));
    argboxes.extend_from_slice(&r_args);
    argbox_types.extend(std::iter::repeat(Type::Ref).take(r_args.len()));
    let mut allboxes = build_allboxes(
        funcptr,
        &argboxes,
        &argbox_types,
        original_call_descr.arg_types(),
    );

    // pyjitpl.py:1105-1120 `opimpl_jit_force_quasi_immutable` must run before
    // any fold or residual applies the opcode. In particular,
    // `try_walker_specialize_store_attr` mutates `?` fields while resolving,
    // so moving this below the STORE_ATTR fold would hide the force. The abort
    // resumes by re-running the whole opcode, so no part of its write may have
    // happened yet.
    if ctx.is_authoritative_executor
        && dst_bank == 'v'
        && mapdict_qmut_force_enabled()
        && try_walker_force_quasi_immut_mapdict_write(
            ctx,
            op.pc,
            pyre_helper_kind,
            &i_args,
            &r_args,
        )
        .is_some()
    {
        // A sub-walk abort's resume coordinate names the callee's code object;
        // `self.x = v` inside an inlined callee is reachable, and the flush
        // leg's guard reads this flag.
        ctx.session.borrow_mut().abort_in_subwalk = ctx.fbw_mode.inline_subwalk;
        crate::state::note_force_quasi_immut_abort();
        return Err(DispatchError::ForceQuasiImmutable { pc: op.pc });
    }

    // STORE_ATTR fold (mapdict.py): recognize an existing unboxed
    // integer slot and replace only the generic setattr residual's helper,
    // arguments, and effect.  The transformed CallN continues through the
    // ordinary record + concrete-execute path below; unsupported receivers,
    // descriptors, custom hooks, absent/boxed/float slots, and type-changing
    // values retain the original CallMayForceN unchanged.
    if ctx.is_authoritative_executor
        && original_call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::StoreAttr
    {
        if let (Some(&obj_opref), Some(&value_opref), Some(&code_opref), Some(&namei_opref)) =
            (r_args.first(), r_args.get(1), r_args.get(2), i_args.first())
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                // Deterministic immutable-type raise (`int.x = v`) folds to a
                // traced inline exception construction the optimizer can
                // virtualize — tried before the unboxed-slot store fold; the
                // two shapes are disjoint (type receiver vs instance receiver).
                if let Some(outcome) = try_walker_trace_immutable_type_attr_raise(
                    ctx,
                    op,
                    obj_opref,
                    Some(value_opref),
                    w_code_ptr,
                    namei as usize,
                )? {
                    return Ok(outcome);
                }
                if let Some(specialization) = try_walker_specialize_store_attr(
                    ctx,
                    op.pc,
                    obj_opref,
                    value_opref,
                    w_code_ptr,
                    namei as usize,
                    original_call_descr.get_extra_info(),
                )? {
                    match specialization {
                        WalkerStoreAttrSpecialization::Residual(
                            specialized_descr,
                            specialized_allboxes,
                        ) => {
                            descr = specialized_descr;
                            allboxes = specialized_allboxes;
                        }
                        WalkerStoreAttrSpecialization::Direct => {
                            fbw_mark_foriter_body_effect_since_consume();
                            fbw_bump_executed_effect();
                            return Ok((DispatchOutcome::Continue, op.next_pc));
                        }
                    }
                }
            }
        }
    }

    // DELETE_ATTR immutable-type raise fold — the deletion twin of the
    // STORE_ATTR fold above.  `bh_delete_attr_fn(obj, code, name_idx)`
    // carries no value operand: r_args = [obj, code], i_args = [name_idx].
    if ctx.is_authoritative_executor && pyre_helper_kind == majit_ir::PyreHelperKind::DeleteAttr {
        if let (Some(&obj_opref), Some(&code_opref), Some(&namei_opref)) =
            (r_args.first(), r_args.get(1), i_args.first())
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                if let Some(outcome) = try_walker_trace_immutable_type_attr_raise(
                    ctx,
                    op,
                    obj_opref,
                    None,
                    w_code_ptr,
                    namei as usize,
                )? {
                    return Ok(outcome);
                }
            }
        }
    }

    let call_descr = descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index,
        })?;

    let ei = call_descr.get_extra_info();
    // pyjitpl.py OS_NOT_IN_TRACE guard — see helper docstring
    // for the convergence rationale.
    if let Some(outcome) = do_not_in_trace_call_result(ei, op.pc)? {
        return Ok((outcome, op.next_pc));
    }
    // pyjitpl.py OS_JIT_FORCE_VIRTUAL fail-loud — see
    // `dispatch_residual_call_iRd_kind` for the rationale.
    do_jit_force_virtual_guard(ei, op.pc)?;

    // Method-form `CALL` helpers lower through the mixed int/ref residual
    // shape (`bh_call_fn_N(callable, null_or_self, args...)` carries the call
    // arity in the Int list).  Share the same user-function inline gate as the
    // plain Ref-only residual, but read the concrete Ref shadows from the
    // shifted R-list offset.
    if let Some(inlined) = try_walker_inline_builtin_call(
        ctx,
        op,
        code,
        1 + i_width,
        &r_args,
        ei.pyre_helper,
        dst_bank,
        dst,
    )? {
        return Ok(inlined);
    }

    if let Some(inlined) = try_walker_inline_user_call(
        ctx,
        op,
        code,
        1 + i_width,
        funcptr,
        &r_args,
        call_descr,
        ei.pyre_helper,
        dst_bank,
        dst,
    )? {
        return Ok(inlined);
    }

    // A class is not a `Function`, so the user-call route above declined it.
    if ei.pyre_helper == majit_ir::PyreHelperKind::CallFn {
        if let Some(inlined) =
            try_walker_inline_type_call(ctx, op, code, funcptr, &r_args, call_descr, dst_bank, dst)?
        {
            return Ok(inlined);
        }
    }

    // LoadConst fold: the LOAD_CONST helper (oopspec `LoadConst`, set
    // codewriter-side at flatten.rs
    // `build_residual_call_ir_r_single_ref_plain_insn_from_operands`)
    // re-materializes `co_consts[idx]` on every call.  When both the const
    // index (`i_args[0]`) and the code pointer (`r_args[0]`, the promoted
    // `frame.pycode`) are concrete, fold to the constant ref the call would
    // have produced — the indexed entry is loop-invariant — and suppress the
    // residual.  Falls through to the generic record when either operand is
    // not concrete (the residual stays correct in that case).
    if ei.pyre_helper == majit_ir::PyreHelperKind::LoadConst {
        if let (Some(&idx_opref), Some(&code_opref)) = (i_args.first(), r_args.first()) {
            if let (
                Some(majit_ir::Value::Int(consti)),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
            ) = (
                ctx.trace_ctx.box_value(idx_opref),
                ctx.trace_ctx.box_value(code_opref),
            ) {
                // Read the constant identically to the runtime
                // `bh_load_const_fn`: every constant is the shared
                // `pycode.co_consts_w[index]` object.
                let w_const = unsafe {
                    pyre_interpreter::pycode::w_code_const(
                        w_code_ptr as pyre_object::PyObjectRef,
                        consti as usize,
                    )
                };
                let const_box = ctx.trace_ctx.const_ref(w_const as i64);
                write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, const_box)?;
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
        }
    }

    // LoadGlobal fold (#62): mirror the trait LOAD_GLOBAL cell fast-path so
    // the optimizer hoists the module-global lookup out of the loop instead of
    // keeping the opaque CanRaise residual every iteration.  Requires namei
    // (i_args[0]), namespace (r_args[0]), promoted pycode (r_args[1]) and the
    // live frame (r_args[2]) concrete; the cell-strategy module-dict fast path
    // (name present in the executing frame's globals) and the builtins-cell
    // fast path (name absent from globals, resolved via `frame.get_builtin()`)
    // are both foldable.
    //
    // Handler-bearing bodies: `load_global_fn` is `CallFlavor::Plain` (can
    // raise NameError), so a residual lowering keeps a `GUARD_NO_EXCEPTION`
    // that a `catch_exception/L` in the same body may resume into.  The fold
    // emits the lookup as `ElidableCannotRaise` (the name resolves at trace
    // time and the version watchers fail the loop on any rebind/shadow), so a
    // SUCCESSFUL fold provably can't raise NameError and dropping that guard
    // for this load is sound — the handler can never be entered from it.  We
    // therefore attempt the fold even in handler-bearing bodies and keep the
    // residual (with its guard) only when the fold DECLINES.  The `B3`/builtin
    // raise+catch path needs this so the
    // `raise ValueError`/`except ValueError` class loads fold to const.
    //
    // The fold resolves the `co_names` index the same way
    // `bh_load_global_fn` does and reaches production parity for
    // global-function-call loops when combined with the user-call inlining
    // path. Handler-bearing reachability also includes the builtins fallback.
    if ctx.is_authoritative_executor && ei.pyre_helper == majit_ir::PyreHelperKind::LoadGlobal {
        if let (Some(&namei_opref), Some(&ns_opref), Some(&code_opref)) =
            (i_args.first(), r_args.first(), r_args.get(1))
        {
            if let (
                Some(majit_ir::Value::Int(namei)),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(ns_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
            ) = (
                ctx.trace_ctx.box_value(namei_opref),
                ctx.trace_ctx.box_value(ns_opref),
                ctx.trace_ctx.box_value(code_opref),
            ) {
                let name_idx = (namei as usize) >> 1;
                if !mark_trace_reads_module_global_from_code(
                    ctx.trace_ctx,
                    ns_ptr as pyre_object::PyObjectRef,
                    w_code_ptr,
                    name_idx,
                ) {
                    ctx.trace_ctx.reads_module_global = true;
                }
            } else {
                ctx.trace_ctx.reads_module_global = true;
            }
        } else {
            ctx.trace_ctx.reads_module_global = true;
        }
    }
    if ctx.is_authoritative_executor && ei.pyre_helper == majit_ir::PyreHelperKind::LoadGlobal {
        if let (Some(&namei_opref), Some(&ns_opref), Some(&code_opref)) =
            (i_args.first(), r_args.first(), r_args.get(1))
        {
            // The live frame operand (r_args[2]) is needed for the builtins
            // fallback (`frame.get_builtin()`); it may be absent/unseeded
            // (an inlined callee's `portal_frame_reg`), in which case only
            // the module-dict cell path is attempted.
            let frame_ptr = r_args
                .get(2)
                .and_then(|&f| ctx.trace_ctx.box_value(f))
                .and_then(|v| match v {
                    majit_ir::Value::Ref(majit_ir::GcRef(p)) => Some(p),
                    _ => None,
                })
                .unwrap_or(0);
            if let (
                Some(majit_ir::Value::Int(namei)),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(ns_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
            ) = (
                ctx.trace_ctx.box_value(namei_opref),
                ctx.trace_ctx.box_value(ns_opref),
                ctx.trace_ctx.box_value(code_opref),
            ) {
                if try_walker_load_global_cell_fold(
                    ctx, op.pc, dst, dst_bank, ns_ptr, w_code_ptr, frame_ptr, namei,
                )? {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    // LoadName fold: module-scope LOAD_NAME mirror of the LoadGlobal fold
    // above.  The residual is `bh_load_name_fn(frame, w_name, namei)`, so
    // r_args = [frame, w_name].  `try_walker_load_name_cell_fold` gates module
    // scope at runtime and routes non-module frames back to this residual.
    if ctx.is_authoritative_executor && ei.pyre_helper == majit_ir::PyreHelperKind::LoadName {
        if let (Some(&frame_opref), Some(&name_opref)) = (r_args.first(), r_args.get(1)) {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_name_ptr))),
            ) = (
                ctx.trace_ctx.box_value(frame_opref),
                ctx.trace_ctx.box_value(name_opref),
            ) {
                if !mark_trace_reads_module_global_from_frame_name(
                    ctx.trace_ctx,
                    frame_ptr,
                    w_name_ptr,
                ) {
                    ctx.trace_ctx.reads_module_global = true;
                }
            } else {
                ctx.trace_ctx.reads_module_global = true;
            }
        } else {
            ctx.trace_ctx.reads_module_global = true;
        }
    }

    // The handler-free gate is NOT the raise argument the LoadGlobal fold above
    // rebuts.  That argument transfers cleanly: both folds lower through
    // `emit_namespace_cell_fold`, which fires only on a name already PRESENT in
    // the module dict and watches that slot with `QUASIIMMUT_FIELD` +
    // `GUARD_NOT_INVALIDATED`, so a rebind or a delete fails the loop instead of
    // reaching a NameError, and a successful fold provably cannot raise.
    //
    // The gate is load-bearing because the fold INSTALLS the module dict's
    // `version?` watcher, and that watcher is what makes a later bumping write
    // in the same program abandon the walk with `ForceQuasiImmutable`.  That
    // abort resumes mid-expression off a latched operand mirror, and when one
    // latched slot is unbound the flush declines to the legacy replay, which
    // then REFUSES to re-deliver an in-flight FOR_ITER item once a body effect
    // has committed — the iteration is dropped and its accumulator increment is
    // silently lost.  Lifting the gate to `DELETE_NAME`/`DELETE_GLOBAL` only
    // (the implicit `del e` an `except X as e:` emits) reaches exactly that:
    // `bench/synth/pickle_terminal_raise_resume` then prints 214 under the JIT
    // against 216 interpreted, off one dropped iteration.  Both halves of that
    // chain — the unbound mirror slot and the silent drop the decline falls
    // back to — have to be closed before the handler shape stops standing in.
    //
    // The scan is whole-body, so one `try` anywhere in a module also charges
    // every name access in it a live dict lookup (~83ns each, linear in the
    // count).  Measured on a 200k-iteration
    // `bench/synth/exc_info_module_loop_hot`, folding it is worth 1.87us ->
    // 1.32us per iteration; at the fixture's own 3000 iterations the difference
    // sits under this box's noise floor.
    if ctx.is_authoritative_executor
        && ei.pyre_helper == majit_ir::PyreHelperKind::LoadName
        && !jitcode_has_exception_handler(code)
    {
        if let (Some(&frame_opref), Some(&name_opref)) = (r_args.first(), r_args.get(1)) {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr))),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_name_ptr))),
            ) = (
                ctx.trace_ctx.box_value(frame_opref),
                ctx.trace_ctx.box_value(name_opref),
            ) {
                if try_walker_load_name_cell_fold(ctx, op.pc, dst, dst_bank, frame_ptr, w_name_ptr)?
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    // LoadAttr fold (`mapdict.py LOAD_ATTR_caching`): fold a
    // monomorphic plain instance-attribute read to `guard_class` +
    // `guard_value(map)` + `getfield(storage)` + `getarrayitem(C_index)`,
    // eliding the opaque `getattr_fn` MRO-walk residual.  The residual is
    // `load_attr_fn(obj, code, name_idx)`, so `r_args = [obj, code]` and
    // `i_args = [name_idx]`.  A successful fold provably cannot raise (the map
    // guard proves the attribute is present on this shape), so it is attempted
    // even in handler-bearing bodies; every unfoldable shape falls through to
    // the residual (which keeps its exception guard).
    if ctx.is_authoritative_executor && ei.pyre_helper == majit_ir::PyreHelperKind::LoadAttr {
        if let (Some(&obj_opref), Some(&code_opref), Some(&namei_opref)) =
            (r_args.first(), r_args.get(1), i_args.first())
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                if try_walker_specialize_load_attr(
                    ctx,
                    op.pc,
                    obj_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                // The plain-slot fold declines a `property` (data descriptor);
                // inline its Python getter instead of the opaque residual.
                if let Some(inlined) = try_walker_inline_property_get(
                    ctx,
                    op,
                    code,
                    &r_args,
                    call_descr,
                    obj_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )? {
                    return Ok(inlined);
                }
            }
        }
    }
    // STORE_ATTR property setter: the plain-slot store fold above declines a
    // `property` data descriptor; inline its Python setter instead of the
    // opaque `setattr` residual.  `store_attr_fn` r_args = [obj, value, code].
    if ctx.is_authoritative_executor && ei.pyre_helper == majit_ir::PyreHelperKind::StoreAttr {
        if let (Some(&obj_opref), Some(&value_opref), Some(&code_opref), Some(&namei_opref)) =
            (r_args.first(), r_args.get(1), r_args.get(2), i_args.first())
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                if let Some(inlined) = try_walker_inline_property_set(
                    ctx,
                    op,
                    code,
                    &r_args,
                    call_descr,
                    obj_opref,
                    value_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )? {
                    return Ok(inlined);
                }
            }
        }
    }
    if ctx.is_authoritative_executor
        && ei.pyre_helper == majit_ir::PyreHelperKind::LoadAttr
        && next_op_is_load_method_self_for_attr(code, op, ctx, dst)
    {
        if let (Some(&obj_opref), Some(&code_opref), Some(&namei_opref)) =
            (r_args.first(), r_args.get(1), i_args.first())
        {
            if let (
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
                Some(majit_ir::Value::Int(namei)),
            ) = (
                ctx.trace_ctx.box_value(code_opref),
                ctx.trace_ctx.box_value(namei_opref),
            ) {
                if try_walker_specialize_load_method_attr(
                    ctx,
                    op.pc,
                    obj_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                // `Type.cmethod(...)`: the receiver is a class and the name
                // resolves to a `classmethod`, which `load_method_fast_path`
                // declines (non-instance receiver, non-method descriptor).
                // Write the classmethod's `__func__` so the paired
                // `load_method_self` binds the class and the CALL inlines it.
                if try_walker_specialize_load_classmethod_attr(
                    ctx,
                    op.pc,
                    obj_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                // The `[w_descr, w_obj]` push above is restricted to
                // `flag_method_descriptor` types, so a builtin method on a
                // builtin receiver (`lst.append`) leaves `getattr` to build a
                // `Method`.  Emit that construction instead of the opaque
                // residual so it virtualizes into the following CALL.
                if try_walker_specialize_load_bound_method_attr(
                    ctx,
                    op.pc,
                    obj_opref,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }
    if ctx.is_authoritative_executor && ei.pyre_helper == majit_ir::PyreHelperKind::LoadMethodSelf {
        if let (Some(&namei_opref), Some(&obj_opref), Some(&attr_opref), Some(&code_opref)) =
            (i_args.first(), r_args.first(), r_args.get(1), r_args.get(2))
        {
            let r_len_pc = op.pc + 1 + 1 + i_width;
            let attr_reg = code
                .get(r_len_pc + 1 + 1)
                .copied()
                .map(usize::from)
                .unwrap_or(usize::MAX);
            if let (
                Some(majit_ir::Value::Int(namei)),
                Some(majit_ir::Value::Ref(majit_ir::GcRef(w_code_ptr))),
            ) = (
                ctx.trace_ctx.box_value(namei_opref),
                ctx.trace_ctx.box_value(code_opref),
            ) {
                if try_walker_fold_load_method_self(
                    ctx,
                    op.pc,
                    obj_opref,
                    attr_opref,
                    attr_reg,
                    w_code_ptr,
                    namei as usize,
                    dst,
                    dst_bank,
                )?
                .is_some()
                {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
            }
        }
    }

    replace_movable_load_global_namespace_with_frame_globals(ctx, ei, &mut allboxes);

    // Defer the arg-bound check past the short-circuiting LoadConst /
    // LoadGlobal folds above: each resolves the call to a constant from
    // `i_args`/`r_args` without recording it, so an unbound *trailing* arg
    // is irrelevant when the call folds away.  In particular an inlined
    // callee's `load_global` passes its OWN unseeded `portal_frame_reg`
    // (Path-1, #68); the fold elides that call, so the frame box never
    // needs binding.  Only a call that survives to a genuine record
    // (BoxInt exec, generic residual below) requires every box bound.
    ensure_residual_call_args_bound(&allboxes, op.pc)?;

    // BoxInt fold (#62): `box_int_fn(raw)` allocates a fresh `PyLong`.  The
    // opaque CanRaise residual the generic leg would record blocks the
    // optimizer (no DCE of an unused/round-tripped box).  Emit the
    // virtualizable `new_with_vtable` + `setfield_gc` form (`wrapint`,
    // identical to the BINARY_OP result box) so a following unbox
    // (`getfield_gc_pure`) forwards through the setfield and the box DCEs
    // when it never escapes.  The concrete shadow carries the authentic
    // boxed pointer so downstream specializations still see a concrete int.
    if ei.pyre_helper == majit_ir::PyreHelperKind::BoxInt && dst_bank == 'r' {
        if let Some(&raw_arg) = i_args.first() {
            if let Some(boxed_ptr) = walker_execute_may_force_boxed(ctx, &allboxes, call_descr) {
                let intval =
                    unsafe { pyre_object::w_int_get_value(boxed_ptr as pyre_object::PyObjectRef) };
                let boxed = walker_box_int(ctx, op.pc, raw_arg, intval)?;
                ctx.trace_ctx
                    .set_opref_concrete(boxed, box_int_concrete(intval, boxed_ptr));
                write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, boxed)?;
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
        }
    }

    // #57: speculative int specialization for the BINARY_OP / COMPARE_OP
    // helper (oopspec `BinaryOp` / `CompareOp`, set codewriter-side at
    // flatten.rs `build_residual_call_ir_r_insn_from_operands`).  When both
    // operands are concrete `W_IntObject`, re-emit the guard_class + unbox
    // + int_OP (+ rebox / bool-box) sequence instead of an opaque
    // CALL_MAY_FORCE, matching the retired trait-side int binop / compare
    // paths.  Falls through to the generic record for
    // non-int operands / deferred operators.
    if matches!(
        ei.pyre_helper,
        majit_ir::PyreHelperKind::BinaryOp | majit_ir::PyreHelperKind::CompareOp
    ) {
        if let Some(&tag_opref) = i_args.first() {
            if let Some(majit_ir::Value::Int(op_tag)) = ctx.trace_ctx.box_value(tag_opref) {
                let specialized = if ei.pyre_helper == majit_ir::PyreHelperKind::BinaryOp {
                    let is_subscr = matches!(
                        pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag),
                        Some(pyre_interpreter::bytecode::BinaryOperator::Subscr)
                    );
                    if is_subscr {
                        // A user-instance receiver resolves `__getitem__` on
                        // its own type; inline that body instead of leaving
                        // the subscript an opaque residual.  The storage folds
                        // below are for builtin containers, which this
                        // declines.
                        if let Some(inlined) = try_walker_inline_subscr_getitem(
                            ctx, op, code, funcptr, &r_args, call_descr, dst, dst_bank,
                        )? {
                            return Ok(inlined);
                        }
                        // BINARY_SUBSCR list[int] getitem (int/float storage);
                        // falls through to the generic may-force leg otherwise.
                        try_walker_specialize_subscr(
                            ctx, op.pc, &r_args, &allboxes, call_descr, dst, dst_bank,
                        )?
                    } else {
                        // int specialization first; float (incl. mixed int/float)
                        // as a fallback so two-int operands keep int arithmetic.
                        let mut specialized = try_walker_specialize_binary_op_int(
                            ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                        )?;
                        if specialized.is_none() {
                            // longobject.py `_make_generic_descr_binop` and
                            // `descr_sub` use the rbigint.int_* family for
                            // mixed Long/Int operands.
                            specialized = try_walker_specialize_binary_op_long_int(
                                ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                            )?;
                        }
                        if specialized.is_none() {
                            // `_make_descr_binop` gives shifts with an Int
                            // count their own `_int_lshift` / `_int_rshift`
                            // path before Long/Long.
                            specialized = try_walker_specialize_binary_op_long_int_shift(
                                ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                            )?;
                        }
                        if specialized.is_none() {
                            // `_int_floordiv` / `_int_mod` are the same family:
                            // an Int divisor keeps its machine word instead of
                            // being widened to a bigint, and `_int_mod`'s
                            // result is a machine int rather than a long.
                            specialized = try_walker_specialize_binary_op_long_int_div(
                                ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                            )?;
                        }
                        if specialized.is_none() {
                            // `descr_pow` keeps a `W_IntObject` exponent
                            // unwrapped and calls `rbigint.int_pow`; only a
                            // long exponent reaches `rbigint.pow`.
                            specialized = try_walker_specialize_binary_op_long_int_pow(
                                ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                            )?;
                        }
                        if specialized.is_none() {
                            // W_LongObject operands take the long fast path
                            // before float so bigint arithmetic retains its
                            // payload representation.
                            specialized = try_walker_specialize_binary_op_long(
                                ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                            )?;
                        }
                        if specialized.is_none() {
                            // Two-long true-divide → float fast path
                            // (CallPureF + wrapfloat).
                            specialized = try_walker_specialize_truediv_op_long(
                                ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                            )?;
                        }
                        if specialized.is_none() {
                            specialized = try_walker_specialize_binary_op_float(
                                ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                            )?;
                        }
                        specialized
                    }
                } else if op_tag == 10 && ctx.is_authoritative_executor {
                    // B3: `op_tag == 10` is CHECK_EXC_MATCH
                    // (`bh_compare_fn(exc, match_type, 10)`,
                    // `call_jit.rs`).  Fold the match concretely to a
                    // const bool (the immortal TRUE/FALSE singleton) so the
                    // exception stays virtual and DCEs, eliding the may-force
                    // compare + its truth-extract residual.  Declines (falls
                    // through to the int/float compare attempts, which also
                    // decline for Ref operands → generic residual) when an
                    // operand has no concrete shadow or the match target is
                    // not a valid exception class.
                    let keep_last_exc_for_handler =
                        reads_last_exc_before_next_catch(code, op.next_pc);
                    let folded =
                        try_walker_fold_check_exc_match(ctx, op.pc, &r_args, dst, dst_bank)?;
                    if folded.is_some() && keep_last_exc_for_handler {
                        ctx.last_exc_value = saved_last_exc_value;
                        ctx.last_exc_value_concrete = saved_last_exc_value_concrete;
                    }
                    folded
                } else if (op_tag == 8 || op_tag == 9) && ctx.is_authoritative_executor {
                    // `op_tag` 8 / 9 are `is` / `is_not` (`IS_OP`), which the
                    // codewriter routes through the same `compare_fn`
                    // residual as the six ordinary comparisons.  Fold them to
                    // `ptr_eq` / `ptr_ne` — or, for a self-compare, straight
                    // to the constant — so identity tests stop paying a
                    // may-force call and its `GuardNotForced` per iteration.
                    // Declines (falls through to the generic residual) for an
                    // operand layout whose class compares `is_w` by value.
                    try_walker_fold_is_op(ctx, op.pc, op_tag, &r_args, dst, dst_bank)?
                } else {
                    // int compare first; then long (two-bigint operands keep
                    // bigint comparison); float (incl. mixed int/float) last.
                    match try_walker_specialize_compare_op_int(
                        ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                    )? {
                        Some(()) => Some(()),
                        None => match try_walker_specialize_compare_op_long_int(
                            ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                        )? {
                            Some(()) => Some(()),
                            None => match try_walker_specialize_compare_op_long(
                                ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst, dst_bank,
                            )? {
                                Some(()) => Some(()),
                                None => try_walker_specialize_compare_op_float(
                                    ctx, op.pc, op_tag, &r_args, &allboxes, call_descr, dst,
                                    dst_bank,
                                )?,
                            },
                        },
                    }
                };
                if specialized.is_some() {
                    return Ok((DispatchOutcome::Continue, op.next_pc));
                }
                if ei.pyre_helper == majit_ir::PyreHelperKind::BinaryOp {
                    if let Some(inlined) = try_walker_inline_user_binop(
                        ctx, op, code, op_tag, &r_args, call_descr, dst, dst_bank,
                    )? {
                        return Ok(inlined);
                    }
                }
                if ei.pyre_helper == majit_ir::PyreHelperKind::CompareOp {
                    if let Some(inlined) = try_walker_inline_user_compareop(
                        ctx, op, code, op_tag, &r_args, call_descr, dst, dst_bank,
                    )? {
                        return Ok(inlined);
                    }
                }
            }
        }
    }

    // UNPACK_SEQUENCE fold (#73): read an arity-2 specialised int tuple's
    // elements directly so the unpacked items stay unboxed ints (the trait
    // path's value0/value1 fold).  A non-foldable shape falls through to the
    // opaque residual below — correct, no decline.
    if matches!(
        ei.pyre_helper,
        majit_ir::PyreHelperKind::UnpackSequence | majit_ir::PyreHelperKind::UnpackItem
    ) && try_walker_specialize_unpack(
        ctx,
        op.pc,
        ei.pyre_helper,
        &i_args,
        &r_args,
        &allboxes,
        call_descr,
        dst,
        dst_bank,
    )?
    .is_some()
    {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

    // pyjitpl.py forces-branch sub-case: route release-gil through
    // `direct_call_release_gil`.  Mirrors `dispatch_residual_call_iRd_kind`.
    if ei.is_call_release_gil() {
        if let Some(outcome) = direct_call_release_gil(
            ctx,
            ei,
            &allboxes,
            descr.clone(),
            call_descr,
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iIRd_kind",
        )? {
            return Ok((outcome, op.next_pc));
        }
    } else if let Some(cached) = loopinvariant_lookup(ctx, ei, descr_key, funcptr) {
        // pyjitpl.py EF_LOOPINVARIANT short-circuit; no IR
        // op, no guard, ordering moot.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, cached)?;
    } else {
        let (call_opcode, can_raise, emit_guard_not_forced) =
            select_residual_call_opcode(ei, dst_bank, "dispatch_residual_call_iIRd_kind");
        walker_abort_if_mayforce_null_ref_arg(call_opcode, &allboxes, call_descr, ctx, op.pc)?;

        // pyjitpl.py `vable_and_vrefs_before_residual_call` —
        // records FORCE_TOKEN + SETFIELD_GC IR; runtime heap mutations
        // and the after-call helpers run in the residual-call execution path.
        // See `dispatch_residual_call_iRd_kind` for the upstream-citation
        // walkthrough.
        if emit_guard_not_forced {
            maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
        }

        if matches!(
            call_opcode,
            OpCode::CallI
                | OpCode::CallR
                | OpCode::CallF
                | OpCode::CallN
                | OpCode::CallPureI
                | OpCode::CallPureR
                | OpCode::CallPureF
                | OpCode::CallPureN
                | OpCode::CallLoopinvariantI
                | OpCode::CallLoopinvariantR
                | OpCode::CallLoopinvariantF
                | OpCode::CallLoopinvariantN
        ) {
            ctx.trace_ctx
                .profiler()
                .count_ops(call_opcode, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(call_opcode, majit_metainterp::counters::RECORDED_OPS);
        }
        let recorded = ctx
            .trace_ctx
            .record_op_with_descr(call_opcode, &allboxes, descr.clone());

        // pyjitpl.py `_record_helper_pure` parity — see
        // `dispatch_residual_call_iRd_kind` for the upstream walk.
        try_fold_pure_call_via_executor(ctx, call_opcode, &allboxes, call_descr, recorded);

        // Non-elidable concrete-execute parity — see
        // `dispatch_residual_call_iRd_kind` for the full citation.
        // PyPy-orthodox activation.  PyPy's
        // `_opimpl_residual_call*` concrete-executes EVERY residual
        // call regardless of EI; the `exc` flag only selects the
        // post-call guard shape (`GUARD_EXCEPTION` vs `GUARD_NO_EXCEPTION`)
        // in `handle_possible_exception`.  Pyre matches by always
        // invoking the executor — `try_execute_residual_call_via_executor`
        // self-gates on a fnaddr-sanity check (rejecting unpatched
        // `symbolic_fnaddr_for_path` hashes whose bits ≥ 47 are set)
        // so unregistered helpers degrade gracefully to recording-only
        // instead of SIGBUSing.
        let resid_exec = try_execute_residual_call_via_executor(
            ctx,
            call_opcode,
            &allboxes,
            call_descr,
            recorded,
            op.pc,
            Some((op.next_pc, dst_bank, dst)),
        )?;
        // A decline leaves the call recorded symbolically WITHOUT running
        // it — a side effect only the legacy replay applies, so the
        // walk-end no-replay commit must stay off for this trace (see
        // `fbw_has_unjournaled_effect`).  Pure/elidable calls never reach
        // this dispatcher (they fold via the pure-call executor).
        let resid_raised = match resid_exec {
            ResidualExecOutcome::Executed(result) => result.is_err(),
            ResidualExecOutcome::Declined(cause) => {
                fbw_abort_nested_unjournaled_residual(ctx, op.pc)?;
                fbw_mark_unjournaled_effect(cause);
                false
            }
        };
        debug_assert!(
            !resid_raised || can_raise,
            "dispatch_residual_call_iIRd_kind: helper raised on a \
             `!can_raise` EI — EffectInfo claim/reality mismatch"
        );

        // `_record_helper_varargs` parity — see
        // `dispatch_residual_call_iRd_kind` for the upstream-citation
        // walkthrough.  Same invalidation semantics; only the
        // arglist construction differs (boxes2 = i_args ++ r_args).
        ctx.trace_ctx
            .heapcache_invalidate_caches_varargs(call_opcode, Some(ei), &allboxes);
        // pyjitpl.py _opimpl_residual_call*: result writeback runs
        // BEFORE handle_possible_exception().  See
        // `dispatch_residual_call_iRd_kind` for the full citation.
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
        if emit_guard_not_forced {
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        if can_raise {
            if resid_raised {
                walker_record_guard_exception(ctx, op.pc);
                // pyjitpl.py `handle_possible_exception`
                // routes the raising branch through
                // `finishframe_exception()` immediately after emitting
                // GUARD_EXCEPTION — see iRd_kind for the full rationale.
                let exc = ctx
                    .last_exc_value
                    .expect("resid_raised implies last_exc_value seeded by the Err branch");
                let exc_concrete = ctx.last_exc_value_concrete;
                return Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc));
            } else {
                ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
                // The mixed int/ref residual-call shape must use the same
                // exception-region resume as the one-ref shape above.  In
                // particular, UNPACK_SEQUENCE and UNPACK_EX pass their arity
                // as Int arguments and the sequence as a Ref argument.  If
                // their validation raises after a guard failure, resume at the
                // call's own catch so the enclosing handler sees the error;
                // the generic fallthrough may already be outside the covered
                // exception-table range.
                walker_capture_snapshot_for_last_guard_scoped(
                    ctx,
                    op.pc,
                    GuardCaptureScope {
                        residual_call_catch_resume: true,
                        ..GuardCaptureScope::default()
                    },
                )?;
            }
        }

        if !resid_raised {
            loopinvariant_now_known(ctx, ei, descr_key, funcptr, recorded);
        }
    }

    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `residual_call` shape `iIRFd>X` dispatcher — `_irf_*` arglist with
/// int + ref + float lists before the descr. RPython parity:
/// `pyjitpl.py _opimpl_residual_call3` (`@arguments` argspec
/// `"box", "boxes3", "descr", "orgpc"`) → same
/// `do_residual_or_indirect_call` body as `_call1` / `_call2`. The
/// `boxes3` argcode (`pyjitpl.py`) decodes three adjacent
/// count-prefixed lists into one concatenated `argboxes` array
/// `[i_args..., r_args..., f_args...]`. `_build_allboxes`
/// (`pyjitpl.py`, ported to [`build_allboxes`]) permutes
/// those to match `descr.get_arg_types()` ABI ordering.
///
/// Operand layout `iIRFd>X`:
///   1B funcptr (i) + 1B i-list count + N×1B i-regs + 1B r-list count
///   + M×1B r-regs + 1B f-list count + K×1B f-regs + 2B descr + 1B
///   `>X` dst.
///
/// EffectInfo classification + guard emission match
/// `dispatch_residual_call_iIRd_kind`; all sub-cases (release-gil,
/// loop-invariant, default) route through the same helpers
/// ([`select_residual_call_opcode`], [`direct_call_release_gil`],
/// [`loopinvariant_lookup`] / [`loopinvariant_now_known`]).
#[allow(non_snake_case)]
pub(crate) fn dispatch_residual_call_iIRFd_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // execute_varargs (pyjitpl.py) clear_exception at every
    // residual-call entry; see dispatch_residual_call_iRd_kind.
    ctx.last_exc_value = None;
    ctx.last_exc_value_concrete = ConcreteValue::Null;
    let funcptr = read_int_reg(code, op, 0, ctx)?;
    let (i_args, i_width) = read_int_var_list(code, op, 1, ctx)?;
    let (r_args, r_width) = read_ref_var_list(code, op, 1 + i_width, ctx)?;
    let (f_args, f_width) = read_float_var_list(code, op, 1 + i_width + r_width, ctx)?;
    let descr_offset = 1 + i_width + r_width + f_width;
    let descr_index = decode_descr_index(code, op, descr_offset);
    let descr = read_descr(code, op, descr_offset, ctx)?;
    let call_descr = descr
        .as_call_descr()
        .ok_or(DispatchError::ResidualCallDescrNotCallDescr {
            pc: op.pc,
            descr_index,
        })?;
    let descr_key = descr.index();
    // Void shape `_irf_v/iIRFd` (`pyjitpl.py opimpl_residual_call_irf_v =
    // _opimpl_residual_call3`) has no `>X` dst byte; see
    // `dispatch_residual_call_iRd_kind` for the void operand-layout note.
    let dst = if dst_bank == 'v' {
        0
    } else {
        code[op.pc + 1 + descr_offset + 2] as usize
    };

    // Flat argboxes = i_args ++ r_args ++ f_args (`boxes3` argcode order).
    let mut argboxes: Vec<OpRef> = Vec::with_capacity(i_args.len() + r_args.len() + f_args.len());
    let mut argbox_types: Vec<Type> =
        Vec::with_capacity(i_args.len() + r_args.len() + f_args.len());
    argboxes.extend_from_slice(&i_args);
    argbox_types.extend(std::iter::repeat(Type::Int).take(i_args.len()));
    argboxes.extend_from_slice(&r_args);
    argbox_types.extend(std::iter::repeat(Type::Ref).take(r_args.len()));
    argboxes.extend_from_slice(&f_args);
    argbox_types.extend(std::iter::repeat(Type::Float).take(f_args.len()));
    let allboxes = build_allboxes(funcptr, &argboxes, &argbox_types, call_descr.arg_types());
    ensure_residual_call_args_bound(&allboxes, op.pc)?;

    let ei = call_descr.get_extra_info();
    clear_walk_exception(ctx);
    if let Some(outcome) = do_not_in_trace_call_result(ei, op.pc)? {
        return Ok((outcome, op.next_pc));
    }
    // pyjitpl.py OS_JIT_FORCE_VIRTUAL fail-loud — see
    // `dispatch_residual_call_iRd_kind` for the rationale.
    do_jit_force_virtual_guard(ei, op.pc)?;

    if ei.is_call_release_gil() {
        if let Some(outcome) = direct_call_release_gil(
            ctx,
            ei,
            &allboxes,
            descr.clone(),
            call_descr,
            dst_bank,
            dst,
            op.pc,
            "dispatch_residual_call_iIRFd_kind",
        )? {
            return Ok((outcome, op.next_pc));
        }
    } else if let Some(cached) = loopinvariant_lookup(ctx, ei, descr_key, funcptr) {
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, cached)?;
    } else {
        let (call_opcode, can_raise, emit_guard_not_forced) =
            select_residual_call_opcode(ei, dst_bank, "dispatch_residual_call_iIRFd_kind");
        walker_abort_if_mayforce_null_ref_arg(call_opcode, &allboxes, call_descr, ctx, op.pc)?;

        // pyjitpl.py `vable_and_vrefs_before_residual_call` —
        // records FORCE_TOKEN + SETFIELD_GC IR; runtime heap mutations
        // and the after-call helpers run in the residual-call execution path.
        // See `dispatch_residual_call_iRd_kind` for the upstream-citation
        // walkthrough.
        if emit_guard_not_forced {
            maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);
        }

        if matches!(
            call_opcode,
            OpCode::CallI
                | OpCode::CallR
                | OpCode::CallF
                | OpCode::CallN
                | OpCode::CallPureI
                | OpCode::CallPureR
                | OpCode::CallPureF
                | OpCode::CallPureN
                | OpCode::CallLoopinvariantI
                | OpCode::CallLoopinvariantR
                | OpCode::CallLoopinvariantF
                | OpCode::CallLoopinvariantN
        ) {
            ctx.trace_ctx
                .profiler()
                .count_ops(call_opcode, majit_metainterp::counters::OPS);
            ctx.trace_ctx
                .profiler()
                .count_ops(call_opcode, majit_metainterp::counters::RECORDED_OPS);
        }
        let recorded = ctx
            .trace_ctx
            .record_op_with_descr(call_opcode, &allboxes, descr.clone());

        // pyjitpl.py `_record_helper_pure` parity — see
        // `dispatch_residual_call_iRd_kind` for the upstream walk.
        try_fold_pure_call_via_executor(ctx, call_opcode, &allboxes, call_descr, recorded);
        // `boxes3`-shaped may-force residual (`CallMayForce{R,I,F,N}`):
        // execute concretely under the authoritative walk and stamp the
        // result, identically to the `iRd` / `iIRd` siblings.
        // `do_residual_call` (`pyjitpl.py _opimpl_residual_call3`)
        // is arglist-shape-independent, so the float-arg shape needs the same
        // execution path — e.g. a float-returning helper such as `math.sqrt`
        // records here as `iIRFd>f` (empty i/r lists), and its result must be
        // made concrete so the downstream float math can specialize.  Void
        // float-stores (`irf_v`) are caught inside the helper's
        // `result_type() == Void` arm and deferred (#61), so the compiled
        // loop's re-run does not double-apply the store.

        // Non-elidable concrete-execute parity — see
        // `dispatch_residual_call_iRd_kind` for the full citation.
        // PyPy-orthodox activation.  PyPy's
        // `_opimpl_residual_call*` concrete-executes EVERY residual
        // call regardless of EI; the `exc` flag only selects the
        // post-call guard shape (`GUARD_EXCEPTION` vs `GUARD_NO_EXCEPTION`)
        // in `handle_possible_exception`.  Pyre matches by always
        // invoking the executor — `try_execute_residual_call_via_executor`
        // self-gates on a fnaddr-sanity check (rejecting unpatched
        // `symbolic_fnaddr_for_path` hashes whose bits ≥ 47 are set)
        // so unregistered helpers degrade gracefully to recording-only
        // instead of SIGBUSing.
        let resid_exec = try_execute_residual_call_via_executor(
            ctx,
            call_opcode,
            &allboxes,
            call_descr,
            recorded,
            op.pc,
            Some((op.next_pc, dst_bank, dst)),
        )?;
        // A decline leaves the call recorded symbolically WITHOUT running
        // it — a side effect only the legacy replay applies, so the
        // walk-end no-replay commit must stay off for this trace (see
        // `fbw_has_unjournaled_effect`).  Pure/elidable calls never reach
        // this dispatcher (they fold via the pure-call executor).
        let resid_raised = match resid_exec {
            ResidualExecOutcome::Executed(result) => result.is_err(),
            ResidualExecOutcome::Declined(cause) => {
                fbw_abort_nested_unjournaled_residual(ctx, op.pc)?;
                fbw_mark_unjournaled_effect(cause);
                false
            }
        };
        debug_assert!(
            !resid_raised || can_raise,
            "dispatch_residual_call_iIRFd_kind: helper raised on a \
             `!can_raise` EI — EffectInfo claim/reality mismatch"
        );

        ctx.trace_ctx
            .heapcache_invalidate_caches_varargs(call_opcode, Some(ei), &allboxes);
        write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, recorded)?;
        if emit_guard_not_forced {
            ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
        if can_raise {
            if resid_raised {
                walker_record_guard_exception(ctx, op.pc);
                // pyjitpl.py `handle_possible_exception`
                // routes the raising branch through
                // `finishframe_exception()` immediately after emitting
                // GUARD_EXCEPTION — see iRd_kind for the full rationale.
                let exc = ctx
                    .last_exc_value
                    .expect("resid_raised implies last_exc_value seeded by the Err branch");
                let exc_concrete = ctx.last_exc_value_concrete;
                return Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc));
            } else {
                ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
                walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
            }
        }

        if !resid_raised {
            loopinvariant_now_known(ctx, ei, descr_key, funcptr, recorded);
        }
    }

    Ok((DispatchOutcome::Continue, op.next_pc))
}
