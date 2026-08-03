//! Public trace entrypoint for `pyre`'s JIT portal.
//!
//! `trace_bytecode` drives the authoritative full-body walk
//! (`full_body_walk_trace`): the walker-as-tracer that walks the per-CodeObject
//! JitCode body, combining symbolic IR recording
//! with the per-step concrete frame snapshot.  Any location the walk declines
//! re-interprets without JIT (the trait `PyreMetaInterp` interpret loop is
//! retired).

use majit_metainterp::{MetaInterp, TraceAction, TraceCtx};
use pyre_interpreter::CodeObject;

use crate::state::{PyreMeta, PyreSym, WalkSym};

struct ObjectSlotRoot {
    slot: *mut *mut u8,
    registered: bool,
}

impl ObjectSlotRoot {
    fn new(value: &mut pyre_object::PyObjectRef) -> Self {
        let slot = value as *mut pyre_object::PyObjectRef as *mut *mut u8;
        let registered = unsafe { pyre_object::gc_hook::try_gc_add_root(slot) };
        Self { slot, registered }
    }
}

impl Drop for ObjectSlotRoot {
    fn drop(&mut self) {
        if self.registered {
            pyre_object::gc_hook::try_gc_remove_root(self.slot);
        }
    }
}

/// Roots every element of an owned `Vec<PyObjectRef>` for the vector's
/// lifetime.  The `Vec` must not be resized while this is alive: the roots are
/// the element addresses, and a realloc would move them.
struct ObjectVecRoot {
    slots: Vec<*mut *mut u8>,
}

impl ObjectVecRoot {
    fn new(values: &mut [pyre_object::PyObjectRef]) -> Self {
        let mut slots = Vec::with_capacity(values.len());
        for value in values.iter_mut() {
            let slot = value as *mut pyre_object::PyObjectRef as *mut *mut u8;
            if unsafe { pyre_object::gc_hook::try_gc_add_root(slot) } {
                slots.push(slot);
            }
        }
        Self { slots }
    }
}

impl Drop for ObjectVecRoot {
    fn drop(&mut self) {
        for slot in self.slots.drain(..) {
            pyre_object::gc_hook::try_gc_remove_root(slot);
        }
    }
}

thread_local! {
    /// pyjitpl.py:3048-3091 `raise_continue_running_normally` seam: set
    /// when the authoritative full-body walk committed its end-of-walk
    /// frame state into the trace's concrete frame snapshot
    /// (`flush_walk_end_state_to_frame`).  The portal call sites consume
    /// it via [`take_walk_end_flush_committed`] to decide whether the
    /// returned `FrameBox` carries adoptable end state for the LIVE
    /// frame (no-replay) or still holds the entry state (legacy replay).
    static WALK_END_FLUSH_COMMITTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Which flush leg set [`WALK_END_FLUSH_COMMITTED`], recorded so the
    /// per-walk census can name it (the legs differ in what they resume at,
    /// so "committed" alone does not say whether the region re-runs).
    /// See `WalkEndCommitLeg`.
    static WALK_END_COMMIT_LEG: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    /// A no-handler exception produced by a committed rebuilt callee.  The
    /// portal consumes it as `LoopResult::Done(Err(..))`; keeping it separate
    /// from `ContinueRunningNormally` mirrors `_exit_frame_with_exception`.
    static WALK_END_PROPAGATED_EXCEPTION: std::cell::RefCell<Option<pyre_interpreter::PyError>> =
        const { std::cell::RefCell::new(None) };
    /// Stable address for the in-flight walk-end exception carrier above.
    ///
    /// RPython keeps this value in the translated MIFrame / exception object
    /// graph, which its root walker visits for every mutator. Pyre's
    /// trace→portal seam is genuinely per-thread, but its raw TLS is outside
    /// that graph, so the owning mutator publishes its address to the
    /// collector's STW root-area registry.
    static WALK_END_ROOT_AREA: WalkEndRootArea = WalkEndRootArea {
        propagated_exception: WALK_END_PROPAGATED_EXCEPTION.with(|value| value as *const _),
    };
    /// True at portal trace sites that can consume
    /// `WALK_END_PROPAGATED_EXCEPTION`. Bridge tracing leaves this false and
    /// conservatively retains its legacy preflight.
    static WALK_END_PROPAGATE_ALLOWED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Interpreter restart pc for a full-body walk that closes at a JitCode
    /// marker inside a Python opcode. The compiled-loop key stays at the
    /// merge point's green pc, but the interpreter fallback must resume at
    /// the opcode whose entry stack matches the restored live boxes.
    static WALK_END_RESTART_PC: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

struct WalkEndRootArea {
    propagated_exception: *const std::cell::RefCell<Option<pyre_interpreter::PyError>>,
}

/// Take-and-reset the walk-end flush flag for the trace that just
/// returned from [`trace_bytecode`].
pub fn take_walk_end_flush_committed() -> bool {
    WALK_END_FLUSH_COMMITTED.with(|c| c.replace(false))
}

/// The flush legs that can commit a walk's end state, in the order they are
/// tried in the epilogue.  Recorded per walk so the census can distinguish
/// them: they resume the interpreter at different pcs, and the ones that
/// resume at a CALL re-execute it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(crate) enum WalkEndCommitLeg {
    /// Loop-header end flush for a `CloseLoop`/`Terminate` walk.
    LoopHeader = 1,
    /// Force-time escape flush wrote the resume state into the live frame.
    VableEscape = 2,
    /// gh#467 CALL-forward: resume the OUTER frame AT the CALL that entered
    /// the aborting callee, re-executing the call from scratch.
    EntryCarrierCall = 3,
    /// gh#467 callee-rebuild: resume inside the rebuilt callee frame.
    CalleeRebuild = 4,
    /// Resume at the abort pc itself.
    AbortPc = 5,
    /// Nested-inline decline: resume the outer caller AT its CALL pc, taken
    /// from the framestack root's parent.
    NestedInlineOuterCall = 6,
    /// Kept-stack branch guard flush at the abort pc.
    BranchGuard = 7,
    /// Not a flush leg: the portal/CALL_ASSEMBLER `Terminate` no-replay
    /// shortcut.  It keeps the journal like the legs above, but by a different
    /// caller protocol — the caller consumes the walk's concrete result
    /// instead of adopting a resume pc — so it must NOT set
    /// [`WALK_END_FLUSH_COMMITTED`].  Tagged anyway because the census
    /// otherwise reports these walks as `committed=true leg=0`, which reads as
    /// "no leg" when it means "a commit path outside the leg contract".
    TerminateNoReplay = 8,
    /// `SwitchToBlackhole(ABORT_TOO_LONG)`: the post-step MIFrame image was
    /// converted to blackhole frames and run forward, never replayed from the
    /// trace entry.
    TraceTooLong = 9,
    /// A bridge carrier sub-walk stopped on a walker capability gap after
    /// concrete-executing the reconstructed callee: its frames were converted
    /// and run forward instead of the drain rewinding to the guard.
    CarrierAbort = 10,
    /// An aborting `DispatchError` from the walker capability-gap family
    /// ([`crate::jitcode_dispatch::DispatchError::leaves_complete_image`])
    /// whose mid-opcode MIFrame image was converted to blackhole frames and run
    /// forward.  Same contract as [`Self::TraceTooLong`]; kept a separate leg
    /// both so the census can tell a bounded-length abort from a capability
    /// gap, and because the two take their operand stack from different
    /// sources — see `capture_frame_stack_from_mirror`.
    WalkAbort = 11,
}

/// Where a committing leg puts the interpreter, relative to the effects the
/// walk already applied.
///
/// Committing does two things at once: it keeps the store journal (instead of
/// rolling it back) AND it hands the caller a resume pc.  Those two must agree,
/// so every leg has to say which side of its applied effects the resume pc
/// falls on.
///
/// Upstream's rule is not "never rewind".  `opimpl_str_guard_value`
/// (`pyjitpl.py:1498-1511`) runs a real `do_residual_call` and then records its
/// guard with `resumepc=orgpc`, an earlier pc; `capture_resumedata` stamps that
/// pc into the frame (`pyjitpl.py:2617-2620`).  What upstream forbids is
/// rewinding past an *effectful* residual, and it decides both the permission
/// and the prohibition STATICALLY — the licence is the codewriter-time
/// `EffectInfo.EF_ELIDABLE_CANNOT_RAISE` registration
/// (`jtransform.py:620-630`), and the ban is by opnum: the four guards that can
/// follow a residual take `after_residual_call`, which pins the resume pc to
/// the POST-call `self.pc` (`pyjitpl.py:2599-2602` → `194-198`).  Upstream's
/// only op counters are profiling-only and gate nothing
/// (`jitprof.py:43-44 EmptyProfiler.count_ops` is `pass`).
///
/// So this enum discharges an obligation upstream also has, but in a form
/// upstream does not use: a runtime odometer where upstream uses a declared
/// effect class.  That is a tracked deviation, not the end state — the
/// convergence is a static per-callee effect classification.  Separately, the
/// legs that resume the OUTER frame at a CALL (gh#467, gated by #126/#215's
/// missing inner-frame rebuild) have no upstream counterpart at all;
/// `convert_and_run_from_pyjitpl` (`blackhole.py:1799-1821`) gives every frame
/// its own current pc and splices the callee result in PAST the caller's call
/// (`blackhole.py:1653-1662`), which is the [`WalkEndResume::AfterApplied`]
/// shape.
#[derive(Clone, Copy)]
pub(crate) enum WalkEndResume {
    /// The resume pc is the walk terminal: nothing already applied lies ahead
    /// of it, and nothing behind it re-runs.  "Anywhere in the walk" and
    /// "behind the resume point" coincide here, which is why the sticky
    /// unjournaled flags a terminal leg consults are a sufficient gate.
    Terminal,
    /// The resume pc REWINDS — the interpreter re-runs a region the walk
    /// already ran, while the journal stays committed.  Sound only while the
    /// executed-effect odometer has not moved since `effects_at_resume_point`,
    /// which is sampled AT that pc; otherwise every effect applied since it
    /// runs a second time on top of the committed ones.  [`commit_walk_end`]
    /// enforces this and declines the leg, leaving the legacy path whose
    /// journal rollback makes the replay exactly-once.
    Rewind { effects_at_resume_point: usize },
    /// The resume pc rewinds and the leg has no sample to prove the odometer
    /// stayed put since it.  [`commit_walk_end`] always declines: an unproven
    /// rewind is exactly the shape that double-executes silently.  A leg that
    /// reaches this either has to start sampling its resume point or should
    /// not be committing at all.
    RewindUnproven,
    /// A rewind whose only proof is taken ELSEWHERE and EARLIER: the escape
    /// latch's `escape_opcode_window_clean` runs at the residual
    /// (`residual_call.rs`), not at this commit.  Named rather than spelled
    /// `Terminal` because the resume pc genuinely re-runs its opcode —
    /// `vstack_cur_pypc` is the pc the walk is ABOUT TO ENTER
    /// (`reconcile_vstack_at_boundary` sets it to `new_pypc` after reconciling
    /// the PREVIOUS opcode), and the flush sets `last_instr = pc - 1`
    /// (`state.rs`), so `next_instr()` re-executes that opcode.
    ///
    /// The proof is DECLARED, not measured: the window records each residual's
    /// `EffectInfo` re-runnability class (`EF_ELIDABLE_*` / `EF_LOOPINVARIANT`),
    /// the same axis upstream licenses `resumepc=orgpc` on
    /// (`jtransform.py:620-630`).  A commit-time counter sample would answer a
    /// different question — the forcing residual itself moves the odometer
    /// after the window is read — and upstream's op counters gate nothing
    /// (`jitprof.py:43-44`).
    RewindProvenAtLatch,
    /// The resume pc is AHEAD of what the walk applied — a rebuilt callee
    /// resumed at its own abort pc.  Nothing re-runs; committing is what
    /// *keeps* the applied effects, and rolling back would lose them (the
    /// discarded trace was their only carrier).  Needs no effect gate at all,
    /// which is why upstream's version of this leg is unconditional
    /// (`run_blackhole_interp_to_cancel_tracing` ends `assert False`,
    /// `pyjitpl.py:2956`).
    AfterApplied,
}

/// Set [`WALK_END_FLUSH_COMMITTED`] and record which leg did it.
///
/// Returns whether the commit was taken.  A [`WalkEndResume::Rewind`] leg whose
/// odometer moved since its resume point is declined here rather than
/// committed — the one gate that cannot be left to each leg, because a leg that
/// forgets it produces a silent double-execution rather than a crash.
/// Whether `resume` may be committed — the predicate [`commit_walk_end`]
/// applies.  Exposed separately because a leg whose flush mutates the live
/// frame before it can commit has to consult it FIRST: those flushes have no
/// undo, so declining after one would leave the frame half-adopted.
#[must_use]
pub(crate) fn walk_end_resume_provable(resume: WalkEndResume) -> bool {
    match resume {
        WalkEndResume::Terminal
        | WalkEndResume::AfterApplied
        | WalkEndResume::RewindProvenAtLatch => true,
        WalkEndResume::Rewind {
            effects_at_resume_point,
        } => crate::jitcode_dispatch::fbw_executed_effect_count() == effects_at_resume_point,
        WalkEndResume::RewindUnproven => false,
    }
}

/// Name the path that kept this walk's journal, for the census only.  Every
/// journal-keeping path goes through here so none stays anonymous; the flush
/// legs additionally set [`WALK_END_FLUSH_COMMITTED`] via [`commit_walk_end`],
/// which is what tells the portal the returned `FrameBox` carries adoptable end
/// state.  A path with a different caller protocol must tag WITHOUT that flag.
#[must_use]
pub(crate) fn record_walk_end_leg(leg: WalkEndCommitLeg, resume: WalkEndResume) -> bool {
    if !walk_end_resume_provable(resume) {
        return false;
    }
    WALK_END_COMMIT_LEG.with(|c| c.set(leg as u8));
    true
}

#[must_use]
pub(crate) fn commit_walk_end(leg: WalkEndCommitLeg, resume: WalkEndResume) -> bool {
    if !record_walk_end_leg(leg, resume) {
        return false;
    }
    WALK_END_FLUSH_COMMITTED.with(|c| c.set(true));
    true
}

pub fn take_walk_end_propagated_exception() -> Option<pyre_interpreter::PyError> {
    WALK_END_PROPAGATED_EXCEPTION.with(|c| c.borrow_mut().take())
}

/// Capture this mutator's raw walk-end carrier address for STW root walking.
pub fn capture_walk_end_root_area() -> *const () {
    WALK_END_ROOT_AREA.with(|area| area as *const _ as *const ())
}

/// Root the no-handler exception parked across the trace→portal boundary until
/// `take_walk_end_propagated_exception` consumes it. Forward its GC refs in
/// place without materialising the lazy-null `exc_object`.
pub fn walk_walk_end_propagated_exception(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let data = capture_walk_end_root_area();
    unsafe { walk_walk_end_roots_area(data, visitor) };
}

/// # Safety
/// `data` must come from [`capture_walk_end_root_area`], and the owning
/// mutator must be quiesced when a foreign collector thread calls this.
pub unsafe fn walk_walk_end_roots_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let area = unsafe { &*(data as *const WalkEndRootArea) };
    let exception_cell = unsafe { &*area.propagated_exception };
    // SAFETY: the owner is either synchronously collecting or STW-quiesced;
    // no Rust borrow is live while the collector forwards these raw slots.
    let opt = unsafe { &mut *exception_cell.as_ptr() };
    if let Some(err) = opt.as_mut() {
        err.walk_gc_refs(visitor);
    }
}

thread_local! {
    /// Raw pointer to the `PyreSym` being traced on this thread, or null when no
    /// trace is in flight. Lets [`walk_active_sym_exc_roots`] reach the
    /// trace-time exception carriers (`trace_built_exc` / `last_exc_value` /
    /// `current_exc_value`) during a collection triggered mid-trace. Set at
    /// [`trace_bytecode`] entry, restored on return.
    static ACTIVE_SYM_EXC: std::cell::Cell<*mut PyreSym> =
        const { std::cell::Cell::new(std::ptr::null_mut()) };
}

/// RAII guard restoring the previous [`ACTIVE_SYM_EXC`] on drop, so nested /
/// re-entrant `trace_bytecode` (recursive portal) unwinds to the outer trace's
/// sym rather than leaving a stale or null anchor.
pub(crate) struct ActiveSymExcGuard {
    prev: *mut PyreSym,
}

impl Drop for ActiveSymExcGuard {
    fn drop(&mut self) {
        ACTIVE_SYM_EXC.with(|c| c.set(self.prev));
    }
}

/// Publish `sym` as the active trace's exception-carrier anchor for the lifetime
/// of the returned guard. Called once at [`trace_bytecode`] entry.
pub(crate) fn set_active_sym_exc(sym: *mut PyreSym) -> ActiveSymExcGuard {
    let prev = ACTIVE_SYM_EXC.with(|c| c.replace(sym));
    ActiveSymExcGuard { prev }
}

/// Root the trace-time exception carriers held in the active `PyreSym`.
///
/// Between construction (`sym.trace_built_exc` insert, `state.rs`) and
/// lift-out (`swap_remove` at the raise), a trace-built exception is reachable
/// only through `sym.trace_built_exc`, invisible to the precise collector; an
/// allocating safepoint in that window would otherwise sweep it. `last_exc_value`
/// / `current_exc_value` cover the seeded-raise and reraise paths.
///
/// Mirrors `walk_jit_exc_value`: the carriers are oldgen-stable exceptions
/// (`try_gc_alloc_stable_raw`), so a bare mark-by-value suffices — no forwarded
/// write-back — which means only a shared read of the sym is taken, never a
/// second `&mut` aliasing the tracer's live borrow.
pub fn walk_active_sym_exc_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let sym_ptr = ACTIVE_SYM_EXC.with(|c| c.get());
    if sym_ptr.is_null() {
        return;
    }
    // SAFETY: a collection triggered mid-trace runs on the SAME thread (the
    // allocating thread becomes the collector), so the tracer's `&mut PyreSym`
    // up the stack is a suspended frame. Only a shared `&PyreSym` is formed and
    // oldgen-stable (non-moving) exceptions are marked by value, never writing a
    // forwarded pointer back — matching the accepted `jit_driver_pair_from_root_area`
    // convention in `pyre-jit`.
    let sym = unsafe { &*sym_ptr };
    let carriers = [sym.last_exc_value, sym.current_exc_value];
    for p in carriers
        .into_iter()
        .chain(sym.trace_built_exc.values().copied())
    {
        if p.is_null() {
            continue;
        }
        let mut gcref = majit_ir::GcRef(p as usize);
        visitor(&mut gcref);
        // The carrier is non-moving, so a minor's root visitor no-ops on it
        // and never reaches its fields: young children (tracebacks/args built
        // while tracing) would be left dangling across the minor. Forward the
        // raw child slots explicitly; the writes land in the exception
        // object, not the sym, so the shared-read contract above holds.
        unsafe { pyre_interpreter::eval::walk_raw_exception_roots(p, visitor) };
    }
}

pub fn take_walk_end_restart_pc() -> Option<usize> {
    WALK_END_RESTART_PC.with(|c| c.replace(None))
}

/// Copy the walk-accumulated `TraceCtx.reads_module_global` flag into the
/// trace's `PyreMeta.namespace_dependent` at every `trace_bytecode` return
/// path.  This is the sole authority for the finalized value; `build_meta`
/// seeds it `false` at trace start (the walk hasn't run yet) and the
/// entry-bridge fold ORs the live flag mid-walk, so `false` there is the OR
/// identity.  On mid-walk-compile close paths the tracing ctx was already
/// taken, so `trace_ctx()` is `None` and this no-ops exactly as before — the
/// value was folded into the entry bridge's meta by then.
fn finish_trace_namespace_dependency(meta: &mut MetaInterp<PyreMeta>) {
    let namespace_dependent = meta
        .trace_ctx()
        .map(|ctx| ctx.reads_module_global)
        .unwrap_or(false);
    if let Some(trace_meta) = meta.trace_meta_mut() {
        trace_meta.namespace_dependent = namespace_dependent;
    }
}

thread_local! {
    /// Green keys whose full-body walk deterministically re-reaches a
    /// structural walk decline that must skip future walks of the same entry.
    /// This set is intentionally narrow: transient walker capability gaps
    /// return a plain abort and rely on the normal hotness/abort machinery.
    static FBW_DECLINED_KEYS: std::cell::RefCell<std::collections::HashSet<u64>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
    /// FOR_ITER green keys whose range-only specialization observed a class
    /// mismatch.  This gates only the specialization; the full-body walk must
    /// still retrace and compile its generic residual.
    ///
    /// This is deliberately per-thread, like `FBW_DECLINED_KEYS` and the JIT
    /// driver itself.  A guard failure proves polymorphism only for the
    /// executing thread's trace; no cross-thread mutable JIT state is needed.
    static RANGE_FORITER_DEMOTED: std::cell::RefCell<std::collections::HashSet<u64>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
    /// The current bridge trace's full-body walk hit a deterministic
    /// structural decline.  The walker only knows `(w_code, start_pc)`; the
    /// bridge launcher still has the originating guard descr and consumes this
    /// bit to populate `MetaInterp::declined_bridge_guards`.
    static FBW_BRIDGE_DECLINED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn fbw_declined(key: u64) -> bool {
    FBW_DECLINED_KEYS.with(|s| s.borrow().contains(&key))
}

pub(crate) fn fbw_decline(key: u64) {
    FBW_DECLINED_KEYS.with(|s| {
        s.borrow_mut().insert(key);
    });
}

fn fbw_bridge_decline(ctx: &TraceCtx) {
    if ctx.is_bridge_trace {
        FBW_BRIDGE_DECLINED.with(|c| c.set(true));
    }
}

fn p2_drain_abort() -> TraceAction {
    TraceAction::Abort
}

pub fn take_fbw_bridge_declined() -> bool {
    FBW_BRIDGE_DECLINED.with(|c| c.replace(false))
}

pub(crate) fn range_foriter_demoted(key: u64) -> bool {
    RANGE_FORITER_DEMOTED.with(|s| s.borrow().contains(&key))
}

/// Demote a range FOR_ITER site on the first failure of its class guard —
/// a definitive polymorphism witness.
///
/// Returns `true` when this call performs the demotion (first class
/// mismatch), `false` when the site was already demoted (idempotent
/// re-failure).  `handle_fail` calls this only after confirming the failing
/// descr carries the range marker (`Descr::range_foriter_green_key`), so an
/// unrelated body guard at the same loop can never demote the site.
pub fn range_foriter_demote_once(green_key: u64) -> bool {
    RANGE_FORITER_DEMOTED.with(|s| s.borrow_mut().insert(green_key))
}

fn midbody_post_marker_is_effect_free(code: &CodeObject, start_pc: usize) -> bool {
    (start_pc..code.instructions.len()).all(|pc| {
        let Some((instruction, _)) = pyre_interpreter::decode_instruction_at(code, pc) else {
            return false;
        };
        matches!(
            instruction,
            pyre_interpreter::Instruction::Cache
                | pyre_interpreter::Instruction::ExtendedArg
                | pyre_interpreter::Instruction::Resume { .. }
                | pyre_interpreter::Instruction::Nop
                | pyre_interpreter::Instruction::NotTaken
                | pyre_interpreter::Instruction::LoadConst { .. }
                | pyre_interpreter::Instruction::LoadCommonConstant { .. }
                | pyre_interpreter::Instruction::LoadSmallInt { .. }
                | pyre_interpreter::Instruction::LoadFast { .. }
                | pyre_interpreter::Instruction::LoadFastBorrow { .. }
                | pyre_interpreter::Instruction::LoadFastCheck { .. }
                | pyre_interpreter::Instruction::LoadFastBorrowLoadFastBorrow { .. }
                | pyre_interpreter::Instruction::LoadFastLoadFast { .. }
                | pyre_interpreter::Instruction::StoreFast { .. }
                | pyre_interpreter::Instruction::StoreFastLoadFast { .. }
                | pyre_interpreter::Instruction::StoreFastStoreFast { .. }
                | pyre_interpreter::Instruction::PopTop
                | pyre_interpreter::Instruction::Copy { .. }
                | pyre_interpreter::Instruction::Swap { .. }
                | pyre_interpreter::Instruction::BinaryOp { .. }
                | pyre_interpreter::Instruction::CompareOp { .. }
                | pyre_interpreter::Instruction::IsOp { .. }
                | pyre_interpreter::Instruction::JumpForward { .. }
                | pyre_interpreter::Instruction::JumpBackward { .. }
                | pyre_interpreter::Instruction::JumpBackwardNoInterrupt { .. }
                | pyre_interpreter::Instruction::PopJumpIfFalse { .. }
                | pyre_interpreter::Instruction::PopJumpIfTrue { .. }
                | pyre_interpreter::Instruction::PopJumpIfNone { .. }
                | pyre_interpreter::Instruction::PopJumpIfNotNone { .. }
                | pyre_interpreter::Instruction::MatchMapping
                | pyre_interpreter::Instruction::MatchSequence
                | pyre_interpreter::Instruction::GetLen
                | pyre_interpreter::Instruction::UnpackSequence { .. }
                | pyre_interpreter::Instruction::ReturnValue
        )
    })
}

fn resolve_entry_carrier_call_py_pc(
    outer_jitcode_index: u32,
    call_jitcode_pc: usize,
) -> Option<usize> {
    let outer = crate::state::pyjitcode_for_jitcode_index(outer_jitcode_index as i32);
    outer.filter(|payload| !payload.code_ptr.is_null())?;
    Some(crate::py_coord::resume_py_pc_for_jitcode_word(
        outer_jitcode_index as i32,
        call_jitcode_pc as i32,
    ) as usize)
}

#[derive(Clone, Copy)]
struct MidBodyFlushWords {
    call_py_pc: usize,
    post_call_py_pc: usize,
    callee_py_pc: usize,
}

/// `PYRE_M73_MIDBODY_CARRY_AUDIT` asserts the forward-carried
/// `MidBodyPayload.callee_py_pc` equals the jitcode->py inversion at the
/// midbody flush boundary before the carry is trusted. Off in production; the
/// gated branch is the only added work.
fn midbody_carry_audit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_M73_MIDBODY_CARRY_AUDIT").is_some())
}

fn resolve_midbody_flush_words(
    payload: &crate::jitcode_dispatch::MidBodyPayload,
) -> Option<MidBodyFlushWords> {
    let outer = crate::state::pyjitcode_for_jitcode_index(payload.outer_jitcode_index as i32);
    let callee = crate::state::pyjitcode_for_jitcode_index(payload.callee_jitcode_index as i32);
    let outer = outer.filter(|payload| !payload.code_ptr.is_null())?;
    let callee = callee.filter(|payload| !payload.code_ptr.is_null())?;
    let call_py_pc = crate::py_coord::resume_py_pc_for_jitcode_word(
        payload.outer_jitcode_index as i32,
        payload.call_jitcode_pc as i32,
    ) as usize;
    // #73 walker-as-tracer P1: the callee resume py is read from the scalar
    // forward-carried on the MidBodyPayload (`callee_py_pc`, stamped at capture
    // from the same jitcode->py inversion this once performed). The `callee`
    // pjc lookup above stays for its null-code decline (`?`) and feeds the
    // gated audit, which asserts the carry still equals the live inversion.
    // The midbody callee-rebuild resume path is corpus-cold, so byte-parity
    // rests on encode/resume input identity (same immutable metadata + pc), not
    // on an exercised audit; the gate remains a permanent tripwire.
    if midbody_carry_audit_enabled() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static HITS: AtomicU64 = AtomicU64::new(0);
        let callee_py_pc_convert = crate::py_coord::containing_py_pc_for_jitcode_pc(
            &callee.metadata,
            payload.abort_jitcode_pc,
        ) as usize;
        if HITS.fetch_add(1, Ordering::Relaxed) == 0 {
            eprintln!(
                "[m73-midbody-carry-audit] first callee_py_pc comparison (carry={} convert={callee_py_pc_convert})",
                payload.callee_py_pc
            );
        }
        assert_eq!(
            payload.callee_py_pc, callee_py_pc_convert,
            "PYRE_M73_MIDBODY_CARRY_AUDIT: callee_py_pc carry diverged at jitcode {} pc {}",
            payload.callee_jitcode_index, payload.abort_jitcode_pc
        );
    }
    Some(MidBodyFlushWords {
        call_py_pc,
        post_call_py_pc: crate::jitcode_dispatch::skip_python_trivia_forward(
            unsafe { &*outer.code_ptr },
            call_py_pc + 1,
        ),
        callee_py_pc: payload.callee_py_pc,
    })
}

/// Whether the caller's handler for the aborting CALL can be entered from the
/// operand stack this leg can reconstruct.
///
/// `handle_exception` only ever POPS down to the handler's recorded depth
/// (`eval.rs`, `pyopcode.py:151-173`), so restoring the `below` operands the
/// CALL sat on is enough for any handler that wants at most that many.
fn exception_delivery_stack_is_sourceable(
    handler_depth: u32,
    below_len: usize,
    array_len: usize,
    stack_base: usize,
) -> bool {
    handler_depth as usize <= below_len && array_len >= stack_base + below_len + 1
}

/// Flush the OUTER frame at the CALL that entered the aborting callee and let
/// the interpreter re-execute that whole call.  Returns the committed
/// `call_py_pc`, or `None` when any step declined and the legacy replay stands.
///
/// This resume REWINDS to the CALL, which is sound only while nothing has been
/// applied since it — the latch was set under that gate, and it is re-checked
/// here before the flush mutates the live frame.  Reached either as the carrier
/// in its own right or as the callee-rebuild leg's fallback.
fn try_commit_entry_carrier_call(
    ctx: &TraceCtx,
    cf_addr: usize,
    abort_jit_pc: usize,
    outer_jitcode_index: u32,
    call_jitcode_pc: usize,
    call_stack: &[pyre_object::PyObjectRef],
    entry_executed_effects: usize,
) -> Option<usize> {
    let resume = WalkEndResume::Rewind {
        effects_at_resume_point: entry_executed_effects,
    };
    if !walk_end_resume_provable(resume) {
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-abort-flush] gh#467 CALL-forward declined at \
                 abort_jit_pc={abort_jit_pc} (executed-effect delta since the \
                 outer CALL) — legacy replay kept"
            );
        }
        return None;
    }
    let Some(call_py_pc) = resolve_entry_carrier_call_py_pc(outer_jitcode_index, call_jitcode_pc)
    else {
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-abort-flush] gh#467 CALL-forward declined at \
                 abort_jit_pc={abort_jit_pc} (unresolved outer \
                 jitcode_index={outer_jitcode_index} or null code ptr) — legacy replay kept"
            );
        }
        return None;
    };
    if !crate::state::flush_walk_end_state_at_outer_call(ctx, cf_addr, call_py_pc, call_stack) {
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-abort-flush] gh#467 CALL-forward declined at \
                 call_py_pc={call_py_pc} (depth mismatch / unresolved local / \
                 lastblock) — legacy replay kept"
            );
        }
        return None;
    }
    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
        eprintln!(
            "[fbw-abort-flush] gh#467 CALL-forward COMMIT abort_jit_pc={abort_jit_pc} \
             call_py_pc={call_py_pc} stack_depth={}",
            call_stack.len()
        );
    }
    let committed = commit_walk_end(WalkEndCommitLeg::EntryCarrierCall, resume);
    debug_assert!(committed, "provability re-checked after a pure flush");
    Some(call_py_pc)
}

/// Why a callee rebuild did not commit — and whether the callee body had
/// already executed when that was decided.
///
/// The distinction is load-bearing, not diagnostic.  The caller's fallback is
/// `EntryCarrierCall`, which rewinds the outer frame to its CALL; taking that
/// after `frame.execute_frame` has run the callee runs the callee a SECOND
/// time.  `walk_end_resume_provable` cannot catch it, because the odometer it
/// samples (`FBW_EXECUTED_EFFECT_COUNT`) is walker-side and the plain
/// interpretation inside `execute_frame` never bumps it.
enum MidBodyDecline {
    /// Refused before the callee ran; none of its effects are applied, so
    /// rewinding the caller to its CALL is still sound.
    BeforeRun(&'static str),
    /// The callee body already ran.  Its effects are applied and there is no
    /// journal undo for them, so no rewinding leg may be selected.
    AfterRun(&'static str),
}

impl MidBodyDecline {
    fn reason(&self) -> &'static str {
        match self {
            Self::BeforeRun(reason) | Self::AfterRun(reason) => reason,
        }
    }
}

/// Wrapper that names which narrowing kept a rebuild off leg 4; the reason
/// otherwise vanishes into the entry-carrier fallback.
fn try_commit_midbody_abort(
    ctx: &TraceCtx,
    cf_addr: usize,
    payload: &crate::jitcode_dispatch::MidBodyPayload,
    words: MidBodyFlushWords,
) -> Result<(), MidBodyDecline> {
    let outcome = try_commit_midbody_abort_inner(ctx, cf_addr, payload, words);
    if let Err(decline) = &outcome
        && crate::jitcode_dispatch::fbw_debug_abort_enabled()
    {
        eprintln!(
            "[fbw-abort-flush] gh#467 callee-rebuild NOT COMMITTED at callee_py_pc={} ({}){}",
            words.callee_py_pc,
            decline.reason(),
            match decline {
                MidBodyDecline::AfterRun(_) => " — callee already ran, no rewinding leg eligible",
                MidBodyDecline::BeforeRun(_) => "",
            },
        );
    }
    outcome
}

fn try_commit_midbody_abort_inner(
    ctx: &TraceCtx,
    cf_addr: usize,
    payload: &crate::jitcode_dispatch::MidBodyPayload,
    words: MidBodyFlushWords,
) -> Result<(), MidBodyDecline> {
    // An expression-position call sits on top of operands the payload does not
    // record — it counts only the call's own `[callable, null_or_self, args…]`.
    // The entry fallback's `reconstructed_all_ref_call_stack` is the caller's
    // WHOLE operand stack at that pc, slot-ordered from the stack base, so its
    // prefix is exactly that residue.
    let below = match crate::state::outer_call_operands_below(
        cf_addr,
        words.call_py_pc,
        words.post_call_py_pc,
        payload.call_stack_len,
    ) {
        Some(0) => &[][..],
        Some(n) => {
            let Some(full) = payload
                .entry_fallback
                .as_ref()
                .map(|fallback| fallback.call_stack.as_slice())
                .filter(|full| full.len() == n + payload.call_stack_len)
            else {
                return Err(MidBodyDecline::BeforeRun(
                    "expression-position call with no reconstructed stack below it",
                ));
            };
            &full[..n]
        }
        None => {
            return Err(MidBodyDecline::BeforeRun(
                "caller stack depth does not model this call shape",
            ));
        }
    };
    if !crate::state::can_flush_walk_end_state_after_outer_call(
        ctx,
        cf_addr,
        words.call_py_pc,
        words.post_call_py_pc,
        payload.call_stack_len,
        below,
    ) {
        return Err(MidBodyDecline::BeforeRun(
            "outer call boundary not flushable",
        ));
    }
    let raw = unsafe {
        pyre_interpreter::w_code_get_ptr(payload.w_code) as *const pyre_interpreter::CodeObject
    };
    if raw.is_null() {
        return Err(MidBodyDecline::BeforeRun("null callee code ptr"));
    }
    let code = unsafe { &*raw };
    // Only portal trace sites carry `_exit_frame_with_exception` out of the
    // walk.  Without that licence a raise from the rebuilt callee has nowhere
    // to go, and by then the body has already run — `MidBodyDecline::AfterRun`,
    // from which no rewinding leg is sound (R8).
    //
    // The former preflight admitted a licence-less site whose callee had an
    // empty exception table and an effect-free post-marker region.  Neither
    // implies the body cannot RAISE: a callee with no handler still
    // propagates, so that pair could admit a rebuild that then stranded an
    // effectful Err.  Requiring the licence up front makes the post-run raise
    // unreachable instead of merely unlikely.  Instrumented over
    // pyre/bench/synth, the admitting branch was taken 0 times, so this costs
    // no measured coverage.
    if !WALK_END_PROPAGATE_ALLOWED.with(|c| c.get()) {
        return Err(MidBodyDecline::BeforeRun("no propagate licence"));
    }
    if cf_addr == 0 {
        return Err(MidBodyDecline::BeforeRun("no live caller frame"));
    }
    let ec = unsafe { (*(cf_addr as *const pyre_interpreter::PyFrame)).execution_context }
        as *mut pyre_interpreter::PyExecutionContext;
    if ec.is_null() {
        return Err(MidBodyDecline::BeforeRun("null execution context"));
    }
    let outer = unsafe { &mut *(cf_addr as *mut pyre_interpreter::PyFrame) };
    let outer_stack_base = outer.nlocals() + outer.ncells();
    let outer_code = unsafe { &*pyre_interpreter::pyframe_get_pycode(outer) };
    let outer_handler = pyre_interpreter::pycode::lookup_exceptiontable(
        &outer_code.exceptiontable,
        (words.call_py_pc as u32) * 2,
    );
    {
        // E-G2: this specialization reconstructs only the exact empty
        // operand-stack level used by statement-position calls. A handler
        // preserving any operand below the call remains on legacy replay.
        if let Some((_target, depth, _lasti)) = outer_handler {
            if !exception_delivery_stack_is_sourceable(
                depth,
                below.len(),
                outer.locals_w().as_slice().len(),
                outer_stack_base,
            ) {
                return Err(MidBodyDecline::BeforeRun(
                    "caller handler wants more operands than the call sat on",
                ));
            }
        }
        // G7: materialize every outer local before the rebuilt callee can run.
        // `can_flush_walk_end_state_after_outer_call` already proved all
        // shadow entries sourceable, so no post-effect decline remains.
        if !crate::state::write_back_outer_locals(ctx, cf_addr) {
            return Err(MidBodyDecline::BeforeRun(
                "an outer local is not sourceable",
            ));
        }
    }
    let mut w_code = payload.w_code;
    let mut w_globals = payload.w_globals;
    let _w_code_root = ObjectSlotRoot::new(&mut w_code);
    let _w_globals_root = ObjectSlotRoot::new(&mut w_globals);
    // No positional seed: `finish_for_call_with_globals_obj` only binds
    // `args` into the first `varnames` slots, and every one of them is
    // cleared to PY_NULL and rewritten from `live_locals` just below.
    let frame = match pyre_interpreter::PyFrame::try_new_for_call_with_closure_and_globals_obj(
        w_code as *const (),
        &[],
        w_globals,
        ec,
        pyre_object::PY_NULL,
        pyre_interpreter::pyframe::FrameLocalsArrayAllocation::OldGenGc,
    ) {
        Ok(frame) => frame,
        Err(_) => return Err(MidBodyDecline::BeforeRun("callee frame allocation failed")),
    };
    let mut frame = pyre_interpreter::pyframe::FrameBox::new(frame);
    frame.fix_array_ptrs();
    let _frame_locals_root = pyre_interpreter::pyframe::FrameLocalsRoot::new(frame.as_mut_ptr());

    let Some(crate::jitcode_dispatch::InlineAbortCarrier::MidBody(current)) =
        crate::jitcode_dispatch::fbw_abort_carrier_clone()
    else {
        return Err(MidBodyDecline::BeforeRun("carrier is no longer a MidBody"));
    };
    if current.live_locals.len() != code.varnames.len() {
        return Err(MidBodyDecline::BeforeRun(
            "live_locals length does not match varnames",
        ));
    }
    for slot in &mut frame.locals_w_mut().as_mut_slice()[..code.varnames.len()] {
        *slot = pyre_object::PY_NULL;
    }
    // `_copy_data_from_miframe` restores Ref registers before any scalar
    // boxing allocation; once installed, the rooted frame array owns them.
    for (slot, value) in current.live_locals.iter().enumerate() {
        if let Some(crate::state::ConcreteValue::Ref(value)) = value {
            frame.locals_w_mut().as_mut_slice()[slot] = *value;
        }
    }
    let stack_base = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);
    for (rel, value) in current.live_stack.iter().enumerate() {
        let crate::state::ConcreteValue::Ref(value) = value else {
            return Err(MidBodyDecline::BeforeRun("live stack slot is not a Ref"));
        };
        frame.locals_w_mut().as_mut_slice()[stack_base + rel] = *value;
    }
    // The array is old-gen from birth (`FrameLocalsArrayAllocation::OldGenGc`)
    // and `FrameLocalsRoot` only forwards the field slot, not the items: the
    // young refs just stored need the remembered set to survive the boxing
    // allocations below, and each minor consumes the entry, so re-arm after
    // every batch that follows a possible collection.
    crate::state::frame_array_write_barrier(
        frame.as_mut_ptr() as *mut u8,
        frame.locals_w_mut() as *mut _,
    );
    for (slot, value) in current.live_locals.iter().enumerate() {
        crate::state::frame_array_write_barrier(
            frame.as_mut_ptr() as *mut u8,
            frame.locals_w_mut() as *mut _,
        );
        frame.locals_w_mut().as_mut_slice()[slot] = match value {
            None => pyre_object::PY_NULL,
            Some(crate::state::ConcreteValue::Ref(value)) => *value,
            Some(crate::state::ConcreteValue::Int(value)) => pyre_object::w_int_new(*value),
            Some(crate::state::ConcreteValue::Float(value)) => {
                pyre_object::floatobject::w_float_new(*value)
            }
            Some(crate::state::ConcreteValue::Null | crate::state::ConcreteValue::Bool(_)) => {
                return Err(MidBodyDecline::BeforeRun("live local is Null/Bool"));
            }
        };
    }
    crate::state::frame_array_write_barrier(
        frame.as_mut_ptr() as *mut u8,
        frame.locals_w_mut() as *mut _,
    );
    frame.valuestackdepth = stack_base + current.live_stack.len();
    frame.last_instr = words.callee_py_pc as isize - 1;
    let sys_exc_value_pre = unsafe { (*ec).sys_exc_value };
    // A minor collection inside the callee can move the operands the CALL sat
    // on, so they have to survive it as roots rather than as a stale borrow.
    // Re-reading them from the carrier afterwards would work too, but it can
    // fail — and a failure here is `AfterRun`, from which no leg is sound
    // (R8).  Rooting an owned copy cannot fail, which is the point.
    let mut below_owned: Vec<pyre_object::PyObjectRef> = below.to_vec();
    let _below_root = ObjectVecRoot::new(&mut below_owned);
    // Everything between the preflight above and here — the callee frame
    // allocation, the local writes, the Int/Float boxing — can allocate and
    // therefore collect.  Re-verify immediately before the point of no return
    // so the only window left is the callee's own execution.
    if !crate::state::can_flush_walk_end_state_after_outer_call(
        ctx,
        cf_addr,
        words.call_py_pc,
        words.post_call_py_pc,
        current.call_stack_len,
        &below_owned,
    ) {
        return Err(MidBodyDecline::BeforeRun(
            "outer call boundary stopped being flushable during the rebuild",
        ));
    }
    // A re-entrant trace started inside the rebuilt callee would call
    // `fbw_store_journal_reset`, which clears the abort carrier, the executed
    // effect odometer and every store journal — i.e. exactly the outer walk
    // state this commit depends on.  Worse than losing it: a reset odometer
    // reads back as 0, which `walk_end_resume_provable` can mistake for "no
    // effects since the latch" and treat as proof.  The residual path already
    // suspends continuation across user code for the same reason
    // (`residual_call.rs`); the rebuilt callee is that same situation.
    let ran = {
        let _suspend = majit_metainterp::TraceContinuationSuspendGuard::enter();
        frame.execute_frame(None, None)
    };
    let below_now = &below_owned[..];
    match ran {
        Ok(mut retval) => {
            crate::jitcode_dispatch::fbw_abort_carrier_set_return(retval);
            let _retval_root = ObjectSlotRoot::new(&mut retval);
            // The callee has already RUN, so this is the last `AfterRun` a
            // caller could still hit.  Its whole precondition was verified
            // immediately before `execute_frame` and `below` is rooted across
            // it, so a false here means an invariant broke during the callee —
            // not an ordinary decline.  Make that loud where it is safe to be.
            if crate::state::flush_walk_end_state_after_outer_call(
                ctx,
                cf_addr,
                words.call_py_pc,
                words.post_call_py_pc,
                current.call_stack_len,
                below_now,
                retval,
            ) {
                Ok(())
            } else {
                debug_assert!(
                    false,
                    "post-call flush declined after the callee ran, though its \
                     precondition held immediately before `execute_frame`",
                );
                Err(MidBodyDecline::AfterRun(
                    "post-call caller flush declined AFTER the callee ran",
                ))
            }
        }
        Err(mut operr) => {
            // `_resume_mainloop(current_exc)` returns the exception to the
            // caller frame. Restore the caller's pre-CALL handled-exception
            // state first; PUSH_EXC_INFO/POP_EXCEPT will manage it from the
            // selected handler onward.
            unsafe { (*ec).sys_exc_value = sys_exc_value_pre };
            // No licence check here any more: the preflight above refuses a
            // licence-less site outright, so a raise reaching this point always
            // has somewhere to go.  That is what makes this arm infallible —
            // it was the one `AfterRun` that a caller could actually provoke.
            let outer = unsafe { &mut *(cf_addr as *mut pyre_interpreter::PyFrame) };
            // The handler unwinds from the operand level the CALL raised at,
            // which for an expression-position call is not the empty one.
            let arr_ptr = outer.locals_w_mut() as *mut _;
            outer.locals_w_mut().as_mut_slice()
                [outer_stack_base..outer_stack_base + below_now.len()]
                .copy_from_slice(below_now);
            crate::state::frame_array_write_barrier(cf_addr as *mut u8, arr_ptr);
            outer.last_instr = words.call_py_pc as isize;
            outer.valuestackdepth = outer_stack_base + below_now.len();
            let mut next_instr = words.call_py_pc;
            if pyre_interpreter::eval::handle_exception(outer, &mut operr, &mut next_instr) {
                outer.last_instr = next_instr as isize - 1;
            } else {
                WALK_END_PROPAGATED_EXCEPTION.with(|c| *c.borrow_mut() = Some(operr));
            }
            Ok(())
        }
    }
}

fn start_pc_is_loop_header(code: &pyre_interpreter::CodeObject, start_pc: usize) -> bool {
    use pyre_interpreter::Instruction as I;
    let mut arg_state = pyre_interpreter::OpArgState::default();
    for (pc, unit) in code.instructions.iter().copied().enumerate() {
        let (instr, op_arg) = arg_state.get(unit);
        let delta = match instr {
            I::JumpBackward { delta } | I::JumpBackwardNoInterrupt { delta } => delta,
            _ => continue,
        };
        if pyre_interpreter::jump_target_backward_decoded(code, pc + 1, delta, op_arg) == start_pc {
            return true;
        }
    }
    false
}

/// Trace an entire loop body starting at `start_pc`.
///
/// Drives the authoritative full-body walk (`full_body_walk_trace`): the
/// walker walks the per-CodeObject JitCode body, recording symbolic IR against
/// the per-step concrete frame snapshot.  A location the walk declines
/// re-interprets without JIT (the trait `PyreMetaInterp` interpret loop is
/// retired).
pub fn trace_bytecode<Sym: WalkSym>(
    meta: &mut MetaInterp<PyreMeta>,
    sym: &mut Sym,
    _code: &CodeObject,
    start_pc: usize,
    mut concrete_frame: pyre_interpreter::pyframe::FrameBox,
    live_frame_addr: usize,
    allow_propagate_out: bool,
) -> (TraceAction, pyre_interpreter::pyframe::FrameBox) {
    // `llmodel.py:557` parity — install pyre's `Cpu` impl so the
    // optimizer's `protect_speculative_string` / `bh_strlen` /
    // `bh_strgetitem` family routes through `W_UnicodeObject`-shaped
    // `str_descr` / `unicode_descr` (`pyre_cpu` module).
    meta.set_cpu(crate::pyre_cpu::shared());

    // A stale flag from a prior trace on this thread must not leak into
    // this trace's adoption decision.
    WALK_END_FLUSH_COMMITTED.with(|c| c.set(false));
    WALK_END_COMMIT_LEG.with(|c| c.set(0));
    WALK_END_PROPAGATED_EXCEPTION.with(|c| *c.borrow_mut() = None);
    WALK_END_PROPAGATE_ALLOWED.with(|c| c.set(allow_propagate_out));
    WALK_END_RESTART_PC.with(|c| c.set(None));
    FBW_BRIDGE_DECLINED.with(|c| c.set(false));
    // A prior walk's opcode-effect-window sample must not alias a same-pc
    // opcode of this walk (the escape-flush latch gate reads it).
    crate::jitcode_dispatch::escape_opcode_window_reset();
    // Likewise drop any escape-flush undo a prior attempt failed to consume:
    // "oldest capture wins" is only correct WITHIN one walk attempt — a
    // leftover here would restore wrong-generation locals onto a frame many
    // iterations ahead.
    crate::jitcode_dispatch::discard_escape_flush_undo();
    // `TraceCtx.reads_module_global` needs no reset here: a fresh TraceCtx is
    // built per trace (zero-init `false`), unlike the walk-end TLS flags above.
    // Likewise clear any no-replay finish payload a prior trace left
    // unconsumed. The full-body walk re-clears this in `run_perfn_walk`.
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    // Likewise drop any cross-frame-resume abort request a prior aborted
    // trace left unconsumed.
    let _ = crate::state::take_trace_abort_requested();

    // Publish this trace's `sym` as the exception-carrier root anchor: a
    // collection triggered by an allocating traced opcode marks the trace-built
    // exception held only in `sym.trace_built_exc` (and the seeded/caught
    // `last_exc_value` / `current_exc_value`). The guard restores the prior
    // anchor on every return path, including panics and nested tracing.
    let _active_sym_guard = sym.active_exc_anchor().map(set_active_sym_exc);

    let ctx = meta
        .trace_ctx()
        .expect("trace_bytecode invariant: meta.tracing must be Some during merge_point closure");
    // A multi-frame bridge carrier overrides the trace-start
    // pc with the OUTERMOST (`frames[0]`) resume pc. The passed `start_pc` is
    // the INNERMOST frame's pc (`decode_and_restore_guard_failure` returns
    // `jit_state.next_instr()`), which belongs to the deepest reconstructed
    // callee — NOT the root. The dedicated carrier walker below starts from
    // the root pc and reconstructs the in-flight inline frames.
    let carrier = ctx.take_bridge_inline_carrier();
    let start_pc = if let Some(ref c) = carrier {
        c.root_pc
    } else {
        start_pc
    };
    let lasti_pc = if let Some(ref c) = carrier {
        crate::py_coord::resume_py_pc_for_jitcode_word(c.root_jitcode_index, c.root_pc as i32)
            as usize
    } else {
        start_pc
    };
    // RPython MetaInterp._interpret() parity: the walker (sole tracer)
    // executes as it records over a concrete `PyFrame` snapshot
    // (`snapshot_for_tracing`); the interpreter does not run during tracing.
    // The snapshot copies frame-LOCAL state (abort-safety) while sharing
    // `w_globals`; vable-statics capture reads pointer-valued fields from the
    // live frame (`live_vable_frame_addr` below), not the snapshot copy.
    //
    // The former snapshot double-apply (inline-frame SHARED-heap STOREs
    // leaking during tracing and re-applying on the compiled loop's re-run)
    // is resolved by gap 10: the concrete executor is deleted so STOREs are
    // record-only, and `flush_walk_end_state_to_frame`
    // (`raise_continue_running_normally` parity) advances the real frame so
    // the interpreter resumes AFTER the walked region, not from its start.
    concrete_frame.set_last_instr_from_next_instr(lasti_pc);
    let w_code = concrete_frame.pycode;
    // Read-only diagnostic: `PYRE_DUMP_PERFN_JITCODE=1` dumps the
    // per-CodeObject JitCode body — the byte stream the walker-as-tracer walks
    // so that `miframe.pc == jitcode_pc`.
    if std::env::var_os("PYRE_DUMP_PERFN_JITCODE").is_some() {
        dump_perfn_jitcode_for_trace(w_code, lasti_pc);
    }
    let cf_addr = &*concrete_frame as *const pyre_interpreter::pyframe::PyFrame as usize;
    // The snapshot stands in for concrete stepping only; vable-statics
    // capture must read pointer-valued fields (`debugdata` / `lastblock`)
    // from the live frame the compiled loop will run on.  See the
    // `live_vable_frame_addr` field doc (state.rs).  Set before the
    // full-body-walk leg below so the production tracer sees it.
    //
    // gap 10 slice 2b: set this BEFORE `init_symbolic` so the root vable
    // identity (seed_virtualizable_boxes) is baked against the live frame
    // address, not the discarded snapshot's.
    sym.set_live_vable_frame_addr(live_frame_addr);
    // pyjitpl.py:65 MIFrame.__init__: sym fields populated once at frame
    // construction. Callee (inline) frames are set up by perform_call
    // (trace_opcode.rs) and don't call init_symbolic; this path
    // handles the root frame push.
    sym.init_symbolic(ctx, cf_addr);
    if let Some(ref carrier) = carrier {
        debug_assert_eq!(
            unsafe { (*sym.jitcode()).index as i32 },
            carrier.root_jitcode_index
        );
    }
    // Issue #215 item 2: drive the multiframe bridge-carrier resume via the
    // full-body walker (reconstruct the in-flight callee framestack + walk
    // innermost-first) instead of aborting to a no-JIT re-interpret below.
    if let Some(ref carrier) = carrier {
        // Reconstruct the in-flight callee framestack and drive it
        // innermost-first via the drain sub-walk, which compiles the N-deep
        // carrier (recipes 1..=7) or cleanly deopts to the blackhole.
        let action = drive_bridge_carrier_walk(ctx, sym, w_code, start_pc, cf_addr, carrier);
        finish_trace_namespace_dependency(meta);
        return (action, concrete_frame);
    }
    // Gated diagnostic: `PYRE_WALK_PERFN_JITCODE=1` attempts to walk the
    // per-CodeObject JitCode body via `dispatch_via_miframe` from the resume
    // entry pc, logs how far the symbolic walk gets (terminator outcome vs
    // first `DispatchError` stop), then aborts the trace.  Default-off → zero
    // production change.
    // The generic probe is gated on `carrier.is_none()` because multi-frame
    // bridge resumes are handled by the dedicated carrier walkers above.
    if carrier.is_none() && std::env::var_os("PYRE_WALK_PERFN_JITCODE").is_some() {
        probe_walk_perfn_jitcode(ctx, sym, w_code, start_pc, cf_addr);
        finish_trace_namespace_dependency(meta);
        return (TraceAction::Abort, concrete_frame);
    }
    // The per-CodeObject JitCode body is traced via the authoritative
    // full-body walk — the walker-as-tracer path that makes
    // `miframe.pc == jitcode_pc`.
    //
    // A green key in `FBW_DECLINED_KEYS` had a prior walk fail on a
    // structural walker limitation (the recurring error classes in
    // `full_body_walk_trace`).  `FBW_DECLINED_KEYS` is insert/contains only,
    // so the decline is permanent for this process: retraces bypass the
    // walker and the key re-interprets without JIT instead of being
    // permanently blacklisted (`DONT_TRACE_HERE`).
    if carrier.is_none() && !fbw_declined(crate::driver::make_green_key(w_code, start_pc)) {
        let action = full_body_walk_trace(ctx, sym, w_code, start_pc, cf_addr, WalkJournals::Reset);
        finish_trace_namespace_dependency(meta);
        return (action, concrete_frame);
    }
    // Any path the walker did not trace above re-interprets without JIT for
    // this key. The location stays trace-eligible (no `DONT_TRACE_HERE`).
    crate::jitcode_dispatch::census_record("Trait::DeclinedAbort");
    let action = TraceAction::Decline;
    finish_trace_namespace_dependency(meta);
    (action, concrete_frame)
}

/// Read-only walker-as-tracer diagnostic probe.
///
/// Attempts to walk the per-CodeObject JitCode body via
/// [`crate::jitcode_dispatch::dispatch_via_miframe`] from the resume
/// entry pc and logs how far the symbolic walk
/// gets: a terminator outcome (`Finish` / `CloseLoop` / `SubReturn`)
/// or the first `DispatchError` stop with its pc.
///
/// Diagnostic-only: the caller aborts the trace immediately after this
/// returns, so any IR / merge-point / heap-cache mutation the walk
/// records is discarded with the aborted trace.  The recorder is also
/// rolled back via `cut_trace` to keep the discarded trace tidy.
///
/// Purpose: with walk-capability gaps #1/#2/#3 closed (decode table +
/// vable array ops + jit_merge_point/last_exception/abort handlers),
/// this surfaces the next blocker for the full-body walk — the Path A
/// payload-seeding gap (an op reading a register slot the entry never
/// seeded, e.g. a `goto_if_not` over a non-concrete Int produced by an
/// unfolded `residual_call`).
///
/// Decode the loop-header `jit_merge_point` that governs a bridge resume
/// coordinate and return its green-ref (`gr`) and red (`rr`) register lists.
///
/// These name the jitcode register colors the loop body reads its
/// loop-invariant pycode (`gr`) and frame/ec (`rr`) from.  A mid-loop walk
/// entering PAST the merge point never executes it, so those colors are
/// left `OpRef::NONE` unless explicitly seeded.
///
/// Operand layout `cIRFIRF`: jdindex(`c`, 1 byte) followed by six
/// count-prefixed register lists `gi, gr, gf, ri, rr, rf`.  Returns `None`
/// when no preceding merge point exists (straight-line resume) or the
/// operand stream is truncated.
///
/// A body-guard bridge enters past its loop's merge point, so the governing
/// op is the last merge point at or before `entry`. Main traces enter before
/// the static marker and never use this runtime reconstruction.
///
pub(crate) fn bridge_resume_merge_point_regs(
    code: &[u8],
    entry: usize,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let merge_point_pcs = || {
        crate::jitcode_runtime::decoded_ops(code)
            .filter(|op| op.opname == "jit_merge_point")
            .map(|op| op.pc)
    };
    let mp_pc = merge_point_pcs()
        .filter(|&pc| pc <= entry)
        .max()
        .or_else(|| merge_point_pcs().filter(|&pc| pc >= entry).min())?;
    let mut cursor = mp_pc + 1 + 1; // opcode byte + jdindex (`c`)
    let mut lists: [Vec<u8>; 6] = Default::default();
    for slot in lists.iter_mut() {
        let count = *code.get(cursor)? as usize;
        cursor += 1;
        for _ in 0..count {
            slot.push(*code.get(cursor)?);
            cursor += 1;
        }
    }
    let [_gi, gr, _gf, _ri, rr, _rf] = lists;
    Some((gr, rr))
}

/// Decode the first source `jit_merge_point` at or after a static sidecar
/// entry and return its Ref-green register colors.
///
/// RPython starts `_compile_and_run_once` with the concrete green arguments
/// that selected the warm-state cell. A source-translated marker entry must
/// make the Ref green available to the root MIFrame even when register
/// allocation leaves the pre-marker `pycode` load symbolic. Int greens live
/// in the JitCode constant region and are populated by `copy_constants`; they
/// are deliberately not positional call arguments. The sidecar enters before
/// the marker, so forward selection (unlike bridge resume's backward
/// selection) identifies the marker owned by this green entry.
fn static_entry_merge_point_green_ref_regs(code: &[u8], entry: usize) -> Option<Vec<u8>> {
    let mp_pc = crate::jitcode_runtime::decoded_ops(code)
        .filter(|op| op.opname == "jit_merge_point" && op.pc >= entry)
        .map(|op| op.pc)
        .min()?;
    let mut cursor = mp_pc + 1 + 1; // opcode byte + jdindex (`c`)
    let mut lists: [Vec<u8>; 6] = Default::default();
    for slot in lists.iter_mut() {
        let count = *code.get(cursor)? as usize;
        cursor += 1;
        for _ in 0..count {
            slot.push(*code.get(cursor)?);
            cursor += 1;
        }
    }
    let [_gi, gr, _gf, _ri, _rr, _rf] = lists;
    Some(gr)
}

type PerfnWalkResult = Result<
    (crate::jitcode_dispatch::DispatchOutcome, usize),
    crate::jitcode_dispatch::DispatchError,
>;

/// Shared per-CodeObject full-body walk used by both the read-only
/// diagnostic probe ([`probe_walk_perfn_jitcode`], `authoritative=false`,
/// trace discarded) and the production full-body tracer
/// ([`full_body_walk_trace`], `authoritative=true`, trace kept).
///
/// Returns `(entry, code_len, walk_result)` or `None` when the
/// per-CodeObject setup is unavailable.  The caller owns the post-walk
/// disposition: the probe captures a trace position beforehand and
/// `cut_trace`s + logs; the production path maps `walk_result` to a
/// `TraceAction` and keeps the recording.
/// Per-frame jitcode dispatch shared by the root full-body walk
/// ([`run_perfn_walk`]) and the multiframe bridge-carrier drain
/// ([`drive_bridge_carrier_walk`]).  Resolves the five terminal descrs off
/// `MetaInterpStaticData`, builds the per-CodeObject descr pool + sub-jitcode
/// lookup off `pjc.jitcode.exec.descrs`, and runs `dispatch_via_miframe` from
/// `entry` with the caller-seeded register banks.  Returns
/// `(code_len, walk_result)`; `None` when the terminal descrs are unwired.
fn dispatch_perfn_frame<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    sym: &mut Sym,
    concrete_frame_addr: usize,
    orgpc: usize,
    session: &std::cell::RefCell<crate::jitcode_dispatch::WalkSession>,
    pjc: &std::sync::Arc<crate::PyJitCode>,
    entry: usize,
    argboxes_r: &[majit_ir::OpRef],
    argboxes_i: &[majit_ir::OpRef],
    argboxes_f: &[majit_ir::OpRef],
    authoritative: bool,
) -> Option<(usize, PerfnWalkResult)> {
    // The walk records no descr itself — the compile consumer picks the
    // terminator descr from `finish_arg_types` (`pyjitpl.rs
    // done_with_this_frame_descr_from_types`).  Still require the five terminal
    // descrs to be wired on MetaInterpStaticData: a missing one means setup
    // never ran, so a Finish this walk produced would have nothing to compile
    // against.  Log and bail rather than walk into that.
    {
        let sd = ctx.metainterp_sd();
        if sd.done_with_this_frame_descr_void.is_none()
            || sd.done_with_this_frame_descr_int.is_none()
            || sd.done_with_this_frame_descr_ref.is_none()
            || sd.done_with_this_frame_descr_float.is_none()
            || sd.exit_frame_with_exception_descr_ref.is_none()
        {
            eprintln!("[walk-perfn] terminal descrs not wired; skipping walk");
            return None;
        }
    }

    // Per-fn descr-pool plumbing: the per-CodeObject body resolves `d`/`j`
    // descr operands through its OWN runtime pool (`pjc.jitcode.exec.descrs`,
    // `Vec<RuntimeBhDescr>`), NOT the global `all_descr_refs()`.  Build the
    // index-parallel adapted `descr_refs` and resolve `inline_call` callee
    // jitcodes through the same pool.
    use majit_metainterp::jitcode::RuntimeBhDescr;
    // The per-CodeObject JitCode lives in the process-global jitcode registry
    // (installed by `install_jitcodes` before tracing); `pjc` is an `Arc` clone
    // of that data, so the descr pool (and the callee jitcode bodies it
    // references) outlive this walk.  Extend the borrow to `'static` so the
    // `'static`-bodied `SubJitCodeBody` from `sub_jitcode_lookup` type-checks —
    // mirrors the production arm-entry borrow extension at `trace_opcode.rs`.
    let perfn_descrs: &'static [RuntimeBhDescr] =
        unsafe { &*(pjc.jitcode.exec.descrs.as_slice() as *const [RuntimeBhDescr]) };
    let perfn_descr_refs: Vec<majit_ir::DescrRef> = perfn_descrs
        .iter()
        .enumerate()
        .map(|(i, d)| match d {
            RuntimeBhDescr::Descr(bh) => crate::descr::make_descr_from_bh(bh),
            // `inline_call`'s `d` operand resolves the callee through
            // `JitCodeDescr::jitcode_index()` → `sub_jitcode_lookup`.  Key the
            // descr by its own pool slot `i` so the per-fn lookup below
            // re-reads `exec.descrs[i].as_jitcode()`.  `Call` /
            // `AssemblerToken` pool entries belong to the `BC_CALL_*` /
            // `BC_CALL_ASSEMBLER_*` op families, whose walker handlers read the
            // target straight from `RawDescrPool::PerFn`, not through this
            // adapted `DescrRef` slot; the jitcode-descr stand-in is a
            // fail-loud tripwire for a mis-routed slot.
            RuntimeBhDescr::JitCode(_) => crate::descr::make_jitcode_descr(i),
            RuntimeBhDescr::Call(_) | RuntimeBhDescr::AssemblerToken(_) => {
                crate::descr::make_jitcode_descr(i)
            }
        })
        .collect();

    let sub_jitcode_lookup = |idx: usize| -> Option<crate::jitcode_dispatch::SubJitCodeBody> {
        perfn_descrs
            .get(idx)
            .and_then(|d| d.as_jitcode())
            .map(|jc| crate::jitcode_dispatch::SubJitCodeBody {
                code: jc.code.as_slice(),
                num_regs_r: jc.num_regs_r() as usize,
                num_regs_i: jc.num_regs_i() as usize,
                num_regs_f: jc.num_regs_f() as usize,
                constants_i: jc.constants_i.as_slice(),
                constants_r: jc.constants_r.as_slice(),
                constants_f: jc.constants_f.as_slice(),
            })
    };

    let code = pjc.jitcode.code.as_slice();
    let code_len = code.len();
    let walk_result = crate::jitcode_dispatch::dispatch_via_miframe(
        ctx,
        sym,
        concrete_frame_addr,
        orgpc,
        session,
        code,
        entry,
        &perfn_descr_refs,
        crate::jitcode_dispatch::RawDescrPool::PerFn(perfn_descrs),
        // Authoritative concrete execution: `false` for a read-only probe
        // (trace discarded → re-executing would corrupt live state); `true`
        // for the production full-body tracer (the walk IS the execution).
        authoritative,
        &sub_jitcode_lookup,
        true,
        pjc.jitcode.num_regs_r() as usize,
        pjc.jitcode.num_regs_i() as usize,
        pjc.jitcode.num_regs_f() as usize,
        pjc.jitcode.constants_r.as_slice(),
        pjc.jitcode.constants_i.as_slice(),
        pjc.jitcode.constants_f.as_slice(),
        argboxes_r,
        argboxes_i,
        argboxes_f,
    );
    Some((code_len, walk_result))
}

/// Select a reconstructed frame's walk-entry JitCode offset: prefer the
/// guard-carried `jitcode_pc` decoded from the resume frame only when it belongs
/// to the same JitCode body that will drive the walk. Pyre permits multiple
/// JitCode bodies per code object, so the carried offset is invalid in another
/// body's coordinate space. Upstream `resume.py:1050-1051` uses the same
/// snapshot-selected jitcode for frame construction and its PC. A missing
/// carried coordinate declines at the caller before a bridge is published.
fn select_recipe_entry(
    jitcode_index: i32,
    body_index: i32,
    carried_jitcode_pc: i32,
) -> Option<usize> {
    (carried_jitcode_pc != majit_ir::resumedata::NO_JITCODE_PC && jitcode_index == body_index)
        .then(|| crate::state::resolve_bridge_walk_entry_at(jitcode_index, carried_jitcode_pc))
        .flatten()
}

fn residual_ref_call_before(code: &[u8], entry: usize) -> Option<(usize, usize)> {
    crate::jitcode_runtime::decoded_ops(code)
        .find(|op| op.next_pc == entry && op.opname.starts_with("residual_call"))
        .and_then(|op| {
            op.argcodes
                .ends_with(">r")
                .then(|| code.get(entry - 1).copied())
                .flatten()
                .map(|dst| (op.pc, usize::from(dst)))
        })
}

/// `MIFrame.make_result_of_lastop` pushes a completed call's result at the
/// top of the caller's post-call operand stack.  This semantic slot is
/// independent of the post-regalloc result color.
fn pending_call_result_semantic_slot(nlocals: usize, post_call_depth: usize) -> Option<usize> {
    post_call_depth
        .checked_sub(1)
        .and_then(|top| nlocals.checked_add(top))
}

/// Issue #215 item 2 (P2 drain): drive a multiframe bridge-carrier resume via
/// the full-body walker instead of aborting to a no-JIT re-interpret.
///
/// The carrier reconstructs the in-flight callee framestack
/// (`rebuild_from_resumedata`, resume.py:1042-1057); each callee is rebuilt as
/// a virtualizable the walker can drive (`setup_reconstructed_callee_frame`),
/// then walked innermost-first via [`dispatch_perfn_frame`], threading each
/// frame's return into its parent before the parent walks, until the root
/// walks forward to a terminator.
///
/// Increment 1 (diagnostic): walk only the DEEPEST reconstructed callee
/// (`recipes` is outermost-first, so the last entry is the guard-failing
/// frame), log the outcome, discard the trace, and abort — validates the
/// reconstructed-frame walk plumbing before result-threading + the root walk
/// are wired. The compile leg is unconditional.
/// Thread a reconstructed callee's `SubReturn` value into the root portal's
/// residual-call result register so the subsequent root walk reads it as the
/// call result at `root_pc`.
///
/// `make_result_of_lastop` writes the result to the residual-call body's
/// trailing `>r` destination byte, so use that register when the call ending at
/// `root_pc` can be decoded.  Fall back to the codewriter-baked result-color
/// trivia twin keyed by the call's JitCode pc for older shapes that lack the
/// canonical residual-call encoding.
///
/// The result is always mirrored into `bridge_registers_r` for interior-entry
/// bridge walks, whose Ref-bank seed is color-indexed. Opcode-entry bridge
/// walks rebuild slot-indexed locals and operand-stack values from
/// `bridge_local_oprefs` / `bridge_stack_oprefs`.  RPython's
/// `MIFrame.make_result_of_lastop` puts the return value at the top of the
/// caller's post-call operand stack, so derive that semantic slot from the
/// codewriter's after-residual depth twin.  The result is not a live Variable
/// at this coordinate and therefore need not have a pcdep color→slot entry.
/// Treating its post-regalloc color as a localsplus slot corrupts an unrelated
/// local whenever regalloc coalesces the call result.
/// Returns `false` (caller declines the compile) when the register is
/// unresolved.
fn inject_root_call_result<Sym: WalkSym>(
    sym: &mut Sym,
    root_pc: usize,
    result: majit_ir::OpRef,
) -> bool {
    if sym.jitcode().is_null() {
        return false;
    }
    let payload = unsafe { &(*sym.jitcode()).payload };
    let residual_call = residual_ref_call_before(payload.jitcode.code.as_slice(), root_pc);
    let result_reg = residual_call.map(|(_, dst)| dst).or_else(|| {
        payload
            .result_color_trivia_for_jitcode_pc(root_pc)
            .map(|c| c as usize)
            .filter(|&c| c != u16::MAX as usize)
    });
    let Some(result_reg) = result_reg else {
        return false;
    };
    let nlocals = sym.nlocals();
    if let Some(bridge_regs) = sym.bridge_registers_r_mut().as_mut() {
        if bridge_regs.len() <= result_reg {
            bridge_regs.resize(result_reg + 1, majit_ir::OpRef::NONE);
        }
        bridge_regs[result_reg] = result;
    }
    let post_call_depth = residual_call
        .and_then(|(call_pc, _)| payload.depth_after_residual_for_jitcode_pc(call_pc))
        .map(usize::from);
    let Some(post_call_depth) = post_call_depth else {
        return false;
    };
    let semantic_result_slot = pending_call_result_semantic_slot(nlocals, post_call_depth);
    let Some(semantic_result_slot) = semantic_result_slot else {
        return false;
    };
    if semantic_result_slot < nlocals {
        if let Some(locals) = sym.bridge_local_oprefs_mut().as_mut() {
            if locals.len() <= semantic_result_slot {
                locals.resize(semantic_result_slot + 1, majit_ir::OpRef::NONE);
            }
            locals[semantic_result_slot] = result;
        }
    } else {
        let slot = semantic_result_slot - nlocals;
        let bridge = sym.bridge_stack_oprefs_mut().get_or_insert_with(Vec::new);
        if bridge.len() <= slot {
            bridge.resize(slot + 1, majit_ir::OpRef::NONE);
        }
        bridge[slot] = result;
    }
    // `MIFrame.make_result_of_lastop` both stores the value and advances the
    // caller's stack pointer.  Without the matching depth update,
    // `init_symbolic` sizes its resumed stack from the pre-call depth and
    // truncates the just-injected top slot.
    sym.set_valuestackdepth(nlocals + post_call_depth);
    true
}

/// The paused frame's `except` handler jitcode pc covering its CALL, for the
/// carrier-boundary `finishframe_exception`.  The resume pc is the CALL's
/// post-call `-live-`, with the enclosing `catch_exception/L` sitting behind
/// it.  The handler is accepted on the catch alone, as
/// `pyjitpl.py finishframe_exception` does; where it leads is `finishframe`'s
/// business.
fn carrier_catch_target(code: &[u8], pc: usize, frame: &str) -> Option<usize> {
    // The paused caller's resume pc is the CALL's post-call `-live-`; the
    // `catch_exception/L` for the enclosing try sits BEHIND it (between the
    // CALL's post-call `-live-` and the next op), so scan backward — the same
    // lookup the single-frame exception-edge router uses.
    // `pc` is the CALL's OWN trailing `-live-`, one op short of the
    // block-entry `-live-` the backward scan keys off, so read forward first
    // and keep the backward scan for callers already at that coordinate.
    let candidate = crate::jitcode_dispatch::catch_target_after_resume_live(code, pc)
        .or_else(|| crate::jitcode_dispatch::find_catch_before_resume_live(code, pc));
    if crate::jitcode_dispatch::p2_diag_enabled() {
        eprintln!("[p2-raise] frame={frame} pc={pc} catch_before={candidate:?}");
    }
    candidate
}

/// The `PyTraceback` node a paused middle frame contributes as an exception
/// crosses it on the drain's unwind.
///
/// `pyopcode.py handle_operation_error` runs
/// `pytraceback.record_application_traceback` for EVERY frame the exception
/// passes through, not only the one that catches; the walk-level sub-walk arm
/// records it for an inlined callee the same way.  A frame this drain crosses
/// is reconstructed from resume data and never becomes a real `PyFrame`, so
/// nothing downstream would record it — the node has to be emitted here, and at
/// runtime as well as for the recording pass, or a `tb_frame.f_code.co_name`
/// walk comes up one level short on exactly the deopt iterations.
///
/// The recipe carries no materialized frame, so this takes the fabricating
/// hook: it builds one from the promoted code / globals, which is what
/// `record_inline_application_traceback` falls back to for an unseeded level.
fn record_carrier_crossed_frame_traceback(
    ctx: &mut TraceCtx,
    recipe: &majit_metainterp::ReconstructRecipe,
    exc: majit_ir::OpRef,
    exc_concrete: crate::state::ConcreteValue,
) {
    let crate::state::ConcreteValue::Ref(exc_ptr) = exc_concrete else {
        return;
    };
    if exc_ptr.is_null() {
        return;
    }
    let w_code = crate::state::recover_inline_callee_code(recipe.code_ptr);
    // A synthetic frame carries the sentinel or null instead of a real code
    // object; the hook dereferences it, so both checks precede `is_code`, whose
    // type check would deref the sentinel.
    if w_code.is_null() || w_code as usize == usize::MAX {
        return;
    }
    if !unsafe { pyre_interpreter::pycode::is_code(w_code) } {
        return;
    }
    let w_globals = crate::state::recover_inline_callee_globals(recipe.code_ptr);
    majit_metainterp::record_inline_application_traceback_for_recording(
        exc_ptr as usize as i64,
        w_code as usize as i64,
        w_globals as usize as i64,
        recipe.jitcode_index,
        recipe.jitcode_pc,
    );
    let hook = majit_metainterp::record_inline_application_traceback_hook_address();
    if hook.is_null() || exc.is_none() {
        return;
    }
    let w_code_op = ctx.const_ref(w_code as usize as i64);
    let w_globals_op = ctx.const_ref(w_globals as usize as i64);
    let jitcode_op = ctx.const_int(i64::from(recipe.jitcode_index));
    let opcode_op = ctx.const_int(i64::from(recipe.jitcode_pc));
    ctx.call_void_typed_with_effect(
        hook,
        &[exc, w_code_op, w_globals_op, jitcode_op, opcode_op],
        &[
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Int,
            majit_ir::Type::Int,
        ],
        majit_metainterp::default_effect_info(),
    );
}

fn carrier_root_catch_target<Sym: WalkSym>(sym: &Sym, root_pc: usize) -> Option<usize> {
    if sym.jitcode().is_null() {
        return None;
    }
    let payload = unsafe { &(*sym.jitcode()).payload };
    carrier_catch_target(payload.jitcode.code.as_slice(), root_pc, "root")
}

fn discard_bridge_carrier_walk<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    sym: &mut Sym,
    entry_depth: usize,
    pre_pos: majit_metainterp::recorder::TracePosition,
    pre_virtualref_boxes: &[(majit_ir::OpRef, usize)],
) {
    // `pyframe.py:316-358 execute_frame` closes exactly the frame entered by
    // that invocation in its `finally: executioncontext.leave(...)`.  Close
    // only scopes this carrier walk opened, while their recorder positions
    // are still live.  If the walk already closed a parent scope, preserve
    // only the snapshot prefix that still survives instead of reopening it.
    let restore_depth = ctx.virtualref_boxes_len().min(entry_depth);
    while ctx.virtualref_boxes_len() > entry_depth {
        let before = ctx.virtualref_boxes_len();
        crate::jitcode_dispatch::carrier_ec_leave(ctx, sym, false);
        if ctx.virtualref_boxes_len() == before {
            break;
        }
    }
    ctx.cut_trace(pre_pos);
    ctx.restore_virtualref_boxes(pre_virtualref_boxes[..restore_depth].to_vec());
}

fn drive_bridge_carrier_walk<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    sym: &mut Sym,
    w_code: *const (),
    root_pc: usize,
    cf_addr: usize,
    carrier: &majit_metainterp::BridgeInlineCarrier,
) -> TraceAction {
    let entry_depth = ctx.virtualref_boxes_len();
    let pre_virtualref_boxes = ctx.snapshot_virtualref_boxes();
    let pre_pos = ctx.get_trace_position();
    let session = std::cell::RefCell::new(crate::jitcode_dispatch::WalkSession::default());
    crate::jitcode_dispatch::bool_box_truth_reset();
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    crate::jitcode_dispatch::fbw_store_journal_reset();
    // A prior walk's blackhole image must not be adopted as this drain's
    // continuation; the sub-walks below latch their own.
    crate::jitcode_dispatch::reset_single_frame_blackhole();
    // Odometer baseline for the abort tail: the drain may only rewind to the
    // guard while the reconstructed frames it drove executed nothing
    // irreversible.
    let effects_at_entry = crate::jitcode_dispatch::fbw_executed_effect_count();

    let root_ec = sym.concrete_execution_context();
    if crate::jitcode_dispatch::p2_diag_enabled() {
        let pcs: Vec<usize> = carrier
            .recipes
            .iter()
            .map(|r| {
                crate::py_coord::resume_py_pc_for_jitcode_word(r.jitcode_index, r.jitcode_pc)
                    as usize
            })
            .collect();
        eprintln!(
            "[p2-shape] root_pc={root_pc} n_recipes={} recipe_pcs={pcs:?}",
            carrier.recipes.len()
        );
    }
    let Some(recipe) = carrier.recipes.last() else {
        crate::jitcode_dispatch::census_record("P2Drain::NoRecipes");
        // Churn guard (Task 8): making this class transient retried the same
        // guard 500 times in inline_chain_depth_typeflip.py and
        // p2_local_result_bridge.py (loops_aborted 6 -> 505), so keep only
        // this measured P2 class permanently declined.
        fbw_bridge_decline(ctx);
        discard_bridge_carrier_walk(ctx, sym, entry_depth, pre_pos, &pre_virtualref_boxes);
        return p2_drain_abort();
    };

    // `setup_reconstructed_callee_frame` emits the callee frame vable into the
    // trace and returns `argboxes_r` seeding the portal reds + in-flight
    // operand-stack temps; the `_pending` callee sym is unused on the sub-walk
    // path (the sub-walk drives the callee body off `argboxes_r` + the emitted
    // frame vable, not a callee MIFrame).
    let Some((_pending, argboxes_r)) =
        crate::state::setup_reconstructed_callee_frame(ctx, recipe, root_ec, Vec::new())
    else {
        discard_bridge_carrier_walk(ctx, sym, entry_depth, pre_pos, &pre_virtualref_boxes);
        crate::jitcode_dispatch::census_record("P2Drain::SetupFailed");
        return p2_drain_abort();
    };
    let Some(callee_pjc) = crate::state::pyjitcode_for_code(recipe.code_ptr) else {
        discard_bridge_carrier_walk(ctx, sym, entry_depth, pre_pos, &pre_virtualref_boxes);
        crate::jitcode_dispatch::census_record("P2Drain::NoCalleePjc");
        return p2_drain_abort();
    };
    let entry = select_recipe_entry(
        recipe.jitcode_index,
        callee_pjc.jitcode.index() as i32,
        recipe.jitcode_pc,
    );
    let Some(entry) = entry else {
        discard_bridge_carrier_walk(ctx, sym, entry_depth, pre_pos, &pre_virtualref_boxes);
        crate::jitcode_dispatch::census_record("P2Drain::NoCalleeEntry");
        return p2_drain_abort();
    };
    let callee_w_globals = crate::state::recover_inline_callee_globals(recipe.code_ptr) as usize;
    // The reconstructed callee's local slot concretes (`recipe.concrete_r` is
    // parallel to `registers_r`; locals occupy `[0, nlocals)`), seeded into the
    // sub-walk's local-concrete shadow so a nested self-recursive call's int arg
    // is known.
    let nlocals = recipe.nlocals.min(recipe.concrete_r.len());
    let local_oprefs = &recipe.registers_r[..nlocals.min(recipe.registers_r.len())];
    let local_concretes = &recipe.concrete_r[..nlocals];
    let stack_end = recipe.valuestackdepth.min(recipe.registers_r.len());
    let resumed_stack_oprefs = &recipe.registers_r[nlocals.min(stack_end)..stack_end];
    let concrete_stack_end = recipe.valuestackdepth.min(recipe.concrete_r.len());
    let resumed_stack_concretes =
        &recipe.concrete_r[nlocals.min(concrete_stack_end)..concrete_stack_end];
    // Increment 2b-i: drive the deepest callee as an inline SUB-WALK rooted on
    // the portal `sym` (is_top_level=false), so its `ref_return` surfaces
    // `SubReturn` instead of the top-level `Finish` pyre's own-portal model
    // rejects.  Diagnostic: log the outcome, then abort (trace discarded).
    let walk = crate::jitcode_dispatch::drive_bridge_carrier_subwalk(
        ctx,
        &session,
        sym,
        root_pc,
        &callee_pjc,
        recipe.code_ptr as usize,
        callee_w_globals,
        entry,
        &argboxes_r,
        &recipe.registers_i,
        &recipe.registers_f,
        local_oprefs,
        local_concretes,
        resumed_stack_oprefs,
        resumed_stack_concretes,
        // Depth-N: tell the deepest sub-walk that all shallower frames are
        // paused so its in-callee guard snapshots encode the full
        // [root, ..middles.., deepest] chain (else the blackhole rebuilds a
        // framestack missing the middles on such a guard's deopt).
        if carrier.recipes.len() >= 2
            && carrier.recipes.len() <= crate::jitcode_dispatch::fbw_max_multiframe_depth()
        {
            &carrier.recipes[..carrier.recipes.len() - 1]
        } else {
            &[]
        },
    );
    let deepest_got_exception = matches!(
        &walk,
        Some(Ok((
            crate::jitcode_dispatch::DispatchOutcome::SubRaise { .. },
            _
        )))
    );
    crate::jitcode_dispatch::carrier_ec_leave(ctx, sym, deepest_got_exception);
    // 2b-ii: on a clean single-recipe `SubReturn`, thread the callee result
    // into the root's operand-stack result slot and walk the ROOT top-level to
    // compile the bridge (the recorded callee continuation + the root
    // continuation form one bridge body).
    // Other shapes / outcomes log + abort (trace discarded).
    let subwalk_result = match &walk {
        Some(Ok((crate::jitcode_dispatch::DispatchOutcome::SubReturn { result: Some(r) }, _))) => {
            Some(*r)
        }
        _ => None,
    };
    if let Some(mut result) = subwalk_result {
        // Depth-N: after the deepest callee returns cleanly, drive each paused
        // middle frame forward from second-deepest (recipes[len-2]) out to the
        // outermost (recipes[0]), delivering each callee's result into its
        // caller's residual-call return register (make_result_of_lastop) and
        // rebinding `result` to that caller's own return.  Any non-portable
        // middle shape yields `None` and drops to the journal-rollback abort
        // epilogue below, so a carrier we cannot compile deopts cleanly.
        let n = carrier.recipes.len();
        let want_compile = n >= 1 && n <= crate::jitcode_dispatch::fbw_max_multiframe_depth();
        let mut middles_ok = true;
        if want_compile {
            for i in (0..n.saturating_sub(1)).rev() {
                // recipes[i]'s paused parents are the shallower frames
                // recipes[..i] (the root sits above them all).
                match drive_middle_frame_and_thread(
                    ctx,
                    &session,
                    sym,
                    root_pc,
                    root_ec,
                    &carrier.recipes[i],
                    &carrier.recipes[..i],
                    result,
                ) {
                    Some(mid_result) => result = mid_result,
                    None => {
                        middles_ok = false;
                        break;
                    }
                }
            }
        }
        if want_compile && middles_ok {
            if inject_root_call_result(sym, root_pc, result) {
                crate::jitcode_dispatch::census_record("P2Drain::CompileRoot");
                let root_py_pc = crate::py_coord::resume_py_pc_for_jitcode_word(
                    carrier.root_jitcode_index,
                    root_pc as i32,
                ) as usize;
                // The sub-walks above already applied each frame's eager stores
                // and journaled them; hand those journals to the root walk so its
                // epilogue settles them exactly once.
                return full_body_walk_trace(
                    ctx,
                    sym,
                    w_code,
                    root_py_pc,
                    cf_addr,
                    WalkJournals::Keep,
                );
            }
            crate::jitcode_dispatch::census_record("P2Drain::ResultSlotUnresolved");
        }
    }

    // `finishframe_exception` at the carrier boundary: the inlined callee's
    // sub-walk raised and had no local handler, so it surfaced `SubRaise`.
    // Scan paused middle frames from deepest to shallowest. A frame without a
    // covering handler closes as the exception passes through; a middle-frame
    // handler needs its own continuation walk, so that shape still declines.
    // Once the exception reaches the ROOT frame, deliver it to a covering
    // handler and continue the root walk. Without this the raise is dropped and
    // re-interpreted every iteration (deopt-storm).
    //
    // A root frame with no covering handler is the framestack-exhausted arm of
    // the same walk: `finishframe_exception` runs out of frames to scan and
    // reaches `compile_exit_frame_with_exception`.  Seeding `catch_target: None`
    // routes the root walk to that exit instead of declining — the decline
    // records the bridge guard as permanently undecidable, so every later
    // failure of that guard short-circuits into a full blackhole resume.
    let subwalk_raise = match &walk {
        Some(Ok((crate::jitcode_dispatch::DispatchOutcome::SubRaise { exc, exc_concrete }, _))) => {
            Some((*exc, *exc_concrete))
        }
        _ => None,
    };
    if let Some((exc, exc_concrete)) = subwalk_raise {
        // `finishframe_exception` walks the framestack from the top down,
        // popping each frame whose pc carries no `catch_exception`.  The
        // deepest frame is the sub-walk that just raised and its
        // `carrier_ec_leave` already ran; the middles between it and the root
        // are `recipes[n-2]..recipes[0]`, and the root is handled below.
        //
        // Decide the whole chain BEFORE closing anything: `carrier_ec_leave`
        // performs the concrete leave as well as recording it, so a mid-chain
        // decline that had already closed a frame would leave the interpreter's
        // `topframeref` one level short for the blackhole replay the abort
        // epilogue hands the iteration to.
        let n = carrier.recipes.len();
        let middles = &carrier.recipes[..n.saturating_sub(1)];
        let middles_ok = n >= 1
            && n <= crate::jitcode_dispatch::fbw_max_multiframe_depth()
            && middles.iter().rev().all(|middle| {
                let Some(middle_pjc) = crate::state::pyjitcode_for_code(middle.code_ptr) else {
                    crate::jitcode_dispatch::census_record("P2Drain::NoMiddlePjc");
                    return false;
                };
                // A middle that catches has to be ENTERED at its handler and
                // walked on from there, which can put every shallower frame
                // back on the value path.  Decline that shape; the exception
                // passes straight through the rest.
                if carrier_catch_target(
                    middle_pjc.jitcode.code.as_slice(),
                    middle.jitcode_pc as usize,
                    "middle",
                )
                .is_some()
                {
                    crate::jitcode_dispatch::census_record("P2Drain::MiddleCatchesRaise");
                    return false;
                }
                true
            });
        if middles_ok {
            // Deepest-first, the order the exception actually crosses them, so
            // the nodes prepend into the same chain order the interpreter
            // builds.
            for middle in middles.iter().rev() {
                record_carrier_crossed_frame_traceback(ctx, middle, exc, exc_concrete);
                crate::jitcode_dispatch::carrier_ec_leave(ctx, sym, true);
            }
            let catch_target = carrier_root_catch_target(sym, root_pc);
            crate::jitcode_dispatch::set_carrier_raise_seed(
                crate::jitcode_dispatch::CarrierRaiseSeed {
                    exc,
                    exc_concrete,
                    catch_target,
                },
            );
            crate::jitcode_dispatch::census_record(if catch_target.is_some() {
                "P2Drain::CompileRootRaise"
            } else {
                "P2Drain::CompileRootRaiseEscape"
            });
            let root_py_pc = crate::py_coord::resume_py_pc_for_jitcode_word(
                carrier.root_jitcode_index,
                root_pc as i32,
            ) as usize;
            let action =
                full_body_walk_trace(ctx, sym, w_code, root_py_pc, cf_addr, WalkJournals::Keep);
            // Defensive: `dispatch_via_miframe` consumes the seed, but a
            // walk that early-declines before reaching it would leave the
            // seed standing and leak it into a later unrelated walk. Clear
            // any residual seed so exactly this walk can observe it.
            let _ = crate::jitcode_dispatch::take_carrier_raise_seed();
            return action;
        }
    }

    let p2_diag = crate::jitcode_dispatch::p2_diag_enabled();
    match &walk {
        Some(Ok((outcome, end_pc))) => {
            if p2_diag {
                eprintln!(
                    "[p2-drain] callee sub-walk OK recipe_py_pc={} entry={entry} end_pc={end_pc} outcome={outcome:?}",
                    crate::py_coord::resume_py_pc_for_jitcode_word(
                        recipe.jitcode_index,
                        recipe.jitcode_pc
                    )
                );
            }
            crate::jitcode_dispatch::census_record("P2Drain::SubWalkOk");
        }
        Some(Err(e)) => {
            if p2_diag {
                let raw_code = recipe.code_ptr as *const CodeObject;
                let (callee_name, source_path) = if raw_code.is_null() {
                    ("<unknown>", "<unknown>")
                } else {
                    unsafe {
                        (
                            (*raw_code).obj_name.as_str(),
                            (*raw_code).source_path.as_str(),
                        )
                    }
                };
                let stop_op = match e {
                    crate::jitcode_dispatch::DispatchError::LoopBearingCalleeInlineUnsupported {
                        pc,
                    }
                    | crate::jitcode_dispatch::DispatchError::AbortPermanentMarkerReached {
                        pc,
                    } => crate::jitcode_runtime::decode_op_at(
                        callee_pjc.jitcode.code.as_slice(),
                        *pc,
                    )
                    .map(|op| op.opname)
                    .unwrap_or("<undecodable>"),
                    _ => "<n/a>",
                };
                eprintln!(
                    "[p2-drain] callee sub-walk STOP callee={callee_name} \
                     source={source_path} recipe_py_pc={} entry={entry} \
                     stop_op={stop_op} err={e:?}",
                    crate::py_coord::resume_py_pc_for_jitcode_word(
                        recipe.jitcode_index,
                        recipe.jitcode_pc
                    )
                );
            }
            crate::jitcode_dispatch::census_record("P2Drain::SubWalkStop");
        }
        None => {
            crate::jitcode_dispatch::census_record("P2Drain::SubWalkSetupNone");
        }
    }

    // `pyjitpl.py:2949 run_blackhole_interp_to_cancel_tracing` →
    // `blackhole.py:1799 convert_and_run_from_pyjitpl`.  The sub-walk above is
    // the reconstructed callee's ONE real execution (`drive_bridge_frame_subwalk`
    // is an authoritative executor), so once its odometer has moved the drain
    // may not hand the guard back to a blackhole resume from `rd_numb`: the
    // store journal unwinds the eager stores, but nothing unwinds a residual
    // call that wrote the heap or entered a Python frame, and the guard resume
    // re-runs the callee from the same coordinate.  Upstream has no such
    // rewind — `_handle_guard_failure` ends `assert False, "should always
    // raise"` (`pyjitpl.py:2956`).  Drive the frames the sub-walk reached
    // instead; they were latched at its stop coordinate.
    //
    // Ordering: adopt BEFORE `discard_bridge_carrier_walk`, whose
    // `carrier_ec_leave` closes the scopes the chain's frames still belong to.
    // A declined adopt leaves everything to the rollback below, which is the
    // pre-existing behaviour.
    let live_root_addr = sym.live_vable_frame_addr();
    let adopted = crate::jitcode_dispatch::fbw_executed_effect_count() != effects_at_entry
        && try_adopt_blackhole(ctx, cf_addr, live_root_addr, WalkEndCommitLeg::CarrierAbort);
    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
        eprintln!(
            "[p2-drain-abort] effects={} adopted={adopted}",
            crate::jitcode_dispatch::fbw_executed_effect_count() - effects_at_entry,
        );
    }
    discard_bridge_carrier_walk(ctx, sym, entry_depth, pre_pos, &pre_virtualref_boxes);
    crate::jitcode_dispatch::bool_box_truth_reset();
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    if adopted {
        // The chain ran the callee forward from where the sub-walk stopped, so
        // the eager stores it journaled stand exactly once.
        crate::jitcode_dispatch::fbw_store_journal_commit();
    } else {
        // Non-commit epilogue: the sub-walk concrete-executed the reconstructed
        // callee, and the blackhole replays it from the guard, so restore the
        // pre-walk heap rather than dropping the journals (which would leave every
        // eager store standing to be applied a second time).
        crate::jitcode_dispatch::fbw_store_journal_rollback();
    }
    p2_drain_abort()
}

/// Middle-frame drive for the DEFAULT drain: reconstruct one paused middle frame
/// (`middle`), deliver its callee's `child_result` into its residual-call return
/// register (`make_result_of_lastop`), and walk it forward to its own
/// `SubReturn`.  `paused_parents` are the frames shallower than `middle` (root
/// sits above them) so its in-callee guard snapshots encode the full paused
/// chain.  Returns the middle's result on a clean return, or `None` on any
/// non-portable shape (recording a `P2Drain::*` census) so the caller falls
/// through to the drain's journal-rollback abort epilogue.
///
/// The failure arms do NOT cut/reset the journal here: the drain's epilogue
/// rolls it back (every driven frame's eager stores) for the blackhole replay,
/// so a reset would leave those stores double-applied.  Mirrors resume.py:
/// 1049-1056 `finishframe` delivering the child result into the parent's dst.
#[allow(clippy::too_many_arguments)]
fn drive_middle_frame_and_thread<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    session: &std::cell::RefCell<crate::jitcode_dispatch::WalkSession>,
    sym: &mut Sym,
    root_pc: usize,
    root_ec: *const pyre_interpreter::PyExecutionContext,
    middle: &majit_metainterp::ReconstructRecipe,
    paused_parents: &[majit_metainterp::ReconstructRecipe],
    child_result: majit_ir::OpRef,
) -> Option<majit_ir::OpRef> {
    let Some((_pending, middle_argboxes_r)) =
        crate::state::setup_reconstructed_callee_frame(ctx, middle, root_ec, Vec::new())
    else {
        crate::jitcode_dispatch::census_record("P2Drain::MiddleSetupFailed");
        return None;
    };
    let Some(middle_pjc) = crate::state::pyjitcode_for_code(middle.code_ptr) else {
        crate::jitcode_dispatch::census_record("P2Drain::NoMiddlePjc");
        return None;
    };
    let middle_entry = select_recipe_entry(
        middle.jitcode_index,
        middle_pjc.jitcode.index() as i32,
        middle.jitcode_pc,
    );
    let Some(middle_entry) = middle_entry else {
        crate::jitcode_dispatch::census_record("P2Drain::NoMiddleEntry");
        return None;
    };
    let middle_w_globals = crate::state::recover_inline_callee_globals(middle.code_ptr) as usize;
    let middle_nlocals = middle.nlocals.min(middle.concrete_r.len());
    let middle_local_oprefs = &middle.registers_r[..middle_nlocals.min(middle.registers_r.len())];
    let middle_local_concretes = &middle.concrete_r[..middle_nlocals];
    let middle_stack_end = middle.valuestackdepth.min(middle.registers_r.len());
    let middle_stack_oprefs =
        &middle.registers_r[middle_nlocals.min(middle_stack_end)..middle_stack_end];
    let middle_concrete_stack_end = middle.valuestackdepth.min(middle.concrete_r.len());
    let middle_stack_concretes = &middle.concrete_r
        [middle_nlocals.min(middle_concrete_stack_end)..middle_concrete_stack_end];
    let middle_walk = crate::jitcode_dispatch::drive_bridge_middle_frame(
        ctx,
        session,
        sym,
        root_pc,
        &middle_pjc,
        middle.code_ptr as usize,
        middle_w_globals,
        middle_entry,
        &middle_argboxes_r,
        &middle.registers_i,
        &middle.registers_f,
        middle_local_oprefs,
        middle_local_concretes,
        middle_stack_oprefs,
        middle_stack_concretes,
        paused_parents,
        child_result,
    );
    let got_exception = matches!(
        &middle_walk,
        Some(Ok((
            crate::jitcode_dispatch::DispatchOutcome::SubRaise { .. },
            _
        )))
    );
    crate::jitcode_dispatch::carrier_ec_leave(ctx, sym, got_exception);
    match middle_walk {
        Some(Ok((
            crate::jitcode_dispatch::DispatchOutcome::SubReturn {
                result: Some(mid_result),
            },
            _,
        ))) => Some(mid_result),
        _ => {
            crate::jitcode_dispatch::census_record("P2Drain::MiddleDriveFailed");
            None
        }
    }
}

fn seed_loop_entry_ref_slots(
    pcdep: &[(u8, u16, u16)],
    num_vable_scalars: usize,
    mut box_at: impl FnMut(usize) -> Option<majit_ir::OpRef>,
    mut seed: impl FnMut(u8, u16, majit_ir::OpRef),
) {
    for &(bank, color, slot) in pcdep {
        if bank != 1 {
            continue;
        }
        if let Some(opref) = box_at(num_vable_scalars + slot as usize)
            && !opref.is_none()
        {
            seed(color as u8, slot, opref);
        }
    }
}

/// Hand a blackhole `ContinueRunningNormally` back the way upstream does:
/// resume on the frame the blackhole has been writing, not on one rebuilt from
/// the terminal register banks.
///
/// `handle_jitexception` (`warmspot.py:970-982`) re-enters the portal with the
/// greens and reds the exception carries; the frame is among the reds, and
/// nothing reconstructs its contents.  It can afford that because the
/// blackhole's virtualizable ops are write-through —
/// `bhimpl_setarrayitem_vable_r` and `bhimpl_setfield_vable_i`
/// (`blackhole.py:1390-1490`) fetch the array out of the virtualizable and
/// store into it — so by the time `bhimpl_jit_merge_point` raises
/// (`blackhole.py:1068-1069`) the frame already holds the driven region's
/// result.  That raise only fires at the bottommost level, which is the portal
/// frame: `convert_and_run_from_pyjitpl` links `framestack[0]` last, so
/// `nextblackholeinterp is None` names the root and no other.
///
/// pyre's walked frame has two representations, so "the frame the blackhole was
/// writing" has to be named.  The drive's vable ops address the frame held in
/// the jitcode's own frame register, which the caller gates to be the LIVE
/// root; the portal epilogue then publishes the SNAPSHOT
/// (`eval.rs` `restore_resume_state_from(&executed_frame)`).  Mirroring live
/// into the snapshot is what makes that epilogue hand over the drive's result
/// instead of the walk's stale copy — the same move, for the same reason, that
/// the escape leg makes.
///
/// `last_instr` is the one coordinate the drive does not leave resume-ready: it
/// stops at the merge point having recorded the pc it is AT, and resuming there
/// needs the pc before it.  Mirroring without this normalization resumes one
/// opcode past the loop header — measured on
/// `getframe_root_loop_force_blackhole_crn` and
/// `getframe_root_loop_force_while_merge`, which then produce no output at all.
///
/// ⭐Infallible, and that is the whole point.  The register-image rebuild this
/// replaces had to validate pcdep coverage, register-bank bounds and NULL Refs,
/// and every one of those checks could only fail AFTER the drive had already
/// executed the region — a decline that hands an executed region back to a
/// replay that runs it again.  Upstream has no such path, and neither does
/// this.
///
/// Both adopt arms come here.  The multi-frame one has already folded its live
/// root into the snapshot by the time it does, so the mirror is a no-op there
/// and only the coordinate lands; routing it through anyway keeps one
/// definition of what adopting a CRN means.
fn adopt_blackhole_crn(snapshot: usize, live_root: usize, resume_py_pc: usize) {
    unsafe {
        let live = &*(live_root as *const pyre_interpreter::PyFrame);
        let snap = &mut *(snapshot as *mut pyre_interpreter::PyFrame);
        snap.restore_resume_state_from(live);
        snap.last_instr = resume_py_pc as isize - 1;
    }
}

/// ⚠️`drive_single_frame_blackhole` **executes** — `bh.resume_mainloop()`
/// (`jitdriver.rs`) runs the region, allocating and calling.  So every decline
/// AFTER the drive discards work that already happened and returns to a caller
/// that replays it: `try_adopt_force_blackhole` reports `false`, and
/// `run_perfn_walk`'s epilogue then takes the committed escape pc or the legacy
/// replay, both of which re-enter the region.  The locals publish below carries
/// its own undo for exactly that reason; the `mfdbg!` sibling names the same
/// hazard at the multi-frame `resume_py_pc` check as the reason not to add a
/// check there.
///
/// Declines therefore split at the drive:
/// - BEFORE it are free — nothing has run.
/// - AFTER it are the double-apply class, and worse than "a lost
///   optimization": the undo the publish below carries restores the frame, but
///   the region's residual calls already mutated the heap and no undo reaches
///   that.  Measured with the CRN arm forced to decline on
///   `getframe_root_loop_force_blackhole_crn_nonidempotent`: the accumulator
///   comes back right at 199990000 and the list holds 20005 entries for 20000
///   iterations, one extra per declined drive.
///
/// ⭐So this path now has NO post-drive decline, which is what makes it safe to
/// drive loop-bearing bodies at all.  Every arm adopts: `DoneWithThisFrame*`
/// and `ExitFrameWithExceptionRef` record a concrete frame result, and
/// `ContinueRunningNormally` hands the frame over through
/// [`adopt_blackhole_crn`], which validates nothing because it rebuilds
/// nothing. An empty green list is a codewriter contract violation with no
/// upstream counterpart, so it fails loudly instead of returning to replay
/// after the drive has executed.
///
/// Over `pyre/bench/synth` that is 50 frame-terminal adoptions plus 15 CRN
/// adoptions (five each from the three `getframe_root_loop_force_*` fixtures),
/// zero declines, and no corpus file's output differing from the
/// interpreter's.
fn try_adopt_single_frame_blackhole(
    ctx: &mut TraceCtx,
    cf_addr: usize,
    live_root_addr: usize,
    commit_leg: WalkEndCommitLeg,
) -> bool {
    macro_rules! sfdbg {
        ($($a:tt)*) => {
            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                eprintln!("[s1-adopt] {}", format!($($a)*));
            }
        };
    }
    // A zero-effect overlong walk deliberately permits entry replay even when
    // no blackhole image was representable. Only an effectful TraceTooLong
    // carries the irrevocable, fully preflighted adoption contract.
    let trace_too_long = commit_leg == WalkEndCommitLeg::TraceTooLong
        && crate::jitcode_dispatch::fbw_executed_effect_count() != 0;
    // Both full-image legs resume the blackhole at a post-step pc it may leave
    // by arbitrary control flow, so the root operand stack has to be published
    // with the locals (`fill_trace_too_long_register_banks`).  The vable-escape
    // leg instead resumes immediately after one forcing residual and keeps its
    // narrower resume-marker image.
    //
    // `WalkAbort` deliberately does NOT join `trace_too_long` above: that flag
    // promotes a post-latch decline to a release assert, which the too-long arm
    // earns by aborting only once `latch_abort_blackhole` has succeeded
    // (`trace_too_long_abort_safe`).  A capability-gap abort fires regardless of
    // the latch, so its adopt must still be able to decline into legacy replay.
    let publishes_root_stack =
        commit_leg == WalkEndCommitLeg::TraceTooLong || commit_leg == WalkEndCommitLeg::WalkAbort;
    let Some(mut latched) = crate::jitcode_dispatch::take_single_frame_blackhole() else {
        assert!(
            !trace_too_long,
            "preflighted trace-too-long blackhole image disappeared before adoption"
        );
        return false;
    };
    let jitcode_index = match i32::try_from(latched.miframe.jitcode.index()) {
        Ok(index) => index,
        Err(_) => {
            assert!(
                !trace_too_long,
                "preflighted trace-too-long jitcode index no longer fits i32"
            );
            return false;
        }
    };
    if ctx.virtualizable_info().is_none() {
        assert!(
            !trace_too_long,
            "preflighted trace-too-long virtualizable info disappeared"
        );
        return false;
    }
    // The resumed jitcodes operate on `PyFrame`, regardless of which
    // translator-state vinfo happens to remain installed on `TraceCtx` at the
    // walk epilogue.  RPython gets this from each vable field descriptor's
    // `get_vinfo()`; retain the canonical PyFrame Arc for the whole drive.
    let pyframe_vinfo = crate::frame_layout::build_pyframe_virtualizable_info();
    let virtualizable_info = std::sync::Arc::as_ptr(&pyframe_vinfo);
    let Some(stack_base) = crate::state::concrete_nlocals(cf_addr) else {
        assert!(
            !trace_too_long,
            "preflighted trace-too-long frame lost its concrete locals base"
        );
        return false;
    };
    // Two gates used to sit here, and both were scaffolding around a decline
    // that no longer exists.
    //
    // One refused to drive any body containing a loop header, so that no drive
    // could reach a `jit_merge_point` and hence the `ContinueRunningNormally`
    // arm — the only arm that could reject the image.  The other pre-checked
    // pcdep trivia purely to pre-empt the pcdep lookup the register-image
    // rebuild did, deciding before the drive what would otherwise have been
    // decided after the region had already run.  The CRN arm rejects nothing now
    // ([`adopt_blackhole_crn`]) and reads no pcdep, so all either gate could do
    // is refuse an adopt that would have succeeded.
    //
    // Worth recording why gating was never the answer: a post-drive decline is
    // unsound in a way no undo can repair.  The publish below carries an undo,
    // and it restores the frame exactly — but the drive also ran the region's
    // residual calls, and their heap effects are not frame state.  Measured
    // with the CRN arm forced to decline on
    // `getframe_root_loop_force_blackhole_crn_nonidempotent`: `total` comes
    // back correct at 199990000 while the list holds 20005 entries for 20000
    // iterations, one extra per declined drive.  The undo makes that failure
    // QUIETER, not smaller — before it the accumulator was visibly wrong too.
    // So the decline had to go, not be gated around.
    //
    // The escape flush that ran ahead of the forcing residual is all-or-
    // nothing, and its decline is what the vable-escape latch is gated on
    // (`committed_frame_escape_pc().is_none()`).  The LOCALS half is not
    // optional: every LOAD_FAST lowers to
    // `getarrayitem_vable_r` on the frame the latched image's register names,
    // so without it the drive reads whatever that frame held before the walk
    // began, and a local the walk assigned comes back null.  Publish that half
    // here, and withdraw it if it cannot complete — the withdrawal is honest
    // only because nothing has run at that point.
    //
    // Trace-too-long is also post-step: its MIFrame `resume_pc` and concrete
    // banks already describe the next jitcode instruction.  The detached
    // `snapshot_for_tracing` therefore owns the matching active operand stack
    // and Python-frame coordinate.  RPython's MIFrame and red frame are one
    // coherent per-call state at this boundary; publish the snapshot stack to
    // the same live red frame before driving instead of leaving only
    // `valuestackdepth` to advance over null slots.
    //
    // Which frame gets it is an identity question, and the walked frame has two
    // representations: `cf_addr` is the `snapshot_for_tracing` copy the walk
    // steps concretely, `live_root_addr` the frame the compiled loop runs on.
    // The register is recovered the same way `try_adopt_multi_frame_blackhole`
    // recovers `per_frame[0]`, and `seed_virtualizable_boxes` bakes that root
    // vable identity against the live address whenever there is one, so the
    // comparison uses the same address under the same fallback.  Code-object
    // equality would not do: two invocations of one function share a code
    // object, and the shadow belongs to exactly one of them.
    let root_addr = if live_root_addr != 0 {
        live_root_addr
    } else {
        cf_addr
    };
    let (frame_reg, _) = crate::state::portal_red_regs_at(jitcode_index);
    let vable_frame = latched
        .miframe
        .ref_values
        .get(frame_reg as usize)
        .copied()
        .flatten()
        .unwrap_or(0) as usize;
    if vable_frame == 0 || vable_frame != root_addr {
        assert!(
            !trace_too_long,
            "preflighted trace-too-long frame identity changed before adoption"
        );
        return false;
    }
    let Some(mut locals_undo) = crate::state::capture_frame_locals(vable_frame) else {
        assert!(
            !trace_too_long,
            "preflighted trace-too-long frame locals became uncapturable"
        );
        return false;
    };
    let mut captured_stack = if publishes_root_stack {
        // The two sources side by side, so the reason this leg has its own is
        // checkable at runtime rather than argued.  Measured on the one adopt
        // in `list_length_hint_validate`: `snapshot-array=[0]` where
        // `mirror=[0x99651ab18]` — the array reads NULL, the mirror holds the
        // live operand.
        if commit_leg == WalkEndCommitLeg::WalkAbort
            && crate::jitcode_dispatch::fbw_debug_abort_enabled()
        {
            let from_array = crate::state::capture_frame_stack_for_publish(cf_addr, vable_frame)
                .map(|stack| stack.roots_snapshot());
            let from_mirror = latched
                .mirror_stack
                .as_ref()
                .map(|mirror| (mirror.py_pc, mirror.slots.clone()));
            eprintln!("[wa-stack] snapshot-array={from_array:x?} mirror={from_mirror:x?}");
        }
        // `ABORT_TOO_LONG` stops at an opcode boundary, where the snapshot
        // array is the image RPython would copy.  `WalkAbort` stops INSIDE an
        // opcode, and for a root walk that array was never written at all — it
        // still holds the pre-walk stack (see `capture_frame_stack_from_mirror`).
        // Take the walker's OpRef mirror, which the latch resolved while the
        // concrete side tables were still live.
        let captured = if commit_leg == WalkEndCommitLeg::WalkAbort {
            latched.mirror_stack.as_ref().and_then(|mirror| {
                crate::state::capture_frame_stack_from_mirror(
                    vable_frame,
                    mirror.py_pc,
                    &mirror.slots,
                )
            })
        } else {
            crate::state::capture_frame_stack_for_publish(cf_addr, vable_frame)
        };
        match captured {
            Some(stack) => Some(stack),
            None => {
                assert!(
                    !trace_too_long,
                    "preflighted trace-too-long operand stack became uncapturable"
                );
                if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-blackhole] single-frame operand stack not capturable \
                         (leg={commit_leg:?}) — legacy replay kept"
                    );
                }
                return false;
            }
        }
    } else {
        None
    };
    // `take_single_frame_blackhole` removed the image from the TLS root
    // walker. `write_back_outer_locals` may box Int/Float locals and collect
    // before the blackhole driver installs its own packed roots. Bridge that
    // interval explicitly, including the detached snapshot's active operand
    // stack, and copy forwarding updates into the MIFrame.
    let image_ref_locations: Vec<usize> = latched
        .miframe
        .ref_values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.map(|_| index))
        .collect();
    let mut image_ref_roots: Vec<i64> = image_ref_locations
        .iter()
        .map(|&index| latched.miframe.ref_values[index].expect("location came from Some"))
        .collect();
    let image_exception_root = (latched.last_exc_value != 0).then(|| {
        let index = image_ref_roots.len();
        image_ref_roots.push(latched.last_exc_value);
        index
    });
    let root_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
    unsafe {
        majit_gc::shadow_stack::push_resume_ref_roots(image_ref_roots.as_mut_slice());
        majit_gc::shadow_stack::push_resume_ref_roots(locals_undo.as_mut_slice());
        if let Some(stack) = captured_stack.as_mut() {
            majit_gc::shadow_stack::push_resume_ref_roots(stack.roots_mut());
        }
    }
    if !crate::state::write_back_outer_locals(ctx, vable_frame) {
        crate::state::restore_frame_locals(vable_frame, &locals_undo);
        majit_gc::shadow_stack::pop_resume_ref_roots_to(root_depth);
        assert!(
            !trace_too_long,
            "preflighted trace-too-long locals publication declined"
        );
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!("[fbw-blackhole] single-frame locals publish declined — legacy replay kept");
        }
        return false;
    }
    for (&index, &forwarded) in image_ref_locations.iter().zip(&image_ref_roots) {
        latched.miframe.ref_values[index] = Some(forwarded);
    }
    if let Some(index) = image_exception_root {
        latched.last_exc_value = image_ref_roots[index];
    }
    let committed_root_addr = latched
        .miframe
        .ref_values
        .get(frame_reg as usize)
        .copied()
        .flatten()
        .expect("preflighted frame register disappeared after forwarding")
        as usize;
    if let Some(stack) = captured_stack.as_ref() {
        if !crate::state::publish_captured_frame_stack(committed_root_addr, stack) {
            crate::state::restore_frame_locals(committed_root_addr, &locals_undo);
            majit_gc::shadow_stack::pop_resume_ref_roots_to(root_depth);
            assert!(
                !trace_too_long,
                "preflighted trace-too-long operand-stack publication declined"
            );
            return false;
        }
    }
    // Locals plus the no-allocation stack publication are the last recoverable
    // adoption gates. Their undo image only needs rooting through this point;
    // after it the blackhole may execute irreversible effects and the handoff
    // must commit or fail loudly, never return to trace-entry replay.
    majit_gc::shadow_stack::pop_resume_ref_roots_to(root_depth);
    drop(locals_undo);
    drop(image_ref_roots);
    drop(captured_stack);
    let terminal = majit_metainterp::drive_single_frame_blackhole(
        &mut latched.miframe,
        majit_metainterp::blackhole::StateFieldLayout::default(),
        virtualizable_info,
        // RPython threads the current MIFrame's own red virtualizable through
        // blackhole execution. Keep the explicit driver field aligned with
        // the same live frame carried in the MIFrame register; the tracing
        // snapshot is only the epilogue's committed copy.
        committed_root_addr as i64,
        stack_base,
        ctx.metainterp_sd().as_ref(),
        latched.last_exc_value,
        latched.raising_exception,
    );
    // The blackhole roots and forwards its Ref bank in place. A collection
    // during the drive can therefore move the live frame away from the
    // pre-drive `committed_root_addr`; recover the authoritative post-drive
    // identity from the same per-frame red register before dereferencing it.
    let forwarded_root_addr = terminal
        .registers_r
        .get(frame_reg as usize)
        .copied()
        .filter(|&addr| addr != 0)
        .expect("post-blackhole frame register lost the live frame identity")
        as usize;
    // Vable opcodes address the frame carried in the MIFrame's red register,
    // i.e. the live frame whose locals were published above. Fold every
    // blackhole write back into the tracing snapshot before the portal
    // epilogue propagates that snapshot. This matters for terminal exceptions:
    // their traceback keeps `tb_frame.f_locals` observable after the walk.
    if forwarded_root_addr != cf_addr {
        unsafe {
            (*(cf_addr as *mut pyre_interpreter::PyFrame)).restore_resume_state_from(
                &*(forwarded_root_addr as *const pyre_interpreter::PyFrame),
            );
        }
    }

    // Its own prefix: this is not a decline, and reading it as one is exactly
    // the confusion that made the CRN arm look unreachable.
    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
        eprintln!(
            "[s1-drive-outcome] {}",
            match terminal.outcome {
                majit_metainterp::jitexc::JitException::ContinueRunningNormally { .. } =>
                    "ContinueRunningNormally",
                majit_metainterp::jitexc::JitException::DoneWithThisFrameVoid =>
                    "DoneWithThisFrameVoid",
                majit_metainterp::jitexc::JitException::DoneWithThisFrameInt(_) =>
                    "DoneWithThisFrameInt",
                majit_metainterp::jitexc::JitException::DoneWithThisFrameRef(_) =>
                    "DoneWithThisFrameRef",
                majit_metainterp::jitexc::JitException::DoneWithThisFrameFloat(_) =>
                    "DoneWithThisFrameFloat",
                majit_metainterp::jitexc::JitException::ExitFrameWithExceptionRef(_) =>
                    "ExitFrameWithExceptionRef",
            }
        );
    }
    match terminal.outcome {
        majit_metainterp::jitexc::JitException::ContinueRunningNormally {
            ref green_int, ..
        } => {
            // The portal's `jit_merge_point` carries `py_pc` as its first int
            // green. Upstream indexes that green unconditionally
            // (`warmspot.py:973-976`); a missing one is a codewriter invariant
            // failure, never a licence to replay an already-driven region.
            let &resume_py_pc = green_int.first().expect(
                "post-blackhole ContinueRunningNormally has no resume pc; \
                 replay is forbidden after blackhole effects",
            );
            let resume_py_pc = resume_py_pc as usize;
            adopt_blackhole_crn(cf_addr, forwarded_root_addr, resume_py_pc);
            sfdbg!("adopted with green resume_py_pc={resume_py_pc}");
            WALK_END_RESTART_PC.with(|slot| slot.set(Some(resume_py_pc)));
        }
        majit_metainterp::jitexc::JitException::DoneWithThisFrameVoid => {
            crate::jitcode_dispatch::fbw_finish_concrete_set(crate::state::ConcreteValue::Null);
        }
        majit_metainterp::jitexc::JitException::DoneWithThisFrameInt(value) => {
            crate::jitcode_dispatch::fbw_finish_concrete_set(crate::state::ConcreteValue::Int(
                value,
            ));
        }
        majit_metainterp::jitexc::JitException::DoneWithThisFrameRef(value) => {
            crate::jitcode_dispatch::fbw_finish_concrete_set(crate::state::ConcreteValue::Ref(
                value.as_usize() as pyre_object::PyObjectRef,
            ));
        }
        majit_metainterp::jitexc::JitException::DoneWithThisFrameFloat(value) => {
            crate::jitcode_dispatch::fbw_finish_concrete_set(crate::state::ConcreteValue::Float(
                value,
            ));
        }
        majit_metainterp::jitexc::JitException::ExitFrameWithExceptionRef(value) => {
            publish_terminal_raise_coordinate(
                jitcode_index,
                terminal.last_opcode_position,
                &[cf_addr, forwarded_root_addr],
            );
            crate::jitcode_dispatch::fbw_finish_raise_set(crate::state::ConcreteValue::Ref(
                value.as_usize() as pyre_object::PyObjectRef,
            ));
        }
    }
    let _ = crate::jitcode_dispatch::take_committed_frame_escape_pc();
    crate::jitcode_dispatch::discard_escape_flush_undo();
    crate::jitcode_dispatch::fbw_foriter_inflight_clear();
    // The blackhole ran the region to a frame terminal, so the resume is
    // the frame's RESULT, not a pc that re-runs anything.
    let _ = commit_walk_end(commit_leg, WalkEndResume::Terminal);
    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
        eprintln!(
            "[fbw-blackhole] adopted single-frame terminal at jitcode_index={} \
             position={} last_opcode_position={}",
            jitcode_index, terminal.position, terminal.last_opcode_position,
        );
    }
    true
}

fn try_adopt_multi_frame_blackhole(
    ctx: &mut TraceCtx,
    cf_addr: usize,
    live_root_addr: usize,
    commit_leg: WalkEndCommitLeg,
) -> bool {
    // Every arm below returns to the legacy escape/replay path.  Name each one
    // under `PYRE_FBW_DEBUG_ABORT`, the way `build_multi_frame_miframe`'s
    // `s2dbg!` names its own: a silent decline is indistinguishable from the
    // adopt never being reached, and the two want different fixes.
    //
    // ⚠️The declines are NOT interchangeable — see
    // [`try_adopt_single_frame_blackhole`].  Those before
    // `drive_multi_frame_blackhole` are free; those after it discard a chain
    // that already ran and hand back to a caller that replays it, and no undo
    // reaches the heap effects that chain committed.
    //
    // No post-drive decline survives here. An empty `green_int` is a codewriter
    // invariant failure rather than a runtime outcome; the three checks that
    // used to sit beside it existed only for the retired register-image
    // rebuild.
    //
    // Note that they could not have been hoisted instead.  The rebuild's index
    // here is `terminal.jitcode_index` = `bh.jitcode.index()` of whichever
    // interpreter the chain ended in (`BlackholeTerminalImage::take_from`), and
    // the run can enter jitcodes that are not levels of `latched.framestack`,
    // so pre-checking the levels would have added a decline without discharging
    // the post-drive one.  Retiring the rebuild is what discharged them.
    macro_rules! mfdbg {
        ($($a:tt)*) => {
            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                eprintln!("[s2-adopt-decline] {}", format!($($a)*));
            }
        };
    }
    let Some(mut latched) = crate::jitcode_dispatch::take_multi_frame_blackhole() else {
        mfdbg!("no latched multi-frame image");
        return false;
    };
    let depth = latched.framestack.len();
    if ctx.virtualizable_info().is_none() {
        mfdbg!("no virtualizable_info");
        return false;
    }
    let pyframe_vinfo = crate::frame_layout::build_pyframe_virtualizable_info();
    let virtualizable_info = std::sync::Arc::as_ptr(&pyframe_vinfo);
    let Some(stack_base) = crate::state::concrete_nlocals(cf_addr) else {
        mfdbg!("cf_addr {cf_addr:#x} has no concrete nlocals");
        return false;
    };
    // Recover each level's concrete PyFrame ptr and its own fastlocals base.
    // `build_multi_frame_miframe` emits one frame per inline level's CALLER
    // plus the innermost callee, so `frames.last()` is the frame being
    // sub-walked and `frames[0]` is either the walked frame `cf_addr` itself
    // or — when the inline sits inside a residual call the walk descended
    // into — that intermediate frame.  Any level whose frame register is not
    // a concrete PyFrame declines the adopt (legacy replay handles it).
    let mut per_frame: Vec<(i64, usize)> = Vec::with_capacity(latched.framestack.frames.len());
    for (index, frame) in latched.framestack.frames.iter().enumerate() {
        let Ok(jitcode_index) = i32::try_from(frame.jitcode.index()) else {
            mfdbg!(
                "frame {index}: jitcode index {} out of range",
                frame.jitcode.index()
            );
            return false;
        };
        let frame_reg = crate::state::portal_red_regs_at(jitcode_index).0;
        if frame_reg == u16::MAX {
            mfdbg!("frame {index}: jitcode {jitcode_index} has no portal frame reg");
            return false;
        }
        let Some(frame_ptr) = frame.ref_values.get(frame_reg as usize).copied().flatten() else {
            mfdbg!("frame {index}: reg {frame_reg} unstamped");
            return false;
        };
        if frame_ptr == 0 {
            mfdbg!("frame {index}: reg {frame_reg} is null");
            return false;
        }
        let Some(frame_stack_base) = crate::state::concrete_nlocals(frame_ptr as usize) else {
            mfdbg!("frame {index}: {frame_ptr:#x} has no concrete nlocals");
            return false;
        };
        per_frame.push((frame_ptr, frame_stack_base));
    }
    // The walked frame has two representations here: `cf_addr` is the
    // `snapshot_for_tracing` copy the walk steps concretely, and
    // `live_root_addr` is the frame the compiled loop runs on.  `per_frame[0]`
    // is recovered from the trace's frame register, and
    // `seed_virtualizable_boxes` bakes that root vable identity against the
    // live address whenever there is one (`state.rs`), so an identity question
    // has to be asked against the same address under the same fallback.
    let root_addr = if live_root_addr != 0 {
        live_root_addr
    } else {
        cf_addr
    };
    // Frame-identity collapse guard.  Every level must be a distinct frame and
    // only `frames[0]` may be the walked frame: a level whose frame register
    // resolved to the root would make the relink below write a `f_backref`
    // cycle and run an inner level against the root's own virtualizable.  No
    // producer is known — an unseeded inline level has no frame object, so its
    // register never resolves to a concrete `PyFrame` and the recovery loop
    // above declines it — but the failure mode is silent, so decline rather
    // than rely on the absence.  The chain is two or three levels, so the
    // pairwise scan is free.
    for i in 0..per_frame.len() {
        if i > 0 && per_frame[i].0 == root_addr as i64 {
            mfdbg!("frame {i}: is the walked frame {root_addr:#x}");
            return false;
        }
        if per_frame[..i].iter().any(|&(ptr, _)| ptr == per_frame[i].0) {
            mfdbg!("frame {i}: {:#x} repeats an earlier level", per_frame[i].0);
            return false;
        }
    }
    // Identity gate: the recovered chain has to be rooted at the frame this
    // walk is stepping.  A chain rooted at an intermediate frame means the walk
    // descended into a residual call and inlined inside it, and then
    // `resume_py_pc` is a coordinate in `frames[0]`'s code while the restart
    // moves the frame.  There is no upstream counterpart to imitate:
    // `convert_and_run_from_pyjitpl` converts the whole framestack
    // unconditionally, and its `frames[0]` is always the portal frame, so this
    // shape cannot arise there.
    //
    // Passing this gate is not by itself sufficient for the adopt to be right.
    // The walker executes residuals CONCRETELY while an inline push never runs
    // the interpreter's call sequence, so `ec.topframeref` used to still name
    // the CALLER while an inlined callee body ran: a `sys._getframe` that is
    // itself the escaping residual read the wrong frame at walk time, and
    // adopting committed that answer where the legacy escape/replay path
    // discarded it.  `walker_ec_enter` / `walker_ec_leave` publish the callee
    // frame on the execution context at the inlined-call push, and
    // `ResidualFrameChainGuard` brackets the residual itself, which closes that
    // hole; a `sys._getframe` executed later, inside the blackhole, was already
    // correct because each level is published as it runs.
    // `synth/getframe_while_escaping_read_frame_identity` guards both readings.
    // The root gate stays regardless: a residual-intermediate chain is not an
    // MIFrame stack rooted at this portal and cannot reuse this walk's restart
    // coordinate.
    mfdbg!(
        "chain root={root_addr:#x} cf_addr={cf_addr:#x} levels=[{}]",
        per_frame
            .iter()
            .map(|&(p, b)| format!("{p:#x}/nl{b}"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    if per_frame.first().map(|&(frame_ptr, _)| frame_ptr) != Some(root_addr as i64) {
        mfdbg!(
            "chain rooted at {:#x}, not the walked frame {root_addr:#x}",
            per_frame.first().map(|&(p, _)| p).unwrap_or(0),
        );
        return false;
    }
    // `ExecutionContext::enter` parity for the resumed chain:
    // `frames[i].f_backref = frames[i - 1]`.  The blackhole re-executes each
    // level's residual `sys._getframe` against `ec.topframeref`/`f_backref`, so
    // the chain must be live before the run; it also stays live afterwards for
    // a frame the residual captured.  `frames[0]` is the walked frame, already
    // entered by its own caller, so the `ptr::eq` skip leaves its `f_backref`
    // alone — which is exactly why the root operand must be `root_addr` and not
    // the snapshot.  The snapshot is freed at the end of this walk, so a link
    // to it would survive as a dangling `f_back` for any later
    // `sys._getframe().f_back` or traceback walk.
    //
    // The links are recorded as they are overwritten, because they only hold
    // for a chain that is actually driven.  An adopt keeps them; every decline
    // below returns to legacy escape/replay, which never entered these levels,
    // so leaving a synthetic `f_back` behind would show the abandoned chain to
    // anything that still reaches one of these frames — a `sys._getframe().f_back`
    // walk or a traceback the walk handed out.
    let relink_barrier = |callee: *mut pyre_interpreter::PyFrame| {
        // `enter` stores into a frame whose allocation barrier is still in
        // effect; these frames were built many collections ago, so each
        // store needs its own remembered-set entry.
        if pyre_object::gc_hook::try_gc_owns_object(callee as *mut u8) {
            pyre_object::gc_hook::try_gc_write_barrier(callee as *mut u8);
        }
    };
    let mut saved_links: Vec<(
        *mut pyre_interpreter::PyFrame,
        *mut pyre_interpreter::PyFrame,
    )> = Vec::with_capacity(per_frame.len());
    unsafe {
        for i in 0..per_frame.len() {
            let callee = per_frame[i].0 as *mut pyre_interpreter::PyFrame;
            let f_back = if i == 0 {
                root_addr as i64
            } else {
                per_frame[i - 1].0
            } as *mut pyre_interpreter::PyFrame;
            if std::ptr::eq(callee, f_back) {
                continue;
            }
            saved_links.push((callee, (*callee).f_backref));
            (*callee).f_backref = f_back;
            relink_barrier(callee);
        }
    }
    let restore_links = |saved: &[(
        *mut pyre_interpreter::PyFrame,
        *mut pyre_interpreter::PyFrame,
    )]| {
        for &(callee, f_back) in saved {
            unsafe {
                (*callee).f_backref = f_back;
            }
            relink_barrier(callee);
        }
    };
    // Frame 0 is the walked frame, and the escape flush that ran ahead of the
    // forcing residual declined — that decline is what the latch is gated on.
    // Its LOCALS half is not optional for frame 0's level either: every
    // LOAD_FAST lowers to `getarrayitem_vable_r` on `per_frame[0]`, so without
    // it the level reads what the live frame held before the walk began.  Same
    // publish and same withdrawal as the single-frame arm.
    //
    // Every INNER level already owns the concrete red frame created at the
    // inline push.  Its standard-vable writes are mirrored directly onto that
    // frame while walking, matching RPython's one-red-frame-per-MIFrame shape;
    // no slot side-table or root-frame anchor is involved.  Frame 0 remains
    // the sole detached-snapshot case and therefore needs the explicit publish
    // below.
    let Some(mut locals_undo) = crate::state::capture_frame_locals(root_addr) else {
        mfdbg!("frame 0: {root_addr:#x} locals not capturable");
        restore_links(&saved_links);
        return false;
    };
    let mut root_stack = if latched.publish_root_stack {
        // Same split as the single-frame arm.  The multi-frame `WalkAbort`
        // latch carries no mirror (its `ctx` is the innermost callee, not
        // frame 0), so this declines and the abort keeps the legacy replay.
        let captured = if commit_leg == WalkEndCommitLeg::WalkAbort {
            latched.mirror_stack.as_ref().and_then(|mirror| {
                crate::state::capture_frame_stack_from_mirror(
                    root_addr,
                    mirror.py_pc,
                    &mirror.slots,
                )
            })
        } else {
            crate::state::capture_frame_stack_for_publish(cf_addr, root_addr)
        };
        let Some(stack) = captured else {
            mfdbg!("frame 0: active stack not capturable (leg={commit_leg:?})");
            restore_links(&saved_links);
            return false;
        };
        Some(stack)
    } else {
        None
    };
    // Taking the latch removes it from the TLS extra-root walker.  Root every
    // MIFrame Ref bank and the pending exception across root-locals boxing,
    // then copy forwarding updates back before the blackhole copies the banks.
    // This is the multi-frame counterpart of the packed image roots in
    // `try_adopt_single_frame_blackhole`.
    let image_ref_locations: Vec<(usize, usize)> = latched
        .framestack
        .frames
        .iter()
        .enumerate()
        .flat_map(|(frame_index, frame)| {
            frame
                .ref_values
                .iter()
                .enumerate()
                .filter_map(move |(reg_index, value)| value.map(|_| (frame_index, reg_index)))
        })
        .collect();
    let mut image_ref_roots: Vec<i64> = image_ref_locations
        .iter()
        .map(|&(frame_index, reg_index)| {
            latched.framestack.frames[frame_index].ref_values[reg_index]
                .expect("location came from Some")
        })
        .collect();
    let image_exception_root = (latched.last_exc_value != 0).then(|| {
        let index = image_ref_roots.len();
        image_ref_roots.push(latched.last_exc_value);
        index
    });
    let undo_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
    unsafe {
        majit_gc::shadow_stack::push_resume_ref_roots(image_ref_roots.as_mut_slice());
        majit_gc::shadow_stack::push_resume_ref_roots(locals_undo.as_mut_slice());
        if let Some(stack) = root_stack.as_mut() {
            majit_gc::shadow_stack::push_resume_ref_roots(stack.roots_mut());
        }
    }
    if !crate::state::write_back_outer_locals(ctx, root_addr) {
        crate::state::restore_frame_locals(root_addr, &locals_undo);
        majit_gc::shadow_stack::pop_resume_ref_roots_to(undo_depth);
        restore_links(&saved_links);
        mfdbg!("frame 0: {root_addr:#x} locals publish declined");
        return false;
    }
    for (&(frame_index, reg_index), &forwarded) in image_ref_locations.iter().zip(&image_ref_roots)
    {
        latched.framestack.frames[frame_index].ref_values[reg_index] = Some(forwarded);
    }
    if let Some(index) = image_exception_root {
        latched.last_exc_value = image_ref_roots[index];
    }
    if let Some(stack) = root_stack.as_ref()
        && !crate::state::publish_captured_frame_stack(root_addr, stack)
    {
        crate::state::restore_frame_locals(root_addr, &locals_undo);
        majit_gc::shadow_stack::pop_resume_ref_roots_to(undo_depth);
        restore_links(&saved_links);
        mfdbg!("frame 0: active stack publish declined");
        return false;
    }
    // As in the single-frame path, locals publication is the final recoverable
    // gate. Once the roots are released and the chain starts running, adoption
    // is irrevocable.
    majit_gc::shadow_stack::pop_resume_ref_roots_to(undo_depth);
    drop(locals_undo);
    drop(image_ref_roots);
    drop(root_stack);
    let ec = unsafe {
        (*(cf_addr as *mut pyre_interpreter::PyFrame)).execution_context
            as *mut pyre_interpreter::PyExecutionContext
    };
    // Rooted for the whole drive: frames themselves never move, but once the
    // tracer stores a `JitVirtualRef` in the chain the displaced value is a
    // nursery object, and a collection inside the drive would leave the
    // restore below writing back a pre-move pointer.
    let saved_root =
        majit_gc::shadow_stack::push(majit_ir::GcRef(unsafe { (*ec).topframeref } as usize));
    // `enter`: publish `ec.topframeref = <this level's frame>` before it runs.
    let set_topframeref = |frame_ptr: i64| unsafe {
        (*ec).topframeref = frame_ptr as *mut pyre_interpreter::PyFrame;
    };
    // EXPERIMENT: multi-frame runs full outer-frame bodies, so it needs a
    // full-coverage dispatch table, not the inline-call-only builder.
    let (mut mf_builder, _unwired) =
        crate::jitcode_runtime::build_default_bh_builder_with_unwired_report();
    mf_builder.cpu = Some(majit_metainterp::blackhole::pyre_production_cpu());
    let majit_metainterp::MultiFrameBlackholeResult {
        outcome,
        terminal: mf_terminal,
    } = majit_metainterp::drive_multi_frame_blackhole(
        &mut mf_builder,
        &mut latched.framestack,
        majit_metainterp::blackhole::StateFieldLayout::default(),
        virtualizable_info,
        // `per_frame` is `Some` just below, and `convert_and_run_from_pyjitpl`
        // then overrides every level's `virtualizable_ptr` /
        // `virtualizable_stack_base` from it (`blackhole.rs`), so this pair
        // never reaches a blackhole level.  Frame 0's vable writes land on
        // `per_frame[0]` — the live frame — which is also what `on_enter_level`
        // publishes as `ec.topframeref`.
        cf_addr as i64,
        stack_base,
        ctx.metainterp_sd().as_ref(),
        latched.last_exc_value,
        latched.raising_exception,
        Some(per_frame.as_slice()),
        Some(&set_topframeref as &dyn Fn(i64)),
    );
    // `leave`: restore `ec.topframeref` to the portal after the inline chain,
    // reading the root back so an in-place forward during the drive is kept.
    let saved_topframeref =
        majit_gc::shadow_stack::get(saved_root).0 as *mut pyre_interpreter::PyFrame;
    majit_gc::shadow_stack::pop_to(saved_root);
    unsafe {
        (*ec).topframeref = saved_topframeref;
    }
    // Frame 0's blackhole level ran against `per_frame[0]`, the LIVE frame,
    // because `convert_and_run_from_pyjitpl` overrides each level's
    // `virtualizable_ptr` from `per_frame` — so its `setfield_vable` stores
    // landed there, not in the snapshot.  EVERY adopted arm below sets
    // `WALK_END_FLUSH_COMMITTED`, and the portal epilogue then copies the
    // SNAPSHOT's whole locals array onto the live frame
    // (`restore_resume_state_from`), which would revert every one of those
    // stores.  Fold them into the snapshot before the arms so it is once again
    // the committed image, which is the contract the epilogue and the
    // single-frame arm both assume.  This has to sit outside the match: a
    // `DoneWithThisFrame*` or `ExitFrameWithExceptionRef` terminal writes no
    // resume state of its own, and for the exception terminal the traceback
    // keeps the root frame reachable, so a stale copy is observable through
    // `tb_frame.f_locals` long after the walk.
    if root_addr != cf_addr {
        unsafe {
            (*(cf_addr as *mut pyre_interpreter::PyFrame))
                .restore_resume_state_from(&*(root_addr as *const pyre_interpreter::PyFrame));
        }
    }
    match outcome {
        majit_metainterp::jitexc::JitException::ContinueRunningNormally {
            ref green_int, ..
        } => {
            let &resume_py_pc = green_int.first().expect(
                "post-blackhole ContinueRunningNormally has no resume pc; \
                 replay is forbidden after blackhole effects",
            );
            let resume_py_pc = resume_py_pc as usize;
            // `resume_py_pc` is deliberately NOT checked.  It is a
            // `depth_at_py_pc` index written through to
            // `frame.last_instr = resume_py_pc - 1`, so 0 is an ordinary
            // coordinate — the single-frame arm hands it to the CRN handoff
            // unfiltered.  Rejecting it here would also decline *after* the
            // chain has been driven, returning to a legacy replay that
            // re-executes what the chain already committed.
            //
            // `cf_addr` needs no check either: the pre-drive
            // `concrete_nlocals(cf_addr)` above already returns `None` for a
            // zero frame (`state.rs` `(frame != 0).then_some(..)?`).  The
            // decline that used to sit here was unreachable AND post-drive,
            // i.e. it could only ever have been an instance of the hazard the
            // comment above names.
            debug_assert_ne!(
                cf_addr, 0,
                "pre-drive concrete_nlocals admitted a zero frame"
            );
            // Same handoff as the single-frame arm, for the same reason: the
            // chain's levels wrote their frames through the blackhole's
            // write-through vable ops, and a CRN is raised only at the
            // bottommost level, which `convert_and_run_from_pyjitpl` makes the
            // portal frame.  So the frame already holds the driven result and
            // nothing has to be rebuilt from the terminal register banks.
            //
            // The fold above the match has already made the snapshot the
            // committed image, so the mirror inside is a no-op here and only
            // the resume coordinate is left to set.  Taking the same path
            // anyway keeps one definition of what adopting a CRN means.
            //
            // This retires three post-drive declines at once — the absent
            // terminal image, the out-of-range terminal jitcode index, and the
            // register-image rebuild's own rejection — each of which handed a
            // chain that had already run back to a replay.
            adopt_blackhole_crn(cf_addr, root_addr, resume_py_pc);
            WALK_END_RESTART_PC.with(|slot| slot.set(Some(resume_py_pc)));
        }
        majit_metainterp::jitexc::JitException::DoneWithThisFrameVoid => {
            crate::jitcode_dispatch::fbw_finish_concrete_set(crate::state::ConcreteValue::Null);
        }
        majit_metainterp::jitexc::JitException::DoneWithThisFrameInt(value) => {
            crate::jitcode_dispatch::fbw_finish_concrete_set(crate::state::ConcreteValue::Int(
                value,
            ));
        }
        majit_metainterp::jitexc::JitException::DoneWithThisFrameRef(value) => {
            crate::jitcode_dispatch::fbw_finish_concrete_set(crate::state::ConcreteValue::Ref(
                value.as_usize() as pyre_object::PyObjectRef,
            ));
        }
        majit_metainterp::jitexc::JitException::DoneWithThisFrameFloat(value) => {
            crate::jitcode_dispatch::fbw_finish_concrete_set(crate::state::ConcreteValue::Float(
                value,
            ));
        }
        majit_metainterp::jitexc::JitException::ExitFrameWithExceptionRef(value) => {
            // The terminal image is the BOTTOMMOST level, which
            // `convert_and_run_from_pyjitpl` made the portal frame — i.e. the
            // one `cf_addr` / `root_addr` represent, the frame the raise is
            // about to leave.  Same publication as the single-frame arm.
            if let Some(image) = mf_terminal.as_ref()
                && let Ok(jitcode_index) = i32::try_from(image.jitcode_index)
            {
                publish_terminal_raise_coordinate(
                    jitcode_index,
                    image.last_opcode_position,
                    &[cf_addr, root_addr],
                );
            }
            crate::jitcode_dispatch::fbw_finish_raise_set(crate::state::ConcreteValue::Ref(
                value.as_usize() as pyre_object::PyObjectRef,
            ));
        }
    }
    let _ = crate::jitcode_dispatch::take_committed_frame_escape_pc();
    crate::jitcode_dispatch::discard_escape_flush_undo();
    crate::jitcode_dispatch::fbw_foriter_inflight_clear();
    // Same as the single-frame adoption: a frame terminal, not a resume pc.
    let _ = commit_walk_end(commit_leg, WalkEndResume::Terminal);
    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
        eprintln!("[fbw-blackhole] adopted multi-frame terminal depth={depth}");
    }
    true
}

/// Publish the Python coordinate of the op a blackhole frame terminal raised at
/// into every representation of that frame.
///
/// The interpreter takes the raise back from here and looks the frame's
/// exception table up at `last_instr` (`handle_exception`) — pyre's Python-level
/// handler search is interpreter machinery, so the blackhole's own
/// `handle_exception_in_frame` scan (jitcode `catch_exception`, i.e. a residual
/// helper's RPython `try`) never saw it.  The frame's coordinate is still the
/// walk's at this point; without the raising op's pc the lookup misses and a
/// `try`/`finally` around the escaped region does not run.  This is the same
/// coordinate the terminal's traceback node carries.
fn publish_terminal_raise_coordinate(
    jitcode_index: i32,
    last_opcode_position: usize,
    frames: &[usize],
) {
    let Ok(position) = i32::try_from(last_opcode_position) else {
        return;
    };
    let Some(py_pc) =
        crate::py_coord::containing_py_pc_for_jitcode_pc_public(jitcode_index, position)
    else {
        return;
    };
    for &addr in frames {
        if addr != 0 {
            // SAFETY: every address here is a live `PyFrame` the adoption just
            // published its register image into.
            unsafe {
                (*(addr as *mut pyre_interpreter::PyFrame)).last_instr = py_pc as isize;
            }
        }
    }
}

/// Kill switch for the `WalkAbort` leg: `PYRE_WALKABORT_OFF=1` puts every
/// capability-gap abort back on the legacy entry replay.
///
/// Kept because this leg's blast radius is every non-carrier walk abort and it
/// commits irrevocably once the blackhole runs, so an operator needs a way to
/// take it out without a rebuild.  It is also the cheap A/B for a class of bug
/// this leg sits in the middle of: one binary, one env var, no layout variable
/// — the same shape as the `PYRE_ANCHOR_STRICT` probe.
fn walk_abort_leg_enabled() -> bool {
    std::env::var_os("PYRE_WALKABORT_OFF").is_none()
}

fn try_adopt_blackhole(
    ctx: &mut TraceCtx,
    cf_addr: usize,
    live_root_addr: usize,
    commit_leg: WalkEndCommitLeg,
) -> bool {
    try_adopt_multi_frame_blackhole(ctx, cf_addr, live_root_addr, commit_leg)
        || try_adopt_single_frame_blackhole(ctx, cf_addr, live_root_addr, commit_leg)
}

fn blackhole_terminal_error(error: &crate::jitcode_dispatch::DispatchError) -> bool {
    matches!(
        error,
        crate::jitcode_dispatch::DispatchError::VableEscapedDuringResidualCall { .. }
            | crate::jitcode_dispatch::DispatchError::TraceTooLong { .. }
    ) || error.leaves_complete_image()
}

fn run_perfn_walk<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    sym: &mut Sym,
    w_code: *const (),
    start_pc: usize,
    cf_addr: usize,
    authoritative: bool,
) -> Option<(usize, usize, PerfnWalkResult)> {
    let session = std::cell::RefCell::new(crate::jitcode_dispatch::WalkSession::default());
    let Some(pjc) = crate::state::pyjitcode_for_code(w_code) else {
        eprintln!("[walk-perfn] no per-CodeObject PyJitCode for code={w_code:?}");
        return None;
    };
    // The green stays in Python-bytecode coordinates for merge-point matching;
    // the codewrite-time trace-entry sidecar carries its JitCode coordinate for
    // plain-portal function entries and loop headers. A bridge starts at its
    // guard resume py_pc, outside that sidecar by construction.
    let is_plain_portal = !ctx.is_bridge_trace;
    let is_loop_header =
        !pjc.code_ptr.is_null() && start_pc_is_loop_header(unsafe { &*pjc.code_ptr }, start_pc);
    let is_entry_green = start_pc == 0 || is_loop_header;
    let uses_entry_sidecar = is_plain_portal && is_entry_green;
    let sidecar_entry = pjc.merge_entry_for(start_pc);
    let entry_coord = if sym.bridge_walk_entry_pc().is_some() {
        // Guard resume with a carried jitcode coordinate: the walk enters at
        // the carried offset (override below).
        sym.bridge_walk_entry_pc()
    } else if uses_entry_sidecar {
        // The resume-marker table forward-carries a py_pc that emitted no op of
        // its own to the next py that did.  When lowering stopped early — an
        // unported opcode ends the body with `abort_permanent` — every later py
        // is unlowered, so that carry crosses the truncation and hands back the
        // abort block.  The block's own coordinate is the unported instruction,
        // far BEFORE `start_pc`: walking from it reaches the marker at once and
        // `fbw_abort_resume_py_pc` back-translates to that earlier py, rewinding
        // the live frame so every instruction between it and `start_pc` runs a
        // second time.  Require the coordinate to round-trip forward instead —
        // a trace-entry green heads its own block, so its marker back-translates
        // to itself, and a genuine trivia carry lands on a LATER py.  Anything
        // else is a body that does not encode `start_pc`, which is exactly the
        // decline below.
        sidecar_entry.filter(|&off| {
            crate::py_coord::containing_py_pc_for_jitcode_pc(&pjc.metadata, off) as usize
                >= start_pc
        })
    } else {
        // Every non-entry resume carries its own JitCode coordinate. Without
        // one the site-specific decline below rejects the walk.
        None
    };
    let Some(entry_coord) = entry_coord else {
        // This already-built body does not encode
        // `start_pc` as a resume coordinate, so the same body walked from
        // the same entry recurs identically on every retrace.  Decline the
        // key so retraces interpret without JIT instead of re-walking and
        // re-aborting each iteration; mirrors the `built_as_portal=false`
        // structural decline below.
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[walk-perfn] no jitcode entry for start_pc={start_pc} (n_py_instrs={}); declining walk",
                pjc.metadata.n_py_instrs as usize
            );
        }
        fbw_bridge_decline(ctx);
        fbw_decline(crate::driver::make_green_key(w_code, start_pc));
        return None;
    };
    // A kept-stack branch-guard bridge resumes at the guard's OWN mid-opcode
    // jitcode offset (`setup_bridge_sym` resolved it into
    // `sym.bridge_walk_entry_pc`, the same coordinate the blackhole
    // `setposition`s to) — NOT the opcode-entry resume marker for `start_pc`.
    // Resuming at the entry marker re-executes the whole opcode from the top,
    // reading abstract-register colors that were live at entry but dead
    // (recolored / already consumed) at the guard, which the guard's resume
    // data never preserved. See the field doc on `PyreSym::bridge_walk_entry_pc`.
    let entry = sym.bridge_walk_entry_pc().unwrap_or(entry_coord);
    if let Some(entry_depth) = pjc.depth_for_jitcode_pc_pred(entry) {
        let stack_base = crate::state::concrete_nlocals(cf_addr).unwrap_or(sym.nlocals());
        let live_stack = sym.valuestackdepth().saturating_sub(stack_base);
        // A mismatch is unsound only when the carried coordinate IS the
        // resume marker.  That is the marker-inside-super-instruction shape:
        // the live frame has advanced through the super-instruction while the
        // predecessor depth twin still describes the marker's pre-op stack.
        // An interior branch resume resolves *through* a marker but carries
        // its own jitcode offset, so it may legitimately have a different
        // live depth and must continue walking.
        let entry_is_resume_marker = pjc.resume_marker_for_jitcode_pc(entry) == Some(entry);
        if live_stack != entry_depth as usize && entry_is_resume_marker {
            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                let marker_py =
                    crate::py_coord::containing_py_pc_for_jitcode_pc(&pjc.metadata, entry);
                eprintln!(
                    "[fbw-abort] start_pc={start_pc} entry={entry} live_stack={live_stack} \\
                     entry_depth={entry_depth} marker_py={marker_py}; declining marker-entry walk"
                );
            }
            fbw_bridge_decline(ctx);
            fbw_decline(crate::driver::make_green_key(w_code, start_pc));
            return None;
        }
    }
    // The full-body walk drives a PORTAL trace, so the body must carry the
    // portal entry INPUT SHAPE (`FrameInputs::Portal`: `[frame, ec]` red inputs
    // + the frame-vable locals prologue). Every drained per-code jitcode is
    // Portal-shaped (`built_as_portal` records the
    // input shape, independent of true-portal-ness), so this decline narrows to
    // the only remaining shapeless case: a skeleton jitcode with no portal
    // input shape (pyjitcode.rs `skeleton`).
    if !pjc.metadata.built_as_portal {
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-abort] start_pc={start_pc} jitcode has no portal input shape \
                 (built_as_portal=false); declining walk"
            );
        }
        fbw_bridge_decline(ctx);
        fbw_decline(crate::driver::make_green_key(w_code, start_pc));
        return None;
    }

    let is_bridge_trace = ctx.is_bridge_trace;

    // setup_call argbox: seed r0 = the standard virtualizable identity box
    // (`virtualizable_boxes[-1]`, the `InputArgRef(SYM_FRAME_IDX)` that
    // `init_symbolic` seeded) — the SAME OpRef the retired per-opcode arm
    // entry seeded as r0 (= `sym.frame`, and
    // `sym.frame == OpRef::input_arg_typed(SYM_FRAME_IDX, Ref)`).  A fresh
    // `const_ref(cf_addr)` would be a DIFFERENT OpRef than the identity box,
    // so `concrete_of_opref`'s standard-vable resolution (trace_ctx.rs:1842,
    // keyed on `== standard_virtualizable_box()`) would miss and every vable
    // read would fall through to the nonstandard GETFIELD_GC leg.  Falls back
    // to `const_ref` only when no virtualizable is bound.
    //
    // NOTE: seeding r0 is NECESSARY but not sufficient for
    // the mid-loop resume entry (pc=107, after the loop-header
    // `jit_merge_point` @ pc=94).  The loop body reads its vable from a
    // post-merge LOOP-INPUT register (the merge-point reds), NOT from r0; that
    // register is left `OpRef::NONE` because the probe enters past the merge
    // point and never binds the reds.  `concrete_of_opref(NONE)` returns the
    // `GcRef(usize::MAX)` sentinel → `is_nonstandard_virtualizable` takes the
    // nonstandard leg → `getarrayitem_vable` returns `Value::Void` even though
    // the virtualizable SHADOW entry is correct.  Closing that needs the live
    // loop-input registers seeded at walk entry, not just r0.
    let frame_box = ctx
        .standard_virtualizable_box()
        .unwrap_or_else(|| ctx.const_ref(cf_addr as i64));
    // Seed the loop's live INPUT registers so the
    // post-merge-point loop body resolves its loop-invariant reads.  The
    // walk enters PAST the loop-header `jit_merge_point`, which would
    // otherwise leave those colors `OpRef::NONE` (→ sentinel concrete →
    // nonstandard-virtualizable Void leg on the first `getarrayitem_vable`).
    // Decode the merge point's green-ref (`gr` = [pycode]) and red (`rr` =
    // [frame, ec], portal jitdriver convention) register lists and seed each
    // named color.  Int greens (`gi` = next_instr, is_being_profiled) live
    // in the int CONSTANT region and are seeded by `copy_constants` inside
    // `dispatch_via_miframe`, so they need no entry seed.  `frame` is the
    // standard-vable identity box (so the body's vable reads hit the
    // standard fast path); `pycode`/`ec` are const-refs to the live
    // pointers.  `argboxes_r[i] -> top_regs_r[i]` is the seed channel.
    let ec_box = ctx.const_ref(sym.concrete_execution_context() as i64);
    let pycode_box = ctx.const_ref(w_code as i64);
    let static_entry_green_ref_regs = if is_bridge_trace {
        None
    } else {
        static_entry_merge_point_green_ref_regs(pjc.jitcode.code.as_slice(), entry)
    };
    let portal_frame_reg = pjc.metadata.portal_frame_reg;
    let portal_ec_reg = pjc.metadata.portal_ec_reg;
    let argboxes_r: Vec<majit_ir::OpRef> = {
        let mut v = vec![majit_ir::OpRef::NONE; 1];
        let mut seed = |reg: u8, val: majit_ir::OpRef| {
            let reg = reg as usize;
            if reg >= v.len() {
                v.resize(reg + 1, majit_ir::OpRef::NONE);
            }
            v[reg] = val;
        };
        // Colors holding the red virtualizable identity (frame) + ec — the
        // standard virtualizable. The #124 operand-stack override below must
        // not overwrite these (a kept temp never lives in a red-input color).
        let mut reserved_red_colors: Vec<u8> = Vec::new();
        if !is_bridge_trace {
            // C1 marker entry starts at the static sidecar coordinate BEFORE
            // the source marker.  Seed only the root JitCode's formal red
            // inputs at their per-JitCode metadata colors. The marker's Int
            // greens remain in the constant region; its Ref green is seeded
            // below from this frame's concrete PyJitCode identity.
            let frame_color = if portal_frame_reg != u16::MAX {
                portal_frame_reg as u8
            } else {
                1
            };
            let ec_color = if portal_ec_reg != u16::MAX {
                portal_ec_reg as u8
            } else {
                2
            };
            seed(frame_color, frame_box);
            seed(ec_color, ec_box);
            // `_compile_and_run_once` selected this trace with concrete
            // `[next_instr, is_being_profiled, pycode]` greens.  Thread the
            // Ref green into the marker's allocated color just as RPython's
            // root MIFrame receives its green arguments.  This is per-frame:
            // `pycode_box` belongs to this PyJitCode/frame, never a shared
            // portal anchor.  The pre-marker instructions may overwrite it
            // with the same value; seeding is required when their vable load
            // remains symbolic (the wasm regression caught that shape).
            if let Some(gr) = static_entry_green_ref_regs.as_ref() {
                if let Some(&pycode_color) = gr.first() {
                    seed(pycode_color, pycode_box);
                }
            }
            reserved_red_colors.push(frame_color);
            reserved_red_colors.push(ec_color);
        } else {
            match bridge_resume_merge_point_regs(pjc.jitcode.code.as_slice(), entry) {
                Some((gr, rr)) => {
                    if let Some(&r) = gr.first() {
                        seed(r, pycode_box);
                    }
                    if let Some(&r) = rr.first() {
                        seed(r, frame_box);
                        reserved_red_colors.push(r);
                    }
                    if let Some(&r) = rr.get(1) {
                        seed(r, ec_box);
                        reserved_red_colors.push(r);
                    }
                }
                // Straight-line entry, no governing loop header (e.g. a
                // non-looping function like `fib` or a leaf method): seed the
                // portal red args `[frame, ec]` at the AUTHORITATIVE
                // post-regalloc colors the codewriter recorded
                // (`metadata.portal_frame_reg` / `portal_ec_reg`), the same
                // colors the loop-header `jit_merge_point` `rr` list carries.
                // The earlier positional `[pycode=r0, frame=r1, ec=r2]`
                // convention only coincided with regalloc for an nlocals==1
                // function (fib: frame=r1); a 2-local leaf method (`value()`)
                // places frame at r2 / ec at r3, so the positional seed put
                // `ec_box` in the frame color and every `getfield/getarrayitem
                // _vable` of a local took the nonstandard-virtualizable leg
                // (internal promote `GuardValue` + force store-back against a
                // frame that is not the standard virtualizable).
                // pycode (the jitdriver's green ref) is read from the frame's
                // `pycode` field via `getfield_vable`, so it needs no register
                // seed once `frame` resolves to the standard virtualizable; the
                // r0 seed is retained as a defensive best-effort (overwritten by
                // the entry prologue's first dst in practice).
                //
                None => {
                    seed(0, pycode_box);
                    let frame_color = if portal_frame_reg != u16::MAX {
                        portal_frame_reg as u8
                    } else {
                        1
                    };
                    let ec_color = if portal_ec_reg != u16::MAX {
                        portal_ec_reg as u8
                    } else {
                        2
                    };
                    seed(frame_color, frame_box);
                    seed(ec_color, ec_box);
                    reserved_red_colors.push(frame_color);
                    reserved_red_colors.push(ec_color);
                }
            }
        }
        // A loop trace starts at the per-CodeObject sidecar coordinate near
        // the loop header, not at the function entry that originally defined
        // every live MIFrame register.  RPython reaches that header with the
        // same MIFrame and therefore retains all live register contents.  The
        // sidecar walk creates a fresh register bank, so reconstruct each
        // per-PC Ref color from its authoritative slot in the red frame's
        // virtualizable array before dispatching the header.  This is the
        // non-bridge counterpart of `setup_bridge_sym`'s color/slot inversion;
        // without it loop-carried collection operands remain `OpRef::NONE`.
        if !is_bridge_trace {
            let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
            if let Some(pcdep) =
                crate::state::pcdep_trivia_at(pjc.jitcode.index() as i32, entry as i32)
            {
                let mut live_slot_seeds = Vec::new();
                seed_loop_entry_ref_slots(
                    &pcdep,
                    nvs,
                    |index| ctx.virtualizable_box_at(index),
                    |color, slot, opref| live_slot_seeds.push((color, slot, opref)),
                );
                for (color, slot, opref) in live_slot_seeds {
                    seed(color, opref);
                    // RPython's MIFrame register is a Box carrying both its
                    // symbolic identity and recording-time value.  The
                    // sidecar reconstruction above restores the first half;
                    // stamp the live PyFrame slot onto the same OpRef so an
                    // inlined callee receives its concrete argument identity.
                    // Without this, a dynamic call such as pickle's
                    // dispatch[key](self) loses `self` at the loop header;
                    // a raising callee then mutates concrete state but cannot
                    // recover the exception class, aborting and replaying the
                    // already-applied mutation.
                    if cf_addr != 0 {
                        let frame =
                            unsafe { &*(cf_addr as *const pyre_interpreter::pyframe::PyFrame) };
                        if let Some(&value) = frame.locals_w().as_slice().get(slot as usize)
                            && !value.is_null()
                        {
                            ctx.set_opref_concrete(
                                opref,
                                majit_ir::Value::Ref(majit_ir::GcRef(value as usize)),
                            );
                        }
                    }
                }
            }
        }
        // The pcdep inversion above is the complete loop-entry seed.  FOR_ITER
        // also emits an in-loop `getarrayitem_vable_r`, so its iterator is
        // refreshed from the same frame owner on every iteration.
        //
        // #124: a bridge enters mid-body, where the loop-header merge-point
        // colors seeded above (the loop's green pycode / red frame+ec) are
        // reused for live operand-stack temps — the kept conditional-
        // expression / short-circuit / chained-compare value.  Leaving e.g.
        // the pycode green at the kept temp's color feeds a stale code object
        // into its binary op (`unsupported operand type(s) for +: 'code' and
        // 'int'`).  Seed the guard's live abstract-register colors from the
        // resume-data OpRefs setup_bridge_sym resolved.  Locals (read through
        // the vable) and frame/ec (at their own colors) keep the seeding above.
        //
        // A kept operand-stack temp never occupies a red-input color, so skip
        // `reserved_red_colors`: seeding a temp over the frame color overwrites
        // the standard virtualizable identity and forces every later `vable_*`
        // op onto the nonstandard leg, promoting and storing back against the
        // wrong frame.
        if is_bridge_trace {
            if sym.bridge_walk_entry_pc().is_some() {
                // Kept-stack branch guard resumed at the guard's own jitcode
                // offset (`entry` above).  The live registers there are the
                // guard-time abstract-register colors the resume data decoded
                // into `bridge_registers_r` (color-indexed, `consume_boxes`
                // parity, resume.py:1055) — the SAME bank the blackhole's
                // `init_register_files` + resume fill would hold.  Seed each
                // non-NONE color directly; the `nlocals + depth` slot→color
                // shortcut below is wrong here because a kept temp's abstract
                // color is not `nlocals + depth` under free register coloring.
                if let Some(bridge_regs_r) = sym.bridge_registers_r() {
                    for (color, &opref) in bridge_regs_r.iter().enumerate() {
                        if opref.is_none() {
                            continue;
                        }
                        let color = color as u8;
                        if reserved_red_colors.contains(&color) {
                            // `bridge_registers_r` is the authoritative guard-time
                            // register coloring.  Free register allocation reuses
                            // the portal EC color for a live operand at PCs where
                            // the trace has no live EC read (the same collision the
                            // guard-failure resume handles at
                            // jitcode_dispatch.rs:6994 via `semantic_idx.is_none()`
                            // — otherwise `fib(n-1) + fib(n-2)` resumes the left
                            // operand as the EC and SIGSEGVs).  When the bridge
                            // names a genuine operand here (an opref other than the
                            // pre-seeded `ec_box`/`frame_box`), skipping strands the
                            // stale `ConstPtr(ec)` in the color the resumed body
                            // reads as its operand.  Seed the real operand.
                            //
                            // The FRAME color (`reserved_red_colors[0]`) keeps its
                            // skip unconditionally: its `frame_box` is the standard
                            // virtualizable identity and overwriting it forces every
                            // later `vable_*` op onto the nonstandard leg (#124),
                            // promoting the wrong frame.  The EC
                            // color carries no such identity — the EC stays
                            // recoverable from the frame — so reseeding it is safe.
                            // This applies regardless of tagged-int state: a leaf
                            // callee whose LOAD_DEREF result is colored onto the EC
                            // register (`return CELL + 1`) strands `ConstPtr(ec)` in
                            // the add's LHS when the cell payload's class flips and a
                            // guard-failure bridge resumes here — the residual add
                            // then dereferences the stale pointer and SIGSEGVs.
                            let is_frame_color =
                                reserved_red_colors.first().copied() == Some(color);
                            let bridge_names_operand =
                                !is_frame_color && opref != ec_box && opref != frame_box;
                            if bridge_names_operand {
                                seed(color, opref);
                            }
                            continue;
                        }
                        seed(color, opref);
                    }
                }
            } else if let Some(bridge_stack) = sym.bridge_stack_oprefs() {
                // Non-branch-guard resume at the opcode-entry
                // marker: the walk re-executes the opcode from the top, reading
                // its operand-stack inputs POSITIONALLY — `registers_r[nlocals +
                // stack_idx]` (trace_opcode.rs `stack_slot_reg_idx`) — so
                // the slot-indexed `bridge_stack` tail seeds color `nlocals + i`.
                //
                // The reserved-red skip is per-PC-wrong here.  `portal_frame_reg`
                // / `portal_ec_reg` are a SINGLE global pair naming where the
                // reds live at PORTAL ENTRY, not at an arbitrary interior resume
                // PC; under free register coloring the reds sit at the
                // operand-stack base `[nlocals, nlocals + 1]`, exactly the colors
                // a shallow live operand stack occupies.  A live (non-NONE)
                // `bridge_stack[i]` is the authoritative per-PC witness that slot
                // `nlocals + i` holds a real operand-stack value at THIS PC — and
                // a single color cannot simultaneously hold that value and the
                // red, so the red is provably not live at color `nlocals + i`
                // here (its identity is recovered through the frame field, e.g.
                // `ensure_execution_context` / the standard-vable box, not this
                // register).  Seeding the temp is therefore correct and the skip
                // is what dropped it: on `re/_parser.py:append` (nlocals=2) the
                // live callable at slot 2 = color 2 = `portal_frame_reg` was
                // skipped, so `residual_call` read the stale entry frame/ec seed
                // as its callable → SIGSEGV in dispatch_callable /
                // `exception_is_valid_obj_as_class_w` (#389 with-gate probe).
                //
                // A NONE `bridge_stack[i]` (dead/empty slot) leaves the color's
                // red seed intact — the red genuinely still owns the color there.
                let nl = sym.nlocals();
                for (i, &opref) in bridge_stack.iter().enumerate() {
                    if !opref.is_none() {
                        let color = (nl + i) as u8;
                        seed(color, opref);
                    }
                }
                // The value just returned by an in-flight callee is the one
                // non-pcdep operand at this opcode-entry coordinate.  RPython
                // has already installed it in the caller MIFrame's result
                // register via `make_result_of_lastop`; mirror that exact
                // color from the resumed bank.  Slot-position seeding above
                // cannot recover it under free register coloring (for
                // Fraction.__rpow__, semantic slot 3 is color r9).
                let pending_result_color =
                    residual_ref_call_before(pjc.jitcode.code.as_slice(), entry)
                        .map(|(_, dst)| dst)
                        .or_else(|| {
                            pjc.result_color_trivia_for_jitcode_pc(entry)
                                .map(usize::from)
                                .filter(|&color| color != usize::from(u16::MAX))
                        });
                if let Some((color, opref)) = pending_result_color.and_then(|color| {
                    sym.bridge_registers_r()
                        .and_then(|regs| regs.get(color).copied())
                        .filter(|opref| !opref.is_none())
                        .map(|opref| (color, opref))
                }) {
                    seed(color as u8, opref);
                }
            }
        }
        v
    };

    // Int-bank seed for a kept-stack branch-guard bridge: the guard reads its
    // condition from an Int register (the `b < 9` compare result) that ran
    // BEFORE the guard, so resuming at the guard offset requires it from the
    // resume data. `setup_bridge_sym` decoded the Int bank color-indexed into
    // `sym.registers_i` (concrete already stamped there); pass it positionally
    // so `dispatch_via_miframe` writes `top_regs_i[color] = value`. Empty for a
    // non-branch-guard resume (`bridge_walk_entry_pc == None`), where the walk
    // enters at the opcode boundary with no live mid-opcode Int temps.
    let argboxes_i: Vec<majit_ir::OpRef> = if sym.bridge_walk_entry_pc().is_some() {
        // Clamp to the jitcode's Int register count: `sym.registers_i` may carry
        // trailing scratch/constant colors beyond `num_regs_i`, and
        // `dispatch_via_miframe` rejects an argbox list longer than the callee
        // bank (`InlineCallIntArityMismatch`). Only the leading `num_regs_i`
        // colors are real Int registers the walk reads.
        let num_regs_i = pjc.jitcode.num_regs_i() as usize;
        let mut v = sym.registers_i().to_vec();
        v.truncate(num_regs_i);
        v
    } else {
        Vec::new()
    };

    // resume.py:1036-1038 `_callback_f` parity: seed the Float bank the same
    // way as argboxes_i. Empty for a non-marker resume; `truncate(num_regs_f)`
    // strips the copy_constants tail exactly as the Int build does.
    let argboxes_f: Vec<majit_ir::OpRef> = if sym.bridge_walk_entry_pc().is_some() {
        let num_regs_f = pjc.jitcode.num_regs_f() as usize;
        let mut v = sym.registers_f().to_vec();
        v.truncate(num_regs_f);
        v
    } else {
        Vec::new()
    };

    // resume.py:1049-1056 constructs a frame and consumes its register stream
    // against that frame's JitCode body. Do not interpret a carried offset in
    // another installed body for the same code object: its register colors are
    // a foreign coordinate space. This is a runtime decline, not a green-key
    // disable, because a later bridge may carry a matching body identity.
    if is_bridge_trace
        && sym.bridge_walk_entry_pc().is_some()
        && pjc.jitcode.index() as i32 != sym.bridge_walk_entry_jitcode_index()
    {
        crate::jitcode_dispatch::census_record("Fbw::BridgeEntryForeignJitcode");
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-bridge-decline] foreign resume-marker jitcode: body_index={} \\
                 bridge_jitcode_index={} entry={}",
                pjc.jitcode.index(),
                sym.bridge_walk_entry_jitcode_index(),
                entry,
            );
        }
        fbw_bridge_decline(ctx);
        return None;
    }

    // The foreign-JitCode check above is a first-class runtime decline. Below,
    // resume.py:1017-1057 `consume_boxes` / `enumerate_vars` fills every live
    // Ref, Int, and Float register from the marker's stream. With the total
    // bridge seed above, an uncovered color is a codewriter/seed defect.
    // `_callback_r` writes `next_ref()` directly (resume.py:1032-1034), so a
    // restored null Ref is covered; only `OpRef::NONE` is absent. `Some` is
    // the resume-marker discriminator: setup_bridge_sym sets it only when the
    // carried frame pc is a decodable `live/` offset.
    if is_bridge_trace && sym.bridge_walk_entry_pc().is_some() {
        let live = crate::state::frame_liveness_reg_indices_by_bank_from_pc(
            pjc.jitcode.index() as i32,
            entry as i32,
        );
        let uncovered = live
            .ref_
            .iter()
            .copied()
            .find(|&color| {
                argboxes_r
                    .get(color as usize)
                    .is_none_or(|opref| opref.is_none())
            })
            .map(|color| ("Ref", color))
            .or_else(|| {
                live.int
                    .iter()
                    .copied()
                    .find(|&color| {
                        argboxes_i
                            .get(color as usize)
                            .is_none_or(|opref| opref.is_none())
                    })
                    .map(|color| ("Int", color))
            })
            .or_else(|| {
                live.float
                    .iter()
                    .copied()
                    .find(|&color| {
                        argboxes_f
                            .get(color as usize)
                            .is_none_or(|opref| opref.is_none())
                    })
                    .map(|color| ("Float", color))
            });
        debug_assert!(
            uncovered.is_none(),
            "consume_boxes totality violated: jitcode_index={} entry={} uncovered={uncovered:?}",
            pjc.jitcode.index(),
            entry,
        );
        if let Some((bank, color)) = uncovered {
            // read_int_reg/read_float_reg (jitcode_dispatch/mod.rs:1907-1936)
            // only bounds-check. An uncovered color would record OpRef::NONE
            // into an operation and can become a SIGSEGV or miscompile; retain
            // this cold decline as the release airbag.
            let census_key = match bank {
                "Ref" => "Fbw::BridgeEntryLiveRegUnseeded::Ref",
                "Int" => "Fbw::BridgeEntryLiveRegUnseeded::Int",
                "Float" => "Fbw::BridgeEntryLiveRegUnseeded::Float",
                _ => unreachable!("unknown bridge register bank"),
            };
            crate::jitcode_dispatch::census_record(census_key);
            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                eprintln!(
                    "[fbw-bridge-decline] uncovered resume-marker live register: \
                     jitcode_index={} entry={} bank={bank} color={color}",
                    pjc.jitcode.index(),
                    entry,
                );
            }
            fbw_bridge_decline(ctx);
            return None;
        }
    }

    let Some((code_len, mut walk_result)) = dispatch_perfn_frame(
        ctx,
        sym,
        cf_addr,
        start_pc,
        &session,
        &pjc,
        entry,
        &argboxes_r,
        &argboxes_i,
        &argboxes_f,
        authoritative,
    ) else {
        return None;
    };

    // Full-body-walk loop close: the walker's `jit_merge_point` handler
    // produces RPython-style reds (`jump_args = [frame, ec]`, len 2 for the
    // portal jitdriver), but pyre's runtime closes loops against the
    // EXPLICIT scalar inputarg vector
    // `[frame, ec, next_instr, code, valuestackdepth, debugdata, lastblock,
    //  namespace, locals..., stack...]` (len >= NUM_SCALAR_INPUTARGS).
    // `validate_close_with_jump_args` (state.rs) rejects the reds shape, so
    // rebuild the explicit vector via `close_loop_args_at`, matching
    // `reached_loop_header` (trace_opcode.rs close path). The
    // loop-carried local/stack OpRefs come from the virtualizable shadow in
    // the TraceCtx (`virtualizable_box_at`, maintained by the authoritative
    // walk's vable ops), NOT from the walk's private register file, so the
    // shadow is live here even though `sym.registers_r` is not.
    //
    // Authoritative only: `close_loop_args_at` records SameAs ops and flushes
    // the virtualizable shadow to the concrete frame heap, which the
    // read-only probe (trace discarded) must not do.
    if authoritative {
        let close_loop_restart_pc = match &walk_result {
            Ok((
                crate::jitcode_dispatch::DispatchOutcome::CloseLoop {
                    loop_header_pc,
                    loop_header_marker_jit_pc,
                    ..
                },
                _end_pc,
            )) => Some(loop_header_marker_jit_pc.map_or(*loop_header_pc, |marker| {
                let marker_py =
                    crate::py_coord::containing_py_pc_for_jitcode_pc(&pjc.metadata, marker)
                        as usize;
                if marker_py == *loop_header_pc
                    && pjc.merge_entry_for(*loop_header_pc) != Some(marker)
                {
                    *loop_header_pc + 1
                } else {
                    marker_py
                }
            })),
            _ => None,
        };
        if let Ok((
            crate::jitcode_dispatch::DispatchOutcome::CloseLoop {
                jump_args,
                loop_header_pc,
                loop_header_marker_jit_pc,
            },
            _end_pc,
        )) = &mut walk_result
        {
            let loop_header_pc = *loop_header_pc;
            let restart_pc = close_loop_restart_pc.expect("close loop has a restart pc");
            WALK_END_RESTART_PC.with(|c| c.set(Some(restart_pc)));
            // `close_loop_args_at` reads the loop-header `orgpc` for the
            // last_instr anchor, so pass that coordinate explicitly.
            *jump_args = sym.close_loop_args_at(
                ctx,
                cf_addr,
                loop_header_pc,
                Some(loop_header_pc),
                *loop_header_marker_jit_pc,
            );
        }
        // pyjitpl.py:3048-3091 raise_continue_running_normally parity: a
        // walk that ends at a merge point hands the interpreter (and the
        // compiled loop's heap-reloading preamble) the END-of-walk frame
        // state, so the walked iteration — whose residual calls executed
        // concretely — is not re-run.  After `close_loop_args_at` (whose
        // jump-arg derivation reads the pre-walk frame) is the one safe
        // commit point.  All-or-nothing inside the helper; a `false`
        // return keeps the legacy replay.
        //
        // Commit preconditions:
        //   - no unjournaled effect (a symbolically recorded residual
        //     call only the replay applies);
        //   - the frame flush resolves every live slot (all-or-nothing);
        // then the committed flag routes the portal to adopt the end
        // state instead of replaying.  The store-journal epilogue below
        // settles the walk's eager list stores either way (commit keeps
        // them, non-commit rolls them back for the replay).
        if let Ok((outcome, _end_pc)) = &walk_result {
            let header_pc = match outcome {
                crate::jitcode_dispatch::DispatchOutcome::CloseLoop { .. } => close_loop_restart_pc,
                crate::jitcode_dispatch::DispatchOutcome::CompileTracePending {
                    loop_header_pc,
                } => Some(*loop_header_pc),
                _ => None,
            };
            if let Some(header_pc) = header_pc {
                if crate::jitcode_dispatch::fbw_has_unjournaled_effect() {
                    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-end-flush] declined at header_pc={header_pc} \
                             (unjournaled effect) — legacy replay kept"
                        );
                    }
                } else if crate::state::flush_walk_loop_end_state_to_frame(ctx, cf_addr, header_pc)
                {
                    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-end-flush] COMMIT header_pc={header_pc} bridge={} \
                             journal_len={} outcome={outcome:?}",
                            ctx.is_bridge_trace,
                            crate::jitcode_dispatch::fbw_store_journal_len(),
                        );
                    }
                    // The loop header IS where this walk ended; the sticky
                    // unjournaled check above is the whole gate.
                    let _ = commit_walk_end(WalkEndCommitLeg::LoopHeader, WalkEndResume::Terminal);
                } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-end-flush] declined at header_pc={header_pc} (shadow slot \
                         without concrete / depth / lastblock) — legacy replay kept"
                    );
                }
            }
        }

        // Inline-callee forward abort, or an `abort_permanent` marker abort
        // (DELETE_FAST and the other emit_abort_permanent opcodes).  The
        // marker's contract is "resume
        // the interpreter AT this unsupported opcode and run it" — codewriter
        // stores `last_instr = py_pc - 1` for the blackhole.  On the
        // full-body walk that recorded write is discarded with the aborted
        // trace, while the walk already executed the region's residual side
        // effects concretely, so the legacy `ContinueRunningNormally` replays
        // them from entry → double-execution (e.g. a `del`-bearing method
        // whose prior STORE_ATTR ran once during the walk, then again on
        // replay).  Flush the abort-point frame (locals + last_instr) so the
        // portal resumes at the unsupported opcode instead of replaying.
        // The marker-only fallback uses the same no-unjournaled-effect
        // predicate as the CloseLoop end-flush above.  A latched inline-callee
        // forward abort has already distinguished an outside mark from a mark
        // inside its discarded attempt.
        let live_root_addr = sym.live_vable_frame_addr();
        let trace_too_long_adopted =
            matches!(
                &walk_result,
                Err(crate::jitcode_dispatch::DispatchError::TraceTooLong { .. })
            ) && try_adopt_blackhole(ctx, cf_addr, live_root_addr, WalkEndCommitLeg::TraceTooLong);
        // Every OTHER aborting error is the same situation: the walk executed
        // residuals concretely and then stopped, so replaying the region from
        // the trace entry re-applies those effects.  `convert_and_run_from_
        // pyjitpl` (`blackhole.py:1799`) finishes the frames instead, which is
        // what the latch staged.
        //
        // Four classes keep their own, more precise recovery and are excluded
        // so this general leg cannot pre-empt them:
        // `VableEscapedDuringResidualCall` latches a narrower resume-marker
        // image and has an escape-pc fallback (arm below);
        // `AbortPermanentMarkerReached` / `LoopBearingCalleeInlineUnsupported`
        // route to the gh#467 CALL-forward carrier, which resumes the OUTER
        // frame at its CALL rather than inside the discarded callee attempt;
        // `ForceQuasiImmutable` resumes AT the forcing opcode via
        // `flush_qmut_abort_state` (arm below), which re-runs the write the
        // walk stopped in front of instead of finishing the frame past it.
        //
        // And only for an abort whose image is COMPLETE
        // (`DispatchError::leaves_complete_image`).  Pyre's walker has a whole
        // family of aborts upstream has no counterpart for — the abort IS the
        // report that a register / concrete / descr could not be resolved — and
        // for those the MIFrame the latch would build is missing exactly the
        // value the blackhole resumes on.  Measured over the 353 synth
        // fixtures with the classification removed, 351 are unchanged and two
        // break: `list_length_hint_validate` adopts its one
        // `RegisterReadUnbound` walk (`pc=456 reg=3 bank=r`) and underflows the
        // operand stack, and `getframe_while_subwalk_decline_shapes` — a
        // fixture pinned precisely so that "the decline stays a decline rather
        // than silently becoming a wrong answer" — adopts an inline-escape
        // shape whose caller banks are incomplete and dies on an unwired
        // blackhole opcode.
        let walk_abort_adopted = !trace_too_long_adopted
            && matches!(&walk_result, Err(error) if error.leaves_complete_image() && !matches!(
                error,
                crate::jitcode_dispatch::DispatchError::TraceTooLong { .. }
                    | crate::jitcode_dispatch::DispatchError::VableEscapedDuringResidualCall { .. }
                    | crate::jitcode_dispatch::DispatchError::AbortPermanentMarkerReached { .. }
                    | crate::jitcode_dispatch::DispatchError::LoopBearingCalleeInlineUnsupported { .. }
                    | crate::jitcode_dispatch::DispatchError::ForceQuasiImmutable { .. }
            ))
            && walk_abort_leg_enabled()
            && try_adopt_blackhole(ctx, cf_addr, live_root_addr, WalkEndCommitLeg::WalkAbort);
        if walk_abort_adopted && crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!("[fbw-blackhole] adopted WALK_ABORT forward resume");
        }
        let vable_escaped = matches!(
            &walk_result,
            Err(crate::jitcode_dispatch::DispatchError::VableEscapedDuringResidualCall { .. })
        );
        let force_blackhole_adopted = vable_escaped
            && try_adopt_blackhole(ctx, cf_addr, live_root_addr, WalkEndCommitLeg::VableEscape);
        let mut escape_pc_adopted = false;
        if trace_too_long_adopted && crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!("[fbw-blackhole] adopted ABORT_TOO_LONG forward resume");
        }
        if !force_blackhole_adopted
            && vable_escaped
            && let Some((resume_py_pc, escape_kind)) =
                crate::jitcode_dispatch::take_committed_frame_escape_pc()
        {
            escape_pc_adopted = true;
            // BOTH flushes inside `flush_active_frame_escape` rewind: they take
            // the same `py_pc` and the same `last_instr = pc - 1`, so the
            // escaping opcode re-runs either way.  They differ in whether the
            // mid-expression operand stack was reconstructed, and — the part
            // that matters here — in whether any gate ran at all.  The latched
            // path is gated by `escape_opcode_window_clean` back at the
            // residual; the merge-point fallback had no gate anywhere, so it
            // has nothing to offer this contract and is refused.
            let resume = match escape_kind {
                crate::jitcode_dispatch::EscapeResumeKind::Exact => {
                    WalkEndResume::RewindProvenAtLatch
                }
                crate::jitcode_dispatch::EscapeResumeKind::RerunsOpcode => {
                    WalkEndResume::RewindUnproven
                }
            };
            if commit_walk_end(WalkEndCommitLeg::VableEscape, resume) {
                crate::jitcode_dispatch::discard_escape_flush_undo();
                // The force-time escape flush wrote the resume state into the
                // LIVE frame (the frame the callee inspected).  The portal
                // epilogue propagates `executed_frame` → live on a committed
                // flush, so mirror the live frame's resume state into the walk
                // snapshot to make that copy the identity.
                let live = sym.live_vable_frame_addr();
                if live != 0 && cf_addr != 0 && live != cf_addr {
                    unsafe {
                        (*(cf_addr as *mut pyre_interpreter::PyFrame)).restore_resume_state_from(
                            &*(live as *const pyre_interpreter::PyFrame),
                        );
                    }
                }
                // The committed flush owns the iteration count (the resume pc
                // is PAST the FOR_ITER consume); drop any in-flight item so
                // the legacy deliver cannot re-apply one.
                crate::jitcode_dispatch::fbw_foriter_inflight_clear();
                WALK_END_RESTART_PC.with(|c| c.set(Some(resume_py_pc)));
            } else {
                // Put the live frame back to its pre-flush state: the legacy
                // replay's contract is that the frame still holds pre-walk
                // state, and its journal rollback makes the replay exactly-once.
                crate::jitcode_dispatch::restore_escape_flush_undo();
                if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-abort-flush] escape flush declined at \
                         resume_py_pc={resume_py_pc} (merge-point fallback re-runs the \
                         escaping opcode) — legacy replay kept"
                    );
                }
            }
        }
        // The force arm withdrew its commit and DEFERRED the frame restore to
        // here.  Neither continuation that keeps the flushed frame ran, so the
        // walk falls back to replaying the traced region from its entry: put
        // the pre-flush locals / operand stack / resume coordinate back so the
        // replay re-derives them instead of compounding onto the walk's
        // mid-region values.  When a blackhole terminal or a committed escape
        // pc DID take over, the flush stands — `virtualizable.py:101-138
        // write_boxes` has no undo once the vable is forced, and the resumed
        // interpreter reads its fastlocals straight out of that array.
        // `walk_abort_adopted` is a blackhole terminal in exactly that sense:
        // the chain ran forward from the flushed frame, so restoring the
        // pre-flush image here would roll the vable back underneath it.
        if crate::jitcode_dispatch::take_escape_flush_undo_pending()
            && !force_blackhole_adopted
            && !escape_pc_adopted
            && !walk_abort_adopted
        {
            crate::jitcode_dispatch::restore_escape_flush_undo();
        }
        let call_forward_abort = match &walk_result {
            Err(crate::jitcode_dispatch::DispatchError::AbortPermanentMarkerReached { pc }) => {
                Some((*pc, true))
            }
            Err(crate::jitcode_dispatch::DispatchError::LoopBearingCalleeInlineUnsupported {
                pc,
            }) => Some((*pc, false)),
            _ => None,
        };
        let mut committed_entry_carrier_call_py_pc = None;
        if let Some((abort_jit_pc, is_marker_abort)) = call_forward_abort {
            // gh#467: a supported abort fired inside a TOP-level inline
            // sub-walk whose callee executed no concrete effect
            // (`try_walker_inline_user_call` latched the carrier only under
            // that gate).  The nested-unjournaled-decline class means the
            // residual did not execute; its callee attempt can be discarded
            // with any inside-only unjournaled mark.  Flush the OUTER frame
            // at the CALL that entered the callee and resume the interpreter
            // forward — re-executing the whole call from scratch — instead
            // of the legacy replay from loop entry, which double-applies the
            // non-journaled pre-CALL store.  The abort's `abort_jit_pc` is a
            // CALLEE coordinate with no meaning in the outer py_pc tables,
            // so the outer CALL py_pc and operand stack come from the latch.
            // Convergence of `run_blackhole_interp_to_cancel_tracing`
            // (`pyjitpl.py:2949`), minus the inner-frame rebuild (#126/#215).
            let carrier = crate::jitcode_dispatch::fbw_abort_carrier_clone();
            match carrier.as_ref() {
                Some(crate::jitcode_dispatch::InlineAbortCarrier::Entry {
                    outer_jitcode_index,
                    call_jitcode_pc,
                    call_stack,
                    entry_executed_effects,
                }) => {
                    committed_entry_carrier_call_py_pc = try_commit_entry_carrier_call(
                        ctx,
                        cf_addr,
                        abort_jit_pc,
                        *outer_jitcode_index,
                        *call_jitcode_pc,
                        call_stack,
                        *entry_executed_effects,
                    );
                }
                Some(crate::jitcode_dispatch::InlineAbortCarrier::MidBody(payload))
                    if (is_marker_abort
                        && payload.abort_kind
                            == crate::jitcode_dispatch::MidBodyAbortKind::Marker)
                        || (!is_marker_abort
                            && payload.abort_kind
                                == crate::jitcode_dispatch::MidBodyAbortKind::Structural) =>
                {
                    let rebuilt = match resolve_midbody_flush_words(payload) {
                        Some(words) => {
                            let outcome = try_commit_midbody_abort(ctx, cf_addr, payload, words);
                            if outcome.is_ok() && crate::jitcode_dispatch::fbw_debug_abort_enabled()
                            {
                                eprintln!(
                                    "[fbw-abort-flush] gh#467 callee-rebuild COMMIT \
                                     abort_jit_pc={abort_jit_pc} callee_py_pc={} \
                                     call_py_pc={} post_call_py_pc={}",
                                    words.callee_py_pc, words.call_py_pc, words.post_call_py_pc,
                                );
                            }
                            outcome
                        }
                        None => {
                            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                                eprintln!(
                                    "[fbw-abort-flush] gh#467 callee-rebuild declined at \
                                     abort_jit_pc={abort_jit_pc} (unresolved carried jitcode \
                                     identity or null code ptr)",
                                );
                            }
                            // Nothing was rebuilt, so nothing of the callee ran.
                            Err(MidBodyDecline::BeforeRun("unresolved jitcode identity"))
                        }
                    };
                    match rebuilt {
                        Ok(()) => {
                            // This leg resumes INSIDE the rebuilt callee at its
                            // abort pc — ahead of what the callee applied, not
                            // behind it.  Nothing re-runs; committing is what
                            // keeps those effects.
                            let _ = commit_walk_end(
                                WalkEndCommitLeg::CalleeRebuild,
                                WalkEndResume::AfterApplied,
                            );
                        }
                        Err(MidBodyDecline::BeforeRun(_)) => {
                            if let Some(fallback) = payload.entry_fallback.as_ref() {
                                // The rebuild declined before running anything
                                // and the entry latch's gate had held, so
                                // rewinding to the outer CALL is still open.
                                // Falling through to the legacy replay instead
                                // would re-apply the non-journaled pre-CALL
                                // stores.
                                committed_entry_carrier_call_py_pc = try_commit_entry_carrier_call(
                                    ctx,
                                    cf_addr,
                                    abort_jit_pc,
                                    payload.outer_jitcode_index,
                                    payload.call_jitcode_pc,
                                    &fallback.call_stack,
                                    fallback.entry_executed_effects,
                                );
                            }
                        }
                        Err(MidBodyDecline::AfterRun(_)) => {
                            // The callee body already executed.  `EntryCarrierCall`
                            // rewinds the outer frame to its CALL, which would run
                            // that body a SECOND time — the gh#467 double-apply.
                            // Its `walk_end_resume_provable` re-check does not stop
                            // it: that samples `FBW_EXECUTED_EFFECT_COUNT`, which is
                            // walker-side, and the plain interpretation inside
                            // `execute_frame` never bumps it.  So take no leg.
                            //
                            // ⚠️This NARROWS the hazard, it does not close it: the
                            // legacy replay this falls through to re-enters the
                            // outer frame at its entry and re-runs the CALL too.
                            // The callee's effects are user code and the store
                            // journal does not cover them, so neither branch can
                            // undo them.  Closing it means making the post-run path
                            // infallible — every one of these declines already has a
                            // pre-run counterpart (`can_flush_walk_end_state_after_
                            // outer_call`, the propagate licence), so reaching here
                            // means a pre-check was too weak, not that a new
                            // fallback is needed.
                        }
                    }
                }
                None if is_marker_abort => {
                    if crate::jitcode_dispatch::fbw_has_unjournaled_effect()
                        || session.borrow().abort_in_subwalk
                    {
                        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-abort-flush] declined at abort_jit_pc={abort_jit_pc} \
                                     (unjournaled effect or inline sub-walk) — legacy replay kept"
                            );
                        }
                    } else if let Some(resume_py_pc) =
                        crate::jitcode_dispatch::fbw_abort_resume_py_pc(sym, abort_jit_pc)
                    {
                        if crate::state::flush_walk_end_state_to_frame(ctx, cf_addr, resume_py_pc) {
                            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                                eprintln!(
                                    "[fbw-abort-flush] COMMIT abort_jit_pc={abort_jit_pc} \
                                         resume_py_pc={resume_py_pc}"
                                );
                            }
                            // The abort pc IS where the walk stopped; the
                            // unjournaled/sub-walk check above is the gate.
                            let _ =
                                commit_walk_end(WalkEndCommitLeg::AbortPc, WalkEndResume::Terminal);
                        } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-abort-flush] declined at resume_py_pc={resume_py_pc} \
                                     (shadow slot without concrete / depth / lastblock) — legacy replay kept"
                            );
                        }
                    }
                }
                _ if crate::jitcode_dispatch::fbw_debug_abort_enabled() => {
                    eprintln!(
                        "[fbw-abort-flush] gh#467 CALL-forward declined at \
                             abort_jit_pc={abort_jit_pc} (no carrier) — legacy replay kept"
                    );
                }
                _ => {}
            }
            if carrier.is_some() {
                crate::jitcode_dispatch::fbw_abort_carrier_clear();
            }
        }
        if let Err(crate::jitcode_dispatch::DispatchError::LoopBearingCalleeInlineUnsupported {
            pc,
        }) = &walk_result
        {
            let abort_jit_pc = *pc;
            // A walk commits at most ONE leg.  The carrier block above may
            // already have taken `CalleeRebuild`, which resumes INSIDE the
            // rebuilt callee, past what it applied; rewinding the caller to its
            // CALL on top of that runs the whole callee a second time.  Nothing
            // else keeps the two apart — this leg's own gates are inclusion
            // tests on the residual odometer, and `walk_end_resume_provable`
            // cannot see effects applied by the plain interpretation the
            // rebuild resumed into (the `MidBodyDecline::AfterRun` argument,
            // which the carrier block applies only to its own fallback).
            if WALK_END_FLUSH_COMMITTED.with(|c| c.get()) {
                crate::jitcode_dispatch::fbw_abort_outer_resume_reset();
                if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-abort-flush] declined at abort_jit_pc={abort_jit_pc} \
                             (a carrier leg already committed this walk)"
                    );
                }
            } else if !crate::jitcode_dispatch::fbw_executed_nonpure_residual() {
                if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-abort-flush] declined at abort_jit_pc={abort_jit_pc} \
                             (no executed non-pure residual) — legacy replay kept"
                    );
                }
            } else if crate::jitcode_dispatch::fbw_has_unjournaled_effect() {
                if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-abort-flush] declined at abort_jit_pc={abort_jit_pc} \
                             (unjournaled effect) — legacy replay kept"
                    );
                }
            } else if let Some((jitcode_index, call_jitcode_pc, effects_at_resume_point)) =
                crate::jitcode_dispatch::fbw_abort_outer_resume_take()
            {
                // Like the entry carrier, this resume re-executes the outer
                // CALL, so it needs the same zero-delta proof — the one the
                // latch sampled at that CALL.
                let resume = WalkEndResume::Rewind {
                    effects_at_resume_point,
                };
                let pjc = crate::state::pyjitcode_for_jitcode_index(jitcode_index as i32);
                if !walk_end_resume_provable(resume) {
                    crate::jitcode_dispatch::fbw_abort_outer_stack_overrides_clear();
                    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-abort-flush] declined at abort_jit_pc={abort_jit_pc} \
                                 (executed-effect delta since the outer CALL) — legacy replay kept"
                        );
                    }
                } else if let Some(pjc) = pjc {
                    let resume_py_pc = crate::py_coord::containing_py_pc_for_jitcode_pc(
                        &pjc.metadata,
                        call_jitcode_pc,
                    ) as usize;
                    if committed_entry_carrier_call_py_pc == Some(resume_py_pc) {
                        crate::jitcode_dispatch::fbw_abort_outer_stack_overrides_clear();
                        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-abort-flush] skipped at resume_py_pc={resume_py_pc} \
                                     (entry carrier already handled same resume)"
                            );
                        }
                    } else {
                        // Flush while the overrides stay rooted in
                        // FBW_ABORT_OUTER_STACK_OVERRIDES (the flush boxes Int/Float
                        // locals — an allocation that can move the nursery-resident
                        // override refs; the area walker forwards them in place),
                        // then clear the cell.
                        let committed =
                            crate::jitcode_dispatch::fbw_abort_outer_stack_overrides_with(
                                |stack_overrides| {
                                    crate::state::flush_walk_end_state_to_frame_with_stack_overrides(
                                        ctx,
                                        cf_addr,
                                        resume_py_pc,
                                        stack_overrides,
                                    )
                                },
                            );
                        crate::jitcode_dispatch::fbw_abort_outer_stack_overrides_clear();
                        if committed {
                            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                                eprintln!(
                                    "[fbw-abort-flush] COMMIT abort_jit_pc={abort_jit_pc} \
                                         resume_py_pc={resume_py_pc} (nested inline decline)"
                                );
                            }
                            let committed =
                                commit_walk_end(WalkEndCommitLeg::NestedInlineOuterCall, resume);
                            debug_assert!(committed, "provability re-checked after a pure flush");
                        } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-abort-flush] declined at resume_py_pc={resume_py_pc} \
                                     (shadow slot without concrete / depth / lastblock) — legacy replay kept"
                            );
                        }
                    }
                } else {
                    crate::jitcode_dispatch::fbw_abort_outer_stack_overrides_clear();
                    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-abort-flush] declined at abort_jit_pc={abort_jit_pc} \
                                 (unresolved outer jitcode_index={jitcode_index}) — legacy replay kept"
                        );
                    }
                }
            } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                eprintln!(
                    "[fbw-abort-flush] declined at abort_jit_pc={abort_jit_pc} \
                         (no outer caller resume pc) — legacy replay kept"
                );
            }
        }

        // `SwitchToBlackhole(ABORT_FORCE_QUASIIMMUT)` (pyjitpl.py:1116) landed
        // as a resume, not a replay.  Upstream's blackhole picks up at the
        // `-live-` in front of the forcing write with every earlier residual
        // already applied and never re-runs one; the walker's equivalent is to
        // keep the journal and resume the interpreter AT the Python opcode the
        // write belongs to, which has not run yet (the abort fires before the
        // residual executes).  Falling through to the plain `Abort` instead
        // rolls the journal back and replays the walked region from its start,
        // re-executing every residual the walk already ran.
        if let Err(crate::jitcode_dispatch::DispatchError::ForceQuasiImmutable { pc }) =
            &walk_result
        {
            let abort_jit_pc = *pc;
            // A recorded-but-unexecuted residual is applied only by the replay
            // this leg removes, so committing would DROP it; a sub-walk abort's
            // resume coordinate names the callee's code object, not `sym`'s.
            if crate::jitcode_dispatch::fbw_has_unjournaled_effect()
                || session.borrow().abort_in_subwalk
            {
                if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-qmut-flush] declined at abort_jit_pc={abort_jit_pc} \
                         (unjournaled effect or inline sub-walk) — legacy replay kept"
                    );
                }
            } else if let Some((resume_py_pc, oprefs)) =
                crate::jitcode_dispatch::fbw_qmut_abort_stack_take()
            {
                // The resume RE-RUNS this opcode (`last_instr = pc - 1`), so it
                // owes the `Rewind` proof: nothing of it may have been applied
                // yet.  The sample is taken at the opcode boundary itself, and
                // its absence (no boundary crossed, or a different opcode)
                // leaves the leg unprovable — `walk_end_resume_provable`
                // declines and the legacy replay stands.
                let resume =
                    match crate::jitcode_dispatch::fbw_opcode_entry_effects_at(resume_py_pc) {
                        Some(effects_at_resume_point) => WalkEndResume::Rewind {
                            effects_at_resume_point,
                        },
                        None => WalkEndResume::RewindUnproven,
                    };
                if !walk_end_resume_provable(resume) {
                    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-qmut-flush] declined at resume_py_pc={resume_py_pc} \
                             (opcode already applied an effect, or no entry sample) \
                             — legacy replay kept"
                        );
                    }
                } else if crate::jitcode_dispatch::flush_qmut_abort_state(
                    ctx,
                    cf_addr,
                    resume_py_pc,
                    &oprefs,
                ) {
                    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-qmut-flush] COMMIT abort_jit_pc={abort_jit_pc} \
                             resume_py_pc={resume_py_pc}"
                        );
                    }
                    let committed = commit_walk_end(WalkEndCommitLeg::AbortPc, resume);
                    debug_assert!(committed, "provability re-checked after a pure flush");
                } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-qmut-flush] declined at resume_py_pc={resume_py_pc} \
                         (operand slot without concrete / depth / lastblock) — legacy replay kept"
                    );
                }
            } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                eprintln!(
                    "[fbw-qmut-flush] declined at abort_jit_pc={abort_jit_pc} \
                     (no operand-stack mirror latched) — legacy replay kept"
                );
            }
        }

        // #32 S2: a kept-stack branch guard whose not-taken arm cannot be
        // restored for the COMPILED trace aborts (`AbortPermanent` for the
        // unrestorable-Ref shape, decline + `Abort` for the depth>1
        // invalid-mirror shape), but the authoritative walk's symbolic shadow
        // IS complete at the abort pc (the hazard is about the JIT resume
        // snapshot, not the interpreter-side shadow).  Flush that end state to
        // the live frame so the interpreter resumes at the abort pc with the
        // walked iterations already counted, instead of discarding the walk
        // and dropping an in-flight FOR_ITER item via the conservative #30
        // header-guard drop (or, for the `Unsupported` shape, re-executing the
        // walk's residual effects from loop entry).  Same
        // no-unjournaled-effect / no-sub-walk predicate and same all-or-nothing
        // `flush_walk_end_state_to_frame` gate as the CloseLoop / marker legs;
        // when the flush declines (a slot the shadow cannot resolve) the legacy
        // drop stands (the residual S3 case).
        let kept_stack_abort_pc = match &walk_result {
            Err(
                crate::jitcode_dispatch::DispatchError::BranchGuardUnrestorableKeptStackPermanent {
                    pc,
                },
            ) => Some((*pc, false)),
            Err(crate::jitcode_dispatch::DispatchError::BranchGuardKeptStackUnsupported { pc }) => {
                Some((*pc, true))
            }
            _ => None,
        };
        if let Some((pc, is_unsupported)) = kept_stack_abort_pc {
            let abort_jit_pc = pc;
            if crate::jitcode_dispatch::fbw_has_unjournaled_effect()
                || session.borrow().abort_in_subwalk
            {
                if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-branch-flush] declined at abort_jit_pc={abort_jit_pc} \
                             (unjournaled effect or inline sub-walk) — legacy drop kept"
                    );
                }
            } else if let Some(resume_py_pc) =
                crate::jitcode_dispatch::fbw_abort_resume_py_pc(sym, abort_jit_pc)
            {
                // Two kept-stack branch aborts reach this leg (`is_unsupported`
                // came from the `kept_stack_abort_pc` match).  Both resume at a
                // FOR_ITER header whose walk already advanced the iterator; they
                // differ in whether the consumed item's body ran.
                let committed = if is_unsupported {
                    if crate::jitcode_dispatch::fbw_foriter_inflight_completed_at_resume(
                        cf_addr,
                        resume_py_pc,
                    ) {
                        // The consumed item's body already ran, so resume at
                        // the FOR_ITER header without re-delivering it.
                        crate::state::flush_walk_end_state_to_frame(ctx, cf_addr, resume_py_pc)
                    } else {
                        // A nested inner FOR_ITER can carry the enclosing
                        // iterator as its kept stack before the consumed
                        // item's body runs.  Mirror Shape A and deliver that
                        // in-flight item exactly once.
                        let push = crate::jitcode_dispatch::fbw_foriter_inflight_take_for_resume(
                            cf_addr,
                            resume_py_pc,
                        );
                        push.is_some()
                            && crate::state::flush_walk_end_state_to_frame_with_item(
                                ctx,
                                cf_addr,
                                resume_py_pc,
                                push,
                            )
                    }
                } else {
                    // Shape A — a `BranchGuardUnrestorableKeptStackPermanent`
                    // abort resumes AT a FOR_ITER header whose consumed item is
                    // in flight (`body_pc == resume_py_pc + 1`, the opcode
                    // there really is a FOR_ITER): the walk advanced the
                    // iterator but the item is not yet on the flushed (header)
                    // stack, so deliver it (push + reposition to the body) so
                    // the body runs once.  Commit ONLY when an item is
                    // delivered — a Permanent abort not at such a header keeps
                    // the legacy drop byte-identically (the residual S3 case),
                    // so every other abort shape (and the whole flag-OFF path)
                    // is untouched.
                    let push = crate::jitcode_dispatch::fbw_foriter_inflight_take_for_resume(
                        cf_addr,
                        resume_py_pc,
                    );
                    push.is_some()
                        && crate::state::flush_walk_end_state_to_frame_with_item(
                            ctx,
                            cf_addr,
                            resume_py_pc,
                            push,
                        )
                };
                if committed {
                    // The flush owns the iteration count; drop any remaining
                    // in-flight items so the legacy deliver cannot re-apply
                    // one (exactly-once).
                    crate::jitcode_dispatch::fbw_foriter_inflight_clear();
                    // The abort pc is where the walk stopped, and the in-flight
                    // item is delivered exactly once by the flush above.
                    let _ = commit_walk_end(WalkEndCommitLeg::BranchGuard, WalkEndResume::Terminal);
                    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-branch-flush] COMMIT abort_jit_pc={abort_jit_pc} \
                                 resume_py_pc={resume_py_pc} (delivered in-flight FOR_ITER item)"
                        );
                    }
                } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                    eprintln!(
                        "[fbw-branch-flush] declined at resume_py_pc={resume_py_pc} \
                             (shadow slot without concrete / depth / lastblock) — legacy drop kept"
                    );
                }
            }
        }
    }

    // No-replay portal exit for a loop-free function trace: a `Terminate`
    // walk whose top-level `*_return` captured a concrete result hands that
    // result to the portal directly (`eval.rs` consumes the stash) instead
    // of re-running the freshly compiled trace for the SAME invocation —
    // the walk already executed every residual call concretely, consuming
    // any side-effecting callee (e.g. a tokenizer's `get`), so a re-run
    // would re-read the mutated heap and deopt.  Declined when an
    // unjournaled effect (a symbolically recorded residual only the legacy
    // replay applies) is present: drop the capture so the portal degrades
    // to `ContinueRunningNormally`.  This shares its predicate with the
    // store-journal commit below so the two decisions never disagree.
    //
    // A guard-failure BRIDGE `Terminate` walk takes the same shortcut when
    // the bridge tracer armed it (`fbw_bridge_noreplay_armed`): the caller
    // hands the captured concrete result forward as `DoneWithThisFrame`
    // rather than rewinding the live frame to the guard pc and re-running the
    // region through the `ContinueRunningNormally` re-entry — which would
    // execute every residual a second time and double-apply any
    // callee-internal side effect (#177).  Both bridge callers arm any resume,
    // single- or multi-frame — `Terminate` is always the LIVE frame's return
    // (an inlined callee's return is a `SubReturn` inside the walk), so its
    // concrete result is what that frame hands its caller either way.  The
    // general guard path consumes the kept stash as a terminal
    // `BridgeResolution`; the CALL_ASSEMBLER callback routes it through a
    // back-to-back blackhole hook that returns it as the CA callee's result
    // under a live-frame identity check, without rebuilding a framestack.
    // A committed journal therefore never strands into a guard-state re-run:
    // the three decisions (this predicate, the journal commit below, and the
    // caller's consume-vs-rewind) stay in agreement.
    //
    // The shortcut itself has no upstream counterpart, and needs none:
    // `pyjitpl.py:2937-2947 _handle_guard_failure` resumes by continuing from
    // the framestack `interpret()` already holds and never re-runs a region, so
    // there is no store journal to commit and no concrete to keep.  pyre's
    // walker instead executes and journals residual effects while tracing, so
    // "do not run that region twice" has to be decided here.
    let terminate_no_replay = (!is_bridge_trace
        || crate::jitcode_dispatch::fbw_bridge_noreplay_armed())
        && matches!(
            &walk_result,
            Ok((crate::jitcode_dispatch::DispatchOutcome::Terminate, _))
        )
        && crate::jitcode_dispatch::fbw_finish_concrete_peek().is_some()
        && !crate::jitcode_dispatch::fbw_has_unjournaled_effect();
    // A terminal blackhole adoption is possible for the two dedicated
    // handoff errors and for the general WalkAbort family.  The latter was
    // added after this no-replay predicate and must be classified through the
    // SAME complete-image allow-list as the adoption leg above; otherwise a
    // `DoneWithThisFrame` result is discarded and interpreter replay resumes
    // one opcode past RETURN_VALUE.
    let blackhole_terminal_no_replay = matches!(
        &walk_result,
        Err(error) if blackhole_terminal_error(error)
    ) && WALK_END_FLUSH_COMMITTED.with(|slot| slot.get())
        && crate::jitcode_dispatch::fbw_finish_concrete_peek().is_some();
    if !terminate_no_replay && !blackhole_terminal_no_replay {
        crate::jitcode_dispatch::fbw_finish_concrete_reset();
    }
    // The one journal-keeping path outside the flush legs: it sets no flush
    // flag (the caller consumes the concrete result instead of adopting a
    // resume pc), so tag it here or the census reports it as `leg=0`.
    // `blackhole_terminal_no_replay` needs no tag — it only refines a walk
    // whose VableEscape leg already committed, and retagging would erase it.
    if terminate_no_replay {
        let _ = record_walk_end_leg(WalkEndCommitLeg::TerminateNoReplay, WalkEndResume::Terminal);
    }

    // Store-journal epilogue, on EVERY walk exit (commit, declined
    // commit, walk error): a committed walk keeps its eagerly executed
    // list stores (drop the undo log); any other exit returns control to
    // a replay-from-start path, which re-executes the walked region and
    // must find the pre-walk heap — roll the stores back.  A
    // `terminate_no_replay` exit also keeps the stores: the portal returns
    // the walk's result without replaying, exactly like the loop-flush
    // commit.
    if is_bridge_trace && crate::jitcode_dispatch::fbw_debug_abort_enabled() {
        let outcome_kind = match &walk_result {
            Ok((crate::jitcode_dispatch::DispatchOutcome::Continue, _)) => "Continue",
            Ok((crate::jitcode_dispatch::DispatchOutcome::Terminate, _)) => "Terminate",
            Ok((crate::jitcode_dispatch::DispatchOutcome::SubReturn { .. }, _)) => "SubReturn",
            Ok((crate::jitcode_dispatch::DispatchOutcome::SubRaise { .. }, _)) => "SubRaise",
            Ok((crate::jitcode_dispatch::DispatchOutcome::SwitchToBlackhole { .. }, _)) => {
                "SwitchToBlackhole"
            }
            Ok((crate::jitcode_dispatch::DispatchOutcome::CloseLoop { .. }, _)) => "CloseLoop",
            Ok((crate::jitcode_dispatch::DispatchOutcome::CompileTracePending { .. }, _)) => {
                "CompileTracePending"
            }
            Ok((_, _)) => "OtherOk",
            Err(_) => "Err",
        };
        eprintln!(
            "[fbw-bridge-epilogue] committed={} store_journal_len={} unjournaled={} outcome={}",
            WALK_END_FLUSH_COMMITTED.with(|c| c.get()),
            crate::jitcode_dispatch::fbw_store_journal_len(),
            crate::jitcode_dispatch::fbw_has_unjournaled_effect(),
            outcome_kind,
        );
    }
    let committed = WALK_END_FLUSH_COMMITTED.with(|c| c.get()) || terminate_no_replay;
    let journal = crate::jitcode_dispatch::fbw_store_journal_len();
    if committed {
        crate::jitcode_dispatch::fbw_store_journal_commit();
        crate::jitcode_dispatch::fbw_exit_last_instr_commit();
        // A committed bridge recording keeps its advanced iterator cursor (the
        // compiled bridge / adopted end state owns the iteration count).
        crate::jitcode_dispatch::fbw_bridge_iter_journal_clear();
    } else {
        crate::jitcode_dispatch::fbw_store_journal_rollback();
        // The exit coordinate the walk published goes back too: this replay
        // resumes the frame from its pre-walk state and derives the next
        // instruction from that field, so a kept exit coordinate would restart
        // it past its own return or raise.
        crate::jitcode_dispatch::fbw_exit_last_instr_rollback();
        // A bridge/retrace recording that does not commit restores the
        // iterator cursor it eagerly advanced, so the interpreter resume
        // re-consumes the in-flight item exactly once (no drop).
        crate::jitcode_dispatch::fbw_bridge_iter_journal_rollback();
    }
    if authoritative {
        let mut end = match &walk_result {
            Ok((outcome, _)) => format!("{outcome:?}"),
            Err(error) => format!("{error:?}"),
        };
        if let Some(at) = end.find(|c: char| matches!(c, '(' | '{' | ' ')) {
            end.truncate(at);
        }
        let (unj_val, unj_sym) = crate::jitcode_dispatch::fbw_unjournaled_kinds();
        let (exec_v, exec_mf, exec_pl) = crate::jitcode_dispatch::fbw_executed_residual_counts();
        // `effects` is the gh#467 executed-effect odometer: residual calls that
        // were not provably side-effect free AND either wrote the live heap or
        // entered a Python frame.  A `committed=false` walk rolls its own store
        // journal back but cannot undo these, so `committed=false effects>0`
        // marks a walk whose caller is about to replay an irreversible region.
        let effects = crate::jitcode_dispatch::fbw_executed_effect_count();
        // The same record, into a static the wasm host reads back through the
        // `pyre_fbw_diag` export.  The guest cannot see `PYRE_FBW_CENSUS`, so
        // without this the wasm target has no walk-level observability at all.
        let leg = WALK_END_COMMIT_LEG.with(|c| c.get());
        fbw_diag::record(
            &end,
            committed,
            ctx.is_bridge_trace,
            effects,
            journal,
            exec_mf,
            leg,
        );
        if std::env::var_os("PYRE_FBW_CENSUS").is_some() {
            eprintln!(
                "[fbw-census] end={end} committed={committed} leg={leg} bridge={} \
                 unj_val={unj_val} unj_sym={unj_sym} exec_v={exec_v} exec_mf={exec_mf} \
                 exec_pl={exec_pl} effects={effects} journal={journal}",
                ctx.is_bridge_trace,
            );
        }
    }

    Some((entry, code_len, walk_result))
}

/// Read-only walker-as-tracer diagnostic probe, non-authoritative.
///
/// Runs the per-CodeObject full-body walk via [`run_perfn_walk`] in
/// non-authoritative mode, logs how far the symbolic walk got (terminator
/// outcome vs first `DispatchError` stop), then rolls the recorder back so
/// the diagnostic leaves no partial trace.  `PYRE_PROBE_AUTHORITATIVE=1`
/// opts into authoritative execution for diagnosis ONLY (verifying the
/// walk advances past the loop `goto_if_not`); it corrupts the live
/// frame/iterator state because the probe still discards the trace, so it
/// must never be set outside a throwaway run.
fn probe_walk_perfn_jitcode<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    sym: &mut Sym,
    w_code: *const (),
    start_pc: usize,
    cf_addr: usize,
) {
    let authoritative = std::env::var_os("PYRE_PROBE_AUTHORITATIVE").is_some();
    // Capture the trace position BEFORE the walk so `cut_trace` rolls back
    // every op the diagnostic recorded (the walk discards its trace).
    let pre_pos = ctx.get_trace_position();
    let Some((entry, code_len, walk_result)) =
        run_perfn_walk(ctx, sym, w_code, start_pc, cf_addr, authoritative)
    else {
        return;
    };
    match &walk_result {
        Ok((outcome, end_pc)) => eprintln!(
            "[walk-perfn] entry={entry} code_len={code_len} OK end_pc={end_pc} outcome={outcome:?}"
        ),
        Err(e) => {
            eprintln!("[walk-perfn] entry={entry} code_len={code_len} STOP err={e:?}");
        }
    }

    // Roll the recorder back so the aborted trace leaves no partial ops.
    ctx.cut_trace(pre_pos);
    // The probe discards its trace; clear the walk-local bool-box-truth map and
    // stashed Finish payload an authoritative probe walk may have recorded so
    // they cannot leak into the next walk (the production tracer clears these
    // at entry, but the probe never runs through that path).
    crate::jitcode_dispatch::bool_box_truth_reset();
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    crate::jitcode_dispatch::fbw_store_journal_reset();
}

/// True when a loop body in `w_code` contains an `abort_permanent` marker.
///
/// An `abort_permanent` inside a loop body (e.g. the `SWAP` an `a < b < c`
/// chained comparison lowers to, or any other unported in-loop opcode)
/// corrupts the authoritative full-body walk: the unsupported op breaks the
/// loop-input register seeding, so the walk mis-evaluates the loop guard,
/// exits the loop on the first pass, and concretely executes the post-loop
/// tail — double-running its side effects and leaving the frame positioned
/// past the loop.  The walk's reactive `abort_permanent` decline
/// never fires because the corrupted guard exits before reaching the
/// marker.  The scan is anchored at the merge point governing the loop being
/// traced, so a marker in a preceding sibling loop cannot decline a clean
/// following loop.  When the loop is not inside a `try`, the scan is scoped to
/// ops after that merge point and before the loop's final back-edge, so neither
/// a prologue-only marker (e.g. `COPY_FREE_VARS` ahead of a clean hot loop) nor
/// a post-loop marker over-declines the loop.  A loop covered by an exception
/// handler keeps the full-tail scan so a post-loop `abort_permanent` still
/// declines it, because compiled-loop delivery of an uncaught raise to the
/// handler is not yet supported.
///
/// The scan exempts one marker class: `LOAD_FAST_CHECK`'s null arm.  The
/// decline above is a refusal over an *unported* opcode, whose arm bails
/// without modelling the stack effect the walk then keeps interpreting.
/// `LOAD_FAST_CHECK` is ported: the codewriter splits it on `ptr_nonzero`,
/// compiles the bound arm normally, and sends the null arm to a dead-end block.
/// A walk that finds the local bound takes the bound arm and never sees the
/// marker; a walk that finds it unbound reaches the marker with the seeding
/// intact and declines reactively, the ordinary path.  Neither leaves the loop
/// guard mis-seeded, so the static refusal has nothing to prevent.
///
/// The widened `loop_in_try` tail is where this bites: putting the loop inside
/// a `try` is itself what makes the tail read a loop variable through
/// `LOAD_FAST_CHECK` rather than `LOAD_FAST` — the raise can reach the handler
/// before the loop assigns the slot, so the slot is only conditionally bound on
/// the join — and declining over that marker meant the widening's own predicate
/// manufactured the marker it tripped on.
///
/// The owning opcode comes from `abort_permanent_py_pc_by_jit_pc`, the exact
/// marker inverse, NOT from `py_floor_by_jit_pc`: the null arm's block is
/// emitted after the whole body, and the floor table keys each Python PC to its
/// FIRST jitcode offset, so it attributes that late block to whichever opcode
/// last opened a segment.
fn loop_body_abort_permanent_pc(w_code: *const (), start_pc: usize) -> Option<usize> {
    let Some(pjc) = crate::state::pyjitcode_for_code(w_code) else {
        return None;
    };
    let code = pjc.jitcode.code.as_slice();
    let Some(merge_point) = pjc
        .merge_entry_for(start_pc)
        .and_then(|entry| {
            crate::jitcode_runtime::decoded_ops(code)
                .filter(|op| op.opname == "jit_merge_point" && op.pc >= entry)
                .map(|op| op.pc)
                .min()
        })
        .or_else(|| {
            crate::jitcode_runtime::decoded_ops(code)
                .find(|op| op.opname == "jit_merge_point")
                .map(|op| op.pc)
        })
    else {
        return None;
    };

    let mut back_edge_end: Option<usize> = None;
    let mut abort_permanent_pcs: Vec<usize> = Vec::new();
    for op in crate::jitcode_runtime::decoded_ops(code).filter(|op| op.pc > merge_point) {
        if op.opname == "abort_permanent" {
            abort_permanent_pcs.push(op.pc);
        }
        if op.opname.starts_with("goto") && op.argcodes.ends_with('L') {
            let target = u16::from_le_bytes([code[op.next_pc - 2], code[op.next_pc - 1]]) as usize;
            if target <= merge_point {
                back_edge_end = Some(back_edge_end.map_or(op.pc, |end| end.max(op.pc)));
            }
        }
    }

    // `start_pc` is a code-unit index; the exception table lookup takes byte offsets (×2).
    let loop_in_try = unsafe {
        pyre_interpreter::pycode::w_code_lookup_exceptiontable(
            w_code as pyre_object::PyObjectRef,
            (start_pc as u32) * 2,
        )
    }
    .is_some();

    // `PYRE_FBW_LOOPBODY_SCAN_LOOP_ONLY` drops the `loop_in_try` widening
    // entirely, the counterpart of `PYRE_FBW_LOOPBODY_SCAN_FULL`, so the whole
    // carve-out stays measurable without a rebuild.
    let scan_end = if !std::env::var_os("PYRE_FBW_LOOPBODY_SCAN_LOOP_ONLY").is_some()
        && (loop_in_try || std::env::var_os("PYRE_FBW_LOOPBODY_SCAN_FULL").is_some())
    {
        code.len()
    } else {
        back_edge_end.unwrap_or(code.len())
    };
    abort_permanent_pcs
        .into_iter()
        .filter(|pc| *pc < scan_end)
        .find(|pc| !marker_is_load_fast_check_null_arm(w_code, &pjc, *pc))
}

/// The Python PC and decoded opcode that emitted the `abort_permanent` at
/// `marker_jit_pc`, via the exact `abort_permanent_py_pc_by_jit_pc` inverse.
///
/// `None` whenever the owning opcode cannot be named — a jitcode whose metadata
/// carries no marker inverse (skeleton / fixture), or a `marker_jit_pc` that is
/// not a marker at all.  Every caller treats that as "keep declining", the
/// conservative direction.
fn abort_permanent_owner(
    w_code: *const (),
    pjc: &crate::pyjitcode::PyJitCodePayload,
    marker_jit_pc: usize,
) -> Option<(usize, pyre_interpreter::Instruction)> {
    let table = &pjc.metadata.abort_permanent_py_pc_by_jit_pc;
    let idx = table
        .binary_search_by_key(&(marker_jit_pc as u32), |&(off, _)| off)
        .ok()?;
    let py_pc = table[idx].1 as usize;
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const CodeObject
    };
    if raw_code.is_null() {
        return None;
    }
    let (instr, _) = pyre_interpreter::decode_instruction_at(unsafe { &*raw_code }, py_pc)?;
    Some((py_pc, instr))
}

/// True when the `abort_permanent` at `marker_jit_pc` is the dead-end arm
/// `LOAD_FAST_CHECK` emits for a conditionally-bound slot (`codewriter.rs`
/// `Instruction::LoadFastCheck`, the `emit_abort_permanent!(py_pc,
/// closes_block)` leg).
fn marker_is_load_fast_check_null_arm(
    w_code: *const (),
    pjc: &crate::pyjitcode::PyJitCodePayload,
    marker_jit_pc: usize,
) -> bool {
    matches!(
        abort_permanent_owner(w_code, pjc, marker_jit_pc),
        Some((_, pyre_interpreter::Instruction::LoadFastCheck { .. }))
    )
}

/// `"py_pc=47 LoadFastCheck"` for the `PYRE_FBW_DEBUG_ABORT` line, or
/// `"owner=?"` when the inverse cannot name it.
///
/// A bare jitcode offset does not say which opcode declined the frame, and the
/// floor/block-head tables answer a different question (they key a *resume*
/// coordinate, and a marker in a block emitted after the body floors to
/// whichever PC last opened a segment).  Naming it here is what turns a
/// `loops_aborted` row into an opcode to port.
fn describe_abort_permanent_owner(w_code: *const (), marker_jit_pc: usize) -> String {
    let Some(pjc) = crate::state::pyjitcode_for_code(w_code) else {
        return "owner=?".to_string();
    };
    match abort_permanent_owner(w_code, &pjc, marker_jit_pc) {
        Some((py_pc, instr)) => format!("py_pc={py_pc} {instr:?}"),
        None => "owner=?".to_string(),
    }
}

struct CalleeAbortPermanentHit {
    callee_name: String,
    marker_jit_pc: usize,
}

fn collect_loop_body_referenced_roots(
    code: &CodeObject,
    start_pc: usize,
) -> Option<(
    std::collections::HashSet<String>,
    std::collections::HashSet<usize>,
)> {
    use pyre_interpreter::Instruction as I;

    let mut back_edge_pc: Option<usize> = None;
    let mut arg_state = pyre_interpreter::OpArgState::default();
    for (pc, unit) in code.instructions.iter().copied().enumerate() {
        let (instr, op_arg) = arg_state.get(unit);
        let delta = match instr {
            I::JumpBackward { delta } | I::JumpBackwardNoInterrupt { delta } => delta,
            _ => continue,
        };
        if pyre_interpreter::jump_target_backward_decoded(code, pc + 1, delta, op_arg) == start_pc {
            back_edge_pc = Some(back_edge_pc.map_or(pc, |old| old.max(pc)));
        }
    }
    let back_edge_pc = back_edge_pc?;

    let mut global_names = std::collections::HashSet::new();
    let mut local_slots = std::collections::HashSet::new();
    let mut scan_pc = |pc: usize| {
        let Some((instr, op_arg)) = pyre_interpreter::decode_instruction_at(code, pc) else {
            return;
        };
        match instr {
            I::LoadName { namei } => {
                let name_idx = namei.get(op_arg) as usize;
                if let Some(name) = code.names.get(name_idx) {
                    global_names.insert(name.to_string());
                }
            }
            I::LoadGlobal { namei } => {
                let name_idx = (namei.get(op_arg) as usize) >> 1;
                if let Some(name) = code.names.get(name_idx) {
                    global_names.insert(name.to_string());
                }
            }
            I::LoadFast { var_num }
            | I::LoadFastBorrow { var_num }
            | I::LoadFastCheck { var_num }
            | I::LoadFastAndClear { var_num } => {
                local_slots.insert(var_num.get(op_arg).as_usize());
            }
            I::LoadDeref { i } => {
                local_slots.insert(i.get(op_arg).as_usize());
            }
            I::LoadFastBorrowLoadFastBorrow { var_nums } | I::LoadFastLoadFast { var_nums } => {
                let pair = var_nums.get(op_arg);
                local_slots.insert(u32::from(pair.idx_1()) as usize);
                local_slots.insert(u32::from(pair.idx_2()) as usize);
            }
            I::StoreFastLoadFast { var_nums } => {
                let pair = var_nums.get(op_arg);
                local_slots.insert(u32::from(pair.idx_2()) as usize);
            }
            _ => {}
        }
    };

    for pc in start_pc..=back_edge_pc {
        scan_pc(pc);
    }

    let loop_start_byte = (start_pc as u32) * 2;
    let loop_end_byte = ((back_edge_pc + 1) as u32) * 2;
    for entry in pyre_interpreter::pycode::decode_exceptiontable(&code.exceptiontable) {
        if entry.start < loop_start_byte || entry.end > loop_end_byte {
            continue;
        }
        let handler_start = (entry.target / 2) as usize;
        for pc in handler_start..code.instructions.len() {
            scan_pc(pc);
            let Some((instr, _)) = pyre_interpreter::decode_instruction_at(code, pc) else {
                continue;
            };
            if matches!(
                instr,
                I::Reraise { .. }
                    | I::JumpForward { .. }
                    | I::JumpBackward { .. }
                    | I::JumpBackwardNoInterrupt { .. }
                    | I::ReturnValue
            ) {
                break;
            }
        }
    }

    Some((global_names, local_slots))
}

/// True when the hot loop body in `w_code` inline-calls — transitively — a
/// user function whose per-fn jitcode body carries an `abort_permanent`
/// marker.
///
/// [`loop_body_abort_permanent_pc`] only scans the top-level per-CodeObject
/// jitcode, so an `abort_permanent` reached through an inlined callee slips
/// past it.  That gap causes a walk-time double-apply: a non-journaled
/// concrete heap store (dict/attr/set item, list `extend`, …) in the loop
/// body executes concretely, then an inline-eligible user CALL later in the
/// same body is inline-attempted; the callee sub-walk hits `abort_permanent`
/// and routes the whole walk to abort; the epilogue rolls back the store
/// journal and REPLAYS FROM LOOP ENTRY, so the non-journaled store — which
/// the journal never captured — re-executes and the loop over-counts (e.g.
/// `300001` instead of `300000`).  Declining the walk up front, before it
/// executes anything, avoids the double-apply: the location re-interprets
/// without JIT (correct, at interpreter speed).
///
/// This is an OVER-DECLINING stopgap, and static: it mirrors the inline path's
/// static eligibility gates, but can still decline on a function referenced by
/// a loop-body `LOAD_GLOBAL`/local-slot load even when the executed path does
/// not actually call it.  Call-site-dependent gates such as the passed-argument
/// count and recursion depth cannot be resolved by this scan, so a callee that
/// would fail one of those gates can also over-decline.  A hot loop that calls
/// an otherwise inline-eligible helper whose body contains an unported op
/// (`match`, `async`, chained-compare `SWAP`, …) — even on a rarely taken path
/// — now runs interpreted in full, not just the aborting call.  The orthodox
/// mechanism has no up-front scan at all: an unsupported op raises
/// `SwitchToBlackhole` mid-trace and
/// `run_blackhole_interp_to_cancel_tracing` (pyjitpl.py:2949) converts the live
/// framestack and continues FORWARD in the blackhole interpreter, so nothing
/// replays and nothing double-applies.  This decline holds until that
/// forward-resume convergence (#126/#215) lets an inlined-callee abort resume
/// the outer walk in place instead of rolling back to loop entry.
///
/// The scan resolves candidate callees CONCRETELY from the live frame (the
/// walk has not run yet, so no store has executed).  Two root seed sources,
/// both scoped to identifiers the loop body or its in-loop handlers read:
/// - module globals named by loop-body `LOAD_GLOBAL` or `LOAD_NAME`;
/// - ROOT-frame fastlocals + closure cells whose slots are read by loop-body
///   local-load opcodes.
///
/// Each plain-function value first passes the inline path's static closure,
/// positional-parameter, jitcode-body, and Ref-register-capacity gates.  Its
/// per-fn jitcode body is then scanned end-to-end for `abort_permanent` (the
/// marker can sit at any pc, ahead of the callee's own merge point).
/// Non-aborting eligible callees are enqueued and their own referenced
/// functions scanned transitively through THEIR globals, guarded by a
/// scan-local visited set.  The root `w_code` is pre-marked visited — its own
/// loop-body marker is already handled by [`loop_body_abort_permanent_pc`].
///
/// Frame-local seeding is ROOT-frame only; a deeper (not-yet-pushed) callee's
/// locals are not available up front.  Callees reached via attribute access,
/// container elements, or another call's return value, and callees local to a
/// deeper frame, stay unresolvable before the walk — those rely on the
/// deferred #126/#215 forward-resume convergence rather than this stopgap.
fn loop_inlines_abort_permanent_callee(
    w_code: *const (),
    start_pc: usize,
    cf_addr: usize,
) -> Option<CalleeAbortPermanentHit> {
    // Gate: only scan when the top-level loop body (ops after the first
    // `jit_merge_point`) contains a `residual_call*` op.  Every inline-eligible
    // user call lowers to a residual_call, so a call-free loop cannot
    // inline-abort — skipping it avoids resolving globals for the common case.
    let Some(pjc) = crate::state::pyjitcode_for_code(w_code) else {
        return None;
    };
    let mut seen_merge_point = false;
    let mut has_residual_call = false;
    for op in crate::jitcode_runtime::decoded_ops(pjc.jitcode.code.as_slice()) {
        if op.opname == "jit_merge_point" {
            seen_merge_point = true;
        } else if seen_merge_point && op.opname.starts_with("residual_call") {
            has_residual_call = true;
            break;
        }
    }
    if !has_residual_call || cf_addr == 0 {
        return None;
    }

    // Process one concrete candidate value shared by both seed paths (globals
    // and frame slots): gate it to a plain user function, scan its whole
    // jitcode body for `abort_permanent`, and otherwise enqueue it for
    // transitive resolution through its own globals.  Returns `true` iff the
    // candidate's body carries the marker.
    //
    // SAFETY: `cand` is a live concrete `PyObjectRef` read from the frame or a
    // module dict before the walk mutates anything.
    unsafe fn consider_candidate(
        cand: pyre_object::PyObjectRef,
        function_type_addr: usize,
        visited: &mut std::collections::HashSet<*const ()>,
        queue: &mut std::collections::VecDeque<(*const (), pyre_object::PyObjectRef)>,
    ) -> Option<CalleeAbortPermanentHit> {
        // A tagged immediate int is never a FUNCTION_TYPE callee; skip it
        // before the `ob_type` deref below (which reads an even-aligned heap
        // pointer). Reaches here via the globals-dict scan and via a cell
        // whose contents are a tagged int.
        if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(cand) {
            return None;
        }
        // Only plain user functions inline (mirrors the inline path's exact
        // FUNCTION_TYPE gate); builtins carry no CodeObject.
        if cand.is_null() || (*cand).ob_type as *const () as usize != function_type_addr {
            return None;
        }
        let callee_w_code = pyre_interpreter::function_get_code(cand);
        if callee_w_code.is_null() {
            return None;
        }
        // A FUNCTION_TYPE object can wrap a BuiltinCode, not a CodeObject:
        // `make_builtin_function*` (gateway.rs:701) puts such a function into
        // module globals (e.g. `from sys import getsizeof`).  Feeding its
        // BuiltinCode to `sub_jitcode_body_for_code` / `w_code_get_ptr` casts it
        // as a PyCode and derefs garbage, so reject it before the scan — a
        // builtin carries no traceable body and never inlines.
        if pyre_interpreter::is_builtin_code(callee_w_code as pyre_object::PyObjectRef) {
            return None;
        }
        let Some((callee_w_code, nparams, has_closure)) =
            crate::jitcode_dispatch::resolve_inlinable_callee(cand)
        else {
            return None;
        };
        if has_closure || nparams == 0 {
            return None;
        }
        let Some(body) = crate::state::sub_jitcode_body_for_code(callee_w_code) else {
            return None;
        };
        if nparams > body.num_regs_r || !visited.insert(callee_w_code) {
            return None;
        }
        for op in crate::jitcode_runtime::decoded_ops(body.code) {
            if op.opname == "abort_permanent" {
                let raw_code =
                    pyre_interpreter::w_code_get_ptr(callee_w_code as pyre_object::PyObjectRef)
                        as *const CodeObject;
                let callee_name = if raw_code.is_null() {
                    "<unknown>".to_string()
                } else {
                    (*raw_code).obj_name.as_str().to_owned()
                };
                return Some(CalleeAbortPermanentHit {
                    callee_name,
                    marker_jit_pc: op.pc,
                });
            }
        }
        // Transitive: resolve this callee's own referenced functions in its own
        // globals.
        let callee_globals = pyre_interpreter::function_get_globals_obj(cand);
        if !callee_globals.is_null() {
            queue.push_back((callee_w_code, callee_globals));
        }
        None
    }

    // SAFETY: `cf_addr` is the live `PyFrame` pointer the portal passed to the
    // walk; its `w_globals` is the module dict and its locals/cells region is
    // initialised.  All callee resolution reads live concrete objects before
    // the walk mutates anything.
    unsafe {
        let cf = &*(cf_addr as *const pyre_interpreter::pyframe::PyFrame);
        let root_globals = cf.w_globals;
        if root_globals.is_null() {
            return None;
        }
        let raw_code = pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const CodeObject;
        if raw_code.is_null() {
            return None;
        }
        let Some((loop_body_global_names, loop_body_local_slots)) =
            collect_loop_body_referenced_roots(&*raw_code, start_pc)
        else {
            return None;
        };
        let function_type_addr = &pyre_interpreter::FUNCTION_TYPE as *const _ as usize;
        let mut visited: std::collections::HashSet<*const ()> = std::collections::HashSet::new();
        // The root's own loop-body `abort_permanent` is handled upstream.
        visited.insert(w_code);
        // BFS over (code wrapper ptr, globals in which its `co_names` resolve).
        let mut queue: std::collections::VecDeque<(*const (), pyre_object::PyObjectRef)> =
            std::collections::VecDeque::new();

        for name in &loop_body_global_names {
            let Some(cand) =
                pyre_object::dictmultiobject::w_dict_getitem_str(root_globals, name.as_str())
            else {
                continue;
            };
            if let Some(hit) =
                consider_candidate(cand, function_type_addr, &mut visited, &mut queue)
            {
                return Some(hit);
            }
        }

        // Seed from the root frame's fastlocals + closure cells: a helper
        // passed as an argument/local or held in a cell is not in `co_names`, so
        // resolve it directly from the loop-body-referenced slots in the
        // frame's initialised locals/cells region.  Stop at `stack_base()` —
        // operand-stack slots beyond it are uninitialised.
        let slots = cf.locals_w().as_slice();
        let bound = cf.stack_base().min(slots.len());
        for (slot_idx, &slot) in slots[..bound].iter().enumerate() {
            if !loop_body_local_slots.contains(&slot_idx) {
                continue;
            }
            if slot.is_null() {
                continue;
            }
            // A tagged immediate int is neither a cell nor a FUNCTION_TYPE
            // callee; skip it before `is_cell(slot)` derefs its `ob_type`.
            if pyre_object::tagged_int::CAN_BE_TAGGED
                && pyre_object::tagged_int::is_tagged_int(slot)
            {
                continue;
            }
            // A closure cell holds the function indirectly; unwrap it.
            let value = if pyre_object::is_cell(slot) {
                pyre_object::w_cell_get(slot)
            } else {
                slot
            };
            if let Some(hit) =
                consider_candidate(value, function_type_addr, &mut visited, &mut queue)
            {
                return Some(hit);
            }
        }

        while let Some((code_ptr, globals)) = queue.pop_front() {
            let raw = pyre_interpreter::w_code_get_ptr(code_ptr as pyre_object::PyObjectRef)
                as *const CodeObject;
            if raw.is_null() {
                continue;
            }
            for name in (*raw).names.iter() {
                let Some(cand) =
                    pyre_object::dictmultiobject::w_dict_getitem_str(globals, name.as_str())
                else {
                    continue;
                };
                if let Some(hit) =
                    consider_candidate(cand, function_type_addr, &mut visited, &mut queue)
                {
                    return Some(hit);
                }
            }
        }
    }
    None
}

/// True when the loop header is about to resume a generator whose body uses
/// SEND delegation.
///
/// The full-body walker executes iterator advances while recording.  A SEND
/// marker aborts that walk, but the delegated iterator is shared with the live
/// generator frame, so replaying from the loop header would observe the
/// already-advanced delegate and drop values.  The upstream blackhole path
/// continues forward from the live generator frame.  Until that forward
/// resume is available here, decline before the first shared-iterator advance.
fn loop_iterates_send_generator(cf_addr: usize, start_pc: usize) -> bool {
    if cf_addr == 0 {
        return false;
    }
    unsafe {
        let frame = &*(cf_addr as *const pyre_interpreter::pyframe::PyFrame);
        let code = frame.code();
        if !matches!(
            pyre_interpreter::decode_instruction_at(code, start_pc),
            Some((pyre_interpreter::Instruction::ForIter { .. }, _))
        ) || frame.valuestackdepth <= frame.stack_base()
        {
            return false;
        }

        let iterator = frame.peek();
        if !pyre_object::generator::is_generator_or_coroutine(iterator) {
            return false;
        }
        let generator_frame = pyre_object::generator::w_generator_get_frame(iterator)
            as *const pyre_interpreter::pyframe::PyFrame;
        if generator_frame.is_null() {
            return false;
        }
        let generator_code = (*generator_frame).code();
        (0..generator_code.instructions.len()).any(|pc| {
            matches!(
                pyre_interpreter::decode_instruction_at(generator_code, pc),
                Some((pyre_interpreter::Instruction::Send { .. }, _))
            )
        })
    }
}

/// Whether [`full_body_walk_trace`] starts a fresh walk or continues one that
/// has already applied eager stores.
enum WalkJournals {
    /// Fresh walk: drop the journals + unjournaled-effect flag a prior aborted
    /// walk left behind, so this walk's commit cannot apply them.
    Reset,
    /// Continuation of a drain sub-walk that already concrete-executed a
    /// reconstructed callee.  Keep its journals so THIS walk's epilogue settles
    /// them — commit keeps the stores, non-commit rolls them back for the
    /// blackhole replay.  Dropping them would leave every eager store standing
    /// while the blackhole replays the callee from the guard, applying it a
    /// second time.  The unjournaled-effect flag carries over for the same
    /// reason: a sub-walk that only recorded a residual must still block the
    /// root's commit.
    Keep,
}

/// Production full-body tracer.
///
/// Drives the per-CodeObject JitCode body via [`run_perfn_walk`] in
/// authoritative mode AS the production trace — the walk IS the concrete
/// execution, so unlike the probe it keeps the recorded trace.  Maps the
/// walk outcome to a [`TraceAction`] for the caller to compile.
///
/// Conservative mapping: only `CloseLoop` — the validated
/// end-to-end case (the four loop benches close under authoritative) — is
/// mapped to a real `CloseLoopWithArgs`; every other outcome (`Terminate`
/// finish-arg recovery, `SubReturn`/`SubRaise`, `SwitchToBlackhole`, any
/// `DispatchError`) aborts the trace and returns to interpretation.
fn full_body_walk_trace<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    sym: &mut Sym,
    w_code: *const (),
    start_pc: usize,
    cf_addr: usize,
    journals: WalkJournals,
) -> TraceAction {
    // #125: decline up front when a loop body carries an `abort_permanent`
    // marker.  The authoritative walk would otherwise mis-seed the loop
    // guard, exit early, and concretely double-execute the post-loop tail;
    // declining before the walk reaches the unported op avoids frame
    // corruption and returns to interpretation.
    if let Some(abort_pc) = loop_body_abort_permanent_pc(w_code, start_pc) {
        // Tag the decline so `PYRE_FBW_DEBUG_ABORT` census attributes it to the
        // up-front `abort_permanent` scan, not the trait retry fall-through
        // (`Trait::DeclinedAbort`).  Without this the real declining class is
        // invisible to the census.
        crate::jitcode_dispatch::census_record("FullBodyWalk::LoopBodyAbortPermanent");
        if std::env::var_os("PYRE_FBW_DEBUG_ABORT").is_some() {
            let owner = describe_abort_permanent_owner(w_code, abort_pc);
            eprintln!("[fbw-abort] start_pc={start_pc} abort_permanent_pc={abort_pc} {owner}");
        }
        fbw_decline(crate::driver::make_green_key(w_code, start_pc));
        return TraceAction::Decline;
    }
    if loop_iterates_send_generator(cf_addr, start_pc) {
        crate::jitcode_dispatch::census_record("FullBodyWalk::SendGeneratorIterator");
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-abort] start_pc={start_pc} SEND generator iterator; declining before delegation"
            );
        }
        fbw_decline(crate::driver::make_green_key(w_code, start_pc));
        return TraceAction::Decline;
    }
    // Sibling defense to the above, transitively through inlined callees: a
    // non-journaled concrete store in the loop body followed by an
    // inline-eligible CALL whose callee body carries `abort_permanent` would
    // abort the walk, roll back the store journal, and replay from loop entry
    // — re-executing the non-journaled store.  Decline up front, before the
    // walk runs anything.  (See `loop_inlines_abort_permanent_callee`.)
    if let Some(hit) = loop_inlines_abort_permanent_callee(w_code, start_pc, cf_addr) {
        crate::jitcode_dispatch::census_record("FullBodyWalk::CalleeAbortPermanent");
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-abort] start_pc={start_pc} callee={} abort_permanent_jit_pc={}; \
                 declining callee-abort walk",
                hit.callee_name, hit.marker_jit_pc
            );
        }
        fbw_decline(crate::driver::make_green_key(w_code, start_pc));
        return TraceAction::Decline;
    }
    // Register the initial merge point with typed input-arg boxes so the trace head
    // carries the portal's entry signature (`inputarg_types()`).  Without
    // it the compiled loop's entry args don't match what the portal
    // supplies, so the portal cannot enter the compiled loop and re-traces
    // every iteration (the observed spin).
    // Clear the walk-local bool-box-truth map left by a prior aborted walk so
    // it cannot leak into this one.
    crate::jitcode_dispatch::bool_box_truth_reset();
    // Slice b: clear any Finish payload a prior
    // aborted walk's top-level `*_return` arm may have stashed, so a stale
    // value cannot leak into this walk's `Terminate` handling.
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    crate::jitcode_dispatch::reset_single_frame_blackhole();
    crate::jitcode_dispatch::fbw_executed_nonpure_residual_reset();
    crate::jitcode_dispatch::fbw_executed_body_residual_reset();
    crate::jitcode_dispatch::fbw_abort_outer_resume_reset();
    // Clear the prior walk's store journal + unjournaled-effect flag so
    // dropped (aborted) entries cannot be applied by this walk's commit.
    // A continuation keeps them instead: see [`WalkJournals`].
    if matches!(journals, WalkJournals::Reset) {
        crate::jitcode_dispatch::fbw_store_journal_reset();
    }
    // A bridge resumes mid-loop from a guard failure; its input args are the
    // guard's resumedata, already seeded into the bridge sym by
    // `setup_bridge_sym`.  PyPy's `interpret()` (rebuild_state_after_failure →
    // continue) registers NO merge point at the resume pc: the bridge walks
    // forward until it reaches an existing compiled loop header and closes as
    // a bridge there.  Registering a merge point at `start_pc` would instead
    // treat the resume pc as a fresh loop header (the portal entry signature),
    // which only a MAIN trace should do.  So skip it for bridges.
    if !ctx.is_bridge_trace {
        let start_key = crate::driver::make_green_key(w_code, start_pc);
        let input_types = ctx.inputarg_types();
        let input_args: Vec<majit_metainterp::GreenBox> = input_types
            .iter()
            .enumerate()
            .map(|(i, &tp)| {
                majit_metainterp::GreenBox::new(majit_ir::OpRef::input_arg_typed(i as u32, tp), tp)
            })
            .collect();
        ctx.add_merge_point(start_key, input_args, start_pc);
    }
    let walk_result = run_perfn_walk(ctx, sym, w_code, start_pc, cf_addr, true);
    // A guard snapshot emitted during the walk may have hit a resume
    // coordinate the jitcode resume markers cannot encode (#124/#130) and requested
    // an abort (`state::request_trace_abort`).  The walker does not poll the
    // flag mid-walk, so honor it here before mapping the outcome — otherwise a
    // walk that reaches a terminator would compile a trace carrying the bad
    // guard. Discard the trace before mapping the terminal outcome.
    if crate::state::take_trace_abort_requested() {
        crate::jitcode_dispatch::census_record("TraceAbortRequested");
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-abort] start_pc={start_pc} unencodable cross-frame resume coordinate (#124/#130)"
            );
        }
        return TraceAction::Abort;
    }
    if ctx.is_bridge_trace && crate::jitcode_dispatch::p2_diag_enabled() {
        ctx.dump_trace_ops_diag("carrier-root-walk-end");
    }
    let action = match walk_result {
        Some((_entry, _code_len, Ok((outcome, _end_pc)))) => match outcome {
            crate::jitcode_dispatch::DispatchOutcome::CloseLoop {
                jump_args,
                loop_header_pc,
                ..
            } => {
                // Mirror trace_bytecode's post-interpret CloseLoop green-key
                // handling: a loop header other than start_pc retargets the
                // green key to the true merge point (cut-to-inner-loop);
                // start_pc closes at the trace head.
                if loop_header_pc != start_pc {
                    let target_key = crate::driver::make_green_key(w_code, loop_header_pc);
                    ctx.set_green_key(target_key, (w_code as usize, loop_header_pc));
                    ctx.header_pc = loop_header_pc;
                    ctx.cut_inner_green_key = Some(target_key);
                } else {
                    let key = crate::driver::make_green_key(w_code, start_pc);
                    ctx.set_green_key(key, (w_code as usize, start_pc));
                    ctx.header_pc = start_pc;
                }
                TraceAction::CloseLoopWithArgs {
                    jump_args,
                    loop_header_pc: Some(loop_header_pc),
                }
            }
            crate::jitcode_dispatch::DispatchOutcome::Terminate => {
                // A loop-free portal exit: the top-level `*_return` reached
                // `done_with_this_frame` with no back-edge.  The return arm
                // routed through `fbw_terminate_with_finish`, which re-boxed the
                // result to Type::Ref, recorded the vable store-back +
                // GUARD_NOT_FORCED_2, and stashed the finish payload.  Build the
                // portal-exit FINISH from it so the compile pipeline records
                // FINISH from `finish_args` (matching `StepResult::Return` in
                // trace_opcode.rs).  No payload → `Abort`.
                let finish_is_exception = crate::jitcode_dispatch::fbw_finish_is_exception();
                match crate::jitcode_dispatch::fbw_finish_payload_take() {
                    // A top-level `void_return/` stashes a `Type::Void`-marked
                    // payload: the portal exits with no value, so build a
                    // FINISH with empty args.  The compile pipeline maps an
                    // empty `finish_arg_types` to `done_with_this_frame_descr_void`
                    // (pyjitpl.rs `done_with_this_frame_descr_from_types`).
                    Some((_, majit_ir::Type::Void)) => TraceAction::Finish {
                        finish_args: vec![],
                        finish_arg_types: vec![],
                        exit_with_exception: false,
                        exc_value: 0,
                    },
                    // A top-level uncaught raise stashes the exception box as an
                    // `is_exception` payload (`fbw_terminate_with_raise`): build
                    // the portal-exit FINISH against
                    // `exit_frame_with_exception_descr` (mirror of the trait
                    // tracer's `compile_exit_frame_with_exception`,
                    // pyjitpl.py:3238-3242) so the frame exits carrying the
                    // exception to the caller instead of aborting the bridge.
                    Some((exc, _)) if finish_is_exception => TraceAction::Finish {
                        finish_args: vec![exc],
                        finish_arg_types: vec![majit_ir::Type::Ref],
                        exit_with_exception: true,
                        // The full-body walk delivers its own uncaught raise
                        // through `WALK_END_PROPAGATED_EXCEPTION` (:557), so the
                        // `raise` half of `finishframe_exception` is already
                        // covered and no value needs to ride the action.
                        exc_value: 0,
                    },
                    Some((finish_value, finish_type)) => TraceAction::Finish {
                        finish_args: vec![finish_value],
                        finish_arg_types: vec![finish_type],
                        exit_with_exception: false,
                        exc_value: 0,
                    },
                    None => {
                        crate::jitcode_dispatch::census_record("Terminate::NoFinishPayload");
                        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-abort] start_pc={start_pc} Terminate without finish payload (ungated portal exit)"
                            );
                        }
                        TraceAction::Abort
                    }
                }
            }
            crate::jitcode_dispatch::DispatchOutcome::CompileTracePending { .. } => {
                // pyjitpl.py:3095 raise_if_successful parity: the walker's
                // in-walk `compile_trace` already compiled+installed the
                // trace as a (entry) bridge jumping into an existing loop;
                // hand the dedicated action back so the driver neither
                // compiles nor aborts this session again.
                TraceAction::CompileTrace
            }
            other => {
                crate::jitcode_dispatch::census_record("Outcome::Other");
                if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                    eprintln!("[fbw-abort] start_pc={start_pc} outcome={other:?}");
                }
                TraceAction::Abort
            }
        },
        Some((_entry, _code_len, Err(e))) => {
            // Record the key in `FBW_DECLINED_KEYS` only for errors that
            // deterministically re-reach a structural decline of the same
            // entry.  Transient walker capability gaps retain the plain
            // `Abort` without declining so a capability that lands mid-run can
            // still pick the location up.
            use crate::jitcode_dispatch::DispatchError as DE;
            crate::jitcode_dispatch::census_record(e.variant_name());
            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                eprintln!("[fbw-abort] start_pc={start_pc} Err={e:?}");
            }
            match e {
                // A kept-stack branch guard whose not-taken arm reads an
                // unrestorable boxed Ref register is a structural abort.
                // Keeping the permanent mapping is behavior-neutral: a plain
                // `Abort` reaches the same `DONT_TRACE_HERE` terminal state
                // through pyre's abort ceiling.
                DE::BranchGuardUnrestorableKeptStackPermanent { .. } => TraceAction::AbortPermanent,
                // #57 (Finding #1): a non-journalable in-place container mutation
                // in a FOR_ITER body cannot be rolled back on abort, so this
                // location can never trace soundly — interpret it permanently
                // (the loop runs correctly under the interpreter).
                DE::InplaceContainerMutationUnsupported { .. } => TraceAction::AbortPermanent,
                DE::AbortPermanentMarkerReached { .. } => TraceAction::AbortPermanent,
                DE::GuardSnapshotVableUntyped { .. }
                | DE::MayForceNullRefArgUnsupported { .. }
                | DE::BranchGuardKeptStackUnsupported { .. }
                | DE::LoopBearingCalleeInlineUnsupported { .. }
                | DE::UnfoldableListAppendResidualUnsupported { .. }
                // Plain, retryable: `ABORT_FORCE_QUASIIMMUT` abandons THIS
                // attempt because the version was live when the walk met the
                // write.  The forcing already ran, so the next attempt finds a
                // stabilised namespace and traces — the reason PyPy reports
                // thousands of these and still compiles the loop.  Retiring the
                // location would be strictly wrong.
                | DE::ForceQuasiImmutable { .. }
                | DE::ResidualCallArgUnbound { .. } => TraceAction::Abort,
                // #68 multiframe: a data-dependent
                // `goto_if_not` whose branch input is not concrete at trace-time
                // recurs identically on every retrace of this entry (the same
                // jitcode walked from the same start_pc reaches the same
                // non-concrete branch operand).  Relaxing the inline predicate
                // lets a portal trace (e.g. a callee independently traced as its
                // own origin) walk PAST its prior `LoopBearing` decline and reach
                // such a branch, which would otherwise re-trace unbounded (each
                // re-walk executes the body's residual calls before failing) —
                // an unbounded slowdown. Decline it so the location interprets
                // instead.
                // A bridge entry is keyed on the guard descr, which the
                // green-key cell never gates, so the green-key decline alone
                // leaves `must_compile_with_values` re-firing this
                // structurally-undecidable bridge every
                // `DEFAULT_TRACE_EAGERNESS` failures forever — each retry
                // re-walking the whole body and executing its residual calls
                // concretely. Record the bridge-guard decline too, the way
                // `ExcEdgeNoInFrameCatch` below does.
                DE::GotoIfNotValueNotConcrete { .. } => {
                    fbw_decline(crate::driver::make_green_key(w_code, start_pc));
                    fbw_bridge_decline(ctx);
                    TraceAction::Abort
                }
                // The exc-edge routing decision is `find_catch_for_exc_resume`
                // over `(jitcode_code, position)` alone, so the same guard
                // reaches it on every retrace — the premise the `AbortPermanent`
                // decline below is written for.  It cannot take that mapping:
                // the abort is raised before any recording precisely so the
                // guard resumes via the blackhole, which is the correct
                // handling, not a location to retire.  Record only the
                // bridge-guard decline, so the guard stops re-walking the whole
                // body (executing its residual calls concretely) to re-derive a
                // static answer.
                DE::ExcEdgeNoInFrameCatch { .. } => {
                    fbw_bridge_decline(ctx);
                    TraceAction::Abort
                }
                _ => TraceAction::Abort,
            }
        }
        None => {
            crate::jitcode_dispatch::census_record("RunPerfnWalkNone");
            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                eprintln!("[fbw-abort] start_pc={start_pc} run_perfn_walk returned None");
            }
            if fbw_declined(crate::driver::make_green_key(w_code, start_pc))
                || (ctx.is_bridge_trace && FBW_BRIDGE_DECLINED.with(|c| c.get()))
            {
                TraceAction::Decline
            } else {
                TraceAction::Abort
            }
        }
    };
    // A permanent abort is a property of the walked jitcode body, not of the
    // guard's runtime values: the same `(jitcode, resume_pc)` reaches the same
    // marker on every retrace.  The loop-header entry stops re-entering because
    // `abort_trace(true)` flips its cell to `DONT_TRACE_HERE`, but a bridge
    // entry is keyed on the guard descr, which that cell never gates — so
    // without this the guard keeps re-firing `must_compile` every
    // `trace_eagerness` failures and each retry walks the whole body, executing
    // its residual calls concretely, before failing again.  Record the decline
    // through the same channel the pre-walk structural declines use.
    if matches!(action, TraceAction::AbortPermanent) {
        fbw_bridge_decline(ctx);
    }
    action
}

/// Walker-as-tracer diagnostic dump.
///
/// Dumps the per-CodeObject JitCode body the walker-as-tracer walks for
/// `miframe.pc == jitcode_pc`.  The per-CodeObject JitCode is built BEFORE
/// this point by `register_portal_jitdriver` → `make_jitcodes`
/// (`pyre/pyre-jit/src/eval.rs`), so it is available here.
///
/// Read-only: logs the body op stream + entry offset, mutates nothing.
fn dump_perfn_jitcode_for_trace(w_code: *const (), start_pc: usize) {
    let Some(pjc) = crate::state::pyjitcode_for_code(w_code) else {
        eprintln!("[perfn-jitcode] no per-CodeObject PyJitCode for code={w_code:?}");
        return;
    };
    let code = pjc.jitcode.code.as_slice();
    let entry = pjc.merge_entry_for(start_pc);
    eprintln!(
        "[perfn-jitcode] code_len={} n_py_instrs={} start_pc={} entry_jitcode_pc={:?} \
         num_regs_r={} num_regs_i={} num_regs_f={} portal_frame_reg={} portal_ec_reg={} \
         built_as_portal={}",
        code.len(),
        pjc.metadata.n_py_instrs as usize,
        start_pc,
        entry,
        pjc.jitcode.num_regs_r(),
        pjc.jitcode.num_regs_i(),
        pjc.jitcode.num_regs_f(),
        pjc.metadata.portal_frame_reg,
        pjc.metadata.portal_ec_reg,
        pjc.metadata.built_as_portal,
    );
    let cap = std::env::var("PYRE_DUMP_PERFN_JITCODE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 1)
        .unwrap_or(80);
    let mut count = 0usize;
    let mut last_next = 0usize;
    let mut histogram: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for op in crate::jitcode_runtime::decoded_ops(code) {
        if count < cap {
            eprintln!(
                "[perfn-jitcode]   pc={:>4} next={:>4} {}/{} bytes={:?}",
                op.pc,
                op.next_pc,
                op.opname,
                op.argcodes,
                &code[op.pc + 1..op.next_pc.min(code.len())]
            );
        }
        *histogram.entry(op.key.to_string()).or_default() += 1;
        count += 1;
        last_next = op.next_pc;
    }
    let clean = last_next == code.len();
    eprintln!("[perfn-jitcode] TOTAL ops={count} last_next={last_next} clean_eof={clean}");
    for (key, n) in &histogram {
        eprintln!("[perfn-jitcode] HIST {n:>4} {key}");
    }
    if !clean && last_next < code.len() {
        let stop_byte = code[last_next];
        eprintln!(
            "[perfn-jitcode] STOP at pc={last_next}: byte=0x{stop_byte:02x} opname={:?}",
            crate::jitcode_runtime::opname_for_byte(stop_byte),
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::pyjitpl::semantic_fallthrough_pc;
    use pyre_interpreter::bytecode::Instruction;
    use pyre_interpreter::compile_exec;
    use pyre_interpreter::decode_instruction_at;

    #[test]
    fn complete_image_walk_abort_keeps_blackhole_terminal_result() {
        use crate::jitcode_dispatch::DispatchError;

        assert!(super::blackhole_terminal_error(
            &DispatchError::MayForceNullRefArgUnsupported { pc: 17 }
        ));
        assert!(super::blackhole_terminal_error(
            &DispatchError::VableEscapedDuringResidualCall { pc: 17 }
        ));
        assert!(super::blackhole_terminal_error(
            &DispatchError::TraceTooLong { pc: 17, ops: 1 }
        ));
        assert!(!super::blackhole_terminal_error(
            &DispatchError::RegisterReadUnbound {
                pc: 17,
                reg: 3,
                bank: "r",
            }
        ));
    }

    #[test]
    fn pending_call_result_uses_post_call_stack_top_not_register_color() {
        // The motivating coalesced shape has result color 0, nlocals=1, and
        // post-call operand depth 4.  The result belongs to semantic slot 4;
        // writing color 0 as slot 0 would overwrite the caller's local.
        assert_eq!(super::pending_call_result_semantic_slot(1, 4), Some(4));
        assert_eq!(super::pending_call_result_semantic_slot(1, 0), None);
    }

    #[test]
    fn walk_end_root_area_forwards_a_quiesced_foreign_mutator() {
        let (area_tx, area_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let (roots_tx, roots_rx) = std::sync::mpsc::channel();
        let owner = std::thread::spawn(move || {
            let mut err = pyre_interpreter::PyError::new(
                pyre_interpreter::PyErrorKind::RuntimeError,
                "foreign mutator root",
            );
            err.w_name_context = 0x3000 as pyre_object::PyObjectRef;
            err.w_obj_context = 0x4000 as pyre_object::PyObjectRef;
            super::WALK_END_PROPAGATED_EXCEPTION.with(|slot| {
                *slot.borrow_mut() = Some(err);
            });
            area_tx
                .send(super::capture_walk_end_root_area() as usize)
                .unwrap();
            resume_rx.recv().unwrap();
            super::WALK_END_PROPAGATED_EXCEPTION.with(|slot| {
                let err = slot.borrow_mut().take().unwrap();
                roots_tx
                    .send((err.w_name_context as usize, err.w_obj_context as usize))
                    .unwrap();
            });
        });

        let area = area_rx.recv().unwrap() as *const ();
        // The owner is blocked after publishing its stable TLS addresses,
        // matching the mutator quiescence required by the STW registry.
        unsafe {
            super::walk_walk_end_roots_area(area, &mut |root| {
                *root = majit_ir::GcRef(root.as_usize() + 0x20);
            });
        }
        resume_tx.send(()).unwrap();
        assert_eq!(roots_rx.recv().unwrap(), (0x3020, 0x4020));
        owner.join().unwrap();
    }

    /// The walk-end commit contract: only a resume pc that does not precede
    /// something the walk already applied may keep the store journal.
    #[test]
    fn walk_end_commit_refuses_an_unproven_or_stale_rewind() {
        use super::{WalkEndResume, walk_end_resume_provable};
        let live = crate::jitcode_dispatch::fbw_executed_effect_count();

        assert!(walk_end_resume_provable(WalkEndResume::Terminal));
        assert!(walk_end_resume_provable(WalkEndResume::AfterApplied));
        assert!(walk_end_resume_provable(WalkEndResume::Rewind {
            effects_at_resume_point: live,
        }));
        assert!(
            !walk_end_resume_provable(WalkEndResume::Rewind {
                effects_at_resume_point: live.wrapping_sub(1),
            }),
            "an odometer delta since the resume point means the region re-runs its effects",
        );
        assert!(
            !walk_end_resume_provable(WalkEndResume::RewindUnproven),
            "a rewind with no resume-point sample cannot be proven and must decline",
        );
    }

    #[test]
    fn static_marker_entry_recovers_ref_green_register_color() {
        let marker = *crate::jitcode_runtime::insns_opname_to_byte()
            .get("jit_merge_point/cIRFIRF")
            .expect("jit_merge_point must be registered");
        let code = [
            marker, 0, // jdindex
            2, 7, 8, // gi: next_instr, is_being_profiled
            1, 9, // gr: pycode
            0, // gf
            0, // ri
            2, 10, 11, // rr: frame, ec
            0,  // rf
        ];

        assert_eq!(
            super::static_entry_merge_point_green_ref_regs(&code, 0),
            Some(vec![9])
        );
        assert_eq!(
            super::static_entry_merge_point_green_ref_regs(&code, code.len()),
            None,
            "a sidecar after the marker must not bind an earlier sibling marker"
        );
    }

    #[test]
    fn loop_sidecar_seeds_ref_colors_from_live_frame_slots() {
        let slot0 = majit_ir::OpRef::input_arg_typed(3, majit_ir::Type::Ref);
        let slot2 = majit_ir::OpRef::input_arg_typed(4, majit_ir::Type::Ref);
        let none = majit_ir::OpRef::NONE;
        let frame_boxes = [slot0, none, slot2];
        let pcdep = [
            (1, 7, 0),
            (0, 8, 1), // Int bank is not a Ref seed.
            (1, 9, 1), // An absent frame slot stays absent.
            (1, 4, 2),
        ];
        let mut seeded = Vec::new();

        super::seed_loop_entry_ref_slots(
            &pcdep,
            crate::virtualizable_gen::NUM_VABLE_SCALARS,
            |index| {
                index
                    .checked_sub(crate::virtualizable_gen::NUM_VABLE_SCALARS)
                    .and_then(|slot| frame_boxes.get(slot).copied())
            },
            |color, _slot, opref| seeded.push((color, opref)),
        );

        assert_eq!(seeded, vec![(7, slot0), (4, slot2)]);
    }

    #[test]
    fn test_semantic_fallthrough_pc_skips_branch_trivia() {
        let mut source = String::from("def f(x, y):\n    if x < y:\n");
        for i in 0..400 {
            source.push_str(&format!("        z{i} = {i}\n"));
        }
        source.push_str("    return 0\n");
        source.push_str("f(1, 2)\n");

        let module = compile_exec(&source).expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                pyre_interpreter::ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let branch_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::PopJumpIfFalse { .. }, _))
                )
            })
            .expect("test bytecode should contain POP_JUMP_IF_FALSE");

        let fallthrough_pc = semantic_fallthrough_pc(&code, branch_pc);
        let fallthrough_instruction = decode_instruction_at(&code, fallthrough_pc)
            .map(|(instruction, _)| instruction)
            .expect("semantic fallthrough should decode");

        assert!(
            !matches!(
                fallthrough_instruction,
                Instruction::ExtendedArg
                    | Instruction::Resume { .. }
                    | Instruction::Nop
                    | Instruction::Cache
                    | Instruction::NotTaken
            ),
            "semantic fallthrough must skip bytecode trivia"
        );
    }

    fn named_function_code(source: &str, name: &str) -> pyre_interpreter::CodeObject {
        fn find_in(
            code: &pyre_interpreter::CodeObject,
            name: &str,
        ) -> Option<pyre_interpreter::CodeObject> {
            for constant in code.constants.iter() {
                if let pyre_interpreter::ConstantData::Code { code: inner } = constant {
                    if inner.obj_name.as_str() == name {
                        return Some((**inner).clone());
                    }
                    if let Some(found) = find_in(inner, name) {
                        return Some(found);
                    }
                }
            }
            None
        }
        let module = compile_exec(source).expect("test code should compile");
        find_in(&module, name)
            .unwrap_or_else(|| panic!("test source should contain function {name}"))
    }

    #[test]
    fn start_pc_is_loop_header_detects_while_target() {
        let src = "def f(a, b):\n    if a <= 0:\n        total = 0\n        i = 0\n        while i < b:\n            total = total + i\n            i = i + 1\n        return total\n    return f(a - 1, b)\n";
        let code = named_function_code(src, "f");

        // Locate the loop header (the JumpBackward target).
        let mut arg_state = pyre_interpreter::OpArgState::default();
        let mut loop_header: Option<usize> = None;
        for (pc, unit) in code.instructions.iter().copied().enumerate() {
            let (instr, op_arg) = arg_state.get(unit);
            if let pyre_interpreter::Instruction::JumpBackward { delta }
            | pyre_interpreter::Instruction::JumpBackwardNoInterrupt { delta } = instr
            {
                loop_header = Some(pyre_interpreter::jump_target_backward_decoded(
                    &code,
                    pc + 1,
                    delta,
                    op_arg,
                ));
                break;
            }
        }
        let loop_header = loop_header.expect("the while loop must emit a JumpBackward");
        assert!(
            super::start_pc_is_loop_header(&code, loop_header),
            "the JumpBackward target must be recognized as a loop header"
        );
        assert!(
            !super::start_pc_is_loop_header(&code, 0),
            "function entry must not be recognized as the loop header"
        );
    }

    #[test]
    fn forward_exception_delivery_needs_a_handler_the_call_operands_can_fill() {
        // Statement position: nothing below the call, empty-stack handler.
        assert!(super::exception_delivery_stack_is_sourceable(0, 0, 8, 7));
        assert!(!super::exception_delivery_stack_is_sourceable(1, 0, 9, 7));
        // Expression position: one operand below, handler wanting 0 or 1.
        assert!(super::exception_delivery_stack_is_sourceable(1, 1, 9, 7));
        assert!(super::exception_delivery_stack_is_sourceable(0, 1, 9, 7));
        assert!(!super::exception_delivery_stack_is_sourceable(2, 1, 9, 7));
        // The array must hold the restored operands plus the pushed exception.
        assert!(!super::exception_delivery_stack_is_sourceable(0, 0, 7, 7));
        assert!(!super::exception_delivery_stack_is_sourceable(1, 1, 8, 7));
    }
}

// ── Guest-visible walk-outcome tally ─────────────────────────────
//
// `wasm32-unknown-unknown` cannot read the environment, so `PYRE_FBW_CENSUS`
// and `PYRE_FBW_DEBUG_ABORT` — the two channels that make a walk-replay RCA
// tractable on the native backends — are inert inside the guest.  Mirror the
// shape `majit_backend_wasm::BRIDGE_DIAG` and `majit_metainterp::MC_DIAG`
// already use for this: a static tally the host reads through a wasm EXPORT
// (`pyre_fbw_diag`).  An export cannot shift the module's function-index
// space, unlike an import, which would break the JIT's baked indices.
//
// Slot layout is duplicated in the runner's decoder
// (`pyre-wasm-runner/src/main.rs`).
pub mod fbw_diag {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Authoritative walks run so far.  Counts every walk, including those
    /// past the end of the ring.
    pub const WALKS: usize = 0;
    /// Walks that rolled their store journal back while the gh#467 odometer
    /// says they executed a residual that wrote the live heap or entered a
    /// Python frame — their caller is about to replay an irreversible region.
    pub const ROLLED_BACK_WITH_EFFECTS: usize = 1;
    /// Reachability of the two walk-end flush legs whose resume pc can precede
    /// an effect the walk already recorded, and of the hazardous subset of
    /// each.  The native corpus reaches neither leg, so these say whether the
    /// wasm target does.
    pub const MIDBODY_LATCH: usize = 2;
    pub const MIDBODY_LATCH_NEW_UNJOURNALED: usize = 3;
    pub const ESCAPE_PLAIN_FALLBACK: usize = 4;
    pub const ESCAPE_PLAIN_FALLBACK_UNCLEAN: usize = 5;
    pub const ESCAPE_PORTAL_ONLY: usize = 6;
    pub const ESCAPE_PUBLISHED_CALLEE_ONLY: usize = 7;
    pub const ESCAPE_PORTAL_AND_PUBLISHED_CALLEE: usize = 8;
    pub const ESCAPE_FORCE_BY_PORTAL: usize = 9;
    pub const ESCAPE_FORCE_BY_CALLEE_ONLY: usize = 10;

    /// One ring entry per walk: four slots of outcome name (8 ASCII bytes per
    /// slot, little-endian) followed by one slot of packed counters.  A `u64`
    /// export cannot carry a string, and the outcome set is far too large to
    /// spend a tally slot per variant.
    pub const RING_BASE: usize = 11;
    pub const RING_ENTRIES: usize = 24;
    pub const RING_STRIDE: usize = 5;
    pub const NAME_SLOTS: usize = 4;

    /// Bit positions inside a ring entry's counter slot.
    pub const FLAG_VALID: u64 = 1 << 0;
    pub const FLAG_COMMITTED: u64 = 1 << 1;
    pub const FLAG_BRIDGE: u64 = 1 << 2;
    pub const SHIFT_EFFECTS: u32 = 8;
    pub const SHIFT_JOURNAL: u32 = 24;
    pub const SHIFT_EXEC_MF: u32 = 40;
    /// `WalkEndCommitLeg` discriminant (0 when the walk did not commit).
    pub const SHIFT_LEG: u32 = 56;
    pub const FIELD_MASK: u64 = 0xffff;

    const LEN: usize = RING_BASE + RING_ENTRIES * RING_STRIDE;

    static FBW_DIAG: [AtomicU64; LEN] = {
        const Z: AtomicU64 = AtomicU64::new(0);
        [Z; LEN]
    };

    /// Bump one of the reachability tallies.
    pub(crate) fn bump(i: usize) {
        FBW_DIAG[i].fetch_add(1, Ordering::Relaxed);
    }

    /// Read one slot (out-of-range reads as 0).  Surfaced to the wasm host
    /// through the `pyre_fbw_diag` export in the `pyre-wasm` crate.
    pub fn get(i: usize) -> u64 {
        FBW_DIAG
            .get(i)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Append one walk's census record.  Walks past `RING_ENTRIES` bump the
    /// totals but leave the ring alone, so the first walks — the ones that
    /// shape the trace — are the ones kept.
    pub(crate) fn record(
        name: &str,
        committed: bool,
        bridge: bool,
        effects: usize,
        journal: usize,
        exec_mf: u32,
        leg: u8,
    ) {
        let index = FBW_DIAG[WALKS].fetch_add(1, Ordering::Relaxed) as usize;
        if !committed && effects > 0 {
            FBW_DIAG[ROLLED_BACK_WITH_EFFECTS].fetch_add(1, Ordering::Relaxed);
        }
        if index >= RING_ENTRIES {
            return;
        }
        let entry = RING_BASE + index * RING_STRIDE;
        let bytes = name.as_bytes();
        for slot in 0..NAME_SLOTS {
            let mut packed = 0u64;
            for byte in 0..8 {
                if let Some(&b) = bytes.get(slot * 8 + byte) {
                    packed |= (b as u64) << (byte * 8);
                }
            }
            FBW_DIAG[entry + slot].store(packed, Ordering::Relaxed);
        }
        let flags = FLAG_VALID
            | if committed { FLAG_COMMITTED } else { 0 }
            | ((effects as u64).min(FIELD_MASK) << SHIFT_EFFECTS)
            | if bridge { FLAG_BRIDGE } else { 0 }
            | ((journal as u64).min(FIELD_MASK) << SHIFT_JOURNAL)
            | ((exec_mf as u64).min(FIELD_MASK) << SHIFT_EXEC_MF)
            | ((leg as u64) << SHIFT_LEG);
        FBW_DIAG[entry + NAME_SLOTS].store(flags, Ordering::Relaxed);
    }
}
