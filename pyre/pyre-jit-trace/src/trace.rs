//! Public trace entrypoint for `pyre`'s JIT portal.
//!
//! `trace_bytecode` drives the authoritative full-body walk
//! (`full_body_walk_trace`): the walker-as-tracer that walks the per-CodeObject
//! JitCode body, combining symbolic IR recording
//! with the per-step concrete frame snapshot.  Any location the walk declines
//! re-interprets without JIT (the trait `PyreMetaInterp` interpret loop is
//! retired, gap-10 of issue #73 Phase 6).

use majit_metainterp::{MetaInterp, TraceAction, TraceCtx};
use pyre_interpreter::CodeObject;

use crate::state::{PyreMeta, PyreSym};

thread_local! {
    /// pyjitpl.py:3048-3091 `raise_continue_running_normally` seam: set
    /// when the authoritative full-body walk committed its end-of-walk
    /// frame state into the trace's concrete frame snapshot
    /// (`flush_walk_end_state_to_frame`).  The portal call sites consume
    /// it via [`take_walk_end_flush_committed`] to decide whether the
    /// returned `FrameBox` carries adoptable end state for the LIVE
    /// frame (no-replay) or still holds the entry state (legacy replay).
    static WALK_END_FLUSH_COMMITTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Take-and-reset the walk-end flush flag for the trace that just
/// returned from [`trace_bytecode`].
pub fn take_walk_end_flush_committed() -> bool {
    WALK_END_FLUSH_COMMITTED.with(|c| c.replace(false))
}

thread_local! {
    /// Green keys whose full-body walk failed on a structural walker
    /// limitation (the recurring `DispatchError` classes listed in
    /// `full_body_walk_trace`).  A retrace of such a key skips the walk and
    /// re-interprets without JIT instead of re-walking and re-failing every
    /// hot encounter; the location stays trace-eligible (not
    /// `DONT_TRACE_HERE`), so upstream's invariant that an abort never marks a
    /// location untraceable (pyjitpl.py:2392 aborted_tracing) holds.  A walker
    /// improvement that closes one of those `DispatchError` classes shrinks
    /// this set.
    static FBW_DECLINED_KEYS: std::cell::RefCell<std::collections::HashSet<u64>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

fn fbw_declined(key: u64) -> bool {
    FBW_DECLINED_KEYS.with(|s| s.borrow().contains(&key))
}

fn fbw_decline(key: u64) {
    FBW_DECLINED_KEYS.with(|s| {
        s.borrow_mut().insert(key);
    });
}

/// Trace an entire loop body starting at `start_pc`.
///
/// Drives the authoritative full-body walk (`full_body_walk_trace`): the
/// walker walks the per-CodeObject JitCode body, recording symbolic IR against
/// the per-step concrete frame snapshot.  A location the walk declines
/// re-interprets without JIT (the trait `PyreMetaInterp` interpret loop is
/// retired, gap-10 of issue #73 Phase 6).
pub fn trace_bytecode(
    meta: &mut MetaInterp<PyreMeta>,
    sym: &mut PyreSym,
    _code: &CodeObject,
    start_pc: usize,
    mut concrete_frame: pyre_interpreter::pyframe::FrameBox,
    live_frame_addr: usize,
) -> (TraceAction, pyre_interpreter::pyframe::FrameBox) {
    // `llmodel.py:557` parity — install pyre's `Cpu` impl so the
    // optimizer's `protect_speculative_string` / `bh_strlen` /
    // `bh_strgetitem` family routes through `W_UnicodeObject`-shaped
    // `str_descr` / `unicode_descr` (`pyre_cpu` module).
    meta.set_cpu(crate::pyre_cpu::shared());

    // A stale flag from a prior trace on this thread must not leak into
    // this trace's adoption decision.
    WALK_END_FLUSH_COMMITTED.with(|c| c.set(false));
    // Likewise clear any no-replay finish payload a prior trace left
    // unconsumed.  The FBW walk re-clears this in `run_perfn_walk`; the
    // trait leg (`trace_step_result_to_action`) has no such epilogue, so
    // reset here covers both before either can stash a capture.
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    // Likewise drop any cross-frame-resume abort request a prior aborted
    // trace left unconsumed (`metainterp::interpret` clears it on the normal
    // path; this guards the paths that exit before the poll runs).
    let _ = crate::state::take_trace_abort_requested();

    let ctx = meta
        .trace_ctx()
        .expect("trace_bytecode invariant: meta.tracing must be Some during merge_point closure");
    // A multi-frame bridge carrier overrides the trace-start
    // pc with the OUTERMOST (`frames[0]`) resume pc. The passed `start_pc` is
    // the INNERMOST frame's pc (`decode_and_restore_guard_failure` returns
    // `jit_state.next_instr()`), which belongs to the deepest reconstructed
    // callee — NOT the root. A carrier resume now re-interprets without JIT (the
    // trait leg that reconstructed + pushed the callees is retired, gap-10), so
    // the walker dispatch below routes it to a plain abort; the root pc is the
    // relevant trace-start either way.
    let carrier = ctx.take_bridge_inline_carrier();
    let start_pc = if let Some(ref c) = carrier {
        c.root_pc
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
    concrete_frame.set_last_instr_from_next_instr(start_pc);
    let w_code = concrete_frame.pycode;
    // Issue #73 walker-as-tracer foundation probe (read-only).
    // `PYRE_DUMP_PERFN_JITCODE=1` dumps the per-CodeObject JitCode body
    // — the byte stream the walker-as-tracer must learn to walk so that
    // `miframe.pc == jitcode_pc` and `pc_map` can retire.  See
    // `project_issue73_architecture_walker_as_tracer_2026_05_28`.
    if std::env::var_os("PYRE_DUMP_PERFN_JITCODE").is_some() {
        dump_perfn_jitcode_for_trace(w_code, start_pc);
    }
    let cf_addr = &*concrete_frame as *const pyre_interpreter::pyframe::PyFrame as usize;
    // The snapshot stands in for concrete stepping only; vable-statics
    // capture must read pointer-valued fields (`debugdata` / `lastblock`)
    // from the live frame the compiled loop will run on.  See the
    // `live_vable_frame_addr` field doc (state.rs).  Set before the
    // full-body-walk leg below so the walker path (the production tracer
    // post-#73) sees it as well as the trait `interpret()` leg.
    //
    // gap 10 slice 2b: set this BEFORE `init_symbolic` so the root vable
    // identity (seed_virtualizable_boxes) is baked against the live frame
    // address, not the discarded snapshot's.
    sym.live_vable_frame_addr = live_frame_addr;
    // pyjitpl.py:65 MIFrame.__init__: sym fields populated once at frame
    // construction. Callee (inline) frames are set up by perform_call
    // (trace_opcode.rs:3323-3424) and don't call init_symbolic; this path
    // handles the root frame push.
    sym.init_symbolic(ctx, cf_addr);
    // Issue #215 item 2: drive the multiframe bridge-carrier resume via the
    // full-body walker (reconstruct the in-flight callee framestack + walk
    // innermost-first) instead of aborting to a no-JIT re-interpret below.
    if let Some(ref carrier) = carrier {
        // A multi-frame bridge resume is driven by the orthodox framestack
        // trampoline (`rebuild_from_resumedata` resume.py:1042-1057 + the
        // continuous interpret loop): reconstruct the resumed callee framestack
        // and walk it forward. Without it a present carrier falls through to the
        // degenerate `Trait::CarrierAbort` below, which never compiles the bridge
        // and re-aborts on every guard failure. The `PYRE_P2_DRAIN`
        // sub-walk+inject shape is a separate unsound deviation, kept gated off.
        if crate::state::p2_drain_enabled() {
            let action = drive_bridge_carrier_walk(ctx, sym, w_code, start_pc, cf_addr, carrier);
            return (action, concrete_frame);
        }
        let action = drive_bridge_framestack_walk(ctx, sym, w_code, start_pc, cf_addr, carrier);
        return (action, concrete_frame);
    }
    // Issue #73 walker-as-tracer foundation probe (slice #1, gated).
    // `PYRE_WALK_PERFN_JITCODE=1` attempts to walk the per-CodeObject
    // JitCode body via `dispatch_via_miframe` from the resume entry pc,
    // logs how far the symbolic walk gets (terminator outcome vs first
    // `DispatchError` stop), then aborts the trace.  Default-off → zero
    // production change.  Produces the Path A (payload-seeding) gap
    // inventory on a live bench now that walk-capability gaps #1/#2/#3
    // are closed.  See
    // `project_issue73_architecture_walker_as_tracer_2026_05_28`.
    // Both walker entries below are gated on `carrier.is_none()`: a
    // multi-frame bridge resume carries reconstructed inline-callee
    // recipes that only the trait path assembles+pushes (the carrier
    // drain before `interpret()` below — `rebuild_from_resumedata`
    // resume.py:1049-1056).  The walker has no multi-Python-frame
    // reconstruction yet (#68); entering it would walk the outer root
    // frame at `root_pc` instead of the deepest resumed callee.
    if carrier.is_none() && std::env::var_os("PYRE_WALK_PERFN_JITCODE").is_some() {
        probe_walk_perfn_jitcode(ctx, sym, w_code, start_pc, cf_addr);
        return (TraceAction::Abort, concrete_frame);
    }
    // Issue #73 Phase 5 production flip: the per-CodeObject JitCode body is
    // traced via the authoritative full-body walk — the walker-as-tracer
    // path that makes `miframe.pc == jitcode_pc` and lets `pc_map` retire.
    // `PYRE_FULL_BODY_WALK=0` opts back into the trait
    // `metainterp.interpret` loop below (transition escape hatch; the trait
    // tracer is deleted in Phase 6).
    //
    // A green key in `FBW_DECLINED_KEYS` had a prior walk fail on a
    // structural walker limitation (the recurring error classes in
    // `full_body_walk_trace`); the retrace routes through the trait
    // tracer below instead of permanently blacklisting the location
    // (`DONT_TRACE_HERE`).  Tracing aborts must not mark a location
    // untraceable — upstream aborts or switches to the blackhole and the
    // location stays eligible (pyjitpl.py:2392 aborted_tracing /
    // blackhole switch); the trait leg is pyre's transitional stand-in
    // until the walker covers those shapes (deleted with the trait in
    // Phase 6).
    if carrier.is_none()
        && std::env::var_os("PYRE_FULL_BODY_WALK").as_deref() != Some(std::ffi::OsStr::new("0"))
        && !fbw_declined(crate::driver::make_green_key(w_code, start_pc))
    {
        let action = full_body_walk_trace(ctx, sym, w_code, start_pc, cf_addr);
        return (action, concrete_frame);
    }
    // gap-10: the trait tracer (`PyreMetaInterp` / `owned_concrete_frame`
    // interpret loop) is retired.  Any path the walker did not trace above — an
    // `fbw_declined` key whose walk hit a structural limit, a
    // `PYRE_FULL_BODY_WALK=0` opt-out, or a multi-frame bridge `carrier` resume
    // (reconstructed only by the deleted trait leg) — re-interprets without JIT
    // for this key.  The location stays trace-eligible (no `DONT_TRACE_HERE`);
    // the next hot encounter re-walks.
    crate::jitcode_dispatch::census_record(if carrier.is_some() {
        "Trait::CarrierAbort"
    } else {
        "Trait::DeclinedAbort"
    });
    (TraceAction::Abort, concrete_frame)
}

/// Issue #73 walker-as-tracer foundation probe (slice #1).
///
/// Attempts to walk the per-CodeObject JitCode body via
/// [`crate::jitcode_dispatch::dispatch_via_miframe`] from the resume
/// entry pc (`pc_map[start_pc]`) and logs how far the symbolic walk
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
/// unfolded `residual_call`).  See
/// `project_issue73_architecture_walker_as_tracer_2026_05_28`.
/// Decode the loop-header `jit_merge_point` that governs the resume
/// coordinate `entry` (the nearest one with `pc < entry`) and return its
/// green-ref (`gr`) and red (`rr`) register lists.
///
/// These name the jitcode register colors the loop body reads its
/// loop-invariant pycode (`gr`) and frame/ec (`rr`) from.  A mid-loop walk
/// entering PAST the merge point never executes it, so those colors are
/// left `OpRef::NONE` unless explicitly seeded — the 51d.1 / B1 blocker.
///
/// Operand layout `cIRFIRF`: jdindex(`c`, 1 byte) followed by six
/// count-prefixed register lists `gi, gr, gf, ri, rr, rf`.  Returns `None`
/// when no preceding merge point exists (straight-line resume) or the
/// operand stream is truncated.
fn loop_header_merge_point_regs(code: &[u8], entry: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    // The merge point governing `entry`.  A body-guard resume enters PAST the
    // merge point, so the merge point precedes `entry` and the last `op.pc <=
    // entry` is the governing one.  A HEADER-entry resume keys to the header
    // PC's own leading `-live-` marker, inserted immediately before the
    // header's first insn, which sits just BEFORE the `jit_merge_point` op;
    // there `entry < mp_pc`, so fall back to the first merge point at or after
    // `entry`.  A straight-line function has no merge point and returns `None`.
    let mp_pc = crate::jitcode_runtime::decoded_ops(code)
        .filter(|op| op.opname == "jit_merge_point")
        .map(|op| op.pc)
        .filter(|&pc| pc <= entry)
        .max()
        .or_else(|| {
            crate::jitcode_runtime::decoded_ops(code)
                .filter(|op| op.opname == "jit_merge_point")
                .map(|op| op.pc)
                .filter(|&pc| pc >= entry)
                .min()
        })?;
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
/// `entry` with the caller-seeded `argboxes_r`.  Returns
/// `(code_len, walk_result)`; `None` when the terminal descrs are unwired.
fn dispatch_perfn_frame(
    mi: &mut crate::state::MIFrame,
    pjc: &std::sync::Arc<crate::PyJitCode>,
    entry: usize,
    argboxes_r: &[majit_ir::OpRef],
    authoritative: bool,
) -> Option<(usize, PerfnWalkResult)> {
    // Resolve the five terminal descrs off MetaInterpStaticData so the
    // walk's Finish / exit-with-exception records carry production descr
    // identities.  A missing one means setup never ran — log and bail
    // rather than feed placeholder descrs.
    let (done_void, done_int, done_ref, done_float, exit_exc_ref) = {
        let sd = mi.ctx().metainterp_sd();
        match (
            sd.done_with_this_frame_descr_void.clone(),
            sd.done_with_this_frame_descr_int.clone(),
            sd.done_with_this_frame_descr_ref.clone(),
            sd.done_with_this_frame_descr_float.clone(),
            sd.exit_frame_with_exception_descr_ref.clone(),
        ) {
            (Some(v), Some(i), Some(r), Some(f), Some(e)) => (v, i, r, f, e),
            _ => {
                eprintln!("[walk-perfn] terminal descrs not wired; skipping walk");
                return None;
            }
        }
    };

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
        mi,
        code,
        entry,
        &perfn_descr_refs,
        crate::jitcode_dispatch::RawDescrPool::PerFn(perfn_descrs),
        // Authoritative concrete execution: `false` for a read-only probe
        // (trace discarded → re-executing would corrupt live state); `true`
        // for the production full-body tracer (the walk IS the execution).
        authoritative,
        &sub_jitcode_lookup,
        done_ref,
        done_int,
        done_float,
        done_void,
        exit_exc_ref,
        true,
        pjc.jitcode.num_regs_r() as usize,
        pjc.jitcode.num_regs_i() as usize,
        pjc.jitcode.num_regs_f() as usize,
        pjc.jitcode.constants_r.as_slice(),
        pjc.jitcode.constants_i.as_slice(),
        pjc.jitcode.constants_f.as_slice(),
        argboxes_r,
        &[],
        &[],
    );
    Some((code_len, walk_result))
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
/// are wired.  Gated behind `PYRE_P2_DRAIN` (default off → unchanged behavior).
/// Thread a reconstructed callee's `SubReturn` value into the root portal's
/// operand-stack result slot so the subsequent root walk (`run_perfn_walk`'s
/// `bridge_stack_oprefs` seeding) reads it as the call result at `root_pc`.
///
/// The result lands at the codewriter-precomputed result color for the call's
/// return pc (`result_color_at_pc_at`), mapped to its `bridge_stack_oprefs`
/// stack slot (`color - nlocals`).  Returns `false` (caller declines the
/// compile) when the color is unresolved or sits below the operand stack.
fn inject_root_call_result(sym: &mut PyreSym, root_pc: usize, result: majit_ir::OpRef) -> bool {
    if sym.jitcode.is_null() {
        return false;
    }
    let jitcode_index = unsafe { (*sym.jitcode).index as i32 };
    let Some(result_color) = crate::state::result_color_at_pc_at(jitcode_index, root_pc) else {
        return false;
    };
    let nlocals = sym.nlocals;
    if result_color < nlocals {
        return false;
    }
    let slot = result_color - nlocals;
    let bridge = sym.bridge_stack_oprefs.get_or_insert_with(Vec::new);
    if bridge.len() <= slot {
        bridge.resize(slot + 1, majit_ir::OpRef::NONE);
    }
    bridge[slot] = result;
    true
}

fn drive_bridge_carrier_walk(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    w_code: *const (),
    root_pc: usize,
    cf_addr: usize,
    carrier: &majit_metainterp::BridgeInlineCarrier,
) -> TraceAction {
    crate::jitcode_dispatch::bool_box_truth_reset();
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    crate::jitcode_dispatch::fbw_store_journal_reset();

    let root_ec = sym.concrete_execution_context;
    if std::env::var_os("PYRE_P2_DIAG").is_some() {
        let pcs: Vec<usize> = carrier.recipes.iter().map(|r| r.pc).collect();
        eprintln!(
            "[p2-shape] root_pc={root_pc} n_recipes={} recipe_pcs={pcs:?}",
            carrier.recipes.len()
        );
    }
    let Some(recipe) = carrier.recipes.last() else {
        crate::jitcode_dispatch::census_record("P2Drain::NoRecipes");
        return TraceAction::Abort;
    };

    let pre_pos = ctx.get_trace_position();
    // `setup_reconstructed_callee_frame` emits the callee frame vable into the
    // trace and returns `argboxes_r` seeding the portal reds + in-flight
    // operand-stack temps; the `_pending` callee sym/concrete frame is unused on
    // the sub-walk path (the sub-walk drives the callee body off `argboxes_r` +
    // the emitted frame vable, not a callee MIFrame).
    let Some((_pending, argboxes_r)) =
        crate::state::setup_reconstructed_callee_frame(ctx, recipe, root_ec, Vec::new())
    else {
        ctx.cut_trace(pre_pos);
        crate::jitcode_dispatch::census_record("P2Drain::SetupFailed");
        return TraceAction::Abort;
    };
    let Some(callee_pjc) = crate::state::pyjitcode_for_code(recipe.code_ptr) else {
        ctx.cut_trace(pre_pos);
        crate::jitcode_dispatch::census_record("P2Drain::NoCalleePjc");
        return TraceAction::Abort;
    };
    let Some(entry) = callee_pjc.resume_jitcode_pc_for(recipe.pc) else {
        ctx.cut_trace(pre_pos);
        crate::jitcode_dispatch::census_record("P2Drain::NoCalleeEntry");
        return TraceAction::Abort;
    };
    let callee_w_globals = crate::state::recover_inline_callee_globals(recipe.code_ptr) as usize;
    // The reconstructed callee's local slot concretes (`recipe.concrete_r` is
    // parallel to `registers_r`; locals occupy `[0, nlocals)`), seeded into the
    // sub-walk's local-concrete shadow so a nested self-recursive call's int arg
    // is known.
    let nlocals = recipe.nlocals.min(recipe.concrete_r.len());
    let local_concretes = &recipe.concrete_r[..nlocals];

    // Increment 2b-i: drive the deepest callee as an inline SUB-WALK rooted on
    // the portal `sym` (is_top_level=false), so its `ref_return` surfaces
    // `SubReturn` instead of the top-level `Finish` pyre's own-portal model
    // rejects.  Diagnostic: log the outcome, then abort (trace discarded).
    let walk = crate::jitcode_dispatch::drive_bridge_carrier_subwalk(
        ctx,
        sym,
        root_pc,
        &callee_pjc,
        recipe.code_ptr as usize,
        callee_w_globals,
        entry,
        &argboxes_r,
        local_concretes,
    );
    // 2b-ii: on a clean single-recipe `SubReturn`, thread the callee result
    // into the root's operand-stack result slot and walk the ROOT top-level to
    // compile the bridge (the recorded callee continuation + the root
    // continuation form one bridge body).  Gated on `PYRE_P2_COMPILE` (requires
    // the authoritative sub-walk that produced the `SubReturn`); other shapes /
    // outcomes log + abort (trace discarded).
    let subwalk_result = match &walk {
        Some(Ok((crate::jitcode_dispatch::DispatchOutcome::SubReturn { result: Some(r) }, _))) => {
            Some(*r)
        }
        _ => None,
    };
    if let Some(result) = subwalk_result {
        if carrier.recipes.len() == 1 && std::env::var_os("PYRE_P2_COMPILE").is_some() {
            if inject_root_call_result(sym, root_pc, result) {
                crate::jitcode_dispatch::census_record("P2Drain::CompileRoot");
                return full_body_walk_trace(ctx, sym, w_code, root_pc, cf_addr);
            }
            crate::jitcode_dispatch::census_record("P2Drain::ResultSlotUnresolved");
        }
    }

    match &walk {
        Some(Ok((outcome, end_pc))) => {
            eprintln!(
                "[p2-drain] callee sub-walk OK recipe.pc={} entry={entry} end_pc={end_pc} outcome={outcome:?}",
                recipe.pc
            );
            crate::jitcode_dispatch::census_record("P2Drain::SubWalkOk");
        }
        Some(Err(e)) => {
            eprintln!(
                "[p2-drain] callee sub-walk STOP recipe.pc={} entry={entry} err={e:?}",
                recipe.pc
            );
            crate::jitcode_dispatch::census_record("P2Drain::SubWalkStop");
        }
        None => {
            crate::jitcode_dispatch::census_record("P2Drain::SubWalkSetupNone");
        }
    }

    ctx.cut_trace(pre_pos);
    crate::jitcode_dispatch::bool_box_truth_reset();
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    crate::jitcode_dispatch::fbw_store_journal_reset();
    TraceAction::Abort
}

/// Shape A orthodox multi-frame bridge resume: the default driver for a
/// single-recipe (depth-1) carrier.
///
/// A driver-managed framestack trampoline that mirrors RPython's
/// `rebuild_from_resumedata` (resume.py:1042-1057) + continuous interpret loop:
///
///   1. The reconstructed callee framestack is held in the DRIVER (owned
///      register banks per frame), not in `WalkContext` — the walker's register
///      banks are borrowed slices (`&'frame mut [OpRef]`), so a `Vec<Frame>`
///      field is borrow-check infeasible; the framestack lives here.
///   2. Each frame is driven FORWARD from its resume pc via a single-frame
///      `walk()` (`drive_bridge_carrier_subwalk` shape). Because the frame
///      traces forward, its own recursive calls fold to a live `CALL_ASSEMBLER`
///      (the self-recursive fold, `try_walker_call_assembler_self_recursive`) —
///      NOT unrolled into frame reconstruction. The framestack walk therefore
///      supersedes bounded unroll for the resumed recursion.
///   3. A frame's `SubReturn` is delivered into its PARENT frame's dst register
///      via `make_result_of_lastop` (`pyjitpl.py:258-275`) — the parent then
///      resumes forward from its own resume pc with the child result live in
///      its register. Innermost→outermost; the outermost callee's result lands
///      in the ROOT portal frame, which resumes at `root_pc`.
///
/// This dissolves the (n-1)-vs-`fib(n-1)` provenance bug (the return is a live
/// `CALL_ASSEMBLER` result, not a reconstructed-frame arg slot) and the
/// missing-`CALL_ASSEMBLER` bug (recursion stays a call boundary).
///
/// A single-recipe carrier drives the outer continuation via
/// [`drive_outer_continuation_and_map`] and compiles the whole cross-frame
/// bridge. `recipes.len() != 1` (depth≥2, #343) and any setup/continuation
/// failure fall through to `SafeAbortReconstruction`, which cuts the whole
/// reconstruction and re-interprets with the correct result (no SEGV). The
/// [`drive_bridge_carrier_walk`] sub-walk+inject shape (`PYRE_P2_DRAIN`) is a
/// separate unsound deviation, kept gated off.
fn drive_bridge_framestack_walk(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    w_code: *const (),
    root_pc: usize,
    cf_addr: usize,
    carrier: &majit_metainterp::BridgeInlineCarrier,
) -> TraceAction {
    crate::jitcode_dispatch::bool_box_truth_reset();
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    crate::jitcode_dispatch::fbw_store_journal_reset();

    if std::env::var_os("PYRE_P2_DIAG").is_some() {
        let pcs: Vec<usize> = carrier.recipes.iter().map(|r| r.pc).collect();
        eprintln!(
            "[p2-framestack] root_pc={root_pc} n_recipes={} recipe_pcs={pcs:?}",
            carrier.recipes.len()
        );
    }

    let Some(recipe) = carrier.recipes.last() else {
        crate::jitcode_dispatch::census_record("P2Framestack::NoRecipes");
        return TraceAction::Abort;
    };

    let root_ec = sym.concrete_execution_context;
    let pre_pos = ctx.get_trace_position();
    // Reconstruct the deepest resumed callee frame vable + its `argboxes_r`
    // portal reds (mirror of the `drive_bridge_carrier_walk` setup).
    let Some((_pending, argboxes_r)) =
        crate::state::setup_reconstructed_callee_frame(ctx, recipe, root_ec, Vec::new())
    else {
        ctx.cut_trace(pre_pos);
        crate::jitcode_dispatch::census_record("P2Framestack::SetupFailed");
        return TraceAction::Abort;
    };
    let Some(callee_pjc) = crate::state::pyjitcode_for_code(recipe.code_ptr) else {
        ctx.cut_trace(pre_pos);
        crate::jitcode_dispatch::census_record("P2Framestack::NoCalleePjc");
        return TraceAction::Abort;
    };
    let Some(entry) = callee_pjc.resume_jitcode_pc_for(recipe.pc) else {
        ctx.cut_trace(pre_pos);
        crate::jitcode_dispatch::census_record("P2Framestack::NoCalleeEntry");
        return TraceAction::Abort;
    };
    let callee_w_globals = crate::state::recover_inline_callee_globals(recipe.code_ptr) as usize;
    let nlocals = recipe.nlocals.min(recipe.concrete_r.len());
    let local_concretes = &recipe.concrete_r[..nlocals];

    let pos_after_setup = ctx.get_trace_position();
    if std::env::var_os("PYRE_P2_DIAG").is_some() {
        let root_entry = crate::state::pyjitcode_for_code(w_code)
            .and_then(|pjc| pjc.resume_jitcode_pc_for(root_pc));
        eprintln!(
            "[p2-fs] callee_entry(jit)={entry} callee.pc(py)={} root_pc(py)={root_pc} root_entry(jit)={root_entry:?} pos_pre={pre_pos:?} pos_after_setup={pos_after_setup:?}",
            recipe.pc
        );
    }

    // Drive the deepest reconstructed callee FORWARD from its resume pc. The
    // sub-walk holds `CarrierResumeGuard` for its lifetime, so a nested
    // self-recursive call folds to a live `CALL_ASSEMBLER` instead of
    // re-unrolling the call tree.
    let walk = crate::jitcode_dispatch::drive_bridge_carrier_subwalk(
        ctx,
        sym,
        root_pc,
        &callee_pjc,
        recipe.code_ptr as usize,
        callee_w_globals,
        entry,
        &argboxes_r,
        local_concretes,
    );
    let subwalk_result = match &walk {
        Some(Ok((crate::jitcode_dispatch::DispatchOutcome::SubReturn { result: Some(r) }, _))) => {
            Some(*r)
        }
        _ => None,
    };
    if std::env::var_os("PYRE_P2_DIAG").is_some() {
        let pos_after_subwalk = ctx.get_trace_position();
        eprintln!(
            "[p2-fs] subwalk outcome={:?} result={subwalk_result:?} pos_after_subwalk={pos_after_subwalk:?}",
            walk.as_ref()
                .map(|r| r.as_ref().map(|(o, pc)| (format!("{o:?}"), *pc)))
        );
        // Dump the ops the sub-walk recorded into `ctx` (pre_pos..now) to confirm
        // the returned SubReturn result traces to the boxed ADD of the two live
        // CALL_ASSEMBLER (the recursive fib(n-1)/fib(n-2) results), not a
        // reconstructed-frame arg-slot read.
        ctx.dump_trace_ops_diag("framestack-subwalk-end");
    }

    // The sub-walk drives the deepest reconstructed callee frame (WITH its
    // emitted vable) forward and records into `ctx`: it emits the two live
    // `CALL_ASSEMBLER` for the callee recursion ([p2-ca] EMIT=2) and its
    // in-callee guards encode resume snapshots against the paused root
    // (`FullBodySnapshotSymGuard`, snapshot_data_len>0), returning a live
    // `SubReturn` result. The vable is load-bearing: local reads lower to
    // `getarrayitem_vable`, which aborts `VableBoxNotSeeded` on an unseeded base
    // — the orthodox resume rebuilds the frame virtualizable
    // (`rebuild_from_resumedata` resume.py:1042 fills the jitcode registers; the
    // Python locals live in the rebuilt vable).
    //
    // #41 continuous cross-frame walk: after the deepest callee sub-walk returns
    // its result, continue the OUTER (root portal) frame forward from its resume
    // pc WITHOUT cutting — appending to the sub-walk's `ctx` so the sub-walk's
    // live `CALL_ASSEMBLER` continuation stays in the compiled bridge. The callee
    // result is delivered into the outer's call-dst register
    // (`make_result_of_lastop`), never a resume color, so the outer resumes with
    // the 1st-call result live and records its 2nd call + ADD + return.
    if let Some(result) = subwalk_result {
        // A single-recipe (depth-1) reconstruction continues the OUTER frame
        // forward and compiles the whole cross-frame bridge. `recipes.len() != 1`
        // (depth≥2) and any continuation-setup failure fall through to the clean
        // `SafeAbortReconstruction` below (correct no-JIT re-interpret).
        if carrier.recipes.len() == 1 {
            if let Some(action) = drive_outer_continuation_and_map(
                ctx, sym, w_code, root_pc, cf_addr, result, pre_pos,
            ) {
                return action;
            }
        }
    }

    let _ = subwalk_result;
    ctx.cut_trace(pre_pos);
    crate::jitcode_dispatch::bool_box_truth_reset();
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    crate::jitcode_dispatch::fbw_store_journal_reset();
    crate::jitcode_dispatch::census_record("P2Framestack::SafeAbortReconstruction");
    TraceAction::Abort
}

/// #41: set up + drive the outer (root portal) continuation and map its outcome
/// to a `TraceAction`.  Returns `Some(action)` when the continuation produced a
/// compilable bridge (or a definite terminal decision), `None` when setup could
/// not proceed so the caller falls through to its clean abort (which cuts the
/// whole reconstruction).  Delivery is by physical call-dst register, decoded
/// from the residual-call op whose `next_pc` is the outer resume entry.
fn drive_outer_continuation_and_map(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    w_code: *const (),
    root_pc: usize,
    _cf_addr: usize,
    result: majit_ir::OpRef,
    pre_pos: majit_metainterp::recorder::TracePosition,
) -> Option<TraceAction> {
    let root_pjc = crate::state::pyjitcode_for_code(w_code)?;
    let entry = root_pjc.resume_jitcode_pc_for(root_pc)?;
    // Decode the call-dst register: the op whose `next_pc == entry` is the
    // residual call the outer resumes after; its `>r` dst is the last operand
    // byte (`code[entry-1]`).
    let code = root_pjc.jitcode.code.as_slice();
    let call_dst_reg = {
        let mut found = None;
        for op in crate::jitcode_runtime::decoded_ops(code) {
            if op.next_pc == entry {
                if op.opname.starts_with("residual_call") && op.argcodes.ends_with(">r") {
                    found = code.get(entry - 1).map(|&b| b as usize);
                }
                break;
            }
        }
        found
    }?;
    let frame_reg = {
        let r = root_pjc.metadata.portal_frame_reg;
        if r != u16::MAX { r as usize } else { 1 }
    };
    let frame_box = ctx
        .standard_virtualizable_box()
        .unwrap_or_else(|| ctx.const_ref(_cf_addr as i64));
    // `w_code` is the root frame's PyCode wrapper; `recover_inline_callee_globals`
    // keys the `code_ptr → live wrapper` registry by RAW code identity, so resolve
    // the raw pointer first.
    let root_code_ptr =
        unsafe { pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef) };
    let root_w_globals = crate::state::recover_inline_callee_globals(root_code_ptr) as usize;

    if std::env::var_os("PYRE_P2_DIAG").is_some() {
        eprintln!(
            "[p2-fs] outer-continuation entry(jit)={entry} call_dst_reg={call_dst_reg} frame_reg={frame_reg} result={result:?} frame_box={frame_box:?}"
        );
    }

    let outcome = crate::jitcode_dispatch::drive_outer_frame_continuation(
        ctx,
        sym,
        &root_pjc,
        w_code as usize,
        root_w_globals,
        root_pc,
        entry,
        frame_box,
        frame_reg,
        result,
        call_dst_reg,
    );

    if crate::state::take_trace_abort_requested() {
        crate::jitcode_dispatch::census_record("P2Framestack::OuterTraceAbortRequested");
        ctx.cut_trace(pre_pos);
        return Some(TraceAction::Abort);
    }
    if std::env::var_os("PYRE_P2_DIAG").is_some() {
        ctx.dump_trace_ops_diag("framestack-outer-walk-end");
        eprintln!(
            "[p2-fs] outer outcome={:?}",
            outcome
                .as_ref()
                .map(|r| r.as_ref().map(|(o, pc)| (format!("{o:?}"), *pc)))
        );
    }

    match outcome {
        Some(Ok((crate::jitcode_dispatch::DispatchOutcome::Terminate, _end_pc))) => {
            match crate::jitcode_dispatch::fbw_finish_payload_take() {
                Some((_, majit_ir::Type::Void)) => {
                    let key = crate::driver::make_green_key(w_code, root_pc);
                    ctx.set_green_key(key, (w_code as usize, root_pc));
                    Some(TraceAction::Finish {
                        finish_args: vec![],
                        finish_arg_types: vec![],
                        exit_with_exception: false,
                    })
                }
                Some((finish_value, finish_type)) => {
                    let key = crate::driver::make_green_key(w_code, root_pc);
                    ctx.set_green_key(key, (w_code as usize, root_pc));
                    crate::jitcode_dispatch::census_record("P2Framestack::OuterFinish");
                    if std::env::var_os("PYRE_P2_DIAG").is_some() {
                        eprintln!(
                            "[p2-fs] COMPILE Finish finish_value={finish_value:?} type={finish_type:?}"
                        );
                    }
                    Some(TraceAction::Finish {
                        finish_args: vec![finish_value],
                        finish_arg_types: vec![finish_type],
                        exit_with_exception: false,
                    })
                }
                None => {
                    crate::jitcode_dispatch::census_record("P2Framestack::OuterNoFinishPayload");
                    if std::env::var_os("PYRE_P2_DIAG").is_some() {
                        eprintln!("[p2-fs] outer Terminate but NO finish payload -> abort");
                    }
                    ctx.cut_trace(pre_pos);
                    Some(TraceAction::Abort)
                }
            }
        }
        other => {
            if std::env::var_os("PYRE_P2_DIAG").is_some() {
                eprintln!("[p2-fs] outer non-terminate outcome, aborting: {other:?}");
            }
            crate::jitcode_dispatch::census_record("P2Framestack::OuterNonTerminate");
            ctx.cut_trace(pre_pos);
            Some(TraceAction::Abort)
        }
    }
}

fn run_perfn_walk(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    w_code: *const (),
    start_pc: usize,
    cf_addr: usize,
    authoritative: bool,
) -> Option<(usize, usize, PerfnWalkResult)> {
    let Some(pjc) = crate::state::pyjitcode_for_code(w_code) else {
        eprintln!("[walk-perfn] no per-CodeObject PyJitCode for code={w_code:?}");
        return None;
    };
    let Some(entry) = pjc.resume_jitcode_pc_for(start_pc) else {
        // The frozen pc_map of this already-built body does not encode
        // `start_pc` as a resume coordinate, so the same body walked from
        // the same entry recurs identically on every retrace.  Decline the
        // key (route its retraces to the trait tracer via FBW_DECLINED_KEYS)
        // instead of re-walking and re-aborting each iteration; mirrors the
        // `built_as_portal=false` structural decline below.
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[walk-perfn] no jitcode entry for start_pc={start_pc} (pc_map_len={}); declining walk",
                pjc.metadata.pc_map.len()
            );
        }
        fbw_decline(crate::driver::make_green_key(w_code, start_pc));
        return None;
    };
    // The full-body walk drives a PORTAL trace, so the body must carry
    // the portal entry shape (`FrameInputs::Portal`: `[frame, ec]` red
    // inputs + the frame-vable locals prologue).  A body first compiled
    // as a plain CALLEE (`FrameInputs::Frame` — `get_jitcode` builds the
    // shape from `jitdriver_sd_from_portal_graph` at compile time, and a
    // function discovered through another function's call compiles
    // before it becomes a portal) reads its params from caller-seeded
    // registers; the portal red seeding below would land `ec_box` in a
    // PARAMETER color and the walk would record the ExecutionContext
    // const as the function's argument — garbage baked into the trace
    // (previously masked only when the unseeded color happened to stay
    // `OpRef::NONE` and aborted as `ResidualCallArgUnbound`).  The
    // installed body is frozen once trace-side resume data references
    // it, so it cannot be swapped for a portal rebuild here; decline
    // permanently like the other structural `FBW_DECLINED_KEYS` classes
    // and let the trait tracer compile this function.
    if !pjc.metadata.built_as_portal {
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-abort] start_pc={start_pc} jitcode body compiled as plain callee \
                 (built_as_portal=false); declining walk"
            );
        }
        fbw_decline(crate::driver::make_green_key(w_code, start_pc));
        return None;
    }

    let is_bridge_trace = ctx.is_bridge_trace;
    let mut mi = crate::state::MIFrame::from_sym(ctx, sym, cf_addr, start_pc, start_pc);

    // setup_call argbox: seed r0 = the standard virtualizable identity box
    // (`virtualizable_boxes[-1]`, the `InputArgRef(SYM_FRAME_IDX)` that
    // `init_symbolic` seeded) — the SAME OpRef production's arm entry uses
    // (`dispatch_via_miframe_at_opcode_entry` seeds r0 = `sym.frame`, and
    // `sym.frame == OpRef::input_arg_typed(SYM_FRAME_IDX, Ref)`).  A fresh
    // `const_ref(cf_addr)` would be a DIFFERENT OpRef than the identity box,
    // so `concrete_of_opref`'s standard-vable resolution (trace_ctx.rs:1842,
    // keyed on `== standard_virtualizable_box()`) would miss and every vable
    // read would fall through to the nonstandard GETFIELD_GC leg.  Falls back
    // to `const_ref` only when no virtualizable is bound.
    //
    // NOTE (51d.1 root cause): seeding r0 is NECESSARY but not sufficient for
    // the mid-loop resume entry (pc=107, after the loop-header
    // `jit_merge_point` @ pc=94).  The loop body reads its vable from a
    // post-merge LOOP-INPUT register (the merge-point reds), NOT from r0; that
    // register is left `OpRef::NONE` because the probe enters past the merge
    // point and never binds the reds.  `concrete_of_opref(NONE)` returns the
    // `GcRef(usize::MAX)` sentinel → `is_nonstandard_virtualizable` takes the
    // nonstandard leg → `getarrayitem_vable` returns `Value::Void` even though
    // the virtualizable SHADOW entry is correct.  Closing that needs the live
    // loop-input registers seeded at walk entry (task #53), not just r0.
    let frame_box = mi
        .ctx()
        .standard_virtualizable_box()
        .unwrap_or_else(|| mi.ctx().const_ref(cf_addr as i64));
    // 51d.1 (B1 blocker): seed the loop's live INPUT registers so the
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
    let ec_box = mi.ctx().const_ref(sym.concrete_execution_context as i64);
    let pycode_box = mi.ctx().const_ref(w_code as i64);
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
        match loop_header_merge_point_regs(pjc.jitcode.code.as_slice(), entry) {
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
            // (internal promote `GuardValue` + force store-back, no resume
            // snapshot → `NonStandardVableFinishPortalUnsupported` abort).
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
        // #124: a bridge enters mid-body, where the loop-header merge-point
        // colors seeded above (the loop's green pycode / red frame+ec) are
        // reused for live operand-stack temps — the kept conditional-
        // expression / short-circuit / chained-compare value.  Leaving e.g.
        // the pycode green at the kept temp's color feeds a stale code object
        // into its binary op (`unsupported operand type(s) for +: 'code' and
        // 'int'`).  Override the kept operand-stack colors
        // [nlocals..nlocals+stack_only) with the resume-data OpRefs
        // setup_bridge_sym resolved; in that semantic prefix the abstract-
        // register color equals the semantic slot.  Locals (read through the
        // vable) and frame/ec (at their own colors) keep the seeding above.
        //
        // The `nl + i` slot→color shortcut collides with a red-input color
        // when `portal_frame_reg == nlocals` (e.g. fib: frame=r1, nlocals=1,
        // so operand-stack slot 0 → color 1 == frame).  A kept operand-stack
        // temp never occupies a red-input color, so skip those: seeding a temp
        // over the frame color overwrites the standard virtualizable identity
        // and forces every later `vable_*` op onto the nonstandard leg
        // (NonStandardVableFinishPortalUnsupported abort).
        if is_bridge_trace {
            if let Some(ref bridge_stack) = sym.bridge_stack_oprefs {
                let nl = sym.nlocals;
                for (i, &opref) in bridge_stack.iter().enumerate() {
                    if !opref.is_none() {
                        let color = (nl + i) as u8;
                        if reserved_red_colors.contains(&color) {
                            continue;
                        }
                        seed(color, opref);
                    }
                }
            }
        } else {
            // Loop trace: seed operand-stack loop-input registers from
            // sym.registers_r.  RPython's MIFrame registers are populated
            // by executing every opcode (including jit_merge_point, which
            // binds the reds); pyre's walker enters PAST the merge point,
            // so loop-carried operand-stack variables (e.g. the FOR_ITER
            // iterator on TOS) are left OpRef::NONE without this seed.
            // The vable shadow (virtualizable_boxes) holds the correct
            // InputArgRef for each slot, but vable_getarrayitem_ref routes
            // through is_nonstandard_virtualizable when the reader OpRef's
            // concrete is the usize::MAX sentinel (= NONE), so the
            // standard fast path is never reached.  Seed the operand-stack
            // colors (nlocals .. nlocals+stack_depth) from sym.registers_r
            // — the same InputArgRef values init_symbolic placed there.
            // Clamp to the jitcode's num_regs_r so the argboxes vector
            // never exceeds the callee register bank size.
            //
            // Also stamp each seeded InputArgRef with its concrete pointer
            // from the live frame's `locals_cells_stack_w` so downstream
            // `box_value`/`concrete_of_opref` consumers (e.g. residual-call
            // arg resolution in `try_execute_residual_call_via_executor`)
            // can resolve loop-input operand-stack values.  Without this,
            // a `ForIterNext(iterator)` residual declines execution because
            // the iterator arg's `box_value` returns `None`, leaving the
            // `ptr_nonzero` result unstemped → `GotoIfNotValueNotConcrete`.
            let nl = sym.nlocals;
            let num_regs_r = pjc.jitcode.num_regs_r() as usize;
            let stack_depth = sym.registers_r.len().saturating_sub(nl);
            // Read the live frame's locals_cells_stack_w for concrete ptrs.
            let frame_ref = unsafe { &*(cf_addr as *const pyre_interpreter::pyframe::PyFrame) };
            let frame_slots: &[pyre_object::PyObjectRef] = unsafe {
                if !frame_ref.locals_cells_stack_w.is_null() {
                    (*frame_ref.locals_cells_stack_w).as_slice()
                } else {
                    &[]
                }
            };
            for i in 0..stack_depth {
                let color = nl + i;
                if color >= num_regs_r {
                    break;
                }
                let opref = sym.registers_r[nl + i];
                if !opref.is_none() {
                    if reserved_red_colors.contains(&(color as u8)) {
                        continue;
                    }
                    seed(color as u8, opref);
                    // Stamp the concrete pointer from the live frame slot.
                    let frame_idx = nl + i;
                    if frame_idx < frame_slots.len() {
                        let ptr = frame_slots[frame_idx];
                        if !ptr.is_null() {
                            mi.ctx().set_opref_concrete(
                                opref,
                                majit_ir::Value::Ref(majit_ir::GcRef(ptr as usize)),
                            );
                        }
                    }
                }
            }
        }
        v
    };

    let Some((code_len, mut walk_result)) =
        dispatch_perfn_frame(&mut mi, &pjc, entry, &argboxes_r, authoritative)
    else {
        return None;
    };

    // Full-body-walk loop close: the walker's `jit_merge_point` handler
    // produces RPython-style reds (`jump_args = [frame, ec]`, len 2 for the
    // portal jitdriver), but pyre's runtime closes loops against the
    // EXPLICIT scalar inputarg vector
    // `[frame, ec, next_instr, code, valuestackdepth, debugdata, lastblock,
    //  namespace, locals..., stack...]` (len >= NUM_SCALAR_INPUTARGS).
    // `validate_close_with_jump_args` (state.rs) rejects the reds shape, so
    // rebuild the explicit vector via `close_loop_args_at`, mirroring the
    // trait path's `reached_loop_header` (trace_opcode.rs close path).  The
    // loop-carried local/stack OpRefs come from the virtualizable shadow in
    // the TraceCtx (`virtualizable_box_at`, maintained by the authoritative
    // walk's vable ops), NOT from the walk's private register file, so the
    // shadow is live here even though `sym.registers_r` is not.
    //
    // Authoritative only: `close_loop_args_at` records SameAs ops and flushes
    // the virtualizable shadow to the concrete frame heap, which the
    // read-only probe (trace discarded) must not do.
    if authoritative {
        if let Ok((
            crate::jitcode_dispatch::DispatchOutcome::CloseLoop {
                jump_args,
                loop_header_pc,
            },
            _end_pc,
        )) = &mut walk_result
        {
            let loop_header_pc = *loop_header_pc;
            // `close_loop_args_at` reads `self.orgpc` for the
            // portal-bridge vsd lookup + last_instr anchor; the merge point
            // closes at the loop header, so anchor orgpc there.
            mi.orgpc = loop_header_pc;
            *jump_args = mi.close_loop_args_at(ctx, Some(loop_header_pc));
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
        // `PYRE_FBW_END_FLUSH=0` opts out for bisection.
        if std::env::var_os("PYRE_FBW_END_FLUSH").as_deref() != Some(std::ffi::OsStr::new("0")) {
            if let Ok((outcome, _end_pc)) = &walk_result {
                let header_pc = match outcome {
                    crate::jitcode_dispatch::DispatchOutcome::CloseLoop {
                        loop_header_pc, ..
                    } => Some(*loop_header_pc),
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
                    } else if crate::state::flush_walk_end_state_to_frame(ctx, cf_addr, header_pc) {
                        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-end-flush] COMMIT header_pc={header_pc} bridge={} \
                                 journal_len={} outcome={outcome:?}",
                                ctx.is_bridge_trace,
                                crate::jitcode_dispatch::fbw_store_journal_len(),
                            );
                        }
                        WALK_END_FLUSH_COMMITTED.with(|c| c.set(true));
                    } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-end-flush] declined at header_pc={header_pc} (shadow slot \
                             without concrete / depth / lastblock) — legacy replay kept"
                        );
                    }
                }
            }
        }

        // `abort_permanent` marker abort (DELETE_FAST and the other
        // emit_abort_permanent opcodes): the marker's contract is "resume
        // the interpreter AT this unsupported opcode and run it" — codewriter
        // stores `last_instr = py_pc - 1` for the blackhole.  On the
        // full-body walk that recorded write is discarded with the aborted
        // trace, while the walk already executed the region's residual side
        // effects concretely, so the legacy `ContinueRunningNormally` replays
        // them from entry → double-execution (e.g. a `del`-bearing method
        // whose prior STORE_ATTR ran once during the walk, then again on
        // replay).  Flush the abort-point frame (locals + last_instr) so the
        // portal resumes at the unsupported opcode instead of replaying —
        // same mechanism and same no-unjournaled-effect predicate as the
        // CloseLoop end-flush above.  `PYRE_FBW_ABORT_FLUSH=0` opts out.
        if std::env::var_os("PYRE_FBW_ABORT_FLUSH").as_deref() != Some(std::ffi::OsStr::new("0")) {
            if let Err(crate::jitcode_dispatch::DispatchError::AbortPermanentMarkerReached { pc }) =
                &walk_result
            {
                let abort_jit_pc = *pc;
                if crate::jitcode_dispatch::fbw_has_unjournaled_effect()
                    || crate::jitcode_dispatch::fbw_abort_in_subwalk()
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
                        WALK_END_FLUSH_COMMITTED.with(|c| c.set(true));
                    } else if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                        eprintln!(
                            "[fbw-abort-flush] declined at resume_py_pc={resume_py_pc} \
                             (shadow slot without concrete / depth / lastblock) — legacy replay kept"
                        );
                    }
                }
            }
        }

        // #32 S2: a kept-stack branch guard whose not-taken arm cannot be
        // restored for the COMPILED trace aborts (`AbortPermanent`), but the
        // authoritative walk's symbolic shadow IS complete at the abort pc (the
        // hazard is about the JIT resume snapshot, not the interpreter-side
        // shadow).  Flush that end state to the live frame so the interpreter
        // resumes at the abort pc with the walked iterations already counted,
        // instead of discarding the walk and dropping an in-flight FOR_ITER
        // item via the conservative #30 header-guard drop.  Same
        // no-unjournaled-effect / no-sub-walk predicate and same all-or-nothing
        // `flush_walk_end_state_to_frame` gate as the CloseLoop / marker legs;
        // when the flush declines (a slot the shadow cannot resolve) the legacy
        // drop stands (the residual S3 case).  `PYRE_FBW_BRANCH_FLUSH=0` opts
        // out.
        if std::env::var_os("PYRE_FBW_BRANCH_FLUSH").as_deref() != Some(std::ffi::OsStr::new("0")) {
            if let Err(
                crate::jitcode_dispatch::DispatchError::BranchGuardUnrestorableKeptStackPermanent {
                    pc,
                },
            ) = &walk_result
            {
                let abort_jit_pc = *pc;
                if crate::jitcode_dispatch::fbw_has_unjournaled_effect()
                    || crate::jitcode_dispatch::fbw_abort_in_subwalk()
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
                    // Shape A — the abort resumes AT a FOR_ITER header whose
                    // consumed item is in flight (`body_pc == resume_py_pc + 1`,
                    // and the opcode there really is a FOR_ITER): the walk
                    // advanced the iterator but the item is not yet on the
                    // flushed (header) stack, so deliver it (push + reposition to
                    // the body) so the body runs once.
                    let push = crate::jitcode_dispatch::fbw_foriter_inflight_take_for_resume(
                        cf_addr,
                        resume_py_pc,
                    );
                    // Commit ONLY for Shape A — the abort resumes at an
                    // in-flight FOR_ITER header and an item is delivered.  This
                    // keeps the leg strictly a FOR_ITER-continuation delivery: a
                    // `BranchGuardUnrestorableKeptStackPermanent` that is not at
                    // such a header (a non-FOR_ITER trace, or an in-flight item
                    // whose header is not the resume pc — the consumed item then
                    // sits between the header and the resume pc but is not on the
                    // shadow stack, so adopting it would resume with a stale TOS
                    // and re-run the loop) keeps the legacy drop byte-identically
                    // (the residual S3 case).  So every other abort shape is
                    // untouched, including the entire flag-OFF path.
                    let committed = push.is_some()
                        && crate::state::flush_walk_end_state_to_frame_with_item(
                            ctx,
                            cf_addr,
                            resume_py_pc,
                            push,
                        );
                    if committed {
                        // The flush owns the iteration count; drop any remaining
                        // in-flight items so the legacy deliver cannot re-apply
                        // one (exactly-once).
                        crate::jitcode_dispatch::fbw_foriter_inflight_clear();
                        WALK_END_FLUSH_COMMITTED.with(|c| c.set(true));
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
    // Bridge walks are excluded: only the loop-free function portal consumes
    // the finish stash without replaying.  A bridge `Terminate` walk's caller
    // resumes the region through the blackhole, so keeping the stores here
    // would double-apply them (once eagerly, once in the blackhole replay).
    let terminate_no_replay = crate::jitcode_dispatch::fbw_no_replay_exit_enabled()
        && !is_bridge_trace
        && matches!(
            &walk_result,
            Ok((crate::jitcode_dispatch::DispatchOutcome::Terminate, _))
        )
        && crate::jitcode_dispatch::fbw_finish_concrete_peek().is_some()
        && !crate::jitcode_dispatch::fbw_has_unjournaled_effect();
    if !terminate_no_replay {
        crate::jitcode_dispatch::fbw_finish_concrete_reset();
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
    if WALK_END_FLUSH_COMMITTED.with(|c| c.get()) || terminate_no_replay {
        crate::jitcode_dispatch::fbw_store_journal_commit();
    } else {
        crate::jitcode_dispatch::fbw_store_journal_rollback();
    }

    Some((entry, code_len, walk_result))
}

/// Issue #73 walker-as-tracer foundation probe (slice #1, read-only).
///
/// Runs the per-CodeObject full-body walk via [`run_perfn_walk`] in
/// non-authoritative mode, logs how far the symbolic walk got (terminator
/// outcome vs first `DispatchError` stop), then rolls the recorder back so
/// the diagnostic leaves no partial trace.  `PYRE_PROBE_AUTHORITATIVE=1`
/// opts into authoritative execution for diagnosis ONLY (verifying the
/// walk advances past the loop `goto_if_not`); it corrupts the live
/// frame/iterator state because the probe still discards the trace, so it
/// must never be set outside a throwaway run.
fn probe_walk_perfn_jitcode(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
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
/// past the loop (#125).  The walk's reactive `abort_permanent` decline
/// never fires because the corrupted guard exits before reaching the
/// marker.  The scan is scoped to ops at/after the first `jit_merge_point`
/// (the inner loop header) so a prologue-only marker (e.g. `COPY_FREE_VARS`
/// ahead of a clean hot loop) does not over-decline.
fn loop_body_has_abort_permanent(w_code: *const ()) -> bool {
    let Some(pjc) = crate::state::pyjitcode_for_code(w_code) else {
        return false;
    };
    let code = pjc.jitcode.code.as_slice();
    let mut seen_merge_point = false;
    for op in crate::jitcode_runtime::decoded_ops(code) {
        if op.opname == "jit_merge_point" {
            seen_merge_point = true;
        } else if seen_merge_point && op.opname == "abort_permanent" {
            return true;
        }
    }
    false
}

/// Issue #73 production full-body tracer (Phase 5 flip, gated).
///
/// `PYRE_FULL_BODY_WALK=1` drives the per-CodeObject JitCode body via
/// [`run_perfn_walk`] in authoritative mode AS the production trace — the
/// walk IS the concrete execution, so unlike the probe it keeps the
/// recorded trace.  Maps the walk outcome to a [`TraceAction`] for the
/// caller to compile.
///
/// Conservative mapping (first slice): only `CloseLoop` — the validated
/// end-to-end case (the four loop benches close under authoritative) — is
/// mapped to a real `CloseLoopWithArgs`; every other outcome (`Terminate`
/// finish-arg recovery, `SubReturn`/`SubRaise`, `SwitchToBlackhole`, any
/// `DispatchError`) aborts the trace so the portal falls back to the trait
/// tracer.  Default-off → the trait `metainterp.interpret` path is
/// untouched.  The remaining flip blocker is guard-snapshot/resume
/// correctness, which this harness exists to validate.
fn full_body_walk_trace(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    w_code: *const (),
    start_pc: usize,
    cf_addr: usize,
) -> TraceAction {
    // #125: decline up front when a loop body carries an `abort_permanent`
    // marker.  The authoritative walk would otherwise mis-seed the loop
    // guard, exit early, and concretely double-execute the post-loop tail;
    // routing to the trait tracer (which handles the unported op) is the
    // same outcome the reactive in-walk `abort_permanent` decline reaches,
    // minus the frame corruption.
    if loop_body_has_abort_permanent(w_code) {
        // Tag the decline so `PYRE_FBW_DEBUG_ABORT` census attributes it to the
        // up-front `abort_permanent` scan, not the trait retry fall-through
        // (`Trait::DeclinedAbort`).  Without this the real declining class is
        // invisible to the census.
        crate::jitcode_dispatch::census_record("FullBodyWalk::LoopBodyAbortPermanent");
        fbw_decline(crate::driver::make_green_key(w_code, start_pc));
        return TraceAction::Abort;
    }
    // Mirror the trait path (trace_bytecode pre-interpret): register the
    // initial merge point with typed input-arg boxes so the trace head
    // carries the portal's entry signature (`inputarg_types()`).  Without
    // it the compiled loop's entry args don't match what the portal
    // supplies, so the portal cannot enter the compiled loop and re-traces
    // every iteration (the observed spin).
    // Clear the walk-local bool-box-truth map left by a prior aborted walk so
    // it cannot leak into this one.
    crate::jitcode_dispatch::bool_box_truth_reset();
    // Slice b (PYRE_FBW_CALL_ASSEMBLER): clear any Finish payload a prior
    // aborted walk's top-level `*_return` arm may have stashed, so a stale
    // value cannot leak into this walk's `Terminate` handling.
    crate::jitcode_dispatch::fbw_finish_payload_reset();
    // Clear the prior walk's store journal + unjournaled-effect flag so
    // dropped (aborted) entries cannot be applied by this walk's commit.
    crate::jitcode_dispatch::fbw_store_journal_reset();
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
    // coordinate the jitcode pc_map cannot encode (#124/#130) and requested
    // an abort (`state::request_trace_abort`).  The walker does not poll the
    // flag mid-walk, so honor it here before mapping the outcome — otherwise a
    // walk that reaches a terminator would compile a trace carrying the bad
    // guard.  Discarding the trace matches the trait leg's `interpret()` poll.
    if crate::state::take_trace_abort_requested() {
        crate::jitcode_dispatch::census_record("TraceAbortRequested");
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-abort] start_pc={start_pc} unencodable cross-frame resume coordinate (#124/#130)"
            );
        }
        return TraceAction::Abort;
    }
    if ctx.is_bridge_trace && std::env::var_os("PYRE_P2_DIAG").is_some() {
        ctx.dump_trace_ops_diag("carrier-root-walk-end");
    }
    match walk_result {
        Some((_entry, _code_len, Ok((outcome, _end_pc)))) => match outcome {
            crate::jitcode_dispatch::DispatchOutcome::CloseLoop {
                jump_args,
                loop_header_pc,
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
                // `done_with_this_frame` with no back-edge.  Under the
                // PYRE_FBW_CALL_ASSEMBLER gate the return arm routed through
                // `fbw_terminate_with_finish`, which re-boxed the result to
                // Type::Ref, recorded the vable store-back + GUARD_NOT_FORCED_2,
                // and stashed the finish payload.  Build the portal-exit FINISH
                // from it so the compile pipeline records FINISH from
                // `finish_args` (mirror of the trait `StepResult::Return`
                // path, trace_opcode.rs).  Ungated → no payload → `Abort`
                // exactly as before the slice.
                match crate::jitcode_dispatch::fbw_finish_payload_take() {
                    // A top-level `void_return/` stashes a `Type::Void`-marked
                    // payload: the portal exits with no value, so build a
                    // FINISH with empty args.  The compile pipeline maps an
                    // empty `finish_arg_types` to `done_with_this_frame_descr_void`
                    // (pyjitpl.rs `done_with_this_frame_descr_from_types`),
                    // matching the trait tracer's `BC_VOID_RETURN` action.
                    Some((_, majit_ir::Type::Void)) => TraceAction::Finish {
                        finish_args: vec![],
                        finish_arg_types: vec![],
                        exit_with_exception: false,
                    },
                    Some((finish_value, finish_type)) => TraceAction::Finish {
                        finish_args: vec![finish_value],
                        finish_arg_types: vec![finish_type],
                        exit_with_exception: false,
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
                // compiles nor aborts this session again — the trait-leg
                // equivalent is `trace_step_result_to_action`'s
                // `compile_trace_success_pending()` branch.
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
            // Structural walker limitations recur identically on every
            // retrace of this location (the same jitcode walked from the same
            // entry produces the same error), so route the key's retraces to
            // the trait tracer (`FBW_DECLINED_KEYS` → the trait leg of
            // `trace_bytecode`) instead of thrashing futile deep re-walks —
            // each of which executes the body's residual calls concretely
            // before failing at the unsupported resume / exception / closure
            // shape.  Permanently blacklisting (`AbortPermanent` →
            // `DONT_TRACE_HERE`) is wrong here: it leaves the location
            // interpreting forever (a try-protected raise in a hot loop
            // deopt-storms past any timeout), and upstream never marks a
            // location untraceable on an abort (pyjitpl.py:2392
            // aborted_tracing).  These are the multi-session-blocked
            // shapes (resume snapshot #124, exception-handler resume #51c,
            // closure NULL-self #60, unported raise marker, a residual
            // arg register the walk never binds); other errors retain the
            // plain `Abort` without declining so a capability that lands
            // mid-run can still pick the location up.
            use crate::jitcode_dispatch::DispatchError as DE;
            crate::jitcode_dispatch::census_record(e.variant_name());
            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                eprintln!("[fbw-abort] start_pc={start_pc} Err={e:?}");
            }
            match e {
                // A kept-stack branch guard whose not-taken arm reads an
                // unrestorable boxed Ref register miscompiles identically in
                // the trait leg, so re-routing there (the `fbw_decline` +
                // recoverable `Abort` path below) would crash there too.
                // Mark the location `DONT_TRACE_HERE` so it interprets
                // permanently — correct, matching the pre-#416/#420 decline.
                DE::BranchGuardUnrestorableKeptStackPermanent { .. } => TraceAction::AbortPermanent,
                // #57 (Finding #1): a non-journalable in-place container mutation
                // in a FOR_ITER body cannot be rolled back on abort, so this
                // location can never trace soundly — interpret it permanently
                // (the loop runs correctly under the interpreter).
                DE::InplaceContainerMutationUnsupported { .. } => TraceAction::AbortPermanent,
                DE::AbortPermanentMarkerReached { .. }
                | DE::GuardSnapshotVableUntyped { .. }
                | DE::MayForceNullRefArgUnsupported { .. }
                | DE::BranchGuardKeptStackUnsupported { .. }
                | DE::NonStandardVableFinishPortalUnsupported { .. }
                | DE::LoopBearingCalleeInlineUnsupported { .. }
                | DE::UnfoldableListAppendResidualUnsupported { .. }
                | DE::ResidualCallArgUnbound { .. } => {
                    fbw_decline(crate::driver::make_green_key(w_code, start_pc));
                    TraceAction::Abort
                }
                // #68 multiframe (`PYRE_FBW_INLINE_MULTIFRAME`): a data-dependent
                // `goto_if_not` whose branch input is not concrete at trace-time
                // recurs identically on every retrace of this entry (the same
                // jitcode walked from the same start_pc reaches the same
                // non-concrete branch operand).  Relaxing the inline predicate
                // lets a portal trace (e.g. a callee independently traced as its
                // own origin) walk PAST its prior `LoopBearing` decline and reach
                // such a branch, which would otherwise re-trace unbounded (each
                // re-walk executes the body's residual calls before failing) —
                // a slowdown worse than the trait leg.  Decline it permanently to
                // the trait leg, mirroring the default path's behavior for the
                // same location.  Gated on the flag so the default path's plain
                // `Abort` (a capability landing mid-run can still pick it up) is
                // byte-identical.
                DE::GotoIfNotValueNotConcrete { .. }
                    if crate::jitcode_dispatch::fbw_inline_multiframe_enabled() =>
                {
                    fbw_decline(crate::driver::make_green_key(w_code, start_pc));
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
            TraceAction::Abort
        }
    }
}

/// Issue #73 walker-as-tracer foundation probe.
///
/// Dumps the per-CodeObject JitCode body that the walker-as-tracer must
/// walk for `miframe.pc == jitcode_pc` (the precondition for retiring
/// `pc_map`).  The per-CodeObject JitCode is built BEFORE this point by
/// `register_portal_jitdriver` → `make_jitcodes`
/// (`pyre/pyre-jit/src/eval.rs:3924`), so it is available here.
///
/// Read-only: logs the body op stream + entry offset, mutates nothing.
fn dump_perfn_jitcode_for_trace(w_code: *const (), start_pc: usize) {
    let Some(pjc) = crate::state::pyjitcode_for_code(w_code) else {
        eprintln!("[perfn-jitcode] no per-CodeObject PyJitCode for code={w_code:?}");
        return;
    };
    let code = pjc.jitcode.code.as_slice();
    let entry = pjc.resume_jitcode_pc_for(start_pc);
    eprintln!(
        "[perfn-jitcode] code_len={} pc_map_len={} start_pc={} entry_jitcode_pc={:?} \
         num_regs_r={} num_regs_i={} num_regs_f={} portal_frame_reg={} portal_ec_reg={} \
         built_as_portal={}",
        code.len(),
        pjc.metadata.pc_map.len(),
        start_pc,
        entry,
        pjc.jitcode.num_regs_r(),
        pjc.jitcode.num_regs_i(),
        pjc.jitcode.num_regs_f(),
        pjc.metadata.portal_frame_reg,
        pjc.metadata.portal_ec_reg,
        pjc.metadata.built_as_portal,
    );
    let mut count = 0usize;
    let mut last_next = 0usize;
    let mut histogram: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for op in crate::jitcode_runtime::decoded_ops(code) {
        if count < 80 {
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
}
