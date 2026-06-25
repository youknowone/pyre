//! Public trace entrypoint for `pyre`'s JIT portal.
//!
//! `trace_bytecode` drives the authoritative full-body walk
//! (`full_body_walk_trace`): the walker-as-tracer that walks the per-CodeObject
//! JitCode body via `MIFrame::trace_code_step`, combining symbolic IR recording
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
    // RPython MetaInterp._interpret() parity: root frame owns a concrete
    // PyFrame snapshot. MetaInterp drives both symbolic tracing AND
    // concrete execution — the interpreter does not run during tracing.
    //
    // KNOWN DIVERGENCE (live miscompile, memory
    // `cf-executor-into-walker-epic-2026-06-08`): tracing runs on a SNAPSHOT
    // (`snapshot_for_tracing`), not the real frame, and the compiled loop
    // re-runs the traced iteration from the loop header.  RPython's `_interpret`
    // advances the SINGLE real frame so the compiled loop resumes AFTER the
    // traced iterations.  For inline-frame SHARED-heap STOREs this re-run
    // double-applies (see `metainterp::concrete_execute_step`).
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
    let mp_pc = crate::jitcode_runtime::decoded_ops(code)
        .filter(|op| op.opname == "jit_merge_point" && op.pc < entry)
        .map(|op| op.pc)
        .max()?;
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
        match loop_header_merge_point_regs(pjc.jitcode.code.as_slice(), entry) {
            Some((gr, rr)) => {
                if let Some(&r) = gr.first() {
                    seed(r, pycode_box);
                }
                if let Some(&r) = rr.first() {
                    seed(r, frame_box);
                }
                if let Some(&r) = rr.get(1) {
                    seed(r, ec_box);
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
                if portal_frame_reg != u16::MAX {
                    seed(portal_frame_reg as u8, frame_box);
                } else {
                    seed(1, frame_box);
                }
                if portal_ec_reg != u16::MAX {
                    seed(portal_ec_reg as u8, ec_box);
                } else {
                    seed(2, ec_box);
                }
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
        if is_bridge_trace {
            if let Some(ref bridge_stack) = sym.bridge_stack_oprefs {
                let nl = sym.nlocals;
                for (i, &opref) in bridge_stack.iter().enumerate() {
                    if !opref.is_none() {
                        seed((nl + i) as u8, opref);
                    }
                }
            }
        }
        v
    };

    // Per-fn descr-pool plumbing: the per-CodeObject body
    // resolves `d`/`j` descr operands through its OWN runtime pool
    // (`pjc.jitcode.exec.descrs`, `Vec<RuntimeBhDescr>`), NOT the global
    // `all_descr_refs()`.  Build the index-parallel adapted `descr_refs`
    // and resolve `inline_call` callee jitcodes through the same pool.
    use majit_metainterp::jitcode::RuntimeBhDescr;
    // The per-CodeObject JitCode lives in the process-global jitcode
    // registry (installed by `install_jitcodes` before tracing); `pjc` is
    // an `Arc` clone of that data, so the descr pool (and the callee
    // jitcode bodies it references) outlive this diagnostic walk.
    // Extend the borrow to `'static` so the `'static`-bodied
    // `SubJitCodeBody` from `sub_jitcode_lookup` type-checks — mirrors the
    // production arm-entry borrow extension at `trace_opcode.rs:6735`.
    let perfn_descrs: &'static [RuntimeBhDescr] =
        unsafe { &*(pjc.jitcode.exec.descrs.as_slice() as *const [RuntimeBhDescr]) };
    let perfn_descr_refs: Vec<majit_ir::DescrRef> = perfn_descrs
        .iter()
        .enumerate()
        .map(|(i, d)| match d {
            RuntimeBhDescr::Descr(bh) => crate::descr::make_descr_from_bh(bh),
            // `inline_call`'s `d` operand resolves the callee through
            // `JitCodeDescr::jitcode_index()` → `sub_jitcode_lookup`.
            // Key the descr by its own pool slot `i` so the per-fn
            // lookup below re-reads `exec.descrs[i].as_jitcode()`.
            RuntimeBhDescr::JitCode(_) => crate::descr::make_jitcode_descr(i),
            // `Call` / `AssemblerToken` pool entries belong to the
            // `BC_CALL_*` / `BC_CALL_ASSEMBLER_*` op families, whose
            // walker handlers read the target straight from the raw
            // per-fn pool (`RawDescrPool::PerFn`), not through this
            // adapted `DescrRef` slot; every `residual_call` `d` slot
            // the codewriter emits is a `Descr(CanonicalBhDescr)` call
            // descr (zero `ResidualCallDescrNotCallDescr` across the
            // bench + synth suites with the walk default-on).  The
            // jitcode-descr stand-in is a fail-loud tripwire: a
            // mis-routed slot surfaces a clean typed error at the first
            // such op instead of mis-dispatching (pinned by the
            // FailDescr-fixture unit test in `jitcode_dispatch.rs`).
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
    let mut walk_result = crate::jitcode_dispatch::dispatch_via_miframe(
        &mut mi,
        code,
        entry,
        &perfn_descr_refs,
        crate::jitcode_dispatch::RawDescrPool::PerFn(perfn_descrs),
        // The diagnostic probe discards the trace (`cut_trace` + Abort)
        // and the bench then runs interpreted, so the walker must NOT be
        // the authoritative executor — executing may-force calls here
        // would corrupt the live frame/iterator state `cut_trace` cannot
        // roll back.  Concrete may-force execution lands with the
        // production flip, not under the probe.
        //
        // (51d diagnosis, 2026-05-29: even with authoritative=true the walk
        // STOPs at the loop `goto_if_not` because the boxed-PyLong compare
        // may-force arg is non-concrete.  Root-caused with PYRE_DIAG_VGAI:
        // the loop body's `getarrayitem_vable_r` (a `LOAD_FAST` of a boxed
        // local) returns `Value::Void` NOT because the virtualizable shadow
        // is wrong — the shadow ENTRY is the correct concrete Ref — but
        // because the VABLE operand register read returns `OpRef::NONE`.  The
        // post-merge-point loop body reads the frame from a LOOP-INPUT
        // register bound by the `jit_merge_point` reds @ pc=94; the probe
        // enters at pc=107 (past the merge point) seeding only r0, so that
        // register stays NONE → `concrete_of_opref(NONE)` = `GcRef(usize::MAX)`
        // sentinel → `is_nonstandard_virtualizable` takes the Void leg.  Fix =
        // seed the live loop-input registers at walk entry, NOT a shadow/
        // stack-depth issue (task #53).  Two further cascade gaps sit above
        // it: a non-pure `CallR` result left symbolic (task #54), and
        // may-force execution — now wired into BOTH residual dispatchers.)
        //
        // Authoritative concrete execution: `false` for the read-only probe
        // (trace discarded → re-executing would corrupt live state); `true`
        // for the production full-body tracer (the walk IS the execution, so
        // there is no double-run and no rollback to miss).
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
        &argboxes_r,
        &[],
        &[],
    );

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
    let terminate_no_replay = crate::jitcode_dispatch::fbw_no_replay_exit_enabled()
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
                DE::AbortPermanentMarkerReached { .. }
                | DE::GuardSnapshotVableUntyped { .. }
                | DE::MayForceNullRefArgUnsupported { .. }
                | DE::BranchGuardKeptStackUnsupported { .. }
                | DE::NonStandardVableFinishPortalUnsupported { .. }
                | DE::LoopBearingCalleeInlineUnsupported { .. }
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
         built_as_portal={} merge_point_pc={:?}",
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
        pjc.merge_point_pc,
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
