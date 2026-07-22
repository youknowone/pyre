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
/// (`walker_capture_multi_frame_inline_snapshot`) unrolls before folding to
/// the `CALL_ASSEMBLER` tail.  Bounded to 1: a depth-≥2 unroll of a tree
/// recursion (e.g. `fib`, whose `n < 2` base-case guard fails per call) does
/// not stay in compiled code — the unrolled trace guard-fails and deopts to
/// the blackhole on essentially every recursive call (measured: two orders of
/// magnitude more blackhole resumes than the folded path, ~20-30× slower).
/// This mirrors `max_unroll_recursion` bounding the same runaway: past the
/// bound the recursive call folds straight to `CALL_ASSEMBLER`
/// (`_opimpl_recursive_call` → `do_recursive_call`, `pyjitpl.py`)
/// rather than continuing to unroll the call tree.  `try_multiframe`
/// (`inline_depth < fbw_max_multiframe_depth()`) therefore only fires at the
/// top inline level by default. (The depth-≥2 blackhole-resume crash that
/// previously blocked this path was a GC-rooting gap in the nested `run()`
/// chain, fixed by rooting the whole pending `nextblackholeinterp` chain across
/// `run()`; raising the bound is now a performance, not a soundness, question.)
pub(crate) fn fbw_max_multiframe_depth() -> usize {
    static DEPTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("PYRE_FBW_MULTIFRAME_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 2)
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

/// `PYRE_FBW_INLINE_MULTIFRAME` (#68): inline branch-bearing callees with a
/// multi-frame guard snapshot instead of declining them to interpretation
/// (`LoopBearingCalleeInlineUnsupported`).  Default-on; `PYRE_FBW_INLINE_MULTIFRAME=0`
/// (or `false`) is the rollback escape hatch.  The multi-frame snapshot
/// encode↔decode contract for walker-emitted callee-frame guards is validated
/// byte-exact (function_calls + corpus) on both backends.
pub(crate) fn fbw_inline_multiframe_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_INLINE_MULTIFRAME") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

/// `PYRE_FBW_NSVABLE_MULTIFRAME` (#73): publish the `_nonstandard_virtualizable`
/// promote guard through the full multi-frame resume chain (each paused caller
/// plus the callee's own coordinate) instead of the single-frame sentinel
/// collapse in [`walker_capture_inline_nonstandard_vable_guard`], which cannot
/// resolve a JitCode resume word and unconditionally aborts every inline
/// sub-walk emit of this guard.  Default-on; `PYRE_FBW_NSVABLE_MULTIFRAME=0`
/// (or `false`) restores the sentinel decline as the rollback escape hatch.
pub(crate) fn fbw_nsvable_multiframe_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_NSVABLE_MULTIFRAME") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

/// `PYRE_FBW_REC_MULTIFRAME` (default ON; `=0`/`false` opts out): route
/// primary-trace self-recursive Python calls through the multiframe inline path
/// while below `PYRE_FBW_MULTIFRAME_DEPTH`, instead of folding immediately to
/// the recursive portal `CALL_ASSEMBLER`.
///
/// RPython parity: `opimpl_recursive_call` / `do_recursive_call`
/// (`pyjitpl.py`) inline within `max_unroll_recursion`; only once the
/// cap is reached does recursion fall back to the assembler-call path.  The
/// prior fold-only default (every self-recursive call cut straight to
/// `CALL_ASSEMBLER`) was the pyre deviation; inlining below the depth bound is
/// the parity behavior, so this is default-on.  The depth bound
/// (`fbw_max_multiframe_depth`, default 1) still caps how deep the inline
/// unrolls before falling back to `CALL_ASSEMBLER`.
pub(crate) fn fbw_rec_multiframe_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_REC_MULTIFRAME") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

/// `PYRE_FBW_BRIDGE_REC_INLINE` (default ON) — on a plain root bridge walk,
/// lift the bridge-trace decline that keeps an exact-integer arithmetic callee a
/// single residual call, letting the bridge inline one self-recursive level
/// exactly as a primary trace does: the call falls through to the self-recursive
/// unroll gate and the multiframe seed instead of returning a residual.  The
/// miscompile hazard it admits — a bridge-inlined int-binop callee's second
/// virtual frame operand stack has no red bridge input, so an overflow/exception
/// resume path can leave a NULL vable stack slot — is contained by the
/// seed-success precondition plus the `n_parents == n_callees` snapshot valve
/// fallbacks.  A/B across the bench corpus is byte-parity clean and adds no
/// `loops_aborted` or `internal_compile_panics`; fib_recursive gains one inlined
/// bridge (guard_failures 407 -> 406, bridges_compiled 2 -> 3) for a ~10% win.
/// `=0`/`false` opts back out.
pub(crate) fn fbw_bridge_rec_inline_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_BRIDGE_REC_INLINE") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

/// Full-portal recursive-call cutover (`PYRE_FBW_REC_MUTUAL_CUTOVER`): at the
/// inline-unroll cap, route a recursive callee (self OR mutual) through
/// `get_assembler_token` → `compile_tmp_callback` (warmstate.py,
/// compile.py) so a not-yet-compiled callee still enters via a real
/// CALL_ASSEMBLER tmp-callback token instead of poisoning the trace with
/// `LoopBearingCalleeInlineUnsupported`.  Mirrors the `build_jit_driver_pair`
/// gate of the same name (eval.rs); default ON, `=0` opts out.
pub(crate) fn fbw_rec_mutual_cutover_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_REC_MUTUAL_CUTOVER") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

/// `PYRE_FBW_LOOP_CALLEE_CA` (general loop-bearing-callee →
/// CALL_ASSEMBLER): when a multi-frame inlined callee sub-walk reaches the
/// callee's own `jit_merge_point` and a compiled loop token already exists
/// for that green key, emit a `CALL_ASSEMBLER` into it (mirror of
/// `opimpl_recursive_call_assembler`) instead of declining the enclosing
/// trace (`JitMergePointGreenKeyUnresolved`). Default-ON;
/// `=0`/`false` opts out.
///
/// The default-ON flip rides the same CALL_ASSEMBLER / residual-executor /
/// virtualizable machinery already shipping default-ON through the
/// self-recursive arm ([`try_walker_call_assembler_self_recursive`],
/// `PYRE_FBW_REC_CA` default-ON). The only extension here is the callee
/// frame shape: a multi-frame inline frame built by
/// `emit_new_pyframe_inline_with_params` that can hold Ref locals, vs the
/// self-recursive arm's int-only `emit_new_pyframe_inline_self_recursive`
/// frame. A four-lens GC-rooting audit established the two frame builders are
/// content-agnostically rooted identically — same `pyframe_size_descr()`,
/// same `pyobject_gcarray_descr()` locals array, same malloc-then-store
/// ordering, the materialized virtualizable frame is JUMP-loop-carried so its
/// slot is in every inner residual-call gcmap (`get_gcmap`), and the runtime
/// `PyFrame`/array GC type registration traces frame->array->elements with no
/// int-vs-ref branch anywhere. A historical GC-stress SEGV (a freed,
/// not-forwarded receiver under nursery pressure) reproduced only on
/// layout-shifting diagnostic-probe builds; on clean binaries it does not
/// reproduce across the GC-stress matrix (r1/r5/r6/r2/r4 × nursery
/// {default,1M,256K,64K,16K,4K} × dynasm+x86, all clean) — consistent with a
/// diagnostic-build layout artifact, and content-agnostic rooting rules out a
/// ref-specific defect in this chain.
pub(crate) fn fbw_loop_callee_ca_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_LOOP_CALLEE_CA") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

/// `PYRE_FBW_VABLE_SCALAR_CA` (default OFF) — sub-mode of
/// [`fbw_loop_callee_ca_enabled`]. When on, the loop-callee
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

/// `PYRE_FBW_RAISE` (default ON) — the FBW walker owns the Python raise/except
/// loop.  The twin NULL-ref guards exempt the trailing `cause` sentinel of a
/// [`PyreHelperKind::RaiseVarargs`] residual so the walker records the raise.
/// Now that the trait tracer is retired, declining instead
/// re-interprets without JIT (a hot raise/except loop would time out), so the
/// walker must own the raise path; `PYRE_FBW_RAISE=0` opts back to declining.
pub(crate) fn fbw_raise_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_RAISE") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

/// `PYRE_FBW_BUILTIN_FOLD` (default ON) — gates the LOAD_GLOBAL cell fold's
/// reachability inside handler-bearing bodies and its builtins-fallback arm.
/// When ON, `dispatch_residual_call_iRd_kind` attempts
/// [`try_walker_load_global_cell_fold`] even when the body contains a
/// `catch_exception` (the fold emits an `ElidableCannotRaise` lookup so the
/// dropped `GUARD_NO_EXCEPTION` is moot for a SUCCESSFUL fold — a declined
/// fold keeps the residual+guard), and the fold resolves names absent from
/// the module dict through `frame.get_builtin()` (e.g. `raise ValueError` /
/// `except ValueError`).  `PYRE_FBW_BUILTIN_FOLD=0` restores the legacy
/// handler-free-only behavior.
pub(crate) fn fbw_builtin_fold_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_BUILTIN_FOLD") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

/// `PYRE_FBW_LOADATTR_FOLD` — gate the full-body-walker LOAD_ATTR fast path
/// ([`try_walker_specialize_load_attr`]).  When on, a monomorphic plain
/// instance-attribute read folds to guards + inline storage read instead of the
/// opaque `getattr_fn` residual.  Default ON (`0`/`false` opts out as the kill
/// switch); verified byte-exact on synth dynasm + cranelift and GC-soak clean
/// under `PYPY_GC_NURSERY=131072` on instance-heavy benches.
pub(crate) fn fbw_loadattr_fold_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_LOADATTR_FOLD") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

/// `PYRE_FBW_STOREATTR_FOLD` — gate the full-body-walker STORE_ATTR fast path
/// ([`try_walker_specialize_store_attr`]).  When on, a plain same-type unboxed
/// integer store folds to guards + a non-forcing raw longlong-list write
/// instead of the forcing `setattr_fn` residual (dropping the force token,
/// vable spill, and the value re-box).  Default ON (`0`/`false` opts out as the
/// kill switch, independent of the read fold since the write executes a
/// concrete heap mutation).
pub(crate) fn fbw_storeattr_fold_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_STOREATTR_FOLD") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

/// `PYRE_FBW_LOADMETHOD_FOLD` — gate the full-body-walker method-cache fold.
/// When on, a monomorphic `obj.method(...)` dispatch folds the LOAD_ATTR
/// method lookup to a constant descriptor plus guards, and folds the paired
/// `load_method_self` residual to its constant binding decision.  Default ON;
/// `0`/`false` opts out as the kill switch.
pub(crate) fn fbw_loadmethod_fold_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_LOADMETHOD_FOLD") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

/// `PYRE_FBW_DELETE_FAST` — gate the full-body-walker DELETE_FAST lowering.
/// Default ON; `0`/`false` opts back into the existing `abort_permanent`
/// marker fallback for unsupported shapes.
pub(crate) fn fbw_delete_fast_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_DELETE_FAST") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

/// `PYRE_FBW_INLINE_NSFOLD` (default ON) — gates resolving an inlined callee's
/// `getfield_vable_r` namespace(idx5)/pycode(idx1) read from the callee's
/// compile-time [`InlineCalleeConsts`] on the MULTIFRAME path (seeded virtual
/// frame), not just the strict path (unseeded frame).  Without it, the seeded
/// virtual frame's vable read misses the heapcache forward — the codewriter's
/// per-fn vable descr identity differs from the seeding descr
/// (`pyframe_w_globals_obj_descr`) — and records a non-const `GetfieldGcR`,
/// leaving the LOAD_GLOBAL fold's namespace operand non-concrete so a
/// loop-bearing inlined callee's `load_global` (e.g. nbody `advance()`'s
/// `len(bodies)`) stays a residual that the nested-unjournaled-residual abort
/// declines.  `=0` restores the strict-path-only behavior.
pub(crate) fn fbw_inline_nsfold_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_INLINE_NSFOLD") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
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

/// `PYRE_FBW_STACK_LIVEREG` (default ON) — for branch-guard operand-stack
/// snapshot slots, prefer the live Ref register (`pyjitpl.py`
/// `get_list_of_active_boxes` reads `self.registers_r[index]`) when the
/// guard PC's per-PC color map proves that color owns the same stack slot.
/// `=0` restores the shadow-first pyre-local order everywhere.
pub(crate) fn fbw_stack_livereg_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_FBW_STACK_LIVEREG") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

thread_local! {
    /// Finish payload stashed by a top-level `*_return` arm under the
    /// `PYRE_FBW_CALL_ASSEMBLER` gate, read back by
    /// [`crate::trace::full_body_walk_trace`] to build a
    /// `TraceAction::Finish` for a loop-free (Finish-terminated) portal.
    ///
    /// `(finish_value, finish_arg_type)` — the re-boxed return value and
    /// its `Type::Ref` portal-exit type.  `None` outside the gated path,
    /// so the default-off walk maps `Terminate -> Abort` exactly as before.
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

/// Whether the Finish-portal compile route is enabled.  Cached so
/// the per-`*_return` read and the `full_body_walk_trace` read see a
/// single consistent value.  Default ON; `PYRE_FBW_CALL_ASSEMBLER=0` opts
/// back into the pre-Finish-portal path (bare
/// `Terminate` -> `Abort`) as a transition escape hatch.
pub(crate) fn fbw_call_assembler_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PYRE_FBW_CALL_ASSEMBLER").as_deref() != Ok("0"))
}

/// Whether the no-replay portal exit is enabled (a loop-free function
/// trace that reached `done_with_this_frame` returns its captured concrete
/// result directly instead of re-running the freshly compiled trace for
/// the SAME invocation).  Default ON; `PYRE_FBW_NO_REPLAY_EXIT=0` opts
/// back into the legacy `ContinueRunningNormally` replay for bisection.
pub(crate) fn fbw_no_replay_exit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PYRE_FBW_NO_REPLAY_EXIT").as_deref() != Ok("0"))
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
    // B3 (`PYRE_FBW_RAISE`): drop any inline-built-exception OpRef keys a
    // prior aborted walk recorded, so they cannot match a same-numbered
    // OpRef minted by this walk's recorder.
    FBW_BUILT_EXC.with(|s| s.borrow_mut().clear());
    // B3 (`PYRE_FBW_RAISE`): drop any unbalanced PUSH_EXC_INFO prev saves a
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
    // `PYRE_FBW_NESTED_RESID_ABORT=0` opts back into the prior
    // (miscompiling) mark-and-replay behavior for A/B.
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        std::env::var_os("PYRE_FBW_NESTED_RESID_ABORT").as_deref()
            != Some(std::ffi::OsStr::new("0"))
    });
    // RPython `do_residual_call` runs the residual executor at any framestack
    // depth (`pyjitpl.py`). Exempt only the self-recursive
    // `CALL_ASSEMBLER` fold's concrete-stamp executor from this pyre-local
    // nested-decline guard, which is for FOREIGN unjournaled residuals.
    let in_selfrec_fold = SELFREC_CA_FOLD_ACTIVE.with(|c| c.get());
    let in_exception_string_inline = EXCEPTION_STRING_INLINE_ACTIVE.with(|c| c.get());
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
    // residual inlines.  The hazard scan is last so the cheap flag checks
    // short-circuit it.
    if enabled
        && !in_selfrec_fold
        && !in_exception_string_inline
        && !ctx.session.borrow().framestack.is_empty()
        && fbw_inline_callee_hazardous(ctx)
    {
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
        return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc });
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

/// Whether an inline callee can be replayed from its caller's CALL boundary
/// without duplicating a live-heap effect.  The inline sub-walk's deopt
/// snapshot does not yet carry its own callee frame, so this is deliberately
/// stricter than ordinary inlining: unknown calls and every live-heap write
/// decline up front.
///
/// A `new_with_vtable/d>r` result is fresh within this body.  Its
/// initialization write is benign only when the target field is immutable;
/// `wrapint` is the important instance (`W_IntObject.intval`).  Freshness may
/// pass through `ref_copy`, but every other Ref-producing instruction clears
/// it, so a later `setfield_gc` cannot accidentally be classified as an
/// initialization of an earlier allocation.
pub(crate) fn fbw_callee_body_side_effect_free(
    body_code: &[u8],
    args_all_numeric: bool,
    num_regs_i: usize,
    constants_i: &[i64],
    callee_descr_refs: &[DescrRef],
) -> bool {
    let mut fresh_ref_regs = [false; u8::MAX as usize + 1];
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return false;
        };

        if d.opname.starts_with("residual_call") {
            let Some(descr_index) = residual_call_descr_index_in_body(body_code, &d) else {
                return false;
            };
            let Some(call_descr) = callee_descr_refs
                .get(descr_index)
                .and_then(|descr| descr.as_call_descr())
            else {
                return false;
            };
            let ei = call_descr.get_extra_info();
            // `ForIterNext` is deliberately not accepted here: it advances the
            // shared heap iterator irreversibly (no journal undo), so replaying
            // a callee that contains it from the caller's CALL boundary would
            // double-consume.  A FOR_ITER-bearing body is declined anyway — its
            // mandatory `GET_ITER` (`MayForce`) predecessor fails this scan
            // first — so this only removes a latent landmine, not live inlines.
            let provably_side_effect_free =
                ei.check_is_elidable() || ei.extraeffect == majit_ir::ExtraEffect::LoopInvariant;
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
                return false;
            }
        } else if d.opname.starts_with("setfield_gc") {
            // Canonical setfield shapes are `r<value>d`: the target ref is
            // operand 0 and the field descr is operand 2.
            let Some(&target_reg) = body_code.get(d.pc + 1) else {
                return false;
            };
            let descr_index = decode_descr_index(body_code, &d, 2);
            let immutable_field = callee_descr_refs
                .get(descr_index)
                .and_then(|descr| descr.as_field_descr())
                .is_some_and(|field| field.is_immutable());
            if !fresh_ref_regs[target_reg as usize] || !immutable_field {
                return false;
            }
        } else if d.opname.starts_with("setarrayitem_gc")
            || d.opname.starts_with("setinteriorfield_gc")
            || d.opname.starts_with("raw_store")
            || d.opname.starts_with("cond_call")
            || d.opname.starts_with("call_assembler")
            || d.opname.starts_with("inline_call")
        {
            // Array/interior/raw stores and non-residual call forms cannot be
            // proven replay-safe from this single callee body.
            return false;
        }

        // The result byte is always the final operand for `>r` forms.
        if d.argcodes.ends_with(">r") {
            let Some(&dst) = body_code.get(d.next_pc.saturating_sub(1)) else {
                return false;
            };
            fresh_ref_regs[dst as usize] = d.key == "new_with_vtable/d>r"
                || (d.key == "ref_copy/r>r"
                    && body_code
                        .get(d.pc + 1)
                        .is_some_and(|src| fresh_ref_regs[*src as usize]));
        }
        pc = d.next_pc;
    }
    true
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
