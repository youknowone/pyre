#![allow(
    dead_code,
    unpredictable_function_pointer_comparisons,
    unused_imports,
    unused_mut,
    unused_variables
)]

//! `majit-meta`: Meta-tracing automation layer for the majit JIT framework.
//!
//! Provides [`MetaInterp`] — a high-level JIT engine that handles the full
//! lifecycle: warm counting → tracing → optimization → compilation → execution.
//!
//! Interpreter authors only need to:
//! 1. Call [`MetaInterp::on_back_edge`] at backward jumps
//! 2. Record IR ops via [`TraceCtx`] during tracing
//! 3. Provide state extraction/restoration logic
//!
//! Everything else (constant management, FailDescr/CallDescr creation,
//! optimizer invocation, backend compilation, I/O buffering) is automated.
//!
//! Most modules below mirror `rpython/jit/metainterp/*.py` by file stem.
//! Local Rust boundaries are kept only where the upstream structure is
//! split across crates or Python runtime machinery:
//!
//! * `jit` is the user-facing half of `rpython/rlib/jit.py`; the
//!   translator half lives in `majit_translate::rlib::jit`.
//! * `call_descr` holds runtime call-descr constructors for the
//!   `call.py` / backend `calldescrof` surface.
//! * `box_trace` holds pyre's boxed primitive trace helper shared by
//!   `pyre-jit` and `pyre-jit-trace`.
//! * `cpu` re-exports the backend `model.py::AbstractCPU` surface
//!   threaded through metainterp optimizers.
//! * `io_buffer`, `jit_state`, `trace_ctx`, and `parity` are pyre
//!   runtime/test boundaries with no same-named upstream file.
//! * `jitcode` and `recorder` are transitional runtime ABI boundaries
//!   around canonical translate-side `jitcode.py` / `opencoder.py`
//!   ports; their module docs describe the remaining migration path.

extern crate self as majit_metainterp;

use majit_ir::{OpRef, Type};

/// Runtime surrogate for RPython's lltype `STRUCT` identity.
///
/// Descriptor caches are process-global, so this only needs to be stable for
/// the lifetime of this process. Rust's `TypeId` is exactly that identity and
/// is independent of whether callers spell the type with an absolute or a
/// relative module path. Keep GC-managed and raw layouts distinct, matching
/// RPython's distinct `GcStruct(T)` / `Struct(T)` lltypes.
#[doc(hidden)]
pub fn __pyre_struct_type_id<T: 'static>(is_gc_managed: bool) -> u64 {
    use std::any::TypeId;
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    TypeId::of::<T>().hash(&mut hasher);
    if !is_gc_managed {
        "raw".hash(&mut hasher);
    }
    hasher.finish()
}

pub mod blackhole;
pub mod box_trace;
pub(crate) mod call_descr;
pub(crate) mod compile;
pub mod counter;
pub use majit_backend::model as cpu;
pub use majit_ir::Value;
pub use majit_ir::debug;
pub mod executor;
pub mod gc;
pub mod graphpage;
pub mod greenfield;
pub mod heapcache;
pub mod history;
pub(crate) mod io_buffer;
pub mod jit;
mod jit_state;
pub mod jitcode;
mod jitdriver;
pub mod jitexc;
pub mod jitprof;
pub mod logger;
pub mod memmgr;
pub mod opencoder;
pub mod optimize;
pub mod optimizeopt;
pub(crate) mod parity;
mod pyjitpl;
pub mod quasiimmut;
pub mod recorder;
pub mod resoperation;
pub mod resume;
pub(crate) mod ruleopt;
pub mod support;
mod trace_ctx;
pub mod virtualizable;
pub mod virtualref;
pub mod walkvirtual;
pub mod warmspot;
pub mod warmstate;

pub use call_descr::{
    CANNOT_RAISE_NO_HEAP_EFFECT_INFO, ELIDABLE_CANNOT_RAISE_EFFECT_INFO, ELIDABLE_EFFECT_INFO,
    ELIDABLE_OR_MEMERROR_EFFECT_INFO, EffectInfoSlot, INT_PY_DIV_EFFECT_INFO,
    INT_PY_MOD_EFFECT_INFO, LOOPINVARIANT_EFFECT_INFO, cannot_raise_effect_info,
    default_effect_info, effect_info_for_slot, forces_virtual_or_virtualizable_effect_info,
    make_call_assembler_descr, make_call_assembler_descr_with_vable, make_call_descr,
    make_call_descr_from_target_slot, make_call_descr_with_effect, nursery_alloc_effect_info,
};
pub use compile::{
    make_fail_descr, make_fail_descr_typed, make_finish_fail_descr_typed,
    make_resume_guard_descr_range_foriter,
};
pub use io_buffer::{
    emit_commit_io, encode_decimal_i64, io_buffer_commit, io_buffer_discard, io_buffer_write,
    io_buffer_write_fmt, jit_write_number_i64, jit_write_utf8_codepoint,
};
pub use jit_state::{
    DeoptMaterializationCache, JitState, PendingFieldWriteLayout, ResidualVirtualizableSync,
    ResumeDataResult, bridge_decode_red,
};
pub use jitcode::{
    BC_GOTO, JitArgKind, JitCallArg, JitCode, JitCodeBuilder, LivenessInfo, RuntimeBhDescr, insns,
    live_slots_for_state_field_jit, set_global_build_descr_pool,
};
pub use jitdriver::{
    DeclarativeJitDriver, JitDriver, JitDriverStaticData, TraceContinuationSuspendGuard,
    trace_continuation_suspended,
};
pub use majit_backend::CompiledTraceInfo;
pub use pyjitpl::{eval_binop_f, eval_binop_i, eval_float_cmp, eval_unary_f, eval_unary_i};
// Re-export the canonical translate-side Assembler so macro-emitted
// state-field JIT setup (e.g. `__JitMeta::install_canonical_liveness`)
// can build a fresh Assembler without forcing each user crate to
// declare a `majit-translate` dependency.  The same pattern is used
// for `JitCode` / `BhDescr` re-exports above (`jitcode/mod.rs:4`).
pub use majit_translate::codewriter::assembler::Assembler;
pub use parity::{TraceParityCase, assert_trace_parity, normalize_ops, normalize_trace};
pub use pyjitpl::{
    BackEdgeAction, BridgeRetraceResult, ClosureRuntime, ClosureRuntimeWithResolver,
    CompileOutcome, CompiledExitLayout, CompiledTerminalExitLayout, CompiledTraceLayout,
    DeadFrameArtifacts, DetailedDriverRunOutcome, InlineDecision, JitCodeMachine, JitCodeRuntime,
    JitCodeSym, JitHooks, JitStats, MIFrame, MIFrameStack, MetaInterp, MetaInterpGlobalData,
    MetaInterpStaticData, RawCompileResult, StandaloneFrameStack, build_state_field_snapshot,
    call_int_function, call_ref_function, call_void_function, counters,
    struct_fields_write_effect_info, trace_jitcode, trace_jitcode_from_merge_point,
    trace_jitcode_with_args, trace_jitcode_with_args_and_runtime,
};
pub use quasiimmut::QuasiImmut;
pub use trace_ctx::BridgeInlineCarrier;
pub use trace_ctx::GreenBox;
pub use trace_ctx::MergePoint;
pub use trace_ctx::ReconstructRecipe;
pub use trace_ctx::TraceCtx;

/// Compute green key from code pointer and PC.
/// Must use the same hash as the front-end's make_green_key — the full
/// `JitCell.get_uhash` over the pypyjit green tuple, `is_being_profiled`
/// folded to 0 (warmstate.py:584-593).
pub fn green_key_from_code_ptr(code_ptr: usize, pc: usize) -> u64 {
    majit_ir::pypyjit_greenkey_uhash(pc, false, code_ptr as u64)
}

/// Whether `MAJIT_LOG` is set, cached at first access.
///
/// `std::env::var_os` acquires a global env lock and walks the env table on
/// every call. The flag never changes after process startup, so checking it
/// from hot dispatch paths (e.g. `run_compiled_code_inner` per bridge hop)
/// shows up in profiles. The `LazyLock` caches the boolean.
pub fn majit_log_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("MAJIT_LOG").is_some());
    *ENABLED
}

/// Strict JIT mode: a non-`InvalidLoop` panic during compilation is a bug and
/// must fail loudly rather than silently degrade to the interpreter and mask
/// the bug behind correct output. Enabled in debug builds (`cargo test`) and
/// whenever `MAJIT_STRICT` is set (release benches / CI); off in plain release
/// so production keeps graceful degradation. Cached like `majit_log_enabled`.
pub fn jit_strict_mode() -> bool {
    static STRICT: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        cfg!(debug_assertions) || std::env::var_os("MAJIT_STRICT").is_some()
    });
    *STRICT
}

// ── Cached diagnostic env-var helpers ────────────────────────────────
//
// Each env var is read once and cached via OnceLock so hot paths
// (back-edge, guard-failure, optimizer) never re-acquire the global
// env lock.

pub fn closedbg_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_CLOSEDBG").is_some())
}

pub fn bh_debug_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_BH_DEBUG").is_some())
}

pub fn callee_rca_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("PYRE_CALLEE_RCA").is_some())
}

pub fn nbody_debug_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("PYRE_NBODY_DEBUG").is_some())
}

pub fn mptrace_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_MPTRACE").is_some())
}

pub fn pcseq_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_PCSEQ").is_some())
}

pub fn tldbg_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_TLDBG").is_some())
}

pub fn heapdbg_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_HEAPDBG").is_some())
}

/// Per-op trace of `run_to_end`'s dispatch loop (frame depth, pc, raw opcode).
/// Diagnostic for pinpointing the op that faults a hardware-signal crash
/// (SIGBUS/SIGSEGV) which `catch_unwind` cannot capture.
pub fn optrace_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_OPTRACE").is_some())
}

pub fn diag_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_DIAG").is_some())
}

thread_local! {
    static PORTAL_CRN_HOOK: std::cell::Cell<Option<fn(usize, usize) -> bool>> =
        const { std::cell::Cell::new(None) };
}

/// Install a thread-local hook called when blackhole resume reaches
/// `ContinueRunningNormally`. The hook receives `(target_pc, green_pc)` and
/// returns true if the portal host handled the CRN itself.
pub fn set_portal_crn_hook(hook: Option<fn(usize, usize) -> bool>) {
    PORTAL_CRN_HOOK.with(|cell| cell.set(hook));
}

pub(crate) fn handle_portal_crn_hook(target_pc: usize, green_pc: usize) -> bool {
    PORTAL_CRN_HOOK.with(|cell| cell.get().is_some_and(|hook| hook(target_pc, green_pc)))
}

pub fn log_jtet_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_LOG_JTET").is_some())
}

pub fn smallir_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_SMALLIR").is_some())
}

pub fn log_opt_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_LOG_OPT").is_some())
}

pub fn bridge_debug_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_BRIDGE_DEBUG").is_some())
}

pub fn no_unroll_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("PYRE_NO_UNROLL").is_some())
}

/// `PYRE_ORIGINAL_BOXES`: default true, only disabled by `0` or `false`.
pub fn original_boxes_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| match std::env::var_os("PYRE_ORIGINAL_BOXES") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => true,
    })
}

pub fn stall_window() -> u64 {
    static VAL: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
        std::env::var("MAJIT_STALL_WINDOW")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000_000)
    });
    *VAL
}

pub fn step_limit() -> u64 {
    static VAL: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
        std::env::var("MAJIT_STEP_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8_000_000)
    });
    *VAL
}

/// Result of tracing a single instruction.
///
/// Returned by the interpreter's `trace_instruction()` function
/// to indicate what the framework should do next.
#[derive(Debug)]
pub enum TraceAction {
    /// Continue tracing the next instruction.
    Continue,
    /// reached_loop_header() compiled the current trace into an existing
    /// target and tracing must stop immediately.
    ///
    /// RPython parity: pyjitpl.py says compile_trace() "raises in case it
    /// works". pyre surfaces that control-flow edge explicitly.
    CompileTrace,
    /// Close the loop (back-edge to header detected).
    CloseLoop,
    /// Close the loop with explicit jump arguments supplied by the tracer.
    ///
    /// RPython parity: the tracer can also pass the explicit loop-header PC
    /// (the backward-jump target / reached loop header).  This lets the
    /// tracing context retarget its green key from the true merge point,
    /// instead of trying to recover it later from virtualizable state.
    CloseLoopWithArgs {
        jump_args: Vec<OpRef>,
        loop_header_pc: Option<usize>,
    },
    /// Finish the trace with terminal output values.
    ///
    /// `exit_with_exception = true` maps to
    /// `pyjitpl.py:3238 MetaInterp.compile_exit_frame_with_exception` —
    /// the FINISH uses `sd.exit_frame_with_exception_descr_ref` and the
    /// classifier routes to `JitException::ExitFrameWithExceptionRef`.
    /// `false` maps to
    /// `pyjitpl.py:3198 MetaInterp.compile_done_with_this_frame` —
    /// FINISH uses `sd.done_with_this_frame_descr_<kind>`.
    Finish {
        finish_args: Vec<OpRef>,
        finish_arg_types: Vec<Type>,
        exit_with_exception: bool,
    },
    /// Close and compile a segmented loop (force_finish_trace).
    /// pyjitpl.py:1622 _create_segmented_trace_and_blackhole parity.
    /// The trace has GUARD_ALWAYS_FAILS + unreachable FINISH appended.
    /// compile_simple_loop inserts a LABEL at entry for bridge attachment.
    SegmentedLoop,
    /// Abort the current trace (recoverable — may retry later).
    Abort,
    /// Abort the current trace permanently (never trace this location again).
    AbortPermanent,
    /// A loop back-edge was reached inside an inline callee frame whose
    /// loop already has compiled code (opimpl_jit_merge_point
    /// portal_call_depth>0, pyjitpl.py:1579-1602). The metainterp must
    /// pop the inline frame (finishframe(None)) and record a
    /// CALL_ASSEMBLER into the loop token from the parent frame
    /// (do_recursive_call assembler_call=True), then continue tracing
    /// the parent (ChangeFrame).
    RecursiveCallAssembler { green_key: u64, target_pc: usize },
}

/// Marker macro for the tracing merge point.
///
/// When used with `#[jit_interp]`, this is replaced with `driver.merge_point(...)`.
/// When used standalone, this is a no-op (interpreter runs without tracing).
#[macro_export]
macro_rules! jit_merge_point {
    () => {};
    ($($tt:tt)*) => {};
}

/// Marker macro for the back-edge entry point.
///
/// When used with `#[jit_interp]`, this is replaced with `driver.back_edge(...)`.
/// When used standalone, this is a no-op.
#[macro_export]
macro_rules! can_enter_jit {
    ($($tt:tt)*) => {};
}

/// Marker macro for a recursive portal re-entry (a self-recursive JIT call).
///
/// `recursive_portal_call!(driver, green0, green1, ...)` re-enters the
/// enclosing `#[jit_interp]` portal with the given green key (the greens in
/// jitdriver declaration order). It is the explicit-intrinsic analog of
/// tl.py:177 `res = interp(code, pc + offset)` and of the codewriter's
/// `recursive_call_*` opcode (jtransform.py:522 `handle_recursive_call`,
/// recognised upstream by `funcptr is jd.portal_runner_ptr`, call.py:363).
///
/// Inside `#[jit_interp]` the proc macro rewrites every occurrence:
/// - the transformed (concrete) function calls the `recursive_entry`
///   function declared in the attribute, forwarding the greens positionally;
/// - the dispatch JitCode emits `BC_RECURSIVE_CALL_*`, which the metainterp
///   routes through the inline / CALL_ASSEMBLER / residual decision seams.
///
/// So this `macro_rules!` body is never expanded in a correctly-configured
/// portal; it fails loud if the intrinsic is used without a `recursive_entry`
/// declaration (or outside `#[jit_interp]`).
#[macro_export]
macro_rules! recursive_portal_call {
    ($($tt:tt)*) => {
        ::core::compile_error!(
            "recursive_portal_call! is only valid inside a #[jit_interp] portal \
             declaring `recursive_entry = <fn path>`"
        )
    };
}

/// Assure the JIT that `func(args...)` will produce `result`.
/// `func` must be an elidable function.
///
/// rlib/jit.py:1224 — `record_known_result(result, func, *args)`
///
/// At runtime (non-JIT), verifies `func(args) == result` (debug builds).
/// The jitcode_lower proc-macro intercepts this macro invocation and
/// emits a `record_known_result_{i|r}` opcode with func and args visible
/// as separate operands — matching RPython's rtyper decomposition.
///
/// Usage: `record_known_result!(result, my_elidable_fn, arg1, arg2)`
#[macro_export]
macro_rules! record_known_result {
    ($result:expr, $func:path $(, $arg:expr)*) => {
        // rlib/jit.py:1229-1232 — untranslated consistency check
        debug_assert_eq!(
            $func($($arg),*), $result,
            "record_known_result: func(...) != result"
        );
    };
}

/// rlib/jit.py:1301 — `conditional_call(condition, function, *args)`
///
/// At runtime: `if condition { function(args...) }`.
/// The jitcode_lower proc-macro intercepts this macro invocation and
/// emits a `conditional_call_ir_v` opcode with func and args as
/// separate operands — matching RPython's ConditionalCallEntry decomposition.
///
/// Usage: `conditional_call!(cond, my_func, arg1, arg2)`
#[macro_export]
macro_rules! conditional_call {
    ($condition:expr, $func:path $(, $arg:expr)*) => {
        if $condition {
            $func($($arg),*);
        }
    };
}

/// rlib/jit.py:1322 — `conditional_call_elidable(value, function, *args)`
///
/// At runtime: `if value is falsy { value = function(args...) }; return value`.
/// The jitcode_lower proc-macro intercepts this macro invocation and
/// emits a `conditional_call_value_ir_{i|r}` opcode with func and args as
/// separate operands.
///
/// Usage: `let v = conditional_call_elidable!(cached, compute_fn, arg1, arg2)`
#[macro_export]
macro_rules! conditional_call_elidable {
    ($value:expr, $func:path $(, $arg:expr)*) => {{
        let __val = $value;
        if __val == 0 {
            $func($($arg),*)
        } else {
            __val
        }
    }};
}

/// Hash a green key from i64 slice values, all-Int convention.
///
/// Uses the same algorithm as [`GreenKey::hash_u64`](majit_ir::GreenKey::hash_u64),
/// so callers can compute a key hash without constructing a full `GreenKey`.
/// warmstate.py:584-593 `JitCell.get_uhash` — Int-only path.
///
/// Callers that have non-Int greens (Float / Ref) must use
/// [`green_key_hash_typed`] instead; the per-type
/// `equal_whatever`/`hash_whatever` differs from the Int default and a
/// bare-i64 hash would collide with an Int-typed key carrying the same bits.
#[inline]
pub fn green_key_hash(values: &[i64]) -> u64 {
    majit_ir::GreenKey::new(values.to_vec()).hash_u64()
}

/// Hash a green key from `(i64 bits, GreenType)` slices.
///
/// `warmstate.py:575 _green_args_spec` keys per-type
/// `equal_whatever`/`hash_whatever` off the green's lltype, so a Float
/// green hashes as `f64::from_bits(bits)`-aware and a Ref green hashes
/// as identity over the pointer bits.  Mirrors the typed schema that
/// `#[jit_interp]` macro-emitted code now produces via
/// `GreenKey::with_types`.
#[inline]
pub fn green_key_hash_typed(values: &[i64], types: &[majit_ir::GreenType]) -> u64 {
    debug_assert_eq!(values.len(), types.len());
    majit_ir::GreenKey::with_types(values.to_vec(), types.to_vec()).hash_u64()
}

// ── we_are_jitted / JIT mode flag ──
// Re-exported from majit-codegen so both meta and backend can access it.
pub use majit_backend::{JittedGuard, set_jitted, we_are_jitted};

// ── rstack criticalcode hooks ──
// rpython/translator/c/src/stack.h:42-43 LL_stack_criticalcode_start/stop.
// Used by blackhole_from_resumedata / handle_async_forcing /
// initialize_state_from_guard_failure to suppress StackOverflow during
// critical sections that would leave virtual references dangling.
//
// The actual implementation lives in pyre-interpreter (the interpreter
// owns the rpy_stacktoobig struct). majit-metainterp cannot depend on
// pyre-interpreter directly — pyre depends on majit, not the other way
// — so the interpreter registers the two hooks at startup.
use std::sync::OnceLock;

static CRITICALCODE_START_FN: OnceLock<fn()> = OnceLock::new();
static CRITICALCODE_STOP_FN: OnceLock<fn()> = OnceLock::new();
static STACK_ALMOST_FULL_FN: OnceLock<fn() -> bool> = OnceLock::new();

/// Register the `_stack_criticalcode_start` / `_stack_criticalcode_stop`
/// hooks the interpreter implements. Called once at JIT install time.
pub fn register_criticalcode_hooks(start: fn(), stop: fn()) {
    let _ = CRITICALCODE_START_FN.set(start);
    let _ = CRITICALCODE_STOP_FN.set(stop);
}

/// Register the `rstack.stack_almost_full` hook the interpreter
/// implements against its `PYRE_STACKTOOBIG` budget. Called once at
/// JIT install time. When no hook is registered, [`stack_almost_full`]
/// returns `false` — matching RPython's untranslated fallback in
/// `rpython/rlib/rstack.py:76-77`.
pub fn register_stack_almost_full_hook(f: fn() -> bool) {
    let _ = STACK_ALMOST_FULL_FN.set(f);
}

/// Diagnostic-only guard-failure → bridge-trace gate tallies, read out via
/// the `pyre_jit_mc_diag` guest export. Index legend: 0 = must_compile_with_values
/// entered, 1 = declined_bridge_guards short-circuit, 2 = descr_addr==0 skip,
/// 3 = status-busy skip, 4 = jitcounter FIRED (true), 5 = stack_almost_full
/// returned true, 6 = start_retrace_from_guard entered, 7 = start_retrace bailed
/// (source loop evicted: compiled_loops miss), 8 = compile_bridge entered (trace
/// closed → backend request path), 9 = compile_bridge InvalidLoop discard, 10 =
/// compile_bridge retrace_requested return, 11 = compile_bridge arity giveup
/// return (JUMP args != target LABEL args), 12 = start_bridge_tracing entered,
/// 13 = sbt early: descr not FailDescr, 14 = sbt early: no owning jct, 15 = sbt
/// early: no compiled_meta, 16 = sbt early: !can_trace, 17 = sbt early:
/// fail_values too short.
pub static MC_DIAG: [std::sync::atomic::AtomicU64; 18] = {
    const Z: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    [Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z]
};

/// Read an `MC_DIAG` tally (saturating). Surfaced via `pyre_jit_mc_diag`.
pub fn mc_diag(i: usize) -> u64 {
    MC_DIAG
        .get(i)
        .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(0)
}

#[inline]
pub fn mc_diag_bump(i: usize) {
    MC_DIAG[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// rpython/rlib/rstack.py:75-90 `stack_almost_full`. Returns `true` if
/// the stack is more than 15/16ths full against the recursion-limit
/// budget. Dispatches to the interpreter-registered hook; in tests or
/// standalone binaries without the interpreter's stack-check layer,
/// returns `false` (rstack.py:76-77 `if not we_are_translated: return
/// False`).
#[inline]
pub fn stack_almost_full() -> bool {
    if let Some(f) = STACK_ALMOST_FULL_FN.get() {
        let r = f();
        if r {
            mc_diag_bump(5); // stack_almost_full returned true
        }
        r
    } else {
        false
    }
}

/// rpython/translator/c/src/stack.h:42 `LL_stack_criticalcode_start`.
/// No-op if the hook is not registered (tests / standalone binaries
/// that don't install the interpreter's stack-check layer).
#[inline]
pub fn criticalcode_start() {
    if let Some(f) = CRITICALCODE_START_FN.get() {
        f();
    }
}

/// rpython/translator/c/src/stack.h:43 `LL_stack_criticalcode_stop`.
#[inline]
pub fn criticalcode_stop() {
    if let Some(f) = CRITICALCODE_STOP_FN.get() {
        f();
    }
}

/// RAII guard wrapping [`criticalcode_start`] / [`criticalcode_stop`].
///
/// RPython's `rstack._stack_criticalcode_start()` uses try/finally to
/// guarantee the matching `_stop()` runs on every exit path (including
/// exceptions). Rust's equivalent is `Drop`: this guard calls
/// `criticalcode_stop()` in its destructor so ordinary returns,
/// `?`-propagated errors, and `panic!` unwind all re-enable the
/// `report_error` flag. Matches rpython/jit/metainterp/resume.py:1315
/// + rpython/jit/metainterp/pyjitpl.py:3281 +
/// rpython/jit/metainterp/compile.py:976 `try/finally` semantics.
pub struct CriticalCodeGuard {
    _private: (),
}

impl CriticalCodeGuard {
    /// Enter a critical section. The returned guard must be held for
    /// the duration of the section; dropping it re-enables stack-
    /// overflow reporting, even if the drop is triggered by panic
    /// unwinding.
    #[inline]
    #[must_use = "CriticalCodeGuard re-enables stack overflow reporting only on drop — binding it to `_` drops it immediately, defeating the guard"]
    pub fn enter() -> Self {
        criticalcode_start();
        CriticalCodeGuard { _private: () }
    }
}

impl Drop for CriticalCodeGuard {
    #[inline]
    fn drop(&mut self) {
        criticalcode_stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn green_key_hash_deterministic() {
        let a = green_key_hash(&[10, 20]);
        let b = green_key_hash(&[10, 20]);
        assert_eq!(a, b);
    }

    #[test]
    fn green_key_hash_different_values() {
        let a = green_key_hash(&[10, 20]);
        let b = green_key_hash(&[10, 21]);
        assert_ne!(a, b);
    }

    #[test]
    fn green_key_hash_matches_green_key() {
        let hash = green_key_hash(&[42, 7]);
        let gk = majit_ir::GreenKey::new(vec![42, 7]);
        assert_eq!(hash, gk.hash_u64());
    }

    #[test]
    fn green_key_hash_typed_diverges_from_all_int_for_float_greens() {
        let bits = (3.14f64).to_bits() as i64;
        let untyped = green_key_hash(&[bits]);
        let typed = green_key_hash_typed(&[bits], &[majit_ir::GreenType::Float]);
        // hash_whatever(Float, bits) vs hash_whatever(Int, bits) — distinct
        // per `warmstate.py:566 _green_args_spec` per-type lookup.
        assert_ne!(
            untyped, typed,
            "Float-typed hash must not collide with Int-typed hash on the same bits",
        );
    }

    #[test]
    fn green_key_hash_typed_matches_with_types() {
        let bits = (3.14f64).to_bits() as i64;
        let hash = green_key_hash_typed(
            &[bits, 42],
            &[majit_ir::GreenType::Float, majit_ir::GreenType::Int],
        );
        let gk = majit_ir::GreenKey::with_types(
            vec![bits, 42],
            vec![majit_ir::GreenType::Float, majit_ir::GreenType::Int],
        );
        assert_eq!(hash, gk.hash_u64());
    }
}
pub(crate) mod resumecode;
