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
pub mod recorder;
pub mod resoperation;
pub mod resume;
pub mod resume_box_reader;
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
    make_call_assembler_descr, make_call_descr, make_call_descr_from_target_slot,
    make_call_descr_sized_with_effect, make_call_descr_with_effect, nursery_alloc_effect_info,
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
    DeoptMaterializationCache, JitState, PendingFieldWriteLayout, ResumeDataResult,
    bridge_decode_red,
};
pub use jitcode::{
    BC_GOTO, JitArgKind, JitCallArg, JitCode, JitCodeBuilder, RuntimeBhDescr,
    init_global_build_descr_pool, insns, live_slots_for_state_field_jit,
};
pub use jitdriver::{
    DeclarativeJitDriver, JitDriver, JitDriverStaticData, MultiFrameBlackholeResult,
    PendingAbortBlackhole, SingleFrameBlackholeResult, TraceContinuationSuspendGuard,
    current_state_field_fvc_epoch, drive_multi_frame_blackhole, drive_single_frame_blackhole,
    no_bridge_enabled, trace_continuation_suspended,
};
pub use majit_backend::CompiledTraceInfo;
pub use pyjitpl::{eval_binop_f, eval_binop_i, eval_float_cmp, eval_unary_f, eval_unary_i};
// Re-export the canonical translate-side Assembler so macro-emitted
// state-field JIT setup (e.g. `__JitMeta_<fn>::install_canonical_liveness`)
// can build a fresh Assembler without forcing each user crate to
// declare a `majit-translate` dependency.  The same pattern is used
// for `JitCode` / `BhDescr` re-exports above (`jitcode/mod.rs:4`).
pub use majit_translate::codewriter::assembler::Assembler;
pub use parity::{TraceParityCase, assert_trace_parity, normalize_ops, normalize_trace};
/// The walker's own `getfield_gc` / `setfield_gc` descr resolution
/// (`blackhole.py:1432-1483` reads the descr straight out of the constant
/// pool).  Exported so the descr-identity census can compare it against the
/// pool-side resolution without re-deriving a second copy of the logic.
pub use pyjitpl::dispatch::field_descr_ref_from_bh;
pub use pyjitpl::{
    BackEdgeAction, BridgeCompileResult, BridgeRetraceResult, ClosureRuntime,
    ClosureRuntimeWithResolver, CompileOutcome, CompiledExitLayout, CompiledTerminalExitLayout,
    CompiledTraceLayout, DeadFrameArtifacts, DetailedDriverRunOutcome, InlineDecision,
    JitCodeMachine, JitCodeRuntime, JitCodeSym, JitHooks, JitStats, MIFrame, MIFrameStack,
    MetaInterp, MetaInterpGlobalData, MetaInterpStaticData, RawCompileResult, StandaloneFrameStack,
    build_state_field_snapshot, call_int_function, call_ref_function, call_void_function, counters,
    record_application_traceback_for_recording, record_application_traceback_hook_address,
    record_discarded_level_traceback_for_recording, record_discarded_level_traceback_hook_address,
    record_inline_application_traceback_for_recording,
    record_inline_application_traceback_hook_address, set_record_application_traceback_hook,
    set_record_discarded_level_traceback_hook, set_record_inline_application_traceback_hook,
    struct_fields_write_effect_info, trace_jitcode, trace_jitcode_from_merge_point,
    trace_jitcode_with_args, trace_jitcode_with_args_and_runtime,
};
pub use resume_box_reader::{
    BridgeVirtualCache, decode_fieldnum, default_bridge_array_descr, emit_pending_field_op,
    materialize_bridge_virtual, rebuilt_value_to_opref, replay_pending_fields,
    seed_bridge_virtualizable_boxes,
};
pub use trace_ctx::BridgeInlineCarrier;
pub use trace_ctx::GreenBox;
pub use trace_ctx::MergePoint;
pub use trace_ctx::ReconstructRecipe;
pub use trace_ctx::TraceCtx;
pub use trace_ctx::VableArrayStore;
pub use trace_ctx::VableEntryWrite;

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

/// Constness of every index arriving at `TraceCtx::get_arrayitem_vable_index`.
/// See that function for what the counts do and do not establish — in
/// particular, it is a shared callee and the reading cannot be attributed to a
/// caller family without pairing it with a call-site probe.
pub fn vable_idx_probe_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("PYRE_VABLE_IDX_PROBE").is_some())
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

/// A dispatch arm whose body `#[jit_interp]` could not lower, so the macro
/// substituted a bare `BC_ABORT` sub-JitCode for it.
///
/// The substitution is deliberate — `make_jitcodes()` builds the portal even
/// when one opcode lowers to a residual the tracer cannot follow, so that
/// opcode aborts the trace instead of disabling the JIT for every other
/// opcode.  What was missing is any record of WHICH opcode.  At execution the
/// only surviving signal is `abort_trace` falling back to
/// `AbortReason::Generic` → `counters::ABORT_BRIDGE` → `MC_DIAG` slot 41,
/// which `jitprof.rs` documents as overwhelmingly unclassified; by then the
/// arm's identity is gone.  This channel keeps it, named, from the point the
/// stub is built.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DegradedDispatchArm {
    /// The machine's declared `state = T` type name (`VmStateF`), which is
    /// what identifies one `#[jit_interp]` mainloop among several.
    pub interp: &'static str,
    /// The arm's match pattern as written in the source (`OP_RETURN_F`).
    pub arm: &'static str,
    /// Why the body did not lower, staged by the macro at the emitting site.
    pub reason: &'static str,
}

/// The mechanism that refused a dispatch arm, as a value a gate can compare.
///
/// [`DegradedDispatchArm::reason`] is prose staged by the macro at the emitting
/// site, naming both the refusing mechanism and the offending source. A gate
/// pinning the whole string breaks on every rewording; a gate pinning only the
/// arm NAME cannot see a change of mechanism at all.
///
/// Pinning only the arm name is insufficient because the arm can remain
/// degraded while its refusing mechanism changes. Tests that care about the
/// mechanism compare this enum as well as the arm name.
// `Ord` so a gate can sort `(arm, RefusalKind)` pairs into a stable order before
// comparing. Arm names lead every such tuple, so the derived variant order never
// decides a comparison; it exists to make the pair sortable at all.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum RefusalKind {
    /// The arm writes a green this lowering path cannot carry back to the
    /// caller.  This refusal stops lowering, so it is the arm's OUTERMOST
    /// blocker; any later ones are reported behind it in the same `reason`.
    /// Read them with [`refusal_kinds`], not [`refusal_kind`].
    GreenWriteback,
    /// The lowerer has no expression for one of the arm's statements.
    UnlowerableStmt,
    /// The arm encloses a `break`/`continue` that cannot be lowered in place.
    EnclosedBreakContinue,
    /// The arm encloses a `return` that cannot be lowered in place.  Sibling of
    /// [`Self::EnclosedBreakContinue`]; the lowerer guards the two separately.
    EnclosedReturn,
    /// The arm body has no statements to lower.
    EmptyBody,
    /// The arm has no `pc` binding for the pc-return writeback.
    NoPcBinding,
    /// Lowering resolved an unsupported call policy at install time.
    ///
    /// The only family raised at INSTALL rather than by the statement lowerer,
    /// so it carries no `arm body {what}: {spelling}` shape and names no
    /// offending statement — a gate keyed on a source snippet gets nothing here.
    UnsupportedCallPolicy,
    /// The lowerer's fallback wording, reached when a refusal site never set a
    /// specific reason.
    ///
    /// Deliberately its own variant rather than folded into
    /// [`Self::Unclassified`]: the fallback's own doc keeps the exact old string
    /// so that sites a refactor never reached stay greppable.  Reaching it means
    /// "some refusal site is still unconverted", which is a different fact from
    /// "majit grew a mechanism nobody has classified".
    UnreachedLoweringFallback,
    /// No known fragment matched.
    ///
    /// Deliberate, and the reason this is an enum with an explicit fallthrough
    /// rather than a permissive default: bucketing an unrecognised reason into a
    /// known family reproduces the defect this type exists to close — a new
    /// mechanism arriving under an unchanged value, invisible to every gate.
    /// Reaching this variant should fail a gate, not be tolerated by one.
    Unclassified,
}

/// Classify a [`DegradedDispatchArm::reason`] by the mechanism that refused it.
///
/// Keyed on the shortest fragment that names the mechanism, so rewording the
/// prose around it does not break the gates.  Rewording a *fragment* is
/// expected to require an edit here: re-point it rather than relaxing a caller's
/// assertion.
///
/// One home, because the alternative was measured too — this classifier existed
/// as four copied literals across the example gates and was headed for nine.  N
/// copies of a predicate drift apart silently and no single reader can see the
/// divergence, which is the same disease the gates themselves exist to catch.
/// `tests/degraded_arm_refusal_kind.rs` pins the mapping against reasons
/// recorded from the example crates.
/// The families are the macro's whole reachable refusal vocabulary, read off the
/// producers rather than off the reasons that happen to be emitted today — three
/// of the eight are observed in the example corpus and five are not.  Listing
/// only the observed ones would make [`RefusalKind::Unclassified`] mean "not
/// seen yet" instead of "majit grew a mechanism", which is the false alarm this
/// type exists to avoid.
/// Joins the refusals accumulated into one [`DegradedDispatchArm::reason`].
///
/// Cross-crate contract: mirrors `REFUSAL_SEPARATOR` in majit-macros'
/// `jitcode_lower::lower_stmt`. A proc-macro crate cannot export a value to its
/// runtime, so the two literals are kept in step by
/// `accumulated_reason_splits_into_its_refusals` in
/// `tests/degraded_arm_refusal_kind.rs`, which splits a reason recorded by a
/// real crate and would read one segment where it expects two.
pub const REFUSAL_SEPARATOR: &str = " || ";

/// The refusals in `reason`, in the order lowering hit them.
pub fn refusal_reasons(reason: &str) -> impl Iterator<Item = &str> {
    reason.split(REFUSAL_SEPARATOR)
}

/// Every refusal's family, in order. `refusal_kind(r) == refusal_kinds(r)[0]`.
///
/// This is the accessor that makes the family distribution a measurement rather
/// than a lower bound: an arm reports its outermost refusal in `reason`'s head,
/// and the rest of the string is what lowering found behind it.
pub fn refusal_kinds(reason: &str) -> Vec<RefusalKind> {
    refusal_reasons(reason).map(refusal_kind_of_one).collect()
}

/// Classify the FIRST refusal.
///
/// Must split before matching, and this is not a stylistic preference. The
/// classifier below is an ORDERED chain of `contains` tests, so on an
/// accumulated reason an un-split match would answer with whichever fragment
/// the chain tests earliest — not with the refusal lowering actually hit first.
/// Keeping the first refusal at the head of the string is necessary for that
/// and not sufficient: without this split, adding accumulation silently
/// re-classifies every previously landed pin.
pub fn refusal_kind(reason: &str) -> RefusalKind {
    refusal_kind_of_one(refusal_reasons(reason).next().unwrap_or(reason))
}

fn refusal_kind_of_one(reason: &str) -> RefusalKind {
    if reason.contains("encloses a `return`") {
        RefusalKind::EnclosedReturn
    } else if reason.contains("encloses a `break`") {
        RefusalKind::EnclosedBreakContinue
    } else if reason.contains("writes a green") {
        RefusalKind::GreenWriteback
    } else if reason.contains("cannot express") {
        RefusalKind::UnlowerableStmt
    } else if reason.contains("no statements to lower") {
        RefusalKind::EmptyBody
    } else if reason.contains("no `pc` binding") {
        RefusalKind::NoPcBinding
    } else if reason.contains("unsupported call policy") {
        RefusalKind::UnsupportedCallPolicy
    } else if reason.contains("could not be lowered to a sub-JitCode") {
        RefusalKind::UnreachedLoweringFallback
    } else {
        RefusalKind::Unclassified
    }
}

static DEGRADED_DISPATCH_ARMS: std::sync::Mutex<Vec<DegradedDispatchArm>> =
    std::sync::Mutex::new(Vec::new());

/// Record that `arm` of `interp` was emitted as an abort stub.
///
/// Called from the `__dispatch_jitcode_*` body, so it fires when the dispatch
/// JitCode is installed rather than when the trace later walks into the stub.
/// Entries are deduplicated by content: a dispatch JitCode may be built more
/// than once per process, and the fact reported is per-arm, not per-build.
pub fn record_degraded_dispatch_arm(interp: &'static str, arm: &'static str, reason: &'static str) {
    let entry = DegradedDispatchArm {
        interp,
        arm,
        reason,
    };
    let mut arms = DEGRADED_DISPATCH_ARMS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if arms.contains(&entry) {
        return;
    }
    if majit_log_enabled() {
        eprintln!(
            "[jit] degraded dispatch arm: {}::{} lowered to an abort stub ({})",
            entry.interp, entry.arm, entry.reason
        );
    }
    arms.push(entry);
}

/// Snapshot of every dispatch arm recorded as degraded so far.
pub fn degraded_dispatch_arms() -> Vec<DegradedDispatchArm> {
    DEGRADED_DISPATCH_ARMS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// The shape of a compiled loop body, as a tier gate needs to read it.
///
/// A compile counter says a trace was *compiled*; an op count says the body is
/// not the degenerate `Finish()`. Neither says the body is a **loop**. A body
/// can be several ops long, carry no back edge, and be cut short by a guard
/// whose only outcome is a bail-out — a compiled trace that can never run a
/// second iteration. That is what a bare `compiles > 0 && ops_after > 1`
/// recipe accepts, and it is what this type is for.
///
/// Build it from the opcodes `JitDriver::set_on_compile_loop` hands the
/// callback, and read it in the same lock window as the counters: the shape is
/// as process-global as they are.
///
/// Only meaningful when the gate's subject **contains a loop**. On a
/// straight-line program a body with no back edge is the correct answer, not a
/// defect, and asserting [`Self::closes_a_loop`] there rejects a healthy
/// compile.
///
/// `Label` is deliberately not part of this. An optimized body can close its
/// back edge without one — tinyframe's carries a `Jump` and no `Label` — so a
/// predicate that also demanded a `Label` would reject a healthy body.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoopBodyShape {
    /// The body carries a `Jump`: it reaches its own back edge.
    pub has_jump: bool,
    /// The body carries a `GuardAlwaysFails`: a guard with no passing outcome,
    /// so control leaves the compiled body where it sits.
    pub has_always_fails: bool,
}

impl LoopBodyShape {
    /// Read the shape off an optimized body.
    pub fn of(opcodes: &[majit_ir::OpCode]) -> Self {
        Self {
            has_jump: opcodes.contains(&majit_ir::OpCode::Jump),
            has_always_fails: opcodes.contains(&majit_ir::OpCode::GuardAlwaysFails),
        }
    }

    /// The body reaches a back edge and is not cut short by a guard that
    /// cannot pass.
    pub fn closes_a_loop(self) -> bool {
        self.has_jump && !self.has_always_fails
    }

    /// Why [`Self::closes_a_loop`] is false, phrased for an assertion message.
    /// `None` when it is true.
    ///
    /// Both fields get their own arm, including the one where both are false at
    /// once. A first-match chain would report only `has_jump` there, and that is
    /// the weaker of the two facts: "no `Jump`" is true of every straight-line
    /// body, while `GuardAlwaysFails` names the specific pathology. The reader
    /// of a failing gate has nothing but this string.
    pub fn why_not(self) -> Option<&'static str> {
        match (self.has_jump, self.has_always_fails) {
            (true, false) => None,
            (true, true) => {
                Some("carries a `GuardAlwaysFails`, so control leaves it rather than looping")
            }
            (false, true) => Some(
                "carries no `Jump` AND carries a `GuardAlwaysFails`: it neither reaches a back \
                 edge nor has a passing outcome at that guard",
            ),
            // Also the [`Default`] value, which the gates reset to before the
            // hook runs — so this arm reads as "the hook never fired" too.
            (false, false) => Some("carries no `Jump`, so it never reaches a back edge"),
        }
    }
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
        /// The concrete exception object this finish escapes with, as a raw
        /// GC-ref word (`0` when `exit_with_exception` is false).
        ///
        /// `pyjitpl.py:2530-2562 finishframe_exception` snapshots
        /// `excvalue = self.last_exc_value` *before* calling
        /// `compile_exit_frame_with_exception(self.last_exc_box)` and then
        /// raises `jitexc.ExitFrameWithExceptionRef(excvalue)`, which
        /// `warmspot.py:998-1005` re-raises out of `ll_portal_runner`.  The
        /// compile half consumes only the symbolic `finish_args`; the raise
        /// half needs the value, so it travels alongside them.
        exc_value: i64,
    },
    /// Close and compile a segmented loop (force_finish_trace).
    /// pyjitpl.py:1622 _create_segmented_trace_and_blackhole parity.
    /// The trace has GUARD_ALWAYS_FAILS + unreachable FINISH appended.
    /// compile_simple_loop inserts a LABEL at entry for bridge attachment.
    SegmentedLoop,
    /// The else-arm of the same split (pyjitpl.py:1665-1668): the segmented
    /// trace is a guard-origin bridge, so there is no merge point to make a
    /// loop out of and it is closed as an ordinary bridge instead.
    ///
    /// ```python
    /// target_token = compile.compile_trace(metainterp, metainterp.resumekey,
    ///                                      [exception_box])
    /// if target_token is not token:
    ///     compile.giveup()
    /// ```
    ///
    /// The trace has GUARD_ALWAYS_FAILS appended but NOT the FINISH — the
    /// driver's compile records it, so that it carries
    /// `sd.exit_frame_with_exception_descr_ref` (the `token` upstream
    /// compares the returned target against).  `exception_box` is the
    /// operand it finishes with, the same box the loop arm records.
    SegmentedBridge { exception_box: OpRef },
    /// Abort the current trace (recoverable — may retry later).
    Abort,
    /// Decline the current trace before compilation and return to residual
    /// execution without charging a trace abort.
    Decline,
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
///
/// Folds `green_uhash_step` over the slices directly rather than building a
/// [`majit_ir::GreenKey`] to hash and drop — the key was never returned, so the
/// two `to_vec()`s were pure overhead. Equals
/// `GreenKey::with_types(values.to_vec(), types.to_vec()).hash_u64()`,
/// including the short-`types` padding: `GreenKey::get_uhash` reads types with
/// `.get(i).unwrap_or(Int)`, so a ragged call pads rather than truncates.
#[inline]
pub fn green_key_hash_typed(values: &[i64], types: &[majit_ir::GreenType]) -> u64 {
    debug_assert_eq!(values.len(), types.len());
    let mut x = majit_ir::GREEN_UHASH_SEED;
    for (i, &value) in values.iter().enumerate() {
        let tp = types.get(i).copied().unwrap_or(majit_ir::GreenType::Int);
        x = majit_ir::green_uhash_step(x, tp, value);
    }
    x
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

/// Number of `MC_DIAG` slots. Declared once so the counter array and
/// `MC_DIAG_LABELS` cannot drift in length — a mismatch is a compile error.
pub const MC_DIAG_SLOTS: usize = 78;

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
/// fail_values too short, 18 = compile_and_run_once entered from a back edge,
/// 19 = compile_and_run_once entered from a function entry, 20 =
/// compile_and_run_once early-out: portal jitcode unavailable, 21 =
/// compile_and_run_once early-out: no merge entry at the target pc, 22 =
/// compile_and_run_once early-out: tracing did not start, 23 =
/// should_trace_function_entry declined (cell compiled or tracing), 24 =
/// should_trace_function_entry declined (dead procedure token cleanup), 25 =
/// should_trace_function_entry answered from the counter tick, 26 = walk steps
/// recorded past `trace_limit` whose abort was suppressed because the walk had
/// already executed an unrollbackable effect, 27 = the walker's in-trace
/// `compile_trace` attempt did not take and fell through to the merge-point
/// scan, 28 = `compile_loop` gave the trace up at its own
/// `has_compiled_targets`, 29 = `compile_trace` cancelled with no
/// front target token to jump at, 30 = `compile_trace` cancelled because the
/// origin guard's loop is no longer compiled, 31 = the interp-origin entry
/// bridge did not compile, 32 = `compile_trace` had neither a guard origin nor
/// entry-bridge data to close against, 33 = `compile_trace` was called with no
/// tracing session.
///
/// 34-39 split slot 31 by which of `compile_entry_bridge`'s exits answered:
/// 34 = the target green key has no compiled loop, 35 = its JitCellToken is
/// dead, 36 = `optimize_bridge` raised `InvalidLoop`, 37 = the optimizer asked
/// for a retrace instead, 38 = the backend's own `compile_loop` panicked,
/// 39 = the backend returned `Err` from it. Slot 31 stays their total. The six
/// are reachable for different reasons and only
/// the split says which; the guest has no stderr, so `MAJIT_LOG`'s per-exit
/// lines never reach a wasm run and this is the only channel that does.
///
/// 40-46 are the `Counters.ABORT_*` histogram of `aborted_tracing`, in
/// `counters` order (`ABORT_TOO_LONG` .. `ABORT_SEGMENTED_TRACE`), with 46 for
/// a reason outside that range. Their sum is the `loops_aborted` contribution
/// of tracing aborts, so it also separates those from the backend-`Err`
/// aborts, which do not pass through here. The reason already reaches
/// `profiler.count(reason)`; these slots are the same value on the one channel
/// a wasm guest can be read through.
///
/// 47-49 split slot 41 (`ABORT_BRIDGE`) by which `compile.giveup()` raised it,
/// since upstream spends that one reason id on every generic bail: 47 = the
/// optimizer deferred an `InvalidLoop` on the root trace, 48 = the root
/// `compile_loop` returned `Err`, 49 = a bridge compile came back `Aborted`.
///
/// 70 completes that decomposition from the other end: the abort where *no*
/// reason was staged at all and the code fell back to `AbortReason::Generic`,
/// counted where the fallback is chosen (`pyjitpl.rs abort_trace`,
/// `jitdriver.rs`'s reason ladder). Slot 41 is the total of every
/// `aborted_tracing(ABORT_BRIDGE)` and cannot separate a real bridge giveup
/// from that default, which is what made `abrt_bridge=1` read as evidence of
/// bridge activity that was not there.
///
/// 70 is NOT `41 - (47+48+49)`, and that subtraction must not be used:
/// 47-49 are bumped at the RAISE, immediately before
/// `return Err(SwitchToBlackhole::giveup())`, while 41 is bumped at the CATCH
/// inside `aborted_tracing`. A giveup raised on a path that never reaches a
/// catch is counted by one and not the other, in a direction nothing
/// announces. Each slot counts its own event at its own site, so a
/// disagreement between them is readable rather than silent — if 47+48+49 ever
/// exceeds what 41 can account for, that difference is a giveup that was
/// raised and never accounted, which has no other detector.
///
/// 50 = `close_bridge` declined while closing a bridge, 51 = bridge close found
/// no compiled target, 52 = abort after a declined bridge attempt, 53 =
/// `compile_trace` called `compile_bridge` and it returned false.
///
/// 54-56 are the merge-point path's three silent recovery rules, each counted
/// where it FIRES rather than where it is called, so a slot reading 0 over the
/// corpus is the evidence needed to replace the rule with a `debug_assert!`:
/// 54 = `has_merge_point_with_shape_assert` rejected a SAME-green-key merge
/// point because its `green_boxes` length differed from `live_args_len`, where
/// `pyjitpl.py:3020 assert len(original_boxes) == len(live_arg_boxes)` asserts
/// instead of filtering; 55 = `register_retrace_merge_point` declined to
/// register because some jump arg carried no intrinsic type, where
/// `pyjitpl.py:3059-3060 self.current_merge_points.append((live_arg_boxes,
/// start))` appends unconditionally; 56 = `close_header_pc` fell back to
/// `self.header_pc` because the walk recorded no close greens, where
/// `pyjitpl.py:3021 same_greenkey(original_boxes, live_arg_boxes,
/// num_green_args)` always compares the actual closing boxes. 55 and 56 are
/// gated behind a non-zero `retrace_limit` (default 0, `warmstate.rs`
/// DEFAULT_RETRACE_LIMIT / `rpython/rlib/jit.py:595`), so a default-parameter
/// run reading 0 on them proves nothing.
///
/// 57 = the unroll pass abandoned a retrace because `jump_to_preamble` would
/// have landed a body JUMP on a preamble LABEL of a different arity
/// (`compile.py:334`). Non-zero means a retrace was built and discarded; see
/// the preamble-arity item.
///
/// 58/59 = the FBW walker's inline sub-walk decided to open (58) / close (59) a
/// `portal_trace_positions` entry.  They count the DECISIONS, not the appends:
/// an abort retires the log mid-sub-walk, so an append tally would read 58 far
/// above 59 for a reason that says nothing about the pairing.  As decisions,
/// `58 != 59` at process exit means exactly one thing — a sub-walk exit path
/// skipped its close, and `find_biggest_function` mis-sizes every frame after
/// it.  A `debug_assert!` cannot see that; the imbalance is only visible across
/// a whole run.
///
/// 61-63 are `maybe_start_tracing`'s two early returns and their denominator,
/// which is 61 and is bumped unconditionally at entry so the two refusals can
/// be read as fractions rather than bare totals: 62 = `sync_before` returned
/// false, 63 = `live_values_match_descriptor` returned false. Both return
/// before `on_back_edge_typed`, so `61 - 62 - 63` is the number of calls that
/// reach the hotness counter at all. The two are counted separately because a
/// deferral past the counter is worth the cost of whichever refusal dominates,
/// and nothing currently says which does; a slot reading 0 across the corpus
/// says its refusal never fires, which is a different design than one that
/// fires often.
///
/// 64-66 split slot 23, which is bumped on the disjunction
/// `cell.is_compiled() || cell.is_tracing()` and so cannot attribute: the
/// `is_compiled()` term fires on every function-entry probe of every
/// already-compiled key, which is the normal high-rate case, so slot 23
/// climbs in a healthy tree. Slot 23 stays their total. The two terms are
/// evaluated independently rather than short-circuited, and both are counted,
/// so a cell that is compiled *and* tracing bumps 64 and 65 both — they
/// partition nothing and must not be subtracted from each other.
/// 64 = the `is_compiled()` term was true, 65 = the `is_tracing()` term was
/// true.
///
/// CORRECTED. An earlier version of this legend said 66 was the
/// discriminator and 65 alone was not, on the premise that "a cell's
/// `JC_TRACING` is legitimately set while a trace is genuinely running, so a
/// non-zero 65 is the healthy reading". **That premise is false at the only
/// production call site.** `should_trace_function_entry` is reached from
/// exactly one production caller, `pyre-jit`'s `try_function_entry_jit`, which
/// guards on `!driver.is_tracing()`. That resolves to
/// `MetaInterp::tracing.is_some()` — one global `Option`, not a per-cell flag —
/// so while the engine is tracing the caller returns a frame earlier and this
/// gate is never reached.
///
/// ⇒ **In production, every bump of 65 is a cell holding `JC_TRACING` while no
/// trace is running.** 65 is itself the leak signal; there is no healthy 65 at
/// this site. (The function can still be called mid-trace directly, and the
/// unit tests do, so the distinction is production-path-specific.)
///
/// 66 splits those leaks by AGE rather than into leak-vs-healthy. It counts the
/// subset of 65 whose `cell.tracing_generation` is strictly older than the warm
/// state's, i.e. a session a later `start_tracing_cell` superseded. Only
/// `start_tracing_cell`/`start_tracing_cell_for_key` increment the generation;
/// the `mark_as_being_traced` pair stamp it without incrementing. So `65 > 0`
/// with `66 == 0` is a flag leaking from the most recent session — if anything
/// the more direct miss, since that is exactly the clear the tracing teardown
/// is responsible for.
///
/// A stale `JC_TRACING` can only sit on a cell that once started tracing. The
/// door counters alone do not prove that the examined cell ever started a
/// trace. `caro_funcentry` must also be nonzero. The keys that trace at back
/// edges are distinct from function-entry keys because `pc` is folded as a green
/// (`majit_ir::pypyjit_greenkey_uhash`), so a loop-header pc and an entry pc
/// are different keys. The probed set and the ever-traced set were disjoint.
///
/// ⇒ **TWO WITNESSES ARE NEEDED AND THEY ARE NOT THE SAME.**
///
/// * DOOR-RAN: slots 23, 24 and 25 are each bumped at exactly one site, all
///   three inside `should_trace_function_entry`, so `23 + 24 + 25 > 0` proves
///   the gate executed. **Necessary, not sufficient.** It is also a LOWER BOUND
///   on entries, not a denominator — two `DONT_TRACE_HERE` exits are uncounted
///   — so it cannot carry a rate either.
/// * ARMED: `caro_funcentry` (slot 19) must be `> 0`, or no probed cell was
///   ever in a position to hold the flag. The bump sits at the top of
///   pyre-jit's `compile_and_run_once` above every early return, so a zero
///   means that call was never reached, not that it returned early. But the
///   CALL is unconditional while the SLOT is selected by the `start` arm
///   (`BackEdge => 18`, `FunctionEntry => 19`, `eval.rs:8991-8994`), so 19
///   counts function-entry starts ONLY — a back-edge-only workload leaves 19
///   at 0 while 18 climbs. Read 19, never 18 + 19, and never "the arm was
///   entered".
///
/// The two witnesses sit in one function and in this order:
/// `try_function_entry_jit` calls the door at `eval.rs:9397` and reaches
/// `compile_and_run_once(.., FunctionEntry)` 235 lines later at `:9632`. So in
/// the worked example above the door ran 2540 times and control never once
/// reached the compile call — every probe declined at the door or between it
/// and `:9632`. That is the mechanism behind "the probed set and the
/// ever-traced set were disjoint", and it is also why 23 + 24 + 25 cannot
/// stand in for 19: they are counted on the near side of that gap.
///
/// `65 == 0` refutes only with **both**. With either missing the reading is
/// NOT EXERCISED — never "clean". The general form, from the first three
/// revisions of this legend, which failed the same way: an exercise witness
/// must witness the ARMING POPULATION, not that the instrument ran.
///
/// CORRECTED A THIRD TIME, and this one is a different failure. The
/// condition above was right; the sentence describing slot 19 said it was
/// "bumped unconditionally at the top of `compile_and_run_once`", which
/// drops the arm selection and so describes a witness with twice the
/// coverage of the one that exists. The verdict does not move — `19 == 0`
/// is still NOT EXERCISED either way — but the reader's model of the run
/// does, in two ways that matter. On a back-edge workload `18 > 0` with
/// `19 == 0` is flatly impossible under the wrong reading, so a reader
/// resolves the contradiction by distrusting a healthy instrument or the
/// run itself; and a reader who believes any `compile_and_run_once` entry
/// bumps 19 will accept `18` (or `18 + 19`) as an arming witness, which it
/// is not. **A witness's stated domain is part of the witness, and it is
/// the part that gets applied.** (Found by sizes-2 against a census of
/// production readouts; re-verified here at `eval.rs:8991-8994`.)
///
/// And 64 is narrower than "a compiled callee was probed": the caller has
/// already excluded `has_runnable_compiled_loop` (a driver-side meta table)
/// while 64 reads `cell.is_compiled()` (the warmstate cell token). The two
/// disagree in both directions, so **64 counts cell-vs-meta disagreement.**
pub static MC_DIAG: [std::sync::atomic::AtomicU64; MC_DIAG_SLOTS] = {
    // `AtomicU64` is not `Copy`, but a repeat expression accepts a path to a
    // const item, so the length is taken from `MC_DIAG_SLOTS` rather than from
    // a spelled-out row of elements that has to be recounted by hand.
    const Z: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    [Z; MC_DIAG_SLOTS]
};

/// Short label per [`MC_DIAG`] slot, in index order, so a tally cannot be added
/// without naming it. Readers join these with the counter values.
pub const MC_DIAG_LABELS: [&str; MC_DIAG_SLOTS] = [
    "mc_entered",
    "decl_shortcircuit",
    "descr0_skip",
    "busy_skip",
    "FIRED",
    "stack_full",
    "retrace_entered",
    "retrace_bailed",
    "cb_entered",
    "cb_invalidloop",
    "cb_retrace_req",
    "cb_arity_giveup",
    "sbt_entered",
    "sbt_not_faildescr",
    "sbt_no_jct",
    "sbt_no_meta",
    "sbt_cant_trace",
    "sbt_short_vals",
    "caro_backedge",
    "caro_funcentry",
    "caro_no_portal",
    "caro_no_merge_entry",
    "caro_not_tracing",
    "stfe_cell_busy",
    "stfe_dead_token",
    "stfe_tick",
    "toolong_suppressed",
    "wct_declined",
    "cl_hct_giveup",
    "ct_no_front_token",
    "ct_origin_loop_gone",
    "ct_entry_bridge_failed",
    "ct_no_entry_data",
    "ct_not_tracing",
    "ceb_no_loop",
    "ceb_dead_token",
    "ceb_invalidloop",
    "ceb_retrace_req",
    "ceb_backend_panic",
    "ceb_backend_err",
    "abrt_too_long",
    "abrt_bridge",
    "abrt_bad_loop",
    "abrt_escape",
    "abrt_force_qmut",
    "abrt_segmented",
    "abrt_other",
    "giveup_invalidloop",
    "giveup_compileloop_err",
    "giveup_bridge_aborted",
    "bridge_declined_close",
    "bridge_no_targets_close",
    "abort_after_declined",
    "ct_compile_bridge_false",
    "mp_shape_filtered",
    "retrace_mp_untyped",
    "close_hdr_fallback",
    "retrace_arity_giveup",
    "ptp_push",
    "ptp_pop",
    "forced_never_compiled",
    "mst_entered",
    "mst_sync_before_false",
    "mst_live_values_mismatch",
    "stfe_declined_compiled",
    "stfe_declined_tracing",
    "stfe_declined_tracing_stale",
    // Appended, never inserted: a slot's index is its position here, and every
    // `mc_diag_bump(N)` in the tree names that position.
    "bridge_unattempted_close",
    // The producer side of the cross-loop close, which slots 50 and 67 consume.
    // They sum: 68 = 69 + (target not compiled), and 69 = 70 + (already latched
    // declined). Without 68 a zero at 70 cannot say whether the decision was
    // never reached or was reached and answered no — and the corpus reads the
    // second case, not the first.
    "xloop_close_decision_reached",
    "xloop_close_target_compiled",
    "xloop_close_published",
    "abrt_unclassified_default",
    // The three outcomes of one `InvalidLoop` reaching `compile_loop`'s handler.
    // They PARTITION that event: the handler either cancels (72) or runs the
    // unroll-free retry, which rescues (73) or abandons (74). So
    // `72 + 73 + 74` is the number of `InvalidLoop`s raised anywhere in the
    // optimizer — a total that `MAJIT_LOG=1` prints independently as
    // `[jit] abort trace at key=… (InvalidLoop: …)`, which makes these three
    // cross-checkable against a channel that does not go through them.
    //
    // They are sited at the CONVERGENCE point, not at any raise site. The
    // optimizer constructs `InvalidLoop` at ~24 places across six files, in at
    // least two families that reach different crates, so a counter at any one
    // raise site measures its own family and reads as a corpus figure. Every
    // one of those sites arrives here.
    //
    // WHICH OF 72 AND 73 ANSWERS "how often was the unrolled compile
    // abandoned" IS DECIDED BY `max_unroll_loops`, AND AT A LIMIT OF 0 IT IS
    // 73. The handler cancels while `cancelled_too_many_times()` is false, i.e.
    // while `cancel_count <= max_unroll_loops`, so at a limit of L:
    //
    //   L > 0  — the first L events for a key land in 72 and tracing continues;
    //            only the L+1'th runs the retry. 73 then counts keys that
    //            reached the retry rather than abandoned compiles, and a key
    //            that never exceeds L never appears in 73 at all.
    //   L == 0 — `1 > 0` holds on the very first `InvalidLoop`, so 72 is never
    //            bumped and every abandoned unrolled compile goes straight to
    //            the retry. 73 is then EXACT, and a zero in 72 is 0-out-of-0
    //            rather than a report that nothing was abandoned.
    //
    // A crate that never passes `"max_unroll_loops"` to `jit::set_param` takes
    // L from `warmstate::DEFAULT_MAX_UNROLL_LOOPS`, so read that constant before
    // reading 72. `TraceCtx::declined_cross_loop_closes` draws its own
    // consequence from the same one.
    "unroll_cancelled_invalid_loop",
    "unroll_free_retry_rescued",
    "unroll_free_retry_failed",
    "qmut_deps_simple_loop",
    "qmut_deps_entry_bridge",
    "qmut_deps_blackhole_arm",
];

/// Render every [`MC_DIAG`] tally as space-separated `label=count` pairs.
pub fn mc_diag_summary() -> String {
    MC_DIAG_LABELS
        .iter()
        .enumerate()
        .map(|(i, label)| format!("{label}={}", mc_diag(i)))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Per-guard deopt census, keyed by `(green_key, fail_index)`.
///
/// `guard_failures` alone cannot tell a guard that keeps returning to the
/// runtime because no bridge was ever attached to it from a wide spread of
/// cold guards that each fail below `trace_eagerness`. The two want opposite
/// fixes, and the distinction is a distribution, not a total.
///
/// Off unless `MAJIT_GUARD_CENSUS` is set: the map write sits on the deopt
/// path, which is exactly the path under study.
static GUARD_CENSUS: std::sync::Mutex<Option<Vec<((u64, u32), u64)>>> = std::sync::Mutex::new(None);

/// Turn the census on where `MAJIT_GUARD_CENSUS` cannot be read.
///
/// `wasm32-unknown-unknown` has a permanently empty `std::env`, so the variable
/// never selects anything from inside the guest and the one distribution that
/// distinguishes a bridge-less hot guard from a spread of cold ones is
/// unreadable on exactly the target whose `guard_failures` totals disagree with
/// native. The host runner calls this through `pyre_jit_guard_census_enable`
/// before the run instead.
pub fn guard_census_enable() {
    GUARD_CENSUS_FORCED.store(true, std::sync::atomic::Ordering::Relaxed);
}

static GUARD_CENSUS_FORCED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn guard_census_enabled() -> bool {
    if GUARD_CENSUS_FORCED.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MAJIT_GUARD_CENSUS").is_some())
}

pub fn guard_census_record(green_key: u64, fail_index: u32) {
    if !guard_census_enabled() {
        return;
    }
    let Ok(mut slot) = GUARD_CENSUS.lock() else {
        return;
    };
    let rows = slot.get_or_insert_with(Vec::new);
    match rows.iter_mut().find(|(k, _)| *k == (green_key, fail_index)) {
        Some((_, count)) => *count += 1,
        None => rows.push(((green_key, fail_index), 1)),
    }
}

/// Render the census as `distinct=N total=M` plus the heaviest guards.
pub fn guard_census_summary(top: usize) -> String {
    let Ok(slot) = GUARD_CENSUS.lock() else {
        return "guard_census=lock-poisoned".to_string();
    };
    let Some(rows) = slot.as_ref() else {
        return "guard_census=off".to_string();
    };
    let mut rows = rows.clone();
    rows.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    let total: u64 = rows.iter().map(|(_, c)| c).sum();
    let heaviest = rows
        .iter()
        .take(top)
        .map(|((key, fail_index), count)| format!("{key}/{fail_index}:{count}"))
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "guard_census distinct={} total={total} top={heaviest}",
        rows.len()
    )
}

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

    /// The direct fold must equal the `GreenKey`-building form it replaced, at
    /// every arity and type mix — not just the one pair pinned above.
    ///
    /// Str / Unicode are excluded: `hash_whatever` routes them through a
    /// registered resolver, so they are not hashable from a bare i64 here.
    ///
    /// The short-`types` padding branch (`.get(i).unwrap_or(Int)`) is NOT
    /// covered: `debug_assert_eq!` makes a ragged call panic in a test build,
    /// so that branch is reachable only in release. It is preserved by
    /// construction, mirroring `GreenKey::get_uhash`.
    #[test]
    fn green_key_hash_typed_equals_the_greenkey_form_across_arities_and_types() {
        use majit_ir::GreenType;
        let palette = [GreenType::Int, GreenType::Float, GreenType::Ref];
        let values: [i64; 6] = [0, 1, -1, i64::MAX, i64::MIN, (2.5f64).to_bits() as i64];

        let mut checked = 0usize;
        for arity in 0..=values.len() {
            for (offset, tp) in palette.iter().enumerate() {
                let vals: Vec<i64> = values[..arity].to_vec();
                // Rotate the type assignment so a given arity is exercised with
                // several distinct type vectors, not one uniform one.
                let types: Vec<GreenType> = (0..arity)
                    .map(|i| palette[(i + offset) % palette.len()])
                    .collect();
                let folded = green_key_hash_typed(&vals, &types);
                let built = majit_ir::GreenKey::with_types(vals.clone(), types.clone()).hash_u64();
                assert_eq!(
                    folded, built,
                    "arity {arity}, offset {offset} (lead {tp:?}): fold {folded} != built {built}",
                );
                checked += 1;
            }
        }
        // Guard the guard: a silently empty sweep would assert nothing.
        assert_eq!(checked, 7 * 3, "sweep did not cover the intended cases");
    }

    /// All four `(has_jump, has_always_fails)` combinations, because the
    /// both-false one is the arm a first-match chain loses and it is the only
    /// one that has to name two facts.
    #[test]
    fn why_not_names_every_false_term() {
        let shape = |has_jump, has_always_fails| LoopBodyShape {
            has_jump,
            has_always_fails,
        };

        assert_eq!(shape(true, false).why_not(), None);
        assert!(shape(true, false).closes_a_loop());

        let jump_and_fails = shape(true, true).why_not().expect("does not close a loop");
        assert!(
            jump_and_fails.contains("GuardAlwaysFails") && !jump_and_fails.contains("no `Jump`"),
            "a body with a back edge must not be told it has none: {jump_and_fails}",
        );

        let neither = shape(false, true).why_not().expect("does not close a loop");
        assert!(
            neither.contains("no `Jump`") && neither.contains("GuardAlwaysFails"),
            "both terms are false, so both must be named: {neither}",
        );

        let no_jump = shape(false, false)
            .why_not()
            .expect("does not close a loop");
        assert!(
            no_jump.contains("no `Jump`") && !no_jump.contains("GuardAlwaysFails"),
            "nothing failed at a guard here, so the message must not claim one: {no_jump}",
        );

        // The reset sentinel the gates write before the hook runs is this arm.
        assert_eq!(LoopBodyShape::default().why_not(), Some(no_jump));
    }

    /// `of()` reads both fields off the same slice, so a body carrying both
    /// opcodes must set both — the census that feeds every gate.
    #[test]
    fn shape_of_reads_both_opcodes() {
        use majit_ir::OpCode;

        assert_eq!(LoopBodyShape::of(&[]), LoopBodyShape::default());
        assert_eq!(
            LoopBodyShape::of(&[OpCode::Jump]),
            LoopBodyShape {
                has_jump: true,
                has_always_fails: false,
            },
        );
        assert_eq!(
            LoopBodyShape::of(&[OpCode::GuardAlwaysFails, OpCode::Jump]),
            LoopBodyShape {
                has_jump: true,
                has_always_fails: true,
            },
        );
    }
}
pub(crate) mod resumecode;
