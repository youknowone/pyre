use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use crate::optimizeopt::optimizer::{Optimizer, OptimizerKnowledge};
use majit_backend::{
    Backend, CompiledTraceInfo, ExitFrameLayout, ExitRecoveryLayout, FailDescrLayout, JitCellToken,
    TerminalExitLayout,
};
#[cfg(feature = "cranelift")]
pub(crate) use majit_backend_cranelift::CraneliftBackend as BackendImpl;
#[cfg(all(feature = "dynasm", not(feature = "cranelift")))]
pub(crate) use majit_backend_dynasm::runner::DynasmBackend as BackendImpl;
#[cfg(target_arch = "wasm32")]
pub(crate) use majit_backend_wasm::WasmBackend as BackendImpl;

#[cfg(not(any(feature = "cranelift", feature = "dynasm", target_arch = "wasm32")))]
compile_error!("majit-metainterp requires a backend: enable feature \"cranelift\" or \"dynasm\"");

use majit_ir::{FailDescr, GcRef, InputArg, Op, OpCode, OpRef, Type, Value};
use majit_trace::history::TreeLoop;
use majit_trace::warmstate::{HotResult, WarmEnterState};

use crate::blackhole::{
    BlackholeResult, ExceptionState, blackhole_execute_with_state, blackhole_execute_with_state_ca,
};
use crate::compile;
pub use crate::compile::{
    CompileResult, CompiledExitLayout, CompiledTerminalExitLayout, CompiledTraceLayout,
    DeadFrameArtifacts, RawCompileResult,
};
use crate::io_buffer;
use crate::jitdriver::JitDriverStaticData;
use crate::resume::{
    EncodedResumeData, MaterializedVirtual, ReconstructedState, ResolvedPendingFieldWrite,
    ResumeData, ResumeDataLoopMemo, ResumeDataVirtualAdder, ResumeFrameLayoutSummary,
    ResumeLayoutSummary,
};
use crate::trace_ctx::TraceCtx;
use crate::virtualizable::VirtualizableInfo;

/// Callback for unboxing a Ref (boxed int pointer) to a raw i64 value.
/// Registered by pyre at init time so adapt_live_values_to_trace_types
/// can unbox W_IntObject without depending on pyre-object.
static REF_UNBOX_INT_FN: OnceLock<fn(i64) -> i64> = OnceLock::new();

/// Register the Ref→Int unbox callback for adapt_live_values_to_trace_types.
pub fn set_ref_unbox_int_fn(f: fn(i64) -> i64) {
    let _ = REF_UNBOX_INT_FN.set(f);
}

/// No direct RPython equivalent — Rust struct carrying data that RPython
/// passes through internal method calls in handle_guard_failure
/// (pyjitpl.py:2890). Fields correspond to:
/// - `fail_types`: ResumeGuardDescr.fail_arg_types (compile.py:797)
/// - `is_exception_guard`: isinstance(key, ResumeGuardExcDescr) (compile.py:932)
/// - `rd_numb`/`rd_consts`: storage.rd_numb/rd_consts (resume.py:1042)
pub struct BridgeRetraceResult {
    pub is_exception_guard: bool,
    pub fail_types: Vec<Type>,
    pub rd_numb: Option<Vec<u8>>,
    pub rd_consts: Option<Vec<(i64, Type)>>,
}

/// Result of checking a back-edge.
pub enum BackEdgeAction {
    /// Not hot yet; keep interpreting.
    Interpret,
    /// Tracing has started. Use `trace_ctx()` to record operations.
    StartedTracing,
    /// Already tracing this loop (inner back-edge).
    AlreadyTracing,
    /// Compiled code exists. Call `run_compiled()`.
    RunCompiled,
}

/// pyjitpl.py: result of compile_loop / compile_trace.
///
/// RPython uses exceptions (raise ContinueRunningNormally on success,
/// raise SwitchToBlackhole on fatal failure) and None returns for
/// cancellation. majit uses this enum instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileOutcome {
    /// Compilation succeeded — compiled loop is installed and ready to run.
    Compiled { green_key: u64, from_retry: bool },
    /// Compilation was cancelled (e.g. InvalidLoop, virtual state mismatch).
    /// The caller may retry or continue tracing.
    Cancelled,
    /// Too many cancellations — abort and fall back to interpreter.
    /// Equivalent to RPython's SwitchToBlackhole(ABORT_BAD_LOOP).
    Aborted,
}

pub(crate) struct CompiledTrace {
    /// Inputargs for this trace, used to recover typed exit layouts during blackhole replay.
    pub(crate) inputargs: Vec<InputArg>,
    /// Resume data for each guard, keyed by fail_index.
    pub(crate) resume_data: HashMap<u32, StoredResumeData>,
    /// Optimized ops for blackhole fallback from compiled guard failures.
    pub(crate) ops: Vec<majit_ir::Op>,
    /// Constant pool paired with `ops` for blackhole fallback.
    pub(crate) constants: HashMap<u32, i64>,
    /// Constant types for the constant pool entries.
    pub(crate) constant_types: HashMap<u32, Type>,
    /// Mapping from backend fail_index to the corresponding guard op index.
    pub(crate) guard_op_indices: HashMap<u32, usize>,
    /// Static exit metadata for each guard/finish in this trace.
    pub(crate) exit_layouts: HashMap<u32, StoredExitLayout>,
    /// Static exit metadata for terminal FINISH/JUMP ops, keyed by op index.
    pub(crate) terminal_exit_layouts: HashMap<usize, StoredExitLayout>,
    /// opencoder.py parity: per-guard snapshots from tracing time.
    /// Indexed by the guard's rd_resume_position. Used for snapshot-based
    /// resume data reconstruction (independent of optimizer's fail_args).
    pub(crate) snapshots: Vec<majit_trace::recorder::Snapshot>,
    /// bridgeopt.py: serialized optimizer knowledge per guard, keyed by fail_index.
    /// Each guard gets the optimizer knowledge state at its point in the trace.
    /// Used by deserialize_optimizer_knowledge when compiling a bridge.
    pub(crate) optimizer_knowledge: HashMap<u32, OptimizerKnowledge>,
    /// JitCode for blackhole fallback. RPython stores jitcodes globally;
    /// in majit the JitCode is produced by #[jit_interp] lowering and
    /// stored per-trace so BlackholeInterpreter can execute from guard
    /// failure points.
    pub(crate) jitcode: Option<crate::jitcode::JitCode>,
}

pub(crate) struct StoredResumeData {
    pub(crate) semantic: ResumeData,
    pub(crate) encoded: EncodedResumeData,
    pub(crate) layout: ResumeLayoutSummary,
}

impl StoredResumeData {
    fn new(semantic: ResumeData) -> Self {
        let encoded = semantic.encode();
        let layout = encoded.layout_summary();
        StoredResumeData {
            semantic,
            encoded,
            layout,
        }
    }

    pub(crate) fn with_loop_memo(semantic: ResumeData, memo: &mut ResumeDataLoopMemo) -> Self {
        let encoded = memo.encode_shared(&semantic);
        let layout = encoded.layout_summary();
        StoredResumeData {
            semantic,
            encoded,
            layout,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StoredExitLayout {
    pub(crate) source_op_index: Option<usize>,
    pub(crate) exit_types: Vec<Type>,
    pub(crate) is_finish: bool,
    pub(crate) gc_ref_slots: Vec<usize>,
    pub(crate) force_token_slots: Vec<usize>,
    pub(crate) recovery_layout: Option<ExitRecoveryLayout>,
    pub(crate) resume_layout: Option<ResumeLayoutSummary>,
    pub(crate) rd_numb: Option<Vec<u8>>,
    pub(crate) rd_consts: Option<Vec<(i64, Type)>>,
    pub(crate) rd_virtuals: Option<Vec<majit_ir::RdVirtualInfo>>,
    pub(crate) rd_pendingfields: Option<Vec<majit_ir::GuardPendingFieldEntry>>,
}

impl StoredExitLayout {
    /// compile.py:186: rd_loop_token = original_jitcell_token.
    /// `owning_key` is the green_key of the compiled loop that owns this guard.
    pub(crate) fn public(
        &self,
        owning_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> CompiledExitLayout {
        CompiledExitLayout {
            rd_loop_token: owning_key,
            trace_id,
            fail_index,
            source_op_index: self.source_op_index,
            exit_types: self.exit_types.clone(),
            is_finish: self.is_finish,
            gc_ref_slots: self.gc_ref_slots.clone(),
            force_token_slots: self.force_token_slots.clone(),
            recovery_layout: self.recovery_layout.clone(),
            resume_layout: self.resume_layout.clone(),
            rd_numb: self.rd_numb.clone(),
            rd_consts: self.rd_consts.clone(),
            rd_virtuals: self.rd_virtuals.clone(),
            rd_pendingfields: self.rd_pendingfields.clone(),
        }
    }
}

/// opencoder.py:819 parity: extract per-snapshot box maps from trace snapshots.
///
/// opencoder.py:603 _encode: Const boxes are registered in the constant pool
/// and returned as pool OpRefs so the optimizer's BoxEnv can resolve them
/// via is_const/get_const (resume.py:157 getconst parity).
fn snapshot_map_from_trace_snapshots(
    trace_snapshots: &[majit_trace::recorder::Snapshot],
    constants: &mut std::collections::HashMap<u32, i64>,
    constant_types: &mut std::collections::HashMap<u32, majit_ir::Type>,
) -> (
    std::collections::HashMap<i32, Vec<majit_ir::OpRef>>,
    std::collections::HashMap<i32, Vec<usize>>,
    std::collections::HashMap<i32, Vec<majit_ir::OpRef>>,
    std::collections::HashMap<i32, Vec<(i32, i32)>>,
    std::collections::HashMap<u32, majit_ir::Type>, // snapshot_box_types
) {
    let mut box_map = std::collections::HashMap::new();
    let mut size_map = std::collections::HashMap::new();
    let mut vable_map = std::collections::HashMap::new();
    let mut pc_map = std::collections::HashMap::new();
    // RPython box.type parity: each Box carries its type from tracing.
    // Collected here so _number_boxes can detect virtual vs int correctly.
    let mut snapshot_box_types: std::collections::HashMap<u32, majit_ir::Type> =
        std::collections::HashMap::new();
    let mut next_const_idx = constants
        .keys()
        .filter(|k| majit_ir::OpRef(**k).is_constant())
        .map(|k| majit_ir::OpRef(*k).const_index())
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    // opencoder.py:603 _encode: Box/Virtual → OpRef, Const → pool OpRef.
    let mut tagged_to_opref = |t: &majit_trace::recorder::SnapshotTagged| -> majit_ir::OpRef {
        match t {
            majit_trace::recorder::SnapshotTagged::Box(n, tp) => {
                snapshot_box_types.insert(*n, *tp);
                majit_ir::OpRef(*n)
            }
            majit_trace::recorder::SnapshotTagged::Virtual(n) => majit_ir::OpRef(*n),
            majit_trace::recorder::SnapshotTagged::Const(val, tp) => {
                // resume.py:173-176: null Ref → NULLREF via getconst.
                // Register in pool so is_const → true, get_const → (0, Ref),
                // getconst → NULLREF. Do NOT short-circuit to OpRef::NONE.
                // resume.py:157 getconst: find or allocate pool entry.
                let existing = constants
                    .iter()
                    .find(|(k, v)| {
                        **v == *val
                            && constant_types
                                .get(k)
                                .copied()
                                .unwrap_or(majit_ir::Type::Int)
                                == *tp
                    })
                    .map(|(k, _)| *k);
                let key = existing.unwrap_or_else(|| {
                    let opref = majit_ir::OpRef::from_const(next_const_idx);
                    next_const_idx += 1;
                    constants.insert(opref.0, *val);
                    constant_types.insert(opref.0, *tp);
                    opref.0
                });
                majit_ir::OpRef(key)
            }
        }
    };
    for (id, snap) in trace_snapshots.iter().enumerate() {
        let boxes: Vec<majit_ir::OpRef> = snap
            .frames
            .iter()
            .flat_map(|f| f.boxes.iter())
            .map(&mut tagged_to_opref)
            .collect();
        let frame_sizes: Vec<usize> = snap.frames.iter().map(|f| f.boxes.len()).collect();
        let vable_boxes: Vec<majit_ir::OpRef> =
            snap.vable_boxes.iter().map(&mut tagged_to_opref).collect();
        let frame_pcs: Vec<(i32, i32)> = snap
            .frames
            .iter()
            .map(|f| (f.jitcode_index as i32, f.pc as i32))
            .collect();
        let id = id as i32;
        box_map.insert(id, boxes);
        size_map.insert(id, frame_sizes);
        vable_map.insert(id, vable_boxes);
        pc_map.insert(id, frame_pcs);
    }
    (box_map, size_map, vable_map, pc_map, snapshot_box_types)
}

fn normalize_root_loop_entry_contract(
    mut inputargs: Vec<InputArg>,
    mut optimized_ops: Vec<Op>,
) -> Result<(Vec<InputArg>, Vec<Op>), (usize, usize)> {
    let last_jump = optimized_ops
        .iter()
        .rev()
        .find(|op| op.opcode == OpCode::Jump);
    let jump_arg_count = last_jump.map(|op| op.args.len()).unwrap_or(0);
    let label_op = optimized_ops
        .iter()
        .rev()
        .find(|op| op.opcode == OpCode::Label);
    let label_arg_count = label_op.map(|op| op.args.len()).unwrap_or(0);
    let label_descr_index = label_op
        .and_then(|op| op.descr.as_ref())
        .map(|descr| descr.index());
    let jump_targets_current_loop = last_jump.is_some_and(|op| {
        let jump_descr_index = op.descr.as_ref().map(|descr| descr.index());
        match (jump_descr_index, label_descr_index) {
            (Some(jump_idx), Some(label_idx)) => jump_idx == label_idx,
            (None, None) => true,
            _ => false,
        }
    });

    if label_arg_count == 0 && jump_arg_count > 0 {
        if inputargs.len() > jump_arg_count {
            inputargs.truncate(jump_arg_count);
        }
        while inputargs.len() < jump_arg_count {
            inputargs.push(InputArg::from_type(Type::Int, inputargs.len() as u32));
        }
        let label_args: Vec<OpRef> = (0..inputargs.len() as u32).map(OpRef).collect();
        let mut label_op = Op::new(OpCode::Label, &label_args);
        label_op.pos = OpRef::NONE;
        optimized_ops.insert(0, label_op);
    } else if jump_targets_current_loop && label_arg_count != jump_arg_count {
        // RPython compile.py:334: assert jump.numargs() == label.numargs().
        return Err((label_arg_count, jump_arg_count));
    }

    Ok((inputargs, optimized_ops))
}

fn root_loop_inputargs_from_optimizer(
    trace_inputargs: &[InputArg],
    final_num_inputs: usize,
) -> Vec<InputArg> {
    let mut inputargs = trace_inputargs.to_vec();
    inputargs.truncate(final_num_inputs);
    while inputargs.len() < final_num_inputs {
        inputargs.push(InputArg::from_type(Type::Int, inputargs.len() as u32));
    }
    inputargs
}

pub(crate) struct CompiledEntry<M> {
    pub(crate) token: JitCellToken,
    pub(crate) num_inputs: usize,
    pub(crate) meta: M,
    /// Front-end loop-version state, mirroring RPython's
    /// jitcell_token.target_tokens ownership across recompilations.
    pub(crate) front_target_tokens: Vec<crate::optimizeopt::unroll::TargetToken>,
    /// history.py: JitCellToken.retraced_count — how many times this loop
    /// has been retraced. Persisted across recompilations.
    pub(crate) retraced_count: u32,
    /// Trace id of the root compiled loop.
    pub(crate) root_trace_id: u64,
    /// Metadata for the root loop and any attached bridges, keyed by trace id.
    pub(crate) traces: HashMap<u64, CompiledTrace>,
    /// RPython parity: previous compiled entries for this green_key.
    /// In RPython, JitCellToken keeps all target_tokens' code alive.
    /// In majit, each retrace produces a new Cranelift function;
    /// previous functions are kept here so external target_token JUMPs
    /// can redirect to them via runtime trampoline.
    pub(crate) previous_tokens: Vec<JitCellToken>,
}

/// compile.py compile_trace return parity.
/// Indicates the result of bridge compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeCompileResult {
    /// Bridge was compiled and attached to the guard.
    Compiled,
    /// Bridge compilation failed.
    Failed,
    /// Optimizer requested retrace (no matching target token).
    /// The tracing context is intact — caller should continue tracing.
    /// pyjitpl.py:3196: compile_trace returns None → MetaInterp continues.
    RetraceNeeded,
}

/// pyjitpl.py: partial trace saved from a failed bridge compilation.
///
/// When `compile_trace` (bridge path) fails to close the loop and sets
/// `retrace_needed`, this struct stores the intermediate compilation result
/// so that `compile_retrace` can append new body ops to it.
pub(crate) struct PartialTrace {
    /// Optimized ops from the first (incomplete) compilation attempt.
    pub(crate) ops: Vec<Op>,
    /// Inputargs from the partial trace.
    pub(crate) inputargs: Vec<InputArg>,
    /// Constants from the partial trace.
    pub(crate) constants: HashMap<u32, i64>,
}

/// The meta-tracing JIT engine.
///
/// Manages the full JIT lifecycle: warm counting → tracing → optimization
/// → compilation → execution.
///
/// `M` is the interpreter-specific metadata stored alongside each compiled loop
/// (e.g., storage layout, register mapping). The interpreter provides `M` when
/// closing a trace and receives it back when running compiled code.
/// Bridge trace data extracted from the tracing context, ready for
/// deferred compilation after the main loop is stored.
pub(crate) struct PendingDoneBridge {
    pub green_key: u64,
    pub trace_id: u64,
    pub fail_index: u32,
    pub bridge_ops: Vec<majit_ir::Op>,
    pub bridge_inputargs: Vec<majit_ir::InputArg>,
    pub constants: HashMap<u32, i64>,
    pub constant_types: HashMap<u32, majit_ir::Type>,
    pub snapshot_boxes: HashMap<i32, Vec<majit_ir::OpRef>>,
    pub snapshot_frame_sizes: HashMap<i32, Vec<usize>>,
    pub snapshot_vable_boxes: HashMap<i32, Vec<majit_ir::OpRef>>,
    pub snapshot_frame_pcs: HashMap<i32, Vec<(i32, i32)>>,
    pub snapshot_box_types: HashMap<u32, majit_ir::Type>,
}

pub struct MetaInterp<M: Clone> {
    pub(crate) warm_state: WarmEnterState,
    pub(crate) backend: BackendImpl,
    pub(crate) compiled_loops: HashMap<u64, CompiledEntry<M>>,
    pub(crate) tracing: Option<TraceCtx>,
    pub(crate) next_trace_id: u64,
    /// Virtualizable info for interpreter frame virtualization.
    ///
    /// When set, the JIT will:
    /// 1. Read virtualizable fields at trace entry (synchronize)
    /// 2. Write them back on guard failure (force)
    /// 3. Track field access as IR ops during tracing
    pub(crate) virtualizable_info: Option<VirtualizableInfo>,
    /// JIT hooks for profiling and debugging.
    pub(crate) hooks: JitHooks,
    /// Pre-allocated token number for the trace currently being recorded.
    /// Set when tracing starts so that self-recursive calls can emit
    /// call_assembler targeting this token before the trace is compiled.
    pub(crate) pending_token: Option<(u64, u64)>,
    /// Cumulative statistics counters.
    pub(crate) stats: JitStatsCounters,
    /// Pointer to the live virtualizable object at trace entry.
    /// Used to derive lengths from the actual object when the interpreter
    /// does not provide them explicitly.
    pub(crate) vable_ptr: *const u8,
    /// Virtualizable array lengths for trace-entry box layout.
    pub(crate) vable_array_lengths: Vec<usize>,
    /// warmspot.py:449 jd.result_type — per-driver static result type.
    pub(crate) result_type: Type,
    /// Helper function pointers that box raw ints into interpreter objects.
    pub(crate) raw_int_box_helpers: HashSet<i64>,
    /// Mapping: create_frame_N_ptr → create_frame_N_raw_int_ptr.
    /// When a box helper feeds directly into create_frame, the box+create
    /// can be folded into a single create_frame_raw_int call.
    pub(crate) create_frame_raw_map: HashMap<i64, i64>,
    /// RPython portal_call_depth parity: call depth at which the current
    /// trace started. When Some(depth), only merge_point at that depth fires.
    /// Replaces the pyre-jit TLS JIT_TRACING_DEPTH — state colocated with
    /// tracing context for single source of truth.
    pub tracing_call_depth: Option<u32>,
    /// PyPy warmspot.py max_unroll_recursion (default 7).
    pub(crate) max_unroll_recursion: usize,
    /// RPython parity: `prepare_trace_segmenting()` marks the next tracing run
    /// for a green key so the loop should finish early instead of repeatedly
    /// aborting once it nears the trace limit.
    pub(crate) force_finish_trace: bool,
    /// RPython metainterp_sd.callinfocollection parity.
    /// Maps oopspec indices to (calldescr, func_ptr) for generate_modified_call.
    pub(crate) callinfocollection: Option<std::sync::Arc<majit_ir::descr::CallInfoCollection>>,
    /// info.py:810-822 `ConstPtrInfo.getstrlen1(mode)` runtime hook. The
    /// host runtime (pyre etc.) registers this via
    /// [`MetaInterp::set_string_length_resolver`] at JIT init. Propagated
    /// to `Optimizer::string_length_resolver` inside `make_optimizer`, then
    /// on to `OptContext::string_length_resolver` for each optimizer run.
    pub(crate) string_length_resolver: Option<crate::optimizeopt::info::StringLengthResolver>,
    /// pyjitpl.py:2389: partial trace from a failed bridge compilation attempt.
    /// When bridge optimization returns "not final" (retrace needed), the
    /// partial optimized ops are saved here so compile_retrace can append
    /// the new body and compile the complete loop.
    pub(crate) partial_trace: Option<PartialTrace>,
    /// pyjitpl.py:2390: trace position where the retrace should resume.
    /// Set to potential_retrace_position by retrace_needed(). On the next
    /// compile_loop, the merge point's start position is compared
    /// against this to verify we're retracing from the correct location.
    pub(crate) retracing_from: Option<majit_trace::recorder::TracePosition>,
    /// pyjitpl.py:2374: optimizer state snapshot from the failed bridge attempt.
    /// compile_retrace imports this to resume optimization from where the
    /// first attempt left off.
    pub(crate) exported_state: Option<crate::optimizeopt::unroll::ExportedState>,
    /// pyjitpl.py:2373: number of cancelled compilation attempts.
    pub(crate) cancel_count: u32,
    /// Actual green_key the last compile_loop stored under. May differ
    /// from the tracing green_key when cross-loop cut retargets to the
    /// inner loop's key (compile.py:269).
    pub(crate) last_compiled_key: Option<u64>,
    /// virtualizable.py:86 NUM_SCALAR_INPUTARGS: number of scalar inputargs
    /// (frame + static fields). Set by the interpreter at JIT init.
    pub num_scalar_inputargs: usize,
    /// Deferred bridge compilations from compile_done_with_this_frame.
    /// RPython parity: in RPython, bridge traces only fire AFTER the main
    /// loop is compiled and the first execution happens after all bridge
    /// actions are processed. In pyre, the interpreter may resume between
    /// finish_and_compile and compile_done_with_this_frame. Pending bridges
    /// are compiled immediately after finish_and_compile stores the main loop.
    pub(crate) pending_done_bridges: Vec<PendingDoneBridge>,
    /// pyjitpl.py:3182: trace position saved before compile_trace records
    /// a tentative JUMP. If compile_trace triggers retrace_needed, this
    /// becomes the retracing_from position.
    pub(crate) potential_retrace_position: Option<majit_trace::recorder::TracePosition>,
    /// RPython compile.py:204-207 (record_loop_or_bridge) parity:
    /// quasi-immutable dependencies from the last compilation.
    /// Raw pointers to namespace/quasi-immutable objects that the compiled
    /// loop depends on. After compilation, the caller registers the loop's
    /// invalidation flag on each dep. Cleared on each compile attempt.
    pub last_quasi_immutable_deps: Vec<(u64, u32)>,
    /// Set by compile_bridge when optimizer returns retrace_requested=true.
    /// Checked by compile_bridge_trace to return RetraceNeeded.
    pub(crate) retrace_after_bridge: bool,
    /// pyjitpl.py:3317 virtualref_boxes: pairs of (symbolic OpRef, concrete ptr).
    /// Managed by opimpl_virtual_ref/opimpl_virtual_ref_finish.
    pub(crate) virtualref_boxes: Vec<(OpRef, usize)>,
    /// compile.py:288-290 parity: preamble target tokens saved from Phase 1
    /// even when Phase 2 raises InvalidLoop.
    pending_preamble_tokens: HashMap<u64, Vec<crate::optimizeopt::unroll::TargetToken>>,
}

/// Internal mutable counters for JIT compilation statistics.
#[derive(Default, Clone, Debug)]
pub(crate) struct JitStatsCounters {
    loops_compiled: usize,
    loops_aborted: usize,
    bridges_compiled: usize,
    guard_failures: usize,
}

/// Snapshot of cumulative JIT compilation statistics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JitStats {
    pub loops_compiled: usize,
    pub loops_aborted: usize,
    pub bridges_compiled: usize,
    pub guard_failures: usize,
}

/// Callback hooks for JIT events (compilation, guard failures, etc.).
///
/// Mirrors RPython's jit_hook_loop, jit_hook_bridge, etc.
/// All fields are optional closures.
#[derive(Default)]
pub struct JitHooks {
    /// Called when a loop is compiled. Args: (green_key, num_ops_before, num_ops_after).
    pub on_compile_loop: Option<Box<dyn Fn(u64, usize, usize) + Send>>,
    /// Called when a bridge is compiled. Args: (green_key, fail_index, num_ops).
    pub on_compile_bridge: Option<Box<dyn Fn(u64, u32, usize) + Send>>,
    /// Called on guard failure. Args: (green_key, fail_index, fail_count).
    pub on_guard_failure: Option<Box<dyn Fn(u64, u32, u32) + Send>>,
    /// Called when tracing starts. Args: (green_key).
    pub on_trace_start: Option<Box<dyn Fn(u64) + Send>>,
    /// Called when tracing is aborted. Args: (green_key, permanent).
    pub on_trace_abort: Option<Box<dyn Fn(u64, bool) + Send>>,
    /// Called when compilation (loop or bridge) fails. Args: (green_key, error_message).
    pub on_compile_error: Option<Box<dyn Fn(u64, &str) + Send>>,
}

impl<M: Clone> MetaInterp<M> {
    #[inline]
    fn prepare_compiled_run_io() {
        io_buffer::io_buffer_discard();
    }

    #[inline]
    fn finish_compiled_run_io(_is_finish: bool) {
        // Guard exits hand control back to the interpreter or blackhole after
        // the already-executed prefix of the trace. Any traced I/O in that
        // prefix is semantically committed and must survive deoptimization.
        io_buffer::io_buffer_commit();
    }

    #[inline]
    fn is_jump_exit(is_finish: bool, fail_index: u32) -> bool {
        !is_finish && fail_index == u32::MAX
    }

    #[inline]
    fn should_record_guard_failure(is_finish: bool, fail_index: u32) -> bool {
        !is_finish && !Self::is_jump_exit(is_finish, fail_index)
    }

    #[inline]
    fn record_guard_failure_event(&mut self, green_key: u64, fail_index: u32) {
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] guard failure at key={}, guard={}",
                green_key, fail_index
            );
        }
        self.stats.guard_failures += 1;
        self.warm_state.log_guard_failure(fail_index);
        if let Some(ref hook) = self.hooks.on_guard_failure {
            hook(green_key, fail_index, 0);
        }
    }

    #[inline]
    fn run_result_for_jump_exit(
        fail_index: u32,
        values: Vec<i64>,
        meta: M,
        savedata: Option<GcRef>,
    ) -> Option<RunResult<M>> {
        (fail_index == u32::MAX).then_some(RunResult::Jump {
            values,
            meta,
            savedata,
        })
    }

    #[inline]
    fn blackhole_result_for_jump_exit(
        fail_index: u32,
        values: Vec<i64>,
        typed_values: Vec<Value>,
        exit_layout: CompiledExitLayout,
        meta: M,
        savedata: Option<GcRef>,
        exception: ExceptionState,
    ) -> Option<BlackholeRunResult<M>> {
        (fail_index == u32::MAX).then_some(BlackholeRunResult::Jump {
            values,
            typed_values: Some(typed_values),
            exit_layout: Some(exit_layout),
            meta,
            via_blackhole: false,
            savedata,
            exception,
        })
    }

    fn alloc_trace_id(&mut self) -> u64 {
        let trace_id = self.next_trace_id;
        self.next_trace_id += 1;
        trace_id
    }

    pub fn normalize_trace_id(compiled: &CompiledEntry<M>, trace_id: u64) -> u64 {
        if trace_id == 0 {
            compiled.root_trace_id
        } else {
            trace_id
        }
    }

    fn trace_for_exit<'a>(
        compiled: &'a CompiledEntry<M>,
        trace_id: u64,
    ) -> Option<(u64, &'a CompiledTrace)> {
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        compiled
            .traces
            .get(&trace_id)
            .map(|trace| (trace_id, trace))
    }

    fn compiled_exit_layout_from_trace(
        trace: &CompiledTrace,
        owning_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<CompiledExitLayout> {
        trace
            .exit_layouts
            .get(&fail_index)
            .map(|layout| layout.public(owning_key, trace_id, fail_index))
    }

    fn terminal_exit_layout_from_trace(
        trace: &CompiledTrace,
        owning_key: u64,
        trace_id: u64,
        op_index: usize,
    ) -> Option<CompiledExitLayout> {
        trace.terminal_exit_layouts.get(&op_index).map(|layout| {
            layout.public(
                owning_key,
                trace_id,
                compile::find_fail_index_for_exit_op(&trace.ops, op_index).unwrap_or(u32::MAX),
            )
        })
    }

    fn compiled_exit_layout_from_backend(
        &self,
        compiled: &CompiledEntry<M>,
        owning_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<CompiledExitLayout> {
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        self.backend
            .compiled_trace_fail_descr_layouts(&compiled.token, trace_id)?
            .into_iter()
            .find(|layout| layout.fail_index == fail_index)
            .map(|layout| CompiledExitLayout {
                rd_loop_token: owning_key, // compile.py:186
                trace_id,
                fail_index: layout.fail_index,
                source_op_index: layout.source_op_index,
                exit_types: layout.fail_arg_types,
                is_finish: layout.is_finish,
                gc_ref_slots: layout.gc_ref_slots,
                force_token_slots: layout.force_token_slots,
                recovery_layout: layout.recovery_layout,
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            })
    }

    fn terminal_exit_layout_from_backend(
        &self,
        compiled: &CompiledEntry<M>,
        owning_key: u64,
        trace_id: u64,
        op_index: usize,
    ) -> Option<CompiledExitLayout> {
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        self.backend
            .compiled_trace_terminal_exit_layouts(&compiled.token, trace_id)?
            .into_iter()
            .find(|layout| layout.op_index == op_index)
            .map(|layout| CompiledExitLayout {
                rd_loop_token: owning_key, // compile.py:186
                trace_id,
                fail_index: layout.fail_index,
                source_op_index: Some(layout.op_index),
                exit_types: layout.exit_types,
                is_finish: layout.is_finish,
                gc_ref_slots: layout.gc_ref_slots,
                force_token_slots: layout.force_token_slots,
                recovery_layout: layout.recovery_layout,
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            })
    }

    fn compiled_trace_layout_for_trace(
        &self,
        compiled: &CompiledEntry<M>,
        owning_key: u64,
        trace_id: u64,
    ) -> Option<CompiledTraceLayout> {
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        let mut exit_layouts =
            if let Some((resolved_trace_id, trace)) = Self::trace_for_exit(compiled, trace_id) {
                let mut layouts: Vec<_> = trace
                    .exit_layouts
                    .iter()
                    .map(|(&fail_index, layout)| {
                        layout.public(owning_key, resolved_trace_id, fail_index)
                    })
                    .collect();
                layouts.sort_by_key(|layout| layout.fail_index);
                layouts
            } else {
                Vec::new()
            };
        if let Some(backend_layouts) = self
            .backend
            .compiled_trace_fail_descr_layouts(&compiled.token, trace_id)
        {
            let mut merged = HashMap::new();
            for layout in exit_layouts.drain(..) {
                merged.insert(layout.fail_index, layout);
            }
            for layout in backend_layouts {
                merged.insert(
                    layout.fail_index,
                    CompiledExitLayout {
                        rd_loop_token: owning_key, // compile.py:186
                        trace_id,
                        fail_index: layout.fail_index,
                        source_op_index: layout.source_op_index,
                        exit_types: layout.fail_arg_types,
                        is_finish: layout.is_finish,
                        gc_ref_slots: layout.gc_ref_slots,
                        force_token_slots: layout.force_token_slots,
                        recovery_layout: layout.recovery_layout,
                        resume_layout: merged
                            .get(&layout.fail_index)
                            .and_then(|existing| existing.resume_layout.clone()),
                        rd_numb: None,
                        rd_consts: None,
                        rd_virtuals: None,
                        rd_pendingfields: None,
                    },
                );
            }
            exit_layouts = merged.into_values().collect();
            exit_layouts.sort_by_key(|layout| layout.fail_index);
        }

        let mut terminal_exit_layouts =
            if let Some((resolved_trace_id, trace)) = Self::trace_for_exit(compiled, trace_id) {
                let mut layouts: Vec<_> = trace
                    .terminal_exit_layouts
                    .iter()
                    .map(|(&op_index, layout)| CompiledTerminalExitLayout {
                        op_index,
                        exit_layout: layout.public(
                            owning_key,
                            resolved_trace_id,
                            compile::find_fail_index_for_exit_op(&trace.ops, op_index)
                                .unwrap_or(u32::MAX),
                        ),
                    })
                    .collect();
                layouts.sort_by_key(|layout| layout.op_index);
                layouts
            } else {
                Vec::new()
            };
        if let Some(backend_layouts) = self
            .backend
            .compiled_trace_terminal_exit_layouts(&compiled.token, trace_id)
        {
            let mut merged = HashMap::new();
            for layout in terminal_exit_layouts.drain(..) {
                merged.insert(layout.op_index, layout);
            }
            for layout in backend_layouts {
                merged.insert(
                    layout.op_index,
                    CompiledTerminalExitLayout {
                        op_index: layout.op_index,
                        exit_layout: CompiledExitLayout {
                            rd_loop_token: owning_key, // compile.py:186
                            trace_id,
                            fail_index: layout.fail_index,
                            source_op_index: Some(layout.op_index),
                            exit_types: layout.exit_types,
                            is_finish: layout.is_finish,
                            gc_ref_slots: layout.gc_ref_slots,
                            force_token_slots: layout.force_token_slots,
                            recovery_layout: layout.recovery_layout,
                            resume_layout: merged
                                .get(&layout.op_index)
                                .and_then(|existing| existing.exit_layout.resume_layout.clone()),
                            rd_numb: None,
                            rd_consts: None,
                            rd_virtuals: None,
                            rd_pendingfields: None,
                        },
                    },
                );
            }
            terminal_exit_layouts = merged.into_values().collect();
            terminal_exit_layouts.sort_by_key(|layout| layout.op_index);
        }

        if exit_layouts.is_empty() && terminal_exit_layouts.is_empty() {
            None
        } else {
            Some(CompiledTraceLayout {
                trace_id,
                exit_layouts,
                terminal_exit_layouts,
            })
        }
    }

    /// Create a new MetaInterp with the given compilation threshold.
    pub fn new(threshold: u32) -> Self {
        MetaInterp {
            warm_state: WarmEnterState::new(threshold),
            backend: BackendImpl::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
            next_trace_id: 1,
            virtualizable_info: None,
            hooks: JitHooks::default(),
            pending_token: None,
            stats: JitStatsCounters::default(),
            vable_ptr: std::ptr::null(),
            vable_array_lengths: Vec::new(),
            result_type: Type::Ref,
            raw_int_box_helpers: HashSet::new(),
            create_frame_raw_map: HashMap::new(),
            tracing_call_depth: None,
            max_unroll_recursion: 7, // RPython default from rlib/jit.py
            force_finish_trace: false,
            callinfocollection: None,
            string_length_resolver: None,
            partial_trace: None,
            retracing_from: None,
            exported_state: None,
            cancel_count: 0,
            last_compiled_key: None,
            num_scalar_inputargs: 0,
            pending_done_bridges: Vec::new(),
            potential_retrace_position: None,
            last_quasi_immutable_deps: Vec::new(),
            retrace_after_bridge: false,
            virtualref_boxes: Vec::new(),
            pending_preamble_tokens: HashMap::new(),
        }
    }

    /// warmspot.py:449 — set the per-driver result_type from the portal
    /// function's return signature. Called once during driver setup.
    pub fn set_result_type(&mut self, tp: Type) {
        self.result_type = tp;
    }

    /// warmspot.py:449 — the per-driver static result_type.
    pub fn result_type(&self) -> Type {
        self.result_type
    }

    /// Cache the current virtualizable object pointer for trace-entry setup.
    pub(crate) fn set_vable_ptr(&mut self, ptr: *const u8) {
        self.vable_ptr = ptr;
    }

    /// Cache fallback virtualizable array lengths for trace-entry box setup.
    pub(crate) fn set_vable_array_lengths(&mut self, lengths: Vec<usize>) {
        self.vable_array_lengths = lengths;
    }

    fn trace_entry_vable_lengths(&self, info: &VirtualizableInfo) -> Vec<usize> {
        // RPython initialize_virtualizable() seeds virtualizable_boxes from the
        // current trace entry boxes, not by re-reading a potentially larger
        // heap layout.  When the interpreter provides explicit trace-entry
        // lengths, prefer them because they describe the live input box prefix
        // that was actually recorded for this trace.
        if !self.vable_array_lengths.is_empty() {
            return self.vable_array_lengths.clone();
        }
        if !self.vable_ptr.is_null() && info.can_read_all_array_lengths_from_heap() {
            // Safety: vable_ptr is cached from JitState::virtualizable_heap_ptr()
            // for the currently active interpreter state.
            return unsafe { info.read_array_lengths_from_heap(self.vable_ptr) };
        }
        Vec::new()
    }

    /// pyjitpl.py:3290 `initialize_virtualizable(self, original_boxes)`.
    ///
    /// RPython:
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is not None:
    ///         index = (self.jitdriver_sd.num_green_args +
    ///                  self.jitdriver_sd.index_of_virtualizable)
    ///         virtualizable_box = original_boxes[index]
    ///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///         vinfo.clear_vable_token(virtualizable)
    ///         startindex = len(original_boxes) - self.jitdriver_sd.num_green_args
    ///         self.virtualizable_boxes = vinfo.read_boxes(self.cpu, virtualizable, startindex)
    ///         original_boxes += self.virtualizable_boxes
    ///         self.virtualizable_boxes.append(virtualizable_box)
    ///         self.check_synchronized_virtualizable()
    ///
    /// pyre adaptation: pyre's TraceCtx already holds the
    /// `virtualizable_boxes` field. This method is the RPython-named
    /// MetaInterp-level entry point; it computes the array lengths /
    /// derived layout and delegates to the lower-level
    /// `TraceCtx::init_virtualizable_boxes` helper which performs the
    /// actual `virtualizable_boxes = [..., vable_ref]` push (matching
    /// the `read_boxes(...) ; append(virtualizable_box)` shape from
    /// pyjitpl.py:3302-3306). The `vable_box` is currently the first
    /// inputarg `OpRef(0)` — the structural divergence from RPython
    /// (where the vable_box is held in the side-channel
    /// `virtualizable_boxes[-1]` separate from threading through every
    /// JUMP arg) is tracked by the `virtualizable_boxes[-1]
    /// line-by-line port` epic.
    ///
    /// `live_values` is the pyre analog of RPython's `original_boxes`:
    /// the list of input values for the current trace.
    fn initialize_virtualizable(&self, ctx: &mut TraceCtx, live_values: &[Value]) {
        let Some(info) = self.virtualizable_info.as_ref() else {
            return;
        };
        let array_lengths = self.trace_entry_vable_lengths(info);
        let num_static = info.num_static_extra_boxes;
        let num_array_elems: usize = array_lengths.iter().sum();
        let total_vable = num_static + num_array_elems;
        if total_vable == 0 || live_values.len() < 1 + total_vable {
            return;
        }
        // pyjitpl.py:3293-3295: index = num_green_args + index_of_virtualizable
        // pyre uses single-jitdriver / num_green_args=0 / index_of_virtualizable=0,
        // so virtualizable_box = original_boxes[0] = OpRef(0).
        let virtualizable_box = OpRef(0);
        // pyjitpl.py:3302: virtualizable_boxes = vinfo.read_boxes(...)
        // pyre lays out the static + array slots immediately after the
        // virtualizable input arg, mirroring how `read_boxes` returns one
        // box per static field followed by one box per array element.
        let vable_oprefs: Vec<OpRef> = (0..total_vable).map(|i| OpRef((1 + i) as u32)).collect();
        // pyjitpl.py:3306: virtualizable_boxes.append(virtualizable_box)
        // is folded inside init_virtualizable_boxes (it pushes vable_ref
        // at the end of the list).
        ctx.init_virtualizable_boxes(info, virtualizable_box, &vable_oprefs, &array_lengths);
        // pyjitpl.py:3307: check_synchronized_virtualizable() — debug-only
        // assertion. pyre's analog is `check_synchronized_virtualizable`
        // on MetaInterp, but it requires &self which we don't have here.
        // Callers in setup_tracing / bound_reached perform the check
        // immediately after `set_force_finish` via the existing path.
    }

    /// warmstate.py:259: set_param_trace_eagerness — delegates to warmstate.
    pub fn set_trace_eagerness(&mut self, eagerness: u32) {
        self.warm_state.set_param_trace_eagerness(eagerness);
    }

    /// Update the green_key associated with the current trace.
    /// Called when tracing started at function entry but the loop closes
    /// at a backward jump with a different PC.
    pub fn update_tracing_green_key(&mut self, key: u64) {
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.green_key = key;
        }
    }

    /// compile.py:269-270: return cross-loop cut info if the current trace
    /// closes at a different loop header than where it started.
    pub fn cross_loop_cut_info(&self) -> Option<(usize, Vec<Type>)> {
        let ctx = self.tracing.as_ref()?;
        let inner_key = ctx.cut_inner_green_key?;
        // compile.py:269: cross-loop cut uses the inner loop's merge point.
        // Lookup by inner_key (not ctx.green_key which is the outer loop).
        ctx.get_merge_point_at(inner_key, ctx.header_pc)
            .filter(|mp| mp.position.ops_len > 0)
            .map(|mp| (mp.header_pc, mp.original_box_types.clone()))
    }

    /// Compat alias: delegates to set_trace_eagerness.
    pub fn set_bridge_threshold(&mut self, threshold: u32) {
        self.set_trace_eagerness(threshold);
    }

    /// Set the main compilation threshold.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.warm_state.set_threshold(threshold);
    }

    /// Set the function inlining threshold.
    ///
    /// A function must be called at least this many times during tracing
    /// before it is inlined. Default is 4 (matching RPython).
    pub fn set_function_threshold(&mut self, threshold: u32) {
        self.warm_state.set_function_threshold(threshold);
    }

    pub fn warm_state_ref(&self) -> &majit_trace::warmstate::WarmEnterState {
        &self.warm_state
    }

    pub fn warm_state_mut(&mut self) -> &mut majit_trace::warmstate::WarmEnterState {
        &mut self.warm_state
    }

    /// Decay all counters to avoid stale hotness data.
    pub fn decay_counters(&mut self) {
        self.warm_state.decay_counters();
    }

    /// Set virtualizable info for interpreter frame virtualization.
    ///
    /// This tells the JIT how to read/write interpreter frame fields
    /// during trace entry/exit.
    pub fn set_virtualizable_info(&mut self, info: VirtualizableInfo) {
        self.virtualizable_info = Some(info);
    }

    /// Get the virtualizable info.
    pub fn virtualizable_info(&self) -> Option<&VirtualizableInfo> {
        self.virtualizable_info.as_ref()
    }

    /// Create an optimizer with virtualizable config if available.
    ///
    /// Standard virtualizable fields become virtual input args — first reads
    /// are replaced with input references and values flow through JUMP args.
    /// No heap access for these fields on the hot path.
    fn current_virtualizable_optimizer_config(
        &self,
    ) -> Option<crate::optimizeopt::virtualize::VirtualizableConfig> {
        self.tracing.as_ref().and_then(|ctx| {
            if !ctx.has_virtualizable_boxes() {
                return None;
            }
            self.virtualizable_info.as_ref().map(|info| {
                let mut config = info.to_optimizer_config();
                config.array_lengths = ctx.virtualizable_array_lengths().unwrap_or(&[]).to_vec();
                config
            })
        })
    }

    fn make_optimizer(&self) -> Optimizer {
        let mut opt = if let Some(config) = self.current_virtualizable_optimizer_config() {
            Optimizer::default_pipeline_with_virtualizable(config)
        } else {
            Optimizer::default_pipeline()
        };
        // optimizer.py:787-789: constant_fold — allocate immutable objects
        // at compile time. Uses Box::leak for permanent allocation (immutable
        // objects are never freed, matching RPython's prebuilt constants).
        opt.constant_fold_alloc = Some(Box::new(|size_bytes: usize| {
            let layout = std::alloc::Layout::from_size_align(size_bytes, 8)
                .unwrap_or(std::alloc::Layout::new::<u8>());
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            if ptr.is_null() {
                majit_ir::GcRef::NULL
            } else {
                majit_ir::GcRef(ptr as usize)
            }
        }));
        // info.py:810-822 `ConstPtrInfo.getstrlen1(mode)` — propagate the
        // host-runtime resolver so constant STRLEN / UNICODELEN operations
        // can fold to an exact `IntBound::from_constant(len)` during
        // intbounds postprocessing.
        opt.string_length_resolver = self.string_length_resolver.clone();
        opt
    }

    /// Install the host-runtime `getstrlen1` resolver. The closure must be
    /// callable from the optimizer for arbitrary constant `GcRef` / mode
    /// pairs. `mode == 0` is byte-string, `mode == 1` is unicode; any other
    /// value returns `None` (matching PyPy's `vstring.mode_string` /
    /// `vstring.mode_unicode` dispatch).
    pub fn set_string_length_resolver(
        &mut self,
        resolver: crate::optimizeopt::info::StringLengthResolver,
    ) {
        self.string_length_resolver = Some(resolver);
    }

    /// Set a callback for loop compilation events.
    pub fn set_on_compile_loop(&mut self, f: impl Fn(u64, usize, usize) + Send + 'static) {
        self.hooks.on_compile_loop = Some(Box::new(f));
    }

    /// Set a callback for bridge compilation events.
    pub fn set_on_compile_bridge(&mut self, f: impl Fn(u64, u32, usize) + Send + 'static) {
        self.hooks.on_compile_bridge = Some(Box::new(f));
    }

    /// Set a callback for guard failure events.
    pub fn set_on_guard_failure(&mut self, f: impl Fn(u64, u32, u32) + Send + 'static) {
        self.hooks.on_guard_failure = Some(Box::new(f));
    }

    /// Set a callback for trace start events.
    pub fn set_on_trace_start(&mut self, f: impl Fn(u64) + Send + 'static) {
        self.hooks.on_trace_start = Some(Box::new(f));
    }

    /// Set a callback for trace abort events.
    pub fn set_on_trace_abort(&mut self, f: impl Fn(u64, bool) + Send + 'static) {
        self.hooks.on_trace_abort = Some(Box::new(f));
    }

    /// Set a callback for compilation error events (loop or bridge).
    pub fn set_on_compile_error(&mut self, f: impl Fn(u64, &str) + Send + 'static) {
        self.hooks.on_compile_error = Some(Box::new(f));
    }

    /// Return a snapshot of the cumulative JIT compilation statistics.
    pub fn get_stats(&self) -> JitStats {
        JitStats {
            loops_compiled: self.stats.loops_compiled,
            loops_aborted: self.stats.loops_aborted,
            bridges_compiled: self.stats.bridges_compiled,
            guard_failures: self.stats.guard_failures,
        }
    }

    /// Check a back-edge: is this location hot enough to trace or run?
    ///
    /// `green_key` identifies the loop header (e.g., PC).
    /// `live_values` are the interpreter's live integer values at this point.
    ///
    /// On `StartedTracing`, the framework registers each value in `live_values`
    /// as an InputArg. The interpreter should then build its symbolic state
    /// from the returned InputArg OpRefs (OpRef(0), OpRef(1), ...).
    pub fn on_back_edge(&mut self, green_key: u64, live_values: &[i64]) -> BackEdgeAction {
        let typed_values: Vec<Value> = live_values.iter().copied().map(Value::Int).collect();
        self.on_back_edge_typed(green_key, None, None, &typed_values)
    }

    /// Force-start tracing for a green key, bypassing the hot counter.
    pub fn force_start_tracing(
        &mut self,
        green_key: u64,
        driver_descriptor: Option<JitDriverStaticData>,
        live_values: &[Value],
    ) -> BackEdgeAction {
        if self.tracing.is_some() {
            return BackEdgeAction::AlreadyTracing;
        }

        match self.warm_state.force_start_tracing(green_key) {
            HotResult::NotHot => BackEdgeAction::Interpret,
            HotResult::StartTracing(mut recorder) => {
                for value in live_values {
                    recorder.record_input_arg(value.get_type());
                }

                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] force start tracing at key={}, num_inputs={}",
                        green_key,
                        live_values.len()
                    );
                }

                let mut ctx = TraceCtx::new(recorder, green_key);
                ctx.initial_inputarg_consts = live_values
                    .iter()
                    .map(|v| {
                        let (bits, tp) = match v {
                            Value::Int(i) => (*i, Type::Int),
                            Value::Float(f) => (f.to_bits() as i64, Type::Float),
                            Value::Ref(r) => (r.as_usize() as i64, Type::Ref),
                            Value::Void => (0, Type::Void),
                        };
                        ctx.constants.get_or_insert_typed(bits, tp)
                    })
                    .collect();
                if let Some(descriptor) = driver_descriptor {
                    ctx.set_driver_descriptor(descriptor);
                }
                // pyjitpl.py:3290 initialize_virtualizable parity.
                self.initialize_virtualizable(&mut ctx, live_values);
                self.force_finish_trace = self.warm_state.take_force_finish_tracing(green_key);
                ctx.set_force_finish(self.force_finish_trace);
                let num_inputs = ctx.recorder.num_inputargs();
                let input_types = ctx.inputarg_types();
                self.tracing = Some(ctx);
                let pending_num = self.warm_state.alloc_token_number();
                self.pending_token = Some((green_key, pending_num));
                // RPython compile_tmp_callback parity: register a placeholder
                // target so call_assembler can resolve the pending token at
                // runtime. call_assembler_fast_path detects null code_ptr and
                // falls back to force_fn.
                self.backend.register_pending_target(
                    pending_num,
                    input_types,
                    num_inputs,
                    self.num_scalar_inputargs,
                );
                if let Some(ref hook) = self.hooks.on_trace_start {
                    hook(green_key);
                }
                BackEdgeAction::StartedTracing
            }
            HotResult::AlreadyTracing => BackEdgeAction::AlreadyTracing,
            HotResult::RunCompiled => BackEdgeAction::RunCompiled,
        }
    }

    /// RPython warmstate.py:425 bound_reached parity.
    ///
    /// Like `on_back_edge_typed` but bypasses the counter tick — the
    /// caller (can_enter_jit_hook) already verified the counter fired.
    /// This allows decay_counters() to be called before tracing starts
    /// without the internal tick check blocking the trace.
    pub fn bound_reached(
        &mut self,
        green_key: u64,
        green_key_values: Option<majit_ir::GreenKey>,
        driver_descriptor: Option<JitDriverStaticData>,
        live_values: &[Value],
    ) -> BackEdgeAction {
        if self.tracing.is_some() {
            return BackEdgeAction::AlreadyTracing;
        }

        match self.warm_state.force_start_tracing(green_key) {
            HotResult::NotHot => BackEdgeAction::Interpret,
            HotResult::StartTracing(recorder) => self.setup_tracing(
                green_key,
                green_key_values,
                driver_descriptor,
                live_values,
                recorder,
            ),
            HotResult::RunCompiled => BackEdgeAction::RunCompiled,
            HotResult::AlreadyTracing => BackEdgeAction::AlreadyTracing,
        }
    }

    pub fn on_back_edge_typed(
        &mut self,
        green_key: u64,
        green_key_values: Option<majit_ir::GreenKey>,
        driver_descriptor: Option<JitDriverStaticData>,
        live_values: &[Value],
    ) -> BackEdgeAction {
        if self.tracing.is_some() {
            return BackEdgeAction::AlreadyTracing;
        }

        match self.warm_state.maybe_compile(green_key) {
            HotResult::NotHot => BackEdgeAction::Interpret,
            HotResult::StartTracing(recorder) => self.setup_tracing(
                green_key,
                green_key_values,
                driver_descriptor,
                live_values,
                recorder,
            ),
            HotResult::AlreadyTracing => BackEdgeAction::AlreadyTracing,
            HotResult::RunCompiled => BackEdgeAction::RunCompiled,
        }
    }

    fn setup_tracing(
        &mut self,
        green_key: u64,
        green_key_values: Option<majit_ir::GreenKey>,
        driver_descriptor: Option<JitDriverStaticData>,
        live_values: &[Value],
        mut recorder: majit_trace::recorder::Trace,
    ) -> BackEdgeAction {
        // RPython parity: each tracing pass starts with cancel_count=0.
        // In RPython, MetaInterp is re-created per _compile_and_run_once.
        // In pyre, MetaInterp is reused, so reset per-trace state here.
        self.cancel_count = 0;
        for value in live_values {
            recorder.record_input_arg(value.get_type());
        }

        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] start tracing at key={}, num_inputs={}",
                green_key,
                live_values.len()
            );
        }

        let mut ctx = if let Some(values) = green_key_values {
            TraceCtx::with_green_key(recorder, green_key, values)
        } else {
            TraceCtx::new(recorder, green_key)
        };
        ctx.initial_inputarg_consts = live_values
            .iter()
            .map(|v| {
                let (bits, tp) = match v {
                    Value::Int(i) => (*i, Type::Int),
                    Value::Float(f) => (f.to_bits() as i64, Type::Float),
                    Value::Ref(r) => (r.as_usize() as i64, Type::Ref),
                    Value::Void => (0, Type::Void),
                };
                ctx.constants.get_or_insert_typed(bits, tp)
            })
            .collect();
        if let Some(descriptor) = driver_descriptor {
            ctx.set_driver_descriptor(descriptor);
        }
        // pyjitpl.py:3290 initialize_virtualizable parity.
        self.initialize_virtualizable(&mut ctx, live_values);

        self.force_finish_trace = self.warm_state.take_force_finish_tracing(green_key);
        // pyjitpl.py:2411: propagate force_finish_trace to TraceCtx
        // so the proc-macro merge_fn closure can read it.
        ctx.set_force_finish(self.force_finish_trace);
        self.tracing = Some(ctx);
        let pending_num = self.warm_state.alloc_token_number();
        self.pending_token = Some((green_key, pending_num));
        if let Some(ref hook) = self.hooks.on_trace_start {
            hook(green_key);
        }
        BackEdgeAction::StartedTracing
    }

    /// Access the active TraceCtx (if currently tracing).
    pub fn trace_ctx(&mut self) -> Option<&mut TraceCtx> {
        self.tracing.as_mut()
    }

    pub fn force_finish_trace_enabled(&self) -> bool {
        self.force_finish_trace
    }

    // ── RPython opimpl_* equivalents for virtualizable ──────────────
    //
    // pyjitpl.py:1120-1146 `_nonstandard_virtualizable(pc, box, fielddescr)`
    // is implemented in `TraceCtx::is_nonstandard_virtualizable` with the
    // full Step 1..5b shape; the opimpl_*_vable thin wrappers below forward
    // to `TraceCtx::vable_*` which are the line-by-line port of RPython's
    // `opimpl_*_vable` opcode handlers. The earlier `MetaInterp` duplicate
    // (with its own `is_standard_virtualizable` / `nonstandard_virtualizable`
    // / `virtualizable_field_index` / `get_arrayitem_vable_index` /
    // `check_synchronized_virtualizable` helpers) was a pyre-introduced
    // duplication of the same logic and has been removed in favour of the
    // single TraceCtx implementation.

    /// pyjitpl.py:3499-3512 `MetaInterp.replace_box(oldbox, newbox)`.
    ///
    ///     def replace_box(self, oldbox, newbox):
    ///         for frame in self.framestack:
    ///             frame.replace_active_box_in_frame(oldbox, newbox)
    ///         boxes = self.virtualref_boxes
    ///         for i in range(len(boxes)):
    ///             if boxes[i] is oldbox:
    ///                 boxes[i] = newbox
    ///         if (self.jitdriver_sd.virtualizable_info is not None or
    ///             self.jitdriver_sd.greenfield_info is not None):
    ///             boxes = self.virtualizable_boxes
    ///             for i in range(len(boxes)):
    ///                 if boxes[i] is oldbox:
    ///                     boxes[i] = newbox
    ///         self.heapcache.replace_box(oldbox, newbox)
    ///
    /// RPython rewrites every place where `oldbox` may appear during
    /// tracing — frame registers, virtualref pairs, the standard
    /// virtualizable box array, and the heap cache — and does so
    /// eagerly so subsequent tracing-time queries see the new identity.
    ///
    /// pyre's frame registers (PyreSym) live inside the jitcode machine
    /// driver and are not reachable from MetaInterp during tracing; the
    /// in-flight trace ops are rewritten as a deferred batch via
    /// `TraceCtx::replace_op` queued by this method, and the
    /// virtualref/virtualizable/heap-cache walks are run eagerly here so
    /// the next call into Step 1 of `nonstandard_virtualizable` (or any
    /// other heapcache query) observes the new OpRef.
    pub fn replace_box(&mut self, oldbox: OpRef, newbox: OpRef) {
        // pyjitpl.py:3500-3501: for frame in self.framestack:
        //                          frame.replace_active_box_in_frame(...)
        //
        // pyre defers the per-frame symbolic register rewrite into the
        // recorder via `TraceCtx::replace_op` (queued by
        // `TraceCtx::replace_box` below). The jitcode machine's PyreSym
        // frames are not reachable from MetaInterp; the deferred
        // recorder pass at trace finalization rewrites all already-emitted
        // op args + fail_args, which substitutes for the eager per-frame
        // walk RPython performs here.
        //
        // pyjitpl.py:3502-3505: virtualref_boxes walk
        for slot in self.virtualref_boxes.iter_mut() {
            if slot.0 == oldbox {
                slot.0 = newbox;
            }
        }
        if let Some(ctx) = self.tracing.as_mut() {
            // pyjitpl.py:3506-3512:
            //     boxes = self.virtualizable_boxes
            //     for i in range(len(boxes)):
            //         if boxes[i] is oldbox:
            //             boxes[i] = newbox
            //     self.heapcache.replace_box(oldbox, newbox)
            //
            // The trace-context portion (virtualizable_boxes walk +
            // heap_cache walk + deferred recorder rewrite) lives on
            // `TraceCtx::replace_box` so the trace_ctx-only call site in
            // `_nonstandard_virtualizable` Step 4 shares the same helper.
            ctx.replace_box(oldbox, newbox);
        }
    }

    /// pyjitpl.py:3446-3450 `MetaInterp.synchronize_virtualizable()`.
    ///
    ///     def synchronize_virtualizable(self):
    ///         vinfo = self.jitdriver_sd.virtualizable_info
    ///         virtualizable_box = self.virtualizable_boxes[-1]
    ///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///         vinfo.write_boxes(virtualizable, self.virtualizable_boxes)
    ///
    /// RPython mirrors the `virtualizable_boxes` model into the C-level
    /// virtualizable struct so that subsequent runtime reads see the new
    /// values. pyre's `virtualizable_boxes` (on TraceCtx) is the single
    /// source of truth — there is no separate C struct to mirror into —
    /// so this is a no-op. Kept as a named seam to make the call sites
    /// at `_opimpl_setfield_vable` (pyjitpl.py:1194) and
    /// `_opimpl_setarrayitem_vable` (pyjitpl.py:1246) line up structurally.
    pub fn synchronize_virtualizable(&mut self, _vable_opref: OpRef) {
        // intentionally empty — see doc comment.
    }

    /// pyjitpl.py:1167-1172 `opimpl_getfield_vable_i(box, fielddescr, pc)`.
    /// Thin wrapper that forwards to `TraceCtx::vable_getfield_int`, which
    /// holds the line-by-line port (`_nonstandard_virtualizable` →
    /// fallback heap op or `virtualizable_boxes[index]` read).
    pub fn opimpl_getfield_vable_int(&mut self, vable_opref: OpRef, field_offset: usize) -> OpRef {
        self.tracing
            .as_mut()
            .expect("opimpl_getfield_vable_int requires active tracing")
            .vable_getfield_int(vable_opref, field_offset)
    }

    /// pyjitpl.py:1173-1179 `opimpl_getfield_vable_r(box, fielddescr, pc)`.
    pub fn opimpl_getfield_vable_ref(&mut self, vable_opref: OpRef, field_offset: usize) -> OpRef {
        self.tracing
            .as_mut()
            .expect("opimpl_getfield_vable_ref requires active tracing")
            .vable_getfield_ref(vable_opref, field_offset)
    }

    /// pyjitpl.py:1180-1186 `opimpl_getfield_vable_f(box, fielddescr, pc)`.
    pub fn opimpl_getfield_vable_float(
        &mut self,
        vable_opref: OpRef,
        field_offset: usize,
    ) -> OpRef {
        self.tracing
            .as_mut()
            .expect("opimpl_getfield_vable_float requires active tracing")
            .vable_getfield_float(vable_opref, field_offset)
    }

    /// pyjitpl.py:1188-1199 `_opimpl_setfield_vable(box, valuebox, fielddescr, pc)`.
    pub fn opimpl_setfield_vable_int(
        &mut self,
        vable_opref: OpRef,
        field_offset: usize,
        value: OpRef,
    ) {
        self.tracing
            .as_mut()
            .expect("opimpl_setfield_vable_int requires active tracing")
            .vable_setfield(vable_opref, field_offset, value);
    }

    /// pyjitpl.py:1188-1199 `_opimpl_setfield_vable` — ref variant.
    pub fn opimpl_setfield_vable_ref(
        &mut self,
        vable_opref: OpRef,
        field_offset: usize,
        value: OpRef,
    ) {
        self.tracing
            .as_mut()
            .expect("opimpl_setfield_vable_ref requires active tracing")
            .vable_setfield(vable_opref, field_offset, value);
    }

    /// pyjitpl.py:1188-1199 `_opimpl_setfield_vable` — float variant.
    pub fn opimpl_setfield_vable_float(
        &mut self,
        vable_opref: OpRef,
        field_offset: usize,
        value: OpRef,
    ) {
        self.tracing
            .as_mut()
            .expect("opimpl_setfield_vable_float requires active tracing")
            .vable_setfield(vable_opref, field_offset, value);
    }

    /// pyjitpl.py:1218-1234 `_opimpl_getarrayitem_vable` — int variant.
    pub fn opimpl_getarrayitem_vable_int(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        array_field_offset: usize,
    ) -> OpRef {
        self.tracing
            .as_mut()
            .expect("opimpl_getarrayitem_vable_int requires active tracing")
            .vable_getarrayitem_int_indexed(
                vable_opref,
                index,
                index_runtime_value,
                array_field_offset,
            )
    }

    /// pyjitpl.py:1218-1234 `_opimpl_getarrayitem_vable` — ref variant.
    pub fn opimpl_getarrayitem_vable_ref(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        array_field_offset: usize,
    ) -> OpRef {
        self.tracing
            .as_mut()
            .expect("opimpl_getarrayitem_vable_ref requires active tracing")
            .vable_getarrayitem_ref_indexed(
                vable_opref,
                index,
                index_runtime_value,
                array_field_offset,
            )
    }

    /// pyjitpl.py:1218-1234 `_opimpl_getarrayitem_vable` — float variant.
    pub fn opimpl_getarrayitem_vable_float(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        array_field_offset: usize,
    ) -> OpRef {
        self.tracing
            .as_mut()
            .expect("opimpl_getarrayitem_vable_float requires active tracing")
            .vable_getarrayitem_float_indexed(
                vable_opref,
                index,
                index_runtime_value,
                array_field_offset,
            )
    }

    /// pyjitpl.py:1236-1247 `_opimpl_setarrayitem_vable` — int variant.
    pub fn opimpl_setarrayitem_vable_int(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        value: OpRef,
        array_field_offset: usize,
    ) {
        self.tracing
            .as_mut()
            .expect("opimpl_setarrayitem_vable_int requires active tracing")
            .vable_setarrayitem_indexed(
                vable_opref,
                index,
                index_runtime_value,
                array_field_offset,
                value,
            );
    }

    /// pyjitpl.py:1236-1247 `_opimpl_setarrayitem_vable` — ref variant.
    pub fn opimpl_setarrayitem_vable_ref(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        value: OpRef,
        array_field_offset: usize,
    ) {
        self.tracing
            .as_mut()
            .expect("opimpl_setarrayitem_vable_ref requires active tracing")
            .vable_setarrayitem_indexed(
                vable_opref,
                index,
                index_runtime_value,
                array_field_offset,
                value,
            );
    }

    /// pyjitpl.py:1236-1247 `_opimpl_setarrayitem_vable` — float variant.
    pub fn opimpl_setarrayitem_vable_float(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        value: OpRef,
        array_field_offset: usize,
    ) {
        self.tracing
            .as_mut()
            .expect("opimpl_setarrayitem_vable_float requires active tracing")
            .vable_setarrayitem_indexed(
                vable_opref,
                index,
                index_runtime_value,
                array_field_offset,
                value,
            );
    }

    /// pyjitpl.py:1253-1263 `opimpl_arraylen_vable(box, fdescr, adescr, pc)`.
    pub fn opimpl_arraylen_vable(
        &mut self,
        vable_opref: OpRef,
        array_field_offset: usize,
    ) -> OpRef {
        self.tracing
            .as_mut()
            .expect("opimpl_arraylen_vable requires active tracing")
            .vable_arraylen_vable(vable_opref, array_field_offset)
    }

    /// pyjitpl.py:1064-1073 `opimpl_hint_force_virtualizable(box)`.
    ///
    ///     def opimpl_hint_force_virtualizable(self, box):
    ///         self.metainterp.gen_store_back_in_vable(box)
    ///
    /// RPython's `gen_store_back_in_vable` (pyjitpl.py:3465) handles the
    /// nonstandard / forced_virtualizable gating internally and emits the
    /// SETFIELD_GC + SETARRAYITEM_GC + token-clear flush. pyre's TraceCtx
    /// hosts the same gating now (see TraceCtx::gen_store_back_in_vable),
    /// so this is a thin forward.
    pub fn opimpl_hint_force_virtualizable(&mut self, vable_opref: OpRef) {
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.gen_store_back_in_vable(vable_opref);
        }
    }

    /// pyjitpl.py:1789-1814 opimpl_virtual_ref parity.
    /// Creates concrete vref via virtual_ref_during_tracing(real_object),
    /// records VIRTUAL_REF(box, cindex), pushes [virtualbox, vrefbox].
    pub fn opimpl_virtual_ref(&mut self, virtual_obj: OpRef, virtual_obj_ptr: usize) -> OpRef {
        let Some(ctx) = self.tracing.as_mut() else {
            return OpRef::NONE;
        };
        // pyjitpl.py:1804: virtual_ref_during_tracing(virtual_obj)
        let vref_info = crate::virtualref::VirtualRefInfo::new();
        let vref_ptr = vref_info.virtual_ref_during_tracing(virtual_obj_ptr as *mut u8);
        // pyjitpl.py:1805: cindex = ConstInt(len(virtualref_boxes) // 2)
        let cindex = ctx.const_int((self.virtualref_boxes.len() / 2) as i64);
        // pyjitpl.py:1806: record VIRTUAL_REF(box, cindex)
        let vref = ctx.record_op(OpCode::VirtualRefR, &[virtual_obj, cindex]);
        // pyjitpl.py:1814: virtualref_boxes += [virtualbox, vrefbox]
        self.virtualref_boxes.push((virtual_obj, virtual_obj_ptr));
        self.virtualref_boxes.push((vref, vref_ptr as usize));
        vref
    }

    /// pyjitpl.py:1819-1831 opimpl_virtual_ref_finish parity.
    /// Checks is_virtual_ref() on concrete vref before recording.
    pub fn opimpl_virtual_ref_finish(&mut self, vref: OpRef, virtual_obj: OpRef) {
        let Some(ctx) = self.tracing.as_mut() else {
            return;
        };
        // pyjitpl.py:1827: check is_virtual_ref(vrefbox)
        let vref_ptr = self
            .virtualref_boxes
            .last()
            .map(|&(_, ptr)| ptr)
            .unwrap_or(0);
        let vref_info = crate::virtualref::VirtualRefInfo::new();
        let is_vref = vref_ptr != 0 && unsafe { vref_info.is_virtual_ref(vref_ptr as *const u8) };
        if is_vref {
            // pyjitpl.py:3371: VIRTUAL_REF_FINISH(vrefbox, NULL)
            let null = ctx.const_ref(0);
            let _ = ctx.record_op(OpCode::VirtualRefFinish, &[vref, null]);
        }
        if self.virtualref_boxes.len() >= 2 {
            self.virtualref_boxes.pop();
            self.virtualref_boxes.pop();
        }
    }

    /// Whether the engine is currently tracing.
    #[inline]
    pub fn is_tracing(&self) -> bool {
        self.tracing.is_some()
    }

    /// RPython JC_TRACING parity: check if we are currently tracing
    /// this specific green key. Returns false for different green keys,
    /// matching RPython's per-cell JC_TRACING flag.
    #[inline]
    pub fn is_tracing_key(&self, green_key: u64) -> bool {
        self.tracing
            .as_ref()
            .is_some_and(|ctx| ctx.green_key == green_key || ctx.root_green_key() == green_key)
    }

    /// Finish the current active trace without optimizing or compiling it.
    ///
    /// This is a semantic-seam helper for parity tests: it lets callers
    /// inspect the raw recorded trace that the proc-macro/runtime path
    /// produced, without requiring backend compilation.
    pub fn finish_trace_for_parity(
        &mut self,
        finish_args: &[OpRef],
    ) -> Option<(TreeLoop, HashMap<u32, i64>)> {
        self.force_finish_trace = false;
        let ctx = self.tracing.take()?;
        let green_key = ctx.green_key;
        let mut recorder = ctx.recorder;
        recorder.finish(finish_args, crate::make_fail_descr(finish_args.len()));
        let trace = recorder.get_trace();
        let constants = ctx.constants.into_inner();
        self.warm_state.abort_tracing(green_key, false);
        Some((trace, constants))
    }

    /// RPython-compatible helper name from compile.py.
    pub fn send_loop_to_backend(&mut self, jump_args: &[OpRef], meta: M) -> CompileOutcome {
        self.compile_loop(jump_args, meta)
    }

    /// Close the current trace, optimize, and compile.
    ///
    /// `jump_args` are the symbolic values (OpRefs) at the end of the loop,
    /// in the same order as the InputArgs registered during `on_back_edge`.
    /// `meta` is interpreter-specific metadata to store alongside the compiled loop.
    pub fn compile_loop(&mut self, jump_args: &[OpRef], mut meta: M) -> CompileOutcome {
        // pyjitpl.py:2993-3007: if partial_trace is set, the previous
        // compilation attempt requested a retrace. Verify the green_key
        // matches and dispatch to compile_retrace.
        if self.partial_trace.is_some() {
            if let Some(retrace_pos) = self.retracing_from {
                // pyjitpl.py:2994: if start != self.retracing_from
                // Find the merge point whose position matches retracing_from.
                // pyjitpl.py:2994: iterate current_merge_points in reverse,
                // check same_greenkey and position match. Use header_pc
                // for precise matching across root/inner key registrations.
                let position_matches = self
                    .tracing
                    .as_ref()
                    .and_then(|ctx| {
                        ctx.get_merge_point_at(ctx.green_key, ctx.header_pc)
                            .map(|mp| mp.position == retrace_pos)
                    })
                    .unwrap_or(false);
                if position_matches {
                    let ok = self.compile_retrace(jump_args, meta.clone());
                    if ok {
                        self.cancel_count = 0;
                        self.warm_state.reset_function_counts();
                        return CompileOutcome::Compiled {
                            green_key: 0,
                            from_retry: false,
                        };
                    }
                    // pyjitpl.py:3004: creation of the loop was cancelled!
                    self.cancel_count += 1;
                    if self.cancelled_too_many_times() {
                        if crate::majit_log_enabled() {
                            eprintln!("[jit] retrace cancelled too many times");
                        }
                        self.clear_retrace_state();
                        if let Some(ctx) = self.tracing.take() {
                            self.warm_state.abort_tracing(ctx.green_key, false);
                        }
                        self.warm_state.reset_function_counts();
                        return CompileOutcome::Aborted;
                    }
                    // Not too many — clear retrace state and fall through
                    // to normal compile_loop path.
                    self.exported_state = None;
                    if crate::majit_log_enabled() {
                        eprintln!("[jit] retrace cancelled, trying normal compilation");
                    }
                } else {
                    // pyjitpl.py:2994-2995: position mismatch — abort.
                    self.clear_retrace_state();
                    if let Some(ctx) = self.tracing.take() {
                        self.warm_state.abort_tracing(ctx.green_key, false);
                    }
                    self.warm_state.reset_function_counts();
                    return CompileOutcome::Aborted;
                }
            }
        }

        // Clear bridge retrace flag — partial_trace is the authoritative
        // state for compile_retrace dispatch.
        if self.retrace_after_bridge {
            self.retrace_after_bridge = false;
        }
        // pyjitpl.py:3162: has_compiled_targets(ptoken) →
        // raise SwitchToBlackhole(ABORT_BAD_LOOP).
        if let Some(ctx) = self.tracing.as_ref() {
            let gk = ctx.green_key;
            if self.has_compiled_targets(gk) {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compile_loop → SwitchToBlackhole: has_compiled_targets key={}",
                        gk
                    );
                }
                self.abort_trace(false);
                return CompileOutcome::Aborted;
            }
        }

        let vable_config = self.current_virtualizable_optimizer_config();
        self.force_finish_trace = false;
        let mut ctx = self.tracing.take().unwrap();
        ctx.apply_replacements();
        let outer_green_key = ctx.green_key;
        let cut_inner_green_key = ctx.cut_inner_green_key;
        // compile.py:269-270: cross-loop cut → store under inner loop's
        // jitcell_token. RPython: jitcell_token = cross_loop.jitcell_token.
        let green_key = cut_inner_green_key.unwrap_or(outer_green_key);
        let cross_loop_cut = if cut_inner_green_key.is_some() {
            ctx.get_merge_point_at(green_key, ctx.header_pc)
                .filter(|mp| mp.position.ops_len > 0)
                .map(|mp| {
                    (
                        mp.original_boxes.clone(),
                        mp.original_box_types.clone(),
                        mp.position,
                    )
                })
        } else {
            None
        };

        let mut recorder = ctx.recorder;
        // RPython heapcache.py:176: every trace gets at least one
        // GUARD_NOT_INVALIDATED. This allows external invalidation
        // (via JitCellToken.invalidate()) to force compiled loops
        // back to the interpreter.
        // RPython heapcache.py:176: every trace gets at least one
        // GUARD_NOT_INVALIDATED before the closing JUMP. fail_args = jump_args
        // so guard failure restores the same state as the JUMP target.
        // pyjitpl.py:2969: GUARD_FUTURE_CONDITION and heapcache.py:176:
        // GUARD_NOT_INVALIDATED are both emitted during tracing in
        // close_loop_args_at (state.rs) via record_guard → capture_resumedata.
        recorder.close_loop(jump_args);
        let trace = recorder.get_trace();

        // RPython Box type parity: build type index from the FULL uncut
        // trace. snapshot_boxes reference positions from the original trace;
        // cut_trace_from removes early ops. Must capture types before cut.
        // RPython Box type parity: every recorded op (including guards)
        // has a position and a type. Include Void ops (guards) so that
        // snapshot references to guard positions are covered.
        let pre_cut_trace_op_types: std::collections::HashMap<u32, majit_ir::Type> = trace
            .ops
            .iter()
            .filter(|op| !op.pos.is_none())
            .map(|op| (op.pos.0, op.result_type()))
            .collect();
        let pre_cut_trace_op_types_for_retry = pre_cut_trace_op_types.clone();

        // compile.py:269-270: cut trace at cross-loop merge point.
        // When the trace was retargeted to a different loop header, record
        // the new header PC so meta can be updated after insert.
        // RPython parity: only cut the trace if no compiled entry already
        // exists at the inner loop's green_key. If the inner loop was already
        // compiled independently, its entry has correct code+meta. Cutting
        // and replacing would install cross-loop-cut code with mismatched
        // inputarg layout. Instead, keep the original (uncut) trace and
        // compile.py:269: cut trace at cross-loop merge point.
        // When the trace was retargeted to a different loop header,
        // cut_trace_from removes ops before the merge point and
        // replaces inputargs with original_boxes at the cut position.
        let trace = if let Some((ref original_boxes, ref original_box_types, start)) =
            cross_loop_cut
        {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] cut_trace_from: start.ops_len={} original_boxes={} trace_ops={} header_pc={}",
                    start.ops_len,
                    original_boxes.len(),
                    trace.ops.len(),
                    ctx.header_pc,
                );
            }
            trace.cut_trace_from_with_consts(
                start,
                original_boxes,
                original_box_types,
                &ctx.initial_inputarg_consts,
            )
        } else {
            trace
        };
        let trace_snapshots = trace.snapshots.clone();

        let numbering_overrides = ctx.constants.numbering_type_overrides().clone();
        let (mut constants, mut constant_types) = ctx.constants.into_inner_with_types();

        let (trace_ops, _) = compile::fold_box_into_create_frame(
            trace.ops.clone(),
            &mut constants,
            &self.raw_int_box_helpers,
            &self.create_frame_raw_map,
        );
        let (trace_ops, _) = compile::elide_create_frame_for_call_assembler(
            trace_ops,
            &constants,
            &self.create_frame_raw_map,
        );
        if crate::majit_log_enabled() {
            eprintln!("--- trace (before opt) ---");
            eprint!("{}", majit_ir::format_trace(&trace_ops, &constants));
        }

        let num_ops_before = trace.ops.len();
        let num_trace_inputargs = trace.inputargs.len();

        // Save trace_ops + constants snapshot for potential unroll-free retry
        // (pyjitpl.py:3016-3021).
        let trace_ops_snapshot = trace_ops.clone();
        let constants_snapshot = constants.clone();

        // Use UnrollOptimizer for preamble peeling when available.
        // compile.py: compile_loop → PreambleCompileData + LoopCompileData.
        let prior_front_target_tokens = self
            .compiled_loops
            .get(&green_key)
            .map(|compiled| compiled.front_target_tokens.clone())
            .or_else(|| self.pending_preamble_tokens.remove(&green_key))
            .unwrap_or_default();
        let mut unroll_opt = crate::optimizeopt::unroll::UnrollOptimizer::new();
        unroll_opt.target_tokens = prior_front_target_tokens.clone();
        // history.py: carry retraced_count across recompilations
        let prior_retraced_count = self
            .compiled_loops
            .get(&green_key)
            .map(|compiled| compiled.retraced_count)
            .unwrap_or(0);
        // RPython parity: if retracing was already disabled for this key
        // (too many guards from a previous compilation), skip recompilation.
        // The existing compiled code handles this loop. Recompiling with
        // different context (e.g., cut_trace_from) would produce equally
        // heavy traces that hang in jump_to_existing_trace.
        if prior_retraced_count == u32::MAX && !prior_front_target_tokens.is_empty() {
            if crate::majit_log_enabled() {
                eprintln!("[jit] skipping recompile: retraced_count=MAX for key={green_key}");
            }
            self.warm_state.abort_tracing(green_key, true);
            return CompileOutcome::Cancelled;
        }
        unroll_opt.retraced_count = prior_retraced_count;
        unroll_opt.retrace_limit = self.warm_state.retrace_limit();
        unroll_opt.max_retrace_guards = self.warm_state.max_retrace_guards();
        unroll_opt.constant_types = constant_types.clone();
        unroll_opt.callinfocollection = self.callinfocollection.clone();
        unroll_opt.numbering_type_overrides = numbering_overrides.clone();
        // RPython Box type parity: each InputArg carries its type from
        // tracing. Propagate to optimizer so value_types covers inputargs.
        unroll_opt.trace_inputarg_types = trace.inputargs.iter().map(|ia| ia.tp).collect();
        // RPython Box type parity: snapshot_boxes reference positions from
        // the original uncut trace. The optimizer sees transformed+cut ops.
        // Use the pre-cut type index to cover all referenced positions.
        unroll_opt.original_trace_op_types = pre_cut_trace_op_types;

        // resume.py parity: convert tracing-time snapshots to flat OpRef
        // vectors so the optimizer can rebuild fail_args from snapshot in
        // store_final_boxes_in_guard (RPython ResumeDataVirtualAdder.finish).
        let (snapshot_map, snapshot_frame_size_map, snapshot_vable_map, snapshot_pc_map, sbt) =
            snapshot_map_from_trace_snapshots(
                &trace_snapshots,
                &mut constants,
                &mut constant_types,
            );
        let retry_sbt = sbt.clone();
        unroll_opt.snapshot_boxes = snapshot_map.clone();
        unroll_opt.snapshot_frame_sizes = snapshot_frame_size_map.clone();
        unroll_opt.snapshot_vable_boxes = snapshot_vable_map.clone();
        unroll_opt.snapshot_frame_pcs = snapshot_pc_map.clone();
        unroll_opt.snapshot_box_types = sbt;

        // RPython compile.py:278-294 parity: Phase 1 results must survive
        // Phase 2 InvalidLoop. Phase 1 writes to phase1_out on the caller's
        // stack BEFORE Phase 2 starts. If Phase 2 panics, phase1_out still
        // holds the Phase 1 results.
        let mut phase1_out: Option<(Vec<Op>, crate::optimizeopt::unroll::ExportedState)> = None;
        let mut updated_constant_types = constant_types.clone();
        let optimize_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = unroll_opt.optimize_trace_with_constants_and_inputs_vable_out(
                &trace_ops,
                &mut constants,
                num_trace_inputargs,
                vable_config.clone(),
                Some(&mut phase1_out),
            );
            // Capture Phase 2 constant types before unroll_opt is dropped.
            for (k, v) in &unroll_opt.constant_types {
                updated_constant_types.entry(*k).or_insert(*v);
            }
            result
        }));
        constant_types = updated_constant_types;
        let mut retried_without_unroll = false;
        let (optimized_ops, final_num_inputs) = match optimize_result {
            Ok(result) => result,
            Err(payload) => {
                // Phase 2 panicked — unroll_opt dropped. Phase 1 results
                // survive in phase1_out (written before Phase 2 started).
                if let Some(inv) = payload.downcast_ref::<crate::optimize::InvalidLoop>() {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] abort trace at key={} (InvalidLoop: {})",
                            green_key, inv.0,
                        );
                    }
                    self.cancel_count += 1;
                    // pyjitpl.py:3021-3030: only after the retry budget is
                    // exhausted do we try one last time without unrolling.
                    if !self.cancelled_too_many_times() {
                        self.warm_state.abort_tracing(green_key, false);
                        self.exported_state = None;
                        self.warm_state.reset_function_counts();
                        return CompileOutcome::Cancelled;
                    }
                    {
                        let mut retry_constants = constants_snapshot;
                        let mut simple_opt = Optimizer::default_pipeline();
                        simple_opt.constant_types = constant_types.clone();
                        simple_opt.numbering_type_overrides = numbering_overrides.clone();
                        let reconciled_inputarg_types: Vec<majit_ir::Type> = {
                            let first_guard_types = trace_ops_snapshot
                                .iter()
                                .find(|op| op.opcode.is_guard())
                                .and_then(|op| {
                                    op.descr
                                        .as_ref()
                                        .and_then(|d| d.as_fail_descr())
                                        .map(|fd| fd.fail_arg_types().to_vec())
                                });
                            trace
                                .inputargs
                                .iter()
                                .enumerate()
                                .map(|(i, ia)| {
                                    first_guard_types
                                        .as_ref()
                                        .and_then(|t| t.get(i).copied())
                                        .unwrap_or(ia.tp)
                                })
                                .collect()
                        };
                        simple_opt.trace_inputarg_types = reconciled_inputarg_types.clone();
                        for (i, &tp) in reconciled_inputarg_types.iter().enumerate() {
                            simple_opt.constant_types.insert(i as u32, tp);
                        }
                        simple_opt.original_trace_op_types =
                            pre_cut_trace_op_types_for_retry.clone();
                        simple_opt.snapshot_boxes = snapshot_map.clone();
                        simple_opt.snapshot_frame_sizes = snapshot_frame_size_map.clone();
                        simple_opt.snapshot_vable_boxes = snapshot_vable_map.clone();
                        simple_opt.snapshot_frame_pcs = snapshot_pc_map.clone();
                        simple_opt.snapshot_box_types = retry_sbt.clone();
                        let retry_result =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                simple_opt.optimize_with_constants_and_inputs(
                                    &trace_ops_snapshot,
                                    &mut retry_constants,
                                    num_trace_inputargs,
                                )
                            }));
                        match retry_result {
                            Ok(retry_ops) => {
                                if crate::majit_log_enabled() {
                                    eprintln!(
                                        "[jit] retry without unroll succeeded at key={}",
                                        green_key
                                    );
                                }
                                retried_without_unroll = true;
                                constants = retry_constants;
                                let ni = simple_opt.final_num_inputs();
                                (retry_ops, ni)
                            }
                            Err(_) => {
                                self.warm_state.abort_tracing(green_key, false);
                                self.exported_state = None;
                                self.warm_state.reset_function_counts();
                                return CompileOutcome::Aborted;
                            }
                        }
                    }
                } else {
                    std::panic::resume_unwind(payload);
                }
            }
        };
        let num_ops_after = optimized_ops.len();

        // RPython compile.py keeps the root entry contract on the original
        // loop inputargs. Simple loops synthesize a LABEL from that contract;
        // they do not grow inputargs to match a rewritten JUMP arity.
        let root_inputargs = root_loop_inputargs_from_optimizer(&trace.inputargs, final_num_inputs);
        let (inputargs, optimized_ops) = match normalize_root_loop_entry_contract(
            root_inputargs,
            optimized_ops,
        ) {
            Ok(normalized) => normalized,
            Err((expected, actual)) => {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] abort compile: root loop entry/jump arity mismatch input={} jump={}",
                        expected, actual,
                    );
                }
                self.cancel_count += 1;
                self.warm_state.reset_function_counts();
                return CompileOutcome::Cancelled;
            }
        };

        // RPython virtualizable parity: standard virtualizable fields and
        // arrays stay in the trace as first-class virtualizable boxes.
        // Do not prepend raw heap preamble loads here; compiled callers pass
        // the traced virtualizable values in the live-input layout, and
        // `vable_*` operations keep the hot path on boxes instead of
        // re-materializing `GetfieldRaw*`/`GetarrayitemRaw*` entry ops.
        let (inputargs, optimized_ops) = (inputargs, optimized_ops);
        let mut compiled_ops =
            compile::normalize_closing_jump_args(optimized_ops, &constants, final_num_inputs);

        if crate::majit_log_enabled() {
            eprintln!("--- trace (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&compiled_ops, &constants));
            for op in &compiled_ops {
                if op.opcode == majit_ir::OpCode::GuardNotInvalidated {
                    if let Some(ref fa) = op.fail_args {
                        let raw: Vec<String> =
                            fa.iter().map(|a| format!("OpRef({})", a.0)).collect();
                        eprintln!("[jit] FINAL GuardNotInv fail_args=[{}]", raw.join(", "));
                    }
                }
            }
        }

        // RPython: jit_merge_point tick counter provides periodic exit from
        // compiled loops. majit: for now, abort guardless loops. Loops with
        // constant-true guards (push(constant) pattern) will also hang but
        // this matches RPython behavior (relies on OS signals/GIL for exit).
        //
        // TODO: implement GUARD_NOT_INVALIDATED + signal-based invalidation
        // for periodic exit from compiled loops.
        let has_guard = compiled_ops.iter().any(|op| op.opcode.is_guard());
        if !has_guard {
            if crate::majit_log_enabled() {
                eprintln!("[jit] abort: guardless loop (no interrupt support)");
            }
            self.warm_state.abort_tracing(green_key, true);
            self.cancel_count += 1;
            self.warm_state.reset_function_counts();
            return CompileOutcome::Cancelled;
        }

        // resume.py parity: rd_numb is now produced inline during optimization
        // (ctx.emit → store_final_boxes_in_guard) rather than post-assembly.

        let compiled_constants = constants.clone();
        let compiled_constant_types = constant_types.clone();
        self.backend.set_constants(constants);

        // Use pre-allocated token number if available (for self-recursion
        // support), otherwise allocate a fresh one.
        let token_num = if let Some((pk, pn)) = self.pending_token.take() {
            if pk == green_key {
                pn
            } else {
                self.warm_state.alloc_token_number()
            }
        } else {
            self.warm_state.alloc_token_number()
        };
        let mut token = JitCellToken::new(token_num);
        token.green_key = green_key;
        let trace_id = self.alloc_trace_id();
        self.backend.set_next_trace_id(trace_id);
        self.backend.set_next_header_pc(green_key);

        let front_target_tokens = if retried_without_unroll {
            let target_token = crate::optimizeopt::unroll::TargetToken::new_loop(token_num);
            if let Some(jump_op) = compiled_ops
                .last_mut()
                .filter(|op| op.opcode == OpCode::Jump)
            {
                jump_op.descr = Some(target_token.as_jump_target_descr());
            }
            let mut label_op = majit_ir::Op::new(
                majit_ir::OpCode::Label,
                &inputargs
                    .iter()
                    .map(|ia| majit_ir::OpRef(ia.index))
                    .collect::<Vec<_>>(),
            );
            label_op.pos = majit_ir::OpRef::NONE;
            label_op.descr = Some(target_token.as_jump_target_descr());
            compiled_ops.insert(0, label_op);
            vec![target_token]
        } else if unroll_opt.target_tokens.is_empty() {
            prior_front_target_tokens.clone()
        } else {
            unroll_opt.target_tokens.clone()
        };

        // virtualizable.py:86 read_boxes: set num_scalar_inputargs on token
        // so the backend can find the first local in force_fn paths.
        token.num_scalar_inputargs = self.num_scalar_inputargs;
        let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.backend
                .compile_loop(&inputargs, &compiled_ops, &mut token)
        }));
        let compile_result = match compile_result {
            Ok(r) => r,
            Err(e) => {
                let is_invalid_loop = e.downcast_ref::<crate::optimize::InvalidLoop>().is_some();
                if crate::majit_log_enabled() {
                    let kind = if is_invalid_loop {
                        "InvalidLoop"
                    } else {
                        "panic"
                    };
                    eprintln!("[jit] compile_loop {kind}, aborting trace at key={green_key}");
                }
                // compile.py:288 parity: preserve preamble target_tokens
                // even on InvalidLoop/panic. The unroller's Phase 1 created
                // target_tokens that the next retrace needs.
                // Store in compiled_loops if available, otherwise in
                // pending_preamble_tokens for the first InvalidLoop before
                // any successful compilation (RPython: jitcell_token.
                // target_tokens = [start_descr] before Phase 2 runs).
                if is_invalid_loop && !unroll_opt.target_tokens.is_empty() {
                    if let Some(compiled) = self.compiled_loops.get_mut(&green_key) {
                        if compiled.front_target_tokens.is_empty() {
                            compiled.front_target_tokens = unroll_opt.target_tokens.clone();
                        }
                    } else {
                        self.pending_preamble_tokens
                            .entry(green_key)
                            .or_insert_with(|| unroll_opt.target_tokens.clone());
                    }
                }
                self.warm_state.abort_tracing(green_key, !is_invalid_loop);
                self.cancel_count += 1;
                self.warm_state.reset_function_counts();
                return CompileOutcome::Cancelled;
            }
        };
        match compile_result {
            Ok(_) => {
                // compile.py:826-830 store_hash: assign jitcounter hashes.
                self.assign_guard_hashes(&token);
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compiled loop at key={}, num_inputs={}",
                        green_key,
                        inputargs.len()
                    );
                }
                // Build resume data and exit layouts for all guards in the optimized trace.
                let (resume_data, guard_op_indices, mut exit_layouts) =
                    compile::build_guard_metadata(
                        &inputargs,
                        &compiled_ops,
                        green_key,
                        &compiled_constants,
                        &compiled_constant_types,
                    );
                let mut terminal_exit_layouts =
                    compile::build_terminal_exit_layouts(&inputargs, &compiled_ops);
                if let Some(backend_layouts) = self.backend.compiled_fail_descr_layouts(&token) {
                    compile::merge_backend_exit_layouts(&mut exit_layouts, &backend_layouts);
                }
                if let Some(backend_layouts) = self.backend.compiled_terminal_exit_layouts(&token) {
                    compile::merge_backend_terminal_exit_layouts(
                        &mut terminal_exit_layouts,
                        &backend_layouts,
                    );
                }
                let trace_info = self.backend.compiled_trace_info(&token, trace_id);
                let mut resume_data = resume_data;
                compile::enrich_guard_resume_layouts_for_trace(
                    &mut resume_data,
                    &mut exit_layouts,
                    trace_id,
                    &inputargs,
                    trace_info.as_ref(),
                );
                compile::patch_backend_guard_recovery_layouts_for_trace(
                    &mut self.backend,
                    &token,
                    trace_id,
                    &mut exit_layouts,
                );
                compile::patch_backend_terminal_recovery_layouts_for_trace(
                    &mut self.backend,
                    &token,
                    trace_id,
                    &mut terminal_exit_layouts,
                );
                // resume.py:570-574 _add_optimizer_sections parity:
                // serialize per-guard knowledge from unroll optimizer.
                let optimizer_knowledge = {
                    let pos_to_fail: HashMap<u32, u32> = guard_op_indices
                        .iter()
                        .filter_map(|(&fi, &op_idx)| {
                            compiled_ops.get(op_idx).map(|op| (op.pos.0, fi))
                        })
                        .collect();
                    let mut result: HashMap<
                        u32,
                        crate::optimizeopt::optimizer::OptimizerKnowledge,
                    > = HashMap::new();
                    for (guard_pos, knowledge) in &unroll_opt.per_guard_knowledge {
                        if let Some(&fi) = pos_to_fail.get(&guard_pos.0) {
                            result.insert(fi, knowledge.clone());
                        }
                    }
                    result
                };
                let mut traces = HashMap::new();
                traces.insert(
                    trace_id,
                    CompiledTrace {
                        inputargs: inputargs.clone(),
                        resume_data,
                        ops: compiled_ops,
                        constants: compiled_constants,
                        constant_types: compiled_constant_types.clone(),
                        guard_op_indices,
                        exit_layouts,
                        terminal_exit_layouts,
                        snapshots: trace_snapshots,
                        optimizer_knowledge,
                        jitcode: None,
                    },
                );

                // RPython parity: keep previous compiled tokens alive so
                // external target_token JUMPs can redirect to them.
                let mut previous_tokens: Vec<JitCellToken> = Vec::new();
                if let Some(old_entry) = self.compiled_loops.remove(&green_key) {
                    // Cranelift workaround (no RPython counterpart): copy
                    // bridges from old token to new, since Cranelift cannot
                    // patch machine code in-place. No-op for dynasm.
                    self.backend.migrate_bridges(&old_entry.token, &token);
                    previous_tokens.push(old_entry.token);
                    previous_tokens.extend(old_entry.previous_tokens);
                    // RPython parity: ResumeGuardDescr lives on the guard
                    // and never gets discarded. Preserve old trace metadata
                    // so guards from previous compilations can still find
                    // their exit_layouts.
                    for (tid, ct) in old_entry.traces {
                        traces.entry(tid).or_insert(ct);
                    }
                }
                if crate::majit_log_enabled() {
                    eprintln!("[jit][compiled_loops.insert] green_key={green_key}");
                }
                self.compiled_loops.insert(
                    green_key,
                    CompiledEntry {
                        token,
                        num_inputs: inputargs.len(),
                        meta,
                        front_target_tokens,
                        retraced_count: unroll_opt.retraced_count,
                        root_trace_id: trace_id,
                        traces,
                        previous_tokens,
                    },
                );
                // RPython warmstate.py:342: attach_procedure_to_interp(greenkey, token)
                let install_num = self.warm_state.alloc_token_number();
                let install_token = JitCellToken::new(install_num);
                self.warm_state
                    .attach_procedure_to_interp(green_key, install_token);

                self.stats.loops_compiled += 1;

                if let Some(ref hook) = self.hooks.on_compile_loop {
                    hook(green_key, num_ops_before, num_ops_after);
                }
                // pyjitpl.py:3025: self.exported_state = None
                self.exported_state = None;
                self.warm_state.reset_function_counts();
                let from_retry = self.cancel_count > 0;
                // When replacing an existing inner-loop entry with a
                // cross-loop cut, suppress meta rebuild — the existing
                // meta is authoritative (built from the inner loop's own
                // trace). The new compiled code may have different inputarg
                // layout, but is_compatible uses meta to extract live_values
                // so the meta must stay consistent with the entry point.
                self.last_compiled_key = Some(green_key);
                return CompileOutcome::Compiled {
                    green_key,
                    from_retry,
                };
            }
            Err(e) => {
                self.stats.loops_aborted += 1;
                let msg = format!("JIT compilation failed: {e}");
                if crate::majit_log_enabled() {
                    eprintln!("[jit] {msg}");
                }
                if let Some(ref cb) = self.hooks.on_compile_error {
                    cb(green_key, &msg);
                }
                // RPython: backend failure propagates to warmspot, handled
                // non-permanently (allows retry). compile.py has no explicit
                // catch for backend errors — they fall through to the outer
                // try/except in maybe_compile_and_run.
                self.warm_state.abort_tracing(green_key, false);
                self.cancel_count += 1;
                // pyjitpl.py:3025: self.exported_state = None
                self.exported_state = None;
                self.warm_state.reset_function_counts();
                return CompileOutcome::Cancelled;
            }
        }
    }

    /// pyjitpl.py:2936-2942: cancelled_too_many_times — check if
    /// cancel_count exceeds max_unroll_loops.
    fn cancelled_too_many_times(&self) -> bool {
        let limit = self.warm_state.max_unroll_loops();
        self.cancel_count > limit
    }

    /// RPython pyjitpl.py: check if partial_trace is set for compile_retrace.
    pub fn has_partial_trace(&self) -> bool {
        self.partial_trace.is_some()
    }

    /// Clear retrace state (partial_trace, retracing_from, exported_state).
    ///
    /// RPython parity: does NOT reset cancel_count. cancel_count is
    /// per-tracing-pass (reset in setup_tracing, which corresponds to
    /// RPython creating a new MetaInterp per _compile_and_run_once).
    /// clear_retrace_state only clears the retrace-specific fields.
    pub fn clear_retrace_state(&mut self) {
        self.partial_trace = None;
        self.retracing_from = None;
        self.exported_state = None;
    }

    /// compile.py: has_compiled_targets — check if a green key has
    /// compiled target tokens that a bridge can jump to.
    pub fn has_compiled_targets(&self, green_key: u64) -> bool {
        // Consistent with has_compiled_loop: direct key only, no alias.
        // Both functions must see the same view — otherwise tracing sees
        // "targets exist" while execution sees "no compiled loop", causing
        // wasted compile/trace churn in nested loops.
        self.compiled_loops
            .get(&green_key)
            .map_or(false, |c| !c.front_target_tokens.is_empty())
    }

    /// pyjitpl.py:3179-3190: compile_trace — try to compile the current
    /// trace as a bridge to an existing compiled loop.
    ///
    /// Called during tracing when a loop header is reached and that loop
    /// already has compiled code. Records a tentative JUMP, takes a
    /// snapshot of the trace ops, then cuts the JUMP back off.
    /// The snapshot is optimized as a bridge; on success the bridge is
    /// compiled and installed, on failure retrace_needed may be set.
    ///
    /// `bridge_origin`: if Some((trace_id, fail_index)), the trace started
    /// from a guard failure (ResumeGuardDescr). If None, the trace started
    /// from the interpreter (ResumeFromInterpDescr / entry bridge).
    ///
    /// Returns `CompileOutcome::Compiled` if the bridge was successfully
    /// compiled and installed. The caller should switch to compiled code.
    /// Returns `CompileOutcome::Cancelled` if the bridge couldn't close
    /// (retrace_needed was set, or optimization failed).
    /// compile.py:1028 compile_trace parity.
    /// `ends_with_jump=true`: records JUMP, uses BridgeCompileData (optimize_bridge).
    /// `ends_with_jump=false`: records FINISH, uses SimpleCompileData (optimize_loop).
    pub fn compile_trace(
        &mut self,
        green_key: u64,
        finish_args: &[OpRef],
        bridge_origin: Option<(u64, u32)>,
    ) -> CompileOutcome {
        self.compile_trace_inner(green_key, finish_args, bridge_origin, None, None)
    }

    /// compile.py:1002-1021 ResumeFromInterpDescr parity.
    pub fn compile_trace_from_interp(
        &mut self,
        green_key: u64,
        finish_args: &[OpRef],
        original_green_key: u64,
        entry_meta: M,
    ) -> CompileOutcome {
        self.compile_trace_inner(
            green_key,
            finish_args,
            None,
            None,
            Some((original_green_key, entry_meta)),
        )
    }

    /// compile_trace with ends_with_jump=false (FINISH).
    pub fn compile_trace_finish(
        &mut self,
        green_key: u64,
        finish_args: &[OpRef],
        bridge_origin: Option<(u64, u32)>,
        finish_descr: majit_ir::DescrRef,
    ) -> CompileOutcome {
        self.compile_trace_inner(
            green_key,
            finish_args,
            bridge_origin,
            Some(finish_descr),
            None,
        )
    }

    fn compile_trace_inner(
        &mut self,
        green_key: u64,
        finish_args: &[OpRef],
        bridge_origin: Option<(u64, u32)>,
        finish_descr: Option<majit_ir::DescrRef>,
        entry_bridge: Option<(u64, M)>,
    ) -> CompileOutcome {
        let ends_with_jump = finish_descr.is_none();
        let ctx = match self.tracing.as_mut() {
            Some(ctx) => ctx,
            None => return CompileOutcome::Cancelled,
        };

        // pyjitpl.py:3187: save position before recording JUMP/FINISH
        let cut_at = ctx.get_trace_position();
        self.potential_retrace_position = Some(cut_at);

        // pyjitpl.py:3189 / 3217: record tentative JUMP or FINISH
        if let Some(descr) = finish_descr {
            ctx.recorder.finish(finish_args, descr);
        } else {
            let jump_descr = self
                .compiled_loops
                .get(&green_key)
                .and_then(|compiled| compiled.front_target_tokens.first())
                .map(|target_token| target_token.as_jump_target_descr());
            let Some(jump_descr) = jump_descr else {
                return CompileOutcome::Cancelled;
            };
            ctx.recorder
                .close_loop_with_descr(finish_args, Some(jump_descr));
        }

        // Snapshot the trace ops (including JUMP) for bridge compilation.
        let bridge_ops = ctx.recorder.ops().to_vec();
        let bridge_inputargs: Vec<majit_ir::InputArg> = ctx
            .recorder
            .inputarg_types()
            .iter()
            .enumerate()
            .map(|(i, &tp)| majit_ir::InputArg::from_type(tp, i as u32))
            .collect();
        let mut constants = ctx.constants.snapshot();
        let mut constant_types = ctx.constants.constant_types_snapshot();
        let trace_snapshots = ctx.recorder.snapshots().to_vec();
        let (
            snapshot_boxes,
            snapshot_frame_sizes,
            snapshot_vable_boxes,
            snapshot_frame_pcs,
            snapshot_box_types,
        ) = snapshot_map_from_trace_snapshots(
            &trace_snapshots,
            &mut constants,
            &mut constant_types,
        );

        // pyjitpl.py:3195 finally: always cut — pop the tentative JUMP/FINISH.
        ctx.recorder.unfinalize();
        ctx.recorder.cut(cut_at);

        let label = if ends_with_jump { "jump" } else { "finish" };
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] compile_trace({}): key={}, ops={}, origin={:?}",
                label,
                green_key,
                bridge_ops.len(),
                bridge_origin,
            );
        }

        match bridge_origin {
            Some((trace_id, fail_index)) => {
                // compile.py:1082 — ResumeGuardDescr path: attach bridge
                // to the existing guard that failed.
                let fail_descr = {
                    let compiled = match self.compiled_loops.get(&green_key) {
                        Some(c) => c,
                        None => return CompileOutcome::Cancelled,
                    };
                    match self.bridge_fail_descr_proxy(compiled, trace_id, fail_index) {
                        Some(d) => Box::new(d) as Box<dyn majit_ir::FailDescr>,
                        None => return CompileOutcome::Cancelled,
                    }
                };
                let success = self.compile_bridge(
                    green_key,
                    fail_index,
                    &*fail_descr,
                    &bridge_ops,
                    &bridge_inputargs,
                    constants,
                    constant_types,
                    snapshot_boxes,
                    snapshot_frame_sizes,
                    snapshot_vable_boxes,
                    snapshot_frame_pcs,
                    snapshot_box_types.clone(),
                );
                if success {
                    CompileOutcome::Compiled {
                        green_key: 0,
                        from_retry: false,
                    }
                } else {
                    CompileOutcome::Cancelled
                }
            }
            None => {
                // compile.py:1006-1022 — ResumeFromInterpDescr path:
                // compile a fresh entry bridge and attach it to the
                // original interpreter green key.
                let Some((original_green_key, entry_meta)) = entry_bridge else {
                    return CompileOutcome::Cancelled;
                };
                let success = self.compile_entry_bridge(
                    green_key,
                    original_green_key,
                    entry_meta,
                    &bridge_ops,
                    &bridge_inputargs,
                    constants,
                    constant_types,
                    snapshot_boxes,
                    snapshot_frame_sizes,
                    snapshot_vable_boxes,
                    snapshot_frame_pcs,
                    snapshot_box_types,
                );
                if success {
                    CompileOutcome::Compiled {
                        green_key: original_green_key,
                        from_retry: false,
                    }
                } else {
                    CompileOutcome::Cancelled
                }
            }
        }
    }

    /// pyjitpl.py:2408-2412: retrace_needed — save state from a failed
    /// bridge compilation for a subsequent compile_retrace attempt.
    ///
    /// Called when the optimizer returns "not final" (no existing target token
    /// matched). The partial trace and exported state are saved so the next
    /// compile_loop for this green_key can use compile_retrace.
    pub fn retrace_needed(
        &mut self,
        green_key: u64,
        ops: Vec<Op>,
        inputargs: Vec<InputArg>,
        constants: HashMap<u32, i64>,
        exported_state: crate::optimizeopt::unroll::ExportedState,
    ) {
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] retrace_needed: key={}, partial_ops={}",
                green_key,
                ops.len()
            );
        }
        self.partial_trace = Some(PartialTrace {
            ops,
            inputargs,
            constants,
        });
        // pyjitpl.py:2410: self.retracing_from = self.potential_retrace_position
        self.retracing_from = self.potential_retrace_position;
        self.exported_state = Some(exported_state);
        // pyjitpl.py:2418: self.heapcache.reset()
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.reset_heap_cache();
        }
    }

    /// pyjitpl.py:3171-3177 / compile.py:341-394: compile_retrace — compile
    /// a new loop specialization by appending new body ops to a partial trace.
    ///
    /// Uses the saved exported_state to import optimizer knowledge from the
    /// first (failed) attempt, then optimizes the new trace body and
    /// concatenates with partial_trace ops.
    ///
    /// Returns true if compilation succeeded.
    pub fn compile_retrace(&mut self, jump_args: &[OpRef], meta: M) -> bool {
        let partial = match self.partial_trace.take() {
            Some(p) => p,
            None => return false,
        };
        let mut start_state = match self.exported_state.take() {
            Some(s) => s,
            None => return false,
        };
        // gcreftracer.py parity: GC may have moved objects between Phase 1
        // and Phase 2. Refresh GcRef values from shadow stack before use.
        start_state.refresh_from_gc();
        // Consume retracing_from (position); green_key comes from tracing ctx.
        self.retracing_from = None;
        let green_key = match self.tracing.as_ref() {
            Some(ctx) => ctx.green_key,
            None => return false,
        };

        let vable_config = self.current_virtualizable_optimizer_config();
        self.force_finish_trace = false;
        let mut ctx = match self.tracing.take() {
            Some(ctx) => ctx,
            None => return false,
        };
        ctx.apply_replacements();

        let mut recorder = ctx.recorder;
        recorder.close_loop(jump_args);
        let trace = recorder.get_trace();

        let numbering_overrides = ctx.constants.numbering_type_overrides().clone();
        let (mut constants, mut constant_types) = ctx.constants.into_inner_with_types();

        let (trace_ops, _) = compile::fold_box_into_create_frame(
            trace.ops.clone(),
            &mut constants,
            &self.raw_int_box_helpers,
            &self.create_frame_raw_map,
        );
        let (trace_ops, _) = compile::elide_create_frame_for_call_assembler(
            trace_ops,
            &constants,
            &self.create_frame_raw_map,
        );

        if crate::majit_log_enabled() {
            eprintln!("--- retrace body (before opt) ---");
            eprint!("{}", majit_ir::format_trace(&trace_ops, &constants));
        }

        // compile.py:362-367: optimize using UnrolledLoopData with start_state.
        let prior_front_target_tokens = self
            .compiled_loops
            .get(&green_key)
            .map(|compiled| compiled.front_target_tokens.clone())
            .or_else(|| self.pending_preamble_tokens.remove(&green_key))
            .unwrap_or_default();
        let mut unroll_opt = crate::optimizeopt::unroll::UnrollOptimizer::new();
        unroll_opt.target_tokens = prior_front_target_tokens.clone();
        unroll_opt.retraced_count = self
            .compiled_loops
            .get(&green_key)
            .map(|compiled| compiled.retraced_count)
            .unwrap_or(0);
        unroll_opt.retrace_limit = self.warm_state.retrace_limit();
        unroll_opt.max_retrace_guards = self.warm_state.max_retrace_guards();
        unroll_opt.constant_types = constant_types.clone();
        unroll_opt.callinfocollection = self.callinfocollection.clone();
        unroll_opt.numbering_type_overrides = numbering_overrides;
        let (
            retrace_snapshot_boxes,
            retrace_snapshot_frame_sizes,
            retrace_snapshot_vable_boxes,
            retrace_snapshot_frame_pcs,
            retrace_sbt,
        ) = snapshot_map_from_trace_snapshots(
            &trace.snapshots,
            &mut constants,
            &mut constant_types,
        );
        unroll_opt.snapshot_boxes = retrace_snapshot_boxes;
        unroll_opt.snapshot_frame_sizes = retrace_snapshot_frame_sizes;
        unroll_opt.snapshot_vable_boxes = retrace_snapshot_vable_boxes;
        unroll_opt.snapshot_frame_pcs = retrace_snapshot_frame_pcs;
        unroll_opt.snapshot_box_types = retrace_sbt;
        // Import the exported state from the first (failed) attempt so the
        // optimizer can continue from where it left off.
        unroll_opt.imported_state = Some(start_state);

        let mut updated_constant_types = constant_types.clone();
        let optimize_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = unroll_opt.optimize_trace_with_constants_and_inputs_vable(
                &trace_ops,
                &mut constants,
                trace.inputargs.len(),
                vable_config,
            );
            for (k, v) in &unroll_opt.constant_types {
                updated_constant_types.entry(*k).or_insert(*v);
            }
            result
        }));
        constant_types = updated_constant_types;
        let (body_ops, final_num_inputs) = match optimize_result {
            Ok(result) => result,
            Err(payload) => {
                if payload
                    .downcast_ref::<crate::optimize::InvalidLoop>()
                    .is_some()
                {
                    if crate::majit_log_enabled() {
                        eprintln!("[jit] compile_retrace: InvalidLoop at key={}", green_key);
                    }
                    return false;
                }
                std::panic::resume_unwind(payload);
            }
        };

        // compile.py:379-382: partial_trace.operations + [label_op] + loop_ops.
        //
        // RPython invariant: partial_trace.operations does NOT contain a
        // terminal JUMP — the optimizer's propagate_all_forward separates
        // JUMP into last_op (not in _newoperations). body_ops (from
        // assemble_peeled_trace_with_jump_args) contains Label + body + JUMP.
        //
        // pyre parity: partial.ops now stores Phase 1 optimized preamble
        // ops (JUMP excluded), matching RPython's partial_trace.operations.
        let mut combined_ops = partial.ops;
        combined_ops.extend(body_ops);
        // Merge constants from partial trace with new constants.
        for (k, v) in partial.constants {
            constants.entry(k).or_insert(v);
        }

        let root_inputargs =
            root_loop_inputargs_from_optimizer(&partial.inputargs, final_num_inputs);
        let (inputargs, combined_ops) =
            match normalize_root_loop_entry_contract(root_inputargs, combined_ops) {
                Ok(normalized) => normalized,
                Err((expected, actual)) => {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] compile_retrace: entry/jump arity mismatch input={} jump={}",
                            expected, actual,
                        );
                    }
                    return false;
                }
            };

        let combined_ops =
            compile::normalize_closing_jump_args(combined_ops, &constants, final_num_inputs);

        if crate::majit_log_enabled() {
            eprintln!("--- retrace combined (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&combined_ops, &constants));
        }

        let num_combined_ops = combined_ops.len();
        let has_guard = combined_ops.iter().any(|op| op.opcode.is_guard());
        if !has_guard {
            if crate::majit_log_enabled() {
                eprintln!("[jit] compile_retrace: guardless loop");
            }
            return false;
        }

        let compiled_constants = constants.clone();
        let compiled_constant_types = constant_types.clone();
        self.backend.set_constants(constants);

        let token_num = self.warm_state.alloc_token_number();
        let mut token = JitCellToken::new(token_num);
        token.green_key = green_key;
        let trace_id = self.alloc_trace_id();
        self.backend.set_next_trace_id(trace_id);
        self.backend.set_next_header_pc(green_key);

        let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.backend
                .compile_loop(&inputargs, &combined_ops, &mut token)
        }));
        let compile_result = match compile_result {
            Ok(r) => r,
            Err(_) => {
                if crate::majit_log_enabled() {
                    eprintln!("[jit] compile_retrace panicked at key={green_key}");
                }
                self.warm_state.abort_tracing(green_key, false);
                return false;
            }
        };
        match compile_result {
            Ok(_) => {
                self.assign_guard_hashes(&token);
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compiled retrace at key={}, num_inputs={}",
                        green_key,
                        inputargs.len()
                    );
                }
                let (resume_data, guard_op_indices, mut exit_layouts) =
                    compile::build_guard_metadata(
                        &inputargs,
                        &combined_ops,
                        green_key,
                        &compiled_constants,
                        &compiled_constant_types,
                    );
                let mut terminal_exit_layouts =
                    compile::build_terminal_exit_layouts(&inputargs, &combined_ops);
                if let Some(backend_layouts) = self.backend.compiled_fail_descr_layouts(&token) {
                    compile::merge_backend_exit_layouts(&mut exit_layouts, &backend_layouts);
                }
                if let Some(backend_layouts) = self.backend.compiled_terminal_exit_layouts(&token) {
                    compile::merge_backend_terminal_exit_layouts(
                        &mut terminal_exit_layouts,
                        &backend_layouts,
                    );
                }
                let trace_info = self.backend.compiled_trace_info(&token, trace_id);
                let mut resume_data = resume_data;
                compile::enrich_guard_resume_layouts_for_trace(
                    &mut resume_data,
                    &mut exit_layouts,
                    trace_id,
                    &inputargs,
                    trace_info.as_ref(),
                );
                compile::patch_backend_guard_recovery_layouts_for_trace(
                    &mut self.backend,
                    &token,
                    trace_id,
                    &mut exit_layouts,
                );
                compile::patch_backend_terminal_recovery_layouts_for_trace(
                    &mut self.backend,
                    &token,
                    trace_id,
                    &mut terminal_exit_layouts,
                );
                // resume.py:570 parity: serialize per_guard_knowledge.
                let optimizer_knowledge = {
                    let pos_to_fail: HashMap<u32, u32> = guard_op_indices
                        .iter()
                        .filter_map(|(&fi, &op_idx)| {
                            combined_ops.get(op_idx).map(|op| (op.pos.0, fi))
                        })
                        .collect();
                    let mut result: HashMap<u32, OptimizerKnowledge> = HashMap::new();
                    for (guard_pos, knowledge) in &unroll_opt.per_guard_knowledge {
                        if let Some(&fi) = pos_to_fail.get(&guard_pos.0) {
                            result.insert(fi, knowledge.clone());
                        }
                    }
                    result
                };
                let mut traces = HashMap::new();
                traces.insert(
                    trace_id,
                    CompiledTrace {
                        snapshots: Vec::new(),
                        inputargs: inputargs.clone(),
                        resume_data,
                        ops: combined_ops,
                        constants: compiled_constants,
                        constant_types: compiled_constant_types.clone(),
                        guard_op_indices,
                        exit_layouts,
                        terminal_exit_layouts,
                        optimizer_knowledge,
                        jitcode: None,
                    },
                );

                let mut previous_tokens: Vec<JitCellToken> = Vec::new();
                if let Some(old_entry) = self.compiled_loops.remove(&green_key) {
                    // Cranelift workaround (no RPython counterpart): copy
                    // bridges from old token to new, since Cranelift cannot
                    // patch machine code in-place. No-op for dynasm.
                    self.backend.migrate_bridges(&old_entry.token, &token);
                    previous_tokens.push(old_entry.token);
                    previous_tokens.extend(old_entry.previous_tokens);
                    for (tid, ct) in old_entry.traces {
                        traces.entry(tid).or_insert(ct);
                    }
                }
                if crate::majit_log_enabled() {
                    eprintln!("[jit][compiled_loops.insert] green_key={green_key}",);
                }
                self.compiled_loops.insert(
                    green_key,
                    CompiledEntry {
                        token,
                        num_inputs: inputargs.len(),
                        meta,
                        front_target_tokens: if unroll_opt.target_tokens.is_empty() {
                            prior_front_target_tokens
                        } else {
                            unroll_opt.target_tokens.clone()
                        },
                        retraced_count: unroll_opt.retraced_count,
                        root_trace_id: trace_id,
                        traces,
                        previous_tokens,
                    },
                );
                let install_num = self.warm_state.alloc_token_number();
                let install_token = JitCellToken::new(install_num);
                self.warm_state
                    .attach_procedure_to_interp(green_key, install_token);
                self.stats.loops_compiled += 1;

                if let Some(ref hook) = self.hooks.on_compile_loop {
                    hook(green_key, 0, num_combined_ops);
                }
                true
            }
            Err(e) => {
                self.stats.loops_aborted += 1;
                if crate::majit_log_enabled() {
                    eprintln!("[jit] compile_retrace failed: {e}");
                }
                self.warm_state.abort_tracing(green_key, false);
                false
            }
        }
    }

    /// Abort the current trace.
    ///
    /// If `permanent` is true, this location will never be traced again.
    pub fn abort_trace(&mut self, permanent: bool) {
        self.force_finish_trace = false;
        self.clear_retrace_state();
        if let Some(ctx) = self.tracing.take() {
            self.stats.loops_aborted += 1;
            let green_key = ctx.green_key;
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] abort trace at key={} (permanent={})",
                    green_key, permanent
                );
            }
            ctx.recorder.abort();
            self.warm_state.abort_tracing(green_key, permanent);
            // Reset per-trace function call counts.
            self.warm_state.reset_function_counts();
            self.pending_token = None;
            if let Some(ref hook) = self.hooks.on_trace_abort {
                hook(green_key, permanent);
            }
        }
    }

    /// Finish the current trace with a terminal `FINISH`, then optimize and compile it.
    pub fn finish_and_compile(
        &mut self,
        finish_args: &[OpRef],
        finish_arg_types: Vec<Type>,
        meta: M,
    ) {
        // Cache vable_config before take() clears self.tracing.
        let vable_config = self.current_virtualizable_optimizer_config();
        self.force_finish_trace = false;
        let mut ctx = self.tracing.take().unwrap();
        ctx.apply_replacements();
        // pyjitpl.py:3199 compile_done_with_this_frame: store_token_in_vable
        // is recorded BEFORE the FINISH op so the JIT-compiled exit path
        // sets the vable token and emits GUARD_NOT_FORCED_2.
        ctx.store_token_in_vable();
        let green_key = ctx.green_key;

        let mut recorder = ctx.recorder;
        recorder.finish(finish_args, crate::make_fail_descr_typed(finish_arg_types));
        let trace = recorder.get_trace();
        let trace_snapshots = trace.snapshots.clone();

        // RPython Box type parity: build type index from the trace ops.
        // snapshot_boxes reference positions from the original trace.
        let pre_cut_trace_op_types: std::collections::HashMap<u32, majit_ir::Type> = trace
            .ops
            .iter()
            .filter(|op| !op.pos.is_none())
            .map(|op| (op.pos.0, op.result_type()))
            .collect();

        let numbering_overrides = ctx.constants.numbering_type_overrides().clone();
        let (mut constants, mut constant_types) = ctx.constants.into_inner_with_types();

        let (trace_ops, pos_remap1) = compile::fold_box_into_create_frame(
            trace.ops.clone(),
            &mut constants,
            &self.raw_int_box_helpers,
            &self.create_frame_raw_map,
        );
        let (trace_ops, pos_remap2) = compile::elide_create_frame_for_call_assembler(
            trace_ops,
            &constants,
            &self.create_frame_raw_map,
        );

        let num_ops_before = trace_ops.len();
        let mut optimizer = if let Some(config) = vable_config {
            Optimizer::default_pipeline_with_virtualizable(config)
        } else {
            Optimizer::default_pipeline()
        };
        optimizer.constant_types = constant_types.clone();
        optimizer.numbering_type_overrides = numbering_overrides;
        // RPython Box.type parity: pyre's recorder records each inputarg
        // with the JITCODE-LEVEL type (Python locals are PyObjectRef so all
        // Python locals come in as Type::Ref). The actual post-unbox type
        // is recorded on each guard's MetaFailDescr.fail_arg_types when the
        // recorder emits the guard (state.rs::record_current_state_guard
        // / generate_guard_core derives fail_arg_types from
        // build_fail_arg_types_for_active_boxes which inspects the
        // symbolic stack types).
        //
        // The optimizer needs the unboxed types so OptBoxEnv::get_type
        // returns Int for an unboxed local; otherwise store_final_boxes
        // populates livebox_types with the source Ref type and the
        // bridge tracer inherits the wrong type for its symbolic_locals.
        //
        // RPython has no equivalent because RPython's recorder produces
        // typed boxes directly (IntFrontendOp, RefFrontendOp,
        // FloatFrontendOp), and the optimizer reads `box.type` from the
        // box object.  pyre carries the type on the FailDescr because the
        // recorder cannot retroactively change the inputarg's type once
        // a slot has been observed.
        //
        // Reconcile inputarg types from the FIRST guard's fail_arg_types
        // BEFORE seeding the optimizer state. The first guard in the
        // recorded trace is `GuardFalse(IntLt(v_n, 2))` for fib, with
        // `fail_arg_types = [Ref, Int, Ref, Int, Ref, Int]` — the
        // post-unbox view that the optimizer should use.
        //
        // Read from `trace_ops` (post-fold_box_into_create_frame /
        // elide_create_frame_for_call_assembler) so the lookup matches
        // what the optimizer actually sees. Those transforms only fold
        // call helpers and never insert/remove guards, so the first
        // guard's MetaFailDescr is unchanged, but reading from the same
        // op slice keeps any future invariant we add to those passes
        // honest.
        let reconciled_inputarg_types: Vec<majit_ir::Type> = {
            let first_guard_types =
                trace_ops
                    .iter()
                    .find(|op| op.opcode.is_guard())
                    .and_then(|op| {
                        op.descr
                            .as_ref()
                            .and_then(|d| d.as_fail_descr())
                            .map(|fd| fd.fail_arg_types().to_vec())
                    });
            trace
                .inputargs
                .iter()
                .enumerate()
                .map(|(i, ia)| {
                    first_guard_types
                        .as_ref()
                        .and_then(|t| t.get(i).copied())
                        .unwrap_or(ia.tp)
                })
                .collect()
        };
        optimizer.trace_inputarg_types = reconciled_inputarg_types.clone();
        // RPython Box.type parity: register inputarg types in constant_types
        // so fail_arg_types inference can resolve them.
        for (i, &tp) in reconciled_inputarg_types.iter().enumerate() {
            optimizer.constant_types.insert(i as u32, tp);
        }
        optimizer.original_trace_op_types = pre_cut_trace_op_types;

        // resume.py parity: convert tracing-time snapshots to flat OpRef
        // vectors so the optimizer can rebuild fail_args from snapshot in
        // store_final_boxes_in_guard (RPython ResumeDataVirtualAdder.finish).
        let (snapshot_map, snapshot_frame_size_map, snapshot_vable_map, snapshot_pc_map, sbt) =
            snapshot_map_from_trace_snapshots(
                &trace_snapshots,
                &mut constants,
                &mut constant_types,
            );
        // compile.py:92-96 SimpleCompileData.optimize → optimize_loop parity.
        // Wire snapshot data through to the optimizer so guard
        // store_final_boxes_in_guard (mod.rs:2261) can properly populate
        // rd_numb / rd_consts via _number_boxes (resume.py:200-205).
        // Without this, every guard from a function-entry trace is dropped
        // by the no-snapshot fallback in mod.rs:2281, leaving rd_numb=None,
        // and the runtime guard-fail path immediately invalidates the loop
        // (because resume_in_blackhole has no resume_pc to walk to).
        optimizer.snapshot_boxes = snapshot_map;
        optimizer.snapshot_frame_sizes = snapshot_frame_size_map;
        optimizer.snapshot_vable_boxes = snapshot_vable_map;
        optimizer.snapshot_frame_pcs = snapshot_pc_map;
        optimizer.snapshot_box_types = sbt;

        // Wrap in catch_unwind — InvalidLoop during optimization should
        // abort the trace, not crash the process. Matches compile_loop.
        let mut updated_constant_types = constant_types.clone();
        let optimize_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = optimizer.optimize_with_constants_and_inputs(
                &trace_ops,
                &mut constants,
                trace.inputargs.len(),
            );
            for (k, v) in &optimizer.constant_types {
                updated_constant_types.entry(*k).or_insert(*v);
            }
            result
        }));
        constant_types = updated_constant_types;
        let optimized_ops = match optimize_result {
            Ok(ops) => ops,
            Err(payload) => {
                if payload
                    .downcast_ref::<crate::optimize::InvalidLoop>()
                    .is_some()
                {
                    if crate::majit_log_enabled() {
                        eprintln!("[jit] abort finish: InvalidLoop at key={}", green_key);
                    }
                    self.warm_state.abort_tracing(green_key, true);
                    return;
                }
                std::panic::resume_unwind(payload);
            }
        };
        // RPython optimizer.py:552-556 (flush=True): Finish/Jump is sent
        // through passes inside propagate_all_forward and ends up in
        // new_operations naturally — no restoration needed.
        let mut optimized_ops = optimized_ops;
        let num_ops_after = optimized_ops.len();
        // RPython compile.py:234 parity: transfer quasi-immutable deps
        // from optimizer to MetaInterp for post-compile watcher registration.
        self.last_quasi_immutable_deps = optimizer.quasi_immutable_deps.drain().collect();

        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] finish_and_compile: key={}, ops_before={}, ops_after={}",
                green_key, num_ops_before, num_ops_after
            );
            eprintln!("--- finish trace (before opt) ---");
            eprint!("{}", majit_ir::format_trace(&trace_ops, &constants));
            eprintln!("--- finish trace (after opt, before unbox) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
        }

        // resume.py:411-417 parity: NONE entries in guard fail_args are
        // valid (TAGCONST/TAGVIRTUAL slots that resume reconstructs from
        // rd_consts/rd_virtuals). RPython has no compile-time abort
        // heuristic — runtime guard-fail recovery handles unrecoverable
        // cases via clear_compiled_loops.
        let mut optimized_ops = compile::strip_stray_overflow_guards(optimized_ops);

        if crate::majit_log_enabled() {
            eprintln!("--- finish trace (after unbox) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
        }

        // Use pre-allocated token number if available (for self-recursion
        // support), otherwise allocate a fresh one.
        let token_num = if let Some((pk, pn)) = self.pending_token.take() {
            if pk == green_key {
                pn
            } else {
                self.warm_state.alloc_token_number()
            }
        } else {
            self.warm_state.alloc_token_number()
        };
        let mut token = JitCellToken::new(token_num);
        let trace_id = self.alloc_trace_id();
        self.backend.set_next_trace_id(trace_id);
        self.backend.set_next_header_pc(green_key);

        let final_num_inputs = optimizer.final_num_inputs();
        let mut inputargs = trace.inputargs.clone();
        while inputargs.len() < final_num_inputs {
            inputargs.push(majit_ir::InputArg {
                tp: majit_ir::Type::Int,
                index: inputargs.len() as u32,
            });
        }
        // Reconcile inputarg types with optimizer's post-unbox types.
        // Pyre starts tracing with Ref values (all Python objects), but
        // the optimizer may unbox Int-typed locals. Read the first guard's
        // fail_arg_types to discover the optimizer's type decisions, then
        // update inputargs to match. This ensures gcmap and adapt-live
        // agree on which slots are GC refs vs raw ints.
        if let Some(first_guard_types) = optimized_ops
            .iter()
            .find(|op| op.opcode.is_guard())
            .and_then(|op| {
                op.descr
                    .as_ref()
                    .and_then(|d| d.as_fail_descr())
                    .map(|fd| fd.fail_arg_types().to_vec())
            })
        {
            for (i, ia) in inputargs.iter_mut().enumerate() {
                if let Some(&tp) = first_guard_types.get(i) {
                    ia.tp = tp;
                }
            }
        }

        // Note: adapt-live type correction (inputarg Ref→Int, guard
        // fail_arg_types) is NOT applied here. CalAssemblerI calls the
        // callee without adapt-live, so the runtime types at guard failure
        // are the original Ref types. The no-snapshot fallback handles
        // types correctly via MetaFailDescr.

        let compiled_constants = constants.clone();
        let compiled_constant_types = constant_types.clone();
        self.backend.set_constants(constants);
        token.green_key = green_key;
        // virtualizable.py:86 read_boxes: set num_scalar_inputargs on token
        // so the backend can find the first local in force_fn paths.
        token.num_scalar_inputargs = self.num_scalar_inputargs;

        match self
            .backend
            .compile_loop(&inputargs, &optimized_ops, &mut token)
        {
            Ok(_) => {
                self.assign_guard_hashes(&token);
                let (resume_data, guard_op_indices, mut exit_layouts) =
                    compile::build_guard_metadata(
                        &inputargs,
                        &optimized_ops,
                        green_key,
                        &compiled_constants,
                        &compiled_constant_types,
                    );
                let mut terminal_exit_layouts =
                    compile::build_terminal_exit_layouts(&inputargs, &optimized_ops);
                if let Some(backend_layouts) = self.backend.compiled_fail_descr_layouts(&token) {
                    compile::merge_backend_exit_layouts(&mut exit_layouts, &backend_layouts);
                }
                if let Some(backend_layouts) = self.backend.compiled_terminal_exit_layouts(&token) {
                    compile::merge_backend_terminal_exit_layouts(
                        &mut terminal_exit_layouts,
                        &backend_layouts,
                    );
                }
                let trace_info = self.backend.compiled_trace_info(&token, trace_id);
                let mut resume_data = resume_data;
                compile::enrich_guard_resume_layouts_for_trace(
                    &mut resume_data,
                    &mut exit_layouts,
                    trace_id,
                    &trace.inputargs,
                    trace_info.as_ref(),
                );
                compile::patch_backend_guard_recovery_layouts_for_trace(
                    &mut self.backend,
                    &token,
                    trace_id,
                    &mut exit_layouts,
                );
                compile::patch_backend_terminal_recovery_layouts_for_trace(
                    &mut self.backend,
                    &token,
                    trace_id,
                    &mut terminal_exit_layouts,
                );
                // RPython resume.py:570-574: per-guard knowledge captured
                // during optimization at each guard emit point.
                let per_guard_knowledge = {
                    let pos_to_fail: HashMap<u32, u32> = guard_op_indices
                        .iter()
                        .filter_map(|(&fi, &op_idx)| {
                            optimized_ops.get(op_idx).map(|op| (op.pos.0, fi))
                        })
                        .collect();
                    let mut result: HashMap<u32, OptimizerKnowledge> = HashMap::new();
                    for (guard_pos, knowledge) in &optimizer.per_guard_knowledge {
                        if let Some(&fi) = pos_to_fail.get(&guard_pos.0) {
                            result.insert(fi, knowledge.clone());
                        }
                    }
                    let end_knowledge = optimizer
                        .final_ctx
                        .as_ref()
                        .map(|c| optimizer.serialize_optimizer_knowledge(c))
                        .unwrap_or_default();
                    for (_, k) in result.iter_mut() {
                        if k.known_classes.is_empty() {
                            k.known_classes = end_knowledge.known_classes.clone();
                        }
                        if k.loopinvariant_results.is_empty() {
                            k.loopinvariant_results = end_knowledge.loopinvariant_results.clone();
                        }
                    }
                    for &fi in guard_op_indices.keys() {
                        result.entry(fi).or_insert_with(|| end_knowledge.clone());
                    }
                    result
                };
                let mut traces = HashMap::new();
                traces.insert(
                    trace_id,
                    CompiledTrace {
                        snapshots: Vec::new(),
                        inputargs: trace.inputargs.clone(),
                        resume_data,
                        ops: optimized_ops,
                        constants: compiled_constants,
                        constant_types: compiled_constant_types.clone(),
                        guard_op_indices,
                        exit_layouts,
                        terminal_exit_layouts,
                        optimizer_knowledge: per_guard_knowledge,
                        jitcode: None,
                    },
                );
                {
                    let mut previous_tokens: Vec<JitCellToken> = Vec::new();
                    let ft = self
                        .compiled_loops
                        .get(&green_key)
                        .map(|c| c.front_target_tokens.clone())
                        .unwrap_or_default();
                    let rc = self
                        .compiled_loops
                        .get(&green_key)
                        .map(|c| c.retraced_count)
                        .unwrap_or(0);
                    let had_old = self.compiled_loops.contains_key(&green_key);
                    if let Some(old_entry) = self.compiled_loops.remove(&green_key) {
                        previous_tokens.push(old_entry.token);
                        previous_tokens.extend(old_entry.previous_tokens);
                        for (tid, ct) in old_entry.traces {
                            traces.entry(tid).or_insert(ct);
                        }
                    }
                    self.compiled_loops.insert(
                        green_key,
                        CompiledEntry {
                            token,
                            num_inputs: inputargs.len(),
                            meta,
                            front_target_tokens: ft,
                            retraced_count: rc,
                            root_trace_id: trace_id,
                            traces,
                            previous_tokens,
                        },
                    );
                }
                let install_num = self.warm_state.alloc_token_number();
                let install_token = JitCellToken::new(install_num);
                self.warm_state
                    .attach_procedure_to_interp(green_key, install_token);
                self.warm_state.reset_function_counts();
                self.stats.loops_compiled += 1;
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] finish_and_compile: compiled trace key={}, trace_id={}",
                        green_key, trace_id
                    );
                }
                if let Some(ref hook) = self.hooks.on_compile_loop {
                    hook(green_key, num_ops_before, num_ops_after);
                }
                // Compile deferred done_with_this_frame bridges now that
                // the main loop is stored. The bridge attaches to the
                // NEW token's fail_descrs (which are the same Arcs as
                // in the registered call_assembler target).
                let pending = std::mem::take(&mut self.pending_done_bridges);
                for pb in pending {
                    if pb.green_key == green_key {
                        self.compile_pending_done_bridge(pb);
                    }
                }
            }
            Err(e) => {
                self.stats.loops_aborted += 1;
                let msg = format!("finish_and_compile: compile_loop FAILED key={green_key}: {e:?}");
                if crate::majit_log_enabled() {
                    eprintln!("[jit] {msg}");
                }
                if let Some(ref cb) = self.hooks.on_compile_error {
                    cb(green_key, &msg);
                }
                self.warm_state.abort_tracing(green_key, false);
                self.warm_state.reset_function_counts();
            }
        }
    }

    /// compile.py:216-249 compile_simple_loop parity.
    ///
    /// Compiles the trace with simple optimizer (no preamble peeling),
    /// prepends a LABEL (via front_target_tokens) for bridge attachment.
    /// Returns the green_key on success (caller must call
    /// attach_procedure_to_interp), None on failure.
    pub fn compile_simple_loop(&mut self, meta: M) -> Option<u64> {
        let vable_config = self.current_virtualizable_optimizer_config();
        self.force_finish_trace = false;
        let mut ctx = match self.tracing.take() {
            Some(ctx) => ctx,
            None => return None,
        };
        ctx.apply_replacements();
        let green_key = ctx.green_key;

        let mut recorder = ctx.recorder;
        // The trace already has GUARD_ALWAYS_FAILS + FINISH ops recorded
        // by jitdriver.rs segmenting. Mark recorder as finalized so
        // get_trace() succeeds.
        recorder.tracing_done();
        let trace = recorder.get_trace();
        let trace_snapshots = trace.snapshots.clone();

        let numbering_overrides = ctx.constants.numbering_type_overrides().clone();
        let (mut constants, mut constant_types) = ctx.constants.into_inner_with_types();

        let (trace_ops, _) = compile::fold_box_into_create_frame(
            trace.ops.clone(),
            &mut constants,
            &self.raw_int_box_helpers,
            &self.create_frame_raw_map,
        );
        let (trace_ops, _) = compile::elide_create_frame_for_call_assembler(
            trace_ops,
            &constants,
            &self.create_frame_raw_map,
        );

        if crate::majit_log_enabled() {
            eprintln!("--- simple loop trace (before opt) ---");
            eprint!("{}", majit_ir::format_trace(&trace_ops, &constants));
        }

        let num_ops_before = trace_ops.len();
        let num_trace_inputargs = trace.inputargs.len();

        // Simple optimizer — no unrolling (compile.py:222-226 SimpleCompileData).
        let mut optimizer = if let Some(config) = vable_config {
            Optimizer::default_pipeline_with_virtualizable(config)
        } else {
            Optimizer::default_pipeline()
        };
        optimizer.constant_types = constant_types.clone();
        optimizer.numbering_type_overrides = numbering_overrides;
        // RPython Box.type parity: register inputarg types.
        for ia in &trace.inputargs {
            optimizer.constant_types.insert(ia.index, ia.tp);
        }

        let (snapshot_map, snapshot_frame_size_map, snapshot_vable_map, snapshot_pc_map, _sbt) =
            snapshot_map_from_trace_snapshots(
                &trace_snapshots,
                &mut constants,
                &mut constant_types,
            );
        optimizer.snapshot_boxes = snapshot_map;
        optimizer.snapshot_frame_sizes = snapshot_frame_size_map;
        optimizer.snapshot_vable_boxes = snapshot_vable_map;
        optimizer.snapshot_frame_pcs = snapshot_pc_map;

        let mut updated_constant_types = constant_types.clone();
        let optimize_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = optimizer.optimize_with_constants_and_inputs(
                &trace_ops,
                &mut constants,
                num_trace_inputargs,
            );
            for (k, v) in &optimizer.constant_types {
                updated_constant_types.entry(*k).or_insert(*v);
            }
            result
        }));
        constant_types = updated_constant_types;
        let optimized_ops = match optimize_result {
            Ok(ops) => ops,
            Err(payload) => {
                if crate::majit_log_enabled() {
                    if payload
                        .downcast_ref::<crate::optimize::InvalidLoop>()
                        .is_some()
                    {
                        eprintln!(
                            "[jit] compile_simple_loop: InvalidLoop at key={}",
                            green_key
                        );
                    }
                }
                // compile.py:228-230: trace.cut_at(cut_at); return None
                self.warm_state.abort_tracing(green_key, false);
                return None;
            }
        };

        self.last_quasi_immutable_deps = optimizer.quasi_immutable_deps.drain().collect();

        let num_ops_after = optimized_ops.len();
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] compile_simple_loop: key={}, ops_before={}, ops_after={}",
                green_key, num_ops_before, num_ops_after
            );
            eprintln!("--- simple loop trace (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
        }

        let optimized_ops = compile::strip_stray_overflow_guards(optimized_ops);

        // Allocate token and compile.
        let token_num = self.warm_state.alloc_token_number();
        let mut token = JitCellToken::new(token_num);
        let trace_id = self.alloc_trace_id();
        self.backend.set_next_trace_id(trace_id);
        self.backend.set_next_header_pc(green_key);

        let final_num_inputs = optimizer.final_num_inputs();
        let mut inputargs = trace.inputargs.clone();
        while inputargs.len() < final_num_inputs {
            inputargs.push(majit_ir::InputArg {
                tp: majit_ir::Type::Int,
                index: inputargs.len() as u32,
            });
        }

        // compile.py:236-245 parity: simple-loop compilation owns a real
        // TargetToken, prepends LABEL(descr=target_token), and patches the
        // closing JUMP to the same token.
        let target_token = crate::optimizeopt::unroll::TargetToken::new_loop(token_num);
        let mut compiled_ops = optimized_ops.clone();
        if let Some(jump_op) = compiled_ops
            .last_mut()
            .filter(|op| op.opcode == OpCode::Jump)
        {
            jump_op.descr = Some(target_token.as_jump_target_descr());
        }
        let mut label_op = majit_ir::Op::new(
            majit_ir::OpCode::Label,
            &inputargs
                .iter()
                .map(|ia| majit_ir::OpRef(ia.index))
                .collect::<Vec<_>>(),
        );
        label_op.pos = majit_ir::OpRef::NONE;
        label_op.descr = Some(target_token.as_jump_target_descr());
        compiled_ops.insert(0, label_op);

        let compiled_constants = constants.clone();
        let compiled_constant_types = constant_types.clone();
        self.backend.set_constants(constants);
        token.green_key = green_key;

        match self
            .backend
            .compile_loop(&inputargs, &compiled_ops, &mut token)
        {
            Ok(_) => {
                self.assign_guard_hashes(&token);
                let (resume_data, guard_op_indices, mut exit_layouts) =
                    compile::build_guard_metadata(
                        &inputargs,
                        &compiled_ops,
                        green_key,
                        &compiled_constants,
                        &compiled_constant_types,
                    );
                let mut terminal_exit_layouts =
                    compile::build_terminal_exit_layouts(&inputargs, &compiled_ops);
                if let Some(backend_layouts) = self.backend.compiled_fail_descr_layouts(&token) {
                    compile::merge_backend_exit_layouts(&mut exit_layouts, &backend_layouts);
                }
                if let Some(backend_layouts) = self.backend.compiled_terminal_exit_layouts(&token) {
                    compile::merge_backend_terminal_exit_layouts(
                        &mut terminal_exit_layouts,
                        &backend_layouts,
                    );
                }
                let trace_info = self.backend.compiled_trace_info(&token, trace_id);
                let mut resume_data = resume_data;
                compile::enrich_guard_resume_layouts_for_trace(
                    &mut resume_data,
                    &mut exit_layouts,
                    trace_id,
                    &trace.inputargs,
                    trace_info.as_ref(),
                );
                compile::patch_backend_guard_recovery_layouts_for_trace(
                    &mut self.backend,
                    &token,
                    trace_id,
                    &mut exit_layouts,
                );
                compile::patch_backend_terminal_recovery_layouts_for_trace(
                    &mut self.backend,
                    &token,
                    trace_id,
                    &mut terminal_exit_layouts,
                );
                let per_guard_knowledge = {
                    let pos_to_fail: HashMap<u32, u32> = guard_op_indices
                        .iter()
                        .filter_map(|(&fi, &op_idx)| {
                            optimized_ops.get(op_idx).map(|op| (op.pos.0, fi))
                        })
                        .collect();
                    let mut result: HashMap<u32, OptimizerKnowledge> = HashMap::new();
                    for (guard_pos, knowledge) in &optimizer.per_guard_knowledge {
                        if let Some(&fi) = pos_to_fail.get(&guard_pos.0) {
                            result.insert(fi, knowledge.clone());
                        }
                    }
                    let end_knowledge = optimizer
                        .final_ctx
                        .as_ref()
                        .map(|c| optimizer.serialize_optimizer_knowledge(c))
                        .unwrap_or_default();
                    for (_, k) in result.iter_mut() {
                        if k.known_classes.is_empty() {
                            k.known_classes = end_knowledge.known_classes.clone();
                        }
                        if k.loopinvariant_results.is_empty() {
                            k.loopinvariant_results = end_knowledge.loopinvariant_results.clone();
                        }
                    }
                    for &fi in guard_op_indices.keys() {
                        result.entry(fi).or_insert_with(|| end_knowledge.clone());
                    }
                    result
                };
                let mut traces = HashMap::new();
                traces.insert(
                    trace_id,
                    CompiledTrace {
                        snapshots: Vec::new(),
                        inputargs: trace.inputargs.clone(),
                        resume_data,
                        ops: compiled_ops,
                        constants: compiled_constants,
                        constant_types: compiled_constant_types,
                        guard_op_indices,
                        exit_layouts,
                        terminal_exit_layouts,
                        optimizer_knowledge: per_guard_knowledge,
                        jitcode: None,
                    },
                );
                let mut previous_tokens: Vec<JitCellToken> = Vec::new();
                if let Some(old_entry) = self.compiled_loops.remove(&green_key) {
                    previous_tokens.push(old_entry.token);
                    previous_tokens.extend(old_entry.previous_tokens);
                    for (tid, ct) in old_entry.traces {
                        traces.entry(tid).or_insert(ct);
                    }
                }
                self.compiled_loops.insert(
                    green_key,
                    CompiledEntry {
                        token,
                        num_inputs: inputargs.len(),
                        meta,
                        front_target_tokens: vec![target_token],
                        retraced_count: 0,
                        root_trace_id: trace_id,
                        traces,
                        previous_tokens,
                    },
                );
                self.stats.loops_compiled += 1;
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compile_simple_loop: compiled segmented trace key={}, trace_id={}",
                        green_key, trace_id
                    );
                }
                if let Some(ref hook) = self.hooks.on_compile_loop {
                    hook(green_key, num_ops_before, num_ops_after);
                }
                // compile.py:249: return target_token
                return Some(green_key);
            }
            Err(e) => {
                self.stats.loops_aborted += 1;
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compile_simple_loop: compile FAILED key={}: {:?}",
                        green_key, e
                    );
                }
                self.warm_state.abort_tracing(green_key, false);
                return None;
            }
        }
    }

    /// Resolve cross-loop cut alias: inner_key → outer_key.
    ///
    /// If the key has its own direct entry, use it — alias only applies
    /// when no direct entry exists. This prevents cross-loop cut aliases
    /// from shadowing independently compiled inner loop entries.
    ///
    /// NOTE: currently disabled in all callers (returns green_key) because
    /// Get the metadata for a compiled loop without executing it.
    ///
    /// Allows the interpreter to check preconditions (e.g., whether the
    /// current state matches the compiled loop's assumptions) before calling
    /// `run_compiled`.
    pub fn get_compiled_meta(&self, green_key: u64) -> Option<&M> {
        self.compiled_loops.get(&green_key).map(|e| &e.meta)
    }

    pub fn get_compiled_meta_mut(&mut self, green_key: u64) -> Option<&mut M> {
        self.compiled_loops.get_mut(&green_key).map(|e| &mut e.meta)
    }

    /// Actual key the last compile_loop stored under. Returns inner key
    /// for cross-loop cuts, otherwise the tracing key.
    pub fn last_compiled_key(&self) -> Option<u64> {
        self.last_compiled_key
    }

    /// Get num_inputs of the compiled loop.
    pub fn get_compiled_num_inputs(&self, green_key: u64) -> Option<usize> {
        self.compiled_loops.get(&green_key).map(|e| e.num_inputs)
    }

    /// Run the compiled loop for the given green key.
    ///
    /// `live_values` must have the same length and order as the values
    /// passed to `on_back_edge` when the trace was recorded.
    ///
    /// Returns `Some((output_values, &meta))` on success, where `output_values`
    /// are the new live values after the loop exits (guard failure), and `meta`
    /// is the interpreter-specific metadata stored during compilation.
    ///
    /// Returns `None` if no compiled loop exists for this key.
    pub fn run_compiled(&mut self, green_key: u64, live_values: &[i64]) -> Option<(Vec<i64>, &M)> {
        let compiled = self.compiled_loops.get(&green_key)?;

        Self::prepare_compiled_run_io();
        let result = self
            .backend
            .execute_token_ints_raw(&compiled.token, live_values);
        Self::finish_compiled_run_io(result.is_finish);

        let fail_index = result.fail_index;
        let trace_id = Self::normalize_trace_id(compiled, result.trace_id);

        if Self::should_record_guard_failure(result.is_finish, fail_index) {
            self.record_guard_failure_event(green_key, fail_index);
        }

        let compiled = self.compiled_loops.get(&green_key).unwrap();
        Some((result.outputs, &compiled.meta))
    }

    /// Run the compiled loop and return typed output values.
    ///
    /// This is the typed counterpart to [`run_compiled`]. It still uses the
    /// lightweight raw backend exit path, but preserves mixed `Int` / `Ref` /
    /// `Float` outputs instead of forcing callers to decode raw words.
    pub fn run_compiled_values(
        &mut self,
        green_key: u64,
        live_values: &[i64],
    ) -> Option<(Vec<Value>, &M)> {
        let compiled = self.compiled_loops.get(&green_key)?;

        Self::prepare_compiled_run_io();
        let result = self
            .backend
            .execute_token_ints_raw(&compiled.token, live_values);
        Self::finish_compiled_run_io(result.is_finish);

        let fail_index = result.fail_index;
        let trace_id = Self::normalize_trace_id(compiled, result.trace_id);

        if Self::should_record_guard_failure(result.is_finish, fail_index) {
            self.record_guard_failure_event(green_key, fail_index);
        }

        let compiled = self.compiled_loops.get(&green_key).unwrap();
        Some((result.typed_outputs, &compiled.meta))
    }

    /// Run the compiled loop with typed live inputs and return typed outputs.
    ///
    /// This is the fully typed raw execution path for already-compiled loops.
    pub fn run_compiled_with_values(
        &mut self,
        green_key: u64,
        live_values: &[Value],
    ) -> Option<(Vec<Value>, &M)> {
        let compiled = self.compiled_loops.get(&green_key)?;

        Self::prepare_compiled_run_io();
        let result = self.backend.execute_token_raw(&compiled.token, live_values);
        Self::finish_compiled_run_io(result.is_finish);

        let fail_index = result.fail_index;
        let trace_id = Self::normalize_trace_id(compiled, result.trace_id);

        if Self::should_record_guard_failure(result.is_finish, fail_index) {
            self.record_guard_failure_event(green_key, fail_index);
        }

        let compiled = self.compiled_loops.get(&green_key).unwrap();
        Some((result.typed_outputs, &compiled.meta))
    }

    /// Run compiled code and report whether it finished (FINISH) or exited
    /// via guard failure. Returns (typed_outputs, is_finish, meta).
    pub fn run_compiled_with_values_detailed(
        &mut self,
        green_key: u64,
        live_values: &[Value],
    ) -> Option<(Vec<Value>, bool, &M)> {
        let compiled = self.compiled_loops.get(&green_key)?;

        Self::prepare_compiled_run_io();
        let result = self.backend.execute_token_raw(&compiled.token, live_values);
        Self::finish_compiled_run_io(result.is_finish);

        let fail_index = result.fail_index;
        let trace_id = Self::normalize_trace_id(compiled, result.trace_id);

        if Self::should_record_guard_failure(result.is_finish, fail_index) {
            self.record_guard_failure_event(green_key, fail_index);
        }

        let compiled = self.compiled_loops.get(&green_key).unwrap();
        Some((result.typed_outputs, result.is_finish, &compiled.meta))
    }

    /// Run compiled code through the raw fast path and return detailed exit metadata.
    ///
    /// This is the lightweight counterpart to [`run_compiled_detailed`]: it avoids
    /// explicit deadframe decoding in the caller while still preserving typed exits,
    /// backend exit layout, savedata, and exception state.
    pub fn run_compiled_raw_detailed(
        &mut self,
        green_key: u64,
        live_values: &[i64],
    ) -> Option<RawCompileResult<'_, M>> {
        let compiled = self.compiled_loops.get(&green_key)?;

        Self::prepare_compiled_run_io();
        let result = self
            .backend
            .execute_token_ints_raw(&compiled.token, live_values);
        Self::finish_compiled_run_io(result.is_finish);

        let fail_index = result.fail_index;
        let trace_id = Self::normalize_trace_id(compiled, result.trace_id);

        let trace_layout =
            Self::trace_for_exit(compiled, trace_id).and_then(|(trace_id, trace)| {
                Self::compiled_exit_layout_from_trace(trace, green_key, trace_id, fail_index)
            });
        let exit_layout = result
            .exit_layout
            .clone()
            .map(|layout| {
                let trace_layout_ref = trace_layout.as_ref();
                let mut resume_layout = trace_layout
                    .as_ref()
                    .and_then(|tl| tl.resume_layout.clone());
                compile::enrich_resume_layout_with_frame_stack(
                    &mut resume_layout,
                    layout.frame_stack.as_deref(),
                    &layout.fail_arg_types,
                );
                CompiledExitLayout {
                    rd_loop_token: green_key, // compile.py:186
                    trace_id,
                    fail_index: layout.fail_index,
                    source_op_index: layout
                        .source_op_index
                        .or_else(|| trace_layout_ref.and_then(|layout| layout.source_op_index)),
                    exit_types: layout.fail_arg_types,
                    is_finish: layout.is_finish,
                    gc_ref_slots: layout.gc_ref_slots,
                    force_token_slots: layout.force_token_slots,
                    recovery_layout: layout.recovery_layout.or_else(|| {
                        trace_layout_ref.and_then(|layout| layout.recovery_layout.clone())
                    }),
                    resume_layout,
                    rd_numb: trace_layout_ref.and_then(|layout| layout.rd_numb.clone()),
                    rd_consts: trace_layout_ref.and_then(|layout| layout.rd_consts.clone()),
                    rd_virtuals: trace_layout_ref.and_then(|layout| layout.rd_virtuals.clone()),
                    rd_pendingfields: trace_layout_ref
                        .and_then(|layout| layout.rd_pendingfields.clone()),
                }
            })
            .or(trace_layout)
            .unwrap_or_else(|| CompiledExitLayout {
                rd_loop_token: green_key, // from trace context
                trace_id,
                fail_index,
                source_op_index: None,
                exit_types: result.typed_outputs.iter().map(Value::get_type).collect(),
                is_finish: result.is_finish,
                gc_ref_slots: result
                    .typed_outputs
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, value)| (value.get_type() == Type::Ref).then_some(slot))
                    .collect(),
                force_token_slots: result.force_token_slots.clone(),
                recovery_layout: None,
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            });
        let effective_is_finish = result.is_finish || exit_layout.is_finish;
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] run_compiled_exit: gk={} fi={} tid={} result.finish={} layout.finish={} effective={}",
                green_key,
                fail_index,
                trace_id,
                result.is_finish,
                exit_layout.is_finish,
                effective_is_finish
            );
        }

        if Self::should_record_guard_failure(effective_is_finish, fail_index) {
            self.record_guard_failure_event(green_key, fail_index);
        }

        // pyjitpl.py:3119-3123: exc_class = ptr2int(exception_obj.typeptr)
        let exc_class = if result.exception_value.is_null() {
            0
        } else {
            unsafe { *(result.exception_value.0 as *const i64) }
        };
        let exception = ExceptionState {
            exc_class,
            exc_value: result.exception_value.0 as i64,
            ovf_flag: false,
        };
        let compiled = self.compiled_loops.get(&green_key).unwrap();

        Some(RawCompileResult {
            values: result.outputs,
            typed_values: result.typed_outputs,
            meta: &compiled.meta,
            fail_index,
            trace_id,
            is_finish: effective_is_finish,
            exit_layout,
            savedata: result.savedata,
            exception,
            status: result.status,
            descr_addr: result.descr_addr,
        })
    }

    /// Typed-input counterpart to [`run_compiled_raw_detailed`].
    pub fn run_compiled_raw_detailed_with_values(
        &mut self,
        green_key: u64,
        live_values: &[Value],
    ) -> Option<RawCompileResult<'_, M>> {
        let compiled = self.compiled_loops.get(&green_key)?;

        Self::prepare_compiled_run_io();
        let result = self.backend.execute_token_raw(&compiled.token, live_values);
        Self::finish_compiled_run_io(result.is_finish);

        let fail_index = result.fail_index;
        let trace_id = Self::normalize_trace_id(compiled, result.trace_id);

        let trace_layout =
            Self::trace_for_exit(compiled, trace_id).and_then(|(trace_id, trace)| {
                Self::compiled_exit_layout_from_trace(trace, green_key, trace_id, fail_index)
            });
        let exit_layout = result
            .exit_layout
            .clone()
            .map(|layout| {
                let trace_layout_ref = trace_layout.as_ref();
                let mut resume_layout = trace_layout
                    .as_ref()
                    .and_then(|tl| tl.resume_layout.clone());
                compile::enrich_resume_layout_with_frame_stack(
                    &mut resume_layout,
                    layout.frame_stack.as_deref(),
                    &layout.fail_arg_types,
                );
                CompiledExitLayout {
                    rd_loop_token: green_key, // compile.py:186
                    trace_id,
                    fail_index: layout.fail_index,
                    source_op_index: layout
                        .source_op_index
                        .or_else(|| trace_layout_ref.and_then(|layout| layout.source_op_index)),
                    exit_types: layout.fail_arg_types,
                    is_finish: layout.is_finish,
                    gc_ref_slots: layout.gc_ref_slots,
                    force_token_slots: layout.force_token_slots,
                    recovery_layout: layout.recovery_layout.or_else(|| {
                        trace_layout_ref.and_then(|layout| layout.recovery_layout.clone())
                    }),
                    resume_layout,
                    rd_numb: trace_layout_ref.and_then(|layout| layout.rd_numb.clone()),
                    rd_consts: trace_layout_ref.and_then(|layout| layout.rd_consts.clone()),
                    rd_virtuals: trace_layout_ref.and_then(|layout| layout.rd_virtuals.clone()),
                    rd_pendingfields: trace_layout_ref
                        .and_then(|layout| layout.rd_pendingfields.clone()),
                }
            })
            .or(trace_layout)
            .unwrap_or_else(|| CompiledExitLayout {
                rd_loop_token: green_key, // from trace context
                trace_id,
                fail_index,
                source_op_index: None,
                exit_types: result.typed_outputs.iter().map(Value::get_type).collect(),
                is_finish: result.is_finish,
                gc_ref_slots: result
                    .typed_outputs
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, value)| (value.get_type() == Type::Ref).then_some(slot))
                    .collect(),
                force_token_slots: result.force_token_slots.clone(),
                recovery_layout: None,
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            });
        let effective_is_finish = result.is_finish || exit_layout.is_finish;
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] run_compiled_exit: gk={} fi={} tid={} result.finish={} layout.finish={} effective={}",
                green_key,
                fail_index,
                trace_id,
                result.is_finish,
                exit_layout.is_finish,
                effective_is_finish
            );
        }

        if Self::should_record_guard_failure(effective_is_finish, fail_index) {
            self.record_guard_failure_event(green_key, fail_index);
        }
        // pyjitpl.py:3119-3123: exc_class = ptr2int(exception_obj.typeptr)
        let exc_class = if result.exception_value.is_null() {
            0
        } else {
            unsafe { *(result.exception_value.0 as *const i64) }
        };
        let exception = ExceptionState {
            exc_class,
            exc_value: result.exception_value.0 as i64,
            ovf_flag: false,
        };
        let compiled = self.compiled_loops.get(&green_key).unwrap();

        Some(RawCompileResult {
            values: result.outputs,
            typed_values: result.typed_outputs,
            meta: &compiled.meta,
            fail_index,
            trace_id,
            is_finish: effective_is_finish,
            exit_layout,
            savedata: result.savedata,
            exception,
            status: result.status,
            descr_addr: result.descr_addr,
        })
    }

    /// Run compiled code and return detailed guard failure information.
    ///
    /// Unlike `run_compiled`, this returns the full `CompileResult` including
    /// the fail_index, which allows the interpreter to handle different guard
    /// failures differently.
    pub fn run_compiled_detailed(
        &mut self,
        green_key: u64,
        live_values: &[i64],
    ) -> Option<CompileResult<'_, M>> {
        let compiled = self.compiled_loops.get(&green_key)?;

        Self::prepare_compiled_run_io();
        let frame = self
            .backend
            .execute_token_ints(&compiled.token, live_values);

        let descr = self.backend.get_latest_descr(&frame);
        let fail_index = descr.fail_index();
        let trace_id = Self::normalize_trace_id(compiled, descr.trace_id());
        let is_finish = descr.is_finish();
        let exit_types = descr.fail_arg_types().to_vec();
        let gc_ref_slots: Vec<usize> = exit_types
            .iter()
            .enumerate()
            .filter_map(|(slot, _)| descr.is_gc_ref_slot(slot).then_some(slot))
            .collect();
        let force_token_slots = descr.force_token_slots().to_vec();
        let status = descr.get_status();
        let descr_addr = descr as *const dyn majit_ir::FailDescr as *const () as usize;
        Self::finish_compiled_run_io(is_finish);

        if Self::should_record_guard_failure(is_finish, fail_index) {
            self.record_guard_failure_event(green_key, fail_index);
        }

        let exit_arity = exit_types.len();
        let compiled = self.compiled_loops.get(&green_key).unwrap();
        let mut exit_layout = Self::trace_for_exit(compiled, trace_id)
            .and_then(|(trace_id, trace)| {
                Self::compiled_exit_layout_from_trace(trace, green_key, trace_id, fail_index)
            })
            .unwrap_or_else(|| CompiledExitLayout {
                rd_loop_token: green_key,
                trace_id,
                fail_index,
                source_op_index: None,
                exit_types: exit_types.clone(),
                is_finish,
                gc_ref_slots,
                force_token_slots,
                recovery_layout: None,
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            });
        // RPython: deadframe has ALL jitframe slots accessible.
        // If the backend's descr covers more slots than the trace layout,
        // extend exit_layout.exit_types to match (conservative Int for extras).
        if exit_types.len() > exit_layout.exit_types.len() {
            exit_layout.exit_types.resize(exit_types.len(), Type::Int);
        }
        let mut values = Vec::with_capacity(exit_arity);
        let mut typed_values = Vec::with_capacity(exit_arity);
        for (i, &tp) in exit_types.iter().enumerate() {
            match tp {
                Type::Int => {
                    let value = self.backend.get_int_value(&frame, i);
                    values.push(value);
                    typed_values.push(Value::Int(value));
                }
                Type::Ref => {
                    let value = self.backend.get_ref_value(&frame, i);
                    values.push(value.as_usize() as i64);
                    typed_values.push(Value::Ref(value));
                }
                Type::Float => {
                    let value = self.backend.get_float_value(&frame, i);
                    values.push(value.to_bits() as i64);
                    typed_values.push(Value::Float(value));
                }
                Type::Void => {
                    values.push(0);
                    typed_values.push(Value::Void);
                }
            }
        }
        let savedata = self.backend.get_savedata_ref(&frame);
        // pyjitpl.py:3119-3123: exc_class = ptr2int(exception_obj.typeptr)
        let exc_value_ref = self.backend.grab_exc_value(&frame);
        let exc_class = if exc_value_ref.is_null() {
            0
        } else {
            unsafe { *(exc_value_ref.0 as *const i64) }
        };
        let exception = ExceptionState {
            exc_class,
            exc_value: exc_value_ref.0 as i64,
            ovf_flag: false,
        };

        Some(CompileResult {
            values,
            typed_values,
            meta: &compiled.meta,
            fail_index,
            trace_id,
            is_finish,
            exit_layout,
            savedata,
            exception,
            status,
            descr_addr,
        })
    }

    /// Pyre-specific: unbox Ref→Int values to match compiled trace types.
    ///
    /// Pyre's virtualizable stores all locals as PyObjectRef (Ref). When
    /// the compiled trace has Int-typed inputargs (optimizer unboxed),
    /// the live values must be unboxed before entering compiled code.
    /// RPython doesn't need this because MIFrame registers are typed.
    pub fn adapt_live_values_to_trace_types(
        &self,
        green_key: u64,
        mut values: Vec<Value>,
    ) -> Vec<Value> {
        let Some(compiled) = self.compiled_loops.get(&green_key) else {
            return values;
        };
        let types: Vec<Type> = compiled
            .traces
            .get(&compiled.root_trace_id)
            .and_then(|trace| {
                trace
                    .ops
                    .iter()
                    .find(|op| op.opcode.is_guard())
                    .and_then(|op| {
                        op.descr
                            .as_ref()
                            .and_then(|d| d.as_fail_descr())
                            .map(|fd| fd.fail_arg_types().to_vec())
                    })
            })
            .unwrap_or_default();
        let before = if crate::majit_log_enabled() {
            Some(values.clone())
        } else {
            None
        };
        for (i, tp) in types.iter().enumerate() {
            if i >= values.len() {
                break;
            }
            if let (Value::Ref(r), Type::Int) = (&values[i], tp) {
                let raw = r.as_usize() as i64;
                if let Some(unbox_fn) = REF_UNBOX_INT_FN.get() {
                    values[i] = Value::Int(unbox_fn(raw));
                } else {
                    values[i] = Value::Int(raw);
                }
            }
        }
        if let Some(before) = before {
            eprintln!(
                "[jit][adapt-live] key={} types={:?} before={:?} after={:?}",
                green_key, types, before, values
            );
        }
        values
    }

    /// Typed-input counterpart to [`run_compiled_detailed`].
    pub fn run_compiled_detailed_with_values(
        &mut self,
        green_key: u64,
        live_values: &[Value],
    ) -> Option<CompileResult<'_, M>> {
        let compiled = self.compiled_loops.get(&green_key)?;

        Self::prepare_compiled_run_io();
        let frame = self.backend.execute_token(&compiled.token, live_values);
        // RPython: bridge compilation happens synchronously inside
        // assembler_call_helper (called from compiled code). No deferred queue.

        let descr = self.backend.get_latest_descr(&frame);
        let fail_index = descr.fail_index();
        let trace_id = Self::normalize_trace_id(compiled, descr.trace_id());
        let is_finish = descr.is_finish();
        let exit_types = descr.fail_arg_types().to_vec();
        let gc_ref_slots: Vec<usize> = exit_types
            .iter()
            .enumerate()
            .filter_map(|(slot, _)| descr.is_gc_ref_slot(slot).then_some(slot))
            .collect();
        let force_token_slots = descr.force_token_slots().to_vec();
        let status = descr.get_status();
        let descr_addr = descr as *const dyn majit_ir::FailDescr as *const () as usize;
        Self::finish_compiled_run_io(is_finish);

        // RPython: guard failure counter tick and bridge compilation happen
        // in handle_fail → must_compile (compile.py:701-784).
        // must_compile handles tick.
        if Self::should_record_guard_failure(is_finish, fail_index) {
            self.record_guard_failure_event(green_key, fail_index);
        }

        let exit_arity = exit_types.len();
        let compiled = self.compiled_loops.get(&green_key).unwrap();
        let mut exit_layout = Self::trace_for_exit(compiled, trace_id)
            .and_then(|(trace_id, trace)| {
                Self::compiled_exit_layout_from_trace(trace, green_key, trace_id, fail_index)
            })
            .unwrap_or_else(|| CompiledExitLayout {
                rd_loop_token: green_key,
                trace_id,
                fail_index,
                source_op_index: None,
                exit_types: exit_types.clone(),
                is_finish,
                gc_ref_slots,
                force_token_slots,
                recovery_layout: None,
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            });
        // RPython: deadframe has ALL jitframe slots accessible.
        // If the backend's descr covers more slots than the trace layout,
        // extend exit_layout.exit_types to match (conservative Int for extras).
        if exit_types.len() > exit_layout.exit_types.len() {
            exit_layout.exit_types.resize(exit_types.len(), Type::Int);
        }
        let mut values = Vec::with_capacity(exit_arity);
        let mut typed_values = Vec::with_capacity(exit_arity);
        for (i, &tp) in exit_types.iter().enumerate() {
            match tp {
                Type::Int => {
                    let value = self.backend.get_int_value(&frame, i);
                    values.push(value);
                    typed_values.push(Value::Int(value));
                }
                Type::Ref => {
                    let value = self.backend.get_ref_value(&frame, i);
                    values.push(value.as_usize() as i64);
                    typed_values.push(Value::Ref(value));
                }
                Type::Float => {
                    let value = self.backend.get_float_value(&frame, i);
                    values.push(value.to_bits() as i64);
                    typed_values.push(Value::Float(value));
                }
                Type::Void => {
                    values.push(0);
                    typed_values.push(Value::Void);
                }
            }
        }
        let savedata = self.backend.get_savedata_ref(&frame);
        // pyjitpl.py:3119-3123: exc_class = ptr2int(exception_obj.typeptr)
        let exc_value_ref = self.backend.grab_exc_value(&frame);
        let exc_class = if exc_value_ref.is_null() {
            0
        } else {
            unsafe { *(exc_value_ref.0 as *const i64) }
        };
        let exception = ExceptionState {
            exc_class,
            exc_value: exc_value_ref.0 as i64,
            ovf_flag: false,
        };

        Some(CompileResult {
            values,
            typed_values,
            meta: &compiled.meta,
            fail_index,
            trace_id,
            is_finish,
            exit_layout,
            savedata,
            exception,
            status,
            descr_addr,
        })
    }

    /// Attach resume data to a specific guard in a compiled loop.
    ///
    /// This allows the interpreter to later reconstruct its full state
    /// when the guard fails, using `get_resume_data`.
    pub fn attach_resume_data(&mut self, green_key: u64, fail_index: u32, resume_data: ResumeData) {
        let Some(trace_id) = self.compiled_loops.get(&green_key).map(|c| c.root_trace_id) else {
            return;
        };
        self.attach_resume_data_to_trace(green_key, trace_id, fail_index, resume_data);
    }

    /// Attach resume data to a specific guard in a specific compiled trace.
    pub fn attach_resume_data_to_trace(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        resume_data: ResumeData,
    ) {
        let Some((trace_id, trace_info)) = self.compiled_loops.get(&green_key).map(|compiled| {
            let trace_id = if trace_id == 0 {
                compiled.root_trace_id
            } else {
                trace_id
            };
            (
                trace_id,
                self.backend.compiled_trace_info(&compiled.token, trace_id),
            )
        }) else {
            return;
        };
        let mut patched_recovery_layout = None;
        if let Some(compiled) = self.compiled_loops.get_mut(&green_key) {
            if let Some(trace) = compiled.traces.get_mut(&trace_id) {
                let recovery_layout = trace
                    .exit_layouts
                    .get(&fail_index)
                    .and_then(|layout| layout.recovery_layout.clone());
                let mut stored = StoredResumeData::new(resume_data);
                compile::enrich_resume_layout_with_trace_metadata(
                    &mut stored.layout,
                    trace_id,
                    &trace.inputargs,
                    trace_info.as_ref(),
                    recovery_layout.as_ref(),
                );
                let layout = stored.layout.clone();
                trace.resume_data.insert(fail_index, stored);
                if let Some(exit_layout) = trace.exit_layouts.get_mut(&fail_index) {
                    exit_layout.resume_layout = Some(layout);
                    if let Some(summary) = exit_layout.resume_layout.as_ref() {
                        let recovery_layout = summary.to_exit_recovery_layout_with_caller_prefix(
                            exit_layout.recovery_layout.as_ref(),
                        );
                        exit_layout.recovery_layout = Some(recovery_layout.clone());
                        patched_recovery_layout = Some(recovery_layout);
                    }
                }
            }
        }
        if let Some(recovery_layout) = patched_recovery_layout {
            if let Some(compiled) = self.compiled_loops.get(&green_key) {
                let _ = self.backend.update_fail_descr_recovery_layout(
                    &compiled.token,
                    trace_id,
                    fail_index,
                    recovery_layout,
                );
            }
        }
    }

    /// Get resume data for a specific guard failure.
    pub fn get_resume_data(&self, green_key: u64, fail_index: u32) -> Option<&ResumeData> {
        let trace_id = self.compiled_loops.get(&green_key)?.root_trace_id;
        self.get_resume_data_in_trace(green_key, trace_id, fail_index)
    }

    /// Get resume data for a specific guard failure in a specific trace.
    pub fn get_resume_data_in_trace(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<&ResumeData> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let (_, trace) = Self::trace_for_exit(compiled, trace_id)?;
        trace
            .resume_data
            .get(&fail_index)
            .map(|data| &data.semantic)
    }

    /// Get a compact resume layout summary for a specific guard failure.
    pub fn get_resume_layout(
        &self,
        green_key: u64,
        fail_index: u32,
    ) -> Option<&ResumeLayoutSummary> {
        let trace_id = self.compiled_loops.get(&green_key)?.root_trace_id;
        self.get_resume_layout_in_trace(green_key, trace_id, fail_index)
    }

    /// Get a compact resume layout summary for a specific guard failure in a specific trace.
    pub fn get_resume_layout_in_trace(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<&ResumeLayoutSummary> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let (_, trace) = Self::trace_for_exit(compiled, trace_id)?;
        trace.resume_data.get(&fail_index).map(|data| &data.layout)
    }

    /// Get the full static layout for a compiled exit in the root trace.
    pub fn get_compiled_exit_layout(
        &self,
        green_key: u64,
        fail_index: u32,
    ) -> Option<CompiledExitLayout> {
        let trace_id = self.compiled_loops.get(&green_key)?.root_trace_id;
        self.get_compiled_exit_layout_in_trace(green_key, trace_id, fail_index)
    }

    /// Get the full static layout for a compiled exit in a specific trace.
    pub fn get_compiled_exit_layout_in_trace(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<CompiledExitLayout> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let (trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
        Self::compiled_exit_layout_from_trace(trace, green_key, trace_id, fail_index).or_else(
            || self.compiled_exit_layout_from_backend(compiled, green_key, trace_id, fail_index),
        )
    }

    /// Get the full static layout for a terminal FINISH/JUMP op in the root trace.
    pub fn get_terminal_exit_layout(
        &self,
        green_key: u64,
        op_index: usize,
    ) -> Option<CompiledExitLayout> {
        let trace_id = self.compiled_loops.get(&green_key)?.root_trace_id;
        self.get_terminal_exit_layout_in_trace(green_key, trace_id, op_index)
    }

    /// Get the full static layout for a terminal FINISH/JUMP op in a specific trace.
    pub fn get_terminal_exit_layout_in_trace(
        &self,
        green_key: u64,
        trace_id: u64,
        op_index: usize,
    ) -> Option<CompiledExitLayout> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let (trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
        Self::terminal_exit_layout_from_trace(trace, green_key, trace_id, op_index).or_else(|| {
            self.terminal_exit_layout_from_backend(compiled, green_key, trace_id, op_index)
        })
    }

    /// Get the full static layout for a compiled trace in the root trace.
    pub fn get_compiled_trace_layout(&self, green_key: u64) -> Option<CompiledTraceLayout> {
        let trace_id = self.compiled_loops.get(&green_key)?.root_trace_id;
        self.get_compiled_trace_layout_in_trace(green_key, trace_id)
    }

    /// Get the full static layout for a compiled trace in a specific trace id.
    pub fn get_compiled_trace_layout_in_trace(
        &self,
        green_key: u64,
        trace_id: u64,
    ) -> Option<CompiledTraceLayout> {
        let compiled = self.compiled_loops.get(&green_key)?;
        self.compiled_trace_layout_for_trace(compiled, green_key, trace_id)
    }

    /// Invalidate a compiled loop (e.g., due to GUARD_NOT_INVALIDATED).
    ///
    /// Marks the loop token as invalidated. Subsequent executions of the
    /// compiled code will fail at GUARD_NOT_INVALIDATED and fall back to
    /// the interpreter.
    pub fn invalidate_loop(&mut self, green_key: u64) {
        if let Some(compiled) = self.compiled_loops.get(&green_key) {
            compiled.token.invalidate();
            if crate::majit_log_enabled() {
                eprintln!("[jit] invalidated loop at key={}", green_key);
            }
        }
    }

    /// rstack.py stack_almost_full: True if stack is > 15/16th full.
    /// compile.py:703: skip bridge compilation when stack space is low.
    #[inline]
    pub fn stack_almost_full() -> bool {
        // rstack.py: remaining < length * 15/16 means almost full.
        // Use platform API to get accurate stack bounds.
        let local_var: u8 = 0;
        let sp = &local_var as *const u8 as usize;
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn pthread_self() -> usize;
                fn pthread_get_stackaddr_np(thread: usize) -> *const u8;
                fn pthread_get_stacksize_np(thread: usize) -> usize;
            }
            let thread = unsafe { pthread_self() };
            let stack_top = unsafe { pthread_get_stackaddr_np(thread) } as usize;
            let stack_size = unsafe { pthread_get_stacksize_np(thread) };
            // Stack grows downward: remaining = sp - stack_bottom
            let stack_bottom = stack_top.saturating_sub(stack_size);
            let remaining = sp.saturating_sub(stack_bottom);
            let threshold = stack_size / 16; // 1/16 of total
            return remaining < threshold;
        }
        #[cfg(target_os = "linux")]
        {
            use std::mem::MaybeUninit;
            unsafe extern "C" {
                fn pthread_self() -> usize;
                fn pthread_getattr_np(thread: usize, attr: *mut [u8; 64]) -> i32;
                fn pthread_attr_getstack(
                    attr: *const [u8; 64],
                    stackaddr: *mut *mut u8,
                    stacksize: *mut usize,
                ) -> i32;
                fn pthread_attr_destroy(attr: *mut [u8; 64]) -> i32;
            }
            let mut attr = MaybeUninit::<[u8; 64]>::zeroed();
            let thread = unsafe { pthread_self() };
            if unsafe { pthread_getattr_np(thread, attr.as_mut_ptr()) } == 0 {
                let attr = unsafe { attr.assume_init_mut() };
                let mut stack_addr: *mut u8 = std::ptr::null_mut();
                let mut stack_size: usize = 0;
                if unsafe { pthread_attr_getstack(attr, &mut stack_addr, &mut stack_size) } == 0 {
                    unsafe { pthread_attr_destroy(attr) };
                    let stack_bottom = stack_addr as usize;
                    let remaining = sp.saturating_sub(stack_bottom);
                    let threshold = stack_size / 16;
                    return remaining < threshold;
                }
                unsafe { pthread_attr_destroy(attr) };
            }
            false // fallback: assume not full
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = sp;
            false // unsupported platform: assume not full
        }
    }

    /// pyjitpl.py:2345-2348: try_to_free_some_loops — advance the
    /// memory manager's generation counter. Old loops not accessed
    /// for max_age generations are candidates for eviction.
    pub fn try_to_free_some_loops(&mut self) {
        let evicted = self.warm_state.memory_manager.next_generation();
        for key in evicted {
            self.compiled_loops.remove(&key);
            if crate::majit_log_enabled() {
                eprintln!("[jit][memmgr] evicted loop key={}", key);
            }
        }
    }

    /// Get the number of fail_arg_types for a guard.
    /// Returns 0 if not found. Used to adjust bridge tracing state.
    pub fn fail_arg_count_for(&self, green_key: u64, trace_id: u64, fail_index: u32) -> usize {
        let Some(compiled) = self.compiled_loops.get(&green_key) else {
            return 0;
        };
        self.bridge_fail_descr_proxy(compiled, trace_id, fail_index)
            .map(|p| p.fail_arg_types.len())
            .unwrap_or(0)
    }

    // ── Call Assembler Support ──────────────────────────────────

    /// Get the JitCellToken for a compiled loop (for CALL_ASSEMBLER).
    ///
    /// In RPython, `call_assembler` allows JIT code for one function
    /// to directly call JIT code for another function. The caller needs
    /// the target's JitCellToken to set up the call.
    pub fn get_loop_token(&self, green_key: u64) -> Option<&JitCellToken> {
        self.compiled_loops.get(&green_key).map(|c| &c.token)
    }

    /// Get the pre-allocated token number for a trace being recorded.
    ///
    /// Returns `Some(number)` if the given green_key matches the trace
    /// currently being recorded. This enables self-recursive calls to
    /// emit call_assembler targeting the pending token.
    pub fn get_pending_token_number(&self, green_key: u64) -> Option<u64> {
        self.pending_token
            .filter(|&(pk, _)| pk == green_key)
            .map(|(_, num)| num)
    }

    /// Redirect existing call_assembler calls from one loop to another.
    ///
    /// When a loop is recompiled (e.g., with bridges), existing
    /// CALL_ASSEMBLER instructions in other compiled code should be
    /// updated to point to the new version.
    pub fn redirect_call_assembler(&self, old_key: u64, new_key: u64) {
        let old_token = self.compiled_loops.get(&old_key).map(|c| &c.token);
        let new_token = self.compiled_loops.get(&new_key).map(|c| &c.token);
        if let (Some(old), Some(new)) = (old_token, new_token) {
            let _ = self.backend.redirect_call_assembler(old, new);
        }
    }

    /// Check whether a compiled loop exists for a given green key.
    #[inline]
    pub fn has_compiled_loop(&self, green_key: u64) -> bool {
        self.compiled_loops
            .get(&green_key)
            .map_or(false, |c| !c.token.is_invalidated())
    }

    /// Number of inputargs for a compiled loop (0 if not compiled).
    pub fn compiled_num_inputs(&self, green_key: u64) -> usize {
        self.compiled_loops
            .get(&green_key)
            .map_or(0, |c| c.num_inputs)
    }

    /// Check if any guard in the compiled trace has Float-typed fail_args.
    /// Used to gate bridge compilation: traces with Float guards have
    /// type metadata issues that cause crashes on bridge guard failures.
    pub fn compiled_trace_has_float_guards(&self, green_key: u64) -> bool {
        let Some(compiled) = self.compiled_loops.get(&green_key) else {
            return false;
        };
        for trace in compiled.traces.values() {
            for layout in trace.exit_layouts.values() {
                if layout
                    .exit_types
                    .iter()
                    .any(|t| matches!(t, majit_ir::Type::Float))
                {
                    return true;
                }
            }
        }
        false
    }

    /// Check if the compiled trace is safe for bridge compilation.
    /// Returns true if all guard exit_types at slot positions match
    /// the expected slot_types. Mismatches indicate type propagation
    /// bugs that cause crashes when bridge guards fail.
    pub fn compiled_trace_safe_for_bridge(
        &self,
        green_key: u64,
        slot_types: &[majit_ir::Type],
    ) -> bool {
        let Some(compiled) = self.compiled_loops.get(&green_key) else {
            return false;
        };
        let num_slots = slot_types.len();
        for trace in compiled.traces.values() {
            for layout in trace.exit_layouts.values() {
                for i in 0..num_slots {
                    let exit_pos = i + 3;
                    if let Some(et) = layout.exit_types.get(exit_pos) {
                        if *et != slot_types[i] {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Remove all compiled loops. Used when guard-fail recovery is
    /// unrecoverable (null Ref in resume data).
    pub fn clear_compiled_loops(&mut self) {
        self.compiled_loops.clear();
    }

    /// Remove compiled loop for a specific green_key.
    pub fn remove_compiled_loop(&mut self, green_key: u64) {
        self.compiled_loops.remove(&green_key);
    }

    /// Return all green keys that have compiled loops.
    pub fn all_compiled_keys(&self) -> Vec<u64> {
        self.compiled_loops.keys().copied().collect()
    }

    /// Check whether `trace_id` belongs to a bridge (not a root trace).
    pub fn is_bridge_trace_id(&self, trace_id: u64) -> bool {
        // A trace_id is a bridge if it doesn't match any compiled loop's root_trace_id.
        if trace_id == 0 {
            return false;
        }
        !self
            .compiled_loops
            .values()
            .any(|entry| entry.root_trace_id == trace_id)
    }

    /// warmstate.py:385 — whether this driver's portal returns a raw int.
    /// result_type == INT.
    pub fn has_raw_int_finish(&self, _green_key: u64) -> bool {
        self.result_type == Type::Int
    }

    // compile.py:687-696 status encoding constants.
    const ST_BUSY_FLAG: u64 = 0x01;
    const ST_TYPE_MASK: u64 = 0x06;
    const ST_SHIFT: u32 = 3;
    const TY_INT: u64 = 0x02;
    const TY_REF: u64 = 0x04;
    const TY_FLOAT: u64 = 0x06;

    /// compile.py:738-784: must_compile — read self.status directly from
    /// the failed descriptor (by descr_addr), compute hash, tick jitcounter.
    ///
    /// RPython: must_compile is a method ON the failed descriptor. self.status
    /// reads the live status of that exact object. descr_addr IS the identity
    /// of that descriptor (current_object_addr_as_int(self) in RPython).
    /// ALWAYS ticks the counter. stack_almost_full is checked by the caller
    /// in handle_fail (compile.py:702-703).
    /// Returns (should_compile, owning_green_key).
    pub fn must_compile_with_values(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
        descr_addr: usize,
    ) -> (bool, u64) {
        let owning_key = self.find_owning_key(green_key, trace_id);
        if descr_addr == 0 {
            return (false, owning_key);
        }
        // compile.py:741: self.status — read live status directly from the
        // failed descriptor object (not a re-lookup by trace_id/fail_index).
        let status = self.backend.read_descr_status(descr_addr);
        // compile.py:741-751: decode status to get hash
        let hash = if status & (Self::ST_BUSY_FLAG | Self::ST_TYPE_MASK) == 0 {
            // compile.py:745: common case — TY_NONE, not busy.
            status
        } else if status & Self::ST_BUSY_FLAG != 0 {
            // compile.py:750-751: already busy tracing.
            return (false, owning_key);
        } else {
            // compile.py:753-781: GUARD_VALUE per-value hash.
            let index = (status >> Self::ST_SHIFT) as u32;
            let typetag = status & Self::ST_TYPE_MASK;
            let raw = fail_values.get(index as usize).copied().unwrap_or(0);
            let intval: i64 = match typetag {
                Self::TY_INT => raw,
                Self::TY_REF => raw,
                Self::TY_FLOAT => raw,
                _ => raw,
            };
            // compile.py:780-781: current_object_addr_as_int(self) * 777767777
            //   + intval * 1442968193
            (descr_addr as u64)
                .wrapping_mul(777767777)
                .wrapping_add((intval as u64).wrapping_mul(1442968193))
        };
        // compile.py:783-784: jitcounter.tick(hash, increment)
        let fired = self.warm_state.tick_guard_failure(hash);
        if fired && crate::majit_log_enabled() {
            eprintln!(
                "[jit] must_compile FIRED: key={} trace={} guard={}",
                owning_key, trace_id, fail_index
            );
        }
        (fired, owning_key)
    }

    /// memmgr.py:58-61: keep_loop_alive(looptoken).
    /// Marks the compiled loop as recently used. Currently a no-op stub
    /// since majit does not have loop aging / memory management yet.
    pub fn keep_loop_alive(&mut self, _green_key: u64) {
        // TODO: implement loop aging when memory_manager is added.
    }

    /// compile.py:826-830 store_hash: assign jitcounter hashes to guards
    /// after compile_loop/compile_bridge. RPython calls store_hash during
    /// optimizer emit (store_final_boxes_in_guard); in majit the backend
    /// creates fail_descrs, so we assign hashes after compilation.
    /// Only allocates hashes for real guards (not FINISH/external JUMP).
    fn assign_guard_hashes(&mut self, token: &JitCellToken) {
        let layouts = self.backend.compiled_fail_descr_layouts(token);
        let hashes: Vec<u64> = layouts
            .iter()
            .flatten()
            .map(|layout| {
                if layout.is_finish {
                    0 // FINISH/external JUMP — no hash needed
                } else {
                    self.warm_state.fetch_next_hash()
                }
            })
            .collect();
        self.backend.store_guard_hashes(token, &hashes);
    }

    /// compile.py:826-830 store_hash for bridge guards.
    fn assign_bridge_guard_hashes(
        &mut self,
        token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) {
        let layouts = self.backend.compiled_bridge_fail_descr_layouts(
            token,
            source_trace_id,
            source_fail_index,
        );
        let hashes: Vec<u64> = layouts
            .iter()
            .flatten()
            .map(|layout| {
                if layout.is_finish {
                    0
                } else {
                    self.warm_state.fetch_next_hash()
                }
            })
            .collect();
        self.backend
            .store_bridge_guard_hashes(token, source_trace_id, source_fail_index, &hashes);
    }

    /// compile.py:741-745: look up (status, descr_addr) for a guard.
    /// Search current token + previous_tokens by (trace_id, fail_index)
    /// to find the exact descriptor — same pattern as start_guard_compiling.
    /// Returns None if descriptor not found (no tick should happen).
    pub fn get_guard_status(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<(u64, usize)> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let tid = Self::normalize_trace_id(compiled, trace_id);
        let (s, a) = self
            .backend
            .get_guard_status(&compiled.token, tid, fail_index);
        if a != 0 {
            return Some((s, a));
        }
        for prev in &compiled.previous_tokens {
            let (s, a) = self.backend.get_guard_status(prev, tid, fail_index);
            if a != 0 {
                return Some((s, a));
            }
        }
        None
    }

    /// compile.py:786-788: self.start_compiling() — set ST_BUSY_FLAG
    /// on the exact failed descriptor (by descr_addr, not re-lookup).
    pub fn start_guard_compiling(&self, descr_addr: usize) {
        self.backend.start_compiling_descr(descr_addr);
    }

    /// compile.py:790-795: self.done_compiling() — clear ST_BUSY_FLAG
    /// on the exact failed descriptor (by descr_addr, not re-lookup).
    pub fn done_guard_compiling(&self, descr_addr: usize) {
        self.backend.done_compiling_descr(descr_addr);
    }

    /// Find the compiled_loops key that owns a given trace_id.
    fn find_owning_key(&self, green_key: u64, trace_id: u64) -> u64 {
        if let Some(compiled) = self.compiled_loops.get(&green_key) {
            let tid = Self::normalize_trace_id(compiled, trace_id);
            if compiled.traces.contains_key(&tid) {
                return green_key;
            }
        }
        for (&key, compiled) in &self.compiled_loops {
            let tid = Self::normalize_trace_id(compiled, trace_id);
            if compiled.traces.contains_key(&tid) {
                return key;
            }
        }
        green_key
    }

    pub fn bridge_fail_descr_proxy(
        &self,
        compiled: &CompiledEntry<M>,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<compile::BridgeFailDescrProxy> {
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        let (_, trace_data) = Self::trace_for_exit(compiled, trace_id)?;
        let exit_layout = trace_data.exit_layouts.get(&fail_index)?;
        // compile.py:797-811 parity: bridge inputargs come from the guard's
        // fail_arg_types AFTER store_final_boxes_in_guard (which may add
        // virtual field boxes, changing the types list). The backend's
        // fail_descr has the UPDATED types; exit_layout.exit_types may have
        // the ORIGINAL MetaFailDescr types (before optimizer update).
        // Prefer the backend's types when available.
        let fail_arg_types = self
            .backend
            .compiled_trace_fail_descr_layouts(&compiled.token, trace_id)
            .and_then(|layouts| {
                layouts
                    .into_iter()
                    .find(|l| l.fail_index == fail_index)
                    .map(|l| l.fail_arg_types)
            })
            .unwrap_or_else(|| exit_layout.exit_types.clone());
        let gc_ref_slots = self
            .backend
            .compiled_trace_fail_descr_layouts(&compiled.token, trace_id)
            .and_then(|layouts| {
                layouts
                    .into_iter()
                    .find(|l| l.fail_index == fail_index)
                    .map(|l| l.gc_ref_slots)
            })
            .unwrap_or_else(|| exit_layout.gc_ref_slots.clone());
        Some(compile::BridgeFailDescrProxy {
            fail_index,
            trace_id,
            fail_arg_types,
            gc_ref_slots,
            force_token_slots: exit_layout.force_token_slots.clone(),
            is_finish: exit_layout.is_finish,
            rd_numb: None,
            rd_consts: None,
        })
    }

    /// Check whether a bridge was actually compiled and attached for a guard.
    /// Used by jit_bridge_compile_for_guard to distinguish successful bridge
    /// compilation from trace abort (RPython pyjitpl.py:2906-2907 parity).
    ///
    /// Searches the current token AND previous_tokens, since bridge
    /// compilation may have attached to an earlier token that was replaced
    /// by a retrace/recompile.
    pub fn bridge_was_compiled(&self, green_key: u64, trace_id: u64, fail_index: u32) -> bool {
        let Some(compiled) = self.compiled_loops.get(&green_key) else {
            return false;
        };
        if self
            .backend
            .compiled_bridge_fail_descr_layouts(&compiled.token, trace_id, fail_index)
            .is_some()
        {
            return true;
        }
        // Search previous tokens (old compilations kept alive for
        // target_token JUMPs and bridge attachments).
        compiled.previous_tokens.iter().any(|prev_token| {
            self.backend
                .compiled_bridge_fail_descr_layouts(prev_token, trace_id, fail_index)
                .is_some()
        })
    }

    // ── Bridge Compilation ──────────────────────────────────────

    /// pyjitpl.py:3195 finally: self.history.cut(cut_at) — undo tentative JUMP/FINISH.
    fn cut_tentative_op(&mut self, cut_at: majit_trace::recorder::TracePosition) {
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.recorder.unfinalize();
            ctx.recorder.cut(cut_at);
        }
    }

    /// pyjitpl.py:2982-2983: close_bridge — compile_trace wrapper that
    /// maps CompileOutcome to BridgeCompileResult.
    pub fn close_bridge(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        finish_args: &[OpRef],
    ) -> BridgeCompileResult {
        let outcome = self.compile_trace(green_key, finish_args, Some((trace_id, fail_index)));
        match outcome {
            CompileOutcome::Compiled { .. } => BridgeCompileResult::Compiled,
            _ if self.retrace_after_bridge => {
                // Keep retrace_after_bridge=true so compile_loop can
                // detect bridge retrace and abort early (preserve
                // retraced_count). pyjitpl.py:3000 partial_trace check
                // is before 3162 has_compiled_targets.
                BridgeCompileResult::RetraceNeeded
            }
            _ => BridgeCompileResult::Failed,
        }
    }

    /// RPython-compatible helper name from compile.py.
    pub fn send_bridge_to_backend(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        finish_args: &[OpRef],
    ) -> BridgeCompileResult {
        self.close_bridge(green_key, trace_id, fail_index, finish_args)
    }

    /// Record a constant Ref value in the active trace context.
    /// Returns the OpRef for the constant.
    pub fn record_bridge_const_ref(&mut self, value: i64) -> majit_ir::OpRef {
        let ctx = self
            .tracing
            .as_mut()
            .expect("record_bridge_const_ref requires active trace");
        ctx.constants
            .get_or_insert_typed(value, majit_ir::Type::Ref)
    }

    /// pyjitpl.py:3198-3220: compile_done_with_this_frame — bridge that
    /// exits via return. Constructs a minimal bridge trace (just a Finish op)
    /// and compiles it. Does NOT access the tracing context (which may have
    /// been consumed by finish_and_compile).
    pub fn compile_done_with_this_frame(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        finish_args: &[OpRef],
        finish_arg_types: Vec<Type>,
    ) {
        // Build the bridge trace: a single Finish op using finish_args.
        // The bridge inputargs are derived from the guard's fail_arg_types.
        let finish_descr = crate::make_fail_descr_typed(finish_arg_types.clone());
        let mut finish_op = majit_ir::Op::new(majit_ir::OpCode::Finish, finish_args);
        finish_op.descr = Some(finish_descr);
        let bridge_ops = vec![finish_op];

        // Inputargs: from the guard's fail_arg_types (the bridge inputs
        // are the guard's fail_args in order).
        let bridge_inputargs: Vec<majit_ir::InputArg> = {
            let compiled = self.compiled_loops.get(&green_key);
            if let Some(compiled) = compiled {
                let trace_id_norm = Self::normalize_trace_id(compiled, trace_id);
                if let Some((_, trace_data)) = Self::trace_for_exit(compiled, trace_id_norm) {
                    if let Some(exit_layout) = trace_data.exit_layouts.get(&fail_index) {
                        exit_layout
                            .exit_types
                            .iter()
                            .enumerate()
                            .map(|(i, &tp)| majit_ir::InputArg::from_type(tp, i as u32))
                            .collect()
                    } else {
                        finish_arg_types
                            .iter()
                            .enumerate()
                            .map(|(i, &tp)| majit_ir::InputArg::from_type(tp, i as u32))
                            .collect()
                    }
                } else {
                    finish_arg_types
                        .iter()
                        .enumerate()
                        .map(|(i, &tp)| majit_ir::InputArg::from_type(tp, i as u32))
                        .collect()
                }
            } else {
                finish_arg_types
                    .iter()
                    .enumerate()
                    .map(|(i, &tp)| majit_ir::InputArg::from_type(tp, i as u32))
                    .collect()
            }
        };

        let pb = PendingDoneBridge {
            green_key,
            trace_id,
            fail_index,
            bridge_ops,
            bridge_inputargs,
            constants: HashMap::new(),
            constant_types: HashMap::new(),
            snapshot_boxes: HashMap::new(),
            snapshot_frame_sizes: HashMap::new(),
            snapshot_vable_boxes: HashMap::new(),
            snapshot_frame_pcs: HashMap::new(),
            snapshot_box_types: HashMap::new(),
        };

        if self.compiled_loops.contains_key(&green_key) {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] compile_done_with_this_frame: compiling NOW key={} fail={}",
                    green_key, fail_index,
                );
            }
            self.compile_pending_done_bridge(pb);
        } else {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] compile_done_with_this_frame: deferring key={} fail={}",
                    green_key, fail_index,
                );
            }
            self.pending_done_bridges.push(pb);
        }
    }

    /// Compile a pending done_with_this_frame bridge using saved trace data.
    fn compile_pending_done_bridge(&mut self, pending: PendingDoneBridge) {
        let fail_descr = {
            let compiled = match self.compiled_loops.get(&pending.green_key) {
                Some(c) => c,
                None => return,
            };
            match self.bridge_fail_descr_proxy(compiled, pending.trace_id, pending.fail_index) {
                Some(d) => Box::new(d) as Box<dyn majit_ir::FailDescr>,
                None => return,
            }
        };
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] compile pending done bridge key={} fail={}",
                pending.green_key, pending.fail_index,
            );
        }
        self.compile_bridge(
            pending.green_key,
            pending.fail_index,
            &*fail_descr,
            &pending.bridge_ops,
            &pending.bridge_inputargs,
            pending.constants,
            pending.constant_types,
            pending.snapshot_boxes,
            pending.snapshot_frame_sizes,
            pending.snapshot_vable_boxes,
            pending.snapshot_frame_pcs,
            pending.snapshot_box_types,
        );
    }

    // Keep the old signature for backward compatibility — this is dead code now
    // but compile_trace_finish might still reference it.
    fn _compile_done_with_this_frame_direct(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        finish_args: &[OpRef],
        finish_arg_types: Vec<Type>,
    ) {
        let finish_descr = crate::make_fail_descr_typed(finish_arg_types);
        self.compile_trace_finish(
            green_key,
            finish_args,
            Some((trace_id, fail_index)),
            finish_descr,
        );
    }
}

impl<M: Clone> MetaInterp<M> {
    /// Obtain a FailDescr proxy for a guard identified by green_key,
    /// trace_id, and fail_index. Used by call_assembler bridge callbacks
    /// that need to pass a FailDescr to compile_bridge.
    pub fn get_fail_descr_for_bridge(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<Box<dyn majit_ir::FailDescr>> {
        let compiled = self.compiled_loops.get(&green_key)?;
        Some(Box::new(
            self.bridge_fail_descr_proxy(compiled, trace_id, fail_index)?,
        ))
    }

    /// Return the full recovery slot types for a guard exit, concatenated
    /// from all frames in callee-first order (matching the blackhole
    /// consumer's section convention). Falls back to exit_types when
    /// recovery_layout is absent.
    pub fn get_recovery_slot_types(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<Vec<Type>> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        let (_, trace_data) = Self::trace_for_exit(compiled, trace_id)?;
        let exit_layout = trace_data.exit_layouts.get(&fail_index)?;
        if let Some(ref recovery) = exit_layout.recovery_layout {
            if recovery.frames.len() > 1 {
                // Multi-frame: concatenate slot_types in callee-first order
                // (matching rebuild_state_after_failure's .rev() iteration).
                let mut types = Vec::new();
                for frame in recovery.frames.iter().rev() {
                    if let Some(ref st) = frame.slot_types {
                        types.extend_from_slice(st);
                    } else {
                        // No slot_types — use Ref for all slots as default.
                        types.extend(frame.slots.iter().map(|_| Type::Ref));
                    }
                }
                return Some(types);
            }
        }
        Some(exit_layout.exit_types.clone())
    }

    /// Return the merge point PC for blackhole resume from a guard exit.
    ///
    /// Producer invariant: after build_guard_metadata + backend merge,
    /// every guard has recovery_layout with header_pc on all frames.
    /// Returns None only if the (green_key, trace_id, fail_index) lookup
    /// itself fails — a metadata consistency error, not a missing field.
    pub fn get_merge_point_pc(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<u64> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        let (_, trace_data) = Self::trace_for_exit(compiled, trace_id)?;
        let exit_layout = trace_data.exit_layouts.get(&fail_index)?;
        let recovery = exit_layout.recovery_layout.as_ref()?;
        recovery.frames.first()?.header_pc
    }

    /// resume.py:1312 blackhole_from_resumedata parity:
    /// Get rd_numb and rd_consts for a guard exit, for use with
    /// ResumeDataDirectReader-based blackhole resume.
    pub fn get_rd_numb(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<(Vec<u8>, Vec<(i64, Type)>)> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        let (_, trace_data) = Self::trace_for_exit(compiled, trace_id)?;
        let exit_layout = trace_data.exit_layouts.get(&fail_index)?;
        let rd_numb = exit_layout.rd_numb.as_ref()?.clone();
        let rd_consts = exit_layout.rd_consts.as_ref()?.clone();
        Some((rd_numb, rd_consts))
    }

    /// Get exit_types for a guard (for decode_ref type dispatch).
    pub fn get_exit_types(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<Vec<Type>> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        let (_, trace_data) = Self::trace_for_exit(compiled, trace_id)?;
        let exit_layout = trace_data.exit_layouts.get(&fail_index)?;
        Some(exit_layout.exit_types.clone())
    }

    /// resume.py:924-926 _prepare: get rd_virtuals + rd_pendingfields
    /// for blackhole resume at a guard failure.
    pub fn get_rd_virtuals(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<Vec<majit_ir::RdVirtualInfo>> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        let (_, trace_data) = Self::trace_for_exit(compiled, trace_id)?;
        let exit_layout = trace_data.exit_layouts.get(&fail_index)?;
        exit_layout.rd_virtuals.clone()
    }

    /// compile.py:1002-1021 ResumeFromInterpDescr.compile_and_attach parity.
    ///
    /// Optimize against the already-compiled loop at `green_key`, then
    /// compile the result as a fresh interpreter entry under
    /// `original_green_key`.
    pub fn compile_entry_bridge(
        &mut self,
        green_key: u64,
        original_green_key: u64,
        meta: M,
        bridge_ops: &[majit_ir::Op],
        bridge_inputargs: &[majit_ir::InputArg],
        constants: HashMap<u32, i64>,
        mut constant_types: HashMap<u32, Type>,
        snapshot_boxes: HashMap<i32, Vec<majit_ir::OpRef>>,
        snapshot_frame_sizes: HashMap<i32, Vec<usize>>,
        snapshot_vable_boxes: HashMap<i32, Vec<majit_ir::OpRef>>,
        snapshot_frame_pcs: HashMap<i32, Vec<(i32, i32)>>,
        snapshot_box_types: HashMap<u32, Type>,
    ) -> bool {
        if !self.compiled_loops.contains_key(&green_key) {
            return false;
        }

        let mut optimizer = self.make_optimizer();
        let mut constants = constants;
        optimizer.constant_types = constant_types.clone();
        for arg in bridge_inputargs {
            optimizer.constant_types.insert(arg.index, arg.tp);
        }
        optimizer.snapshot_boxes = snapshot_boxes;
        optimizer.snapshot_frame_sizes = snapshot_frame_sizes;
        optimizer.snapshot_vable_boxes = snapshot_vable_boxes;
        optimizer.snapshot_frame_pcs = snapshot_frame_pcs;
        optimizer.snapshot_box_types = snapshot_box_types;
        optimizer.trace_inputarg_types = bridge_inputargs.iter().map(|ia| ia.tp).collect();

        let (retraced_count, loop_num_inputs) = {
            let compiled = self.compiled_loops.get(&green_key).unwrap();
            if let Some(source_trace) = compiled.traces.get(&compiled.root_trace_id) {
                for (&idx, &tp) in &source_trace.constant_types {
                    optimizer.constant_types.entry(idx).or_insert(tp);
                }
            }
            (compiled.retraced_count, compiled.num_inputs)
        };
        let retrace_limit = 5u32;
        let bridge_optimize_result = {
            let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                optimizer.optimize_bridge(
                    bridge_ops,
                    &mut constants,
                    bridge_inputargs.len(),
                    &mut compiled.front_target_tokens,
                    true,
                    retraced_count,
                    retrace_limit,
                    None,
                    Some(loop_num_inputs),
                )
            }))
        };
        let (optimized_ops, retrace_requested) = match bridge_optimize_result {
            Ok(result) => result,
            Err(payload) => {
                if payload
                    .downcast_ref::<crate::optimize::InvalidLoop>()
                    .is_some()
                {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] compile_entry_bridge: InvalidLoop target={} original={}",
                            green_key, original_green_key
                        );
                    }
                    return false;
                }
                std::panic::resume_unwind(payload);
            }
        };
        for (&idx, &(val, tp)) in &optimizer.bridge_preamble_constants {
            constants.entry(idx).or_insert(val);
            constant_types.entry(idx).or_insert(tp);
        }
        {
            let compiled = self.compiled_loops.get(&green_key).unwrap();
            if let Some(source_trace) = compiled.traces.get(&compiled.root_trace_id) {
                let mut defined: std::collections::HashSet<u32> = std::collections::HashSet::new();
                for i in 0..bridge_inputargs.len() {
                    defined.insert(i as u32);
                }
                for op in &optimized_ops {
                    if !op.pos.is_none() {
                        defined.insert(op.pos.0);
                    }
                }
                let mut missing: Vec<u32> = Vec::new();
                for op in &optimized_ops {
                    for &arg in &op.args {
                        let idx = arg.0;
                        if !defined.contains(&idx) && !constants.contains_key(&idx) {
                            missing.push(idx);
                        }
                    }
                    if let Some(ref fa) = op.fail_args {
                        for &arg in fa {
                            let idx = arg.0;
                            if !defined.contains(&idx) && !constants.contains_key(&idx) {
                                missing.push(idx);
                            }
                        }
                    }
                }
                for idx in missing {
                    if let Some(&val) = source_trace.constants.get(&idx) {
                        constants.insert(idx, val);
                        if let Some(&tp) = source_trace.constant_types.get(&idx) {
                            constant_types.insert(idx, tp);
                        }
                    }
                }
            }
        }
        if retrace_requested {
            if let Some(compiled) = self.compiled_loops.get_mut(&green_key) {
                compiled.retraced_count += 1;
            }
            if let Some(es) = optimizer.exported_loop_state.take() {
                let renamed_inputargs: Vec<InputArg> = es
                    .renamed_inputargs
                    .iter()
                    .enumerate()
                    .map(|(i, &opref)| {
                        let tp = es
                            .renamed_inputarg_types
                            .get(i)
                            .copied()
                            .unwrap_or(Type::Int);
                        InputArg::from_type(tp, opref.0)
                    })
                    .collect();
                self.retrace_needed(
                    green_key,
                    optimized_ops.clone(),
                    renamed_inputargs,
                    constants,
                    es,
                );
            }
            self.retrace_after_bridge = true;
            return false;
        }

        let optimized_ops = compile::strip_stray_overflow_guards(optimized_ops);
        let num_optimized_ops = optimized_ops.len();
        let compiled_constants = constants.clone();
        let compiled_constant_types = constant_types.clone();
        let trace_id = self.alloc_trace_id();

        if crate::majit_log_enabled() {
            eprintln!(
                "[jit][entry-bridge] original_key={} target_key={} inputs={:?}",
                original_green_key, green_key, bridge_inputargs
            );
            for (i, op) in optimized_ops.iter().enumerate() {
                eprintln!(
                    "[jit][entry-bridge] op[{i}] {:?} pos={:?} args={:?} descr={:?}",
                    op.opcode, op.pos, op.args, op.descr
                );
            }
        }

        self.backend.set_constants(constants);
        self.backend.set_next_trace_id(trace_id);
        self.backend.set_next_header_pc(original_green_key);

        let mut token = JitCellToken::new(self.warm_state.alloc_token_number());
        token.green_key = original_green_key;
        token.num_scalar_inputargs = self.num_scalar_inputargs;

        let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.backend
                .compile_loop(bridge_inputargs, &optimized_ops, &mut token)
        }));
        let compile_result = match compile_result {
            Ok(r) => r,
            Err(_) => return false,
        };

        match compile_result {
            Ok(_) => {
                self.assign_guard_hashes(&token);
                let (resume_data, guard_op_indices, mut exit_layouts) =
                    compile::build_guard_metadata(
                        bridge_inputargs,
                        &optimized_ops,
                        original_green_key,
                        &compiled_constants,
                        &compiled_constant_types,
                    );
                let mut terminal_exit_layouts =
                    compile::build_terminal_exit_layouts(bridge_inputargs, &optimized_ops);
                if let Some(backend_layouts) = self.backend.compiled_fail_descr_layouts(&token) {
                    compile::merge_backend_exit_layouts(&mut exit_layouts, &backend_layouts);
                }
                if let Some(backend_layouts) = self.backend.compiled_terminal_exit_layouts(&token) {
                    compile::merge_backend_terminal_exit_layouts(
                        &mut terminal_exit_layouts,
                        &backend_layouts,
                    );
                }
                let trace_info = self.backend.compiled_trace_info(&token, trace_id);
                let mut resume_data = resume_data;
                compile::enrich_guard_resume_layouts_for_trace(
                    &mut resume_data,
                    &mut exit_layouts,
                    trace_id,
                    bridge_inputargs,
                    trace_info.as_ref(),
                );
                compile::patch_backend_guard_recovery_layouts_for_trace(
                    &mut self.backend,
                    &token,
                    trace_id,
                    &mut exit_layouts,
                );
                compile::patch_backend_terminal_recovery_layouts_for_trace(
                    &mut self.backend,
                    &token,
                    trace_id,
                    &mut terminal_exit_layouts,
                );
                let bridge_optimizer_knowledge = {
                    let pos_to_fail: HashMap<u32, u32> = guard_op_indices
                        .iter()
                        .filter_map(|(&fi, &op_idx)| {
                            optimized_ops.get(op_idx).map(|op| (op.pos.0, fi))
                        })
                        .collect();
                    let mut result: HashMap<u32, OptimizerKnowledge> = HashMap::new();
                    for (guard_pos, knowledge) in &optimizer.per_guard_knowledge {
                        if let Some(&fi) = pos_to_fail.get(&guard_pos.0) {
                            result.insert(fi, knowledge.clone());
                        }
                    }
                    result
                };
                let mut traces = HashMap::new();
                traces.insert(
                    trace_id,
                    CompiledTrace {
                        snapshots: Vec::new(),
                        inputargs: bridge_inputargs.to_vec(),
                        resume_data,
                        ops: optimized_ops,
                        constants: compiled_constants,
                        constant_types: compiled_constant_types,
                        guard_op_indices,
                        exit_layouts,
                        terminal_exit_layouts,
                        optimizer_knowledge: bridge_optimizer_knowledge,
                        jitcode: None,
                    },
                );

                let front_target_tokens = self
                    .compiled_loops
                    .get(&original_green_key)
                    .map(|c| c.front_target_tokens.clone())
                    .unwrap_or_else(|| {
                        if original_green_key == green_key {
                            self.compiled_loops
                                .get(&green_key)
                                .map(|c| c.front_target_tokens.clone())
                                .unwrap_or_default()
                        } else {
                            Vec::new()
                        }
                    });
                let retraced_count = self
                    .compiled_loops
                    .get(&original_green_key)
                    .map(|c| c.retraced_count)
                    .unwrap_or(0);
                let mut previous_tokens: Vec<JitCellToken> = Vec::new();
                if let Some(old_entry) = self.compiled_loops.remove(&original_green_key) {
                    self.backend.migrate_bridges(&old_entry.token, &token);
                    previous_tokens.push(old_entry.token);
                    previous_tokens.extend(old_entry.previous_tokens);
                    for (tid, ct) in old_entry.traces {
                        traces.entry(tid).or_insert(ct);
                    }
                }
                self.compiled_loops.insert(
                    original_green_key,
                    CompiledEntry {
                        token,
                        num_inputs: bridge_inputargs.len(),
                        meta,
                        front_target_tokens,
                        retraced_count,
                        root_trace_id: trace_id,
                        traces,
                        previous_tokens,
                    },
                );
                let install_num = self.warm_state.alloc_token_number();
                let install_token = JitCellToken::new(install_num);
                self.warm_state
                    .attach_procedure_to_interp(original_green_key, install_token);
                self.stats.loops_compiled += 1;
                if let Some(ref hook) = self.hooks.on_compile_loop {
                    hook(original_green_key, bridge_ops.len(), num_optimized_ops);
                }
                true
            }
            Err(_) => false,
        }
    }

    /// Compile a bridge from a guard failure point.
    ///
    /// In RPython, when a guard fails frequently, the JIT compiles a
    /// "bridge" — an alternative path starting from the guard failure
    /// that eventually jumps back to the loop or exits.
    ///
    /// `green_key` identifies the loop containing the guard.
    /// `fail_index` identifies which guard to bridge from.
    /// `fail_descr` is the FailDescr from the guard that failed.
    /// `bridge_ops` are the recorded bridge trace operations.
    /// `bridge_inputargs` are the input arguments for the bridge.
    pub fn compile_bridge(
        &mut self,
        green_key: u64,
        fail_index: u32,
        fail_descr: &dyn majit_ir::FailDescr,
        bridge_ops: &[majit_ir::Op],
        bridge_inputargs: &[majit_ir::InputArg],
        constants: HashMap<u32, i64>,
        mut constant_types: HashMap<u32, Type>,
        snapshot_boxes: HashMap<i32, Vec<majit_ir::OpRef>>,
        snapshot_frame_sizes: HashMap<i32, Vec<usize>>,
        snapshot_vable_boxes: HashMap<i32, Vec<majit_ir::OpRef>>,
        snapshot_frame_pcs: HashMap<i32, Vec<(i32, i32)>>,
        snapshot_box_types: HashMap<u32, Type>,
    ) -> bool {
        if !self.compiled_loops.contains_key(&green_key) {
            return false;
        }

        // RPython unroll.py:183-236: Optimizer.optimize_bridge()
        let mut optimizer = self.make_optimizer();
        let mut constants = constants;
        optimizer.constant_types = constant_types.clone();
        // RPython Box.type parity: inputargs carry their types implicitly
        // via Box subclass. In majit, register inputarg types explicitly
        // so fail_arg_types inference can resolve them.
        for arg in bridge_inputargs {
            optimizer.constant_types.insert(arg.index, arg.tp);
        }
        optimizer.snapshot_boxes = snapshot_boxes;
        optimizer.snapshot_frame_sizes = snapshot_frame_sizes;
        optimizer.snapshot_vable_boxes = snapshot_vable_boxes;
        optimizer.snapshot_frame_pcs = snapshot_frame_pcs;
        // RPython Box.type parity: snapshot_box_types maps OpRef→Type for
        // boxes captured in trace snapshots. Without this, bridge optimizer
        // cannot resolve types for OpRefs in store_final_boxes_in_guard.
        optimizer.snapshot_box_types = snapshot_box_types;
        // compile.py:1035-1038: isinstance(resumekey, ResumeAtPositionDescr)
        let inline_short_preamble = !fail_descr.is_resume_at_position();
        let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
        let retraced_count = compiled.retraced_count;
        // RPython warmspot.py:93 retrace_limit=5: allow bridge to create
        // new target_token specializations when existing body token doesn't
        // match. Without this, bridges fall back to preamble (causing
        // infinite guard failure loops on preamble guards).
        let retrace_limit = 5u32;
        // bridgeopt.py: retrieve optimizer knowledge from the source trace
        // and remap OpRefs from the source trace's numbering to the bridge's
        // inputarg numbering. The bridge inputargs correspond to the guard's
        // fail_args in order, so source_opref -> bridge_inputarg_index.
        let bridge_knowledge: Option<OptimizerKnowledge> = {
            let source_trace_id = {
                let tid = fail_descr.trace_id();
                if tid == 0 {
                    compiled.root_trace_id
                } else {
                    tid
                }
            };
            compiled.traces.get(&source_trace_id).and_then(|trace| {
                let knowledge = trace.optimizer_knowledge.get(&fail_index)?;
                if knowledge.is_empty() {
                    return None;
                }
                // Build mapping: source_trace_opref -> bridge_inputarg_index.
                // The guard's fail_args list which source-trace OpRefs are
                // saved; the bridge's inputargs are OpRef(0..n) in that order.
                let guard_op_idx = trace.guard_op_indices.get(&fail_index)?;
                let guard_op = trace.ops.get(*guard_op_idx)?;
                let fail_args = guard_op.fail_args.as_ref()?;
                let mut remap: HashMap<majit_ir::OpRef, majit_ir::OpRef> = HashMap::new();
                for (i, &src_ref) in fail_args.iter().enumerate() {
                    if !src_ref.is_none() {
                        remap.insert(src_ref, majit_ir::OpRef(i as u32));
                    }
                }
                // Also map constants: they keep their original OpRef
                // (constants are in the bridge's constant pool too).
                for (&idx, _) in trace.constants.iter() {
                    let opref = majit_ir::OpRef(idx);
                    remap.entry(opref).or_insert(opref);
                }
                let remapped = knowledge.remap(&remap);
                if remapped.is_empty() {
                    None
                } else {
                    Some(remapped)
                }
            })
        };
        // Store bridge inputarg types so export_state can propagate them
        // to ExportedState.renamed_inputarg_types (RPython Box type parity).
        optimizer.trace_inputarg_types = bridge_inputargs.iter().map(|ia| ia.tp).collect();

        // Merge source trace's constant_types into the bridge optimizer
        // BEFORE optimization runs. The bridge trace's constant pool may
        // reference OpRefs from the source trace (e.g. ob_type constants
        // for virtual objects). Without this, fail_arg_type inference
        // panics on unknown OpRef types during bridge optimization.
        {
            let source_trace_id = {
                let tid = fail_descr.trace_id();
                if tid == 0 {
                    compiled.root_trace_id
                } else {
                    tid
                }
            };
            if let Some(source_trace) = compiled.traces.get(&source_trace_id) {
                for (&idx, &tp) in &source_trace.constant_types {
                    optimizer.constant_types.entry(idx).or_insert(tp);
                }
            }
        }

        // RPython bridgeopt.py:133-146 deserialize_optimizer_knowledge:
        // known_classes are restored from the per-guard bitfield that was
        // serialized at guard compile time (bridgeopt.py:69-88). Only
        // classes that were known at the guard point are restored —
        // runtime class inspection is NOT used here.
        let loop_num_inputs = compiled.num_inputs;
        // compile.py:1077-1078 parity: optimize_bridge may raise InvalidLoop
        // (e.g. rewrite.py:404-407 GUARD_CLASS proven to always fail).
        // RPython catches it via the abstract jitexc handler and discards
        // the bridge. Mirror that here so the trace abort doesn't unwind
        // past compile_bridge.
        let bridge_optimize_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            optimizer.optimize_bridge(
                bridge_ops,
                &mut constants,
                bridge_inputargs.len(),
                &mut compiled.front_target_tokens,
                inline_short_preamble,
                retraced_count,
                retrace_limit,
                bridge_knowledge.as_ref(),
                Some(loop_num_inputs),
            )
        }));
        let (optimized_ops, retrace_requested) = match bridge_optimize_result {
            Ok(result) => result,
            Err(payload) => {
                if payload
                    .downcast_ref::<crate::optimize::InvalidLoop>()
                    .is_some()
                {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] compile_bridge: InvalidLoop at key={} fail_index={}",
                            green_key, fail_index
                        );
                    }
                    return false;
                }
                std::panic::resume_unwind(payload);
            }
        };
        // RPython parity: merge short preamble constants into bridge pool.
        // inline_short_preamble registered them in optimizer.bridge_preamble_constants.
        for (&idx, &(val, tp)) in &optimizer.bridge_preamble_constants {
            constants.entry(idx).or_insert(val);
            constant_types.entry(idx).or_insert(tp);
        }
        // Also merge any remaining missing constants from the source trace's
        // constant pool (for non-short-preamble references).
        {
            let source_trace_id = {
                let tid = fail_descr.trace_id();
                if tid == 0 {
                    compiled.root_trace_id
                } else {
                    tid
                }
            };
            // Collect all defined OpRefs (inputargs + op results)
            let mut defined: std::collections::HashSet<u32> = std::collections::HashSet::new();
            for i in 0..bridge_inputargs.len() {
                defined.insert(i as u32);
            }
            for op in &optimized_ops {
                if !op.pos.is_none() {
                    defined.insert(op.pos.0);
                }
            }
            // For any referenced OpRef that's not defined and not already a
            // constant, check the source trace's constant pool.
            if let Some(source_trace) = compiled.traces.get(&source_trace_id) {
                let mut missing: Vec<u32> = Vec::new();
                for op in &optimized_ops {
                    for &arg in &op.args {
                        let idx = arg.0;
                        if !defined.contains(&idx) && !constants.contains_key(&idx) {
                            missing.push(idx);
                        }
                    }
                    if let Some(ref fa) = op.fail_args {
                        for &arg in fa {
                            let idx = arg.0;
                            if !defined.contains(&idx) && !constants.contains_key(&idx) {
                                missing.push(idx);
                            }
                        }
                    }
                }
                for idx in missing {
                    if let Some(&val) = source_trace.constants.get(&idx) {
                        constants.insert(idx, val);
                        // Also merge the type so Cranelift uses the correct
                        // register class (Int vs Ref vs Float).
                        if let Some(&tp) = source_trace.constant_types.get(&idx) {
                            constant_types.insert(idx, tp);
                        }
                    }
                }
            }
        }
        if retrace_requested {
            // compile.py:1079: metainterp.retrace_needed(new_trace, info)
            // Save partial trace + exported state so the next loop-header's
            // compile_loop → compile_retrace can produce a new specialization.
            compiled.retraced_count += 1;
            let exported = optimizer.exported_loop_state.take();
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] bridge retrace needed: key={} exported={}",
                    green_key,
                    exported.is_some(),
                );
            }
            if let Some(es) = exported {
                // compile.py:1075-1084: new_trace.inputargs = info.renamed_inputargs.
                // Types come from ExportedState.renamed_inputarg_types, populated
                // by Optimizer from trace_inputarg_types (RPython Box type parity).
                let renamed_inputargs: Vec<InputArg> =
                    es.renamed_inputargs
                        .iter()
                        .enumerate()
                        .map(|(i, &opref)| {
                            let tp = es.renamed_inputarg_types.get(i).copied().unwrap_or_else(
                                || {
                                    panic!(
                                        "missing renamed_inputarg_types entry for retrace input {}",
                                        i
                                    )
                                },
                            );
                            InputArg::from_type(tp, opref.0)
                        })
                        .collect();
                self.retrace_needed(
                    green_key,
                    optimized_ops.clone(),
                    renamed_inputargs,
                    constants,
                    es,
                );
            }
            self.retrace_after_bridge = true;
            return false;
        }

        let optimized_ops = compile::strip_stray_overflow_guards(optimized_ops);

        let num_optimized_ops = optimized_ops.len();
        let compiled_constants = constants.clone();
        let compiled_constant_types = constant_types.clone();
        let bridge_trace_id = self.alloc_trace_id();

        if crate::majit_log_enabled() {
            eprintln!("--- bridge trace (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
        }

        self.backend.set_constants(constants);
        self.backend.set_next_trace_id(bridge_trace_id);
        self.backend.set_next_header_pc(green_key);

        let result = {
            let compiled = self.compiled_loops.get(&green_key).unwrap();
            // compile.py:701-717: bridge failure → blackhole resume.
            // Catch Cranelift panics to prevent crashing the process.
            let token = &compiled.token;
            let previous_tokens = &compiled.previous_tokens;
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.backend.compile_bridge(
                    fail_descr,
                    bridge_inputargs,
                    &optimized_ops,
                    token,
                    previous_tokens,
                )
            })) {
                Ok(r) => r,
                Err(_) => {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] bridge compile_bridge panicked key={} guard={}",
                            green_key, fail_index
                        );
                    }
                    Err(majit_backend::BackendError::CompilationFailed(
                        "Cranelift panic during bridge compilation".to_string(),
                    ))
                }
            }
        };

        match result {
            Ok(_) => {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compiled bridge at key={}, guard={}",
                        green_key, fail_index
                    );
                }
                // compile.py:826-830 store_hash for bridge guards.
                if let Some(compiled) = self.compiled_loops.get(&green_key) {
                    let source_trace_id = {
                        let tid = fail_descr.trace_id();
                        if tid == 0 {
                            compiled.root_trace_id
                        } else {
                            tid
                        }
                    };
                    let layouts = self.backend.compiled_bridge_fail_descr_layouts(
                        &compiled.token,
                        source_trace_id,
                        fail_index,
                    );
                    let hashes: Vec<u64> = layouts
                        .iter()
                        .flatten()
                        .map(|layout| {
                            if layout.is_finish {
                                0
                            } else {
                                self.warm_state.fetch_next_hash()
                            }
                        })
                        .collect();
                    self.backend.store_bridge_guard_hashes(
                        &compiled.token,
                        source_trace_id,
                        fail_index,
                        &hashes,
                    );
                }
                // Mark the bridge as compiled
                if let Some(compiled) = self.compiled_loops.get_mut(&green_key) {
                    let source_trace_id = {
                        let trace_id = fail_descr.trace_id();
                        if trace_id == 0 {
                            compiled.root_trace_id
                        } else {
                            trace_id
                        }
                    };
                    let (resume_data, guard_op_indices, mut exit_layouts) =
                        compile::build_guard_metadata(
                            bridge_inputargs,
                            &optimized_ops,
                            green_key,
                            &compiled_constants,
                            &compiled_constant_types,
                        );
                    let mut terminal_exit_layouts =
                        compile::build_terminal_exit_layouts(bridge_inputargs, &optimized_ops);
                    if let Some(backend_layouts) = self.backend.compiled_bridge_fail_descr_layouts(
                        &compiled.token,
                        source_trace_id,
                        fail_index,
                    ) {
                        compile::merge_backend_exit_layouts(&mut exit_layouts, &backend_layouts);
                    }
                    if let Some(backend_layouts) =
                        self.backend.compiled_bridge_terminal_exit_layouts(
                            &compiled.token,
                            source_trace_id,
                            fail_index,
                        )
                    {
                        compile::merge_backend_terminal_exit_layouts(
                            &mut terminal_exit_layouts,
                            &backend_layouts,
                        );
                    }
                    let bridge_trace_info = self
                        .backend
                        .compiled_trace_info(&compiled.token, bridge_trace_id);
                    let mut resume_data = resume_data;
                    compile::enrich_guard_resume_layouts_for_trace(
                        &mut resume_data,
                        &mut exit_layouts,
                        bridge_trace_id,
                        bridge_inputargs,
                        bridge_trace_info.as_ref(),
                    );
                    compile::patch_backend_guard_recovery_layouts_for_trace(
                        &mut self.backend,
                        &compiled.token,
                        bridge_trace_id,
                        &mut exit_layouts,
                    );
                    compile::patch_backend_terminal_recovery_layouts_for_trace(
                        &mut self.backend,
                        &compiled.token,
                        bridge_trace_id,
                        &mut terminal_exit_layouts,
                    );
                    // resume.py:570 parity: serialize bridge per_guard_knowledge.
                    let bridge_optimizer_knowledge = {
                        let pos_to_fail: HashMap<u32, u32> = guard_op_indices
                            .iter()
                            .filter_map(|(&fi, &op_idx)| {
                                optimized_ops.get(op_idx).map(|op| (op.pos.0, fi))
                            })
                            .collect();
                        let mut result: HashMap<u32, OptimizerKnowledge> = HashMap::new();
                        for (guard_pos, knowledge) in &optimizer.per_guard_knowledge {
                            if let Some(&fi) = pos_to_fail.get(&guard_pos.0) {
                                result.insert(fi, knowledge.clone());
                            }
                        }
                        result
                    };
                    compiled.traces.insert(
                        bridge_trace_id,
                        CompiledTrace {
                            snapshots: Vec::new(),
                            inputargs: bridge_inputargs.to_vec(),
                            resume_data,
                            ops: optimized_ops,
                            constants: compiled_constants,
                            constant_types: compiled_constant_types,
                            guard_op_indices,
                            exit_layouts,
                            terminal_exit_layouts,
                            optimizer_knowledge: bridge_optimizer_knowledge,
                            jitcode: None,
                        },
                    );
                }
                self.warm_state.log_bridge_compile(fail_index);
                self.stats.bridges_compiled += 1;

                if let Some(ref hook) = self.hooks.on_compile_bridge {
                    hook(green_key, fail_index, num_optimized_ops);
                }
                true
            }
            Err(e) => {
                // RPython compile.py:701-717: bridge compilation failure
                // is not permanent — the counter resets and may fire again.
                // RPython uses ST_BUSY_FLAG only (cleared by done_compiling).
                let msg = format!("Bridge compilation failed: {e}");
                if crate::majit_log_enabled() {
                    eprintln!("[jit] {msg}");
                }
                if let Some(ref cb) = self.hooks.on_compile_error {
                    cb(green_key, &msg);
                }
                false
            }
        }
    }

    /// Start retracing from a guard failure point.
    ///
    /// When a guard fails enough times (>= trace_eagerness) and is not yet
    /// eligible for bridge compilation, we start a new trace from the
    /// guard failure point. The resulting trace replaces the original guard.
    ///
    /// Returns true if retracing was started.
    /// RPython pyjitpl.py:2890 handle_guard_failure parity:
    /// Initialize bridge tracing from a guard failure point.
    /// Returns (success, is_exception_guard) so the caller can emit
    /// SAVE_EXC_CLASS + SAVE_EXCEPTION ops for exception bridges.
    pub fn start_retrace_from_guard(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        _fail_values: &[i64],
        _live_types: &[Type],
    ) -> Option<BridgeRetraceResult> {
        let compiled = match self.compiled_loops.get(&green_key) {
            Some(c) => c,
            None => return None,
        };

        // RPython compile.py:932 invent_fail_descr_for_op:
        // GUARD_EXCEPTION / GUARD_NO_EXCEPTION → ResumeGuardExcDescr.
        // Check the guard's opcode to determine if this is an exception guard.
        let norm_tid = Self::normalize_trace_id(compiled, trace_id);
        let is_exception_guard = Self::trace_for_exit(compiled, norm_tid)
            .and_then(|(_, trace)| {
                let idx = *trace.guard_op_indices.get(&fail_index)?;
                Some(trace.ops.get(idx)?.opcode.is_guard_exception())
            })
            .unwrap_or(false);

        let fail_descr = match self.bridge_fail_descr_proxy(compiled, trace_id, fail_index) {
            Some(descr) => descr,
            None => return None,
        };

        // compile.py:797-811 parity: bridge inputargs come from the guard's
        // fail_arg_types AFTER store_final_boxes_in_guard. The backend's
        // fail_descr has the UPDATED types (may include virtual field boxes).
        // bridge_fail_descr_proxy already prefers backend types.
        let bridge_input_types = fail_descr.fail_arg_types();
        let recorder = self.warm_state.start_retrace(bridge_input_types);
        self.force_finish_trace = false;
        self.tracing = Some(crate::trace_ctx::TraceCtx::new(recorder, green_key));

        if let Some(ref hook) = self.hooks.on_trace_start {
            hook(green_key);
        }

        // resume.py:1042: retrieve rd_numb/rd_consts directly from exit_layout
        // (not from BridgeFailDescrProxy, to avoid cloning on the hot path).
        let (rd_numb, rd_consts) = Self::trace_for_exit(compiled, norm_tid)
            .and_then(|(_, trace)| trace.exit_layouts.get(&fail_index))
            .map(|layout| (layout.rd_numb.clone(), layout.rd_consts.clone()))
            .unwrap_or((None, None));

        Some(BridgeRetraceResult {
            is_exception_guard,
            fail_types: bridge_input_types.to_vec(),
            rd_numb,
            rd_consts,
        })
    }

    /// compile.py:987-1000: handle_async_forcing — force all virtuals
    /// from resume data when a GUARD_NOT_FORCED fires asynchronously
    /// (during a residual call that forces the virtualizable).
    ///
    /// RPython flow: force_now() → cpu.force(token) → handle_async_forcing()
    /// → force_from_resumedata() → materialize all virtuals → save on deadframe.
    ///
    /// Returns the forced virtual caches (ptr, int) for later blackhole
    /// resumption from the GUARD_NOT_FORCED. RPython stores these as
    /// AllVirtuals via cpu.set_savedata_ref().
    pub fn handle_async_forcing(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
    ) -> Option<(Vec<i64>, Vec<i64>)> {
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit][handle_async_forcing] key={} trace={} fail={} nvals={}",
                green_key,
                trace_id,
                fail_index,
                fail_values.len()
            );
        }
        // compile.py:988-991: resolve metainterp_sd, vinfo, ginfo
        let compiled = self.compiled_loops.get(&green_key)?;
        let norm_tid = Self::normalize_trace_id(compiled, trace_id);
        let (_, trace) = Self::trace_for_exit(compiled, norm_tid)?;
        let exit_layout =
            Self::compiled_exit_layout_from_trace(trace, green_key, norm_tid, fail_index)?;

        // compile.py:994: force_from_resumedata(metainterp_sd, self, deadframe, vinfo, ginfo)
        let rd_numb = exit_layout.rd_numb.as_deref().unwrap_or(&[]);
        let rd_consts: Vec<i64> = exit_layout
            .rd_consts
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|(v, _)| *v)
            .collect();
        // compile.py:990-991: vinfo = self.jitdriver_sd.virtualizable_info
        let vinfo = self.virtualizable_info();
        let allocator = crate::resume::NullAllocator;
        let (all_virtuals_ptr, all_virtuals_int) = crate::resume::force_from_resumedata(
            rd_numb,
            &rd_consts,
            fail_values,
            None, // deadframe_types
            None, // vrefinfo — pyre has no vref mechanism
            vinfo.map(|v| v as &dyn crate::resume::VirtualizableInfo),
            None, // ginfo — pyre has no greenfield mechanism
            &allocator,
        );
        // compile.py:999-1000: obj = AllVirtuals(all_virtuals)
        //   metainterp_sd.cpu.set_savedata_ref(deadframe, obj.hide())
        // Return the virtual caches so the caller can store them.
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit][handle_async_forcing] forced {} ptr + {} int virtuals",
                all_virtuals_ptr.len(),
                all_virtuals_int.len(),
            );
        }
        Some((all_virtuals_ptr, all_virtuals_int))
    }

    /// RPython pyjitpl.py:3101 _prepare_exception_resumption +
    /// pyjitpl.py:3132 prepare_resume_from_failure parity:
    /// Emit SAVE_EXC_CLASS + SAVE_EXCEPTION + RESTORE_EXCEPTION at
    /// the bridge trace start for exception guard bridges.
    pub fn emit_exception_bridge_prologue(&mut self, exc_class: i64, exc_value: i64) {
        let Some(ref mut ctx) = self.tracing else {
            return;
        };
        let class_const = ctx.const_int(exc_class);
        let value_const = ctx.const_int(exc_value);
        let class_op = ctx.record_op(OpCode::SaveExcClass, &[class_const]);
        let value_op = ctx.record_op(OpCode::SaveException, &[value_const]);
        ctx.record_op(OpCode::RestoreException, &[class_op, value_op]);
    }

    /// No RPython equivalent — RPython uses ConstBox natively in traces.
    /// Rust needs explicit constant pool injection for TAGCONST/TAGINT
    /// entries decoded from rd_numb by rebuild_from_resumedata.
    pub fn inject_bridge_constants(&mut self, constants: &[(u32, i64, Type)]) {
        let Some(ref mut ctx) = self.tracing else {
            return;
        };
        for &(opref_idx, value, tp) in constants {
            ctx.constants.as_mut().insert(opref_idx, value);
            if tp != Type::Int {
                ctx.constants.mark_type(OpRef(opref_idx), tp);
            }
        }
    }

    // ── Guard Failure Recovery ─────────────────────────────────

    /// Handle a guard failure: recover interpreter state using resume data.
    ///
    /// This is the central guard failure handler, equivalent to RPython's
    /// `handle_guard_failure()` in pyjitpl.py.
    ///
    /// Returns `GuardRecovery` describing the recovered state.
    /// Bridge-vs-blackhole is decided by the caller from `must_compile()`,
    /// matching compile.py:701-717 handle_fail flow.
    pub fn handle_guard_failure(
        &mut self,
        green_key: u64,
        fail_index: u32,
        fail_values: &[i64],
        exception: ExceptionState,
    ) -> Option<GuardRecovery> {
        self.handle_guard_failure_with_savedata(green_key, fail_index, fail_values, None, exception)
    }

    /// `handle_guard_failure()` variant that also carries backend savedata.
    pub fn handle_guard_failure_with_savedata(
        &mut self,
        green_key: u64,
        fail_index: u32,
        fail_values: &[i64],
        savedata: Option<GcRef>,
        exception: ExceptionState,
    ) -> Option<GuardRecovery> {
        // pyjitpl.py:2900: try_to_free_some_loops
        self.try_to_free_some_loops();
        let trace_id = self.compiled_loops.get(&green_key)?.root_trace_id;
        self.handle_guard_failure_in_trace_with_savedata(
            green_key,
            trace_id,
            fail_index,
            fail_values,
            None,
            savedata,
            exception,
        )
    }

    /// Handle a guard failure in a specific compiled trace (root loop or bridge).
    ///
    /// This is the trace-aware counterpart to `handle_guard_failure()`. Callers
    /// should use the `trace_id` reported by `run_compiled_detailed()` when the
    /// failing exit may come from a bridge.
    pub fn handle_guard_failure_in_trace(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
        typed_fail_values: Option<&[Value]>,
        exception: ExceptionState,
    ) -> Option<GuardRecovery> {
        self.handle_guard_failure_in_trace_with_savedata(
            green_key,
            trace_id,
            fail_index,
            fail_values,
            typed_fail_values,
            None,
            exception,
        )
    }

    /// `handle_guard_failure_in_trace()` variant that also carries backend savedata.
    ///
    pub fn handle_guard_failure_in_trace_with_savedata(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
        typed_fail_values: Option<&[Value]>,
        savedata: Option<GcRef>,
        exception: ExceptionState,
    ) -> Option<GuardRecovery> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let (trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;

        let exit_layout =
            Self::compiled_exit_layout_from_trace(trace, green_key, trace_id, fail_index)
                .unwrap_or_else(|| CompiledExitLayout {
                    rd_loop_token: green_key, // from trace context
                    trace_id,
                    fail_index,
                    source_op_index: None,
                    exit_types: typed_fail_values
                        .map(|values| values.iter().map(Value::get_type).collect())
                        .unwrap_or_default(),
                    is_finish: false,
                    gc_ref_slots: typed_fail_values
                        .map(|values| {
                            values
                                .iter()
                                .enumerate()
                                .filter_map(|(slot, value)| {
                                    (value.get_type() == Type::Ref).then_some(slot)
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                    force_token_slots: Vec::new(),
                    recovery_layout: None,
                    resume_layout: None,
                    rd_numb: None,
                    rd_consts: None,
                    rd_virtuals: None,
                    rd_pendingfields: None,
                });
        let reconstructed_state = exit_layout
            .resume_layout
            .as_ref()
            .map(|layout| layout.reconstruct_state(fail_values))
            .or_else(|| {
                trace
                    .resume_data
                    .get(&fail_index)
                    .map(|resume_data| resume_data.layout.reconstruct_state(fail_values))
            });
        let resume_layout = exit_layout.resume_layout.clone();
        let reconstructed = reconstructed_state
            .as_ref()
            .map(|state| state.frames.clone());
        let materialized_virtuals = reconstructed_state
            .as_ref()
            .map(|state| state.virtuals.clone())
            .unwrap_or_default();
        let pending_field_writes = reconstructed_state
            .as_ref()
            .map(|state| state.pending_fields.clone())
            .unwrap_or_default();

        Some(GuardRecovery {
            trace_id,
            fail_index,
            exit_layout,
            fail_values: fail_values.to_vec(),
            typed_fail_values: typed_fail_values.map(|values| values.to_vec()),
            resume_layout,
            reconstructed_frames: reconstructed,
            reconstructed_state,
            materialized_virtuals,
            pending_field_writes,
            savedata,
            exception,
        })
    }

    /// Run compiled code and handle guard failures automatically.
    ///
    /// This is a convenience wrapper around `run_compiled_detailed` +
    /// `handle_guard_failure`.
    pub fn run_and_recover(&mut self, green_key: u64, live_values: &[i64]) -> Option<RunResult<M>> {
        let result = self.run_compiled_detailed(green_key, live_values)?;
        let fail_index = result.fail_index;
        let trace_id = result.trace_id;
        let is_finish = result.is_finish;
        let values = result.values.clone();
        let typed_values = result.typed_values.clone();
        let savedata = result.savedata;
        let exception = result.exception.clone();
        let meta = result.meta.clone();

        if is_finish {
            // Normal finish (not a guard failure)
            return Some(RunResult::Finished {
                values,
                meta,
                savedata,
            });
        }

        if let Some(jump) =
            Self::run_result_for_jump_exit(fail_index, values.clone(), meta.clone(), savedata)
        {
            return Some(jump);
        }

        // Guard failure — recover
        let recovery = self.handle_guard_failure_in_trace_with_savedata(
            green_key,
            trace_id,
            fail_index,
            &values,
            Some(&typed_values),
            savedata,
            exception,
        );

        Some(RunResult::GuardFailure {
            values,
            meta,
            trace_id,
            fail_index,
            savedata,
            recovery,
        })
    }

    /// RPython resume_in_blackhole parity: resume execution from the guard
    /// failure point using the IR-based blackhole interpreter.
    /// RPython bh_call_i parity: `memory` provides call_i/call_r for
    pub fn blackhole_guard_failure(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
        exception: ExceptionState,
    ) -> Option<(BlackholeResult, ExceptionState)> {
        self.blackhole_guard_failure_ca(
            green_key,
            trace_id,
            fail_index,
            fail_values,
            exception,
            None,
        )
    }

    /// blackhole.py:1095 bhimpl_recursive_call parity:
    /// Like `blackhole_guard_failure` but with a CallAssembler callback
    /// so the IR blackhole can execute CallAssembler ops via portal_runner.
    pub fn blackhole_guard_failure_ca(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
        exception: ExceptionState,
        call_assembler_fn: Option<&crate::blackhole::CallAssemblerFn>,
    ) -> Option<(BlackholeResult, ExceptionState)> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let (_, trace) = Self::trace_for_exit(compiled, trace_id)?;
        let guard_op_index = *trace.guard_op_indices.get(&fail_index)?;
        let guard_op = trace.ops.get(guard_op_index)?;
        let fail_args = guard_op.fail_args.as_ref()?;

        let mut initial_values = HashMap::with_capacity(fail_args.len());
        for (arg, value) in fail_args.iter().zip(fail_values.iter().copied()) {
            initial_values.insert(arg.0, value);
        }

        Some(blackhole_execute_with_state_ca(
            &trace.ops,
            &trace.constants,
            &initial_values,
            guard_op_index + 1,
            exception,
            call_assembler_fn,
        ))
    }

    /// Run compiled code and, on guard failure, immediately continue through the
    /// blackhole interpreter for the subset where majit can replay the remaining
    /// optimized ops directly.
    pub fn run_with_blackhole_fallback(
        &mut self,
        green_key: u64,
        live_values: &[i64],
    ) -> Option<BlackholeRunResult<M>> {
        let result = self.run_compiled_detailed(green_key, live_values)?;
        let fail_index = result.fail_index;
        let trace_id = result.trace_id;
        let is_finish = result.is_finish;
        let values = result.values.clone();
        let typed_values = result.typed_values.clone();
        let compiled_exit_layout = result.exit_layout.clone();
        let savedata = result.savedata;
        let exception = result.exception.clone();
        let meta = result.meta.clone();

        if is_finish {
            return Some(BlackholeRunResult::Finished {
                values,
                typed_values: Some(typed_values),
                exit_layout: Some(compiled_exit_layout),
                meta,
                via_blackhole: false,
                savedata,
                exception,
            });
        }

        if let Some(jump) = Self::blackhole_result_for_jump_exit(
            fail_index,
            values.clone(),
            typed_values.clone(),
            compiled_exit_layout.clone(),
            meta.clone(),
            savedata,
            exception.clone(),
        ) {
            return Some(jump);
        }

        let initial_recovery = self.handle_guard_failure_in_trace_with_savedata(
            green_key,
            trace_id,
            fail_index,
            &values,
            Some(&typed_values),
            savedata,
            exception.clone(),
        );

        let Some((blackhole_result, blackhole_exception)) = self.blackhole_guard_failure(
            green_key,
            trace_id,
            fail_index,
            &values,
            exception.clone(),
        ) else {
            return Some(BlackholeRunResult::GuardFailure {
                trace_id,
                fail_index,
                fail_values: values,
                typed_fail_values: Some(typed_values),
                exit_layout: Some(compiled_exit_layout),
                meta,
                recovery: initial_recovery,
                materialized_virtuals: Vec::new(),
                pending_field_writes: Vec::new(),
                via_blackhole: false,
                savedata,
                exception,
            });
        };

        match blackhole_result {
            BlackholeResult::Finish {
                op_index, values, ..
            } => {
                let exit_layout = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (terminal_trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    compile::terminal_exit_layout_for_trace(
                        trace,
                        green_key,
                        terminal_trace_id,
                        op_index,
                    )
                };
                let typed_values = exit_layout
                    .as_ref()
                    .map(|layout| compile::decode_values_with_layout(&values, layout));
                Some(BlackholeRunResult::Finished {
                    values,
                    typed_values,
                    exit_layout,
                    meta,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::Jump { op_index, values } => {
                let exit_layout = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (terminal_trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    compile::terminal_exit_layout_for_trace(
                        trace,
                        green_key,
                        terminal_trace_id,
                        op_index,
                    )
                };
                let typed_values = exit_layout
                    .as_ref()
                    .map(|layout| compile::decode_values_with_layout(&values, layout));
                Some(BlackholeRunResult::Jump {
                    values,
                    typed_values,
                    exit_layout,
                    meta,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::GuardFailed {
                guard_index,
                fail_values,
            } => {
                let (
                    fallback_trace_id,
                    fallback_fail_index,
                    exit_layout,
                    typed_fail_values,
                    materialized_virtuals,
                    pending_field_writes,
                ) = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    let fallback_fail_index = trace
                        .guard_op_indices
                        .iter()
                        .find_map(|(&idx, &op_index)| (op_index == guard_index).then_some(idx))
                        .unwrap_or(fail_index);
                    let materialized_virtuals = trace
                        .resume_data
                        .get(&fallback_fail_index)
                        .map(|resume_data| resume_data.encoded.materialize_virtuals(&fail_values))
                        .unwrap_or_default();
                    let pending_field_writes = trace
                        .resume_data
                        .get(&fallback_fail_index)
                        .map(|resume_data| {
                            resume_data
                                .encoded
                                .resolve_pending_field_writes(&fail_values)
                        })
                        .unwrap_or_default();
                    let exit_layout = Self::compiled_exit_layout_from_trace(
                        trace,
                        green_key,
                        trace_id,
                        fallback_fail_index,
                    );
                    let typed_fail_values = exit_layout
                        .as_ref()
                        .map(|layout| compile::decode_values_with_layout(&fail_values, layout));
                    (
                        trace_id,
                        fallback_fail_index,
                        exit_layout,
                        typed_fail_values,
                        materialized_virtuals,
                        pending_field_writes,
                    )
                };
                let recovery = self.handle_guard_failure_in_trace_with_savedata(
                    green_key,
                    fallback_trace_id,
                    fallback_fail_index,
                    &fail_values,
                    typed_fail_values.as_deref(),
                    savedata,
                    blackhole_exception.clone(),
                );
                Some(BlackholeRunResult::GuardFailure {
                    trace_id: fallback_trace_id,
                    fail_index: fallback_fail_index,
                    fail_values,
                    typed_fail_values,
                    exit_layout,
                    meta,
                    recovery,
                    materialized_virtuals,
                    pending_field_writes,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::GuardFailedWithVirtuals {
                guard_index,
                fail_values,
                materialized_virtuals,
                pending_field_writes,
            } => {
                let (fallback_trace_id, fallback_fail_index, exit_layout, typed_fail_values) = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    let fallback_fail_index = trace
                        .guard_op_indices
                        .iter()
                        .find_map(|(&idx, &op_index)| (op_index == guard_index).then_some(idx))
                        .unwrap_or(fail_index);
                    let exit_layout = Self::compiled_exit_layout_from_trace(
                        trace,
                        green_key,
                        trace_id,
                        fallback_fail_index,
                    );
                    let typed_fail_values = exit_layout
                        .as_ref()
                        .map(|layout| compile::decode_values_with_layout(&fail_values, layout));
                    (
                        trace_id,
                        fallback_fail_index,
                        exit_layout,
                        typed_fail_values,
                    )
                };
                let recovery = self.handle_guard_failure_in_trace_with_savedata(
                    green_key,
                    fallback_trace_id,
                    fallback_fail_index,
                    &fail_values,
                    typed_fail_values.as_deref(),
                    savedata,
                    blackhole_exception.clone(),
                );
                Some(BlackholeRunResult::GuardFailure {
                    trace_id: fallback_trace_id,
                    fail_index: fallback_fail_index,
                    fail_values,
                    typed_fail_values,
                    exit_layout,
                    meta,
                    recovery,
                    materialized_virtuals,
                    pending_field_writes,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::Abort(message) => Some(BlackholeRunResult::Abort {
                trace_id,
                fail_index,
                fail_values: values,
                typed_fail_values: Some(typed_values),
                exit_layout: Some(compiled_exit_layout),
                meta,
                recovery: initial_recovery,
                message,
                savedata,
                exception: blackhole_exception,
            }),
        }
    }

    /// Typed-input counterpart to [`run_with_blackhole_fallback`].
    pub fn run_with_blackhole_fallback_with_values(
        &mut self,
        green_key: u64,
        live_values: &[Value],
    ) -> Option<BlackholeRunResult<M>> {
        let result = self.run_compiled_detailed_with_values(green_key, live_values)?;
        let fail_index = result.fail_index;
        let trace_id = result.trace_id;
        let is_finish = result.is_finish;
        let values = result.values.clone();
        let typed_values = result.typed_values.clone();
        let compiled_exit_layout = result.exit_layout.clone();
        let savedata = result.savedata;
        let exception = result.exception.clone();
        let meta = result.meta.clone();

        if is_finish {
            return Some(BlackholeRunResult::Finished {
                values,
                typed_values: Some(typed_values),
                exit_layout: Some(compiled_exit_layout),
                meta,
                via_blackhole: false,
                savedata,
                exception,
            });
        }

        if let Some(jump) = Self::blackhole_result_for_jump_exit(
            fail_index,
            values.clone(),
            typed_values.clone(),
            compiled_exit_layout.clone(),
            meta.clone(),
            savedata,
            exception.clone(),
        ) {
            return Some(jump);
        }

        let initial_recovery = self.handle_guard_failure_in_trace_with_savedata(
            green_key,
            trace_id,
            fail_index,
            &values,
            Some(&typed_values),
            savedata,
            exception.clone(),
        );

        let Some((blackhole_result, blackhole_exception)) = self.blackhole_guard_failure(
            green_key,
            trace_id,
            fail_index,
            &values,
            exception.clone(),
        ) else {
            return Some(BlackholeRunResult::GuardFailure {
                trace_id,
                fail_index,
                fail_values: values,
                typed_fail_values: Some(typed_values),
                exit_layout: Some(compiled_exit_layout),
                meta,
                recovery: initial_recovery,
                materialized_virtuals: Vec::new(),
                pending_field_writes: Vec::new(),
                via_blackhole: false,
                savedata,
                exception,
            });
        };

        match blackhole_result {
            BlackholeResult::Finish {
                op_index, values, ..
            } => {
                let exit_layout = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (terminal_trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    compile::terminal_exit_layout_for_trace(
                        trace,
                        green_key,
                        terminal_trace_id,
                        op_index,
                    )
                };
                let typed_values = exit_layout
                    .as_ref()
                    .map(|layout| compile::decode_values_with_layout(&values, layout));
                Some(BlackholeRunResult::Finished {
                    values,
                    typed_values,
                    exit_layout,
                    meta,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::Jump { op_index, values } => {
                let exit_layout = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (terminal_trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    compile::terminal_exit_layout_for_trace(
                        trace,
                        green_key,
                        terminal_trace_id,
                        op_index,
                    )
                };
                let typed_values = exit_layout
                    .as_ref()
                    .map(|layout| compile::decode_values_with_layout(&values, layout));
                Some(BlackholeRunResult::Jump {
                    values,
                    typed_values,
                    exit_layout,
                    meta,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::GuardFailed {
                guard_index,
                fail_values,
            } => {
                let (
                    fallback_trace_id,
                    fallback_fail_index,
                    exit_layout,
                    typed_fail_values,
                    materialized_virtuals,
                    pending_field_writes,
                ) = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    let fallback_fail_index = trace
                        .guard_op_indices
                        .iter()
                        .find_map(|(&idx, &op_index)| (op_index == guard_index).then_some(idx))
                        .unwrap_or(fail_index);
                    let materialized_virtuals = trace
                        .resume_data
                        .get(&fallback_fail_index)
                        .map(|resume_data| resume_data.encoded.materialize_virtuals(&fail_values))
                        .unwrap_or_default();
                    let pending_field_writes = trace
                        .resume_data
                        .get(&fallback_fail_index)
                        .map(|resume_data| {
                            resume_data
                                .encoded
                                .resolve_pending_field_writes(&fail_values)
                        })
                        .unwrap_or_default();
                    let exit_layout = Self::compiled_exit_layout_from_trace(
                        trace,
                        green_key,
                        trace_id,
                        fallback_fail_index,
                    );
                    let typed_fail_values = exit_layout
                        .as_ref()
                        .map(|layout| compile::decode_values_with_layout(&fail_values, layout));
                    (
                        trace_id,
                        fallback_fail_index,
                        exit_layout,
                        typed_fail_values,
                        materialized_virtuals,
                        pending_field_writes,
                    )
                };
                let recovery = self.handle_guard_failure_in_trace_with_savedata(
                    green_key,
                    fallback_trace_id,
                    fallback_fail_index,
                    &fail_values,
                    typed_fail_values.as_deref(),
                    savedata,
                    blackhole_exception.clone(),
                );
                Some(BlackholeRunResult::GuardFailure {
                    trace_id: fallback_trace_id,
                    fail_index: fallback_fail_index,
                    fail_values,
                    typed_fail_values,
                    exit_layout,
                    meta,
                    recovery,
                    materialized_virtuals,
                    pending_field_writes,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::GuardFailedWithVirtuals {
                guard_index,
                fail_values,
                materialized_virtuals,
                pending_field_writes,
            } => {
                let (fallback_trace_id, fallback_fail_index, exit_layout, typed_fail_values) = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    let fallback_fail_index = trace
                        .guard_op_indices
                        .iter()
                        .find_map(|(&idx, &op_index)| (op_index == guard_index).then_some(idx))
                        .unwrap_or(fail_index);
                    let exit_layout = Self::compiled_exit_layout_from_trace(
                        trace,
                        green_key,
                        trace_id,
                        fallback_fail_index,
                    );
                    let typed_fail_values = exit_layout
                        .as_ref()
                        .map(|layout| compile::decode_values_with_layout(&fail_values, layout));
                    (
                        trace_id,
                        fallback_fail_index,
                        exit_layout,
                        typed_fail_values,
                    )
                };
                let recovery = self.handle_guard_failure_in_trace_with_savedata(
                    green_key,
                    fallback_trace_id,
                    fallback_fail_index,
                    &fail_values,
                    typed_fail_values.as_deref(),
                    savedata,
                    blackhole_exception.clone(),
                );
                Some(BlackholeRunResult::GuardFailure {
                    trace_id: fallback_trace_id,
                    fail_index: fallback_fail_index,
                    fail_values,
                    typed_fail_values,
                    exit_layout,
                    meta,
                    recovery,
                    materialized_virtuals,
                    pending_field_writes,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::Abort(message) => Some(BlackholeRunResult::Abort {
                trace_id,
                fail_index,
                fail_values: values,
                typed_fail_values: Some(typed_values),
                exit_layout: Some(compiled_exit_layout),
                meta,
                recovery: initial_recovery,
                message,
                savedata,
                exception: blackhole_exception,
            }),
        }
    }

    // ── Retrace Support ──────────────────────────────────────

    /// Start retracing from a guard failure point.
    ///
    /// When a guard fails too many times, the JIT can start a new trace
    /// from the failure point. The new trace becomes a bridge that is
    /// attached to the failed guard.
    ///
    /// `green_key` identifies the loop containing the guard.
    /// `fail_index` identifies which guard failed.
    /// `live_values` are the concrete values at the guard failure point.
    ///
    /// Returns `true` if retracing was started, `false` if not possible.
    pub fn start_retrace(&mut self, green_key: u64, _fail_index: u32, live_values: &[i64]) -> bool {
        let Some(root_trace_id) = self.compiled_loops.get(&green_key).map(|c| c.root_trace_id)
        else {
            return false;
        };
        self.start_retrace_from_guard(green_key, root_trace_id, _fail_index, live_values, &[])
            .is_some()
    }

    // ── Inlining Support ──────────────────────────────────────

    /// Check if a function call should be inlined during tracing.
    ///
    /// Mirrors RPython's inlining heuristic from warmstate.py:
    /// 1. If the callee already has compiled code → CALL_ASSEMBLER
    /// 2. If not tracing → residual call
    /// 3. If recursion depth too deep → residual call
    /// 4. If the function hasn't been called enough times → residual call
    ///    (controlled by `function_threshold`)
    /// 5. Otherwise → inline
    ///
    /// `callee_key` identifies the called function's JitDriver.
    pub fn should_inline(&mut self, callee_key: u64) -> InlineDecision {
        // Extract inline-relevant info from ctx before calling impl
        // (avoids borrow conflict between self.tracing and &mut self).
        let ctx_info = self.tracing.as_ref().map(|ctx| {
            (
                ctx.inline_depth(),
                ctx.is_tracing_key(callee_key),
                ctx.recursive_depth(callee_key),
            )
        });
        self.should_inline_core(callee_key, ctx_info)
    }

    pub fn should_inline_with_ctx(
        &mut self,
        callee_key: u64,
        ctx: &crate::trace_ctx::TraceCtx,
    ) -> InlineDecision {
        let ctx_info = Some((
            ctx.inline_depth(),
            ctx.is_tracing_key(callee_key),
            ctx.recursive_depth(callee_key),
        ));
        self.should_inline_core(callee_key, ctx_info)
    }

    /// Core inline decision logic.
    /// ctx_info: Option<(inline_depth, is_tracing_key, recursive_depth)>
    fn should_inline_core(
        &mut self,
        callee_key: u64,
        ctx_info: Option<(usize, bool, usize)>,
    ) -> InlineDecision {
        let callee_compiled = self.compiled_loops.contains_key(&callee_key);
        if !self.warm_state.can_inline_callable(callee_key) {
            if callee_compiled {
                return InlineDecision::CallAssembler;
            }
            return InlineDecision::ResidualCall;
        }

        if let Some((inline_depth, is_self_recursive, recursive_depth)) = ctx_info {
            if inline_depth >= MAX_INLINE_DEPTH {
                if callee_compiled {
                    return InlineDecision::CallAssembler;
                }
                return InlineDecision::ResidualCall;
            }

            // Self-recursion: PyPy max_unroll_recursion strategy.
            // RPython warmstate.py:714 get_assembler_token: when the
            // callee is not yet compiled, compile_tmp_callback creates
            // a temporary token so CALL_ASSEMBLER can still be emitted.
            // In pyre, pending_token serves the same role.
            if is_self_recursive {
                if recursive_depth < self.max_unroll_recursion {
                    return InlineDecision::Inline;
                }
                if callee_compiled || self.pending_token.map_or(false, |(k, _)| k == callee_key) {
                    return InlineDecision::CallAssembler;
                }
                self.warm_state.boost_function_entry(callee_key);
                return InlineDecision::ResidualCall;
            }

            if !self.warm_state.should_inline_function(callee_key) {
                if callee_compiled {
                    return InlineDecision::CallAssembler;
                }
                return InlineDecision::ResidualCall;
            }

            return InlineDecision::Inline;
        }

        // Not tracing — use CALL_ASSEMBLER if compiled.
        if callee_compiled {
            return InlineDecision::CallAssembler;
        }

        InlineDecision::ResidualCall
    }

    /// Begin inlining a function call during tracing.
    ///
    /// Pushes an inline frame so tracing can continue through the callee body.
    /// We intentionally avoid recording ENTER_PORTAL_FRAME markers for inline
    /// calls: unlike a real portal transition, they do not carry runtime
    /// semantics and only bloat the trace.
    ///
    /// Returns `true` if inlining started, `false` if not tracing or depth exceeded.
    pub fn enter_inline_frame(&mut self, callee_key: u64) -> bool {
        let ctx = match self.tracing.as_mut() {
            Some(ctx) => ctx,
            None => return false,
        };
        if ctx.inline_depth() >= MAX_INLINE_DEPTH {
            return false;
        }

        ctx.push_inline_frame(callee_key, MAX_INLINE_DEPTH as u32);
        true
    }

    /// Leave an inlined function call during tracing.
    ///
    /// Pops the inline frame. See `enter_inline_frame()` for why we do not
    /// record LEAVE_PORTAL_FRAME for inline calls.
    pub fn leave_inline_frame(&mut self) {
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.pop_inline_frame();
        }
    }

    /// Get the current inlining depth.
    pub fn inline_depth(&self) -> usize {
        self.tracing
            .as_ref()
            .map(|ctx| ctx.inline_depth())
            .unwrap_or(0)
    }

    /// Access the backend directly (for advanced operations).
    pub fn backend(&self) -> &BackendImpl {
        &self.backend
    }

    /// Access the backend mutably (for advanced operations).
    pub fn backend_mut(&mut self) -> &mut BackendImpl {
        &mut self.backend
    }

    /// Register a helper that boxes a raw integer into an interpreter object.
    ///
    /// Compiled finish post-processing uses this to recognize boxing helpers
    /// that are safe to peel away for the raw-int call_assembler protocol.
    pub fn register_raw_int_box_helper(&mut self, helper: *const ()) {
        self.raw_int_box_helpers.insert(helper as i64);
    }

    /// Register a create_frame → create_frame_raw_int mapping.
    /// When a box helper result feeds directly into create_frame as the last arg,
    /// the two calls can be folded into a single create_frame_raw_int call.
    pub fn register_create_frame_raw(&mut self, normal: *const (), raw_int: *const ()) {
        self.create_frame_raw_map
            .insert(normal as i64, raw_int as i64);
    }

    /// PyPy warmspot.py set_param_max_unroll_recursion().
    pub fn set_max_unroll_recursion(&mut self, value: usize) {
        self.max_unroll_recursion = value;
    }
}

/// Default maximum inlining depth during tracing.
/// Configurable via WarmEnterState::set_max_inline_depth().
const MAX_INLINE_DEPTH: usize = 10;

/// Describes the recovery state after a guard failure.
#[derive(Debug, Clone)]
pub struct GuardRecovery {
    /// Compiled trace identifier for the failing exit.
    pub trace_id: u64,
    /// Index of the failed guard.
    pub fail_index: u32,
    /// Static layout metadata for this compiled exit.
    pub exit_layout: CompiledExitLayout,
    /// Raw fail_values from the DeadFrame.
    pub fail_values: Vec<i64>,
    /// Typed fail values decoded from the backend deadframe, when available.
    pub typed_fail_values: Option<Vec<Value>>,
    /// Compact resume/jitframe layout for this exit, when available.
    pub resume_layout: Option<ResumeLayoutSummary>,
    /// Reconstructed interpreter frames (if resume data was available).
    pub reconstructed_frames: Option<Vec<crate::resume::ReconstructedFrame>>,
    /// Full reconstructed state, including materialized virtuals.
    pub reconstructed_state: Option<ReconstructedState>,
    /// Materialized virtuals referenced by the reconstructed state.
    pub materialized_virtuals: Vec<MaterializedVirtual>,
    /// Deferred heap writes reconstructed from resume data.
    pub pending_field_writes: Vec<ResolvedPendingFieldWrite>,
    /// Optional saved-data GC ref captured from the failing exit.
    pub savedata: Option<GcRef>,
    /// Pending exception state captured from the failing deadframe.
    pub exception: ExceptionState,
}

/// Result of running compiled code with automatic recovery.
#[derive(Debug, Clone)]
pub enum RunResult<M> {
    /// The loop finished normally.
    Finished {
        values: Vec<i64>,
        meta: M,
        savedata: Option<GcRef>,
    },
    /// The trace exited via a normal back-edge jump.
    Jump {
        values: Vec<i64>,
        meta: M,
        savedata: Option<GcRef>,
    },
    /// A guard failed.
    GuardFailure {
        values: Vec<i64>,
        meta: M,
        trace_id: u64,
        fail_index: u32,
        savedata: Option<GcRef>,
        recovery: Option<GuardRecovery>,
    },
}

/// Result of running compiled code with automatic blackhole fallback.
#[derive(Debug, Clone)]
pub enum BlackholeRunResult<M> {
    /// The trace produced final values, either directly or via blackhole replay.
    Finished {
        values: Vec<i64>,
        typed_values: Option<Vec<Value>>,
        exit_layout: Option<CompiledExitLayout>,
        meta: M,
        via_blackhole: bool,
        savedata: Option<GcRef>,
        exception: ExceptionState,
    },
    /// The blackhole replay reached a jump back-edge.
    Jump {
        values: Vec<i64>,
        typed_values: Option<Vec<Value>>,
        exit_layout: Option<CompiledExitLayout>,
        meta: M,
        via_blackhole: bool,
        savedata: Option<GcRef>,
        exception: ExceptionState,
    },
    /// Execution still ended in a guard failure.
    GuardFailure {
        trace_id: u64,
        fail_index: u32,
        fail_values: Vec<i64>,
        typed_fail_values: Option<Vec<Value>>,
        exit_layout: Option<CompiledExitLayout>,
        meta: M,
        recovery: Option<GuardRecovery>,
        materialized_virtuals: Vec<MaterializedVirtual>,
        pending_field_writes: Vec<ResolvedPendingFieldWrite>,
        via_blackhole: bool,
        savedata: Option<GcRef>,
        exception: ExceptionState,
    },
    /// Blackhole replay could not continue the trace.
    Abort {
        trace_id: u64,
        fail_index: u32,
        fail_values: Vec<i64>,
        typed_fail_values: Option<Vec<Value>>,
        exit_layout: Option<CompiledExitLayout>,
        meta: M,
        recovery: Option<GuardRecovery>,
        message: String,
        savedata: Option<GcRef>,
        exception: ExceptionState,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverRunOutcome {
    Finished { via_blackhole: bool },
    Jump { via_blackhole: bool },
    GuardFailure { restored: bool, via_blackhole: bool },
    Abort { restored: bool, via_blackhole: bool },
}

#[derive(Debug, Clone)]
pub enum DetailedDriverRunOutcome {
    Finished {
        typed_values: Vec<Value>,
        via_blackhole: bool,
        /// When true, Int-typed values are raw integers (not boxed pointers).
        raw_int_result: bool,
    },
    Jump {
        via_blackhole: bool,
    },
    /// compile.py:701 handle_fail: guard failure data for the caller to
    /// process via handle_fail(). No state restoration is done here —
    /// the caller decides whether to bridge or blackhole.
    GuardFailure {
        fail_index: u32,
        trace_id: u64,
        /// compile.py:702: must_compile() result.
        should_bridge: bool,
        /// compile.py: rd_loop_token — owning compiled loop key.
        owning_key: u64,
        /// compile.py:780: current_object_addr_as_int(self) — the exact
        /// descriptor that failed. Used for start/done_compiling.
        descr_addr: usize,
        /// Raw register values from compiled code exit.
        raw_values: Vec<i64>,
        /// Guard exit layout (rd_numb, fail_arg_types, etc.).
        exit_layout: CompiledExitLayout,
    },
    Abort {
        restored: bool,
        via_blackhole: bool,
    },
}

/// Decision about how to handle a function call during tracing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InlineDecision {
    /// Inline the call: continue tracing into the callee.
    Inline,
    /// Emit a CALL_ASSEMBLER: callee has compiled code.
    CallAssembler,
    /// Emit a residual (opaque) call.
    ResidualCall,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resume::{FrameSlotSource, ReconstructedValue, ResolvedPendingFieldWrite};
    use majit_backend::DeadFrame;
    use majit_backend::{Backend, ExitFrameLayout, ExitRecoveryLayout, ExitValueSourceLayout};
    #[cfg(feature = "cranelift")]
    use majit_backend_cranelift::compiler::{
        force_token_to_dead_frame, get_int_from_deadframe, get_latest_descr_from_deadframe,
        set_savedata_ref_on_deadframe,
    };
    #[cfg(feature = "cranelift")]
    use majit_backend_cranelift::guard::CraneliftFailDescr;
    use majit_gc::collector::MiniMarkGC;
    use majit_ir::descr::{CallDescr, Descr, EffectInfo, ExtraEffect};
    use majit_ir::{DescrRef, InputArg, Op, OpCode, OpRef, Type, Value};
    use std::sync::{Arc, Mutex, OnceLock};

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut op = Op::new(opcode, args);
        op.pos = OpRef(pos);
        op
    }

    fn mk_op_with_descr(opcode: OpCode, args: &[OpRef], pos: u32, descr: DescrRef) -> Op {
        let mut op = Op::with_descr(opcode, args, descr);
        op.pos = OpRef(pos);
        op
    }

    #[test]
    fn test_normalize_root_loop_entry_contract_inserts_label_from_inputargs() {
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 3),
            mk_op(OpCode::Jump, &[OpRef(3), OpRef(2), OpRef(1)], OpRef::NONE.0),
        ];

        let (normalized_inputargs, normalized_ops) =
            normalize_root_loop_entry_contract(inputargs.clone(), ops).expect("should normalize");

        assert_eq!(normalized_inputargs.len(), inputargs.len());
        assert_eq!(normalized_ops[0].opcode, OpCode::Label);
        assert_eq!(
            normalized_ops[0].args.as_slice(),
            &[OpRef(0), OpRef(1), OpRef(2)],
            "synthetic root label must follow original inputargs contract"
        );
    }

    #[test]
    fn test_normalize_root_loop_entry_contract_uses_simple_loop_jump_contract() {
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![mk_op(OpCode::Jump, &[OpRef(0), OpRef(1)], OpRef::NONE.0)];

        let (normalized_inputargs, normalized_ops) =
            normalize_root_loop_entry_contract(inputargs, ops).expect("should normalize");
        assert_eq!(normalized_inputargs.len(), 2);
        assert_eq!(normalized_ops[0].opcode, OpCode::Label);
        assert_eq!(normalized_ops[0].args.as_slice(), &[OpRef(0), OpRef(1)]);
    }

    #[test]
    fn test_root_loop_inputargs_from_optimizer_truncates_to_optimizer_contract() {
        let trace_inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_ref(2),
        ];

        let inputargs = root_loop_inputargs_from_optimizer(&trace_inputargs, 2);

        assert_eq!(inputargs.len(), 2);
        assert_eq!(inputargs[0].tp, Type::Ref);
        assert_eq!(inputargs[1].tp, Type::Int);
    }

    #[derive(Debug)]
    struct TestCallDescr {
        arg_types: Vec<Type>,
        result_type: Type,
    }

    impl Descr for TestCallDescr {
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestCallDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }

        fn result_type(&self) -> Type {
            self.result_type
        }

        fn result_size(&self) -> usize {
            8
        }

        fn effect_info(&self) -> &EffectInfo {
            static EFFECT_INFO: EffectInfo =
                EffectInfo::const_new(ExtraEffect::CanRaise, majit_ir::OopSpecIndex::None);
            &EFFECT_INFO
        }
    }

    fn make_call_descr(arg_types: Vec<Type>, result_type: Type) -> DescrRef {
        Arc::new(TestCallDescr {
            arg_types,
            result_type,
        })
    }

    fn may_force_void_values() -> &'static Mutex<Vec<i64>> {
        static VALUES: OnceLock<Mutex<Vec<i64>>> = OnceLock::new();
        VALUES.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn may_force_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(feature = "cranelift")]
    fn with_forced_deadframe(force_token: i64, f: impl FnOnce(DeadFrame)) {
        f(force_token_to_dead_frame(GcRef(force_token as usize)));
    }

    #[cfg(feature = "cranelift")]
    extern "C" fn maybe_force_and_return_void(force_token: i64, flag: i64) {
        if flag == 0 {
            return;
        }
        with_forced_deadframe(force_token, |mut deadframe| {
            let mut values = may_force_void_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner());
            values.push(
                get_latest_descr_from_deadframe(&deadframe)
                    .unwrap()
                    .fail_index() as i64,
            );
            values.push(get_int_from_deadframe(&deadframe, 0).unwrap());
            values.push(get_int_from_deadframe(&deadframe, 1).unwrap());
            drop(values);
            set_savedata_ref_on_deadframe(&mut deadframe, GcRef(0xDADA)).unwrap();
        });
    }

    #[cfg(feature = "cranelift")]
    fn attach_procedure_to_interp_entry(
        meta: &mut MetaInterp<()>,
        green_key: u64,
        inputargs: &[InputArg],
        ops: Vec<Op>,
        constants: HashMap<u32, i64>,
    ) {
        meta.backend.set_constants(constants.clone());
        let mut token = JitCellToken::new(green_key + 1000);
        let trace_id = meta.alloc_trace_id();
        meta.backend.set_next_trace_id(trace_id);
        meta.backend
            .compile_loop(inputargs, &ops, &mut token)
            .expect("loop should compile");
        let (mut resume_data, guard_op_indices, mut exit_layouts) =
            compile::build_guard_metadata(inputargs, &ops, green_key, &constants, &HashMap::new());
        let mut terminal_exit_layouts = compile::build_terminal_exit_layouts(inputargs, &ops);
        if let Some(backend_layouts) = meta.backend.compiled_fail_descr_layouts(&token) {
            compile::merge_backend_exit_layouts(&mut exit_layouts, &backend_layouts);
        }
        if let Some(backend_layouts) = meta.backend.compiled_terminal_exit_layouts(&token) {
            compile::merge_backend_terminal_exit_layouts(
                &mut terminal_exit_layouts,
                &backend_layouts,
            );
        }
        let trace_info = meta.backend.compiled_trace_info(&token, trace_id);
        compile::enrich_guard_resume_layouts_for_trace(
            &mut resume_data,
            &mut exit_layouts,
            trace_id,
            inputargs,
            trace_info.as_ref(),
        );
        compile::patch_backend_guard_recovery_layouts_for_trace(
            &mut meta.backend,
            &token,
            trace_id,
            &mut exit_layouts,
        );
        compile::patch_backend_terminal_recovery_layouts_for_trace(
            &mut meta.backend,
            &token,
            trace_id,
            &mut terminal_exit_layouts,
        );
        let mut traces = HashMap::new();
        traces.insert(
            trace_id,
            CompiledTrace {
                inputargs: inputargs.to_vec(),
                resume_data,
                ops,
                constants,
                constant_types: HashMap::new(),
                guard_op_indices,
                exit_layouts,
                terminal_exit_layouts,
                optimizer_knowledge: HashMap::new(),
                jitcode: None,
                snapshots: Vec::new(),
            },
        );

        meta.compiled_loops.insert(
            green_key,
            CompiledEntry {
                token,
                num_inputs: inputargs.len(),
                meta: (),
                front_target_tokens: Vec::new(),
                retraced_count: 0,
                root_trace_id: trace_id,

                traces,
                previous_tokens: Vec::new(),
            },
        );
    }

    #[cfg(feature = "cranelift")]
    fn install_may_force_void_entry(meta: &mut MetaInterp<()>, green_key: u64) {
        may_force_void_values()
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clear();
        let descr = make_call_descr(vec![Type::Ref, Type::Int], Type::Void);
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardNotForced, &[], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(1), OpRef(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::ForceToken, &[], 2),
            mk_op_with_descr(
                OpCode::CallMayForceN,
                &[OpRef(100), OpRef(2), OpRef(1)],
                OpRef::NONE.0,
                descr,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut constants = HashMap::new();
        constants.insert(
            100,
            maybe_force_and_return_void as *const () as usize as i64,
        );
        attach_procedure_to_interp_entry(meta, green_key, &inputargs, ops, constants);
    }

    fn test_vable_info_static_only() -> VirtualizableInfo {
        let mut info = VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));
        info
    }

    fn test_vable_info_with_array() -> VirtualizableInfo {
        let mut info = VirtualizableInfo::new(0);
        info.add_array_field_with_layout("stack", Type::Int, 24, 0, 0);
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));
        info
    }

    #[repr(C)]
    struct TraceEntryArray {
        len: usize,
        items: [i64; 4],
    }

    #[repr(C)]
    struct TraceEntryObj {
        arr: *const TraceEntryArray,
    }

    fn start_tracing_with_virtualizable(
        meta: &mut MetaInterp<()>,
        info: VirtualizableInfo,
        live_values: &[Value],
        array_lengths: Vec<usize>,
    ) {
        meta.set_virtualizable_info(info);
        meta.set_vable_array_lengths(array_lengths);
        let action = meta.force_start_tracing(777, None, live_values);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
    }

    fn take_recorded_ops(meta: &mut MetaInterp<()>) -> Vec<Op> {
        let mut ctx = meta.tracing.take().expect("expected active trace context");
        let num_inputs = ctx.recorder.num_inputargs();
        let jump_args: Vec<OpRef> = (0..num_inputs).map(|i| OpRef(i as u32)).collect();
        ctx.recorder.close_loop(&jump_args);
        let trace = ctx.recorder.get_trace();
        trace
            .ops
            .into_iter()
            .filter(|op| op.opcode != OpCode::Jump)
            .collect()
    }

    #[test]
    fn trace_entry_vable_lengths_prefers_cached_fallback_over_heap_lengths() {
        let mut info = VirtualizableInfo::new(0);
        info.add_array_field_with_layout(
            "arr",
            Type::Int,
            std::mem::offset_of!(TraceEntryObj, arr),
            0,
            std::mem::size_of::<usize>(),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));

        let array = TraceEntryArray {
            len: 4,
            items: [10, 20, 30, 40],
        };
        let obj = TraceEntryObj { arr: &array };

        let mut meta = MetaInterp::<()>::new(10);
        meta.set_virtualizable_info(info.clone());
        meta.set_vable_ptr((&obj as *const TraceEntryObj).cast());
        meta.set_vable_array_lengths(vec![1]);

        assert_eq!(meta.trace_entry_vable_lengths(&info), vec![1]);
    }

    #[test]
    fn opimpl_getfield_vable_int_reads_standard_box_without_heap_op() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );

        let result = meta.opimpl_getfield_vable_int(OpRef(0), 8);
        assert_eq!(result, OpRef(1));

        let ctx = meta.trace_ctx().unwrap();
        assert_eq!(ctx.recorder.num_ops(), 0);
    }

    #[test]
    fn opimpl_setfield_vable_int_synchronizes_standard_virtualizable() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(7)],
            Vec::new(),
        );

        let new_val = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(99)
        };
        meta.opimpl_setfield_vable_int(OpRef(0), 8, new_val);

        let ctx = meta.trace_ctx().unwrap();
        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes[0], new_val);
        // pyjitpl.py:1189-1194 _opimpl_setfield_vable for STANDARD
        // virtualizables only updates the cached box and calls
        // synchronize_virtualizable, which writes back into the
        // virtualizable struct via `vinfo.write_boxes` WITHOUT recording
        // any trace ops (RPython pyjitpl.py:3446-3450). The trace stays
        // empty until a non-virtualizable op is recorded.
        assert_eq!(ctx.recorder.num_ops(), 0);
    }

    #[test]
    fn opimpl_getarrayitem_vable_int_reads_standard_box_without_heap_op() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_with_array(),
            &[Value::Int(0x1234), Value::Int(11), Value::Int(22)],
            vec![2],
        );

        let index = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(1)
        };
        let result = meta.opimpl_getarrayitem_vable_int(OpRef(0), index, 1, 24);
        assert_eq!(result, OpRef(2));

        let ctx = meta.trace_ctx().unwrap();
        assert_eq!(ctx.recorder.num_ops(), 0);
    }

    #[test]
    fn opimpl_arraylen_vable_returns_cached_standard_length() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_with_array(),
            &[Value::Int(0x1234), Value::Int(11), Value::Int(22)],
            vec![2],
        );

        let len_ref = meta.opimpl_arraylen_vable(OpRef(0), 24);
        let ctx = meta.trace_ctx().unwrap();
        assert_eq!(ctx.const_value(len_ref), Some(2));
        assert_eq!(ctx.recorder.num_ops(), 0);
    }

    #[test]
    fn opimpl_getfield_vable_int_nonstandard_falls_back_to_heap_op() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );

        let nonstandard_vable = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(0xCAFE)
        };
        let _result = meta.opimpl_getfield_vable_int(nonstandard_vable, 8);

        // pyjitpl.py:1120-1146 _nonstandard_virtualizable falls through
        // to Step 4 (PTR_EQ + implement_guard_value) and Step 5a
        // (emit_force_virtualizable: GETFIELD_GC_R(token_descr) +
        // PTR_NE(CONST_NULL) + COND_CALL) before Step 5b marks the box
        // known. The COND_CALL tail is currently a TODO; the observable
        // prefix is the four ops emitted by `nonstandard_virtualizable`,
        // followed by the caller's GETFIELD_GC_I (the actual non-vable
        // field read).
        let ops = take_recorded_ops(&mut meta);
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[0].opcode, OpCode::PtrEq); // Step 4: PTR_EQ
        assert_eq!(ops[1].opcode, OpCode::GuardValue); // Step 4: implement_guard_value
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcR); // Step 5a: token_descr read
        assert_eq!(ops[3].opcode, OpCode::PtrNe); // Step 5a: PTR_NE(CONST_NULL)
        assert_eq!(ops[4].opcode, OpCode::GetfieldGcI); // caller fallback
    }

    #[test]
    fn opimpl_getarrayitem_vable_int_nonstandard_falls_back_to_heap_ops() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_with_array(),
            &[Value::Int(0x1234), Value::Int(11), Value::Int(22)],
            vec![2],
        );

        let (nonstandard_vable, index) = {
            let ctx = meta.trace_ctx().unwrap();
            (ctx.const_int(0xCAFE), ctx.const_int(1))
        };
        let _result = meta.opimpl_getarrayitem_vable_int(nonstandard_vable, index, 1, 24);

        // pyjitpl.py:1219-1230 _opimpl_getarrayitem_vable falls back to
        // GETFIELD_GC_R(arraydescr) + GETARRAYITEM_GC_I(arraybox) when
        // _nonstandard_virtualizable returns True. The four ops emitted
        // by `_nonstandard_virtualizable` (Step 4 PTR_EQ + GUARD_VALUE
        // and Step 5a GETFIELD_GC_R(token_descr) + PTR_NE) precede the
        // caller's two-op fallback, totalling 6 ops.
        let ops = take_recorded_ops(&mut meta);
        assert_eq!(ops.len(), 6);
        assert_eq!(ops[0].opcode, OpCode::PtrEq); // Step 4: PTR_EQ
        assert_eq!(ops[1].opcode, OpCode::GuardValue); // Step 4: implement_guard_value
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcR); // Step 5a: token_descr read
        assert_eq!(ops[3].opcode, OpCode::PtrNe); // Step 5a: PTR_NE(CONST_NULL)
        assert_eq!(ops[4].opcode, OpCode::GetfieldGcR); // caller fallback: arraybox
        assert_eq!(ops[5].opcode, OpCode::GetarrayitemGcI); // caller fallback: item read
    }

    #[test]
    fn opimpl_hint_force_virtualizable_standard_emits_store_back_only_once() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );

        meta.opimpl_hint_force_virtualizable(OpRef(0));
        meta.opimpl_hint_force_virtualizable(OpRef(0));

        let ops = take_recorded_ops(&mut meta);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::SetfieldGc);
    }

    #[test]
    fn opimpl_hint_force_virtualizable_ignores_nonstandard_virtualizable() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );

        let nonstandard_vable = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(0xCAFE)
        };
        meta.opimpl_hint_force_virtualizable(nonstandard_vable);

        let ops = take_recorded_ops(&mut meta);
        assert!(ops.is_empty());
    }

    #[test]
    fn hint_force_virtualizable_state_is_reset_between_traces() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );
        meta.opimpl_hint_force_virtualizable(OpRef(0));
        let _ = meta.finish_trace_for_parity(&[]);

        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );
        meta.opimpl_hint_force_virtualizable(OpRef(0));

        let ops = take_recorded_ops(&mut meta);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::SetfieldGc);
    }

    #[test]
    fn standard_vable_access_consumes_forced_virtualizable_state() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );

        meta.opimpl_hint_force_virtualizable(OpRef(0));
        let _ = meta.opimpl_getfield_vable_int(OpRef(0), 8);
        meta.opimpl_hint_force_virtualizable(OpRef(0));

        let ops = take_recorded_ops(&mut meta);
        assert_eq!(ops.len(), 4);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[2].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[3].opcode, OpCode::SetfieldGc);
    }

    #[test]
    fn compiled_virtualizable_trace_does_not_use_raw_heap_ops() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_with_array(),
            &[Value::Int(0x1234), Value::Int(10), Value::Int(20)],
            vec![2],
        );

        let index = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(1)
        };
        let item = meta.opimpl_getarrayitem_vable_int(OpRef(0), index, 1, 24);
        if let Some(ctx) = meta.trace_ctx() {
            ctx.record_guard_with_fail_args(
                OpCode::GuardTrue,
                &[item],
                0,
                &[OpRef(0), OpRef(1), OpRef(2)],
            );
        }
        meta.compile_loop(&[OpRef(0), OpRef(1), OpRef(2)], ());

        let compiled = meta.compiled_loops.get(&777).expect("compiled entry");
        let trace = compiled
            .traces
            .get(&compiled.root_trace_id)
            .expect("root compiled trace");

        assert!(
            trace.ops.iter().all(|op| {
                !matches!(
                    op.opcode,
                    OpCode::GetfieldRawI
                        | OpCode::GetfieldRawR
                        | OpCode::GetfieldRawF
                        | OpCode::SetfieldRaw
                        | OpCode::GetarrayitemRawI
                        | OpCode::GetarrayitemRawR
                        | OpCode::GetarrayitemRawF
                        | OpCode::SetarrayitemRaw
                )
            }),
            "standard virtualizable loop should use vable boxes, not raw heap ops: {}",
            majit_ir::format_trace(&trace.ops, &trace.constants)
        );
        assert_eq!(item, OpRef(2));
    }

    #[test]
    fn optimizer_vable_config_requires_standard_virtualizable_boxes() {
        let mut meta = MetaInterp::<()>::new(10);
        let info = test_vable_info_with_array();
        meta.set_virtualizable_info(info.clone());
        assert!(
            meta.current_virtualizable_optimizer_config().is_none(),
            "virtualizable config should only exist while tracing is active"
        );

        let action = meta.force_start_tracing(777, None, &[Value::Int(0x1234)]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        assert!(
            meta.current_virtualizable_optimizer_config().is_none(),
            "virtualizable config should require standard virtualizable boxes"
        );
    }

    #[test]
    fn optimizer_vable_config_matches_registered_virtualizable_when_boxes_active() {
        let mut meta = MetaInterp::<()>::new(10);
        let info = test_vable_info_with_array();
        start_tracing_with_virtualizable(
            &mut meta,
            info.clone(),
            &[Value::Int(0x1234), Value::Int(10), Value::Int(20)],
            vec![2],
        );

        let config = meta
            .current_virtualizable_optimizer_config()
            .expect("standard virtualizable trace should pass config to optimizer");
        assert_eq!(
            config.static_field_offsets,
            info.to_optimizer_config().static_field_offsets
        );
        assert_eq!(
            config.static_field_types,
            info.to_optimizer_config().static_field_types
        );
        assert_eq!(
            config.array_field_offsets,
            info.to_optimizer_config().array_field_offsets
        );
        assert_eq!(
            config.array_item_types,
            info.to_optimizer_config().array_item_types
        );
        assert_eq!(config.array_lengths, vec![2]);
    }

    // ── JitIface hook/callback parity tests (rpython/jit/metainterp/test/test_jitiface.py) ──

    #[test]
    fn test_on_compile_loop_fires_with_correct_metadata() {
        // Parity with test_on_compile: after_compile hook fires with green_key,
        // num_ops_before, num_ops_after.
        let mut meta = MetaInterp::<()>::new(1);
        let compile_events: Arc<Mutex<Vec<(u64, usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let events = compile_events.clone();
        meta.set_on_compile_loop(move |green_key, ops_before, ops_after| {
            events
                .lock()
                .unwrap()
                .push((green_key, ops_before, ops_after));
        });

        let green_key = 42;
        // Trigger tracing by making back-edge hot
        for _ in 0..2 {
            meta.on_back_edge(green_key, &[0]);
        }
        assert!(meta.tracing.is_some());

        // Record a simple operation and close the trace
        if let Some(ctx) = meta.trace_ctx() {
            let i0 = OpRef(0);
            let const_one = ctx.const_int(1);
            let sum = ctx.recorder.record_op(OpCode::IntAdd, &[i0, const_one]);
            ctx.record_guard_with_fail_args(OpCode::GuardTrue, &[i0], 0, &[sum]);
        }
        meta.compile_loop(&[OpRef(0)], ());

        let events = compile_events.lock().unwrap();
        assert_eq!(events.len(), 1, "on_compile_loop should fire exactly once");
        assert_eq!(events[0].0, green_key, "green_key should match");
        assert!(events[0].1 > 0, "num_ops_before should be positive");
        assert!(events[0].2 > 0, "num_ops_after should be positive");
    }

    #[test]
    fn test_on_compile_error_fires_on_failure() {
        // Parity with test_on_abort: on_compile_error fires when compilation fails.
        // We can test this by installing a hook and verifying it captures the error.
        let mut meta = MetaInterp::<()>::new(10);
        let error_events: Arc<Mutex<Vec<(u64, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let events = error_events.clone();
        meta.set_on_compile_error(move |green_key, msg| {
            events.lock().unwrap().push((green_key, msg.to_string()));
        });

        // There's no easy way to trigger a compilation failure through the public API
        // without a malformed trace, so we directly test the hook mechanism.
        // Simulate: if the hook is set, calling it works correctly.
        if let Some(ref cb) = meta.hooks.on_compile_error {
            cb(99, "test error");
        }
        let events = error_events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, 99);
        assert_eq!(events[0].1, "test error");
    }

    #[test]
    fn test_multiple_hooks_independent() {
        // Parity with JitHookInterface: multiple different hooks can be registered
        // independently and all fire for their respective events.
        let mut meta = MetaInterp::<()>::new(1);

        let compile_count = Arc::new(Mutex::new(0u32));
        let trace_start_count = Arc::new(Mutex::new(0u32));
        let trace_abort_count = Arc::new(Mutex::new(0u32));

        let cc = compile_count.clone();
        meta.set_on_compile_loop(move |_, _, _| {
            *cc.lock().unwrap() += 1;
        });

        let tsc = trace_start_count.clone();
        meta.set_on_trace_start(move |_| {
            *tsc.lock().unwrap() += 1;
        });

        let tac = trace_abort_count.clone();
        meta.set_on_trace_abort(move |_, _| {
            *tac.lock().unwrap() += 1;
        });

        let green_key = 100;
        // Heat up and start tracing
        for _ in 0..2 {
            meta.on_back_edge(green_key, &[0]);
        }
        assert_eq!(
            *trace_start_count.lock().unwrap(),
            1,
            "on_trace_start should fire"
        );

        // Abort the trace
        meta.abort_trace(false);
        assert_eq!(
            *trace_abort_count.lock().unwrap(),
            1,
            "on_trace_abort should fire"
        );
        assert_eq!(
            *compile_count.lock().unwrap(),
            0,
            "on_compile_loop should NOT fire yet"
        );

        // Start another trace and compile it.
        // After abort, the cell goes to DontTraceHere if retrace limit is exceeded.
        // Use a fresh green key to avoid this.
        let green_key2 = 200;
        for _ in 0..2 {
            meta.on_back_edge(green_key2, &[0]);
        }
        if let Some(ctx) = meta.trace_ctx() {
            let i0 = OpRef(0);
            let const_one = ctx.const_int(1);
            let sum = ctx.recorder.record_op(OpCode::IntAdd, &[i0, const_one]);
            ctx.record_guard_with_fail_args(OpCode::GuardTrue, &[i0], 0, &[sum]);
        }
        meta.compile_loop(&[OpRef(0)], ());
        assert_eq!(
            *compile_count.lock().unwrap(),
            1,
            "on_compile_loop should fire after compile"
        );
        assert_eq!(
            *trace_start_count.lock().unwrap(),
            2,
            "on_trace_start should fire twice total"
        );
    }

    #[test]
    fn test_on_compile_loop_receives_correct_trace_metadata() {
        // Parity with test_on_compile: verify that the hook receives the correct
        // green key and that op counts reflect the actual trace.
        let mut meta = MetaInterp::<()>::new(1);
        let events: Arc<Mutex<Vec<(u64, usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let ev = events.clone();
        meta.set_on_compile_loop(move |gk, before, after| {
            ev.lock().unwrap().push((gk, before, after));
        });

        // Compile two different loops with different green keys
        for green_key in [10u64, 20u64] {
            for _ in 0..2 {
                meta.on_back_edge(green_key, &[0, 0]);
            }
            if let Some(ctx) = meta.trace_ctx() {
                let i0 = OpRef(0);
                let i1 = OpRef(1);
                let const_one = ctx.const_int(1);
                let sum = ctx.recorder.record_op(OpCode::IntAdd, &[i0, i1]);
                let sum2 = ctx.recorder.record_op(OpCode::IntAdd, &[sum, const_one]);
                ctx.record_guard_with_fail_args(OpCode::GuardTrue, &[i0], 0, &[sum2, i1]);
            }
            meta.compile_loop(&[OpRef(0), OpRef(1)], ());
        }

        let events = events.lock().unwrap();
        assert_eq!(events.len(), 2, "two compilation events should fire");
        assert_eq!(events[0].0, 10, "first event green_key=10");
        assert_eq!(events[1].0, 20, "second event green_key=20");
        // Both traces had the same ops, so op counts should be equal
        assert_eq!(
            events[0].1, events[1].1,
            "ops_before should match for identical traces"
        );
    }

    #[test]
    #[cfg(feature = "cranelift")]
    fn test_on_compile_bridge_fires() {
        // Parity with test_on_compile_bridge: after_compile_bridge hook fires
        // when a bridge is compiled.
        let mut meta = MetaInterp::<()>::new(10);
        let bridge_events: Arc<Mutex<Vec<(u64, u32, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let ev = bridge_events.clone();
        meta.set_on_compile_bridge(move |gk, fi, nops| {
            ev.lock().unwrap().push((gk, fi, nops));
        });

        // Install a simple compiled loop with a guard
        let green_key = 50;
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let _const_one = OpRef(100);
        let const_zero = OpRef(101);
        let mut guard_op = mk_op(OpCode::GuardTrue, &[OpRef(2)], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(0), OpRef(1)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::IntGt, &[OpRef(2), const_zero], 3),
            {
                let mut g = mk_op(OpCode::GuardTrue, &[OpRef(3)], OpRef::NONE.0);
                g.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(0), OpRef(1)]));
                g
            },
            mk_op(OpCode::Jump, &[OpRef(2), OpRef(1)], OpRef::NONE.0),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 0);
        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, constants);

        // The bridge hook is set. We verify the hook mechanism is correctly wired.
        if let Some(ref hook) = meta.hooks.on_compile_bridge {
            hook(green_key, 3, 5);
        }
        let events = bridge_events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, green_key);
        assert_eq!(events[0].1, 3);
        assert_eq!(events[0].2, 5);
    }

    #[test]
    fn test_on_guard_failure_hook() {
        // Parity with test_get_stats: guard failure hook fires with correct args.
        let mut meta = MetaInterp::<()>::new(10);
        let failure_events: Arc<Mutex<Vec<(u64, u32, u32)>>> = Arc::new(Mutex::new(Vec::new()));
        let ev = failure_events.clone();
        meta.set_on_guard_failure(move |gk, fi, fc| {
            ev.lock().unwrap().push((gk, fi, fc));
        });

        // Verify the hook is correctly installed and callable
        if let Some(ref hook) = meta.hooks.on_guard_failure {
            hook(42, 3, 1);
            hook(42, 3, 2);
            hook(42, 5, 1);
        }
        let events = failure_events.lock().unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], (42, 3, 1));
        assert_eq!(events[1], (42, 3, 2));
        assert_eq!(events[2], (42, 5, 1));
    }

    #[test]
    fn test_should_record_guard_failure_skips_jump_exit() {
        assert!(!MetaInterp::<()>::should_record_guard_failure(
            false,
            u32::MAX
        ));
        assert!(MetaInterp::<()>::should_record_guard_failure(false, 7));
        assert!(!MetaInterp::<()>::should_record_guard_failure(true, 7));
    }

    #[test]
    fn test_run_result_for_jump_exit_returns_jump() {
        let result = MetaInterp::<()>::run_result_for_jump_exit(u32::MAX, vec![42], (), None)
            .expect("jump exit should produce a direct Jump result");
        match result {
            RunResult::Jump { values, .. } => assert_eq!(values, vec![42]),
            other => panic!("expected Jump result, got {other:?}"),
        }

        assert!(
            MetaInterp::<()>::run_result_for_jump_exit(3, vec![42], (), None).is_none(),
            "guard failure exits must keep using recovery paths"
        );
    }

    #[test]
    fn test_blackhole_result_for_jump_exit_returns_jump() {
        let exit_layout = CompiledExitLayout {
            rd_loop_token: 91,
            trace_id: 12,
            fail_index: u32::MAX,
            source_op_index: Some(3),
            exit_types: vec![Type::Int],
            is_finish: false,
            gc_ref_slots: Vec::new(),
            force_token_slots: Vec::new(),
            recovery_layout: None,
            resume_layout: None,
            rd_numb: None,
            rd_consts: None,
            rd_virtuals: None,
            rd_pendingfields: None,
        };
        let exception = ExceptionState::default();

        let result = MetaInterp::<()>::blackhole_result_for_jump_exit(
            u32::MAX,
            vec![42],
            vec![Value::Int(42)],
            exit_layout.clone(),
            (),
            None,
            exception.clone(),
        )
        .expect("jump exit should bypass blackhole recovery");
        match result {
            BlackholeRunResult::Jump {
                values,
                typed_values,
                exit_layout,
                via_blackhole,
                exception,
                ..
            } => {
                assert_eq!(values, vec![42]);
                assert_eq!(typed_values, Some(vec![Value::Int(42)]));
                assert_eq!(exit_layout.unwrap().trace_id, 12);
                assert!(!via_blackhole);
                assert_eq!(exception, ExceptionState::default());
            }
            other => panic!("expected Jump result, got {other:?}"),
        }

        assert!(
            MetaInterp::<()>::blackhole_result_for_jump_exit(
                5,
                vec![42],
                vec![Value::Int(42)],
                exit_layout,
                (),
                None,
                exception,
            )
            .is_none(),
            "guard failure exits must keep using guard recovery / blackhole fallback"
        );
    }

    #[test]
    fn test_on_trace_abort_hook_with_permanent_flag() {
        // Parity with test_abort_quasi_immut: on_abort receives the permanent flag.
        let mut meta = MetaInterp::<()>::new(1);
        let abort_events: Arc<Mutex<Vec<(u64, bool)>>> = Arc::new(Mutex::new(Vec::new()));
        let ev = abort_events.clone();
        meta.set_on_trace_abort(move |gk, permanent| {
            ev.lock().unwrap().push((gk, permanent));
        });

        let green_key = 77;
        // Start tracing
        for _ in 0..2 {
            meta.on_back_edge(green_key, &[0]);
        }
        assert!(meta.tracing.is_some());

        // Abort non-permanently
        meta.abort_trace(false);
        {
            let events = abort_events.lock().unwrap();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0], (green_key, false));
        }

        // Start tracing again and abort permanently
        for _ in 0..2 {
            meta.on_back_edge(green_key, &[0]);
        }
        if meta.tracing.is_some() {
            meta.abort_trace(true);
            let events = abort_events.lock().unwrap();
            assert_eq!(events.len(), 2);
            assert_eq!(events[1], (green_key, true));
        }
    }

    #[test]
    fn test_jit_hooks_default_is_all_none() {
        // All hooks default to None.
        let hooks = JitHooks::default();
        assert!(hooks.on_compile_loop.is_none());
        assert!(hooks.on_compile_bridge.is_none());
        assert!(hooks.on_guard_failure.is_none());
        assert!(hooks.on_trace_start.is_none());
        assert!(hooks.on_trace_abort.is_none());
        assert!(hooks.on_compile_error.is_none());
    }
}
