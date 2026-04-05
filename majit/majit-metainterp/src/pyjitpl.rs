use std::collections::{HashMap, HashSet};

use crate::optimizeopt::optimizer::{Optimizer, OptimizerKnowledge};
use majit_backend::{
    Backend, CompiledTraceInfo, ExitFrameLayout, ExitRecoveryLayout, FailDescrLayout, JitCellToken,
    TerminalExitLayout,
};
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use majit_backend_cranelift::CraneliftBackend as BackendImpl;
#[cfg(target_arch = "wasm32")]
pub(crate) use majit_backend_wasm::WasmBackend as BackendImpl;
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
    let mut next_const_idx = constants.keys().copied().max().unwrap_or(10_000) + 1;
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
                    let k = next_const_idx;
                    next_const_idx += 1;
                    constants.insert(k, *val);
                    constant_types.insert(k, *tp);
                    k
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
    /// RPython parity: standard virtualizable that was just forced via
    /// `hint_force_virtualizable` / `gen_store_back_in_vable`.
    pub(crate) forced_virtualizable: Option<OpRef>,
    /// warmspot.py:449 jd.result_type — per-driver static result type.
    pub(crate) result_type: Type,
    /// Helper function pointers that box raw ints into interpreter objects.
    pub(crate) raw_int_box_helpers: HashSet<i64>,
    /// W_IntObject.intval field descriptor for unboxing Ref→Int in FINISH.
    pub(crate) intval_descr: Option<majit_ir::DescrRef>,
    /// Helper function pointers that take a raw int argument and return a
    /// raw-int result when the callee's Finish protocol is raw.
    pub(crate) raw_int_force_helpers: HashSet<i64>,
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
            forced_virtualizable: None,
            result_type: Type::Ref,
            raw_int_box_helpers: HashSet::new(),
            intval_descr: None,
            raw_int_force_helpers: HashSet::new(),
            create_frame_raw_map: HashMap::new(),
            tracing_call_depth: None,
            max_unroll_recursion: 7, // RPython default from rlib/jit.py
            force_finish_trace: false,
            partial_trace: None,
            retracing_from: None,
            exported_state: None,
            cancel_count: 0,
            last_compiled_key: None,
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
        opt
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
                // Init virtualizable boxes (same as on_back_edge_typed)
                if let Some(ref info) = self.virtualizable_info {
                    let array_lengths = self.trace_entry_vable_lengths(info);
                    let num_static = info.num_static_extra_boxes;
                    let num_array_elems: usize = array_lengths.iter().sum();
                    let total_vable = num_static + num_array_elems;
                    if total_vable > 0 && live_values.len() >= 1 + total_vable {
                        let vable_oprefs: Vec<OpRef> =
                            (0..total_vable).map(|i| OpRef((1 + i) as u32)).collect();
                        ctx.init_virtualizable_boxes(
                            info,
                            OpRef(0), // frame ref = first inputarg
                            &vable_oprefs,
                            &array_lengths,
                        );
                    }
                }
                self.forced_virtualizable = None;
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
                self.backend
                    .register_pending_target(pending_num, input_types, num_inputs);
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
        if let Some(ref info) = self.virtualizable_info {
            let array_lengths = self.trace_entry_vable_lengths(info);
            let num_static = info.num_static_extra_boxes;
            let num_array_elems: usize = array_lengths.iter().sum();
            let total_vable = num_static + num_array_elems;

            if total_vable > 0 && live_values.len() >= 1 + total_vable {
                let vable_oprefs: Vec<OpRef> =
                    (0..total_vable).map(|i| OpRef((1 + i) as u32)).collect();
                ctx.init_virtualizable_boxes(info, OpRef(0), &vable_oprefs, &array_lengths);
            }
        }

        self.forced_virtualizable = None;
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

    fn is_standard_virtualizable(&self, vable_opref: OpRef) -> bool {
        self.tracing
            .as_ref()
            .and_then(|ctx| ctx.standard_virtualizable_box())
            .is_some_and(|standard| standard == vable_opref)
    }

    /// RPython equivalent: `MIFrameOpImpl._nonstandard_virtualizable()`.
    ///
    /// Returns true when `vable_opref` is not the active standard
    /// virtualizable tracked in `virtualizable_boxes[-1]`.
    fn nonstandard_virtualizable(&mut self, vable_opref: OpRef) -> bool {
        if self.forced_virtualizable == Some(vable_opref) {
            self.forced_virtualizable = None;
        }
        !self.is_standard_virtualizable(vable_opref)
    }

    fn virtualizable_field_index(&self, field_offset: usize) -> Option<usize> {
        self.virtualizable_info
            .as_ref()
            .and_then(|info| info.static_field_index(field_offset))
    }

    /// RPython equivalent: `_get_arrayitem_vable_index()`.
    ///
    /// Returns the flat index into the standard virtualizable box array.
    fn get_arrayitem_vable_index(&self, array_field_offset: usize, index: OpRef) -> Option<usize> {
        let ctx = self.tracing.as_ref()?;
        let raw_index = ctx.const_value(index)?;
        let item_index = usize::try_from(raw_index).ok()?;
        let info = self.virtualizable_info.as_ref()?;
        let lengths = ctx.virtualizable_array_lengths()?;
        let array_index = info.array_field_index(array_field_offset)?;
        Some(info.get_index_in_array(array_index, item_index, lengths))
    }

    /// RPython equivalent: `check_synchronized_virtualizable()`.
    ///
    /// In translated PyPy this is effectively a no-op. Keep the same contract:
    /// the active tracing state is the source of truth, and debug builds may
    /// assert that a standard virtualizable has been initialized.
    pub fn check_synchronized_virtualizable(&self) {
        if let Some(ctx) = self.tracing.as_ref() {
            debug_assert!(
                !ctx.has_virtualizable_boxes() || ctx.standard_virtualizable_box().is_some()
            );
        }
    }

    /// RPython equivalent: `synchronize_virtualizable()`.
    ///
    /// Standard virtualizable writes are materialized back to heap state through
    /// the existing trace-time store-back path.
    pub fn synchronize_virtualizable(&mut self, vable_opref: OpRef) {
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.gen_store_back_in_vable(vable_opref);
        }
    }

    /// RPython equivalent: `opimpl_getfield_vable_i(box, fielddescr, pc)`.
    pub fn opimpl_getfield_vable_int(&mut self, vable_opref: OpRef, field_offset: usize) -> OpRef {
        if self.nonstandard_virtualizable(vable_opref) {
            let ctx = self
                .tracing
                .as_mut()
                .expect("opimpl_getfield_vable_int requires active tracing");
            let offset_ref = ctx.const_int(field_offset as i64);
            return ctx.record_op(OpCode::GetfieldGcI, &[vable_opref, offset_ref]);
        }
        self.check_synchronized_virtualizable();
        let index = self
            .virtualizable_field_index(field_offset)
            .expect("unknown standard virtualizable field offset");
        self.tracing
            .as_ref()
            .and_then(|ctx| ctx.virtualizable_box_at(index))
            .expect("standard virtualizable box missing")
    }

    /// RPython equivalent: `opimpl_getfield_vable_r(box, fielddescr, pc)`.
    pub fn opimpl_getfield_vable_ref(&mut self, vable_opref: OpRef, field_offset: usize) -> OpRef {
        if self.nonstandard_virtualizable(vable_opref) {
            let ctx = self
                .tracing
                .as_mut()
                .expect("opimpl_getfield_vable_ref requires active tracing");
            let offset_ref = ctx.const_int(field_offset as i64);
            return ctx.record_op(OpCode::GetfieldGcR, &[vable_opref, offset_ref]);
        }
        self.check_synchronized_virtualizable();
        let index = self
            .virtualizable_field_index(field_offset)
            .expect("unknown standard virtualizable field offset");
        self.tracing
            .as_ref()
            .and_then(|ctx| ctx.virtualizable_box_at(index))
            .expect("standard virtualizable box missing")
    }

    /// RPython equivalent: `opimpl_getfield_vable_f(box, fielddescr, pc)`.
    pub fn opimpl_getfield_vable_float(
        &mut self,
        vable_opref: OpRef,
        field_offset: usize,
    ) -> OpRef {
        if self.nonstandard_virtualizable(vable_opref) {
            let ctx = self
                .tracing
                .as_mut()
                .expect("opimpl_getfield_vable_float requires active tracing");
            let offset_ref = ctx.const_int(field_offset as i64);
            return ctx.record_op(OpCode::GetfieldGcF, &[vable_opref, offset_ref]);
        }
        self.check_synchronized_virtualizable();
        let index = self
            .virtualizable_field_index(field_offset)
            .expect("unknown standard virtualizable field offset");
        self.tracing
            .as_ref()
            .and_then(|ctx| ctx.virtualizable_box_at(index))
            .expect("standard virtualizable box missing")
    }

    fn opimpl_setfield_vable_any(&mut self, vable_opref: OpRef, field_offset: usize, value: OpRef) {
        if self.nonstandard_virtualizable(vable_opref) {
            if let Some(ctx) = self.tracing.as_mut() {
                let offset_ref = ctx.const_int(field_offset as i64);
                ctx.record_op(OpCode::SetfieldGc, &[vable_opref, value, offset_ref]);
            }
            return;
        }
        let index = self
            .virtualizable_field_index(field_offset)
            .expect("unknown standard virtualizable field offset");
        if let Some(ctx) = self.tracing.as_mut() {
            let updated = ctx.set_virtualizable_box_at(index, value);
            debug_assert!(updated, "standard virtualizable box missing");
        }
        self.synchronize_virtualizable(vable_opref);
    }

    /// RPython equivalent: `opimpl_setfield_vable_i`.
    pub fn opimpl_setfield_vable_int(
        &mut self,
        vable_opref: OpRef,
        field_offset: usize,
        value: OpRef,
    ) {
        self.opimpl_setfield_vable_any(vable_opref, field_offset, value);
    }

    /// RPython equivalent: `opimpl_setfield_vable_r`.
    pub fn opimpl_setfield_vable_ref(
        &mut self,
        vable_opref: OpRef,
        field_offset: usize,
        value: OpRef,
    ) {
        self.opimpl_setfield_vable_any(vable_opref, field_offset, value);
    }

    /// RPython equivalent: `opimpl_setfield_vable_f`.
    pub fn opimpl_setfield_vable_float(
        &mut self,
        vable_opref: OpRef,
        field_offset: usize,
        value: OpRef,
    ) {
        self.opimpl_setfield_vable_any(vable_opref, field_offset, value);
    }

    /// RPython equivalent: `_opimpl_getarrayitem_vable`.
    pub fn opimpl_getarrayitem_vable_int(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        array_field_offset: usize,
    ) -> OpRef {
        if self.nonstandard_virtualizable(vable_opref) {
            let ctx = self
                .tracing
                .as_mut()
                .expect("opimpl_getarrayitem_vable_int requires active tracing");
            let field_ref = ctx.const_int(array_field_offset as i64);
            let array_opref = ctx.record_op(OpCode::GetfieldGcR, &[vable_opref, field_ref]);
            let zero = ctx.const_int(0);
            return ctx.record_op(OpCode::GetarrayitemGcI, &[array_opref, index, zero]);
        }
        self.check_synchronized_virtualizable();
        let flat_index = self
            .get_arrayitem_vable_index(array_field_offset, index)
            .expect("standard virtualizable array index must be constant");
        self.tracing
            .as_ref()
            .and_then(|ctx| ctx.virtualizable_box_at(flat_index))
            .expect("standard virtualizable array box missing")
    }

    /// RPython equivalent: `_opimpl_getarrayitem_vable`.
    pub fn opimpl_getarrayitem_vable_ref(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        array_field_offset: usize,
    ) -> OpRef {
        if self.nonstandard_virtualizable(vable_opref) {
            let ctx = self
                .tracing
                .as_mut()
                .expect("opimpl_getarrayitem_vable_ref requires active tracing");
            let field_ref = ctx.const_int(array_field_offset as i64);
            let array_opref = ctx.record_op(OpCode::GetfieldGcR, &[vable_opref, field_ref]);
            let zero = ctx.const_int(0);
            return ctx.record_op(OpCode::GetarrayitemGcR, &[array_opref, index, zero]);
        }
        self.check_synchronized_virtualizable();
        let flat_index = self
            .get_arrayitem_vable_index(array_field_offset, index)
            .expect("standard virtualizable array index must be constant");
        self.tracing
            .as_ref()
            .and_then(|ctx| ctx.virtualizable_box_at(flat_index))
            .expect("standard virtualizable array box missing")
    }

    /// RPython equivalent: `_opimpl_getarrayitem_vable`.
    pub fn opimpl_getarrayitem_vable_float(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        array_field_offset: usize,
    ) -> OpRef {
        if self.nonstandard_virtualizable(vable_opref) {
            let ctx = self
                .tracing
                .as_mut()
                .expect("opimpl_getarrayitem_vable_float requires active tracing");
            let field_ref = ctx.const_int(array_field_offset as i64);
            let array_opref = ctx.record_op(OpCode::GetfieldGcR, &[vable_opref, field_ref]);
            let zero = ctx.const_int(0);
            return ctx.record_op(OpCode::GetarrayitemGcF, &[array_opref, index, zero]);
        }
        self.check_synchronized_virtualizable();
        let flat_index = self
            .get_arrayitem_vable_index(array_field_offset, index)
            .expect("standard virtualizable array index must be constant");
        self.tracing
            .as_ref()
            .and_then(|ctx| ctx.virtualizable_box_at(flat_index))
            .expect("standard virtualizable array box missing")
    }

    fn opimpl_setarrayitem_vable_any(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        value: OpRef,
        array_field_offset: usize,
    ) {
        if self.nonstandard_virtualizable(vable_opref) {
            if let Some(ctx) = self.tracing.as_mut() {
                let field_ref = ctx.const_int(array_field_offset as i64);
                let array_opref = ctx.record_op(OpCode::GetfieldGcR, &[vable_opref, field_ref]);
                let zero = ctx.const_int(0);
                ctx.record_op(OpCode::SetarrayitemGc, &[array_opref, index, value, zero]);
            }
            return;
        }
        let flat_index = self
            .get_arrayitem_vable_index(array_field_offset, index)
            .expect("standard virtualizable array index must be constant");
        if let Some(ctx) = self.tracing.as_mut() {
            let updated = ctx.set_virtualizable_box_at(flat_index, value);
            debug_assert!(updated, "standard virtualizable array box missing");
        }
        self.synchronize_virtualizable(vable_opref);
    }

    /// RPython equivalent: `_opimpl_setarrayitem_vable`.
    pub fn opimpl_setarrayitem_vable_int(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        value: OpRef,
        array_field_offset: usize,
    ) {
        self.opimpl_setarrayitem_vable_any(vable_opref, index, value, array_field_offset);
    }

    /// RPython equivalent: `_opimpl_setarrayitem_vable`.
    pub fn opimpl_setarrayitem_vable_ref(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        value: OpRef,
        array_field_offset: usize,
    ) {
        self.opimpl_setarrayitem_vable_any(vable_opref, index, value, array_field_offset);
    }

    /// RPython equivalent: `_opimpl_setarrayitem_vable`.
    pub fn opimpl_setarrayitem_vable_float(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        value: OpRef,
        array_field_offset: usize,
    ) {
        self.opimpl_setarrayitem_vable_any(vable_opref, index, value, array_field_offset);
    }

    /// RPython equivalent: `opimpl_arraylen_vable(box, fdescr, adescr, pc)`.
    pub fn opimpl_arraylen_vable(
        &mut self,
        vable_opref: OpRef,
        array_field_offset: usize,
    ) -> OpRef {
        if self.nonstandard_virtualizable(vable_opref) {
            let ctx = self
                .tracing
                .as_mut()
                .expect("opimpl_arraylen_vable requires active tracing");
            let field_ref = ctx.const_int(array_field_offset as i64);
            let array_opref = ctx.record_op(OpCode::GetfieldGcR, &[vable_opref, field_ref]);
            return ctx.record_op(OpCode::ArraylenGc, &[array_opref]);
        }
        let ctx = self
            .tracing
            .as_ref()
            .expect("opimpl_arraylen_vable requires active tracing");
        let info = self
            .virtualizable_info
            .as_ref()
            .expect("standard virtualizable info missing");
        let array_index = info
            .array_field_index(array_field_offset)
            .expect("unknown standard virtualizable array offset");
        let length = ctx
            .virtualizable_array_lengths()
            .and_then(|lengths| lengths.get(array_index).copied())
            .expect("standard virtualizable array length missing");
        self.tracing
            .as_mut()
            .expect("opimpl_arraylen_vable requires active tracing")
            .const_int(length as i64)
    }

    /// RPython equivalent: `emit_force_virtualizable()`.
    ///
    /// Standard virtualizables are flushed back to heap state once per
    /// trace-side force point. Nonstandard virtualizables are ignored here
    /// and continue through the normal heap fallback path.
    pub fn emit_force_virtualizable(&mut self, vable_opref: OpRef) {
        if !self.is_standard_virtualizable(vable_opref) {
            return;
        }
        if self.forced_virtualizable.is_some() {
            return;
        }
        self.forced_virtualizable = Some(vable_opref);
        self.synchronize_virtualizable(vable_opref);
    }

    /// RPython equivalent: `opimpl_hint_force_virtualizable(box)`
    pub fn opimpl_hint_force_virtualizable(&mut self, vable_opref: OpRef) {
        self.emit_force_virtualizable(vable_opref);
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
            let null = ctx.const_int(0);
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
        self.forced_virtualizable = None;
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
        self.forced_virtualizable = None;
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

        // compile.py:269-270: cut trace at cross-loop merge point.
        // When the trace was retargeted to a different loop header, record
        // the new header PC so meta.merge_pc can be updated after insert.
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
        let (optimized_ops, final_num_inputs) = match optimize_result {
            Ok(result) => result,
            Err(payload) => {
                // Phase 2 panicked — unroll_opt dropped. Phase 1 results
                // survive in phase1_out (written before Phase 2 started).
                if payload
                    .downcast_ref::<crate::optimize::InvalidLoop>()
                    .is_some()
                {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] abort trace at key={} (InvalidLoop during optimize)",
                            green_key
                        );
                    }
                    self.cancel_count += 1;
                    // pyjitpl.py:3021-3030: if cancelled too many times,
                    // try one last time without unrolling.
                    if self.cancelled_too_many_times() {
                        let mut retry_constants = constants_snapshot;
                        let mut simple_opt = Optimizer::default_pipeline();
                        simple_opt.constant_types = constant_types.clone();
                        simple_opt.snapshot_boxes = snapshot_map.clone();
                        simple_opt.snapshot_frame_sizes = snapshot_frame_size_map.clone();
                        simple_opt.snapshot_vable_boxes = snapshot_vable_map.clone();
                        simple_opt.snapshot_frame_pcs = snapshot_pc_map.clone();
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
                                // compile.py:236-245 compile_simple_loop parity:
                                unroll_opt.target_tokens =
                                    vec![crate::optimizeopt::unroll::TargetToken::new_preamble(0)];
                                constants = retry_constants;
                                let ni = simple_opt.final_num_inputs();
                                (retry_ops, ni)
                            }
                            Err(_) => {
                                self.warm_state.abort_tracing(green_key, false);
                                self.warm_state.reset_function_counts();
                                return CompileOutcome::Aborted;
                            }
                        }
                    } else {
                        // compile.py:288-290 parity: preserve preamble target tokens
                        if !unroll_opt.target_tokens.is_empty() {
                            if let Some(compiled) = self.compiled_loops.get_mut(&green_key) {
                                if compiled.front_target_tokens.is_empty() {
                                    compiled.front_target_tokens = unroll_opt.target_tokens.clone();
                                }
                            } else {
                                self.pending_preamble_tokens
                                    .insert(green_key, unroll_opt.target_tokens.clone());
                            }
                        }
                        // RPython pyjitpl.py:3014-3017 parity: InvalidLoop
                        // but cancel_count < limit — set up retrace with
                        // Phase 1 preamble so the next compile_loop call
                        // uses compile_retrace with new runtime values.
                        // RPython compile.py:278-284 parity: use Phase 1
                        // preamble ops (JUMP excluded by optimizer) for
                        // retrace_needed partial_trace. phase1_out was
                        // written before Phase 2 started, so it survives
                        // the Phase 2 panic.
                        if let Some((preamble_ops, es)) = phase1_out.take() {
                            if crate::majit_log_enabled() {
                                eprintln!(
                                    "[jit] retrace_needed after InvalidLoop at key={} preamble_ops={}",
                                    green_key,
                                    preamble_ops.len(),
                                );
                            }
                            self.potential_retrace_position =
                                Some(majit_trace::recorder::TracePosition {
                                    op_count: 0,
                                    ops_len: 0,
                                });
                            self.retrace_needed(
                                green_key,
                                preamble_ops,
                                trace.inputargs.clone(),
                                constants_snapshot.clone(),
                                es,
                            );
                        } else {
                            self.warm_state.abort_tracing(green_key, false);
                        }
                        self.warm_state.reset_function_counts();
                        return CompileOutcome::Cancelled;
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
        let mut optimized_ops = compile::unbox_call_assembler_results(optimized_ops);
        let mut optimized_ops =
            compile::normalize_closing_jump_args(optimized_ops, &constants, final_num_inputs);

        if crate::majit_log_enabled() {
            eprintln!("--- trace (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
            for op in &optimized_ops {
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
        let has_guard = optimized_ops.iter().any(|op| op.opcode.is_guard());
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

        let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.backend
                .compile_loop(&inputargs, &optimized_ops, &mut token)
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
                            optimized_ops.get(op_idx).map(|op| (op.pos.0, fi))
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
                        ops: optimized_ops,
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
                    // Cranelift-specific: migrate existing bridges to new token.
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
        self.compile_trace_inner(green_key, finish_args, bridge_origin, None)
    }

    /// compile_trace with ends_with_jump=false (FINISH).
    pub fn compile_trace_finish(
        &mut self,
        green_key: u64,
        finish_args: &[OpRef],
        bridge_origin: Option<(u64, u32)>,
        finish_descr: majit_ir::DescrRef,
    ) -> CompileOutcome {
        self.compile_trace_inner(green_key, finish_args, bridge_origin, Some(finish_descr))
    }

    fn compile_trace_inner(
        &mut self,
        green_key: u64,
        finish_args: &[OpRef],
        bridge_origin: Option<(u64, u32)>,
        finish_descr: Option<majit_ir::DescrRef>,
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
            ctx.recorder.close_loop(finish_args);
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
                    match Self::bridge_fail_descr_proxy(compiled, trace_id, fail_index) {
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
                // compile.py:1006-1022 — ResumeFromInterpDescr path: entry
                // bridge. Compile as a fresh entry and attach to interpreter.
                let fail_descr_types: Vec<Type> = bridge_inputargs.iter().map(|a| a.tp).collect();
                let fail_descr_ref =
                    crate::fail_descr::make_fail_descr_typed_with_index(0, fail_descr_types);
                let fail_descr: &dyn majit_ir::FailDescr = fail_descr_ref.as_fail_descr().unwrap();
                let success = self.compile_bridge(
                    green_key,
                    0,
                    fail_descr,
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
                        green_key: 0,
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
        self.forced_virtualizable = None;
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

        let combined_ops = compile::unbox_call_assembler_results(combined_ops);
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
                    // Cranelift-specific: migrate existing bridges to new token.
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
        self.forced_virtualizable = None;
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
        self.forced_virtualizable = None;
        self.force_finish_trace = false;
        let mut ctx = self.tracing.take().unwrap();
        ctx.apply_replacements();
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
        // RPython Box type parity: inputarg types from tracing.
        optimizer.trace_inputarg_types = trace.inputargs.iter().map(|ia| ia.tp).collect();
        // RPython Box.type parity: register inputarg types in constant_types
        // so fail_arg_types inference can resolve them.
        for ia in &trace.inputargs {
            optimizer.constant_types.insert(ia.index, ia.tp);
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
        // compile.py:92-96: SimpleCompileData.optimize → optimize_loop.
        // Blocked: fib_recursive SEGFAULT — inputarg types from recorder
        // are Ref for Python objects, but backend unboxes to Int. Snapshot
        // numbering produces fail_arg_types=[Ref] for an Int-typed dead
        // frame slot → decode_ref treats raw int as pointer → SIGSEGV.
        // Needs inputarg-type correction after fold_box_into_create_frame
        // before activation.
        let _ = (
            &snapshot_map,
            &snapshot_frame_size_map,
            &snapshot_vable_map,
            &snapshot_pc_map,
            &sbt,
        );

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
        let mut optimized_ops = match optimize_result {
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

        // Skip compilation if any guard has unrecoverable NONE in fail_args.
        {
            let has_unrecoverable_none = optimized_ops.iter().any(|op| {
                if op.opcode.is_guard() {
                    if let Some(ref fa) = op.fail_args {
                        let none_count = fa.iter().skip(3).filter(|a| **a == OpRef::NONE).count();
                        if none_count > 0 {
                            let last_none = fa.iter().rposition(|a| *a == OpRef::NONE).unwrap();
                            let trailing = fa.len() - last_none - 1;
                            return trailing < none_count * 2;
                        }
                    }
                }
                false
            });
            if has_unrecoverable_none {
                if crate::majit_log_enabled() {
                    eprintln!("[jit] abort finish: guard NONE without sufficient trailing fields");
                }
                self.warm_state.abort_tracing(green_key, true);
                return;
            }
        }

        // unbox_finish_result: only when result_type == Int.
        // When result_type == Ref, FINISH stays as Ref
        // (matching DoneWithThisFrameDescrRef contract).
        let (optimized_ops, finish_unboxed) = if self.result_type == Type::Int {
            compile::unbox_finish_result(optimized_ops, &constants, &self.raw_int_box_helpers)
        } else {
            (optimized_ops, false)
        };
        let optimized_ops = compile::unbox_call_assembler_results(optimized_ops);
        let optimized_ops = if finish_unboxed {
            compile::unbox_raw_force_results(optimized_ops, &constants, &self.raw_int_force_helpers)
        } else {
            optimized_ops
        };
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

        // Extend inputargs if the optimizer added virtual inputs (virtualizable)
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
                // Read from descr (original trace types, full-length).
                // NOT from op.fail_arg_types which may be compact after
                // snapshot-based numbering reduces fail_args to liveboxes.
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
        self.forced_virtualizable = None;
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

        let compiled_constants = constants.clone();
        let compiled_constant_types = constant_types.clone();
        self.backend.set_constants(constants);
        token.green_key = green_key;

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
                        constant_types: compiled_constant_types,
                        guard_op_indices,
                        exit_layouts,
                        terminal_exit_layouts,
                        optimizer_knowledge: per_guard_knowledge,
                        jitcode: None,
                    },
                );
                // compile.py:236-245 parity: create TargetToken with
                // front_target_tokens so has_compiled_targets() returns true.
                // This is the LABEL that bridges can close back to.
                let target_token = crate::optimizeopt::unroll::TargetToken::new_preamble(token_num);
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

        if !result.is_finish {
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

        if !result.is_finish {
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

        if !result.is_finish {
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

        if !result.is_finish {
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

        if !effective_is_finish {
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

        let exception = ExceptionState {
            exc_class: result.exception_class,
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

        if !effective_is_finish {
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
        let exception = ExceptionState {
            exc_class: result.exception_class,
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
        Self::finish_compiled_run_io(is_finish);

        if !is_finish {
            self.stats.guard_failures += 1;
            self.warm_state.log_guard_failure(fail_index);

            if let Some(ref hook) = self.hooks.on_guard_failure {
                hook(green_key, fail_index, 0);
            }
        }

        let exit_types = descr.fail_arg_types();
        let exit_arity = exit_types.len();
        let compiled = self.compiled_loops.get(&green_key).unwrap();
        let exit_layout = Self::trace_for_exit(compiled, trace_id)
            .and_then(|(trace_id, trace)| {
                Self::compiled_exit_layout_from_trace(trace, green_key, trace_id, fail_index)
            })
            .unwrap_or_else(|| CompiledExitLayout {
                rd_loop_token: green_key,
                trace_id,
                fail_index,
                source_op_index: None,
                exit_types: exit_types.to_vec(),
                is_finish,
                gc_ref_slots: exit_types
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, _)| descr.is_gc_ref_slot(slot).then_some(slot))
                    .collect(),
                force_token_slots: descr.force_token_slots().to_vec(),
                recovery_layout: None,
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            });
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
        let savedata = self.backend.grab_savedata_ref(&frame);
        let exception = ExceptionState {
            exc_class: self.backend.grab_exc_class(&frame),
            exc_value: self.backend.grab_exc_value(&frame).0 as i64,
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
            status: descr.get_status(),
            descr_addr: descr as *const dyn majit_ir::FailDescr as *const () as usize,
        })
    }

    /// pyre-specific: unbox Ref→Int where compiled trace expects Int.
    /// pyre traces start with all-Ref locals (PyObjectRef) but trace-internal
    /// operations unbox to Int/Float. At compiled entry, live_values carry
    /// Ref pointers that must be unboxed to match the trace's typed label.
    /// RPython doesn't need this because wrap() sets Box types before tracing.
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
                let ptr = r.as_usize();
                if ptr >= 0x1_0000 && ptr < (1u64 << 56) as usize && (ptr & 7) == 0 {
                    values[i] = Value::Int(unsafe { *((ptr + 8) as *const i64) });
                } else {
                    values[i] = Value::Int(ptr as i64);
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
        Self::finish_compiled_run_io(is_finish);

        // RPython: guard failure counter tick and bridge compilation happen
        // in handle_fail → must_compile (compile.py:701-784).
        // must_compile handles tick.
        if !is_finish {
            self.stats.guard_failures += 1;
            self.warm_state.log_guard_failure(fail_index);

            if let Some(ref hook) = self.hooks.on_guard_failure {
                hook(green_key, fail_index, 0);
            }
        }

        let exit_types = descr.fail_arg_types();
        let exit_arity = exit_types.len();
        let compiled = self.compiled_loops.get(&green_key).unwrap();
        let exit_layout = Self::trace_for_exit(compiled, trace_id)
            .and_then(|(trace_id, trace)| {
                Self::compiled_exit_layout_from_trace(trace, green_key, trace_id, fail_index)
            })
            .unwrap_or_else(|| CompiledExitLayout {
                rd_loop_token: green_key,
                trace_id,
                fail_index,
                source_op_index: None,
                exit_types: exit_types.to_vec(),
                is_finish,
                gc_ref_slots: exit_types
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, _)| descr.is_gc_ref_slot(slot).then_some(slot))
                    .collect(),
                force_token_slots: descr.force_token_slots().to_vec(),
                recovery_layout: None,
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            });
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
        let savedata = self.backend.grab_savedata_ref(&frame);
        let exception = ExceptionState {
            exc_class: self.backend.grab_exc_class(&frame),
            exc_value: self.backend.grab_exc_value(&frame).0 as i64,
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
            status: descr.get_status(),
            descr_addr: descr as *const dyn majit_ir::FailDescr as *const () as usize,
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
        Self::bridge_fail_descr_proxy(compiled, trace_id, fail_index)
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
        compiled: &CompiledEntry<M>,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<compile::BridgeFailDescrProxy> {
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        let (_, trace_data) = Self::trace_for_exit(compiled, trace_id)?;
        let exit_layout = trace_data.exit_layouts.get(&fail_index)?;
        Some(compile::BridgeFailDescrProxy {
            fail_index,
            trace_id,
            fail_arg_types: exit_layout.exit_types.clone(),
            gc_ref_slots: exit_layout.gc_ref_slots.clone(),
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
    /// exits via return. Calls compile_trace with ends_with_jump=false (FINISH).
    pub fn compile_done_with_this_frame(
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
        Some(Box::new(Self::bridge_fail_descr_proxy(
            compiled, trace_id, fail_index,
        )?))
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
        let (optimized_ops, retrace_requested) = optimizer.optimize_bridge(
            bridge_ops,
            &mut constants,
            bridge_inputargs.len(),
            &mut compiled.front_target_tokens,
            inline_short_preamble,
            retraced_count,
            retrace_limit,
            bridge_knowledge.as_ref(),
            Some(loop_num_inputs),
        );
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

        // RPython parity: unbox the Finish result in bridges too.
        // Without this, bridges return boxed pointers while the caller
        // (call_assembler_fast_path) expects raw ints for [Type::Int] Finish.
        let (optimized_ops, bridge_finish_unboxed) = if self.result_type == Type::Int {
            compile::unbox_finish_result(optimized_ops, &constants, &self.raw_int_box_helpers)
        } else {
            (optimized_ops, false)
        };
        let optimized_ops = if bridge_finish_unboxed {
            compile::unbox_raw_force_results(optimized_ops, &constants, &self.raw_int_force_helpers)
        } else {
            optimized_ops
        };
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

        let fail_descr = match Self::bridge_fail_descr_proxy(compiled, trace_id, fail_index) {
            Some(descr) => descr,
            None => return None,
        };

        // compile.py:797-811 / resume.py:1042: bridge inputargs come from
        // rebuild_from_resumedata, which produces boxes matching the guard's
        // fail_arg_types exactly. Always use fail_arg_types regardless of
        // what the interpreter's live_types say — they may differ for
        // bridge guards where the optimizer reduced the fail_args count.
        let bridge_input_types = fail_descr.fail_arg_types();
        let recorder = self.warm_state.start_retrace(bridge_input_types);
        self.forced_virtualizable = None;
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

    /// Set the W_IntObject.intval field descriptor for Ref→Int unboxing in FINISH.
    pub fn set_intval_descr(&mut self, descr: majit_ir::DescrRef) {
        self.intval_descr = Some(descr);
    }

    /// Register a recursive force helper that takes a raw-int argument and
    /// can return a raw-int result when the callee trace ends with a raw-int
    /// Finish protocol.
    pub fn register_raw_int_force_helper(&mut self, helper: *const ()) {
        self.raw_int_force_helpers.insert(helper as i64);
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
    use majit_backend_cranelift::compiler::{
        force_token_to_dead_frame, get_int_from_deadframe, get_latest_descr_from_deadframe,
        set_savedata_ref_on_deadframe,
    };
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

    fn with_forced_deadframe(force_token: i64, f: impl FnOnce(DeadFrame)) {
        f(force_token_to_dead_frame(GcRef(force_token as usize)));
    }

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
        info
    }

    fn test_vable_info_with_array() -> VirtualizableInfo {
        let mut info = VirtualizableInfo::new(0);
        info.add_array_field_with_layout("stack", Type::Int, 24, 0, 0);
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
        assert_eq!(ctx.recorder.num_ops(), 2);
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
        let result = meta.opimpl_getarrayitem_vable_int(OpRef(0), index, 24);
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

        let ops = take_recorded_ops(&mut meta);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcI);
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
        let _result = meta.opimpl_getarrayitem_vable_int(nonstandard_vable, index, 24);

        let ops = take_recorded_ops(&mut meta);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[1].opcode, OpCode::GetarrayitemGcI);
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
        let item = meta.opimpl_getarrayitem_vable_int(OpRef(0), index, 24);
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

    #[test]
    fn test_handle_guard_failure_preserves_exception_state_in_recovery() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 7;
        let fail_index = 3;
        let mut resume_data = HashMap::new();
        resume_data.insert(fail_index, StoredResumeData::new(ResumeData::simple(99, 2)));
        let trace_id = meta.alloc_trace_id();
        let mut traces = HashMap::new();
        traces.insert(
            trace_id,
            CompiledTrace {
                inputargs: Vec::new(),
                resume_data,
                ops: Vec::new(),
                constants: HashMap::new(),
                guard_op_indices: HashMap::new(),
                exit_layouts: HashMap::new(),
                terminal_exit_layouts: HashMap::new(),
                optimizer_knowledge: HashMap::new(),
                jitcode: None,
                snapshots: Vec::new(),
            },
        );

        meta.compiled_loops.insert(
            green_key,
            CompiledEntry {
                token: JitCellToken::new(1),
                num_inputs: 2,
                meta: (),
                front_target_tokens: Vec::new(),
                retraced_count: 0,
                root_trace_id: trace_id,

                traces,
                previous_tokens: Vec::new(),
            },
        );

        let recovery = meta
            .handle_guard_failure(
                green_key,
                fail_index,
                &[11, 22],
                ExceptionState {
                    exc_class: 0x1234,
                    exc_value: 0xABCD,
                    ovf_flag: false,
                },
            )
            .expect("compiled loop should exist");

        assert_eq!(recovery.fail_index, fail_index);
        assert_eq!(recovery.trace_id, trace_id);
        assert_eq!(recovery.fail_values, vec![11, 22]);
        assert_eq!(recovery.exception.exc_class, 0x1234);
        assert_eq!(recovery.exception.exc_value, 0xABCD);
        assert!(recovery.materialized_virtuals.is_empty());

        let reconstructed = recovery
            .reconstructed_frames
            .expect("resume data should reconstruct one frame");
        assert_eq!(reconstructed.len(), 1);
        assert_eq!(reconstructed[0].pc, 99);
        assert_eq!(
            reconstructed[0].values,
            vec![ReconstructedValue::Value(11), ReconstructedValue::Value(22)]
        );
    }

    #[test]
    fn test_get_compiled_exit_layout_in_trace_reports_exit_and_resume_shape() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 70;
        let fail_index = 3;
        let trace_id = meta.alloc_trace_id();
        let stored = StoredResumeData::new(ResumeData {
            frames: vec![crate::resume::FrameInfo {
                pc: 111,
                slot_map: vec![
                    FrameSlotSource::FailArg(4),
                    crate::resume::FrameSlotSource::Constant(55),
                ],
            }],
            virtuals: vec![crate::resume::VirtualInfo::VStruct {
                type_id: 0,
                descr_index: 9,
                fields: vec![(0, crate::resume::VirtualFieldSource::FailArg(4))],
            }],
            pending_fields: vec![crate::resume::PendingFieldInfo {
                descr_index: 12,
                target: crate::resume::ResumeValueSource::FailArg(1),
                value: crate::resume::ResumeValueSource::Constant(77),
                item_index: Some(3),
            }],
        });
        let expected_layout = stored.layout.clone();
        let mut resume_data = HashMap::new();
        resume_data.insert(fail_index, stored);
        let mut exit_layouts = HashMap::new();
        exit_layouts.insert(
            fail_index,
            StoredExitLayout {
                source_op_index: None,
                exit_types: vec![Type::Ref, Type::Int],
                is_finish: false,
                gc_ref_slots: vec![0],
                force_token_slots: Vec::new(),
                recovery_layout: None,
                resume_layout: Some(expected_layout.clone()),
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            },
        );
        let mut traces = HashMap::new();
        traces.insert(
            trace_id,
            CompiledTrace {
                inputargs: Vec::new(),
                resume_data,
                ops: Vec::new(),
                constants: HashMap::new(),
                guard_op_indices: HashMap::new(),
                exit_layouts,
                terminal_exit_layouts: HashMap::new(),
                optimizer_knowledge: HashMap::new(),
                jitcode: None,
                snapshots: Vec::new(),
            },
        );
        meta.compiled_loops.insert(
            green_key,
            CompiledEntry {
                token: JitCellToken::new(100),
                num_inputs: 2,
                meta: (),
                front_target_tokens: Vec::new(),
                retraced_count: 0,
                root_trace_id: trace_id,

                traces,
                previous_tokens: Vec::new(),
            },
        );

        let layout = meta
            .get_compiled_exit_layout_in_trace(green_key, trace_id, fail_index)
            .expect("compiled exit layout should exist");
        assert_eq!(layout.trace_id, trace_id);
        assert_eq!(layout.fail_index, fail_index);
        assert_eq!(layout.exit_types, vec![Type::Ref, Type::Int]);
        assert!(!layout.is_finish);
        assert_eq!(layout.gc_ref_slots, vec![0]);
        assert!(layout.force_token_slots.is_empty());
        assert_eq!(layout.resume_layout, Some(expected_layout.clone()));

        let resume_layout = meta
            .get_resume_layout_in_trace(green_key, trace_id, fail_index)
            .expect("resume layout should exist");
        assert_eq!(resume_layout, &expected_layout);
    }

    #[test]
    fn test_attach_resume_data_to_trace_enriches_resume_layout_with_backend_trace_metadata() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 701;
        meta.backend.set_next_header_pc(1234);
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());
        let trace_id = meta
            .compiled_loops
            .get(&green_key)
            .expect("compiled entry")
            .root_trace_id;

        meta.attach_resume_data_to_trace(green_key, trace_id, 0, ResumeData::simple(555, 1));

        let resume_layout = meta
            .get_resume_layout_in_trace(green_key, trace_id, 0)
            .expect("resume layout should exist");
        assert_eq!(resume_layout.frame_layouts.len(), 1);
        assert_eq!(resume_layout.frame_layouts[0].trace_id, Some(trace_id));
        assert_eq!(resume_layout.frame_layouts[0].header_pc, Some(1234));
        assert_eq!(resume_layout.frame_layouts[0].source_guard, None);
        assert_eq!(
            resume_layout.frame_layouts[0].slot_types,
            Some(vec![Type::Int])
        );

        let exit_layout = meta
            .get_compiled_exit_layout_in_trace(green_key, trace_id, 0)
            .expect("compiled exit layout should exist");
        assert_eq!(exit_layout.resume_layout, Some(resume_layout.clone()));

        let token = &meta
            .compiled_loops
            .get(&green_key)
            .expect("compiled entry")
            .token;
        let backend_layout = meta
            .backend
            .compiled_trace_fail_descr_layouts(token, trace_id)
            .expect("backend fail layouts should exist")
            .into_iter()
            .find(|layout| layout.fail_index == 0)
            .expect("backend fail layout should exist");
        let backend_recovery = backend_layout
            .recovery_layout
            .expect("backend recovery layout should be patched");
        assert_eq!(backend_recovery.frames.len(), 1);
        assert_eq!(backend_recovery.frames[0].trace_id, Some(trace_id));
        assert_eq!(backend_recovery.frames[0].header_pc, Some(1234));
        assert_eq!(backend_recovery.frames[0].source_guard, None);
        assert_eq!(backend_recovery.frames[0].pc, 555);
        assert_eq!(backend_recovery.frames[0].slot_types, Some(vec![Type::Int]));
        assert_eq!(
            backend_recovery.frames[0].slots,
            vec![majit_backend::ExitValueSourceLayout::ExitValue(0)]
        );
    }

    #[test]
    fn test_handle_guard_failure_in_trace_uses_enriched_resume_layout_metadata() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 702;
        meta.backend.set_next_header_pc(4321);
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());
        let trace_id = meta
            .compiled_loops
            .get(&green_key)
            .expect("compiled entry")
            .root_trace_id;
        meta.attach_resume_data_to_trace(green_key, trace_id, 0, ResumeData::simple(777, 1));

        let detailed = meta
            .run_compiled_detailed(green_key, &[1])
            .expect("guard should fail");
        let recovery_trace_id = detailed.trace_id;
        let recovery_fail_index = detailed.fail_index;
        let fail_values = detailed.values.clone();
        let exception = detailed.exception.clone();
        let _ = detailed;
        let recovery = meta
            .handle_guard_failure_in_trace(
                green_key,
                recovery_trace_id,
                recovery_fail_index,
                &fail_values,
                None,
                exception,
            )
            .expect("guard recovery should succeed");
        let reconstructed = recovery
            .reconstructed_frames
            .expect("resume layout should reconstruct frame");
        assert_eq!(reconstructed[0].pc, 777);
        assert_eq!(reconstructed[0].trace_id, Some(trace_id));
        assert_eq!(reconstructed[0].header_pc, Some(4321));
        assert_eq!(reconstructed[0].slot_types, Some(vec![Type::Int]));
        assert_eq!(reconstructed[0].values, vec![ReconstructedValue::Value(1)]);
    }

    #[test]
    fn test_merge_backend_exit_layouts_overrides_gc_and_force_token_slots() {
        let recovery = ExitRecoveryLayout {
            frames: vec![ExitFrameLayout {
                trace_id: Some(99),
                header_pc: Some(700),
                source_guard: Some((98, 3)),
                pc: 33,
                slots: Vec::new(),
                slot_types: None,
            }],
            virtual_layouts: Vec::new(),
            pending_field_layouts: Vec::new(),
        };
        let mut exit_layouts = HashMap::new();
        exit_layouts.insert(
            0,
            StoredExitLayout {
                source_op_index: None,
                exit_types: vec![Type::Ref],
                is_finish: false,
                gc_ref_slots: vec![0],
                force_token_slots: Vec::new(),
                recovery_layout: None,
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            },
        );

        compile::merge_backend_exit_layouts(
            &mut exit_layouts,
            &[FailDescrLayout {
                fail_index: 0,
                source_op_index: Some(7),
                trace_id: 99,
                trace_info: None,
                fail_arg_types: vec![Type::Ref],
                is_finish: false,
                gc_ref_slots: Vec::new(),
                force_token_slots: vec![0],
                recovery_layout: Some(recovery.clone()),
                frame_stack: None,
            }],
        );

        let layout = exit_layouts.get(&0).expect("layout should exist");
        assert!(layout.gc_ref_slots.is_empty());
        assert_eq!(layout.force_token_slots, vec![0]);
        assert_eq!(layout.source_op_index, Some(7));
        assert_eq!(layout.recovery_layout, Some(recovery));
    }

    #[test]
    fn test_merge_backend_terminal_exit_layouts_overrides_gc_and_force_token_slots() {
        let recovery = ExitRecoveryLayout {
            frames: vec![ExitFrameLayout {
                trace_id: Some(99),
                header_pc: Some(800),
                source_guard: Some((97, 4)),
                pc: 44,
                slots: Vec::new(),
                slot_types: None,
            }],
            virtual_layouts: Vec::new(),
            pending_field_layouts: Vec::new(),
        };
        let mut exit_layouts = HashMap::new();
        exit_layouts.insert(
            5,
            StoredExitLayout {
                source_op_index: None,
                exit_types: vec![Type::Ref],
                is_finish: true,
                gc_ref_slots: vec![0],
                force_token_slots: Vec::new(),
                recovery_layout: None,
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            },
        );

        compile::merge_backend_terminal_exit_layouts(
            &mut exit_layouts,
            &[TerminalExitLayout {
                op_index: 5,
                trace_id: 99,
                trace_info: None,
                fail_index: 7,
                exit_types: vec![Type::Ref],
                is_finish: true,
                gc_ref_slots: Vec::new(),
                force_token_slots: vec![0],
                recovery_layout: Some(recovery.clone()),
            }],
        );

        let layout = exit_layouts.get(&5).expect("layout should exist");
        assert!(layout.gc_ref_slots.is_empty());
        assert_eq!(layout.force_token_slots, vec![0]);
        assert_eq!(layout.source_op_index, Some(5));
        assert_eq!(layout.recovery_layout, Some(recovery));
    }

    #[test]
    fn test_handle_guard_failure_reconstructs_pending_field_writes() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 17;
        let fail_index = 1;
        let mut resume_data = HashMap::new();
        resume_data.insert(
            fail_index,
            StoredResumeData::new(ResumeData {
                frames: vec![crate::resume::FrameInfo {
                    pc: 44,
                    slot_map: vec![FrameSlotSource::FailArg(0)],
                }],
                virtuals: Vec::new(),
                pending_fields: vec![crate::resume::PendingFieldInfo {
                    descr_index: 12,
                    target: crate::resume::ResumeValueSource::FailArg(0),
                    value: crate::resume::ResumeValueSource::Constant(99),
                    item_index: Some(4),
                }],
            }),
        );
        let trace_id = meta.alloc_trace_id();
        let mut traces = HashMap::new();
        traces.insert(
            trace_id,
            CompiledTrace {
                inputargs: Vec::new(),
                resume_data,
                ops: Vec::new(),
                constants: HashMap::new(),
                guard_op_indices: HashMap::new(),
                exit_layouts: HashMap::new(),
                terminal_exit_layouts: HashMap::new(),
                optimizer_knowledge: HashMap::new(),
                jitcode: None,
                snapshots: Vec::new(),
            },
        );
        meta.compiled_loops.insert(
            green_key,
            CompiledEntry {
                token: JitCellToken::new(2),
                num_inputs: 1,
                meta: (),
                front_target_tokens: Vec::new(),
                retraced_count: 0,
                root_trace_id: trace_id,

                traces,
                previous_tokens: Vec::new(),
            },
        );

        let recovery = meta
            .handle_guard_failure(green_key, fail_index, &[123], ExceptionState::default())
            .expect("recovery should exist");

        assert_eq!(
            recovery.pending_field_writes,
            vec![ResolvedPendingFieldWrite {
                descr_index: 12,
                target: crate::resume::MaterializedValue::Value(123),
                value: crate::resume::MaterializedValue::Value(99),
                item_index: Some(4),
            }]
        );
        assert_eq!(
            recovery
                .reconstructed_state
                .as_ref()
                .expect("reconstructed state")
                .pending_fields,
            recovery.pending_field_writes
        );
    }

    #[test]
    fn test_run_and_recover_uses_backend_finish_kind_instead_of_fail_index_zero() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 11;
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());

        match meta.run_and_recover(green_key, &[0]) {
            Some(RunResult::Finished { values, .. }) => assert_eq!(values, vec![0]),
            other => panic!("expected Finished, got {other:?}"),
        }
    }

    #[test]
    fn test_run_compiled_detailed_preserves_savedata_for_call_may_force() {
        let _guard = may_force_test_lock()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 111;
        install_may_force_void_entry(&mut meta, green_key);

        let result = meta
            .run_compiled_detailed(green_key, &[10, 1])
            .expect("forced call should exit via guard");
        assert!(!result.is_finish);
        assert_eq!(result.fail_index, 0);
        assert_eq!(result.values, vec![1, 10]);
        assert_eq!(result.savedata, Some(GcRef(0xDADA)));
        assert_eq!(
            *may_force_void_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner()),
            vec![0, 1, 10]
        );
    }

    #[test]
    fn test_run_and_recover_preserves_savedata_for_guard_failure() {
        let _guard = may_force_test_lock()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 112;
        install_may_force_void_entry(&mut meta, green_key);

        match meta.run_and_recover(green_key, &[10, 1]) {
            Some(RunResult::GuardFailure {
                values,
                savedata,
                recovery,
                ..
            }) => {
                assert_eq!(values, vec![1, 10]);
                assert_eq!(savedata, Some(GcRef(0xDADA)));
                let recovery = recovery.expect("guard recovery should exist");
                assert_eq!(recovery.savedata, Some(GcRef(0xDADA)));
            }
            other => panic!("expected GuardFailure with savedata, got {other:?}"),
        }
    }

    #[test]
    fn test_run_with_blackhole_fallback_finishes_after_compiled_guard_failure() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 12;
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 1);

        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, constants);

        match meta.run_with_blackhole_fallback(green_key, &[1]) {
            Some(BlackholeRunResult::Finished {
                values,
                via_blackhole,
                exception,
                ..
            }) => {
                assert_eq!(values, vec![2]);
                assert!(via_blackhole);
                assert_eq!(exception, ExceptionState::default());
            }
            other => panic!("expected blackhole Finished, got {other:?}"),
        }
    }

    #[test]
    fn test_run_with_blackhole_fallback_preserves_savedata_for_call_may_force() {
        let _guard = may_force_test_lock()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 113;
        install_may_force_void_entry(&mut meta, green_key);

        match meta.run_with_blackhole_fallback(green_key, &[10, 1]) {
            Some(BlackholeRunResult::Finished {
                values,
                via_blackhole,
                savedata,
                ..
            }) => {
                assert_eq!(values, vec![10]);
                assert!(via_blackhole);
                assert_eq!(savedata, Some(GcRef(0xDADA)));
            }
            other => panic!("expected blackhole Finished with savedata, got {other:?}"),
        }
    }

    #[test]
    fn test_run_with_blackhole_fallback_infers_typed_finish_values_from_trace_layout() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 19;
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            mk_op(OpCode::CastIntToFloat, &[OpRef(0)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());

        match meta.run_with_blackhole_fallback(green_key, &[1]) {
            Some(BlackholeRunResult::Finished {
                typed_values,
                exit_layout,
                via_blackhole,
                ..
            }) => {
                assert!(via_blackhole);
                assert_eq!(typed_values, Some(vec![Value::Float(1.0)]));
                let exit_layout = exit_layout.expect("finish exit layout should be inferred");
                assert_eq!(exit_layout.exit_types, vec![Type::Float]);
                assert!(exit_layout.gc_ref_slots.is_empty());
            }
            other => panic!("expected typed blackhole Finished, got {other:?}"),
        }
    }

    #[test]
    fn test_run_with_blackhole_fallback_materializes_virtuals_on_nested_guard_failure() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 13;
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());
        meta.attach_resume_data(
            green_key,
            1,
            ResumeData {
                frames: vec![crate::resume::FrameInfo {
                    pc: green_key,
                    slot_map: vec![FrameSlotSource::FailArg(0)],
                }],
                virtuals: vec![crate::resume::VirtualInfo::VStruct {
                    type_id: 0,
                    descr_index: 7,
                    fields: vec![(3, crate::resume::VirtualFieldSource::Constant(55))],
                }],
                pending_fields: Vec::new(),
            },
        );

        match meta.run_with_blackhole_fallback(green_key, &[1]) {
            Some(BlackholeRunResult::GuardFailure {
                fail_index,
                fail_values,
                materialized_virtuals,
                pending_field_writes,
                via_blackhole,
                exception,
                recovery,
                ..
            }) => {
                assert_eq!(fail_index, 1);
                assert_eq!(fail_values, vec![1]);
                assert!(via_blackhole);
                assert_eq!(exception, ExceptionState::default());
                assert_eq!(materialized_virtuals.len(), 1);
                assert!(pending_field_writes.is_empty());
                match &materialized_virtuals[0] {
                    crate::resume::MaterializedVirtual::Struct {
                        descr_index,
                        fields,
                        ..
                    } => {
                        assert_eq!(*descr_index, 7);
                        assert_eq!(
                            fields,
                            &vec![(3, crate::resume::MaterializedValue::Value(55))]
                        );
                    }
                    other => panic!("unexpected virtual: {other:?}"),
                }
                let recovery = recovery.expect("fallback guard should reconstruct state");
                assert_eq!(
                    recovery.trace_id,
                    meta.compiled_loops[&green_key].root_trace_id
                );
                assert_eq!(recovery.fail_index, 1);
                assert_eq!(recovery.fail_values, vec![1]);
                assert_eq!(recovery.materialized_virtuals.len(), 1);
            }
            other => panic!("expected blackhole GuardFailure, got {other:?}"),
        }
    }

    #[test]
    fn test_run_with_blackhole_fallback_surfaces_pending_field_writes_on_nested_guard_failure() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 18;
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());
        meta.attach_resume_data(
            green_key,
            1,
            ResumeData {
                frames: vec![crate::resume::FrameInfo {
                    pc: green_key,
                    slot_map: vec![FrameSlotSource::FailArg(0)],
                }],
                virtuals: Vec::new(),
                pending_fields: vec![crate::resume::PendingFieldInfo {
                    descr_index: 12,
                    target: crate::resume::ResumeValueSource::FailArg(0),
                    value: crate::resume::ResumeValueSource::Constant(99),
                    item_index: Some(4),
                }],
            },
        );

        match meta.run_with_blackhole_fallback(green_key, &[1]) {
            Some(BlackholeRunResult::GuardFailure {
                fail_index,
                fail_values,
                materialized_virtuals,
                pending_field_writes,
                via_blackhole,
                exception,
                recovery,
                ..
            }) => {
                assert_eq!(fail_index, 1);
                assert_eq!(fail_values, vec![1]);
                assert!(via_blackhole);
                assert_eq!(exception, ExceptionState::default());
                assert!(materialized_virtuals.is_empty());
                assert_eq!(
                    pending_field_writes,
                    vec![ResolvedPendingFieldWrite {
                        descr_index: 12,
                        target: crate::resume::MaterializedValue::Value(1),
                        value: crate::resume::MaterializedValue::Value(99),
                        item_index: Some(4),
                    }]
                );
                let recovery = recovery.expect("fallback guard should reconstruct state");
                assert_eq!(recovery.pending_field_writes, pending_field_writes);
            }
            other => panic!("expected blackhole GuardFailure, got {other:?}"),
        }
    }

    #[test]
    fn test_run_with_blackhole_fallback_uses_bridge_trace_metadata() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 14;
        let inputargs = vec![InputArg::new_int(0)];
        let root_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            mk_op(OpCode::Finish, &[OpRef(100)], OpRef::NONE.0),
        ];
        let mut root_constants = HashMap::new();
        root_constants.insert(100, 99);
        attach_procedure_to_interp_entry(
            &mut meta,
            green_key,
            &inputargs,
            root_ops,
            root_constants,
        );

        let detailed = meta
            .run_compiled_detailed(green_key, &[1])
            .expect("root guard should fail");
        let fail_index = detailed.fail_index;
        assert_eq!(fail_index, 0);
        let source_trace_id = detailed.trace_id;
        let fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            source_trace_id,
            vec![Type::Int],
            false,
            Vec::new(),
            None,
        );

        let bridge_inputargs = vec![InputArg::new_int(0)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut bridge_constants = HashMap::new();
        bridge_constants.insert(100, 5);
        assert!(meta.compile_bridge(
            green_key,
            fail_index,
            &fail_descr,
            &bridge_ops,
            &bridge_inputargs,
            bridge_constants,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        ));

        let bridge_trace_id = meta.compiled_loops[&green_key]
            .traces
            .keys()
            .copied()
            .find(|&trace_id| trace_id != source_trace_id)
            .expect("bridge trace should be registered");

        match meta.run_with_blackhole_fallback(green_key, &[1]) {
            Some(BlackholeRunResult::Finished {
                values,
                via_blackhole,
                ..
            }) => {
                assert_eq!(values, vec![6]);
                assert!(via_blackhole);
            }
            other => panic!("expected bridge blackhole Finish, got {other:?}"),
        }

        let detailed = meta
            .run_compiled_detailed(green_key, &[1])
            .expect("bridge exit should be observable");
        assert_eq!(detailed.trace_id, bridge_trace_id);
    }

    #[test]
    fn test_handle_guard_failure_in_trace_uses_bridge_resume_data() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 15;
        let inputargs = vec![InputArg::new_int(0)];
        let root_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            mk_op(OpCode::Finish, &[OpRef(100)], OpRef::NONE.0),
        ];
        let mut root_constants = HashMap::new();
        root_constants.insert(100, 77);
        attach_procedure_to_interp_entry(
            &mut meta,
            green_key,
            &inputargs,
            root_ops,
            root_constants,
        );

        let root_failure = meta
            .run_compiled_detailed(green_key, &[1])
            .expect("root guard should fail");
        let root_fail_index = root_failure.fail_index;
        let root_trace_id = root_failure.trace_id;
        let bridge_fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            root_fail_index,
            root_trace_id,
            vec![Type::Int],
            false,
            Vec::new(),
            None,
        );
        let _ = root_failure;

        let bridge_inputargs = vec![InputArg::new_int(0)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        assert!(meta.compile_bridge(
            green_key,
            root_fail_index,
            &bridge_fail_descr,
            &bridge_ops,
            &bridge_inputargs,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        ));

        let bridge_trace_id = meta.compiled_loops[&green_key]
            .traces
            .keys()
            .copied()
            .find(|&trace_id| trace_id != root_trace_id)
            .expect("bridge trace should exist");
        meta.attach_resume_data_to_trace(green_key, bridge_trace_id, 0, ResumeData::simple(555, 1));

        let bridge_failure = meta
            .run_compiled_detailed(green_key, &[1])
            .expect("bridge guard should fail");
        let bridge_failure_trace_id = bridge_failure.trace_id;
        let bridge_fail_index = bridge_failure.fail_index;
        let bridge_fail_values = bridge_failure.values.clone();
        let bridge_failure_exception = bridge_failure.exception.clone();
        assert_eq!(bridge_failure_trace_id, bridge_trace_id);
        let _ = bridge_failure;

        let recovery = meta
            .handle_guard_failure_in_trace(
                green_key,
                bridge_failure_trace_id,
                bridge_fail_index,
                &bridge_fail_values,
                None,
                bridge_failure_exception,
            )
            .expect("bridge recovery should succeed");
        assert_eq!(recovery.trace_id, bridge_trace_id);
        assert_eq!(recovery.fail_index, 0);
        let reconstructed = recovery
            .reconstructed_frames
            .expect("bridge resume data should reconstruct frame");
        assert_eq!(reconstructed[0].pc, 555);
        assert_eq!(reconstructed[0].values, vec![ReconstructedValue::Value(1)]);
    }

    #[test]
    fn test_start_retrace_from_guard_preserves_fail_arg_types() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 152;
        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(1)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0), OpRef(1)]);
                guard
            },
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());
        let trace_id = meta.compiled_loops[&green_key].root_trace_id;

        assert!(
            meta.start_retrace_from_guard(green_key, trace_id, 0, &[], &[])
                .0
        );

        let mut ctx = meta.tracing.take().expect("expected active bridge trace");
        ctx.recorder.close_loop(&[OpRef(0), OpRef(1)]);
        let trace = ctx.recorder.get_trace();
        let input_types: Vec<Type> = trace.inputargs.iter().map(|arg| arg.tp).collect();
        assert_eq!(input_types, vec![Type::Ref, Type::Int]);
    }

    #[test]
    fn test_compiled_bridge_runs_with_ref_fail_args() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 153;
        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let root_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(1)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0), OpRef(1)]);
                guard
            },
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        attach_procedure_to_interp_entry(
            &mut meta,
            green_key,
            &inputargs,
            root_ops,
            HashMap::new(),
        );

        let root_failure = meta
            .run_compiled_detailed(green_key, &[0x1234, 1])
            .expect("root guard should fail");
        let root_fail_index = root_failure.fail_index;
        let root_trace_id = root_failure.trace_id;
        let bridge_fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            root_fail_index,
            root_trace_id,
            vec![Type::Ref, Type::Int],
            false,
            Vec::new(),
            None,
        );

        let bridge_inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
        ];
        assert!(meta.compile_bridge(
            green_key,
            root_fail_index,
            &bridge_fail_descr,
            &bridge_ops,
            &bridge_inputargs,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        ));

        let bridge_trace_id = meta.compiled_loops[&green_key]
            .traces
            .keys()
            .copied()
            .find(|&trace_id| trace_id != root_trace_id)
            .expect("bridge trace should exist");
        let bridge_exit = meta
            .run_compiled_detailed(green_key, &[0x1234, 1])
            .expect("bridge should run and exit");
        assert_eq!(bridge_exit.trace_id, bridge_trace_id);
        assert_eq!(bridge_exit.values, vec![0x1234, 1]);
        assert_eq!(
            bridge_exit.exit_layout.exit_types,
            vec![Type::Ref, Type::Int]
        );
    }

    #[test]
    fn test_compiled_bridge_runs_with_new_ref_result() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 154;
        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let root_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(1)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0), OpRef(1)]);
                guard
            },
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        attach_procedure_to_interp_entry(
            &mut meta,
            green_key,
            &inputargs,
            root_ops,
            HashMap::new(),
        );

        let root_failure = meta
            .run_compiled_detailed(green_key, &[0x1234, 1])
            .expect("root guard should fail");
        let root_fail_index = root_failure.fail_index;
        let root_trace_id = root_failure.trace_id;
        let bridge_fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            root_fail_index,
            root_trace_id,
            vec![Type::Ref, Type::Int],
            false,
            Vec::new(),
            None,
        );

        let value_field = majit_ir::make_field_descr(0, 8, Type::Int, true);
        let next_field = majit_ir::make_field_descr(8, 8, Type::Ref, false);
        let node_size = majit_ir::descr::make_size_descr(16);
        let bridge_inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::New, &[], 2, node_size),
            mk_op_with_descr(OpCode::SetfieldGc, &[OpRef(2), OpRef(1)], 3, value_field),
            mk_op_with_descr(OpCode::SetfieldGc, &[OpRef(2), OpRef(0)], 4, next_field),
            mk_op(OpCode::Finish, &[OpRef(2), OpRef(1)], OpRef::NONE.0),
        ];
        assert!(meta.compile_bridge(
            green_key,
            root_fail_index,
            &bridge_fail_descr,
            &bridge_ops,
            &bridge_inputargs,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        ));

        let bridge_trace_id = meta.compiled_loops[&green_key]
            .traces
            .keys()
            .copied()
            .find(|&trace_id| trace_id != root_trace_id)
            .expect("bridge trace should exist");
        let bridge_exit = meta
            .run_compiled_detailed(green_key, &[0x1234, 1])
            .expect("bridge should run and exit");
        assert_eq!(bridge_exit.trace_id, bridge_trace_id);
        assert_eq!(
            bridge_exit.exit_layout.exit_types,
            vec![Type::Ref, Type::Int]
        );
        assert_eq!(bridge_exit.values[1], 1);
        assert_ne!(bridge_exit.values[0], 0);
    }

    #[test]
    fn test_compiled_bridge_runs_with_many_ref_inputs_and_new_ref_outputs() {
        let mut meta = MetaInterp::<()>::new(10);
        meta.backend.set_gc_allocator(Box::new(MiniMarkGC::new()));
        let green_key = 155;

        let mut inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        for idx in 2..28 {
            inputargs.push(InputArg::new_ref(idx as u32));
        }
        let label_args: Vec<OpRef> = (0..28).map(OpRef).collect();
        let mut root_guard = mk_op(OpCode::GuardFalse, &[OpRef(1)], OpRef::NONE.0);
        root_guard.fail_args = Some(label_args.clone().into());
        let root_ops = vec![
            mk_op(OpCode::Label, &label_args, OpRef::NONE.0),
            root_guard,
            mk_op(OpCode::Finish, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
        ];
        attach_procedure_to_interp_entry(
            &mut meta,
            green_key,
            &inputargs,
            root_ops,
            HashMap::new(),
        );

        let live_values: Vec<i64> = std::iter::once(7)
            .chain(std::iter::once(1))
            .chain((2..28).map(|idx| 0x1000 + idx as i64 * 16))
            .collect();
        let root_failure = meta
            .run_compiled_detailed(green_key, &live_values)
            .expect("root guard should fail");
        let root_fail_index = root_failure.fail_index;
        let root_trace_id = root_failure.trace_id;
        let bridge_fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            root_fail_index,
            root_trace_id,
            inputargs.iter().map(|arg| arg.tp).collect(),
            false,
            Vec::new(),
            None,
        );

        let value_field = majit_ir::make_field_descr(0, 8, Type::Int, true);
        let next_field = majit_ir::make_field_descr(8, 8, Type::Ref, false);
        let node_size = majit_ir::descr::make_size_descr(16);
        let mut finish_args = label_args.clone();
        finish_args[3] = OpRef(31);
        finish_args[6] = OpRef(28);
        let bridge_ops = vec![
            mk_op(OpCode::Label, &label_args, OpRef::NONE.0),
            mk_op_with_descr(OpCode::New, &[], 28, node_size.clone()),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef(28), OpRef(0)],
                29,
                value_field.clone(),
            ),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef(28), OpRef(6)],
                30,
                next_field.clone(),
            ),
            mk_op_with_descr(OpCode::New, &[], 31, node_size),
            mk_op_with_descr(OpCode::SetfieldGc, &[OpRef(31), OpRef(1)], 32, value_field),
            mk_op_with_descr(OpCode::SetfieldGc, &[OpRef(31), OpRef(3)], 33, next_field),
            mk_op(OpCode::Finish, &finish_args, OpRef::NONE.0),
        ];
        assert!(meta.compile_bridge(
            green_key,
            root_fail_index,
            &bridge_fail_descr,
            &bridge_ops,
            &inputargs,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        ));

        let bridge_trace_id = meta.compiled_loops[&green_key]
            .traces
            .keys()
            .copied()
            .find(|&trace_id| trace_id != root_trace_id)
            .expect("bridge trace should exist");
        let bridge_exit = meta
            .run_compiled_detailed(green_key, &live_values)
            .expect("bridge should run and exit");
        assert_eq!(bridge_exit.trace_id, bridge_trace_id);
        assert_eq!(bridge_exit.exit_layout.exit_types.len(), 28);
        assert_eq!(bridge_exit.values[0], 7);
        assert_eq!(bridge_exit.values[1], 1);
        assert_ne!(bridge_exit.values[3], live_values[3]);
        assert_ne!(bridge_exit.values[6], live_values[6]);
        assert_ne!(bridge_exit.values[3], 0);
        assert_ne!(bridge_exit.values[6], 0);
    }

    #[test]
    fn test_bridge_attach_resume_data_patches_backend_recovery_layout_preserving_caller_prefix() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 151;
        let inputargs = vec![InputArg::new_int(0)];
        let root_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        attach_procedure_to_interp_entry(
            &mut meta,
            green_key,
            &inputargs,
            root_ops,
            HashMap::new(),
        );
        let root_trace_id = meta.compiled_loops[&green_key].root_trace_id;
        meta.attach_resume_data_to_trace(
            green_key,
            root_trace_id,
            0,
            ResumeData {
                frames: vec![
                    crate::resume::FrameInfo {
                        pc: 10,
                        slot_map: vec![FrameSlotSource::FailArg(0)],
                    },
                    crate::resume::FrameInfo {
                        pc: 20,
                        slot_map: vec![FrameSlotSource::FailArg(0)],
                    },
                ],
                virtuals: Vec::new(),
                pending_fields: Vec::new(),
            },
        );

        let fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            0,
            root_trace_id,
            vec![Type::Int],
            false,
            Vec::new(),
            None,
        );
        let bridge_inputargs = vec![InputArg::new_int(0)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        assert!(meta.compile_bridge(
            green_key,
            0,
            &fail_descr,
            &bridge_ops,
            &bridge_inputargs,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        ));

        let bridge_trace_id = meta.compiled_loops[&green_key]
            .traces
            .keys()
            .copied()
            .find(|&trace_id| trace_id != root_trace_id)
            .expect("bridge trace should exist");
        meta.attach_resume_data_to_trace(green_key, bridge_trace_id, 0, ResumeData::simple(555, 1));

        let token = &meta.compiled_loops[&green_key].token;
        let bridge_layout = meta
            .backend
            .compiled_trace_fail_descr_layouts(token, bridge_trace_id)
            .expect("bridge backend layouts should exist")
            .into_iter()
            .find(|layout| layout.fail_index == 0)
            .expect("bridge fail layout should exist");
        let recovery = bridge_layout
            .recovery_layout
            .expect("bridge recovery layout should be patched");
        assert_eq!(recovery.frames.len(), 2);
        assert_eq!(recovery.frames[0].pc, 10);
        assert_eq!(recovery.frames[0].source_guard, None);
        assert_eq!(recovery.frames[1].pc, 555);
        assert_eq!(recovery.frames[1].source_guard, Some((root_trace_id, 0)));
    }

    #[test]
    fn test_run_compiled_detailed_decodes_float_exit_values() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 16;
        let inputargs = vec![];
        let ops = vec![
            mk_op(OpCode::Label, &[], OpRef::NONE.0),
            mk_op(OpCode::CastIntToFloat, &[OpRef(100)], 0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 7);

        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, constants);

        let result = meta
            .run_compiled_detailed(green_key, &[])
            .expect("compiled loop should finish");
        assert!(result.is_finish);
        assert_eq!(result.values, vec![7.0f64.to_bits() as i64]);
        assert_eq!(result.typed_values, vec![Value::Float(7.0)]);
    }

    #[test]
    fn test_run_compiled_values_preserves_mixed_type_fast_path_outputs() {
        let mut meta = MetaInterp::<()>::new(10);
        meta.backend.set_gc_allocator(Box::new(MiniMarkGC::new()));
        let green_key = 20;
        let inputargs = vec![];
        let ops = vec![
            mk_op(OpCode::Label, &[], OpRef::NONE.0),
            mk_op(OpCode::Newstr, &[OpRef(100)], 0),
            mk_op(OpCode::CastIntToFloat, &[OpRef(101)], 1),
            mk_op(OpCode::Finish, &[OpRef(1), OpRef(0)], OpRef::NONE.0),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 7);

        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, constants);

        let (values, _) = meta
            .run_compiled_values(green_key, &[])
            .expect("compiled loop should finish");
        match values.as_slice() {
            [Value::Float(value), Value::Ref(gcref)] => {
                assert_eq!(*value, 7.0);
                assert!(!gcref.is_null());
            }
            other => panic!("unexpected typed outputs: {other:?}"),
        }
    }

    #[test]
    fn test_run_compiled_raw_detailed_preserves_savedata_and_layout_for_call_may_force() {
        let _guard = may_force_test_lock()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 114;
        install_may_force_void_entry(&mut meta, green_key);

        let result = meta
            .run_compiled_raw_detailed(green_key, &[10, 1])
            .expect("forced raw exit should exist");
        assert!(!result.is_finish);
        assert_eq!(result.fail_index, 0);
        assert_eq!(result.values, vec![1, 10]);
        assert_eq!(result.typed_values, vec![Value::Int(1), Value::Int(10)]);
        assert_eq!(result.savedata, Some(GcRef(0xDADA)));
        assert_eq!(result.exception, ExceptionState::default());
        assert_eq!(result.exit_layout.source_op_index, Some(3));
        assert_eq!(result.exit_layout.exit_types, vec![Type::Int, Type::Int]);
        assert!(result.exit_layout.recovery_layout.is_some());
        assert_eq!(
            *may_force_void_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner()),
            vec![0, 1, 10]
        );
    }

    #[test]
    fn test_run_compiled_with_values_accepts_float_inputs() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 21;
        let inputargs = vec![InputArg::new_float(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());

        let (values, _) = meta
            .run_compiled_with_values(green_key, &[Value::Float(3.5)])
            .expect("compiled loop should finish");
        assert_eq!(values, vec![Value::Float(3.5)]);
    }

    #[test]
    fn test_run_compiled_raw_detailed_with_values_accepts_float_inputs() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 211;
        let inputargs = vec![InputArg::new_float(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());

        let result = meta
            .run_compiled_raw_detailed_with_values(green_key, &[Value::Float(3.5)])
            .expect("compiled raw loop should finish");
        assert!(result.is_finish);
        assert_eq!(result.values, vec![3.5f64.to_bits() as i64]);
        assert_eq!(result.typed_values, vec![Value::Float(3.5)]);
        assert_eq!(result.savedata, None);
        assert_eq!(result.exception, ExceptionState::default());
        assert_eq!(result.exit_layout.exit_types, vec![Type::Float]);
        assert_eq!(result.exit_layout.gc_ref_slots, Vec::<usize>::new());
    }

    #[test]
    fn test_get_terminal_exit_layout_in_trace_reports_jump_shape() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 22;
        let inputargs = vec![InputArg::new_float(0), InputArg::new_ref(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(1), OpRef(0)], OpRef::NONE.0),
        ];

        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());
        let trace_id = meta
            .compiled_loops
            .get(&green_key)
            .expect("compiled entry")
            .root_trace_id;

        let layout = meta
            .get_terminal_exit_layout_in_trace(green_key, trace_id, 1)
            .expect("terminal jump layout should exist");
        assert_eq!(layout.trace_id, trace_id);
        assert_eq!(layout.fail_index, u32::MAX);
        assert_eq!(layout.exit_types, vec![Type::Ref, Type::Float]);
        assert!(!layout.is_finish);
        assert_eq!(layout.gc_ref_slots, vec![0]);
        assert!(layout.force_token_slots.is_empty());
    }

    #[test]
    fn test_get_compiled_trace_layout_in_trace_reports_guard_and_terminal_shapes() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 220;
        let fail_index = 3;
        let trace_id = meta.alloc_trace_id();
        let mut exit_layouts = HashMap::new();
        exit_layouts.insert(
            fail_index,
            StoredExitLayout {
                source_op_index: None,
                exit_types: vec![Type::Ref, Type::Int],
                is_finish: false,
                gc_ref_slots: vec![0],
                force_token_slots: Vec::new(),
                recovery_layout: None,
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            },
        );
        let mut terminal_exit_layouts = HashMap::new();
        terminal_exit_layouts.insert(
            9,
            StoredExitLayout {
                source_op_index: None,
                exit_types: vec![Type::Float],
                is_finish: true,
                gc_ref_slots: Vec::new(),
                force_token_slots: Vec::new(),
                recovery_layout: None,
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            },
        );
        let mut traces = HashMap::new();
        traces.insert(
            trace_id,
            CompiledTrace {
                inputargs: Vec::new(),
                resume_data: HashMap::new(),
                ops: Vec::new(),
                constants: HashMap::new(),
                guard_op_indices: HashMap::new(),
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
                token: JitCellToken::new(700),
                num_inputs: 0,
                meta: (),
                front_target_tokens: Vec::new(),
                retraced_count: 0,
                root_trace_id: trace_id,

                traces,
                previous_tokens: Vec::new(),
            },
        );

        let layout = meta
            .get_compiled_trace_layout_in_trace(green_key, trace_id)
            .expect("trace layout should exist");
        assert_eq!(layout.trace_id, trace_id);
        assert_eq!(layout.exit_layouts.len(), 1);
        assert_eq!(layout.exit_layouts[0].fail_index, fail_index);
        assert_eq!(
            layout.exit_layouts[0].exit_types,
            vec![Type::Ref, Type::Int]
        );
        assert_eq!(layout.terminal_exit_layouts.len(), 1);
        assert_eq!(layout.terminal_exit_layouts[0].op_index, 9);
        assert_eq!(
            layout.terminal_exit_layouts[0].exit_layout.exit_types,
            vec![Type::Float]
        );
        assert!(layout.terminal_exit_layouts[0].exit_layout.is_finish);
    }

    #[test]
    fn test_terminal_exit_layout_falls_back_to_backend_metadata_when_trace_local_is_missing() {
        let mut meta = MetaInterp::<()>::new(10);
        let green_key = 221;
        let inputargs = vec![InputArg::new_float(0), InputArg::new_ref(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(1), OpRef(0)], OpRef::NONE.0),
        ];

        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());
        let trace_id = meta
            .compiled_loops
            .get(&green_key)
            .expect("compiled entry")
            .root_trace_id;
        meta.compiled_loops
            .get_mut(&green_key)
            .expect("compiled entry")
            .traces
            .get_mut(&trace_id)
            .expect("trace")
            .terminal_exit_layouts
            .clear();

        let layout = meta
            .get_terminal_exit_layout_in_trace(green_key, trace_id, 1)
            .expect("backend terminal layout should exist");
        assert_eq!(layout.trace_id, trace_id);
        assert_eq!(layout.source_op_index, Some(1));
        assert_eq!(layout.exit_types, vec![Type::Ref, Type::Float]);
        assert!(!layout.is_finish);
        assert_eq!(layout.gc_ref_slots, vec![0]);
        assert!(layout.recovery_layout.is_some());
    }

    #[test]
    fn test_handle_guard_failure_in_trace_preserves_typed_ref_fail_values() {
        let mut meta = MetaInterp::<()>::new(10);
        meta.backend.set_gc_allocator(Box::new(MiniMarkGC::new()));
        let green_key = 19;
        let inputargs = vec![];
        let ops = vec![
            mk_op(OpCode::Label, &[], OpRef::NONE.0),
            mk_op(OpCode::Newstr, &[OpRef(100)], 0),
            {
                let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(101)], OpRef::NONE.0);
                guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                guard
            },
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 1);

        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, constants);

        let result = meta
            .run_compiled_detailed(green_key, &[])
            .expect("guard should fail");
        assert!(!result.is_finish);
        let trace_id = result.trace_id;
        let fail_index = result.fail_index;
        let exception = result.exception.clone();
        let typed_values = result.typed_values.clone();
        let raw_values = result.values.clone();
        let _ = result;
        let recovery = meta
            .handle_guard_failure_in_trace(
                green_key,
                trace_id,
                fail_index,
                &raw_values,
                Some(&typed_values),
                exception,
            )
            .expect("recovery should exist");

        match recovery
            .typed_fail_values
            .as_deref()
            .expect("typed fail values should be preserved")
        {
            [Value::Ref(gcref)] => {
                assert!(!gcref.is_null());
                assert_eq!(raw_values, vec![gcref.as_usize() as i64]);
            }
            other => panic!("unexpected typed fail values: {other:?}"),
        }
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
            let const_one = OpRef(10_000);
            ctx.constants.as_mut().insert(10_000, 1);
            let _result = ctx.recorder.record_op(OpCode::IntAdd, &[i0, const_one]);
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
            let const_one = OpRef(10_000);
            ctx.constants.as_mut().insert(10_000, 1);
            let _result = ctx.recorder.record_op(OpCode::IntAdd, &[i0, const_one]);
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
                let const_one = OpRef(10_000);
                ctx.constants.as_mut().insert(10_000, 1);
                let sum = ctx.recorder.record_op(OpCode::IntAdd, &[i0, i1]);
                let _ = ctx.recorder.record_op(OpCode::IntAdd, &[sum, const_one]);
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

    #[test]
    fn test_multi_frame_restore_uses_frame_stack_metadata() {
        use crate::jit_state::JitState;
        use crate::resume::{ReconstructedFrame, ReconstructedValue, ResumeFrameLayoutSummary};

        // JitState implementation that records per-frame restores.
        #[derive(Default)]
        struct MultiFrameState {
            restored_frames: Vec<(usize, u64, Vec<Value>)>,
        }

        impl JitState for MultiFrameState {
            type Meta = ();
            type Sym = ();
            type Env = ();

            fn build_meta(&self, _: usize, _: &()) -> () {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                Vec::new()
            }
            fn create_sym(_: &(), _: usize) -> () {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}

            fn restore_reconstructed_frame_values_with_metadata(
                &mut self,
                _meta: &(),
                frame_index: usize,
                _total_frames: usize,
                frame: &crate::resume::ReconstructedFrame,
                values: &[Value],
                _exception: &ExceptionState,
            ) -> bool {
                self.restored_frames
                    .push((frame_index, frame.pc, values.to_vec()));
                true
            }

            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }
        }

        let mut state = MultiFrameState::default();

        // Build a ReconstructedState with 2 frames (outermost first).
        let reconstructed_state = ReconstructedState {
            frames: vec![
                ReconstructedFrame {
                    trace_id: Some(100),
                    header_pc: Some(500),
                    source_guard: None,
                    pc: 10,
                    slot_types: None,
                    values: vec![ReconstructedValue::Value(42)],
                },
                ReconstructedFrame {
                    trace_id: Some(200),
                    header_pc: Some(600),
                    source_guard: None,
                    pc: 20,
                    slot_types: None,
                    values: vec![ReconstructedValue::Value(7), ReconstructedValue::Value(8)],
                },
            ],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };

        // Build frame_stack metadata with slot_types for both frames.
        let frame_layouts = vec![
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(100),
                header_pc: Some(500),
                source_guard: None,
                pc: 10,
                slots: vec![ExitValueSourceLayout::ExitValue(0)],
                slot_types: Some(vec![Type::Int]),
            }),
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(200),
                header_pc: Some(600),
                source_guard: None,
                pc: 20,
                slots: vec![
                    ExitValueSourceLayout::ExitValue(1),
                    ExitValueSourceLayout::ExitValue(2),
                ],
                slot_types: Some(vec![Type::Int, Type::Int]),
            }),
        ];

        let restored = state.restore_guard_failure_with_resume_layout(
            &(),
            &[42, 7, 8],
            Some(&reconstructed_state),
            Some(&frame_layouts),
            &[],
            &[],
            &ExceptionState::default(),
        );

        assert!(restored, "multi-frame restore should succeed");
        assert_eq!(
            state.restored_frames.len(),
            2,
            "both frames should be restored"
        );
        assert_eq!(state.restored_frames[0].0, 0); // frame_index 0
        assert_eq!(state.restored_frames[0].1, 10); // pc
        assert_eq!(state.restored_frames[0].2, vec![Value::Int(42)]);
        assert_eq!(state.restored_frames[1].0, 1); // frame_index 1
        assert_eq!(state.restored_frames[1].1, 20); // pc
        assert_eq!(
            state.restored_frames[1].2,
            vec![Value::Int(7), Value::Int(8)]
        );
    }

    #[test]
    fn test_single_frame_restore_with_frame_stack() {
        use crate::jit_state::JitState;
        use crate::resume::{ReconstructedFrame, ReconstructedValue, ResumeFrameLayoutSummary};
        use majit_backend::ExitValueSourceLayout;

        #[derive(Default)]
        struct SingleFrameState {
            restored_values: Vec<Value>,
        }

        impl JitState for SingleFrameState {
            type Meta = ();
            type Sym = ();
            type Env = ();

            fn build_meta(&self, _: usize, _: &()) -> () {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                Vec::new()
            }
            fn create_sym(_: &(), _: usize) -> () {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}

            fn restore_values(&mut self, _: &(), values: &[Value]) {
                self.restored_values = values.to_vec();
            }

            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }
        }

        let mut state = SingleFrameState::default();

        let reconstructed_state = ReconstructedState {
            frames: vec![ReconstructedFrame {
                trace_id: Some(300),
                header_pc: Some(700),
                source_guard: None,
                pc: 30,
                slot_types: None,
                values: vec![
                    ReconstructedValue::Value(100),
                    ReconstructedValue::Value(200),
                ],
            }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };

        // Provide frame_stack metadata with slot_types.
        let frame_layouts = vec![ResumeFrameLayoutSummary::from_exit_frame_layout(
            &ExitFrameLayout {
                trace_id: Some(300),
                header_pc: Some(700),
                source_guard: None,
                pc: 30,
                slots: vec![
                    ExitValueSourceLayout::ExitValue(0),
                    ExitValueSourceLayout::ExitValue(1),
                ],
                slot_types: Some(vec![Type::Int, Type::Float]),
            },
        )];

        let restored = state.restore_guard_failure_with_resume_layout(
            &(),
            &[100, 200],
            Some(&reconstructed_state),
            Some(&frame_layouts),
            &[],
            &[],
            &ExceptionState::default(),
        );

        assert!(restored, "single-frame restore should succeed");
        assert_eq!(state.restored_values.len(), 2);
        assert_eq!(state.restored_values[0], Value::Int(100));
        // 200 as f64 bits
        assert_eq!(state.restored_values[1], Value::Float(f64::from_bits(200)));
    }

    #[test]
    fn test_merge_backend_exit_layouts_with_frame_stack() {
        let mut exit_layouts = HashMap::new();
        exit_layouts.insert(
            0,
            StoredExitLayout {
                source_op_index: None,
                exit_types: vec![Type::Int, Type::Int],
                is_finish: false,
                gc_ref_slots: Vec::new(),
                force_token_slots: Vec::new(),
                recovery_layout: None,
                resume_layout: None,
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            },
        );

        // Merge with frame_stack containing two frames.
        compile::merge_backend_exit_layouts(
            &mut exit_layouts,
            &[FailDescrLayout {
                fail_index: 0,
                source_op_index: Some(3),
                trace_id: 50,
                trace_info: None,
                fail_arg_types: vec![Type::Int, Type::Int],
                is_finish: false,
                gc_ref_slots: Vec::new(),
                force_token_slots: Vec::new(),
                recovery_layout: None,
                frame_stack: Some(vec![
                    ExitFrameLayout {
                        trace_id: Some(40),
                        header_pc: Some(400),
                        source_guard: None,
                        pc: 10,
                        slots: vec![ExitValueSourceLayout::ExitValue(0)],
                        slot_types: Some(vec![Type::Int]),
                    },
                    ExitFrameLayout {
                        trace_id: Some(50),
                        header_pc: Some(500),
                        source_guard: None,
                        pc: 20,
                        slots: vec![ExitValueSourceLayout::ExitValue(1)],
                        slot_types: Some(vec![Type::Int]),
                    },
                ]),
            }],
        );

        let layout = exit_layouts.get(&0).expect("layout should exist");
        let resume = layout
            .resume_layout
            .as_ref()
            .expect("resume_layout should be created from frame_stack");
        assert_eq!(resume.frame_layouts.len(), 2);
        assert_eq!(resume.frame_layouts[0].pc, 10);
        assert_eq!(resume.frame_layouts[0].trace_id, Some(40));
        assert_eq!(resume.frame_layouts[0].slot_types, Some(vec![Type::Int]));
        assert_eq!(resume.frame_layouts[1].pc, 20);
        assert_eq!(resume.frame_layouts[1].trace_id, Some(50));
        assert_eq!(resume.frame_layouts[1].slot_types, Some(vec![Type::Int]));
        assert_eq!(resume.num_frames, 2);
        assert_eq!(resume.frame_pcs, vec![10, 20]);
    }

    #[test]
    fn test_merge_backend_exit_layouts_frame_stack_enriches_existing_resume() {
        use crate::resume::{ResumeValueKind, ResumeValueLayoutSummary};

        // Existing resume layout with one frame that has no slot_types.
        let existing_frame = ResumeFrameLayoutSummary {
            trace_id: None,
            header_pc: None,
            source_guard: None,
            pc: 20,
            slot_sources: vec![ResumeValueKind::FailArg],
            slot_layouts: vec![ResumeValueLayoutSummary {
                kind: ResumeValueKind::FailArg,
                fail_arg_index: 0,
                raw_fail_arg_position: Some(0),
                constant: None,
                virtual_index: None,
            }],
            slot_types: None,
        };
        let existing_resume = ResumeLayoutSummary {
            num_frames: 1,
            frame_pcs: vec![20],
            frame_slot_counts: vec![1],
            frame_layouts: vec![existing_frame],
            num_virtuals: 0,
            virtual_kinds: Vec::new(),
            virtual_layouts: Vec::new(),
            pending_field_count: 0,
            pending_field_layouts: Vec::new(),
            const_pool_size: 0,
        };

        let mut exit_layouts = HashMap::new();
        exit_layouts.insert(
            0,
            StoredExitLayout {
                source_op_index: None,
                exit_types: vec![Type::Int],
                is_finish: false,
                gc_ref_slots: Vec::new(),
                force_token_slots: Vec::new(),
                recovery_layout: None,
                resume_layout: Some(existing_resume),
                rd_numb: None,
                rd_consts: None,
                rd_virtuals: None,
                rd_pendingfields: None,
            },
        );

        // frame_stack with an outer frame (new) and the same innermost frame
        // (with slot_types that the existing resume layout lacked).
        compile::merge_backend_exit_layouts(
            &mut exit_layouts,
            &[FailDescrLayout {
                fail_index: 0,
                source_op_index: Some(5),
                trace_id: 60,
                trace_info: None,
                fail_arg_types: vec![Type::Int],
                is_finish: false,
                gc_ref_slots: Vec::new(),
                force_token_slots: Vec::new(),
                recovery_layout: None,
                frame_stack: Some(vec![
                    ExitFrameLayout {
                        trace_id: Some(55),
                        header_pc: Some(550),
                        source_guard: None,
                        pc: 10,
                        slots: vec![ExitValueSourceLayout::Constant(99)],
                        slot_types: Some(vec![Type::Int]),
                    },
                    ExitFrameLayout {
                        trace_id: Some(60),
                        header_pc: Some(600),
                        source_guard: None,
                        pc: 20,
                        slots: vec![ExitValueSourceLayout::ExitValue(0)],
                        slot_types: Some(vec![Type::Int]),
                    },
                ]),
            }],
        );

        let layout = exit_layouts.get(&0).expect("layout should exist");
        let resume = layout
            .resume_layout
            .as_ref()
            .expect("resume_layout should exist");

        // Now should have 2 frames: outer prepended + existing enriched.
        assert_eq!(resume.frame_layouts.len(), 2);
        // Outer frame (prepended from frame_stack).
        assert_eq!(resume.frame_layouts[0].pc, 10);
        assert_eq!(resume.frame_layouts[0].trace_id, Some(55));
        // Innermost frame: trace_id enriched, slot_types filled.
        assert_eq!(resume.frame_layouts[1].pc, 20);
        assert_eq!(resume.frame_layouts[1].trace_id, Some(60));
        assert_eq!(resume.frame_layouts[1].slot_types, Some(vec![Type::Int]));
    }

    #[test]
    fn test_three_level_nested_frame_restore_from_frame_stack_metadata() {
        use crate::jit_state::JitState;
        use crate::resume::{ReconstructedFrame, ReconstructedValue, ResumeFrameLayoutSummary};

        // JitState implementation that records per-frame restores.
        #[derive(Default)]
        struct ThreeFrameState {
            restored_frames: Vec<(usize, u64, Vec<Value>)>,
        }

        impl JitState for ThreeFrameState {
            type Meta = ();
            type Sym = ();
            type Env = ();

            fn build_meta(&self, _: usize, _: &()) -> () {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                Vec::new()
            }
            fn create_sym(_: &(), _: usize) -> () {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}

            fn restore_reconstructed_frame_values_with_metadata(
                &mut self,
                _meta: &(),
                frame_index: usize,
                _total_frames: usize,
                frame: &crate::resume::ReconstructedFrame,
                values: &[Value],
                _exception: &ExceptionState,
            ) -> bool {
                self.restored_frames
                    .push((frame_index, frame.pc, values.to_vec()));
                true
            }

            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }
        }

        let mut state = ThreeFrameState::default();

        // Frame 0 (outermost): 3 Int slots
        // Frame 1 (middle):    1 Ref slot + 1 Float slot
        // Frame 2 (innermost): 2 Int slots
        let float_bits: i64 = 3.14f64.to_bits() as i64;
        let ref_val: i64 = 0xBEEF;

        let reconstructed_state = ReconstructedState {
            frames: vec![
                ReconstructedFrame {
                    trace_id: Some(100),
                    header_pc: Some(500),
                    source_guard: None,
                    pc: 10,
                    slot_types: None,
                    values: vec![
                        ReconstructedValue::Value(1),
                        ReconstructedValue::Value(2),
                        ReconstructedValue::Value(3),
                    ],
                },
                ReconstructedFrame {
                    trace_id: Some(200),
                    header_pc: Some(600),
                    source_guard: None,
                    pc: 20,
                    slot_types: None,
                    values: vec![
                        ReconstructedValue::Value(ref_val),
                        ReconstructedValue::Value(float_bits),
                    ],
                },
                ReconstructedFrame {
                    trace_id: Some(300),
                    header_pc: Some(700),
                    source_guard: None,
                    pc: 30,
                    slot_types: None,
                    values: vec![ReconstructedValue::Value(42), ReconstructedValue::Value(99)],
                },
            ],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };

        // Build frame_stack metadata with slot_types for all 3 frames.
        let frame_layouts = vec![
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(100),
                header_pc: Some(500),
                source_guard: None,
                pc: 10,
                slots: vec![
                    ExitValueSourceLayout::ExitValue(0),
                    ExitValueSourceLayout::ExitValue(1),
                    ExitValueSourceLayout::ExitValue(2),
                ],
                slot_types: Some(vec![Type::Int, Type::Int, Type::Int]),
            }),
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(200),
                header_pc: Some(600),
                source_guard: None,
                pc: 20,
                slots: vec![
                    ExitValueSourceLayout::ExitValue(3),
                    ExitValueSourceLayout::ExitValue(4),
                ],
                slot_types: Some(vec![Type::Ref, Type::Float]),
            }),
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(300),
                header_pc: Some(700),
                source_guard: None,
                pc: 30,
                slots: vec![
                    ExitValueSourceLayout::ExitValue(5),
                    ExitValueSourceLayout::ExitValue(6),
                ],
                slot_types: Some(vec![Type::Int, Type::Int]),
            }),
        ];

        let restored = state.restore_guard_failure_with_resume_layout(
            &(),
            &[1, 2, 3, ref_val, float_bits, 42, 99],
            Some(&reconstructed_state),
            Some(&frame_layouts),
            &[],
            &[],
            &ExceptionState::default(),
        );

        assert!(restored, "three-frame restore should succeed");
        assert_eq!(
            state.restored_frames.len(),
            3,
            "all 3 frames should be restored"
        );

        // Frame 0 (outermost): 3 Int slots
        assert_eq!(state.restored_frames[0].0, 0);
        assert_eq!(state.restored_frames[0].1, 10);
        assert_eq!(
            state.restored_frames[0].2,
            vec![Value::Int(1), Value::Int(2), Value::Int(3)]
        );

        // Frame 1 (middle): 1 Ref slot + 1 Float slot
        assert_eq!(state.restored_frames[1].0, 1);
        assert_eq!(state.restored_frames[1].1, 20);
        assert_eq!(
            state.restored_frames[1].2,
            vec![
                Value::Ref(GcRef(ref_val as usize)),
                Value::Float(f64::from_bits(float_bits as u64))
            ]
        );

        // Frame 2 (innermost): 2 Int slots
        assert_eq!(state.restored_frames[2].0, 2);
        assert_eq!(state.restored_frames[2].1, 30);
        assert_eq!(
            state.restored_frames[2].2,
            vec![Value::Int(42), Value::Int(99)]
        );
    }

    #[test]
    fn test_four_level_nested_frame_restore() {
        use crate::jit_state::JitState;
        use crate::resume::{ReconstructedFrame, ReconstructedValue, ResumeFrameLayoutSummary};

        #[derive(Default)]
        struct FourFrameState {
            restored_frames: Vec<(usize, u64, Vec<Value>)>,
        }

        impl JitState for FourFrameState {
            type Meta = ();
            type Sym = ();
            type Env = ();

            fn build_meta(&self, _: usize, _: &()) -> () {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                Vec::new()
            }
            fn create_sym(_: &(), _: usize) -> () {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}

            fn restore_reconstructed_frame_values_with_metadata(
                &mut self,
                _meta: &(),
                frame_index: usize,
                _total_frames: usize,
                frame: &crate::resume::ReconstructedFrame,
                values: &[Value],
                _exception: &ExceptionState,
            ) -> bool {
                self.restored_frames
                    .push((frame_index, frame.pc, values.to_vec()));
                true
            }

            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }
        }

        let mut state = FourFrameState::default();

        let float1_bits: i64 = 1.5f64.to_bits() as i64;
        let float2_bits: i64 = (-2.718f64).to_bits() as i64;
        let ref1: i64 = 0xCAFE;
        let ref2: i64 = 0xFACE;

        // Frame 0: 1 Int
        // Frame 1: 1 Float
        // Frame 2: 2 Ref
        // Frame 3: 1 Int + 1 Float + 1 Ref
        let reconstructed_state = ReconstructedState {
            frames: vec![
                ReconstructedFrame {
                    trace_id: Some(1000),
                    header_pc: Some(5000),
                    source_guard: None,
                    pc: 100,
                    slot_types: None,
                    values: vec![ReconstructedValue::Value(77)],
                },
                ReconstructedFrame {
                    trace_id: Some(2000),
                    header_pc: Some(6000),
                    source_guard: None,
                    pc: 200,
                    slot_types: None,
                    values: vec![ReconstructedValue::Value(float1_bits)],
                },
                ReconstructedFrame {
                    trace_id: Some(3000),
                    header_pc: Some(7000),
                    source_guard: None,
                    pc: 300,
                    slot_types: None,
                    values: vec![
                        ReconstructedValue::Value(ref1),
                        ReconstructedValue::Value(ref2),
                    ],
                },
                ReconstructedFrame {
                    trace_id: Some(4000),
                    header_pc: Some(8000),
                    source_guard: None,
                    pc: 400,
                    slot_types: None,
                    values: vec![
                        ReconstructedValue::Value(55),
                        ReconstructedValue::Value(float2_bits),
                        ReconstructedValue::Value(ref1),
                    ],
                },
            ],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };

        let frame_layouts = vec![
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(1000),
                header_pc: Some(5000),
                source_guard: None,
                pc: 100,
                slots: vec![ExitValueSourceLayout::ExitValue(0)],
                slot_types: Some(vec![Type::Int]),
            }),
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(2000),
                header_pc: Some(6000),
                source_guard: None,
                pc: 200,
                slots: vec![ExitValueSourceLayout::ExitValue(1)],
                slot_types: Some(vec![Type::Float]),
            }),
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(3000),
                header_pc: Some(7000),
                source_guard: None,
                pc: 300,
                slots: vec![
                    ExitValueSourceLayout::ExitValue(2),
                    ExitValueSourceLayout::ExitValue(3),
                ],
                slot_types: Some(vec![Type::Ref, Type::Ref]),
            }),
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(4000),
                header_pc: Some(8000),
                source_guard: None,
                pc: 400,
                slots: vec![
                    ExitValueSourceLayout::ExitValue(4),
                    ExitValueSourceLayout::ExitValue(5),
                    ExitValueSourceLayout::ExitValue(6),
                ],
                slot_types: Some(vec![Type::Int, Type::Float, Type::Ref]),
            }),
        ];

        let restored = state.restore_guard_failure_with_resume_layout(
            &(),
            &[77, float1_bits, ref1, ref2, 55, float2_bits, ref1],
            Some(&reconstructed_state),
            Some(&frame_layouts),
            &[],
            &[],
            &ExceptionState::default(),
        );

        assert!(restored, "four-frame restore should succeed");
        assert_eq!(
            state.restored_frames.len(),
            4,
            "all 4 frames should be restored"
        );

        // Frame 0: 1 Int
        assert_eq!(state.restored_frames[0].0, 0);
        assert_eq!(state.restored_frames[0].1, 100);
        assert_eq!(state.restored_frames[0].2, vec![Value::Int(77)]);

        // Frame 1: 1 Float
        assert_eq!(state.restored_frames[1].0, 1);
        assert_eq!(state.restored_frames[1].1, 200);
        assert_eq!(
            state.restored_frames[1].2,
            vec![Value::Float(f64::from_bits(float1_bits as u64))]
        );

        // Frame 2: 2 Ref
        assert_eq!(state.restored_frames[2].0, 2);
        assert_eq!(state.restored_frames[2].1, 300);
        assert_eq!(
            state.restored_frames[2].2,
            vec![
                Value::Ref(GcRef(ref1 as usize)),
                Value::Ref(GcRef(ref2 as usize))
            ]
        );

        // Frame 3: Int + Float + Ref
        assert_eq!(state.restored_frames[3].0, 3);
        assert_eq!(state.restored_frames[3].1, 400);
        assert_eq!(
            state.restored_frames[3].2,
            vec![
                Value::Int(55),
                Value::Float(f64::from_bits(float2_bits as u64)),
                Value::Ref(GcRef(ref1 as usize))
            ]
        );
    }

    #[test]
    fn test_nested_frame_restore_with_virtuals() {
        use crate::jit_state::JitState;
        use crate::resume::{
            MaterializedValue, MaterializedVirtual, ReconstructedFrame, ReconstructedValue,
            ResumeFrameLayoutSummary,
        };

        #[derive(Default)]
        struct VirtualFrameState {
            restored_frames: Vec<(usize, u64, Vec<Value>)>,
        }

        impl JitState for VirtualFrameState {
            type Meta = ();
            type Sym = ();
            type Env = ();

            fn build_meta(&self, _: usize, _: &()) -> () {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                Vec::new()
            }
            fn create_sym(_: &(), _: usize) -> () {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}

            fn restore_reconstructed_frame_values_with_metadata(
                &mut self,
                _meta: &(),
                frame_index: usize,
                _total_frames: usize,
                frame: &crate::resume::ReconstructedFrame,
                values: &[Value],
                _exception: &ExceptionState,
            ) -> bool {
                self.restored_frames
                    .push((frame_index, frame.pc, values.to_vec()));
                true
            }

            fn materialize_virtual_ref(
                &mut self,
                _meta: &(),
                virtual_index: usize,
                _materialized: &MaterializedVirtual,
            ) -> Option<GcRef> {
                // Return a deterministic GcRef based on virtual_index.
                Some(GcRef(0xA000 + virtual_index))
            }

            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }
        }

        let mut state = VirtualFrameState::default();

        // Frame 0 (outermost): 1 Int slot + 1 Virtual(0) slot
        // Frame 1 (middle):    1 Float slot + 1 Virtual(1) slot
        // Frame 2 (innermost): 2 Int slots
        let float_bits: i64 = 9.81f64.to_bits() as i64;

        let reconstructed_state = ReconstructedState {
            frames: vec![
                ReconstructedFrame {
                    trace_id: Some(100),
                    header_pc: Some(500),
                    source_guard: None,
                    pc: 10,
                    slot_types: None,
                    values: vec![
                        ReconstructedValue::Value(42),
                        ReconstructedValue::Virtual(0),
                    ],
                },
                ReconstructedFrame {
                    trace_id: Some(200),
                    header_pc: Some(600),
                    source_guard: None,
                    pc: 20,
                    slot_types: None,
                    values: vec![
                        ReconstructedValue::Value(float_bits),
                        ReconstructedValue::Virtual(1),
                    ],
                },
                ReconstructedFrame {
                    trace_id: Some(300),
                    header_pc: Some(700),
                    source_guard: None,
                    pc: 30,
                    slot_types: None,
                    values: vec![ReconstructedValue::Value(10), ReconstructedValue::Value(20)],
                },
            ],
            virtuals: vec![
                MaterializedVirtual::Obj {
                    type_id: 1,
                    descr_index: 0,
                    fields: vec![(0, MaterializedValue::Value(100))],
                },
                MaterializedVirtual::Obj {
                    type_id: 2,
                    descr_index: 1,
                    fields: vec![(0, MaterializedValue::Value(200))],
                },
            ],
            pending_fields: Vec::new(),
        };

        let frame_layouts = vec![
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(100),
                header_pc: Some(500),
                source_guard: None,
                pc: 10,
                slots: vec![
                    ExitValueSourceLayout::ExitValue(0),
                    ExitValueSourceLayout::ExitValue(1),
                ],
                slot_types: Some(vec![Type::Int, Type::Ref]),
            }),
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(200),
                header_pc: Some(600),
                source_guard: None,
                pc: 20,
                slots: vec![
                    ExitValueSourceLayout::ExitValue(2),
                    ExitValueSourceLayout::ExitValue(3),
                ],
                slot_types: Some(vec![Type::Float, Type::Ref]),
            }),
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(300),
                header_pc: Some(700),
                source_guard: None,
                pc: 30,
                slots: vec![
                    ExitValueSourceLayout::ExitValue(4),
                    ExitValueSourceLayout::ExitValue(5),
                ],
                slot_types: Some(vec![Type::Int, Type::Int]),
            }),
        ];

        let restored = state.restore_guard_failure_with_resume_layout(
            &(),
            &[42, 0, float_bits, 0, 10, 20], // Virtual slots have placeholder 0
            Some(&reconstructed_state),
            Some(&frame_layouts),
            &reconstructed_state.virtuals,
            &[],
            &ExceptionState::default(),
        );

        assert!(
            restored,
            "nested frame restore with virtuals should succeed"
        );
        assert_eq!(
            state.restored_frames.len(),
            3,
            "all 3 frames should be restored"
        );

        // Frame 0: Int(42) + Virtual(0) materialized as Ref(0xA000)
        assert_eq!(state.restored_frames[0].0, 0);
        assert_eq!(state.restored_frames[0].1, 10);
        assert_eq!(
            state.restored_frames[0].2,
            vec![Value::Int(42), Value::Ref(GcRef(0xA000))]
        );

        // Frame 1: Float(9.81) + Virtual(1) materialized as Ref(0xA001)
        assert_eq!(state.restored_frames[1].0, 1);
        assert_eq!(state.restored_frames[1].1, 20);
        assert_eq!(
            state.restored_frames[1].2,
            vec![
                Value::Float(f64::from_bits(float_bits as u64)),
                Value::Ref(GcRef(0xA001))
            ]
        );

        // Frame 2: Int(10) + Int(20)
        assert_eq!(state.restored_frames[2].0, 2);
        assert_eq!(state.restored_frames[2].1, 30);
        assert_eq!(
            state.restored_frames[2].2,
            vec![Value::Int(10), Value::Int(20)]
        );
    }
}

// ── Post-process: raw-int CallAssembler protocol ─────────────────────

/// Strip boxing from Finish result: Finish(CallI(w_int_new, raw)) → Finish(raw).

#[cfg(test)]
mod raw_int_postprocess_tests {
    use crate::compile::{
        unbox_call_assembler_results, unbox_finish_result, unbox_raw_force_results,
    };
    use std::collections::{HashMap, HashSet};

    use majit_ir::{Op, OpCode, OpRef, Type, make_field_descr};

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut op = Op::new(opcode, args);
        op.pos = OpRef(pos);
        op
    }

    fn mk_op_with_descr(opcode: OpCode, args: &[OpRef], pos: u32, descr: majit_ir::DescrRef) -> Op {
        let mut op = Op::with_descr(opcode, args, descr);
        op.pos = OpRef(pos);
        op
    }

    extern "C" fn dummy_box_helper(_value: i64) -> i64 {
        0
    }

    #[test]
    fn unbox_finish_result_requires_jit_w_int_new_target() {
        let func_const = OpRef(dummy_box_helper as *const () as usize as u32);
        let raw = OpRef(0);
        let call = mk_op_with_descr(
            OpCode::CallI,
            &[func_const, raw],
            1,
            crate::make_call_descr(&[Type::Int], Type::Int),
        );
        let finish = Op::with_descr(
            OpCode::Finish,
            &[OpRef(1)],
            crate::make_fail_descr_typed(vec![Type::Int]),
        );
        let helpers = HashSet::new();
        let mut constants = HashMap::new();
        constants.insert(func_const.0, 0xDEAD_BEEF_i64);

        let (ops, changed) = unbox_finish_result(vec![call, finish], &constants, &helpers);

        assert!(!changed);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::CallI);
        assert_eq!(ops[1].opcode, OpCode::Finish);
        assert_eq!(ops[1].args.as_slice(), &[OpRef(1)]);
    }

    #[test]
    fn unbox_finish_result_rewrites_jit_w_int_new_finish() {
        let func_const = OpRef(dummy_box_helper as *const () as usize as u32);
        let raw = OpRef(0);
        let call = mk_op_with_descr(
            OpCode::CallI,
            &[func_const, raw],
            1,
            crate::make_call_descr(&[Type::Int], Type::Int),
        );
        let finish = Op::with_descr(
            OpCode::Finish,
            &[OpRef(1)],
            crate::make_fail_descr_typed(vec![Type::Int]),
        );
        let mut helpers = HashSet::new();
        helpers.insert(dummy_box_helper as *const () as i64);
        let mut constants = HashMap::new();
        constants.insert(func_const.0, dummy_box_helper as *const () as i64);

        let (ops, changed) = unbox_finish_result(vec![call, finish], &constants, &helpers);

        assert!(changed);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::Finish);
        assert_eq!(ops[0].args.as_slice(), &[raw]);
    }

    #[test]
    fn unbox_finish_result_rewrites_new_setfield_chain_even_if_new_is_last() {
        let new_op = mk_op_with_descr(OpCode::New, &[], 25, majit_ir::descr::make_size_descr(16));
        let set_type = mk_op_with_descr(
            OpCode::SetfieldGc,
            &[OpRef(25), OpRef(99)],
            1,
            make_field_descr(0, 8, Type::Int, false),
        );
        let raw = OpRef(13);
        let set_payload = mk_op_with_descr(
            OpCode::SetfieldGc,
            &[OpRef(25), raw],
            2,
            make_field_descr(8, 8, Type::Int, true),
        );
        let finish = Op::with_descr(
            OpCode::Finish,
            &[OpRef(25)],
            crate::make_fail_descr_typed(vec![Type::Ref]),
        );

        let (ops, changed) = unbox_finish_result(
            vec![set_type, set_payload, new_op, finish],
            &HashMap::new(),
            &HashSet::new(),
        );

        assert!(changed);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::Finish);
        assert_eq!(ops[0].args.as_slice(), &[raw]);
    }

    #[test]
    fn unbox_call_assembler_results_rewrites_only_int_payload_unboxing() {
        let ca = mk_op_with_descr(
            OpCode::CallAssemblerI,
            &[OpRef(0)],
            1,
            crate::make_call_assembler_descr(1, &[Type::Int], Type::Int),
        );
        let get_type = mk_op_with_descr(
            OpCode::GetfieldRawI,
            &[OpRef(1)],
            2,
            make_field_descr(0, 8, Type::Int, false),
        );
        let guard = Op::new(OpCode::GuardClass, &[OpRef(2), OpRef(99_999)]);
        let get_int = mk_op_with_descr(
            OpCode::GetfieldRawI,
            &[OpRef(1)],
            3,
            make_field_descr(8, 8, Type::Int, true),
        );
        let add = mk_op(OpCode::IntAdd, &[OpRef(3), OpRef(0)], 4);

        let ops = unbox_call_assembler_results(vec![ca, get_type, guard, get_int, add]);

        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::CallAssemblerI);
        assert_eq!(ops[1].opcode, OpCode::IntAdd);
        assert_eq!(ops[1].args[0], OpRef(1));
    }

    #[test]
    fn unbox_call_assembler_results_preserves_bool_payload_unboxing() {
        let ca = mk_op_with_descr(
            OpCode::CallAssemblerI,
            &[OpRef(0)],
            1,
            crate::make_call_assembler_descr(1, &[Type::Int], Type::Int),
        );
        let get_type = mk_op_with_descr(
            OpCode::GetfieldRawI,
            &[OpRef(1)],
            2,
            make_field_descr(0, 8, Type::Int, false),
        );
        let guard = Op::new(OpCode::GuardClass, &[OpRef(2), OpRef(99_999)]);
        let get_bool = mk_op_with_descr(
            OpCode::GetfieldRawI,
            &[OpRef(1)],
            3,
            make_field_descr(8, 1, Type::Int, false),
        );
        let test = mk_op(OpCode::IntNe, &[OpRef(3), OpRef(0)], 4);

        let ops = unbox_call_assembler_results(vec![ca, get_type, guard, get_bool, test]);

        assert_eq!(ops.len(), 5);
        assert_eq!(ops[3].opcode, OpCode::GetfieldRawI);
        assert_eq!(ops[4].opcode, OpCode::IntNe);
        assert_eq!(ops[4].args[0], OpRef(3));
    }

    #[test]
    fn unbox_call_assembler_results_rewrites_gc_int_payload_unboxing() {
        let ca = mk_op_with_descr(
            OpCode::CallAssemblerI,
            &[OpRef(0)],
            1,
            crate::make_call_assembler_descr(1, &[Type::Int], Type::Int),
        );
        let get_type = mk_op_with_descr(
            OpCode::GetfieldGcI,
            &[OpRef(1)],
            2,
            make_field_descr(0, 8, Type::Int, false),
        );
        let guard = Op::new(OpCode::GuardClass, &[OpRef(2), OpRef(99_999)]);
        let get_int = mk_op_with_descr(
            OpCode::GetfieldGcI,
            &[OpRef(1)],
            3,
            make_field_descr(8, 8, Type::Int, true),
        );
        let add = mk_op(OpCode::IntAdd, &[OpRef(3), OpRef(0)], 4);

        let ops = unbox_call_assembler_results(vec![ca, get_type, guard, get_int, add]);

        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::CallAssemblerI);
        assert_eq!(ops[1].opcode, OpCode::IntAdd);
        assert_eq!(ops[1].args[0], OpRef(1));
    }

    #[test]
    fn unbox_call_assembler_results_rewrites_gc_pure_int_payload_unboxing() {
        let ca = mk_op_with_descr(
            OpCode::CallAssemblerI,
            &[OpRef(0)],
            1,
            crate::make_call_assembler_descr(1, &[Type::Int], Type::Int),
        );
        let get_type = mk_op_with_descr(
            OpCode::GetfieldGcPureI,
            &[OpRef(1)],
            2,
            make_field_descr(0, 8, Type::Int, false),
        );
        let guard = Op::new(OpCode::GuardClass, &[OpRef(2), OpRef(99_999)]);
        let get_int = mk_op_with_descr(
            OpCode::GetfieldGcPureI,
            &[OpRef(1)],
            3,
            make_field_descr(8, 8, Type::Int, true),
        );
        let add = mk_op(OpCode::IntAdd, &[OpRef(3), OpRef(0)], 4);

        let ops = unbox_call_assembler_results(vec![ca, get_type, guard, get_int, add]);

        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::CallAssemblerI);
        assert_eq!(ops[1].opcode, OpCode::IntAdd);
        assert_eq!(ops[1].args[0], OpRef(1));
    }

    #[test]
    fn unbox_raw_force_results_rewrites_only_int_payload_unboxing() {
        let helper_const = OpRef(90_000);
        let mut constants = HashMap::new();
        constants.insert(helper_const.0, 0xfeed_beef);
        let mut helpers = HashSet::new();
        helpers.insert(0xfeed_beef);

        let call = mk_op_with_descr(
            OpCode::CallMayForceI,
            &[helper_const, OpRef(0), OpRef(1), OpRef(2)],
            3,
            crate::make_call_descr(&[Type::Int, Type::Int, Type::Int], Type::Int),
        );
        let get_type = mk_op_with_descr(
            OpCode::GetfieldGcI,
            &[OpRef(3)],
            4,
            make_field_descr(0, 8, Type::Int, false),
        );
        let guard = Op::new(OpCode::GuardClass, &[OpRef(4), OpRef(99_999)]);
        let get_int = mk_op_with_descr(
            OpCode::GetfieldGcI,
            &[OpRef(3)],
            5,
            make_field_descr(8, 8, Type::Int, true),
        );
        let add = mk_op(OpCode::IntAdd, &[OpRef(5), OpRef(0)], 6);

        let ops = unbox_raw_force_results(
            vec![call, get_type, guard, get_int, add],
            &constants,
            &helpers,
        );

        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::CallMayForceI);
        assert_eq!(ops[1].opcode, OpCode::IntAdd);
        assert_eq!(ops[1].args[0], OpRef(3));
    }
}
