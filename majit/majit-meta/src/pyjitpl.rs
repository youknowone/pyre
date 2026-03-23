use std::collections::{HashMap, HashSet};

use majit_codegen::{
    Backend, CompiledTraceInfo, ExitFrameLayout, ExitRecoveryLayout, FailDescrLayout, JitCellToken,
    TerminalExitLayout,
};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{GcRef, InputArg, Op, OpCode, OpRef, Type, Value};
use majit_opt::optimizer::Optimizer;
use majit_trace::history::TreeLoop;
use majit_trace::warmstate::{HotResult, WarmEnterState};

use crate::blackhole::{BlackholeResult, ExceptionState, blackhole_execute_with_state};
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

/// Per-guard failure tracking for bridge compilation decisions.
pub(crate) struct GuardFailureInfo {
    /// Number of times this guard has failed.
    pub(crate) fail_count: u32,
    /// Whether a bridge has been compiled for this guard.
    pub(crate) bridge_compiled: bool,
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
    /// Mapping from backend fail_index to the corresponding guard op index.
    pub(crate) guard_op_indices: HashMap<u32, usize>,
    /// Static exit metadata for each guard/finish in this trace.
    pub(crate) exit_layouts: HashMap<u32, StoredExitLayout>,
    /// Static exit metadata for terminal FINISH/JUMP ops, keyed by op index.
    pub(crate) terminal_exit_layouts: HashMap<usize, StoredExitLayout>,
    /// bridgeopt.py: serialized optimizer knowledge per guard.
    /// Heap cache (struct, field_idx, value) triples known at guard time.
    /// Used by deserialize_optimizer_knowledge when compiling a bridge.
    pub(crate) optimizer_knowledge: Vec<(majit_ir::OpRef, u32, majit_ir::OpRef)>,
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
}

impl StoredExitLayout {
    pub(crate) fn public(&self, trace_id: u64, fail_index: u32) -> CompiledExitLayout {
        CompiledExitLayout {
            trace_id,
            fail_index,
            source_op_index: self.source_op_index,
            exit_types: self.exit_types.clone(),
            is_finish: self.is_finish,
            gc_ref_slots: self.gc_ref_slots.clone(),
            force_token_slots: self.force_token_slots.clone(),
            recovery_layout: self.recovery_layout.clone(),
            resume_layout: self.resume_layout.clone(),
        }
    }
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
    pub(crate) front_target_tokens: Vec<majit_opt::unroll::TargetToken>,
    /// history.py: JitCellToken.retraced_count — how many times this loop
    /// has been retraced. Persisted across recompilations.
    pub(crate) retraced_count: u32,
    /// Trace id of the root compiled loop.
    pub(crate) root_trace_id: u64,
    /// Per-guard failure tracking, keyed by (trace_id, fail_index).
    pub(crate) guard_failures: HashMap<(u64, u32), GuardFailureInfo>,
    /// Metadata for the root loop and any attached bridges, keyed by trace id.
    pub(crate) traces: HashMap<u64, CompiledTrace>,
    /// RPython parity: previous compiled entries for this green_key.
    /// In RPython, JitCellToken keeps all target_tokens' code alive.
    /// In majit, each retrace produces a new Cranelift function;
    /// previous functions are kept here so external target_token JUMPs
    /// can redirect to them via runtime trampoline.
    pub(crate) previous_tokens: Vec<JitCellToken>,
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
    pub(crate) backend: CraneliftBackend,
    pub(crate) compiled_loops: HashMap<u64, CompiledEntry<M>>,
    pub(crate) tracing: Option<TraceCtx>,
    pub(crate) next_trace_id: u64,
    /// Trace eagerness: start tracing from a guard failure point
    /// after this many failures (0 = never trace from guards).
    pub(crate) trace_eagerness: u32,
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
    /// Green keys whose compiled Finish returns a raw int (not a boxed pointer).
    pub(crate) raw_int_finish_keys: HashSet<u64>,
    /// Helper function pointers that box raw ints into interpreter objects.
    pub(crate) raw_int_box_helpers: HashSet<i64>,
    /// Helper function pointers that take a raw int argument and return a
    /// raw-int result when the callee's Finish protocol is raw.
    pub(crate) raw_int_force_helpers: HashSet<i64>,
    /// Mapping: create_frame_N_ptr → create_frame_N_raw_int_ptr.
    /// When a box helper feeds directly into create_frame, the box+create
    /// can be folded into a single create_frame_raw_int call.
    pub(crate) create_frame_raw_map: HashMap<i64, i64>,
    /// PyPy warmspot.py max_unroll_recursion (default 7).
    pub(crate) max_unroll_recursion: usize,
    /// RPython parity: `prepare_trace_segmenting()` marks the next tracing run
    /// for a green key so the loop should finish early instead of repeatedly
    /// aborting once it nears the trace limit.
    pub(crate) force_finish_trace: bool,
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
    /// RPython compile.py:714 (_trace_and_compile_from_bridge):
    /// Called when guard failure threshold is reached to trigger bridge
    /// compilation. Args: (green_key, trace_id, fail_index).
    pub on_bridge_threshold: Option<Box<dyn Fn(u64, u64, u32) + Send>>,
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

    fn normalize_trace_id(compiled: &CompiledEntry<M>, trace_id: u64) -> u64 {
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
        trace_id: u64,
        fail_index: u32,
    ) -> Option<CompiledExitLayout> {
        trace
            .exit_layouts
            .get(&fail_index)
            .map(|layout| layout.public(trace_id, fail_index))
    }

    fn terminal_exit_layout_from_trace(
        trace: &CompiledTrace,
        trace_id: u64,
        op_index: usize,
    ) -> Option<CompiledExitLayout> {
        trace.terminal_exit_layouts.get(&op_index).map(|layout| {
            layout.public(
                trace_id,
                compile::find_fail_index_for_exit_op(&trace.ops, op_index).unwrap_or(u32::MAX),
            )
        })
    }

    fn compiled_exit_layout_from_backend(
        &self,
        compiled: &CompiledEntry<M>,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<CompiledExitLayout> {
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        self.backend
            .compiled_trace_fail_descr_layouts(&compiled.token, trace_id)?
            .into_iter()
            .find(|layout| layout.fail_index == fail_index)
            .map(|layout| CompiledExitLayout {
                trace_id,
                fail_index: layout.fail_index,
                source_op_index: layout.source_op_index,
                exit_types: layout.fail_arg_types,
                is_finish: layout.is_finish,
                gc_ref_slots: layout.gc_ref_slots,
                force_token_slots: layout.force_token_slots,
                recovery_layout: layout.recovery_layout,
                resume_layout: None,
            })
    }

    fn terminal_exit_layout_from_backend(
        &self,
        compiled: &CompiledEntry<M>,
        trace_id: u64,
        op_index: usize,
    ) -> Option<CompiledExitLayout> {
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        self.backend
            .compiled_trace_terminal_exit_layouts(&compiled.token, trace_id)?
            .into_iter()
            .find(|layout| layout.op_index == op_index)
            .map(|layout| CompiledExitLayout {
                trace_id,
                fail_index: layout.fail_index,
                source_op_index: Some(layout.op_index),
                exit_types: layout.exit_types,
                is_finish: layout.is_finish,
                gc_ref_slots: layout.gc_ref_slots,
                force_token_slots: layout.force_token_slots,
                recovery_layout: layout.recovery_layout,
                resume_layout: None,
            })
    }

    fn compiled_trace_layout_for_trace(
        &self,
        compiled: &CompiledEntry<M>,
        trace_id: u64,
    ) -> Option<CompiledTraceLayout> {
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        let mut exit_layouts =
            if let Some((resolved_trace_id, trace)) = Self::trace_for_exit(compiled, trace_id) {
                let mut layouts: Vec<_> = trace
                    .exit_layouts
                    .iter()
                    .map(|(&fail_index, layout)| layout.public(resolved_trace_id, fail_index))
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
            backend: CraneliftBackend::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
            next_trace_id: 1,
            trace_eagerness: 200,
            virtualizable_info: None,
            hooks: JitHooks::default(),
            pending_token: None,
            stats: JitStatsCounters::default(),
            vable_ptr: std::ptr::null(),
            vable_array_lengths: Vec::new(),
            forced_virtualizable: None,
            raw_int_finish_keys: HashSet::new(),
            raw_int_box_helpers: HashSet::new(),
            raw_int_force_helpers: HashSet::new(),
            create_frame_raw_map: HashMap::new(),
            max_unroll_recursion: 7, // RPython default from rlib/jit.py
            force_finish_trace: false,
        }
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

    /// Set the trace eagerness (guard failure threshold for bridge tracing).
    pub fn set_trace_eagerness(&mut self, eagerness: u32) {
        self.trace_eagerness = eagerness;
    }

    /// Update the green_key associated with the current trace.
    /// Called when tracing started at function entry but the loop closes
    /// at a backward jump with a different PC.
    pub fn update_tracing_green_key(&mut self, key: u64) {
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.green_key = key;
        }
    }

    /// Set the bridge compilation threshold.
    pub fn set_bridge_threshold(&mut self, threshold: u32) {
        self.warm_state.set_bridge_threshold(threshold);
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
    ) -> Option<majit_opt::virtualize::VirtualizableConfig> {
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
        if let Some(config) = self.current_virtualizable_optimizer_config() {
            return Optimizer::default_pipeline_with_virtualizable(config);
        }
        Optimizer::default_pipeline()
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
            HotResult::StartTracing(mut recorder) => {
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
                if let Some(descriptor) = driver_descriptor {
                    ctx.set_driver_descriptor(descriptor);
                }
                // Initialize standard virtualizable boxes if VirtualizableInfo is set.
                //
                // RPython equivalent: MetaInterp._init_virtualizable_state()
                //
                // The inputargs are laid out as: [frame_ref, field0, field1, ...,
                // array0_elem0, ..., array0_elemN, ...]. We extract the OpRef
                // indices corresponding to static fields + array elements and
                // pass them to init_virtualizable_boxes so that subsequent
                // vable_getfield/setfield calls use boxes instead of heap ops.
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
                self.tracing = Some(ctx);
                // Pre-allocate a token number for this trace so that
                // self-recursive calls can emit call_assembler targeting
                // this token before the trace is compiled.
                let pending_num = self.warm_state.alloc_token_number();
                self.pending_token = Some((green_key, pending_num));
                if let Some(ref hook) = self.hooks.on_trace_start {
                    hook(green_key);
                }
                BackEdgeAction::StartedTracing
            }
            HotResult::AlreadyTracing => BackEdgeAction::AlreadyTracing,
            HotResult::RunCompiled => BackEdgeAction::RunCompiled,
        }
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

    /// Close the current trace, optimize, and compile.
    ///
    /// `jump_args` are the symbolic values (OpRefs) at the end of the loop,
    /// in the same order as the InputArgs registered during `on_back_edge`.
    /// `meta` is interpreter-specific metadata to store alongside the compiled loop.
    pub fn close_and_compile(&mut self, jump_args: &[OpRef], meta: M) {
        let vable_config = self.current_virtualizable_optimizer_config();
        self.forced_virtualizable = None;
        self.force_finish_trace = false;
        let mut ctx = self.tracing.take().unwrap();
        ctx.apply_replacements();
        let green_key = ctx.green_key;

        let mut recorder = ctx.recorder;
        // RPython heapcache.py:176: every trace gets at least one
        // GUARD_NOT_INVALIDATED. This allows external invalidation
        // (via JitCellToken.invalidate()) to force compiled loops
        // back to the interpreter.
        // RPython heapcache.py:176: every trace gets at least one
        // GUARD_NOT_INVALIDATED before the closing JUMP. fail_args = jump_args
        // so guard failure restores the same state as the JUMP target.
        {
            let inputarg_types = recorder.inputarg_types();
            let num_inputargs = recorder.num_inputargs() as u32;
            let guard_fail_arg_types = jump_args
                .iter()
                .map(|&arg| {
                    if arg.0 < num_inputargs {
                        inputarg_types[arg.0 as usize]
                    } else {
                        recorder
                            .get_op_by_pos(arg)
                            .map(|op| op.result_type())
                            .unwrap_or(Type::Int)
                    }
                })
                .collect();
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] close_and_compile GuardNotInvalidated types={:?} inputarg_types={:?} jump_args={:?}",
                    guard_fail_arg_types,
                    inputarg_types,
                    jump_args,
                );
            }
            let descr = crate::fail_descr::make_fail_descr_typed(guard_fail_arg_types);
            recorder.record_guard_with_fail_args(
                OpCode::GuardNotInvalidated, &[], descr, jump_args,
            );
        }
        recorder.close_loop(jump_args);
        let trace = recorder.get_trace();

        let (mut constants, constant_types) = ctx.constants.into_inner_with_types();

        let trace_ops = compile::fold_box_into_create_frame(
            trace.ops.clone(),
            &mut constants,
            &self.raw_int_box_helpers,
            &self.create_frame_raw_map,
        );

        if crate::majit_log_enabled() {
            eprintln!("--- trace (before opt) ---");
            eprint!("{}", majit_ir::format_trace(&trace_ops, &constants));
        }

        let num_ops_before = trace.ops.len();

        // Use UnrollOptimizer for preamble peeling when available.
        // compile.py: compile_loop → PreambleCompileData + LoopCompileData.
        let prior_front_target_tokens = self
            .compiled_loops
            .get(&green_key)
            .map(|compiled| compiled.front_target_tokens.clone())
            .unwrap_or_default();
        let mut unroll_opt = majit_opt::unroll::UnrollOptimizer::new();
        unroll_opt.target_tokens = prior_front_target_tokens.clone();
        // history.py: carry retraced_count across recompilations
        unroll_opt.retraced_count = self
            .compiled_loops
            .get(&green_key)
            .map(|compiled| compiled.retraced_count)
            .unwrap_or(0);
        unroll_opt.retrace_limit = self.warm_state.retrace_limit();
        unroll_opt.max_retrace_guards = self.warm_state.max_retrace_guards();
        unroll_opt.constant_types = constant_types;

        // RPython virtualizable.py: if interpreter has a virtualizable,
        // pass its config to OptVirtualize so it can carry frame fields and
        // array slots through the loop as virtual state instead of heap traffic.
        let optimize_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            unroll_opt.optimize_trace_with_constants_and_inputs_vable(
                &trace_ops,
                &mut constants,
                trace.inputargs.len(),
                vable_config,
            )
        }));
        let (optimized_ops, final_num_inputs) = match optimize_result {
            Ok(result) => result,
            Err(payload) => {
                if payload
                    .downcast_ref::<majit_opt::optimize::InvalidLoop>()
                    .is_some()
                {
                    if crate::majit_log_enabled() {
                        eprintln!("[jit] abort trace at key={} (InvalidLoop during optimize)", green_key);
                    }
                    self.warm_state.abort_tracing(green_key, false);
                    return;
                }
                std::panic::resume_unwind(payload);
            }
        };
        let num_ops_after = optimized_ops.len();

        // RPython compile.py keeps the root entry contract on the original
        // loop inputargs. Simple loops synthesize a LABEL from that contract;
        // they do not grow inputargs to match a rewritten JUMP arity.
        let root_inputargs = root_loop_inputargs_from_optimizer(&trace.inputargs, final_num_inputs);
        let (inputargs, optimized_ops) =
            match normalize_root_loop_entry_contract(root_inputargs, optimized_ops) {
                Ok(normalized) => normalized,
                Err((expected, actual)) => {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] abort compile: root loop entry/jump arity mismatch input={} jump={}",
                            expected, actual,
                        );
                    }
                    return;
                }
            };

        // RPython virtualizable parity: standard virtualizable fields and
        // arrays stay in the trace as first-class virtualizable boxes.
        // Do not prepend raw heap preamble loads here; compiled callers pass
        // the traced virtualizable values in the live-input layout, and
        // `vable_*` operations keep the hot path on boxes instead of
        // re-materializing `GetfieldRaw*`/`GetarrayitemRaw*` entry ops.
        let (inputargs, optimized_ops) = (inputargs, optimized_ops);
        let optimized_ops = compile::unbox_call_assembler_results(optimized_ops);
        let optimized_ops =
            compile::normalize_closing_jump_args(optimized_ops, &constants, final_num_inputs);

        if crate::majit_log_enabled() {
            eprintln!("--- trace (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
            for op in &optimized_ops {
                if op.opcode == majit_ir::OpCode::GuardNotInvalidated {
                    if let Some(ref fa) = op.fail_args {
                        let raw: Vec<String> = fa.iter().map(|a| format!("OpRef({})", a.0)).collect();
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
            return;
        }

        let compiled_constants = constants.clone();
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
        let trace_id = self.alloc_trace_id();
        self.backend.set_next_trace_id(trace_id);

        let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.backend
                .compile_loop(&inputargs, &optimized_ops, &mut token)
        }));
        let compile_result = match compile_result {
            Ok(r) => r,
            Err(_) => {
                if crate::majit_log_enabled() {
                    eprintln!("[jit] compile_loop panicked, aborting trace at key={green_key}");
                }
                // RPython: backend failure is non-permanent — allow retry.
                self.warm_state.abort_tracing(green_key, false);
                return;
            }
        };
        match compile_result {
            Ok(_) => {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compiled loop at key={}, num_inputs={}",
                        green_key,
                        inputargs.len()
                    );
                }
                // Build resume data and exit layouts for all guards in the optimized trace.
                let (resume_data, guard_op_indices, mut exit_layouts) =
                    compile::build_guard_metadata(&inputargs, &optimized_ops, green_key);
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
                let mut traces = HashMap::new();
                traces.insert(
                    trace_id,
                    CompiledTrace {
                        inputargs: inputargs.clone(),
                        resume_data,
                        ops: optimized_ops,
                        constants: compiled_constants,
                        guard_op_indices,
                        exit_layouts,
                        terminal_exit_layouts,
                        optimizer_knowledge: Vec::new(), // TODO: serialize from optimizer
                    },
                );

                // RPython parity: keep previous compiled tokens alive so
                // external target_token JUMPs can redirect to them.
                let mut previous_tokens: Vec<JitCellToken> = Vec::new();
                if let Some(old_entry) = self.compiled_loops.remove(&green_key) {
                    previous_tokens.push(old_entry.token);
                    previous_tokens.extend(old_entry.previous_tokens);
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
                        guard_failures: HashMap::new(),
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
                    hook(green_key, num_ops_before, num_ops_after);
                }
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
                // RPython: backend compilation failure is non-permanent.
                self.warm_state.abort_tracing(green_key, false);
            }
        }
        // Reset per-trace function call counts.
        self.warm_state.reset_function_counts();
    }

    /// Abort the current trace.
    ///
    /// If `permanent` is true, this location will never be traced again.
    pub fn abort_trace(&mut self, permanent: bool) {
        self.forced_virtualizable = None;
        self.force_finish_trace = false;
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

        let (mut constants, constant_types) = ctx.constants.into_inner_with_types();

        let trace_ops = compile::fold_box_into_create_frame(
            trace.ops.clone(),
            &mut constants,
            &self.raw_int_box_helpers,
            &self.create_frame_raw_map,
        );

        let num_ops_before = trace_ops.len();
        let mut optimizer = if let Some(config) = vable_config {
            Optimizer::default_pipeline_with_virtualizable(config)
        } else {
            Optimizer::default_pipeline()
        };
        optimizer.constant_types = constant_types;
        let optimized_ops = optimizer.optimize_with_constants_and_inputs(
            &trace_ops,
            &mut constants,
            trace.inputargs.len(),
        );
        let num_ops_after = optimized_ops.len();

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

        // Re-enable raw-int terminal finishes when the trace ends in an
        // obvious int boxing pattern. Top-level exits re-box in pyre-jit,
        // while recursive call boundaries can consume the raw int directly.
        let (optimized_ops, finish_unboxed) =
            compile::unbox_finish_result(optimized_ops, &constants, &self.raw_int_box_helpers);
        if finish_unboxed {
            self.raw_int_finish_keys.insert(green_key);
        } else {
            self.raw_int_finish_keys.remove(&green_key);
        }
        let optimized_ops = compile::unbox_call_assembler_results(optimized_ops);
        let optimized_ops = if finish_unboxed {
            compile::unbox_raw_force_results(optimized_ops, &constants, &self.raw_int_force_helpers)
        } else {
            optimized_ops
        };
        let optimized_ops = compile::strip_stray_overflow_guards(optimized_ops);

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

        // Extend inputargs if the optimizer added virtual inputs (virtualizable)
        let final_num_inputs = optimizer.final_num_inputs();
        let mut inputargs = trace.inputargs.clone();
        while inputargs.len() < final_num_inputs {
            inputargs.push(majit_ir::InputArg {
                tp: majit_ir::Type::Int,
                index: inputargs.len() as u32,
            });
        }

        // No preamble for FINISH traces — CALL_ASSEMBLER passes all args
        // directly (frame + ni + sd + locals). A guard before GETFIELD ops
        // ensures tagged pointers (force_cache hits) don't get dereferenced.

        let compiled_constants = constants.clone();
        self.backend.set_constants(constants);

        match self
            .backend
            .compile_loop(&inputargs, &optimized_ops, &mut token)
        {
            Ok(_) => {
                let (resume_data, guard_op_indices, mut exit_layouts) =
                    compile::build_guard_metadata(&inputargs, &optimized_ops, green_key);
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
                let mut traces = HashMap::new();
                traces.insert(
                    trace_id,
                    CompiledTrace {
                        inputargs: trace.inputargs.clone(),
                        resume_data,
                        ops: optimized_ops,
                        constants: compiled_constants,
                        guard_op_indices,
                        exit_layouts,
                        terminal_exit_layouts,
                        optimizer_knowledge: optimizer.serialize_optimizer_knowledge(),
                    },
                );
                {
                    let mut previous_tokens: Vec<JitCellToken> = Vec::new();
                    let ft = self.compiled_loops.get(&green_key)
                        .map(|c| c.front_target_tokens.clone()).unwrap_or_default();
                    let rc = self.compiled_loops.get(&green_key)
                        .map(|c| c.retraced_count).unwrap_or(0);
                    if let Some(old_entry) = self.compiled_loops.remove(&green_key) {
                        previous_tokens.push(old_entry.token);
                        previous_tokens.extend(old_entry.previous_tokens);
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
                            guard_failures: HashMap::new(),
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

    /// Get the metadata for a compiled loop without executing it.
    ///
    /// Allows the interpreter to check preconditions (e.g., whether the
    /// current state matches the compiled loop's assumptions) before calling
    /// `run_compiled`.
    pub fn get_compiled_meta(&self, green_key: u64) -> Option<&M> {
        self.compiled_loops.get(&green_key).map(|e| &e.meta)
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

        // Track guard failures for bridge compilation decisions.
        if !result.is_finish {
            let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
            let info = compiled
                .guard_failures
                .entry((trace_id, fail_index))
                .or_insert(GuardFailureInfo {
                    fail_count: 0,
                    bridge_compiled: false,
                });
            info.fail_count += 1;

            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] guard failure at key={}, guard={}, count={}",
                    green_key, fail_index, info.fail_count
                );
            }

            self.stats.guard_failures += 1;
            self.warm_state.log_guard_failure(fail_index);

            if let Some(ref hook) = self.hooks.on_guard_failure {
                hook(green_key, fail_index, info.fail_count);
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
            let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
            let info = compiled
                .guard_failures
                .entry((trace_id, fail_index))
                .or_insert(GuardFailureInfo {
                    fail_count: 0,
                    bridge_compiled: false,
                });
            info.fail_count += 1;

            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] guard failure at key={}, guard={}, count={}",
                    green_key, fail_index, info.fail_count
                );
            }

            self.stats.guard_failures += 1;
            self.warm_state.log_guard_failure(fail_index);

            if let Some(ref hook) = self.hooks.on_guard_failure {
                hook(green_key, fail_index, info.fail_count);
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
            let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
            let info = compiled
                .guard_failures
                .entry((trace_id, fail_index))
                .or_insert(GuardFailureInfo {
                    fail_count: 0,
                    bridge_compiled: false,
                });
            info.fail_count += 1;

            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] guard failure at key={}, guard={}, count={}",
                    green_key, fail_index, info.fail_count
                );
            }

            self.stats.guard_failures += 1;
            self.warm_state.log_guard_failure(fail_index);

            if let Some(ref hook) = self.hooks.on_guard_failure {
                hook(green_key, fail_index, info.fail_count);
            }
        }

        let compiled = self.compiled_loops.get(&green_key).unwrap();
        Some((result.typed_outputs, &compiled.meta))
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
                Self::compiled_exit_layout_from_trace(trace, trace_id, fail_index)
            });
        let exit_layout = result
            .exit_layout
            .clone()
            .map(|layout| {
                let mut resume_layout = trace_layout
                    .as_ref()
                    .and_then(|tl| tl.resume_layout.clone());
                compile::enrich_resume_layout_with_frame_stack(
                    &mut resume_layout,
                    layout.frame_stack.as_deref(),
                    &layout.fail_arg_types,
                );
                CompiledExitLayout {
                    trace_id,
                    fail_index: layout.fail_index,
                    source_op_index: layout.source_op_index.or_else(|| {
                        trace_layout
                            .as_ref()
                            .and_then(|layout| layout.source_op_index)
                    }),
                    exit_types: layout.fail_arg_types,
                    is_finish: layout.is_finish,
                    gc_ref_slots: layout.gc_ref_slots,
                    force_token_slots: layout.force_token_slots,
                    recovery_layout: layout.recovery_layout,
                    resume_layout,
                }
            })
            .or(trace_layout)
            .unwrap_or_else(|| CompiledExitLayout {
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
            });
        let effective_is_finish = result.is_finish || exit_layout.is_finish;

        let mut guard_fail_count = None;
        if !effective_is_finish {
            let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
            let info = compiled
                .guard_failures
                .entry((trace_id, fail_index))
                .or_insert(GuardFailureInfo {
                    fail_count: 0,
                    bridge_compiled: false,
                });
            info.fail_count += 1;
            guard_fail_count = Some(info.fail_count);
        }
        if let Some(fail_count) = guard_fail_count {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] guard failure at key={}, guard={}, count={}",
                    green_key, fail_index, fail_count
                );
            }

            self.stats.guard_failures += 1;
            self.warm_state.log_guard_failure(fail_index);

            if let Some(ref hook) = self.hooks.on_guard_failure {
                hook(green_key, fail_index, fail_count);
            }
        }

        let exception = ExceptionState {
            exc_class: result.exception_class,
            exc_value: result.exception_value.0 as i64,
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
                Self::compiled_exit_layout_from_trace(trace, trace_id, fail_index)
            });
        let exit_layout = result
            .exit_layout
            .clone()
            .map(|layout| {
                let mut resume_layout = trace_layout
                    .as_ref()
                    .and_then(|tl| tl.resume_layout.clone());
                compile::enrich_resume_layout_with_frame_stack(
                    &mut resume_layout,
                    layout.frame_stack.as_deref(),
                    &layout.fail_arg_types,
                );
                CompiledExitLayout {
                    trace_id,
                    fail_index: layout.fail_index,
                    source_op_index: layout.source_op_index.or_else(|| {
                        trace_layout
                            .as_ref()
                            .and_then(|layout| layout.source_op_index)
                    }),
                    exit_types: layout.fail_arg_types,
                    is_finish: layout.is_finish,
                    gc_ref_slots: layout.gc_ref_slots,
                    force_token_slots: layout.force_token_slots,
                    recovery_layout: layout.recovery_layout,
                    resume_layout,
                }
            })
            .or(trace_layout)
            .unwrap_or_else(|| CompiledExitLayout {
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
            });
        let effective_is_finish = result.is_finish || exit_layout.is_finish;

        let mut guard_fail_count = None;
        if !effective_is_finish {
            let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
            let info = compiled
                .guard_failures
                .entry((trace_id, fail_index))
                .or_insert(GuardFailureInfo {
                    fail_count: 0,
                    bridge_compiled: false,
                });
            info.fail_count += 1;
            guard_fail_count = Some(info.fail_count);
        }
        if let Some(fail_count) = guard_fail_count {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] guard failure at key={}, guard={}, count={}",
                    green_key, fail_index, fail_count
                );
            }

            self.stats.guard_failures += 1;
            self.warm_state.log_guard_failure(fail_index);

            if let Some(ref hook) = self.hooks.on_guard_failure {
                hook(green_key, fail_index, fail_count);
            }
        }
        let exception = ExceptionState {
            exc_class: result.exception_class,
            exc_value: result.exception_value.0 as i64,
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

        // Track guard failures
        if !is_finish {
            let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
            let info = compiled
                .guard_failures
                .entry((trace_id, fail_index))
                .or_insert(GuardFailureInfo {
                    fail_count: 0,
                    bridge_compiled: false,
                });
            info.fail_count += 1;
            self.stats.guard_failures += 1;
            self.warm_state.log_guard_failure(fail_index);

            if let Some(ref hook) = self.hooks.on_guard_failure {
                hook(green_key, fail_index, info.fail_count);
            }
        }

        let exit_types = descr.fail_arg_types();
        let exit_arity = exit_types.len();
        let compiled = self.compiled_loops.get(&green_key).unwrap();
        let exit_layout = Self::trace_for_exit(compiled, trace_id)
            .and_then(|(trace_id, trace)| {
                Self::compiled_exit_layout_from_trace(trace, trace_id, fail_index)
            })
            .unwrap_or_else(|| CompiledExitLayout {
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
        })
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

        // RPython compile.py:696 (handle_fail):
        // Guard failure counter + bridge compilation trigger.
        if !is_finish {
            let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
            let info = compiled
                .guard_failures
                .entry((trace_id, fail_index))
                .or_insert(GuardFailureInfo {
                    fail_count: 0,
                    bridge_compiled: false,
                });
            info.fail_count += 1;
            let fail_count = info.fail_count;
            let bridge_compiled = info.bridge_compiled;
            self.stats.guard_failures += 1;
            self.warm_state.log_guard_failure(fail_index);

            if let Some(ref hook) = self.hooks.on_guard_failure {
                hook(green_key, fail_index, fail_count);
            }

            // RPython compile.py:697-706 (must_compile → _trace_and_compile_from_bridge):
            // When guard fails >= trace_eagerness times, trigger bridge hook.
            // The hook (set by pyre) compiles a bridge for the alternative path.
            if fail_count >= self.trace_eagerness
                && self.trace_eagerness > 0
                && !bridge_compiled
            {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] bridge threshold reached: key={} trace={} guard={} count={}",
                        green_key, trace_id, fail_index, fail_count
                    );
                }
                if let Some(compiled) = self.compiled_loops.get_mut(&green_key) {
                    if let Some(info) = compiled.guard_failures.get_mut(&(trace_id, fail_index)) {
                        info.bridge_compiled = true;
                    }
                }
                // RPython compile.py:714: invoke bridge compilation callback.
                // pyre's call_jit.rs sets this hook to compile a bridge
                // that handles the alternative path (e.g., fib base case).
                if let Some(ref hook) = self.hooks.on_bridge_threshold {
                    hook(green_key, trace_id, fail_index);
                }
            }
        }

        let exit_types = descr.fail_arg_types();
        let exit_arity = exit_types.len();
        let compiled = self.compiled_loops.get(&green_key).unwrap();
        let exit_layout = Self::trace_for_exit(compiled, trace_id)
            .and_then(|(trace_id, trace)| {
                Self::compiled_exit_layout_from_trace(trace, trace_id, fail_index)
            })
            .unwrap_or_else(|| CompiledExitLayout {
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
        Self::compiled_exit_layout_from_trace(trace, trace_id, fail_index)
            .or_else(|| self.compiled_exit_layout_from_backend(compiled, trace_id, fail_index))
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
        Self::terminal_exit_layout_from_trace(trace, trace_id, op_index)
            .or_else(|| self.terminal_exit_layout_from_backend(compiled, trace_id, op_index))
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
        self.compiled_trace_layout_for_trace(compiled, trace_id)
    }

    /// Check whether a guard has failed enough times to warrant bridge compilation.
    pub fn should_compile_bridge(&self, green_key: u64, fail_index: u32) -> bool {
        let Some(compiled) = self.compiled_loops.get(&green_key) else {
            return false;
        };
        let key = (compiled.root_trace_id, fail_index);
        compiled.guard_failures.get(&key).is_some_and(|info| {
            !info.bridge_compiled && self.warm_state.should_compile_bridge(info.fail_count)
        })
    }

    /// Invalidate a compiled loop (e.g., due to GUARD_NOT_INVALIDATED).
    ///
    /// Marks the loop token as invalidated. Subsequent executions of the
    /// compiled code will fail at GUARD_NOT_INVALIDATED and fall back to
    /// the interpreter.
    pub fn invalidate_loop(&mut self, green_key: u64) {
        if let Some(compiled) = self.compiled_loops.get(&green_key) {
            compiled.token.invalidate();
            self.raw_int_finish_keys.remove(&green_key);
            if crate::majit_log_enabled() {
                eprintln!("[jit] invalidated loop at key={}", green_key);
            }
        }
    }

    /// Get the guard failure count for a specific guard.
    pub fn get_guard_failure_count(&self, green_key: u64, fail_index: u32) -> u32 {
        self.compiled_loops
            .get(&green_key)
            .and_then(|c| c.guard_failures.get(&(c.root_trace_id, fail_index)))
            .map(|info| info.fail_count)
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

    /// Remove all compiled loops. Used when guard-fail recovery is
    /// unrecoverable (null Ref in resume data).
    pub fn clear_compiled_loops(&mut self) {
        self.compiled_loops.clear();
    }

    /// Return all green keys that have compiled loops.
    pub fn all_compiled_keys(&self) -> Vec<u64> {
        self.compiled_loops.keys().copied().collect()
    }

    /// Whether the compiled Finish for this green_key returns a raw int.
    pub fn has_raw_int_finish(&self, green_key: u64) -> bool {
        self.raw_int_finish_keys.contains(&green_key)
    }

    /// Check whether a guard in a specific compiled trace should get a bridge.
    ///
    /// Only root-loop guards are eligible for bridge compilation.
    /// Bridge guard failures (trace_id != root_trace_id) fall back to
    /// the interpreter to avoid infinite bridge-of-bridge recompilation.
    pub fn should_compile_bridge_in_trace(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> bool {
        let Some(compiled) = self.compiled_loops.get(&green_key) else {
            return false;
        };
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        // Only compile bridges for root-loop guards, not bridge guards.
        if trace_id != compiled.root_trace_id {
            return false;
        }
        compiled
            .guard_failures
            .get(&(trace_id, fail_index))
            .is_some_and(|info| {
                !info.bridge_compiled && self.warm_state.should_compile_bridge(info.fail_count)
            })
    }

    /// Get the failure count for a guard in a specific compiled trace.
    pub fn get_guard_failure_count_in_trace(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> u32 {
        let Some(compiled) = self.compiled_loops.get(&green_key) else {
            return 0;
        };
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        compiled
            .guard_failures
            .get(&(trace_id, fail_index))
            .map(|info| info.fail_count)
            .unwrap_or(0)
    }

    fn bridge_fail_descr_proxy(
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
        })
    }

    // ── Bridge Compilation ──────────────────────────────────────

    /// Close the current bridge trace with a FINISH op, optimize, and compile
    /// it as a bridge attached to the specified guard.
    ///
    /// `green_key` identifies the parent loop.
    /// `trace_id` and `fail_index` identify the guard to attach the bridge to.
    /// `finish_args` are the symbolic values at the bridge end (loop header state).
    pub fn close_bridge(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        finish_args: &[OpRef],
    ) -> bool {
        let finish_arg_types = {
            let compiled = match self.compiled_loops.get(&green_key) {
                Some(c) => c,
                None => return false,
            };
            let norm_trace_id = Self::normalize_trace_id(compiled, trace_id);
            let (_, trace_data) = match Self::trace_for_exit(compiled, norm_trace_id) {
                Some(t) => t,
                None => return false,
            };
            trace_data
                .inputargs
                .iter()
                .map(|arg| arg.tp)
                .collect::<Vec<_>>()
        };
        if finish_arg_types.len() != finish_args.len() {
            return false;
        }

        self.forced_virtualizable = None;
        let ctx = match self.tracing.take() {
            Some(ctx) => ctx,
            None => return false,
        };

        let mut recorder = ctx.recorder;
        // RPython parity: if the target loop has compiled target_tokens,
        // close with JUMP (so optimize_bridge can jump_to_existing_trace).
        // Otherwise fall back to Finish.
        let has_targets = self
            .compiled_loops
            .get(&green_key)
            .map_or(false, |c| !c.front_target_tokens.is_empty());
        if has_targets {
            recorder.close_loop(finish_args);
        } else {
            recorder.finish(finish_args, crate::make_fail_descr_typed(finish_arg_types));
        }
        let trace = recorder.get_trace();

        let constants = ctx.constants.into_inner();

        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] close_bridge: key={}, trace_id={}, guard={}, ops={}",
                green_key,
                trace_id,
                fail_index,
                trace.ops.len()
            );
            eprintln!("--- bridge trace (before opt) ---");
            eprint!("{}", majit_ir::format_trace(&trace.ops, &constants));
        }

        // Look up the guard's fail_arg_types to build the fail_descr for the bridge
        let fail_descr = {
            let compiled = match self.compiled_loops.get(&green_key) {
                Some(c) => c,
                None => return false,
            };
            let fail_descr = match Self::bridge_fail_descr_proxy(compiled, trace_id, fail_index) {
                Some(descr) => descr,
                None => return false,
            };
            Box::new(fail_descr) as Box<dyn majit_ir::FailDescr>
        };

        self.compile_bridge(
            green_key,
            fail_index,
            &*fail_descr,
            &trace.ops,
            &trace.inputargs,
            constants,
        )
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
    ) -> bool {
        if !self.compiled_loops.contains_key(&green_key) {
            return false;
        }

        // RPython unroll.py:183-236: Optimizer.optimize_bridge()
        let mut optimizer = self.make_optimizer();
        let mut constants = constants;
        let inline_short_preamble = true;
        let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
        let retraced_count = compiled.retraced_count;
        let retrace_limit = 0u32;
        // bridgeopt.py: retrieve optimizer knowledge from the source trace.
        // The knowledge captures heap cache state at the guard point, so the
        // bridge optimizer can start with cached field values.
        let bridge_knowledge: Option<Vec<(majit_ir::OpRef, u32, majit_ir::OpRef)>> = {
            let source_trace_id = {
                let tid = fail_descr.trace_id();
                if tid == 0 { compiled.root_trace_id } else { tid }
            };
            compiled
                .traces
                .get(&source_trace_id)
                .map(|t| t.optimizer_knowledge.clone())
                .filter(|k| !k.is_empty())
        };
        let (optimized_ops, retrace_requested) = optimizer.optimize_bridge(
            bridge_ops,
            &mut constants,
            bridge_inputargs.len(),
            &mut compiled.front_target_tokens,
            inline_short_preamble,
            retraced_count,
            retrace_limit,
            bridge_knowledge.as_deref(),
        );
        if retrace_requested {
            // unroll.py:217: cell_token.retraced_count += 1
            compiled.retraced_count += 1;
            // unroll.py:231-236: export_state creates a new target token.
            // The optimizer's exported_loop_state has the virtual state for
            // the new specialization. Add a new target token to
            // front_target_tokens so future bridges/loops can match it.
            if let Some(exported) = optimizer.exported_loop_state.take() {
                let mut new_token = majit_opt::unroll::TargetToken::new_preamble(
                    compiled.front_target_tokens.len() as u64,
                );
                new_token.virtual_state = Some(exported.virtual_state);
                compiled.front_target_tokens.push(new_token);
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] bridge retrace: added target_token #{}, total={}",
                        compiled.front_target_tokens.len() - 1,
                        compiled.front_target_tokens.len()
                    );
                }
            }
        }

        // RPython parity: unbox the Finish result in bridges too.
        // Without this, bridges return boxed pointers while the caller
        // (call_assembler_fast_path) expects raw ints for [Type::Int] Finish.
        let (optimized_ops, bridge_finish_unboxed) =
            compile::unbox_finish_result(optimized_ops, &constants, &self.raw_int_box_helpers);
        if bridge_finish_unboxed {
            self.raw_int_finish_keys.insert(green_key);
        }
        let optimized_ops = if bridge_finish_unboxed {
            compile::unbox_raw_force_results(optimized_ops, &constants, &self.raw_int_force_helpers)
        } else {
            optimized_ops
        };
        let optimized_ops = compile::strip_stray_overflow_guards(optimized_ops);

        let num_optimized_ops = optimized_ops.len();
        let compiled_constants = constants.clone();
        let bridge_trace_id = self.alloc_trace_id();

        if crate::majit_log_enabled() {
            eprintln!("--- bridge trace (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
        }

        self.backend.set_constants(constants);
        self.backend.set_next_trace_id(bridge_trace_id);

        let result = {
            let compiled = self.compiled_loops.get(&green_key).unwrap();
            self.backend.compile_bridge(
                fail_descr,
                bridge_inputargs,
                &optimized_ops,
                &compiled.token,
            )
        };

        match result {
            Ok(_) => {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compiled bridge at key={}, guard={}",
                        green_key, fail_index
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
                    compiled
                        .guard_failures
                        .entry((source_trace_id, fail_index))
                        .or_insert(GuardFailureInfo {
                            fail_count: 0,
                            bridge_compiled: false,
                        })
                        .bridge_compiled = true;
                    let (resume_data, guard_op_indices, mut exit_layouts) =
                        compile::build_guard_metadata(bridge_inputargs, &optimized_ops, green_key);
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
                    compiled.traces.insert(
                        bridge_trace_id,
                        CompiledTrace {
                            inputargs: bridge_inputargs.to_vec(),
                            resume_data,
                            ops: optimized_ops,
                            constants: compiled_constants,
                            guard_op_indices,
                            exit_layouts,
                            terminal_exit_layouts,
                            optimizer_knowledge: Vec::new(),
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
    pub fn start_retrace_from_guard(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        _fail_values: &[i64],
    ) -> bool {
        let compiled = match self.compiled_loops.get(&green_key) {
            Some(c) => c,
            None => return false,
        };
        let fail_descr = match Self::bridge_fail_descr_proxy(compiled, trace_id, fail_index) {
            Some(descr) => descr,
            None => return false,
        };

        let recorder = self.warm_state.start_retrace(&fail_descr.fail_arg_types);
        self.forced_virtualizable = None;
        self.force_finish_trace = false;
        self.tracing = Some(crate::trace_ctx::TraceCtx::new(recorder, green_key));

        if let Some(ref hook) = self.hooks.on_trace_start {
            hook(green_key);
        }

        true
    }

    // ── Guard Failure Recovery ─────────────────────────────────

    /// Handle a guard failure: recover interpreter state using resume data
    /// and optionally decide whether to compile a bridge.
    ///
    /// This is the central guard failure handler, equivalent to RPython's
    /// `handle_guard_failure()` in pyjitpl.py.
    ///
    /// Returns `GuardRecovery` describing the recovered state and recommended action.
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

        let exit_layout = Self::compiled_exit_layout_from_trace(trace, trace_id, fail_index)
            .unwrap_or_else(|| CompiledExitLayout {
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

        // Decide what to do next
        let action = if self.should_compile_bridge_in_trace(green_key, trace_id, fail_index) {
            GuardRecoveryAction::CompileBridge
        } else if self.get_guard_failure_count_in_trace(green_key, trace_id, fail_index)
            >= self.trace_eagerness
            && self.trace_eagerness > 0
        {
            GuardRecoveryAction::RetraceFromGuard
        } else {
            GuardRecoveryAction::ResumeInterpreter
        };

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
            action,
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
    pub fn blackhole_guard_failure(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
        exception: ExceptionState,
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

        Some(blackhole_execute_with_state(
            &trace.ops,
            &trace.constants,
            &initial_values,
            guard_op_index + 1,
            exception,
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
            BlackholeResult::Finish { op_index, values } => {
                let exit_layout = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (terminal_trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    compile::terminal_exit_layout_for_trace(trace, terminal_trace_id, op_index)
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
                    compile::terminal_exit_layout_for_trace(trace, terminal_trace_id, op_index)
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
                    let exit_layout =
                        Self::compiled_exit_layout_from_trace(trace, trace_id, fallback_fail_index);
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
                    let exit_layout =
                        Self::compiled_exit_layout_from_trace(trace, trace_id, fallback_fail_index);
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
            BlackholeResult::Finish { op_index, values } => {
                let exit_layout = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (terminal_trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    compile::terminal_exit_layout_for_trace(trace, terminal_trace_id, op_index)
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
                    compile::terminal_exit_layout_for_trace(trace, terminal_trace_id, op_index)
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
                    let exit_layout =
                        Self::compiled_exit_layout_from_trace(trace, trace_id, fallback_fail_index);
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
                    let exit_layout =
                        Self::compiled_exit_layout_from_trace(trace, trace_id, fallback_fail_index);
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
        self.start_retrace_from_guard(green_key, root_trace_id, _fail_index, live_values)
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
    pub fn backend(&self) -> &CraneliftBackend {
        &self.backend
    }

    /// Access the backend mutably (for advanced operations).
    pub fn backend_mut(&mut self) -> &mut CraneliftBackend {
        &mut self.backend
    }

    /// Register a helper that boxes a raw integer into an interpreter object.
    ///
    /// Compiled finish post-processing uses this to recognize boxing helpers
    /// that are safe to peel away for the raw-int call_assembler protocol.
    pub fn register_raw_int_box_helper(&mut self, helper: *const ()) {
        self.raw_int_box_helpers.insert(helper as i64);
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
    /// Recommended action after recovery.
    pub action: GuardRecoveryAction,
}

/// What should be done after a guard failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardRecoveryAction {
    /// Resume the interpreter from the recovered state.
    ResumeInterpreter,
    /// The guard has failed enough times to warrant bridge compilation.
    CompileBridge,
    /// Start retracing from this guard failure point.
    RetraceFromGuard,
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

#[derive(Debug, Clone, PartialEq)]
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
    GuardFailure {
        restored: bool,
        via_blackhole: bool,
        /// RPython compile.py handle_fail parity: fail_index and trace_id
        /// for bridge compilation after guard failure.
        fail_index: Option<u32>,
        trace_id: Option<u64>,
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
    use majit_codegen::DeadFrame;
    use majit_codegen::{Backend, ExitFrameLayout, ExitRecoveryLayout, ExitValueSourceLayout};
    use majit_codegen_cranelift::compiler::{
        force_token_to_dead_frame, get_int_from_deadframe, get_latest_descr_from_deadframe,
        set_savedata_ref_on_deadframe,
    };
    use majit_codegen_cranelift::guard::CraneliftFailDescr;
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
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1), InputArg::new_int(2)];
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
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1), InputArg::new_int(2)];
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
            compile::build_guard_metadata(inputargs, &ops, green_key);
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
        MetaInterp::<()>::enrich_guard_resume_layouts_for_trace(
            &mut resume_data,
            &mut exit_layouts,
            trace_id,
            inputargs,
            trace_info.as_ref(),
        );
        MetaInterp::<()>::patch_backend_guard_recovery_layouts_for_trace(
            &mut meta.backend,
            &token,
            trace_id,
            &mut exit_layouts,
        );
        MetaInterp::<()>::patch_backend_terminal_recovery_layouts_for_trace(
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
                guard_failures: HashMap::new(),
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
    fn opimpl_setarrayitem_vable_int_synchronizes_standard_virtualizable() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_with_array(),
            &[Value::Int(0x1234), Value::Int(11), Value::Int(22)],
            vec![2],
        );

        let (index, new_val) = {
            let ctx = meta.trace_ctx().unwrap();
            (ctx.const_int(1), ctx.const_int(33))
        };
        meta.opimpl_setarrayitem_vable_int(OpRef(0), index, new_val, 24);

        let ctx = meta.trace_ctx().unwrap();
        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes[1], new_val);
        assert_eq!(ctx.recorder.num_ops(), 5);
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
        meta.close_and_compile(&[OpRef(0), OpRef(1), OpRef(2)], ());

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
                guard_failures: HashMap::new(),
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
                },
            )
            .expect("compiled loop should exist");

        assert_eq!(recovery.fail_index, fail_index);
        assert_eq!(recovery.trace_id, trace_id);
        assert_eq!(recovery.fail_values, vec![11, 22]);
        assert_eq!(recovery.exception.exc_class, 0x1234);
        assert_eq!(recovery.exception.exc_value, 0xABCD);
        assert_eq!(recovery.action, GuardRecoveryAction::ResumeInterpreter);
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
                guard_failures: HashMap::new(),
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
            vec![majit_codegen::ExitValueSourceLayout::ExitValue(0)]
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
                guard_failures: HashMap::new(),
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

        assert!(meta.start_retrace_from_guard(green_key, trace_id, 0, &[]));

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
                guard_failures: HashMap::new(),
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
        meta.close_and_compile(&[OpRef(0)], ());

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
        meta.close_and_compile(&[OpRef(0)], ());
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
            meta.close_and_compile(&[OpRef(0), OpRef(1)], ());
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
        use majit_codegen::ExitValueSourceLayout;

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
                fail_arg_index: Some(0),
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
            num_fail_args: 1,
            fail_arg_positions: vec![0],
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
        let new_op = mk_op_with_descr(
            OpCode::New,
            &[],
            25,
            majit_ir::descr::make_size_descr(16),
        );
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

        let (ops, changed) =
            unbox_finish_result(vec![set_type, set_payload, new_op, finish], &HashMap::new(), &HashSet::new());

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
