use std::collections::HashMap;

use majit_codegen::{
    Backend, CompiledTraceInfo, ExitRecoveryLayout, FailDescrLayout, LoopToken, TerminalExitLayout,
};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{GcRef, InputArg, OpCode, OpRef, Type, Value};
use majit_opt::optimizer::Optimizer;
use majit_trace::trace::Trace;
use majit_trace::warmstate::{HotResult, WarmState};

use crate::blackhole::{blackhole_execute_with_state, BlackholeResult, ExceptionState};
use crate::io_buffer;
use crate::resume::{
    EncodedResumeData, MaterializedVirtual, ReconstructedState, ResolvedPendingFieldWrite,
    ResumeData, ResumeDataBuilder, ResumeDataLoopMemo, ResumeLayoutSummary,
};
use crate::trace_ctx::{JitDriverDescriptor, TraceCtx};
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

/// Detailed result from running compiled code, including guard failure info.
///
/// Mirrors RPython's handle_guard_failure in pyjitpl.py.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledExitLayout {
    /// Compiled trace identifier for this exit (root loop or bridge).
    pub trace_id: u64,
    /// Backend fail-index for this exit.
    pub fail_index: u32,
    /// Trace op index that produced this exit, when known by the backend.
    pub source_op_index: Option<usize>,
    /// Typed layout of the raw exit slots produced by the backend.
    pub exit_types: Vec<Type>,
    /// Whether this exit is a FINISH rather than a guard failure.
    pub is_finish: bool,
    /// Exit slot indices that hold rooted GC references.
    pub gc_ref_slots: Vec<usize>,
    /// Exit slot indices that carry opaque FORCE_TOKEN handles.
    pub force_token_slots: Vec<usize>,
    /// Raw backend-origin recovery layout for this exit, when available.
    pub recovery_layout: Option<ExitRecoveryLayout>,
    /// Compact resume/jitframe layout attached to this exit, when available.
    pub resume_layout: Option<ResumeLayoutSummary>,
}

pub struct CompileResult<'a, M> {
    /// The live values at the point of guard failure (or loop finish).
    pub values: Vec<i64>,
    /// Typed exit values decoded from the backend deadframe.
    pub typed_values: Vec<Value>,
    /// The interpreter-specific metadata for this loop.
    pub meta: &'a M,
    /// Backend fail-index for this exit.
    pub fail_index: u32,
    /// Compiled trace identifier for this exit (root loop or bridge).
    pub trace_id: u64,
    /// Whether this exit is a FINISH rather than a guard failure.
    pub is_finish: bool,
    /// Static layout metadata for this compiled exit.
    pub exit_layout: CompiledExitLayout,
    /// Optional saved-data GC ref captured from the backend exit.
    pub savedata: Option<GcRef>,
    /// Pending exception state captured from the backend deadframe.
    pub exception: ExceptionState,
}

pub struct RawCompileResult<'a, M> {
    /// The live values at the point of guard failure (or loop finish).
    pub values: Vec<i64>,
    /// Typed exit values decoded from the backend exit.
    pub typed_values: Vec<Value>,
    /// The interpreter-specific metadata for this loop.
    pub meta: &'a M,
    /// Backend fail-index for this exit.
    pub fail_index: u32,
    /// Compiled trace identifier for this exit (root loop or bridge).
    pub trace_id: u64,
    /// Whether this exit is a FINISH rather than a guard failure.
    pub is_finish: bool,
    /// Static layout metadata for this compiled exit.
    pub exit_layout: CompiledExitLayout,
    /// Optional saved-data GC ref captured from the backend exit.
    pub savedata: Option<GcRef>,
    /// Pending exception state captured from the backend exit.
    pub exception: ExceptionState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledTerminalExitLayout {
    pub op_index: usize,
    pub exit_layout: CompiledExitLayout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledTraceLayout {
    pub trace_id: u64,
    pub exit_layouts: Vec<CompiledExitLayout>,
    pub terminal_exit_layouts: Vec<CompiledTerminalExitLayout>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeadFrameArtifacts {
    pub values: Vec<i64>,
    pub typed_values: Vec<Value>,
    pub exit_layout: CompiledExitLayout,
    pub savedata: Option<GcRef>,
    pub exception: ExceptionState,
}

/// Per-guard failure tracking for bridge compilation decisions.
struct GuardFailureInfo {
    /// Number of times this guard has failed.
    fail_count: u32,
    /// Whether a bridge has been compiled for this guard.
    bridge_compiled: bool,
}

struct CompiledTrace {
    /// Inputargs for this trace, used to recover typed exit layouts during blackhole replay.
    inputargs: Vec<InputArg>,
    /// Resume data for each guard, keyed by fail_index.
    resume_data: HashMap<u32, StoredResumeData>,
    /// Optimized ops for blackhole fallback from compiled guard failures.
    ops: Vec<majit_ir::Op>,
    /// Constant pool paired with `ops` for blackhole fallback.
    constants: HashMap<u32, i64>,
    /// Mapping from backend fail_index to the corresponding guard op index.
    guard_op_indices: HashMap<u32, usize>,
    /// Static exit metadata for each guard/finish in this trace.
    exit_layouts: HashMap<u32, StoredExitLayout>,
    /// Static exit metadata for terminal FINISH/JUMP ops, keyed by op index.
    terminal_exit_layouts: HashMap<usize, StoredExitLayout>,
}

struct StoredResumeData {
    semantic: ResumeData,
    encoded: EncodedResumeData,
    layout: ResumeLayoutSummary,
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

    fn with_loop_memo(semantic: ResumeData, memo: &mut ResumeDataLoopMemo) -> Self {
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
struct StoredExitLayout {
    source_op_index: Option<usize>,
    exit_types: Vec<Type>,
    is_finish: bool,
    gc_ref_slots: Vec<usize>,
    force_token_slots: Vec<usize>,
    recovery_layout: Option<ExitRecoveryLayout>,
    resume_layout: Option<ResumeLayoutSummary>,
}

impl StoredExitLayout {
    fn public(&self, trace_id: u64, fail_index: u32) -> CompiledExitLayout {
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

struct CompiledEntry<M> {
    token: LoopToken,
    num_inputs: usize,
    meta: M,
    /// Trace id of the root compiled loop.
    root_trace_id: u64,
    /// Per-guard failure tracking, keyed by (trace_id, fail_index).
    guard_failures: HashMap<(u64, u32), GuardFailureInfo>,
    /// Metadata for the root loop and any attached bridges, keyed by trace id.
    traces: HashMap<u64, CompiledTrace>,
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
    warm_state: WarmState,
    backend: CraneliftBackend,
    compiled_loops: HashMap<u64, CompiledEntry<M>>,
    tracing: Option<TraceCtx>,
    next_trace_id: u64,
    /// Trace eagerness: start tracing from a guard failure point
    /// after this many failures (0 = never trace from guards).
    trace_eagerness: u32,
    /// Virtualizable info for interpreter frame virtualization.
    ///
    /// When set, the JIT will:
    /// 1. Read virtualizable fields at trace entry (synchronize)
    /// 2. Write them back on guard failure (force)
    /// 3. Track field access as IR ops during tracing
    virtualizable_info: Option<VirtualizableInfo>,
    /// JIT hooks for profiling and debugging.
    hooks: JitHooks,
    /// Pre-allocated token number for the trace currently being recorded.
    /// Set when tracing starts so that self-recursive calls can emit
    /// call_assembler targeting this token before the trace is compiled.
    pending_token: Option<(u64, u64)>,
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
                find_fail_index_for_exit_op(&trace.ops, op_index).unwrap_or(u32::MAX),
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
                            find_fail_index_for_exit_op(&trace.ops, op_index).unwrap_or(u32::MAX),
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

    fn enrich_guard_resume_layouts_for_trace(
        resume_data: &mut HashMap<u32, StoredResumeData>,
        exit_layouts: &mut HashMap<u32, StoredExitLayout>,
        trace_id: u64,
        inputargs: &[InputArg],
        trace_info: Option<&CompiledTraceInfo>,
    ) {
        for (fail_index, stored) in resume_data.iter_mut() {
            let recovery_layout = exit_layouts
                .get(fail_index)
                .and_then(|layout| layout.recovery_layout.clone());
            enrich_resume_layout_with_trace_metadata(
                &mut stored.layout,
                trace_id,
                inputargs,
                trace_info,
                recovery_layout.as_ref(),
            );
            if let Some(exit_layout) = exit_layouts.get_mut(fail_index) {
                exit_layout.resume_layout = Some(stored.layout.clone());
            }
        }
    }

    fn patch_backend_guard_recovery_layouts_for_trace(
        backend: &mut CraneliftBackend,
        token: &LoopToken,
        trace_id: u64,
        exit_layouts: &mut HashMap<u32, StoredExitLayout>,
    ) {
        for (&fail_index, exit_layout) in exit_layouts.iter_mut() {
            let Some(resume_layout) = exit_layout.resume_layout.as_ref() else {
                continue;
            };
            let recovery_layout = resume_layout
                .to_exit_recovery_layout_with_caller_prefix(exit_layout.recovery_layout.as_ref());
            if backend.update_fail_descr_recovery_layout(
                token,
                trace_id,
                fail_index,
                recovery_layout.clone(),
            ) {
                exit_layout.recovery_layout = Some(recovery_layout);
            }
        }
    }

    fn patch_backend_terminal_recovery_layouts_for_trace(
        backend: &mut CraneliftBackend,
        token: &LoopToken,
        trace_id: u64,
        terminal_exit_layouts: &mut HashMap<usize, StoredExitLayout>,
    ) {
        for (&op_index, exit_layout) in terminal_exit_layouts.iter_mut() {
            let Some(resume_layout) = exit_layout.resume_layout.as_ref() else {
                continue;
            };
            let recovery_layout = resume_layout
                .to_exit_recovery_layout_with_caller_prefix(exit_layout.recovery_layout.as_ref());
            if backend.update_terminal_exit_recovery_layout(
                token,
                trace_id,
                op_index,
                recovery_layout.clone(),
            ) {
                exit_layout.recovery_layout = Some(recovery_layout);
            }
        }
    }

    /// Create a new MetaInterp with the given compilation threshold.
    pub fn new(threshold: u32) -> Self {
        MetaInterp {
            warm_state: WarmState::new(threshold),
            backend: CraneliftBackend::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
            next_trace_id: 1,
            trace_eagerness: 200,
            virtualizable_info: None,
            hooks: JitHooks::default(),
            pending_token: None,
        }
    }

    /// Set the trace eagerness (guard failure threshold for bridge tracing).
    pub fn set_trace_eagerness(&mut self, eagerness: u32) {
        self.trace_eagerness = eagerness;
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
        driver_descriptor: Option<JitDriverDescriptor>,
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
                self.tracing = Some(ctx);
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

    pub fn on_back_edge_typed(
        &mut self,
        green_key: u64,
        green_key_values: Option<majit_ir::GreenKey>,
        driver_descriptor: Option<JitDriverDescriptor>,
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

    /// Whether the engine is currently tracing.
    #[inline]
    pub fn is_tracing(&self) -> bool {
        self.tracing.is_some()
    }

    /// Finish the current active trace without optimizing or compiling it.
    ///
    /// This is a semantic-seam helper for parity tests: it lets callers
    /// inspect the raw recorded trace that the proc-macro/runtime path
    /// produced, without requiring backend compilation.
    pub fn finish_trace_for_parity(
        &mut self,
        finish_args: &[OpRef],
    ) -> Option<(Trace, HashMap<u32, i64>)> {
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
        let ctx = self.tracing.take().unwrap();
        let green_key = ctx.green_key;

        let mut recorder = ctx.recorder;
        recorder.close_loop(jump_args);
        let trace = recorder.get_trace();

        let mut optimizer = Optimizer::default_pipeline();
        let mut constants = ctx.constants.into_inner();

        if crate::majit_log_enabled() {
            eprintln!("--- trace (before opt) ---");
            eprint!("{}", majit_ir::format_trace(&trace.ops, &constants));
        }

        let num_ops_before = trace.ops.len();
        let optimized_ops = optimizer.optimize_with_constants_and_inputs(
            &trace.ops,
            &mut constants,
            trace.inputargs.len(),
        );
        let num_ops_after = optimized_ops.len();

        if crate::majit_log_enabled() {
            eprintln!("--- trace (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
        }

        let compiled_constants = constants.clone();
        self.backend.set_constants(constants);

        // Use pre-allocated token number if available (for self-recursion
        // support), otherwise allocate a fresh one.
        let token_num = if let Some((pk, pn)) = self.pending_token.take() {
            if pk == green_key { pn } else { self.warm_state.alloc_token_number() }
        } else {
            self.warm_state.alloc_token_number()
        };
        let mut token = LoopToken::new(token_num);
        let trace_id = self.alloc_trace_id();
        self.backend.set_next_trace_id(trace_id);

        match self
            .backend
            .compile_loop(&trace.inputargs, &optimized_ops, &mut token)
        {
            Ok(_) => {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compiled loop at key={}, num_inputs={}",
                        green_key,
                        trace.inputargs.len()
                    );
                }
                // Build resume data and exit layouts for all guards in the optimized trace.
                let (resume_data, guard_op_indices, mut exit_layouts) =
                    build_guard_metadata(&trace.inputargs, &optimized_ops, green_key);
                let mut terminal_exit_layouts =
                    build_terminal_exit_layouts(&trace.inputargs, &optimized_ops);
                if let Some(backend_layouts) = self.backend.compiled_fail_descr_layouts(&token) {
                    merge_backend_exit_layouts(&mut exit_layouts, &backend_layouts);
                }
                if let Some(backend_layouts) = self.backend.compiled_terminal_exit_layouts(&token) {
                    merge_backend_terminal_exit_layouts(
                        &mut terminal_exit_layouts,
                        &backend_layouts,
                    );
                }
                let trace_info = self.backend.compiled_trace_info(&token, trace_id);
                let mut resume_data = resume_data;
                Self::enrich_guard_resume_layouts_for_trace(
                    &mut resume_data,
                    &mut exit_layouts,
                    trace_id,
                    &trace.inputargs,
                    trace_info.as_ref(),
                );
                Self::patch_backend_guard_recovery_layouts_for_trace(
                    &mut self.backend,
                    &token,
                    trace_id,
                    &mut exit_layouts,
                );
                Self::patch_backend_terminal_recovery_layouts_for_trace(
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
                    },
                );

                self.compiled_loops.insert(
                    green_key,
                    CompiledEntry {
                        token,
                        num_inputs: trace.inputargs.len(),
                        meta,
                        root_trace_id: trace_id,
                        guard_failures: HashMap::new(),
                        traces,
                    },
                );
                let install_num = self.warm_state.alloc_token_number();
                let install_token = LoopToken::new(install_num);
                self.warm_state.install_compiled(green_key, install_token);

                if let Some(ref hook) = self.hooks.on_compile_loop {
                    hook(green_key, num_ops_before, num_ops_after);
                }
            }
            Err(e) => {
                let msg = format!("JIT compilation failed: {e}");
                if crate::majit_log_enabled() {
                    eprintln!("[jit] {msg}");
                }
                if let Some(ref cb) = self.hooks.on_compile_error {
                    cb(green_key, &msg);
                }
                self.warm_state.abort_tracing(green_key, true);
            }
        }
        // Reset per-trace function call counts.
        self.warm_state.reset_function_counts();
    }

    /// Abort the current trace.
    ///
    /// If `permanent` is true, this location will never be traced again.
    pub fn abort_trace(&mut self, permanent: bool) {
        if let Some(ctx) = self.tracing.take() {
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
        let ctx = self.tracing.take().unwrap();
        let green_key = ctx.green_key;

        let mut recorder = ctx.recorder;
        recorder.finish(
            finish_args,
            crate::make_fail_descr_typed(finish_arg_types),
        );
        let trace = recorder.get_trace();

        let mut optimizer = Optimizer::default_pipeline();
        let mut constants = ctx.constants.into_inner();

        let num_ops_before = trace.ops.len();
        let optimized_ops = optimizer.optimize_with_constants_and_inputs(
            &trace.ops,
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
            eprint!("{}", majit_ir::format_trace(&trace.ops, &constants));
            eprintln!("--- finish trace (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
        }

        let compiled_constants = constants.clone();
        self.backend.set_constants(constants);

        // Use pre-allocated token number if available (for self-recursion
        // support), otherwise allocate a fresh one.
        let token_num = if let Some((pk, pn)) = self.pending_token.take() {
            if pk == green_key { pn } else { self.warm_state.alloc_token_number() }
        } else {
            self.warm_state.alloc_token_number()
        };
        let mut token = LoopToken::new(token_num);
        let trace_id = self.alloc_trace_id();
        self.backend.set_next_trace_id(trace_id);

        match self
            .backend
            .compile_loop(&trace.inputargs, &optimized_ops, &mut token)
        {
            Ok(_) => {
                let (resume_data, guard_op_indices, mut exit_layouts) =
                    build_guard_metadata(&trace.inputargs, &optimized_ops, green_key);
                let mut terminal_exit_layouts =
                    build_terminal_exit_layouts(&trace.inputargs, &optimized_ops);
                if let Some(backend_layouts) = self.backend.compiled_fail_descr_layouts(&token) {
                    merge_backend_exit_layouts(&mut exit_layouts, &backend_layouts);
                }
                if let Some(backend_layouts) = self.backend.compiled_terminal_exit_layouts(&token) {
                    merge_backend_terminal_exit_layouts(
                        &mut terminal_exit_layouts,
                        &backend_layouts,
                    );
                }
                let trace_info = self.backend.compiled_trace_info(&token, trace_id);
                let mut resume_data = resume_data;
                Self::enrich_guard_resume_layouts_for_trace(
                    &mut resume_data,
                    &mut exit_layouts,
                    trace_id,
                    &trace.inputargs,
                    trace_info.as_ref(),
                );
                Self::patch_backend_guard_recovery_layouts_for_trace(
                    &mut self.backend,
                    &token,
                    trace_id,
                    &mut exit_layouts,
                );
                Self::patch_backend_terminal_recovery_layouts_for_trace(
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
                    },
                );
                self.compiled_loops.insert(
                    green_key,
                    CompiledEntry {
                        token,
                        num_inputs: trace.inputargs.len(),
                        meta,
                        root_trace_id: trace_id,
                        guard_failures: HashMap::new(),
                        traces,
                    },
                );
                let install_num = self.warm_state.alloc_token_number();
                let install_token = LoopToken::new(install_num);
                self.warm_state.install_compiled(green_key, install_token);
                self.warm_state.reset_function_counts();
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

            self.warm_state.log_guard_failure(fail_index);

            if let Some(ref hook) = self.hooks.on_guard_failure {
                hook(green_key, fail_index, info.fail_count);
            }
        }

        let compiled = self.compiled_loops.get(&green_key).unwrap();
        let trace_layout =
            Self::trace_for_exit(compiled, trace_id).and_then(|(trace_id, trace)| {
                Self::compiled_exit_layout_from_trace(trace, trace_id, fail_index)
            });
        let exit_layout = result
            .exit_layout
            .map(|layout| CompiledExitLayout {
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
                resume_layout: trace_layout
                    .as_ref()
                    .and_then(|layout| layout.resume_layout.clone()),
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
        let exception = ExceptionState {
            exc_class: result.exception_class,
            exc_value: result.exception_value.0 as i64,
        };

        Some(RawCompileResult {
            values: result.outputs,
            typed_values: result.typed_outputs,
            meta: &compiled.meta,
            fail_index,
            trace_id,
            is_finish: result.is_finish,
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

            self.warm_state.log_guard_failure(fail_index);

            if let Some(ref hook) = self.hooks.on_guard_failure {
                hook(green_key, fail_index, info.fail_count);
            }
        }

        let compiled = self.compiled_loops.get(&green_key).unwrap();
        let trace_layout =
            Self::trace_for_exit(compiled, trace_id).and_then(|(trace_id, trace)| {
                Self::compiled_exit_layout_from_trace(trace, trace_id, fail_index)
            });
        let exit_layout = result
            .exit_layout
            .map(|layout| CompiledExitLayout {
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
                resume_layout: trace_layout
                    .as_ref()
                    .and_then(|layout| layout.resume_layout.clone()),
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
        let exception = ExceptionState {
            exc_class: result.exception_class,
            exc_value: result.exception_value.0 as i64,
        };

        Some(RawCompileResult {
            values: result.outputs,
            typed_values: result.typed_outputs,
            meta: &compiled.meta,
            fail_index,
            trace_id,
            is_finish: result.is_finish,
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

        let descr = self.backend.get_latest_descr(&frame);
        let fail_index = descr.fail_index();
        let trace_id = Self::normalize_trace_id(compiled, descr.trace_id());
        let is_finish = descr.is_finish();
        Self::finish_compiled_run_io(is_finish);

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
                enrich_resume_layout_with_trace_metadata(
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

    /// Get the LoopToken for a compiled loop (for CALL_ASSEMBLER).
    ///
    /// In RPython, `call_assembler` allows JIT code for one function
    /// to directly call JIT code for another function. The caller needs
    /// the target's LoopToken to set up the call.
    pub fn get_loop_token(&self, green_key: u64) -> Option<&LoopToken> {
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
            self.backend.redirect_call_assembler(old, new);
        }
    }

    /// Check whether a compiled loop exists for a given green key.
    #[inline]
    pub fn has_compiled_loop(&self, green_key: u64) -> bool {
        self.compiled_loops.contains_key(&green_key)
    }

    /// Check whether a guard in a specific compiled trace should get a bridge.
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

    // ── Bridge Compilation ──────────────────────────────────────
}

/// Proxy FailDescr used when compiling bridges from guard failure points.
/// Carries enough information for the backend to locate the original guard.
#[derive(Debug)]
struct BridgeFailDescrProxy {
    fail_index: u32,
    trace_id: u64,
    fail_arg_types: Vec<Type>,
}

impl majit_ir::Descr for BridgeFailDescrProxy {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn majit_ir::FailDescr> {
        Some(self)
    }
}

impl majit_ir::FailDescr for BridgeFailDescrProxy {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }
    fn trace_id(&self) -> u64 {
        self.trace_id
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
        let trace_id = Self::normalize_trace_id(compiled, trace_id);
        // Verify the guard exists
        let _trace = compiled.traces.get(&trace_id)?;
        Some(Box::new(BridgeFailDescrProxy {
            fail_index,
            trace_id,
            fail_arg_types: vec![Type::Int], // call_assembler guard fail_args = [frame_ptr]
        }))
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

        // Optimize the bridge trace
        let mut optimizer = Optimizer::default_pipeline();
        let mut constants = constants;
        let optimized_ops = optimizer.optimize_with_constants_and_inputs(
            bridge_ops,
            &mut constants,
            bridge_inputargs.len(),
        );
        let num_optimized_ops = optimized_ops.len();
        let compiled_constants = constants.clone();
        let bridge_trace_id = self.alloc_trace_id();

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
                        build_guard_metadata(bridge_inputargs, &optimized_ops, green_key);
                    let mut terminal_exit_layouts =
                        build_terminal_exit_layouts(bridge_inputargs, &optimized_ops);
                    if let Some(backend_layouts) = self.backend.compiled_bridge_fail_descr_layouts(
                        &compiled.token,
                        source_trace_id,
                        fail_index,
                    ) {
                        merge_backend_exit_layouts(&mut exit_layouts, &backend_layouts);
                    }
                    if let Some(backend_layouts) =
                        self.backend.compiled_bridge_terminal_exit_layouts(
                            &compiled.token,
                            source_trace_id,
                            fail_index,
                        )
                    {
                        merge_backend_terminal_exit_layouts(
                            &mut terminal_exit_layouts,
                            &backend_layouts,
                        );
                    }
                    let bridge_trace_info = self
                        .backend
                        .compiled_trace_info(&compiled.token, bridge_trace_id);
                    let mut resume_data = resume_data;
                    Self::enrich_guard_resume_layouts_for_trace(
                        &mut resume_data,
                        &mut exit_layouts,
                        bridge_trace_id,
                        bridge_inputargs,
                        bridge_trace_info.as_ref(),
                    );
                    Self::patch_backend_guard_recovery_layouts_for_trace(
                        &mut self.backend,
                        &compiled.token,
                        bridge_trace_id,
                        &mut exit_layouts,
                    );
                    Self::patch_backend_terminal_recovery_layouts_for_trace(
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
                        },
                    );
                }
                self.warm_state.log_bridge_compile(fail_index);

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

        let trace_id = Self::normalize_trace_id(compiled, trace_id);

        // Find the guard's fail_arg_types to set up inputs
        let (_, trace) = match Self::trace_for_exit(compiled, trace_id) {
            Some(t) => t,
            None => return false,
        };

        let guard_op_index = match trace.guard_op_indices.get(&fail_index) {
            Some(&idx) => idx,
            None => return false,
        };

        let guard_op = match trace.ops.get(guard_op_index) {
            Some(op) => op,
            None => return false,
        };

        let num_inputs = guard_op.fail_args.as_ref().map(|fa| fa.len()).unwrap_or(0);

        // Create a new trace recorder for the retrace
        let recorder = self.warm_state.start_retrace(num_inputs);
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

    fn blackhole_guard_failure(
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
                    terminal_exit_layout_for_trace(trace, terminal_trace_id, op_index)
                };
                let typed_values = exit_layout
                    .as_ref()
                    .map(|layout| decode_values_with_layout(&values, layout));
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
                    terminal_exit_layout_for_trace(trace, terminal_trace_id, op_index)
                };
                let typed_values = exit_layout
                    .as_ref()
                    .map(|layout| decode_values_with_layout(&values, layout));
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
                        .map(|layout| decode_values_with_layout(&fail_values, layout));
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
                        .map(|layout| decode_values_with_layout(&fail_values, layout));
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
                    terminal_exit_layout_for_trace(trace, terminal_trace_id, op_index)
                };
                let typed_values = exit_layout
                    .as_ref()
                    .map(|layout| decode_values_with_layout(&values, layout));
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
                    terminal_exit_layout_for_trace(trace, terminal_trace_id, op_index)
                };
                let typed_values = exit_layout
                    .as_ref()
                    .map(|layout| decode_values_with_layout(&values, layout));
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
                        .map(|layout| decode_values_with_layout(&fail_values, layout));
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
                        .map(|layout| decode_values_with_layout(&fail_values, layout));
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
        if self.tracing.is_some() {
            return false; // already tracing
        }

        if !self.compiled_loops.contains_key(&green_key) {
            return false;
        }

        // Create a new trace recorder for the bridge
        let mut recorder = majit_trace::recorder::TraceRecorder::new();
        for _ in live_values {
            recorder.record_input_arg(Type::Int);
        }

        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] start retrace at key={}, num_inputs={}",
                green_key,
                live_values.len()
            );
        }

        self.tracing = Some(TraceCtx::new(recorder, green_key));
        true
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
        // If the callee already has compiled code, don't inline —
        // use CALL_ASSEMBLER instead.
        if self.compiled_loops.contains_key(&callee_key) {
            return InlineDecision::CallAssembler;
        }

        // If we're already tracing, we can potentially inline.
        if self.tracing.is_some() {
            // Check recursion depth (max inlining depth = 10).
            if let Some(ref ctx) = self.tracing {
                if ctx.inline_depth() >= MAX_INLINE_DEPTH {
                    return InlineDecision::ResidualCall;
                }
            }

            // Check function_threshold: has the function been called enough
            // times to warrant inlining? This mirrors RPython's
            // function_threshold heuristic in warmstate.py.
            if !self.warm_state.should_inline_function(callee_key) {
                return InlineDecision::ResidualCall;
            }

            return InlineDecision::Inline;
        }

        // Otherwise, emit a regular residual call.
        InlineDecision::ResidualCall
    }

    /// Begin inlining a function call during tracing.
    ///
    /// Records ENTER_PORTAL_FRAME marker and pushes an inline frame.
    /// The caller should then trace the callee's body normally.
    /// Call `leave_inline_frame()` when the callee returns.
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

        let key_ref = ctx.const_int(callee_key as i64);
        ctx.record_op(OpCode::EnterPortalFrame, &[key_ref]);
        ctx.push_inline_frame(callee_key, MAX_INLINE_DEPTH as u32);
        true
    }

    /// Leave an inlined function call during tracing.
    ///
    /// Records LEAVE_PORTAL_FRAME marker and pops the inline frame.
    pub fn leave_inline_frame(&mut self) {
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.record_op(OpCode::LeavePortalFrame, &[]);
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
}

/// Default maximum inlining depth during tracing.
/// Configurable via WarmState::set_max_inline_depth().
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
    },
    Jump {
        via_blackhole: bool,
    },
    GuardFailure {
        restored: bool,
        via_blackhole: bool,
    },
    Abort {
        restored: bool,
        via_blackhole: bool,
    },
}

/// Build guard metadata for a compiled trace.
///
/// The backend numbers every guard and finish in a single exit table, so this
/// helper mirrors that numbering and records only the guard entries that need
/// resume data plus the corresponding op index for blackhole fallback.
fn build_guard_metadata(
    inputargs: &[InputArg],
    ops: &[majit_ir::Op],
    pc: u64,
) -> (
    HashMap<u32, StoredResumeData>,
    HashMap<u32, usize>,
    HashMap<u32, StoredExitLayout>,
) {
    let mut result = HashMap::new();
    let mut guard_op_indices = HashMap::new();
    let mut exit_layouts = HashMap::new();
    let mut fail_index = 0u32;
    let mut resume_memo = ResumeDataLoopMemo::new();
    let mut value_types: HashMap<u32, Type> =
        inputargs.iter().map(|arg| (arg.index, arg.tp)).collect();

    for (op_idx, op) in ops.iter().enumerate() {
        if !op.pos.is_none() && op.result_type() != Type::Void {
            value_types.insert(op.pos.0, op.result_type());
        }

        let is_guard = op.opcode.is_guard();
        let is_finish = op.opcode == OpCode::Finish;
        if !is_guard && !is_finish {
            continue;
        }

        if is_guard {
            guard_op_indices.insert(fail_index, op_idx);
        }

        let exit_types: Vec<Type> = if is_finish {
            op.args
                .iter()
                .map(|opref| value_types.get(&opref.0).copied().unwrap_or(Type::Int))
                .collect()
        } else if let Some(ref fail_args) = op.fail_args {
            fail_args
                .iter()
                .map(|opref| value_types.get(&opref.0).copied().unwrap_or(Type::Int))
                .collect()
        } else {
            inputargs.iter().map(|arg| arg.tp).collect()
        };

        let mut resume_layout = None;
        if is_guard {
            if let Some(ref fail_args) = op.fail_args {
                let mut builder = ResumeDataBuilder::new();
                builder.push_frame(pc);

                for (slot_idx, _) in fail_args.iter().enumerate() {
                    builder.map_slot(slot_idx, slot_idx);
                }

                let stored = StoredResumeData::with_loop_memo(builder.build(), &mut resume_memo);
                resume_layout = Some(stored.layout.clone());
                result.insert(fail_index, stored);
            }
        }

        exit_layouts.insert(
            fail_index,
            StoredExitLayout {
                source_op_index: Some(op_idx),
                gc_ref_slots: exit_types
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, tp)| (*tp == Type::Ref).then_some(slot))
                    .collect(),
                force_token_slots: Vec::new(),
                exit_types,
                is_finish,
                recovery_layout: None,
                resume_layout,
            },
        );
        fail_index += 1;
    }

    (result, guard_op_indices, exit_layouts)
}

fn merge_backend_exit_layouts(
    exit_layouts: &mut HashMap<u32, StoredExitLayout>,
    backend_layouts: &[FailDescrLayout],
) {
    for layout in backend_layouts {
        let entry = exit_layouts
            .entry(layout.fail_index)
            .or_insert_with(|| StoredExitLayout {
                source_op_index: layout.source_op_index,
                exit_types: layout.fail_arg_types.clone(),
                is_finish: layout.is_finish,
                gc_ref_slots: layout.gc_ref_slots.clone(),
                force_token_slots: layout.force_token_slots.clone(),
                recovery_layout: layout.recovery_layout.clone(),
                resume_layout: None,
            });
        entry.source_op_index = layout.source_op_index;
        entry.exit_types = layout.fail_arg_types.clone();
        entry.is_finish = layout.is_finish;
        entry.gc_ref_slots = layout.gc_ref_slots.clone();
        entry.force_token_slots = layout.force_token_slots.clone();
        entry.recovery_layout = layout.recovery_layout.clone();
    }
}

fn merge_backend_terminal_exit_layouts(
    terminal_exit_layouts: &mut HashMap<usize, StoredExitLayout>,
    backend_layouts: &[TerminalExitLayout],
) {
    for layout in backend_layouts {
        let entry = terminal_exit_layouts
            .entry(layout.op_index)
            .or_insert_with(|| StoredExitLayout {
                source_op_index: Some(layout.op_index),
                exit_types: layout.exit_types.clone(),
                is_finish: layout.is_finish,
                gc_ref_slots: layout.gc_ref_slots.clone(),
                force_token_slots: layout.force_token_slots.clone(),
                recovery_layout: layout.recovery_layout.clone(),
                resume_layout: None,
            });
        entry.source_op_index = Some(layout.op_index);
        entry.exit_types = layout.exit_types.clone();
        entry.is_finish = layout.is_finish;
        entry.gc_ref_slots = layout.gc_ref_slots.clone();
        entry.force_token_slots = layout.force_token_slots.clone();
        entry.recovery_layout = layout.recovery_layout.clone();
    }
}

fn enrich_resume_layout_with_trace_metadata(
    layout: &mut ResumeLayoutSummary,
    trace_id: u64,
    inputargs: &[InputArg],
    trace_info: Option<&CompiledTraceInfo>,
    recovery_layout: Option<&ExitRecoveryLayout>,
) {
    if layout.frame_layouts.is_empty() {
        return;
    }

    if let Some(recovery_layout) = recovery_layout {
        let shared_frames = layout.frame_layouts.len().min(recovery_layout.frames.len());
        for offset in 0..shared_frames {
            let layout_index = layout.frame_layouts.len() - 1 - offset;
            let recovery_index = recovery_layout.frames.len() - 1 - offset;
            let recovery_frame = &recovery_layout.frames[recovery_index];
            let frame = &mut layout.frame_layouts[layout_index];
            if frame.trace_id.is_none() {
                frame.trace_id = recovery_frame.trace_id;
            }
            if frame.header_pc.is_none() {
                frame.header_pc = recovery_frame.header_pc;
            }
            if frame.source_guard.is_none() {
                frame.source_guard = recovery_frame.source_guard;
            }
            let needs_slot_types = match frame.slot_types.as_ref() {
                Some(slot_types) => slot_types.len() != frame.slot_layouts.len(),
                None => true,
            };
            if needs_slot_types
                && recovery_frame
                    .slot_types
                    .as_ref()
                    .is_some_and(|slot_types| slot_types.len() == frame.slot_layouts.len())
            {
                frame.slot_types = recovery_frame.slot_types.clone();
            }
        }
    }

    let last_index = layout.frame_layouts.len() - 1;
    let innermost = &mut layout.frame_layouts[last_index];
    if innermost.trace_id.is_none() {
        innermost.trace_id = Some(trace_id);
    }
    if innermost.header_pc.is_none() {
        innermost.header_pc = trace_info.map(|info| info.header_pc);
    }
    if innermost.source_guard.is_none() {
        innermost.source_guard = trace_info.and_then(|info| info.source_guard);
    }
    let needs_slot_types = match innermost.slot_types.as_ref() {
        Some(slot_types) => slot_types.len() != innermost.slot_layouts.len(),
        None => true,
    };
    if needs_slot_types && inputargs.len() == innermost.slot_layouts.len() {
        innermost.slot_types = Some(inputargs.iter().map(|arg| arg.tp).collect());
    }
}

fn build_trace_value_maps(
    inputargs: &[InputArg],
    ops: &[majit_ir::Op],
) -> (HashMap<u32, Type>, HashMap<u32, OpCode>) {
    let mut value_types: HashMap<u32, Type> =
        inputargs.iter().map(|arg| (arg.index, arg.tp)).collect();
    let mut producers = HashMap::new();
    for op in ops {
        if !op.pos.is_none() && op.result_type() != Type::Void {
            value_types.insert(op.pos.0, op.result_type());
            producers.insert(op.pos.0, op.opcode);
        }
    }
    (value_types, producers)
}

fn find_fail_index_for_exit_op(ops: &[majit_ir::Op], op_index: usize) -> Option<u32> {
    let mut fail_index = 0u32;
    for (idx, op) in ops.iter().enumerate() {
        if op.opcode.is_guard() || op.opcode == OpCode::Finish {
            if idx == op_index {
                return Some(fail_index);
            }
            fail_index += 1;
        }
    }
    None
}

fn infer_terminal_exit_layout(
    inputargs: &[InputArg],
    ops: &[majit_ir::Op],
    trace_id: u64,
    op_index: usize,
) -> Option<CompiledExitLayout> {
    let op = ops.get(op_index)?;
    let is_finish = op.opcode == OpCode::Finish;
    if !is_finish && op.opcode != OpCode::Jump {
        return None;
    }
    let fail_index = find_fail_index_for_exit_op(ops, op_index).unwrap_or(u32::MAX);
    let (value_types, producers) = build_trace_value_maps(inputargs, ops);
    let exit_types: Vec<Type> = op
        .args
        .iter()
        .map(|opref| value_types.get(&opref.0).copied().unwrap_or(Type::Int))
        .collect();
    let force_token_slots: Vec<usize> = op
        .args
        .iter()
        .enumerate()
        .filter_map(|(slot, opref)| {
            producers
                .get(&opref.0)
                .copied()
                .filter(|opcode| *opcode == OpCode::ForceToken)
                .map(|_| slot)
        })
        .collect();
    let gc_ref_slots: Vec<usize> = exit_types
        .iter()
        .enumerate()
        .filter_map(|(slot, tp)| {
            (*tp == Type::Ref && !force_token_slots.contains(&slot)).then_some(slot)
        })
        .collect();
    Some(CompiledExitLayout {
        trace_id,
        fail_index,
        source_op_index: Some(op_index),
        exit_types,
        is_finish,
        gc_ref_slots,
        force_token_slots,
        recovery_layout: None,
        resume_layout: None,
    })
}

fn build_terminal_exit_layouts(
    inputargs: &[InputArg],
    ops: &[majit_ir::Op],
) -> HashMap<usize, StoredExitLayout> {
    let mut layouts = HashMap::new();
    for (op_index, op) in ops.iter().enumerate() {
        if op.opcode != OpCode::Finish && op.opcode != OpCode::Jump {
            continue;
        }
        if let Some(layout) = infer_terminal_exit_layout(inputargs, ops, 0, op_index) {
            layouts.insert(
                op_index,
                StoredExitLayout {
                    source_op_index: Some(op_index),
                    exit_types: layout.exit_types,
                    is_finish: layout.is_finish,
                    gc_ref_slots: layout.gc_ref_slots,
                    force_token_slots: layout.force_token_slots,
                    recovery_layout: None,
                    resume_layout: None,
                },
            );
        }
    }
    layouts
}

fn terminal_exit_layout_for_trace(
    trace: &CompiledTrace,
    trace_id: u64,
    op_index: usize,
) -> Option<CompiledExitLayout> {
    if let Some(layout) = trace.terminal_exit_layouts.get(&op_index) {
        return Some(layout.public(
            trace_id,
            find_fail_index_for_exit_op(&trace.ops, op_index).unwrap_or(u32::MAX),
        ));
    }
    if let Some(fail_index) = find_fail_index_for_exit_op(&trace.ops, op_index) {
        if let Some(layout) = trace.exit_layouts.get(&fail_index) {
            return Some(layout.public(trace_id, fail_index));
        }
    }
    infer_terminal_exit_layout(&trace.inputargs, &trace.ops, trace_id, op_index)
}

fn decode_values_with_layout(raw_values: &[i64], layout: &CompiledExitLayout) -> Vec<Value> {
    layout
        .exit_types
        .iter()
        .enumerate()
        .map(|(index, tp)| {
            let raw = raw_values.get(index).copied().unwrap_or(0);
            match tp {
                Type::Int => Value::Int(raw),
                Type::Ref => Value::Ref(GcRef(raw as usize)),
                Type::Float => Value::Float(f64::from_bits(raw as u64)),
                Type::Void => Value::Void,
            }
        })
        .collect()
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
    use majit_codegen::{Backend, ExitFrameLayout, ExitRecoveryLayout};
    use majit_codegen_cranelift::compiler::{
        force_token_to_dead_frame, get_int_from_deadframe, get_latest_descr_from_deadframe,
        set_savedata_ref_on_deadframe,
    };
    use majit_codegen_cranelift::guard::CraneliftFailDescr;
    use majit_gc::collector::MiniMarkGC;
    use majit_ir::descr::{CallDescr, Descr, EffectInfo, ExtraEffect};
    use majit_ir::{DescrRef, InputArg, Op, OpCode, OpRef, Type};
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
            static EFFECT_INFO: EffectInfo = EffectInfo {
                extra_effect: ExtraEffect::CanRaise,
                oopspec_index: majit_ir::OopSpecIndex::None,
            };
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
            values.push(get_latest_descr_from_deadframe(&deadframe).unwrap().fail_index() as i64);
            values.push(get_int_from_deadframe(&deadframe, 0).unwrap());
            values.push(get_int_from_deadframe(&deadframe, 1).unwrap());
            drop(values);
            set_savedata_ref_on_deadframe(&mut deadframe, GcRef(0xDADA)).unwrap();
        });
    }

    fn install_compiled_entry(
        meta: &mut MetaInterp<()>,
        green_key: u64,
        inputargs: &[InputArg],
        ops: Vec<Op>,
        constants: HashMap<u32, i64>,
    ) {
        meta.backend.set_constants(constants.clone());
        let mut token = LoopToken::new(green_key + 1000);
        let trace_id = meta.alloc_trace_id();
        meta.backend.set_next_trace_id(trace_id);
        meta.backend
            .compile_loop(inputargs, &ops, &mut token)
            .expect("loop should compile");
        let (mut resume_data, guard_op_indices, mut exit_layouts) =
            build_guard_metadata(inputargs, &ops, green_key);
        let mut terminal_exit_layouts = build_terminal_exit_layouts(inputargs, &ops);
        if let Some(backend_layouts) = meta.backend.compiled_fail_descr_layouts(&token) {
            merge_backend_exit_layouts(&mut exit_layouts, &backend_layouts);
        }
        if let Some(backend_layouts) = meta.backend.compiled_terminal_exit_layouts(&token) {
            merge_backend_terminal_exit_layouts(&mut terminal_exit_layouts, &backend_layouts);
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
                root_trace_id: trace_id,
                guard_failures: HashMap::new(),
                traces,
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
        install_compiled_entry(meta, green_key, &inputargs, ops, constants);
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
                token: LoopToken::new(1),
                num_inputs: 2,
                meta: (),
                root_trace_id: trace_id,
                guard_failures: HashMap::new(),
                traces,
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
                token: LoopToken::new(100),
                num_inputs: 2,
                meta: (),
                root_trace_id: trace_id,
                guard_failures: HashMap::new(),
                traces,
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
        install_compiled_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());
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
        install_compiled_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());
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

        merge_backend_exit_layouts(
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

        merge_backend_terminal_exit_layouts(
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
                token: LoopToken::new(2),
                num_inputs: 1,
                meta: (),
                root_trace_id: trace_id,
                guard_failures: HashMap::new(),
                traces,
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

        install_compiled_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());

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

        install_compiled_entry(&mut meta, green_key, &inputargs, ops, constants);

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

        install_compiled_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());

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

        install_compiled_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());
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

        install_compiled_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());
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
        install_compiled_entry(&mut meta, green_key, &inputargs, root_ops, root_constants);

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
        install_compiled_entry(&mut meta, green_key, &inputargs, root_ops, root_constants);

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
        install_compiled_entry(&mut meta, green_key, &inputargs, root_ops, HashMap::new());
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

        install_compiled_entry(&mut meta, green_key, &inputargs, ops, constants);

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

        install_compiled_entry(&mut meta, green_key, &inputargs, ops, constants);

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

        install_compiled_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());

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

        install_compiled_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());

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

        install_compiled_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());
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
                token: LoopToken::new(700),
                num_inputs: 0,
                meta: (),
                root_trace_id: trace_id,
                guard_failures: HashMap::new(),
                traces,
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

        install_compiled_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());
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

        install_compiled_entry(&mut meta, green_key, &inputargs, ops, constants);

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
}
