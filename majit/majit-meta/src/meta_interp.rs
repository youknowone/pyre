use std::collections::HashMap;

use majit_codegen::{Backend, LoopToken};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{OpRef, Type, Value};
use majit_opt::optimizer::Optimizer;
use majit_trace::trace::Trace;
use majit_trace::warmstate::{HotResult, WarmState};

use crate::io_buffer;
use crate::resume::ResumeData;
use crate::trace_ctx::TraceCtx;

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
pub struct CompileResult<'a, M> {
    /// The live values at the point of guard failure (or loop finish).
    pub values: Vec<i64>,
    /// The interpreter-specific metadata for this loop.
    pub meta: &'a M,
    /// Index of the failed guard (0 for Finish, >0 for guard failures).
    pub fail_index: u32,
}

/// Per-guard failure tracking for bridge compilation decisions.
struct GuardFailureInfo {
    /// Number of times this guard has failed.
    fail_count: u32,
    /// Whether a bridge has been compiled for this guard.
    bridge_compiled: bool,
}

struct CompiledEntry<M> {
    token: LoopToken,
    num_inputs: usize,
    meta: M,
    /// Per-guard failure tracking, keyed by fail_index.
    guard_failures: HashMap<u32, GuardFailureInfo>,
    /// Resume data for each guard, keyed by fail_index.
    resume_data: HashMap<u32, ResumeData>,
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
    /// Trace eagerness: start tracing from a guard failure point
    /// after this many failures (0 = never trace from guards).
    trace_eagerness: u32,
}

impl<M: Clone> MetaInterp<M> {
    /// Create a new MetaInterp with the given compilation threshold.
    pub fn new(threshold: u32) -> Self {
        MetaInterp {
            warm_state: WarmState::new(threshold),
            backend: CraneliftBackend::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
            trace_eagerness: 200,
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

    /// Decay all counters to avoid stale hotness data.
    pub fn decay_counters(&mut self) {
        self.warm_state.decay_counters();
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
        if self.tracing.is_some() {
            return BackEdgeAction::AlreadyTracing;
        }

        match self.warm_state.maybe_compile(green_key) {
            HotResult::NotHot => BackEdgeAction::Interpret,
            HotResult::StartTracing(mut recorder) => {
                // Register all live values as InputArgs
                for _ in live_values {
                    recorder.record_input_arg(Type::Int);
                }

                if std::env::var("MAJIT_LOG").is_ok() {
                    eprintln!(
                        "[jit] start tracing at key={}, num_inputs={}",
                        green_key,
                        live_values.len()
                    );
                }

                self.tracing = Some(TraceCtx::new(recorder, green_key));
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

        if std::env::var("MAJIT_LOG").is_ok() {
            eprintln!("--- trace (before opt) ---");
            eprint!("{}", majit_ir::format_trace(&trace.ops, &constants));
        }

        let optimized_ops = optimizer.optimize_with_constants(&trace.ops, &mut constants);

        if std::env::var("MAJIT_LOG").is_ok() {
            eprintln!("--- trace (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
        }

        self.backend.set_constants(constants);

        let token_num = self.warm_state.alloc_token_number();
        let mut token = LoopToken::new(token_num);

        match self
            .backend
            .compile_loop(&trace.inputargs, &optimized_ops, &mut token)
        {
            Ok(_) => {
                if std::env::var("MAJIT_LOG").is_ok() {
                    eprintln!(
                        "[jit] compiled loop at key={}, num_inputs={}",
                        green_key,
                        trace.inputargs.len()
                    );
                }
                self.compiled_loops.insert(
                    green_key,
                    CompiledEntry {
                        token,
                        num_inputs: trace.inputargs.len(),
                        meta,
                        guard_failures: HashMap::new(),
                        resume_data: HashMap::new(),
                    },
                );
                let install_num = self.warm_state.alloc_token_number();
                let install_token = LoopToken::new(install_num);
                self.warm_state.install_compiled(green_key, install_token);
            }
            Err(e) => {
                eprintln!("JIT compilation failed: {e}");
                self.warm_state.abort_tracing(green_key, true);
            }
        }
    }

    /// Abort the current trace.
    ///
    /// If `permanent` is true, this location will never be traced again.
    pub fn abort_trace(&mut self, permanent: bool) {
        if let Some(ctx) = self.tracing.take() {
            let green_key = ctx.green_key;
            if std::env::var("MAJIT_LOG").is_ok() {
                eprintln!(
                    "[jit] abort trace at key={} (permanent={})",
                    green_key, permanent
                );
            }
            ctx.recorder.abort();
            self.warm_state.abort_tracing(green_key, permanent);
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

        let args: Vec<Value> = live_values.iter().map(|&v| Value::Int(v)).collect();

        io_buffer::io_buffer_discard();
        let _jitted = majit_codegen::JittedGuard::enter();
        let frame = self.backend.execute_token(&compiled.token, &args);
        drop(_jitted);
        io_buffer::io_buffer_discard();

        // Read guard failure information
        let descr = self.backend.get_latest_descr(&frame);
        let fail_index = descr.fail_index();

        // Track guard failures for bridge compilation decisions
        if fail_index > 0 {
            let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
            let info = compiled.guard_failures.entry(fail_index).or_insert(GuardFailureInfo {
                fail_count: 0,
                bridge_compiled: false,
            });
            info.fail_count += 1;

            if std::env::var("MAJIT_LOG").is_ok() {
                eprintln!(
                    "[jit] guard failure at key={}, guard={}, count={}",
                    green_key, fail_index, info.fail_count
                );
            }

            self.warm_state.log_guard_failure(fail_index);
        }

        let compiled = self.compiled_loops.get(&green_key).unwrap();
        let mut output = Vec::with_capacity(compiled.num_inputs);
        for i in 0..compiled.num_inputs {
            output.push(self.backend.get_int_value(&frame, i));
        }

        Some((output, &compiled.meta))
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

        let args: Vec<Value> = live_values.iter().map(|&v| Value::Int(v)).collect();

        io_buffer::io_buffer_discard();
        let _jitted = majit_codegen::JittedGuard::enter();
        let frame = self.backend.execute_token(&compiled.token, &args);
        drop(_jitted);
        io_buffer::io_buffer_discard();

        let descr = self.backend.get_latest_descr(&frame);
        let fail_index = descr.fail_index();

        // Track guard failures
        if fail_index > 0 {
            let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
            let info = compiled.guard_failures.entry(fail_index).or_insert(GuardFailureInfo {
                fail_count: 0,
                bridge_compiled: false,
            });
            info.fail_count += 1;
            self.warm_state.log_guard_failure(fail_index);
        }

        let compiled = self.compiled_loops.get(&green_key).unwrap();
        let mut values = Vec::with_capacity(compiled.num_inputs);
        for i in 0..compiled.num_inputs {
            values.push(self.backend.get_int_value(&frame, i));
        }

        Some(CompileResult {
            values,
            meta: &compiled.meta,
            fail_index,
        })
    }

    /// Attach resume data to a specific guard in a compiled loop.
    ///
    /// This allows the interpreter to later reconstruct its full state
    /// when the guard fails, using `get_resume_data`.
    pub fn attach_resume_data(
        &mut self,
        green_key: u64,
        fail_index: u32,
        resume_data: ResumeData,
    ) {
        if let Some(compiled) = self.compiled_loops.get_mut(&green_key) {
            compiled.resume_data.insert(fail_index, resume_data);
        }
    }

    /// Get resume data for a specific guard failure.
    pub fn get_resume_data(&self, green_key: u64, fail_index: u32) -> Option<&ResumeData> {
        self.compiled_loops
            .get(&green_key)
            .and_then(|c| c.resume_data.get(&fail_index))
    }

    /// Check whether a guard has failed enough times to warrant bridge compilation.
    pub fn should_compile_bridge(&self, green_key: u64, fail_index: u32) -> bool {
        self.compiled_loops
            .get(&green_key)
            .and_then(|c| c.guard_failures.get(&fail_index))
            .is_some_and(|info| {
                !info.bridge_compiled
                    && self.warm_state.should_compile_bridge(info.fail_count)
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
            if std::env::var("MAJIT_LOG").is_ok() {
                eprintln!("[jit] invalidated loop at key={}", green_key);
            }
        }
    }

    /// Get the guard failure count for a specific guard.
    pub fn get_guard_failure_count(&self, green_key: u64, fail_index: u32) -> u32 {
        self.compiled_loops
            .get(&green_key)
            .and_then(|c| c.guard_failures.get(&fail_index))
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
    pub fn has_compiled_loop(&self, green_key: u64) -> bool {
        self.compiled_loops.contains_key(&green_key)
    }

    // ── Bridge Compilation ──────────────────────────────────────

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
        let optimized_ops = optimizer.optimize_with_constants(bridge_ops, &mut constants);

        self.backend.set_constants(constants);

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
                if std::env::var("MAJIT_LOG").is_ok() {
                    eprintln!(
                        "[jit] compiled bridge at key={}, guard={}",
                        green_key, fail_index
                    );
                }
                // Mark the bridge as compiled
                if let Some(compiled) = self.compiled_loops.get_mut(&green_key) {
                    if let Some(info) = compiled.guard_failures.get_mut(&fail_index) {
                        info.bridge_compiled = true;
                    }
                }
                self.warm_state.log_bridge_compile(fail_index);
                true
            }
            Err(e) => {
                eprintln!("Bridge compilation failed: {e}");
                false
            }
        }
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
    ) -> Option<GuardRecovery> {
        let compiled = self.compiled_loops.get(&green_key)?;

        // Reconstruct interpreter state from resume data if available
        let reconstructed = if let Some(resume_data) = compiled.resume_data.get(&fail_index) {
            let frames = resume_data.reconstruct(fail_values);
            Some(frames)
        } else {
            None
        };

        // Decide what to do next
        let action = if self.should_compile_bridge(green_key, fail_index) {
            GuardRecoveryAction::CompileBridge
        } else if self.get_guard_failure_count(green_key, fail_index) >= self.trace_eagerness
            && self.trace_eagerness > 0
        {
            GuardRecoveryAction::RetraceFromGuard
        } else {
            GuardRecoveryAction::ResumeInterpreter
        };

        Some(GuardRecovery {
            fail_index,
            fail_values: fail_values.to_vec(),
            reconstructed_frames: reconstructed,
            action,
        })
    }

    /// Run compiled code and handle guard failures automatically.
    ///
    /// This is a convenience wrapper around `run_compiled_detailed` +
    /// `handle_guard_failure`.
    pub fn run_and_recover(
        &mut self,
        green_key: u64,
        live_values: &[i64],
    ) -> Option<RunResult<M>> {
        let result = self.run_compiled_detailed(green_key, live_values)?;
        let fail_index = result.fail_index;
        let values = result.values.clone();
        let meta = result.meta.clone();

        if fail_index == 0 {
            // Normal finish (not a guard failure)
            return Some(RunResult::Finished { values, meta });
        }

        // Guard failure — recover
        let recovery = self.handle_guard_failure(green_key, fail_index, &values);

        Some(RunResult::GuardFailure {
            values,
            meta,
            fail_index,
            recovery,
        })
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
    pub fn start_retrace(
        &mut self,
        green_key: u64,
        _fail_index: u32,
        live_values: &[i64],
    ) -> bool {
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

        if std::env::var("MAJIT_LOG").is_ok() {
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
    /// In RPython, the meta-interpreter decides whether to trace into
    /// a called function (inline it) or emit a residual call. Factors:
    /// - Does the callee have a JitDriver (is it jittable)?
    /// - Has the callee been compiled before?
    /// - Is the call recursion depth too deep?
    ///
    /// `callee_key` identifies the called function's JitDriver.
    pub fn should_inline(&self, callee_key: u64) -> InlineDecision {
        // If the callee already has compiled code, don't inline —
        // use CALL_ASSEMBLER instead.
        if self.compiled_loops.contains_key(&callee_key) {
            return InlineDecision::CallAssembler;
        }

        // If we're already tracing, we can potentially inline.
        if self.tracing.is_some() {
            return InlineDecision::Inline;
        }

        // Otherwise, emit a regular residual call.
        InlineDecision::ResidualCall
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

/// Describes the recovery state after a guard failure.
#[derive(Debug, Clone)]
pub struct GuardRecovery {
    /// Index of the failed guard.
    pub fail_index: u32,
    /// Raw fail_values from the DeadFrame.
    pub fail_values: Vec<i64>,
    /// Reconstructed interpreter frames (if resume data was available).
    pub reconstructed_frames: Option<Vec<crate::resume::ReconstructedFrame>>,
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
    },
    /// A guard failed.
    GuardFailure {
        values: Vec<i64>,
        meta: M,
        fail_index: u32,
        recovery: Option<GuardRecovery>,
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
