use crate::jit_state::JitState;
use crate::meta_interp::{BackEdgeAction, JitHooks, MetaInterp};
use crate::trace_ctx::TraceCtx;
use crate::virtualizable::VirtualizableInfo;
use crate::TraceAction;

/// High-level JIT driver that automates the tracing lifecycle.
///
/// Wraps [`MetaInterp`] and manages symbolic state, replacing ~80 lines
/// of boilerplate in each interpreter's mainloop with two method calls:
///
/// - [`merge_point`](JitDriver::merge_point) — replaces the tracing hook
///   (`jit_merge_point` in RPython)
/// - [`back_edge`](JitDriver::back_edge) — replaces back-edge handling
///   (`can_enter_jit` in RPython)
///
/// # Type Parameter
///
/// `S` implements [`JitState`] and defines how the interpreter's live
/// state maps to/from the JIT's representation.
pub struct JitDriver<S: JitState> {
    meta: MetaInterp<S::Meta>,
    sym: Option<S::Sym>,
    trace_meta: Option<S::Meta>,
}

impl<S: JitState> JitDriver<S> {
    /// Create a new JitDriver with the given hot-counting threshold.
    pub fn new(threshold: u32) -> Self {
        JitDriver {
            meta: MetaInterp::new(threshold),
            sym: None,
            trace_meta: None,
        }
    }

    /// Whether the driver is currently tracing.
    pub fn is_tracing(&self) -> bool {
        self.meta.is_tracing()
    }

    /// Tracing hook — call at the top of the dispatch loop.
    ///
    /// If tracing is active, calls `trace_fn` with the active [`TraceCtx`]
    /// and symbolic state, then handles the result automatically:
    ///
    /// - `CloseLoop` → validates depths, collects jump args, compiles
    /// - `Abort` → aborts trace (may retry later)
    /// - `AbortPermanent` → aborts trace permanently
    /// - `Continue` → no action
    ///
    /// # Example
    ///
    /// ```ignore
    /// driver.merge_point(|ctx, sym| {
    ///     trace_instruction(ctx, sym, program, pc, &state)
    /// });
    /// ```
    pub fn merge_point<F>(&mut self, trace_fn: F)
    where
        F: FnOnce(&mut TraceCtx, &mut S::Sym) -> TraceAction,
    {
        if !self.meta.is_tracing() {
            return;
        }

        // Phase 1: split-borrow self into meta (for ctx) and sym, run closure
        let action = {
            let ctx = self.meta.trace_ctx().unwrap();
            let sym = self.sym.as_mut().unwrap();
            trace_fn(ctx, sym)
        }; // ctx and sym references dropped here

        // Phase 2: handle trace result with full access to self
        match action {
            TraceAction::Continue => {}
            TraceAction::CloseLoop => {
                let trace_meta = self.trace_meta.as_ref().unwrap();
                let sym = self.sym.as_ref().unwrap();
                if S::validate_close(sym, trace_meta) {
                    let jump_args = S::collect_jump_args(sym);
                    let meta = self.trace_meta.take().unwrap();
                    self.meta.close_and_compile(&jump_args, meta);
                } else {
                    self.meta.abort_trace(false);
                    self.trace_meta = None;
                }
                self.sym = None;
            }
            TraceAction::Abort => {
                self.meta.abort_trace(false);
                self.sym = None;
                self.trace_meta = None;
            }
            TraceAction::AbortPermanent => {
                self.meta.abort_trace(true);
                self.sym = None;
                self.trace_meta = None;
            }
        }
    }

    /// Back-edge handler — call when a backward jump is detected.
    ///
    /// Handles hot counting, trace start, and compiled code execution.
    /// Returns `true` if compiled code ran successfully and the interpreter
    /// should jump to `target_pc` with restored state.
    ///
    /// `pre_run` is called before executing compiled code (e.g., to flush
    /// buffered I/O).
    ///
    /// # Example
    ///
    /// ```ignore
    /// if target <= pc {
    ///     if driver.back_edge(target, &mut state, program, || writer.flush()) {
    ///         pc = target;
    ///         continue;
    ///     }
    /// }
    /// ```
    pub fn back_edge(
        &mut self,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> bool {
        if self.meta.is_tracing() || !state.can_trace() {
            return false;
        }

        let meta = state.build_meta(target_pc, env);
        let live = state.extract_live(&meta);

        match self.meta.on_back_edge(target_pc as u64, &live) {
            BackEdgeAction::Interpret => false,
            BackEdgeAction::StartedTracing => {
                self.sym = Some(S::create_sym(&meta, target_pc));
                self.trace_meta = Some(meta);
                false
            }
            BackEdgeAction::AlreadyTracing => false,
            BackEdgeAction::RunCompiled => {
                // Check compatibility before executing
                let compatible = self
                    .meta
                    .get_compiled_meta(target_pc as u64)
                    .is_some_and(|m| state.is_compatible(m));
                if !compatible {
                    return false;
                }

                pre_run();

                let live = state.extract_live(&meta);
                if let Some((new_values, run_meta)) =
                    self.meta.run_compiled(target_pc as u64, &live)
                {
                    let restore_meta = run_meta.clone();
                    state.restore(&restore_meta, &new_values);
                    return true;
                }
                false
            }
        }
    }

    /// Set virtualizable info for frame virtualization.
    pub fn set_virtualizable_info(&mut self, info: VirtualizableInfo) {
        self.meta.set_virtualizable_info(info);
    }

    /// Set the compilation threshold.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.meta.set_threshold(threshold);
    }

    /// Set the bridge compilation threshold.
    pub fn set_bridge_threshold(&mut self, threshold: u32) {
        self.meta.set_bridge_threshold(threshold);
    }

    /// Set the trace eagerness (guard failure threshold for bridge tracing).
    pub fn set_trace_eagerness(&mut self, eagerness: u32) {
        self.meta.set_trace_eagerness(eagerness);
    }

    /// Set a callback for loop compilation events.
    pub fn set_on_compile_loop(&mut self, f: impl Fn(u64, usize, usize) + Send + 'static) {
        self.meta.set_on_compile_loop(f);
    }

    /// Set a callback for guard failure events.
    pub fn set_on_guard_failure(&mut self, f: impl Fn(u64, u32, u32) + Send + 'static) {
        self.meta.set_on_guard_failure(f);
    }

    /// Get direct access to the underlying MetaInterp.
    pub fn meta_interp(&self) -> &MetaInterp<S::Meta> {
        &self.meta
    }

    /// Get mutable access to the underlying MetaInterp.
    pub fn meta_interp_mut(&mut self) -> &mut MetaInterp<S::Meta> {
        &mut self.meta
    }

    /// Invalidate a compiled loop, forcing fallback to interpretation.
    pub fn invalidate_loop(&mut self, green_key: u64) {
        self.meta.invalidate_loop(green_key);
    }

    /// Check whether a compiled loop exists for a given green key.
    pub fn has_compiled_loop(&self, green_key: u64) -> bool {
        self.meta.has_compiled_loop(green_key)
    }
}
