use crate::blackhole::ExceptionState;
use crate::jit_state::JitState;
use crate::meta_interp::{
    BackEdgeAction, CompiledExitLayout, DetailedDriverRunOutcome, InlineDecision, MetaInterp,
};
use crate::resume::ResumeLayoutSummary;
use crate::trace_ctx::{DeclarativeJitDriver, JitDriverDescriptor, TraceCtx};
use crate::virtualizable::VirtualizableInfo;
use crate::TraceAction;
use majit_ir::{GreenKey, Type, Value};

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
    descriptor: Option<JitDriverDescriptor>,
}

impl<S: JitState> JitDriver<S> {
    /// Create a new JitDriver with the given hot-counting threshold.
    pub fn new(threshold: u32) -> Self {
        JitDriver {
            meta: MetaInterp::new(threshold),
            sym: None,
            trace_meta: None,
            descriptor: None,
        }
    }

    pub fn with_descriptor(threshold: u32, descriptor: JitDriverDescriptor) -> Self {
        let mut driver = Self::new(threshold);
        driver.descriptor = Some(descriptor);
        driver
    }

    /// Get compiled loop metadata for the given green key.
    pub fn get_compiled_meta(&self, green_key: u64) -> Option<&S::Meta> {
        self.meta.get_compiled_meta(green_key)
    }

    pub fn with_declarative<D: DeclarativeJitDriver>(
        threshold: u32,
        green_types: &[Type],
        red_types: &[Type],
    ) -> Result<Self, &'static str> {
        let descriptor = D::descriptor(green_types, red_types)?;
        Ok(Self::with_descriptor(threshold, descriptor))
    }

    /// Whether the driver is currently tracing.
    #[inline]
    pub fn is_tracing(&self) -> bool {
        self.meta.is_tracing()
    }

    /// The green key of the active trace, if any.
    pub fn current_trace_green_key(&mut self) -> Option<u64> {
        self.meta.trace_ctx().map(|ctx| ctx.green_key())
    }

    /// Abort the active trace and clear driver-owned symbolic state.
    pub fn abort_current_trace(&mut self, permanent: bool) {
        if self.meta.is_tracing() {
            self.meta.abort_trace(permanent);
            self.sym = None;
            self.trace_meta = None;
        }
    }

    /// Tracing hook — call at the top of the dispatch loop.
    ///
    /// If tracing is active, calls `trace_fn` with the active [`TraceCtx`]
    /// and symbolic state, then handles the result automatically:
    ///
    /// - `CloseLoop` → validates depths, collects jump args, compiles
    /// - `Finish` → compiles a terminal trace ending in `FINISH`
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
    #[inline]
    pub fn merge_point<F>(&mut self, trace_fn: F)
    where
        F: FnOnce(&mut TraceCtx, &mut S::Sym) -> TraceAction,
    {
        if !self.meta.is_tracing() {
            return;
        }
        if self.sym.is_none() || self.trace_meta.is_none() {
            self.meta.abort_trace(false);
            self.sym = None;
            self.trace_meta = None;
            return;
        }

        // Phase 1: split-borrow self into meta (for ctx) and sym, run closure
        let action = {
            let Some(ctx) = self.meta.trace_ctx() else {
                self.sym = None;
                self.trace_meta = None;
                return;
            };
            let Some(sym) = self.sym.as_mut() else {
                self.meta.abort_trace(false);
                self.trace_meta = None;
                return;
            };
            trace_fn(ctx, sym)
        }; // ctx and sym references dropped here

        // Phase 2: handle trace result with full access to self
        match action {
            TraceAction::Continue => {}
            TraceAction::CloseLoop => {
                let Some(trace_meta) = self.trace_meta.as_ref() else {
                    self.meta.abort_trace(false);
                    self.sym = None;
                    self.trace_meta = None;
                    return;
                };
                let Some(sym) = self.sym.as_ref() else {
                    self.meta.abort_trace(false);
                    self.trace_meta = None;
                    return;
                };
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
            TraceAction::CloseLoopWithArgs { jump_args } => {
                let Some(trace_meta) = self.trace_meta.as_ref() else {
                    self.meta.abort_trace(false);
                    self.sym = None;
                    self.trace_meta = None;
                    return;
                };
                let Some(sym) = self.sym.as_ref() else {
                    self.meta.abort_trace(false);
                    self.trace_meta = None;
                    return;
                };
                if S::validate_close_with_jump_args(sym, trace_meta, &jump_args) {
                    let meta = self.trace_meta.take().unwrap();
                    self.meta.close_and_compile(&jump_args, meta);
                } else {
                    self.meta.abort_trace(false);
                    self.trace_meta = None;
                }
                self.sym = None;
            }
            TraceAction::Finish {
                finish_args,
                finish_arg_types,
            } => {
                let meta = self.trace_meta.take().unwrap();
                self.meta
                    .finish_and_compile(&finish_args, finish_arg_types, meta);
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
        self.back_edge_internal(target_pc as u64, None, target_pc, state, env, pre_run)
    }

    pub fn back_edge_keyed(
        &mut self,
        green_key: u64,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> bool {
        self.back_edge_internal(green_key, None, target_pc, state, env, pre_run)
    }

    pub fn back_edge_structured(
        &mut self,
        green_key: GreenKey,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> bool {
        let key = green_key.hash_u64();
        self.back_edge_internal(key, Some(green_key), target_pc, state, env, pre_run)
    }

    pub fn back_edge_or_run_compiled(
        &mut self,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<DetailedDriverRunOutcome> {
        self.back_edge_or_run_compiled_internal(
            target_pc as u64,
            None,
            target_pc,
            state,
            env,
            pre_run,
        )
    }

    pub fn back_edge_or_run_compiled_keyed(
        &mut self,
        green_key: u64,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<DetailedDriverRunOutcome> {
        self.back_edge_or_run_compiled_internal(green_key, None, target_pc, state, env, pre_run)
    }

    pub fn back_edge_declarative<D: DeclarativeJitDriver>(
        &mut self,
        green_values: &[i64],
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Result<bool, &'static str> {
        let green_key = D::green_key(green_values)?;
        Ok(self.back_edge_structured(green_key, target_pc, state, env, pre_run))
    }

    fn back_edge_internal(
        &mut self,
        green_key: u64,
        structured_green_key: Option<GreenKey>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> bool {
        if self.meta.is_tracing() || !state.can_trace() {
            return false;
        }

        if self.meta.has_compiled_loop(green_key) {
            let compiled_meta = self.meta.get_compiled_meta(green_key).unwrap();
            let descriptor = self.driver_descriptor_for(state, compiled_meta);
            let live_values = {
                if !state.is_compatible(compiled_meta) {
                    return false;
                }
                if !self.sync_before(state, compiled_meta, descriptor.as_ref()) {
                    return false;
                }
                let live_values = state.extract_live_values(compiled_meta);
                if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
                    return false;
                }
                live_values
            };

            pre_run();

            if let Some((new_values, run_meta)) =
                self.meta.run_compiled_with_values(green_key, &live_values)
            {
                let run_meta = run_meta.clone();
                state.restore_values(&run_meta, &new_values);
                let run_descriptor = self.driver_descriptor_for(state, &run_meta);
                self.sync_after(state, &run_meta, run_descriptor.as_ref());
                return true;
            }
            return false;
        }

        self.maybe_start_tracing(green_key, structured_green_key, target_pc, state, env);
        false
    }

    fn back_edge_or_run_compiled_internal(
        &mut self,
        green_key: u64,
        structured_green_key: Option<GreenKey>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<DetailedDriverRunOutcome> {
        if self.meta.is_tracing() || !state.can_trace() {
            return None;
        }

        if self.meta.has_compiled_loop(green_key) {
            return Some(self.run_compiled_detailed_keyed(green_key, state, pre_run));
        }

        self.maybe_start_tracing(green_key, structured_green_key, target_pc, state, env);
        None
    }

    /// Force-start tracing for a function entry.
    ///
    /// Bypasses the WarmState hot counter (the caller already did its own
    /// counting). Used for function-entry JIT where the threshold is
    /// controlled externally.
    pub fn force_start_tracing(
        &mut self,
        green_key: u64,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) {
        let meta = state.build_meta(target_pc, env);
        let descriptor = self.driver_descriptor_for(state, &meta);
        if !self.sync_before(state, &meta, descriptor.as_ref()) {
            return;
        }
        let live_values = state.extract_live_values(&meta);
        if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
            return;
        }

        match self
            .meta
            .force_start_tracing(green_key, descriptor, &live_values)
        {
            BackEdgeAction::StartedTracing => {
                self.sym = Some(S::create_sym(&meta, target_pc));
                self.trace_meta = Some(meta);
            }
            _ => {}
        }
    }

    fn maybe_start_tracing(
        &mut self,
        green_key: u64,
        structured_green_key: Option<GreenKey>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) {
        let meta = state.build_meta(target_pc, env);
        let descriptor = self.driver_descriptor_for(state, &meta);
        if !self.sync_before(state, &meta, descriptor.as_ref()) {
            return;
        }
        let live_values = state.extract_live_values(&meta);
        if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
            return;
        }

        match self.meta.on_back_edge_typed(
            green_key,
            structured_green_key,
            descriptor,
            &live_values,
        ) {
            BackEdgeAction::Interpret => {}
            BackEdgeAction::StartedTracing => {
                self.sym = Some(S::create_sym(&meta, target_pc));
                self.trace_meta = Some(meta);
            }
            BackEdgeAction::AlreadyTracing | BackEdgeAction::RunCompiled => {}
        }
    }

    fn driver_descriptor_for(&self, state: &S, meta: &S::Meta) -> Option<JitDriverDescriptor> {
        self.descriptor
            .clone()
            .or_else(|| state.driver_descriptor(meta))
    }

    fn live_values_match_descriptor(
        descriptor: Option<&JitDriverDescriptor>,
        live_values: &[Value],
    ) -> bool {
        let Some(descriptor) = descriptor else {
            return true;
        };
        let reds = descriptor.reds();
        reds.len() == live_values.len()
            && reds
                .iter()
                .zip(live_values.iter())
                .all(|(var, value)| var.tp == value.get_type())
    }

    fn sync_before(
        &self,
        state: &mut S,
        meta: &S::Meta,
        descriptor: Option<&JitDriverDescriptor>,
    ) -> bool {
        let Some(descriptor) = descriptor else {
            return true;
        };
        let Some(virtualizable) = descriptor.virtualizable() else {
            return true;
        };
        state.sync_named_virtualizable_before_jit(
            meta,
            &virtualizable.name,
            self.meta.virtualizable_info(),
        )
    }

    fn sync_after(&self, state: &mut S, meta: &S::Meta, descriptor: Option<&JitDriverDescriptor>) {
        let Some(descriptor) = descriptor else {
            return;
        };
        let Some(virtualizable) = descriptor.virtualizable() else {
            return;
        };
        state.sync_named_virtualizable_after_jit(
            meta,
            &virtualizable.name,
            self.meta.virtualizable_info(),
        );
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

    pub fn run_compiled_with_blackhole_fallback_keyed(
        &mut self,
        green_key: u64,
        state: &mut S,
        pre_run: impl FnOnce(),
    ) -> crate::meta_interp::DriverRunOutcome {
        let Some(meta) = self.meta.get_compiled_meta(green_key).cloned() else {
            return crate::meta_interp::DriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };
        let descriptor = self.driver_descriptor_for(state, &meta);
        if !state.is_compatible(&meta) || !self.sync_before(state, &meta, descriptor.as_ref()) {
            return crate::meta_interp::DriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let live_values = state.extract_live_values(&meta);
        if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
            return crate::meta_interp::DriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        pre_run();
        let Some(result) = self
            .meta
            .run_with_blackhole_fallback_with_values(green_key, &live_values)
        else {
            return crate::meta_interp::DriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };

        match result {
            crate::meta_interp::BlackholeRunResult::Finished {
                values,
                typed_values,
                meta,
                via_blackhole,
                ..
            } => {
                if let Some(values) = typed_values.as_ref() {
                    state.restore_values(&meta, values);
                } else {
                    state.restore(&meta, &values);
                }
                self.sync_after(state, &meta, descriptor.as_ref());
                crate::meta_interp::DriverRunOutcome::Finished { via_blackhole }
            }
            crate::meta_interp::BlackholeRunResult::Jump {
                values,
                typed_values,
                exit_layout,
                meta,
                via_blackhole,
                ..
            } => {
                if let Some(values) = typed_values.as_ref() {
                    state.restore_values(&meta, values);
                } else if let Some(layout) = exit_layout.as_ref() {
                    state.restore_values(&meta, &Self::decode_exit_layout_values(&values, layout));
                } else if let Some(values) =
                    Self::decode_descriptor_values(descriptor.as_ref(), &values)
                {
                    state.restore_values(&meta, &values);
                } else {
                    state.restore(&meta, &values);
                }
                self.sync_after(state, &meta, descriptor.as_ref());
                crate::meta_interp::DriverRunOutcome::Jump { via_blackhole }
            }
            crate::meta_interp::BlackholeRunResult::GuardFailure {
                fail_values,
                typed_fail_values,
                exit_layout,
                meta,
                recovery,
                via_blackhole,
                ..
            } => {
                let patched_resume_layout = recovery.as_ref().and_then(|recovery| {
                    recovery.resume_layout.as_ref().and_then(|layout| {
                        Self::resume_layout_with_descriptor_slot_types(descriptor.as_ref(), layout)
                    })
                });
                let restored = if let Some(recovery) = recovery.as_ref() {
                    state.restore_guard_failure_with_resume_layout(
                        &meta,
                        &fail_values,
                        recovery.reconstructed_state.as_ref(),
                        patched_resume_layout
                            .as_ref()
                            .or(recovery.resume_layout.as_ref())
                            .map(|layout| layout.frame_layouts.as_slice()),
                        &recovery.materialized_virtuals,
                        &recovery.pending_field_writes,
                        &recovery.exception,
                    )
                } else if let Some(values) = typed_fail_values.as_ref() {
                    state.restore_guard_failure_values(&meta, values, &ExceptionState::default())
                } else if let Some(layout) = exit_layout.as_ref() {
                    state.restore_guard_failure_values(
                        &meta,
                        &Self::decode_exit_layout_values(&fail_values, layout),
                        &ExceptionState::default(),
                    )
                } else if let Some(values) =
                    Self::decode_descriptor_values(descriptor.as_ref(), &fail_values)
                {
                    state.restore_guard_failure_values(&meta, &values, &ExceptionState::default())
                } else {
                    state.restore_guard_failure_with_resume_layout(
                        &meta,
                        &fail_values,
                        None,
                        None,
                        &[],
                        &[],
                        &ExceptionState::default(),
                    )
                };
                if restored {
                    self.sync_after(state, &meta, descriptor.as_ref());
                }
                crate::meta_interp::DriverRunOutcome::GuardFailure {
                    restored,
                    via_blackhole,
                }
            }
            crate::meta_interp::BlackholeRunResult::Abort { .. } => {
                crate::meta_interp::DriverRunOutcome::Abort {
                    restored: false,
                    via_blackhole: true,
                }
            }
        }
    }

    pub fn run_compiled_with_blackhole_fallback_declarative<D: DeclarativeJitDriver>(
        &mut self,
        green_values: &[i64],
        state: &mut S,
        pre_run: impl FnOnce(),
    ) -> Result<crate::meta_interp::DriverRunOutcome, &'static str> {
        let green_key = D::green_key(green_values)?;
        Ok(self.run_compiled_with_blackhole_fallback_keyed(green_key.hash_u64(), state, pre_run))
    }

    pub fn run_compiled_detailed_keyed(
        &mut self,
        green_key: u64,
        state: &mut S,
        pre_run: impl FnOnce(),
    ) -> DetailedDriverRunOutcome {
        let Some(meta) = self.meta.get_compiled_meta(green_key).cloned() else {
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };
        let descriptor = self.driver_descriptor_for(state, &meta);
        if !state.is_compatible(&meta) || !self.sync_before(state, &meta, descriptor.as_ref()) {
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let live_values = state.extract_live_values(&meta);
        if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        pre_run();
        let Some(result) = self
            .meta
            .run_compiled_detailed_with_values(green_key, &live_values)
        else {
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };

        if result.is_finish {
            return DetailedDriverRunOutcome::Finished {
                typed_values: result.typed_values,
                via_blackhole: false,
            };
        }

        let exit_meta = result.meta.clone();
        state.restore_values(&exit_meta, &result.typed_values);
        self.sync_after(state, &exit_meta, descriptor.as_ref());
        DetailedDriverRunOutcome::Jump {
            via_blackhole: false,
        }
    }

    fn decode_descriptor_values(
        descriptor: Option<&JitDriverDescriptor>,
        raw_values: &[i64],
    ) -> Option<Vec<Value>> {
        let descriptor = descriptor?;
        let reds = descriptor.reds();
        if reds.len() != raw_values.len() {
            return None;
        }
        Some(
            reds.iter()
                .zip(raw_values.iter().copied())
                .map(|(var, raw)| match var.tp {
                    Type::Int => Value::Int(raw),
                    Type::Ref => Value::Ref(majit_ir::GcRef(raw as usize)),
                    Type::Float => Value::Float(f64::from_bits(raw as u64)),
                    Type::Void => Value::Void,
                })
                .collect(),
        )
    }

    fn decode_exit_layout_values(raw_values: &[i64], layout: &CompiledExitLayout) -> Vec<Value> {
        layout
            .exit_types
            .iter()
            .enumerate()
            .map(|(index, tp)| {
                let raw = raw_values.get(index).copied().unwrap_or(0);
                match tp {
                    Type::Int => Value::Int(raw),
                    Type::Ref => Value::Ref(majit_ir::GcRef(raw as usize)),
                    Type::Float => Value::Float(f64::from_bits(raw as u64)),
                    Type::Void => Value::Void,
                }
            })
            .collect()
    }

    fn resume_layout_with_descriptor_slot_types(
        descriptor: Option<&JitDriverDescriptor>,
        resume_layout: &ResumeLayoutSummary,
    ) -> Option<ResumeLayoutSummary> {
        let descriptor = descriptor?;
        let red_types: Vec<Type> = descriptor.reds().iter().map(|var| var.tp).collect();
        let last = resume_layout.frame_layouts.last()?;
        if last.slot_types.is_some() || last.slot_layouts.len() != red_types.len() {
            return None;
        }
        let mut patched = resume_layout.clone();
        if let Some(last) = patched.frame_layouts.last_mut() {
            last.slot_types = Some(red_types);
        }
        Some(patched)
    }

    /// Invalidate a compiled loop, forcing fallback to interpretation.
    pub fn invalidate_loop(&mut self, green_key: u64) {
        self.meta.invalidate_loop(green_key);
    }

    /// Check whether a compiled loop exists for a given green key.
    #[inline]
    pub fn has_compiled_loop(&self, green_key: u64) -> bool {
        self.meta.has_compiled_loop(green_key)
    }

    /// Get the loop token number for a compiled loop.
    pub fn get_loop_token_number(&self, green_key: u64) -> Option<u64> {
        self.meta.get_loop_token(green_key).map(|t| t.number)
    }

    /// Get the pre-allocated token number for the trace being recorded.
    ///
    /// Returns `Some(number)` if `green_key` matches the current trace's
    /// target, enabling self-recursive call_assembler emission.
    pub fn get_pending_token_number(&self, green_key: u64) -> Option<u64> {
        self.meta.get_pending_token_number(green_key)
    }

    /// Decide how to handle a function call during tracing.
    ///
    /// Returns `Inline` if the callee should be traced through,
    /// `CallAssembler` if it already has compiled code, or
    /// `ResidualCall` if it should be left as an opaque call.
    pub fn should_inline(&mut self, callee_key: u64) -> InlineDecision {
        self.meta.should_inline(callee_key)
    }

    /// Begin inlining a function call during tracing.
    ///
    /// Records EnterPortalFrame and pushes an inline frame.
    /// Returns `true` if inlining started successfully.
    pub fn enter_inline_frame(&mut self, callee_key: u64) -> bool {
        self.meta.enter_inline_frame(callee_key)
    }

    /// End an inlined function call during tracing.
    ///
    /// Records LeavePortalFrame and pops the inline frame.
    pub fn leave_inline_frame(&mut self) {
        self.meta.leave_inline_frame()
    }

    /// Get the current inlining depth.
    pub fn inline_depth(&self) -> usize {
        self.meta.inline_depth()
    }

    /// Start bridge tracing from a guard failure point.
    ///
    /// Builds meta/sym from the interpreter state at `resume_pc` and begins
    /// a retrace on the MetaInterp. Returns `true` if tracing was started.
    pub fn start_bridge_tracing(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        state: &mut S,
        env: &S::Env,
        resume_pc: usize,
    ) -> bool {
        let meta = state.build_meta(resume_pc, env);
        let live_values = state.extract_live(& meta);

        if !self.meta.start_retrace(green_key, fail_index, &live_values) {
            return false;
        }

        self.sym = Some(S::create_sym(&meta, resume_pc));
        self.trace_meta = Some(meta);
        true
    }

    /// Close an active bridge trace and compile it.
    ///
    /// Collects the symbolic jump args from the current sym (which represent
    /// the loop header's live state), then calls `close_bridge_with_finish`
    /// on MetaInterp to finish, optimize, and compile the bridge.
    pub fn close_bridge_trace(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> bool {
        let sym = match self.sym.take() {
            Some(sym) => sym,
            None => return false,
        };
        self.trace_meta = None;

        let finish_args = S::collect_jump_args(&sym);

        self.meta
            .close_bridge_with_finish(green_key, trace_id, fail_index, &finish_args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resume::{ReconstructedFrame, ReconstructedState, ReconstructedValue};
    use majit_ir::{GcRef, OpCode, OpRef, Type, Value};

    #[derive(Default)]
    struct TypedRestoreState {
        live_values: Vec<i64>,
        restored_values: Vec<Value>,
        restore_called: bool,
    }

    #[derive(Default)]
    struct TypedInputState {
        raw_live_values: Vec<i64>,
        typed_live_values: Vec<Value>,
        restored_values: Vec<Value>,
        raw_restore_calls: usize,
        typed_restore_calls: usize,
    }

    #[derive(Default)]
    struct FrameMetadataState {
        seen_trace_id: Option<u64>,
        seen_header_pc: Option<u64>,
        seen_source_guard: Option<(u64, u32)>,
        restored_values: Vec<Value>,
    }

    impl JitState for TypedRestoreState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            self.live_values.clone()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {
            self.restore_called = true;
        }

        fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
            self.restored_values = values.to_vec();
        }

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    impl JitState for TypedInputState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            self.raw_live_values.clone()
        }

        fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
            self.typed_live_values.clone()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {
            self.raw_restore_calls += 1;
        }

        fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
            self.typed_restore_calls += 1;
            self.restored_values = values.to_vec();
        }

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    impl JitState for FrameMetadataState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            Vec::new()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {}

        fn restore_reconstructed_frame_values_with_metadata(
            &mut self,
            _meta: &Self::Meta,
            _frame_index: usize,
            _total_frames: usize,
            frame: &ReconstructedFrame,
            values: &[Value],
            _exception: &ExceptionState,
        ) -> bool {
            self.seen_trace_id = frame.trace_id;
            self.seen_header_pc = frame.header_pc;
            self.seen_source_guard = frame.source_guard;
            self.restored_values = values.to_vec();
            true
        }

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    #[test]
    fn back_edge_uses_typed_restore_values_on_compiled_fast_path() {
        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        let key = 7u64;

        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::StartedTracing
        ));

        {
            let ctx = driver.meta.trace_ctx().expect("trace ctx should exist");
            let cond = ctx.const_int(1);
            let value = ctx.const_int(7);
            let float = ctx.record_op(OpCode::CastIntToFloat, &[value]);
            ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[cond], 0, &[float]);
        }
        driver.meta.close_and_compile(&[], ());
        assert!(driver.has_compiled_loop(key));

        let mut state = TypedRestoreState {
            live_values: vec![1],
            ..Default::default()
        };
        assert!(driver.back_edge(key as usize, &mut state, &(), || {}));
        assert!(!state.restore_called);
        assert_eq!(state.restored_values, vec![Value::Float(7.0)]);
    }

    #[test]
    fn blackhole_jump_reports_via_blackhole_even_with_typed_restore_values() {
        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        let key = 9u64;

        assert!(matches!(
            driver.meta.on_back_edge(key, &[1]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[1]),
            BackEdgeAction::StartedTracing
        ));

        let cond = {
            let ctx = driver.meta.trace_ctx().expect("trace ctx should exist");
            let cond = ctx.const_int(1);
            ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[cond], 0, &[cond]);
            cond
        };
        driver.meta.close_and_compile(&[cond], ());
        assert!(driver.has_compiled_loop(key));

        let mut state = TypedRestoreState {
            live_values: vec![1],
            ..Default::default()
        };
        match driver.run_compiled_with_blackhole_fallback_keyed(key, &mut state, || {}) {
            crate::meta_interp::DriverRunOutcome::Jump { via_blackhole } => {
                assert!(via_blackhole);
            }
            other => panic!("expected Jump outcome, got {other:?}"),
        }
        assert_eq!(state.restored_values, vec![Value::Int(1)]);
        assert!(!state.restore_called);
    }

    #[test]
    fn run_compiled_detailed_keyed_uses_typed_live_inputs() {
        let mut driver = JitDriver::<TypedInputState>::new(1);
        let key = 11u64;
        let typed_live_values = vec![Value::Ref(GcRef(0x1234)), Value::Float(3.5)];

        assert!(matches!(
            driver
                .meta
                .on_back_edge_typed(key, None, None, &typed_live_values),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver
                .meta
                .on_back_edge_typed(key, None, None, &typed_live_values),
            BackEdgeAction::StartedTracing
        ));

        let cond = {
            let ctx = driver.meta.trace_ctx().expect("trace ctx should exist");
            let cond = ctx.const_int(1);
            ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[cond], 0, &[OpRef(0), OpRef(1)]);
            cond
        };
        driver.meta.close_and_compile(&[OpRef(0), OpRef(1)], ());
        assert!(driver.has_compiled_loop(key));

        let mut state = TypedInputState {
            raw_live_values: vec![0, 0],
            typed_live_values: typed_live_values.clone(),
            ..Default::default()
        };
        match driver.run_compiled_detailed_keyed(key, &mut state, || {}) {
            DetailedDriverRunOutcome::Jump { via_blackhole } => {
                assert!(!via_blackhole);
            }
            other => panic!("expected Jump outcome, got {other:?}"),
        }
        assert_eq!(state.restored_values, typed_live_values);
        assert_eq!(state.raw_restore_calls, 0);
        assert_eq!(state.typed_restore_calls, 1);
        let _ = cond;
    }

    #[test]
    fn blackhole_fallback_uses_typed_live_inputs() {
        let mut driver = JitDriver::<TypedInputState>::new(1);
        let key = 12u64;
        let typed_live_values = vec![Value::Ref(GcRef(0x5678)), Value::Float(6.25)];

        assert!(matches!(
            driver
                .meta
                .on_back_edge_typed(key, None, None, &typed_live_values),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver
                .meta
                .on_back_edge_typed(key, None, None, &typed_live_values),
            BackEdgeAction::StartedTracing
        ));

        let cond = {
            let ctx = driver.meta.trace_ctx().expect("trace ctx should exist");
            let cond = ctx.const_int(1);
            ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[cond], 0, &[OpRef(0), OpRef(1)]);
            cond
        };
        driver.meta.close_and_compile(&[OpRef(0), OpRef(1)], ());
        assert!(driver.has_compiled_loop(key));

        let mut state = TypedInputState {
            raw_live_values: vec![0, 0],
            typed_live_values: typed_live_values.clone(),
            ..Default::default()
        };
        match driver.run_compiled_with_blackhole_fallback_keyed(key, &mut state, || {}) {
            crate::meta_interp::DriverRunOutcome::Jump { via_blackhole } => {
                assert!(via_blackhole);
            }
            other => panic!("expected Jump outcome, got {other:?}"),
        }
        assert_eq!(state.restored_values, typed_live_values);
        assert_eq!(state.raw_restore_calls, 0);
        assert_eq!(state.typed_restore_calls, 1);
        let _ = cond;
    }

    #[test]
    fn generic_guard_restore_uses_embedded_reconstructed_frame_slot_types() {
        let mut state = TypedRestoreState::default();
        let reconstructed_state = ReconstructedState {
            frames: vec![ReconstructedFrame {
                trace_id: Some(77),
                header_pc: Some(88),
                source_guard: Some((70, 7)),
                pc: 99,
                slot_types: Some(vec![Type::Ref, Type::Float]),
                values: vec![
                    ReconstructedValue::Value(0x1234),
                    ReconstructedValue::Value(6.25f64.to_bits() as i64),
                ],
            }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };

        assert!(
            <TypedRestoreState as JitState>::restore_guard_failure_with_resume_layout(
                &mut state,
                &(),
                &[],
                Some(&reconstructed_state),
                None,
                &[],
                &[],
                &ExceptionState::default(),
            )
        );
        assert!(!state.restore_called);
        assert_eq!(
            state.restored_values,
            vec![Value::Ref(GcRef(0x1234)), Value::Float(6.25)]
        );
    }

    #[test]
    fn generic_guard_restore_passes_embedded_reconstructed_frame_metadata() {
        let mut state = FrameMetadataState::default();
        let reconstructed_state = ReconstructedState {
            frames: vec![ReconstructedFrame {
                trace_id: Some(701),
                header_pc: Some(1701),
                source_guard: Some((700, 0)),
                pc: 99,
                slot_types: Some(vec![Type::Int]),
                values: vec![ReconstructedValue::Value(44)],
            }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };

        assert!(
            <FrameMetadataState as JitState>::restore_guard_failure_with_resume_layout(
                &mut state,
                &(),
                &[],
                Some(&reconstructed_state),
                None,
                &[],
                &[],
                &ExceptionState::default(),
            )
        );
        assert_eq!(state.seen_trace_id, Some(701));
        assert_eq!(state.seen_header_pc, Some(1701));
        assert_eq!(state.seen_source_guard, Some((700, 0)));
        assert_eq!(state.restored_values, vec![Value::Int(44)]);
    }
}
