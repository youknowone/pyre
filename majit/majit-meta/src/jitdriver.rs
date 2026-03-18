use crate::blackhole::ExceptionState;
use crate::jit_state::JitState;
use crate::pyjitpl::{
    BackEdgeAction, CompiledExitLayout, DetailedDriverRunOutcome, InlineDecision, JitStats,
    MetaInterp,
};
use crate::resume::ResumeLayoutSummary;
use crate::trace_ctx::{DeclarativeJitDriver, JitDriverStaticData, TraceCtx};
use crate::virtualizable::VirtualizableInfo;
use crate::TraceAction;
use majit_ir::OpRef;
use majit_ir::{GreenKey, Type, Value};

/// A named entry point registered with a [`JitDriver`].
///
/// Multiple functions can share the same JitDriver and compiled loops
/// by registering additional entry points with distinct green key schemas.
/// This models the warmspot multi-entry pattern where several
/// `jit_merge_point` calls in different functions share one driver.
#[derive(Debug, Clone)]
pub struct EntryPoint {
    /// Human-readable name of this entry point (e.g., function name).
    pub name: String,
    /// Green key type schema for this entry point.
    pub schema: Vec<Type>,
}

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
/// PyPy warmspot.py JitDriver equivalent.
///
/// Manages the tracing/compilation lifecycle for a single jitdriver.
/// PyPy names: JitDriver(greens=[...], reds=[...], is_recursive=True).
pub struct JitDriver<S: JitState> {
    meta: MetaInterp<S::Meta>,
    sym: Option<S::Sym>,
    trace_meta: Option<S::Meta>,
    descriptor: Option<JitDriverStaticData>,
    /// Bridge tracing state: (green_key, trace_id, fail_index).
    bridge_info: Option<(u64, u64, u32)>,
    /// Additional entry points sharing this driver's compiled loops.
    entry_points: Vec<EntryPoint>,
    /// Key of the loop just compiled — skip first execution to avoid
    /// re-entering in the same back-edge iteration (RPython compiled_new).
    just_compiled_key: Option<u64>,
    /// PyPy JitDriver(is_recursive=True): enables max_unroll_recursion
    /// for recursive portal calls (pyjitpl.py _opimpl_recursive_call).
    is_recursive: bool,
}

impl<S: JitState> JitDriver<S> {
    /// Create a new JitDriver with the given hot-counting threshold.
    pub fn new(threshold: u32) -> Self {
        JitDriver {
            meta: MetaInterp::new(threshold),
            sym: None,
            trace_meta: None,
            descriptor: None,
            bridge_info: None,
            entry_points: Vec::new(),
            just_compiled_key: None,
            is_recursive: false,
        }
    }

    pub fn with_descriptor(threshold: u32, descriptor: JitDriverStaticData) -> Self {
        let mut driver = Self::new(threshold);
        let greens: Vec<Type> = descriptor.greens().iter().map(|v| v.tp).collect();
        driver.descriptor = Some(descriptor);
        driver.entry_points.push(EntryPoint {
            name: "primary".to_string(),
            schema: greens,
        });
        driver
    }

    /// Get compiled loop metadata for the given green key.
    pub fn get_compiled_meta(&self, green_key: u64) -> Option<&S::Meta> {
        self.meta.get_compiled_meta(green_key)
    }

    /// Get the compiled loop's num_inputs (after preamble patching).
    pub fn get_compiled_num_inputs(&self, green_key: u64) -> Option<usize> {
        self.meta.get_compiled_num_inputs(green_key)
    }

    /// Set the virtualizable object pointer for compile-time array length reading.
    ///
    /// RPython parity: compile.py reads array lengths from the actual object.
    pub fn set_vable_ptr(&mut self, ptr: *const u8) {
        self.meta.set_vable_ptr(ptr);
    }

    /// Register an interpreter boxing helper for the raw-int finish protocol.
    pub fn register_raw_int_box_helper(&mut self, helper: *const ()) {
        self.meta.register_raw_int_box_helper(helper);
    }

    /// Register a create_frame_N → create_frame_N_raw_int mapping for box folding.
    pub fn register_create_frame_raw(&mut self, normal: *const (), raw_int: *const ()) {
        self.meta.register_create_frame_raw(normal, raw_int);
    }

    /// PyPy JitDriver(is_recursive=True).
    /// Enables max_unroll_recursion for recursive portal calls.
    pub fn set_is_recursive(&mut self, value: bool) {
        self.is_recursive = value;
    }

    /// PyPy warmspot.py set_param_max_unroll_recursion().
    pub fn set_max_unroll_recursion(&mut self, value: usize) {
        self.meta.set_max_unroll_recursion(value);
    }

    /// Whether this driver was declared recursive.
    pub fn is_recursive(&self) -> bool {
        self.is_recursive
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

    /// Whether the driver is currently tracing a bridge.
    #[inline]
    pub fn is_bridge_tracing(&self) -> bool {
        self.bridge_info.is_some()
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
            self.bridge_info = None;
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
                // Bridge tracing: close as bridge instead of loop.
                if let Some((bridge_key, bridge_trace_id, bridge_fail_index)) =
                    self.bridge_info.take()
                {
                    let sym = self.sym.take();
                    self.trace_meta = None;
                    if let Some(sym) = sym {
                        let finish_args = S::collect_jump_args(&sym);
                        self.meta.close_bridge_with_finish(
                            bridge_key,
                            bridge_trace_id,
                            bridge_fail_index,
                            &finish_args,
                        );
                    } else {
                        self.meta.abort_trace(false);
                    }
                    return;
                }
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
                self.bridge_info = None;
            }
            TraceAction::AbortPermanent => {
                self.meta.abort_trace(true);
                self.sym = None;
                self.trace_meta = None;
                self.bridge_info = None;
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
            let compiled_meta = self.meta.get_compiled_meta(green_key).unwrap().clone();
            let descriptor = self.driver_descriptor_for(state, &compiled_meta);
            let live_values = {
                if !state.is_compatible(&compiled_meta) {
                    return false;
                }
                if !self.sync_before(state, &compiled_meta, descriptor.as_ref()) {
                    return false;
                }
                let live_values = state.extract_live_values(&compiled_meta);
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
    /// Bypasses the WarmEnterState hot counter (the caller already did its own
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

    fn driver_descriptor_for(&self, state: &S, meta: &S::Meta) -> Option<JitDriverStaticData> {
        self.descriptor
            .clone()
            .or_else(|| state.driver_descriptor(meta))
    }

    fn live_values_match_descriptor(
        descriptor: Option<&JitDriverStaticData>,
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
        &mut self,
        state: &mut S,
        meta: &S::Meta,
        descriptor: Option<&JitDriverStaticData>,
    ) -> bool {
        let Some(descriptor) = descriptor else {
            return true;
        };
        let Some(virtualizable) = descriptor.virtualizable() else {
            return true;
        };
        let ok = state.sync_named_virtualizable_before_jit(
            meta,
            &virtualizable.name,
            self.meta.virtualizable_info(),
        );
        if !ok {
            return false;
        }
        // Auto-set vable_ptr + array_lengths from state.
        //
        // RPython compile.py reads lengths from the actual virtualizable
        // object via vinfo.get_array_length(). Here we use two sources:
        // 1. Primary: JitState::virtualizable_array_lengths() (interpreter knows)
        // 2. Fallback: VirtualizableInfo::get_array_length() (from object header)
        //
        // This eliminates the need for callers to manually call
        // driver.set_vable_array_lengths().
        let vable_name = virtualizable.name.clone();
        let info_clone = self.meta.virtualizable_info().cloned();
        if let Some(ref info) = info_clone {
            if let Some(ptr) = state.virtualizable_heap_ptr(meta, &vable_name, info) {
                self.meta.set_vable_ptr(ptr.cast_const());
            }
            // Prefer JitState-provided lengths (interpreter always knows).
            // Fall back to reading from object header if available.
            if let Some(lengths) = state.virtualizable_array_lengths(meta, &vable_name, info) {
                self.meta.set_vable_array_lengths(lengths);
            }
        }
        true
    }

    fn sync_after(&self, state: &mut S, meta: &S::Meta, descriptor: Option<&JitDriverStaticData>) {
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

    /// Register an additional entry point for this driver.
    ///
    /// Multiple functions can share the same JitDriver and compiled loops.
    /// Each entry point has a name and a green key schema describing the
    /// types of its green (loop-invariant) variables.
    pub fn register_entry_point(&mut self, name: &str, green_key_schema: &[Type]) {
        self.entry_points.push(EntryPoint {
            name: name.to_string(),
            schema: green_key_schema.to_vec(),
        });
    }

    /// Return all registered entry points.
    pub fn entry_points(&self) -> &[EntryPoint] {
        &self.entry_points
    }

    /// Look up an entry point by name.
    pub fn find_entry_point(&self, name: &str) -> Option<&EntryPoint> {
        self.entry_points.iter().find(|ep| ep.name == name)
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

    /// Set a JIT parameter by name at runtime.
    ///
    /// Supported parameters:
    /// - `"threshold"` — compilation hot-count threshold
    /// - `"trace_eagerness"` — guard failure count before bridge tracing
    /// - `"bridge_threshold"` — guard failure count before bridge compilation
    /// - `"function_threshold"` — function call count before inlining
    ///
    /// Unknown parameter names are silently ignored.
    pub fn set_param(&mut self, name: &str, value: i64) {
        match name {
            "threshold" => self.meta.set_threshold(value as u32),
            "trace_eagerness" => self.meta.set_trace_eagerness(value as u32),
            "bridge_threshold" => self.meta.set_bridge_threshold(value as u32),
            "function_threshold" => self.meta.set_function_threshold(value as u32),
            _ => {} // unknown params silently ignored
        }
    }

    /// Return a snapshot of the cumulative JIT compilation statistics.
    pub fn get_stats(&self) -> JitStats {
        self.meta.get_stats()
    }

    /// Get direct access to the underlying MetaInterp.
    pub fn meta_interp(&self) -> &MetaInterp<S::Meta> {
        &self.meta
    }

    /// Get mutable access to the underlying MetaInterp.
    pub fn meta_interp_mut(&mut self) -> &mut MetaInterp<S::Meta> {
        &mut self.meta
    }

    /// Check if a function was boosted for fast tracing.
    pub fn is_function_boosted(&self, callee_key: u64) -> bool {
        self.meta.warm_state_ref().is_boosted(callee_key)
    }

    pub fn run_compiled_with_blackhole_fallback_keyed(
        &mut self,
        green_key: u64,
        state: &mut S,
        pre_run: impl FnOnce(),
    ) -> crate::pyjitpl::DriverRunOutcome {
        let Some(meta) = self.meta.get_compiled_meta(green_key).cloned() else {
            return crate::pyjitpl::DriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };
        let descriptor = self.driver_descriptor_for(state, &meta);
        if !state.is_compatible(&meta) || !self.sync_before(state, &meta, descriptor.as_ref()) {
            return crate::pyjitpl::DriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let live_values = state.extract_live_values(&meta);
        if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
            return crate::pyjitpl::DriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        pre_run();
        let Some(result) = self
            .meta
            .run_with_blackhole_fallback_with_values(green_key, &live_values)
        else {
            return crate::pyjitpl::DriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };

        match result {
            crate::pyjitpl::BlackholeRunResult::Finished {
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
                crate::pyjitpl::DriverRunOutcome::Finished { via_blackhole }
            }
            crate::pyjitpl::BlackholeRunResult::Jump {
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
                crate::pyjitpl::DriverRunOutcome::Jump { via_blackhole }
            }
            crate::pyjitpl::BlackholeRunResult::GuardFailure {
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
                crate::pyjitpl::DriverRunOutcome::GuardFailure {
                    restored,
                    via_blackhole,
                }
            }
            crate::pyjitpl::BlackholeRunResult::Abort { .. } => {
                crate::pyjitpl::DriverRunOutcome::Abort {
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
    ) -> Result<crate::pyjitpl::DriverRunOutcome, &'static str> {
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
                raw_int_result: self.meta.has_raw_int_finish(green_key),
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
        descriptor: Option<&JitDriverStaticData>,
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
        descriptor: Option<&JitDriverStaticData>,
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

    /// Whether the compiled finish for this loop exits with a raw int.
    pub fn has_raw_int_finish(&self, green_key: u64) -> bool {
        self.meta.has_raw_int_finish(green_key)
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

    /// Inline decision with externally-held ctx (merge_point callback).
    pub fn should_inline_with_ctx(
        &mut self,
        callee_key: u64,
        ctx: &crate::trace_ctx::TraceCtx,
    ) -> InlineDecision {
        self.meta.should_inline_with_ctx(callee_key, ctx)
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

    /// RPython equivalent: `opimpl_hint_force_virtualizable(box)`
    ///
    /// Call during tracing when the interpreter encounters
    /// `hint_force_virtualizable`. Emits IR to flush virtualizable
    /// boxes back to the heap.
    pub fn opimpl_hint_force_virtualizable(&mut self, vable_opref: OpRef) {
        self.meta.opimpl_hint_force_virtualizable(vable_opref);
    }

    /// Start bridge tracing from a guard failure point.
    ///
    /// Uses the compiled loop's stored meta so that the sym's
    /// storage_layout matches the parent loop's inputargs format.
    /// `resume_pc` is where interpretation resumes after the guard failure.
    /// `loop_header_pc` is the parent loop's back-edge target.
    pub fn start_bridge_tracing(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        state: &mut S,
        _env: &S::Env,
        _resume_pc: usize,
        loop_header_pc: usize,
    ) -> bool {
        // Use the compiled loop's meta so the bridge's FINISH args
        // match the parent loop's inputargs exactly.
        let Some(loop_meta) = self.meta.get_compiled_meta(green_key).cloned() else {
            return false;
        };

        // Verify the current state is compatible with the loop's meta.
        // If guard failure changed storage depths, bridge tracing cannot
        // produce FINISH args matching the parent loop — skip.
        if !state.is_compatible(&loop_meta) {
            return false;
        }

        let live_values = state.extract_live(&loop_meta);

        if !self.meta.start_retrace(green_key, fail_index, &live_values) {
            return false;
        }

        self.sym = Some(S::create_sym(&loop_meta, loop_header_pc));
        self.trace_meta = Some(loop_meta);
        self.bridge_info = Some((green_key, trace_id, fail_index));
        true
    }

    /// Generic back-edge runner for storage-pool interpreters.
    ///
    /// Encapsulates the common pattern: check tracing state, hash green key,
    /// try compiled execution, and handle guard failure / bridge compilation.
    ///
    /// The caller provides:
    /// - `green_values`: green key as i64 slice (e.g., `[target_pc, selected]`)
    /// - `target_pc`: the back-edge target PC
    /// - `pre_run`: callback to execute before compiled code (e.g., flush I/O)
    /// - `on_guard_failure`: callback to restore interpreter state from guard
    ///   failure values; returns `Some(resume_pc)` on success
    ///
    /// Returns `Some(resume_pc)` if compiled code ran or guard state was restored.
    pub fn run_back_edge_generic(
        &mut self,
        green_values: &[i64],
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
        on_guard_failure: impl FnOnce(&mut S, &S::Meta, &[i64]) -> Option<usize>,
    ) -> Option<usize> {
        if self.is_tracing() || !state.can_trace() {
            return None;
        }

        let key_hash = crate::green_key_hash(green_values);

        // Skip first execution after compilation to avoid re-entering
        // the loop we just compiled in the same back-edge iteration.
        // RPython equivalent: warmstate.py compiled_new flag.
        if self.just_compiled_key == Some(key_hash) {
            self.just_compiled_key = None;
            return None;
        }

        if !self.has_compiled_loop(key_hash) {
            // Fast path: if this key is already marked DONT_TRACE_HERE, skip
            // the expensive GreenKey allocation and tracing attempt.
            if self.meta.warm_state_ref().get_cell_state(key_hash)
                == majit_trace::warmstate::BaseJitCellState::DontTraceHere
            {
                return None;
            }
            let green_key = GreenKey::new(green_values.to_vec());
            let ran = self.back_edge_structured(green_key, target_pc, state, env, pre_run);
            // If tracing just finished and compiled a loop, mark it so the
            // next back_edge_generic call skips the first execution.
            if self.has_compiled_loop(key_hash) {
                self.just_compiled_key = Some(key_hash);
            }
            return ran.then_some(target_pc);
        }

        let meta = self.meta.get_compiled_meta(key_hash)?;
        if !state.is_compatible(meta) {
            return None;
        }
        let meta = meta.clone();
        let live_values = state.extract_live_values(&meta);
        pre_run();

        let result = self
            .meta
            .run_compiled_raw_detailed_with_values(key_hash, &live_values)?;
        let is_finish = result.is_finish;
        let fail_index = result.fail_index;
        let trace_id = result.trace_id;
        let result_meta = result.meta.clone();
        let typed_values = result.typed_values;
        let raw_values = result.values;

        if is_finish || fail_index == u32::MAX {
            state.restore_values(&result_meta, &typed_values);
            return Some(target_pc);
        }

        let should_bridge = self
            .meta
            .should_compile_bridge_in_trace(key_hash, trace_id, fail_index);
        if should_bridge {
            if let Some(resume_pc) = on_guard_failure(state, &result_meta, &raw_values) {
                self.start_bridge_tracing(
                    key_hash, trace_id, fail_index, state, env, resume_pc, target_pc,
                );
                return Some(resume_pc);
            }
            return None;
        }

        on_guard_failure(state, &result_meta, &raw_values)
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
            crate::pyjitpl::DriverRunOutcome::Jump { via_blackhole } => {
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
            crate::pyjitpl::DriverRunOutcome::Jump { via_blackhole } => {
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

    #[test]
    fn test_set_param_threshold() {
        let mut driver = JitDriver::<TypedRestoreState>::new(10);
        // Initially threshold is 10 — not hot after 2 ticks.
        let key = 100u64;
        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::Interpret
        ));

        // Lower threshold to 1 via set_param — next tick should start tracing.
        driver.set_param("threshold", 1);
        let key2 = 200u64;
        assert!(matches!(
            driver.meta.on_back_edge(key2, &[]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key2, &[]),
            BackEdgeAction::StartedTracing
        ));
    }

    #[test]
    fn test_get_stats_after_compile() {
        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        let key = 300u64;

        // Stats should be zero initially.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 0);
        assert_eq!(stats.loops_aborted, 0);

        // Warm up and start tracing.
        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::StartedTracing
        ));

        // Record minimal trace and compile.
        {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let _val = ctx.const_int(42);
        }
        driver.meta.close_and_compile(&[], ());
        assert!(driver.has_compiled_loop(key));

        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 1);
        assert_eq!(stats.loops_aborted, 0);
        assert_eq!(stats.bridges_compiled, 0);
    }

    #[test]
    fn test_set_param_unknown_ignored() {
        let mut driver = JitDriver::<TypedRestoreState>::new(5);
        // Unknown params should not panic or cause any side effects.
        driver.set_param("nonexistent_param", 999);
        driver.set_param("", 0);
        driver.set_param("enable_opts", 1);

        // Driver should still work normally after unknown params.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 0);
    }

    // ── Event-driven JitHookInterface parity tests ──
    // These exercise the full compilation pipeline (tracing → optimize → Cranelift)
    // and verify hooks fire with correct metadata, unlike wiring-only tests that
    // call hook closures directly.

    #[test]
    fn test_hook_on_compile_fires_through_real_pipeline() {
        use std::sync::{Arc, Mutex};

        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        let compile_events: Arc<Mutex<Vec<(u64, usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let events = compile_events.clone();
        driver.set_on_compile_loop(move |green_key, ops_before, ops_after| {
            events
                .lock()
                .unwrap()
                .push((green_key, ops_before, ops_after));
        });

        let key = 42u64;

        // Warm up: first back_edge increments counter, second starts tracing.
        // Pass one live value so OpRef(0) = i0 is available.
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::StartedTracing
        ));

        // Record a real trace: IntAdd + GuardFalse + JUMP
        {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef(0); // input arg from on_back_edge
            let c1 = ctx.const_int(1);
            let sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
            let cond = ctx.const_int(0);
            ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[cond], 0, &[sum]);
        }

        // Close and compile through the real Cranelift pipeline.
        driver.meta.close_and_compile(&[OpRef(0)], ());
        assert!(driver.has_compiled_loop(key));

        let events = compile_events.lock().unwrap();
        assert_eq!(events.len(), 1, "on_compile_loop should fire exactly once");
        assert_eq!(events[0].0, key, "green_key should match");
        assert!(events[0].1 > 0, "num_ops_before should be positive");
        assert!(events[0].2 > 0, "num_ops_after should be positive");
    }

    #[test]
    fn test_hook_get_stats_matches_real_compile_count() {
        let mut driver = JitDriver::<TypedRestoreState>::new(1);

        // Initially zero.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 0);
        assert_eq!(stats.bridges_compiled, 0);

        // Compile two distinct loops.
        for key in [100u64, 200u64] {
            assert!(matches!(
                driver.meta.on_back_edge(key, &[0]),
                BackEdgeAction::Interpret
            ));
            assert!(matches!(
                driver.meta.on_back_edge(key, &[0]),
                BackEdgeAction::StartedTracing
            ));
            {
                let ctx = driver.meta.trace_ctx().expect("should be tracing");
                let i0 = OpRef(0);
                let c1 = ctx.const_int(1);
                let _sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
            }
            driver.meta.close_and_compile(&[OpRef(0)], ());
            assert!(driver.has_compiled_loop(key));
        }

        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 2);
        assert_eq!(stats.loops_aborted, 0);
    }

    #[test]
    fn test_hook_on_compile_bridge_fires_through_real_pipeline() {
        use std::sync::{Arc, Mutex};

        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        let bridge_events: Arc<Mutex<Vec<(u64, u32, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let ev = bridge_events.clone();
        driver.meta.set_on_compile_bridge(move |gk, fi, nops| {
            ev.lock().unwrap().push((gk, fi, nops));
        });

        let key = 50u64;

        // Step 1: Compile a loop with a guard on a non-constant condition.
        // The guard must survive optimization, so use an input arg as condition.
        assert!(matches!(
            driver.meta.on_back_edge(key, &[1]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[1]),
            BackEdgeAction::StartedTracing
        ));
        {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef(0); // input arg (non-constant = won't be folded)
            let c1 = ctx.const_int(1);
            let sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
            // Guard on the input arg itself — optimizer cannot prove it's true,
            // so the guard survives optimization.
            ctx.record_guard_with_fail_args(OpCode::GuardTrue, &[i0], 0, &[sum]);
        }
        driver.meta.close_and_compile(&[OpRef(0)], ());
        assert!(driver.has_compiled_loop(key));

        // Identify the fail_index assigned to the first guard in the optimized trace.
        // build_guard_metadata assigns fail_index sequentially starting from 0.
        let fail_index = 0u32;

        // Step 2: Start bridge tracing via start_retrace (simulates guard failure path).
        assert!(driver.meta.start_retrace(key, fail_index, &[0]));

        // Step 3: Record a bridge trace and compile it via close_bridge_with_finish.
        {
            let ctx = driver.meta.trace_ctx().expect("should be tracing bridge");
            let i0 = OpRef(0); // bridge input from start_retrace
            let c2 = ctx.const_int(2);
            let _sum = ctx.record_op(OpCode::IntAdd, &[i0, c2]);
        }
        let trace_id = 0u64; // will be normalized to root_trace_id
        let compiled = driver
            .meta
            .close_bridge_with_finish(key, trace_id, fail_index, &[OpRef(0)]);
        assert!(compiled, "bridge should compile successfully");

        let events = bridge_events.lock().unwrap();
        assert_eq!(
            events.len(),
            1,
            "on_compile_bridge should fire exactly once"
        );
        assert_eq!(events[0].0, key, "bridge green_key should match");
        assert_eq!(events[0].1, fail_index, "bridge fail_index should match");
        assert!(events[0].2 > 0, "bridge num_ops should be positive");

        // Stats should reflect 1 loop + 1 bridge.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 1);
        assert_eq!(stats.bridges_compiled, 1);
    }

    #[test]
    fn test_hook_on_compile_error_fires_on_real_failure() {
        use std::sync::{Arc, Mutex};

        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        let error_events: Arc<Mutex<Vec<(u64, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let ev = error_events.clone();
        driver.meta.set_on_compile_error(move |gk, msg| {
            ev.lock().unwrap().push((gk, msg.to_string()));
        });

        let key = 99u64;

        // Warm up and start tracing.
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::StartedTracing
        ));

        // Record a trace with a guard whose fail_args include OpRef::NONE.
        // The Cranelift backend rejects this with BackendError::Unsupported,
        // which triggers the on_compile_error hook through the real pipeline.
        {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef(0);
            // Guard on input arg so optimizer cannot fold it away.
            ctx.record_guard_with_fail_args(
                OpCode::GuardTrue,
                &[i0],
                0,
                &[OpRef::NONE], // invalid: causes BackendError
            );
        }
        driver.meta.close_and_compile(&[OpRef(0)], ());

        // The loop should NOT have been compiled (error path).
        assert!(!driver.has_compiled_loop(key));

        let events = error_events.lock().unwrap();
        // The error hook should have fired.
        assert_eq!(
            events.len(),
            1,
            "on_compile_error should fire exactly once on compilation failure"
        );
        assert_eq!(events[0].0, key, "error green_key should match");
        assert!(!events[0].1.is_empty(), "error message should be non-empty");

        // Stats should reflect an aborted loop.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 0);
        assert_eq!(stats.loops_aborted, 1);
    }

    // ── Multi-entry point lifecycle tests ──
    // Parity with warmspot.py multi-driver entry semantics: multiple functions
    // can share the same JitDriver and compiled loops via register_entry_point.

    #[test]
    fn test_multi_entry_point_registration() {
        let mut driver = JitDriver::<TypedRestoreState>::new(1);

        // Initially no entry points.
        assert!(driver.entry_points().is_empty());

        // Register two entry points with different green key schemas.
        driver.register_entry_point("loop_main", &[Type::Int]);
        driver.register_entry_point("loop_helper", &[Type::Int, Type::Ref]);

        assert_eq!(driver.entry_points().len(), 2);

        // Verify first entry point.
        let ep0 = driver
            .find_entry_point("loop_main")
            .expect("should find loop_main");
        assert_eq!(ep0.name, "loop_main");
        assert_eq!(ep0.schema, vec![Type::Int]);

        // Verify second entry point.
        let ep1 = driver
            .find_entry_point("loop_helper")
            .expect("should find loop_helper");
        assert_eq!(ep1.name, "loop_helper");
        assert_eq!(ep1.schema, vec![Type::Int, Type::Ref]);

        // Non-existent entry point returns None.
        assert!(driver.find_entry_point("nonexistent").is_none());
    }

    #[test]
    fn test_multi_entry_points_share_compiled_loops() {
        // Two entry points register different functions on the same driver.
        // A loop compiled from one entry point is visible to back_edge calls
        // from any entry point, since they share the same MetaInterp and
        // compiled loop table (keyed by green_key, not by entry point name).
        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        driver.register_entry_point("func_a", &[Type::Int]);
        driver.register_entry_point("func_b", &[Type::Int]);

        // Both entry points are registered.
        assert_eq!(driver.entry_points().len(), 2);

        // Compile a loop from "func_a"'s perspective using green_key=10.
        let key = 10u64;
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::StartedTracing
        ));

        {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef(0);
            let c1 = ctx.const_int(1);
            let sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
            let cond = ctx.const_int(0);
            ctx.record_guard_with_fail_args(OpCode::GuardFalse, &[cond], 0, &[sum]);
        }
        driver.meta.close_and_compile(&[OpRef(0)], ());
        assert!(driver.has_compiled_loop(key));

        // "func_b" can see the compiled loop via the same green_key.
        // The compiled loop is shared — it doesn't matter which entry point
        // triggered the compilation.
        assert!(
            driver.has_compiled_loop(key),
            "compiled loop should be visible from any entry point"
        );

        // A different green key from "func_b" does not see the loop.
        let other_key = 20u64;
        assert!(
            !driver.has_compiled_loop(other_key),
            "different green_key should not match"
        );

        // The driver stats reflect exactly one compiled loop.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 1);
    }
}
