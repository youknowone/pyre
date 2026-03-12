use majit_ir::{DescrRef, GreenKey, JitDriverVar, OpCode, OpRef, Type, VarKind};
use majit_trace::recorder::TraceRecorder;

use crate::call_descr::make_call_descr;
use crate::constant_pool::ConstantPool;
use crate::fail_descr::make_fail_descr;
use crate::symbolic_stack::SymbolicStack;
use crate::TraceAction;

/// Descriptor for a JitDriver's variable layout.
///
/// Mirrors RPython's `JitDriver(greens=[...], reds=[...])`:
/// - `greens` are compile-time constants identifying the loop header
/// - `reds` are runtime values carried as InputArgs
///
/// The interpreter declares this once per JitDriver and passes it to
/// MetaInterp for structured green/red handling.
#[derive(Clone, Debug)]
pub struct JitDriverDescriptor {
    /// All variables in declaration order.
    pub vars: Vec<JitDriverVar>,
    /// Optional name of the virtualizable red variable.
    pub virtualizable: Option<String>,
}

impl JitDriverDescriptor {
    /// Create a descriptor from green and red variable lists.
    pub fn new(greens: Vec<(&str, Type)>, reds: Vec<(&str, Type)>) -> Self {
        Self::with_virtualizable(greens, reds, None)
    }

    /// Create a descriptor with optional virtualizable metadata.
    pub fn with_virtualizable(
        greens: Vec<(&str, Type)>,
        reds: Vec<(&str, Type)>,
        virtualizable: Option<&str>,
    ) -> Self {
        let mut vars = Vec::new();
        for (name, tp) in greens {
            vars.push(JitDriverVar::green(name, tp));
        }
        for (name, tp) in reds {
            vars.push(JitDriverVar::red(name, tp));
        }
        JitDriverDescriptor {
            vars,
            virtualizable: virtualizable.map(str::to_string),
        }
    }

    /// Get only the green variables.
    pub fn greens(&self) -> Vec<&JitDriverVar> {
        self.vars
            .iter()
            .filter(|v| v.kind == VarKind::Green)
            .collect()
    }

    /// Get only the red variables.
    pub fn reds(&self) -> Vec<&JitDriverVar> {
        self.vars
            .iter()
            .filter(|v| v.kind == VarKind::Red)
            .collect()
    }

    /// Number of green variables.
    pub fn num_greens(&self) -> usize {
        self.vars
            .iter()
            .filter(|v| v.kind == VarKind::Green)
            .count()
    }

    /// Number of red variables.
    pub fn num_reds(&self) -> usize {
        self.vars.iter().filter(|v| v.kind == VarKind::Red).count()
    }

    /// Get the virtualizable variable, if any.
    pub fn virtualizable(&self) -> Option<&JitDriverVar> {
        let name = self.virtualizable.as_deref()?;
        self.vars.iter().find(|var| var.name == name)
    }
}

/// Trait implemented by declarative `#[jit_driver]` marker types.
///
/// This provides a stable seam between proc-macro-generated driver metadata
/// and the runtime `JitDriver` orchestration layer.
pub trait DeclarativeJitDriver {
    const GREENS: &'static [&'static str];
    const REDS: &'static [&'static str];
    const NUM_VARS: usize;
    const NUM_GREENS: usize;
    const NUM_REDS: usize;
    const VIRTUALIZABLE: Option<&'static str>;

    fn descriptor(
        green_types: &[Type],
        red_types: &[Type],
    ) -> Result<JitDriverDescriptor, &'static str>;

    fn green_key(values: &[i64]) -> Result<GreenKey, &'static str>;
}

/// Tracing context: wraps TraceRecorder + ConstantPool with convenience API.
///
/// The interpreter uses this during trace recording to:
/// - Record IR operations
/// - Manage constants (with deduplication)
/// - Record guards (with auto-generated FailDescr)
/// - Record function calls (with auto-generated CallDescr)
pub struct TraceCtx {
    pub(crate) recorder: TraceRecorder,
    pub(crate) green_key: u64,
    pub(crate) constants: ConstantPool,
    /// Stack of inlined function frames (callee green_keys).
    inline_frames: Vec<u64>,
    /// Structured green key values (if provided by the interpreter).
    green_key_values: Option<GreenKey>,
    /// Declarative driver layout metadata, if provided by the interpreter.
    driver_descriptor: Option<JitDriverDescriptor>,
}

impl TraceCtx {
    pub(crate) fn new(recorder: TraceRecorder, green_key: u64) -> Self {
        TraceCtx {
            recorder,
            green_key,
            constants: ConstantPool::new(),
            inline_frames: Vec::new(),
            green_key_values: None,
            driver_descriptor: None,
        }
    }

    /// Create a TraceCtx with a structured green key.
    pub(crate) fn with_green_key(
        recorder: TraceRecorder,
        green_key: u64,
        green_key_values: GreenKey,
    ) -> Self {
        TraceCtx {
            recorder,
            green_key,
            constants: ConstantPool::new(),
            inline_frames: Vec::new(),
            green_key_values: Some(green_key_values),
            driver_descriptor: None,
        }
    }

    /// Get the current inlining depth.
    pub fn inline_depth(&self) -> usize {
        self.inline_frames.len()
    }

    /// Push an inline frame (entering a callee).
    /// Returns false if the max inline depth has been exceeded.
    pub(crate) fn push_inline_frame(&mut self, callee_key: u64, max_depth: u32) -> bool {
        if (self.inline_frames.len() as u32) >= max_depth {
            return false;
        }
        self.inline_frames.push(callee_key);
        true
    }

    /// Pop an inline frame (returning from a callee).
    pub(crate) fn pop_inline_frame(&mut self) {
        self.inline_frames.pop();
    }

    /// Get or create a constant OpRef for a given i64 value.
    pub fn const_int(&mut self, value: i64) -> OpRef {
        self.constants.get_or_insert(value)
    }

    /// Record a regular IR operation.
    pub fn record_op(&mut self, opcode: OpCode, args: &[OpRef]) -> OpRef {
        self.recorder.record_op(opcode, args)
    }

    /// Record an operation with a descriptor (e.g., calls).
    pub fn record_op_with_descr(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
    ) -> OpRef {
        self.recorder.record_op_with_descr(opcode, args, descr)
    }

    /// Record a guard with auto-generated FailDescr.
    ///
    /// `num_live` is the number of live integer values (for the FailDescr).
    pub fn record_guard(&mut self, opcode: OpCode, args: &[OpRef], num_live: usize) -> OpRef {
        let descr = make_fail_descr(num_live);
        self.recorder.record_guard(opcode, args, descr)
    }

    /// Record a guard with explicit fail_args.
    pub fn record_guard_with_fail_args(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        num_live: usize,
        fail_args: &[OpRef],
    ) -> OpRef {
        let descr = make_fail_descr(num_live);
        self.recorder
            .record_guard_with_fail_args(opcode, args, descr, fail_args)
    }

    /// Record a void-returning function call (CallN).
    ///
    /// Automatically registers the function pointer as a constant and
    /// creates a CallDescr. The interpreter doesn't need to manage
    /// function pointer constants or CallDescr implementations.
    pub fn call_void(&mut self, func_ptr: *const (), args: &[OpRef]) {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Void);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallN, &call_args, descr);
    }

    /// Record an integer-returning function call (CallI).
    ///
    /// Same convenience as `call_void` but returns an OpRef for the result.
    pub fn call_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Int);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallI, &call_args, descr)
    }

    /// Whether the trace has exceeded the maximum allowed length.
    pub fn is_too_long(&self) -> bool {
        self.recorder.is_too_long()
    }

    /// The green key hash (loop header PC) for this trace.
    pub fn green_key(&self) -> u64 {
        self.green_key
    }

    /// The structured green key values, if provided.
    pub fn green_key_values(&self) -> Option<&GreenKey> {
        self.green_key_values.as_ref()
    }

    /// Set the structured green key values.
    pub fn set_green_key_values(&mut self, values: GreenKey) {
        self.green_key_values = Some(values);
    }

    /// The declarative JitDriver descriptor, if provided.
    pub fn driver_descriptor(&self) -> Option<&JitDriverDescriptor> {
        self.driver_descriptor.as_ref()
    }

    /// Attach declarative JitDriver metadata to the active trace.
    pub fn set_driver_descriptor(&mut self, descriptor: JitDriverDescriptor) {
        self.driver_descriptor = Some(descriptor);
    }

    /// Record a promote: emit GuardValue to specialize on a runtime value.
    ///
    /// In RPython this is `jit.promote(x)` — it records a `GUARD_VALUE`
    /// that asserts the runtime value equals the constant captured during
    /// tracing. After the guard, the optimizer treats the value as constant.
    ///
    /// `opref` is the traced value, `runtime_value` is the current concrete
    /// value seen at trace time.
    pub fn promote_int(&mut self, opref: OpRef, runtime_value: i64, num_live: usize) -> OpRef {
        let const_ref = self.const_int(runtime_value);
        self.record_guard(OpCode::GuardValue, &[opref, const_ref], num_live);
        const_ref
    }

    /// Record a ref-typed promote (GUARD_VALUE for GC references).
    pub fn promote_ref(&mut self, opref: OpRef, runtime_value: i64, num_live: usize) -> OpRef {
        let const_ref = self.const_int(runtime_value);
        self.record_guard(OpCode::GuardValue, &[opref, const_ref], num_live);
        const_ref
    }

    /// Record a call to an elidable (pure) function.
    ///
    /// In RPython, `@jit.elidable` marks a function whose result depends
    /// only on its arguments and has no side effects. The optimizer can
    /// constant-fold calls where all args are constants, or CSE identical calls.
    ///
    /// This records a CALL_PURE_I (or CALL_PURE_R/CALL_PURE_N) which the
    /// optimizer's pure pass can eliminate.
    pub fn call_elidable_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Int);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallPureI, &call_args, descr)
    }

    /// Record a void-returning call to a may-force function (e.g., one that
    /// may trigger GC or exceptions).
    ///
    /// In RPython this is `call_may_force` — a call that may force virtualizable
    /// frames or raise exceptions. Must be followed by `GUARD_NOT_FORCED`.
    pub fn call_may_force_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Int);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallMayForceI, &call_args, descr)
    }

    /// Record a virtualizable field read (GETFIELD_GC_I/R/F).
    ///
    /// During tracing, reading a virtualizable field is recorded as a
    /// GETFIELD_GC operation. The optimizer's virtualize pass may later
    /// eliminate this operation if the field is virtual.
    pub fn vable_getfield_int(&mut self, vable_opref: OpRef, field_offset: usize) -> OpRef {
        let offset_ref = self.const_int(field_offset as i64);
        self.record_op(OpCode::GetfieldGcI, &[vable_opref, offset_ref])
    }

    /// Record a virtualizable field write (SETFIELD_GC).
    pub fn vable_setfield(&mut self, vable_opref: OpRef, field_offset: usize, value: OpRef) {
        let offset_ref = self.const_int(field_offset as i64);
        self.record_op(OpCode::SetfieldGc, &[vable_opref, offset_ref, value]);
    }

    /// Record a virtualizable ref field read (GETFIELD_GC_R).
    pub fn vable_getfield_ref(&mut self, vable_opref: OpRef, field_offset: usize) -> OpRef {
        let offset_ref = self.const_int(field_offset as i64);
        self.record_op(OpCode::GetfieldGcR, &[vable_opref, offset_ref])
    }

    /// Record a virtualizable float field read (GETFIELD_GC_F).
    pub fn vable_getfield_float(&mut self, vable_opref: OpRef, field_offset: usize) -> OpRef {
        let offset_ref = self.const_int(field_offset as i64);
        self.record_op(OpCode::GetfieldGcF, &[vable_opref, offset_ref])
    }

    /// Record a virtualizable array item read (GETARRAYITEM_GC_I).
    pub fn vable_getarrayitem_int(&mut self, array_opref: OpRef, index: OpRef) -> OpRef {
        let zero = self.const_int(0); // descr placeholder
        self.record_op(OpCode::GetarrayitemGcI, &[array_opref, index, zero])
    }

    /// Record a virtualizable array item read (GETARRAYITEM_GC_R).
    pub fn vable_getarrayitem_ref(&mut self, array_opref: OpRef, index: OpRef) -> OpRef {
        let zero = self.const_int(0); // descr placeholder
        self.record_op(OpCode::GetarrayitemGcR, &[array_opref, index, zero])
    }

    /// Record a virtualizable array item read (GETARRAYITEM_GC_F).
    pub fn vable_getarrayitem_float(&mut self, array_opref: OpRef, index: OpRef) -> OpRef {
        let zero = self.const_int(0); // descr placeholder
        self.record_op(OpCode::GetarrayitemGcF, &[array_opref, index, zero])
    }

    /// Record a virtualizable array item write (SETARRAYITEM_GC).
    pub fn vable_setarrayitem(&mut self, array_opref: OpRef, index: OpRef, value: OpRef) {
        let zero = self.const_int(0); // descr placeholder
        self.record_op(OpCode::SetarrayitemGc, &[array_opref, index, value, zero]);
    }

    /// Record a ref-returning call to a may-force function.
    pub fn call_may_force_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Ref);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallMayForceR, &call_args, descr)
    }

    /// Record a void-returning call to a may-force function.
    pub fn call_may_force_void(&mut self, func_ptr: *const (), args: &[OpRef]) {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Void);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallMayForceN, &call_args, descr);
    }

    /// Record a call with GIL release (for C extensions / external libs).
    ///
    /// In RPython this is `call_release_gil`. The GIL is released before the
    /// call and reacquired after. Used for long-running C functions.
    pub fn call_release_gil_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Int);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallReleaseGilI, &call_args, descr)
    }

    /// Record a call to a loop-invariant function.
    ///
    /// The result is cached for the duration of one loop iteration.
    /// In RPython, `@jit.loop_invariant` marks such functions.
    pub fn call_loopinvariant_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Int);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallLoopinvariantI, &call_args, descr)
    }

    /// Record GUARD_NOT_FORCED (must follow a call_may_force).
    pub fn guard_not_forced(&mut self, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardNotForced, &[], num_live)
    }

    /// Record GUARD_NO_EXCEPTION (check no pending exception).
    pub fn guard_no_exception(&mut self, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardNoException, &[], num_live)
    }

    /// Record GUARD_NOT_INVALIDATED (check loop not invalidated).
    pub fn guard_not_invalidated(&mut self, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardNotInvalidated, &[], num_live)
    }

    // ── Generic typed call ──────────────────────────────────────────

    /// Record a function call with explicit argument and return types.
    ///
    /// All type-specific call convenience methods delegate to this.
    /// `opcode` selects the call family (CallI/R/F/N, CallPureI/R/F/N, etc.).
    pub fn call_typed(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr = make_call_descr(arg_types, ret_type);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(opcode, &call_args, descr)
    }

    // ── Ref/Float call variants ─────────────────────────────────────

    /// Record a ref-returning function call (CallR).
    pub fn call_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Ref);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallR, &call_args, descr)
    }

    /// Record a float-returning function call (CallF).
    pub fn call_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Float);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallF, &call_args, descr)
    }

    /// Record a ref-returning elidable (pure) call (CallPureR).
    pub fn call_elidable_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Ref);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallPureR, &call_args, descr)
    }

    /// Record a float-returning elidable (pure) call (CallPureF).
    pub fn call_elidable_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Float);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallPureF, &call_args, descr)
    }

    /// Record a float-returning may-force call (CallMayForceF).
    pub fn call_may_force_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Float);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallMayForceF, &call_args, descr)
    }

    /// Record a void-returning GIL-release call (CallReleaseGilN).
    pub fn call_release_gil_void(&mut self, func_ptr: *const (), args: &[OpRef]) {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Void);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallReleaseGilN, &call_args, descr);
    }

    /// Record a ref-returning GIL-release call (CallReleaseGilR).
    pub fn call_release_gil_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Ref);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallReleaseGilR, &call_args, descr)
    }

    /// Record a float-returning GIL-release call (CallReleaseGilF).
    pub fn call_release_gil_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Float);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallReleaseGilF, &call_args, descr)
    }

    /// Record a ref-returning loop-invariant call (CallLoopinvariantR).
    pub fn call_loopinvariant_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Ref);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallLoopinvariantR, &call_args, descr)
    }

    /// Record a float-returning loop-invariant call (CallLoopinvariantF).
    pub fn call_loopinvariant_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_descr(&arg_types, Type::Float);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CallLoopinvariantF, &call_args, descr)
    }

    // ── Exception handling ──────────────────────────────────────────

    /// Record GUARD_EXCEPTION: assert that the pending exception matches
    /// the given class, and produce a ref to the exception value.
    pub fn guard_exception(&mut self, exc_class: OpRef, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardException, &[exc_class], num_live)
    }

    /// Record SAVE_EXCEPTION: capture the pending exception value as a ref.
    pub fn save_exception(&mut self) -> OpRef {
        self.record_op(OpCode::SaveException, &[])
    }

    /// Record SAVE_EXC_CLASS: capture the pending exception's class as an int.
    pub fn save_exc_class(&mut self) -> OpRef {
        self.record_op(OpCode::SaveExcClass, &[])
    }

    /// Record RESTORE_EXCEPTION: restore exception state from saved
    /// class and value refs.
    pub fn restore_exception(&mut self, exc_class: OpRef, exc_value: OpRef) {
        self.record_op(OpCode::RestoreException, &[exc_class, exc_value]);
    }

    // ── Object allocation ───────────────────────────────────────────

    /// Record NEW: allocate a new object described by `descr`.
    pub fn record_new(&mut self, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::New, &[], descr)
    }

    /// Record NEW_WITH_VTABLE: allocate a new object with an explicit vtable pointer.
    pub fn record_new_with_vtable(&mut self, vtable: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::NewWithVtable, &[vtable], descr)
    }

    /// Record NEW_ARRAY: allocate a new array with the given length.
    pub fn record_new_array(&mut self, length: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::NewArray, &[length], descr)
    }

    /// Record NEW_ARRAY_CLEAR: allocate a zero-initialized array.
    pub fn record_new_array_clear(&mut self, length: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::NewArrayClear, &[length], descr)
    }

    // ── Overflow-checked arithmetic ────────────────────────────────

    /// Record overflow-checked integer add + GuardNoOverflow.
    ///
    /// Returns the result OpRef. On overflow at trace time, the caller
    /// should abort tracing.
    pub fn int_add_ovf(&mut self, lhs: OpRef, rhs: OpRef, num_live: usize) -> OpRef {
        let result = self.record_op(OpCode::IntAddOvf, &[lhs, rhs]);
        self.record_guard(OpCode::GuardNoOverflow, &[], num_live);
        result
    }

    /// Record overflow-checked integer sub + GuardNoOverflow.
    pub fn int_sub_ovf(&mut self, lhs: OpRef, rhs: OpRef, num_live: usize) -> OpRef {
        let result = self.record_op(OpCode::IntSubOvf, &[lhs, rhs]);
        self.record_guard(OpCode::GuardNoOverflow, &[], num_live);
        result
    }

    /// Record overflow-checked integer mul + GuardNoOverflow.
    pub fn int_mul_ovf(&mut self, lhs: OpRef, rhs: OpRef, num_live: usize) -> OpRef {
        let result = self.record_op(OpCode::IntMulOvf, &[lhs, rhs]);
        self.record_guard(OpCode::GuardNoOverflow, &[], num_live);
        result
    }

    // ── String operations ───────────────────────────────────────────

    /// Record NEWSTR: allocate a new string with given length.
    pub fn newstr(&mut self, length: OpRef) -> OpRef {
        self.record_op(OpCode::Newstr, &[length])
    }

    /// Record STRLEN: get string length.
    pub fn strlen(&mut self, string: OpRef) -> OpRef {
        self.record_op(OpCode::Strlen, &[string])
    }

    /// Record STRGETITEM: read character at index.
    pub fn strgetitem(&mut self, string: OpRef, index: OpRef) -> OpRef {
        self.record_op(OpCode::Strgetitem, &[string, index])
    }

    /// Record STRSETITEM: write character at index.
    pub fn strsetitem(&mut self, string: OpRef, index: OpRef, value: OpRef) {
        self.record_op(OpCode::Strsetitem, &[string, index, value]);
    }

    /// Record COPYSTRCONTENT: copy characters between strings.
    pub fn copystrcontent(
        &mut self,
        src: OpRef,
        dst: OpRef,
        src_start: OpRef,
        dst_start: OpRef,
        length: OpRef,
    ) {
        self.record_op(
            OpCode::Copystrcontent,
            &[src, dst, src_start, dst_start, length],
        );
    }

    /// Record STRHASH: compute string hash.
    pub fn strhash(&mut self, string: OpRef) -> OpRef {
        self.record_op(OpCode::Strhash, &[string])
    }

    // ── Convenience methods for common trace patterns ───────────────

    /// Pop two operands, record a binary operation, push the result.
    ///
    /// Handles the common stack pattern: `a, b → op(a, b)`.
    /// Note: pops in stack order (top first), but passes to IR as `[second, first]`
    /// so that the left operand comes first.
    pub fn trace_binop(&mut self, stack: &mut SymbolicStack, opcode: OpCode) {
        let r1 = stack.pop().unwrap();
        let r2 = stack.pop().unwrap();
        let result = self.record_op(opcode, &[r2, r1]);
        stack.push(result);
    }

    /// Push a constant integer value onto the symbolic stack.
    pub fn trace_push_const(&mut self, stack: &mut SymbolicStack, value: i64) {
        let opref = self.const_int(value);
        stack.push(opref);
    }

    /// Pop one value and call a void function with it.
    ///
    /// Common pattern for output operations (e.g., POPNUM, POPCHAR).
    pub fn trace_call_void_1(&mut self, stack: &mut SymbolicStack, func_ptr: *const ()) {
        let value = stack.pop().unwrap();
        self.call_void(func_ptr, &[value]);
    }

    /// Pop a boolean-like stack value, record the matching guard, and return
    /// whether tracing should continue or close the loop.
    ///
    /// `branch_taken` is the runtime branch result, while `taken_when_true`
    /// describes whether the interpreter takes the branch on a non-zero value.
    pub fn trace_branch_guard(
        &mut self,
        stack: &mut SymbolicStack,
        branch_taken: bool,
        taken_when_true: bool,
        num_live: usize,
        close_loop_on_taken: bool,
    ) -> TraceAction {
        let cond = stack.pop().unwrap();
        let opcode = if branch_taken == taken_when_true {
            OpCode::GuardTrue
        } else {
            OpCode::GuardFalse
        };
        self.record_guard(opcode, &[cond], num_live);
        if branch_taken && close_loop_on_taken {
            TraceAction::CloseLoop
        } else {
            TraceAction::Continue
        }
    }
}
