use majit_ir::{DescrRef, GreenKey, JitDriverVar, OpCode, OpRef, Type, VarKind};
use majit_trace::recorder::TraceRecorder;

use majit_codegen::LoopToken;

use crate::call_descr::{make_call_assembler_descr, make_call_descr};
use crate::constant_pool::ConstantPool;
use crate::fail_descr::{make_fail_descr, make_fail_descr_typed};
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

/// A single virtualizable field to synchronize before/after a residual call.
///
/// `field_descr_idx` identifies the field descriptor, `value` is the current
/// symbolic value (OpRef) that should be written to the heap before the call.
/// `field_type` determines which GETFIELD_GC variant to use when re-reading.
#[derive(Debug, Clone, Copy)]
pub struct VableSyncField {
    /// Descriptor index identifying the virtualizable field.
    pub field_descr_idx: u32,
    /// Current symbolic value to write to heap before the call.
    pub value: OpRef,
    /// Type of the field (Int, Ref, Float).
    pub field_type: Type,
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
    /// Create a standalone TraceCtx for testing or external use.
    pub fn for_test(num_inputs: usize) -> Self {
        let mut recorder = TraceRecorder::new();
        for _ in 0..num_inputs {
            recorder.record_input_arg(majit_ir::Type::Int);
        }
        Self::new(recorder, 0)
    }

    /// Take the recorder out of this context (consumes self).
    pub fn into_recorder(self) -> TraceRecorder {
        self.recorder
    }

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

    /// Record a guard with explicit typed fail_args.
    pub fn record_guard_typed_with_fail_args(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        fail_arg_types: Vec<Type>,
        fail_args: &[OpRef],
    ) -> OpRef {
        let descr = make_fail_descr_typed(fail_arg_types);
        self.recorder
            .record_guard_with_fail_args(opcode, args, descr, fail_args)
    }

    /// Record a void-returning function call (CallN).
    ///
    /// Automatically registers the function pointer as a constant and
    /// creates a CallDescr. The interpreter doesn't need to manage
    /// function pointer constants or CallDescr implementations.
    pub fn call_void(&mut self, func_ptr: *const (), args: &[OpRef]) {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_void_typed(func_ptr, args, &arg_types);
    }

    /// Record an integer-returning function call (CallI).
    ///
    /// Same convenience as `call_void` but returns an OpRef for the result.
    pub fn call_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_int_typed(func_ptr, args, &arg_types)
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
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_elidable_int_typed(func_ptr, args, &arg_types)
    }

    /// Record a void-returning call to a may-force function (e.g., one that
    /// may trigger GC or exceptions).
    ///
    /// In RPython this is `call_may_force` — a call that may force virtualizable
    /// frames or raise exceptions. Must be followed by `GUARD_NOT_FORCED`.
    pub fn call_may_force_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_may_force_int_typed(func_ptr, args, &arg_types)
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

    /// Record a virtualizable field read with an explicit field descriptor.
    pub fn vable_getfield_int_descr(&mut self, vable_opref: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::GetfieldGcI, &[vable_opref], descr)
    }

    /// Record a virtualizable field write (SETFIELD_GC).
    pub fn vable_setfield(&mut self, vable_opref: OpRef, field_offset: usize, value: OpRef) {
        let offset_ref = self.const_int(field_offset as i64);
        self.record_op(OpCode::SetfieldGc, &[vable_opref, offset_ref, value]);
    }

    /// Record a virtualizable field write with an explicit field descriptor.
    pub fn vable_setfield_descr(&mut self, vable_opref: OpRef, value: OpRef, descr: DescrRef) {
        self.record_op_with_descr(OpCode::SetfieldGc, &[vable_opref, value], descr);
    }

    /// Record a virtualizable ref field read (GETFIELD_GC_R).
    pub fn vable_getfield_ref(&mut self, vable_opref: OpRef, field_offset: usize) -> OpRef {
        let offset_ref = self.const_int(field_offset as i64);
        self.record_op(OpCode::GetfieldGcR, &[vable_opref, offset_ref])
    }

    /// Record a virtualizable ref field read with an explicit field descriptor.
    pub fn vable_getfield_ref_descr(&mut self, vable_opref: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], descr)
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

    /// Record a virtualizable array item read with an explicit array descriptor.
    pub fn vable_getarrayitem_int_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        descr: DescrRef,
    ) -> OpRef {
        self.record_op_with_descr(OpCode::GetarrayitemGcI, &[array_opref, index], descr)
    }

    /// Record a virtualizable array item read (GETARRAYITEM_GC_R).
    pub fn vable_getarrayitem_ref(&mut self, array_opref: OpRef, index: OpRef) -> OpRef {
        let zero = self.const_int(0); // descr placeholder
        self.record_op(OpCode::GetarrayitemGcR, &[array_opref, index, zero])
    }

    /// Record a virtualizable array item read with an explicit array descriptor.
    pub fn vable_getarrayitem_ref_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        descr: DescrRef,
    ) -> OpRef {
        self.record_op_with_descr(OpCode::GetarrayitemGcR, &[array_opref, index], descr)
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

    /// Record a virtualizable array item write with an explicit array descriptor.
    pub fn vable_setarrayitem_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        value: OpRef,
        descr: DescrRef,
    ) {
        self.record_op_with_descr(OpCode::SetarrayitemGc, &[array_opref, index, value], descr);
    }

    /// Record a ref-returning call to a may-force function.
    pub fn call_may_force_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_may_force_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a void-returning call to a may-force function.
    pub fn call_may_force_void(&mut self, func_ptr: *const (), args: &[OpRef]) {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_may_force_void_typed(func_ptr, args, &arg_types);
    }

    /// Record a call with GIL release (for C extensions / external libs).
    ///
    /// In RPython this is `call_release_gil`. The GIL is released before the
    /// call and reacquired after. Used for long-running C functions.
    pub fn call_release_gil_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_release_gil_int_typed(func_ptr, args, &arg_types)
    }

    /// Record a call to a loop-invariant function.
    ///
    /// The result is cached for the duration of one loop iteration.
    /// In RPython, `@jit.loop_invariant` marks such functions.
    pub fn call_loopinvariant_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_loopinvariant_int_typed(func_ptr, args, &arg_types)
    }

    /// Record GUARD_NOT_FORCED (must follow a call_may_force).
    pub fn guard_not_forced(&mut self, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardNotForced, &[], num_live)
    }

    // ── CALL_MAY_FORCE with virtualizable synchronization ─────────

    /// Emit SETFIELD_GC ops to flush virtualizable fields to heap,
    /// then a CALL_MAY_FORCE, then GETFIELD_GC ops to re-read them.
    ///
    /// This is the RPython `vable_and_vrefs_before_residual_call` /
    /// `vable_after_residual_call` pattern: virtualizable state is
    /// written to the heap before the callee can observe it, and
    /// re-read afterwards because the callee may have modified it.
    ///
    /// Returns `(call_result, updated_fields)` where `updated_fields`
    /// contains the new OpRefs for each virtualizable field after the
    /// call. The caller must use these new OpRefs for subsequent
    /// operations instead of the stale pre-call values.
    pub fn call_may_force_with_vable_sync_int(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        vable_ref: OpRef,
        sync_fields: &[VableSyncField],
        num_live: usize,
    ) -> (OpRef, Vec<(u32, OpRef)>) {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_may_force_with_vable_sync_int_typed(
            func_ptr,
            args,
            &arg_types,
            vable_ref,
            sync_fields,
            num_live,
        )
    }

    /// Typed variant of [`call_may_force_with_vable_sync_int`].
    pub fn call_may_force_with_vable_sync_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        vable_ref: OpRef,
        sync_fields: &[VableSyncField],
        num_live: usize,
    ) -> (OpRef, Vec<(u32, OpRef)>) {
        self.emit_vable_sync_before(vable_ref, sync_fields);
        let result = self.call_may_force_int_typed(func_ptr, args, arg_types);
        let updated = self.emit_vable_sync_after(vable_ref, sync_fields);
        self.guard_not_forced(num_live);
        (result, updated)
    }

    /// Ref-returning variant of call_may_force with virtualizable sync.
    pub fn call_may_force_with_vable_sync_ref(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        vable_ref: OpRef,
        sync_fields: &[VableSyncField],
        num_live: usize,
    ) -> (OpRef, Vec<(u32, OpRef)>) {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_may_force_with_vable_sync_ref_typed(
            func_ptr,
            args,
            &arg_types,
            vable_ref,
            sync_fields,
            num_live,
        )
    }

    /// Typed ref-returning variant.
    pub fn call_may_force_with_vable_sync_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        vable_ref: OpRef,
        sync_fields: &[VableSyncField],
        num_live: usize,
    ) -> (OpRef, Vec<(u32, OpRef)>) {
        self.emit_vable_sync_before(vable_ref, sync_fields);
        let result = self.call_may_force_ref_typed(func_ptr, args, arg_types);
        let updated = self.emit_vable_sync_after(vable_ref, sync_fields);
        self.guard_not_forced(num_live);
        (result, updated)
    }

    /// Void-returning variant of call_may_force with virtualizable sync.
    pub fn call_may_force_with_vable_sync_void(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        vable_ref: OpRef,
        sync_fields: &[VableSyncField],
        num_live: usize,
    ) -> Vec<(u32, OpRef)> {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_may_force_with_vable_sync_void_typed(
            func_ptr,
            args,
            &arg_types,
            vable_ref,
            sync_fields,
            num_live,
        )
    }

    /// Typed void-returning variant.
    pub fn call_may_force_with_vable_sync_void_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        vable_ref: OpRef,
        sync_fields: &[VableSyncField],
        num_live: usize,
    ) -> Vec<(u32, OpRef)> {
        self.emit_vable_sync_before(vable_ref, sync_fields);
        self.call_may_force_void_typed(func_ptr, args, arg_types);
        let updated = self.emit_vable_sync_after(vable_ref, sync_fields);
        self.guard_not_forced(num_live);
        updated
    }

    /// Float-returning variant of call_may_force with virtualizable sync.
    pub fn call_may_force_with_vable_sync_float(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        vable_ref: OpRef,
        sync_fields: &[VableSyncField],
        num_live: usize,
    ) -> (OpRef, Vec<(u32, OpRef)>) {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_may_force_with_vable_sync_float_typed(
            func_ptr,
            args,
            &arg_types,
            vable_ref,
            sync_fields,
            num_live,
        )
    }

    /// Typed float-returning variant.
    pub fn call_may_force_with_vable_sync_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        vable_ref: OpRef,
        sync_fields: &[VableSyncField],
        num_live: usize,
    ) -> (OpRef, Vec<(u32, OpRef)>) {
        self.emit_vable_sync_before(vable_ref, sync_fields);
        let result = self.call_may_force_float_typed(func_ptr, args, arg_types);
        let updated = self.emit_vable_sync_after(vable_ref, sync_fields);
        self.guard_not_forced(num_live);
        (result, updated)
    }

    /// Emit SETFIELD_GC for each virtualizable field before a residual call.
    fn emit_vable_sync_before(&mut self, vable_ref: OpRef, sync_fields: &[VableSyncField]) {
        for field in sync_fields {
            let descr = crate::fail_descr::make_fail_descr(field.field_descr_idx as usize);
            self.record_op_with_descr(OpCode::SetfieldGc, &[vable_ref, field.value], descr);
        }
    }

    /// Emit GETFIELD_GC for each virtualizable field after a residual call.
    ///
    /// Returns updated (field_descr_idx, new_opref) pairs.
    fn emit_vable_sync_after(
        &mut self,
        vable_ref: OpRef,
        sync_fields: &[VableSyncField],
    ) -> Vec<(u32, OpRef)> {
        sync_fields
            .iter()
            .map(|field| {
                let opcode = OpCode::getfield_for_type(field.field_type);
                let descr = crate::fail_descr::make_fail_descr(field.field_descr_idx as usize);
                let new_ref = self.record_op_with_descr(opcode, &[vable_ref], descr);
                (field.field_descr_idx, new_ref)
            })
            .collect()
    }

    /// Callback-based virtualizable sync for CALL_MAY_FORCE.
    ///
    /// Uses JitState's `sync_virtualizable_before/after_residual_call`
    /// methods to emit the appropriate SETFIELD/GETFIELD ops. This is
    /// the preferred API for interpreters that implement the JitState
    /// virtualizable sync hooks.
    ///
    /// Returns `(call_result, after_fields)` where `after_fields` are
    /// the (field_index, new_opref) pairs from the after-sync callback.
    pub fn call_may_force_with_jitstate_sync_int<S: crate::jit_state::JitState>(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        state: &S,
        num_live: usize,
    ) -> (OpRef, Vec<(u32, OpRef)>) {
        state.sync_virtualizable_before_residual_call(self);
        let result = self.call_may_force_int_typed(func_ptr, args, arg_types);
        let updated = state.sync_virtualizable_after_residual_call(self);
        self.guard_not_forced(num_live);
        (result, updated)
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

    pub fn call_void_typed(&mut self, func_ptr: *const (), args: &[OpRef], arg_types: &[Type]) {
        let _ = self.call_typed(OpCode::CallN, func_ptr, args, arg_types, Type::Void);
    }

    pub fn call_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallI, func_ptr, args, arg_types, Type::Int)
    }

    pub fn call_elidable_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallPureI, func_ptr, args, arg_types, Type::Int)
    }

    // ── Ref/Float call variants ─────────────────────────────────────

    /// Record a ref-returning function call (CallR).
    pub fn call_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning function call (CallF).
    pub fn call_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_float_typed(func_ptr, args, &arg_types)
    }

    /// Record a ref-returning elidable (pure) call (CallPureR).
    pub fn call_elidable_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_elidable_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning elidable (pure) call (CallPureF).
    pub fn call_elidable_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_elidable_float_typed(func_ptr, args, &arg_types)
    }

    pub fn call_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallR, func_ptr, args, arg_types, Type::Ref)
    }

    pub fn call_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallF, func_ptr, args, arg_types, Type::Float)
    }

    pub fn call_elidable_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallPureR, func_ptr, args, arg_types, Type::Ref)
    }

    pub fn call_elidable_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallPureF, func_ptr, args, arg_types, Type::Float)
    }

    fn call_family_typed(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        self.call_typed(opcode, func_ptr, args, arg_types, ret_type)
    }

    pub fn call_may_force_void_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Void),
            func_ptr,
            args,
            arg_types,
            Type::Void,
        );
    }

    pub fn call_may_force_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Int),
            func_ptr,
            args,
            arg_types,
            Type::Int,
        )
    }

    pub fn call_may_force_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Ref),
            func_ptr,
            args,
            arg_types,
            Type::Ref,
        )
    }

    /// Record a float-returning may-force call (CallMayForceF).
    pub fn call_may_force_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_may_force_float_typed(func_ptr, args, &arg_types)
    }

    pub fn call_may_force_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Float),
            func_ptr,
            args,
            arg_types,
            Type::Float,
        )
    }

    /// Record a void-returning GIL-release call (CallReleaseGilN).
    pub fn call_release_gil_void(&mut self, func_ptr: *const (), args: &[OpRef]) {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_release_gil_void_typed(func_ptr, args, &arg_types);
    }

    /// Record a ref-returning GIL-release call (CallReleaseGilR).
    pub fn call_release_gil_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_release_gil_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning GIL-release call (CallReleaseGilF).
    pub fn call_release_gil_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_release_gil_float_typed(func_ptr, args, &arg_types)
    }

    /// Record a ref-returning loop-invariant call (CallLoopinvariantR).
    pub fn call_loopinvariant_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_loopinvariant_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning loop-invariant call (CallLoopinvariantF).
    pub fn call_loopinvariant_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_loopinvariant_float_typed(func_ptr, args, &arg_types)
    }

    pub fn call_release_gil_void_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_family_typed(
            OpCode::call_release_gil_for_type(Type::Void),
            func_ptr,
            args,
            arg_types,
            Type::Void,
        );
    }

    pub fn call_release_gil_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_release_gil_for_type(Type::Int),
            func_ptr,
            args,
            arg_types,
            Type::Int,
        )
    }

    pub fn call_release_gil_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_release_gil_for_type(Type::Ref),
            func_ptr,
            args,
            arg_types,
            Type::Ref,
        )
    }

    pub fn call_release_gil_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_release_gil_for_type(Type::Float),
            func_ptr,
            args,
            arg_types,
            Type::Float,
        )
    }

    pub fn call_loopinvariant_void_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_family_typed(
            OpCode::call_loopinvariant_for_type(Type::Void),
            func_ptr,
            args,
            arg_types,
            Type::Void,
        );
    }

    pub fn call_loopinvariant_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_loopinvariant_for_type(Type::Int),
            func_ptr,
            args,
            arg_types,
            Type::Int,
        )
    }

    pub fn call_loopinvariant_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_loopinvariant_for_type(Type::Ref),
            func_ptr,
            args,
            arg_types,
            Type::Ref,
        )
    }

    pub fn call_loopinvariant_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_loopinvariant_for_type(Type::Float),
            func_ptr,
            args,
            arg_types,
            Type::Float,
        )
    }

    // ── CALL_ASSEMBLER ────────────────────────────────────────────

    fn call_assembler_typed(
        &mut self,
        opcode: OpCode,
        target: &LoopToken,
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        let descr = make_call_assembler_descr(target.number, arg_types, ret_type);
        self.record_op_with_descr(opcode, args, descr)
    }

    /// Emit CALL_ASSEMBLER_I by token number, without needing a `&LoopToken`.
    ///
    /// Assumes all args are `Type::Int`. For mixed-type args, use
    /// `call_assembler_int_by_number_typed` instead.
    pub fn call_assembler_int_by_number(&mut self, target_number: u64, args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_assembler_descr(target_number, &arg_types, Type::Int);
        self.record_op_with_descr(OpCode::CallAssemblerI, args, descr)
    }

    /// Emit CALL_ASSEMBLER_I by token number with explicit arg types.
    pub fn call_assembler_int_by_number_typed(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        let descr = make_call_assembler_descr(target_number, arg_types, Type::Int);
        self.record_op_with_descr(OpCode::CallAssemblerI, args, descr)
    }

    /// Emit CALL_ASSEMBLER_N (void). Assumes all args are `Type::Int`.
    /// For mixed-type args, use `call_assembler_void_typed`.
    pub fn call_assembler_void(&mut self, target: &LoopToken, args: &[OpRef]) {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_assembler_void_typed(target, args, &arg_types);
    }

    /// Emit CALL_ASSEMBLER_I. Assumes all args are `Type::Int`.
    /// For mixed-type args, use `call_assembler_int_typed`.
    pub fn call_assembler_int(&mut self, target: &LoopToken, args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_assembler_int_typed(target, args, &arg_types)
    }

    /// Emit CALL_ASSEMBLER_R. Assumes all args are `Type::Int`.
    /// For mixed-type args, use `call_assembler_ref_typed`.
    pub fn call_assembler_ref(&mut self, target: &LoopToken, args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_assembler_ref_typed(target, args, &arg_types)
    }

    /// Emit CALL_ASSEMBLER_F. Assumes all args are `Type::Int`.
    /// For mixed-type args, use `call_assembler_float_typed`.
    pub fn call_assembler_float(&mut self, target: &LoopToken, args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_assembler_float_typed(target, args, &arg_types)
    }

    pub fn call_assembler_void_typed(
        &mut self,
        target: &LoopToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Void),
            target,
            args,
            arg_types,
            Type::Void,
        );
    }

    pub fn call_assembler_int_typed(
        &mut self,
        target: &LoopToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Int),
            target,
            args,
            arg_types,
            Type::Int,
        )
    }

    pub fn call_assembler_ref_typed(
        &mut self,
        target: &LoopToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Ref),
            target,
            args,
            arg_types,
            Type::Ref,
        )
    }

    pub fn call_assembler_float_typed(
        &mut self,
        target: &LoopToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Float),
            target,
            args,
            arg_types,
            Type::Float,
        )
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

    // ── Virtual references ────────────────────────────────────────

    /// Record VIRTUAL_REF_R: create a virtual reference (ref-typed result).
    ///
    /// `virtual_obj` is the real object being wrapped.
    /// `force_token` is the force token for the current JIT frame.
    ///
    /// The optimizer replaces this with a virtual struct, so if the vref
    /// never escapes, no allocation happens.
    pub fn virtual_ref_r(&mut self, virtual_obj: OpRef, force_token: OpRef) -> OpRef {
        self.record_op(OpCode::VirtualRefR, &[virtual_obj, force_token])
    }

    /// Record VIRTUAL_REF_I: create a virtual reference (int-typed result).
    pub fn virtual_ref_i(&mut self, virtual_obj: OpRef, force_token: OpRef) -> OpRef {
        self.record_op(OpCode::VirtualRefI, &[virtual_obj, force_token])
    }

    /// Record VIRTUAL_REF_FINISH: finalize a virtual reference.
    ///
    /// `vref` is the virtual reference to finalize.
    /// `virtual_obj` is the real object (or NULL/0 if the frame is being left normally).
    pub fn virtual_ref_finish(&mut self, vref: OpRef, virtual_obj: OpRef) {
        self.record_op(OpCode::VirtualRefFinish, &[vref, virtual_obj]);
    }

    /// Record FORCE_TOKEN: capture the current JIT frame address.
    pub fn force_token(&mut self) -> OpRef {
        self.record_op(OpCode::ForceToken, &[])
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

#[cfg(test)]
mod tests {
    use super::*;
    use majit_codegen::LoopToken;
    use majit_ir::Type;

    extern "C" fn dummy_call_target() {}

    fn make_ctx_with_mixed_inputs() -> (TraceCtx, [OpRef; 3]) {
        let mut recorder = TraceRecorder::new();
        let r = recorder.record_input_arg(Type::Ref);
        let f = recorder.record_input_arg(Type::Float);
        let i = recorder.record_input_arg(Type::Int);
        (TraceCtx::new(recorder, 0), [r, f, i])
    }

    fn take_single_call_descr(ctx: TraceCtx, jump_args: &[OpRef]) -> (Vec<Type>, OpCode) {
        let mut recorder = ctx.recorder;
        recorder.close_loop(jump_args);
        let trace = recorder.get_trace();
        let call_op = &trace.ops[0];
        let arg_types = call_op
            .descr
            .as_ref()
            .and_then(|descr| descr.as_call_descr())
            .expect("call op should carry CallDescr")
            .arg_types()
            .to_vec();
        (arg_types, call_op.opcode)
    }

    fn take_single_call_op(ctx: TraceCtx, jump_args: &[OpRef]) -> majit_ir::Op {
        let mut recorder = ctx.recorder;
        recorder.close_loop(jump_args);
        let mut trace = recorder.get_trace();
        trace.ops.remove(0)
    }

    #[test]
    fn call_may_force_typed_preserves_mixed_arg_types() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_may_force_ref_typed(
            dummy_call_target as *const (),
            &args,
            &[Type::Ref, Type::Float, Type::Int],
        );
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallMayForceR);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_release_gil_typed_preserves_mixed_arg_types() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_release_gil_float_typed(
            dummy_call_target as *const (),
            &args,
            &[Type::Ref, Type::Float, Type::Int],
        );
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallReleaseGilF);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_loopinvariant_typed_preserves_mixed_arg_types() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_loopinvariant_int_typed(
            dummy_call_target as *const (),
            &args,
            &[Type::Ref, Type::Float, Type::Int],
        );
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallLoopinvariantI);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_assembler_typed_preserves_mixed_arg_types_and_target_token() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let token = LoopToken::new(777);
        let _ = ctx.call_assembler_ref_typed(&token, &args, &[Type::Ref, Type::Float, Type::Int]);
        let op = take_single_call_op(ctx, &args);
        assert_eq!(op.opcode, OpCode::CallAssemblerR);
        assert_eq!(op.args.as_slice(), &args);
        let call_descr = op
            .descr
            .as_ref()
            .and_then(|descr| descr.as_call_descr())
            .expect("call op should carry CallDescr");
        assert_eq!(call_descr.arg_types(), &[Type::Ref, Type::Float, Type::Int]);
        assert_eq!(call_descr.call_target_token(), Some(777));
    }

    fn take_all_ops(ctx: TraceCtx) -> Vec<majit_ir::Op> {
        let mut recorder = ctx.recorder;
        let num_inputs = recorder.num_inputargs();
        let jump_args: Vec<OpRef> = (0..num_inputs).map(|i| OpRef(i as u32)).collect();
        recorder.close_loop(&jump_args);
        let trace = recorder.get_trace();
        // Return only non-JUMP ops
        trace
            .ops
            .iter()
            .filter(|op| op.opcode != OpCode::Jump)
            .cloned()
            .collect()
    }

    #[test]
    fn call_may_force_with_vable_sync_emits_setfield_before_and_getfield_after() {
        let mut recorder = TraceRecorder::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let field_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        let sync_fields = [VableSyncField {
            field_descr_idx: 42,
            value: field_val,
            field_type: Type::Int,
        }];

        let (result, updated) = ctx.call_may_force_with_vable_sync_int(
            dummy_call_target as *const (),
            &[field_val],
            vable,
            &sync_fields,
            2,
        );

        // result should be a valid OpRef
        assert!(result.0 > 0);

        // updated should contain one field with the new OpRef
        assert_eq!(updated.len(), 1);
        assert_eq!(updated[0].0, 42);
        // the new OpRef should differ from the original field_val
        assert_ne!(updated[0].1, field_val);

        let ops = take_all_ops(ctx);
        // Expected sequence:
        //   SetfieldGc (before)
        //   CallMayForceI
        //   GetfieldGcI (after)
        //   GuardNotForced
        assert!(ops.len() >= 4);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceI);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_vable_sync_void_emits_correct_sequence() {
        let mut recorder = TraceRecorder::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let field_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        let sync_fields = [VableSyncField {
            field_descr_idx: 10,
            value: field_val,
            field_type: Type::Int,
        }];

        let updated = ctx.call_may_force_with_vable_sync_void(
            dummy_call_target as *const (),
            &[field_val],
            vable,
            &sync_fields,
            2,
        );

        assert_eq!(updated.len(), 1);
        assert_eq!(updated[0].0, 10);

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceN);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_vable_sync_multiple_fields() {
        let mut recorder = TraceRecorder::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let int_val = recorder.record_input_arg(Type::Int);
        let ref_val = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(recorder, 0);

        let sync_fields = [
            VableSyncField {
                field_descr_idx: 0,
                value: int_val,
                field_type: Type::Int,
            },
            VableSyncField {
                field_descr_idx: 1,
                value: ref_val,
                field_type: Type::Ref,
            },
        ];

        let (_, updated) = ctx.call_may_force_with_vable_sync_ref(
            dummy_call_target as *const (),
            &[int_val],
            vable,
            &sync_fields,
            3,
        );

        assert_eq!(updated.len(), 2);
        assert_eq!(updated[0].0, 0);
        assert_eq!(updated[1].0, 1);

        let ops = take_all_ops(ctx);
        // 2x SetfieldGc + CallMayForceR + 2x GetfieldGc + GuardNotForced
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[2].opcode, OpCode::CallMayForceR);
        assert_eq!(ops[3].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[4].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[5].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_empty_vable_sync_behaves_like_plain_call() {
        let mut recorder = TraceRecorder::new();
        let val = recorder.record_input_arg(Type::Int);
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(recorder, 0);

        let (result, updated) = ctx.call_may_force_with_vable_sync_int(
            dummy_call_target as *const (),
            &[val],
            vable,
            &[],
            1,
        );

        // No sync fields => no extra ops
        assert!(result.0 > 0);
        assert!(updated.is_empty());

        let ops = take_all_ops(ctx);
        // Just CallMayForceI + GuardNotForced
        assert_eq!(ops[0].opcode, OpCode::CallMayForceI);
        assert_eq!(ops[1].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_vable_sync_float_field() {
        let mut recorder = TraceRecorder::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let float_val = recorder.record_input_arg(Type::Float);
        let mut ctx = TraceCtx::new(recorder, 0);

        let sync_fields = [VableSyncField {
            field_descr_idx: 5,
            value: float_val,
            field_type: Type::Float,
        }];

        let (_, updated) = ctx.call_may_force_with_vable_sync_float(
            dummy_call_target as *const (),
            &[float_val],
            vable,
            &sync_fields,
            2,
        );

        assert_eq!(updated.len(), 1);
        assert_eq!(updated[0].0, 5);

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceF);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcF);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_default_noop() {
        use crate::jit_state::JitState;

        #[derive(Default)]
        struct NoVableState;

        impl JitState for NoVableState {
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
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }
        }

        let mut recorder = TraceRecorder::new();
        let val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);
        let state = NoVableState;

        let (result, updated) = ctx.call_may_force_with_jitstate_sync_int(
            dummy_call_target as *const (),
            &[val],
            &[Type::Int],
            &state,
            1,
        );

        // Default JitState does no sync => no extra ops
        assert!(result.0 > 0);
        assert!(updated.is_empty());

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::CallMayForceI);
        assert_eq!(ops[1].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_custom_impl() {
        use crate::jit_state::JitState;

        struct VableState {
            vable_ref: OpRef,
            field_val: OpRef,
        }

        impl JitState for VableState {
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
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }

            fn sync_virtualizable_before_residual_call(&self, ctx: &mut TraceCtx) {
                // Write field 0 to heap
                ctx.vable_setfield(self.vable_ref, 0, self.field_val);
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                ctx: &mut TraceCtx,
            ) -> Vec<(u32, OpRef)> {
                // Re-read field 0 from heap
                let new_ref = ctx.vable_getfield_int(self.vable_ref, 0);
                vec![(0, new_ref)]
            }
        }

        let mut recorder = TraceRecorder::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let field_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);
        let state = VableState {
            vable_ref: vable,
            field_val,
        };

        let (result, updated) = ctx.call_may_force_with_jitstate_sync_int(
            dummy_call_target as *const (),
            &[field_val],
            &[Type::Int],
            &state,
            2,
        );

        assert!(result.0 > 0);
        assert_eq!(updated.len(), 1);
        assert_eq!(updated[0].0, 0);

        let ops = take_all_ops(ctx);
        // SetfieldGc(before) + CallMayForceI + GetfieldGcI(after) + GuardNotForced
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceI);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }
}
