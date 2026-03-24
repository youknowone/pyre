//! JitState implementation for pyre.
//!
//! `PyreJitState` bridges the interpreter's `PyFrame` with majit's JIT
//! infrastructure. It extracts live values from the frame, restores them
//! after compiled code runs, and provides the meta/sym types for tracing.

use majit_ir::{DescrRef, OpCode, OpRef, Type, Value};
use majit_meta::virtualizable::VirtualizableInfo;
use majit_meta::{JitDriverStaticData, JitState, ResidualVirtualizableSync, TraceAction, TraceCtx};

use pyre_bytecode::bytecode::{BinaryOperator, CodeObject, ComparisonOperator, Instruction};
use pyre_interp::frame::PendingInlineResult;
use pyre_object::PyObjectRef;
use pyre_object::boolobject::w_bool_get_value;
use pyre_object::pyobject::{
    BOOL_TYPE, DICT_TYPE, FLOAT_TYPE, INT_TYPE, LIST_TYPE, NONE_TYPE, PyType, TUPLE_TYPE, is_bool,
    is_dict, is_float, is_int, is_list, is_none, is_tuple,
};
use pyre_object::rangeobject::RANGE_ITER_TYPE;
use pyre_object::strobject::is_str;
use pyre_object::{
    PY_NULL, w_bool_from, w_float_get_value, w_int_get_value, w_int_new,
    w_list_can_append_without_realloc, w_list_is_inline_storage, w_list_len, w_list_new,
    w_list_uses_float_storage, w_list_uses_int_storage, w_list_uses_object_storage,
    w_str_get_value, w_tuple_len,
};
use pyre_objspace::truth_value as objspace_truth_value;

/// Traced value — RPython `FrontendOp(position, _resint/_resref/_resfloat)` parity.
///
/// Carries both the symbolic IR reference (OpRef) and the concrete
/// execution value (ConcreteValue). Created by opcode handlers that
/// compute concrete results alongside IR recording.
#[derive(Clone, Copy, Debug)]
pub struct TracedBox {
    pub opref: OpRef,
    pub concrete: ConcreteValue,
}

impl TracedBox {
    pub fn new(opref: OpRef, concrete: ConcreteValue) -> Self {
        Self { opref, concrete }
    }

    pub fn opref_only(opref: OpRef) -> Self {
        Self {
            opref,
            concrete: ConcreteValue::Null,
        }
    }
}

/// Typed concrete value — RPython `FrontendOp._resint/_resref/_resfloat` parity.
///
/// Python bytecode uses untyped locals, so we use a tagged enum instead of
/// RPython's separate `registers_i/r/f` arrays. Each variant corresponds to
/// one of RPython's Box types: `BoxInt`, `BoxPtr`, `BoxFloat`.
#[derive(Clone, Copy, Debug)]
pub enum ConcreteValue {
    Int(i64),
    Float(f64),
    Ref(PyObjectRef),
    Null,
}

impl ConcreteValue {
    /// Convert from PyObjectRef (unbox if possible).
    pub fn from_pyobj(obj: PyObjectRef) -> Self {
        if obj.is_null() {
            return ConcreteValue::Null;
        }
        unsafe {
            if is_int(obj) {
                ConcreteValue::Int(w_int_get_value(obj))
            } else if is_float(obj) {
                ConcreteValue::Float(w_float_get_value(obj))
            } else {
                ConcreteValue::Ref(obj)
            }
        }
    }

    /// Convert to PyObjectRef (box if needed).
    pub fn to_pyobj(self) -> PyObjectRef {
        match self {
            ConcreteValue::Int(v) => w_int_new(v),
            ConcreteValue::Float(v) => pyre_object::w_float_new(v),
            ConcreteValue::Ref(obj) => obj,
            ConcreteValue::Null => PY_NULL,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, ConcreteValue::Null)
    }

    /// RPython box.getint() parity.
    pub fn getint(&self) -> Option<i64> {
        match self {
            ConcreteValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// RPython box.getfloatstorage() parity.
    pub fn getfloat(&self) -> Option<f64> {
        match self {
            ConcreteValue::Float(v) => Some(*v),
            ConcreteValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// RPython box.getref_base() parity.
    pub fn getref(&self) -> PyObjectRef {
        self.to_pyobj()
    }

    /// Convert to majit IR Type.
    pub fn ir_type(&self) -> Type {
        match self {
            ConcreteValue::Int(_) => Type::Int,
            ConcreteValue::Float(_) => Type::Float,
            ConcreteValue::Ref(_) => Type::Ref,
            ConcreteValue::Null => Type::Ref,
        }
    }

    /// Truth value (RPython box.getint() != 0 for goto_if_not).
    pub fn is_truthy(&self) -> bool {
        match self {
            ConcreteValue::Int(v) => *v != 0,
            ConcreteValue::Float(v) => *v != 0.0,
            ConcreteValue::Ref(obj) => objspace_truth_value(*obj),
            ConcreteValue::Null => false,
        }
    }
}

/// Convert a bytecode constant to ConcreteValue.
pub fn load_const_concrete(constant: &pyre_bytecode::bytecode::ConstantData) -> ConcreteValue {
    use pyre_bytecode::bytecode::ConstantData;
    match constant {
        ConstantData::Integer { value } => match i64::try_from(value).ok() {
            Some(v) => ConcreteValue::Int(v),
            None => ConcreteValue::Ref(pyre_object::w_long_new(value.clone())),
        },
        ConstantData::Float { value } => ConcreteValue::Float(*value),
        ConstantData::Boolean { value } => ConcreteValue::Int(*value as i64),
        ConstantData::Str { value } => ConcreteValue::Ref(pyre_object::w_str_new(
            value.as_str().expect("non-UTF-8 string constant"),
        )),
        ConstantData::None => ConcreteValue::Ref(pyre_object::w_none()),
        _ => ConcreteValue::Null,
    }
}

use pyre_runtime::{
    ArithmeticOpcodeHandler, BranchOpcodeHandler, ConstantOpcodeHandler, ControlFlowOpcodeHandler,
    IterOpcodeHandler, LocalOpcodeHandler, NamespaceOpcodeHandler, OpcodeStepExecutor, PyBigInt,
    PyError, PyNamespace, PyObjectArray, SharedOpcodeHandler, StackOpcodeHandler,
    TruthOpcodeHandler, decode_instruction_at, execute_opcode_step, is_builtin_func, is_func,
    range_iter_continues, w_builtin_func_name, w_func_get_code_ptr, w_func_get_globals,
};

use crate::jit::descr::{
    bool_boolval_descr, dict_len_descr, float_floatval_descr, int_intval_descr,
    list_float_items_heap_cap_descr, list_float_items_len_descr, list_float_items_ptr_descr,
    list_int_items_heap_cap_descr, list_int_items_len_descr, list_int_items_ptr_descr,
    list_items_heap_cap_descr, list_items_len_descr, list_items_ptr_descr, list_strategy_descr,
    make_array_descr, make_field_descr, namespace_values_len_descr, namespace_values_ptr_descr,
    ob_type_descr, range_iter_current_descr, range_iter_step_descr, range_iter_stop_descr,
    str_len_descr, tuple_items_len_descr, tuple_items_ptr_descr, w_float_size_descr,
    w_int_size_descr,
};
use crate::jit::frame_layout::{
    PYFRAME_LOCALS_CELLS_STACK_OFFSET, PYFRAME_NAMESPACE_OFFSET, PYFRAME_NEXT_INSTR_OFFSET,
    PYFRAME_VALUESTACKDEPTH_OFFSET, build_pyframe_virtualizable_info,
};
use crate::jit::helpers::{
    TraceHelperAccess, emit_box_float_inline, emit_trace_bool_value_from_truth,
};

/// Interpreter state exposed to the JIT framework.
///
/// Built from `PyFrame` before calling `back_edge`, and synced back
/// after compiled code runs.
pub struct PyreJitState {
    /// Opaque pointer to the owning PyFrame heap object.
    pub frame: usize,
    /// Current instruction index.
    pub next_instr: usize,
    /// Absolute index into `locals_cells_stack_w` marking stack top.
    /// Starts at `nlocals` (empty stack), grows upward on push.
    pub valuestackdepth: usize,
}

/// Meta information for a trace — describes the shape of the code being traced.
#[derive(Clone)]
pub struct PyreMeta {
    /// Instruction index at the merge point (green key).
    pub merge_pc: usize,
    /// Number of fast local variable slots.
    pub num_locals: usize,
    /// Sorted namespace keys expected by this trace.
    pub ns_keys: Vec<String>,
    /// Absolute valuestackdepth at the merge point.
    pub valuestackdepth: usize,
    /// Whether the optimizer uses the virtualizable mechanism.
    /// When true, guard fail_args follow the layout:
    ///   [frame, next_instr, valuestackdepth, l0..lN, s0..sM]
    pub has_virtualizable: bool,
    /// RPython resume.py parity: typed live boxes for locals + stack.
    /// Order is locals first, then stack-only slots.
    pub slot_types: Vec<Type>,
}

/// Symbolic state during tracing.
///
/// `frame` maps to a live IR `OpRef`. Symbolic frame field tracking
/// (locals, stack, valuestackdepth, next_instr) persists across instructions.
/// Locals and stack are virtualized (carried through JUMP args);
/// only next_instr and valuestackdepth are synced before guards / loop close.
#[derive(Clone)]
pub struct PyreSym {
    /// OpRef for the owning PyFrame pointer.
    pub frame: OpRef,
    // ── Persistent symbolic frame field tracking ──
    // These fields survive across per-instruction TraceFrameState lifetimes.
    pub(crate) symbolic_locals: Vec<OpRef>,
    pub(crate) symbolic_stack: Vec<OpRef>,
    pub(crate) symbolic_local_types: Vec<Type>,
    pub(crate) symbolic_stack_types: Vec<Type>,
    pub(crate) pending_next_instr: Option<usize>,
    pub(crate) locals_cells_stack_array_ref: OpRef,
    /// Absolute index into the unified array (starts at nlocals).
    pub(crate) valuestackdepth: usize,
    /// Number of local variable slots (cached from code object).
    pub(crate) nlocals: usize,
    pub(crate) symbolic_initialized: bool,
    pub(crate) vable_next_instr: OpRef,
    pub(crate) vable_valuestackdepth: OpRef,
    pub(crate) symbolic_namespace_slots: std::collections::HashMap<usize, OpRef>,
    /// Base OpRef index for virtualizable array slots.
    /// When set, symbolic_locals[i] = OpRef(vable_array_base + i),
    /// symbolic_stack[j] = OpRef(vable_array_base + nlocals + j).
    pub(crate) vable_array_base: Option<u32>,
    /// RPython goto_if_not fusion: cached raw truth OpRef from the
    /// most recent int/float comparison. POP_JUMP_IF_FALSE/TRUE
    /// consumes this to emit GUARD_TRUE/GUARD_FALSE directly,
    /// bypassing bool object creation.
    pub(crate) last_comparison_truth: Option<OpRef>,
    /// Concrete truth value paired with `last_comparison_truth`.
    /// RPython pyjitpl.py passes the computed condbox directly into
    /// goto_if_not and reads `box.getint()` there, instead of reloading a
    /// stack value later. Keep the concrete branch direction alongside the
    /// raw traced truth so POP_JUMP_IF can consume the same pair.
    pub(crate) last_comparison_concrete_truth: Option<bool>,
    /// Concrete box most recently popped through the shared stack API.
    /// This mirrors RPython's `goto_if_not(box)` ownership: branch truth
    /// consumers should inspect the popped box, not reload from the
    /// post-pop concrete stack.
    pub(crate) last_popped_concrete_value: Option<PyObjectRef>,
    /// Popped branch source currently being converted to truth.
    /// Generic guards emitted during `truth_value(value)` must still capture
    /// the pre-pop stack shape, matching RPython's goto_if_not(box).
    pub(crate) pending_branch_value: Option<OpRef>,
    pub(crate) transient_value_types: std::collections::HashMap<OpRef, Type>,
    // ── MIFrame concrete Box tracking (RPython registers_i/r/f parity) ──
    // Concrete Python object values for locals and stack, tracked in
    // parallel with symbolic_locals/symbolic_stack. Each opcode handler
    // updates these alongside the symbolic OpRefs so that guard decisions,
    // branch directions, and call results use internally tracked values
    // instead of reading from an external PyFrame snapshot.
    pub(crate) concrete_locals: Vec<ConcreteValue>,
    pub(crate) concrete_stack: Vec<ConcreteValue>,
    /// Concrete value to be consumed by the next push_value() call.
    pub(crate) pending_concrete_push: Option<ConcreteValue>,
    /// Frame metadata extracted at trace start — avoids stale snapshot reads.
    /// RPython MIFrame.jitcode parity.
    pub(crate) concrete_code: *const pyre_bytecode::CodeObject,
    /// Namespace for global lookups.
    pub(crate) concrete_namespace: *mut pyre_runtime::PyNamespace,
    /// Execution context pointer (for creating callee frames).
    pub(crate) concrete_execution_context: *const pyre_runtime::PyExecutionContext,
}

/// Trace-time view over the virtualizable `PyFrame`.
///
/// Per-instruction wrapper that borrows persistent symbolic state from
/// `PyreSym` via raw pointer. The symbolic tracking (locals, stack,
/// valuestackdepth, next_instr) lives in PyreSym and survives across
/// instructions; this struct provides the per-instruction context
/// (ctx, fallthrough_pc, concrete_frame).
pub(crate) struct TraceFrameState {
    ctx: *mut TraceCtx,
    sym: *mut PyreSym,
    concrete_frame: usize,
    ob_type_fd: DescrRef,
    fallthrough_pc: usize,
    /// PyPy capture_resumedata: parent frame fail_args for multi-frame guards.
    pub(crate) parent_fail_args: Option<Vec<OpRef>>,
    pub(crate) parent_fail_arg_types: Option<Vec<Type>>,
    pending_inline_frame: Option<PendingInlineFrame>,
}

fn code_has_backward_jump(code: &CodeObject) -> bool {
    for pc in 0..code.instructions.len() {
        let Some((instruction, _)) = decode_instruction_at(code, pc) else {
            continue;
        };
        if matches!(
            instruction,
            Instruction::JumpBackward { .. } | Instruction::JumpBackwardNoInterrupt { .. }
        ) {
            return true;
        }
    }
    false
}

fn instruction_consumes_comparison_truth(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::PopJumpIfFalse { .. } | Instruction::PopJumpIfTrue { .. }
    )
}

fn instruction_is_trivia_between_compare_and_branch(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::ExtendedArg
            | Instruction::Resume { .. }
            | Instruction::Nop
            | Instruction::Cache
            | Instruction::NotTaken
    )
}

/// Environment context — currently unused.
pub struct PyreEnv;

fn pyobject_array_descr() -> DescrRef {
    make_array_descr(0, 8, Type::Ref, false)
}

fn int_array_descr() -> DescrRef {
    make_array_descr(0, 8, Type::Int, true)
}

fn float_array_descr() -> DescrRef {
    make_array_descr(0, 8, Type::Float, false)
}

fn frame_locals_cells_stack_descr() -> DescrRef {
    make_field_descr(PYFRAME_LOCALS_CELLS_STACK_OFFSET, 8, Type::Int, false)
}

fn frame_stack_depth_descr() -> DescrRef {
    make_field_descr(PYFRAME_VALUESTACKDEPTH_OFFSET, 8, Type::Int, true)
}

fn frame_next_instr_descr() -> DescrRef {
    make_field_descr(PYFRAME_NEXT_INSTR_OFFSET, 8, Type::Int, true)
}

fn frame_namespace_descr() -> DescrRef {
    make_field_descr(PYFRAME_NAMESPACE_OFFSET, 8, Type::Int, false)
}

pub(crate) fn trace_ob_type_descr() -> DescrRef {
    ob_type_descr()
}

fn box_traced_raw_int(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    crate::jit::helpers::emit_box_int_inline(
        ctx,
        value,
        w_int_size_descr(),
        ob_type_descr(),
        int_intval_descr(),
        &INT_TYPE as *const _ as i64,
    )
}

fn note_inline_trace_too_long(
    callee_key: u64,
    caller_function_key: u64,
    root_trace_key: u64,
    err: &PyError,
) {
    if err.message != "inline trace aborted" {
        return;
    }
    let (driver, _) = crate::eval::driver_pair();
    let warm_state = driver.meta_interp_mut().warm_state_mut();
    warm_state.disable_noninlinable_function(callee_key);
    if callee_key == caller_function_key {
        // RPython converges the next portal entry quickly after marking the
        // huge function non-inlinable. Mirror that by priming the warmstate-
        // owned function-entry counter instead of only bumping a backedge
        // hot counter.
        warm_state.boost_function_entry(caller_function_key);
    } else {
        warm_state.trace_next_iteration(root_trace_key);
    }
    if majit_meta::majit_log_enabled() {
        eprintln!(
            "[jit][trace-through] disable_noninlinable_function key={} caller_function_key={} root_trace_key={} same_key={}",
            callee_key,
            caller_function_key,
            root_trace_key,
            callee_key == caller_function_key
        );
    }
}

fn current_trace_green_key(state: &mut TraceFrameState) -> u64 {
    state.with_ctx(|_, ctx| ctx.green_key())
}

fn root_trace_green_key(state: &mut TraceFrameState) -> u64 {
    state.with_ctx(|_, ctx| ctx.root_green_key())
}

fn biggest_inline_trace_key(state: &mut TraceFrameState) -> Option<u64> {
    state.with_ctx(|_, ctx| ctx.find_biggest_inline_function())
}

fn note_root_trace_too_long(green_key: u64) {
    let (driver, _) = crate::eval::driver_pair();
    let warm_state = driver.meta_interp_mut().warm_state_mut();
    warm_state.trace_next_iteration(green_key);
    warm_state.mark_force_finish_tracing(green_key);
    if majit_meta::majit_log_enabled() {
        eprintln!(
            "[jit][trace-too-long] trace_next_iteration + mark_force_finish_tracing key={}",
            green_key
        );
    }
}

fn box_traced_raw_float(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_box_float_inline(
        ctx,
        value,
        w_float_size_descr(),
        ob_type_descr(),
        float_floatval_descr(),
        &FLOAT_TYPE as *const _ as i64,
    )
}

fn ensure_boxed_for_ca(ctx: &mut TraceCtx, state: &TraceFrameState, value: OpRef) -> OpRef {
    match state.value_type(value) {
        Type::Int => box_traced_raw_int(ctx, value),
        Type::Float => box_traced_raw_float(ctx, value),
        Type::Ref | Type::Void => value,
    }
}

fn box_value_for_python_helper(
    state: &mut TraceFrameState,
    ctx: &mut TraceCtx,
    value: OpRef,
) -> OpRef {
    match state.value_type(value) {
        Type::Int => box_traced_raw_int(ctx, value),
        Type::Float => box_traced_raw_float(ctx, value),
        Type::Ref | Type::Void => value,
    }
}

fn box_args_for_python_helper(
    state: &mut TraceFrameState,
    ctx: &mut TraceCtx,
    args: &[OpRef],
) -> Vec<OpRef> {
    args.iter()
        .map(|&arg| box_value_for_python_helper(state, ctx, arg))
        .collect()
}

fn try_trace_const_pure_int_field(
    ctx: &mut TraceCtx,
    obj: OpRef,
    descr: &DescrRef,
) -> Option<OpRef> {
    if !descr.is_always_pure() {
        return None;
    }
    let ptr = ctx.const_value(obj)?;
    if ptr == 0 {
        return None;
    }
    let field = descr.as_field_descr()?;
    let addr = ptr as usize + field.offset();
    let value = unsafe {
        match (field.field_size(), field.is_field_signed()) {
            (8, _) => *(addr as *const i64),
            (4, true) => *(addr as *const i32) as i64,
            (4, false) => *(addr as *const u32) as i64,
            (2, true) => *(addr as *const i16) as i64,
            (2, false) => *(addr as *const u16) as i64,
            (1, true) => *(addr as *const i8) as i64,
            (1, false) => *(addr as *const u8) as i64,
            _ => return None,
        }
    };
    Some(ctx.const_int(value))
}

fn try_trace_const_boxed_int(
    ctx: &mut TraceCtx,
    value: OpRef,
    concrete_value: PyObjectRef,
) -> Option<OpRef> {
    if ctx.const_value(value) != Some(concrete_value as i64) {
        return None;
    }
    unsafe {
        if is_int(concrete_value) {
            return Some(ctx.const_int(w_int_get_value(concrete_value)));
        }
        if is_bool(concrete_value) {
            return Some(ctx.const_int(if w_bool_get_value(concrete_value) {
                1
            } else {
                0
            }));
        }
    }
    None
}

fn try_trace_const_boxed_float(
    ctx: &mut TraceCtx,
    value: OpRef,
    concrete_value: PyObjectRef,
) -> Option<OpRef> {
    if ctx.const_value(value) != Some(concrete_value as i64) {
        return None;
    }
    unsafe {
        is_float(concrete_value)
            .then(|| ctx.const_int(w_float_get_value(concrete_value).to_bits() as i64))
    }
}

fn trace_gc_object_int_field(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    if let Some(folded) = try_trace_const_pure_int_field(ctx, obj, &descr) {
        return folded;
    }
    let opcode = if descr.is_always_pure() {
        OpCode::GetfieldGcPureI
    } else {
        OpCode::GetfieldGcI
    };
    ctx.record_op_with_descr(opcode, &[obj], descr)
}

fn trace_gc_object_type_field(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    trace_gc_object_int_field(ctx, obj, descr)
}

unsafe fn objspace_compare_ints(
    lhs_obj: PyObjectRef,
    rhs_obj: PyObjectRef,
    op: ComparisonOperator,
) -> bool {
    let lhs = w_int_get_value(lhs_obj);
    let rhs = w_int_get_value(rhs_obj);
    match op {
        ComparisonOperator::Less => lhs < rhs,
        ComparisonOperator::LessOrEqual => lhs <= rhs,
        ComparisonOperator::Greater => lhs > rhs,
        ComparisonOperator::GreaterOrEqual => lhs >= rhs,
        ComparisonOperator::Equal => lhs == rhs,
        ComparisonOperator::NotEqual => lhs != rhs,
    }
}

unsafe fn objspace_compare_floats(
    lhs_obj: PyObjectRef,
    rhs_obj: PyObjectRef,
    op: ComparisonOperator,
) -> bool {
    let lhs = w_float_get_value(lhs_obj);
    let rhs = w_float_get_value(rhs_obj);
    match op {
        ComparisonOperator::Less => lhs < rhs,
        ComparisonOperator::LessOrEqual => lhs <= rhs,
        ComparisonOperator::Greater => lhs > rhs,
        ComparisonOperator::GreaterOrEqual => lhs >= rhs,
        ComparisonOperator::Equal => lhs == rhs,
        ComparisonOperator::NotEqual => lhs != rhs,
    }
}

pub(crate) fn frame_locals_cells_stack_array(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(
        OpCode::GetfieldRawI,
        &[frame],
        frame_locals_cells_stack_descr(),
    )
}

pub(crate) fn frame_namespace_ptr(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(OpCode::GetfieldRawI, &[frame], frame_namespace_descr())
}

/// Read from frame's locals_cells_stack_w array.
/// Uses GcR (Ref-typed) to match RPython's GETARRAYITEM_GC_R,
/// ensuring the optimizer knows these are boxed pointers.
pub(crate) fn trace_array_getitem_value(ctx: &mut TraceCtx, array: OpRef, index: OpRef) -> OpRef {
    ctx.record_op_with_descr(
        OpCode::GetarrayitemGcR,
        &[array, index],
        pyobject_array_descr(),
    )
}

/// Read from frame's locals_cells_stack_w — namespace access path.
pub(crate) fn trace_raw_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    ctx.record_op_with_descr(
        OpCode::GetarrayitemGcR,
        &[array, index],
        pyobject_array_descr(),
    )
}

pub(crate) fn trace_raw_int_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    ctx.record_op_with_descr(OpCode::GetarrayitemRawI, &[array, index], int_array_descr())
}

pub(crate) fn trace_raw_float_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    ctx.record_op_with_descr(
        OpCode::GetarrayitemRawF,
        &[array, index],
        float_array_descr(),
    )
}

/// Write to frame's locals_cells_stack_w array.
/// Uses Gc (GC-typed) to match RPython's SETARRAYITEM_GC.
pub(crate) fn trace_raw_array_setitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    ctx.record_op_with_descr(
        OpCode::SetarrayitemGc,
        &[array, index, value],
        pyobject_array_descr(),
    );
}

pub(crate) fn trace_raw_int_array_setitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    ctx.record_op_with_descr(
        OpCode::SetarrayitemRaw,
        &[array, index, value],
        int_array_descr(),
    );
}

pub(crate) fn trace_raw_float_array_setitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    ctx.record_op_with_descr(
        OpCode::SetarrayitemRaw,
        &[array, index, value],
        float_array_descr(),
    );
}

fn is_boxed_int_value(concrete_value: PyObjectRef) -> bool {
    !concrete_value.is_null() && unsafe { is_int(concrete_value) }
}

pub(crate) fn frame_get_next_instr(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(OpCode::GetfieldRawI, &[frame], frame_next_instr_descr())
}

pub(crate) fn frame_get_stack_depth(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(OpCode::GetfieldRawI, &[frame], frame_stack_depth_descr())
}

/// Read a value from the unified `locals_cells_stack_w` at the given absolute index.
/// Get the concrete top-of-stack value for the RETURN_VALUE opcode.
pub(crate) fn concrete_return_value(frame: usize) -> Option<PyObjectRef> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let vsd = unsafe {
        *(frame_ptr.add(crate::jit::frame_layout::PYFRAME_VALUESTACKDEPTH_OFFSET) as *const usize)
    };
    if vsd == 0 {
        return None;
    }
    concrete_stack_value(frame, vsd - 1)
}

/// Read a value from the unified `locals_cells_stack_w` at the given absolute index.
pub(crate) fn concrete_stack_value(frame: usize, abs_idx: usize) -> Option<PyObjectRef> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let arr =
        unsafe { &*(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET) as *const PyObjectArray) };
    arr.as_slice().get(abs_idx).copied()
}

/// Return nlocals for the given frame (from the unified array's total length and code object).
pub(crate) fn concrete_nlocals(frame: usize) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let code_ptr = unsafe {
        *(frame_ptr.add(crate::jit::frame_layout::PYFRAME_CODE_OFFSET)
            as *const *const pyre_bytecode::CodeObject)
    };
    if code_ptr.is_null() {
        return None;
    }
    Some(unsafe { (&(*code_ptr).varnames).len() })
}

/// Return nlocals as the "locals_len" for virtualizable.
pub(crate) fn concrete_locals_len(frame: usize) -> Option<usize> {
    concrete_nlocals(frame)
}

/// Return the absolute valuestackdepth.
pub(crate) fn concrete_stack_depth(frame: usize) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    Some(unsafe { *(frame_ptr.add(PYFRAME_VALUESTACKDEPTH_OFFSET) as *const usize) })
}

pub(crate) fn concrete_namespace_slot(frame: usize, name: &str) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let namespace_ptr =
        unsafe { *(frame_ptr.add(PYFRAME_NAMESPACE_OFFSET) as *const *mut PyNamespace) };
    let namespace = (!namespace_ptr.is_null()).then_some(unsafe { &*namespace_ptr })?;
    namespace.slot_of(name)
}

pub(crate) fn concrete_namespace_value(frame: usize, idx: usize) -> Option<PyObjectRef> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let namespace_ptr =
        unsafe { *(frame_ptr.add(PYFRAME_NAMESPACE_OFFSET) as *const *mut PyNamespace) };
    let namespace = (!namespace_ptr.is_null()).then_some(unsafe { &*namespace_ptr })?;
    namespace.get_slot(idx)
}

fn namespace_slot_direct(ns: *mut PyNamespace, name: &str) -> Option<usize> {
    if ns.is_null() {
        return None;
    }
    unsafe { &*ns }.slot_of(name)
}

fn namespace_value_direct(ns: *mut PyNamespace, idx: usize) -> Option<PyObjectRef> {
    if ns.is_null() {
        return None;
    }
    unsafe { &*ns }.get_slot(idx)
}

pub(crate) fn record_current_state_guard(
    ctx: &mut TraceCtx,
    frame: OpRef,
    next_instr: OpRef,
    stack_depth: OpRef,
    locals: &[OpRef],
    stack: &[OpRef],
    opcode: OpCode,
    args: &[OpRef],
) {
    let mut fail_args = vec![frame, next_instr, stack_depth];
    fail_args.extend_from_slice(locals);
    fail_args.extend_from_slice(stack);
    let slot_types = std::iter::repeat_n(Type::Ref, fail_args.len().saturating_sub(3));
    let fail_arg_types = virtualizable_fail_arg_types(slot_types);
    ctx.record_guard_typed_with_fail_args(opcode, args, fail_arg_types, &fail_args);
}

fn virtualizable_fail_arg_types(slot_types: impl IntoIterator<Item = Type>) -> Vec<Type> {
    let mut types = vec![Type::Ref, Type::Int, Type::Int];
    types.extend(slot_types);
    types
}

fn concrete_value_type(value: PyObjectRef) -> Type {
    if value.is_null() {
        return Type::Ref;
    }
    if !looks_like_heap_ref(value) {
        // Non-heap value (e.g. PY_NULL sentinel or leaked raw int)
        return Type::Int;
    }
    if unsafe { is_int(value) } {
        return Type::Int;
    }
    if unsafe { is_float(value) } {
        return Type::Float;
    }
    Type::Ref
}

/// RPython parity: virtualizable array slots are always GCREF (Ref).
/// Even W_IntObject values are Ref — the trace unboxes via GetfieldGcPureI.
fn concrete_virtualizable_slot_type(_value: PyObjectRef) -> Type {
    Type::Ref
}

fn looks_like_heap_ref(value: PyObjectRef) -> bool {
    let addr = value as usize;
    let word_align = std::mem::align_of::<usize>() - 1;
    addr >= 0x1_0000 && addr < (1usize << 56) && (addr & word_align) == 0
}

fn extract_concrete_typed_value(slot_type: Type, value: PyObjectRef) -> Value {
    match slot_type {
        Type::Int => {
            if value.is_null() {
                Value::Int(0)
            } else if looks_like_heap_ref(value) && unsafe { is_int(value) } {
                Value::Int(unsafe { w_int_get_value(value) })
            } else {
                Value::Int(value as i64)
            }
        }
        Type::Float => {
            if value.is_null() {
                Value::Float(0.0)
            } else if looks_like_heap_ref(value) && unsafe { is_float(value) } {
                Value::Float(unsafe { pyre_object::floatobject::w_float_get_value(value) })
            } else {
                Value::Float(f64::from_bits(value as u64))
            }
        }
        Type::Ref | Type::Void => Value::Ref(majit_ir::GcRef(value as usize)),
    }
}

fn concrete_slot_types(frame: usize, num_locals: usize, valuestackdepth: usize) -> Vec<Type> {
    let stack_only = valuestackdepth.saturating_sub(num_locals);
    let mut types = Vec::with_capacity(num_locals + stack_only);
    for idx in 0..num_locals {
        types.push(
            concrete_stack_value(frame, idx)
                .map(concrete_virtualizable_slot_type)
                .unwrap_or(Type::Ref),
        );
    }
    for stack_idx in 0..stack_only {
        types.push(
            concrete_stack_value(frame, num_locals + stack_idx)
                .map(concrete_virtualizable_slot_type)
                .unwrap_or(Type::Ref),
        );
    }
    types
}

fn boxed_slot_i64_for_type(slot_type: Type, raw: i64) -> PyObjectRef {
    match slot_type {
        Type::Int => w_int_new(raw),
        Type::Float => pyre_object::floatobject::w_float_new(f64::from_bits(raw as u64)),
        Type::Ref | Type::Void => raw as PyObjectRef,
    }
}

fn boxed_slot_value_for_type(slot_type: Type, value: &Value) -> PyObjectRef {
    match (slot_type, value) {
        (Type::Int, Value::Int(v)) => w_int_new(*v),
        (Type::Float, Value::Float(v)) => pyre_object::floatobject::w_float_new(*v),
        (_, Value::Ref(r)) => r.as_usize() as PyObjectRef,
        (_, Value::Int(v)) => *v as PyObjectRef,
        (_, Value::Float(v)) => v.to_bits() as PyObjectRef,
        (_, Value::Void) => PY_NULL,
    }
}

/// RPython parity: virtualizable array slots are always GCREF.
/// When restoring from guard failure, treat ALL values as Ref (PyObjectRef).
/// Int values that look like heap pointers are used directly; otherwise box.
/// RPython parity: virtualizable array slots are always GCREF.
/// When restoring from guard failure, treat ALL values as Ref (PyObjectRef).
/// Int values that look like heap pointers are used directly (they are
/// PyObjectRef that the optimizer typed as Int during unboxing). Zero/null
/// values become PY_NULL. Only true small ints (non-zero, non-pointer)
/// need boxing.
fn boxed_slot_value_as_ref(value: &Value) -> PyObjectRef {
    match value {
        Value::Ref(r) => r.as_usize() as PyObjectRef,
        Value::Int(v) => {
            let addr = *v as usize;
            if addr == 0 {
                PY_NULL
            } else if addr >= 0x1_0000 && addr < (1usize << 56) && (addr & 7) == 0 {
                // Heap pointer — use as PyObjectRef directly
                *v as PyObjectRef
            } else {
                // Small int — box it
                w_int_new(*v)
            }
        }
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v),
        Value::Void => PY_NULL,
    }
}

fn boxed_slot_value_from_runtime_kind(value: &Value) -> PyObjectRef {
    match value {
        Value::Int(v) => w_int_new(*v),
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v),
        Value::Ref(r) => r.as_usize() as PyObjectRef,
        Value::Void => PY_NULL,
    }
}

fn fail_arg_opref_for_typed_value(ctx: &mut TraceCtx, value: Value) -> OpRef {
    match value {
        Value::Int(v) => ctx.const_int(v),
        Value::Float(v) => ctx.const_int(v.to_bits() as i64),
        Value::Ref(r) => ctx.const_int(r.as_usize() as i64),
        Value::Void => ctx.const_int(PY_NULL as i64),
    }
}

fn pending_inline_result_from_concrete(
    result_type: Type,
    concrete_result: PyObjectRef,
) -> PendingInlineResult {
    match result_type {
        Type::Int => PendingInlineResult::Int(unsafe { w_int_get_value(concrete_result) }),
        Type::Float => PendingInlineResult::Float(unsafe {
            pyre_object::floatobject::w_float_get_value(concrete_result)
        }),
        Type::Ref | Type::Void => PendingInlineResult::Ref(concrete_result),
    }
}

fn materialize_pending_inline_result(result: PendingInlineResult) -> PyObjectRef {
    match result {
        PendingInlineResult::Ref(result) => result,
        PendingInlineResult::Int(value) => w_int_new(value),
        PendingInlineResult::Float(value) => pyre_object::floatobject::w_float_new(value),
    }
}

fn frame_callable_arg_types(nargs: usize) -> Vec<Type> {
    let mut types = Vec::with_capacity(2 + nargs);
    types.push(Type::Ref);
    types.push(Type::Ref);
    for _ in 0..nargs {
        types.push(Type::Ref);
    }
    types
}

fn one_arg_callee_frame_helper(arg_type: Type, is_self_recursive: bool) -> (*const (), Vec<Type>) {
    match (is_self_recursive, arg_type) {
        (true, Type::Int) => (
            crate::call_jit::jit_create_self_recursive_callee_frame_1_raw_int as *const (),
            vec![Type::Ref, Type::Int],
        ),
        (true, _) => (
            crate::call_jit::jit_create_self_recursive_callee_frame_1 as *const (),
            vec![Type::Ref, Type::Ref],
        ),
        (false, Type::Int) => (
            crate::call_jit::jit_create_callee_frame_1_raw_int as *const (),
            vec![Type::Ref, Type::Ref, Type::Int],
        ),
        (false, _) => (
            crate::call_jit::jit_create_callee_frame_1 as *const (),
            vec![Type::Ref, Type::Ref, Type::Ref],
        ),
    }
}

fn fail_arg_types_for_virtualizable_state(len: usize) -> Vec<Type> {
    let mut types = Vec::with_capacity(len);
    for idx in 0..len {
        if idx == 0 {
            types.push(Type::Ref);
        } else if idx < 3 {
            types.push(Type::Int);
        } else {
            types.push(Type::Ref);
        }
    }
    types
}

fn frame_entry_arg_types_from_slot_types(slot_types: &[Type]) -> Vec<Type> {
    if slot_types.is_empty() {
        vec![Type::Ref]
    } else {
        let mut types = Vec::with_capacity(3 + slot_types.len());
        types.push(Type::Ref);
        types.push(Type::Int);
        types.push(Type::Int);
        types.extend(slot_types.iter().copied());
        types
    }
}

fn pending_entry_slot_types_from_args(
    arg_types: &[Type],
    callee_nlocals: usize,
    callee_stack_only: usize,
) -> Vec<Type> {
    let mut slot_types = Vec::with_capacity(callee_nlocals + callee_stack_only);
    slot_types.extend(arg_types.iter().copied().take(callee_nlocals));
    while slot_types.len() < callee_nlocals {
        slot_types.push(Type::Ref);
    }
    while slot_types.len() < callee_nlocals + callee_stack_only {
        slot_types.push(Type::Ref);
    }
    slot_types
}

fn synthesize_fresh_callee_entry_args(
    ctx: &mut TraceCtx,
    callee_frame: OpRef,
    args: &[OpRef],
    callee_nlocals: usize,
) -> Vec<OpRef> {
    // Fresh entry: next_instr=0, valuestackdepth=nlocals (empty stack)
    let mut ca_args = vec![
        callee_frame,
        ctx.const_int(0),
        ctx.const_int(callee_nlocals as i64),
    ];
    ca_args.extend(args.iter().copied().take(callee_nlocals));
    let null = ctx.const_int(PY_NULL as i64);
    while ca_args.len() < 3 + callee_nlocals {
        ca_args.push(null);
    }
    ca_args
}

impl PyreSym {
    pub(crate) fn new_uninit(frame: OpRef) -> Self {
        Self {
            frame,
            symbolic_locals: Vec::new(),
            symbolic_stack: Vec::new(),
            symbolic_local_types: Vec::new(),
            symbolic_stack_types: Vec::new(),
            pending_next_instr: None,
            locals_cells_stack_array_ref: OpRef::NONE,
            valuestackdepth: 0,
            nlocals: 0,
            symbolic_initialized: false,
            vable_next_instr: OpRef::NONE,
            vable_valuestackdepth: OpRef::NONE,
            symbolic_namespace_slots: std::collections::HashMap::new(),
            vable_array_base: None,
            last_comparison_truth: None,
            last_comparison_concrete_truth: None,
            last_popped_concrete_value: None,
            pending_branch_value: None,
            transient_value_types: std::collections::HashMap::new(),
            concrete_locals: Vec::new(),
            concrete_stack: Vec::new(),
            pending_concrete_push: None,
            // concrete_code and concrete_namespace initialized below
            concrete_code: std::ptr::null(),
            concrete_namespace: std::ptr::null_mut(),
            concrete_execution_context: std::ptr::null(),
        }
    }

    /// Initialize symbolic tracking state on first trace instruction.
    /// Subsequent calls are no-ops (state persists across instructions).
    pub(crate) fn init_symbolic(&mut self, ctx: &mut TraceCtx, concrete_frame: usize) {
        if self.symbolic_initialized {
            return;
        }
        let nlocals = concrete_nlocals(concrete_frame).unwrap_or(0);
        if majit_meta::majit_log_enabled() {
            eprintln!(
                "[jit][init-sym] concrete_frame={:#x} nlocals={} vable_base={:?}",
                concrete_frame, nlocals, self.vable_array_base
            );
        }
        let valuestackdepth = concrete_stack_depth(concrete_frame).unwrap_or(nlocals);
        let stack_only_depth = valuestackdepth.saturating_sub(nlocals);
        self.nlocals = nlocals;
        self.locals_cells_stack_array_ref = if self.vable_array_base.is_some() {
            OpRef::NONE
        } else {
            frame_locals_cells_stack_array(ctx, self.frame)
        };
        self.symbolic_locals = if let Some(base) = self.vable_array_base {
            (0..nlocals).map(|i| OpRef(base + i as u32)).collect()
        } else {
            vec![OpRef::NONE; nlocals]
        };
        if self.symbolic_local_types.len() != nlocals {
            self.symbolic_local_types = concrete_slot_types(concrete_frame, nlocals, nlocals);
        }
        self.symbolic_stack = if let Some(base) = self.vable_array_base {
            let stack_base = base + nlocals as u32;
            (0..stack_only_depth)
                .map(|i| OpRef(stack_base + i as u32))
                .collect()
        } else {
            vec![OpRef::NONE; stack_only_depth]
        };
        if self.symbolic_stack_types.len() != stack_only_depth {
            self.symbolic_stack_types =
                concrete_slot_types(concrete_frame, nlocals, valuestackdepth)
                    .into_iter()
                    .skip(nlocals)
                    .collect();
        }
        self.pending_next_instr = None;
        self.valuestackdepth = valuestackdepth;
        // MIFrame concrete Box tracking: populate concrete value arrays
        // from the concrete frame snapshot (RPython MIFrame.setup_call parity).
        self.concrete_locals = (0..nlocals)
            .map(|i| {
                ConcreteValue::from_pyobj(
                    concrete_stack_value(concrete_frame, i).unwrap_or(PY_NULL),
                )
            })
            .collect();
        self.concrete_stack = (0..stack_only_depth)
            .map(|i| {
                ConcreteValue::from_pyobj(
                    concrete_stack_value(concrete_frame, nlocals + i).unwrap_or(PY_NULL),
                )
            })
            .collect();
        // Extract frame metadata pointers for use without concrete_frame
        if concrete_frame != 0 {
            let frame = unsafe { &*(concrete_frame as *const pyre_interp::frame::PyFrame) };
            self.concrete_code = frame.code;
            self.concrete_namespace = frame.namespace;
            self.concrete_execution_context = frame.execution_context;
        }
        self.symbolic_initialized = true;
    }

    /// Stack-only depth (number of values on the operand stack).
    #[inline]
    pub(crate) fn stack_only_depth(&self) -> usize {
        self.valuestackdepth.saturating_sub(self.nlocals)
    }

    pub(crate) fn value_type_of(&self, value: OpRef) -> Type {
        if value.is_none() {
            return Type::Ref;
        }
        if let Some(value_type) = self.transient_value_types.get(&value).copied() {
            return value_type;
        }
        if let Some(idx) = self.symbolic_locals.iter().position(|&slot| slot == value) {
            return self
                .symbolic_local_types
                .get(idx)
                .copied()
                .unwrap_or(Type::Ref);
        }
        let stack_only = self.stack_only_depth().min(self.symbolic_stack.len());
        if let Some(idx) = self.symbolic_stack[..stack_only]
            .iter()
            .position(|&slot| slot == value)
        {
            return self
                .symbolic_stack_types
                .get(idx)
                .copied()
                .unwrap_or(Type::Ref);
        }
        Type::Ref
    }

    /// Read a concrete value from the Box arrays using an absolute
    /// unified-array index (0..nlocals = locals, nlocals.. = stack).
    pub(crate) fn concrete_value_at(&self, abs_idx: usize) -> ConcreteValue {
        if abs_idx < self.nlocals {
            self.concrete_locals
                .get(abs_idx)
                .copied()
                .unwrap_or(ConcreteValue::Null)
        } else {
            let stack_idx = abs_idx - self.nlocals;
            self.concrete_stack
                .get(stack_idx)
                .copied()
                .unwrap_or(ConcreteValue::Null)
        }
    }

    /// Read as PyObjectRef (convenience for legacy code).
    pub(crate) fn concrete_pyobj_at(&self, abs_idx: usize) -> PyObjectRef {
        self.concrete_value_at(abs_idx).to_pyobj()
    }
}

impl TraceFrameState {
    /// Get the concrete return value from the frame's stack top.
    fn concrete_stack_value_at_return(&self) -> Option<PyObjectRef> {
        // MIFrame Box tracking: read from concrete_stack
        let s = self.sym();
        if s.valuestackdepth > 0 {
            let v = s.concrete_value_at(s.valuestackdepth - 1);
            if !v.is_null() {
                return Some(v.to_pyobj());
            }
        }
        // Fallback: read from concrete frame snapshot
        concrete_return_value(self.concrete_frame)
    }

    fn next_instruction_consumes_comparison_truth(&self) -> bool {
        let code = unsafe { &*self.sym().concrete_code };
        // RPython optimize_goto_if_not works on the semantic successor,
        // not on bytecode trivia like EXTENDED_ARG/NOT_TAKEN/CACHE.
        let mut pc = self.fallthrough_pc;
        loop {
            match decode_instruction_at(code, pc) {
                Some((instruction, _))
                    if instruction_is_trivia_between_compare_and_branch(instruction) =>
                {
                    pc += 1
                }
                Some((instruction, _)) => {
                    return instruction_consumes_comparison_truth(instruction);
                }
                None => return false,
            }
        }
    }

    pub(crate) fn from_sym(
        ctx: &mut TraceCtx,
        sym: &mut PyreSym,
        concrete_frame: usize,
        fallthrough_pc: usize,
    ) -> Self {
        debug_assert!(
            concrete_frame != 0,
            "concrete_frame must be a valid frame pointer"
        );
        sym.init_symbolic(ctx, concrete_frame);
        Self {
            ctx,
            sym,
            concrete_frame,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        }
    }

    pub(crate) fn ctx(&mut self) -> &mut TraceCtx {
        unsafe { &mut *self.ctx }
    }

    fn with_ctx<R>(&mut self, f: impl FnOnce(&mut Self, &mut TraceCtx) -> R) -> R {
        let ctx = self.ctx;
        unsafe { f(self, &mut *ctx) }
    }

    #[inline]
    fn sym(&self) -> &PyreSym {
        unsafe { &*self.sym }
    }

    #[inline]
    fn sym_mut(&mut self) -> &mut PyreSym {
        unsafe { &mut *self.sym }
    }

    pub(crate) fn frame(&self) -> OpRef {
        self.sym().frame
    }

    /// PyPy interp_jit.py:89 — hint(self.valuestackdepth, promote=True).
    /// No-op: GuardValue on vsd causes segfault in finish traces.
    /// The virtualizable mechanism implicitly promotes via JUMP args.
    pub(crate) fn promote_valuestackdepth(&mut self, _concrete_frame: usize) {}

    fn materialize_fail_arg_slot(
        &mut self,
        ctx: &mut TraceCtx,
        slot: OpRef,
        slot_type: Type,
        abs_idx: usize,
    ) -> OpRef {
        if !slot.is_none() {
            return slot;
        }
        let concrete_value = self.concrete_at_or_frame(abs_idx).unwrap_or(PY_NULL);
        let typed_value = extract_concrete_typed_value(slot_type, concrete_value);
        fail_arg_opref_for_typed_value(ctx, typed_value)
    }

    /// Build fail_args for the current frame only (no multi-frame header).
    /// RPython resume.py keeps holes out of live failargs; for pyre's
    /// full-frame snapshot, materialize any symbolic holes from the concrete
    /// frame before recording the guard.
    fn build_single_frame_fail_args(&mut self, ctx: &mut TraceCtx) -> Vec<OpRef> {
        self.flush_to_frame(ctx);
        let (
            frame,
            next_instr,
            stack_depth,
            nlocals,
            local_values,
            local_types,
            stack_values,
            stack_types,
        ) = {
            let s = self.sym();
            let stack_only = s.stack_only_depth();
            (
                s.frame,
                s.vable_next_instr,
                s.vable_valuestackdepth,
                s.nlocals,
                s.symbolic_locals.clone(),
                s.symbolic_local_types.clone(),
                s.symbolic_stack[..stack_only.min(s.symbolic_stack.len())].to_vec(),
                s.symbolic_stack_types[..stack_only.min(s.symbolic_stack_types.len())].to_vec(),
            )
        };
        let mut fa = vec![frame, next_instr, stack_depth];
        for (idx, slot) in local_values.into_iter().enumerate() {
            let slot_type = local_types.get(idx).copied().unwrap_or(Type::Ref);
            fa.push(self.materialize_fail_arg_slot(ctx, slot, slot_type, idx));
        }
        for (stack_idx, slot) in stack_values.into_iter().enumerate() {
            let slot_type = stack_types.get(stack_idx).copied().unwrap_or(Type::Ref);
            fa.push(self.materialize_fail_arg_slot(ctx, slot, slot_type, nlocals + stack_idx));
        }
        fa
    }

    fn build_single_frame_fail_arg_types(&self) -> Vec<Type> {
        let s = self.sym();
        let stack_only = s.stack_only_depth();
        if s.vable_array_base.is_some() {
            // RPython parity: virtualizable array slots are always GCREF.
            // Both locals and stack slots must be Ref in fail_arg_types
            // so that guard-failure restoration boxes raw values correctly.
            let total_slots = s.symbolic_local_types.len() + stack_only;
            virtualizable_fail_arg_types(std::iter::repeat_n(Type::Ref, total_slots))
        } else {
            let slot_types = s.symbolic_local_types.iter().copied().chain(
                s.symbolic_stack_types[..stack_only.min(s.symbolic_stack_types.len())]
                    .iter()
                    .copied(),
            );
            virtualizable_fail_arg_types(slot_types)
        }
    }

    fn remember_value_type(&mut self, value: OpRef, value_type: Type) {
        if value.is_none() {
            return;
        }
        self.sym_mut()
            .transient_value_types
            .insert(value, value_type);
    }

    fn value_type(&self, value: OpRef) -> Type {
        if value.is_none() {
            return Type::Ref;
        }
        self.sym().value_type_of(value)
    }

    fn push_typed_value(&mut self, _ctx: &mut TraceCtx, value: OpRef, value_type: Type) {
        let s = self.sym_mut();
        let stack_idx = s.stack_only_depth();
        if stack_idx >= s.symbolic_stack.len() {
            s.symbolic_stack.resize(stack_idx + 1, OpRef::NONE);
        }
        if stack_idx >= s.symbolic_stack_types.len() {
            s.symbolic_stack_types.resize(stack_idx + 1, Type::Ref);
        }
        s.symbolic_stack[stack_idx] = value;
        s.symbolic_stack_types[stack_idx] = value_type;
        // Persist type for value_type_of lookups after stack pop
        // (e.g., inline callee return → pop → value_type_of on result).
        if !value.is_none() {
            s.transient_value_types.insert(value, value_type);
        }
        // MIFrame Box tracking: store concrete value alongside symbolic
        let concrete = s
            .pending_concrete_push
            .take()
            .unwrap_or(ConcreteValue::Null);
        if stack_idx >= s.concrete_stack.len() {
            s.concrete_stack.resize(stack_idx + 1, ConcreteValue::Null);
        }
        s.concrete_stack[stack_idx] = concrete;
        s.valuestackdepth += 1;
    }

    /// Push a TracedBox (RPython execute_and_record return value parity).
    /// Bypasses pending_concrete_push — concrete value is directly provided.
    fn push_traced_box(&mut self, _ctx: &mut TraceCtx, traced: TracedBox) {
        let value_type = self.value_type(traced.opref);
        let s = self.sym_mut();
        let stack_idx = s.stack_only_depth();
        if stack_idx >= s.symbolic_stack.len() {
            s.symbolic_stack.resize(stack_idx + 1, OpRef::NONE);
        }
        if stack_idx >= s.symbolic_stack_types.len() {
            s.symbolic_stack_types.resize(stack_idx + 1, Type::Ref);
        }
        s.symbolic_stack[stack_idx] = traced.opref;
        s.symbolic_stack_types[stack_idx] = value_type;
        if stack_idx >= s.concrete_stack.len() {
            s.concrete_stack.resize(stack_idx + 1, ConcreteValue::Null);
        }
        s.concrete_stack[stack_idx] = traced.concrete;
        s.valuestackdepth += 1;
    }

    pub(crate) fn push_value(&mut self, _ctx: &mut TraceCtx, value: OpRef) {
        let value_type = self.value_type(value);
        self.push_typed_value(_ctx, value, value_type);
    }

    pub(crate) fn pop_value(&mut self, ctx: &mut TraceCtx) -> Result<OpRef, PyError> {
        let concrete_frame = self.concrete_frame;
        let s = self.sym_mut();
        let nlocals = s.nlocals;
        let stack_idx = s
            .valuestackdepth
            .checked_sub(nlocals + 1)
            .ok_or_else(|| pyre_runtime::stack_underflow_error("trace opcode"))?;
        if s.symbolic_stack[stack_idx] == OpRef::NONE {
            let abs_idx = nlocals + stack_idx;
            let idx_const = ctx.const_int(abs_idx as i64);
            s.symbolic_stack[stack_idx] =
                trace_array_getitem_value(ctx, s.locals_cells_stack_array_ref, idx_const);
        }
        let value = s.symbolic_stack[stack_idx];
        let value_type = s
            .symbolic_stack_types
            .get(stack_idx)
            .copied()
            .unwrap_or(Type::Ref);
        // MIFrame Box tracking: pop concrete value from concrete_stack
        let concrete_popped = s
            .concrete_stack
            .get(stack_idx)
            .copied()
            .unwrap_or(ConcreteValue::Null);
        // Fall back to concrete_frame only if within snapshot's depth
        let concrete_popped_pyobj = if concrete_popped.is_null() {
            let snapshot_depth = concrete_stack_depth(concrete_frame).unwrap_or(0);
            let abs_idx = s.valuestackdepth.saturating_sub(1);
            if abs_idx < snapshot_depth {
                concrete_stack_value(concrete_frame, abs_idx)
            } else {
                None
            }
        } else {
            Some(concrete_popped.to_pyobj())
        };
        s.valuestackdepth -= 1;
        s.last_popped_concrete_value = concrete_popped_pyobj;
        s.transient_value_types.insert(value, value_type);
        Ok(value)
    }

    pub(crate) fn peek_value(
        &mut self,
        ctx: &mut TraceCtx,
        depth: usize,
    ) -> Result<OpRef, PyError> {
        let s = self.sym_mut();
        let nlocals = s.nlocals;
        let stack_idx = s
            .valuestackdepth
            .checked_sub(nlocals + depth + 1)
            .ok_or_else(|| pyre_runtime::stack_underflow_error("trace peek"))?;
        if s.symbolic_stack[stack_idx] == OpRef::NONE {
            let abs_idx = nlocals + stack_idx;
            let idx_const = ctx.const_int(abs_idx as i64);
            s.symbolic_stack[stack_idx] =
                trace_array_getitem_value(ctx, s.locals_cells_stack_array_ref, idx_const);
        }
        let value = s.symbolic_stack[stack_idx];
        let value_type = s
            .symbolic_stack_types
            .get(stack_idx)
            .copied()
            .unwrap_or(Type::Ref);
        s.transient_value_types.insert(value, value_type);
        Ok(value)
    }

    fn push_call_replay_stack(
        &mut self,
        ctx: &mut TraceCtx,
        callable: OpRef,
        args: &[OpRef],
        call_pc: usize,
    ) {
        let null = ctx.const_int(pyre_object::PY_NULL as i64);
        self.push_value(ctx, callable);
        self.push_value(ctx, null);
        for &arg in args {
            self.push_value(ctx, arg);
        }
        self.sym_mut().pending_next_instr = Some(call_pc);
    }

    fn pop_call_replay_stack(
        &mut self,
        ctx: &mut TraceCtx,
        args_len: usize,
    ) -> Result<(), PyError> {
        for _ in 0..(2 + args_len) {
            let _ = self.pop_value(ctx)?;
        }
        self.sym_mut().pending_next_instr = None;
        Ok(())
    }

    pub(crate) fn swap_values(&mut self, ctx: &mut TraceCtx, depth: usize) -> Result<(), PyError> {
        let s = self.sym_mut();
        let stack_only = s.stack_only_depth();
        if depth == 0 || stack_only < depth {
            return Err(PyError::type_error("stack underflow during trace swap"));
        }
        let top_idx = stack_only - 1;
        let other_idx = stack_only - depth;
        let nlocals = s.nlocals;
        if s.symbolic_stack[top_idx] == OpRef::NONE {
            let abs_idx = nlocals + top_idx;
            let idx_const = ctx.const_int(abs_idx as i64);
            s.symbolic_stack[top_idx] =
                trace_array_getitem_value(ctx, s.locals_cells_stack_array_ref, idx_const);
        }
        if s.symbolic_stack[other_idx] == OpRef::NONE {
            let abs_idx = nlocals + other_idx;
            let idx_const = ctx.const_int(abs_idx as i64);
            s.symbolic_stack[other_idx] =
                trace_array_getitem_value(ctx, s.locals_cells_stack_array_ref, idx_const);
        }
        s.symbolic_stack.swap(top_idx, other_idx);
        if top_idx < s.symbolic_stack_types.len() && other_idx < s.symbolic_stack_types.len() {
            s.symbolic_stack_types.swap(top_idx, other_idx);
        }
        // MIFrame Box tracking: swap concrete values too
        if top_idx < s.concrete_stack.len() && other_idx < s.concrete_stack.len() {
            s.concrete_stack.swap(top_idx, other_idx);
        }
        Ok(())
    }

    pub(crate) fn load_local_value(
        &mut self,
        ctx: &mut TraceCtx,
        idx: usize,
    ) -> Result<OpRef, PyError> {
        let s = self.sym_mut();
        if idx >= s.symbolic_locals.len() {
            return Err(PyError::type_error("local index out of range in trace"));
        }
        if s.symbolic_locals[idx] == OpRef::NONE {
            if let Some(base) = s.vable_array_base {
                // Virtualizable: locals[idx] was loaded in the preamble
                // and carried as a JUMP arg. OpRef(base + idx) references it.
                s.symbolic_locals[idx] = OpRef(base + idx as u32);
            } else {
                let idx_const = ctx.const_int(idx as i64);
                s.symbolic_locals[idx] =
                    trace_array_getitem_value(ctx, s.locals_cells_stack_array_ref, idx_const);
            }
        }
        let value = s.symbolic_locals[idx];
        let value_type = s
            .symbolic_local_types
            .get(idx)
            .copied()
            .unwrap_or(Type::Ref);
        s.transient_value_types.insert(value, value_type);
        Ok(value)
    }

    pub(crate) fn store_local_value(
        &mut self,
        ctx: &mut TraceCtx,
        idx: usize,
        mut value: OpRef,
    ) -> Result<(), PyError> {
        let vtype = self.value_type(value);
        let concrete_slot_type = self
            .concrete_at_or_frame(idx)
            .map(concrete_virtualizable_slot_type);
        match (concrete_slot_type, vtype) {
            (Some(Type::Int), Type::Ref) => {
                value = self.trace_guarded_int_payload(ctx, value);
            }
            (Some(Type::Float), Type::Ref) => {
                // Inline unbox with record_guard for correct fail_arg types.
                let float_type_addr = &FLOAT_TYPE as *const _ as i64;
                if value.0 < 10_000 {
                    let ob_type =
                        trace_gc_object_int_field(ctx, value, crate::jit::descr::ob_type_descr());
                    let type_const = ctx.const_int(float_type_addr);
                    self.record_guard(ctx, OpCode::GuardClass, &[ob_type, type_const]);
                }
                value = ctx.record_op_with_descr(
                    OpCode::GetfieldGcPureF,
                    &[value],
                    crate::jit::descr::float_floatval_descr(),
                );
            }
            _ => {}
        }
        let stored_type = self.value_type(value);
        let s = self.sym_mut();
        if idx >= s.symbolic_locals.len() {
            return Err(PyError::type_error("local index out of range in trace"));
        }
        s.symbolic_locals[idx] = value;
        if idx >= s.symbolic_local_types.len() {
            s.symbolic_local_types.resize(idx + 1, Type::Ref);
        }
        // Keep the traced value type for subsequent symbolic loads.
        // The concrete virtualizable frame slot may still hold boxed GCREFs,
        // but within the trace a local can legitimately carry a raw int/float
        // until guard/loop materialization re-boxes it at the boundary.
        s.symbolic_local_types[idx] = stored_type;
        Ok(())
    }

    pub(crate) fn load_namespace_value(
        &mut self,
        ctx: &mut TraceCtx,
        idx: usize,
    ) -> Result<OpRef, PyError> {
        if let Some(value) = self.sym().symbolic_namespace_slots.get(&idx).copied() {
            return Ok(value);
        }
        let frame = self.sym().frame;
        let namespace = frame_namespace_ptr(ctx, frame);
        let len = ctx.record_op_with_descr(
            OpCode::GetfieldRawI,
            &[namespace],
            namespace_values_len_descr(),
        );
        self.guard_len_gt_index(ctx, len, idx);
        let values = ctx.record_op_with_descr(
            OpCode::GetfieldRawI,
            &[namespace],
            namespace_values_ptr_descr(),
        );
        let idx_const = ctx.const_int(idx as i64);
        let value = trace_raw_array_getitem_value(ctx, values, idx_const);
        self.sym_mut().symbolic_namespace_slots.insert(idx, value);
        Ok(value)
    }

    pub(crate) fn store_namespace_value(
        &mut self,
        ctx: &mut TraceCtx,
        idx: usize,
        value: OpRef,
    ) -> Result<(), PyError> {
        let frame = self.sym().frame;
        let namespace = frame_namespace_ptr(ctx, frame);
        let len = ctx.record_op_with_descr(
            OpCode::GetfieldRawI,
            &[namespace],
            namespace_values_len_descr(),
        );
        self.guard_len_gt_index(ctx, len, idx);
        let values = ctx.record_op_with_descr(
            OpCode::GetfieldRawI,
            &[namespace],
            namespace_values_ptr_descr(),
        );
        let idx_const = ctx.const_int(idx as i64);
        trace_raw_array_setitem_value(ctx, values, idx_const, value);
        self.sym_mut().symbolic_namespace_slots.insert(idx, value);
        Ok(())
    }

    pub(crate) fn set_next_instr(&mut self, _ctx: &mut TraceCtx, target: usize) {
        self.sym_mut().pending_next_instr = Some(target);
    }

    pub(crate) fn fallthrough_pc(&self) -> usize {
        self.fallthrough_pc
    }

    pub(crate) fn prepare_fallthrough(&mut self) {
        self.sym_mut().pending_next_instr = Some(self.fallthrough_pc);
    }

    /// Update virtualizable next_instr and valuestackdepth before guards / loop close.
    /// Locals and stack are carried through JUMP args (virtualizable), not flushed to heap.
    pub(crate) fn flush_to_frame(&mut self, ctx: &mut TraceCtx) {
        let fallthrough = self.fallthrough_pc;
        let s = self.sym_mut();
        let pending_pc = s.pending_next_instr.take().or(Some(fallthrough));
        if let Some(pc) = pending_pc {
            s.vable_next_instr = ctx.const_int(pc as i64);
        }
        // valuestackdepth is unconditionally synced (const_int is cheap — deduped by HashMap).
        s.vable_valuestackdepth = ctx.const_int(s.valuestackdepth as i64);
    }

    fn sync_standard_virtualizable_before_residual_call(&mut self, ctx: &mut TraceCtx) {
        let Some(vable_ref) = ctx.standard_virtualizable_box() else {
            return;
        };
        ctx.gen_store_back_in_vable(vable_ref);
        let info = build_pyframe_virtualizable_info();
        let obj_ptr = self.concrete_frame as *mut u8;
        unsafe {
            info.tracing_before_residual_call(obj_ptr);
        }
        let force_token = ctx.force_token();
        ctx.vable_setfield_descr(vable_ref, force_token, info.token_field_descr());
    }

    fn sync_standard_virtualizable_after_residual_call(&self) -> bool {
        let info = build_pyframe_virtualizable_info();
        let obj_ptr = self.concrete_frame as *mut u8;
        unsafe { info.tracing_after_residual_call(obj_ptr) }
    }

    /// Loop-carried values must follow the typed live-state contract used by
    /// PyreMeta::slot_types / restore_values().
    ///
    /// In pyre's typed INT/REF/FLOAT model, integer locals cross a loop JUMP
    /// as raw Int values, not freshly boxed W_Int objects.
    fn materialize_loop_carried_value(
        &mut self,
        ctx: &mut TraceCtx,
        value: OpRef,
        slot_type: Type,
    ) -> OpRef {
        match slot_type {
            Type::Int => match self.value_type(value) {
                Type::Int => value,
                Type::Ref => {
                    // Convert boxed W_Int back to its raw payload so the loop
                    // header sees the typed INT stream expected by restore_values().
                    self.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, value))
                }
                _ => value,
            },
            Type::Ref => match self.value_type(value) {
                Type::Int => {
                    // Virtualizable slots are Ref — re-box raw Int for the
                    // loop header which expects boxed W_IntObject.
                    let int_type_addr = &INT_TYPE as *const _ as i64;
                    crate::jit::generated::trace_box_int(
                        ctx,
                        value,
                        w_int_size_descr(),
                        ob_type_descr(),
                        int_intval_descr(),
                        int_type_addr,
                    )
                }
                Type::Float => {
                    let float_type_addr = &FLOAT_TYPE as *const _ as i64;
                    crate::trace_box_float(
                        ctx,
                        value,
                        w_float_size_descr(),
                        ob_type_descr(),
                        float_floatval_descr(),
                        float_type_addr,
                    )
                }
                _ => value,
            },
            _ => value,
        }
    }

    pub(crate) fn close_loop_args(&mut self, ctx: &mut TraceCtx) -> Vec<OpRef> {
        // RPython parity: loop-carried live state comes from the current
        // virtualizable frame state at the merge point. If symbolic stack
        // accounting drifted during tracing, resync depth/shape from the
        // concrete frame before materializing JUMP args.
        // MIFrame Box tracking: use PyreSym's tracked values, not snapshot.
        let concrete_frame = self.concrete_frame;
        let concrete_nlocals = self.sym().nlocals;
        let concrete_vsd = self.sym().valuestackdepth.max(concrete_nlocals);
        {
            let s = self.sym_mut();
            s.nlocals = concrete_nlocals;
            s.valuestackdepth = concrete_vsd;
            let stack_only = s.stack_only_depth();
            if s.symbolic_local_types.len() != concrete_nlocals {
                s.symbolic_local_types =
                    concrete_slot_types(concrete_frame, concrete_nlocals, concrete_nlocals);
            }
            if s.symbolic_stack_types.len() != stack_only {
                s.symbolic_stack_types =
                    concrete_slot_types(concrete_frame, concrete_nlocals, concrete_vsd)
                        .into_iter()
                        .skip(concrete_nlocals)
                        .collect();
            }
            if s.symbolic_stack.len() < stack_only {
                s.symbolic_stack.resize(stack_only, OpRef::NONE);
            }
        }
        self.flush_to_frame(ctx);
        // If nlocals was lost (e.g., inline tracing reset symbolic_initialized),
        // re-derive from concrete frame. RPython keeps virtualizable_boxes in
        // sync across all frames; pyre needs this fallback.
        // If nlocals was lost (inline tracing reset), recover from
        // the trace's num_inputs. RPython carries virtualizable_boxes
        // across frames; pyre derives nlocals from inputarg count.
        {
            let num_inputs = ctx.num_inputs();
            let s = self.sym_mut();
            if s.nlocals == 0 && s.vable_array_base.is_some() {
                let base = s.vable_array_base.unwrap() as usize;
                let nlocals = num_inputs.saturating_sub(base);
                if nlocals > 0 {
                    s.nlocals = nlocals;
                    s.symbolic_locals = (0..nlocals)
                        .map(|i| OpRef(base as u32 + i as u32))
                        .collect();
                    if s.symbolic_local_types.len() != nlocals {
                        s.symbolic_local_types.resize(nlocals, Type::Ref);
                    }
                }
            }
        }
        let (frame, next_instr, stack_depth, nlocals, locals, stack, local_types, stack_types) = {
            let s = self.sym();
            let stack_only = s.stack_only_depth();
            (
                s.frame,
                s.vable_next_instr,
                s.vable_valuestackdepth,
                s.nlocals,
                s.symbolic_locals.clone(),
                s.symbolic_stack[..stack_only.min(s.symbolic_stack.len())].to_vec(),
                s.symbolic_local_types.clone(),
                s.symbolic_stack_types[..stack_only.min(s.symbolic_stack_types.len())].to_vec(),
            )
        };
        let mut args = vec![frame, next_instr, stack_depth];
        for (idx, value) in locals.into_iter().enumerate() {
            let slot_type = local_types.get(idx).copied().unwrap_or(Type::Ref);
            args.push(self.materialize_loop_carried_value(ctx, value, slot_type));
        }
        for (stack_idx, value) in stack.into_iter().enumerate() {
            let slot_type = stack_types.get(stack_idx).copied().unwrap_or(Type::Ref);
            args.push(self.materialize_loop_carried_value(ctx, value, slot_type));
        }
        // pyjitpl.py:2954 remove_consts_and_duplicates: replace constant
        // and duplicate OpRefs with SameAs ops. This ensures each JUMP arg
        // is a unique non-constant reference, matching the optimizer's
        // expectation that JUMP args are live input values.
        let mut slot_types: Vec<Type> = Vec::with_capacity(args.len() - 3);
        slot_types.extend(local_types.iter().copied());
        slot_types.extend(stack_types.iter().copied());
        Self::remove_consts_and_duplicates(&mut args, 3, &slot_types, ctx);
        args
    }

    /// pyjitpl.py:2934 remove_consts_and_duplicates
    fn remove_consts_and_duplicates(
        args: &mut [OpRef],
        start: usize,
        slot_types: &[Type],
        ctx: &mut TraceCtx,
    ) {
        let mut seen = std::collections::HashSet::new();
        for i in start..args.len() {
            let opref = args[i];
            if opref == OpRef::NONE {
                continue;
            }
            let is_const = opref.0 >= 10_000;
            if is_const || !seen.insert(opref) {
                let tp = slot_types.get(i - start).copied().unwrap_or(Type::Ref);
                let same_as_opcode = majit_ir::OpCode::same_as_for_type(tp);
                let fresh = ctx.record_op(same_as_opcode, &[opref]);
                args[i] = fresh;
            }
        }
    }

    /// Build the current fail_args for guards: [frame, ni, vsd, locals..., stack...]
    /// RPython pyjitpl.py capture_resumedata: always captures full frame state.
    pub(crate) fn current_fail_args(&mut self, ctx: &mut TraceCtx) -> Vec<OpRef> {
        if let Some(ref pfa) = self.parent_fail_args {
            return pfa.clone();
        }
        self.build_single_frame_fail_args(ctx)
    }

    /// PyPy generate_guard + capture_resumedata: uses current_fail_args
    /// which encodes the full framestack for multi-frame resume.
    pub(crate) fn record_guard(&mut self, ctx: &mut TraceCtx, opcode: OpCode, args: &[OpRef]) {
        // If parent_fail_args is set (inlined callee), use parent's
        // state for guard recovery instead of callee's vable fields.
        if let Some(ref pfa) = self.parent_fail_args {
            let types = self
                .parent_fail_arg_types
                .clone()
                .unwrap_or_else(|| fail_arg_types_for_virtualizable_state(pfa.len()));
            ctx.record_guard_typed_with_fail_args(opcode, args, types, pfa);
            return;
        }

        if let Some(branch_value) = self.sym().pending_branch_value {
            let truth = args.first().copied().unwrap_or(OpRef::NONE);
            let concrete_truth = opcode == OpCode::GuardTrue;
            self.record_branch_guard(ctx, branch_value, truth, concrete_truth);
            return;
        }

        self.flush_to_frame(ctx);
        let fail_arg_types = self.build_single_frame_fail_arg_types();
        let fail_args = self.build_single_frame_fail_args(ctx);
        ctx.record_guard_typed_with_fail_args(opcode, args, fail_arg_types, &fail_args);
    }

    pub(crate) fn guard_value(&mut self, ctx: &mut TraceCtx, value: OpRef, expected: i64) {
        let expected = ctx.const_int(expected);
        self.record_guard(ctx, OpCode::GuardValue, &[value, expected]);
    }

    /// Guard value == expected Ref constant (heap pointer).
    /// Uses const_ref so guard fail_args preserve the Ref type.
    pub(crate) fn guard_value_ref(&mut self, ctx: &mut TraceCtx, value: OpRef, expected: i64) {
        let expected = ctx.const_ref(expected);
        self.record_guard(ctx, OpCode::GuardValue, &[value, expected]);
    }

    pub(crate) fn guard_nonnull(&mut self, ctx: &mut TraceCtx, value: OpRef) {
        self.record_guard(ctx, OpCode::GuardNonnull, &[value]);
    }

    pub(crate) fn guard_range_iter(&mut self, ctx: &mut TraceCtx, obj: OpRef) {
        self.guard_object_class(ctx, obj, &RANGE_ITER_TYPE as *const PyType);
    }

    pub(crate) fn record_for_iter_guard(
        &mut self,
        ctx: &mut TraceCtx,
        next: OpRef,
        continues: bool,
    ) {
        // RPython range/xrange iter traces carry the next value as a raw int,
        // not an optional boxed object. Only ref-typed iterators need the
        // optional-value guard here.
        if self.value_type(next) != Type::Ref {
            return;
        }
        let opcode = if continues {
            OpCode::GuardNonnull
        } else {
            OpCode::GuardIsnull
        };
        self.record_guard(ctx, opcode, &[next]);
    }

    pub(crate) fn record_branch_guard(
        &mut self,
        ctx: &mut TraceCtx,
        branch_value: OpRef,
        truth: OpRef,
        concrete_truth: bool,
    ) {
        let opcode = if concrete_truth {
            OpCode::GuardTrue
        } else {
            OpCode::GuardFalse
        };
        if self.parent_fail_args.is_some() {
            self.record_guard(ctx, opcode, &[truth]);
            self.sym_mut().last_popped_concrete_value = None;
            return;
        }

        self.flush_to_frame(ctx);
        let concrete_next_instr = self.sym().pending_next_instr.unwrap_or(self.fallthrough_pc);
        let (
            frame,
            next_instr,
            nlocals,
            local_values,
            local_types,
            stack_values,
            stack_types,
            stack_depth,
        ) = {
            let s = self.sym();
            let stack_only = s.stack_only_depth();
            (
                s.frame,
                if s.vable_next_instr.is_none() {
                    ctx.const_int(concrete_next_instr as i64)
                } else {
                    s.vable_next_instr
                },
                s.nlocals,
                s.symbolic_locals.clone(),
                s.symbolic_local_types.clone(),
                s.symbolic_stack[..stack_only.min(s.symbolic_stack.len())].to_vec(),
                s.symbolic_stack_types[..stack_only.min(s.symbolic_stack_types.len())].to_vec(),
                s.valuestackdepth,
            )
        };

        // RPython pyjitpl.py:514: guard fail_args capture the state AFTER
        // the branch is resolved (comparison result already consumed).
        // The resume PC (next_instr) is the branch target, and vsd does not
        // include the comparison result (already popped by pop_jump_if).
        let mut fail_args = vec![frame, next_instr, ctx.const_int(stack_depth as i64)];
        for (idx, slot) in local_values.into_iter().enumerate() {
            let slot_type = local_types.get(idx).copied().unwrap_or(Type::Ref);
            fail_args.push(self.materialize_fail_arg_slot(ctx, slot, slot_type, idx));
        }
        for (stack_idx, slot) in stack_values.into_iter().enumerate() {
            let slot_type = stack_types.get(stack_idx).copied().unwrap_or(Type::Ref);
            fail_args.push(self.materialize_fail_arg_slot(
                ctx,
                slot,
                slot_type,
                nlocals + stack_idx,
            ));
        }

        let fail_arg_types =
            virtualizable_fail_arg_types(local_types.into_iter().chain(stack_types));
        ctx.record_guard_typed_with_fail_args(opcode, &[truth], fail_arg_types, &fail_args);
        if self.sym().pending_branch_value.is_none() {
            self.sym_mut().last_popped_concrete_value = None;
        }
    }

    fn concrete_popped_value(&self) -> Option<PyObjectRef> {
        self.sym()
            .last_popped_concrete_value
            .or_else(|| self.concrete_at_or_frame(self.sym().valuestackdepth))
    }

    /// Read a concrete value from Box arrays, with safe fallback to concrete_frame.
    /// In full-loop tracing, the concrete_frame snapshot may be stale (its stack
    /// depth doesn't advance), so we only fall back for indices within the
    /// snapshot's actual depth.
    fn concrete_at_or_frame(&self, abs_idx: usize) -> Option<PyObjectRef> {
        let v = self.sym().concrete_value_at(abs_idx);
        if !v.is_null() {
            return Some(v.to_pyobj());
        }
        // Safe fallback: only read from concrete_frame if the index is within
        // the snapshot's actual stack depth (avoids underflow/OOB).
        let snapshot_depth = concrete_stack_depth(self.concrete_frame).unwrap_or(0);
        if abs_idx < snapshot_depth {
            concrete_stack_value(self.concrete_frame, abs_idx)
        } else {
            None
        }
    }

    fn concrete_binary_operands(&self) -> Option<(PyObjectRef, PyObjectRef)> {
        let vsd = self.sym().valuestackdepth;
        Some((
            self.concrete_at_or_frame(vsd)?,
            self.concrete_at_or_frame(vsd + 1)?,
        ))
    }

    fn concrete_store_subscr_operands(&self) -> Option<(PyObjectRef, PyObjectRef, PyObjectRef)> {
        let vsd = self.sym().valuestackdepth;
        Some((
            self.concrete_at_or_frame(vsd)?,
            self.concrete_at_or_frame(vsd + 1)?,
            self.concrete_at_or_frame(vsd + 2)?,
        ))
    }

    fn guard_int_object_value(&mut self, ctx: &mut TraceCtx, int_obj: OpRef, expected: i64) {
        self.guard_object_class(ctx, int_obj, &INT_TYPE as *const PyType);
        let actual_value = trace_gc_object_int_field(ctx, int_obj, int_intval_descr());
        self.guard_value(ctx, actual_value, expected);
    }

    fn guard_int_like_value(&mut self, ctx: &mut TraceCtx, value: OpRef, expected: i64) {
        if self.value_type(value) == Type::Int {
            self.guard_value(ctx, value, expected);
        } else {
            self.guard_int_object_value(ctx, value, expected);
        }
    }

    fn guard_object_class(&mut self, ctx: &mut TraceCtx, obj: OpRef, expected_type: *const PyType) {
        let expected_type = ctx.const_int(expected_type as usize as i64);
        self.record_guard(ctx, OpCode::GuardNonnullClass, &[obj, expected_type]);
    }

    fn trace_guarded_int_payload(&mut self, ctx: &mut TraceCtx, int_obj: OpRef) -> OpRef {
        if self.value_type(int_obj) == Type::Int {
            return int_obj;
        }
        self.guard_object_class(ctx, int_obj, &INT_TYPE as *const PyType);
        let raw = trace_gc_object_int_field(ctx, int_obj, int_intval_descr());
        self.remember_value_type(raw, Type::Int);
        raw
    }

    fn concrete_binary_int_operands(&self) -> Option<(i64, i64)> {
        let (lhs, rhs) = self.concrete_binary_operands()?;
        unsafe {
            if is_int(lhs) && is_int(rhs) {
                Some((w_int_get_value(lhs), w_int_get_value(rhs)))
            } else {
                // Values might be LONG_TYPE (bigint) — not inlinable
                None
            }
        }
    }

    fn concrete_binary_float_operands(&self) -> bool {
        let Some((lhs, rhs)) = self.concrete_binary_operands() else {
            return false;
        };
        unsafe { is_float(lhs) && is_float(rhs) }
    }

    fn concrete_unary_int_operand(&self) -> Option<i64> {
        let value = self.concrete_popped_value()?;
        unsafe {
            if is_int(value) {
                Some(w_int_get_value(value))
            } else {
                None
            }
        }
    }

    fn guard_len_gt_index(&mut self, ctx: &mut TraceCtx, len: OpRef, index: usize) {
        let index = ctx.const_int(index as i64);
        let in_bounds = ctx.record_op(OpCode::IntGt, &[len, index]);
        self.record_guard(ctx, OpCode::GuardTrue, &[in_bounds]);
    }

    fn guard_len_eq(&mut self, ctx: &mut TraceCtx, len: OpRef, expected: usize) {
        self.guard_value(ctx, len, expected as i64);
    }

    fn guard_list_strategy(&mut self, ctx: &mut TraceCtx, obj: OpRef, expected: i64) {
        let strategy = trace_gc_object_int_field(ctx, obj, list_strategy_descr());
        self.guard_value(ctx, strategy, expected);
    }

    /// PyPy list strategies index directly into unwrapped storage with the
    /// runtime integer index; they do not specialize every list access to an
    /// exact constant key. We follow that model here and only guard the
    /// key's sign/bounds for the current trace.
    fn trace_dynamic_list_index(
        &mut self,
        ctx: &mut TraceCtx,
        key: OpRef,
        len: OpRef,
        concrete_key: i64,
    ) -> OpRef {
        let raw_index = if self.value_type(key) == Type::Int {
            key
        } else {
            let fail_args = self.current_fail_args(ctx);
            crate::jit::generated::trace_unbox_int(
                ctx,
                key,
                &INT_TYPE as *const _ as i64,
                ob_type_descr(),
                int_intval_descr(),
                &fail_args,
            )
        };
        let zero = ctx.const_int(0);
        if concrete_key >= 0 {
            let nonnegative = ctx.record_op(OpCode::IntGe, &[raw_index, zero]);
            self.record_guard(ctx, OpCode::GuardTrue, &[nonnegative]);
            let in_bounds = ctx.record_op(OpCode::IntLt, &[raw_index, len]);
            self.record_guard(ctx, OpCode::GuardTrue, &[in_bounds]);
            raw_index
        } else {
            let negative = ctx.record_op(OpCode::IntLt, &[raw_index, zero]);
            self.record_guard(ctx, OpCode::GuardTrue, &[negative]);
            let normalized = ctx.record_op(OpCode::IntAddOvf, &[len, raw_index]);
            self.record_guard(ctx, OpCode::GuardNoOverflow, &[]);
            let in_bounds = ctx.record_op(OpCode::IntGe, &[normalized, zero]);
            self.record_guard(ctx, OpCode::GuardTrue, &[in_bounds]);
            normalized
        }
    }

    fn trace_direct_tuple_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        expected_type: *const PyType,
        items_ptr_descr: DescrRef,
        items_len_descr: DescrRef,
        concrete_index: usize,
    ) -> OpRef {
        self.guard_object_class(ctx, obj, expected_type);
        self.guard_int_like_value(ctx, key, concrete_index as i64);
        let len = trace_gc_object_int_field(ctx, obj, items_len_descr);
        self.guard_len_gt_index(ctx, len, concrete_index);
        let items_ptr = trace_gc_object_int_field(ctx, obj, items_ptr_descr);
        let index = ctx.const_int(concrete_index as i64);
        trace_raw_array_getitem_value(ctx, items_ptr, index)
    }

    fn trace_direct_negative_tuple_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        expected_type: *const PyType,
        items_ptr_descr: DescrRef,
        items_len_descr: DescrRef,
        concrete_key: i64,
        concrete_len: usize,
    ) -> OpRef {
        let normalized = (concrete_len as i64 + concrete_key) as usize;
        self.guard_object_class(ctx, obj, expected_type);
        self.guard_int_like_value(ctx, key, concrete_key);
        let len = trace_gc_object_int_field(ctx, obj, items_len_descr);
        self.guard_len_eq(ctx, len, concrete_len);
        let items_ptr = trace_gc_object_int_field(ctx, obj, items_ptr_descr);
        let index = ctx.const_int(normalized as i64);
        trace_raw_array_getitem_value(ctx, items_ptr, index)
    }

    fn trace_direct_object_list_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        concrete_index: usize,
    ) -> OpRef {
        self.guard_object_class(ctx, obj, &LIST_TYPE as *const PyType);
        self.guard_list_strategy(ctx, obj, 0);
        let len = trace_gc_object_int_field(ctx, obj, list_items_len_descr());
        let index = self.trace_dynamic_list_index(ctx, key, len, concrete_index as i64);
        let items_ptr = trace_gc_object_int_field(ctx, obj, list_items_ptr_descr());
        trace_raw_array_getitem_value(ctx, items_ptr, index)
    }

    fn trace_direct_negative_object_list_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        concrete_key: i64,
        _concrete_len: usize,
    ) -> OpRef {
        self.guard_object_class(ctx, obj, &LIST_TYPE as *const PyType);
        self.guard_list_strategy(ctx, obj, 0);
        let len = trace_gc_object_int_field(ctx, obj, list_items_len_descr());
        let index = self.trace_dynamic_list_index(ctx, key, len, concrete_key);
        let items_ptr = trace_gc_object_int_field(ctx, obj, list_items_ptr_descr());
        trace_raw_array_getitem_value(ctx, items_ptr, index)
    }

    fn trace_direct_int_list_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        concrete_index: usize,
    ) -> OpRef {
        self.guard_object_class(ctx, obj, &LIST_TYPE as *const PyType);
        self.guard_list_strategy(ctx, obj, 1);
        let len = trace_gc_object_int_field(ctx, obj, list_int_items_len_descr());
        let index = self.trace_dynamic_list_index(ctx, key, len, concrete_index as i64);
        let items_ptr = trace_gc_object_int_field(ctx, obj, list_int_items_ptr_descr());
        let raw = trace_raw_int_array_getitem_value(ctx, items_ptr, index);
        self.remember_value_type(raw, Type::Int);
        raw
    }

    fn trace_direct_negative_int_list_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        concrete_key: i64,
        _concrete_len: usize,
    ) -> OpRef {
        self.guard_object_class(ctx, obj, &LIST_TYPE as *const PyType);
        self.guard_list_strategy(ctx, obj, 1);
        let len = trace_gc_object_int_field(ctx, obj, list_int_items_len_descr());
        let index = self.trace_dynamic_list_index(ctx, key, len, concrete_key);
        let items_ptr = trace_gc_object_int_field(ctx, obj, list_int_items_ptr_descr());
        let raw = trace_raw_int_array_getitem_value(ctx, items_ptr, index);
        self.remember_value_type(raw, Type::Int);
        raw
    }

    fn trace_direct_float_list_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        concrete_index: usize,
    ) -> OpRef {
        self.guard_object_class(ctx, obj, &LIST_TYPE as *const PyType);
        self.guard_list_strategy(ctx, obj, 2);
        let len = trace_gc_object_int_field(ctx, obj, list_float_items_len_descr());
        let index = self.trace_dynamic_list_index(ctx, key, len, concrete_index as i64);
        let items_ptr = trace_gc_object_int_field(ctx, obj, list_float_items_ptr_descr());
        let raw = trace_raw_float_array_getitem_value(ctx, items_ptr, index);
        self.remember_value_type(raw, Type::Float);
        raw
    }

    fn trace_direct_negative_float_list_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        concrete_key: i64,
        _concrete_len: usize,
    ) -> OpRef {
        self.guard_object_class(ctx, obj, &LIST_TYPE as *const PyType);
        self.guard_list_strategy(ctx, obj, 2);
        let len = trace_gc_object_int_field(ctx, obj, list_float_items_len_descr());
        let index = self.trace_dynamic_list_index(ctx, key, len, concrete_key);
        let items_ptr = trace_gc_object_int_field(ctx, obj, list_float_items_ptr_descr());
        let raw = trace_raw_float_array_getitem_value(ctx, items_ptr, index);
        self.remember_value_type(raw, Type::Float);
        raw
    }

    fn trace_unpack_known_sequence(
        &mut self,
        ctx: &mut TraceCtx,
        seq: OpRef,
        count: usize,
        expected_type: *const PyType,
        items_ptr_descr: DescrRef,
        items_len_descr: DescrRef,
    ) -> Vec<OpRef> {
        self.guard_object_class(ctx, seq, expected_type);

        let len = trace_gc_object_int_field(ctx, seq, items_len_descr);
        self.guard_value(ctx, len, count as i64);

        let items_ptr = trace_gc_object_int_field(ctx, seq, items_ptr_descr);
        (0..count)
            .map(|idx| {
                let idx = ctx.const_int(idx as i64);
                trace_raw_array_getitem_value(ctx, items_ptr, idx)
            })
            .collect()
    }

    pub(crate) fn unpack_sequence_value(
        &mut self,
        seq: OpRef,
        count: usize,
    ) -> Result<Vec<OpRef>, PyError> {
        let Some(concrete_seq) = self.concrete_popped_value() else {
            return TraceHelperAccess::trace_unpack_sequence(self, seq, count);
        };

        self.with_ctx(|this, ctx| unsafe {
            if is_tuple(concrete_seq) && w_tuple_len(concrete_seq) == count {
                return Ok(this.trace_unpack_known_sequence(
                    ctx,
                    seq,
                    count,
                    &TUPLE_TYPE as *const PyType,
                    tuple_items_ptr_descr(),
                    tuple_items_len_descr(),
                ));
            }
            if is_list(concrete_seq)
                && w_list_uses_object_storage(concrete_seq)
                && w_list_len(concrete_seq) == count
            {
                return Ok(this.trace_unpack_known_sequence(
                    ctx,
                    seq,
                    count,
                    &LIST_TYPE as *const PyType,
                    list_items_ptr_descr(),
                    list_items_len_descr(),
                ));
            }
            TraceHelperAccess::trace_unpack_sequence(this, seq, count)
        })
    }

    pub(crate) fn binary_subscr_value(&mut self, a: OpRef, b: OpRef) -> Result<OpRef, PyError> {
        let Some((concrete_obj, concrete_key)) = self.concrete_binary_operands() else {
            return self.trace_binary_value(a, b, BinaryOperator::Subscr);
        };
        // MIFrame Box tracking: compute concrete subscr result
        if let Ok(result) = pyre_objspace::space::py_getitem(concrete_obj, concrete_key) {
            self.sym_mut().pending_concrete_push = Some(ConcreteValue::from_pyobj(result));
        }

        unsafe {
            if is_int(concrete_key) {
                let index = w_int_get_value(concrete_key);
                return self.with_ctx(|this, ctx| {
                    if is_tuple(concrete_obj) {
                        let concrete_len = w_tuple_len(concrete_obj);
                        if index >= 0 {
                            let index = index as usize;
                            if index < concrete_len {
                                return Ok(this.trace_direct_tuple_getitem(
                                    ctx,
                                    a,
                                    b,
                                    &TUPLE_TYPE as *const PyType,
                                    tuple_items_ptr_descr(),
                                    tuple_items_len_descr(),
                                    index,
                                ));
                            }
                        } else if let Some(abs_index) = index
                            .checked_neg()
                            .and_then(|value| usize::try_from(value).ok())
                        {
                            if abs_index <= concrete_len {
                                return Ok(this.trace_direct_negative_tuple_getitem(
                                    ctx,
                                    a,
                                    b,
                                    &TUPLE_TYPE as *const PyType,
                                    tuple_items_ptr_descr(),
                                    tuple_items_len_descr(),
                                    index,
                                    concrete_len,
                                ));
                            }
                        }
                    } else if is_list(concrete_obj) && w_list_uses_object_storage(concrete_obj) {
                        let concrete_len = w_list_len(concrete_obj);
                        if index >= 0 {
                            let index = index as usize;
                            if index < concrete_len {
                                return Ok(this.trace_direct_object_list_getitem(ctx, a, b, index));
                            }
                        } else if let Some(abs_index) = index
                            .checked_neg()
                            .and_then(|value| usize::try_from(value).ok())
                        {
                            if abs_index <= concrete_len {
                                return Ok(this.trace_direct_negative_object_list_getitem(
                                    ctx,
                                    a,
                                    b,
                                    index,
                                    concrete_len,
                                ));
                            }
                        }
                    } else if is_list(concrete_obj) && w_list_uses_int_storage(concrete_obj) {
                        let concrete_len = w_list_len(concrete_obj);
                        if index >= 0 {
                            let index = index as usize;
                            if index < concrete_len {
                                return Ok(this.trace_direct_int_list_getitem(ctx, a, b, index));
                            }
                        } else if let Some(abs_index) = index
                            .checked_neg()
                            .and_then(|value| usize::try_from(value).ok())
                        {
                            if abs_index <= concrete_len {
                                return Ok(this.trace_direct_negative_int_list_getitem(
                                    ctx,
                                    a,
                                    b,
                                    index,
                                    concrete_len,
                                ));
                            }
                        }
                    } else if is_list(concrete_obj) && w_list_uses_float_storage(concrete_obj) {
                        let concrete_len = w_list_len(concrete_obj);
                        if index >= 0 {
                            let index = index as usize;
                            if index < concrete_len {
                                return Ok(this.trace_direct_float_list_getitem(ctx, a, b, index));
                            }
                        } else if let Some(abs_index) = index
                            .checked_neg()
                            .and_then(|value| usize::try_from(value).ok())
                        {
                            if abs_index <= concrete_len {
                                return Ok(this.trace_direct_negative_float_list_getitem(
                                    ctx,
                                    a,
                                    b,
                                    index,
                                    concrete_len,
                                ));
                            }
                        }
                    }
                    this.trace_binary_value(a, b, BinaryOperator::Subscr)
                });
            }
        }

        self.trace_binary_value(a, b, BinaryOperator::Subscr)
    }

    pub(crate) fn binary_int_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: BinaryOperator,
    ) -> Result<OpRef, PyError> {
        let Some((lhs, rhs)) = self.concrete_binary_int_operands() else {
            return self.trace_binary_value(a, b, op);
        };
        let (lhs_obj, rhs_obj) = self
            .concrete_binary_operands()
            .expect("integer concrete operands should expose boxed operands");

        let op_code = match op {
            BinaryOperator::Add | BinaryOperator::InplaceAdd => OpCode::IntAddOvf,
            BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => OpCode::IntSubOvf,
            BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => OpCode::IntMulOvf,
            BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide => {
                if lhs < 0 || rhs <= 0 {
                    return self.trace_binary_value(a, b, op);
                }
                OpCode::IntFloorDiv
            }
            BinaryOperator::Remainder | BinaryOperator::InplaceRemainder => {
                if lhs < 0 || rhs <= 0 {
                    return self.trace_binary_value(a, b, op);
                }
                OpCode::IntMod
            }
            BinaryOperator::And | BinaryOperator::InplaceAnd => OpCode::IntAnd,
            BinaryOperator::Or | BinaryOperator::InplaceOr => OpCode::IntOr,
            BinaryOperator::Xor | BinaryOperator::InplaceXor => OpCode::IntXor,
            BinaryOperator::Lshift | BinaryOperator::InplaceLshift => {
                let Ok(shift) = u32::try_from(rhs) else {
                    return self.trace_binary_value(a, b, op);
                };
                if shift >= i64::BITS || lhs.checked_shl(shift).is_none() {
                    return self.trace_binary_value(a, b, op);
                }
                OpCode::IntLshift
            }
            BinaryOperator::Rshift | BinaryOperator::InplaceRshift => {
                let Ok(shift) = u32::try_from(rhs) else {
                    return self.trace_binary_value(a, b, op);
                };
                if shift >= i64::BITS {
                    return self.trace_binary_value(a, b, op);
                }
                OpCode::IntRshift
            }
            _ => return self.trace_binary_value(a, b, op),
        };

        // concrete_result no longer needed — generated trace functions
        // handle boxing without concrete values.
        let _ = (lhs, rhs); // suppress unused warnings for edge-case validation above

        let has_overflow = matches!(
            op_code,
            OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf
        );
        self.with_ctx(|this, ctx| {
            let fail_args = this.current_fail_args(ctx);
            let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
            let lhs_raw = if this.value_type(a) == Type::Int {
                a
            } else {
                crate::jit::generated::trace_unbox_int(
                    ctx,
                    a,
                    int_type_addr,
                    crate::jit::descr::ob_type_descr(),
                    crate::jit::descr::int_intval_descr(),
                    &fail_args,
                )
            };
            let rhs_raw = if this.value_type(b) == Type::Int {
                b
            } else {
                crate::jit::generated::trace_unbox_int(
                    ctx,
                    b,
                    int_type_addr,
                    crate::jit::descr::ob_type_descr(),
                    crate::jit::descr::int_intval_descr(),
                    &fail_args,
                )
            };
            let raw_result = ctx.record_op(op_code, &[lhs_raw, rhs_raw]);
            if has_overflow {
                this.record_guard(ctx, OpCode::GuardNoOverflow, &[]);
            }
            // RPython parity: wrapint(space, z) re-boxes the raw int result.
            // Record New(W_IntObject) + SetfieldGc so optimizer can virtualize.
            // pypy/objspace/std/intobject.py:671 wrapint
            let boxed = box_traced_raw_int(ctx, raw_result);
            this.remember_value_type(boxed, Type::Ref);
            Ok(boxed)
        })
    }

    pub(crate) fn binary_float_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: BinaryOperator,
    ) -> Result<OpRef, PyError> {
        let op_code = match op {
            BinaryOperator::Add | BinaryOperator::InplaceAdd => OpCode::FloatAdd,
            BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => OpCode::FloatSub,
            BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => OpCode::FloatMul,
            BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide => {
                OpCode::FloatFloorDiv
            }
            BinaryOperator::Remainder | BinaryOperator::InplaceRemainder => OpCode::FloatMod,
            BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide => OpCode::FloatTrueDiv,
            _ => return self.trace_binary_value(a, b, op),
        };

        self.with_ctx(|this, ctx| {
            let float_type_addr = &FLOAT_TYPE as *const _ as i64;
            // Inline trace_unbox_float but use this.record_guard for correct
            // fail_arg types (generated trace_unbox_float hardcodes all types
            // as Int, losing Float type info needed for guard recovery).
            let unbox_float =
                |this: &mut TraceFrameState, ctx: &mut TraceCtx, obj: OpRef| -> OpRef {
                    // Skip GuardClass for constant objects (ob_type is known at
                    // trace time). RPython optimizer constant-folds these, but
                    // pyre's optimizer doesn't always catch it.
                    if obj.0 < 10_000 {
                        let ob_type =
                            trace_gc_object_int_field(ctx, obj, crate::jit::descr::ob_type_descr());
                        let type_const = ctx.const_int(float_type_addr);
                        this.record_guard(ctx, OpCode::GuardClass, &[ob_type, type_const]);
                    }
                    ctx.record_op_with_descr(
                        OpCode::GetfieldGcPureF,
                        &[obj],
                        crate::jit::descr::float_floatval_descr(),
                    )
                };
            let lhs_raw = if this.value_type(a) == Type::Float {
                a
            } else {
                unbox_float(this, ctx, a)
            };
            let rhs_raw = if this.value_type(b) == Type::Float {
                b
            } else {
                unbox_float(this, ctx, b)
            };
            let result = ctx.record_op(op_code, &[lhs_raw, rhs_raw]);
            this.remember_value_type(result, Type::Float);
            // RPython parity: wrapfloat(space, z) re-boxes the raw float result.
            // pypy/objspace/std/floatobject.py
            let boxed = box_traced_raw_float(ctx, result);
            this.remember_value_type(boxed, Type::Ref);
            Ok(boxed)
        })
    }

    pub(crate) fn compare_value_direct(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: ComparisonOperator,
    ) -> Result<OpRef, PyError> {
        let Some((lhs_obj, rhs_obj)) = self.concrete_binary_operands() else {
            return self.trace_compare_value(a, b, op);
        };

        unsafe {
            if is_int(lhs_obj) && is_int(rhs_obj) {
                let cmp = match op {
                    ComparisonOperator::Less => OpCode::IntLt,
                    ComparisonOperator::LessOrEqual => OpCode::IntLe,
                    ComparisonOperator::Greater => OpCode::IntGt,
                    ComparisonOperator::GreaterOrEqual => OpCode::IntGe,
                    ComparisonOperator::Equal => OpCode::IntEq,
                    ComparisonOperator::NotEqual => OpCode::IntNe,
                };
                return self.with_ctx(|this, ctx| {
                    let fail_args = this.current_fail_args(ctx);
                    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
                    let lhs_raw = if this.value_type(a) == Type::Int {
                        a
                    } else if let Some(raw) = try_trace_const_boxed_int(ctx, a, lhs_obj) {
                        raw
                    } else {
                        crate::jit::generated::trace_unbox_int(
                            ctx,
                            a,
                            int_type_addr,
                            ob_type_descr(),
                            int_intval_descr(),
                            &fail_args,
                        )
                    };
                    let rhs_raw = if this.value_type(b) == Type::Int {
                        b
                    } else if let Some(raw) = try_trace_const_boxed_int(ctx, b, rhs_obj) {
                        raw
                    } else {
                        crate::jit::generated::trace_unbox_int(
                            ctx,
                            b,
                            int_type_addr,
                            ob_type_descr(),
                            int_intval_descr(),
                            &fail_args,
                        )
                    };
                    let truth = ctx.record_op(cmp, &[lhs_raw, rhs_raw]);
                    // RPython goto_if_not fusion: cache truth for
                    // the next POP_JUMP_IF to consume directly.
                    this.sym_mut().last_comparison_truth = Some(truth);
                    this.sym_mut().last_comparison_concrete_truth =
                        Some(objspace_compare_ints(lhs_obj, rhs_obj, op));
                    if this.next_instruction_consumes_comparison_truth() {
                        Ok(truth)
                    } else {
                        Ok(emit_trace_bool_value_from_truth(ctx, truth, false))
                    }
                });
            }
            if is_float(lhs_obj) && is_float(rhs_obj) {
                let cmp = match op {
                    ComparisonOperator::Less => OpCode::FloatLt,
                    ComparisonOperator::LessOrEqual => OpCode::FloatLe,
                    ComparisonOperator::Greater => OpCode::FloatGt,
                    ComparisonOperator::GreaterOrEqual => OpCode::FloatGe,
                    ComparisonOperator::Equal => OpCode::FloatEq,
                    ComparisonOperator::NotEqual => OpCode::FloatNe,
                };
                return self.with_ctx(|this, ctx| {
                    let fail_args = this.current_fail_args(ctx);
                    let float_type_addr = &FLOAT_TYPE as *const _ as i64;
                    let lhs_raw = if this.value_type(a) == Type::Float {
                        a
                    } else {
                        crate::jit::generated::trace_unbox_float(
                            ctx,
                            a,
                            float_type_addr,
                            ob_type_descr(),
                            float_floatval_descr(),
                            &fail_args,
                        )
                    };
                    let rhs_raw = if this.value_type(b) == Type::Float {
                        b
                    } else {
                        crate::jit::generated::trace_unbox_float(
                            ctx,
                            b,
                            float_type_addr,
                            ob_type_descr(),
                            float_floatval_descr(),
                            &fail_args,
                        )
                    };
                    let truth = ctx.record_op(cmp, &[lhs_raw, rhs_raw]);
                    this.sym_mut().last_comparison_truth = Some(truth);
                    this.sym_mut().last_comparison_concrete_truth =
                        Some(objspace_compare_floats(lhs_obj, rhs_obj, op));
                    if this.next_instruction_consumes_comparison_truth() {
                        Ok(truth)
                    } else {
                        Ok(emit_trace_bool_value_from_truth(ctx, truth, false))
                    }
                });
            }
        }

        self.trace_compare_value(a, b, op)
    }

    pub(crate) fn store_subscr_value(
        &mut self,
        obj: OpRef,
        key: OpRef,
        value: OpRef,
    ) -> Result<(), PyError> {
        let Some((concrete_value, concrete_obj, concrete_key)) =
            self.concrete_store_subscr_operands()
        else {
            return self.trace_store_subscr(obj, key, value);
        };

        unsafe {
            if is_list(concrete_obj)
                && w_list_uses_object_storage(concrete_obj)
                && is_int(concrete_key)
            {
                let index = w_int_get_value(concrete_key);
                let concrete_len = w_list_len(concrete_obj);
                if index >= 0 {
                    let index = index as usize;
                    if index < concrete_len {
                        return self.with_ctx(|this, ctx| {
                            this.guard_object_class(ctx, obj, &LIST_TYPE as *const PyType);
                            this.guard_list_strategy(ctx, obj, 0);
                            let len = trace_gc_object_int_field(ctx, obj, list_items_len_descr());
                            let index = this.trace_dynamic_list_index(ctx, key, len, index as i64);
                            let items_ptr =
                                trace_gc_object_int_field(ctx, obj, list_items_ptr_descr());
                            trace_raw_array_setitem_value(ctx, items_ptr, index, value);
                            Ok(())
                        });
                    }
                } else if let Some(abs_index) = index
                    .checked_neg()
                    .and_then(|value| usize::try_from(value).ok())
                {
                    if abs_index <= concrete_len {
                        return self.with_ctx(|this, ctx| {
                            this.guard_object_class(ctx, obj, &LIST_TYPE as *const PyType);
                            this.guard_list_strategy(ctx, obj, 0);
                            let len = trace_gc_object_int_field(ctx, obj, list_items_len_descr());
                            let index = this.trace_dynamic_list_index(ctx, key, len, index);
                            let items_ptr =
                                trace_gc_object_int_field(ctx, obj, list_items_ptr_descr());
                            trace_raw_array_setitem_value(ctx, items_ptr, index, value);
                            Ok(())
                        });
                    }
                }
            } else if is_list(concrete_obj)
                && w_list_uses_int_storage(concrete_obj)
                && is_int(concrete_key)
                && is_int(concrete_value)
            {
                let index = w_int_get_value(concrete_key);
                let concrete_len = w_list_len(concrete_obj);
                if index >= 0 {
                    let index = index as usize;
                    if index < concrete_len {
                        return self.with_ctx(|this, ctx| {
                            this.guard_object_class(ctx, obj, &LIST_TYPE as *const PyType);
                            this.guard_list_strategy(ctx, obj, 1);
                            let len =
                                trace_gc_object_int_field(ctx, obj, list_int_items_len_descr());
                            let index = this.trace_dynamic_list_index(ctx, key, len, index as i64);
                            let items_ptr =
                                trace_gc_object_int_field(ctx, obj, list_int_items_ptr_descr());
                            let fail_args = this.current_fail_args(ctx);
                            let raw = if this.value_type(value) == Type::Int {
                                value
                            } else {
                                crate::jit::generated::trace_unbox_int(
                                    ctx,
                                    value,
                                    &INT_TYPE as *const _ as i64,
                                    ob_type_descr(),
                                    int_intval_descr(),
                                    &fail_args,
                                )
                            };
                            trace_raw_int_array_setitem_value(ctx, items_ptr, index, raw);
                            Ok(())
                        });
                    }
                } else if let Some(abs_index) = index
                    .checked_neg()
                    .and_then(|value| usize::try_from(value).ok())
                {
                    if abs_index <= concrete_len {
                        return self.with_ctx(|this, ctx| {
                            this.guard_object_class(ctx, obj, &LIST_TYPE as *const PyType);
                            this.guard_list_strategy(ctx, obj, 1);
                            let len =
                                trace_gc_object_int_field(ctx, obj, list_int_items_len_descr());
                            let index = this.trace_dynamic_list_index(ctx, key, len, index);
                            let items_ptr =
                                trace_gc_object_int_field(ctx, obj, list_int_items_ptr_descr());
                            let fail_args = this.current_fail_args(ctx);
                            let raw = if this.value_type(value) == Type::Int {
                                value
                            } else {
                                crate::jit::generated::trace_unbox_int(
                                    ctx,
                                    value,
                                    &INT_TYPE as *const _ as i64,
                                    ob_type_descr(),
                                    int_intval_descr(),
                                    &fail_args,
                                )
                            };
                            trace_raw_int_array_setitem_value(ctx, items_ptr, index, raw);
                            Ok(())
                        });
                    }
                }
            } else if is_list(concrete_obj)
                && w_list_uses_float_storage(concrete_obj)
                && is_int(concrete_key)
                && is_float(concrete_value)
            {
                let index = w_int_get_value(concrete_key);
                let concrete_len = w_list_len(concrete_obj);
                if index >= 0 {
                    let index = index as usize;
                    if index < concrete_len {
                        return self.with_ctx(|this, ctx| {
                            this.guard_object_class(ctx, obj, &LIST_TYPE as *const PyType);
                            this.guard_list_strategy(ctx, obj, 2);
                            let len =
                                trace_gc_object_int_field(ctx, obj, list_float_items_len_descr());
                            let index = this.trace_dynamic_list_index(ctx, key, len, index as i64);
                            let items_ptr =
                                trace_gc_object_int_field(ctx, obj, list_float_items_ptr_descr());
                            let fail_args = this.current_fail_args(ctx);
                            let raw = if this.value_type(value) == Type::Float {
                                value
                            } else {
                                crate::jit::generated::trace_unbox_float(
                                    ctx,
                                    value,
                                    &FLOAT_TYPE as *const _ as i64,
                                    ob_type_descr(),
                                    float_floatval_descr(),
                                    &fail_args,
                                )
                            };
                            trace_raw_float_array_setitem_value(ctx, items_ptr, index, raw);
                            Ok(())
                        });
                    }
                } else if let Some(abs_index) = index
                    .checked_neg()
                    .and_then(|value| usize::try_from(value).ok())
                {
                    if abs_index <= concrete_len {
                        return self.with_ctx(|this, ctx| {
                            this.guard_object_class(ctx, obj, &LIST_TYPE as *const PyType);
                            this.guard_list_strategy(ctx, obj, 2);
                            let len =
                                trace_gc_object_int_field(ctx, obj, list_float_items_len_descr());
                            let index = this.trace_dynamic_list_index(ctx, key, len, index);
                            let items_ptr =
                                trace_gc_object_int_field(ctx, obj, list_float_items_ptr_descr());
                            let fail_args = this.current_fail_args(ctx);
                            let raw = if this.value_type(value) == Type::Float {
                                value
                            } else {
                                crate::jit::generated::trace_unbox_float(
                                    ctx,
                                    value,
                                    &FLOAT_TYPE as *const _ as i64,
                                    ob_type_descr(),
                                    float_floatval_descr(),
                                    &fail_args,
                                )
                            };
                            trace_raw_float_array_setitem_value(ctx, items_ptr, index, raw);
                            Ok(())
                        });
                    }
                }
            }
        }

        self.trace_store_subscr(obj, key, value)
    }

    pub(crate) fn list_append_value(&mut self, list: OpRef, value: OpRef) -> Result<(), PyError> {
        let Some(concrete_list) = self.concrete_popped_value() else {
            return self.trace_list_append(list, value);
        };

        unsafe {
            if is_list(concrete_list)
                && w_list_uses_object_storage(concrete_list)
                && w_list_can_append_without_realloc(concrete_list)
            {
                let concrete_len = w_list_len(concrete_list);
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, list, &LIST_TYPE as *const PyType);
                    this.guard_list_strategy(ctx, list, 0);
                    if w_list_is_inline_storage(concrete_list) {
                        let heap_cap = ctx.record_op_with_descr(
                            OpCode::GetfieldGcI,
                            &[list],
                            list_items_heap_cap_descr(),
                        );
                        this.guard_value(ctx, heap_cap, 0);
                    } else {
                        let len = trace_gc_object_int_field(ctx, list, list_items_len_descr());
                        this.guard_value(ctx, len, concrete_len as i64);
                    }
                    let items_ptr = trace_gc_object_int_field(ctx, list, list_items_ptr_descr());
                    let index = ctx.const_int(concrete_len as i64);
                    trace_raw_array_setitem_value(ctx, items_ptr, index, value);
                    let new_len = ctx.const_int((concrete_len + 1) as i64);
                    ctx.record_op_with_descr(
                        OpCode::SetfieldGc,
                        &[list, new_len],
                        list_items_len_descr(),
                    );
                    Ok(())
                });
            } else if is_list(concrete_list)
                && w_list_uses_int_storage(concrete_list)
                && w_list_can_append_without_realloc(concrete_list)
            {
                let concrete_len = w_list_len(concrete_list);
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, list, &LIST_TYPE as *const PyType);
                    this.guard_list_strategy(ctx, list, 1);
                    if w_list_is_inline_storage(concrete_list) {
                        let heap_cap = ctx.record_op_with_descr(
                            OpCode::GetfieldGcI,
                            &[list],
                            list_int_items_heap_cap_descr(),
                        );
                        this.guard_value(ctx, heap_cap, 0);
                    } else {
                        let len = trace_gc_object_int_field(ctx, list, list_int_items_len_descr());
                        this.guard_value(ctx, len, concrete_len as i64);
                    }
                    let items_ptr =
                        trace_gc_object_int_field(ctx, list, list_int_items_ptr_descr());
                    let index = ctx.const_int(concrete_len as i64);
                    let fail_args = this.current_fail_args(ctx);
                    let raw = if this.value_type(value) == Type::Int {
                        value
                    } else {
                        crate::jit::generated::trace_unbox_int(
                            ctx,
                            value,
                            &INT_TYPE as *const _ as i64,
                            ob_type_descr(),
                            int_intval_descr(),
                            &fail_args,
                        )
                    };
                    trace_raw_int_array_setitem_value(ctx, items_ptr, index, raw);
                    let new_len = ctx.const_int((concrete_len + 1) as i64);
                    ctx.record_op_with_descr(
                        OpCode::SetfieldGc,
                        &[list, new_len],
                        list_int_items_len_descr(),
                    );
                    Ok(())
                });
            } else if is_list(concrete_list)
                && w_list_uses_float_storage(concrete_list)
                && w_list_can_append_without_realloc(concrete_list)
            {
                let concrete_len = w_list_len(concrete_list);
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, list, &LIST_TYPE as *const PyType);
                    this.guard_list_strategy(ctx, list, 2);
                    if w_list_is_inline_storage(concrete_list) {
                        let heap_cap = ctx.record_op_with_descr(
                            OpCode::GetfieldGcI,
                            &[list],
                            list_float_items_heap_cap_descr(),
                        );
                        this.guard_value(ctx, heap_cap, 0);
                    } else {
                        let len =
                            trace_gc_object_int_field(ctx, list, list_float_items_len_descr());
                        this.guard_value(ctx, len, concrete_len as i64);
                    }
                    let items_ptr =
                        trace_gc_object_int_field(ctx, list, list_float_items_ptr_descr());
                    let index = ctx.const_int(concrete_len as i64);
                    let fail_args = this.current_fail_args(ctx);
                    let raw = if this.value_type(value) == Type::Float {
                        value
                    } else {
                        crate::jit::generated::trace_unbox_float(
                            ctx,
                            value,
                            &FLOAT_TYPE as *const _ as i64,
                            ob_type_descr(),
                            float_floatval_descr(),
                            &fail_args,
                        )
                    };
                    trace_raw_float_array_setitem_value(ctx, items_ptr, index, raw);
                    let new_len = ctx.const_int((concrete_len + 1) as i64);
                    ctx.record_op_with_descr(
                        OpCode::SetfieldGc,
                        &[list, new_len],
                        list_float_items_len_descr(),
                    );
                    Ok(())
                });
            }
        }

        self.trace_list_append(list, value)
    }

    pub(crate) fn concrete_iter_continues(&self) -> Result<bool, PyError> {
        let concrete_iter = self
            .concrete_at_or_frame(self.sym().valuestackdepth - 1)
            .ok_or_else(|| PyError::type_error("missing concrete iterator during trace"))?;
        range_iter_continues(concrete_iter)
    }

    fn concrete_callable_after_pops(&self) -> Option<PyObjectRef> {
        self.concrete_at_or_frame(self.sym().valuestackdepth)
    }

    fn concrete_call_arg_after_pops(&self, arg_idx: usize) -> Option<PyObjectRef> {
        self.concrete_at_or_frame(self.sym().valuestackdepth + 2 + arg_idx)
    }

    fn trace_known_builtin_call(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
    ) -> Result<OpRef, PyError> {
        self.with_ctx(|this, ctx| {
            let boxed_args = box_args_for_python_helper(this, ctx, args);
            crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, &boxed_args)
        })
    }

    fn direct_len_value(&mut self, callable: OpRef, value: OpRef) -> Result<OpRef, PyError> {
        let Some(concrete_value) = self.concrete_call_arg_after_pops(0) else {
            return self.trace_known_builtin_call(callable, &[value]);
        };

        unsafe {
            if is_str(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &pyre_object::STR_TYPE as *const PyType);
                    let len = trace_gc_object_int_field(ctx, value, str_len_descr());
                    this.remember_value_type(len, Type::Int);
                    Ok(len)
                });
            }
            if is_dict(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &DICT_TYPE as *const PyType);
                    let len = trace_gc_object_int_field(ctx, value, dict_len_descr());
                    this.remember_value_type(len, Type::Int);
                    Ok(len)
                });
            }
            if is_list(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &LIST_TYPE as *const PyType);
                    let len = if w_list_uses_object_storage(concrete_value) {
                        this.guard_list_strategy(ctx, value, 0);
                        trace_gc_object_int_field(ctx, value, list_items_len_descr())
                    } else if w_list_uses_int_storage(concrete_value) {
                        this.guard_list_strategy(ctx, value, 1);
                        trace_gc_object_int_field(ctx, value, list_int_items_len_descr())
                    } else if w_list_uses_float_storage(concrete_value) {
                        this.guard_list_strategy(ctx, value, 2);
                        trace_gc_object_int_field(ctx, value, list_float_items_len_descr())
                    } else {
                        let boxed_value = box_value_for_python_helper(this, ctx, value);
                        return crate::jit::helpers::emit_trace_call_known_builtin(
                            ctx,
                            callable,
                            &[boxed_value],
                        );
                    };
                    this.remember_value_type(len, Type::Int);
                    Ok(len)
                });
            }
            if is_tuple(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &TUPLE_TYPE as *const PyType);
                    let len = trace_gc_object_int_field(ctx, value, tuple_items_len_descr());
                    this.remember_value_type(len, Type::Int);
                    Ok(len)
                });
            }
        }

        self.trace_known_builtin_call(callable, &[value])
    }

    fn direct_abs_value(&mut self, callable: OpRef, value: OpRef) -> Result<OpRef, PyError> {
        let Some(concrete_value) = self.concrete_call_arg_after_pops(0) else {
            return self.trace_known_builtin_call(callable, &[value]);
        };

        unsafe {
            if is_int(concrete_value) {
                let concrete_int = w_int_get_value(concrete_value);
                if concrete_int == i64::MIN {
                    return self.trace_known_builtin_call(callable, &[value]);
                }
                return self.with_ctx(|this, ctx| {
                    let int_value = this.trace_guarded_int_payload(ctx, value);
                    let min_value = ctx.const_int(i64::MIN);
                    let is_min = ctx.record_op(OpCode::IntEq, &[int_value, min_value]);
                    this.record_guard(ctx, OpCode::GuardFalse, &[is_min]);
                    let shift = ctx.const_int((i64::BITS - 1) as i64);
                    let sign = ctx.record_op(OpCode::IntRshift, &[int_value, shift]);
                    let xor = ctx.record_op(OpCode::IntXor, &[int_value, sign]);
                    let abs_value = ctx.record_op(OpCode::IntSub, &[xor, sign]);
                    Ok({
                        let int_type_addr = &INT_TYPE as *const _ as i64;
                        crate::jit::generated::trace_box_int(
                            ctx,
                            abs_value,
                            w_int_size_descr(),
                            ob_type_descr(),
                            int_intval_descr(),
                            int_type_addr,
                        )
                    })
                });
            }
        }

        self.trace_known_builtin_call(callable, &[value])
    }

    fn direct_type_value(&mut self, callable: OpRef, value: OpRef) -> Result<OpRef, PyError> {
        let Some(concrete_value) = self.concrete_call_arg_after_pops(0) else {
            return self.trace_known_builtin_call(callable, &[value]);
        };

        unsafe {
            let concrete_obj_type = (*concrete_value).ob_type;
            let type_name = (*concrete_obj_type).tp_name;
            return self.with_ctx(|this, ctx| {
                this.guard_object_class(ctx, value, concrete_obj_type);
                Ok(ctx.const_int(pyre_object::box_str_constant(type_name) as i64))
            });
        }
    }

    fn direct_isinstance_value(
        &mut self,
        callable: OpRef,
        obj: OpRef,
        type_name: OpRef,
    ) -> Result<OpRef, PyError> {
        let (Some(concrete_obj), Some(concrete_type_name)) = (
            self.concrete_call_arg_after_pops(0),
            self.concrete_call_arg_after_pops(1),
        ) else {
            return self.trace_known_builtin_call(callable, &[obj, type_name]);
        };

        unsafe {
            if is_str(concrete_type_name) {
                let concrete_obj_type = (*concrete_obj).ob_type;
                let concrete_result =
                    (*concrete_obj_type).tp_name == w_str_get_value(concrete_type_name);
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, obj, concrete_obj_type);
                    this.guard_value(ctx, type_name, concrete_type_name as i64);
                    Ok(ctx.const_int(w_bool_from(concrete_result) as i64))
                });
            }
        }

        self.trace_known_builtin_call(callable, &[obj, type_name])
    }

    fn direct_minmax_value(
        &mut self,
        callable: OpRef,
        a: OpRef,
        b: OpRef,
        choose_max: bool,
    ) -> Result<OpRef, PyError> {
        let (Some(concrete_a), Some(concrete_b)) = (
            self.concrete_call_arg_after_pops(0),
            self.concrete_call_arg_after_pops(1),
        ) else {
            return self.trace_known_builtin_call(callable, &[a, b]);
        };

        unsafe {
            if is_int(concrete_a) && is_int(concrete_b) {
                let lhs = w_int_get_value(concrete_a);
                let rhs = w_int_get_value(concrete_b);
                let concrete_result = if choose_max {
                    lhs.max(rhs)
                } else {
                    lhs.min(rhs)
                };
                let concrete_obj = if choose_max {
                    if lhs >= rhs { concrete_a } else { concrete_b }
                } else if lhs <= rhs {
                    concrete_a
                } else {
                    concrete_b
                };
                if w_int_new(concrete_result) != concrete_obj {
                    return self.trace_known_builtin_call(callable, &[a, b]);
                }
                return self.with_ctx(|this, ctx| {
                    let lhs = this.trace_guarded_int_payload(ctx, a);
                    let rhs = this.trace_guarded_int_payload(ctx, b);
                    let cmp = ctx.record_op(OpCode::IntLt, &[lhs, rhs]);
                    let zero = ctx.const_int(0);
                    let mask = ctx.record_op(OpCode::IntSub, &[zero, cmp]);
                    let xor = ctx.record_op(OpCode::IntXor, &[lhs, rhs]);
                    let select_bits = ctx.record_op(OpCode::IntAnd, &[xor, mask]);
                    let selected = if choose_max {
                        ctx.record_op(OpCode::IntXor, &[lhs, select_bits])
                    } else {
                        ctx.record_op(OpCode::IntXor, &[rhs, select_bits])
                    };
                    Ok({
                        let int_type_addr = &INT_TYPE as *const _ as i64;
                        crate::jit::generated::trace_box_int(
                            ctx,
                            selected,
                            w_int_size_descr(),
                            ob_type_descr(),
                            int_intval_descr(),
                            int_type_addr,
                        )
                    })
                });
            }
        }

        self.trace_known_builtin_call(callable, &[a, b])
    }

    pub(crate) fn call_callable_value(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
    ) -> Result<OpRef, PyError> {
        let Some(concrete_callable) = self.concrete_callable_after_pops() else {
            debug_assert!(
                false,
                "concrete_callable should always be available during tracing"
            );
            return self.trace_call_callable(callable, args);
        };

        unsafe {
            if is_builtin_func(concrete_callable) {
                let builtin_name = w_builtin_func_name(concrete_callable);
                if args.len() == 1 {
                    self.with_ctx(|this, ctx| {
                        this.guard_value_ref(ctx, callable, concrete_callable as i64)
                    });
                    if builtin_name == "type" {
                        return self.direct_type_value(callable, args[0]);
                    }
                    if builtin_name == "len" {
                        return self.direct_len_value(callable, args[0]);
                    }
                    if builtin_name == "abs" {
                        return self.direct_abs_value(callable, args[0]);
                    }
                } else if args.len() == 2 && builtin_name == "isinstance" {
                    self.with_ctx(|this, ctx| {
                        this.guard_value_ref(ctx, callable, concrete_callable as i64)
                    });
                    return self.direct_isinstance_value(callable, args[0], args[1]);
                } else if args.len() == 2 && builtin_name == "min" {
                    self.with_ctx(|this, ctx| {
                        this.guard_value_ref(ctx, callable, concrete_callable as i64)
                    });
                    return self.direct_minmax_value(callable, args[0], args[1], false);
                } else if args.len() == 2 && builtin_name == "max" {
                    self.with_ctx(|this, ctx| {
                        this.guard_value_ref(ctx, callable, concrete_callable as i64)
                    });
                    return self.direct_minmax_value(callable, args[0], args[1], true);
                }
                return self.with_ctx(|this, ctx| {
                    this.guard_value_ref(ctx, callable, concrete_callable as i64);
                    let boxed_args = box_args_for_python_helper(this, ctx, args);
                    crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, &boxed_args)
                });
            }
            if is_func(concrete_callable) {
                let callee_code_ptr = w_func_get_code_ptr(concrete_callable) as *const CodeObject;
                let callee_key = crate::eval::make_green_key(callee_code_ptr, 0);
                let callee_code = unsafe { &*callee_code_ptr };
                let callee_has_loop = code_has_backward_jump(callee_code);
                let (driver, _) = crate::eval::driver_pair();
                let nargs = args.len();

                // RPython pyjitpl.py: do_residual_or_indirect_call() follows
                // direct jitcode calls via perform_call() before falling back
                // to residual helpers.  Mirror that for ordinary direct calls:
                // if we know the callee body and it is a small acyclic helper,
                // trace through it directly instead of waiting for
                // should_inline() to bless a helper-boundary inline.
                let root_trace_green_key = root_trace_green_key(self);
                let current_function_key = crate::eval::make_green_key(self.sym().concrete_code, 0);
                let is_self_recursive = callee_key == current_function_key;
                let inline_decision = driver.should_inline(callee_key);
                let inline_framestack_active = self.parent_fail_args.is_some();
                let callee_inline_eligible = driver
                    .meta_interp()
                    .warm_state_ref()
                    .can_inline_callable(callee_key);
                let max_unroll_recursion =
                    driver.meta_interp().warm_state_ref().max_unroll_recursion() as usize;
                let recursive_depth = self.with_ctx(|_, ctx| ctx.recursive_depth(callee_key));
                let concrete_arg0 = if nargs == 1 {
                    self.concrete_call_arg_after_pops(0)
                } else {
                    None
                };
                let callee_prefers_function_entry =
                    crate::call_jit::callable_prefers_function_entry(concrete_callable);
                if is_self_recursive
                    && inline_decision == majit_meta::InlineDecision::Inline
                    && recursive_depth >= max_unroll_recursion
                {
                    driver
                        .meta_interp_mut()
                        .warm_state_mut()
                        .disable_noninlinable_function(callee_key);
                }

                // RPython _opimpl_recursive_call() traces recursive portal
                // calls via perform_call() only until max_unroll_recursion is
                // hit, then forces the residual/assembler path so the current
                // trace can converge. Mirror that here instead of letting
                // self-recursive trace-through run until trace-too-long.
                let tracing_recursive_function_entry =
                    is_self_recursive && root_trace_green_key == current_function_key;
                let can_trace_through = callee_inline_eligible
                    && nargs <= 4
                    && if is_self_recursive {
                        // RPython pyjitpl.py reached_loop_header(): when a
                        // function-entry trace re-enters the same portal, the
                        // recursive call is handed off with
                        // do_recursive_call(..., assembler_call=True)
                        // instead of tracing through the nested portal body as
                        // an ordinary inline helper. Keep pyre on the same
                        // side of that boundary: root function-entry traces do
                        // not recursively trace-through "self"; they converge
                        // to CALL_ASSEMBLER via the pending token path below.
                        !tracing_recursive_function_entry
                            && inline_decision == majit_meta::InlineDecision::Inline
                            && recursive_depth < max_unroll_recursion
                    } else {
                        !callee_prefers_function_entry && !callee_has_loop
                    };

                if majit_meta::majit_log_enabled() {
                    eprintln!(
                        "[jit][direct-call] key={} nargs={} inline_eligible={} self_recursive={} recursive_depth={} max_unroll_recursion={} prefers_func_entry={} has_loop={} inline_active={} can_trace_through={}",
                        callee_key,
                        nargs,
                        callee_inline_eligible,
                        is_self_recursive,
                        recursive_depth,
                        max_unroll_recursion,
                        callee_prefers_function_entry,
                        callee_has_loop,
                        inline_framestack_active,
                        can_trace_through,
                    );
                }

                if callee_prefers_function_entry && !is_self_recursive {
                    let call_pc = self.fallthrough_pc.saturating_sub(1);
                    return self.with_ctx(|this, ctx| {
                        this.guard_value_ref(ctx, callable, concrete_callable as i64);
                        let boxed_args = box_args_for_python_helper(this, ctx, args);
                        let result = crate::jit::helpers::emit_trace_call_known_function(
                            ctx,
                            this.frame(),
                            callable,
                            &boxed_args,
                        )?;
                        this.push_call_replay_stack(ctx, callable, args, call_pc);
                        this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                        this.pop_call_replay_stack(ctx, args.len())?;
                        Ok(result)
                    });
                }

                if can_trace_through {
                    if inline_framestack_active {
                        match self.build_pending_inline_frame(
                            callable,
                            args,
                            concrete_callable,
                            callee_key,
                        ) {
                            Ok(pending) => {
                                self.pending_inline_frame = Some(pending);
                                return self.with_ctx(|_, ctx| {
                                    Ok(ctx.const_int(pyre_object::PY_NULL as i64))
                                });
                            }
                            Err(err) => {
                                if majit_meta::majit_log_enabled() {
                                    eprintln!(
                                        "[jit][trace-through] pending inline frame build failed key={} nargs={} err={}",
                                        callee_key, nargs, err
                                    );
                                }
                            }
                        }
                    }
                    match self.trace_through_callee(callable, args, concrete_callable, callee_key) {
                        Ok(result) => return Ok(result),
                        Err(err) => {
                            let inline_too_long = err.message == "inline trace aborted";
                            note_inline_trace_too_long(
                                callee_key,
                                current_function_key,
                                root_trace_green_key,
                                &err,
                            );
                            if majit_meta::majit_log_enabled() {
                                eprintln!(
                                    "[jit][trace-through] inline trace-through failed key={} nargs={} err={}",
                                    callee_key, nargs, err
                                );
                            }
                            if inline_too_long {
                                return Err(err);
                            }
                        }
                    }
                    // trace-through failed: fall through to the older helper
                    // path so correctness wins over optimization.
                }

                // PyPy converges self-recursion to separate functrace +
                // call_assembler once compiled code exists. However,
                // `_opimpl_recursive_call` still goes through `perform_call()`
                // while tracing the current portal, and only switches to the
                // assembler-call path after recursion is no longer being
                // unrolled in the active trace.
                //
                // pyre does not yet have the full `ChangeFrame`-style
                // self-recursive framestack needed to trace through that path
                // safely. If we emit CALL_ASSEMBLER here for a pending
                // self-recursive token, the backend sees a transient
                // descriptor/input mismatch and rejects the trace. Keep
                // self-recursion on the helper-boundary path below until the
                // framestack implementation is complete.
                if majit_meta::majit_log_enabled() {
                    eprintln!(
                        "[jit][call-check] is_self={} cache_safe={} inline_active={} callee_key={}",
                        is_self_recursive,
                        crate::call_jit::recursive_force_cache_safe(concrete_callable),
                        inline_framestack_active,
                        callee_key
                    );
                }

                if inline_decision == majit_meta::InlineDecision::Inline {
                    if let Some(frame_helper) = crate::call_jit::callee_frame_helper(nargs) {
                        return self.inline_function_call(
                            callable,
                            args,
                            concrete_callable,
                            callee_key,
                            frame_helper,
                        );
                    }
                }

                if majit_meta::majit_log_enabled() {
                    eprintln!(
                        "[jit][call-dispatch] callee_key={} pending_token={:?} loop_token={:?} is_self={}",
                        callee_key,
                        driver.get_pending_token_number(callee_key),
                        driver.get_loop_token_number(callee_key),
                        is_self_recursive
                    );
                }
                if let Some(token_number) = driver.get_pending_token_number(callee_key) {
                    let callee_nlocals = {
                        let code_ptr = w_func_get_code_ptr(concrete_callable) as *const CodeObject;
                        (&*code_ptr).varnames.len()
                    };
                    if nargs == 1 || crate::call_jit::callee_frame_helper(nargs).is_some() {
                        return self.with_ctx(|this, ctx| {
                            if !is_self_recursive {
                                this.guard_value_ref(ctx, callable, concrete_callable as i64);
                            }
                            let self_recursive_raw_arg = if is_self_recursive
                                && nargs == 1
                                && matches!(concrete_arg0, Some(arg) if unsafe { is_int(arg) })
                            {
                                Some(this.trace_guarded_int_payload(ctx, args[0]))
                            } else {
                                None
                            };
                            let callee_frame = if let Some(raw_arg) = self_recursive_raw_arg {
                                let (helper, helper_arg_types) =
                                    one_arg_callee_frame_helper(Type::Int, true);
                                ctx.call_ref_typed(
                                    helper,
                                    &[this.frame(), raw_arg],
                                    &helper_arg_types,
                                )
                            } else if nargs == 1 {
                                let (helper, helper_arg_types) = one_arg_callee_frame_helper(
                                    this.value_type(args[0]),
                                    is_self_recursive,
                                );
                                let helper_args = if is_self_recursive {
                                    vec![this.frame(), args[0]]
                                } else {
                                    vec![this.frame(), callable, args[0]]
                                };
                                ctx.call_ref_typed(helper, &helper_args, &helper_arg_types)
                            } else {
                                let frame_helper =
                                    crate::call_jit::callee_frame_helper(nargs).unwrap();
                                let mut helper_args = if is_self_recursive {
                                    vec![this.frame()]
                                } else {
                                    vec![this.frame(), callable]
                                };
                                helper_args.extend_from_slice(args);
                                let helper_arg_types = frame_callable_arg_types(args.len());
                                ctx.call_ref_typed(frame_helper, &helper_args, &helper_arg_types)
                            };
                            // RPython parity: CallAssemblerI locals must be Ref
                            // (boxed) to match the callee trace's virtualizable
                            // array type. Raw Int args are re-boxed before passing.
                            let ca_locals: Vec<OpRef> =
                                if let Some(raw_arg) = self_recursive_raw_arg {
                                    vec![box_traced_raw_int(ctx, raw_arg)]
                                } else {
                                    args.iter()
                                        .map(|&arg| ensure_boxed_for_ca(ctx, &*this, arg))
                                        .collect()
                                };
                            let ca_args = synthesize_fresh_callee_entry_args(
                                ctx,
                                callee_frame,
                                &ca_locals,
                                callee_nlocals,
                            );
                            let callee_slot_types = pending_entry_slot_types_from_args(
                                &vec![Type::Ref; ca_locals.len()],
                                callee_nlocals,
                                0,
                            );
                            let ca_arg_types =
                                frame_entry_arg_types_from_slot_types(&callee_slot_types);
                            let result = ctx.call_assembler_int_by_number_typed(
                                token_number,
                                &ca_args,
                                &ca_arg_types,
                            );
                            ctx.call_void(
                                crate::call_jit::jit_drop_callee_frame as *const (),
                                &[callee_frame],
                            );
                            let result = if inline_framestack_active {
                                box_traced_raw_int(ctx, result)
                            } else {
                                result
                            };
                            Ok(result)
                        });
                    }
                }

                match inline_decision {
                    majit_meta::InlineDecision::CallAssembler => {
                        // Trace-through: inline callee body instead of CallAssembler.
                        // Guards use parent_fail_args to avoid OpRef::NONE in fail_args.
                        if callee_inline_eligible
                            && !is_self_recursive
                            && nargs <= 4
                            && !callee_has_loop
                        {
                            match self.trace_through_callee(
                                callable,
                                args,
                                concrete_callable,
                                callee_key,
                            ) {
                                Ok(result) => return Ok(result),
                                Err(err) => {
                                    let inline_too_long = err.message == "inline trace aborted";
                                    note_inline_trace_too_long(
                                        callee_key,
                                        current_function_key,
                                        root_trace_green_key,
                                        &err,
                                    );
                                    if majit_meta::majit_log_enabled() {
                                        eprintln!(
                                            "[jit][trace-through] call-assembler fallback trace-through failed key={} nargs={} err={}",
                                            callee_key, nargs, err
                                        );
                                    }
                                    if inline_too_long {
                                        return Err(err);
                                    }
                                }
                            }
                        }
                        // Use compiled loop token only (not pending_token)
                        // to avoid type descriptor mismatches.
                        let Some(token_number) = driver.get_loop_token_number(callee_key) else {
                            let call_pc = self.fallthrough_pc.saturating_sub(1);
                            return self.with_ctx(|this, ctx| {
                                this.guard_value_ref(ctx, callable, concrete_callable as i64);
                                let result = crate::jit::helpers::emit_trace_call_known_function(
                                    ctx,
                                    this.frame(),
                                    callable,
                                    args,
                                )?;
                                this.push_call_replay_stack(ctx, callable, args, call_pc);
                                this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                                this.pop_call_replay_stack(ctx, args.len())?;
                                Ok(result)
                            });
                        };

                        {
                            let callee_meta = driver.get_compiled_meta(callee_key).unwrap_or_else(|| {
                                panic!(
                                    "compiled loop for callee_key={callee_key} is missing compiled meta"
                                )
                            });
                            let callee_nlocals = callee_meta.num_locals;
                            let callee_vsd = callee_meta.valuestackdepth;
                            let callee_stack_only = callee_vsd.saturating_sub(callee_nlocals);
                            let target_num_inputs =
                                driver.get_compiled_num_inputs(callee_key).unwrap_or(1);

                            return self.with_ctx(|this, ctx| {
                                // Self-recursive: no callable guard needed (same function).
                                // Non-self-recursive: guard on callable value.
                                if !is_self_recursive {
                                    this.guard_value_ref(ctx, callable, concrete_callable as i64);
                                }
                                let callee_frame = if args.len() == 1 {
                                    let (helper, helper_arg_types) = one_arg_callee_frame_helper(
                                        this.value_type(args[0]),
                                        is_self_recursive,
                                    );
                                    let helper_args = if is_self_recursive {
                                        vec![this.frame(), args[0]]
                                    } else {
                                        vec![this.frame(), callable, args[0]]
                                    };
                                    ctx.call_ref_typed(helper, &helper_args, &helper_arg_types)
                                } else if let Some(frame_helper) =
                                    crate::call_jit::callee_frame_helper(nargs)
                                {
                                    let mut helper_args = if is_self_recursive {
                                        vec![this.frame()]
                                    } else {
                                        vec![this.frame(), callable]
                                    };
                                    helper_args.extend_from_slice(args);
                                    let helper_arg_types = frame_callable_arg_types(args.len());
                                    ctx.call_ref_typed(
                                        frame_helper,
                                        &helper_args,
                                        &helper_arg_types,
                                    )
                                } else {
                                    // Fallback: can't create callee frame in trace
                                    return Err(pyre_runtime::PyError::type_error(
                                        "call_assembler: no frame helper for nargs",
                                    ));
                                };

                                let ca_args = if target_num_inputs <= 1 {
                                    vec![callee_frame]
                                } else if callee_stack_only == 0 {
                                    synthesize_fresh_callee_entry_args(
                                        ctx,
                                        callee_frame,
                                        args,
                                        callee_nlocals,
                                    )
                                } else {
                                    let callee_ni = frame_get_next_instr(ctx, callee_frame);
                                    let callee_sd = frame_get_stack_depth(ctx, callee_frame);
                                    let mut a = vec![callee_frame, callee_ni, callee_sd];
                                    let callee_arr =
                                        frame_locals_cells_stack_array(ctx, callee_frame);
                                    for i in 0..callee_nlocals {
                                        let idx = ctx.const_int(i as i64);
                                        let val = ctx.record_op_with_descr(
                                            OpCode::GetarrayitemGcR,
                                            &[callee_arr, idx],
                                            pyobject_array_descr(),
                                        );
                                        a.push(val);
                                    }
                                    for i in 0..callee_stack_only {
                                        let idx = ctx.const_int((callee_nlocals + i) as i64);
                                        let val = ctx.record_op_with_descr(
                                            OpCode::GetarrayitemGcR,
                                            &[callee_arr, idx],
                                            pyobject_array_descr(),
                                        );
                                        a.push(val);
                                    }
                                    a
                                };
                                let ca_arg_types =
                                    frame_entry_arg_types_from_slot_types(&callee_meta.slot_types);
                                let result = ctx.call_assembler_int_by_number_typed(
                                    token_number,
                                    &ca_args,
                                    &ca_arg_types,
                                );
                                ctx.call_void(
                                    crate::call_jit::jit_drop_callee_frame as *const (),
                                    &[callee_frame],
                                );
                                let result = if inline_framestack_active
                                    && driver.has_raw_int_finish(callee_key)
                                {
                                    box_traced_raw_int(ctx, result)
                                } else {
                                    result
                                };
                                Ok(result)
                            });
                        }
                    }
                    majit_meta::InlineDecision::Inline => {
                        if let Some(frame_helper) = crate::call_jit::callee_frame_helper(nargs) {
                            return self.inline_function_call(
                                callable,
                                args,
                                concrete_callable,
                                callee_key,
                                frame_helper,
                            );
                        }
                    }
                    majit_meta::InlineDecision::ResidualCall => {}
                }

                let call_pc = self.fallthrough_pc.saturating_sub(1);
                return self.with_ctx(|this, ctx| {
                    this.guard_value_ref(ctx, callable, concrete_callable as i64);
                    let boxed_args = box_args_for_python_helper(this, ctx, args);
                    let result = crate::jit::helpers::emit_trace_call_known_function(
                        ctx,
                        this.frame(),
                        callable,
                        &boxed_args,
                    )?;
                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                    this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                    this.pop_call_replay_stack(ctx, args.len())?;
                    Ok(result)
                });
            }
        }

        self.trace_call_callable(callable, args)
    }

    fn build_pending_inline_frame(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
        concrete_callable: PyObjectRef,
        callee_key: u64,
    ) -> Result<PendingInlineFrame, PyError> {
        use pyre_interp::frame::PyFrame;

        self.with_ctx(|this, ctx| {
            this.guard_value_ref(ctx, callable, concrete_callable as i64);
        });

        let concrete_args: Vec<PyObjectRef> = (0..args.len())
            .map(|i| {
                self.concrete_call_arg_after_pops(i)
                    .unwrap_or(pyre_object::PY_NULL)
            })
            .collect();
        for (idx, arg) in concrete_args.iter().copied().enumerate() {
            if arg.is_null() {
                panic!("pending inline frame lost concrete arg at index {idx}");
            }
        }

        let caller_code = self.sym().concrete_code;
        // Read execution_context and namespace from concrete_frame (the caller).
        // PyreSym's values may belong to a nested callee in inline trace-through.
        let caller = unsafe { &*(self.concrete_frame as *const PyFrame) };
        let code_ptr = unsafe { w_func_get_code_ptr(concrete_callable) } as *const CodeObject;
        let globals = unsafe { w_func_get_globals(concrete_callable) };
        let closure = unsafe { pyre_runtime::w_func_get_closure(concrete_callable) };
        let is_self_recursive = crate::eval::make_green_key(caller_code, 0) == callee_key;
        let mut callee_frame = PyFrame::new_for_call_with_closure(
            code_ptr,
            &concrete_args,
            globals,
            caller.execution_context,
            closure,
        );
        callee_frame.fix_array_ptrs();

        let callee_code = unsafe { &*callee_frame.code };
        let callee_nlocals = callee_code.varnames.len();
        let caller_namespace = caller.namespace;
        let callee_globals = unsafe { w_func_get_globals(concrete_callable) };
        let can_skip_traced_callee_frame = !is_self_recursive
            && callee_globals == caller_namespace
            && callee_nlocals == args.len();

        let (callee_sym, drop_frame_opref) = if can_skip_traced_callee_frame {
            let frame = self.frame();
            let mut sym = PyreSym::new_uninit(frame);
            sym.nlocals = callee_nlocals;
            sym.valuestackdepth = callee_nlocals;
            sym.symbolic_locals = args.to_vec();
            sym.symbolic_local_types = args.iter().map(|&arg| self.value_type(arg)).collect();
            sym.symbolic_stack = Vec::new();
            sym.symbolic_stack_types = Vec::new();
            sym.symbolic_initialized = true;
            let (vable_next_instr, vable_valuestackdepth) =
                self.with_ctx(|_, ctx| (ctx.const_int(0), ctx.const_int(callee_nlocals as i64)));
            sym.vable_next_instr = vable_next_instr;
            sym.vable_valuestackdepth = vable_valuestackdepth;
            (sym, None)
        } else {
            // Create symbolic OpRef for callee frame in trace
            let callee_frame_opref = self.with_ctx(|this, ctx| {
                if args.len() == 1 {
                    let (helper, helper_arg_types) =
                        one_arg_callee_frame_helper(this.value_type(args[0]), is_self_recursive);
                    if is_self_recursive {
                        ctx.call_ref_typed(helper, &[this.frame(), args[0]], &helper_arg_types)
                    } else {
                        ctx.call_ref_typed(
                            helper,
                            &[this.frame(), callable, args[0]],
                            &helper_arg_types,
                        )
                    }
                } else if let Some(frame_helper) = crate::call_jit::callee_frame_helper(args.len())
                {
                    let mut helper_args = vec![this.frame(), callable];
                    helper_args.extend_from_slice(args);
                    let helper_arg_types = frame_callable_arg_types(args.len());
                    ctx.call_ref_typed(frame_helper, &helper_args, &helper_arg_types)
                } else {
                    panic!("no frame helper for {} args", args.len());
                }
            });

            let mut sym = PyreSym::new_uninit(callee_frame_opref);
            sym.nlocals = callee_nlocals;
            sym.valuestackdepth = sym.nlocals;
            sym.symbolic_locals = Vec::with_capacity(sym.nlocals);
            sym.symbolic_local_types = Vec::with_capacity(sym.nlocals);
            for i in 0..sym.nlocals {
                if i < args.len() {
                    sym.symbolic_locals.push(args[i]);
                    sym.symbolic_local_types.push(self.value_type(args[i]));
                } else {
                    sym.symbolic_locals.push(OpRef::NONE);
                    sym.symbolic_local_types.push(Type::Ref);
                }
            }
            sym.symbolic_stack = Vec::new();
            sym.symbolic_stack_types = Vec::new();
            sym.symbolic_initialized = true;
            (sym, Some(callee_frame_opref))
        };

        // GuardNotForced in callee → interpreter must re-execute the CALL.
        // Temporarily restore callable+args to parent stack and set PC to CALL.
        let call_pc = self.fallthrough_pc.saturating_sub(1);
        self.with_ctx(|this, ctx| this.push_call_replay_stack(ctx, callable, args, call_pc));
        self.with_ctx(|this, ctx| this.flush_to_frame(ctx));
        let parent_fail_args = self.with_ctx(|this, ctx| this.build_single_frame_fail_args(ctx));
        let parent_fail_arg_types = self.build_single_frame_fail_arg_types();
        self.with_ctx(|this, ctx| this.pop_call_replay_stack(ctx, args.len()))?;
        Ok(PendingInlineFrame {
            sym: callee_sym,
            concrete_frame: callee_frame,
            drop_frame_opref,
            green_key: callee_key,
            parent_fail_args,
            parent_fail_arg_types,
            nargs: args.len(),
            caller_result_stack_idx: None,
        })
    }

    /// PyPy perform_call + _interpret: trace INTO callee body.
    ///
    /// Instead of emitting an opaque CallMayForce, traces each callee
    /// bytecode instruction into the current trace. Simple functions
    /// like `add(a,b): return a+b` become `IntAdd(a,b)` in the trace.
    fn trace_through_callee(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
        concrete_callable: PyObjectRef,
        callee_key: u64,
    ) -> Result<OpRef, PyError> {
        let pending =
            self.build_pending_inline_frame(callable, args, concrete_callable, callee_key)?;
        let (driver, _) = crate::eval::driver_pair();
        driver.enter_inline_frame(callee_key);

        let ctx_ptr = self.ctx;
        let ctx = unsafe { &mut *ctx_ptr };
        let result = inline_trace_and_execute(ctx, pending);
        let (driver, _) = crate::eval::driver_pair();
        driver.leave_inline_frame();
        let (result_opref, result_type, concrete_result) = result?;
        if majit_meta::majit_log_enabled() {
            eprintln!(
                "[jit][trace-through] key={} result_opref={:?} result_type={:?} concrete_result={:#x}",
                callee_key, result_opref, result_type, concrete_result as usize,
            );
        }
        self.remember_value_type(result_opref, result_type);

        // RPython finishframe()/ChangeFrame parity: once the inlined callee
        // has concretely returned, store its result in the parent frame and
        // advance that caller past the CALL immediately. Do not route the
        // MIFrame Box tracking: store concrete result for next push.
        // In full-loop tracing, the concrete_frame snapshot is stale, so we
        // track results via pending_concrete_push instead of frame stack ops.
        self.sym_mut().pending_concrete_push = Some(ConcreteValue::from_pyobj(concrete_result));

        // RPython finishframe parity: result is stored via pending_concrete_push.
        // No concrete_frame stack manipulation needed — MIFrame Box tracking
        // handles all concrete state internally.

        Ok(result_opref)
    }

    /// CallMayForce-based inline: opaque helper call.
    /// Fallback for self-recursion or when trace-through is not available.
    fn inline_function_call(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
        concrete_callable: PyObjectRef,
        callee_key: u64,
        frame_helper: *const (),
    ) -> Result<OpRef, PyError> {
        let (driver, _) = crate::eval::driver_pair();
        let concrete_arg0 = if args.len() == 1 {
            self.concrete_call_arg_after_pops(0)
        } else {
            None
        };
        // Save CALL instruction PC so GuardNotForced can resume the CALL.
        let call_pc = self.fallthrough_pc.saturating_sub(1);

        let result = self.with_ctx(|this, ctx| {
            this.guard_value_ref(ctx, callable, concrete_callable as i64);

            if args.len() == 1 {
                let result = if matches!(concrete_arg0, Some(arg) if unsafe { is_int(arg) }) {
                    let raw_arg = this.trace_guarded_int_payload(ctx, args[0]);
                    let is_self_recursive = callee_key
                        == unsafe {
                            crate::eval::make_green_key(
                                (*(this.concrete_frame as *const pyre_interp::frame::PyFrame)).code,
                                0,
                            )
                        };
                    // RPython parity: an opaque helper-boundary Python CALL
                    // still produces a boxed object result.  Even if the
                    // callee itself can finish with a raw int, the helper
                    // boxes at the boundary and the trace records a Ref.
                    let force_fn = if is_self_recursive
                        && crate::call_jit::recursive_force_cache_safe(concrete_callable)
                    {
                        crate::call_jit::jit_force_self_recursive_call_argraw_boxed_1 as *const ()
                    } else {
                        crate::call_jit::jit_force_recursive_call_argraw_boxed_1 as *const ()
                    };
                    this.sync_standard_virtualizable_before_residual_call(ctx);
                    // RPython parity: use CALL_ASSEMBLER_I when a token
                    // exists for the callee (compiled or pending).
                    // RPython parity: use CALL_ASSEMBLER_I when a token
                    // exists (compiled or pending).
                    let ca_token = if is_self_recursive {
                        driver
                            .get_loop_token_number(callee_key)
                            .or_else(|| driver.get_pending_token_number(callee_key))
                    } else {
                        None
                    };
                    let result = if let Some(token_number) = ca_token {
                        // RPython: CALL_ASSEMBLER_I — compiled code calls
                        // compiled callee directly. Bridge handles guard
                        // failures (base cases).
                        let (helper, helper_arg_types) =
                            one_arg_callee_frame_helper(Type::Int, true);
                        let callee_frame =
                            ctx.call_ref_typed(helper, &[this.frame(), raw_arg], &helper_arg_types);
                        let callee_nlocals = unsafe {
                            let code_ptr = w_func_get_code_ptr(concrete_callable)
                                as *const pyre_bytecode::CodeObject;
                            (&*code_ptr).varnames.len()
                        };
                        // RPython parity: CallAssemblerI locals must be Ref
                        // (boxed) to match the callee trace's virtualizable
                        // array type. Re-box the raw Int arg.
                        let boxed_arg = box_traced_raw_int(ctx, raw_arg);
                        let ca_args = synthesize_fresh_callee_entry_args(
                            ctx,
                            callee_frame,
                            &[boxed_arg],
                            callee_nlocals,
                        );
                        let callee_slot_types =
                            pending_entry_slot_types_from_args(&[Type::Ref], callee_nlocals, 0);
                        let ca_arg_types =
                            frame_entry_arg_types_from_slot_types(&callee_slot_types);
                        let ca_result = ctx.call_assembler_int_by_number_typed(
                            token_number,
                            &ca_args,
                            &ca_arg_types,
                        );
                        ctx.call_void(
                            crate::call_jit::jit_drop_callee_frame as *const (),
                            &[callee_frame],
                        );
                        // Re-box the raw int result for the caller
                        box_traced_raw_int(ctx, ca_result)
                    } else if force_fn
                        == crate::call_jit::jit_force_self_recursive_call_argraw_boxed_1
                            as *const ()
                    {
                        ctx.call_may_force_ref_typed(
                            force_fn,
                            &[this.frame(), raw_arg],
                            &[Type::Ref, Type::Int],
                        )
                    } else {
                        ctx.call_may_force_ref_typed(
                            force_fn,
                            &[this.frame(), callable, raw_arg],
                            &[Type::Ref, Type::Ref, Type::Int],
                        )
                    };
                    // CallAssemblerI handles guard failures internally
                    // (call_assembler_fast_path). No GuardNotForced needed,
                    // but virtualizable token must be cleaned up.
                    if ca_token.is_some() {
                        this.sync_standard_virtualizable_after_residual_call();
                    } else if !this.sync_standard_virtualizable_after_residual_call() {
                        this.push_call_replay_stack(ctx, callable, args, call_pc);
                        this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                        this.pop_call_replay_stack(ctx, args.len())?;
                    }
                    result
                } else {
                    let force_fn = crate::call_jit::jit_force_recursive_call_1 as *const ();
                    this.sync_standard_virtualizable_before_residual_call(ctx);
                    let result = ctx.call_may_force_ref_typed(
                        force_fn,
                        &[this.frame(), callable, args[0]],
                        &[Type::Ref, Type::Ref, Type::Ref],
                    );
                    if !this.sync_standard_virtualizable_after_residual_call() {
                        this.push_call_replay_stack(ctx, callable, args, call_pc);
                        this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                        this.pop_call_replay_stack(ctx, args.len())?;
                    }
                    result
                };
                Ok(result)
            } else {
                let mut helper_args = vec![this.frame(), callable];
                helper_args.extend_from_slice(args);
                let helper_arg_types = frame_callable_arg_types(args.len());
                let callee_frame =
                    ctx.call_ref_typed(frame_helper, &helper_args, &helper_arg_types);
                let force_fn = crate::call_jit::jit_force_callee_frame as *const ();
                this.sync_standard_virtualizable_before_residual_call(ctx);
                let result = ctx.call_may_force_ref_typed(force_fn, &[callee_frame], &[Type::Ref]);
                if !this.sync_standard_virtualizable_after_residual_call() {
                    // GuardNotForced fail → interpreter must re-execute CALL.
                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                    this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                    this.pop_call_replay_stack(ctx, args.len())?;
                }
                ctx.call_void(
                    crate::call_jit::jit_drop_callee_frame as *const (),
                    &[callee_frame],
                );
                Ok(result)
            }
        });

        result
    }

    pub(crate) fn iter_next_value(&mut self, iter: OpRef) -> Result<OpRef, PyError> {
        let concrete_iter = self
            .concrete_at_or_frame(self.sym().valuestackdepth - 1)
            .ok_or_else(|| PyError::type_error("missing concrete iterator during trace"))?;
        let concrete_continues = range_iter_continues(concrete_iter)?;
        let concrete_step =
            unsafe { (*(concrete_iter as *const pyre_object::rangeobject::W_RangeIterator)).step };
        let concrete_current = unsafe {
            (*(concrete_iter as *const pyre_object::rangeobject::W_RangeIterator)).current
        };
        let concrete_step_positive = concrete_step > 0;
        if concrete_continues {
            if concrete_current.checked_add(concrete_step).is_none() {
                return self.trace_iter_next_value(iter);
            }
        }

        self.with_ctx(|this, ctx| {
            TraceFrameState::guard_range_iter(this, ctx, iter);

            let current = trace_gc_object_int_field(ctx, iter, range_iter_current_descr());
            let stop = trace_gc_object_int_field(ctx, iter, range_iter_stop_descr());
            let step = trace_gc_object_int_field(ctx, iter, range_iter_step_descr());
            let zero = ctx.const_int(0);
            let step_positive = ctx.record_op(OpCode::IntGt, &[step, zero]);
            this.guard_value(ctx, step_positive, concrete_step_positive as i64);

            let continues = if concrete_step_positive {
                ctx.record_op(OpCode::IntLt, &[current, stop])
            } else {
                ctx.record_op(OpCode::IntGt, &[current, stop])
            };
            this.guard_value(ctx, continues, concrete_continues as i64);

            if !concrete_continues {
                this.remember_value_type(zero, Type::Int);
                return Ok(zero);
            }

            let next_current = ctx.record_op(OpCode::IntAddOvf, &[current, step]);
            this.record_guard(ctx, OpCode::GuardNoOverflow, &[]);
            ctx.record_op_with_descr(
                OpCode::SetfieldGc,
                &[iter, next_current],
                range_iter_current_descr(),
            );
            this.remember_value_type(current, Type::Int);
            Ok(current)
        })
    }

    pub(crate) fn concrete_branch_truth_for_value(
        &mut self,
        _value: OpRef,
    ) -> Result<bool, PyError> {
        if let Some(truth) = self.sym_mut().last_comparison_concrete_truth.take() {
            return Ok(truth);
        }
        let concrete_val = self
            .sym_mut()
            .last_popped_concrete_value
            .or_else(|| self.concrete_at_or_frame(self.sym().valuestackdepth))
            .ok_or_else(|| PyError::type_error("missing concrete branch value during trace"))?;
        Ok(objspace_truth_value(concrete_val))
    }

    pub(crate) fn concrete_branch_truth(&mut self) -> Result<bool, PyError> {
        self.concrete_branch_truth_for_value(OpRef::NONE)
    }

    pub(crate) fn truth_value_direct(&mut self, value: OpRef) -> Result<OpRef, PyError> {
        // RPython goto_if_not fusion: if the previous op was a comparison,
        // use the cached raw truth (0/1) directly instead of unboxing the
        // bool object. This eliminates the CallI(bool_helper) + GetfieldGcI
        // round-trip, matching RPython's fused goto_if_not_int_lt pattern.
        if let Some(truth) = self.sym_mut().last_comparison_truth.take() {
            return Ok(truth);
        }

        let cached_concrete_truth = self.concrete_popped_value().map(objspace_truth_value);
        if let Some(truth) = cached_concrete_truth {
            self.sym_mut().last_comparison_concrete_truth = Some(truth);
        }

        if self.value_type(value) == Type::Int {
            return self.with_ctx(|_, ctx| {
                let zero = ctx.const_int(0);
                Ok(ctx.record_op(OpCode::IntNe, &[value, zero]))
            });
        }
        if self.value_type(value) == Type::Float {
            return self.with_ctx(|_, ctx| {
                let zero = ctx.const_int(0);
                let zero_float = ctx.record_op(OpCode::CastIntToFloat, &[zero]);
                Ok(ctx.record_op(OpCode::FloatNe, &[value, zero_float]))
            });
        }

        let Some(concrete_value) = self.concrete_popped_value() else {
            return self.trace_truth_value(value);
        };

        unsafe {
            if is_int(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    let fail_args = this.current_fail_args(ctx);
                    let int_type_addr = &INT_TYPE as *const _ as i64;
                    let int_value =
                        if let Some(raw) = try_trace_const_boxed_int(ctx, value, concrete_value) {
                            raw
                        } else {
                            crate::jit::generated::trace_unbox_int(
                                ctx,
                                value,
                                int_type_addr,
                                ob_type_descr(),
                                int_intval_descr(),
                                &fail_args,
                            )
                        };
                    let zero = ctx.const_int(0);
                    Ok(ctx.record_op(OpCode::IntNe, &[int_value, zero]))
                });
            }
            if is_bool(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    let fail_args = this.current_fail_args(ctx);
                    let bool_type_addr = &BOOL_TYPE as *const _ as i64;
                    let bool_value =
                        if let Some(raw) = try_trace_const_boxed_int(ctx, value, concrete_value) {
                            raw
                        } else {
                            crate::jit::generated::trace_unbox_int(
                                ctx,
                                value,
                                bool_type_addr,
                                ob_type_descr(),
                                bool_boolval_descr(),
                                &fail_args,
                            )
                        };
                    let zero = ctx.const_int(0);
                    Ok(ctx.record_op(OpCode::IntNe, &[bool_value, zero]))
                });
            }
            if is_none(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &NONE_TYPE as *const PyType);
                    Ok(ctx.const_int(0))
                });
            }
            if is_float(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    let fail_args = this.current_fail_args(ctx);
                    let float_type_addr = &FLOAT_TYPE as *const _ as i64;
                    let float_value = crate::jit::generated::trace_unbox_float(
                        ctx,
                        value,
                        float_type_addr,
                        ob_type_descr(),
                        float_floatval_descr(),
                        &fail_args,
                    );
                    let zero = ctx.const_int(0);
                    let zero_float = ctx.record_op(OpCode::CastIntToFloat, &[zero]);
                    Ok(ctx.record_op(OpCode::FloatNe, &[float_value, zero_float]))
                });
            }
            if is_str(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &pyre_object::STR_TYPE as *const PyType);
                    let len =
                        ctx.record_op_with_descr(OpCode::GetfieldRawI, &[value], str_len_descr());
                    let zero = ctx.const_int(0);
                    Ok(ctx.record_op(OpCode::IntNe, &[len, zero]))
                });
            }
            if is_dict(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &DICT_TYPE as *const PyType);
                    let len =
                        ctx.record_op_with_descr(OpCode::GetfieldRawI, &[value], dict_len_descr());
                    let zero = ctx.const_int(0);
                    Ok(ctx.record_op(OpCode::IntNe, &[len, zero]))
                });
            }
            if is_list(concrete_value) && w_list_uses_object_storage(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &LIST_TYPE as *const PyType);
                    let len = ctx.record_op_with_descr(
                        OpCode::GetfieldRawI,
                        &[value],
                        list_items_len_descr(),
                    );
                    let zero = ctx.const_int(0);
                    Ok(ctx.record_op(OpCode::IntNe, &[len, zero]))
                });
            }
            if is_tuple(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &TUPLE_TYPE as *const PyType);
                    let len = ctx.record_op_with_descr(
                        OpCode::GetfieldRawI,
                        &[value],
                        tuple_items_len_descr(),
                    );
                    let zero = ctx.const_int(0);
                    Ok(ctx.record_op(OpCode::IntNe, &[len, zero]))
                });
            }
        }

        self.trace_truth_value(value)
    }

    pub(crate) fn unary_int_value(
        &mut self,
        value: OpRef,
        opcode: OpCode,
    ) -> Result<OpRef, PyError> {
        if self.concrete_unary_int_operand().is_none() {
            return match opcode {
                OpCode::IntNeg => self.trace_unary_negative_value(value),
                OpCode::IntInvert => self.trace_unary_invert_value(value),
                _ => unreachable!("unexpected unary opcode"),
            };
        }

        self.with_ctx(|this, ctx| {
            let payload = if this.value_type(value) == Type::Int {
                value
            } else {
                let fail_args = this.current_fail_args(ctx);
                let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
                crate::jit::generated::trace_unbox_int(
                    ctx,
                    value,
                    int_type_addr,
                    ob_type_descr(),
                    int_intval_descr(),
                    &fail_args,
                )
            };
            if matches!(opcode, OpCode::IntNeg) {
                let min_val = ctx.const_int(i64::MIN);
                let is_min = ctx.record_op(OpCode::IntEq, &[payload, min_val]);
                this.record_guard(ctx, OpCode::GuardFalse, &[is_min]);
            }
            let result = ctx.record_op(opcode, &[payload]);
            this.remember_value_type(result, Type::Int);
            Ok(result)
        })
    }

    pub(crate) fn into_trace_action(
        &mut self,
        result: Result<pyre_runtime::StepResult<OpRef>, PyError>,
    ) -> TraceAction {
        trace_step_result_to_action(self, result)
    }

    pub(crate) fn trace_code_step(&mut self, code: &CodeObject, pc: usize) -> TraceAction {
        if pc >= code.instructions.len() {
            if majit_meta::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step pc_oob pc={} code_len={}",
                    pc,
                    code.instructions.len()
                );
            }
            return TraceAction::Abort;
        }

        let Some((instruction, op_arg)) = decode_instruction_at(code, pc) else {
            if majit_meta::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step decode_failed pc={}",
                    pc
                );
            }
            return TraceAction::Abort;
        };

        // RPython goto_if_not fusion: invalidate comparison truth cache
        // unless this instruction is POP_JUMP_IF_FALSE/TRUE (the consumer).
        // RPython's codewriter fuses COMPARE+BRANCH into one op, so there's
        // no intermediate instruction. In pyre they're separate bytecodes,
        // so we keep the cache alive only for the immediately following jump.
        if !instruction_consumes_comparison_truth(instruction)
            && !instruction_is_trivia_between_compare_and_branch(instruction)
        {
            self.sym_mut().last_comparison_truth = None;
            self.sym_mut().last_comparison_concrete_truth = None;
        }

        self.prepare_fallthrough();
        let step_result = execute_opcode_step(self, code, instruction, op_arg, pc + 1);
        self.into_trace_action(step_result)
    }

    pub(crate) fn trace_code_step_inline(
        &mut self,
        code: &CodeObject,
        pc: usize,
    ) -> InlineTraceStepAction {
        if pc >= code.instructions.len() {
            if majit_meta::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step_inline pc_oob pc={} code_len={}",
                    pc,
                    code.instructions.len()
                );
            }
            return InlineTraceStepAction::Trace(TraceAction::Abort);
        }

        let Some((instruction, op_arg)) = decode_instruction_at(code, pc) else {
            if majit_meta::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step_inline decode_failed pc={}",
                    pc
                );
            }
            return InlineTraceStepAction::Trace(TraceAction::Abort);
        };

        // RPython goto_if_not: invalidate truth cache for non-jump instructions.
        if !instruction_consumes_comparison_truth(instruction)
            && !instruction_is_trivia_between_compare_and_branch(instruction)
        {
            self.sym_mut().last_comparison_truth = None;
            self.sym_mut().last_comparison_concrete_truth = None;
        }

        self.prepare_fallthrough();
        let step_result = execute_opcode_step(self, code, instruction, op_arg, pc + 1);
        let action = match step_result {
            // Inline finishframe() is not a top-level FINISH descriptor
            // boundary. Preserve the traced value and its current symbolic
            // type for the parent frame instead of coercing boxed ints to
            // raw finish-protocol Ints.
            Ok(pyre_runtime::StepResult::Return(value)) => TraceAction::Finish {
                finish_args: vec![value],
                finish_arg_types: vec![self.value_type(value)],
            },
            other => self.into_trace_action(other),
        };
        match action {
            TraceAction::Continue => {
                if let Some(mut pending) = self.pending_inline_frame.take() {
                    let result_idx = self
                        .sym()
                        .stack_only_depth()
                        .checked_sub(1)
                        .expect("pending inline frame missing caller result slot");
                    pending.caller_result_stack_idx = Some(result_idx);
                    InlineTraceStepAction::PushFrame(pending)
                } else {
                    InlineTraceStepAction::Trace(TraceAction::Continue)
                }
            }
            other => InlineTraceStepAction::Trace(other),
        }
    }
}

pub(crate) fn trace_step_result_to_action(
    state: &mut TraceFrameState,
    result: Result<pyre_runtime::StepResult<OpRef>, PyError>,
) -> TraceAction {
    match result {
        Ok(pyre_runtime::StepResult::Continue) => {
            if state.ctx().is_too_long() {
                let green_key = state.ctx().green_key();
                let root_green_key = state.with_ctx(|_, ctx| ctx.root_green_key());
                if let Some(biggest_key) = biggest_inline_trace_key(state) {
                    let (driver, _) = crate::eval::driver_pair();
                    let warm_state = driver.meta_interp_mut().warm_state_mut();
                    warm_state.disable_noninlinable_function(biggest_key);
                    warm_state.trace_next_iteration(root_green_key);
                    if majit_meta::majit_log_enabled() {
                        eprintln!(
                            "[jit][trace-too-long] biggest_inline_key={} trace_next_iteration root_key={}",
                            biggest_key, root_green_key
                        );
                    }
                    return majit_meta::TraceAction::Abort;
                }
                let force_finish_trace = {
                    let (driver, _) = crate::eval::driver_pair();
                    driver.meta_interp().force_finish_trace_enabled()
                };
                if force_finish_trace {
                    let jump_args = state.with_ctx(|this, ctx| this.close_loop_args(ctx));
                    if majit_meta::majit_log_enabled() {
                        eprintln!(
                            "[jit] force_finish_trace: closing loop early at key={}",
                            root_green_key
                        );
                    }
                    return TraceAction::CloseLoopWithArgs {
                        jump_args,
                        loop_header_pc: None,
                    };
                }
                note_root_trace_too_long(root_green_key);
                if majit_meta::majit_log_enabled() {
                    eprintln!(
                        "[jit][abort-reason] trace_too_long key={} root_key={}",
                        green_key, root_green_key
                    );
                }
                TraceAction::Abort
            } else {
                TraceAction::Continue
            }
        }
        Ok(pyre_runtime::StepResult::CloseLoop {
            jump_args,
            loop_header_pc,
        }) => TraceAction::CloseLoopWithArgs {
            jump_args,
            loop_header_pc: Some(loop_header_pc),
        },
        Ok(pyre_runtime::StepResult::Return(value)) => {
            // RPython DoneWithThisFrameDescrInt parity: if the return
            // value is a boxed W_IntObject on the virtualizable stack
            // (type Ref), unbox it to raw Int for the Finish descriptor.
            // This avoids New+SetfieldGc before Finish and matches RPython's
            // done_with_this_frame_descr_int which stores raw ints.
            let (finish_value, finish_type) = match state.value_type(value) {
                Type::Int => (value, Type::Int),
                Type::Float => (value, Type::Float),
                Type::Ref | Type::Void => {
                    let concrete = state.concrete_stack_value_at_return();
                    let is_int = concrete.map_or(false, |v| {
                        !v.is_null() && unsafe { pyre_object::pyobject::is_int(v) }
                    });
                    if is_int {
                        let unboxed =
                            state.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, value));
                        (unboxed, Type::Int)
                    } else {
                        (value, Type::Ref)
                    }
                }
            };
            TraceAction::Finish {
                finish_args: vec![finish_value],
                finish_arg_types: vec![finish_type],
            }
        }
        Ok(pyre_runtime::StepResult::Yield(value)) => {
            let finish_type = match state.value_type(value) {
                Type::Int => Type::Int,
                Type::Float => Type::Float,
                Type::Ref | Type::Void => Type::Ref,
            };
            TraceAction::Finish {
                finish_args: vec![value],
                finish_arg_types: vec![finish_type],
            }
        }
        Err(err) => {
            if majit_meta::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] step_error key={} err={}",
                    state.ctx().green_key(),
                    err
                );
            }
            TraceAction::Abort
        }
    }
}

impl TraceHelperAccess for TraceFrameState {
    fn with_trace_ctx<R>(&mut self, f: impl FnOnce(&mut TraceCtx) -> R) -> R {
        self.with_ctx(|_, ctx| f(ctx))
    }

    fn trace_frame(&self) -> OpRef {
        self.frame()
    }

    fn trace_globals_ptr(&mut self) -> OpRef {
        self.with_ctx(|this, ctx| frame_namespace_ptr(ctx, this.frame()))
    }

    fn trace_record_not_forced_guard(&mut self) {
        self.with_ctx(|this, ctx| {
            this.record_guard(ctx, OpCode::GuardNotForced, &[]);
        });
    }

    fn trace_call_callable(&mut self, callable: OpRef, args: &[OpRef]) -> Result<OpRef, PyError> {
        let frame = self.trace_frame();
        let result = self.with_ctx(|this, ctx| {
            let boxed_args = box_args_for_python_helper(this, ctx, args);
            crate::jit::helpers::emit_trace_call_callable(ctx, frame, callable, &boxed_args)
        })?;
        self.trace_record_not_forced_guard();
        Ok(result)
    }

    fn trace_binary_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: pyre_bytecode::bytecode::BinaryOperator,
    ) -> Result<OpRef, PyError> {
        self.with_ctx(|this, ctx| {
            let lhs = box_value_for_python_helper(this, ctx, a);
            let rhs = box_value_for_python_helper(this, ctx, b);
            crate::jit::helpers::emit_trace_binary_value(ctx, lhs, rhs, op)
        })
    }
}

impl SharedOpcodeHandler for TraceFrameState {
    type Value = OpRef;

    fn push_value(&mut self, value: Self::Value) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            TraceFrameState::push_value(this, ctx, value);
            Ok(())
        })
    }

    fn pop_value(&mut self) -> Result<Self::Value, PyError> {
        self.with_ctx(|this, ctx| TraceFrameState::pop_value(this, ctx))
    }

    fn peek_at(&mut self, depth: usize) -> Result<Self::Value, PyError> {
        // MIFrame Box tracking: set pending concrete from peeked stack position
        let s = self.sym();
        let stack_idx = s.valuestackdepth.checked_sub(s.nlocals + depth + 1);
        if let Some(idx) = stack_idx {
            let concrete = s
                .concrete_stack
                .get(idx)
                .copied()
                .unwrap_or(ConcreteValue::Null);
            self.sym_mut().pending_concrete_push = Some(concrete);
        }
        self.with_ctx(|this, ctx| TraceFrameState::peek_value(this, ctx, depth))
    }

    fn guard_nonnull_value(&mut self, value: Self::Value) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            TraceFrameState::guard_nonnull(this, ctx, value);
            Ok(())
        })
    }

    fn make_function(&mut self, code_obj: Self::Value) -> Result<Self::Value, PyError> {
        self.trace_make_function(code_obj)
    }

    fn call_callable(
        &mut self,
        callable: Self::Value,
        args: &[Self::Value],
    ) -> Result<Self::Value, PyError> {
        // MIFrame Box tracking: compute concrete call result from concrete args.
        // RPython executor.execute_varargs() runs the real function and captures
        // the concrete result. We do the same for both builtins and user functions.
        if let Some(concrete_callable) = self.concrete_callable_after_pops() {
            let concrete_args: Vec<PyObjectRef> = (0..args.len())
                .filter_map(|i| self.concrete_call_arg_after_pops(i))
                .collect();
            if concrete_args.len() == args.len() {
                unsafe {
                    if pyre_runtime::is_builtin_func(concrete_callable) {
                        let func = pyre_runtime::w_builtin_func_get(concrete_callable);
                        let result = func(&concrete_args);
                        self.sym_mut().pending_concrete_push =
                            Some(ConcreteValue::from_pyobj(result));
                    } else if pyre_runtime::is_func(concrete_callable) {
                        // User-defined function: call via plain interpreter
                        // to get concrete result without JIT re-entry.
                        // Concrete call execution — RPython executor.execute_varargs parity.
                        // Depth limit prevents stack overflow on deeply nested calls.
                        use std::cell::Cell;
                        thread_local! {
                            static CONCRETE_CALL_DEPTH: Cell<u32> = Cell::new(0);
                        }
                        let depth = CONCRETE_CALL_DEPTH.with(|d| d.get());
                        if depth < 32 {
                            CONCRETE_CALL_DEPTH.with(|d| d.set(depth + 1));
                            let frame_ptr =
                                self.concrete_frame as *const pyre_interp::frame::PyFrame;
                            let result = pyre_interp::call::call_user_function_plain(
                                unsafe { &*frame_ptr },
                                concrete_callable,
                                &concrete_args,
                            );
                            CONCRETE_CALL_DEPTH.with(|d| d.set(depth));
                            if let Ok(result) = result {
                                self.sym_mut().pending_concrete_push =
                                    Some(ConcreteValue::from_pyobj(result));
                            }
                        }
                    }
                }
            }
        }
        self.call_callable_value(callable, args)
    }

    fn build_list(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        // MIFrame Box tracking: build concrete list from concrete stack values
        let s = self.sym();
        let vsd = s.valuestackdepth;
        let concrete_items: Vec<PyObjectRef> = (0..items.len())
            .map(|i| s.concrete_value_at(vsd + i).to_pyobj())
            .collect();
        if concrete_items.iter().all(|v| !v.is_null()) {
            let list = pyre_runtime::build_list_from_refs(&concrete_items);
            self.sym_mut().pending_concrete_push = Some(ConcreteValue::from_pyobj(list));
        }
        self.trace_build_list(items)
    }

    fn build_tuple(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        // MIFrame Box tracking: build concrete tuple
        let s = self.sym();
        let vsd = s.valuestackdepth;
        let concrete_items: Vec<PyObjectRef> = (0..items.len())
            .map(|i| s.concrete_value_at(vsd + i).to_pyobj())
            .collect();
        if concrete_items.iter().all(|v| !v.is_null()) {
            let tuple = pyre_runtime::build_tuple_from_refs(&concrete_items);
            self.sym_mut().pending_concrete_push = Some(ConcreteValue::from_pyobj(tuple));
        }
        self.trace_build_tuple(items)
    }

    fn build_map(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        // MIFrame Box tracking: build concrete dict
        let s = self.sym();
        let vsd = s.valuestackdepth;
        let concrete_items: Vec<PyObjectRef> = (0..items.len())
            .map(|i| s.concrete_value_at(vsd + i).to_pyobj())
            .collect();
        if concrete_items.iter().all(|v| !v.is_null()) {
            let dict = pyre_runtime::build_map_from_refs(&concrete_items);
            self.sym_mut().pending_concrete_push = Some(ConcreteValue::from_pyobj(dict));
        }
        self.trace_build_map(items)
    }

    fn store_subscr(
        &mut self,
        obj: Self::Value,
        key: Self::Value,
        value: Self::Value,
    ) -> Result<(), PyError> {
        // MIFrame Box tracking: mutate concrete list/dict
        if let Some((c_obj, c_key, c_val)) = self.concrete_store_subscr_operands() {
            let _ = pyre_objspace::space::py_setitem(c_obj, c_key, c_val);
        }
        self.store_subscr_value(obj, key, value)
    }

    fn list_append(&mut self, list: Self::Value, value: Self::Value) -> Result<(), PyError> {
        // MIFrame Box tracking: mutate concrete list
        // list and value are already popped from symbolic stack; their concrete
        // counterparts are in the positions just above current valuestackdepth.
        let s = self.sym();
        let c_list = s.concrete_value_at(s.valuestackdepth).to_pyobj();
        let c_value = s.concrete_value_at(s.valuestackdepth + 1).to_pyobj();
        if !c_list.is_null() && !c_value.is_null() {
            unsafe {
                if pyre_object::is_list(c_list) {
                    pyre_object::w_list_append(c_list, c_value);
                }
            }
        }
        self.list_append_value(list, value)
    }

    fn unpack_sequence(
        &mut self,
        seq: Self::Value,
        count: usize,
    ) -> Result<Vec<Self::Value>, PyError> {
        self.unpack_sequence_value(seq, count)
    }

    fn load_attr(&mut self, obj: Self::Value, name: &str) -> Result<Self::Value, PyError> {
        // MIFrame Box tracking: compute concrete getattr
        if let Some(concrete_obj) = self.concrete_popped_value() {
            if let Ok(result) = pyre_objspace::space::py_getattr(concrete_obj, name) {
                self.sym_mut().pending_concrete_push = Some(ConcreteValue::from_pyobj(result));
            }
        }
        self.trace_load_attr(obj, name)
    }

    fn store_attr(
        &mut self,
        obj: Self::Value,
        name: &str,
        value: Self::Value,
    ) -> Result<(), PyError> {
        self.trace_store_attr(obj, name, value)
    }
}

impl LocalOpcodeHandler for TraceFrameState {
    fn load_local_value(&mut self, idx: usize) -> Result<Self::Value, PyError> {
        // MIFrame Box tracking: set pending concrete from concrete_locals
        let concrete = self
            .sym()
            .concrete_locals
            .get(idx)
            .copied()
            .unwrap_or(ConcreteValue::Null);
        self.sym_mut().pending_concrete_push = Some(concrete);
        self.with_ctx(|this, ctx| TraceFrameState::load_local_value(this, ctx, idx))
    }

    fn load_local_checked_value(&mut self, idx: usize, name: &str) -> Result<Self::Value, PyError> {
        let _ = name;
        // MIFrame Box tracking: set pending concrete from concrete_locals
        let concrete = self
            .sym()
            .concrete_locals
            .get(idx)
            .copied()
            .unwrap_or(ConcreteValue::Null);
        self.sym_mut().pending_concrete_push = Some(concrete);
        let value = self.with_ctx(|this, ctx| TraceFrameState::load_local_value(this, ctx, idx))?;
        if self.value_type(value) == Type::Ref {
            self.with_ctx(|this, ctx| {
                TraceFrameState::guard_nonnull(this, ctx, value);
            });
        }
        Ok(value)
    }

    fn store_local_value(&mut self, idx: usize, value: Self::Value) -> Result<(), PyError> {
        // MIFrame Box tracking: update concrete_locals from last popped value
        let concrete = self
            .sym()
            .last_popped_concrete_value
            .map(ConcreteValue::from_pyobj)
            .unwrap_or(ConcreteValue::Null);
        if idx < self.sym().concrete_locals.len() {
            self.sym_mut().concrete_locals[idx] = concrete;
        }
        self.with_ctx(|this, ctx| TraceFrameState::store_local_value(this, ctx, idx, value))
    }
}

impl NamespaceOpcodeHandler for TraceFrameState {
    fn load_name_value(&mut self, name: &str) -> Result<Self::Value, PyError> {
        let ns = self.sym().concrete_namespace;
        let Some(slot) = namespace_slot_direct(ns, name)
            .or_else(|| concrete_namespace_slot(self.concrete_frame, name))
        else {
            return self.trace_load_name(name);
        };
        // MIFrame Box tracking: set concrete value for global load
        if let Some(cv) = namespace_value_direct(ns, slot)
            .or_else(|| concrete_namespace_value(self.concrete_frame, slot))
        {
            self.sym_mut().pending_concrete_push = Some(ConcreteValue::from_pyobj(cv));
        }
        if let Some(concrete_value) = namespace_value_direct(ns, slot)
            .or_else(|| concrete_namespace_value(self.concrete_frame, slot))
        {
            unsafe {
                if is_func(concrete_value) || is_builtin_func(concrete_value) {
                    return self.with_ctx(|this, ctx| {
                        let loaded = TraceFrameState::load_namespace_value(this, ctx, slot)?;
                        this.guard_value(ctx, loaded, concrete_value as i64);
                        let const_value = ctx.const_int(concrete_value as i64);
                        this.sym_mut()
                            .symbolic_namespace_slots
                            .insert(slot, const_value);
                        Ok(const_value)
                    });
                }
            }
        }
        self.with_ctx(|this, ctx| TraceFrameState::load_namespace_value(this, ctx, slot))
    }

    fn store_name_value(&mut self, name: &str, value: Self::Value) -> Result<(), PyError> {
        let ns = self.sym().concrete_namespace;
        let Some(slot) = namespace_slot_direct(ns, name)
            .or_else(|| concrete_namespace_slot(self.concrete_frame, name))
        else {
            return self.trace_store_name(name, value);
        };
        self.with_ctx(|this, ctx| TraceFrameState::store_namespace_value(this, ctx, slot, value))
    }

    fn null_value(&mut self) -> Result<Self::Value, PyError> {
        self.sym_mut().pending_concrete_push = Some(ConcreteValue::Null);
        self.trace_null_value()
    }
}

impl StackOpcodeHandler for TraceFrameState {
    fn swap_values(&mut self, depth: usize) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| TraceFrameState::swap_values(this, ctx, depth))
    }
}

impl IterOpcodeHandler for TraceFrameState {
    fn ensure_iter_value(&mut self, iter: Self::Value) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            TraceFrameState::guard_range_iter(this, ctx, iter);
            Ok(())
        })
    }

    fn concrete_iter_continues(&mut self, _iter: Self::Value) -> Result<bool, PyError> {
        TraceFrameState::concrete_iter_continues(self)
    }

    fn iter_next_value(&mut self, iter: Self::Value) -> Result<Self::Value, PyError> {
        TraceFrameState::iter_next_value(self, iter)
    }

    fn guard_optional_value(&mut self, next: Self::Value, continues: bool) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            TraceFrameState::record_for_iter_guard(this, ctx, next, continues);
            Ok(())
        })
    }

    fn on_iter_exhausted(&mut self, target: usize) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            TraceFrameState::set_next_instr(this, ctx, target);
            Ok(())
        })
    }
}

impl TruthOpcodeHandler for TraceFrameState {
    type Truth = OpRef;

    fn truth_value(&mut self, value: Self::Value) -> Result<Self::Truth, PyError> {
        self.truth_value_direct(value)
    }

    fn bool_value_from_truth(
        &mut self,
        truth: Self::Truth,
        negate: bool,
    ) -> Result<Self::Value, PyError> {
        // MIFrame Box tracking: compute concrete bool from truth
        if let Some(concrete_truth) = self.sym().last_comparison_concrete_truth {
            let result = if negate {
                !concrete_truth
            } else {
                concrete_truth
            };
            self.sym_mut().pending_concrete_push = Some(ConcreteValue::Int(result as i64));
        }
        self.trace_bool_value_from_truth(truth, negate)
    }
}

impl ControlFlowOpcodeHandler for TraceFrameState {
    fn fallthrough_target(&mut self) -> usize {
        self.fallthrough_pc()
    }

    fn set_next_instr(&mut self, target: usize) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            TraceFrameState::set_next_instr(this, ctx, target);
            Ok(())
        })
    }

    fn close_loop_args(&mut self, target: usize) -> Result<Option<Vec<Self::Value>>, PyError> {
        self.with_ctx(|this, ctx| {
            // RPython reached_loop_header (pyjitpl.py:2988-3030):
            // Check current_merge_points for same_greenkey. If this is
            // an inner loop's first visit, record and continue (unroll).
            let code_ptr =
                unsafe { &*(this.concrete_frame as *const pyre_interp::frame::PyFrame) }.code;
            let back_edge_key = crate::eval::make_green_key(code_ptr, target);
            if !ctx.has_merge_point(back_edge_key) {
                // First visit: record merge point and continue tracing.
                // pyjitpl.py:3030: save live args + types as original_boxes.
                let live_args = TraceFrameState::close_loop_args(this, ctx);
                let live_types = {
                    let s = this.sym();
                    let mut types = vec![Type::Ref, Type::Int, Type::Int]; // frame, ni, vsd
                    types.extend(s.symbolic_local_types.iter().copied());
                    let stack_only = s.stack_only_depth();
                    types.extend(
                        s.symbolic_stack_types[..stack_only.min(s.symbolic_stack_types.len())]
                            .iter()
                            .copied(),
                    );
                    types
                };
                ctx.add_merge_point(back_edge_key, live_args, live_types);
                TraceFrameState::set_next_instr(this, ctx, target);
                if majit_meta::majit_log_enabled() {
                    eprintln!(
                        "[jit][reached_loop_header] first visit, unroll: key={} pc={}",
                        back_edge_key, target
                    );
                }
                return Ok(None); // Continue — inner loop unrolled
            }
            // Same green key seen → close the loop.
            TraceFrameState::set_next_instr(this, ctx, target);
            Ok(Some(TraceFrameState::close_loop_args(this, ctx)))
        })
    }
}

impl BranchOpcodeHandler for TraceFrameState {
    fn enter_branch_truth(&mut self, value: Self::Value) -> Result<(), PyError> {
        self.sym_mut().pending_branch_value = Some(value);
        Ok(())
    }

    fn leave_branch_truth(&mut self) -> Result<(), PyError> {
        let sym = self.sym_mut();
        sym.pending_branch_value = None;
        sym.last_popped_concrete_value = None;
        Ok(())
    }

    fn concrete_truth_as_bool(
        &mut self,
        value: Self::Value,
        _truth: Self::Truth,
    ) -> Result<bool, PyError> {
        TraceFrameState::concrete_branch_truth_for_value(self, value)
    }

    fn guard_truth_value(&mut self, truth: Self::Truth, expect_true: bool) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            let opcode = if expect_true {
                OpCode::GuardTrue
            } else {
                OpCode::GuardFalse
            };
            TraceFrameState::record_guard(this, ctx, opcode, &[truth]);
            Ok(())
        })
    }

    fn record_branch_guard(
        &mut self,
        value: Self::Value,
        truth: Self::Truth,
        concrete_truth: bool,
    ) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            TraceFrameState::record_branch_guard(this, ctx, value, truth, concrete_truth);
            Ok(())
        })
    }
}

impl ArithmeticOpcodeHandler for TraceFrameState {
    fn binary_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: BinaryOperator,
    ) -> Result<Self::Value, PyError> {
        // MIFrame Box tracking: compute concrete result for binary ops
        // Float path
        if let Some((lhs_obj, rhs_obj)) = self.concrete_binary_operands() {
            unsafe {
                let is_float_op = is_float(lhs_obj) || is_float(rhs_obj);
                if is_float_op {
                    let lhs_f = if is_float(lhs_obj) {
                        w_float_get_value(lhs_obj)
                    } else if is_int(lhs_obj) {
                        w_int_get_value(lhs_obj) as f64
                    } else {
                        0.0
                    };
                    let rhs_f = if is_float(rhs_obj) {
                        w_float_get_value(rhs_obj)
                    } else if is_int(rhs_obj) {
                        w_int_get_value(rhs_obj) as f64
                    } else {
                        0.0
                    };
                    let result = match op {
                        BinaryOperator::Add | BinaryOperator::InplaceAdd => lhs_f + rhs_f,
                        BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => lhs_f - rhs_f,
                        BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => lhs_f * rhs_f,
                        BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide
                            if rhs_f != 0.0 =>
                        {
                            lhs_f / rhs_f
                        }
                        BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide
                            if rhs_f != 0.0 =>
                        {
                            (lhs_f / rhs_f).floor()
                        }
                        BinaryOperator::Remainder | BinaryOperator::InplaceRemainder
                            if rhs_f != 0.0 =>
                        {
                            lhs_f % rhs_f
                        }
                        _ => f64::NAN,
                    };
                    if !result.is_nan()
                        || matches!(
                            op,
                            BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide
                        )
                    {
                        self.sym_mut().pending_concrete_push = Some(ConcreteValue::Float(result));
                    }
                }
            }
        }
        // Int path
        if let Some((lhs, rhs)) = self.concrete_binary_int_operands() {
            let result = match op {
                BinaryOperator::Add | BinaryOperator::InplaceAdd => lhs.wrapping_add(rhs),
                BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => lhs.wrapping_sub(rhs),
                BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => lhs.wrapping_mul(rhs),
                BinaryOperator::Remainder | BinaryOperator::InplaceRemainder if rhs != 0 => {
                    ((lhs % rhs) + rhs) % rhs // Python modulo semantics
                }
                BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide if rhs != 0 => {
                    let d = lhs.wrapping_div(rhs);
                    if (lhs ^ rhs) < 0 && d * rhs != lhs {
                        d - 1
                    } else {
                        d
                    }
                }
                BinaryOperator::And | BinaryOperator::InplaceAnd => lhs & rhs,
                BinaryOperator::Or | BinaryOperator::InplaceOr => lhs | rhs,
                BinaryOperator::Xor | BinaryOperator::InplaceXor => lhs ^ rhs,
                BinaryOperator::Lshift | BinaryOperator::InplaceLshift => {
                    lhs.wrapping_shl(rhs as u32)
                }
                BinaryOperator::Rshift | BinaryOperator::InplaceRshift => {
                    lhs.wrapping_shr(rhs as u32)
                }
                _ => 0, // fallback
            };
            self.sym_mut().pending_concrete_push = Some(ConcreteValue::Int(result));
        }
        if matches!(op, BinaryOperator::Subscr) {
            self.binary_subscr_value(a, b)
        } else if self.concrete_binary_float_operands()
            || self.value_type(a) == Type::Float
            || self.value_type(b) == Type::Float
        {
            self.binary_float_value(a, b, op)
        } else {
            self.binary_int_value(a, b, op)
        }
    }

    fn compare_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: ComparisonOperator,
    ) -> Result<Self::Value, PyError> {
        // MIFrame Box tracking: compute concrete comparison result
        if let Some((lhs_obj, rhs_obj)) = self.concrete_binary_operands() {
            unsafe {
                if is_float(lhs_obj) || is_float(rhs_obj) {
                    let lhs_f = if is_float(lhs_obj) {
                        w_float_get_value(lhs_obj)
                    } else if is_int(lhs_obj) {
                        w_int_get_value(lhs_obj) as f64
                    } else {
                        0.0
                    };
                    let rhs_f = if is_float(rhs_obj) {
                        w_float_get_value(rhs_obj)
                    } else if is_int(rhs_obj) {
                        w_int_get_value(rhs_obj) as f64
                    } else {
                        0.0
                    };
                    let result = match op {
                        ComparisonOperator::Less => lhs_f < rhs_f,
                        ComparisonOperator::LessOrEqual => lhs_f <= rhs_f,
                        ComparisonOperator::Greater => lhs_f > rhs_f,
                        ComparisonOperator::GreaterOrEqual => lhs_f >= rhs_f,
                        ComparisonOperator::Equal => lhs_f == rhs_f,
                        ComparisonOperator::NotEqual => lhs_f != rhs_f,
                    };
                    self.sym_mut().pending_concrete_push = Some(ConcreteValue::Int(result as i64));
                }
            }
        }
        if let Some((lhs, rhs)) = self.concrete_binary_int_operands() {
            let result = match op {
                ComparisonOperator::Less => lhs < rhs,
                ComparisonOperator::LessOrEqual => lhs <= rhs,
                ComparisonOperator::Greater => lhs > rhs,
                ComparisonOperator::GreaterOrEqual => lhs >= rhs,
                ComparisonOperator::Equal => lhs == rhs,
                ComparisonOperator::NotEqual => lhs != rhs,
            };
            self.sym_mut().pending_concrete_push = Some(ConcreteValue::Int(result as i64));
        }
        self.compare_value_direct(a, b, op)
    }

    fn unary_negative_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        // MIFrame Box tracking: compute concrete negation
        if let Some(v) = self.concrete_unary_int_operand() {
            self.sym_mut().pending_concrete_push = Some(ConcreteValue::Int(v.wrapping_neg()));
        }
        self.unary_int_value(value, OpCode::IntNeg)
    }

    fn unary_invert_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        // MIFrame Box tracking: compute concrete bitwise invert
        if let Some(v) = self.concrete_unary_int_operand() {
            self.sym_mut().pending_concrete_push = Some(ConcreteValue::Int(!v));
        }
        self.unary_int_value(value, OpCode::IntInvert)
    }
}

impl ConstantOpcodeHandler for TraceFrameState {
    fn int_constant(&mut self, value: i64) -> Result<Self::Value, PyError> {
        self.sym_mut().pending_concrete_push = Some(ConcreteValue::Int(value));
        self.trace_int_constant(value)
    }

    fn bigint_constant(&mut self, value: &PyBigInt) -> Result<Self::Value, PyError> {
        self.sym_mut().pending_concrete_push =
            Some(ConcreteValue::Ref(pyre_object::w_long_new(value.clone())));
        self.trace_bigint_constant(value)
    }

    fn float_constant(&mut self, value: f64) -> Result<Self::Value, PyError> {
        self.sym_mut().pending_concrete_push = Some(ConcreteValue::Float(value));
        self.trace_float_constant(value)
    }

    fn bool_constant(&mut self, value: bool) -> Result<Self::Value, PyError> {
        self.sym_mut().pending_concrete_push = Some(ConcreteValue::Int(value as i64));
        self.trace_bool_constant(value)
    }

    fn str_constant(&mut self, value: &str) -> Result<Self::Value, PyError> {
        self.sym_mut().pending_concrete_push =
            Some(ConcreteValue::Ref(pyre_object::w_str_new(value)));
        self.trace_str_constant(value)
    }

    fn code_constant(&mut self, code: &CodeObject) -> Result<Self::Value, PyError> {
        // Code objects don't have a simple concrete repr; use PY_NULL
        self.trace_code_constant(code)
    }

    fn none_constant(&mut self) -> Result<Self::Value, PyError> {
        self.sym_mut().pending_concrete_push = Some(ConcreteValue::Ref(pyre_object::w_none()));
        self.trace_none_constant()
    }
}

impl OpcodeStepExecutor for TraceFrameState {
    type Error = PyError;

    fn unsupported(
        &mut self,
        instruction: &Instruction,
    ) -> Result<pyre_runtime::StepResult<OpRef>, Self::Error> {
        Err(PyError::type_error(format!(
            "unsupported instruction during trace: {instruction:?}"
        )))
    }
}

impl PyreJitState {
    /// RPython resume.py parity: restore virtualizable frame state directly
    /// from raw output buffer values, bypassing exit_layout type decoding.
    ///
    /// The optimizer may type virtualizable slots as Int (after unboxing),
    /// but the raw values in the output buffer are always the actual Python
    /// objects (PyObjectRef pointers). RPython uses FieldDescr at decode time
    /// to determine types; pyre uses the virtualizable layout to know that
    /// all array slots (locals + stack) are always Ref (GCREF).
    pub fn restore_virtualizable_from_raw(&mut self, raw_values: &[i64]) -> bool {
        if raw_values.is_empty() {
            return false;
        }
        self.frame = raw_values[0] as usize;
        self.next_instr = raw_values.get(1).copied().unwrap_or(0) as usize;
        self.valuestackdepth = raw_values.get(2).copied().unwrap_or(0) as usize;

        let nlocals = self.local_count();
        let stack_only = self.valuestackdepth.saturating_sub(nlocals);
        let mut idx = 3;
        for local_idx in 0..nlocals {
            if let Some(&raw) = raw_values.get(idx) {
                let _ = self.set_local_at(local_idx, raw as PyObjectRef);
            }
            idx += 1;
        }
        for stack_idx in 0..stack_only {
            if let Some(&raw) = raw_values.get(idx) {
                let _ = self.set_stack_at(stack_idx, raw as PyObjectRef);
            }
            idx += 1;
        }
        self.sync_scalar_fields_to_frame()
    }

    /// Returns true if the optimizer virtualizable mechanism is active.
    fn has_virtualizable_info(&self) -> bool {
        // pyre always uses virtualizable (JitDriverStaticData::with_virtualizable)
        true
    }

    fn frame_ptr(&self) -> Option<*mut u8> {
        (self.frame != 0).then_some(self.frame as *mut u8)
    }

    fn frame_array(&self, offset: usize) -> Option<&PyObjectArray> {
        let frame_ptr = self.frame_ptr()?;
        Some(unsafe { &*(frame_ptr.add(offset) as *const PyObjectArray) })
    }

    fn frame_array_mut(&mut self, offset: usize) -> Option<&mut PyObjectArray> {
        let frame_ptr = self.frame_ptr()?;
        Some(unsafe { &mut *(frame_ptr.add(offset) as *mut PyObjectArray) })
    }

    fn read_frame_usize(&self, offset: usize) -> Option<usize> {
        let frame_ptr = self.frame_ptr()?;
        Some(unsafe { *(frame_ptr.add(offset) as *const usize) })
    }

    fn write_frame_usize(&mut self, offset: usize, value: usize) -> bool {
        let Some(frame_ptr) = self.frame_ptr() else {
            return false;
        };
        unsafe {
            *(frame_ptr.add(offset) as *mut usize) = value;
        }
        true
    }

    fn locals_cells_stack_array(&self) -> Option<&PyObjectArray> {
        self.frame_array(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
    }

    fn locals_cells_stack_array_mut(&mut self) -> Option<&mut PyObjectArray> {
        self.frame_array_mut(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
    }

    fn namespace_ptr(&self) -> Option<*mut PyNamespace> {
        let frame_ptr = self.frame_ptr()?;
        let namespace_ptr =
            unsafe { *(frame_ptr.add(PYFRAME_NAMESPACE_OFFSET) as *const *mut PyNamespace) };
        (!namespace_ptr.is_null()).then_some(namespace_ptr)
    }

    fn namespace_len(&self) -> usize {
        let Some(namespace_ptr) = self.namespace_ptr() else {
            return 0;
        };
        unsafe { (*namespace_ptr).len() }
    }

    fn namespace_keys(&self) -> Vec<String> {
        let Some(namespace_ptr) = self.namespace_ptr() else {
            return Vec::new();
        };
        let namespace = unsafe { &*namespace_ptr };
        let mut keys: Vec<String> = namespace.keys().cloned().collect();
        keys.sort();
        keys
    }

    fn restore_single_frame(&mut self, meta: &PyreMeta, values: &[i64]) {
        let Some(&frame) = values.first() else {
            return;
        };
        self.frame = frame as usize;
        if values.len() == 1 {
            let _ = self.refresh_from_frame();
            return;
        }
        if meta.has_virtualizable {
            self.restore_virtualizable_i64(values);
        } else {
            let nlocals = self.local_count();
            let stack_only = self.valuestackdepth.saturating_sub(nlocals);
            let mut idx = 1;
            for local_idx in 0..nlocals {
                if idx < values.len() {
                    let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                    let _ = self
                        .set_local_at(local_idx, boxed_slot_i64_for_type(slot_type, values[idx]));
                }
                idx += 1;
            }
            for i in 0..stack_only {
                if idx < values.len() {
                    let slot_type = meta
                        .slot_types
                        .get(nlocals + i)
                        .copied()
                        .unwrap_or(Type::Ref);
                    let _ = self.set_stack_at(i, boxed_slot_i64_for_type(slot_type, values[idx]));
                }
                idx += 1;
            }
        }
        // Write next_instr and valuestackdepth back to the concrete PyFrame.
        let _ = self.sync_scalar_fields_to_frame();
    }

    pub fn local_at(&self, idx: usize) -> Option<PyObjectRef> {
        self.locals_cells_stack_array()
            .and_then(|arr| arr.as_slice().get(idx).copied())
    }

    /// Number of local variable slots.
    pub fn local_count(&self) -> usize {
        concrete_nlocals(self.frame).unwrap_or(0)
    }

    pub fn set_local_at(&mut self, idx: usize, value: PyObjectRef) -> bool {
        let Some(arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        let Some(slot) = arr.as_mut_slice().get_mut(idx) else {
            return false;
        };
        *slot = value;
        true
    }

    /// Read a stack value at stack-relative index `idx` (0-based from stack bottom).
    pub fn stack_at(&self, idx: usize) -> Option<PyObjectRef> {
        let nlocals = self.local_count();
        self.locals_cells_stack_array()
            .and_then(|arr| arr.as_slice().get(nlocals + idx).copied())
    }

    /// Total capacity of the unified array.
    pub fn array_capacity(&self) -> usize {
        self.locals_cells_stack_array()
            .map(PyObjectArray::len)
            .unwrap_or(0)
    }

    /// Set a stack value at stack-relative index `idx`.
    pub fn set_stack_at(&mut self, idx: usize, value: PyObjectRef) -> bool {
        let nlocals = self.local_count();
        let Some(arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        let Some(slot) = arr.as_mut_slice().get_mut(nlocals + idx) else {
            return false;
        };
        *slot = value;
        true
    }

    fn sync_scalar_fields_to_frame(&mut self) -> bool {
        self.write_frame_usize(PYFRAME_NEXT_INSTR_OFFSET, self.next_instr)
            && self.write_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET, self.valuestackdepth)
    }

    fn refresh_from_frame(&mut self) -> bool {
        let Some(next_instr) = self.read_frame_usize(PYFRAME_NEXT_INSTR_OFFSET) else {
            return false;
        };
        let Some(valuestackdepth) = self.read_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET) else {
            return false;
        };
        if self.locals_cells_stack_array().is_none() {
            return false;
        }

        self.next_instr = next_instr;
        self.valuestackdepth = valuestackdepth;
        true
    }

    /// Restore from virtualizable fail_args format:
    ///   [frame, next_instr, valuestackdepth, l0..lN-1, s0..sM-1]
    fn restore_virtualizable_i64(&mut self, values: &[i64]) {
        let mut idx = 1;

        // Static fields: next_instr, valuestackdepth
        if idx < values.len() {
            self.next_instr = values[idx] as usize;
            idx += 1;
        }
        if idx < values.len() {
            self.valuestackdepth = values[idx] as usize;
            idx += 1;
        }

        let nlocals = self.local_count();

        // Locals follow directly (indices 0..nlocals in unified array)
        for i in 0..nlocals {
            if idx < values.len() {
                let slot_type = concrete_value_type(self.local_at(i).unwrap_or(PY_NULL));
                let _ = self.set_local_at(i, boxed_slot_i64_for_type(slot_type, values[idx]));
                idx += 1;
            }
        }

        // Stack values follow locals (indices nlocals..valuestackdepth)
        let stack_only = self.valuestackdepth.saturating_sub(nlocals);
        for i in 0..stack_only {
            if idx < values.len() {
                let slot_type = concrete_value_type(self.stack_at(i).unwrap_or(PY_NULL));
                let _ = self.set_stack_at(i, boxed_slot_i64_for_type(slot_type, values[idx]));
                idx += 1;
            }
        }
    }

    fn import_virtualizable_state(
        &mut self,
        static_boxes: &[i64],
        array_boxes: &[Vec<i64>],
    ) -> bool {
        let Some(&next_instr) = static_boxes.first() else {
            return false;
        };
        let Some(&valuestackdepth) = static_boxes.get(1) else {
            return false;
        };
        // Single unified array: locals_cells_stack_w
        let Some(unified) = array_boxes.first() else {
            return false;
        };

        self.next_instr = next_instr as usize;
        self.valuestackdepth = valuestackdepth as usize;

        let Some(frame_arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        if frame_arr.len() != unified.len() {
            return false;
        }
        for (dst, &src) in frame_arr.as_mut_slice().iter_mut().zip(unified) {
            *dst = src as PyObjectRef;
        }

        self.sync_scalar_fields_to_frame()
            && self
                .read_frame_usize(PYFRAME_NEXT_INSTR_OFFSET)
                .is_some_and(|next_instr| next_instr == self.next_instr)
            && self
                .read_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET)
                .is_some_and(|vsd| vsd == self.valuestackdepth)
    }

    fn export_virtualizable_state(&self) -> (Vec<i64>, Vec<Vec<i64>>) {
        let static_boxes = vec![self.next_instr as i64, self.valuestackdepth as i64];
        // Single unified array
        let array_boxes = vec![
            self.locals_cells_stack_array()
                .map(|arr| arr.as_slice().iter().map(|&value| value as i64).collect())
                .unwrap_or_default(),
        ];
        (static_boxes, array_boxes)
    }

    pub fn sync_from_virtualizable(&mut self, info: &VirtualizableInfo) -> bool {
        let _ = info;
        // RPython pre-run sync reads the live virtualizable state from the
        // concrete frame; it does not materialize resume-data back into the
        // frame before entering compiled code.  PyreJitState already uses the
        // concrete PyFrame as the source of truth for locals/stack, so the
        // entry sync only needs to refresh scalar fields from that frame.
        self.refresh_from_frame()
    }

    pub fn sync_to_virtualizable(&self, info: &VirtualizableInfo) -> bool {
        let Some(frame_ptr) = self.frame_ptr() else {
            return false;
        };
        let (static_boxes, array_boxes) = self.export_virtualizable_state();
        unsafe {
            info.write_from_resume_data_partial(frame_ptr, &static_boxes, &array_boxes);
            info.clear_vable_token(frame_ptr, |_token| {
                // Force callback — in RPython this forces the virtualizable.
                // In pyre, the frame is already restored above.
            });
        }
        true
    }
}

impl JitState for PyreJitState {
    type Meta = PyreMeta;
    type Sym = PyreSym;
    type Env = PyreEnv;

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        let num_locals = self.local_count();
        let slot_types = concrete_slot_types(self.frame, num_locals, self.valuestackdepth);
        PyreMeta {
            merge_pc: header_pc,
            num_locals,
            ns_keys: self.namespace_keys(),
            valuestackdepth: self.valuestackdepth,
            has_virtualizable: self.has_virtualizable_info(),
            slot_types,
        }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        self.extract_live_values(_meta)
            .into_iter()
            .map(|value| match value {
                Value::Int(v) => v,
                Value::Float(v) => v.to_bits() as i64,
                Value::Ref(r) => r.as_usize() as i64,
                Value::Void => 0,
            })
            .collect()
    }

    fn extract_live_values(&self, meta: &Self::Meta) -> Vec<Value> {
        let nlocals = meta.num_locals;
        let stack_only = meta.valuestackdepth.saturating_sub(nlocals);
        let mut vals = vec![
            Value::Ref(majit_ir::GcRef(self.frame)),
            Value::Int(meta.merge_pc as i64),
            Value::Int(self.valuestackdepth as i64),
        ];
        for i in 0..nlocals {
            let value = self.local_at(i).unwrap_or(PY_NULL);
            let slot_type = meta.slot_types.get(i).copied().unwrap_or(Type::Ref);
            vals.push(extract_concrete_typed_value(slot_type, value));
        }
        for i in 0..stack_only {
            let value = self.stack_at(i).unwrap_or(PY_NULL);
            let slot_type = meta
                .slot_types
                .get(nlocals + i)
                .copied()
                .unwrap_or(Type::Ref);
            vals.push(extract_concrete_typed_value(slot_type, value));
        }
        vals
    }

    fn live_value_types(&self, meta: &Self::Meta) -> Vec<Type> {
        virtualizable_fail_arg_types(meta.slot_types.iter().copied())
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        let mut sym = PyreSym::new_uninit(OpRef(0));
        sym.vable_next_instr = OpRef(1);
        sym.vable_valuestackdepth = OpRef(2);
        // Unified array: locals at base+0..base+nlocals, stack at base+nlocals..
        sym.vable_array_base = Some(3); // starts after frame(0), ni(1), vsd(2)
        sym.nlocals = _meta.num_locals;
        sym.symbolic_local_types =
            _meta.slot_types[.._meta.num_locals.min(_meta.slot_types.len())].to_vec();
        sym.symbolic_stack_types =
            _meta.slot_types[_meta.num_locals.min(_meta.slot_types.len())..].to_vec();
        sym
    }

    fn driver_descriptor(&self, _meta: &Self::Meta) -> Option<JitDriverStaticData> {
        None
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        // RPython warmstate.py: only green key matching + token validity.
        // Concrete slot types are NOT checked — preamble guards catch
        // type mismatches at runtime.
        self.next_instr == meta.merge_pc
            && self.local_count() == meta.num_locals
            && self.namespace_len() == meta.ns_keys.len()
            && self.valuestackdepth == meta.valuestackdepth
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        if values.is_empty() {
            return;
        }

        // Multi-frame format: [num_frames, size_0, data_0..., size_1, data_1...]
        // Detect: values[0] is a small number (1-10) = frame count
        // Legacy: values[0] is a large pointer
        let first = values[0];
        if first >= 1 && first <= 10 && values.len() > 2 {
            let _num_frames = first as usize;
            let outer_size = values[1] as usize;
            if outer_size > 0 && 2 + outer_size <= values.len() {
                // Restore outermost frame only.
                // Inner frame guard failure → interpreter re-executes callee call.
                self.restore_single_frame(meta, &values[2..2 + outer_size]);
                return;
            }
        }

        // Legacy single-frame format
        self.restore_single_frame(meta, values);
    }

    fn restore_values(&mut self, meta: &Self::Meta, values: &[Value]) {
        let Some(frame) = values.first() else {
            return;
        };
        self.frame = value_to_usize(frame);
        if majit_meta::majit_log_enabled() {
            let arg0 = self.local_at(0).and_then(|value| {
                if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
                    return None;
                }
                Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
            });
            eprintln!(
                "[jit][restore_values] before arg0={:?} meta.pc={} meta.vsd={} has_vable={} values={:?}",
                arg0, meta.merge_pc, meta.valuestackdepth, meta.has_virtualizable, values
            );
        }
        if values.len() == 1 {
            let _ = self.refresh_from_frame();
            return;
        }

        if meta.has_virtualizable {
            // RPython parity:
            // - normal compiled loop JUMPs resume at the loop header merge point
            // - guard failures rebuild the current frame state separately via
            //   restore_guard_failure_values()
            //
            // So restore_values() must treat virtualizable jump outcomes as
            // loop-header state, not as "current opcode" state.
            self.next_instr = meta.merge_pc;
            self.valuestackdepth = meta.valuestackdepth;
            let nlocals = self.local_count();
            let stack_only = self.valuestackdepth.saturating_sub(nlocals);
            let mut idx = 3;
            for local_idx in 0..nlocals {
                if let Some(value) = values.get(idx) {
                    let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                    let _ =
                        self.set_local_at(local_idx, boxed_slot_value_for_type(slot_type, value));
                }
                idx += 1;
            }
            for i in 0..stack_only {
                if let Some(value) = values.get(idx) {
                    let slot_type = meta
                        .slot_types
                        .get(nlocals + i)
                        .copied()
                        .unwrap_or(Type::Ref);
                    let _ = self.set_stack_at(i, boxed_slot_value_for_type(slot_type, value));
                }
                idx += 1;
            }
        } else {
            let nlocals = self.local_count();
            let stack_only_depth = meta.valuestackdepth.saturating_sub(nlocals);
            let mut idx = 1;
            for local_idx in 0..nlocals {
                let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                let _ = self.set_local_at(
                    local_idx,
                    boxed_slot_value_for_type(slot_type, &values[idx]),
                );
                idx += 1;
            }
            for i in 0..stack_only_depth {
                let slot_type = meta
                    .slot_types
                    .get(nlocals + i)
                    .copied()
                    .unwrap_or(Type::Ref);
                let _ = self.set_stack_at(i, boxed_slot_value_for_type(slot_type, &values[idx]));
                idx += 1;
            }
            self.valuestackdepth = meta.valuestackdepth;
        }
        let _ = self.sync_scalar_fields_to_frame();
        if majit_meta::majit_log_enabled() {
            let arg0 = self.local_at(0).and_then(|value| {
                if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
                    return None;
                }
                Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
            });
            eprintln!(
                "[jit][restore_values] after arg0={:?} ni={} vsd={}",
                arg0, self.next_instr, self.valuestackdepth
            );
        }
    }

    fn restore_guard_failure_values(
        &mut self,
        meta: &Self::Meta,
        values: &[Value],
        _exception: &majit_meta::blackhole::ExceptionState,
    ) -> bool {
        if !meta.has_virtualizable {
            self.restore_values(meta, values);
            return true;
        }

        let Some(frame) = values.first() else {
            return false;
        };
        self.frame = value_to_usize(frame);
        if values.len() == 1 {
            return self.refresh_from_frame();
        }

        // RPython resume.py rebuilds guard-failure state from the typed
        // INT/REF/FLOAT streams carried by resumedata instead of reusing the
        // trace-entry slot kinds.  Virtualizable guards in pyre can likewise
        // exit with more precise slot kinds than `meta.slot_types`
        // (e.g. raw loop indices), so restore from the runtime value kind.
        self.next_instr = values.get(1).map(value_to_usize).unwrap_or(self.next_instr);
        // RPython parity: blackhole creates a fresh frame from resumedata.
        // In pyre we reuse the same frame, so clamp valuestackdepth to
        // frame array capacity — extra fail_args from virtual materialization
        // can inflate the count beyond bounds.
        let arr_cap = self
            .locals_cells_stack_array()
            .map(|a| unsafe { (*a).len() })
            .unwrap_or(usize::MAX);
        self.valuestackdepth = values
            .get(2)
            .map(value_to_usize)
            .unwrap_or(self.valuestackdepth)
            .min(arr_cap);

        let nlocals = self.local_count();
        let stack_only = self.valuestackdepth.saturating_sub(nlocals);
        let mut idx = 3;
        for local_idx in 0..nlocals {
            if let Some(value) = values.get(idx) {
                // RPython parity: virtualizable array slots are always GCREF.
                // Values with trace-level Int type that are actually heap
                // pointers must be treated as Ref for the Python frame.
                let boxed = boxed_slot_value_from_runtime_kind(value);
                let _ = self.set_local_at(local_idx, boxed);
            }
            idx += 1;
        }
        for stack_idx in 0..stack_only {
            if let Some(value) = values.get(idx) {
                let boxed = boxed_slot_value_from_runtime_kind(value);
                let _ = self.set_stack_at(stack_idx, boxed);
            }
            idx += 1;
        }

        // Clear stale slots beyond valuestackdepth (blackhole fresh frame parity).
        let vsd = self.valuestackdepth;
        if let Some(arr) = self.locals_cells_stack_array_mut() {
            for i in vsd..arr.len() {
                arr[i] = pyre_object::PY_NULL;
            }
        }

        self.sync_scalar_fields_to_frame()
    }

    fn reconstructed_frame_value_types(
        &self,
        meta: &Self::Meta,
        frame_index: usize,
        total_frames: usize,
        _frame_pc: u64,
    ) -> Option<Vec<Type>> {
        let _ = meta;
        if frame_index + 1 != total_frames {
            return None;
        }
        Some(vec![Type::Ref])
    }

    fn restore_reconstructed_frame_values(
        &mut self,
        _meta: &Self::Meta,
        frame_index: usize,
        total_frames: usize,
        frame_pc: u64,
        values: &[Value],
        _exception: &majit_meta::blackhole::ExceptionState,
    ) -> bool {
        if frame_index + 1 != total_frames {
            return true;
        }

        let Some(frame) = values.first() else {
            return false;
        };

        self.frame = value_to_usize(frame);
        if !self.refresh_from_frame() {
            return false;
        }
        self.next_instr = frame_pc as usize;
        self.sync_scalar_fields_to_frame()
    }

    fn virtualizable_heap_ptr(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<*mut u8> {
        self.frame_ptr()
    }

    fn virtualizable_array_lengths(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<Vec<usize>> {
        // RPython initializes virtualizable_boxes from the current trace-entry
        // boxes.  Pyre's trace-entry live boxes only carry the active prefix of
        // `locals_cells_stack_w` (locals/cells plus live stack items), so the
        // fallback length must describe that live prefix instead of the full
        // heap capacity.
        Some(vec![self.valuestackdepth])
    }

    fn sync_virtualizable_before_jit(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> bool {
        // RPython enters compiled code with the concrete virtualizable already
        // authoritative.  PyreJitState does not maintain a detached copy of
        // locals/stack, so importing resume-data back into the frame here
        // would mutate the live PyFrame before the compiled trace runs.
        self.refresh_from_frame()
    }

    fn sync_virtualizable_after_jit(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        info: &VirtualizableInfo,
    ) {
        let Some(frame_ptr) = self.frame_ptr() else {
            return;
        };
        // RPython writes detached virtualizable_boxes back here. Pyre keeps
        // locals/stack in the concrete PyFrame throughout execution, so after
        // restore_values() the heap frame is already authoritative. Replaying
        // exported resume-data back into the frame would reinterpret those
        // live slots as detached boxes and can corrupt normal Jump recovery.
        let _ = self.sync_scalar_fields_to_frame();
        unsafe {
            info.reset_vable_token(frame_ptr);
        }
    }

    fn sync_virtualizable_before_residual_call(&self, ctx: &mut TraceCtx) {
        let info = build_pyframe_virtualizable_info();
        let Some(vable_ref) = ctx.standard_virtualizable_box() else {
            return;
        };
        ctx.gen_store_back_in_vable(vable_ref);
        let Some(obj_ptr) = self.frame_ptr() else {
            return;
        };
        unsafe {
            info.tracing_before_residual_call(obj_ptr);
        }
        let force_token = ctx.force_token();
        ctx.vable_setfield_descr(vable_ref, force_token, info.token_field_descr());
    }

    fn sync_virtualizable_after_residual_call(
        &self,
        _ctx: &mut TraceCtx,
    ) -> ResidualVirtualizableSync {
        let info = build_pyframe_virtualizable_info();
        let Some(obj_ptr) = self.frame_ptr() else {
            return ResidualVirtualizableSync::default();
        };
        let forced = unsafe { info.tracing_after_residual_call(obj_ptr) };
        ResidualVirtualizableSync {
            updated_fields: Vec::new(),
            forced,
        }
    }

    fn import_virtualizable_boxes(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
        static_boxes: &[i64],
        array_boxes: &[Vec<i64>],
    ) -> bool {
        self.import_virtualizable_state(static_boxes, array_boxes)
    }

    fn export_virtualizable_boxes(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<(Vec<i64>, Vec<Vec<i64>>)> {
        Some(self.export_virtualizable_state())
    }

    fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
        unreachable!("pyre closes loops with explicit frame-backed jump args")
    }

    fn collect_typed_jump_args(_sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        unreachable!("pyre closes loops with explicit frame-backed jump args")
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        let _ = (sym, meta);
        true
    }

    fn validate_close_with_jump_args(
        _sym: &Self::Sym,
        meta: &Self::Meta,
        jump_args: &[OpRef],
    ) -> bool {
        let _ = meta;
        // pyre does not close loops by re-reading symbolic stack state from
        // the trace header.  Instead it materializes explicit jump args from
        // the concrete frame-backed virtualizable state at the merge point:
        //   [frame, next_instr, valuestackdepth, locals..., stack...]
        //
        // Retraces/bridges may start from a transient stackful state
        // (e.g. direct-call trace-through in progress) and still legally
        // jump back to a target loop whose merge-point stack is smaller.
        // RPython closes those traces against the target token contract, not
        // the retrace entry state's stack depth.  So for pyre's explicit
        // jump-arg model, the trace-start `meta.valuestackdepth` is not a
        // sound validator here.
        jump_args.len() >= 3
    }

    /// RPython resume.py: materialize a virtual object from resume data.
    ///
    /// Called during guard failure recovery when the optimizer kept an
    /// object virtual (New + SetfieldGc eliminated). The resume mechanism
    /// reconstructs the object so the interpreter can use it.
    fn materialize_virtual_ref(
        &mut self,
        _meta: &Self::Meta,
        _virtual_index: usize,
        materialized: &majit_meta::resume::MaterializedVirtual,
    ) -> Option<majit_ir::GcRef> {
        self.materialize_virtual_ref_from_layout(materialized, &[])
    }

    fn materialize_virtual_ref_with_refs(
        &mut self,
        _meta: &Self::Meta,
        _virtual_index: usize,
        materialized: &majit_meta::resume::MaterializedVirtual,
        materialized_refs: &[Option<majit_ir::GcRef>],
    ) -> Option<majit_ir::GcRef> {
        self.materialize_virtual_ref_from_layout(materialized, materialized_refs)
    }
}

impl PyreJitState {
    fn materialize_virtual_ref_from_layout(
        &mut self,
        materialized: &majit_meta::resume::MaterializedVirtual,
        materialized_refs: &[Option<majit_ir::GcRef>],
    ) -> Option<majit_ir::GcRef> {
        use majit_meta::resume::MaterializedVirtual;

        match materialized {
            MaterializedVirtual::Struct { fields, .. }
            | MaterializedVirtual::Obj { fields, .. } => {
                self.materialize_virtual_object(fields, materialized_refs)
            }
            MaterializedVirtual::RawBuffer { size, entries } => {
                materialize_virtual_raw_buffer(*size, entries, materialized_refs)
            }
            _ => None,
        }
    }

    fn materialize_virtual_object(
        &mut self,
        fields: &[(u32, majit_meta::resume::MaterializedValue)],
        materialized_refs: &[Option<majit_ir::GcRef>],
    ) -> Option<majit_ir::GcRef> {
        let mut ob_type: usize = 0;
        let mut int_payload: i64 = 0;
        let mut list_items_ptr: usize = 0;
        let mut list_items_len: usize = 0;

        for (field_idx, value) in fields {
            let offset = extract_pyre_field_offset(*field_idx);
            let concrete = value.resolve_with_refs(materialized_refs)?;
            match offset {
                Some(0) => ob_type = concrete as usize,
                Some(8) => {
                    if ob_type == &LIST_TYPE as *const _ as usize {
                        list_items_ptr = concrete as usize;
                    } else {
                        int_payload = concrete;
                    }
                }
                Some(16) if ob_type == &LIST_TYPE as *const _ as usize => {
                    list_items_len = concrete as usize;
                }
                _ => {}
            }
        }

        if ob_type == &INT_TYPE as *const _ as usize {
            let ptr = pyre_object::intobject::w_int_new(int_payload);
            Some(majit_ir::GcRef(ptr as usize))
        } else if ob_type == &FLOAT_TYPE as *const _ as usize {
            let ptr = pyre_object::floatobject::w_float_new(f64::from_bits(int_payload as u64));
            Some(majit_ir::GcRef(ptr as usize))
        } else if ob_type == &LIST_TYPE as *const _ as usize {
            let items = if list_items_len == 0 {
                Vec::new()
            } else {
                if list_items_ptr == 0 {
                    return None;
                }
                unsafe {
                    std::slice::from_raw_parts(list_items_ptr as *const PyObjectRef, list_items_len)
                        .to_vec()
                }
            };
            let ptr = w_list_new(items);
            Some(majit_ir::GcRef(ptr as usize))
        } else {
            None
        }
    }
}

fn materialize_virtual_raw_buffer(
    size: usize,
    entries: &[(usize, usize, majit_meta::resume::MaterializedValue)],
    materialized_refs: &[Option<majit_ir::GcRef>],
) -> Option<majit_ir::GcRef> {
    use std::alloc::{Layout, alloc_zeroed};

    let layout = Layout::from_size_align(size.max(1), 8).ok()?;
    let ptr = unsafe { alloc_zeroed(layout) };
    if ptr.is_null() {
        return None;
    }

    for (offset, length, value) in entries {
        let concrete = value.resolve_with_refs(materialized_refs)?;
        unsafe {
            let slot = ptr.add(*offset);
            match *length {
                1 => slot.write(concrete as u8),
                2 => (slot as *mut u16).write_unaligned(concrete as u16),
                4 => (slot as *mut u32).write_unaligned(concrete as u32),
                8 => (slot as *mut u64).write_unaligned(concrete as u64),
                _ => return None,
            }
        }
    }

    Some(majit_ir::GcRef(ptr as usize))
}

fn value_to_ptr(value: &Value) -> PyObjectRef {
    match value {
        Value::Ref(gc_ref) => gc_ref.0 as PyObjectRef,
        Value::Int(n) => *n as PyObjectRef,
        _ => std::ptr::null_mut(),
    }
}

fn value_to_raw(value: &Value) -> usize {
    match value {
        Value::Ref(gc_ref) => gc_ref.0,
        Value::Int(n) => *n as usize,
        Value::Float(f) => f.to_bits() as usize,
        Value::Void => 0,
    }
}

fn value_to_usize(value: &Value) -> usize {
    match value {
        Value::Ref(gc_ref) => gc_ref.0,
        Value::Int(n) => *n as usize,
        _ => 0,
    }
}

/// Extract byte offset from a pyre field descriptor index.
/// Mirrors `extract_field_offset` in majit-opt/virtualize.rs.
fn extract_pyre_field_offset(descr_idx: u32) -> Option<usize> {
    const FIELD_DESCR_TAG: u32 = 0x1000_0000;
    if descr_idx & 0xF000_0000 != FIELD_DESCR_TAG {
        return None;
    }
    Some(((descr_idx >> 4) & 0x000f_ffff) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_meta::JitState;
    use majit_meta::resume::{MaterializedValue, MaterializedVirtual};
    use pyre_object::OB_TYPE_OFFSET;
    use pyre_object::floatobject::w_float_get_value;
    use pyre_object::listobject::w_list_getitem;

    fn empty_meta() -> PyreMeta {
        PyreMeta {
            merge_pc: 0,
            num_locals: 0,
            ns_keys: Vec::new(),
            valuestackdepth: 0,
            slot_types: Vec::new(),
            has_virtualizable: false,
        }
    }

    fn empty_state() -> PyreJitState {
        PyreJitState {
            frame: 0,
            next_instr: 0,
            valuestackdepth: 0,
        }
    }

    #[test]
    fn test_is_boxed_int_value_ignores_py_null_sentinel() {
        assert!(!is_boxed_int_value(PY_NULL));
        let boxed = w_int_new(7);
        assert!(is_boxed_int_value(boxed));
    }

    #[test]
    fn test_trace_ob_type_descr_uses_immutable_header_field_descr() {
        let descr = trace_ob_type_descr();
        let field = descr
            .as_field_descr()
            .expect("ob_type descr must be a field descr");
        assert_eq!(field.offset(), OB_TYPE_OFFSET);
        assert_eq!(field.field_type(), Type::Int);
        assert!(descr.is_always_pure());
        assert!(field.is_immutable());
    }

    #[test]
    fn test_trace_gc_object_type_field_uses_pure_getfield_for_immutable_header() {
        let mut ctx = TraceCtx::for_test(1);
        let obj = OpRef(0);
        let _ = trace_gc_object_type_field(&mut ctx, obj, trace_ob_type_descr());

        let recorder = ctx.into_recorder();
        let op = recorder.last_op().expect("getfield op should be present");
        assert_eq!(op.opcode, OpCode::GetfieldGcPureI);
        assert_eq!(op.args.as_slice(), &[obj]);
    }

    #[test]
    fn test_guard_object_class_uses_guard_nonnull_class() {
        let mut ctx = TraceCtx::for_test(1);
        let obj = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![obj];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;
        sym.symbolic_initialized = true;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: 1,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        state.with_ctx(|this, ctx| {
            this.guard_object_class(ctx, obj, &INT_TYPE as *const PyType);
        });

        let recorder = ctx.into_recorder();
        let op = recorder.last_op().expect("guard op should be present");
        assert_eq!(op.opcode, OpCode::GuardNonnullClass);
        assert_eq!(op.args[0], obj);
    }

    #[test]
    fn test_trace_guarded_int_payload_uses_guard_nonnull_class_and_pure_payload() {
        let mut ctx = TraceCtx::for_test(1);
        let int_obj = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![int_obj];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;
        sym.symbolic_initialized = true;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: 1,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let _ = state.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, int_obj));

        let recorder = ctx.into_recorder();
        let mut saw_guard_nonnull_class = false;
        let mut saw_pure_payload = false;
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if op.opcode == OpCode::GuardNonnullClass {
                saw_guard_nonnull_class = true;
            }
            if op.opcode == OpCode::GetfieldGcPureI && op.args.as_slice() == &[int_obj] {
                saw_pure_payload = true;
            }
        }
        assert!(
            saw_guard_nonnull_class,
            "int payload fast path should guard object class via GuardNonnullClass"
        );
        assert!(
            saw_pure_payload,
            "int payload fast path should read the immutable payload with GetfieldGcPureI"
        );
    }

    #[test]
    fn test_init_symbolic_skips_heap_array_read_for_standard_virtualizable() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(1);
        let mut sym = PyreSym::new_uninit(OpRef(0));
        sym.vable_array_base = Some(3);
        sym.vable_next_instr = OpRef(1);
        sym.vable_valuestackdepth = OpRef(2);

        sym.init_symbolic(&mut ctx, frame_ptr);

        assert_eq!(sym.locals_cells_stack_array_ref, OpRef::NONE);
        let recorder = ctx.into_recorder();
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            assert_ne!(
                op.opcode,
                OpCode::GetfieldRawI,
                "standard virtualizable init should not read locals array from heap"
            );
        }
    }

    #[test]
    fn test_materialize_virtual_ref_reconstructs_float_object() {
        let mut state = empty_state();
        let meta = empty_meta();
        let value = 3.25f64;
        let materialized = MaterializedVirtual::Obj {
            type_id: 0,
            descr_index: crate::jit::descr::w_float_size_descr().index(),
            fields: vec![
                (
                    crate::jit::descr::ob_type_descr().index(),
                    MaterializedValue::Value(&FLOAT_TYPE as *const PyType as usize as i64),
                ),
                (
                    crate::jit::descr::float_floatval_descr().index(),
                    MaterializedValue::Value(value.to_bits() as i64),
                ),
            ],
        };

        let ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &materialized,
            &[],
        )
        .expect("float virtual should materialize");

        unsafe {
            assert!(is_float(ptr.0 as PyObjectRef));
            assert_eq!(w_float_get_value(ptr.0 as PyObjectRef), value);
        }
    }

    #[test]
    fn test_materialize_virtual_ref_reconstructs_list_from_raw_buffer_ref() {
        let mut state = empty_state();
        let meta = empty_meta();
        let first = w_int_new(2);
        let second = w_int_new(4);

        let raw_items = MaterializedVirtual::RawBuffer {
            size: 16,
            entries: vec![
                (0, 8, MaterializedValue::Value(first as i64)),
                (8, 8, MaterializedValue::Value(second as i64)),
            ],
        };
        let raw_ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &raw_items,
            &[],
        )
        .expect("raw items buffer should materialize");

        let list_virtual = MaterializedVirtual::Obj {
            type_id: 0,
            descr_index: 0,
            fields: vec![
                (
                    crate::jit::descr::ob_type_descr().index(),
                    MaterializedValue::Value(&LIST_TYPE as *const PyType as usize as i64),
                ),
                (
                    crate::jit::descr::list_items_ptr_descr().index(),
                    MaterializedValue::VirtualRef(0),
                ),
                (
                    crate::jit::descr::list_items_len_descr().index(),
                    MaterializedValue::Value(2),
                ),
                (
                    crate::jit::descr::list_items_heap_cap_descr().index(),
                    MaterializedValue::Value(2),
                ),
            ],
        };

        let list_ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            1,
            &list_virtual,
            &[Some(raw_ptr)],
        )
        .expect("list virtual should materialize");

        unsafe {
            assert!(is_list(list_ptr.0 as PyObjectRef));
            assert_eq!(
                w_int_get_value(w_list_getitem(list_ptr.0 as PyObjectRef, 0).unwrap()),
                2
            );
            assert_eq!(
                w_int_get_value(w_list_getitem(list_ptr.0 as PyObjectRef, 1).unwrap()),
                4
            );
        }
    }

    #[test]
    fn test_virtualizable_array_lengths_use_live_prefix_depth() {
        let mut state = empty_state();
        state.valuestackdepth = 7;
        let info = build_pyframe_virtualizable_info();

        assert_eq!(
            <PyreJitState as JitState>::virtualizable_array_lengths(
                &state,
                &empty_meta(),
                "frame",
                &info,
            ),
            Some(vec![7])
        );
    }

    #[test]
    fn test_restore_guard_failure_uses_runtime_value_kinds_for_virtualizable_locals() {
        use majit_ir::GcRef;
        use pyre_bytecode::{ConstantData, compile_exec};
        use pyre_interp::frame::PyFrame;

        let module = compile_exec("def f(a, b, c):\n    i = 0\n    return i\nf(1, 2, 3)\n")
            .expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut state = PyreJitState {
            frame: frame_ptr,
            next_instr: 0,
            valuestackdepth: 4,
        };
        let meta = PyreMeta {
            merge_pc: 0,
            num_locals: 4,
            ns_keys: Vec::new(),
            valuestackdepth: 4,
            has_virtualizable: true,
            // Stale trace-entry types: all locals looked boxed refs when the
            // trace started, but guard failure can still carry a raw int in
            // slot `i`.
            slot_types: vec![Type::Ref, Type::Ref, Type::Ref, Type::Ref],
        };
        let values = vec![
            Value::Ref(GcRef(frame_ptr)),
            Value::Int(9),
            Value::Int(4),
            Value::Ref(GcRef(w_int_new(1) as usize)),
            Value::Ref(GcRef(w_int_new(2) as usize)),
            Value::Ref(GcRef(w_int_new(3) as usize)),
            Value::Int(7),
        ];

        assert!(<PyreJitState as JitState>::restore_guard_failure_values(
            &mut state,
            &meta,
            &values,
            &majit_meta::blackhole::ExceptionState::default(),
        ));

        assert_eq!(state.next_instr, 9);
        assert_eq!(state.valuestackdepth, 4);
        let restored_i = state.local_at(3).expect("local i should be restored");
        assert!(unsafe { is_int(restored_i) });
        assert_eq!(unsafe { w_int_get_value(restored_i) }, 7);
    }

    #[test]
    fn test_load_local_checked_value_skips_nonnull_guard_for_typed_int_local() {
        let mut ctx = TraceCtx::for_test(1);
        let local = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![local];
        sym.symbolic_local_types = vec![Type::Int];
        sym.nlocals = 1;
        sym.symbolic_initialized = true;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: 1,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let loaded =
            <TraceFrameState as LocalOpcodeHandler>::load_local_checked_value(&mut state, 0, "j")
                .expect("typed int local should load without guard");
        assert_eq!(loaded, local);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 0);
    }

    #[test]
    fn test_load_local_checked_value_keeps_nonnull_guard_for_ref_local() {
        let mut ctx = TraceCtx::for_test(1);
        let local = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![local];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;
        sym.symbolic_initialized = true;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: 1,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let loaded =
            <TraceFrameState as LocalOpcodeHandler>::load_local_checked_value(&mut state, 0, "b")
                .expect("ref local should load with guard");
        assert_eq!(loaded, local);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 1);
        assert_eq!(
            recorder.last_op().map(|op| op.opcode),
            Some(OpCode::GuardNonnull)
        );
    }

    #[test]
    fn test_store_local_value_preserves_traced_raw_int_type_for_ref_slot() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.locals_cells_stack_w[0] = w_int_new(41);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(1);
        let raw = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![OpRef::NONE];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;
        sym.symbolic_initialized = true;
        sym.transient_value_types.insert(raw, Type::Int);

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        state
            .with_ctx(|this, ctx| this.store_local_value(ctx, 0, raw))
            .expect("store of raw traced int should succeed");
        assert_eq!(state.sym().symbolic_locals[0], raw);
        assert_eq!(state.sym().symbolic_local_types[0], Type::Int);

        let loaded =
            <TraceFrameState as LocalOpcodeHandler>::load_local_checked_value(&mut state, 0, "x")
                .expect("typed raw int local should reload without object guards");
        assert_eq!(loaded, raw);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 0);
    }

    #[test]
    fn test_trace_dynamic_list_index_typed_int_skips_object_unbox() {
        let mut ctx = TraceCtx::for_test(2);
        let key = OpRef(0);
        let len = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![key];
        sym.symbolic_local_types = vec![Type::Int];
        sym.nlocals = 1;
        sym.symbolic_initialized = true;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: 1,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let raw_index = state.with_ctx(|this, ctx| this.trace_dynamic_list_index(ctx, key, len, 2));
        assert_eq!(raw_index, key);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 4);
        assert_eq!(recorder.num_guards(), 2);
    }

    #[test]
    fn test_trace_binary_value_boxes_typed_raw_operands_for_python_helper() {
        let mut ctx = TraceCtx::for_test(2);
        let lhs = OpRef(0);
        let rhs = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![lhs, rhs];
        sym.symbolic_local_types = vec![Type::Float, Type::Int];
        sym.nlocals = 2;
        sym.symbolic_initialized = true;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: 1,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let _ = <TraceFrameState as TraceHelperAccess>::trace_binary_value(
            &mut state,
            lhs,
            rhs,
            BinaryOperator::Power,
        )
        .expect("generic helper call should box raw operands first");

        let recorder = ctx.into_recorder();
        let call = recorder.last_op().expect("call op should be present");
        assert_eq!(call.opcode, OpCode::CallI);
        assert_ne!(call.args[0], lhs);
        assert_ne!(call.args[1], rhs);
    }

    #[test]
    fn test_trace_known_builtin_call_boxes_typed_raw_args_for_python_helper_boundary() {
        let mut ctx = TraceCtx::for_test(2);
        let callable = OpRef(0);
        let arg = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![callable, arg];
        sym.symbolic_local_types = vec![Type::Ref, Type::Int];
        sym.nlocals = 2;
        sym.symbolic_initialized = true;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: 1,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let _ = state
            .trace_known_builtin_call(callable, &[arg])
            .expect("known builtin helper boundary should box raw int args");

        let recorder = ctx.into_recorder();
        let call = recorder.last_op().expect("call op should be present");
        assert!(matches!(
            call.opcode,
            OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN
        ));
        assert_ne!(call.args.last().copied(), Some(arg));
    }

    #[test]
    fn test_compare_value_direct_keeps_raw_truth_for_immediate_branch_consumer() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("if 1 < 2:\n    x = 3\n").expect("test code should compile");
        let compare_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::CompareOp { .. }, _))
                )
            })
            .expect("test bytecode should contain COMPARE_OP");
        let branch_pc = ((compare_pc + 1)..code.instructions.len())
            .find(|&pc| {
                decode_instruction_at(&code, pc)
                    .map(|(instruction, _)| instruction_consumes_comparison_truth(instruction))
                    .unwrap_or(false)
            })
            .expect("test bytecode should contain POP_JUMP_IF after COMPARE_OP");

        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        frame.push(w_int_new(1));
        frame.push(w_int_new(2));
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        let lhs = OpRef(0);
        let rhs = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.valuestackdepth = 0;
        sym.transient_value_types.insert(lhs, Type::Int);
        sym.transient_value_types.insert(rhs, Type::Int);

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: branch_pc,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let result = state
            .compare_value_direct(lhs, rhs, ComparisonOperator::Less)
            .expect("int comparison should trace");
        let truth = state
            .truth_value_direct(result)
            .expect("immediate POP_JUMP consumer should reuse raw truth");

        assert_eq!(result, truth);

        let recorder = ctx.into_recorder();
        let mut saw_cmp = false;
        let mut saw_bool_call = false;
        let mut saw_bool_unbox = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if op.opcode == OpCode::IntLt {
                saw_cmp = true;
            }
            if op.opcode == OpCode::CallR {
                saw_bool_call = true;
            }
            if op.opcode == OpCode::GetfieldGcI {
                saw_bool_unbox = true;
            }
        }
        assert!(
            saw_cmp,
            "branch compare should still emit raw int comparison"
        );
        assert!(
            !saw_bool_call,
            "immediate branch consumer should not allocate a bool object"
        );
        assert!(
            !saw_bool_unbox,
            "immediate branch consumer should not unbox a transient bool object"
        );
    }

    #[test]
    fn test_compare_value_direct_boxes_bool_when_not_immediately_consumed_by_branch() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        frame.push(w_int_new(1));
        frame.push(w_int_new(2));
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        let lhs = OpRef(0);
        let rhs = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.valuestackdepth = 0;
        sym.transient_value_types.insert(lhs, Type::Int);
        sym.transient_value_types.insert(rhs, Type::Int);

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let _ = state
            .compare_value_direct(lhs, rhs, ComparisonOperator::Less)
            .expect("non-branch compare should trace");

        let recorder = ctx.into_recorder();
        let mut saw_bool_call = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if op.opcode == OpCode::CallR {
                saw_bool_call = true;
            }
        }
        assert!(
            saw_bool_call,
            "non-branch compare should continue to materialize a Python bool result"
        );
    }

    #[test]
    fn test_next_instruction_consumes_comparison_truth_skips_extended_arg_trivia() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let mut source = String::from("def f(x, y):\n    if x < y:\n");
        for i in 0..400 {
            source.push_str(&format!("        z{i} = {i}\n"));
        }
        source.push_str("    return 0\n");
        source.push_str("f(1, 2)\n");

        let module = compile_exec(&source).expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                pyre_bytecode::ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let compare_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::CompareOp { .. }, _))
                )
            })
            .expect("test bytecode should contain COMPARE_OP");

        let first_after_compare = decode_instruction_at(&code, compare_pc + 1)
            .map(|(instruction, _)| instruction)
            .expect("bytecode should continue after COMPARE_OP");
        assert!(
            instruction_is_trivia_between_compare_and_branch(first_after_compare),
            "test source should force trivia between COMPARE_OP and POP_JUMP_IF"
        );

        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.valuestackdepth = 0;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: compare_pc + 1,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        assert!(
            state.next_instruction_consumes_comparison_truth(),
            "branch fusion should survive EXTENDED_ARG/other trivia before the branch"
        );
    }

    #[test]
    fn test_trace_code_step_preserves_comparison_truth_across_extended_arg_trivia() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let mut source = String::from("def f(x, y):\n    if x < y:\n");
        for i in 0..400 {
            source.push_str(&format!("        z{i} = {i}\n"));
        }
        source.push_str("    return 0\n");
        source.push_str("f(1, 2)\n");

        let module = compile_exec(&source).expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                pyre_bytecode::ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let compare_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::CompareOp { .. }, _))
                )
            })
            .expect("test bytecode should contain COMPARE_OP");

        let first_after_compare = decode_instruction_at(&code, compare_pc + 1)
            .map(|(instruction, _)| instruction)
            .expect("bytecode should continue after COMPARE_OP");
        assert!(
            instruction_is_trivia_between_compare_and_branch(first_after_compare),
            "test source should force trivia between COMPARE_OP and POP_JUMP_IF"
        );

        let mut frame = Box::new(PyFrame::new(code.clone()));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.last_comparison_truth = Some(OpRef(123));

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: compare_pc + 2,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let action = state.trace_code_step(&code, compare_pc + 1);
        assert!(matches!(action, TraceAction::Continue));
        assert_eq!(
            state.sym().last_comparison_truth,
            Some(OpRef(123)),
            "comparison truth cache should survive trivia until the real branch consumer"
        );
    }

    #[test]
    fn test_concrete_popped_value_reads_last_popped_slot() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.push(w_int_new(11));
        frame.push(w_int_new(22));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth - 1;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let top = state
            .concrete_popped_value()
            .expect("top-of-stack value should be readable");
        assert!(unsafe { is_int(top) });
        assert_eq!(unsafe { w_int_get_value(top) }, 22);
    }

    #[test]
    fn test_pop_value_caches_concrete_popped_box_for_truth_consumers() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;
        use pyre_runtime::SharedOpcodeHandler;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.push(w_int_new(11));
        frame.push(w_int_new(22));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth;
        sym.symbolic_stack = vec![OpRef(10), OpRef(11)];
        sym.symbolic_stack_types = vec![Type::Ref, Type::Ref];

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let _ = SharedOpcodeHandler::pop_value(&mut state).expect("pop should succeed");
        let top = state
            .concrete_popped_value()
            .expect("popped concrete box should stay available after depth change");
        assert_eq!(unsafe { w_int_get_value(top) }, 22);
    }

    #[test]
    fn test_concrete_binary_operands_read_last_two_popped_slots() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("a = [0]\na[0] = 1\n").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.push(w_int_new(11));
        frame.push(w_int_new(22));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth - 2;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let (lhs, rhs) = state
            .concrete_binary_operands()
            .expect("binary operands should be readable");
        assert_eq!(unsafe { w_int_get_value(lhs) }, 11);
        assert_eq!(unsafe { w_int_get_value(rhs) }, 22);
    }

    #[test]
    fn test_concrete_store_subscr_operands_read_last_three_popped_slots() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let module = compile_exec("def f(a, b, c):\n    a[b] = c\nf([], 0, 0)\n")
            .expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                pyre_bytecode::ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");
        let mut frame = Box::new(PyFrame::new(code));
        let value = w_int_new(7);
        let obj = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        let key = w_int_new(1);
        frame.push(value);
        frame.push(obj);
        frame.push(key);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth - 3;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let (concrete_value, concrete_obj, concrete_key) = state
            .concrete_store_subscr_operands()
            .expect("store_subscr operands should be readable");
        assert_eq!(concrete_value, value);
        assert_eq!(concrete_obj, obj);
        assert_eq!(unsafe { w_int_get_value(concrete_key) }, 1);
    }

    #[test]
    fn test_concrete_branch_truth_reads_last_popped_slot() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.push(w_int_new(0));
        frame.push(w_int_new(1));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth - 1;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        assert!(
            state
                .concrete_branch_truth()
                .expect("branch truth should read the popped branch operand")
        );
    }

    #[test]
    fn test_concrete_branch_truth_uses_cached_comparison_truth_without_stack_value() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth;
        sym.last_comparison_truth = Some(OpRef(321));
        sym.last_comparison_concrete_truth = Some(false);

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        assert!(
            !state
                .concrete_branch_truth()
                .expect("cached comparison truth should avoid concrete stack lookup")
        );
        assert_eq!(state.sym().last_comparison_concrete_truth, None);
    }

    #[test]
    fn test_truth_value_direct_caches_concrete_truth_for_raw_int_branch_consumer() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.push(w_int_new(7));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth - 1;
        sym.transient_value_types.insert(OpRef(77), Type::Int);

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let truth = state
            .truth_value_direct(OpRef(77))
            .expect("raw int truth should trace");
        assert!(!truth.is_none());

        state.sym_mut().valuestackdepth = frame.valuestackdepth;
        assert!(
            state
                .concrete_branch_truth()
                .expect("cached raw-int truth should avoid concrete stack lookup")
        );
        assert_eq!(state.sym().last_comparison_concrete_truth, None);
    }

    #[test]
    fn test_record_branch_guard_uses_branch_pc_and_pre_pop_stack_shape() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.next_instr = 456;
        let lower_value = w_int_new(0);
        let branch_value = w_int_new(1);
        frame.push(lower_value);
        frame.push(branch_value);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(3);
        let expected_pc = ctx.const_int(456);
        let expected_vsd = ctx.const_int(2);
        let expected_branch_value = ctx.const_int(branch_value as i64);
        let frame_ref = OpRef(0);
        let lower_stack = OpRef(1);
        let truth = OpRef(2);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth - 1;
        sym.symbolic_stack = vec![lower_stack];
        sym.symbolic_stack_types = vec![Type::Ref];

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 459,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        state.with_ctx(|this, ctx| {
            this.record_branch_guard(ctx, OpRef::NONE, truth, true);
        });

        let recorder = ctx.into_recorder();
        let guard = recorder.last_op().expect("branch guard should be recorded");
        let fail_args = guard
            .fail_args
            .as_ref()
            .expect("branch guard should carry explicit fail args");
        assert_eq!(guard.opcode, OpCode::GuardTrue);
        assert_eq!(
            fail_args.as_slice(),
            &[
                frame_ref,
                expected_pc,
                expected_vsd,
                lower_stack,
                expected_branch_value
            ]
        );
    }

    #[test]
    fn test_generic_guard_during_branch_truth_uses_pre_pop_stack_shape() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.next_instr = 456;
        let lower_value = w_int_new(0);
        let branch_value = w_int_new(1);
        frame.push(lower_value);
        frame.push(branch_value);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(3);
        let expected_pc = ctx.const_int(456);
        let expected_vsd = ctx.const_int(2);
        let expected_branch_value = ctx.const_int(branch_value as i64);
        let frame_ref = OpRef(0);
        let lower_stack = OpRef(1);
        let truth = OpRef(2);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth - 1;
        sym.symbolic_stack = vec![lower_stack];
        sym.symbolic_stack_types = vec![Type::Ref];
        sym.pending_branch_value = Some(OpRef::NONE);

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 459,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        state.sym_mut().pending_branch_value = Some(OpRef::NONE);
        state.with_ctx(|this, ctx| {
            this.sym_mut().pending_branch_value = Some(OpRef::NONE);
            this.record_guard(ctx, OpCode::GuardTrue, &[truth]);
        });

        let recorder = ctx.into_recorder();
        let guard = recorder.last_op().expect("branch guard should be recorded");
        let fail_args = guard
            .fail_args
            .as_ref()
            .expect("branch guard should carry explicit fail args");
        assert_eq!(guard.opcode, OpCode::GuardTrue);
        assert_eq!(
            fail_args.as_slice(),
            &[
                frame_ref,
                expected_pc,
                expected_vsd,
                lower_stack,
                expected_branch_value
            ]
        );
    }

    #[test]
    fn test_branch_truth_keeps_concrete_popped_value_until_leave() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.push(w_int_new(1));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        let frame_ref = OpRef(0);
        let truth = OpRef(1);
        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth;
        sym.symbolic_stack = vec![truth];
        sym.symbolic_stack_types = vec![Type::Int];
        sym.last_popped_concrete_value = Some(w_int_new(1));
        sym.pending_branch_value = Some(truth);

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        state.with_ctx(|this, ctx| {
            this.record_guard(ctx, OpCode::GuardTrue, &[truth]);
        });
        assert_eq!(state.concrete_branch_truth_for_value(truth).unwrap(), true);
        <TraceFrameState as BranchOpcodeHandler>::leave_branch_truth(&mut state).unwrap();
        assert!(state.sym().last_popped_concrete_value.is_none());
    }

    #[test]
    fn test_current_fail_args_flushes_virtualizable_header_before_capture() {
        let mut ctx = TraceCtx::for_test(2);
        let frame_ref = OpRef(0);
        let local0 = OpRef(1);
        let stack0 = OpRef(2);
        let stack1 = OpRef(3);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.symbolic_initialized = true;
        sym.nlocals = 1;
        sym.valuestackdepth = 3;
        sym.vable_next_instr = ctx.const_int(12);
        sym.vable_valuestackdepth = ctx.const_int(1);
        sym.symbolic_locals = vec![local0];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack = vec![stack0, stack1];
        sym.symbolic_stack_types = vec![Type::Ref, Type::Ref];

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: 1,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let fail_args = state.with_ctx(|this, ctx| this.current_fail_args(ctx));
        let expected_pc = state.with_ctx(|_, ctx| ctx.const_int(12));
        let expected_vsd = state.with_ctx(|_, ctx| ctx.const_int(3));

        assert_eq!(
            fail_args.as_slice(),
            &[frame_ref, expected_pc, expected_vsd, local0, stack0, stack1]
        );
    }

    #[test]
    fn test_current_fail_args_materializes_symbolic_holes_from_concrete_frame() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("len(x)").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        let local_ref = w_int_new(41);
        let stack_int = w_int_new(7);
        frame.locals_cells_stack_w[0] = local_ref;
        frame.locals_cells_stack_w[1] = stack_int;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(1);
        let frame_ref = OpRef(0);
        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.symbolic_initialized = true;
        sym.nlocals = 1;
        sym.valuestackdepth = 2;
        sym.vable_next_instr = ctx.const_int(33);
        sym.vable_valuestackdepth = ctx.const_int(0);
        sym.symbolic_locals = vec![OpRef::NONE];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack = vec![OpRef::NONE];
        sym.symbolic_stack_types = vec![Type::Int];

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let fail_args = state.with_ctx(|this, ctx| this.current_fail_args(ctx));
        let expected_pc = state.with_ctx(|_, ctx| ctx.const_int(33));
        let expected_vsd = state.with_ctx(|_, ctx| ctx.const_int(2));
        let expected_local = state.with_ctx(|_, ctx| ctx.const_int(local_ref as i64));
        let expected_stack = state.with_ctx(|_, ctx| ctx.const_int(7));

        assert_eq!(
            fail_args.as_slice(),
            &[
                frame_ref,
                expected_pc,
                expected_vsd,
                expected_local,
                expected_stack
            ]
        );
        assert!(
            fail_args.iter().all(|arg| !arg.is_none()),
            "materialized fail args should not contain OpRef::NONE holes"
        );
    }

    #[test]
    fn test_direct_len_value_returns_typed_raw_len_for_integer_list() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let code = compile_exec("len(x)").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
        }
        let arg_idx = frame.stack_base() + 2;
        frame.locals_cells_stack_w[arg_idx] = list;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        let value = OpRef(0);
        let callable = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![value, callable];
        sym.symbolic_local_types = vec![Type::Ref, Type::Ref];
        sym.nlocals = 2;
        sym.valuestackdepth = frame.stack_base();
        sym.symbolic_initialized = true;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let len = state
            .direct_len_value(callable, value)
            .expect("integer-list len fast path should trace");
        assert!(
            state.sym().transient_value_types.contains_key(&len),
            "len fast path should remember the typed raw result"
        );
        let len_type = state.value_type(len);

        let recorder = ctx.into_recorder();
        assert_ne!(
            recorder.last_op().map(|op| op.opcode),
            Some(OpCode::CallI),
            "len(list) should not fall back to the builtin helper call path"
        );
        let mut saw_len_field = false;
        let mut saw_new = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if op.opcode == OpCode::New {
                saw_new = true;
            }
            if op.opcode == OpCode::GetfieldGcI
                && op.descr.as_ref().map(|d| d.index()) == Some(list_int_items_len_descr().index())
            {
                saw_len_field = true;
            }
        }
        assert_eq!(len_type, Type::Int);
        assert!(
            saw_len_field,
            "list len fast path should read the integer-list length field"
        );
        assert!(
            !saw_new,
            "len(list) should not allocate a W_Int box in the trace"
        );
    }

    #[test]
    fn test_trace_direct_float_list_getitem_uses_gc_field_loads_for_list_object() {
        let mut ctx = TraceCtx::for_test(2);
        let list = OpRef(0);
        let key = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![list, key];
        sym.symbolic_local_types = vec![Type::Ref, Type::Int];
        sym.nlocals = 2;
        sym.symbolic_initialized = true;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: 1,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let result =
            state.with_ctx(|this, ctx| this.trace_direct_float_list_getitem(ctx, list, key, 2));
        assert_eq!(state.value_type(result), Type::Float);

        let recorder = ctx.into_recorder();
        let mut saw_gc_field = false;
        let mut saw_raw_field = false;
        let mut saw_raw_array = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            match op.opcode {
                OpCode::GetfieldGcI => saw_gc_field = true,
                OpCode::GetfieldRawI => saw_raw_field = true,
                OpCode::GetarrayitemRawF => saw_raw_array = true,
                _ => {}
            }
        }
        assert!(saw_gc_field, "list object fields should use GetfieldGcI");
        assert!(
            !saw_raw_field,
            "float-list fast path should not use raw field loads on GC list objects"
        );
        assert!(saw_raw_array, "payload array access stays raw");
    }

    #[test]
    fn test_list_append_value_uses_raw_int_storage_fast_path() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let mut ctx = TraceCtx::for_test(2);
        let list = OpRef(0);
        let value = OpRef(1);
        let concrete_list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        unsafe {
            assert!(w_list_uses_int_storage(concrete_list));
            assert!(w_list_can_append_without_realloc(concrete_list));
        }

        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![list, value];
        sym.symbolic_local_types = vec![Type::Ref, Type::Int];
        sym.nlocals = 2;
        sym.symbolic_initialized = true;
        sym.last_popped_concrete_value = Some(concrete_list);
        sym.transient_value_types.insert(value, Type::Int);

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        state
            .list_append_value(list, value)
            .expect("integer-list append fast path should trace");

        let recorder = ctx.into_recorder();
        let mut saw_raw_setitem = false;
        let mut saw_len_update = false;
        let mut saw_call = false;
        let mut saw_new = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if matches!(
                op.opcode,
                OpCode::CallI | OpCode::CallN | OpCode::CallR | OpCode::CallF
            ) {
                saw_call = true;
            }
            if op.opcode == OpCode::New {
                saw_new = true;
            }
            if op.opcode == OpCode::SetarrayitemRaw
                && op.descr.as_ref().map(|d| d.index()) == Some(int_array_descr().index())
            {
                saw_raw_setitem = true;
            }
            if op.opcode == OpCode::SetfieldGc
                && op.descr.as_ref().map(|d| d.index()) == Some(list_int_items_len_descr().index())
            {
                saw_len_update = true;
            }
        }

        assert!(
            saw_raw_setitem,
            "integer-list append should write through the raw int array"
        );
        assert!(
            saw_len_update,
            "integer-list append should update the integer-list length field"
        );
        assert!(
            !saw_call,
            "integer-list append should not fall back to jit_list_append"
        );
        assert!(
            !saw_new,
            "integer-list append should not allocate a W_Int box in the trace"
        );
    }

    #[test]
    fn test_list_append_value_uses_raw_float_storage_fast_path() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;

        let mut ctx = TraceCtx::for_test(2);
        let list = OpRef(0);
        let value = OpRef(1);
        let concrete_list = w_list_new(vec![
            pyre_object::floatobject::w_float_new(1.5),
            pyre_object::floatobject::w_float_new(2.5),
        ]);
        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        unsafe {
            assert!(w_list_uses_float_storage(concrete_list));
            assert!(w_list_can_append_without_realloc(concrete_list));
        }

        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![list, value];
        sym.symbolic_local_types = vec![Type::Ref, Type::Float];
        sym.nlocals = 2;
        sym.symbolic_initialized = true;
        sym.last_popped_concrete_value = Some(concrete_list);
        sym.transient_value_types.insert(value, Type::Float);

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        state
            .list_append_value(list, value)
            .expect("float-list append fast path should trace");

        let recorder = ctx.into_recorder();
        let mut saw_raw_setitem = false;
        let mut saw_len_update = false;
        let mut saw_call = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if matches!(
                op.opcode,
                OpCode::CallI | OpCode::CallN | OpCode::CallR | OpCode::CallF
            ) {
                saw_call = true;
            }
            if op.opcode == OpCode::SetarrayitemRaw
                && op.descr.as_ref().map(|d| d.index()) == Some(float_array_descr().index())
            {
                saw_raw_setitem = true;
            }
            if op.opcode == OpCode::SetfieldGc
                && op.descr.as_ref().map(|d| d.index())
                    == Some(list_float_items_len_descr().index())
            {
                saw_len_update = true;
            }
        }

        assert!(
            saw_raw_setitem,
            "float-list append should write through the raw float array"
        );
        assert!(
            saw_len_update,
            "float-list append should update the float-list length field"
        );
        assert!(
            !saw_call,
            "float-list append should not fall back to jit_list_append"
        );
    }

    #[test]
    fn test_iter_next_value_for_range_iterator_uses_gc_fields_and_returns_raw_int() {
        use pyre_bytecode::compile_exec;
        use pyre_interp::frame::PyFrame;
        use pyre_object::w_range_iter_new;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        frame.push(w_range_iter_new(0, 2, 1));
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(1);
        let iter = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_stack = vec![iter];
        sym.symbolic_stack_types = vec![Type::Ref];
        sym.valuestackdepth = 1;
        sym.symbolic_initialized = true;

        let mut state = TraceFrameState {
            ctx: &mut ctx,
            sym: &mut sym,
            concrete_frame: frame_ptr,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_fail_args: None,
            parent_fail_arg_types: None,
            pending_inline_frame: None,
        };

        let next = TraceFrameState::iter_next_value(&mut state, iter)
            .expect("range iterator fast path should trace");
        assert_eq!(state.value_type(next), Type::Int);
        <TraceFrameState as IterOpcodeHandler>::guard_optional_value(&mut state, next, true)
            .expect("typed range next should not need optional guard");

        let recorder = ctx.into_recorder();
        let mut saw_getfield_gc = false;
        let mut saw_setfield_gc = false;
        let mut saw_setfield_raw = false;
        let mut saw_getfield_raw = false;
        let mut saw_new = false;
        let mut saw_optional_guard = false;
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            match op.opcode {
                OpCode::GetfieldGcI => saw_getfield_gc = true,
                OpCode::SetfieldGc => saw_setfield_gc = true,
                OpCode::SetfieldRaw => saw_setfield_raw = true,
                OpCode::GetfieldRawI => saw_getfield_raw = true,
                OpCode::New => saw_new = true,
                OpCode::GuardNonnull | OpCode::GuardIsnull => saw_optional_guard = true,
                _ => {}
            }
        }
        assert!(
            saw_getfield_gc,
            "range iterator fields should use GetfieldGcI"
        );
        assert!(
            saw_setfield_gc,
            "range iterator current update should use SetfieldGc"
        );
        assert!(
            !saw_setfield_raw,
            "range iterator fast path should not use raw stores on GC iterator objects"
        );
        assert!(
            !saw_getfield_raw,
            "range iterator fast path should not use raw field loads on GC iterator objects"
        );
        assert!(
            !saw_new,
            "range iterator fast path should not box the yielded int"
        );
        assert!(
            !saw_optional_guard,
            "raw-int range iteration should not emit optional-value guards"
        );
    }
}

// ── Virtualizable configuration ──────────────────────────────────────
//
// PyPy's `pypy/interpreter/pyframe.py` declares:
//
//     _virtualizable_ = ['locals_stack_w[*]', 'valuestackdepth',
//                         'last_instr', ...]
//
// Our Rust equivalent uses explicit byte offsets instead of name-based
// introspection. The JIT optimizer's Virtualize pass uses this info
// to keep frame fields in CPU registers, eliminating heap accesses
// for LoadFast/StoreFast and stack push/pop during compiled code.
//
// The shared frame layout contract now also lives in `pyre-jit/src/frame_layout.rs`
// so the tracer can compute the same offsets without depending on
// `pyre-interp`. Driver registration still happens in `pyre-interp/src/eval.rs`.

/// Mini eval loop that traces AND executes callee bytecodes on the
/// same TraceCtx as the caller, implementing trace-through inlining.
/// A frame on the inline tracing stack (PyPy's MIFrame equivalent).
///
/// PyPy pyjitpl.py: MIFrame holds jitcode + registers + pc + greenkey.
/// Ours holds PyreSym (symbolic state) + concrete PyFrame + OpArgState.
struct MIFrame {
    sym: PyreSym,
    concrete_frame: pyre_interp::frame::PyFrame,
    drop_frame_opref: Option<OpRef>,
    green_key: u64,
    arg_state: pyre_bytecode::bytecode::OpArgState,
    parent_fail_args: Vec<OpRef>,
    parent_fail_arg_types: Vec<Type>,
    caller_result_stack_idx: Option<usize>,
}

struct PendingInlineFrame {
    sym: PyreSym,
    concrete_frame: pyre_interp::frame::PyFrame,
    drop_frame_opref: Option<OpRef>,
    green_key: u64,
    parent_fail_args: Vec<OpRef>,
    parent_fail_arg_types: Vec<Type>,
    nargs: usize,
    caller_result_stack_idx: Option<usize>,
}

enum InlineTraceStepAction {
    Trace(TraceAction),
    PushFrame(PendingInlineFrame),
}

fn execute_inline_residual_call(
    frame: &mut pyre_interp::frame::PyFrame,
    nargs: usize,
) -> Result<(), pyre_runtime::PyError> {
    let required = nargs + 2; // callable + null/self + args
    if frame.valuestackdepth < frame.stack_base() + required {
        return Err(pyre_runtime::PyError::type_error(
            "inline residual call stack underflow",
        ));
    }

    let mut args = Vec::with_capacity(nargs);
    for _ in 0..nargs {
        args.push(frame.pop());
    }
    args.reverse();
    let _null_or_self = frame.pop();
    let callable = frame.pop();
    let result = pyre_interp::call::call_callable_inline_residual(frame, callable, &args)?;
    frame.push(result);
    Ok(())
}

/// PyPy pyjitpl.py `_interpret()` equivalent for inlined calls.
///
/// Uses a framestack (Vec<MIFrame>) instead of recursive Rust calls.
/// When trace_code_step encounters a call that should be inlined,
/// it pushes a new MIFrame. When RETURN_VALUE is traced, it pops
/// and stores the result in the parent via make_result_of_lastop.
///
/// This matches PyPy's ChangeFrame exception pattern:
/// - perform_call → newframe + push + ChangeFrame
/// - finishframe → popframe + make_result_of_lastop + ChangeFrame
fn inline_trace_and_execute(
    ctx: &mut majit_meta::TraceCtx,
    pending: PendingInlineFrame,
) -> Result<(OpRef, Type, PyObjectRef), pyre_runtime::PyError> {
    use pyre_runtime::StepResult;
    let _inline_call_override = pyre_interp::call::inline_call_override_guard();
    let inline_trace_base = ctx.inline_trace_depth();
    ctx.push_inline_trace_position(pending.green_key);
    let mut framestack: Vec<MIFrame> = vec![MIFrame {
        sym: pending.sym,
        concrete_frame: pending.concrete_frame,
        drop_frame_opref: pending.drop_frame_opref,
        green_key: pending.green_key,
        arg_state: pyre_bytecode::bytecode::OpArgState::default(),
        parent_fail_args: pending.parent_fail_args,
        parent_fail_arg_types: pending.parent_fail_arg_types,
        caller_result_stack_idx: pending.caller_result_stack_idx,
    }];

    // PyPy _interpret(): while True: framestack[-1].run_one_step()
    loop {
        let top = framestack.last_mut().unwrap();
        let code = unsafe { &*top.concrete_frame.code };
        let pc = top.concrete_frame.next_instr;
        if pc >= code.instructions.len() {
            return Err(pyre_runtime::PyError::type_error(
                "inline callee fell off end of bytecode",
            ));
        }

        // ── run_one_step: TRACE ──
        let pfa = top.parent_fail_args.clone();
        let pfa_types = top.parent_fail_arg_types.clone();
        let trace_action = {
            let mut fs = TraceFrameState::from_sym(
                ctx,
                &mut top.sym,
                &top.concrete_frame as *const _ as usize,
                pc + 1,
            );
            // PyPy capture_resumedata: callee guards use parent fail_args
            fs.parent_fail_args = Some(pfa);
            fs.parent_fail_arg_types = Some(pfa_types);
            fs.trace_code_step_inline(code, pc)
        };

        match trace_action {
            InlineTraceStepAction::PushFrame(pending) => {
                if majit_meta::majit_log_enabled() {
                    eprintln!(
                        "[jit][inline] push frame depth={} green_key={} nargs={}",
                        framestack.len(),
                        pending.green_key,
                        pending.nargs
                    );
                }
                {
                    let parent = framestack.last_mut().unwrap();
                    for _ in 0..pending.nargs {
                        let value = parent.concrete_frame.pop();
                        if value.is_null() {
                            panic!("inline framestack popped null concrete arg");
                        }
                    }
                    let _null_or_self = parent.concrete_frame.pop();
                    let concrete_callable = parent.concrete_frame.pop();
                    if concrete_callable.is_null() {
                        panic!("inline framestack lost concrete callable");
                    }
                    parent.concrete_frame.next_instr = pc + 1;
                }
                let (driver, _) = crate::eval::driver_pair();
                driver.enter_inline_frame(pending.green_key);
                ctx.push_inline_trace_position(pending.green_key);
                framestack.push(MIFrame {
                    sym: pending.sym,
                    concrete_frame: pending.concrete_frame,
                    drop_frame_opref: pending.drop_frame_opref,
                    green_key: pending.green_key,
                    arg_state: pyre_bytecode::bytecode::OpArgState::default(),
                    parent_fail_args: pending.parent_fail_args,
                    parent_fail_arg_types: pending.parent_fail_arg_types,
                    caller_result_stack_idx: pending.caller_result_stack_idx,
                });
                continue;
            }
            InlineTraceStepAction::Trace(majit_meta::TraceAction::Continue) => {}
            InlineTraceStepAction::Trace(majit_meta::TraceAction::Finish {
                finish_args, ..
            }) => {
                // PyPy finishframe(): pop frame, store result in parent
                let result_opref = finish_args[0];

                // Concrete execute the RETURN_VALUE
                let top = framestack.last_mut().unwrap();
                let code = unsafe { &*top.concrete_frame.code };
                let ni = top.concrete_frame.next_instr;
                let code_unit = code.instructions[ni];
                let (instruction, op_arg) = top.arg_state.get(code_unit);
                top.concrete_frame.next_instr = ni + 1;
                let next = top.concrete_frame.next_instr;
                let step = pyre_runtime::execute_opcode_step(
                    &mut top.concrete_frame,
                    code,
                    instruction,
                    op_arg,
                    next,
                )?;

                match step {
                    StepResult::Return(concrete_result) => {
                        // popframe()
                        let popped = framestack.pop().unwrap();
                        let result_type = popped.sym.value_type_of(result_opref);
                        let concrete_result = materialize_pending_inline_result(
                            pending_inline_result_from_concrete(result_type, concrete_result),
                        );
                        ctx.pop_inline_trace_position();

                        // Drop callee frame in trace
                        if let Some(frame_opref) = popped.drop_frame_opref {
                            ctx.call_void(
                                crate::call_jit::jit_drop_callee_frame as *const (),
                                &[frame_opref],
                            );
                        }

                        if framestack.is_empty() {
                            // Outermost inline frame returned
                            if majit_meta::majit_log_enabled() {
                                eprintln!("[jit][inline] return outermost");
                            }
                            ctx.truncate_inline_trace_positions(inline_trace_base);
                            return Ok((result_opref, result_type, concrete_result));
                        }

                        let (driver, _) = crate::eval::driver_pair();
                        driver.leave_inline_frame();
                        if majit_meta::majit_log_enabled() {
                            eprintln!(
                                "[jit][inline] pop frame, resume parent depth={}",
                                framestack.len()
                            );
                        }

                        // make_result_of_lastop: store result in parent
                        let resume_depth = framestack.len();
                        let parent = framestack.last_mut().unwrap();
                        let result_idx = popped
                            .caller_result_stack_idx
                            .expect("inline child frame lost caller result slot");
                        if majit_meta::majit_log_enabled() {
                            eprintln!(
                                "[jit][inline] write result depth={} idx={} opref={:?} type={:?}",
                                resume_depth, result_idx, result_opref, result_type,
                            );
                        }
                        let slot = parent
                            .sym
                            .symbolic_stack
                            .get_mut(result_idx)
                            .expect("inline caller result slot out of bounds");
                        *slot = result_opref;
                        if result_idx >= parent.sym.symbolic_stack_types.len() {
                            parent
                                .sym
                                .symbolic_stack_types
                                .resize(result_idx + 1, Type::Ref);
                        }
                        parent.sym.symbolic_stack_types[result_idx] = result_type;
                        parent
                            .sym
                            .transient_value_types
                            .insert(result_opref, result_type);
                        parent.concrete_frame.push(concrete_result);
                        continue; // ChangeFrame: resume parent
                    }
                    other => {
                        return Err(pyre_runtime::PyError::type_error(
                            "expected concrete return after trace finish",
                        ));
                    }
                }
            }
            InlineTraceStepAction::Trace(
                majit_meta::TraceAction::Abort | majit_meta::TraceAction::AbortPermanent,
            ) => {
                ctx.truncate_inline_trace_positions(inline_trace_base);
                return Err(pyre_runtime::PyError::type_error("inline trace aborted"));
            }
            InlineTraceStepAction::Trace(
                majit_meta::TraceAction::CloseLoop
                | majit_meta::TraceAction::CloseLoopWithArgs { .. },
            ) => {
                ctx.truncate_inline_trace_positions(inline_trace_base);
                return Err(pyre_runtime::PyError::type_error(
                    "inline callee has loop (not supported)",
                ));
            }
        }

        // ── run_one_step: EXECUTE ──
        let top = framestack.last_mut().unwrap();
        let code = unsafe { &*top.concrete_frame.code };
        let ni = top.concrete_frame.next_instr;
        let code_unit = code.instructions[ni];
        let (instruction, op_arg) = top.arg_state.get(code_unit);
        top.concrete_frame.next_instr = ni + 1;
        let next = top.concrete_frame.next_instr;
        if let Instruction::Call { argc } = instruction {
            if pyre_interp::call::replay_pending_inline_call(
                &mut top.concrete_frame,
                argc.get(op_arg) as usize,
            ) {
                continue;
            }
            execute_inline_residual_call(&mut top.concrete_frame, argc.get(op_arg) as usize)?;
            continue;
        }
        match pyre_runtime::execute_opcode_step(
            &mut top.concrete_frame,
            code,
            instruction,
            op_arg,
            next,
        )? {
            StepResult::Continue => {}
            StepResult::Return(_) | StepResult::Yield(_) => {
                ctx.truncate_inline_trace_positions(inline_trace_base);
                return Err(pyre_runtime::PyError::type_error(
                    "concrete return/yield without trace finish",
                ));
            }
            StepResult::CloseLoop {
                jump_args: _,
                loop_header_pc: _,
            } => {
                ctx.truncate_inline_trace_positions(inline_trace_base);
                return Err(pyre_runtime::PyError::type_error(
                    "inline callee has loop (not supported)",
                ));
            }
        }
    }
}
