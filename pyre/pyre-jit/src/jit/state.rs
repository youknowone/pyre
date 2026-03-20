//! JitState implementation for pyre.
//!
//! `PyreJitState` bridges the interpreter's `PyFrame` with majit's JIT
//! infrastructure. It extracts live values from the frame, restores them
//! after compiled code runs, and provides the meta/sym types for tracing.

use majit_ir::{DescrRef, OpCode, OpRef, Type, Value};
use majit_meta::virtualizable::VirtualizableInfo;
use majit_meta::{JitDriverStaticData, JitState, ResidualVirtualizableSync, TraceAction, TraceCtx};

use pyre_bytecode::bytecode::{BinaryOperator, CodeObject, ComparisonOperator, Instruction};
use pyre_object::PyObjectRef;
use pyre_object::pyobject::{
    BOOL_TYPE, DICT_TYPE, FLOAT_TYPE, INT_TYPE, LIST_TYPE, NONE_TYPE, OB_TYPE_OFFSET, PyType,
    TUPLE_TYPE, is_bool, is_dict, is_float, is_int, is_list, is_none, is_tuple,
};
use pyre_object::rangeobject::RANGE_ITER_TYPE;
use pyre_object::strobject::is_str;
use pyre_object::{
    PY_NULL, w_bool_from, w_int_get_value, w_int_new, w_list_can_append_without_realloc,
    w_list_is_inline_storage, w_list_len, w_list_new, w_list_uses_float_storage,
    w_list_uses_int_storage, w_list_uses_object_storage,
    w_str_get_value, w_tuple_len,
};
use pyre_objspace::truth_value as objspace_truth_value;
use pyre_runtime::{
    ArithmeticOpcodeHandler, BranchOpcodeHandler, ConstantOpcodeHandler, ControlFlowOpcodeHandler,
    IterOpcodeHandler, LocalOpcodeHandler, NamespaceOpcodeHandler, OpcodeStepExecutor, PyBigInt,
    PyError, PyNamespace, PyObjectArray, SharedOpcodeHandler, StackOpcodeHandler,
    TruthOpcodeHandler, decode_instruction_at, execute_opcode_step, is_builtin_func, is_func,
    range_iter_continues, w_builtin_func_name, w_func_get_code_ptr, w_func_get_globals,
};

use crate::jit::descr::{
    bool_boolval_descr, dict_len_descr, float_floatval_descr, int_intval_descr,
    list_float_items_len_descr, list_float_items_ptr_descr, list_int_items_len_descr,
    list_int_items_ptr_descr, list_items_heap_cap_descr, list_items_len_descr,
    list_items_ptr_descr, list_strategy_descr, make_array_descr, make_field_descr,
    namespace_values_len_descr, namespace_values_ptr_descr, ob_type_descr,
    range_iter_current_descr, range_iter_step_descr, range_iter_stop_descr, str_len_descr,
    tuple_items_len_descr, tuple_items_ptr_descr, w_float_size_descr, w_int_size_descr,
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
}

/// Symbolic state during tracing.
///
/// `frame` maps to a live IR `OpRef`. Symbolic frame field tracking
/// (locals, stack, valuestackdepth, next_instr) persists across instructions.
/// Locals and stack are virtualized (carried through JUMP args);
/// only next_instr and valuestackdepth are synced before guards / loop close.
pub struct PyreSym {
    /// OpRef for the owning PyFrame pointer.
    pub frame: OpRef,
    // ── Persistent symbolic frame field tracking ──
    // These fields survive across per-instruction TraceFrameState lifetimes.
    pub(crate) symbolic_locals: Vec<OpRef>,
    pub(crate) symbolic_stack: Vec<OpRef>,
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
    make_field_descr(OB_TYPE_OFFSET, 8, Type::Int, false)
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
    ctx.record_op_with_descr(OpCode::GetarrayitemRawF, &[array, index], float_array_descr())
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
    let fail_arg_types = fail_arg_types_for_virtualizable_state(fail_args.len());
    ctx.record_guard_typed_with_fail_args(opcode, args, fail_arg_types, &fail_args);
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

fn frame_callable_arg_types(nargs: usize) -> Vec<Type> {
    let mut types = Vec::with_capacity(2 + nargs);
    types.push(Type::Ref);
    types.push(Type::Ref);
    for _ in 0..nargs {
        types.push(Type::Ref);
    }
    types
}

fn frame_entry_arg_types(len: usize) -> Vec<Type> {
    if len <= 1 {
        vec![Type::Ref]
    } else {
        fail_arg_types_for_virtualizable_state(len)
    }
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
            pending_next_instr: None,
            locals_cells_stack_array_ref: OpRef::NONE,
            valuestackdepth: 0,
            nlocals: 0,
            symbolic_initialized: false,
            vable_next_instr: OpRef::NONE,
            vable_valuestackdepth: OpRef::NONE,
            symbolic_namespace_slots: std::collections::HashMap::new(),
            vable_array_base: None,
        }
    }

    /// Initialize symbolic tracking state on first trace instruction.
    /// Subsequent calls are no-ops (state persists across instructions).
    pub(crate) fn init_symbolic(&mut self, ctx: &mut TraceCtx, concrete_frame: usize) {
        if self.symbolic_initialized {
            return;
        }
        let nlocals = concrete_nlocals(concrete_frame).unwrap_or(0);
        let valuestackdepth = concrete_stack_depth(concrete_frame).unwrap_or(nlocals);
        let stack_only_depth = valuestackdepth.saturating_sub(nlocals);
        self.nlocals = nlocals;
        self.locals_cells_stack_array_ref = frame_locals_cells_stack_array(ctx, self.frame);
        self.symbolic_locals = if let Some(base) = self.vable_array_base {
            (0..nlocals).map(|i| OpRef(base + i as u32)).collect()
        } else {
            vec![OpRef::NONE; nlocals]
        };
        self.symbolic_stack = if let Some(base) = self.vable_array_base {
            let stack_base = base + nlocals as u32;
            (0..stack_only_depth)
                .map(|i| OpRef(stack_base + i as u32))
                .collect()
        } else {
            vec![OpRef::NONE; stack_only_depth]
        };
        self.pending_next_instr = None;
        self.valuestackdepth = valuestackdepth;
        self.symbolic_initialized = true;
    }

    /// Stack-only depth (number of values on the operand stack).
    #[inline]
    pub(crate) fn stack_only_depth(&self) -> usize {
        self.valuestackdepth.saturating_sub(self.nlocals)
    }
}

impl TraceFrameState {
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

    /// Build fail_args for the current frame only (no multi-frame header).
    /// RPython opencoder.py create_snapshot(): always includes locals + stack.
    fn build_single_frame_fail_args(&self) -> Vec<OpRef> {
        let s = self.sym();
        let mut fa = vec![s.frame, s.vable_next_instr, s.vable_valuestackdepth];
        let stack_only = s.stack_only_depth();
        fa.extend_from_slice(&s.symbolic_locals);
        fa.extend_from_slice(&s.symbolic_stack[..stack_only.min(s.symbolic_stack.len())]);
        fa
    }

    fn build_single_frame_fail_arg_types(&self) -> Vec<Type> {
        let s = self.sym();
        let stack_only = s.stack_only_depth();
        fail_arg_types_for_virtualizable_state(3 + s.symbolic_locals.len() + stack_only)
    }

    pub(crate) fn push_value(&mut self, _ctx: &mut TraceCtx, value: OpRef) {
        let s = self.sym_mut();
        let stack_idx = s.stack_only_depth();
        if stack_idx >= s.symbolic_stack.len() {
            s.symbolic_stack.resize(stack_idx + 1, OpRef::NONE);
        }
        s.symbolic_stack[stack_idx] = value;
        s.valuestackdepth += 1;
    }

    pub(crate) fn pop_value(&mut self, ctx: &mut TraceCtx) -> Result<OpRef, PyError> {
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
        s.valuestackdepth -= 1;
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
        Ok(s.symbolic_stack[stack_idx])
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

    fn pop_call_replay_stack(&mut self, ctx: &mut TraceCtx, args_len: usize) -> Result<(), PyError> {
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
        Ok(s.symbolic_locals[idx])
    }

    pub(crate) fn store_local_value(
        &mut self,
        _ctx: &mut TraceCtx,
        idx: usize,
        value: OpRef,
    ) -> Result<(), PyError> {
        let s = self.sym_mut();
        if idx >= s.symbolic_locals.len() {
            return Err(PyError::type_error("local index out of range in trace"));
        }
        s.symbolic_locals[idx] = value;
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
        let s = self.sym_mut();
        if let Some(pc) = s.pending_next_instr.take() {
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

    fn materialize_loop_carried_value(
        &mut self,
        ctx: &mut TraceCtx,
        value: OpRef,
        concrete_value: Option<PyObjectRef>,
    ) -> OpRef {
        let Some(concrete_value) = concrete_value else {
            return value;
        };
        if is_boxed_int_value(concrete_value) {
            let raw_value = self.trace_guarded_int_payload(ctx, value);
            return crate::jit::helpers::emit_box_int_inline(
                ctx,
                raw_value,
                w_int_size_descr(),
                ob_type_descr(),
                int_intval_descr(),
                &INT_TYPE as *const _ as i64,
            );
        }
        value
    }

    pub(crate) fn close_loop_args(&mut self, ctx: &mut TraceCtx) -> Vec<OpRef> {
        self.flush_to_frame(ctx);
        let (frame, next_instr, stack_depth, nlocals, locals, stack) = {
            let s = self.sym();
            let stack_only = s.stack_only_depth();
            (
                s.frame,
                s.vable_next_instr,
                s.vable_valuestackdepth,
                s.nlocals,
                s.symbolic_locals.clone(),
                s.symbolic_stack[..stack_only.min(s.symbolic_stack.len())].to_vec(),
            )
        };
        let mut args = vec![frame, next_instr, stack_depth];
        for (idx, value) in locals.into_iter().enumerate() {
            let concrete_value = concrete_stack_value(self.concrete_frame, idx);
            args.push(self.materialize_loop_carried_value(ctx, value, concrete_value));
        }
        for (stack_idx, value) in stack.into_iter().enumerate() {
            let concrete_value = concrete_stack_value(self.concrete_frame, nlocals + stack_idx);
            args.push(self.materialize_loop_carried_value(ctx, value, concrete_value));
        }
        args
    }

    /// Build the current fail_args for guards: [frame, ni, vsd, locals..., stack...]
    /// RPython pyjitpl.py capture_resumedata: always captures full frame state.
    pub(crate) fn current_fail_args(&self, _ctx: &mut TraceCtx) -> Vec<OpRef> {
        if let Some(ref pfa) = self.parent_fail_args {
            return pfa.clone();
        }
        let s = self.sym();
        let mut fa = vec![s.frame, s.vable_next_instr, s.vable_valuestackdepth];
        let stack_only = s.stack_only_depth();
        fa.extend_from_slice(&s.symbolic_locals);
        fa.extend_from_slice(&s.symbolic_stack[..stack_only.min(s.symbolic_stack.len())]);
        fa
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

        self.flush_to_frame(ctx);
        let s = self.sym();
        let stack_only = s.stack_only_depth();
        let mut fail_args = vec![s.frame, s.vable_next_instr, s.vable_valuestackdepth];
        fail_args.extend_from_slice(&s.symbolic_locals);
        fail_args.extend_from_slice(&s.symbolic_stack[..stack_only.min(s.symbolic_stack.len())]);
        let fail_arg_types = fail_arg_types_for_virtualizable_state(fail_args.len());
        ctx.record_guard_typed_with_fail_args(opcode, args, fail_arg_types, &fail_args);
    }

    pub(crate) fn guard_value(&mut self, ctx: &mut TraceCtx, value: OpRef, expected: i64) {
        let expected = ctx.const_int(expected);
        self.record_guard(ctx, OpCode::GuardValue, &[value, expected]);
    }

    pub(crate) fn guard_nonnull(&mut self, ctx: &mut TraceCtx, value: OpRef) {
        self.record_guard(ctx, OpCode::GuardNonnull, &[value]);
    }

    pub(crate) fn guard_range_iter(&mut self, ctx: &mut TraceCtx, obj: OpRef) {
        let range_iter_type_ptr = &RANGE_ITER_TYPE as *const PyType as usize as i64;
        let actual_type =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], self.ob_type_fd.clone());
        let expected = ctx.const_int(range_iter_type_ptr);
        self.record_guard(ctx, OpCode::GuardClass, &[actual_type, expected]);
    }

    pub(crate) fn record_for_iter_guard(
        &mut self,
        ctx: &mut TraceCtx,
        next: OpRef,
        continues: bool,
    ) {
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
        truth: OpRef,
        concrete_truth: bool,
    ) {
        let opcode = if concrete_truth {
            OpCode::GuardTrue
        } else {
            OpCode::GuardFalse
        };
        self.record_guard(ctx, opcode, &[truth]);
    }

    fn concrete_popped_value(&self) -> Option<PyObjectRef> {
        concrete_stack_value(self.concrete_frame, self.sym().valuestackdepth)
    }

    fn concrete_binary_operands(&self) -> Option<(PyObjectRef, PyObjectRef)> {
        let vsd = self.sym().valuestackdepth;
        Some((
            concrete_stack_value(self.concrete_frame, vsd)?,
            concrete_stack_value(self.concrete_frame, vsd + 1)?,
        ))
    }

    fn concrete_store_subscr_operands(&self) -> Option<(PyObjectRef, PyObjectRef, PyObjectRef)> {
        let vsd = self.sym().valuestackdepth;
        Some((
            concrete_stack_value(self.concrete_frame, vsd)?,
            concrete_stack_value(self.concrete_frame, vsd + 1)?,
            concrete_stack_value(self.concrete_frame, vsd + 2)?,
        ))
    }

    fn guard_int_object_value(&mut self, ctx: &mut TraceCtx, int_obj: OpRef, expected: i64) {
        self.guard_object_class(ctx, int_obj, &INT_TYPE as *const PyType);
        let actual_value =
            ctx.record_op_with_descr(OpCode::GetfieldGcI, &[int_obj], int_intval_descr());
        self.guard_value(ctx, actual_value, expected);
    }

    fn guard_object_class(&mut self, ctx: &mut TraceCtx, obj: OpRef, expected_type: *const PyType) {
        let actual_type =
            ctx.record_op_with_descr(OpCode::GetfieldGcI, &[obj], self.ob_type_fd.clone());
        let expected_type = ctx.const_int(expected_type as usize as i64);
        self.record_guard(ctx, OpCode::GuardClass, &[actual_type, expected_type]);
    }

    fn trace_guarded_int_payload(&mut self, ctx: &mut TraceCtx, int_obj: OpRef) -> OpRef {
        self.guard_object_class(ctx, int_obj, &INT_TYPE as *const PyType);
        ctx.record_op_with_descr(OpCode::GetfieldGcI, &[int_obj], int_intval_descr())
    }

    fn concrete_binary_int_operands(&self) -> Option<(i64, i64)> {
        let (lhs, rhs) = self.concrete_binary_operands()?;
        unsafe {
            if is_int(lhs) && is_int(rhs) {
                Some((w_int_get_value(lhs), w_int_get_value(rhs)))
            } else {
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
        let strategy = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], list_strategy_descr());
        self.guard_value(ctx, strategy, expected);
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
        let actual_type =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], self.ob_type_fd.clone());
        let expected_type = ctx.const_int(expected_type as usize as i64);
        self.record_guard(ctx, OpCode::GuardClass, &[actual_type, expected_type]);
        self.guard_int_object_value(ctx, key, concrete_index as i64);
        let len = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], items_len_descr);
        self.guard_len_gt_index(ctx, len, concrete_index);
        let items_ptr = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], items_ptr_descr);
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
        let actual_type =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], self.ob_type_fd.clone());
        let expected_type = ctx.const_int(expected_type as usize as i64);
        self.record_guard(ctx, OpCode::GuardClass, &[actual_type, expected_type]);
        self.guard_int_object_value(ctx, key, concrete_key);
        let len = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], items_len_descr);
        self.guard_len_eq(ctx, len, concrete_len);
        let items_ptr = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], items_ptr_descr);
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
        let actual_type =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], self.ob_type_fd.clone());
        let expected_type = ctx.const_int(&LIST_TYPE as *const PyType as usize as i64);
        self.record_guard(ctx, OpCode::GuardClass, &[actual_type, expected_type]);
        self.guard_list_strategy(ctx, obj, 0);
        self.guard_int_object_value(ctx, key, concrete_index as i64);
        let len = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], list_items_len_descr());
        self.guard_len_gt_index(ctx, len, concrete_index);
        let items_ptr = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], list_items_ptr_descr());
        let index = ctx.const_int(concrete_index as i64);
        trace_raw_array_getitem_value(ctx, items_ptr, index)
    }

    fn trace_direct_negative_object_list_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        concrete_key: i64,
        concrete_len: usize,
    ) -> OpRef {
        let normalized = (concrete_len as i64 + concrete_key) as usize;
        let actual_type =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], self.ob_type_fd.clone());
        let expected_type = ctx.const_int(&LIST_TYPE as *const PyType as usize as i64);
        self.record_guard(ctx, OpCode::GuardClass, &[actual_type, expected_type]);
        self.guard_list_strategy(ctx, obj, 0);
        self.guard_int_object_value(ctx, key, concrete_key);
        let len = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], list_items_len_descr());
        self.guard_len_eq(ctx, len, concrete_len);
        let items_ptr = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], list_items_ptr_descr());
        let index = ctx.const_int(normalized as i64);
        trace_raw_array_getitem_value(ctx, items_ptr, index)
    }

    fn trace_direct_int_list_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        concrete_index: usize,
    ) -> OpRef {
        let actual_type =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], self.ob_type_fd.clone());
        let expected_type = ctx.const_int(&LIST_TYPE as *const PyType as usize as i64);
        self.record_guard(ctx, OpCode::GuardClass, &[actual_type, expected_type]);
        self.guard_list_strategy(ctx, obj, 1);
        self.guard_int_object_value(ctx, key, concrete_index as i64);
        let len =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], list_int_items_len_descr());
        self.guard_len_gt_index(ctx, len, concrete_index);
        let items_ptr =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], list_int_items_ptr_descr());
        let index = ctx.const_int(concrete_index as i64);
        let raw = trace_raw_int_array_getitem_value(ctx, items_ptr, index);
        box_traced_raw_int(ctx, raw)
    }

    fn trace_direct_negative_int_list_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        concrete_key: i64,
        concrete_len: usize,
    ) -> OpRef {
        let normalized = (concrete_len as i64 + concrete_key) as usize;
        let actual_type =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], self.ob_type_fd.clone());
        let expected_type = ctx.const_int(&LIST_TYPE as *const PyType as usize as i64);
        self.record_guard(ctx, OpCode::GuardClass, &[actual_type, expected_type]);
        self.guard_list_strategy(ctx, obj, 1);
        self.guard_int_object_value(ctx, key, concrete_key);
        let len =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], list_int_items_len_descr());
        self.guard_len_eq(ctx, len, concrete_len);
        let items_ptr =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], list_int_items_ptr_descr());
        let index = ctx.const_int(normalized as i64);
        let raw = trace_raw_int_array_getitem_value(ctx, items_ptr, index);
        box_traced_raw_int(ctx, raw)
    }

    fn trace_direct_float_list_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        concrete_index: usize,
    ) -> OpRef {
        let actual_type =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], self.ob_type_fd.clone());
        let expected_type = ctx.const_int(&LIST_TYPE as *const PyType as usize as i64);
        self.record_guard(ctx, OpCode::GuardClass, &[actual_type, expected_type]);
        self.guard_list_strategy(ctx, obj, 2);
        self.guard_int_object_value(ctx, key, concrete_index as i64);
        let len =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], list_float_items_len_descr());
        self.guard_len_gt_index(ctx, len, concrete_index);
        let items_ptr =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], list_float_items_ptr_descr());
        let index = ctx.const_int(concrete_index as i64);
        let raw = trace_raw_float_array_getitem_value(ctx, items_ptr, index);
        box_traced_raw_float(ctx, raw)
    }

    fn trace_direct_negative_float_list_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        concrete_key: i64,
        concrete_len: usize,
    ) -> OpRef {
        let normalized = (concrete_len as i64 + concrete_key) as usize;
        let actual_type =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], self.ob_type_fd.clone());
        let expected_type = ctx.const_int(&LIST_TYPE as *const PyType as usize as i64);
        self.record_guard(ctx, OpCode::GuardClass, &[actual_type, expected_type]);
        self.guard_list_strategy(ctx, obj, 2);
        self.guard_int_object_value(ctx, key, concrete_key);
        let len =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], list_float_items_len_descr());
        self.guard_len_eq(ctx, len, concrete_len);
        let items_ptr =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[obj], list_float_items_ptr_descr());
        let index = ctx.const_int(normalized as i64);
        let raw = trace_raw_float_array_getitem_value(ctx, items_ptr, index);
        box_traced_raw_float(ctx, raw)
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
        let actual_type =
            ctx.record_op_with_descr(OpCode::GetfieldRawI, &[seq], self.ob_type_fd.clone());
        let expected = ctx.const_int(expected_type as usize as i64);
        self.record_guard(ctx, OpCode::GuardClass, &[actual_type, expected]);

        let len = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[seq], items_len_descr);
        self.guard_value(ctx, len, count as i64);

        let items_ptr = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[seq], items_ptr_descr);
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
            let result = if has_overflow {
                crate::jit::generated::trace_int_binop_ovf(
                    ctx,
                    a,
                    b,
                    op_code,
                    int_type_addr,
                    crate::jit::descr::ob_type_descr(),
                    crate::jit::descr::int_intval_descr(),
                    crate::jit::descr::w_int_size_descr(),
                    &fail_args,
                )
            } else {
                crate::jit::generated::trace_int_binop(
                    ctx,
                    a,
                    b,
                    op_code,
                    int_type_addr,
                    crate::jit::descr::ob_type_descr(),
                    crate::jit::descr::int_intval_descr(),
                    crate::jit::descr::w_int_size_descr(),
                    &fail_args,
                )
            };
            Ok(result)
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
            let fail_args = this.current_fail_args(ctx);
            let float_type_addr = &FLOAT_TYPE as *const _ as i64;
            let result = crate::jit::generated::trace_float_binop(
                ctx,
                a,
                b,
                op_code,
                float_type_addr,
                crate::jit::descr::ob_type_descr(),
                crate::jit::descr::float_floatval_descr(),
                crate::jit::descr::w_float_size_descr(),
                &fail_args,
            );
            Ok(result)
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
                    let truth = crate::jit::generated::trace_int_compare(
                        ctx,
                        a,
                        b,
                        cmp,
                        int_type_addr,
                        ob_type_descr(),
                        int_intval_descr(),
                        &fail_args,
                    );
                    Ok(emit_trace_bool_value_from_truth(ctx, truth, false))
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
                    let truth = crate::jit::generated::trace_float_compare(
                        ctx,
                        a,
                        b,
                        cmp,
                        float_type_addr,
                        ob_type_descr(),
                        float_floatval_descr(),
                        &fail_args,
                    );
                    Ok(emit_trace_bool_value_from_truth(ctx, truth, false))
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
                            let actual_type = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                this.ob_type_fd.clone(),
                            );
                            let expected_type =
                                ctx.const_int(&LIST_TYPE as *const PyType as usize as i64);
                            this.record_guard(
                                ctx,
                                OpCode::GuardClass,
                                &[actual_type, expected_type],
                            );
                            this.guard_list_strategy(ctx, obj, 0);
                            this.guard_int_object_value(ctx, key, index as i64);
                            let len = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                list_items_len_descr(),
                            );
                            this.guard_len_gt_index(ctx, len, index);
                            let items_ptr = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                list_items_ptr_descr(),
                            );
                            let index = ctx.const_int(index as i64);
                            trace_raw_array_setitem_value(ctx, items_ptr, index, value);
                            Ok(())
                        });
                    }
                } else if let Some(abs_index) = index
                    .checked_neg()
                    .and_then(|value| usize::try_from(value).ok())
                {
                    if abs_index <= concrete_len {
                        let normalized = concrete_len - abs_index;
                        return self.with_ctx(|this, ctx| {
                            let actual_type = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                this.ob_type_fd.clone(),
                            );
                            let expected_type =
                                ctx.const_int(&LIST_TYPE as *const PyType as usize as i64);
                            this.record_guard(
                                ctx,
                                OpCode::GuardClass,
                                &[actual_type, expected_type],
                            );
                            this.guard_list_strategy(ctx, obj, 0);
                            this.guard_int_object_value(ctx, key, index);
                            let len = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                list_items_len_descr(),
                            );
                            this.guard_len_eq(ctx, len, concrete_len);
                            let items_ptr = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                list_items_ptr_descr(),
                            );
                            let index = ctx.const_int(normalized as i64);
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
                            let actual_type = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                this.ob_type_fd.clone(),
                            );
                            let expected_type =
                                ctx.const_int(&LIST_TYPE as *const PyType as usize as i64);
                            this.record_guard(
                                ctx,
                                OpCode::GuardClass,
                                &[actual_type, expected_type],
                            );
                            this.guard_list_strategy(ctx, obj, 1);
                            this.guard_int_object_value(ctx, key, index as i64);
                            let len = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                list_int_items_len_descr(),
                            );
                            this.guard_len_gt_index(ctx, len, index);
                            let items_ptr = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                list_int_items_ptr_descr(),
                            );
                            let fail_args = this.current_fail_args(ctx);
                            let raw = crate::jit::generated::trace_unbox_int(
                                ctx,
                                value,
                                &INT_TYPE as *const _ as i64,
                                ob_type_descr(),
                                int_intval_descr(),
                                &fail_args,
                            );
                            let index = ctx.const_int(index as i64);
                            trace_raw_int_array_setitem_value(ctx, items_ptr, index, raw);
                            Ok(())
                        });
                    }
                } else if let Some(abs_index) = index
                    .checked_neg()
                    .and_then(|value| usize::try_from(value).ok())
                {
                    if abs_index <= concrete_len {
                        let normalized = concrete_len - abs_index;
                        return self.with_ctx(|this, ctx| {
                            let actual_type = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                this.ob_type_fd.clone(),
                            );
                            let expected_type =
                                ctx.const_int(&LIST_TYPE as *const PyType as usize as i64);
                            this.record_guard(
                                ctx,
                                OpCode::GuardClass,
                                &[actual_type, expected_type],
                            );
                            this.guard_list_strategy(ctx, obj, 1);
                            this.guard_int_object_value(ctx, key, index);
                            let len = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                list_int_items_len_descr(),
                            );
                            this.guard_len_eq(ctx, len, concrete_len);
                            let items_ptr = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                list_int_items_ptr_descr(),
                            );
                            let fail_args = this.current_fail_args(ctx);
                            let raw = crate::jit::generated::trace_unbox_int(
                                ctx,
                                value,
                                &INT_TYPE as *const _ as i64,
                                ob_type_descr(),
                                int_intval_descr(),
                                &fail_args,
                            );
                            let index = ctx.const_int(normalized as i64);
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
                            let actual_type = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                this.ob_type_fd.clone(),
                            );
                            let expected_type =
                                ctx.const_int(&LIST_TYPE as *const PyType as usize as i64);
                            this.record_guard(
                                ctx,
                                OpCode::GuardClass,
                                &[actual_type, expected_type],
                            );
                            this.guard_list_strategy(ctx, obj, 2);
                            this.guard_int_object_value(ctx, key, index as i64);
                            let len = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                list_float_items_len_descr(),
                            );
                            this.guard_len_gt_index(ctx, len, index);
                            let items_ptr = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                list_float_items_ptr_descr(),
                            );
                            let fail_args = this.current_fail_args(ctx);
                            let raw = crate::jit::generated::trace_unbox_float(
                                ctx,
                                value,
                                &FLOAT_TYPE as *const _ as i64,
                                ob_type_descr(),
                                float_floatval_descr(),
                                &fail_args,
                            );
                            let index = ctx.const_int(index as i64);
                            trace_raw_float_array_setitem_value(ctx, items_ptr, index, raw);
                            Ok(())
                        });
                    }
                } else if let Some(abs_index) = index
                    .checked_neg()
                    .and_then(|value| usize::try_from(value).ok())
                {
                    if abs_index <= concrete_len {
                        let normalized = concrete_len - abs_index;
                        return self.with_ctx(|this, ctx| {
                            let actual_type = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                this.ob_type_fd.clone(),
                            );
                            let expected_type =
                                ctx.const_int(&LIST_TYPE as *const PyType as usize as i64);
                            this.record_guard(
                                ctx,
                                OpCode::GuardClass,
                                &[actual_type, expected_type],
                            );
                            this.guard_list_strategy(ctx, obj, 2);
                            this.guard_int_object_value(ctx, key, index);
                            let len = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                list_float_items_len_descr(),
                            );
                            this.guard_len_eq(ctx, len, concrete_len);
                            let items_ptr = ctx.record_op_with_descr(
                                OpCode::GetfieldRawI,
                                &[obj],
                                list_float_items_ptr_descr(),
                            );
                            let fail_args = this.current_fail_args(ctx);
                            let raw = crate::jit::generated::trace_unbox_float(
                                ctx,
                                value,
                                &FLOAT_TYPE as *const _ as i64,
                                ob_type_descr(),
                                float_floatval_descr(),
                                &fail_args,
                            );
                            let index = ctx.const_int(normalized as i64);
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
                    let actual_type = ctx.record_op_with_descr(
                        OpCode::GetfieldRawI,
                        &[list],
                        this.ob_type_fd.clone(),
                    );
                    let expected_type = ctx.const_int(&LIST_TYPE as *const PyType as usize as i64);
                    this.record_guard(ctx, OpCode::GuardClass, &[actual_type, expected_type]);
                    this.guard_list_strategy(ctx, list, 0);
                    if w_list_is_inline_storage(concrete_list) {
                        let heap_cap = ctx.record_op_with_descr(
                            OpCode::GetfieldRawI,
                            &[list],
                            list_items_heap_cap_descr(),
                        );
                        this.guard_value(ctx, heap_cap, 0);
                    } else {
                        let len = ctx.record_op_with_descr(
                            OpCode::GetfieldRawI,
                            &[list],
                            list_items_len_descr(),
                        );
                        this.guard_value(ctx, len, concrete_len as i64);
                    }
                    let items_ptr = ctx.record_op_with_descr(
                        OpCode::GetfieldRawI,
                        &[list],
                        list_items_ptr_descr(),
                    );
                    let index = ctx.const_int(concrete_len as i64);
                    trace_raw_array_setitem_value(ctx, items_ptr, index, value);
                    let new_len = ctx.const_int((concrete_len + 1) as i64);
                    ctx.record_op_with_descr(
                        OpCode::SetfieldRaw,
                        &[list, new_len],
                        list_items_len_descr(),
                    );
                    Ok(())
                });
            }
        }

        self.trace_list_append(list, value)
    }

    pub(crate) fn concrete_iter_continues(&self) -> Result<bool, PyError> {
        let concrete_iter =
            concrete_stack_value(self.concrete_frame, self.sym().valuestackdepth - 1)
                .ok_or_else(|| PyError::type_error("missing concrete iterator during trace"))?;
        range_iter_continues(concrete_iter)
    }

    fn concrete_callable_after_pops(&self) -> Option<PyObjectRef> {
        concrete_stack_value(self.concrete_frame, self.sym().valuestackdepth)
    }

    fn concrete_call_arg_after_pops(&self, arg_idx: usize) -> Option<PyObjectRef> {
        concrete_stack_value(
            self.concrete_frame,
            self.sym().valuestackdepth + 2 + arg_idx,
        )
    }

    fn direct_len_value(&mut self, callable: OpRef, value: OpRef) -> Result<OpRef, PyError> {
        let Some(concrete_value) = self.concrete_call_arg_after_pops(0) else {
            return self.with_ctx(|_this, ctx| {
                crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, &[value])
            });
        };

        unsafe {
            if is_str(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &pyre_object::STR_TYPE as *const PyType);
                    let len =
                        ctx.record_op_with_descr(OpCode::GetfieldRawI, &[value], str_len_descr());
                    Ok({
                        let int_type_addr = &INT_TYPE as *const _ as i64;
                        crate::jit::generated::trace_box_int(
                            ctx,
                            len,
                            w_int_size_descr(),
                            ob_type_descr(),
                            int_intval_descr(),
                            int_type_addr,
                        )
                    })
                });
            }
            if is_dict(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &DICT_TYPE as *const PyType);
                    let len =
                        ctx.record_op_with_descr(OpCode::GetfieldRawI, &[value], dict_len_descr());
                    Ok({
                        let int_type_addr = &INT_TYPE as *const _ as i64;
                        crate::jit::generated::trace_box_int(
                            ctx,
                            len,
                            w_int_size_descr(),
                            ob_type_descr(),
                            int_intval_descr(),
                            int_type_addr,
                        )
                    })
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
                    Ok({
                        let int_type_addr = &INT_TYPE as *const _ as i64;
                        crate::jit::generated::trace_box_int(
                            ctx,
                            len,
                            w_int_size_descr(),
                            ob_type_descr(),
                            int_intval_descr(),
                            int_type_addr,
                        )
                    })
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
                    Ok({
                        let int_type_addr = &INT_TYPE as *const _ as i64;
                        crate::jit::generated::trace_box_int(
                            ctx,
                            len,
                            w_int_size_descr(),
                            ob_type_descr(),
                            int_intval_descr(),
                            int_type_addr,
                        )
                    })
                });
            }
        }

        self.with_ctx(|_this, ctx| {
            crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, &[value])
        })
    }

    fn direct_abs_value(&mut self, callable: OpRef, value: OpRef) -> Result<OpRef, PyError> {
        let Some(concrete_value) = self.concrete_call_arg_after_pops(0) else {
            return self.with_ctx(|_this, ctx| {
                crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, &[value])
            });
        };

        unsafe {
            if is_int(concrete_value) {
                let concrete_int = w_int_get_value(concrete_value);
                if concrete_int == i64::MIN {
                    return self.with_ctx(|_this, ctx| {
                        crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, &[value])
                    });
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

        self.with_ctx(|_this, ctx| {
            crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, &[value])
        })
    }

    fn direct_type_value(&mut self, callable: OpRef, value: OpRef) -> Result<OpRef, PyError> {
        let Some(concrete_value) = self.concrete_call_arg_after_pops(0) else {
            return self.with_ctx(|_this, ctx| {
                crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, &[value])
            });
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
            return self.with_ctx(|_this, ctx| {
                crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, &[obj, type_name])
            });
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

        self.with_ctx(|_this, ctx| {
            crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, &[obj, type_name])
        })
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
            return self.with_ctx(|_this, ctx| {
                crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, &[a, b])
            });
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
                    return self.with_ctx(|_this, ctx| {
                        crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, &[a, b])
                    });
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

        self.with_ctx(|_this, ctx| {
            crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, &[a, b])
        })
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
                        this.guard_value(ctx, callable, concrete_callable as i64)
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
                        this.guard_value(ctx, callable, concrete_callable as i64)
                    });
                    return self.direct_isinstance_value(callable, args[0], args[1]);
                } else if args.len() == 2 && builtin_name == "min" {
                    self.with_ctx(|this, ctx| {
                        this.guard_value(ctx, callable, concrete_callable as i64)
                    });
                    return self.direct_minmax_value(callable, args[0], args[1], false);
                } else if args.len() == 2 && builtin_name == "max" {
                    self.with_ctx(|this, ctx| {
                        this.guard_value(ctx, callable, concrete_callable as i64)
                    });
                    return self.direct_minmax_value(callable, args[0], args[1], true);
                }
                return self.with_ctx(|this, ctx| {
                    this.guard_value(ctx, callable, concrete_callable as i64);
                    crate::jit::helpers::emit_trace_call_known_builtin(ctx, callable, args)
                });
            }
            if is_func(concrete_callable) {
                let callee_code_ptr = w_func_get_code_ptr(concrete_callable) as *const CodeObject;
                let callee_key = crate::eval::make_green_key(callee_code_ptr, 0);
                let callee_code = unsafe { &*callee_code_ptr };
                let callee_has_loop = code_has_backward_jump(callee_code);
                let (driver, _) = crate::eval::driver_pair();
                let nargs = args.len();

                // PyPy perform_call: trace INTO callee body when the call
                // should be inlined. Self-recursive portal calls are handled
                // by max_unroll_recursion in should_inline(); if it still says
                // Inline, prefer the same trace-through path before falling
                // back to helper-based CallMayForce.
                let current_green_key = unsafe {
                    let cf = &*(self.concrete_frame as *const pyre_interp::frame::PyFrame);
                    crate::eval::make_green_key(cf.code, 0)
                };
                let is_self_recursive = callee_key == current_green_key;
                let inline_decision = driver.should_inline(callee_key);
                let inline_framestack_active = self.parent_fail_args.is_some();

                // RPython traces through regular callees, but recursive
                // calls converge to separate functraces + call_assembler.
                //
                // We do not yet have full multi-frame resumedata / ChangeFrame
                // recovery for trace-through recursion, so keep self-recursive
                // calls out of the trace-through path.  They still use the
                // regular compiled-call helpers below.
                let can_trace_through = !is_self_recursive && nargs <= 4 && !callee_has_loop;

                if inline_decision == majit_meta::InlineDecision::Inline && can_trace_through {
                    if inline_framestack_active {
                        if let Ok(pending) = self.build_pending_inline_frame(
                            callable,
                            args,
                            concrete_callable,
                            callee_key,
                        ) {
                            self.pending_inline_frame = Some(pending);
                            return self
                                .with_ctx(|_, ctx| Ok(ctx.const_int(pyre_object::PY_NULL as i64)));
                        }
                    }
                    if let Ok(result) =
                        self.trace_through_callee(callable, args, concrete_callable, callee_key)
                    {
                        return Ok(result);
                    }
                    // trace-through failed: fall through to the older helper
                    // path so correctness wins over optimization.
                }

                // PyPy converges self-recursion to separate functrace +
                // call_assembler once compiled code exists. Our current
                // self-recursive "Inline" path is still helper-boundary,
                // so prefer direct CallAssembler as soon as a compiled token
                // exists instead of paying that extra helper layer.
                if majit_meta::majit_log_enabled() {
                    eprintln!(
                        "[jit][call-check] is_self={} cache_safe={} inline_active={} callee_key={}",
                        is_self_recursive,
                        crate::call_jit::recursive_force_cache_safe(concrete_callable),
                        inline_framestack_active,
                        callee_key
                    );
                }
                if is_self_recursive
                    && crate::call_jit::recursive_force_cache_safe(concrete_callable)
                {
                    // RPython parity: compiled code calls compiled code
                    // directly via CALL_ASSEMBLER. Guard failures in the
                    // callee are handled by force_fn → resume_in_blackhole.
                    // This is the fast path matching RPython's
                    // call_assembler / assembler_call_helper pattern.
                    // RPython warmstate.py:714 get_assembler_token: use
                    // compiled token or pending token (compile_tmp_callback).
                    if let Some(token_number) = driver
                        .get_loop_token_number(callee_key)
                        .or_else(|| driver.get_pending_token_number(callee_key))
                    {
                        // For pending token (not yet compiled), use current
                        // trace's own metadata since it's self-recursive.
                        let (callee_nlocals, callee_vsd, target_num_inputs) =
                            if let Some(callee_meta) = driver.get_compiled_meta(callee_key) {
                                (
                                    callee_meta.num_locals,
                                    callee_meta.valuestackdepth,
                                    driver.get_compiled_num_inputs(callee_key).unwrap_or(1),
                                )
                            } else {
                                // Pending token: self-recursive, use caller's info
                                let code = unsafe { &*(w_func_get_code_ptr(concrete_callable) as *const CodeObject) };
                                (code.varnames.len(), code.varnames.len() + 1, 4)
                            };
                        let callee_stack_only = callee_vsd.saturating_sub(callee_nlocals);

                        if let Some(frame_helper) = crate::call_jit::callee_frame_helper(nargs) {
                            return self.with_ctx(|this, ctx| {
                                this.guard_value(ctx, callable, concrete_callable as i64);
                                let callee_frame = if nargs == 1 {
                                    ctx.call_ref_typed(
                                        crate::call_jit::jit_create_self_recursive_callee_frame_1
                                            as *const (),
                                        &[this.frame(), args[0]],
                                        &[Type::Ref, Type::Ref],
                                    )
                                } else {
                                    let mut helper_args = vec![this.frame(), callable];
                                    helper_args.extend_from_slice(args);
                                    let helper_arg_types = frame_callable_arg_types(args.len());
                                    ctx.call_ref_typed(frame_helper, &helper_args, &helper_arg_types)
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
                                let ca_arg_types = frame_entry_arg_types(ca_args.len());
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
                if let Some(token_number) = driver.get_pending_token_number(callee_key)
                {
                    let callee_nlocals = {
                        let code_ptr = w_func_get_code_ptr(concrete_callable) as *const CodeObject;
                        (&*code_ptr).varnames.len()
                    };
                    if let Some(frame_helper) = crate::call_jit::callee_frame_helper(nargs) {
                        return self.with_ctx(|this, ctx| {
                            this.guard_value(ctx, callable, concrete_callable as i64);
                            let callee_frame = if is_self_recursive && nargs == 1 {
                                ctx.call_ref_typed(
                                    crate::call_jit::jit_create_self_recursive_callee_frame_1
                                        as *const (),
                                    &[this.frame(), args[0]],
                                    &[Type::Ref, Type::Ref],
                                )
                            } else {
                                let mut helper_args = vec![this.frame(), callable];
                                helper_args.extend_from_slice(args);
                                let helper_arg_types = frame_callable_arg_types(args.len());
                                ctx.call_ref_typed(frame_helper, &helper_args, &helper_arg_types)
                            };
                            let ca_args = synthesize_fresh_callee_entry_args(
                                ctx,
                                callee_frame,
                                args,
                                callee_nlocals,
                            );
                            let ca_arg_types = frame_entry_arg_types(ca_args.len());
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
                        if !is_self_recursive && nargs <= 4 && !callee_has_loop {
                            if let Ok(result) = self.trace_through_callee(
                                callable,
                                args,
                                concrete_callable,
                                callee_key,
                            ) {
                                return Ok(result);
                            }
                        }
                        let Some(token_number) = driver.get_loop_token_number(callee_key) else {
                            let call_pc = self.fallthrough_pc.saturating_sub(1);
                            return self.with_ctx(|this, ctx| {
                                this.guard_value(ctx, callable, concrete_callable as i64);
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

                        if let Some(frame_helper) = crate::call_jit::callee_frame_helper(nargs)
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
                                this.guard_value(ctx, callable, concrete_callable as i64);
                                let mut helper_args = vec![this.frame(), callable];
                                helper_args.extend_from_slice(args);
                                let helper_arg_types = frame_callable_arg_types(args.len());
                                let callee_frame =
                                    ctx.call_ref_typed(frame_helper, &helper_args, &helper_arg_types);

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
                                let ca_arg_types = frame_entry_arg_types(ca_args.len());
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
                    this.guard_value(ctx, callable, concrete_callable as i64);
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
            this.guard_value(ctx, callable, concrete_callable as i64);
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

        let caller = unsafe { &*(self.concrete_frame as *const PyFrame) };
        let code_ptr = unsafe { w_func_get_code_ptr(concrete_callable) } as *const CodeObject;
        let globals = unsafe { w_func_get_globals(concrete_callable) };
        let closure = unsafe { pyre_runtime::w_func_get_closure(concrete_callable) };
        let is_self_recursive = crate::eval::make_green_key(caller.code, 0) == callee_key;
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
            sym.symbolic_stack = Vec::new();
            sym.symbolic_initialized = true;
            let (vable_next_instr, vable_valuestackdepth) =
                self.with_ctx(|_, ctx| (ctx.const_int(0), ctx.const_int(callee_nlocals as i64)));
            sym.vable_next_instr = vable_next_instr;
            sym.vable_valuestackdepth = vable_valuestackdepth;
            (sym, None)
        } else {
            // Create symbolic OpRef for callee frame in trace
            let callee_frame_opref = self.with_ctx(|this, ctx| {
                if is_self_recursive && args.len() == 1 {
                    ctx.call_ref_typed(
                        crate::call_jit::jit_create_self_recursive_callee_frame_1 as *const (),
                        &[this.frame(), args[0]],
                        &[Type::Ref, Type::Ref],
                    )
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
            for i in 0..sym.nlocals {
                if i < args.len() {
                    sym.symbolic_locals.push(args[i]);
                } else {
                    sym.symbolic_locals.push(OpRef::NONE);
                }
            }
            sym.symbolic_stack = Vec::new();
            sym.symbolic_initialized = true;
            (sym, Some(callee_frame_opref))
        };

        // GuardNotForced in callee → interpreter must re-execute the CALL.
        // Temporarily restore callable+args to parent stack and set PC to CALL.
        let call_pc = self.fallthrough_pc.saturating_sub(1);
        self.with_ctx(|this, ctx| this.push_call_replay_stack(ctx, callable, args, call_pc));
        self.with_ctx(|this, ctx| this.flush_to_frame(ctx));
        let parent_fail_args = self.build_single_frame_fail_args();
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
        let (result_opref, concrete_result) = result?;

        // Write the concrete result onto the owning caller frame so the
        // following concrete CALL replay can consume it without a
        // thread-global side channel.
        let caller_frame =
            unsafe { &mut *(self.concrete_frame as *mut pyre_interp::frame::PyFrame) };
        pyre_interp::call::set_pending_inline_result(caller_frame, concrete_result);

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
        let raw_finish_ready = driver.has_raw_int_finish(callee_key);
        let concrete_arg0 = if args.len() == 1 {
            self.concrete_call_arg_after_pops(0)
        } else {
            None
        };
        // Save CALL instruction PC so GuardNotForced can resume the CALL.
        let call_pc = self.fallthrough_pc.saturating_sub(1);

        let result = self.with_ctx(|this, ctx| {
            this.guard_value(ctx, callable, concrete_callable as i64);

            if args.len() == 1 {
                let result = if matches!(concrete_arg0, Some(arg) if unsafe { is_int(arg) }) {
                    let raw_arg = this.trace_guarded_int_payload(ctx, args[0]);
                    let is_self_recursive = callee_key
                        == unsafe {
                            crate::eval::make_green_key(
                                (*(this.concrete_frame as *const pyre_interp::frame::PyFrame))
                                    .code,
                                0,
                            )
                        };
                    if raw_finish_ready {
                        let force_fn = if is_self_recursive
                            && crate::call_jit::recursive_force_cache_safe(concrete_callable)
                        {
                            crate::call_jit::jit_force_self_recursive_call_raw_1 as *const ()
                        } else {
                            crate::call_jit::jit_force_recursive_call_raw_1 as *const ()
                        };
                        if force_fn
                            == crate::call_jit::jit_force_self_recursive_call_raw_1 as *const ()
                        {
                            this.sync_standard_virtualizable_before_residual_call(ctx);
                            let raw = ctx.call_may_force_int_typed(
                                force_fn,
                                &[this.frame(), raw_arg],
                                &[Type::Ref, Type::Int],
                            );
                            let forced = this.sync_standard_virtualizable_after_residual_call();
                            if this.parent_fail_args.is_some() {
                                let result = box_traced_raw_int(ctx, raw);
                                if !forced {
                                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                                    this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                                    this.pop_call_replay_stack(ctx, args.len())?;
                                }
                                result
                            } else {
                                if !forced {
                                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                                    this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                                    this.pop_call_replay_stack(ctx, args.len())?;
                                }
                                raw
                            }
                        } else {
                            this.sync_standard_virtualizable_before_residual_call(ctx);
                            let raw = ctx.call_may_force_int_typed(
                                force_fn,
                                &[this.frame(), callable, raw_arg],
                                &[Type::Ref, Type::Ref, Type::Int],
                            );
                            let forced = this.sync_standard_virtualizable_after_residual_call();
                            if this.parent_fail_args.is_some() {
                                let result = box_traced_raw_int(ctx, raw);
                                if !forced {
                                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                                    this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                                    this.pop_call_replay_stack(ctx, args.len())?;
                                }
                                result
                            } else {
                                if !forced {
                                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                                    this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                                    this.pop_call_replay_stack(ctx, args.len())?;
                                }
                                raw
                            }
                        }
                    } else {
                        // Keep the helper's return protocol stable for the
                        // lifetime of this trace. Before the callee has a raw
                        // int finish, use the boxed helper with a Ref result,
                        // matching RPython's call_assembler result kind.
                        let force_fn = if is_self_recursive
                            && crate::call_jit::recursive_force_cache_safe(concrete_callable)
                        {
                            crate::call_jit::jit_force_self_recursive_call_argraw_boxed_1
                                as *const ()
                        } else {
                            crate::call_jit::jit_force_recursive_call_argraw_boxed_1 as *const ()
                        };
                        this.sync_standard_virtualizable_before_residual_call(ctx);
                        let result = if force_fn
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
                        if !this.sync_standard_virtualizable_after_residual_call() {
                            this.push_call_replay_stack(ctx, callable, args, call_pc);
                            this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                            this.pop_call_replay_stack(ctx, args.len())?;
                        }
                        result
                    }
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
                let result =
                    ctx.call_may_force_ref_typed(force_fn, &[callee_frame], &[Type::Ref]);
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
        let concrete_iter =
            concrete_stack_value(self.concrete_frame, self.sym().valuestackdepth - 1)
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

            let current =
                ctx.record_op_with_descr(OpCode::GetfieldRawI, &[iter], range_iter_current_descr());
            let stop =
                ctx.record_op_with_descr(OpCode::GetfieldRawI, &[iter], range_iter_stop_descr());
            let step =
                ctx.record_op_with_descr(OpCode::GetfieldRawI, &[iter], range_iter_step_descr());
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
                return Ok(zero);
            }

            let next_current = ctx.record_op(OpCode::IntAddOvf, &[current, step]);
            this.record_guard(ctx, OpCode::GuardNoOverflow, &[]);
            ctx.record_op_with_descr(
                OpCode::SetfieldRaw,
                &[iter, next_current],
                range_iter_current_descr(),
            );
            let int_type_addr = &INT_TYPE as *const _ as i64;
            Ok(crate::jit::generated::trace_box_int(
                ctx,
                current,
                w_int_size_descr(),
                ob_type_descr(),
                int_intval_descr(),
                int_type_addr,
            ))
        })
    }

    pub(crate) fn concrete_branch_truth(&self) -> Result<bool, PyError> {
        let concrete_val = concrete_stack_value(self.concrete_frame, self.sym().valuestackdepth)
            .ok_or_else(|| PyError::type_error("missing concrete branch value during trace"))?;
        Ok(objspace_truth_value(concrete_val))
    }

    pub(crate) fn truth_value_direct(&mut self, value: OpRef) -> Result<OpRef, PyError> {
        let Some(concrete_value) = self.concrete_popped_value() else {
            return self.trace_truth_value(value);
        };

        unsafe {
            if is_int(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    let fail_args = this.current_fail_args(ctx);
                    let int_type_addr = &INT_TYPE as *const _ as i64;
                    let int_value = crate::jit::generated::trace_unbox_int(
                        ctx,
                        value,
                        int_type_addr,
                        ob_type_descr(),
                        int_intval_descr(),
                        &fail_args,
                    );
                    let zero = ctx.const_int(0);
                    Ok(ctx.record_op(OpCode::IntNe, &[int_value, zero]))
                });
            }
            if is_bool(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    let fail_args = this.current_fail_args(ctx);
                    let bool_type_addr = &BOOL_TYPE as *const _ as i64;
                    let bool_value = crate::jit::generated::trace_unbox_int(
                        ctx,
                        value,
                        bool_type_addr,
                        ob_type_descr(),
                        bool_boolval_descr(),
                        &fail_args,
                    );
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
            let fail_args = this.current_fail_args(ctx);
            let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
            let payload = crate::jit::generated::trace_unbox_int(
                ctx,
                value,
                int_type_addr,
                ob_type_descr(),
                int_intval_descr(),
                &fail_args,
            );
            if matches!(opcode, OpCode::IntNeg) {
                let min_val = ctx.const_int(i64::MIN);
                let is_min = ctx.record_op(OpCode::IntEq, &[payload, min_val]);
                this.record_guard(ctx, OpCode::GuardFalse, &[is_min]);
            }
            let result = ctx.record_op(opcode, &[payload]);
            Ok(crate::jit::generated::trace_box_int(
                ctx,
                result,
                w_int_size_descr(),
                ob_type_descr(),
                int_intval_descr(),
                int_type_addr,
            ))
        })
    }

    pub(crate) fn into_trace_action(
        &mut self,
        result: Result<pyre_runtime::StepResult<OpRef>, PyError>,
    ) -> TraceAction {
        trace_step_result_to_action(self.ctx(), result)
    }

    pub(crate) fn trace_code_step(&mut self, code: &CodeObject, pc: usize) -> TraceAction {
        if pc >= code.instructions.len() {
            return TraceAction::Abort;
        }

        let Some((instruction, op_arg)) = decode_instruction_at(code, pc) else {
            return TraceAction::Abort;
        };

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
            return InlineTraceStepAction::Trace(TraceAction::Abort);
        }

        let Some((instruction, op_arg)) = decode_instruction_at(code, pc) else {
            return InlineTraceStepAction::Trace(TraceAction::Abort);
        };

        self.prepare_fallthrough();
        let step_result = execute_opcode_step(self, code, instruction, op_arg, pc + 1);
        let action = self.into_trace_action(step_result);
        match action {
            TraceAction::Continue => {
                if let Some(mut pending) = self.pending_inline_frame.take() {
                    let result_idx = self
                        .sym()
                        .symbolic_stack
                        .len()
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
    ctx: &TraceCtx,
    result: Result<pyre_runtime::StepResult<OpRef>, PyError>,
) -> TraceAction {
    match result {
        Ok(pyre_runtime::StepResult::Continue) => {
            if ctx.is_too_long() {
                TraceAction::Abort
            } else {
                TraceAction::Continue
            }
        }
        Ok(pyre_runtime::StepResult::CloseLoop(jump_args)) => {
            TraceAction::CloseLoopWithArgs { jump_args }
        }
        Ok(pyre_runtime::StepResult::Return(value)) => TraceAction::Finish {
            finish_args: vec![value],
            finish_arg_types: vec![Type::Int],
        },
        Ok(pyre_runtime::StepResult::Yield(value)) => TraceAction::Finish {
            finish_args: vec![value],
            finish_arg_types: vec![Type::Int],
        },
        Err(_) => TraceAction::Abort,
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
        self.call_callable_value(callable, args)
    }

    fn build_list(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        self.trace_build_list(items)
    }

    fn build_tuple(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        self.trace_build_tuple(items)
    }

    fn build_map(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        self.trace_build_map(items)
    }

    fn store_subscr(
        &mut self,
        obj: Self::Value,
        key: Self::Value,
        value: Self::Value,
    ) -> Result<(), PyError> {
        self.store_subscr_value(obj, key, value)
    }

    fn list_append(&mut self, list: Self::Value, value: Self::Value) -> Result<(), PyError> {
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
        self.with_ctx(|this, ctx| TraceFrameState::load_local_value(this, ctx, idx))
    }

    fn store_local_value(&mut self, idx: usize, value: Self::Value) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| TraceFrameState::store_local_value(this, ctx, idx, value))
    }
}

impl NamespaceOpcodeHandler for TraceFrameState {
    fn load_name_value(&mut self, name: &str) -> Result<Self::Value, PyError> {
        let Some(slot) = concrete_namespace_slot(self.concrete_frame, name) else {
            return self.trace_load_name(name);
        };
        if let Some(concrete_value) = concrete_namespace_value(self.concrete_frame, slot) {
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
        let Some(slot) = concrete_namespace_slot(self.concrete_frame, name) else {
            return self.trace_store_name(name, value);
        };
        self.with_ctx(|this, ctx| TraceFrameState::store_namespace_value(this, ctx, slot, value))
    }

    fn null_value(&mut self) -> Result<Self::Value, PyError> {
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

    fn close_loop_args(&mut self, _target: usize) -> Result<Option<Vec<Self::Value>>, PyError> {
        self.with_ctx(|this, ctx| Ok(Some(TraceFrameState::close_loop_args(this, ctx))))
    }
}

impl BranchOpcodeHandler for TraceFrameState {
    fn concrete_truth_as_bool(&mut self, _truth: Self::Truth) -> Result<bool, PyError> {
        TraceFrameState::concrete_branch_truth(self)
    }

    fn guard_truth_value(&mut self, truth: Self::Truth, expect_true: bool) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            TraceFrameState::record_branch_guard(this, ctx, truth, expect_true);
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
        if matches!(op, BinaryOperator::Subscr) {
            self.binary_subscr_value(a, b)
        } else if self.concrete_binary_float_operands() {
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
        self.compare_value_direct(a, b, op)
    }

    fn unary_negative_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        self.unary_int_value(value, OpCode::IntNeg)
    }

    fn unary_invert_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        self.unary_int_value(value, OpCode::IntInvert)
    }
}

impl ConstantOpcodeHandler for TraceFrameState {
    fn int_constant(&mut self, value: i64) -> Result<Self::Value, PyError> {
        self.trace_int_constant(value)
    }

    fn bigint_constant(&mut self, value: &PyBigInt) -> Result<Self::Value, PyError> {
        self.trace_bigint_constant(value)
    }

    fn float_constant(&mut self, value: f64) -> Result<Self::Value, PyError> {
        self.trace_float_constant(value)
    }

    fn bool_constant(&mut self, value: bool) -> Result<Self::Value, PyError> {
        self.trace_bool_constant(value)
    }

    fn str_constant(&mut self, value: &str) -> Result<Self::Value, PyError> {
        self.trace_str_constant(value)
    }

    fn code_constant(&mut self, code: &CodeObject) -> Result<Self::Value, PyError> {
        self.trace_code_constant(code)
    }

    fn none_constant(&mut self) -> Result<Self::Value, PyError> {
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
                    let _ = self.set_local_at(local_idx, values[idx] as pyre_object::PyObjectRef);
                }
                idx += 1;
            }
            for i in 0..stack_only {
                if idx < values.len() {
                    let _ = self.set_stack_at(i, values[idx] as pyre_object::PyObjectRef);
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
                let _ = self.set_local_at(i, values[idx] as PyObjectRef);
                idx += 1;
            }
        }

        // Stack values follow locals (indices nlocals..valuestackdepth)
        let stack_only = self.valuestackdepth.saturating_sub(nlocals);
        for i in 0..stack_only {
            if idx < values.len() {
                let _ = self.set_stack_at(i, values[idx] as PyObjectRef);
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

    fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {
        PyreMeta {
            merge_pc: self.next_instr,
            num_locals: self.local_count(),
            ns_keys: self.namespace_keys(),
            valuestackdepth: self.valuestackdepth,
            has_virtualizable: self.has_virtualizable_info(),
        }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        let nlocals = self.local_count();
        let stack_only = self.valuestackdepth.saturating_sub(nlocals);
        let mut vals = vec![
            self.frame as i64,
            self.next_instr as i64,
            self.valuestackdepth as i64,
        ];
        for i in 0..nlocals {
            vals.push(self.local_at(i).unwrap_or(0 as PyObjectRef) as i64);
        }
        for i in 0..stack_only {
            vals.push(self.stack_at(i).unwrap_or(0 as PyObjectRef) as i64);
        }
        vals
    }

    fn live_value_types(&self, meta: &Self::Meta) -> Vec<Type> {
        let stack_only = meta.valuestackdepth.saturating_sub(meta.num_locals);
        fail_arg_types_for_virtualizable_state(3 + meta.num_locals + stack_only)
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        let mut sym = PyreSym::new_uninit(OpRef(0));
        sym.vable_next_instr = OpRef(1);
        sym.vable_valuestackdepth = OpRef(2);
        // Unified array: locals at base+0..base+nlocals, stack at base+nlocals..
        sym.vable_array_base = Some(3); // starts after frame(0), ni(1), vsd(2)
        sym
    }

    fn driver_descriptor(&self, _meta: &Self::Meta) -> Option<JitDriverStaticData> {
        None
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
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
                arg0,
                meta.merge_pc,
                meta.valuestackdepth,
                meta.has_virtualizable,
                values
            );
        }
        if values.len() == 1 {
            let _ = self.refresh_from_frame();
            return;
        }

        if meta.has_virtualizable {
            // Virtualizable exits carry the current frame state, not the
            // merge-point state from `meta`.
            //
            // In particular, GuardNotForced from function-entry traces resumes
            // the CALL bytecode with a deeper operand stack than
            // `meta.valuestackdepth`.  Using the merge-point depth here drops
            // callable/arg slots and resumes the interpreter with a corrupt
            // stack.  RPython restores from the payload itself.
            let ints: Vec<i64> = values
                .iter()
                .map(|v| match v {
                    Value::Int(i) => *i,
                    Value::Ref(r) => r.as_usize() as i64,
                    _ => 0,
                })
                .collect();
            self.restore_virtualizable_i64(&ints);
        } else {
            let nlocals = self.local_count();
            let stack_only_depth = meta.valuestackdepth.saturating_sub(nlocals);
            let mut idx = 1;
            for local_idx in 0..nlocals {
                let _ = self.set_local_at(local_idx, value_to_ptr(&values[idx]));
                idx += 1;
            }
            for i in 0..stack_only_depth {
                let _ = self.set_stack_at(i, value_to_ptr(&values[idx]));
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
                arg0,
                self.next_instr,
                self.valuestackdepth
            );
        }
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
        // [frame, next_instr, valuestackdepth, locals..., stack...]
        let stack_only = meta.valuestackdepth.saturating_sub(meta.num_locals);
        let expected = 3 + meta.num_locals + stack_only;
        jump_args.len() == expected
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
                    std::slice::from_raw_parts(
                        list_items_ptr as *const PyObjectRef,
                        list_items_len,
                    )
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
    use pyre_object::floatobject::w_float_get_value;
    use pyre_object::listobject::w_list_getitem;

    fn empty_meta() -> PyreMeta {
        PyreMeta {
            merge_pc: 0,
            num_locals: 0,
            ns_keys: Vec::new(),
            valuestackdepth: 0,
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
            assert_eq!(w_int_get_value(w_list_getitem(list_ptr.0 as PyObjectRef, 0).unwrap()), 2);
            assert_eq!(w_int_get_value(w_list_getitem(list_ptr.0 as PyObjectRef, 1).unwrap()), 4);
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
) -> Result<(OpRef, PyObjectRef), pyre_runtime::PyError> {
    use pyre_runtime::StepResult;
    let _inline_call_override = pyre_interp::call::inline_call_override_guard();
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
                            return Ok((result_opref, concrete_result));
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
                        let parent = framestack.last_mut().unwrap();
                        let result_idx = popped
                            .caller_result_stack_idx
                            .expect("inline child frame lost caller result slot");
                        let slot = parent
                            .sym
                            .symbolic_stack
                            .get_mut(result_idx)
                            .expect("inline caller result slot out of bounds");
                        *slot = result_opref;
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
                return Err(pyre_runtime::PyError::type_error("inline trace aborted"));
            }
            InlineTraceStepAction::Trace(
                majit_meta::TraceAction::CloseLoop
                | majit_meta::TraceAction::CloseLoopWithArgs { .. },
            ) => {
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
                return Err(pyre_runtime::PyError::type_error(
                    "concrete return/yield without trace finish",
                ));
            }
            StepResult::CloseLoop(_) => {
                return Err(pyre_runtime::PyError::type_error(
                    "inline callee has loop (not supported)",
                ));
            }
        }
    }
}
