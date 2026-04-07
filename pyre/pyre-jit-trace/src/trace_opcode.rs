//! MIFrame opcode handlers for trace-time JIT.
//!
//! Contains all `impl MIFrame` methods and trait implementations
//! (SharedOpcodeHandler, LocalOpcodeHandler, etc.).

use crate::state::*;

use majit_ir::{DescrRef, GcRef, OpCode, OpRef, Type, Value};
use majit_metainterp::{TraceAction, TraceCtx};

use pyre_interpreter::bytecode::{BinaryOperator, CodeObject, ComparisonOperator, Instruction};

/// floatobject.py:561 `descr_pow` → `_pow(space, x, y)` parity.
///
/// `_pow` in floatobject.py:799-881 takes two raw floats and returns a
/// raw float (can raise OverflowError / ValueError / ZeroDivisionError).
/// The JIT trace records this as `CALL_F(float_pow_jit, lhs, rhs)`
/// (pyjitpl.py:2119-2121 CALL_F branch taken because
/// `check_forces_virtual_or_virtualizable()` is False for ll_math_pow,
/// and `exc=True` because EF_CAN_RAISE), followed by `GUARD_NO_EXCEPTION`
/// via `handle_possible_exception` (pyjitpl.py:1950-1955, 3395).
///
/// ll_math_pow (ll_math.py:260) is the can-raise helper (EF_CAN_RAISE),
/// NOT elidable and NOT force-virtual. Using Rust's native `x.powf(y)`
/// would drop the Python exception semantics (negative base fractional
/// exponent → ValueError, 0.0 raised to negative → ZeroDivisionError,
/// overflow → OverflowError). Using CALL_MAY_FORCE_F would be wrong
/// because the optimizer postpones that family until GUARD_NOT_FORCED
/// arrives (heap.py CALL_MAY_FORCE branch), which is the virtualizable
/// protocol — ll_math_pow does not touch virtualizables.
///
/// Extracted to module level for stable function pointer identity.
///
/// Must match `float_pow_impl` semantics in `baseobjspace.rs`: any
/// divergence would cause the JIT compiled code to produce a different
/// result from the interpreter for the same input (correctness bug).
extern "C" fn float_pow_jit(x: f64, y: f64) -> f64 {
    match pyre_interpreter::float_pow_raw(x, y) {
        Ok(z) => z,
        Err(err) => {
            // llmodel.py:194-199 _store_exception parity: set JIT exception
            // state so the following GuardNoException sees it and fails,
            // propagating the raise into the meta-interpreter.
            let exc_obj = err.to_exc_object();
            majit_backend_cranelift::jit_exc_raise(exc_obj as i64);
            // Return value is discarded by GuardNoException path; use NaN
            // as a safe sentinel in case the guard is elided.
            f64::NAN
        }
    }
}
use pyre_interpreter::truth_value as objspace_truth_value;
use pyre_interpreter::{
    ArithmeticOpcodeHandler, BranchOpcodeHandler, ConstantOpcodeHandler, ControlFlowOpcodeHandler,
    IterOpcodeHandler, LocalOpcodeHandler, NamespaceOpcodeHandler, OpcodeStepExecutor, PyBigInt,
    PyError, PyNamespace, SharedOpcodeHandler, StackOpcodeHandler, TruthOpcodeHandler,
    builtin_code_name, decode_instruction_at, execute_opcode_step, function_get_code,
    function_get_globals, is_builtin_code, is_function, range_iter_continues,
};

use pyre_object::PyObjectRef;
use pyre_object::boolobject::w_bool_get_value;
use pyre_object::listobject::w_list_getitem;
use pyre_object::pyobject::{
    BOOL_TYPE, DICT_TYPE, FLOAT_TYPE, INT_TYPE, LIST_TYPE, NONE_TYPE, PyType, TUPLE_TYPE, is_bool,
    is_dict, is_float, is_int, is_list, is_none, is_tuple,
};
use pyre_object::rangeobject::RANGE_ITER_TYPE;
use pyre_object::strobject::is_str;
use pyre_object::tupleobject::w_tuple_getitem;
use pyre_object::{
    PY_NULL, w_bool_from, w_float_get_value, w_int_get_value, w_int_new,
    w_list_can_append_without_realloc, w_list_is_inline_storage, w_list_len, w_list_new,
    w_list_uses_float_storage, w_list_uses_int_storage, w_list_uses_object_storage,
    w_str_get_value, w_tuple_len,
};

use crate::descr::{
    bool_boolval_descr, dict_len_descr, float_floatval_descr, int_intval_descr,
    list_float_items_heap_cap_descr, list_float_items_len_descr, list_float_items_ptr_descr,
    list_int_items_heap_cap_descr, list_int_items_len_descr, list_int_items_ptr_descr,
    list_items_heap_cap_descr, list_items_len_descr, list_items_ptr_descr, list_strategy_descr,
    namespace_values_len_descr, namespace_values_ptr_descr, ob_type_descr,
    range_iter_current_descr, range_iter_step_descr, range_iter_stop_descr, str_len_descr,
    tuple_items_len_descr, tuple_items_ptr_descr, w_float_size_descr, w_int_size_descr,
};
use crate::frame_layout::PYFRAME_CODE_OFFSET;
use crate::helpers::{TraceHelperAccess, emit_box_float_inline, emit_trace_bool_value_from_truth};
use crate::liveness::liveness_for;

impl MIFrame {
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
        None
    }

    pub(crate) fn next_instruction_consumes_comparison_truth(&self) -> bool {
        let code = unsafe { &*(*self.sym().jitcode).raw_code() };
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

    pub fn from_sym(
        ctx: &mut TraceCtx,
        sym: &mut PyreSym,
        concrete_frame: usize,
        fallthrough_pc: usize,
        opcode_start_pc: usize,
    ) -> Self {
        sym.init_symbolic(ctx, concrete_frame);
        // RPython pyjitpl.py: orgpc = opcode start PC passed to each handler.
        let orgpc = opcode_start_pc;
        Self {
            ctx,
            sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc,
            concrete_frame_addr: concrete_frame,
            orgpc,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
        }
    }

    pub(crate) fn ctx(&mut self) -> &mut TraceCtx {
        unsafe { &mut *self.ctx }
    }

    pub(crate) fn with_ctx<R>(&mut self, f: impl FnOnce(&mut Self, &mut TraceCtx) -> R) -> R {
        let ctx = self.ctx;
        unsafe { f(self, &mut *ctx) }
    }

    #[inline]
    pub(crate) fn sym(&self) -> &PyreSym {
        unsafe { &*self.sym }
    }

    #[inline]
    pub(crate) fn sym_mut(&mut self) -> &mut PyreSym {
        unsafe { &mut *self.sym }
    }

    pub(crate) fn frame(&self) -> OpRef {
        self.sym().frame
    }

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
        let concrete_value = self.concrete_at(abs_idx).unwrap_or(PY_NULL);
        let typed_value = extract_concrete_typed_value(slot_type, concrete_value);
        fail_arg_opref_for_typed_value(ctx, typed_value)
    }

    /// pyjitpl.py:177 get_list_of_active_boxes parity.
    /// Returns compact register boxes for live registers only.
    ///
    /// RPython: both capture (here) and resume (consume_one_section,
    /// resume.py:1381) use the SAME `all_liveness` data, iterating
    /// via LivenessIterator over the same register indices in the
    /// same order. In pyre, both use JitCode.liveness (LivenessInfo),
    /// which now has precise per-local liveness from LiveVars::compute
    /// (RPython liveness.py backward analysis parity).
    fn get_list_of_active_boxes(
        &mut self,
        _ctx: &mut TraceCtx,
        in_a_call: bool,
        after_residual_call: bool,
    ) -> Vec<OpRef> {
        let (nlocals, local_values, stack_values, majit_jitcode) = {
            let s = self.sym();
            let stack_values = if let Some(ref pre_stack) = s.pre_opcode_stack {
                pre_stack.clone()
            } else {
                let stack_only = s.stack_only_depth();
                s.symbolic_stack[..stack_only.min(s.symbolic_stack.len())].to_vec()
            };
            let mjc = unsafe { (*s.jitcode).majit_jitcode };
            (s.nlocals, s.symbolic_locals.clone(), stack_values, mjc)
        };
        // pyjitpl.py:194: in_a_call or after_residual_call → self.pc
        let live_pc = if in_a_call || after_residual_call {
            self.fallthrough_pc
        } else {
            self.orgpc
        };
        // pyjitpl.py:202-203: look up liveness from JitCode (same data
        // that consume_one_section uses via bh.get_current_position_info).
        let liveness_info = if !majit_jitcode.is_null() {
            let jc = unsafe { &*majit_jitcode };
            jc.py_to_jit_pc
                .get(live_pc)
                .and_then(|&jit_pc| jc.liveness.iter().find(|info| info.pc as usize == jit_pc))
        } else {
            None
        };
        if let Some(info) = liveness_info.filter(|i| {
            !i.live_r_regs.is_empty() || !i.live_i_regs.is_empty() || !i.live_f_regs.is_empty()
        }) {
            // pyjitpl.py:216-233: iterate all three register banks in order.
            // RPython: length_i + length_r + length_f = total live boxes.
            // pyre: all Python values are PyObjectRef (GCREF), so live_i_regs
            // and live_f_regs are normally empty. Iterate all three banks for
            // structural parity with RPython's get_list_of_active_boxes.
            let total = info.live_i_regs.len() + info.live_r_regs.len() + info.live_f_regs.len();
            let mut boxes = Vec::with_capacity(total);
            // pyjitpl.py:216-221: int bank (registers_i)
            for &reg_idx in &info.live_i_regs {
                let idx = reg_idx as usize;
                let val = if idx < nlocals {
                    local_values.get(idx).copied().unwrap_or(OpRef::NONE)
                } else {
                    stack_values
                        .get(idx - nlocals)
                        .copied()
                        .unwrap_or(OpRef::NONE)
                };
                boxes.push(val);
            }
            // pyjitpl.py:222-227: ref bank (registers_r)
            for &reg_idx in &info.live_r_regs {
                let idx = reg_idx as usize;
                let val = if idx < nlocals {
                    local_values.get(idx).copied().unwrap_or(OpRef::NONE)
                } else {
                    stack_values
                        .get(idx - nlocals)
                        .copied()
                        .unwrap_or(OpRef::NONE)
                };
                boxes.push(val);
            }
            // pyjitpl.py:228-233: float bank (registers_f)
            for &reg_idx in &info.live_f_regs {
                let idx = reg_idx as usize;
                let val = if idx < nlocals {
                    local_values.get(idx).copied().unwrap_or(OpRef::NONE)
                } else {
                    stack_values
                        .get(idx - nlocals)
                        .copied()
                        .unwrap_or(OpRef::NONE)
                };
                boxes.push(val);
            }
            boxes
        } else {
            // pyjitpl.py:177-234 parity: RPython's get_list_of_active_boxes
            // NEVER returns empty for a guard inside a loop — the codewriter
            // guarantees liveness at every guard position. When majit JitCode
            // liveness is empty or unavailable, fall back to LiveVars.
            let raw_code_ptr = unsafe { (*self.sym().jitcode).raw_code() };
            let live = if !raw_code_ptr.is_null() {
                Some(liveness_for(raw_code_ptr))
            } else {
                None
            };
            let mut boxes = Vec::with_capacity(nlocals + stack_values.len());
            for (idx, slot) in local_values.iter().enumerate() {
                let is_live = live.map_or(!slot.is_none(), |lv| lv.is_local_live(live_pc, idx));
                if is_live {
                    boxes.push(*slot);
                }
            }
            for (idx, slot) in stack_values.iter().enumerate() {
                let is_live = live.map_or(!slot.is_none(), |lv| lv.is_stack_live(live_pc, idx));
                if is_live {
                    boxes.push(*slot);
                }
            }
            boxes
        }
    }

    /// RPython Box.type parity: build fail_arg_types matching compact
    /// active_boxes length. Each box carries its own immutable type.
    /// header = [Ref, Int, Int] (frame, next_instr, valuestackdepth).
    fn build_fail_arg_types_for_active_boxes(&self, active_boxes: &[OpRef]) -> Vec<Type> {
        let mut types = crate::virtualizable_gen::virt_live_value_types(0);
        for &opref in active_boxes {
            types.push(self.value_type(opref));
        }
        types
    }

    fn remember_value_type(&mut self, value: OpRef, value_type: Type) {
        if value.is_none() {
            return;
        }
        self.sym_mut()
            .transient_value_types
            .insert(value, value_type);
    }

    pub(crate) fn value_type(&self, value: OpRef) -> Type {
        if value.is_none() {
            return Type::Ref;
        }
        self.sym().value_type_of(value)
    }

    /// RPython Box push: symbolic OpRef + concrete value together.
    fn push_typed_value(
        &mut self,
        _ctx: &mut TraceCtx,
        value: OpRef,
        value_type: Type,
        concrete: ConcreteValue,
    ) {
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
        if !value.is_none() {
            s.transient_value_types.insert(value, value_type);
        }
        if stack_idx >= s.concrete_stack.len() {
            s.concrete_stack.resize(stack_idx + 1, ConcreteValue::Null);
        }
        s.concrete_stack[stack_idx] = concrete;
        s.valuestackdepth += 1;
    }

    pub(crate) fn push_value(
        &mut self,
        _ctx: &mut TraceCtx,
        value: OpRef,
        concrete: ConcreteValue,
    ) {
        let value_type = self.value_type(value);
        self.push_typed_value(_ctx, value, value_type, concrete);
    }

    pub(crate) fn pop_value(&mut self, ctx: &mut TraceCtx) -> Result<OpRef, PyError> {
        let s = self.sym_mut();
        let nlocals = s.nlocals;
        let stack_idx = s
            .valuestackdepth
            .checked_sub(nlocals + 1)
            .ok_or_else(|| pyre_interpreter::stack_underflow_error("trace opcode"))?;
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
        s.valuestackdepth -= 1;
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
            .ok_or_else(|| pyre_interpreter::stack_underflow_error("trace peek"))?;
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
        self.push_value(ctx, callable, ConcreteValue::Null);
        self.push_value(ctx, null, ConcreteValue::Ref(pyre_object::PY_NULL));
        for &arg in args {
            self.push_value(ctx, arg, ConcreteValue::Null);
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
        value: OpRef,
    ) -> Result<(), PyError> {
        // RPython pyjitpl.py opimpl_setlocal: stores the box directly.
        // Unboxing happens at operation time (binary_float_value, etc.),
        // not at store time. concrete_virtualizable_slot_type always
        // returns Type::Ref (pyre slots are GCREFs), so any
        // concrete-type-based unboxing here was dead code.
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
        // pyjitpl.py:1075-1089: quasi-immutable field pattern.
        // Module globals are effectively quasi-immutable — they rarely change
        // during hot loops. Emit GUARD_NOT_INVALIDATED on first global access
        // so compiled code is invalidated if globals mutate.
        if ctx.heap_cache_mut().check_and_clear_guard_not_invalidated() {
            self.record_guard(ctx, OpCode::GuardNotInvalidated, &[]);
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

    /// Set pending_next_instr for trace advancement (step_root_frame).
    /// This is the NEXT bytecode PC — used by the MetaInterp to advance.
    pub(crate) fn prepare_fallthrough(&mut self) {
        self.sym_mut().pending_next_instr = Some(self.fallthrough_pc);
    }

    /// Set the original PC for the current opcode (RPython orgpc).
    /// All guards within this opcode will use orgpc as their resume PC.
    pub(crate) fn set_orgpc(&mut self, pc: usize) {
        self.orgpc = pc;
    }

    /// Update virtualizable next_instr and valuestackdepth.
    /// RPython parity: always use orgpc (opcode start PC) for next_instr
    /// so gen_store_back_in_vable writes a real opcode PC to the frame,
    /// not a Cache PC. The trace loop advancement uses pending_next_instr
    /// separately (in metainterp.rs step_*_frame).
    pub(crate) fn flush_to_frame(&mut self, ctx: &mut TraceCtx) {
        let resume_pc = self.orgpc;
        let frame_addr = self.concrete_frame_addr;
        let code_ptr = if frame_addr != 0 {
            unsafe { *((frame_addr + PYFRAME_CODE_OFFSET) as *const usize) }
        } else {
            0
        };
        let ns_ptr = self.sym().concrete_namespace as i64;
        let vsd = self.sym().valuestackdepth as i64;
        self.sym_mut()
            .flush_vable_fields(ctx, &[resume_pc as i64, code_ptr as i64, vsd, ns_ptr]);
    }

    /// capture_resumedata(resumepc=orgpc) parity: flush vable fields for guards.
    ///
    /// When pre_opcode_vsd is set, sets vable_next_instr = orgpc and
    /// vable_valuestackdepth = pre-opcode depth. The guard's fail_args
    /// then carry the pre-opcode stack state so the blackhole interpreter
    /// can re-execute the opcode from orgpc.
    ///
    /// Note: `record_branch_guard` does NOT call this — branch guards
    /// build their own fail_args with post-pop state and other_target PC
    /// (see the comment there for why).
    fn flush_to_frame_for_guard(&mut self, ctx: &mut TraceCtx) {
        // RPython capture_resumedata(resumepc=orgpc) parity:
        // Always use orgpc (opcode start PC) as the resume PC.
        // orgpc is set to the current opcode's code unit in from_sym().
        let resume_pc = self.orgpc;
        let frame_addr = self.concrete_frame_addr;
        let code_ptr = if frame_addr != 0 {
            unsafe { *((frame_addr + PYFRAME_CODE_OFFSET) as *const usize) }
        } else {
            0
        };
        let ns_ptr = self.sym().concrete_namespace as i64;
        let vsd = {
            let s = self.sym();
            s.pre_opcode_vsd.unwrap_or(s.valuestackdepth) as i64
        };
        self.sym_mut()
            .flush_vable_fields(ctx, &[resume_pc as i64, code_ptr as i64, vsd, ns_ptr]);
    }

    fn sync_standard_virtualizable_before_residual_call(&mut self, ctx: &mut TraceCtx) {
        let Some(vable_ref) = ctx.standard_virtualizable_box() else {
            return;
        };
        ctx.gen_store_back_in_vable(vable_ref);
        let info = crate::virtualizable_gen::build_virtualizable_info();
        let obj_ptr = self.sym().concrete_vable_ptr;
        unsafe {
            info.tracing_before_residual_call(obj_ptr);
        }
        let force_token = ctx.force_token();
        ctx.vable_setfield_descr(vable_ref, force_token, info.token_field_descr());
        // pyjitpl.py:2575 parity: mark all active virtual references
        // as "in residual call" (TOKEN_TRACING_RESCALL) so that if the
        // callee forces a vref, we detect it after the call.
        self.virtualref_before_residual_call();
    }

    fn sync_standard_virtualizable_after_residual_call(&mut self, ctx: &mut TraceCtx) -> bool {
        let info = crate::virtualizable_gen::build_virtualizable_info();
        let obj_ptr = self.sym().concrete_vable_ptr;
        let vable_forced = unsafe { info.tracing_after_residual_call(obj_ptr) };
        // pyjitpl.py:3400 parity: check if any virtualref was forced
        // during the residual call.
        self.virtualref_after_residual_call(ctx);
        vable_forced
    }

    /// pyjitpl.py:3317-3337 parity: before residual call, set all
    /// active virtualref tokens to TOKEN_TRACING_RESCALL.
    fn virtualref_before_residual_call(&self) {
        let vref_boxes = &self.sym().virtualref_boxes;
        if vref_boxes.is_empty() {
            return;
        }
        // pyjitpl.py:3339: for each pair, call tracing_before on the
        // ODD slot (vrefbox = second element), not the virtual (first).
        let vref_info = majit_metainterp::virtualref::VirtualRefInfo::new();
        // virtualref_boxes = [(virt_sym, virt_ptr), (vref_sym, vref_ptr), ...]
        for pair in vref_boxes.chunks(2) {
            if let Some(&(_vref_sym, vref_ptr)) = pair.get(1) {
                if vref_ptr != 0 {
                    unsafe {
                        vref_info.tracing_before_residual_call(vref_ptr as *mut u8);
                    }
                }
            }
        }
    }

    /// pyjitpl.py:3337-3347 vrefs_after_residual_call parity:
    /// after residual call, check if any virtualref was forced.
    /// If forced, call stop_tracking_virtualref(i) to record
    /// VIRTUAL_REF_FINISH and replace odd slot with CONST_NULL.
    fn virtualref_after_residual_call(&mut self, ctx: &mut TraceCtx) {
        let sym = unsafe { &mut *self.sym };
        if sym.virtualref_boxes.is_empty() {
            return;
        }
        let vref_info = majit_metainterp::virtualref::VirtualRefInfo::new();
        let len = sym.virtualref_boxes.len();
        let mut i = 0;
        while i < len {
            let (vref_opref, vref_ptr) = sym.virtualref_boxes[i + 1];
            if vref_ptr != 0 {
                let forced = unsafe {
                    vref_info.tracing_after_residual_call(
                        vref_ptr as *mut u8,
                        majit_metainterp::virtualref::TOKEN_TRACING_RESCALL,
                    )
                };
                if forced {
                    // pyjitpl.py:3371-3378: stop_tracking_virtualref(i)
                    let virt_opref = sym.virtualref_boxes[i].0;
                    // record VIRTUAL_REF_FINISH(vrefbox, virtualbox)
                    let _ = ctx.record_op(OpCode::VirtualRefFinish, &[vref_opref, virt_opref]);
                    // virtualref_boxes[i+1] = CONST_NULL
                    let null_opref = ctx.const_int(0);
                    sym.virtualref_boxes[i + 1] = (null_opref, 0);
                }
            }
            i += 2;
        }
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
                    crate::generated::trace_box_int(
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
        self.close_loop_args_at(ctx, None)
    }

    pub(crate) fn close_loop_args_at(
        &mut self,
        ctx: &mut TraceCtx,
        target_pc: Option<usize>,
    ) -> Vec<OpRef> {
        // RPython parity: loop-carried live state comes from the current
        // virtualizable frame state at the merge point. If symbolic stack
        // accounting drifted during tracing, resync depth/shape from the
        // concrete frame before materializing JUMP args.
        // MIFrame Box tracking: use PyreSym's tracked values, not snapshot.
        let concrete_nlocals = self.sym().nlocals;
        let concrete_vsd = self.sym().valuestackdepth.max(concrete_nlocals);
        {
            let s = self.sym_mut();
            s.nlocals = concrete_nlocals;
            s.valuestackdepth = concrete_vsd;
            let stack_only = s.stack_only_depth();
            // All slot types forced to Ref (PyObjectRef).
            if s.symbolic_local_types.len() != concrete_nlocals {
                s.symbolic_local_types = vec![Type::Ref; concrete_nlocals];
            }
            if s.symbolic_stack_types.len() != stack_only {
                s.symbolic_stack_types = vec![Type::Ref; stack_only];
            }
            if s.symbolic_stack.len() < stack_only {
                s.symbolic_stack.resize(stack_only, OpRef::NONE);
            }
        }
        self.flush_to_frame(ctx);
        // pyjitpl.py:2973: at a merge point, next_instr should be the TARGET
        // PC, not the last bytecode's orgpc. flush_to_frame sets
        // vable_next_instr from orgpc; override it here.
        if let Some(pc) = target_pc {
            self.sym_mut().vable_next_instr = ctx.const_int(pc as i64);
        }
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
        // RPython close_loop_args parity: JUMP args must match the target
        // label's types (inputarg_types). materialize_loop_carried_value
        // boxes values to match (e.g. Int → Ref for virtualizable locals).
        //
        // For bridge traces, ctx.inputarg_types() returns the bridge's
        // guard fail_arg types, NOT the root loop's label types. The JUMP
        // targets the root loop label, so we must use the root loop token's
        // inputarg_types instead.
        let inputarg_types = {
            let (driver, _) = crate::driver::driver_pair();
            if driver.is_bridge_tracing() {
                if let Some(gk) = driver.current_trace_green_key() {
                    driver
                        .get_loop_token(gk)
                        .map(|token| token.inputarg_types.clone())
                        .unwrap_or_else(|| ctx.inputarg_types())
                } else {
                    ctx.inputarg_types()
                }
            } else {
                ctx.inputarg_types()
            }
        };
        let mut args = vec![frame, next_instr, stack_depth];
        for (idx, value) in locals.into_iter().enumerate() {
            let target_type = inputarg_types
                .get(crate::virtualizable_gen::NUM_SCALAR_INPUTARGS + idx)
                .copied()
                .unwrap_or(Type::Ref);
            // Materialize NONE slots from concrete frame before boxing.
            // RPython's live_arg_boxes never contains holes at loop closure
            // because MIFrame.run_one_step always updates all live registers.
            let value = self.materialize_fail_arg_slot(ctx, value, target_type, idx);
            args.push(self.materialize_loop_carried_value(ctx, value, target_type));
        }
        for (stack_idx, value) in stack.into_iter().enumerate() {
            let target_type = inputarg_types
                .get(crate::virtualizable_gen::NUM_SCALAR_INPUTARGS + nlocals + stack_idx)
                .copied()
                .unwrap_or(Type::Ref);
            let value =
                self.materialize_fail_arg_slot(ctx, value, target_type, nlocals + stack_idx);
            args.push(self.materialize_loop_carried_value(ctx, value, target_type));
        }
        // pyjitpl.py:2934-2940 remove_consts_and_duplicates:
        // Replace constant or duplicate OpRefs with SameAs to give each
        // JUMP arg slot a unique identity. The optimizer's unroll pass
        // needs distinct identities to track values independently.
        {
            use std::collections::HashSet;
            let header_len = 3; // [frame, next_instr, valuestackdepth]
            let mut seen = HashSet::new();
            for i in header_len..args.len() {
                let opref = args[i];
                if opref.is_constant() || !seen.insert(opref) {
                    let tp = inputarg_types
                        .get(i)
                        .copied()
                        .unwrap_or(majit_ir::Type::Ref);
                    let same_as_op = majit_ir::OpCode::same_as_for_type(tp);
                    args[i] = ctx.record_op(same_as_op, &[opref]);
                }
            }
        }
        // pyjitpl.py:2967-2969: generate a dummy GUARD_FUTURE_CONDITION
        // just before the JUMP so that unroll can use it when it's
        // creating artificial guards (patchguardop). record_guard calls
        // capture_resumedata which captures the full framestack +
        // virtualizable_boxes + virtualref_boxes.
        //
        // RPython only emits GUARD_FUTURE_CONDITION here. GUARD_NOT_INVALIDATED
        // is *not* unconditionally emitted before JUMP — pyjitpl.py:1086-1089
        // emits it only inside `opimpl_record_quasi_immutable_field`, after a
        // quasi-immut field read sets `heapcache.need_guard_not_invalidated`.
        // The pyre frontend does the same via `flush_guard_not_invalidated`,
        // so an unconditional emit here would (a) leak resume data for traces
        // that have no quasi-immut dep at all, and (b) leave a runtime guard
        // whose flag is decoupled from any watcher, which can spuriously
        // exit a hot inner loop with no chance of re-tracing.
        self.record_guard(ctx, majit_ir::OpCode::GuardFutureCondition, &[]);
        args
    }

    /// pyjitpl.py:2586 capture_resumedata: build fail_args for CURRENT
    /// top frame. Returns [frame, ni, vsd, active_boxes...] for Cranelift.
    /// RPython always captures the current frame; parent frames are handled
    /// separately via _ensure_parent_resumedata (opencoder.py:819).
    pub(crate) fn current_fail_args(&mut self, ctx: &mut TraceCtx) -> Vec<OpRef> {
        self.flush_to_frame_for_guard(ctx);
        let active_boxes = self.get_list_of_active_boxes(ctx, false, false);
        let s = self.sym();
        let mut fa = vec![s.frame, s.vable_next_instr, s.vable_valuestackdepth];
        fa.extend_from_slice(&active_boxes);
        fa
    }

    /// pyjitpl.py:1087 parity: after a field read that might have set the
    /// needs_guard_not_invalidated flag (quasi-immutable field), emit the
    /// guard with full snapshot via record_guard.
    fn flush_guard_not_invalidated(&mut self, ctx: &mut TraceCtx) {
        if let Some(saved_orgpc) = ctx.pending_guard_not_invalidated_pc() {
            ctx.set_pending_guard_not_invalidated(None);
            // pyjitpl.py:1087 parity: use the field read's orgpc so the
            // snapshot captures the correct liveness state. Also clear
            // pending_branch_value to prevent record_guard from diverting
            // to record_branch_guard (which would emit GuardTrue/False
            // instead of GuardNotInvalidated).
            let current_orgpc = self.orgpc;
            let saved_branch = self.sym_mut().pending_branch_value.take();
            let saved_other = self.sym_mut().pending_branch_other_target.take();
            self.orgpc = saved_orgpc;
            self.record_guard(ctx, OpCode::GuardNotInvalidated, &[]);
            self.orgpc = current_orgpc;
            self.sym_mut().pending_branch_value = saved_branch;
            self.sym_mut().pending_branch_other_target = saved_other;
        }
    }

    /// PyPy generate_guard + capture_resumedata: uses current_fail_args
    /// which encodes the full framestack for multi-frame resume.
    pub(crate) fn record_guard(&mut self, ctx: &mut TraceCtx, opcode: OpCode, args: &[OpRef]) {
        // pyjitpl.py:1087 parity: flush pending guard_not_invalidated
        // before recording any new guard (the quasi-immut guard should be
        // emitted with its own snapshot before the current guard).
        if opcode != OpCode::GuardNotInvalidated {
            self.flush_guard_not_invalidated(ctx);
        }
        // pyjitpl.py:2575-2578: determine after_residual_call from guard opcode.
        // opencoder.py:767: when true, all boxes in top frame are live
        // (liveness filter disabled for residual call guards).
        let after_residual_call = matches!(
            opcode,
            OpCode::GuardException
                | OpCode::GuardNoException
                | OpCode::GuardNotForced
                | OpCode::GuardAlwaysFails
        );
        // opencoder.py:819 capture_resumedata(framestack) parity:
        // Encode the full framestack [callee (top), caller (parent)] into
        // a multi-frame snapshot. The callee's pc is set to resumepc
        // (orgpc), while the caller keeps its original pc (return_point).
        // pyjitpl.py:2597 passes full framestack + vable/vref boxes.
        if !self.parent_frames.is_empty() {
            // pyjitpl.py:2593-2596: top frame pc = resumepc (orgpc)
            self.flush_to_frame_for_guard(ctx);
            // pyjitpl.py:177: active boxes = registers only (no header).
            let callee_active_boxes =
                self.get_list_of_active_boxes(ctx, false, after_residual_call);
            // RPython Box.type parity: snapshot types match the full
            // (un-filtered) active_boxes — constants are part of the
            // snapshot via TAGCONST.
            let callee_snapshot_types_full =
                self.build_fail_arg_types_for_active_boxes(&callee_active_boxes);

            // snapshot.pc must match the liveness PC used for active boxes
            // (get_list_of_active_boxes uses fallthrough_pc when after_residual_call).
            let callee_live_pc = if after_residual_call {
                self.fallthrough_pc
            } else {
                self.orgpc
            };
            // opencoder.py:819-834: snapshot uses active boxes (not fail_args).
            // callee_snapshot_types_full includes header [Ref, Int, Int];
            // snapshot needs only the active_boxes portion.
            let __n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
            let callee_snapshot_types = &callee_snapshot_types_full[__n..];
            let mut frames = vec![majit_trace::recorder::SnapshotFrame {
                jitcode_index: unsafe { (*self.sym().jitcode).index } as u32,
                pc: callee_live_pc as u32,
                boxes: Self::fail_args_to_snapshot_boxes_typed(
                    &callee_active_boxes,
                    callee_snapshot_types,
                    ctx,
                ),
            }];
            // pyjitpl.py:2558-2585 generate_guard parity: record guard with
            // a raw fail_args "hint" (header + active_boxes). The optimizer's
            // store_final_boxes_in_guard (mod.rs:finalize_guard_resume_data)
            // calls ResumeDataLoopMemo::finish() which rebuilds fail_args
            // from the snapshot via _number_boxes — Consts are dropped from
            // liveboxes during numbering (resume.py:202-207 getconst path),
            // so the post-optimizer fail_args naturally satisfy regalloc.py:1203.
            let mut fail_args: Vec<OpRef> = {
                let s = self.sym();
                vec![s.frame, s.vable_next_instr, s.vable_valuestackdepth]
            };
            fail_args.extend_from_slice(&callee_active_boxes);
            let mut types = callee_snapshot_types_full;
            // opencoder.py:806: parent frames keep their original pc.
            // Snapshot boxes = active boxes only (skip [frame, ni, vsd] header).
            for (pfa, pfa_types, pfa_resumepc, pfa_jitcode_index) in &self.parent_frames {
                let __n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
                let parent_active: &[OpRef] = if pfa.len() > __n { &pfa[__n..] } else { &[] };
                let parent_types: &[Type] = if pfa_types.len() > __n {
                    &pfa_types[__n..]
                } else {
                    &[]
                };
                frames.push(majit_trace::recorder::SnapshotFrame {
                    jitcode_index: *pfa_jitcode_index as u32,
                    pc: *pfa_resumepc as u32,
                    boxes: Self::fail_args_to_snapshot_boxes_typed(
                        parent_active,
                        parent_types,
                        ctx,
                    ),
                });
                fail_args.extend_from_slice(pfa);
                if !pfa_types.is_empty() {
                    types.extend_from_slice(pfa_types);
                } else {
                    let pt = fail_arg_types_for_virtualizable_state(pfa.len());
                    types.extend_from_slice(&pt);
                }
            }
            let vable_boxes = Self::build_virtualizable_boxes(self.sym(), ctx);
            // pyjitpl.py:2597: self.virtualref_boxes
            let vref_boxes = Self::build_virtualref_boxes(self.sym(), ctx);
            let snapshot = majit_trace::recorder::Snapshot {
                frames,
                vable_boxes,
                vref_boxes,
            };
            let snapshot_id = ctx.capture_resumedata(snapshot);

            ctx.record_guard_typed_with_fail_args(opcode, args, types, &fail_args);
            ctx.set_last_guard_resume_position(snapshot_id);
            return;
        }

        if let Some(branch_value) = self.sym().pending_branch_value {
            let truth = args.first().copied().unwrap_or(OpRef::NONE);
            let concrete_truth = opcode == OpCode::GuardTrue;
            self.record_branch_guard(ctx, branch_value, truth, concrete_truth);
            return;
        }

        // pyjitpl.py:2586-2596 capture_resumedata(resumepc) parity:
        // Normal guards: resumepc = orgpc (re-execute the opcode from start).
        // after_residual_call guards (GUARD_NOT_FORCED, GUARD_NO_EXCEPTION):
        //   RPython generate_guard passes resumepc=-1, and capture_resumedata
        //   skips the "frame.pc = resumepc" assignment — frame.pc stays at
        //   the auto-advanced next instruction (pyre fallthrough_pc equivalent).
        //   This ensures the liveness PC, header ni, and blackhole resume PC
        //   all point to the instruction AFTER the call, not the call itself.
        let resume_pc = if after_residual_call {
            self.fallthrough_pc
        } else {
            self.orgpc
        };
        self.record_guard_core(ctx, opcode, args, resume_pc, after_residual_call);
    }

    /// Core guard recording with explicit resume PC.
    ///
    /// pyjitpl.py:2586-2602 capture_resumedata parity: temporarily
    /// substitute frame.pc = resume_pc before snapshot capture.
    /// Used by both record_guard (resume_pc=orgpc) and
    /// record_branch_guard (resume_pc=other_target).
    fn record_guard_core(
        &mut self,
        ctx: &mut TraceCtx,
        opcode: OpCode,
        args: &[OpRef],
        resume_pc: usize,
        after_residual_call: bool,
    ) {
        // pyjitpl.py:2594-2596: saved_pc = frame.pc; frame.pc = resumepc
        let saved_orgpc = self.orgpc;
        let saved_ni = self.sym().vable_next_instr;
        let saved_vsd = self.sym().vable_valuestackdepth;
        self.orgpc = resume_pc;

        self.flush_to_frame_for_guard(ctx);
        // pyjitpl.py:177: active boxes = registers only (no header).
        let active_boxes = self.get_list_of_active_boxes(ctx, false, after_residual_call);
        // RPython Box.type parity: snapshot types match the full (un-filtered)
        // active_boxes — snapshot encodes Consts as TAGCONST.
        let snapshot_full_types = self.build_fail_arg_types_for_active_boxes(&active_boxes);
        // pyjitpl.py:2558-2585 generate_guard parity: record a raw fail_args
        // "hint" (header + active_boxes including Consts). The optimizer's
        // store_final_boxes_in_guard rebuilds fail_args from the snapshot
        // via ResumeDataLoopMemo::finish() → _number_boxes, which drops
        // Consts from liveboxes (resume.py:202-207 getconst path). This
        // satisfies regalloc.py:1203 structurally rather than via an
        // ad-hoc tracer-level filter.
        let fail_arg_types = snapshot_full_types.clone();
        let fail_args: Vec<OpRef> = {
            let s = self.sym();
            let mut fa = vec![s.frame, s.vable_next_instr, s.vable_valuestackdepth];
            fa.extend_from_slice(&active_boxes);
            fa
        };

        // The snapshot's frame.pc must match the liveness PC used by
        // get_list_of_active_boxes, so the blackhole decoder applies the
        // same bitvector. after_residual_call → fallthrough_pc, else orgpc.
        let snapshot_live_pc = if after_residual_call {
            self.fallthrough_pc
        } else {
            self.orgpc
        };

        // opencoder.py:767-770: snapshot uses active boxes (not fail_args).
        // snapshot_full_types includes [Ref, Int, Int] header; snapshot
        // needs only the active_boxes portion starting after NUM_SCALAR_INPUTARGS.
        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let snapshot_types = &snapshot_full_types[n..];
        let snapshot_boxes =
            Self::fail_args_to_snapshot_boxes_typed(&active_boxes, snapshot_types, ctx);
        let vable_boxes = Self::build_virtualizable_boxes(self.sym(), ctx);
        let jitcode_index = unsafe { (*self.sym().jitcode).index } as u32;
        let snapshot = majit_trace::recorder::Snapshot {
            frames: vec![majit_trace::recorder::SnapshotFrame {
                jitcode_index,
                pc: snapshot_live_pc as u32,
                boxes: snapshot_boxes,
            }],
            vable_boxes,
            // pyjitpl.py:2597: self.virtualref_boxes
            vref_boxes: Self::build_virtualref_boxes(self.sym(), ctx),
        };
        let snapshot_id = ctx.capture_resumedata(snapshot);

        ctx.record_guard_typed_with_fail_args(opcode, args, fail_arg_types, &fail_args);
        ctx.set_last_guard_resume_position(snapshot_id);

        // pyjitpl.py:2602: frame.pc = saved_pc (restore)
        self.orgpc = saved_orgpc;
        let s = self.sym_mut();
        s.vable_next_instr = saved_ni;
        s.vable_valuestackdepth = saved_vsd;
    }

    /// virtualizable.py:139 _get_virtualizable_field_boxes parity:
    /// [static_fields..., array_items..., virtualizable_ptr].
    /// pyjitpl.py:2586: self.virtualizable_boxes → vable_array.
    /// opencoder.py:603 _encode parity: encode OpRef as SnapshotTagged.
    /// Constant-pool OpRefs → Const(value, type) from pool.
    /// NONE → Const(0, Ref). Regular → Box.
    fn opref_to_snapshot_tagged(
        opref: OpRef,
        ctx: &majit_metainterp::TraceCtx,
    ) -> majit_trace::recorder::SnapshotTagged {
        if opref.is_none() {
            majit_trace::recorder::SnapshotTagged::Const(0, majit_ir::Type::Ref)
        } else if ctx.constant_value(opref).is_some() {
            let val = ctx.constant_value(opref).unwrap_or(0);
            let tp = ctx.const_type(opref).unwrap_or(majit_ir::Type::Int);
            majit_trace::recorder::SnapshotTagged::Const(val, tp)
        } else {
            // resume.py:211,214: box.type for _number_boxes TAGVIRTUAL/TAGBOX.
            let tp = ctx.get_opref_type(opref).unwrap_or(majit_ir::Type::Int);
            majit_trace::recorder::SnapshotTagged::Box(opref.0, tp)
        }
    }

    /// RPython pyjitpl.py:2586 virtualizable_boxes parity.
    ///
    /// RPython creates SEPARATE Box objects for virtualizable_boxes via
    /// read_boxes()/wrap() — these are distinct from frame register boxes.
    /// _number_boxes dedup uses object identity, so vable and frame get
    /// independent TAGBOX indices → deadframe stores both.
    ///
    /// pyre uses the SAME OpRefs for both → _number_boxes dedup merges them
    /// → vable and frame sections share TAGBOX indices. Recovery uses frame
    /// sections with liveness-based mapping (restore_guard_failure_values),
    /// matching RPython's consume_boxes(position_info) architecture.
    ///
    /// Fresh identity approaches (VABLE_FRESH_BIT, VABLE_KEY_OFFSET)
    /// expand num_boxes → larger fail_args → deadframe/exit layout mismatch.
    /// Fix requires backend exit block recompilation after numbering,
    /// or trace-time SameAs emission for fresh vable OpRefs.
    fn build_virtualizable_boxes(
        sym: &PyreSym,
        ctx: &majit_metainterp::TraceCtx,
    ) -> Vec<majit_trace::recorder::SnapshotTagged> {
        // opencoder.py:718-726 _list_of_boxes_virtualizable parity:
        // RPython format: [virtualizable_ptr, static_fields..., array_items...]
        // (virtualizable_ptr moved from end to front).
        // virtualizable.py:86/139 read_boxes / load_list_of_boxes:
        // Memory order: [static_field_0, ..., array_items..., vable_ptr]
        // read_boxes creates fresh Box objects for each field via wrap().
        // opencoder.py:722 _list_of_boxes_virtualizable: reorders
        //   vable_ptr from end to front → snapshot = [vable_ptr, fields..., items...]
        let stack_only = sym.stack_only_depth();
        let mut boxes = Vec::new();
        // opencoder.py:722: virtualizable_ptr FIRST.
        boxes.push(Self::opref_to_snapshot_tagged(sym.frame, ctx));
        // Static fields in declared order (virtualizable.py:90-93).
        for opref in sym.vable_field_oprefs() {
            boxes.push(Self::opref_to_snapshot_tagged(opref, ctx));
        }
        // Array items: locals + stack (virtualizable.py:86 read_boxes).
        let concrete_frame = if !sym.concrete_vable_ptr.is_null() {
            Some(unsafe { &*(sym.concrete_vable_ptr as *const pyre_interpreter::pyframe::PyFrame) })
        } else {
            None
        };
        let full_array_len = concrete_frame
            .map(|f| f.locals_cells_stack_w.len())
            .unwrap_or(sym.symbolic_locals.len() + stack_only);
        for i in 0..full_array_len {
            let opref = if i < sym.symbolic_locals.len() {
                sym.symbolic_locals[i]
            } else {
                let stack_idx = i - sym.nlocals;
                if stack_idx < stack_only && stack_idx < sym.symbolic_stack.len() {
                    sym.symbolic_stack[stack_idx]
                } else {
                    OpRef::NONE
                }
            };
            if !opref.is_none() {
                boxes.push(Self::opref_to_snapshot_tagged(opref, ctx));
            } else if let Some(frame) = concrete_frame {
                let val = frame
                    .locals_cells_stack_w
                    .as_slice()
                    .get(i)
                    .copied()
                    .unwrap_or(pyre_object::PY_NULL);
                boxes.push(majit_trace::recorder::SnapshotTagged::Const(
                    val as i64,
                    Type::Ref,
                ));
            } else {
                boxes.push(Self::opref_to_snapshot_tagged(OpRef::NONE, ctx));
            }
        }
        boxes
    }

    /// pyjitpl.py:2597 virtualref_boxes parity.
    /// pyjitpl.py:2597 virtualref_boxes parity.
    /// Returns pairs of (jit_virtual, real_vref) as SnapshotTagged.
    fn build_virtualref_boxes(
        sym: &PyreSym,
        ctx: &majit_metainterp::TraceCtx,
    ) -> Vec<majit_trace::recorder::SnapshotTagged> {
        sym.virtualref_boxes
            .iter()
            .map(|&(opref, _concrete)| Self::opref_to_snapshot_tagged(opref, ctx))
            .collect()
    }

    /// RPython pyjitpl.py:177 get_list_of_active_boxes parity:
    fn fail_args_to_snapshot_boxes(
        fail_args: &[OpRef],
        ctx: &majit_metainterp::TraceCtx,
    ) -> Vec<majit_trace::recorder::SnapshotTagged> {
        fail_args
            .iter()
            .map(|&opref| Self::opref_to_snapshot_tagged(opref, ctx))
            .collect()
    }

    /// snapshot boxes from active_boxes = [locals, stack].
    /// RPython: each Box carries type ('r'/'i'/'f') — pyre passes types
    /// explicitly so _number_boxes can detect virtual vs int correctly.
    fn fail_args_to_snapshot_boxes_typed(
        active_boxes: &[OpRef],
        types: &[majit_ir::Type],
        ctx: &majit_metainterp::TraceCtx,
    ) -> Vec<majit_trace::recorder::SnapshotTagged> {
        active_boxes
            .iter()
            .enumerate()
            .map(|(i, &opref)| {
                if opref.is_none() {
                    majit_trace::recorder::SnapshotTagged::Const(0, majit_ir::Type::Ref)
                } else if ctx.constant_value(opref).is_some() {
                    let val = ctx.constant_value(opref).unwrap_or(0);
                    let tp = ctx.const_type(opref).unwrap_or(majit_ir::Type::Int);
                    majit_trace::recorder::SnapshotTagged::Const(val, tp)
                } else {
                    let tp = types.get(i).copied().unwrap_or(majit_ir::Type::Ref);
                    majit_trace::recorder::SnapshotTagged::Box(opref.0, tp)
                }
            })
            .collect()
    }

    /// pyjitpl.py:1916-1927 implement_guard_value parity.
    /// executor.py:544-551 constant_from_op(box): dispatches on box.type.
    pub(crate) fn guard_value(&mut self, ctx: &mut TraceCtx, value: OpRef, expected: i64) {
        let expected_ref = match self.value_type(value) {
            majit_ir::Type::Ref => ctx.const_ref(expected),
            _ => ctx.const_int(expected),
        };
        self.record_guard(ctx, OpCode::GuardValue, &[value, expected_ref]);
        // pyjitpl.py:3512: replace_box
        ctx.heap_cache_mut().replace_box(value, expected_ref);
    }

    pub(crate) fn guard_nonnull(&mut self, ctx: &mut TraceCtx, value: OpRef) {
        // heapcache.py:561-565: skip if nullity or class already known
        if ctx.heap_cache().is_nullity_known(value) == Some(true) {
            return;
        }
        if ctx.heap_cache().is_class_known(value) {
            return; // class known implies nonnull
        }
        self.record_guard(ctx, OpCode::GuardNonnull, &[value]);
        ctx.heap_cache_mut().nullity_now_known(value, true);
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
        // heapcache: track nullity after for-iter guard
        ctx.heap_cache_mut().nullity_now_known(next, continues);
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
        if !self.parent_frames.is_empty() {
            self.record_guard(ctx, opcode, &[truth]);
            return;
        }

        // pyjitpl.py:510-520 goto_if_not(box, target, orgpc):
        //   self.metainterp.generate_guard(opnum, box, resumepc=orgpc)
        // blackhole.py:864 bhimpl_goto_if_not(a, target, pc) re-pops the
        // truth from a register and re-decides which arm to take.
        //
        // RPython's goto_if_not is a compound jitcode opcode produced by
        // jtransform.optimize_goto_if_not (jtransform.py:196): COMPARE_OP +
        // GOTO_IF_NOT are fused into a single op that takes the comparison
        // operands directly and branches. The trace's "register" for the
        // truth and the jitcode's "register" agree on type (int).
        //
        // Pyre has separate Python bytecodes for COMPARE_OP and
        // POP_JUMP_IF_FALSE. The trace records them via
        // last_comparison_truth/last_comparison_concrete_truth which is a
        // trace-time emulation of the fused op, but the codewriter still
        // generates two separate JitCode opcodes for the blackhole, and the
        // POP_JUMP_IF_FALSE jitcode reads a Ref (boxed PyBool) from the
        // PyFrame stack via move_r + call_int_typed(truth_fn). If we used
        // orgpc as the resume_pc, the snapshot at orgpc would carry the
        // symbolic stack top — which is a Type::Int raw truth from
        // trace_compare_value's fast path, not a Ref — and the blackhole
        // would call truth_fn on a raw int (NULL pointer dereference).
        //
        // ADAPTATION (pending fused jitcode opcode infrastructure):
        // Resume at other_target (the runtime branch destination) so the
        // blackhole skips POP_JUMP_IF_FALSE entirely and re-enters the
        // interpreter at the not-taken branch. The truth never has to be
        // restored — its only role was to pick the resume_pc at trace time.
        // True line-by-line parity requires implementing
        // jtransform.optimize_goto_if_not at pyre's codewriter level: a
        // fused goto_if_not_int_lt jitcode opcode taking int operands +
        // target label, and the matching bhimpl_goto_if_not_int_lt. See
        // task #30.
        let other_target = self.sym().pending_branch_other_target;
        let resume_pc = {
            let s = self.sym();
            other_target.unwrap_or(s.pending_next_instr.unwrap_or(self.fallthrough_pc))
        };

        // pyjitpl.py:2593-2602: saved_pc = frame.pc; frame.pc = resumepc;
        // capture_resumedata(); frame.pc = saved_pc
        // Save ALL state BEFORE flush (record_guard_core parity).
        let saved_orgpc = self.orgpc;
        let saved_ni = self.sym().vable_next_instr;
        let saved_vsd = self.sym().vable_valuestackdepth;
        self.orgpc = resume_pc;

        self.flush_to_frame_for_guard(ctx);
        // pyjitpl.py:177: get_list_of_active_boxes uses frame.pc for liveness
        let active_boxes = self.get_list_of_active_boxes(ctx, false, false);
        // pyjitpl.py:2558-2585 generate_guard parity: record a raw fail_args
        // "hint". store_final_boxes_in_guard rebuilds the real fail_args from
        // the snapshot via _number_boxes (Consts go to TAGCONST, not liveboxes).
        let fail_arg_types = self.build_fail_arg_types_for_active_boxes(&active_boxes);
        let fail_args: Vec<OpRef> = {
            let s = self.sym();
            let mut fa = vec![s.frame, s.vable_next_instr, s.vable_valuestackdepth];
            fa.extend_from_slice(&active_boxes);
            fa
        };

        // opencoder.py:767-770: snapshot uses active boxes (not fail_args)
        let snapshot_boxes = Self::fail_args_to_snapshot_boxes(&active_boxes, ctx);
        let vable_boxes = Self::build_virtualizable_boxes(self.sym(), ctx);
        let jitcode_index = unsafe { (*self.sym().jitcode).index } as u32;
        let snapshot = majit_trace::recorder::Snapshot {
            frames: vec![majit_trace::recorder::SnapshotFrame {
                jitcode_index,
                pc: resume_pc as u32,
                boxes: snapshot_boxes,
            }],
            vable_boxes,
            vref_boxes: Self::build_virtualref_boxes(self.sym(), ctx),
        };
        let snapshot_id = ctx.capture_resumedata(snapshot);

        ctx.record_guard_typed_with_fail_args(opcode, &[truth], fail_arg_types, &fail_args);
        ctx.set_last_guard_resume_position(snapshot_id);

        // pyjitpl.py:2602: frame.pc = saved_pc (restore all, record_guard_core parity)
        self.orgpc = saved_orgpc;
        let s = self.sym_mut();
        s.vable_next_instr = saved_ni;
        s.vable_valuestackdepth = saved_vsd;
    }

    /// RPython registers[idx] parity: read concrete value from Box arrays.
    fn concrete_at(&self, abs_idx: usize) -> Option<PyObjectRef> {
        let v = self.sym().concrete_value_at(abs_idx);
        if !v.is_null() {
            return Some(v.to_pyobj());
        }
        None
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

    pub(crate) fn guard_object_class(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        expected_type: *const PyType,
    ) {
        // heapcache.py: skip guard if class already known for this object
        if ctx.heap_cache().is_class_known(obj) {
            return;
        }
        let expected_type_const = ctx.const_int(expected_type as usize as i64);
        self.record_guard(ctx, OpCode::GuardNonnullClass, &[obj, expected_type_const]);
        // heapcache.py:470-473: class_now_known sets class + nullity.
        ctx.heap_cache_mut()
            .class_now_known(obj, majit_ir::GcRef(expected_type as usize));
    }

    pub(crate) fn trace_guarded_int_payload(
        &mut self,
        ctx: &mut TraceCtx,
        int_obj: OpRef,
    ) -> OpRef {
        if self.value_type(int_obj) == Type::Int {
            return int_obj;
        }
        self.guard_object_class(ctx, int_obj, &INT_TYPE as *const PyType);
        let raw = trace_gc_object_int_field(ctx, int_obj, int_intval_descr());
        self.remember_value_type(raw, Type::Int);
        raw
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
    pub(crate) fn trace_dynamic_list_index(
        &mut self,
        ctx: &mut TraceCtx,
        key: OpRef,
        len: OpRef,
        concrete_key: i64,
    ) -> OpRef {
        let raw_index = if self.value_type(key) == Type::Int {
            key
        } else {
            crate::state::trace_unbox_int_with_resume(self, ctx, key, &INT_TYPE as *const _ as i64)
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
        let len = trace_arraylen_gc(ctx, obj, items_len_descr);
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
        let len = trace_arraylen_gc(ctx, obj, items_len_descr);
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
        let len = trace_arraylen_gc(ctx, obj, list_items_len_descr());
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
        let len = trace_arraylen_gc(ctx, obj, list_items_len_descr());
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
        let len = trace_arraylen_gc(ctx, obj, list_int_items_len_descr());
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
        let len = trace_arraylen_gc(ctx, obj, list_int_items_len_descr());
        let index = self.trace_dynamic_list_index(ctx, key, len, concrete_key);
        let items_ptr = trace_gc_object_int_field(ctx, obj, list_int_items_ptr_descr());
        let raw = trace_raw_int_array_getitem_value(ctx, items_ptr, index);
        self.remember_value_type(raw, Type::Int);
        raw
    }

    pub(crate) fn trace_direct_float_list_getitem(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        key: OpRef,
        concrete_index: usize,
    ) -> OpRef {
        self.guard_object_class(ctx, obj, &LIST_TYPE as *const PyType);
        self.guard_list_strategy(ctx, obj, 2);
        let len = trace_arraylen_gc(ctx, obj, list_float_items_len_descr());
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
        let len = trace_arraylen_gc(ctx, obj, list_float_items_len_descr());
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

        let len = trace_arraylen_gc(ctx, seq, items_len_descr);
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
        concrete_seq: PyObjectRef,
    ) -> Result<Vec<FrontendOp>, PyError> {
        if concrete_seq.is_null() {
            let oprefs = TraceHelperAccess::trace_unpack_sequence(self, seq, count)?;
            return Ok(oprefs.into_iter().map(FrontendOp::opref_only).collect());
        }

        // Extract concrete items from the sequence for RPython Box parity.
        let concrete_items: Vec<PyObjectRef> = unsafe {
            if is_tuple(concrete_seq) {
                (0..count)
                    .filter_map(|i| w_tuple_getitem(concrete_seq, i as i64))
                    .collect()
            } else if is_list(concrete_seq) && w_list_uses_object_storage(concrete_seq) {
                (0..count)
                    .filter_map(|i| w_list_getitem(concrete_seq, i as i64))
                    .collect()
            } else {
                Vec::new()
            }
        };

        let oprefs = self.with_ctx(|this, ctx| unsafe {
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
        })?;

        Ok(oprefs
            .into_iter()
            .enumerate()
            .map(|(i, opref)| {
                let cv = concrete_items
                    .get(i)
                    .copied()
                    .map(ConcreteValue::from_pyobj)
                    .unwrap_or(ConcreteValue::Null);
                FrontendOp::new(opref, cv)
            })
            .collect())
    }

    pub(crate) fn binary_subscr_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        concrete_obj: PyObjectRef,
        concrete_key: PyObjectRef,
    ) -> Result<FrontendOp, PyError> {
        if concrete_obj.is_null() || concrete_key.is_null() {
            let opref = self.trace_binary_value(a, b, BinaryOperator::Subscr)?;
            return Ok(FrontendOp::new(opref, ConcreteValue::Null));
        }
        // MIFrame Box tracking: compute concrete subscr result
        let subscr_concrete = if let Ok(result) =
            pyre_interpreter::baseobjspace::getitem(concrete_obj, concrete_key)
        {
            ConcreteValue::from_pyobj(result)
        } else {
            ConcreteValue::Null
        };

        unsafe {
            if is_int(concrete_key) {
                let index = w_int_get_value(concrete_key);
                return self
                    .with_ctx(|this, ctx| {
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
                        } else if is_list(concrete_obj) && w_list_uses_object_storage(concrete_obj)
                        {
                            let concrete_len = w_list_len(concrete_obj);
                            if index >= 0 {
                                let index = index as usize;
                                if index < concrete_len {
                                    return Ok(
                                        this.trace_direct_object_list_getitem(ctx, a, b, index)
                                    );
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
                                    return Ok(
                                        this.trace_direct_float_list_getitem(ctx, a, b, index)
                                    );
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
                    })
                    .map(|op| FrontendOp::new(op, subscr_concrete));
            }
        }

        let opref = self.trace_binary_value(a, b, BinaryOperator::Subscr)?;
        Ok(FrontendOp::new(opref, subscr_concrete))
    }

    pub(crate) fn binary_int_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: BinaryOperator,
        concrete_lhs: PyObjectRef,
        concrete_rhs: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // RPython intobject.py:588 descr_add: type-based dispatch via GuardClass + unbox.
        // Extract concrete int values for range checks (FloorDiv, Mod, Shift).
        let concrete = unsafe {
            if concrete_lhs.is_null()
                || concrete_rhs.is_null()
                || !is_int(concrete_lhs)
                || !is_int(concrete_rhs)
            {
                None
            } else {
                Some((w_int_get_value(concrete_lhs), w_int_get_value(concrete_rhs)))
            }
        };

        // Table lookup: BinaryOperator → (OpCode, has_overflow, needs_concrete_check)
        let Some((op_code, has_overflow, needs_concrete_check)) = crate::int_binop_lookup(op)
        else {
            return self.trace_binary_value(a, b, op);
        };

        // Operators needing concrete value validation before trace emission
        if needs_concrete_check {
            match op_code {
                OpCode::IntFloorDiv | OpCode::IntMod => {
                    let Some((lhs, rhs)) = concrete else {
                        return self.trace_binary_value(a, b, op);
                    };
                    if lhs < 0 || rhs <= 0 {
                        return self.trace_binary_value(a, b, op);
                    }
                }
                OpCode::IntLshift => {
                    let Some((lhs, rhs)) = concrete else {
                        return self.trace_binary_value(a, b, op);
                    };
                    let Ok(shift) = u32::try_from(rhs) else {
                        return self.trace_binary_value(a, b, op);
                    };
                    if shift >= i64::BITS {
                        return self.trace_binary_value(a, b, op);
                    }
                    // intobject.py:207 ovfcheck(a << b): if the shift overflows
                    // i64 range, fall back to interpreter for long promotion.
                    let result = lhs.wrapping_shl(shift);
                    if result.wrapping_shr(shift) != lhs {
                        return self.trace_binary_value(a, b, op);
                    }
                }
                OpCode::IntRshift => {
                    let Some((lhs, rhs)) = concrete else {
                        return self.trace_binary_value(a, b, op);
                    };
                    // intobject.py:225: r_uint(b) >= LONG_BIT
                    let Ok(shift) = u32::try_from(rhs) else {
                        return self.trace_binary_value(a, b, op);
                    };
                    if shift >= i64::BITS {
                        // intobject.py:229-231: large shift → 0 or -1
                        let result = if lhs < 0 { -1i64 } else { 0i64 };
                        return self.with_ctx(|this, ctx| {
                            let raw = ctx.const_int(result);
                            let boxed = box_traced_raw_int(ctx, raw);
                            this.remember_value_type(boxed, Type::Ref);
                            Ok(boxed)
                        });
                    }
                }
                _ => {}
            }
        }
        self.with_ctx(|this, ctx| {
            let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
            let lhs_raw = if this.value_type(a) == Type::Int {
                a
            } else {
                crate::state::trace_unbox_int_with_resume(this, ctx, a, int_type_addr)
            };
            let rhs_raw = if this.value_type(b) == Type::Int {
                b
            } else {
                crate::state::trace_unbox_int_with_resume(this, ctx, b, int_type_addr)
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
        concrete_lhs: PyObjectRef,
        concrete_rhs: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        let is_power = matches!(op, BinaryOperator::Power | BinaryOperator::InplacePower);
        let op_code = crate::float_binop_lookup(op);

        // resoperation.py: no FLOAT_POW / FLOAT_FLOORDIV / FLOAT_MOD opcodes.
        // FloorDivide/Remainder → residual Python-level call.
        // Power → raw-float call_may_force (ll_math_pow, EF_CAN_RAISE).
        if op_code.is_none() && !is_power {
            return self.trace_binary_value(a, b, op);
        }

        // Determine which operands are int (need int→float cast) vs float.
        // RPython: float_add etc. call space.float_w() which dispatches on
        // type — int objects go through int2float (CastIntToFloat).
        let lhs_is_int = (!concrete_lhs.is_null() && unsafe { is_int(concrete_lhs) })
            || self.value_type(a) == Type::Int;
        let rhs_is_int = (!concrete_rhs.is_null() && unsafe { is_int(concrete_rhs) })
            || self.value_type(b) == Type::Int;

        self.with_ctx(|this, ctx| {
            let float_type_addr = &FLOAT_TYPE as *const _ as i64;
            let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
            // Unbox a float object to raw f64.
            let unbox_float = |this: &mut MIFrame, ctx: &mut TraceCtx, obj: OpRef| -> OpRef {
                if !ctx.heap_cache().is_class_known(obj) {
                    let type_const = ctx.const_int(float_type_addr);
                    this.record_guard(ctx, OpCode::GuardClass, &[obj, type_const]);
                    ctx.heap_cache_mut()
                        .class_now_known(obj, majit_ir::GcRef(float_type_addr as usize));
                }
                {
                    let ff_descr = crate::descr::float_floatval_descr();
                    let ff_idx = ff_descr.index();
                    if let Some(cached) = ctx.heap_cache().getfield_cached(obj, ff_idx) {
                        cached
                    } else {
                        let r = ctx.record_op_with_descr(OpCode::GetfieldGcPureF, &[obj], ff_descr);
                        ctx.heap_cache_mut().getfield_now_known(obj, ff_idx, r);
                        r
                    }
                }
            };
            // Unbox an int object to raw i64, then CastIntToFloat → f64.
            // RPython: space.float_w(w_int) → float(w_int.intval)
            let unbox_int_to_float =
                |this: &mut MIFrame, ctx: &mut TraceCtx, obj: OpRef| -> OpRef {
                    let raw_int = if this.value_type(obj) == Type::Int {
                        obj
                    } else {
                        crate::state::trace_unbox_int_with_resume(this, ctx, obj, int_type_addr)
                    };
                    ctx.record_op(OpCode::CastIntToFloat, &[raw_int])
                };
            let lhs_raw = if this.value_type(a) == Type::Float {
                a
            } else if lhs_is_int {
                unbox_int_to_float(this, ctx, a)
            } else {
                unbox_float(this, ctx, a)
            };
            let rhs_raw = if this.value_type(b) == Type::Float {
                b
            } else if rhs_is_int {
                unbox_int_to_float(this, ctx, b)
            } else {
                unbox_float(this, ctx, b)
            };
            let result = if is_power {
                // floatobject.py:561 descr_pow → _pow(space, x, y) parity.
                // ll_math_pow (ll_math.py:260) is EF_CAN_RAISE, NOT
                // force_virtual. pyjitpl.py:2084-2121 routes EF_CAN_RAISE
                // calls to `execute_varargs(rop.CALL_F, ..., exc=True, pure=False)`
                // which records CALL_F and then calls handle_possible_exception
                // (pyjitpl.py:1950-1955, 3395) — that generates GUARD_NO_EXCEPTION.
                let call_result = ctx.call_float_typed(
                    float_pow_jit as *const (),
                    &[lhs_raw, rhs_raw],
                    &[Type::Float, Type::Float],
                );
                // pyjitpl.py:3395 GUARD_NO_EXCEPTION from handle_possible_exception.
                this.record_guard(ctx, OpCode::GuardNoException, &[]);
                call_result
            } else {
                ctx.record_op(op_code.unwrap(), &[lhs_raw, rhs_raw])
            };
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
        concrete_lhs: PyObjectRef,
        concrete_rhs: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        let (lhs_obj, rhs_obj) = (concrete_lhs, concrete_rhs);
        if lhs_obj.is_null() || rhs_obj.is_null() {
            return self.trace_compare_value(a, b, op);
        }

        unsafe {
            if is_int(lhs_obj) && is_int(rhs_obj) {
                let cmp = crate::int_compare_lookup(op);
                return self.with_ctx(|this, ctx| {
                    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
                    let lhs_raw = if this.value_type(a) == Type::Int {
                        a
                    } else if let Some(raw) = try_trace_const_boxed_int(ctx, a, lhs_obj) {
                        raw
                    } else {
                        crate::state::trace_unbox_int_with_resume(this, ctx, a, int_type_addr)
                    };
                    let rhs_raw = if this.value_type(b) == Type::Int {
                        b
                    } else if let Some(raw) = try_trace_const_boxed_int(ctx, b, rhs_obj) {
                        raw
                    } else {
                        crate::state::trace_unbox_int_with_resume(this, ctx, b, int_type_addr)
                    };
                    let truth = ctx.record_op(cmp, &[lhs_raw, rhs_raw]);
                    this.remember_value_type(truth, Type::Int);
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
                let cmp = crate::float_compare_lookup(op);
                return self.with_ctx(|this, ctx| {
                    let float_type_addr = &FLOAT_TYPE as *const _ as i64;
                    let lhs_raw = if this.value_type(a) == Type::Float {
                        a
                    } else {
                        crate::state::trace_unbox_float_with_resume(this, ctx, a, float_type_addr)
                    };
                    let rhs_raw = if this.value_type(b) == Type::Float {
                        b
                    } else {
                        crate::state::trace_unbox_float_with_resume(this, ctx, b, float_type_addr)
                    };
                    let truth = ctx.record_op(cmp, &[lhs_raw, rhs_raw]);
                    this.remember_value_type(truth, Type::Int);
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
        concrete_obj: PyObjectRef,
        concrete_key: PyObjectRef,
        concrete_value: PyObjectRef,
    ) -> Result<(), PyError> {
        if concrete_obj.is_null() || concrete_key.is_null() || concrete_value.is_null() {
            return self.trace_store_subscr(obj, key, value);
        }

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
                            let len = trace_arraylen_gc(ctx, obj, list_items_len_descr());
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
                            let len = trace_arraylen_gc(ctx, obj, list_items_len_descr());
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
                            let len = trace_arraylen_gc(ctx, obj, list_int_items_len_descr());
                            let index = this.trace_dynamic_list_index(ctx, key, len, index as i64);
                            let items_ptr =
                                trace_gc_object_int_field(ctx, obj, list_int_items_ptr_descr());
                            let raw = if this.value_type(value) == Type::Int {
                                value
                            } else {
                                crate::state::trace_unbox_int_with_resume(
                                    this,
                                    ctx,
                                    value,
                                    &INT_TYPE as *const _ as i64,
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
                            let len = trace_arraylen_gc(ctx, obj, list_int_items_len_descr());
                            let index = this.trace_dynamic_list_index(ctx, key, len, index);
                            let items_ptr =
                                trace_gc_object_int_field(ctx, obj, list_int_items_ptr_descr());
                            let raw = if this.value_type(value) == Type::Int {
                                value
                            } else {
                                crate::state::trace_unbox_int_with_resume(
                                    this,
                                    ctx,
                                    value,
                                    &INT_TYPE as *const _ as i64,
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
                            let len = trace_arraylen_gc(ctx, obj, list_float_items_len_descr());
                            let index = this.trace_dynamic_list_index(ctx, key, len, index as i64);
                            let items_ptr =
                                trace_gc_object_int_field(ctx, obj, list_float_items_ptr_descr());
                            let raw = if this.value_type(value) == Type::Float {
                                value
                            } else {
                                crate::state::trace_unbox_float_with_resume(
                                    this,
                                    ctx,
                                    value,
                                    &FLOAT_TYPE as *const _ as i64,
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
                            let len = trace_arraylen_gc(ctx, obj, list_float_items_len_descr());
                            let index = this.trace_dynamic_list_index(ctx, key, len, index);
                            let items_ptr =
                                trace_gc_object_int_field(ctx, obj, list_float_items_ptr_descr());
                            let raw = if this.value_type(value) == Type::Float {
                                value
                            } else {
                                crate::state::trace_unbox_float_with_resume(
                                    this,
                                    ctx,
                                    value,
                                    &FLOAT_TYPE as *const _ as i64,
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

    pub(crate) fn list_append_value(
        &mut self,
        list: OpRef,
        value: OpRef,
        concrete_list: PyObjectRef,
    ) -> Result<(), PyError> {
        if concrete_list.is_null() {
            return self.trace_list_append(list, value);
        }

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
                        let len = trace_arraylen_gc(ctx, list, list_items_len_descr());
                        this.guard_value(ctx, len, concrete_len as i64);
                    }
                    let items_ptr = trace_gc_object_int_field(ctx, list, list_items_ptr_descr());
                    let index = ctx.const_int(concrete_len as i64);
                    trace_raw_array_setitem_value(ctx, items_ptr, index, value);
                    let new_len = ctx.const_int((concrete_len + 1) as i64);
                    let len_descr = list_items_len_descr();
                    let len_descr_idx = len_descr.index();
                    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, new_len], len_descr);
                    ctx.heap_cache_mut()
                        .setfield_cached(list, len_descr_idx, new_len);
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
                        let len = trace_arraylen_gc(ctx, list, list_int_items_len_descr());
                        this.guard_value(ctx, len, concrete_len as i64);
                    }
                    let items_ptr =
                        trace_gc_object_int_field(ctx, list, list_int_items_ptr_descr());
                    let index = ctx.const_int(concrete_len as i64);
                    let raw = if this.value_type(value) == Type::Int {
                        value
                    } else {
                        crate::state::trace_unbox_int_with_resume(
                            this,
                            ctx,
                            value,
                            &INT_TYPE as *const _ as i64,
                        )
                    };
                    trace_raw_int_array_setitem_value(ctx, items_ptr, index, raw);
                    let new_len = ctx.const_int((concrete_len + 1) as i64);
                    let len_descr = list_int_items_len_descr();
                    let len_descr_idx = len_descr.index();
                    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, new_len], len_descr);
                    ctx.heap_cache_mut()
                        .setfield_cached(list, len_descr_idx, new_len);
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
                        let len = trace_arraylen_gc(ctx, list, list_float_items_len_descr());
                        this.guard_value(ctx, len, concrete_len as i64);
                    }
                    let items_ptr =
                        trace_gc_object_int_field(ctx, list, list_float_items_ptr_descr());
                    let index = ctx.const_int(concrete_len as i64);
                    let raw = if this.value_type(value) == Type::Float {
                        value
                    } else {
                        crate::state::trace_unbox_float_with_resume(
                            this,
                            ctx,
                            value,
                            &FLOAT_TYPE as *const _ as i64,
                        )
                    };
                    trace_raw_float_array_setitem_value(ctx, items_ptr, index, raw);
                    let new_len = ctx.const_int((concrete_len + 1) as i64);
                    let len_descr = list_float_items_len_descr();
                    let len_descr_idx = len_descr.index();
                    ctx.record_op_with_descr(OpCode::SetfieldGc, &[list, new_len], len_descr);
                    ctx.heap_cache_mut()
                        .setfield_cached(list, len_descr_idx, new_len);
                    Ok(())
                });
            }
        }

        self.trace_list_append(list, value)
    }

    pub(crate) fn concrete_iter_continues(
        &self,
        concrete_iter: PyObjectRef,
    ) -> Result<bool, PyError> {
        range_iter_continues(concrete_iter)
    }

    pub(crate) fn trace_known_builtin_call(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
    ) -> Result<OpRef, PyError> {
        self.with_ctx(|this, ctx| {
            let boxed_args = box_args_for_python_helper(this, ctx, args);
            crate::helpers::emit_trace_call_known_builtin(ctx, callable, &boxed_args)
        })
    }

    pub(crate) fn direct_len_value(
        &mut self,
        callable: OpRef,
        value: OpRef,
        concrete_value: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        if concrete_value.is_null() {
            return self.trace_known_builtin_call(callable, &[value]);
        }

        unsafe {
            if is_str(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &pyre_object::STR_TYPE as *const PyType);
                    let len = trace_arraylen_gc(ctx, value, str_len_descr());
                    this.remember_value_type(len, Type::Int);
                    Ok(len)
                });
            }
            if is_dict(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &DICT_TYPE as *const PyType);
                    let len = trace_arraylen_gc(ctx, value, dict_len_descr());
                    this.remember_value_type(len, Type::Int);
                    Ok(len)
                });
            }
            if is_list(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, value, &LIST_TYPE as *const PyType);
                    let len = if w_list_uses_object_storage(concrete_value) {
                        this.guard_list_strategy(ctx, value, 0);
                        trace_arraylen_gc(ctx, value, list_items_len_descr())
                    } else if w_list_uses_int_storage(concrete_value) {
                        this.guard_list_strategy(ctx, value, 1);
                        trace_arraylen_gc(ctx, value, list_int_items_len_descr())
                    } else if w_list_uses_float_storage(concrete_value) {
                        this.guard_list_strategy(ctx, value, 2);
                        trace_arraylen_gc(ctx, value, list_float_items_len_descr())
                    } else {
                        let boxed_value = box_value_for_python_helper(this, ctx, value);
                        return crate::helpers::emit_trace_call_known_builtin(
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
                    let len = trace_arraylen_gc(ctx, value, tuple_items_len_descr());
                    this.remember_value_type(len, Type::Int);
                    Ok(len)
                });
            }
        }

        self.trace_known_builtin_call(callable, &[value])
    }

    fn direct_abs_value(
        &mut self,
        callable: OpRef,
        value: OpRef,
        concrete_value: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        if concrete_value.is_null() {
            return self.trace_known_builtin_call(callable, &[value]);
        }

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
                        crate::generated::trace_box_int(
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

    fn direct_type_value(
        &mut self,
        callable: OpRef,
        value: OpRef,
        concrete_value: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        if concrete_value.is_null() {
            return self.trace_known_builtin_call(callable, &[value]);
        }

        unsafe {
            let concrete_obj_type = (*concrete_value).ob_type;
            let type_name = (*concrete_obj_type).tp_name;
            return self.with_ctx(|this, ctx| {
                this.guard_object_class(ctx, value, concrete_obj_type);
                // box_str_constant returns a PyObjectRef.
                Ok(ctx.const_ref(pyre_object::box_str_constant(type_name) as i64))
            });
        }
    }

    fn direct_isinstance_value(
        &mut self,
        callable: OpRef,
        obj: OpRef,
        type_name: OpRef,
        concrete_obj: PyObjectRef,
        concrete_type_name: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        if concrete_obj.is_null() || concrete_type_name.is_null() {
            return self.trace_known_builtin_call(callable, &[obj, type_name]);
        }

        unsafe {
            if is_str(concrete_type_name) {
                let concrete_obj_type = (*concrete_obj).ob_type;
                let concrete_result =
                    (*concrete_obj_type).tp_name == w_str_get_value(concrete_type_name);
                return self.with_ctx(|this, ctx| {
                    this.guard_object_class(ctx, obj, concrete_obj_type);
                    this.guard_value(ctx, type_name, concrete_type_name as i64);
                    // w_bool_from returns a PyObjectRef (Ref-typed singleton).
                    Ok(ctx.const_ref(w_bool_from(concrete_result) as i64))
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
        concrete_a: PyObjectRef,
        concrete_b: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        if concrete_a.is_null() || concrete_b.is_null() {
            return self.trace_known_builtin_call(callable, &[a, b]);
        }

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
                        crate::generated::trace_box_int(
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
        concrete_callable: PyObjectRef,
        concrete_args: &[PyObjectRef],
    ) -> Result<OpRef, PyError> {
        if concrete_callable.is_null() {
            debug_assert!(
                false,
                "concrete_callable should always be available during tracing"
            );
            return self.trace_call_callable(callable, args);
        }

        unsafe {
            let is_builtin = is_function(concrete_callable)
                && is_builtin_code(
                    pyre_interpreter::getcode(concrete_callable) as pyre_object::PyObjectRef
                );
            if is_builtin {
                let builtin_name = pyre_interpreter::function_get_name(concrete_callable);
                if args.len() == 1 {
                    let c_arg0 = concrete_args.first().copied().unwrap_or(PY_NULL);
                    self.with_ctx(|this, ctx| {
                        this.guard_value(ctx, callable, concrete_callable as i64)
                    });
                    if builtin_name == "type" {
                        return self.direct_type_value(callable, args[0], c_arg0);
                    }
                    if builtin_name == "len" {
                        return self.direct_len_value(callable, args[0], c_arg0);
                    }
                    if builtin_name == "abs" {
                        return self.direct_abs_value(callable, args[0], c_arg0);
                    }
                } else if args.len() == 2 && builtin_name == "isinstance" {
                    let c_arg0 = concrete_args.first().copied().unwrap_or(PY_NULL);
                    let c_arg1 = concrete_args.get(1).copied().unwrap_or(PY_NULL);
                    self.with_ctx(|this, ctx| {
                        this.guard_value(ctx, callable, concrete_callable as i64)
                    });
                    return self
                        .direct_isinstance_value(callable, args[0], args[1], c_arg0, c_arg1);
                } else if args.len() == 2 && builtin_name == "min" {
                    let c_arg0 = concrete_args.first().copied().unwrap_or(PY_NULL);
                    let c_arg1 = concrete_args.get(1).copied().unwrap_or(PY_NULL);
                    self.with_ctx(|this, ctx| {
                        this.guard_value(ctx, callable, concrete_callable as i64)
                    });
                    return self
                        .direct_minmax_value(callable, args[0], args[1], false, c_arg0, c_arg1);
                } else if args.len() == 2 && builtin_name == "max" {
                    let c_arg0 = concrete_args.first().copied().unwrap_or(PY_NULL);
                    let c_arg1 = concrete_args.get(1).copied().unwrap_or(PY_NULL);
                    self.with_ctx(|this, ctx| {
                        this.guard_value(ctx, callable, concrete_callable as i64)
                    });
                    return self
                        .direct_minmax_value(callable, args[0], args[1], true, c_arg0, c_arg1);
                }
                return self.with_ctx(|this, ctx| {
                    this.guard_value(ctx, callable, concrete_callable as i64);
                    let boxed_args = box_args_for_python_helper(this, ctx, args);
                    crate::helpers::emit_trace_call_known_builtin(ctx, callable, &boxed_args)
                });
            }
            if is_function(concrete_callable) {
                let w_callee_code = unsafe { pyre_interpreter::getcode(concrete_callable) };
                let callee_key = crate::driver::make_green_key(w_callee_code, 0);
                let callee_code = unsafe {
                    &*(pyre_interpreter::w_code_get_ptr(w_callee_code as pyre_object::PyObjectRef)
                        as *const CodeObject)
                };
                let callee_has_loop = code_has_backward_jump(callee_code);
                let (driver, _) = crate::driver::driver_pair();
                let nargs = args.len();

                // RPython pyjitpl.py: do_residual_or_indirect_call() follows
                // direct jitcode calls via perform_call() before falling back
                // to residual helpers.  Mirror that for ordinary direct calls:
                // if we know the callee body and it is a small acyclic helper,
                // trace through it directly instead of waiting for
                // should_inline() to bless a helper-boundary inline.
                let root_trace_green_key = root_trace_green_key(self);
                let current_function_key =
                    crate::driver::make_green_key(unsafe { (*self.sym().jitcode).code }, 0);
                let is_self_recursive = callee_key == current_function_key;
                let inline_decision = driver.should_inline(callee_key);
                let inline_framestack_active = !self.parent_frames.is_empty();
                let callee_inline_eligible = driver
                    .meta_interp()
                    .warm_state_ref()
                    .can_inline_callable(callee_key);
                let max_unroll_recursion =
                    driver.meta_interp().warm_state_ref().max_unroll_recursion() as usize;
                let recursive_depth = self.with_ctx(|_, ctx| ctx.recursive_depth(callee_key));
                let concrete_arg0 = if nargs == 1 {
                    concrete_args.first().copied()
                } else {
                    None
                };
                let callee_prefers_function_entry =
                    (crate::callbacks::get().callable_prefers_function_entry)(concrete_callable);
                if is_self_recursive
                    && inline_decision == majit_metainterp::InlineDecision::Inline
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
                            && inline_decision == majit_metainterp::InlineDecision::Inline
                            && recursive_depth < max_unroll_recursion
                    } else {
                        !callee_prefers_function_entry && !callee_has_loop
                    };

                if majit_metainterp::majit_log_enabled() {
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
                        this.guard_value(ctx, callable, concrete_callable as i64);
                        let boxed_args = box_args_for_python_helper(this, ctx, args);
                        let result = crate::helpers::emit_trace_call_known_function(
                            ctx,
                            this.frame(),
                            callable,
                            &boxed_args,
                        )?;
                        this.push_call_replay_stack(ctx, callable, args, call_pc);
                        this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                        ctx.heap_cache_mut().invalidate_caches_for_escaped();
                        this.pop_call_replay_stack(ctx, args.len())?;
                        Ok(result)
                    });
                }

                if can_trace_through {
                    // RPython perform_call: produce PendingInlineFrame for
                    // the MetaInterp interpret loop to push onto framestack.
                    match self.build_pending_inline_frame(
                        callable,
                        args,
                        concrete_callable,
                        callee_key,
                        concrete_args,
                    ) {
                        Ok(pending) => {
                            self.pending_inline_frame = Some(pending);
                            return self
                                .with_ctx(|_, ctx| Ok(ctx.const_int(pyre_object::PY_NULL as i64)));
                        }
                        Err(err) => {
                            if majit_metainterp::majit_log_enabled() {
                                eprintln!(
                                    "[jit][perform-call] build_pending failed key={} err={}, residual path",
                                    callee_key, err
                                );
                            }
                            // Fall through to residual helper path
                        }
                    }
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
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][call-check] is_self={} cache_safe={} inline_active={} callee_key={}",
                        is_self_recursive,
                        (crate::callbacks::get().recursive_force_cache_safe)(concrete_callable),
                        inline_framestack_active,
                        callee_key
                    );
                }

                if inline_decision == majit_metainterp::InlineDecision::Inline {
                    if let Some(frame_helper) = (crate::callbacks::get().callee_frame_helper)(nargs)
                    {
                        return self.inline_function_call(
                            callable,
                            args,
                            concrete_callable,
                            callee_key,
                            frame_helper,
                            concrete_args,
                        );
                    }
                }

                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][call-dispatch] callee_key={} pending_token={:?} loop_token={:?} is_self={}",
                        callee_key,
                        driver.get_pending_token_number(callee_key),
                        driver.get_loop_token_number(callee_key),
                        is_self_recursive
                    );
                }
                if let Some(token_number) = driver.get_pending_token_number(callee_key) {
                    let callee_nlocals = unsafe {
                        let code_ptr =
                            pyre_interpreter::get_pycode(concrete_callable) as *const CodeObject;
                        (&*code_ptr).varnames.len()
                    };
                    if nargs == 1 || (crate::callbacks::get().callee_frame_helper)(nargs).is_some()
                    {
                        return self.with_ctx(|this, ctx| {
                            if !is_self_recursive {
                                this.guard_value(ctx, callable, concrete_callable as i64);
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
                                    (crate::callbacks::get().callee_frame_helper)(nargs).unwrap();
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
                                crate::callbacks::get().jit_drop_callee_frame,
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
                    majit_metainterp::InlineDecision::CallAssembler => {
                        // Trace-through: inline callee body instead of CallAssembler.
                        // Guards use parent_frames to avoid OpRef::NONE in fail_args.
                        if callee_inline_eligible
                            && !is_self_recursive
                            && nargs <= 4
                            && !callee_has_loop
                        {
                            match self.build_pending_inline_frame(
                                callable,
                                args,
                                concrete_callable,
                                callee_key,
                                concrete_args,
                            ) {
                                Ok(pending) => {
                                    self.pending_inline_frame = Some(pending);
                                    return self.with_ctx(|_, ctx| {
                                        Ok(ctx.const_int(pyre_object::PY_NULL as i64))
                                    });
                                }
                                Err(err) => {
                                    if majit_metainterp::majit_log_enabled() {
                                        eprintln!(
                                            "[jit][perform-call] call-assembler inline failed key={} err={}",
                                            callee_key, err
                                        );
                                    }
                                }
                            }
                        }
                        // Use compiled loop token only (not pending_token)
                        // to avoid type descriptor mismatches.
                        let Some(token_number) = driver.get_loop_token_number(callee_key) else {
                            let call_pc = self.fallthrough_pc.saturating_sub(1);
                            return self.with_ctx(|this, ctx| {
                                this.guard_value(ctx, callable, concrete_callable as i64);
                                let result = crate::helpers::emit_trace_call_known_function(
                                    ctx,
                                    this.frame(),
                                    callable,
                                    args,
                                )?;
                                this.push_call_replay_stack(ctx, callable, args, call_pc);
                                this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                                ctx.heap_cache_mut().invalidate_caches_for_escaped();
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
                                    this.guard_value(ctx, callable, concrete_callable as i64);
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
                                    (crate::callbacks::get().callee_frame_helper)(nargs)
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
                                    return Err(pyre_interpreter::PyError::type_error(
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
                                    crate::callbacks::get().jit_drop_callee_frame,
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
                    majit_metainterp::InlineDecision::Inline => {
                        if let Some(frame_helper) =
                            (crate::callbacks::get().callee_frame_helper)(nargs)
                        {
                            return self.inline_function_call(
                                callable,
                                args,
                                concrete_callable,
                                callee_key,
                                frame_helper,
                                concrete_args,
                            );
                        }
                    }
                    majit_metainterp::InlineDecision::ResidualCall => {}
                }

                let call_pc = self.fallthrough_pc.saturating_sub(1);
                return self.with_ctx(|this, ctx| {
                    this.guard_value(ctx, callable, concrete_callable as i64);
                    let boxed_args = box_args_for_python_helper(this, ctx, args);
                    let result = crate::helpers::emit_trace_call_known_function(
                        ctx,
                        this.frame(),
                        callable,
                        &boxed_args,
                    )?;
                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                    this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                    ctx.heap_cache_mut().invalidate_caches_for_escaped();
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
        passed_concrete_args: &[PyObjectRef],
    ) -> Result<PendingInlineFrame, PyError> {
        use pyre_interpreter::pyframe::PyFrame;

        self.with_ctx(|this, ctx| {
            this.guard_value(ctx, callable, concrete_callable as i64);
        });

        let concrete_args: Vec<PyObjectRef> = passed_concrete_args.to_vec();
        for (_idx, arg) in concrete_args.iter().copied().enumerate() {
            if arg.is_null() {
                return Err(PyError::type_error(
                    "pending inline frame lost concrete arg",
                ));
            }
        }

        let caller_code = unsafe { (*self.sym().jitcode).code };
        let caller_exec_ctx = self.sym().concrete_execution_context;
        let caller_namespace_ptr = self.sym().concrete_namespace;
        let w_code = unsafe { pyre_interpreter::getcode(concrete_callable) };
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const CodeObject
        };
        let globals = unsafe { function_get_globals(concrete_callable) };
        let closure = unsafe { pyre_interpreter::function_get_closure(concrete_callable) };
        let is_self_recursive = crate::driver::make_green_key(caller_code, 0) == callee_key;
        let mut callee_frame = PyFrame::new_for_call_with_closure(
            w_code,
            &concrete_args,
            globals,
            caller_exec_ctx,
            closure,
        );
        callee_frame.fix_array_ptrs();

        let callee_code = unsafe { &*pyre_interpreter::pyframe_get_pycode(&callee_frame) };
        let callee_nlocals = callee_code.varnames.len();
        let caller_namespace = caller_namespace_ptr;
        let callee_globals = unsafe { function_get_globals(concrete_callable) };
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
            // MIFrame Box tracking: set concrete metadata for callee
            sym.concrete_locals = concrete_args
                .iter()
                .map(|&a| ConcreteValue::from_pyobj(a))
                .collect();
            sym.concrete_locals
                .resize(callee_nlocals, ConcreteValue::Null);
            sym.concrete_stack = Vec::new();
            sym.jitcode = jitcode_for(w_code);
            sym.concrete_namespace = callee_globals as *mut PyNamespace;
            sym.concrete_execution_context = self.sym().concrete_execution_context;
            let (vable_next_instr, vable_code, vable_valuestackdepth, vable_namespace) = self
                .with_ctx(|_, ctx| {
                    // w_code and callee_globals are PyObjectRef pointers;
                    // tag them as Ref so the typed constant pool dedupes
                    // them with any other Ref reference to the same address.
                    (
                        ctx.const_int(0),
                        ctx.const_ref(w_code as i64),
                        ctx.const_int(callee_nlocals as i64),
                        ctx.const_ref(callee_globals as i64),
                    )
                });
            sym.vable_next_instr = vable_next_instr;
            sym.vable_code = vable_code;
            sym.vable_valuestackdepth = vable_valuestackdepth;
            sym.vable_namespace = vable_namespace;
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
                } else if let Some(frame_helper) =
                    (crate::callbacks::get().callee_frame_helper)(args.len())
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
            // MIFrame Box tracking: set concrete metadata for callee
            sym.concrete_locals = concrete_args
                .iter()
                .map(|&a| ConcreteValue::from_pyobj(a))
                .collect();
            sym.concrete_locals
                .resize(callee_nlocals, ConcreteValue::Null);
            sym.concrete_stack = Vec::new();
            sym.jitcode = jitcode_for(w_code);
            sym.concrete_namespace = callee_globals as *mut PyNamespace;
            sym.concrete_execution_context = self.sym().concrete_execution_context;
            (sym, Some(callee_frame_opref))
        };

        // pyjitpl.py:2601-2602: parent frame keeps its original pc
        // (return_point_pc = CALL fallthrough). The callee blackhole
        // handles the call; the caller continues AFTER the call returns.
        // Stack is post-dispatch (args consumed, no result yet).
        let return_point_pc = self.fallthrough_pc;
        self.with_ctx(|this, ctx| {
            let ni = ctx.const_int(return_point_pc as i64);
            let s = this.sym_mut();
            s.vable_next_instr = ni;
            s.vable_valuestackdepth = ctx.const_int(s.valuestackdepth as i64);
        });
        // pyjitpl.py:177: active boxes for parent frame.
        // pyjitpl.py:177: parent frame is in_a_call=True (opencoder.py:806).
        let my_active_boxes =
            self.with_ctx(|this, ctx| this.get_list_of_active_boxes(ctx, true, false));
        // fail_args = header + active_boxes (for Cranelift deadframe).
        let my_fail_args = self.with_ctx(|this, _ctx| {
            let s = this.sym();
            let mut fa = vec![s.frame, s.vable_next_instr, s.vable_valuestackdepth];
            fa.extend_from_slice(&my_active_boxes);
            fa
        });
        let my_fail_arg_types = self.build_fail_arg_types_for_active_boxes(&my_active_boxes);
        // opencoder.py:810: parent frame's jitcode.index.
        let my_jitcode_index = unsafe { (*self.sym().jitcode).index };
        // opencoder.py:819-834 parity: accumulate full parent chain.
        // Current frame becomes the newest parent; existing parents follow.
        let mut parent_frames = vec![(
            my_fail_args,
            my_fail_arg_types,
            return_point_pc,
            my_jitcode_index,
        )];
        parent_frames.extend(self.parent_frames.iter().cloned());
        Ok(PendingInlineFrame {
            sym: callee_sym,
            concrete_frame: callee_frame,
            drop_frame_opref,
            green_key: callee_key,
            parent_frames,
            nargs: args.len(),
            caller_result_stack_idx: None,
        })
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
        passed_concrete_args: &[PyObjectRef],
    ) -> Result<OpRef, PyError> {
        let (driver, _) = crate::driver::driver_pair();
        let concrete_arg0 = if args.len() == 1 {
            passed_concrete_args.first().copied()
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
                        == crate::driver::make_green_key(unsafe { (*this.sym().jitcode).code }, 0);
                    // RPython parity: an opaque helper-boundary Python CALL
                    // still produces a boxed object result.  Even if the
                    // callee itself can finish with a raw int, the helper
                    // boxes at the boundary and the trace records a Ref.
                    let force_fn = if is_self_recursive
                        && (crate::callbacks::get().recursive_force_cache_safe)(concrete_callable)
                    {
                        crate::callbacks::get().jit_force_self_recursive_call_argraw_boxed_1
                    } else {
                        crate::callbacks::get().jit_force_recursive_call_argraw_boxed_1
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
                            let code_ptr = pyre_interpreter::get_pycode(concrete_callable)
                                as *const pyre_interpreter::CodeObject;
                            (&*code_ptr).varnames.len()
                        };
                        // RPython pyjitpl.py:3583: pass raw Int arg, return raw Int.
                        let ca_args = synthesize_fresh_callee_entry_args(
                            ctx,
                            callee_frame,
                            &[raw_arg],
                            callee_nlocals,
                        );
                        let callee_slot_types =
                            pending_entry_slot_types_from_args(&[Type::Int], callee_nlocals, 0);
                        let ca_arg_types =
                            frame_entry_arg_types_from_slot_types(&callee_slot_types);
                        let ca_result = ctx.call_assembler_int_by_number_typed(
                            token_number,
                            &ca_args,
                            &ca_arg_types,
                        );
                        ctx.call_void(
                            crate::callbacks::get().jit_drop_callee_frame,
                            &[callee_frame],
                        );
                        this.remember_value_type(ca_result, Type::Int);
                        ca_result
                    } else if force_fn
                        == crate::callbacks::get().jit_force_self_recursive_call_argraw_boxed_1
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
                        this.sync_standard_virtualizable_after_residual_call(ctx);
                    } else if !this.sync_standard_virtualizable_after_residual_call(ctx) {
                        this.push_call_replay_stack(ctx, callable, args, call_pc);
                        this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                        ctx.heap_cache_mut().invalidate_caches_for_escaped();
                        this.pop_call_replay_stack(ctx, args.len())?;
                    }
                    result
                } else {
                    let force_fn = crate::callbacks::get().jit_force_recursive_call_1;
                    this.sync_standard_virtualizable_before_residual_call(ctx);
                    let result = ctx.call_may_force_ref_typed(
                        force_fn,
                        &[this.frame(), callable, args[0]],
                        &[Type::Ref, Type::Ref, Type::Ref],
                    );
                    if !this.sync_standard_virtualizable_after_residual_call(ctx) {
                        this.push_call_replay_stack(ctx, callable, args, call_pc);
                        this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                        ctx.heap_cache_mut().invalidate_caches_for_escaped();
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
                let force_fn = crate::callbacks::get().jit_force_callee_frame;
                this.sync_standard_virtualizable_before_residual_call(ctx);
                let result = ctx.call_may_force_ref_typed(force_fn, &[callee_frame], &[Type::Ref]);
                if !this.sync_standard_virtualizable_after_residual_call(ctx) {
                    // GuardNotForced fail → interpreter must re-execute CALL.
                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                    this.record_guard(ctx, OpCode::GuardNotForced, &[]);
                    ctx.heap_cache_mut().invalidate_caches_for_escaped();
                    this.pop_call_replay_stack(ctx, args.len())?;
                }
                ctx.call_void(
                    crate::callbacks::get().jit_drop_callee_frame,
                    &[callee_frame],
                );
                Ok(result)
            }
        });

        result
    }

    pub(crate) fn iter_next_value(
        &mut self,
        iter: OpRef,
        concrete_iter: PyObjectRef,
    ) -> Result<FrontendOp, PyError> {
        let concrete_continues = range_iter_continues(concrete_iter)?;
        let concrete_step =
            unsafe { (*(concrete_iter as *const pyre_object::rangeobject::W_RangeIterator)).step };
        let concrete_current = unsafe {
            (*(concrete_iter as *const pyre_object::rangeobject::W_RangeIterator)).current
        };
        let concrete_step_positive = concrete_step > 0;
        if concrete_continues {
            if concrete_current.checked_add(concrete_step).is_none() {
                let opref = self.trace_iter_next_value(iter)?;
                return Ok(FrontendOp::opref_only(opref));
            }
        }

        self.with_ctx(|this, ctx| {
            MIFrame::guard_range_iter(this, ctx, iter);

            let current = trace_gc_object_int_field(ctx, iter, range_iter_current_descr());
            let stop = trace_gc_object_int_field(ctx, iter, range_iter_stop_descr());
            let step = trace_gc_object_int_field(ctx, iter, range_iter_step_descr());
            let zero = ctx.const_int(0);
            // pyjitpl.py:1877 opimpl_goto_if_not: boolean guards use
            // GuardTrue/GuardFalse, NOT GuardValue. GuardValue's replace_box
            // propagates a constant substitution that unroll's body-peeling
            // then drops because the peeled body recomputes the value from
            // fresh field reads. GuardTrue only records the truth assertion
            // without constant substitution, so the body's corresponding
            // guard is preserved after unrolling.
            let step_positive = ctx.record_op(OpCode::IntGt, &[step, zero]);
            if concrete_step_positive {
                this.record_guard(ctx, OpCode::GuardTrue, &[step_positive]);
            } else {
                this.record_guard(ctx, OpCode::GuardFalse, &[step_positive]);
            }

            let continues = if concrete_step_positive {
                ctx.record_op(OpCode::IntLt, &[current, stop])
            } else {
                ctx.record_op(OpCode::IntGt, &[current, stop])
            };
            if concrete_continues {
                this.record_guard(ctx, OpCode::GuardTrue, &[continues]);
            } else {
                this.record_guard(ctx, OpCode::GuardFalse, &[continues]);
            }

            if !concrete_continues {
                this.remember_value_type(zero, Type::Int);
                return Ok(FrontendOp::new(zero, ConcreteValue::Int(0)));
            }

            let next_current = ctx.record_op(OpCode::IntAddOvf, &[current, step]);
            this.record_guard(ctx, OpCode::GuardNoOverflow, &[]);
            let ri_descr = range_iter_current_descr();
            let ri_descr_idx = ri_descr.index();
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[iter, next_current], ri_descr);
            ctx.heap_cache_mut()
                .setfield_cached(iter, ri_descr_idx, next_current);
            this.remember_value_type(current, Type::Int);
            Ok(FrontendOp::new(
                current,
                ConcreteValue::Int(concrete_current),
            ))
        })
    }

    pub(crate) fn concrete_branch_truth_for_value(
        &mut self,
        _value: OpRef,
        concrete_val: PyObjectRef,
    ) -> Result<bool, PyError> {
        if let Some(truth) = self.sym_mut().last_comparison_concrete_truth.take() {
            return Ok(truth);
        }
        if concrete_val.is_null() {
            return Err(PyError::type_error(
                "missing concrete branch value during trace",
            ));
        }
        Ok(objspace_truth_value(concrete_val))
    }

    pub(crate) fn concrete_branch_truth(&mut self) -> Result<bool, PyError> {
        self.concrete_branch_truth_for_value(OpRef::NONE, PY_NULL)
    }

    pub(crate) fn truth_value_direct(
        &mut self,
        value: OpRef,
        concrete_val: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // RPython goto_if_not fusion: if the previous op was a comparison,
        // use the cached raw truth (0/1) directly instead of unboxing the
        // bool object. This eliminates the CallI(bool_helper) + GetfieldGcI
        // round-trip, matching RPython's fused goto_if_not_int_lt pattern.
        if let Some(truth) = self.sym_mut().last_comparison_truth.take() {
            return Ok(truth);
        }

        if !concrete_val.is_null() {
            let cached_concrete_truth = objspace_truth_value(concrete_val);
            self.sym_mut().last_comparison_concrete_truth = Some(cached_concrete_truth);
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

        let concrete_value = concrete_val;
        if concrete_value.is_null() {
            return self.trace_truth_value(value);
        }

        unsafe {
            if is_int(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    let int_type_addr = &INT_TYPE as *const _ as i64;
                    let int_value = if let Some(raw) =
                        try_trace_const_boxed_int(ctx, value, concrete_value)
                    {
                        raw
                    } else {
                        crate::state::trace_unbox_int_with_resume(this, ctx, value, int_type_addr)
                    };
                    let zero = ctx.const_int(0);
                    Ok(ctx.record_op(OpCode::IntNe, &[int_value, zero]))
                });
            }
            if is_bool(concrete_value) {
                return self.with_ctx(|this, ctx| {
                    let bool_type_addr = &BOOL_TYPE as *const _ as i64;
                    let bool_value =
                        if let Some(raw) = try_trace_const_boxed_int(ctx, value, concrete_value) {
                            raw
                        } else {
                            crate::state::trace_unbox_int_with_resume_descr(
                                this,
                                ctx,
                                value,
                                bool_type_addr,
                                crate::descr::bool_boolval_descr(),
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
                    let float_type_addr = &FLOAT_TYPE as *const _ as i64;
                    let float_value = crate::state::trace_unbox_float_with_resume(
                        this,
                        ctx,
                        value,
                        float_type_addr,
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
        concrete_value: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        let is_int_operand = !concrete_value.is_null() && unsafe { is_int(concrete_value) };
        if !is_int_operand {
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
                let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
                crate::state::trace_unbox_int_with_resume(this, ctx, value, int_type_addr)
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
        result: Result<pyre_interpreter::StepResult<FrontendOp>, PyError>,
    ) -> TraceAction {
        trace_step_result_to_action(self, result)
    }

    pub fn trace_code_step(&mut self, code: &CodeObject, pc: usize) -> TraceAction {
        if pc >= code.instructions.len() {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step pc_oob pc={} code_len={}",
                    pc,
                    code.instructions.len()
                );
            }
            return TraceAction::Abort;
        }

        let Some((instruction, op_arg)) = decode_instruction_at(code, pc) else {
            if majit_metainterp::majit_log_enabled() {
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

        self.set_orgpc(pc);
        self.prepare_fallthrough();
        // RPython capture_resumedata(resumepc=orgpc): save pre-opcode
        // stack state so guard fail_args reflect the state BEFORE the
        // opcode executes. On guard failure the interpreter re-executes
        // the opcode from orgpc.
        {
            let s = self.sym_mut();
            let stack_only = s.valuestackdepth.saturating_sub(s.nlocals);
            s.pre_opcode_vsd = Some(s.valuestackdepth);
            s.pre_opcode_stack =
                Some(s.symbolic_stack[..stack_only.min(s.symbolic_stack.len())].to_vec());
            s.pre_opcode_stack_types = Some(
                s.symbolic_stack_types[..stack_only.min(s.symbolic_stack_types.len())].to_vec(),
            );
        }
        let step_result = execute_opcode_step(self, code, instruction, op_arg, pc + 1);
        // Clear pre-opcode snapshot immediately after opcode execution.
        // RPython: capture_resumedata is called at each guard; there is
        // no per-opcode snapshot lifecycle. Clearing here ensures the
        // snapshot doesn't leak into subsequent opcodes.
        {
            let s = self.sym_mut();
            s.pre_opcode_vsd = None;
            s.pre_opcode_stack = None;
            s.pre_opcode_stack_types = None;
        }
        // RPython execute_ll_raised parity: store exception in
        // last_exc_value before calling handle_possible_exception.
        if let Err(ref err) = step_result {
            let exc_obj = err.to_exc_object();
            let s = self.sym_mut();
            if s.last_exc_value.is_null() {
                s.last_exc_value = exc_obj;
            }
        }
        // RPython pyjitpl.py:1956-1957 execute_varargs: exc=True ops
        // always call handle_possible_exception, which internally decides
        // GUARD_EXCEPTION vs GUARD_NO_EXCEPTION.
        if instruction_may_raise(instruction) {
            let action = self.handle_possible_exception(code, pc);
            if !matches!(action, TraceAction::Continue) {
                return action;
            }
            // RPython: handle_possible_exception consumes the exception.
            // If it returned Continue, the exception was handled (e.g.,
            // GUARD_EXCEPTION emitted) and tracing continues normally.
            // Treat handled exceptions as successful opcode completion.
            if step_result.is_err() {
                return TraceAction::Continue;
            }
        }
        match step_result {
            Err(ref e) => {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][abort-reason] trace_code_step err at pc={} instr={:?} err={:?}",
                        pc, instruction, e
                    );
                }
                TraceAction::Abort
            }
            other => self.into_trace_action(other),
        }
    }

    /// RPython pyjitpl.py:3380 handle_possible_exception.
    ///
    /// Called after every may-raise opcode. Checks last_exc_value to decide:
    /// - exception raised → GUARD_EXCEPTION + finishframe_exception
    /// - no exception → GUARD_NO_EXCEPTION
    pub(crate) fn handle_possible_exception(
        &mut self,
        code: &CodeObject,
        pc: usize,
    ) -> TraceAction {
        if !self.sym().last_exc_value.is_null() {
            let exc_obj = self.sym().last_exc_value;

            // pyjitpl.py:3382-3384: ALWAYS emit GUARD_EXCEPTION first,
            // regardless of class_of_last_exc_is_const.
            let exc_type_ptr = unsafe {
                (*(exc_obj as *const pyre_object::excobject::W_ExceptionObject))
                    .ob_header
                    .ob_type as i64
            };

            let guard_op = self.with_ctx(|this, ctx| {
                // pyjitpl.py:2575-2578: after_residual_call=true for
                // GuardException — all boxes in top frame are live.
                let after_residual_call = true;
                let resume_pc = this.fallthrough_pc;
                let saved_orgpc = this.orgpc;
                this.orgpc = resume_pc;

                this.flush_to_frame_for_guard(ctx);
                let active_boxes = this.get_list_of_active_boxes(ctx, false, after_residual_call);
                let fail_arg_types = this.build_fail_arg_types_for_active_boxes(&active_boxes);
                let fail_args = {
                    let s = this.sym();
                    let mut fa = vec![s.frame, s.vable_next_instr, s.vable_valuestackdepth];
                    fa.extend_from_slice(&active_boxes);
                    fa
                };

                // capture_resumedata parity: full framestack snapshot.
                // pyjitpl.py:2597: capture_resumedata(self.framestack, ...)
                let __n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
                let jitcode_index = unsafe { (*this.sym().jitcode).index } as u32;
                // fail_arg_types includes header [Ref, Int, Int]; snapshot
                // needs only the active_boxes portion.
                let snapshot_types = &fail_arg_types[__n..];
                let mut frames = vec![majit_trace::recorder::SnapshotFrame {
                    jitcode_index,
                    pc: resume_pc as u32,
                    boxes: Self::fail_args_to_snapshot_boxes_typed(
                        &active_boxes,
                        snapshot_types,
                        ctx,
                    ),
                }];
                let mut all_fail_args = fail_args.clone();
                let mut all_types = fail_arg_types.clone();
                // Include parent frames (RPython: full self.framestack).
                for (pfa, pfa_types, pfa_resumepc, pfa_jitcode_index) in &this.parent_frames {
                    // pyjitpl.py:2586-2602 capture_resumedata: parent
                    // frames' snapshot stores ONLY their active boxes (not
                    // the [frame, ni, vsd] header). When pfa.len() <= __n,
                    // there are zero active boxes — body must be empty.
                    let parent_active: &[OpRef] = if pfa.len() > __n { &pfa[__n..] } else { &[] };
                    let parent_types: &[Type] = if pfa_types.len() > __n {
                        &pfa_types[__n..]
                    } else {
                        &[]
                    };
                    frames.push(majit_trace::recorder::SnapshotFrame {
                        jitcode_index: *pfa_jitcode_index as u32,
                        pc: *pfa_resumepc as u32,
                        boxes: Self::fail_args_to_snapshot_boxes_typed(
                            parent_active,
                            parent_types,
                            ctx,
                        ),
                    });
                    all_fail_args.extend_from_slice(pfa);
                    let pt = if pfa_types.is_empty() {
                        fail_arg_types_for_virtualizable_state(pfa.len())
                    } else {
                        pfa_types.clone()
                    };
                    all_types.extend_from_slice(&pt);
                }
                let vable_boxes = Self::build_virtualizable_boxes(this.sym(), ctx);
                let snapshot = majit_trace::recorder::Snapshot {
                    frames,
                    vable_boxes,
                    vref_boxes: Self::build_virtualref_boxes(this.sym(), ctx),
                };
                let snapshot_id = ctx.capture_resumedata(snapshot);

                let exc_type_const = ctx.const_int(exc_type_ptr);
                let op = ctx.record_guard_typed_with_fail_args(
                    majit_ir::OpCode::GuardException,
                    &[exc_type_const],
                    all_types,
                    &all_fail_args,
                );
                ctx.set_last_guard_resume_position(snapshot_id);

                this.orgpc = saved_orgpc;
                op
            });

            // pyjitpl.py:3385-3392:
            //   val = cast_opaque_ptr(GCREF, self.last_exc_value)
            //   if self.class_of_last_exc_is_const:
            //       self.last_exc_box = ConstPtr(val)
            //   else:
            //       self.last_exc_box = op
            //       op.setref_base(val)
            //   self.class_of_last_exc_is_const = True
            if self.sym().class_of_last_exc_is_const {
                let exc_box = self.with_ctx(|_this, ctx| ctx.const_ref(exc_obj as i64));
                self.sym_mut().last_exc_box = exc_box;
            } else {
                self.sym_mut().last_exc_box = guard_op;
            }
            self.sym_mut().class_of_last_exc_is_const = true;

            self.finishframe_exception(code, pc)
        } else {
            // pyjitpl.py:3397: GUARD_NO_EXCEPTION
            self.with_ctx(|this, ctx| {
                this.record_guard(ctx, majit_ir::OpCode::GuardNoException, &[]);
            });
            TraceAction::Continue
        }
    }

    /// RPython pyjitpl.py:2506 finishframe_exception (single-frame).
    ///
    /// Checks current frame for an exception handler.
    /// If found: unwind stack to handler depth, push exception, continue.
    /// If not found: return Abort (metainterp handles multi-frame unwind).
    fn finishframe_exception(&mut self, code: &CodeObject, pc: usize) -> TraceAction {
        let exc_opref = self.sym().last_exc_box;
        let exc_obj = self.sym().last_exc_value;
        let concrete_frame_addr = self.concrete_frame_addr;

        // pyjitpl.py:2510-2520: scan for catch_exception handler
        // (Python 3.11+ exception table replaces RPython's op_catch_exception)
        if let Some(entry) =
            pyre_interpreter::bytecode::find_exception_handler(&code.exceptiontable, pc as u32)
        {
            let handler_pc = entry.target as usize;
            let handler_depth = entry.depth as usize;
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][finishframe_exception] pc={} handler={} depth={}",
                    pc, handler_pc, handler_depth
                );
            }

            // pyjitpl.py:2506 finishframe_exception: unwind stack to handler,
            // pyjitpl.py:2517: frame.pc = target; raise ChangeFrame
            let ncells = unsafe { (&*code).cellvars.len() + (&*code).freevars.len() };
            let nlocals = self.sym().nlocals;
            let target_stack_len = ncells + handler_depth;
            {
                let s = self.sym_mut();
                s.symbolic_stack.truncate(target_stack_len);
                s.symbolic_stack_types.truncate(target_stack_len);
                s.concrete_stack.truncate(target_stack_len);
                s.valuestackdepth = nlocals + target_stack_len;
                if entry.push_lasti {
                    s.symbolic_stack.push(OpRef::NONE);
                    s.symbolic_stack_types.push(Type::Ref);
                    s.concrete_stack
                        .push(ConcreteValue::Ref(pyre_object::w_int_new(pc as i64)));
                    s.valuestackdepth += 1;
                }
                s.symbolic_stack.push(exc_opref);
                s.symbolic_stack_types.push(Type::Ref);
                s.concrete_stack.push(ConcreteValue::Ref(exc_obj));
                s.valuestackdepth += 1;
            }
            // Sync concrete frame.
            let frame = unsafe { &mut *(concrete_frame_addr as *mut pyre_interpreter::PyFrame) };
            let target_depth = frame.nlocals() + frame.ncells() + handler_depth;
            while frame.valuestackdepth > target_depth {
                frame.pop();
            }
            if entry.push_lasti {
                frame.push(pyre_object::w_int_new(pc as i64));
            }
            frame.push(exc_obj);
            // pyjitpl.py:2518: frame.pc = target; raise ChangeFrame
            self.sym_mut().pending_next_instr = Some(handler_pc);
            TraceAction::Continue
        } else {
            // No handler in this frame — return Abort so metainterp's
            // multi-frame finishframe_exception can pop this frame and
            // try the parent (pyjitpl.py:2520 self.popframe() loop).
            // Root frame with no handler → metainterp emits FINISH
            // (pyjitpl.py:2532 compile_exit_frame_with_exception).
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[jit][finishframe_exception] no handler pc={}", pc);
            }
            TraceAction::Abort
        }
    }

    pub fn trace_code_step_inline(
        &mut self,
        code: &CodeObject,
        pc: usize,
    ) -> InlineTraceStepAction {
        if pc >= code.instructions.len() {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step_inline pc_oob pc={} code_len={}",
                    pc,
                    code.instructions.len()
                );
            }
            return InlineTraceStepAction::Trace(TraceAction::Abort);
        }

        let Some((instruction, op_arg)) = decode_instruction_at(code, pc) else {
            if majit_metainterp::majit_log_enabled() {
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
        // RPython execute_ll_raised parity: store exception in last_exc_value.
        if let Err(ref err) = step_result {
            let exc_obj = err.to_exc_object();
            let s = self.sym_mut();
            if s.last_exc_value.is_null() {
                s.last_exc_value = exc_obj;
            }
        }
        if instruction_may_raise(instruction) {
            let exc_action = self.handle_possible_exception(code, pc);
            if !matches!(exc_action, TraceAction::Continue) {
                return InlineTraceStepAction::Trace(exc_action);
            }
            if step_result.is_err() {
                return InlineTraceStepAction::Trace(TraceAction::Continue);
            }
        }
        let action = match step_result {
            Ok(pyre_interpreter::StepResult::Return(value)) => TraceAction::Finish {
                finish_args: vec![value.opref],
                finish_arg_types: vec![self.value_type(value.opref)],
            },
            Err(_) => TraceAction::Abort,
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
    state: &mut MIFrame,
    result: Result<pyre_interpreter::StepResult<FrontendOp>, PyError>,
) -> TraceAction {
    match result {
        Ok(pyre_interpreter::StepResult::Continue) => {
            if state.ctx().is_too_long() {
                let green_key = state.ctx().green_key();
                let root_green_key = state.with_ctx(|_, ctx| ctx.root_green_key());
                if let Some(biggest_key) = biggest_inline_trace_key(state) {
                    let (driver, _) = crate::driver::driver_pair();
                    let warm_state = driver.meta_interp_mut().warm_state_mut();
                    warm_state.disable_noninlinable_function(biggest_key);
                    warm_state.trace_next_iteration(root_green_key);
                    if majit_metainterp::majit_log_enabled() {
                        eprintln!(
                            "[jit][trace-too-long] biggest_inline_key={} trace_next_iteration root_key={}",
                            biggest_key, root_green_key
                        );
                    }
                    return majit_metainterp::TraceAction::Abort;
                }
                let force_finish_trace = {
                    let (driver, _) = crate::driver::driver_pair();
                    driver.meta_interp().force_finish_trace_enabled()
                };
                if force_finish_trace {
                    let jump_args = state.with_ctx(|this, ctx| this.close_loop_args(ctx));
                    if majit_metainterp::majit_log_enabled() {
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
                if majit_metainterp::majit_log_enabled() {
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
        Ok(pyre_interpreter::StepResult::CloseLoop {
            jump_args,
            loop_header_pc,
        }) => TraceAction::CloseLoopWithArgs {
            jump_args: jump_args.iter().map(|fop| fop.opref).collect(),
            loop_header_pc: Some(loop_header_pc),
        },
        Ok(pyre_interpreter::StepResult::Return(fop)) => {
            // RPython DoneWithThisFrameDescrInt parity: unbox W_IntObject
            // to raw Int for the Finish descriptor.
            let value = fop.opref;
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
        Ok(pyre_interpreter::StepResult::Yield(fop)) => {
            let finish_type = match state.value_type(fop.opref) {
                Type::Int => Type::Int,
                Type::Float => Type::Float,
                Type::Ref | Type::Void => Type::Ref,
            };
            TraceAction::Finish {
                finish_args: vec![fop.opref],
                finish_arg_types: vec![finish_type],
            }
        }
        Err(err) => {
            if majit_metainterp::majit_log_enabled() {
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

impl TraceHelperAccess for MIFrame {
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
            // heapcache.py: invalidate_caches after non-pure calls.
            // The call may have mutated heap state, so cached field
            // values for escaped objects are no longer reliable.
            ctx.heap_cache_mut().invalidate_caches_for_escaped();
        });
    }

    fn trace_call_callable(&mut self, callable: OpRef, args: &[OpRef]) -> Result<OpRef, PyError> {
        let frame = self.trace_frame();
        let result = self.with_ctx(|this, ctx| {
            let boxed_args = box_args_for_python_helper(this, ctx, args);
            crate::helpers::emit_trace_call_callable(ctx, frame, callable, &boxed_args)
        })?;
        self.trace_record_not_forced_guard();
        Ok(result)
    }

    fn trace_binary_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: pyre_interpreter::bytecode::BinaryOperator,
    ) -> Result<OpRef, PyError> {
        self.with_ctx(|this, ctx| {
            let lhs = box_value_for_python_helper(this, ctx, a);
            let rhs = box_value_for_python_helper(this, ctx, b);
            crate::helpers::emit_trace_binary_value(ctx, lhs, rhs, op)
        })
    }
}

impl SharedOpcodeHandler for MIFrame {
    type Value = FrontendOp;

    fn push_value(&mut self, value: Self::Value) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            MIFrame::push_value(this, ctx, value.opref, value.concrete);
            Ok(())
        })
    }

    fn pop_value(&mut self) -> Result<Self::Value, PyError> {
        // RPython Box: pop symbolic + concrete together.
        let s = self.sym();
        let stack_idx = s.valuestackdepth.checked_sub(s.nlocals + 1).unwrap_or(0);
        let concrete = s
            .concrete_stack
            .get(stack_idx)
            .copied()
            .unwrap_or(ConcreteValue::Null);
        let opref = self.with_ctx(|this, ctx| MIFrame::pop_value(this, ctx))?;
        Ok(FrontendOp::new(opref, concrete))
    }

    fn peek_at(&mut self, depth: usize) -> Result<Self::Value, PyError> {
        // RPython Box: peek returns FrontendOp with concrete from stack.
        let s = self.sym();
        let stack_idx = s.valuestackdepth.checked_sub(s.nlocals + depth + 1);
        let concrete = stack_idx
            .and_then(|idx| s.concrete_stack.get(idx).copied())
            .unwrap_or(ConcreteValue::Null);
        let opref = self.with_ctx(|this, ctx| MIFrame::peek_value(this, ctx, depth))?;
        Ok(FrontendOp::new(opref, concrete))
    }

    fn guard_nonnull_value(&mut self, value: Self::Value) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            MIFrame::guard_nonnull(this, ctx, value.opref);
            Ok(())
        })
    }

    fn make_function(&mut self, code_obj: Self::Value) -> Result<Self::Value, PyError> {
        let opref = self.trace_make_function(code_obj.opref)?;
        Ok(FrontendOp::opref_only(opref))
    }

    fn call_callable(
        &mut self,
        callable: Self::Value,
        args: &[Self::Value],
    ) -> Result<Self::Value, PyError> {
        // RPython executor.execute_varargs parity: compute concrete result.
        let concrete_callable = callable.concrete.to_pyobj();
        let concrete_args: Vec<PyObjectRef> = args.iter().map(|a| a.concrete.to_pyobj()).collect();
        let mut result_concrete = ConcreteValue::Null;
        if !concrete_callable.is_null() && concrete_args.iter().all(|v| !v.is_null()) {
            unsafe {
                if pyre_interpreter::is_function(concrete_callable)
                    && pyre_interpreter::is_builtin_code(pyre_interpreter::getcode(
                        concrete_callable,
                    )
                        as pyre_object::PyObjectRef)
                {
                    let code = pyre_interpreter::getcode(concrete_callable);
                    let func = pyre_interpreter::builtin_code_get(code as pyre_object::PyObjectRef);
                    let result = func(&concrete_args).unwrap_or(pyre_object::PY_NULL);
                    result_concrete = ConcreteValue::from_pyobj(result);
                } else if pyre_interpreter::is_function(concrete_callable) {
                    use std::cell::Cell;
                    thread_local! {
                        static CONCRETE_CALL_DEPTH: Cell<u32> = Cell::new(0);
                    }
                    let depth = CONCRETE_CALL_DEPTH.with(|d| d.get());
                    if depth < 32 {
                        CONCRETE_CALL_DEPTH.with(|d| d.set(depth + 1));
                        let exec_ctx = self.sym().concrete_execution_context;
                        let result = pyre_interpreter::call::call_user_function_plain_with_ctx(
                            exec_ctx,
                            concrete_callable,
                            &concrete_args,
                        );
                        CONCRETE_CALL_DEPTH.with(|d| d.set(depth));
                        if let Ok(result) = result {
                            result_concrete = ConcreteValue::from_pyobj(result);
                        }
                    }
                }
            }
        }
        let arg_oprefs: Vec<OpRef> = args.iter().map(|a| a.opref).collect();
        let opref = self.call_callable_value(
            callable.opref,
            &arg_oprefs,
            concrete_callable,
            &concrete_args,
        )?;
        // Inline frame path: call_callable_value returns PY_NULL placeholder.
        // Keep result_concrete if computed by concrete execution above.
        Ok(FrontendOp::new(opref, result_concrete))
    }

    fn build_list(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        let concrete_items: Vec<PyObjectRef> =
            items.iter().map(|i| i.concrete.to_pyobj()).collect();
        let mut result_concrete = ConcreteValue::Null;
        if concrete_items.iter().all(|v| !v.is_null()) {
            let list = pyre_interpreter::build_list_from_refs(&concrete_items);
            result_concrete = ConcreteValue::from_pyobj(list);
        }
        let item_oprefs: Vec<OpRef> = items.iter().map(|i| i.opref).collect();
        let opref = self.trace_build_list(&item_oprefs)?;
        Ok(FrontendOp::new(opref, result_concrete))
    }

    fn build_tuple(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        let concrete_items: Vec<PyObjectRef> =
            items.iter().map(|i| i.concrete.to_pyobj()).collect();
        let mut result_concrete = ConcreteValue::Null;
        if concrete_items.iter().all(|v| !v.is_null()) {
            let tuple = pyre_interpreter::build_tuple_from_refs(&concrete_items);
            result_concrete = ConcreteValue::from_pyobj(tuple);
        }
        let item_oprefs: Vec<OpRef> = items.iter().map(|i| i.opref).collect();
        let opref = self.trace_build_tuple(&item_oprefs)?;
        Ok(FrontendOp::new(opref, result_concrete))
    }

    fn build_map(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        let concrete_items: Vec<PyObjectRef> =
            items.iter().map(|i| i.concrete.to_pyobj()).collect();
        let mut result_concrete = ConcreteValue::Null;
        if concrete_items.iter().all(|v| !v.is_null()) {
            let dict = pyre_interpreter::build_map_from_refs(&concrete_items);
            result_concrete = ConcreteValue::from_pyobj(dict);
        }
        let item_oprefs: Vec<OpRef> = items.iter().map(|i| i.opref).collect();
        let opref = self.trace_build_map(&item_oprefs)?;
        Ok(FrontendOp::new(opref, result_concrete))
    }

    fn store_subscr(
        &mut self,
        obj: Self::Value,
        key: Self::Value,
        value: Self::Value,
    ) -> Result<(), PyError> {
        // RPython MIFrame parity: trace-only, no concrete mutation.
        // Root frame: interpreter executes the real STORE_SUBSCR.
        // Inline frame: MetaInterp.concrete_execute_step handles it.
        self.store_subscr_value(
            obj.opref,
            key.opref,
            value.opref,
            obj.concrete.to_pyobj(),
            key.concrete.to_pyobj(),
            value.concrete.to_pyobj(),
        )
    }

    fn list_append(&mut self, list: Self::Value, value: Self::Value) -> Result<(), PyError> {
        // RPython MIFrame parity: trace-only, no concrete mutation.
        self.list_append_value(list.opref, value.opref, list.concrete.to_pyobj())
    }

    fn unpack_sequence(
        &mut self,
        seq: Self::Value,
        count: usize,
    ) -> Result<Vec<Self::Value>, PyError> {
        self.unpack_sequence_value(seq.opref, count, seq.concrete.to_pyobj())
    }

    fn load_attr(&mut self, obj: Self::Value, name: &str) -> Result<Self::Value, PyError> {
        // Concrete getattr from FrontendOp
        let mut result_concrete = ConcreteValue::Null;
        let c_obj = obj.concrete.to_pyobj();
        if !c_obj.is_null() {
            if let Ok(result) = pyre_interpreter::baseobjspace::getattr(c_obj, name) {
                result_concrete = ConcreteValue::from_pyobj(result);
            }
        }
        let opref = self.trace_load_attr(obj.opref, name)?;
        Ok(FrontendOp::new(opref, result_concrete))
    }

    fn store_attr(
        &mut self,
        obj: Self::Value,
        name: &str,
        value: Self::Value,
    ) -> Result<(), PyError> {
        self.trace_store_attr(obj.opref, name, value.opref)
    }
}

impl LocalOpcodeHandler for MIFrame {
    fn load_local_value(&mut self, idx: usize) -> Result<Self::Value, PyError> {
        let concrete = self
            .sym()
            .concrete_locals
            .get(idx)
            .copied()
            .unwrap_or(ConcreteValue::Null);
        let opref = self.with_ctx(|this, ctx| MIFrame::load_local_value(this, ctx, idx))?;
        Ok(FrontendOp::new(opref, concrete))
    }

    fn load_local_checked_value(&mut self, idx: usize, name: &str) -> Result<Self::Value, PyError> {
        let _ = name;
        let concrete = self
            .sym()
            .concrete_locals
            .get(idx)
            .copied()
            .unwrap_or(ConcreteValue::Null);
        let opref = self.with_ctx(|this, ctx| MIFrame::load_local_value(this, ctx, idx))?;
        if self.value_type(opref) == Type::Ref {
            self.with_ctx(|this, ctx| {
                MIFrame::guard_nonnull(this, ctx, opref);
            });
        }
        Ok(FrontendOp::new(opref, concrete))
    }

    fn store_local_value(&mut self, idx: usize, value: Self::Value) -> Result<(), PyError> {
        // RPython Box: concrete travels inside FrontendOp
        if idx < self.sym().concrete_locals.len() {
            self.sym_mut().concrete_locals[idx] = value.concrete;
        }
        self.with_ctx(|this, ctx| MIFrame::store_local_value(this, ctx, idx, value.opref))
    }
}

impl NamespaceOpcodeHandler for MIFrame {
    fn load_name_value(&mut self, name: &str) -> Result<Self::Value, PyError> {
        let ns = self.sym().concrete_namespace;
        let Some(slot) = namespace_slot_direct(ns, name) else {
            let opref = self.trace_load_name(name)?;
            return Ok(FrontendOp::opref_only(opref));
        };
        let concrete_cv = namespace_value_direct(ns, slot);
        let result_concrete = concrete_cv
            .map(ConcreteValue::from_pyobj)
            .unwrap_or(ConcreteValue::Null);
        if let Some(concrete_value) = concrete_cv {
            if !concrete_value.is_null() {
                // RPython celldict.py @elidable_promote + quasiimmut.py parity:
                //
                // 1. QUASIIMMUT_FIELD(ns, slot) — optimizer collects into
                //    quasi_immutable_deps + emits GUARD_NOT_INVALIDATED.
                //
                // 2. RECORD_KNOWN_RESULT(result, ns, slot) — cache the
                //    trace-time lookup result (RPython call_pure_results).
                //    pyjitpl.py:419: jitcode opname `record_known_result_r`
                //    when result kind is ref (jtransform.py:303-307); same
                //    RECORD_KNOWN_RESULT opcode at the trace level.
                //
                // 3. CALL_PURE_R(ns, slot) — elidable lookup call.
                //    resoperation.py:1214 call_pure_for_descr returns
                //    rop.CALL_PURE_R when the descr's normalized result type
                //    is 'r'; namespace lookup returns a PyObjectRef so the
                //    matching opnum is CALL_PURE_R, not CALL_PURE_I.
                //    RPython record_result_of_call_pure: all args constant
                //    → history.cut() → trace-time constant. OptPure folds
                //    via lookup_known_result → same effect.
                let opref = self.with_ctx(|this, ctx| {
                    // ns and concrete_value are PyObjectRef pointers (Ref-typed).
                    // Use const_ref so the constant pool tracks them with the
                    // correct type — otherwise typed seeding sees them as Int
                    // and optimize_guard_value cannot match a Ref-typed
                    // expected against an Int-typed obj_ref, leaving a
                    // redundant GuardValue with no resume snapshot in the
                    // optimized trace.
                    let ns_const = ctx.const_ref(ns as i64);
                    let slot_const = ctx.const_int(slot as i64);
                    ctx.record_op(OpCode::QuasiimmutField, &[ns_const, slot_const]);
                    let result_const = ctx.const_ref(concrete_value as i64);
                    ctx.record_op(
                        OpCode::RecordKnownResult,
                        &[result_const, ns_const, slot_const],
                    );
                    let call_result = ctx.record_op(OpCode::CallPureR, &[ns_const, slot_const]);
                    this.sym_mut()
                        .symbolic_namespace_slots
                        .insert(slot, call_result);
                    Ok::<_, pyre_interpreter::PyError>(call_result)
                })?;
                return Ok(FrontendOp::new(opref, result_concrete));
            }
        }
        let opref = self.with_ctx(|this, ctx| MIFrame::load_namespace_value(this, ctx, slot))?;
        Ok(FrontendOp::new(opref, result_concrete))
    }

    fn store_name_value(&mut self, name: &str, value: Self::Value) -> Result<(), PyError> {
        let ns = self.sym().concrete_namespace;
        let Some(slot) = namespace_slot_direct(ns, name) else {
            return self.trace_store_name(name, value.opref);
        };
        self.with_ctx(|this, ctx| MIFrame::store_namespace_value(this, ctx, slot, value.opref))
    }

    fn null_value(&mut self) -> Result<Self::Value, PyError> {
        let opref = self.trace_null_value()?;
        Ok(FrontendOp::new(
            opref,
            ConcreteValue::Ref(pyre_object::PY_NULL),
        ))
    }
}

impl StackOpcodeHandler for MIFrame {
    fn swap_values(&mut self, depth: usize) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| MIFrame::swap_values(this, ctx, depth))
    }
}

impl IterOpcodeHandler for MIFrame {
    fn ensure_iter_value(&mut self, iter: Self::Value) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            MIFrame::guard_range_iter(this, ctx, iter.opref);
            Ok(())
        })
    }

    fn concrete_iter_continues(&mut self, iter: Self::Value) -> Result<bool, PyError> {
        let concrete_iter = iter.concrete.to_pyobj();
        MIFrame::concrete_iter_continues(self, concrete_iter)
    }

    fn iter_next_value(&mut self, iter: Self::Value) -> Result<Self::Value, PyError> {
        let concrete_iter = iter.concrete.to_pyobj();
        MIFrame::iter_next_value(self, iter.opref, concrete_iter)
    }

    fn guard_optional_value(&mut self, next: Self::Value, continues: bool) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            MIFrame::record_for_iter_guard(this, ctx, next.opref, continues);
            Ok(())
        })
    }

    fn on_iter_exhausted(&mut self, target: usize) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            MIFrame::set_next_instr(this, ctx, target);
            Ok(())
        })
    }
}

impl TruthOpcodeHandler for MIFrame {
    type Truth = OpRef;

    fn truth_value(&mut self, value: Self::Value) -> Result<Self::Truth, PyError> {
        self.truth_value_direct(value.opref, value.concrete.to_pyobj())
    }

    fn bool_value_from_truth(
        &mut self,
        truth: Self::Truth,
        negate: bool,
    ) -> Result<Self::Value, PyError> {
        let mut result_concrete = ConcreteValue::Null;
        if let Some(concrete_truth) = self.sym().last_comparison_concrete_truth {
            let result = if negate {
                !concrete_truth
            } else {
                concrete_truth
            };
            result_concrete = ConcreteValue::Int(result as i64);
        }
        let opref = self.trace_bool_value_from_truth(truth, negate)?;
        Ok(FrontendOp::new(opref, result_concrete))
    }
}

impl ControlFlowOpcodeHandler for MIFrame {
    fn fallthrough_target(&mut self) -> usize {
        self.fallthrough_pc()
    }

    fn set_next_instr(&mut self, target: usize) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            MIFrame::set_next_instr(this, ctx, target);
            Ok(())
        })
    }

    fn close_loop_args(&mut self, target: usize) -> Result<Option<Vec<Self::Value>>, PyError> {
        self.with_ctx(|this, ctx| {
            // pyjitpl.py:2950-3036: reached_loop_header
            let code_ptr = unsafe { (*this.sym().jitcode).code };
            let back_edge_key = crate::driver::make_green_key(code_ptr, target);
            // pyjitpl.py:2951: self.heapcache.reset()
            ctx.reset_heap_cache();
            // pyjitpl.py:2978-2983: get_procedure_token(greenboxes)
            // RPython uses the CURRENT greenboxes (back_edge_key), not root_key.
            // pyjitpl.py:2979: if not self.partial_trace — skip compile_trace
            // when a retrace is pending (compile_retrace handles it via
            // compile_loop at the final merge point).
            {
                let (driver, _) = crate::driver::driver_pair();
                let has_partial = driver.meta_interp().has_partial_trace();
                let bridge_origin = driver.bridge_origin();
                if !has_partial && driver.meta_interp().has_compiled_targets(back_edge_key) {
                    let jump_args = MIFrame::close_loop_args(this, ctx);
                    let outcome = driver
                        .meta_interp_mut()
                        .compile_trace(back_edge_key, &jump_args, bridge_origin);
                    if matches!(outcome, majit_metainterp::CompileOutcome::Compiled { .. }) {
                        if majit_metainterp::majit_log_enabled() {
                            eprintln!(
                                "[jit][reached_loop_header] compile_trace success: key={} pc={} bridge={:?}",
                                back_edge_key, target, bridge_origin
                            );
                        }
                        MIFrame::set_next_instr(this, ctx, target);
                        return Ok(Some(
                            jump_args.into_iter().map(FrontendOp::opref_only).collect(),
                        ));
                    }
                }
            }
            // pyjitpl.py:2994-3036: search current_merge_points
            if !ctx.has_merge_point(back_edge_key) {
                // pyjitpl.py:3034-3036: no loop found → register and continue
                let live_args = MIFrame::close_loop_args(this, ctx);
                let live_types = {
                    let s = this.sym();
                    let mut types = crate::virtualizable_gen::virt_live_value_types(0);
                    types.extend(s.symbolic_local_types.iter().copied());
                    let stack_only = s.stack_only_depth();
                    types.extend(
                        s.symbolic_stack_types[..stack_only.min(s.symbolic_stack_types.len())]
                            .iter()
                            .copied(),
                    );
                    types
                };
                ctx.add_merge_point(back_edge_key, live_args, live_types, target);
                MIFrame::set_next_instr(this, ctx, target);
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][reached_loop_header] first visit, unroll: key={} pc={}",
                        back_edge_key, target
                    );
                }
                return Ok(None);
            }
            // pyjitpl.py:3002-3030: Found! Compile it as a loop.
            MIFrame::set_next_instr(this, ctx, target);
            let oprefs = MIFrame::close_loop_args(this, ctx);
            Ok(Some(
                oprefs.into_iter().map(FrontendOp::opref_only).collect(),
            ))
        })
    }
}

impl BranchOpcodeHandler for MIFrame {
    fn enter_branch_truth(&mut self, value: Self::Value) -> Result<(), PyError> {
        self.sym_mut().pending_branch_value = Some(value.opref);
        Ok(())
    }

    fn leave_branch_truth(&mut self) -> Result<(), PyError> {
        let sym = self.sym_mut();
        sym.pending_branch_value = None;
        sym.pending_branch_other_target = None;
        Ok(())
    }

    fn set_branch_other_target(&mut self, target: usize) {
        self.sym_mut().pending_branch_other_target = Some(target);
    }

    fn branch_other_target(&self) -> Option<usize> {
        self.sym().pending_branch_other_target
    }

    fn concrete_truth_as_bool(
        &mut self,
        value: Self::Value,
        _truth: Self::Truth,
    ) -> Result<bool, PyError> {
        MIFrame::concrete_branch_truth_for_value(self, value.opref, value.concrete.to_pyobj())
    }

    fn guard_truth_value(&mut self, truth: Self::Truth, expect_true: bool) -> Result<(), PyError> {
        self.with_ctx(|this, ctx| {
            let opcode = if expect_true {
                OpCode::GuardTrue
            } else {
                OpCode::GuardFalse
            };
            MIFrame::record_guard(this, ctx, opcode, &[truth]);
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
            MIFrame::record_branch_guard(this, ctx, value.opref, truth, concrete_truth);
            Ok(())
        })
    }
}

impl ArithmeticOpcodeHandler for MIFrame {
    fn binary_value(
        &mut self,
        a_fop: Self::Value,
        b_fop: Self::Value,
        op: BinaryOperator,
    ) -> Result<Self::Value, PyError> {
        let a = a_fop.opref;
        let b = b_fop.opref;
        let lhs_obj = a_fop.concrete.to_pyobj();
        let rhs_obj = b_fop.concrete.to_pyobj();
        let mut result_concrete = ConcreteValue::Null;
        // MIFrame Box tracking: compute concrete result for binary ops
        // Float path
        if !lhs_obj.is_null() && !rhs_obj.is_null() {
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
                    if let Some(result) = crate::concrete_float_binop(op, lhs_f, rhs_f) {
                        if !result.is_nan()
                            || matches!(
                                op,
                                BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide
                            )
                        {
                            result_concrete = ConcreteValue::Float(result);
                        }
                    }
                }
            }
        }
        // Int path
        if result_concrete.is_null() && !lhs_obj.is_null() && !rhs_obj.is_null() {
            unsafe {
                if is_int(lhs_obj) && is_int(rhs_obj) {
                    let lhs = w_int_get_value(lhs_obj);
                    let rhs = w_int_get_value(rhs_obj);
                    // intobject.py:556 int_add/sub/mul: overflow → long path.
                    // intobject.py:205 _lshift: ovfcheck(a << b).
                    // Use checked arithmetic; on overflow, skip this fast path
                    // and fall through to the objspace slow path.
                    let result: Option<i64> = crate::concrete_int_binop(op, lhs, rhs);
                    if let Some(r) = result {
                        result_concrete = ConcreteValue::Int(r);
                    }
                }
            }
        }
        // Fallback: objspace (BigInt, mixed types)
        if result_concrete.is_null() && !lhs_obj.is_null() && !rhs_obj.is_null() {
            let result = match op {
                BinaryOperator::Add | BinaryOperator::InplaceAdd => {
                    pyre_interpreter::baseobjspace::add(lhs_obj, rhs_obj)
                }
                BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => {
                    pyre_interpreter::baseobjspace::sub(lhs_obj, rhs_obj)
                }
                BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => {
                    pyre_interpreter::baseobjspace::mul(lhs_obj, rhs_obj)
                }
                _ => Err(pyre_interpreter::PyError::type_error("unsupported")),
            };
            if let Ok(r) = result {
                result_concrete = ConcreteValue::from_pyobj(r);
            }
        }
        if matches!(op, BinaryOperator::Subscr) {
            let fop = self.binary_subscr_value(a, b, lhs_obj, rhs_obj)?;
            let concrete = if result_concrete.is_null() {
                fop.concrete
            } else {
                result_concrete
            };
            return Ok(FrontendOp::new(fop.opref, concrete));
        }
        let is_float_path = (!lhs_obj.is_null()
            && !rhs_obj.is_null()
            && unsafe { is_float(lhs_obj) || is_float(rhs_obj) })
            || self.value_type(a) == Type::Float
            || self.value_type(b) == Type::Float;
        let opref = if is_float_path {
            self.binary_float_value(a, b, op, lhs_obj, rhs_obj)?
        } else {
            self.binary_int_value(a, b, op, lhs_obj, rhs_obj)?
        };
        Ok(FrontendOp::new(opref, result_concrete))
    }

    fn compare_value(
        &mut self,
        a_fop: Self::Value,
        b_fop: Self::Value,
        op: ComparisonOperator,
    ) -> Result<Self::Value, PyError> {
        let a = a_fop.opref;
        let b = b_fop.opref;
        let lhs_obj = a_fop.concrete.to_pyobj();
        let rhs_obj = b_fop.concrete.to_pyobj();
        let mut result_concrete = ConcreteValue::Null;
        if !lhs_obj.is_null() && !rhs_obj.is_null() {
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
                    result_concrete = ConcreteValue::Int(result as i64);
                }
            }
        }
        if result_concrete.is_null() && !lhs_obj.is_null() && !rhs_obj.is_null() {
            unsafe {
                if is_int(lhs_obj) && is_int(rhs_obj) {
                    let lhs = w_int_get_value(lhs_obj);
                    let rhs = w_int_get_value(rhs_obj);
                    let result = match op {
                        ComparisonOperator::Less => lhs < rhs,
                        ComparisonOperator::LessOrEqual => lhs <= rhs,
                        ComparisonOperator::Greater => lhs > rhs,
                        ComparisonOperator::GreaterOrEqual => lhs >= rhs,
                        ComparisonOperator::Equal => lhs == rhs,
                        ComparisonOperator::NotEqual => lhs != rhs,
                    };
                    result_concrete = ConcreteValue::Int(result as i64);
                }
            }
        }
        if result_concrete.is_null() && !lhs_obj.is_null() && !rhs_obj.is_null() {
            let cmp_op = match op {
                ComparisonOperator::Less => pyre_interpreter::baseobjspace::CompareOp::Lt,
                ComparisonOperator::LessOrEqual => pyre_interpreter::baseobjspace::CompareOp::Le,
                ComparisonOperator::Greater => pyre_interpreter::baseobjspace::CompareOp::Gt,
                ComparisonOperator::GreaterOrEqual => pyre_interpreter::baseobjspace::CompareOp::Ge,
                ComparisonOperator::Equal => pyre_interpreter::baseobjspace::CompareOp::Eq,
                ComparisonOperator::NotEqual => pyre_interpreter::baseobjspace::CompareOp::Ne,
            };
            if let Ok(r) = pyre_interpreter::baseobjspace::compare(lhs_obj, rhs_obj, cmp_op) {
                result_concrete = ConcreteValue::from_pyobj(r);
            }
        }
        let opref = self.compare_value_direct(a, b, op, lhs_obj, rhs_obj)?;
        Ok(FrontendOp::new(opref, result_concrete))
    }

    fn unary_negative_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        let concrete_val = value.concrete.to_pyobj();
        let mut result_concrete = ConcreteValue::Null;
        if !concrete_val.is_null() && unsafe { is_int(concrete_val) } {
            let v = unsafe { w_int_get_value(concrete_val) };
            result_concrete = ConcreteValue::Int(v.wrapping_neg());
        }
        let opref = self.unary_int_value(value.opref, OpCode::IntNeg, concrete_val)?;
        Ok(FrontendOp::new(opref, result_concrete))
    }

    fn unary_invert_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        let concrete_val = value.concrete.to_pyobj();
        let mut result_concrete = ConcreteValue::Null;
        if !concrete_val.is_null() && unsafe { is_int(concrete_val) } {
            let v = unsafe { w_int_get_value(concrete_val) };
            result_concrete = ConcreteValue::Int(!v);
        }
        let opref = self.unary_int_value(value.opref, OpCode::IntInvert, concrete_val)?;
        Ok(FrontendOp::new(opref, result_concrete))
    }
}

impl ConstantOpcodeHandler for MIFrame {
    fn int_constant(&mut self, value: i64) -> Result<Self::Value, PyError> {
        let concrete = ConcreteValue::Int(value);
        let opref = self.trace_int_constant(value)?;
        Ok(FrontendOp::new(opref, concrete))
    }

    fn bigint_constant(&mut self, value: &PyBigInt) -> Result<Self::Value, PyError> {
        let concrete = ConcreteValue::Ref(pyre_object::w_long_new(value.clone()));
        let opref = self.trace_bigint_constant(value)?;
        Ok(FrontendOp::new(opref, concrete))
    }

    fn float_constant(&mut self, value: f64) -> Result<Self::Value, PyError> {
        let concrete = ConcreteValue::Float(value);
        let opref = self.trace_float_constant(value)?;
        Ok(FrontendOp::new(opref, concrete))
    }

    fn bool_constant(&mut self, value: bool) -> Result<Self::Value, PyError> {
        let concrete = ConcreteValue::Int(value as i64);
        let opref = self.trace_bool_constant(value)?;
        Ok(FrontendOp::new(opref, concrete))
    }

    fn str_constant(&mut self, value: &str) -> Result<Self::Value, PyError> {
        let concrete = ConcreteValue::Ref(pyre_object::w_str_new(value));
        let opref = self.trace_str_constant(value)?;
        Ok(FrontendOp::new(opref, concrete))
    }

    fn code_constant(&mut self, code: &CodeObject) -> Result<Self::Value, PyError> {
        let concrete = ConcreteValue::Ref(code as *const CodeObject as PyObjectRef);
        let opref = self.trace_code_constant(code)?;
        Ok(FrontendOp::new(opref, concrete))
    }

    fn none_constant(&mut self) -> Result<Self::Value, PyError> {
        let concrete = ConcreteValue::Ref(pyre_object::w_none());
        let opref = self.trace_none_constant()?;
        Ok(FrontendOp::new(opref, concrete))
    }
}

impl OpcodeStepExecutor for MIFrame {
    type Error = PyError;

    /// Fix fusion opcode: load two locals with correct concrete tracking.
    /// FrontendOp carries concrete directly — no pending_concrete_push needed.
    fn load_fast_pair_checked(
        &mut self,
        idx1: usize,
        _name1: &str,
        idx2: usize,
        _name2: &str,
    ) -> Result<(), Self::Error> {
        let c1 = self
            .sym()
            .concrete_locals
            .get(idx1)
            .copied()
            .unwrap_or(ConcreteValue::Null);
        let c2 = self
            .sym()
            .concrete_locals
            .get(idx2)
            .copied()
            .unwrap_or(ConcreteValue::Null);
        let v1 = self.with_ctx(|this, ctx| MIFrame::load_local_value(this, ctx, idx1))?;
        SharedOpcodeHandler::push_value(self, FrontendOp::new(v1, c1))?;
        let v2 = self.with_ctx(|this, ctx| MIFrame::load_local_value(this, ctx, idx2))?;
        SharedOpcodeHandler::push_value(self, FrontendOp::new(v2, c2))?;
        Ok(())
    }

    fn to_bool(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// LOAD_ATTR is_method=true — RPython LOOKUP_METHOD parity.
    fn load_method(&mut self, name: &str) -> Result<(), Self::Error> {
        let obj = SharedOpcodeHandler::pop_value(self)?;
        let concrete_obj = obj.concrete.to_pyobj();

        // Abort for instance method calls or unknown concrete.
        let is_instance =
            !concrete_obj.is_null() && unsafe { pyre_object::is_instance(concrete_obj) };
        if is_instance || concrete_obj.is_null() {
            return Err(PyError::type_error(
                "load_method: instance method — needs LOOKUP_METHOD IR (not yet implemented)",
            ));
        }

        // Non-instance path: trace as normal [attr, NULL]
        let mut attr_concrete = ConcreteValue::Null;
        if let Ok(result) = pyre_interpreter::baseobjspace::getattr(concrete_obj, name) {
            attr_concrete = ConcreteValue::from_pyobj(result);
        }
        let attr_opref = self.trace_load_attr(obj.opref, name)?;
        SharedOpcodeHandler::push_value(self, FrontendOp::new(attr_opref, attr_concrete))?;

        let null_opref = self.trace_null_value()?;
        SharedOpcodeHandler::push_value(
            self,
            FrontendOp::new(null_opref, ConcreteValue::Ref(pyre_object::PY_NULL)),
        )
    }

    // RPython exception handler tracing (pyjitpl.py:2506 finishframe_exception):
    // handle_possible_exception emits GUARD_EXCEPTION and continues at the
    // handler PC. These three overrides trace the handler-entry bytecodes.
    //
    //   PUSH_EXC_INFO  → pop exc, push None (prev_exc), push exc (+1 depth)
    //   CHECK_EXC_MATCH → opimpl_goto_if_exception_mismatch (pyjitpl.py:1677)
    //   POP_EXCEPT     → pop prev_exc, clear_exception (pyjitpl.py:2751)

    fn push_exc_info(&mut self) -> Result<(), Self::Error> {
        let exc = <Self as SharedOpcodeHandler>::pop_value(self)?;
        let none_obj = pyre_object::w_none();
        let none_opref = self.with_ctx(|_this, ctx| ctx.const_ref(none_obj as i64));
        <Self as SharedOpcodeHandler>::push_value(
            self,
            FrontendOp::new(none_opref, ConcreteValue::Ref(none_obj)),
        )?;
        <Self as SharedOpcodeHandler>::push_value(self, exc)?;
        let frame =
            unsafe { &mut *(self.concrete_frame_addr as *mut pyre_interpreter::pyframe::PyFrame) };
        let exc_val = frame.pop();
        frame.push(pyre_object::w_none());
        frame.push(exc_val);
        Ok(())
    }

    fn pop_except(&mut self) -> Result<(), Self::Error> {
        let _ = <Self as SharedOpcodeHandler>::pop_value(self).ok();
        let frame =
            unsafe { &mut *(self.concrete_frame_addr as *mut pyre_interpreter::pyframe::PyFrame) };
        frame.pop();
        // RPython pyjitpl.py:2751 clear_exception: exception fully handled.
        let s = self.sym_mut();
        s.last_exc_value = std::ptr::null_mut();
        s.class_of_last_exc_is_const = false;
        Ok(())
    }

    /// RPython pyjitpl.py:1677 opimpl_goto_if_exception_mismatch.
    ///
    /// Pops the expected exception type, checks against last_exc_value,
    /// and pushes the concrete match result. GUARD_EXCEPTION already
    /// verified the class so this usually produces True, but multi-except
    /// blocks may produce False for non-matching clauses.
    fn check_exc_match(&mut self) -> Result<(), Self::Error> {
        let exc_type_val = <Self as SharedOpcodeHandler>::pop_value(self).ok();
        let exc_type_obj = exc_type_val
            .as_ref()
            .map(|v| v.concrete.to_pyobj())
            .unwrap_or(std::ptr::null_mut());

        // pyjitpl.py:1682 rclass.ll_isinstance: isinstance check against
        // last_exc_value. Must match eval.rs check_exc_match exactly so
        // the concrete trace path is correct (multi-except blocks).
        let last_exc = self.sym().last_exc_value;
        let matched = if !last_exc.is_null() && !exc_type_obj.is_null() {
            unsafe {
                if !pyre_object::is_exception(last_exc) {
                    true
                } else {
                    let kind = pyre_object::w_exception_get_kind(last_exc);
                    if pyre_object::is_str(exc_type_obj) {
                        let type_name = pyre_object::w_str_get_value(exc_type_obj);
                        pyre_object::exc_kind_matches(kind, type_name)
                    } else if pyre_interpreter::is_function(exc_type_obj)
                        && pyre_interpreter::is_builtin_code(
                            pyre_interpreter::getcode(exc_type_obj) as pyre_object::PyObjectRef,
                        )
                    {
                        let type_name = pyre_interpreter::function_get_name(exc_type_obj);
                        pyre_object::exc_kind_matches(kind, type_name)
                    } else {
                        true // unrecognized type format → match
                    }
                }
            }
        } else {
            true // fallback: assume match if no concrete info
        };

        let result_obj = pyre_object::w_bool_from(matched);
        let result_opref = self.with_ctx(|_this, ctx| ctx.const_ref(result_obj as i64));
        <Self as SharedOpcodeHandler>::push_value(
            self,
            FrontendOp::new(result_opref, ConcreteValue::Ref(result_obj)),
        )?;
        Ok(())
    }

    /// opimpl_raise (pyjitpl.py:1688).
    fn raise_varargs(&mut self, argc: usize) -> Result<(), Self::Error> {
        if argc == 0 {
            return Err(PyError::runtime_error("bare raise during tracing"));
        }
        let exc_val = <Self as SharedOpcodeHandler>::pop_value(self)?;
        if argc >= 2 {
            let _cause = <Self as SharedOpcodeHandler>::pop_value(self).ok();
        }
        // pyjitpl.py:1688-1693 opimpl_raise:
        //   if not heapcache.is_class_known(exc_value_box):
        //       clsbox = cls_of_box(exc_value_box)
        //       generate_guard(GUARD_CLASS, exc_value_box, clsbox, resumepc=orgpc)
        let concrete_exc = exc_val.concrete.to_pyobj();
        if !concrete_exc.is_null() {
            let exc_class_ptr = unsafe {
                (*(concrete_exc as *const pyre_object::excobject::W_ExceptionObject))
                    .ob_header
                    .ob_type
            };
            self.with_ctx(|this, ctx| {
                // pyjitpl.py:1690-1693: generate_guard(GUARD_CLASS,
                // exc_value_box, clsbox, resumepc=orgpc).
                if !ctx.heap_cache().is_class_known(exc_val.opref) {
                    let cls_const = ctx.const_int(exc_class_ptr as usize as i64);
                    this.record_guard(ctx, OpCode::GuardClass, &[exc_val.opref, cls_const]);
                    ctx.heap_cache_mut()
                        .class_now_known(exc_val.opref, majit_ir::GcRef(exc_class_ptr as usize));
                }
            });
        }
        // RPython pyjitpl.py:2745 execute_ll_raised: store concrete
        // exception for handle_possible_exception / check_exc_match.
        {
            let s = self.sym_mut();
            s.last_exc_value = concrete_exc;
            s.class_of_last_exc_is_const = true;
        }
        Err(PyError::value_error("raised during tracing"))
    }

    /// RPython opimpl_reraise (pyjitpl.py:1701).
    fn reraise(&mut self) -> Result<(), Self::Error> {
        Err(PyError::runtime_error("reraise during tracing"))
    }

    fn unsupported(
        &mut self,
        instruction: &Instruction,
    ) -> Result<pyre_interpreter::StepResult<FrontendOp>, Self::Error> {
        Err(PyError::type_error(format!(
            "unsupported instruction during trace: {instruction:?}"
        )))
    }
}
