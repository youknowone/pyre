use pyre_bytecode::bytecode::{
    BinaryOperator, CodeObject, CodeUnit, ComparisonOperator, ConstantData, Instruction,
    IntrinsicFunction1, IntrinsicFunction2, OpArg, OpArgState,
};

use crate::{
    PyBigInt, PyError, SharedOpcodeHandler, exec_build_list, exec_build_map, exec_build_tuple,
    exec_call, exec_list_append, exec_load_attr, exec_make_function, exec_store_attr,
    exec_store_subscr, exec_unpack_sequence,
};

pub enum StepResult<V> {
    Continue,
    Return(V),
    CloseLoop(Vec<V>),
    Yield(V),
}

pub fn decode_instruction_at(code: &CodeObject, pc: usize) -> Option<(Instruction, OpArg)> {
    let code_unit = *code.instructions.get(pc)?;
    let mut start = pc;
    while start > 0 {
        let prev = code.instructions[start - 1];
        if matches!(prev.op, Instruction::ExtendedArg) {
            start -= 1;
        } else {
            break;
        }
    }

    let mut arg_state = OpArgState::default();
    for idx in start..pc {
        let _ = arg_state.get(code.instructions[idx]);
    }
    Some(arg_state.get(code_unit))
}

pub trait LocalOpcodeHandler: SharedOpcodeHandler {
    fn load_local_value(&mut self, idx: usize) -> Result<Self::Value, PyError>;
    fn load_local_checked_value(&mut self, idx: usize, name: &str) -> Result<Self::Value, PyError> {
        let _ = name;
        let value = self.load_local_value(idx)?;
        self.guard_nonnull_value(value)?;
        Ok(value)
    }
    fn store_local_value(&mut self, idx: usize, value: Self::Value) -> Result<(), PyError>;
}

pub trait NamespaceOpcodeHandler: SharedOpcodeHandler {
    fn load_name_value(&mut self, name: &str) -> Result<Self::Value, PyError>;
    fn load_name_checked_value(&mut self, name: &str) -> Result<Self::Value, PyError> {
        let value = self.load_name_value(name)?;
        self.guard_nonnull_value(value)?;
        Ok(value)
    }
    fn store_name_value(&mut self, name: &str, value: Self::Value) -> Result<(), PyError>;
    fn null_value(&mut self) -> Result<Self::Value, PyError>;
}

pub trait StackOpcodeHandler: SharedOpcodeHandler {
    fn swap_values(&mut self, depth: usize) -> Result<(), PyError>;
}

pub trait IterOpcodeHandler: SharedOpcodeHandler {
    fn ensure_iter_value(&mut self, iter: Self::Value) -> Result<(), PyError>;
    fn concrete_iter_continues(&mut self, iter: Self::Value) -> Result<bool, PyError>;
    fn iter_next_value(&mut self, iter: Self::Value) -> Result<Self::Value, PyError>;
    fn guard_optional_value(
        &mut self,
        value: Self::Value,
        expect_some: bool,
    ) -> Result<(), PyError> {
        let _ = (value, expect_some);
        Ok(())
    }

    fn record_for_iter_guard(&mut self, next: Self::Value, continues: bool) -> Result<(), PyError> {
        self.guard_optional_value(next, continues)
    }

    fn on_iter_exhausted(&mut self, target: usize) -> Result<(), PyError>;
}

pub trait TruthOpcodeHandler: SharedOpcodeHandler {
    type Truth: Copy;

    fn truth_value(&mut self, value: Self::Value) -> Result<Self::Truth, PyError>;
    fn bool_value_from_truth(
        &mut self,
        truth: Self::Truth,
        negate: bool,
    ) -> Result<Self::Value, PyError>;
}

pub trait ControlFlowOpcodeHandler: SharedOpcodeHandler {
    fn fallthrough_target(&mut self) -> usize;
    fn set_next_instr(&mut self, target: usize) -> Result<(), PyError>;
    fn close_loop_args(&mut self, _target: usize) -> Result<Option<Vec<Self::Value>>, PyError> {
        Ok(None)
    }

    fn close_loop(&mut self, target: usize) -> Result<StepResult<Self::Value>, PyError> {
        match self.close_loop_args(target)? {
            Some(args) => Ok(StepResult::CloseLoop(args)),
            None => Ok(StepResult::Continue),
        }
    }

    fn finish_value(&mut self, value: Self::Value) -> Result<StepResult<Self::Value>, PyError> {
        Ok(StepResult::Return(value))
    }
}

pub trait BranchOpcodeHandler: TruthOpcodeHandler + ControlFlowOpcodeHandler {
    fn concrete_truth_as_bool(&mut self, truth: Self::Truth) -> Result<bool, PyError>;
    fn guard_truth_value(&mut self, truth: Self::Truth, expect_true: bool) -> Result<(), PyError> {
        let _ = (truth, expect_true);
        Ok(())
    }

    fn record_branch_guard(
        &mut self,
        truth: Self::Truth,
        concrete_truth: bool,
    ) -> Result<(), PyError> {
        self.guard_truth_value(truth, concrete_truth)
    }
}

pub trait ArithmeticOpcodeHandler: SharedOpcodeHandler {
    fn binary_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: BinaryOperator,
    ) -> Result<Self::Value, PyError>;
    fn compare_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: ComparisonOperator,
    ) -> Result<Self::Value, PyError>;
    fn unary_negative_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError>;
    fn unary_invert_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError>;
}

pub trait ConstantOpcodeHandler: SharedOpcodeHandler {
    fn int_constant(&mut self, value: i64) -> Result<Self::Value, PyError>;
    fn bigint_constant(&mut self, value: &PyBigInt) -> Result<Self::Value, PyError>;
    fn float_constant(&mut self, value: f64) -> Result<Self::Value, PyError>;
    fn bool_constant(&mut self, value: bool) -> Result<Self::Value, PyError>;
    fn str_constant(&mut self, value: &str) -> Result<Self::Value, PyError>;
    fn code_constant(&mut self, code: &CodeObject) -> Result<Self::Value, PyError>;
    fn none_constant(&mut self) -> Result<Self::Value, PyError>;
}

fn load_const_value<H: ConstantOpcodeHandler + ?Sized>(
    handler: &mut H,
    constant: &ConstantData,
) -> Result<H::Value, PyError> {
    match constant {
        ConstantData::Integer { value } => {
            use num_traits::ToPrimitive;
            match value.to_i64() {
                Some(value) => handler.int_constant(value),
                None => handler.bigint_constant(value),
            }
        }
        ConstantData::Float { value } => handler.float_constant(*value),
        ConstantData::Boolean { value } => handler.bool_constant(*value),
        ConstantData::Str { value } => {
            handler.str_constant(value.as_str().expect("non-UTF-8 string constant"))
        }
        ConstantData::Tuple { elements } => {
            let mut items = Vec::with_capacity(elements.len());
            for element in elements {
                items.push(load_const_value(handler, element)?);
            }
            handler.build_tuple(&items)
        }
        ConstantData::Code { code } => handler.code_constant(code),
        ConstantData::None => handler.none_constant(),
        _ => Err(PyError::type_error("unsupported constant")),
    }
}

pub fn exec_load_const<H: ConstantOpcodeHandler + ?Sized>(
    handler: &mut H,
    constant: &ConstantData,
) -> Result<(), PyError> {
    let value = load_const_value(handler, constant)?;
    handler.push_value(value)
}

pub fn exec_load_small_int<H: ConstantOpcodeHandler + ?Sized>(
    handler: &mut H,
    value: i64,
) -> Result<(), PyError> {
    let value = handler.int_constant(value)?;
    handler.push_value(value)
}

pub fn exec_load_fast_checked<H: LocalOpcodeHandler + ?Sized>(
    handler: &mut H,
    idx: usize,
    name: &str,
) -> Result<(), PyError> {
    let value = handler.load_local_checked_value(idx, name)?;
    handler.push_value(value)
}

pub fn exec_load_fast_pair_checked<H: LocalOpcodeHandler + ?Sized>(
    handler: &mut H,
    idx1: usize,
    name1: &str,
    idx2: usize,
    name2: &str,
) -> Result<(), PyError> {
    let v1 = handler.load_local_checked_value(idx1, name1)?;
    let v2 = handler.load_local_checked_value(idx2, name2)?;
    handler.push_value(v1)?;
    handler.push_value(v2)
}

pub fn exec_store_fast<H: LocalOpcodeHandler + ?Sized>(
    handler: &mut H,
    idx: usize,
) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    handler.store_local_value(idx, value)
}

pub fn exec_load_fast_load_fast<H: LocalOpcodeHandler + ?Sized>(
    handler: &mut H,
    idx1: usize,
    idx2: usize,
) -> Result<(), PyError> {
    let v1 = handler.load_local_value(idx1)?;
    let v2 = handler.load_local_value(idx2)?;
    handler.push_value(v1)?;
    handler.push_value(v2)
}

pub fn exec_store_fast_load_fast<H: LocalOpcodeHandler + ?Sized>(
    handler: &mut H,
    store_idx: usize,
    load_idx: usize,
) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    handler.store_local_value(store_idx, value)?;
    let loaded = handler.load_local_value(load_idx)?;
    handler.push_value(loaded)
}

pub fn exec_store_fast_store_fast<H: LocalOpcodeHandler + ?Sized>(
    handler: &mut H,
    idx1: usize,
    idx2: usize,
) -> Result<(), PyError> {
    let v1 = handler.pop_value()?;
    let v2 = handler.pop_value()?;
    handler.store_local_value(idx1, v1)?;
    handler.store_local_value(idx2, v2)
}

pub fn exec_store_name<H: NamespaceOpcodeHandler + ?Sized>(
    handler: &mut H,
    name: &str,
) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    handler.store_name_value(name, value)
}

pub fn exec_load_name<H: NamespaceOpcodeHandler + ?Sized>(
    handler: &mut H,
    name: &str,
) -> Result<(), PyError> {
    let value = handler.load_name_checked_value(name)?;
    handler.push_value(value)
}

pub fn exec_load_global<H: NamespaceOpcodeHandler + ?Sized>(
    handler: &mut H,
    name: &str,
    push_null: bool,
) -> Result<(), PyError> {
    exec_load_name(handler, name)?;
    if push_null {
        let null = handler.null_value()?;
        handler.push_value(null)?;
    }
    Ok(())
}

pub fn exec_pop_top<H: SharedOpcodeHandler + ?Sized>(handler: &mut H) -> Result<(), PyError> {
    let _ = handler.pop_value()?;
    Ok(())
}

pub fn exec_push_null<H: NamespaceOpcodeHandler + ?Sized>(handler: &mut H) -> Result<(), PyError> {
    let null = handler.null_value()?;
    handler.push_value(null)
}

pub fn exec_copy_value<H: SharedOpcodeHandler + ?Sized>(
    handler: &mut H,
    depth: usize,
) -> Result<(), PyError> {
    let value = handler.peek_at(depth - 1)?;
    handler.push_value(value)
}

pub fn exec_swap<H: StackOpcodeHandler + ?Sized>(
    handler: &mut H,
    depth: usize,
) -> Result<(), PyError> {
    handler.swap_values(depth)
}

pub fn exec_get_iter<H: IterOpcodeHandler + ?Sized>(handler: &mut H) -> Result<(), PyError> {
    let iter = handler.peek_at(0)?;
    handler.ensure_iter_value(iter)
}

pub fn exec_for_iter<H: IterOpcodeHandler + ControlFlowOpcodeHandler + ?Sized>(
    handler: &mut H,
    target: usize,
) -> Result<(), PyError> {
    let iter = handler.peek_at(0)?;
    let continues = handler.concrete_iter_continues(iter)?;
    let next = handler.iter_next_value(iter)?;
    if continues {
        let fallthrough = handler.fallthrough_target();
        // On guard failure this bytecode exits through the exhaustion path.
        handler.set_next_instr(target)?;
        handler.record_for_iter_guard(next, true)?;
        handler.set_next_instr(fallthrough)?;
        handler.push_value(next)
    } else {
        handler.record_for_iter_guard(next, false)?;
        handler.on_iter_exhausted(target)
    }
}

pub fn exec_unary_not<H: TruthOpcodeHandler + ?Sized>(handler: &mut H) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    let truth = handler.truth_value(value)?;
    let result = handler.bool_value_from_truth(truth, true)?;
    handler.push_value(result)
}

pub fn exec_binary_op<H: ArithmeticOpcodeHandler + ?Sized>(
    handler: &mut H,
    op: BinaryOperator,
) -> Result<(), PyError> {
    let b = handler.pop_value()?;
    let a = handler.pop_value()?;
    let result = handler.binary_value(a, b, op)?;
    handler.push_value(result)
}

pub fn exec_compare_op<H: ArithmeticOpcodeHandler + ?Sized>(
    handler: &mut H,
    op: ComparisonOperator,
) -> Result<(), PyError> {
    let b = handler.pop_value()?;
    let a = handler.pop_value()?;
    let result = handler.compare_value(a, b, op)?;
    handler.push_value(result)
}

pub fn exec_unary_negative<H: ArithmeticOpcodeHandler + ?Sized>(
    handler: &mut H,
) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    let result = handler.unary_negative_value(value)?;
    handler.push_value(result)
}

pub fn exec_unary_invert<H: ArithmeticOpcodeHandler + ?Sized>(
    handler: &mut H,
) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    let result = handler.unary_invert_value(value)?;
    handler.push_value(result)
}

fn exec_pop_jump_if<H: BranchOpcodeHandler + ?Sized>(
    handler: &mut H,
    target: usize,
    jump_if_true: bool,
) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    let truth = handler.truth_value(value)?;
    let concrete_truth = handler.concrete_truth_as_bool(truth)?;
    let should_jump = concrete_truth == jump_if_true;
    let fallthrough = handler.fallthrough_target();
    if !should_jump {
        handler.set_next_instr(target)?;
    }
    handler.record_branch_guard(truth, concrete_truth)?;
    let next_target = if should_jump { target } else { fallthrough };
    handler.set_next_instr(next_target)
}

pub fn exec_pop_jump_if_false<H: BranchOpcodeHandler + ?Sized>(
    handler: &mut H,
    target: usize,
) -> Result<(), PyError> {
    exec_pop_jump_if(handler, target, false)
}

pub fn exec_pop_jump_if_true<H: BranchOpcodeHandler + ?Sized>(
    handler: &mut H,
    target: usize,
) -> Result<(), PyError> {
    exec_pop_jump_if(handler, target, true)
}

pub fn exec_jump_forward<H: ControlFlowOpcodeHandler + ?Sized>(
    handler: &mut H,
    target: usize,
) -> Result<(), PyError> {
    handler.set_next_instr(target)
}

pub fn exec_jump_backward<H: ControlFlowOpcodeHandler + ?Sized>(
    handler: &mut H,
    target: usize,
) -> Result<StepResult<H::Value>, PyError> {
    handler.set_next_instr(target)?;
    handler.close_loop(target)
}

pub fn exec_return_value<H: ControlFlowOpcodeHandler + ?Sized>(
    handler: &mut H,
) -> Result<StepResult<H::Value>, PyError> {
    let value = handler.pop_value()?;
    handler.finish_value(value)
}

pub trait OpcodeStepExecutor: SharedOpcodeHandler {
    type Error: From<PyError>;

    fn load_const(&mut self, constant: &ConstantData) -> Result<(), Self::Error>
    where
        Self: ConstantOpcodeHandler,
    {
        exec_load_const(self, constant).map_err(Into::into)
    }

    fn load_small_int(&mut self, value: i64) -> Result<(), Self::Error>
    where
        Self: ConstantOpcodeHandler,
    {
        exec_load_small_int(self, value).map_err(Into::into)
    }

    fn load_fast_checked(&mut self, idx: usize, name: &str) -> Result<(), Self::Error>
    where
        Self: LocalOpcodeHandler,
    {
        exec_load_fast_checked(self, idx, name).map_err(Into::into)
    }

    fn load_fast_pair_checked(
        &mut self,
        idx1: usize,
        name1: &str,
        idx2: usize,
        name2: &str,
    ) -> Result<(), Self::Error>
    where
        Self: LocalOpcodeHandler,
    {
        exec_load_fast_pair_checked(self, idx1, name1, idx2, name2).map_err(Into::into)
    }

    fn store_fast(&mut self, idx: usize) -> Result<(), Self::Error>
    where
        Self: LocalOpcodeHandler,
    {
        exec_store_fast(self, idx).map_err(Into::into)
    }

    fn load_fast_load_fast(&mut self, idx1: usize, idx2: usize) -> Result<(), Self::Error>
    where
        Self: LocalOpcodeHandler,
    {
        exec_load_fast_load_fast(self, idx1, idx2).map_err(Into::into)
    }

    fn store_fast_load_fast(&mut self, store_idx: usize, load_idx: usize) -> Result<(), Self::Error>
    where
        Self: LocalOpcodeHandler,
    {
        exec_store_fast_load_fast(self, store_idx, load_idx).map_err(Into::into)
    }

    fn store_fast_store_fast(&mut self, idx1: usize, idx2: usize) -> Result<(), Self::Error>
    where
        Self: LocalOpcodeHandler,
    {
        exec_store_fast_store_fast(self, idx1, idx2).map_err(Into::into)
    }

    fn store_name(&mut self, name: &str) -> Result<(), Self::Error>
    where
        Self: NamespaceOpcodeHandler,
    {
        exec_store_name(self, name).map_err(Into::into)
    }

    fn load_name(&mut self, name: &str) -> Result<(), Self::Error>
    where
        Self: NamespaceOpcodeHandler,
    {
        exec_load_name(self, name).map_err(Into::into)
    }

    fn load_global(&mut self, name: &str, push_null: bool) -> Result<(), Self::Error>
    where
        Self: NamespaceOpcodeHandler,
    {
        exec_load_global(self, name, push_null).map_err(Into::into)
    }

    fn pop_top(&mut self) -> Result<(), Self::Error>
    where
        Self: SharedOpcodeHandler,
    {
        exec_pop_top(self).map_err(Into::into)
    }

    fn push_null(&mut self) -> Result<(), Self::Error>
    where
        Self: NamespaceOpcodeHandler,
    {
        exec_push_null(self).map_err(Into::into)
    }

    fn copy_value(&mut self, depth: usize) -> Result<(), Self::Error>
    where
        Self: SharedOpcodeHandler,
    {
        exec_copy_value(self, depth).map_err(Into::into)
    }

    fn swap(&mut self, depth: usize) -> Result<(), Self::Error>
    where
        Self: StackOpcodeHandler,
    {
        exec_swap(self, depth).map_err(Into::into)
    }

    fn binary_op(&mut self, op: BinaryOperator) -> Result<(), Self::Error>
    where
        Self: ArithmeticOpcodeHandler,
    {
        exec_binary_op(self, op).map_err(Into::into)
    }

    fn compare_op(&mut self, op: ComparisonOperator) -> Result<(), Self::Error>
    where
        Self: ArithmeticOpcodeHandler,
    {
        exec_compare_op(self, op).map_err(Into::into)
    }

    fn unary_negative(&mut self) -> Result<(), Self::Error>
    where
        Self: ArithmeticOpcodeHandler,
    {
        exec_unary_negative(self).map_err(Into::into)
    }

    fn unary_not(&mut self) -> Result<(), Self::Error>
    where
        Self: TruthOpcodeHandler,
    {
        exec_unary_not(self).map_err(Into::into)
    }

    fn unary_invert(&mut self) -> Result<(), Self::Error>
    where
        Self: ArithmeticOpcodeHandler,
    {
        exec_unary_invert(self).map_err(Into::into)
    }

    fn jump_forward(&mut self, target: usize) -> Result<(), Self::Error>
    where
        Self: ControlFlowOpcodeHandler,
    {
        exec_jump_forward(self, target).map_err(Into::into)
    }

    fn jump_backward(
        &mut self,
        target: usize,
    ) -> Result<StepResult<<Self as SharedOpcodeHandler>::Value>, Self::Error>
    where
        Self: ControlFlowOpcodeHandler,
    {
        exec_jump_backward(self, target).map_err(Into::into)
    }

    fn pop_jump_if_false(&mut self, target: usize) -> Result<(), Self::Error>
    where
        Self: BranchOpcodeHandler,
    {
        exec_pop_jump_if_false(self, target).map_err(Into::into)
    }

    fn pop_jump_if_true(&mut self, target: usize) -> Result<(), Self::Error>
    where
        Self: BranchOpcodeHandler,
    {
        exec_pop_jump_if_true(self, target).map_err(Into::into)
    }

    fn make_function(&mut self) -> Result<(), Self::Error>
    where
        Self: SharedOpcodeHandler,
    {
        exec_make_function(self).map_err(Into::into)
    }

    fn call(&mut self, nargs: usize) -> Result<(), Self::Error>
    where
        Self: SharedOpcodeHandler,
    {
        exec_call(self, nargs).map_err(Into::into)
    }

    fn return_value(
        &mut self,
    ) -> Result<StepResult<<Self as SharedOpcodeHandler>::Value>, Self::Error>
    where
        Self: ControlFlowOpcodeHandler,
    {
        exec_return_value(self).map_err(Into::into)
    }

    fn build_list(&mut self, size: usize) -> Result<(), Self::Error>
    where
        Self: SharedOpcodeHandler,
    {
        exec_build_list(self, size).map_err(Into::into)
    }

    fn build_tuple(&mut self, size: usize) -> Result<(), Self::Error>
    where
        Self: SharedOpcodeHandler,
    {
        exec_build_tuple(self, size).map_err(Into::into)
    }

    fn build_map(&mut self, size: usize) -> Result<(), Self::Error>
    where
        Self: SharedOpcodeHandler,
    {
        exec_build_map(self, size).map_err(Into::into)
    }

    fn store_subscr(&mut self) -> Result<(), Self::Error>
    where
        Self: SharedOpcodeHandler,
    {
        exec_store_subscr(self).map_err(Into::into)
    }

    fn list_append(&mut self, depth: usize) -> Result<(), Self::Error>
    where
        Self: SharedOpcodeHandler,
    {
        exec_list_append(self, depth).map_err(Into::into)
    }

    fn unpack_sequence(&mut self, count: usize) -> Result<(), Self::Error>
    where
        Self: SharedOpcodeHandler,
    {
        exec_unpack_sequence(self, count).map_err(Into::into)
    }

    fn load_attr(&mut self, name: &str) -> Result<(), Self::Error>
    where
        Self: SharedOpcodeHandler,
    {
        exec_load_attr(self, name).map_err(Into::into)
    }

    fn store_attr(&mut self, name: &str) -> Result<(), Self::Error>
    where
        Self: SharedOpcodeHandler,
    {
        exec_store_attr(self, name).map_err(Into::into)
    }

    fn get_iter(&mut self) -> Result<(), Self::Error>
    where
        Self: IterOpcodeHandler,
    {
        exec_get_iter(self).map_err(Into::into)
    }

    fn for_iter(&mut self, target: usize) -> Result<(), Self::Error>
    where
        Self: IterOpcodeHandler + ControlFlowOpcodeHandler,
    {
        exec_for_iter(self, target).map_err(Into::into)
    }

    fn end_for(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn pop_iter(&mut self) -> Result<(), Self::Error>
    where
        Self: SharedOpcodeHandler,
    {
        exec_pop_top(self).map_err(Into::into)
    }

    // ── Closures / cells ──
    fn load_deref(&mut self, _idx: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("load_deref not implemented").into())
    }
    fn store_deref(&mut self, _idx: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("store_deref not implemented").into())
    }
    fn load_closure(&mut self, _idx: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("load_closure not implemented").into())
    }
    fn delete_deref(&mut self, _idx: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("delete_deref not implemented").into())
    }

    // ── Exception handling ──
    fn setup_finally(&mut self, _handler: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("setup_finally not implemented").into())
    }
    fn setup_except(&mut self, _handler: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("setup_except not implemented").into())
    }
    fn pop_block(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("pop_block not implemented").into())
    }
    fn raise_varargs(&mut self, _argc: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("raise_varargs not implemented").into())
    }
    fn end_finally(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("end_finally not implemented").into())
    }
    fn exception_handler(&mut self) -> Result<(), Self::Error> {
        Ok(()) // no-op by default
    }

    // ── Import ──
    fn import_name(&mut self, _name: &str) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("import_name not implemented").into())
    }
    fn import_from(&mut self, _name: &str) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("import_from not implemented").into())
    }
    fn import_star(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("import_star not implemented").into())
    }

    // ── Stack manipulation ──
    fn rotate3(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("rotate3 not implemented").into())
    }

    // ── Delete operations ──
    fn delete_fast(&mut self, _idx: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("delete_fast not implemented").into())
    }
    fn delete_subscript(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("delete_subscript not implemented").into())
    }
    fn delete_attr(&mut self, _name: &str) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("delete_attr not implemented").into())
    }
    fn delete_name(&mut self, _name: &str) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("delete_name not implemented").into())
    }
    fn delete_global(&mut self, _name: &str) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("delete_global not implemented").into())
    }

    // Containment / identity
    fn contains_op(&mut self, _invert: pyre_bytecode::bytecode::Invert) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("contains_op not implemented").into())
    }
    fn is_op(&mut self, _invert: pyre_bytecode::bytecode::Invert) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("is_op not implemented").into())
    }

    // Exception handling
    fn push_exc_info(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn pop_except(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn check_exc_match(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("check_exc_match not implemented").into())
    }
    fn reraise(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("reraise not implemented").into())
    }

    // Collections
    fn build_set(&mut self, _count: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("build_set not implemented").into())
    }
    fn build_slice(
        &mut self,
        _argc: pyre_bytecode::bytecode::BuildSliceArgCount,
    ) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("build_slice not implemented").into())
    }
    fn build_string(&mut self, _count: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("build_string not implemented").into())
    }
    fn list_extend(&mut self, _i: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("list_extend not implemented").into())
    }
    fn set_add(&mut self, _i: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("set_add not implemented").into())
    }
    fn dict_merge(&mut self, _i: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("dict_merge not implemented").into())
    }
    fn dict_update(&mut self, _i: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("dict_update not implemented").into())
    }
    fn set_update(&mut self, _i: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("set_update not implemented").into())
    }
    fn map_add(&mut self, _i: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("map_add not implemented").into())
    }

    // Slicing
    fn binary_slice(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("binary_slice not implemented").into())
    }
    fn store_slice(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("store_slice not implemented").into())
    }

    // Boolean
    fn to_bool(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("to_bool not implemented").into())
    }

    // None jumps
    fn pop_jump_if_none(&mut self, _target: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("pop_jump_if_none not implemented").into())
    }
    fn pop_jump_if_not_none(&mut self, _target: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("pop_jump_if_not_none not implemented").into())
    }

    // Closures 3.11+
    fn copy_free_vars(&mut self, _count: usize) -> Result<(), Self::Error> {
        Ok(())
    }
    fn return_generator(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("return_generator not implemented").into())
    }

    // Call variants
    fn call_kw(&mut self, _argc: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("call_kw not implemented").into())
    }
    fn call_function_ex(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("call_function_ex not implemented").into())
    }

    // Class
    fn load_build_class(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("load_build_class not implemented").into())
    }
    fn load_super_attr(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("load_super_attr not implemented").into())
    }
    fn load_locals(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("load_locals not implemented").into())
    }

    // String formatting
    fn format_simple(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("format_simple not implemented").into())
    }
    fn format_with_spec(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("format_with_spec not implemented").into())
    }
    fn convert_value(
        &mut self,
        _conv: pyre_bytecode::bytecode::ConvertValueOparg,
    ) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("convert_value not implemented").into())
    }

    fn get_len(&mut self, _obj: <Self as SharedOpcodeHandler>::Value) -> Result<<Self as SharedOpcodeHandler>::Value, Self::Error> {
        Err(crate::PyError::type_error("get_len not implemented").into())
    }
    fn load_fast_and_clear(&mut self, _idx: usize) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("load_fast_and_clear not implemented").into())
    }
    fn set_function_attribute(&mut self) -> Result<(), Self::Error> {
        let _attr = self.pop_value().map_err(Into::into)?;
        Ok(())
    }
    fn load_from_dict_or_globals(&mut self, _name: &str) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("load_from_dict_or_globals not implemented").into())
    }
    fn load_from_dict_or_deref(&mut self, _idx: usize, _name: &str) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("load_from_dict_or_deref not implemented").into())
    }
    fn match_stub(&mut self) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("pattern matching not implemented").into())
    }
    fn unpack_ex(&mut self, _args: pyre_bytecode::bytecode::UnpackExArgs) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("unpack_ex not implemented").into())
    }

    /// CALL_INTRINSIC_1: single-argument intrinsic operations.
    fn call_intrinsic_1(
        &mut self,
        func: IntrinsicFunction1,
    ) -> Result<(), Self::Error> {
        match func {
            IntrinsicFunction1::UnaryPositive => {
                // PyPy: UNARY_POSITIVE → space.pos(w_value)
                let val = self.pop_value().map_err(Into::into)?;
                let result = self.unary_positive(val)?;
                self.push_value(result).map_err(Into::into)?;
                Ok(())
            }
            IntrinsicFunction1::ListToTuple => {
                let val = self.pop_value().map_err(Into::into)?;
                let result = self.list_to_tuple(val)?;
                self.push_value(result).map_err(Into::into)?;
                Ok(())
            }
            IntrinsicFunction1::ImportStar => {
                // Module is TOS; import_star pops it internally
                self.import_star()?;
                let none = self.none_value()?;
                self.push_value(none).map_err(Into::into)?;
                Ok(())
            }
            IntrinsicFunction1::Print => {
                // sys.displayhook(value)
                let val = self.pop_value().map_err(Into::into)?;
                self.print_expr(val)?;
                let none = self.none_value()?;
                self.push_value(none).map_err(Into::into)?;
                Ok(())
            }
            _ => Err(crate::PyError::type_error(
                &format!("intrinsic function {:?} not implemented", func),
            ).into()),
        }
    }

    /// CALL_INTRINSIC_2: two-argument intrinsic operations.
    fn call_intrinsic_2(
        &mut self,
        func: IntrinsicFunction2,
    ) -> Result<(), Self::Error> {
        match func {
            IntrinsicFunction2::SetFunctionTypeParams => {
                // arg2 = type_params, arg1 = function
                // Set __type_params__ attribute on the function; push function back
                let _type_params = self.pop_value().map_err(Into::into)?;
                // just leave the function on the stack
                Ok(())
            }
            _ => Err(crate::PyError::type_error(
                &format!("intrinsic function {:?} not implemented", func),
            ).into()),
        }
    }

    // ── Intrinsic helper methods ──
    fn unary_positive(&mut self, _val: <Self as SharedOpcodeHandler>::Value) -> Result<<Self as SharedOpcodeHandler>::Value, Self::Error> {
        Err(crate::PyError::type_error("unary_positive not implemented").into())
    }
    fn list_to_tuple(&mut self, _val: <Self as SharedOpcodeHandler>::Value) -> Result<<Self as SharedOpcodeHandler>::Value, Self::Error> {
        Err(crate::PyError::type_error("list_to_tuple not implemented").into())
    }
    fn print_expr(&mut self, _val: <Self as SharedOpcodeHandler>::Value) -> Result<(), Self::Error> {
        Err(crate::PyError::type_error("print_expr not implemented").into())
    }
    fn none_value(&mut self) -> Result<<Self as SharedOpcodeHandler>::Value, Self::Error> {
        Err(crate::PyError::type_error("none_value not implemented").into())
    }

    fn unsupported(
        &mut self,
        instruction: &Instruction,
    ) -> Result<StepResult<<Self as SharedOpcodeHandler>::Value>, Self::Error>;
}

pub fn execute_opcode_step<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
    next_instr: usize,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, E::Error>
where
    E: SharedOpcodeHandler
        + ConstantOpcodeHandler
        + LocalOpcodeHandler
        + NamespaceOpcodeHandler
        + StackOpcodeHandler
        + IterOpcodeHandler
        + TruthOpcodeHandler
        + ControlFlowOpcodeHandler
        + BranchOpcodeHandler
        + ArithmeticOpcodeHandler,
{
    match instruction {
        Instruction::ExtendedArg
        | Instruction::Resume { .. }
        | Instruction::Nop
        | Instruction::Cache
        | Instruction::NotTaken => Ok(StepResult::Continue),

        Instruction::LoadConst { consti } => {
            let const_idx = consti.get(op_arg);
            executor.load_const(&code.constants[const_idx])?;
            Ok(StepResult::Continue)
        }

        Instruction::LoadSmallInt { i } => {
            executor.load_small_int(i.get(op_arg) as i64)?;
            Ok(StepResult::Continue)
        }

        Instruction::LoadFast { var_num } | Instruction::LoadFastBorrow { var_num } => {
            let idx = var_num.get(op_arg).as_usize();
            let name = code.varnames.get(idx).map(|s| s.as_ref()).unwrap_or("<cell>");
            executor.load_fast_checked(idx, name)?;
            Ok(StepResult::Continue)
        }

        Instruction::LoadFastBorrowLoadFastBorrow { var_nums } => {
            let pair = var_nums.get(op_arg);
            let idx1 = u32::from(pair.idx_1()) as usize;
            let idx2 = u32::from(pair.idx_2()) as usize;
            executor.load_fast_pair_checked(
                idx1,
                code.varnames[idx1].as_ref(),
                idx2,
                code.varnames[idx2].as_ref(),
            )?;
            Ok(StepResult::Continue)
        }

        Instruction::StoreFast { var_num } => {
            executor.store_fast(var_num.get(op_arg).as_usize())?;
            Ok(StepResult::Continue)
        }

        Instruction::LoadFastCheck { var_num } => {
            let idx = var_num.get(op_arg).as_usize();
            let name = code.varnames.get(idx).map(|s| s.as_ref()).unwrap_or("<cell>");
            executor.load_fast_checked(idx, name)?;
            Ok(StepResult::Continue)
        }

        Instruction::LoadFastLoadFast { var_nums } => {
            let pair = var_nums.get(op_arg);
            let idx1 = u32::from(pair.idx_1()) as usize;
            let idx2 = u32::from(pair.idx_2()) as usize;
            executor.load_fast_load_fast(idx1, idx2)?;
            Ok(StepResult::Continue)
        }

        Instruction::StoreFastLoadFast { var_nums } => {
            let pair = var_nums.get(op_arg);
            executor.store_fast_load_fast(
                u32::from(pair.idx_1()) as usize,
                u32::from(pair.idx_2()) as usize,
            )?;
            Ok(StepResult::Continue)
        }

        Instruction::StoreFastStoreFast { var_nums } => {
            let pair = var_nums.get(op_arg);
            executor.store_fast_store_fast(
                u32::from(pair.idx_1()) as usize,
                u32::from(pair.idx_2()) as usize,
            )?;
            Ok(StepResult::Continue)
        }

        Instruction::StoreName { namei } | Instruction::StoreGlobal { namei } => {
            let idx = namei.get(op_arg) as usize;
            executor.store_name(code.names[idx].as_ref())?;
            Ok(StepResult::Continue)
        }

        Instruction::LoadName { namei } => {
            let idx = namei.get(op_arg) as usize;
            executor.load_name(code.names[idx].as_ref())?;
            Ok(StepResult::Continue)
        }

        Instruction::LoadGlobal { namei } => {
            let raw = namei.get(op_arg) as usize;
            let name_idx = raw >> 1;
            let push_null = (raw & 1) != 0;
            executor.load_global(code.names[name_idx].as_ref(), push_null)?;
            Ok(StepResult::Continue)
        }

        Instruction::PopTop => {
            executor.pop_top()?;
            Ok(StepResult::Continue)
        }

        Instruction::PushNull => {
            executor.push_null()?;
            Ok(StepResult::Continue)
        }

        Instruction::Copy { i } => {
            executor.copy_value(i.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }

        Instruction::Swap { i } => {
            executor.swap(i.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }

        Instruction::BinaryOp { op } => {
            executor.binary_op(op.get(op_arg))?;
            Ok(StepResult::Continue)
        }

        Instruction::CompareOp { opname } => {
            executor.compare_op(opname.get(op_arg))?;
            Ok(StepResult::Continue)
        }

        Instruction::UnaryNegative => {
            executor.unary_negative()?;
            Ok(StepResult::Continue)
        }

        Instruction::UnaryNot => {
            executor.unary_not()?;
            Ok(StepResult::Continue)
        }

        Instruction::UnaryInvert => {
            executor.unary_invert()?;
            Ok(StepResult::Continue)
        }

        Instruction::JumpForward { delta } => {
            executor.jump_forward(jump_target_forward(
                &code.instructions,
                next_instr,
                delta.get(op_arg).as_usize(),
            ))?;
            Ok(StepResult::Continue)
        }

        Instruction::JumpBackward { delta } => executor.jump_backward(jump_target_backward(
            &code.instructions,
            next_instr,
            delta.get(op_arg).as_usize(),
        )),

        Instruction::PopJumpIfFalse { delta } => {
            executor.pop_jump_if_false(jump_target_forward(
                &code.instructions,
                next_instr,
                delta.get(op_arg).as_usize(),
            ))?;
            Ok(StepResult::Continue)
        }

        Instruction::PopJumpIfTrue { delta } => {
            executor.pop_jump_if_true(jump_target_forward(
                &code.instructions,
                next_instr,
                delta.get(op_arg).as_usize(),
            ))?;
            Ok(StepResult::Continue)
        }

        Instruction::MakeFunction => {
            OpcodeStepExecutor::make_function(executor)?;
            Ok(StepResult::Continue)
        }

        Instruction::Call { argc } => {
            executor.call(argc.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }

        Instruction::ReturnValue => executor.return_value(),

        Instruction::BuildList { count } => {
            OpcodeStepExecutor::build_list(executor, count.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }

        Instruction::BuildTuple { count } => {
            OpcodeStepExecutor::build_tuple(executor, count.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }

        Instruction::BuildMap { count } => {
            OpcodeStepExecutor::build_map(executor, count.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }

        Instruction::StoreSubscr => {
            OpcodeStepExecutor::store_subscr(executor)?;
            Ok(StepResult::Continue)
        }

        Instruction::ListAppend { i } => {
            OpcodeStepExecutor::list_append(executor, i.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }

        Instruction::UnpackSequence { count } => {
            OpcodeStepExecutor::unpack_sequence(executor, count.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }

        Instruction::GetIter => {
            executor.get_iter()?;
            Ok(StepResult::Continue)
        }

        Instruction::ForIter { delta } => {
            executor.for_iter(jump_target_forward(
                &code.instructions,
                next_instr,
                delta.get(op_arg).as_usize(),
            ))?;
            Ok(StepResult::Continue)
        }

        Instruction::EndFor => {
            executor.end_for()?;
            Ok(StepResult::Continue)
        }

        Instruction::PopIter => {
            executor.pop_iter()?;
            Ok(StepResult::Continue)
        }

        Instruction::LoadAttr { namei } => {
            let attr = namei.get(op_arg);
            let name_idx = attr.name_idx() as usize;
            OpcodeStepExecutor::load_attr(executor, code.names[name_idx].as_ref())?;
            Ok(StepResult::Continue)
        }

        Instruction::StoreAttr { namei } => {
            let name_idx = namei.get(op_arg) as usize;
            OpcodeStepExecutor::store_attr(executor, code.names[name_idx].as_ref())?;
            Ok(StepResult::Continue)
        }

        // ── Generators ──
        Instruction::YieldValue { .. } => {
            let value = executor.pop_value()?;
            Ok(StepResult::Yield(value))
        }

        // All other opcodes fall through to unsupported handler.
        // Phase 1 opcodes (closures, exceptions, imports) will be added
        // ── Closures / cells ──
        Instruction::LoadDeref { i } => {
            let idx = i.get(op_arg) as usize;
            executor.load_deref(idx)?;
            Ok(StepResult::Continue)
        }
        Instruction::StoreDeref { i } => {
            let idx = i.get(op_arg) as usize;
            executor.store_deref(idx)?;
            Ok(StepResult::Continue)
        }
        Instruction::DeleteDeref { i } => {
            let idx = i.get(op_arg) as usize;
            executor.delete_deref(idx)?;
            Ok(StepResult::Continue)
        }

        // ── Generators ──
        Instruction::YieldValue { .. } => {
            let value = executor.pop_value()?;
            Ok(StepResult::Yield(value))
        }

        // ── Import ──
        Instruction::ImportName { namei } => {
            let name_idx = namei.get(op_arg) as usize;
            executor.import_name(code.names[name_idx].as_ref())?;
            Ok(StepResult::Continue)
        }
        Instruction::ImportFrom { namei } => {
            let name_idx = namei.get(op_arg) as usize;
            executor.import_from(code.names[name_idx].as_ref())?;
            Ok(StepResult::Continue)
        }

        // ── Containment / identity tests ──
        Instruction::ContainsOp { invert } => {
            let inv = invert.get(op_arg);
            executor.contains_op(inv)?;
            Ok(StepResult::Continue)
        }
        Instruction::IsOp { invert } => {
            let inv = invert.get(op_arg);
            executor.is_op(inv)?;
            Ok(StepResult::Continue)
        }

        // ── Delete subscript ──
        Instruction::DeleteSubscr => {
            executor.delete_subscript()?;
            Ok(StepResult::Continue)
        }

        // ── Exception handling (CPython 3.13) ──
        Instruction::PushExcInfo => {
            executor.push_exc_info()?;
            Ok(StepResult::Continue)
        }
        Instruction::PopExcept => {
            executor.pop_except()?;
            Ok(StepResult::Continue)
        }
        Instruction::CheckExcMatch => {
            executor.check_exc_match()?;
            Ok(StepResult::Continue)
        }
        Instruction::RaiseVarargs { argc } => {
            executor.raise_varargs(argc.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }
        Instruction::Reraise { .. } => {
            executor.reraise()?;
            Ok(StepResult::Continue)
        }

        // ── Collection operations ──
        Instruction::BuildSet { count } => {
            executor.build_set(count.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }
        Instruction::BuildSlice { argc } => {
            executor.build_slice(argc.get(op_arg))?;
            Ok(StepResult::Continue)
        }
        Instruction::BuildString { count } => {
            executor.build_string(count.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }
        Instruction::ListExtend { i } => {
            executor.list_extend(i.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }
        Instruction::SetAdd { i } => {
            executor.set_add(i.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }
        Instruction::DictMerge { i } => {
            executor.dict_merge(i.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }
        Instruction::DictUpdate { i } => {
            executor.dict_update(i.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }
        Instruction::SetUpdate { i } => {
            executor.set_update(i.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }
        Instruction::MapAdd { i } => {
            executor.map_add(i.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }

        // ── Slicing ──
        Instruction::BinarySlice => {
            executor.binary_slice()?;
            Ok(StepResult::Continue)
        }
        Instruction::StoreSlice => {
            executor.store_slice()?;
            Ok(StepResult::Continue)
        }

        // ── Boolean conversion ──
        Instruction::ToBool => {
            executor.to_bool()?;
            Ok(StepResult::Continue)
        }

        // ── None comparison jumps ──
        Instruction::PopJumpIfNone { delta } => {
            let target = u32::from(delta.get(op_arg)) as usize;
            executor.pop_jump_if_none(target)?;
            Ok(StepResult::Continue)
        }
        Instruction::PopJumpIfNotNone { delta } => {
            let target = u32::from(delta.get(op_arg)) as usize;
            executor.pop_jump_if_not_none(target)?;
            Ok(StepResult::Continue)
        }

        // ── Closure 3.11+ ──
        Instruction::MakeCell { i } => {
            // Phase 1: no-op (cell slots initialized in frame constructor)
            Ok(StepResult::Continue)
        }
        Instruction::CopyFreeVars { n } => {
            executor.copy_free_vars(n.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }

        // ── Generator ──
        Instruction::ReturnGenerator => {
            executor.return_generator()?;
            Ok(StepResult::Continue)
        }

        // ── Function call variants ──
        Instruction::CallKw { argc } => {
            executor.call_kw(argc.get(op_arg) as usize)?;
            Ok(StepResult::Continue)
        }
        Instruction::CallFunctionEx => {
            executor.call_function_ex()?;
            Ok(StepResult::Continue)
        }

        // ── Class support ──
        Instruction::LoadBuildClass => {
            executor.load_build_class()?;
            Ok(StepResult::Continue)
        }

        // ── Delete ops ──
        Instruction::DeleteFast { var_num } => {
            executor.delete_fast(u32::from(var_num.get(op_arg)) as usize)?;
            Ok(StepResult::Continue)
        }
        Instruction::DeleteName { namei } => {
            executor.delete_name(code.names[namei.get(op_arg) as usize].as_ref())?;
            Ok(StepResult::Continue)
        }
        Instruction::DeleteGlobal { namei } => {
            executor.delete_global(code.names[namei.get(op_arg) as usize].as_ref())?;
            Ok(StepResult::Continue)
        }
        Instruction::DeleteAttr { namei } => {
            executor.delete_attr(code.names[namei.get(op_arg) as usize].as_ref())?;
            Ok(StepResult::Continue)
        }

        // ── Load super attr ──
        Instruction::LoadSuperAttr { .. } => {
            executor.load_super_attr()?;
            Ok(StepResult::Continue)
        }

        // ── Misc ──
        Instruction::SetupAnnotations => Ok(StepResult::Continue),
        Instruction::LoadLocals => {
            executor.load_locals()?;
            Ok(StepResult::Continue)
        }
        Instruction::Copy { i } => {
            let depth = i.get(op_arg) as usize;
            let val = executor.peek_at(depth - 1)?;
            executor.push_value(val)?;
            Ok(StepResult::Continue)
        }

        // ── String formatting (f-strings) ──
        Instruction::FormatSimple => {
            executor.format_simple()?;
            Ok(StepResult::Continue)
        }
        Instruction::FormatWithSpec => {
            executor.format_with_spec()?;
            Ok(StepResult::Continue)
        }
        Instruction::ConvertValue { oparg: conv } => {
            executor.convert_value(conv.get(op_arg))?;
            Ok(StepResult::Continue)
        }

        // ── Sequence matching ──
        Instruction::GetLen => {
            let obj = executor.peek_at(0)?;
            let len = executor.get_len(obj)?;
            executor.push_value(len)?;
            Ok(StepResult::Continue)
        }

        // ── Loop / generator control ──
        Instruction::JumpBackwardNoInterrupt { delta } => {
            let tgt = u32::from(delta.get(op_arg)) as usize;
            executor.set_next_instr(next_instr - tgt)?;
            Ok(StepResult::Continue)
        }

        // ── Load fast and clear (comprehension scope) ──
        Instruction::LoadFastAndClear { var_num } => {
            let idx = var_num.get(op_arg).as_usize();
            executor.load_fast_and_clear(idx)?;
            Ok(StepResult::Continue)
        }

        // ── Set function attribute (closure, annotations, etc.) ──
        Instruction::SetFunctionAttribute { flag } => {
            let _flag = flag.get(op_arg);
            executor.set_function_attribute()?;
            Ok(StepResult::Continue)
        }

        // ── Scoping ──
        Instruction::LoadFromDictOrGlobals { i } => {
            let idx = i.get(op_arg) as usize;
            executor.load_from_dict_or_globals(code.names[idx].as_ref())?;
            Ok(StepResult::Continue)
        }
        Instruction::LoadFromDictOrDeref { i } => {
            let idx = i.get(op_arg) as usize;
            executor.load_from_dict_or_deref(idx, code.names[idx].as_ref())?;
            Ok(StepResult::Continue)
        }

        // ── Pattern matching (Python 3.10+) ──
        Instruction::MatchMapping | Instruction::MatchSequence => {
            // Phase 1 stub: push False
            executor.match_stub()?;
            Ok(StepResult::Continue)
        }

        // ── Unpack extended ──
        Instruction::UnpackEx { counts } => {
            executor.unpack_ex(counts.get(op_arg))?;
            Ok(StepResult::Continue)
        }

        // ── Intrinsics ──
        Instruction::CallIntrinsic1 { func } => {
            executor.call_intrinsic_1(func.get(op_arg))?;
            Ok(StepResult::Continue)
        }
        Instruction::CallIntrinsic2 { func } => {
            executor.call_intrinsic_2(func.get(op_arg))?;
            Ok(StepResult::Continue)
        }

        // ── Async stubs ──
        Instruction::GetAwaitable { .. } | Instruction::GetAIter | Instruction::GetANext
        | Instruction::EndAsyncFor | Instruction::Send { .. } | Instruction::EndSend
        | Instruction::GetYieldFromIter | Instruction::CleanupThrow => {
            Err(crate::PyError::type_error("async/generator send not yet implemented").into())
        }

        // ── Misc stubs ──
        Instruction::LoadSpecial { .. } => {
            Err(crate::PyError::type_error("LOAD_SPECIAL not yet implemented").into())
        }
        Instruction::ExitInitCheck => Ok(StepResult::Continue),
        Instruction::WithExceptStart => {
            Err(crate::PyError::type_error("WITH_EXCEPT_START not yet implemented").into())
        }

        other => executor.unsupported(&other),
    }
}

fn jump_target_forward(instructions: &[CodeUnit], next_instr: usize, delta: usize) -> usize {
    skip_caches(instructions, next_instr) + delta
}

fn jump_target_backward(instructions: &[CodeUnit], next_instr: usize, delta: usize) -> usize {
    skip_caches(instructions, next_instr) - delta
}

fn skip_caches(instructions: &[CodeUnit], mut pos: usize) -> usize {
    while pos < instructions.len() {
        let mut state = OpArgState::default();
        let (instruction, _) = state.get(instructions[pos]);
        if matches!(instruction, Instruction::Cache) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::decode_instruction_at;
    use pyre_bytecode::bytecode::Instruction;
    use pyre_bytecode::{OpArgState, compile_exec};

    #[test]
    fn decode_instruction_at_matches_forward_decode_across_extended_arg_prefix() {
        let source = (0..400)
            .map(|i| format!("v{i} = {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let code = compile_exec(&source).expect("compile failed");

        let target_pc = code
            .instructions
            .iter()
            .enumerate()
            .find_map(|(pc, unit)| {
                if pc > 0
                    && matches!(code.instructions[pc - 1].op, Instruction::ExtendedArg)
                    && !matches!(unit.op, Instruction::ExtendedArg)
                {
                    Some(pc)
                } else {
                    None
                }
            })
            .expect("expected an instruction with an ExtendedArg prefix");

        let mut forward = OpArgState::default();
        let mut expected = None;
        for (pc, unit) in code.instructions.iter().copied().enumerate() {
            let decoded = forward.get(unit);
            if pc == target_pc {
                expected = Some(decoded);
                break;
            }
        }

        assert_eq!(
            decode_instruction_at(&code, target_pc).map(|(instruction, arg)| {
                (std::mem::discriminant(&instruction), u32::from(arg))
            }),
            expected
                .map(|(instruction, arg)| (std::mem::discriminant(&instruction), u32::from(arg)))
        );
    }
}
