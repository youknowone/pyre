//! JIT helper functions — `extern "C"` wrappers called from compiled traces.
//!
//! The JIT backend (Cranelift) emits C-ABI calls to these functions.
//! Each wraps a pyre-object or pyre-objspace operation with the
//! correct calling convention and integer-based parameter passing.

use majit_ir::{OpCode, OpRef, Type};
use majit_meta::TraceCtx;

use pyre_object::*;
use pyre_objspace::{
    jit_binary_value_from_tag, jit_bool_value_from_truth, jit_compare_value_from_tag, jit_getattr,
    jit_setattr, jit_setitem, jit_truth_value, jit_unary_invert_value, jit_unary_negative_value,
};
use pyre_runtime::{
    PyBigInt, PyError, binary_op_tag, compare_op_tag, jit_range_iter_next_or_null,
    jit_sequence_getitem,
};

pub use pyre_runtime::{
    FlatBuildKind, callable_call_helper, flat_build_helper, jit_build_list_0, jit_build_list_1,
    jit_build_list_2, jit_build_list_3, jit_build_list_4, jit_build_list_5, jit_build_list_6,
    jit_build_list_7, jit_build_list_8, jit_build_map_0, jit_build_map_1, jit_build_map_2,
    jit_build_map_3, jit_build_map_4, jit_build_tuple_0, jit_build_tuple_1, jit_build_tuple_2,
    jit_build_tuple_3, jit_build_tuple_4, jit_build_tuple_5, jit_build_tuple_6, jit_build_tuple_7,
    jit_build_tuple_8, jit_call_callable_0, jit_call_callable_1, jit_call_callable_2,
    jit_call_callable_3, jit_call_callable_4, jit_call_callable_5, jit_call_callable_6,
    jit_call_callable_7, jit_call_callable_8, jit_load_name_from_namespace,
    jit_make_function_from_globals, jit_store_name_to_namespace, known_builtin_call_helper,
    known_function_call_helper, register_jit_function_caller,
};

fn trace_name_args(ctx: &mut TraceCtx, name: &str) -> [OpRef; 2] {
    [
        ctx.const_int(name.as_ptr() as usize as i64),
        ctx.const_int(name.len() as i64),
    ]
}

pub fn emit_trace_call_int(ctx: &mut TraceCtx, helper: *const (), args: &[OpRef]) -> OpRef {
    ctx.call_int(helper, args)
}

pub fn emit_trace_call_int_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    ctx.call_int_typed(helper, args, arg_types)
}

pub fn emit_trace_call_ref(ctx: &mut TraceCtx, helper: *const (), args: &[OpRef]) -> OpRef {
    ctx.call_ref(helper, args)
}

pub fn emit_trace_call_ref_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    ctx.call_ref_typed(helper, args, arg_types)
}

pub fn emit_trace_call_void(ctx: &mut TraceCtx, helper: *const (), args: &[OpRef]) {
    ctx.call_void(helper, args);
}

pub fn emit_trace_call_void_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) {
    ctx.call_void_typed(helper, args, arg_types);
}

pub fn emit_trace_call_may_force_ref_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    ctx.call_may_force_ref_typed(helper, args, arg_types)
}

pub fn emit_trace_build_flat(
    ctx: &mut TraceCtx,
    kind: FlatBuildKind,
    items: &[OpRef],
) -> Result<OpRef, PyError> {
    let helper_count = match kind {
        FlatBuildKind::Map => items.len() / 2,
        FlatBuildKind::List | FlatBuildKind::Tuple => items.len(),
    };
    let Some(helper) = flat_build_helper(kind, helper_count) else {
        let opname = match kind {
            FlatBuildKind::List => "list",
            FlatBuildKind::Tuple => "tuple",
            FlatBuildKind::Map => "map",
        };
        return Err(PyError::type_error(format!(
            "{opname} build arity not supported by JIT"
        )));
    };
    let arg_types = vec![Type::Ref; items.len()];
    Ok(ctx.call_ref_typed(helper, items, &arg_types))
}

pub fn emit_trace_call_callable(
    ctx: &mut TraceCtx,
    frame: OpRef,
    callable: OpRef,
    args: &[OpRef],
) -> Result<OpRef, PyError> {
    let helper = callable_call_helper(args.len())
        .ok_or_else(|| PyError::type_error("call arity not supported by JIT"))?;
    let mut call_args = vec![frame, callable];
    call_args.extend_from_slice(args);
    let mut arg_types = vec![Type::Ref, Type::Ref];
    arg_types.extend(std::iter::repeat_n(Type::Ref, args.len()));
    Ok(ctx.call_may_force_ref_typed(helper, &call_args, &arg_types))
}

pub fn emit_trace_call_known_builtin(
    ctx: &mut TraceCtx,
    callable: OpRef,
    args: &[OpRef],
) -> Result<OpRef, PyError> {
    let helper = known_builtin_call_helper(args.len())
        .ok_or_else(|| PyError::type_error("builtin call arity not supported by JIT"))?;
    let mut call_args = vec![callable];
    call_args.extend_from_slice(args);
    let mut arg_types = vec![Type::Ref];
    arg_types.extend(std::iter::repeat_n(Type::Ref, args.len()));
    Ok(ctx.call_ref_typed(helper, &call_args, &arg_types))
}

pub fn emit_trace_call_known_function(
    ctx: &mut TraceCtx,
    frame: OpRef,
    callable: OpRef,
    args: &[OpRef],
) -> Result<OpRef, PyError> {
    let helper = known_function_call_helper(args.len())
        .ok_or_else(|| PyError::type_error("function call arity not supported by JIT"))?;
    let mut call_args = vec![frame, callable];
    call_args.extend_from_slice(args);
    let mut arg_types = vec![Type::Ref, Type::Ref];
    arg_types.extend(std::iter::repeat_n(Type::Ref, args.len()));
    Ok(ctx.call_may_force_ref_typed(helper, &call_args, &arg_types))
}

pub fn emit_trace_unpack_sequence(
    ctx: &mut TraceCtx,
    seq: OpRef,
    count: usize,
) -> Result<Vec<OpRef>, PyError> {
    let mut items = Vec::with_capacity(count);
    for idx in 0..count {
        let idx_const = ctx.const_int(idx as i64);
        items.push(emit_trace_call_ref_typed(
            ctx,
            jit_sequence_getitem as *const (),
            &[seq, idx_const],
            &[Type::Ref, Type::Int],
        ));
    }
    Ok(items)
}

pub fn emit_trace_load_name_from_namespace(
    ctx: &mut TraceCtx,
    namespace: OpRef,
    name: &str,
) -> OpRef {
    let [name_ptr, name_len] = trace_name_args(ctx, name);
    emit_trace_call_ref_typed(
        ctx,
        jit_load_name_from_namespace as *const (),
        &[namespace, name_ptr, name_len],
        &[Type::Ref, Type::Int, Type::Int],
    )
}

pub fn emit_trace_store_name_to_namespace(
    ctx: &mut TraceCtx,
    namespace: OpRef,
    name: &str,
    value: OpRef,
) {
    let [name_ptr, name_len] = trace_name_args(ctx, name);
    emit_trace_call_void_typed(
        ctx,
        jit_store_name_to_namespace as *const (),
        &[namespace, name_ptr, name_len, value],
        &[Type::Ref, Type::Int, Type::Int, Type::Ref],
    );
}

pub fn emit_trace_truth_value(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_trace_call_int_typed(ctx, jit_truth_value as *const (), &[value], &[Type::Ref])
}

pub fn emit_trace_bool_value_from_truth(ctx: &mut TraceCtx, truth: OpRef, negate: bool) -> OpRef {
    let truth = if negate {
        let one = ctx.const_int(1);
        ctx.record_op(OpCode::IntSub, &[one, truth])
    } else {
        truth
    };
    emit_trace_call_ref_typed(ctx, jit_bool_value_from_truth as *const (), &[truth], &[Type::Int])
}

pub fn emit_trace_binary_value(
    ctx: &mut TraceCtx,
    a: OpRef,
    b: OpRef,
    op: pyre_bytecode::bytecode::BinaryOperator,
) -> Result<OpRef, PyError> {
    let Some(tag) = binary_op_tag(op) else {
        return Err(PyError::type_error(format!(
            "binary operation {op:?} not yet traceable"
        )));
    };
    let tag = ctx.const_int(tag);
    Ok(emit_trace_call_ref_typed(
        ctx,
        jit_binary_value_from_tag as *const (),
        &[a, b, tag],
        &[Type::Ref, Type::Ref, Type::Int],
    ))
}

pub fn emit_trace_compare_value(
    ctx: &mut TraceCtx,
    a: OpRef,
    b: OpRef,
    op: pyre_bytecode::bytecode::ComparisonOperator,
) -> OpRef {
    let tag = ctx.const_int(compare_op_tag(op));
    emit_trace_call_ref_typed(
        ctx,
        jit_compare_value_from_tag as *const (),
        &[a, b, tag],
        &[Type::Ref, Type::Ref, Type::Int],
    )
}

pub fn emit_trace_range_iter_next_or_null(ctx: &mut TraceCtx, iter: OpRef) -> OpRef {
    emit_trace_call_ref_typed(
        ctx,
        jit_range_iter_next_or_null as *const (),
        &[iter],
        &[Type::Ref],
    )
}

pub fn emit_trace_int_constant(ctx: &mut TraceCtx, value: i64) -> OpRef {
    ctx.const_int(w_int_new(value) as i64)
}

pub fn emit_trace_float_constant(ctx: &mut TraceCtx, value: f64) -> OpRef {
    ctx.const_int(box_float_constant(value) as i64)
}

pub fn emit_trace_unary_negative_value(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_trace_call_ref_typed(
        ctx,
        jit_unary_negative_value as *const (),
        &[value],
        &[Type::Ref],
    )
}

pub fn emit_trace_unary_invert_value(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_trace_call_ref_typed(
        ctx,
        jit_unary_invert_value as *const (),
        &[value],
        &[Type::Ref],
    )
}

pub trait TraceHelperAccess {
    fn with_trace_ctx<R>(&mut self, f: impl FnOnce(&mut TraceCtx) -> R) -> R;
    fn trace_frame(&self) -> OpRef;
    fn trace_globals_ptr(&mut self) -> OpRef;
    fn trace_record_not_forced_guard(&mut self);

    fn trace_make_function(&mut self, code_obj: OpRef) -> Result<OpRef, PyError> {
        let globals = self.trace_globals_ptr();
        self.with_trace_ctx(|ctx| {
            Ok(emit_trace_call_ref_typed(
                ctx,
                jit_make_function_from_globals as *const (),
                &[globals, code_obj],
                &[Type::Ref, Type::Ref],
            ))
        })
    }

    fn trace_call_callable(&mut self, callable: OpRef, args: &[OpRef]) -> Result<OpRef, PyError> {
        let frame = self.trace_frame();
        let result =
            self.with_trace_ctx(|ctx| emit_trace_call_callable(ctx, frame, callable, args))?;
        self.trace_record_not_forced_guard();
        Ok(result)
    }

    fn trace_build_list(&mut self, items: &[OpRef]) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| emit_trace_build_flat(ctx, FlatBuildKind::List, items))
    }

    fn trace_build_tuple(&mut self, items: &[OpRef]) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| emit_trace_build_flat(ctx, FlatBuildKind::Tuple, items))
    }

    fn trace_build_map(&mut self, items: &[OpRef]) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| emit_trace_build_flat(ctx, FlatBuildKind::Map, items))
    }

    fn trace_store_subscr(&mut self, obj: OpRef, key: OpRef, value: OpRef) -> Result<(), PyError> {
        self.with_trace_ctx(|ctx| {
            let _ = emit_trace_call_int_typed(
                ctx,
                jit_setitem as *const (),
                &[obj, key, value],
                &[Type::Ref, Type::Ref, Type::Ref],
            );
            Ok(())
        })
    }

    fn trace_load_attr(&mut self, obj: OpRef, name: &str) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| {
            let [name_ptr, name_len] = trace_name_args(ctx, name);
            Ok(emit_trace_call_ref_typed(
                ctx,
                jit_getattr as *const (),
                &[obj, name_ptr, name_len],
                &[Type::Ref, Type::Int, Type::Int],
            ))
        })
    }

    fn trace_store_attr(&mut self, obj: OpRef, name: &str, value: OpRef) -> Result<(), PyError> {
        self.with_trace_ctx(|ctx| {
            let [name_ptr, name_len] = trace_name_args(ctx, name);
            let _ = emit_trace_call_int_typed(
                ctx,
                jit_setattr as *const (),
                &[obj, name_ptr, name_len, value],
                &[Type::Ref, Type::Int, Type::Int, Type::Ref],
            );
            Ok(())
        })
    }

    fn trace_list_append(&mut self, list: OpRef, value: OpRef) -> Result<(), PyError> {
        self.with_trace_ctx(|ctx| {
            emit_trace_call_void_typed(
                ctx,
                jit_list_append as *const (),
                &[list, value],
                &[Type::Ref, Type::Ref],
            );
            Ok(())
        })
    }

    fn trace_unpack_sequence(&mut self, seq: OpRef, count: usize) -> Result<Vec<OpRef>, PyError> {
        self.with_trace_ctx(|ctx| emit_trace_unpack_sequence(ctx, seq, count))
    }

    fn trace_load_name(&mut self, name: &str) -> Result<OpRef, PyError> {
        let globals = self.trace_globals_ptr();
        self.with_trace_ctx(|ctx| Ok(emit_trace_load_name_from_namespace(ctx, globals, name)))
    }

    fn trace_store_name(&mut self, name: &str, value: OpRef) -> Result<(), PyError> {
        let globals = self.trace_globals_ptr();
        self.with_trace_ctx(|ctx| {
            emit_trace_store_name_to_namespace(ctx, globals, name, value);
            Ok(())
        })
    }

    fn trace_null_value(&mut self) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_int(0)))
    }

    fn trace_iter_next_value(&mut self, iter: OpRef) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_range_iter_next_or_null(ctx, iter)))
    }

    fn trace_truth_value(&mut self, value: OpRef) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_truth_value(ctx, value)))
    }

    fn trace_bool_value_from_truth(
        &mut self,
        truth: OpRef,
        negate: bool,
    ) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_bool_value_from_truth(ctx, truth, negate)))
    }

    fn trace_binary_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: pyre_bytecode::bytecode::BinaryOperator,
    ) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| emit_trace_binary_value(ctx, a, b, op))
    }

    fn trace_compare_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: pyre_bytecode::bytecode::ComparisonOperator,
    ) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_compare_value(ctx, a, b, op)))
    }

    fn trace_unary_negative_value(&mut self, value: OpRef) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_unary_negative_value(ctx, value)))
    }

    fn trace_unary_invert_value(&mut self, value: OpRef) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_unary_invert_value(ctx, value)))
    }

    fn trace_int_constant(&mut self, value: i64) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_int_constant(ctx, value)))
    }

    fn trace_bigint_constant(&mut self, value: &PyBigInt) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_int(box_bigint_constant(value) as i64)))
    }

    fn trace_float_constant(&mut self, value: f64) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_float_constant(ctx, value)))
    }

    fn trace_bool_constant(&mut self, value: bool) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_int(w_bool_from(value) as i64)))
    }

    fn trace_str_constant(&mut self, value: &str) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_int(box_str_constant(value) as i64)))
    }

    fn trace_code_constant(&mut self, code: &pyre_bytecode::CodeObject) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_int(pyre_runtime::box_code_constant(code) as i64)))
    }

    fn trace_none_constant(&mut self) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_int(w_none() as i64)))
    }
}

/// Emit inline W_Int creation (New + SetfieldGc) instead of CallI.
///
/// This allows the optimizer to virtualize the W_Int allocation:
/// if the intval field is read back later, the optimizer returns
/// the original raw int directly without allocating.
pub fn emit_box_int_inline(
    ctx: &mut TraceCtx,
    raw_int: OpRef,
    size_descr: majit_ir::DescrRef,
    ob_type_descr: majit_ir::DescrRef,
    intval_descr: majit_ir::DescrRef,
    int_type_addr: i64,
) -> OpRef {
    // Emit: v = New(W_Int)
    let new_op = ctx.record_op_with_descr(OpCode::New, &[], size_descr);
    // Emit: SetfieldGc(v, ob_type, INT_TYPE)
    let type_const = ctx.const_int(int_type_addr);
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, type_const], ob_type_descr);
    // Emit: SetfieldGc(v, intval, raw_int)
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, raw_int], intval_descr);
    new_op
}

pub fn emit_box_float_inline(
    ctx: &mut TraceCtx,
    raw_float: OpRef,
    size_descr: majit_ir::DescrRef,
    ob_type_descr: majit_ir::DescrRef,
    floatval_descr: majit_ir::DescrRef,
    float_type_addr: i64,
) -> OpRef {
    let new_op = ctx.record_op_with_descr(OpCode::New, &[], size_descr);
    let type_const = ctx.const_int(float_type_addr);
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, type_const], ob_type_descr);
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, raw_float], floatval_descr);
    new_op
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyre_bytecode::{ConstantData, compile_exec};
    use pyre_runtime::PyExecutionContext;

    #[test]
    fn test_callable_call_helper_dispatches_builtin_without_trace_side_branching() {
        let namespace = PyExecutionContext::default().fresh_namespace();
        let abs = *namespace.get("abs").expect("abs builtin must exist");
        let result = jit_call_callable_1(0, abs as i64, w_int_new(-11) as i64);
        unsafe {
            assert_eq!(w_int_get_value(result as PyObjectRef), 11);
        }
    }

    #[test]
    fn test_tuple_build_helper_dispatches_items() {
        let result = jit_build_tuple_2(w_int_new(3) as i64, w_int_new(5) as i64);
        let tuple = result as PyObjectRef;
        unsafe {
            assert!(is_tuple(tuple));
            assert_eq!(w_int_get_value(w_tuple_getitem(tuple, 0).unwrap()), 3);
            assert_eq!(w_int_get_value(w_tuple_getitem(tuple, 1).unwrap()), 5);
        }
    }

    #[test]
    fn test_sequence_getitem_helper_dispatches_list_and_tuple() {
        let list = w_list_new(vec![w_int_new(2), w_int_new(4)]);
        let tuple = w_tuple_new(vec![w_int_new(7), w_int_new(9)]);
        unsafe {
            assert_eq!(
                w_int_get_value(jit_sequence_getitem(list as i64, 1) as PyObjectRef),
                4
            );
            assert_eq!(
                w_int_get_value(jit_sequence_getitem(tuple as i64, 0) as PyObjectRef),
                7
            );
        }
    }

    #[test]
    fn test_map_build_helper_dispatches_int_keys() {
        let result = jit_build_map_2(
            w_int_new(1) as i64,
            w_int_new(10) as i64,
            w_int_new(2) as i64,
            w_int_new(20) as i64,
        );
        let dict = result as PyObjectRef;
        unsafe {
            assert!(is_dict(dict));
            assert_eq!(w_int_get_value(w_dict_getitem(dict, 1).unwrap()), 10);
            assert_eq!(w_int_get_value(w_dict_getitem(dict, 2).unwrap()), 20);
        }
    }

    #[test]
    fn test_binary_helper_reuses_objspace_semantics() {
        let result = jit_binary_value_from_tag(w_int_new(9) as i64, w_int_new(4) as i64, 1);
        unsafe {
            assert_eq!(w_int_get_value(result as PyObjectRef), 5);
        }
    }

    #[test]
    fn test_compare_helper_reuses_objspace_semantics() {
        let result = jit_compare_value_from_tag(w_int_new(2) as i64, w_int_new(7) as i64, 0);
        unsafe {
            assert!(w_bool_get_value(result as PyObjectRef));
        }
    }

    #[test]
    fn test_invert_helper_reuses_objspace_semantics() {
        let result = jit_unary_invert_value(w_int_new(5) as i64);
        unsafe {
            assert_eq!(w_int_get_value(result as PyObjectRef), !5);
        }
    }

    #[test]
    fn test_make_function_helper_wraps_code_object() {
        let module = compile_exec("def f(x):\n    return x").expect("compile failed");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } => Some(code.as_ref().clone()),
                _ => None,
            })
            .expect("expected nested function code");
        let code_ptr = Box::into_raw(Box::new(code)) as *const ();
        let code_obj = pyre_runtime::w_code_new(code_ptr);
        let func = jit_make_function_from_globals(0, code_obj as i64) as PyObjectRef;

        unsafe {
            assert!(pyre_runtime::is_func(func));
            assert_eq!(pyre_runtime::w_func_get_code_ptr(func), code_ptr);
        }
    }

    #[test]
    fn test_range_iter_next_helper_uses_runtime_iterator_step() {
        let iter = w_range_iter_new(0, 2, 1);
        let first = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        let second = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        let done = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(first), 0);
            assert_eq!(w_int_get_value(second), 1);
            assert!(done.is_null());
        }
    }
}
