//! _opcode module — PyPy: `pypy/module/_opcode/`.
//!
//! Opcode metadata used by `opcode.py` and `dis.py`.

use pyre_object::*;
use rustpython_compiler_core::bytecode::{AnyOpcode, oparg};

fn try_opcode(raw: i64) -> Option<AnyOpcode> {
    u16::try_from(raw).ok()?.try_into().ok()
}

fn opcode_predicate(
    args: &[PyObjectRef],
    predicate: impl FnOnce(AnyOpcode) -> bool,
) -> Result<PyObjectRef, crate::PyError> {
    let opcode = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("opcode argument is required"))?;
    let raw = crate::baseobjspace::int_w(opcode)?;
    Ok(w_bool_from(try_opcode(raw).is_some_and(predicate)))
}

fn is_valid(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    opcode_predicate(args, |_| true)
}

fn has_arg(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    opcode_predicate(args, |op| op.has_arg())
}

fn has_const(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    opcode_predicate(args, |op| op.has_const())
}

fn has_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    opcode_predicate(args, |op| op.has_name())
}

fn has_jump(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    opcode_predicate(args, |op| op.has_jump())
}

fn has_free(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    opcode_predicate(args, |op| op.has_free())
}

fn has_local(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    opcode_predicate(args, |op| op.has_local())
}

fn has_exc(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    opcode_predicate(args, |op| op.is_block_push())
}

fn stack_effect(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if positional.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "stack_effect() takes at most 2 positional arguments ({} given)",
            positional.len(),
        )));
    }
    let raw = positional
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("stack_effect() missing opcode"))?;
    let raw = crate::baseobjspace::int_w(raw)?;
    let opcode = try_opcode(raw)
        .filter(|op| op.real().is_none_or(|real| real.deopt().is_none()))
        .ok_or_else(|| crate::PyError::value_error("invalid opcode or oparg"))?;

    let oparg = match positional.get(1).copied() {
        Some(value) if unsafe { !is_none(value) } => crate::baseobjspace::int_w(value)?,
        _ => 0,
    };
    let oparg =
        u32::try_from(oparg).map_err(|_| crate::PyError::value_error("invalid opcode or oparg"))?;

    let jump = crate::builtins::kwarg_get(kwargs, "jump");
    let effect = match jump {
        Some(value) if unsafe { !is_none(value) } => {
            if crate::baseobjspace::is_true(value)? {
                opcode.stack_effect_jump(oparg)
            } else {
                opcode.stack_effect(oparg)
            }
        }
        _ => opcode
            .stack_effect(oparg)
            .max(opcode.stack_effect_jump(oparg)),
    };
    Ok(w_int_new(effect as i64))
}

fn get_intrinsic1_descs(_: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mut descriptions = vec![w_str_new("INTRINSIC_1_INVALID")];
    descriptions.extend(oparg::IntrinsicFunction1::iter().map(|value| w_str_new(value.desc())));
    Ok(w_list_new(descriptions))
}

fn get_intrinsic2_descs(_: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mut descriptions = vec![w_str_new("INTRINSIC_2_INVALID")];
    descriptions.extend(oparg::IntrinsicFunction2::iter().map(|value| w_str_new(value.desc())));
    Ok(w_list_new(descriptions))
}

fn get_nb_ops(_: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_list_new(
        oparg::BinaryOperator::iter()
            .map(|value| w_tuple_new(vec![w_str_new(value.desc()), w_str_new(&value.to_string())]))
            .collect(),
    ))
}

fn special_method_names_impl() -> PyObjectRef {
    w_list_new(
        oparg::SpecialMethod::iter()
            .map(|value| w_str_new(&value.to_string()))
            .collect(),
    )
}

crate::py_module! {
    "_opcode",
    inline_functions: {
        fn get_opname(opcode: i64) -> String {
            format!("<{opcode}>")
        }
        fn get_special_method_names() -> PyObjectRef {
            crate::module::_opcode::special_method_names_impl()
        }
    },
    functions: {
        "stack_effect"             / 3 = stack_effect,
        "get_executor"             / 0 = |_| Ok(w_none()),
        "get_specialization_stats" / 0 = |_| Ok(w_none()),
        "get_intrinsic1_descs"     / 0 = get_intrinsic1_descs,
        "get_intrinsic2_descs"     / 0 = get_intrinsic2_descs,
        "get_nb_ops"               / 0 = get_nb_ops,
        "get_executor_count"       / 0 = |_| Ok(w_int_new(0)),
        "get_hot_code"             / 0 = |_| Ok(w_list_new(vec![])),
    },
    extra_init: |ns| {
        // `Python/bytecodes.c` exposes `ENABLE_SPECIALIZATION`; pyre has no
        // CPython-style adaptive specialization, so it reads False — tests
        // gated on `@requires_specialization` then skip.
        crate::module_ns_store(ns, "ENABLE_SPECIALIZATION", w_bool_from(false));
        crate::module_ns_store(ns, "ENABLE_SPECIALIZATION_FT", w_bool_from(false));
        for (name, function) in [
            ("is_valid", is_valid as crate::BuiltinCodeFn),
            ("has_arg", has_arg as crate::BuiltinCodeFn),
            ("has_const", has_const as crate::BuiltinCodeFn),
            ("has_name", has_name as crate::BuiltinCodeFn),
            ("has_jump", has_jump as crate::BuiltinCodeFn),
            ("has_free", has_free as crate::BuiltinCodeFn),
            ("has_local", has_local as crate::BuiltinCodeFn),
            ("has_exc", has_exc as crate::BuiltinCodeFn),
        ] {
            crate::module_ns_store(ns, name, crate::make_builtin_function_with_arity(name, function, 1));
        }
    }
}
