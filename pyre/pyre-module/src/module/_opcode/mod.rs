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
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let opcode = args
        .first()
        .copied()
        .ok_or_else(|| pyre_interpreter::PyError::type_error("opcode argument is required"))?;
    let raw = pyre_interpreter::baseobjspace::int_w(opcode)?;
    Ok(w_bool_from(try_opcode(raw).is_some_and(predicate)))
}

fn is_valid(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |_| true)
}

fn has_arg(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.has_arg())
}

fn has_const(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.has_const())
}

fn has_name(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.has_name())
}

fn has_jump(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.has_jump())
}

fn has_free(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.has_free())
}

fn has_local(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.has_local())
}

fn has_exc(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    opcode_predicate(args, |op| op.is_block_push())
}

fn stack_effect(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    pyre_interpreter::builtins::kwarg_reject_unknown(kwargs, &["jump"], "stack_effect")?;
    if positional.len() > 2 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "stack_effect() takes at most 2 positional arguments ({} given)",
            positional.len(),
        )));
    }
    let raw = positional
        .first()
        .copied()
        .ok_or_else(|| pyre_interpreter::PyError::type_error("stack_effect() missing opcode"))?;
    let raw = pyre_interpreter::baseobjspace::int_w(raw)?;
    let opcode = try_opcode(raw)
        .filter(|op| op.real().is_none_or(|real| real.deopt().is_none()))
        .ok_or_else(|| pyre_interpreter::PyError::value_error("invalid opcode or oparg"))?;

    let oparg = match positional.get(1).copied() {
        Some(value) if unsafe { !is_none(value) } => pyre_interpreter::baseobjspace::int_w(value)?,
        _ => 0,
    };
    let oparg = u32::try_from(oparg)
        .map_err(|_| pyre_interpreter::PyError::value_error("invalid opcode or oparg"))?;

    let jump = pyre_interpreter::builtins::kwarg_get(kwargs, "jump");
    let effect = match jump {
        Some(value) if unsafe { !is_none(value) } => {
            if pyre_interpreter::baseobjspace::is_true(value)? {
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

// Both enums already carry `Invalid = 0`, whose `desc()` is the
// `INTRINSIC_*_INVALID` entry the table starts with, so `iter()` alone produces
// the whole table. Prepending it again shifted every real name one slot up and
// mislabelled every intrinsic in `dis` output.
fn get_intrinsic1_descs(_: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for value in oparg::IntrinsicFunction1::iter() {
        items.push(w_str_new_managed(value.desc()));
    }
    Ok(w_list_new(items.take()))
}

fn get_intrinsic2_descs(_: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for value in oparg::IntrinsicFunction2::iter() {
        items.push(w_str_new_managed(value.desc()));
    }
    Ok(w_list_new(items.take()))
}

fn get_nb_ops(_: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // Each name string and each pair tuple is freshly allocated, so both levels
    // pin as they arrive; the inner bracket closes before its tuple is pinned.
    let mut rows = pyre_object::gc_roots::RootedItems::new();
    for value in oparg::BinaryOperator::iter() {
        let row = {
            let mut names = pyre_object::gc_roots::RootedItems::new();
            names.push(w_str_new_managed(value.desc()));
            names.push(w_str_new_managed(&value.to_string()));
            w_tuple_new(names.take())
        };
        rows.push(row);
    }
    Ok(w_list_new(rows.take()))
}

fn special_method_names_impl() -> PyObjectRef {
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for value in oparg::SpecialMethod::iter() {
        items.push(w_str_new_managed(&value.to_string()));
    }
    w_list_new(items.take())
}

pyre_interpreter::py_module! {
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
        // `stack_effect(opcode, oparg=None, *, jump=None)` — the optional
        // tail and the keyword-only `jump` leave no single natural arity, so
        // the body enforces the count itself.
        "stack_effect"             / * = stack_effect,
        "get_executor"             / 2 = |_| Ok(w_none()),
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
        pyre_interpreter::module_ns_store(ns, "ENABLE_SPECIALIZATION", w_bool_from(false));
        pyre_interpreter::module_ns_store(ns, "ENABLE_SPECIALIZATION_FT", w_bool_from(false));
        for (name, function) in [
            ("is_valid", is_valid as pyre_interpreter::BuiltinCodeFn),
            ("has_arg", has_arg as pyre_interpreter::BuiltinCodeFn),
            ("has_const", has_const as pyre_interpreter::BuiltinCodeFn),
            ("has_name", has_name as pyre_interpreter::BuiltinCodeFn),
            ("has_jump", has_jump as pyre_interpreter::BuiltinCodeFn),
            ("has_free", has_free as pyre_interpreter::BuiltinCodeFn),
            ("has_local", has_local as pyre_interpreter::BuiltinCodeFn),
            ("has_exc", has_exc as pyre_interpreter::BuiltinCodeFn),
        ] {
            pyre_interpreter::module_ns_store(ns, name, pyre_interpreter::make_builtin_function_with_arity(name, function, 1));
        }
    }
}
