//! operator module — PyPy: pypy/module/operator/

use pyre_object::*;

fn op_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "index() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let indexed = crate::baseobjspace::space_index(args[0])?;
    unsafe { Ok(range_bigint_to_obj(range_obj_to_bigint(indexed))) }
}

/// Shared body for the binary-arithmetic thunks (`add`/`sub`/`mul`): a
/// wrong argument count is a `TypeError`, not a panic
/// (`interp_operator.py` `@unwrap_spec` argument checking).  The operand
/// error propagates, matching the `truediv`/`floordiv` thunks.
fn op_binary<F>(args: &[PyObjectRef], name: &str, f: F) -> Result<PyObjectRef, crate::PyError>
where
    F: Fn(PyObjectRef, PyObjectRef) -> Result<PyObjectRef, crate::PyError>,
{
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "{name} expected 2 arguments, got {}",
            args.len()
        )));
    }
    f(args[0], args[1])
}

/// `interp_operator.py:213-219`:
/// ```text
/// @unwrap_spec(default='index')
/// def length_hint(space, w_iterable, default=0):
///     return space.newint(space.length_hint(w_iterable, default))
/// ```
/// `default` defaults to 0, must be unwrapped via `__index__`.  Pyre
/// routes through `crate::baseobjspace::length_hint` (the
/// `space.length_hint` port), so `__length_hint__` priority +
/// negative-result ValueError + default fallback all match PyPy.
fn op_length_hint(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() || args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "length_hint expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }
    let w_iterable = args[0];
    let default = if let Some(&w_default) = args.get(1) {
        let w_index = crate::baseobjspace::space_index(w_default)?;
        crate::baseobjspace::int_w(w_index)?
    } else {
        0
    };
    let n = crate::baseobjspace::length_hint(w_iterable, default)?;
    Ok(w_int_new(n))
}

/// `_compare_digest(a, b)` — constant-time equality of two ASCII strings or
/// two bytes-like objects, used by `hmac` / `secrets`.
fn op_compare_digest(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let read = |obj: PyObjectRef| -> Result<Vec<u8>, crate::PyError> {
        unsafe {
            if is_str(obj) {
                let s = w_str_get_value(obj);
                if !s.is_ascii() {
                    return Err(crate::PyError::type_error(
                        "comparing strings with non-ASCII characters is not supported",
                    ));
                }
                Ok(s.as_bytes().to_vec())
            } else if bytesobject::is_bytes_like(obj) {
                Ok(bytesobject::bytes_like_data(obj).to_vec())
            } else {
                Err(crate::PyError::type_error(
                    "unsupported operand types(s) or combination of types",
                ))
            }
        }
    };
    let a = read(args.first().copied().unwrap_or_else(w_none))?;
    let b = read(args.get(1).copied().unwrap_or_else(w_none))?;
    let mut result = (a.len() ^ b.len()) as u8;
    for i in 0..a.len() {
        result |= a[i] ^ b.get(i).copied().unwrap_or(0);
    }
    Ok(w_bool_from(result == 0))
}

/// `interp_operator.py:204 iconcat` — `a += b` for two subscriptable
/// sequences; either operand missing `__getitem__` is a TypeError that
/// names the left operand.
fn op_iconcat(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "iconcat expected 2 arguments, got {}",
            args.len()
        )));
    }
    if unsafe {
        crate::baseobjspace::lookup(args[0], "__getitem__").is_none()
            || crate::baseobjspace::lookup(args[1], "__getitem__").is_none()
    } {
        return Err(crate::PyError::type_error(format!(
            "'{}' object can't be concatenated",
            crate::baseobjspace::object_functionstr_type_name(args[0])
        )));
    }
    crate::opcode_ops::binary_value(
        args[0],
        args[1],
        crate::bytecode::BinaryOperator::InplaceAdd,
    )
}

// Binary arithmetic / comparison thunks share one shape — call
// `baseobjspace::OP(args[0], args[1])` and unwrap-or-none the result.
// Inline closures below preserve the per-name `assert!` checks.
use crate::baseobjspace::{
    self, CompareOp, add, and_, contains, delitem, floordiv, getitem, invert, is_true, lshift,
    matmul, mod_, mul, neg, or_, pos, pow, rshift, setitem, sub, truediv, xor,
};

crate::py_module! {
    "operator",
    // `countOf` + the `itemgetter`/`attrgetter`/`methodcaller` callable
    // classes are app-level (`pypy/module/operator/app_operator.py`,
    // `moduledef.py` `app_names`), not interp-level.
    appleveldefs: {
        "app_operator.py" => ["countOf", "itemgetter", "attrgetter", "methodcaller"],
    },
    functions: {
        "index"    / 1 = op_index,
        "add"      / 2 = |args| op_binary(args, "add", add),
        "sub"      / 2 = |args| op_binary(args, "sub", sub),
        "mul"      / 2 = |args| op_binary(args, "mul", mul),
        "matmul"   / 2 = |args| op_binary(args, "matmul", matmul),
        "truediv"  / 2 = |args| truediv(args[0], args[1]),
        "floordiv" / 2 = |args| floordiv(args[0], args[1]),
        "mod"      / 2 = |args| mod_(args[0], args[1]),
        "pow"      / 2 = |args| pow(args[0], args[1]),
        "neg"      / 1 = |args| neg(args[0]),
        "pos"      / 1 = |args| pos(args[0]),
        "abs"      / 1 = |args| crate::builtins::builtin_abs(args),
        "invert"   / 1 = |args| invert(args[0]),
        "lshift"   / 2 = |args| lshift(args[0], args[1]),
        "rshift"   / 2 = |args| rshift(args[0], args[1]),
        "and_"     / 2 = |args| and_(args[0], args[1]),
        "or_"      / 2 = |args| or_(args[0], args[1]),
        "xor"      / 2 = |args| xor(args[0], args[1]),
        // interp_operator.py:150-210 — in-place operations, each `space.inplace_X`.
        "iadd"      / 2 = |args| op_binary(args, "iadd", |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceAdd)),
        "isub"      / 2 = |args| op_binary(args, "isub", |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceSubtract)),
        "imul"      / 2 = |args| op_binary(args, "imul", |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceMultiply)),
        "imatmul"   / 2 = |args| op_binary(args, "imatmul", |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceMatrixMultiply)),
        "ifloordiv" / 2 = |args| op_binary(args, "ifloordiv", |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceFloorDivide)),
        "imod"      / 2 = |args| op_binary(args, "imod", |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceRemainder)),
        "itruediv"  / 2 = |args| op_binary(args, "itruediv", |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceTrueDivide)),
        "ipow"      / 2 = |args| op_binary(args, "ipow", |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplacePower)),
        "ilshift"   / 2 = |args| op_binary(args, "ilshift", |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceLshift)),
        "irshift"   / 2 = |args| op_binary(args, "irshift", |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceRshift)),
        "iand"      / 2 = |args| op_binary(args, "iand", |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceAnd)),
        "ior"       / 2 = |args| op_binary(args, "ior", |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceOr)),
        "ixor"      / 2 = |args| op_binary(args, "ixor", |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceXor)),
        "iconcat"   / 2 = op_iconcat,
        "not_"     / 1 = |args| Ok(w_bool_from(!is_true(args[0])?)),
        // interp_operator.py:138
        "truth"    / 1 = |args| Ok(w_bool_from(is_true(args[0])?)),
        "is_"      / 2 = |args| Ok(w_bool_from(std::ptr::eq(args[0], args[1]))),
        "is_not"   / 2 = |args| Ok(w_bool_from(!std::ptr::eq(args[0], args[1]))),
        "contains" / 2 = |args| Ok(w_bool_from(contains(args[0], args[1])?)),
        "getitem"  / 2 = |args| getitem(args[0], args[1]),
        "setitem"  / 3 = |args| { setitem(args[0], args[1], args[2])?; Ok(w_none()) },
        "delitem"  / 2 = |args| { delitem(args[0], args[1])?; Ok(w_none()) },
        // Underscore aliases (__add__ / __sub__ / __mul__ via operator).
        "__add__"  / 2 = |args| { if args.len() != 2 { return Err(crate::PyError::type_error(format!("__add__() takes exactly 2 arguments ({} given)", args.len()))); } Ok(add(args[0], args[1]).unwrap_or(w_none())) },
        "__sub__"  / 2 = |args| { if args.len() != 2 { return Err(crate::PyError::type_error(format!("__sub__() takes exactly 2 arguments ({} given)", args.len()))); } Ok(sub(args[0], args[1]).unwrap_or(w_none())) },
        "__mul__"  / 2 = |args| { if args.len() != 2 { return Err(crate::PyError::type_error(format!("__mul__() takes exactly 2 arguments ({} given)", args.len()))); } Ok(mul(args[0], args[1]).unwrap_or(w_none())) },
        "eq" / 2 = |args| { if args.len() != 2 { return Err(crate::PyError::type_error(format!("eq() takes exactly 2 arguments ({} given)", args.len()))); } Ok(baseobjspace::compare(args[0], args[1], CompareOp::Eq).unwrap_or(w_none())) },
        "lt" / 2 = |args| { if args.len() != 2 { return Err(crate::PyError::type_error(format!("lt() takes exactly 2 arguments ({} given)", args.len()))); } Ok(baseobjspace::compare(args[0], args[1], CompareOp::Lt).unwrap_or(w_none())) },
        "gt" / 2 = |args| { if args.len() != 2 { return Err(crate::PyError::type_error(format!("gt() takes exactly 2 arguments ({} given)", args.len()))); } Ok(baseobjspace::compare(args[0], args[1], CompareOp::Gt).unwrap_or(w_none())) },
        "le" / 2 = |args| baseobjspace::compare(args[0], args[1], CompareOp::Le),
        "ge" / 2 = |args| baseobjspace::compare(args[0], args[1], CompareOp::Ge),
        "ne" / 2 = |args| baseobjspace::compare(args[0], args[1], CompareOp::Ne),
        "length_hint"  / * = op_length_hint,
        "_compare_digest" / 2 = op_compare_digest,
    },
    extra_init: |ns| {
        // `operator.py` tail — bind each dunder name to its operator
        // function (`__lt__ = lt`, `__add__ = add`, …) so `operator.__lt__`
        // resolves like CPython's pure-Python wrapper does.
        const ALIASES: &[(&str, &str)] = &[
            ("__lt__", "lt"), ("__le__", "le"), ("__eq__", "eq"),
            ("__ne__", "ne"), ("__ge__", "ge"), ("__gt__", "gt"),
            ("__not__", "not_"), ("__abs__", "abs"), ("__add__", "add"),
            ("__and__", "and_"), ("__call__", "call"),
            ("__floordiv__", "floordiv"), ("__index__", "index"),
            ("__inv__", "inv"), ("__invert__", "invert"),
            ("__lshift__", "lshift"), ("__mod__", "mod"), ("__mul__", "mul"),
            ("__matmul__", "matmul"), ("__neg__", "neg"), ("__or__", "or_"),
            ("__pos__", "pos"), ("__pow__", "pow"), ("__rshift__", "rshift"),
            ("__sub__", "sub"), ("__truediv__", "truediv"), ("__xor__", "xor"),
            ("__concat__", "concat"), ("__contains__", "contains"),
            ("__delitem__", "delitem"), ("__getitem__", "getitem"),
            ("__setitem__", "setitem"), ("__iadd__", "iadd"),
            ("__iand__", "iand"), ("__iconcat__", "iconcat"),
            ("__ifloordiv__", "ifloordiv"), ("__ilshift__", "ilshift"),
            ("__imod__", "imod"), ("__imul__", "imul"),
            ("__imatmul__", "imatmul"), ("__ior__", "ior"),
            ("__ipow__", "ipow"), ("__irshift__", "irshift"),
            ("__isub__", "isub"), ("__itruediv__", "itruediv"),
            ("__ixor__", "ixor"),
        ];
        for (dunder, src) in ALIASES {
            if let Some(f) = crate::module_ns_get(ns, src) {
                crate::module_ns_store(ns, dunder, f);
            }
        }
    },
}
