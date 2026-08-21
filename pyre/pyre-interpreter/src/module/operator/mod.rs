//! operator module — PyPy: pypy/module/operator/

use pyre_object::*;

fn op_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let indexed = crate::baseobjspace::space_index(args[0])?;
    unsafe { Ok(range_bigint_to_obj(range_obj_to_bigint(indexed))) }
}

/// Shared body for the binary-arithmetic thunks (`add`/`sub`/`mul`).  The
/// operand error propagates, matching the `truediv`/`floordiv` thunks.
fn op_binary<F>(args: &[PyObjectRef], f: F) -> Result<PyObjectRef, crate::PyError>
where
    F: Fn(PyObjectRef, PyObjectRef) -> Result<PyObjectRef, crate::PyError>,
{
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
    let a_obj = args.first().copied().unwrap_or_else(w_none);
    let b_obj = args.get(1).copied().unwrap_or_else(w_none);
    if unsafe { is_str(a_obj) } != unsafe { is_str(b_obj) } {
        return Err(crate::PyError::type_error(
            "unsupported operand types(s) or combination of types",
        ));
    }
    let read = |obj: PyObjectRef| -> Result<Vec<u8>, crate::PyError> {
        unsafe {
            if is_str(obj) {
                // The ASCII check runs on the raw buffer: a lone surrogate is
                // non-ASCII, so it takes the same rejection as any other
                // non-ASCII character.
                let s = w_str_get_wtf8(obj);
                if !s.as_bytes().is_ascii() {
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
    let a = read(a_obj)?;
    let b = read(b_obj)?;
    let mut result = (a.len() ^ b.len()) as u8;
    for i in 0..a.len() {
        result |= a[i] ^ b.get(i).copied().unwrap_or(0);
    }
    Ok(w_bool_from(result == 0))
}

/// `interp_operator.py concat` — `a + b` for two subscriptable
/// sequences.  Either operand missing `__getitem__` raises a bare
/// `TypeError` with no message (`OperationError(space.w_TypeError,
/// space.w_None)`); otherwise the result is `space.add(a, b)`.
fn op_concat(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "concat expected 2 arguments, got {}",
            args.len()
        )));
    }
    if unsafe {
        crate::baseobjspace::lookup(args[0], "__getitem__").is_none()
            || crate::baseobjspace::lookup(args[1], "__getitem__").is_none()
    } {
        return Err(crate::PyError::type_error(String::new()));
    }
    add(args[0], args[1])
}

/// `interp_operator.py iconcat` — `a += b` for two subscriptable
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
    // `Modules/_operator.c` — the accelerator module.  `operator` itself is
    // the pure-Python `lib-python/3/operator.py`, which imports from here.
    // `moduledef.py:5` names it `_operator` too (`applevel_name`).
    "_operator",
    // `moduledef.py` `app_names` — only the `attrgetter`/`itemgetter`/
    // `methodcaller` factory classes (which cannot be plain interp-level
    // functions) stay app-level.  `countOf` is interp-level here, a non-binding
    // builtin delegating to `space.sequence_count`; PyPy keeps it in
    // `app_operator.py`, but `_operator.c` ships `countOf` as a C function, so
    // the interp-level form matches that non-binding exposure.  `indexOf`
    // likewise delegates to `space.sequence_index` (`interp_operator.py`);
    // `concat` (`op_concat`, `interp_operator.py`) guards both operands for
    // `__getitem__`.
    appleveldefs: {
        "app_operator.py" => [
            "itemgetter", "attrgetter", "methodcaller",
        ],
    },
    functions: {
        "index"    / 1 = op_index,
        "add"      / 2 = |args| op_binary(args, add),
        "sub"      / 2 = |args| op_binary(args, sub),
        "mul"      / 2 = |args| op_binary(args, mul),
        "matmul"   / 2 = |args| op_binary(args, matmul),
        "truediv"  / 2 = |args| truediv(args[0], args[1]),
        "floordiv" / 2 = |args| floordiv(args[0], args[1]),
        "mod"      / 2 = |args| mod_(args[0], args[1]),
        "pow"      / 2 = |args| pow(args[0], args[1]),
        "neg"      / 1 = |args| neg(args[0]),
        "pos"      / 1 = |args| pos(args[0]),
        "abs"      / 1 = |args| crate::builtins::builtin_abs(args),
        "invert"   / 1 = |args| invert(args[0]),
        // `inv` is the historical spelling of `invert` — same as `~a`.
        "inv"      / 1 = |args| invert(args[0]),
        "lshift"   / 2 = |args| lshift(args[0], args[1]),
        "rshift"   / 2 = |args| rshift(args[0], args[1]),
        "and_"     / 2 = |args| and_(args[0], args[1]),
        "or_"      / 2 = |args| or_(args[0], args[1]),
        "xor"      / 2 = |args| xor(args[0], args[1]),
        // interp_operator.py:150-210 — in-place operations, each `space.inplace_X`.
        "iadd"      / 2 = |args| op_binary(args, |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceAdd)),
        "isub"      / 2 = |args| op_binary(args, |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceSubtract)),
        "imul"      / 2 = |args| op_binary(args, |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceMultiply)),
        "imatmul"   / 2 = |args| op_binary(args, |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceMatrixMultiply)),
        "ifloordiv" / 2 = |args| op_binary(args, |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceFloorDivide)),
        "imod"      / 2 = |args| op_binary(args, |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceRemainder)),
        "itruediv"  / 2 = |args| op_binary(args, |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceTrueDivide)),
        "ipow"      / 2 = |args| op_binary(args, |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplacePower)),
        "ilshift"   / 2 = |args| op_binary(args, |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceLshift)),
        "irshift"   / 2 = |args| op_binary(args, |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceRshift)),
        "iand"      / 2 = |args| op_binary(args, |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceAnd)),
        "ior"       / 2 = |args| op_binary(args, |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceOr)),
        "ixor"      / 2 = |args| op_binary(args, |a, b| crate::opcode_ops::binary_value(a, b, crate::bytecode::BinaryOperator::InplaceXor)),
        "concat"    / 2 = op_concat,
        "iconcat"   / 2 = op_iconcat,
        "not_"     / 1 = |args| Ok(w_bool_from(!is_true(args[0])?)),
        // interp_operator.py:138
        "truth"    / 1 = |args| Ok(w_bool_from(is_true(args[0])?)),
        "is_"      / 2 = |args| Ok(w_bool_from(std::ptr::eq(args[0], args[1]))),
        "is_not"   / 2 = |args| Ok(w_bool_from(!std::ptr::eq(args[0], args[1]))),
        "is_none"     / 1 = |args| Ok(w_bool_from(std::ptr::eq(args[0], w_none()))),
        "is_not_none" / 1 = |args| Ok(w_bool_from(!std::ptr::eq(args[0], w_none()))),
        "contains" / 2 = |args| Ok(w_bool_from(contains(args[0], args[1])?)),
        "indexOf"  / 2 = |args| baseobjspace::sequence_index(args[0], args[1]),
        "countOf"  / 2 = |args| baseobjspace::sequence_count(args[0], args[1]),
        // `call(obj, /, *args, **kwargs)` == `obj(*args, **kwargs)`.
        // `call_forwarding_args` re-splits the `__pyre_kw__` marker back into
        // keyword arguments before dispatching.
        "call"     / * = |args: &[PyObjectRef]| {
            if args.is_empty() {
                return Err(crate::PyError::type_error(
                    "call expected at least 1 argument, got 0",
                ));
            }
            crate::builtins::call_forwarding_args(args[0], &args[1..])
        },
        "getitem"  / 2 = |args| getitem(args[0], args[1]),
        "setitem"  / 3 = |args| { setitem(args[0], args[1], args[2])?; Ok(w_none()) },
        "delitem"  / 2 = |args| { delitem(args[0], args[1])?; Ok(w_none()) },
        "eq" / 2 = |args| baseobjspace::compare(args[0], args[1], CompareOp::Eq),
        "lt" / 2 = |args| baseobjspace::compare(args[0], args[1], CompareOp::Lt),
        "gt" / 2 = |args| baseobjspace::compare(args[0], args[1], CompareOp::Gt),
        "le" / 2 = |args| baseobjspace::compare(args[0], args[1], CompareOp::Le),
        "ge" / 2 = |args| baseobjspace::compare(args[0], args[1], CompareOp::Ge),
        "ne" / 2 = |args| baseobjspace::compare(args[0], args[1], CompareOp::Ne),
        "length_hint"  / * = op_length_hint,
        "_compare_digest" / 2 = op_compare_digest,
    },
    // The `__lt__ = lt` / `__add__ = add` dunder aliases belong to
    // `operator.py`, which binds them after its `from _operator import *`;
    // this module carries only the names that tail imports.
}
