//! operator module — PyPy: pypy/module/operator/

use pyre_object::*;

fn op_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "index() takes exactly one argument");
    let obj = args[0];
    unsafe {
        if is_int(obj) {
            return Ok(obj);
        }
        if is_bool(obj) {
            return Ok(w_int_new(if w_bool_get_value(obj) { 1 } else { 0 }));
        }
    }
    Ok(crate::call_function_or_identity(obj, "__index__"))
}

fn op_add(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(crate::baseobjspace::add(args[0], args[1]).unwrap_or(w_none()))
}

fn op_sub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(crate::baseobjspace::sub(args[0], args[1]).unwrap_or(w_none()))
}

fn op_mul(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(crate::baseobjspace::mul(args[0], args[1]).unwrap_or(w_none()))
}

fn op_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(
        crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Eq)
            .unwrap_or(w_none()),
    )
}

fn op_lt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(
        crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Lt)
            .unwrap_or(w_none()),
    )
}

fn op_gt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(
        crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Gt)
            .unwrap_or(w_none()),
    )
}

fn op_le(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Le)
}

fn op_ge(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Ge)
}

fn op_ne(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Ne)
}

fn op_truediv(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::truediv(args[0], args[1])
}

fn op_floordiv(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::floordiv(args[0], args[1])
}

fn op_mod(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::mod_(args[0], args[1])
}

fn op_pow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::pow(args[0], args[1])
}

fn op_neg(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::neg(args[0])
}

fn op_pos(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::pos(args[0])
}

fn op_abs(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::builtins::builtin_abs(args)
}

fn op_invert(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::invert(args[0])
}

fn op_lshift(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::lshift(args[0], args[1])
}

fn op_rshift(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::rshift(args[0], args[1])
}

fn op_and(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::and_(args[0], args[1])
}

fn op_or(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::or_(args[0], args[1])
}

fn op_xor(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::xor(args[0], args[1])
}

fn op_not(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_bool_from(!crate::baseobjspace::is_true(args[0])))
}

fn op_truth(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_bool_from(crate::baseobjspace::is_true(args[0])))
}

fn op_is(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_bool_from(std::ptr::eq(args[0], args[1])))
}

fn op_is_not(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_bool_from(!std::ptr::eq(args[0], args[1])))
}

fn op_contains(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_bool_from(crate::baseobjspace::contains(args[0], args[1])?))
}

fn op_getitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::getitem(args[0], args[1])
}

fn op_setitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::setitem(args[0], args[1], args[2])?;
    Ok(w_none())
}

fn op_delitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::delitem(args[0], args[1])?;
    Ok(w_none())
}

fn op_first_or_none(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(if args.is_empty() { w_none() } else { args[0] })
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

crate::py_module! {
    "operator",
    functions: {
        "index"    / 1 = op_index,
        "add"      / 2 = op_add,
        "sub"      / 2 = op_sub,
        "mul"      / 2 = op_mul,
        "truediv"  / 2 = op_truediv,
        "floordiv" / 2 = op_floordiv,
        "mod"      / 2 = op_mod,
        "pow"      / 2 = op_pow,
        "neg"      / 1 = op_neg,
        "pos"      / 1 = op_pos,
        "abs"      / 1 = op_abs,
        "invert"   / 1 = op_invert,
        "lshift"   / 2 = op_lshift,
        "rshift"   / 2 = op_rshift,
        "and_"     / 2 = op_and,
        "or_"      / 2 = op_or,
        "xor"      / 2 = op_xor,
        "not_"     / 1 = op_not,
        // interp_operator.py:138
        "truth"    / 1 = op_truth,
        "is_"      / 2 = op_is,
        "is_not"   / 2 = op_is_not,
        "contains" / 2 = op_contains,
        "getitem"  / 2 = op_getitem,
        "setitem"  / 3 = op_setitem,
        "delitem"  / 2 = op_delitem,
        // Underscore aliases (__add__ / __sub__ / __mul__ via operator).
        "__add__"  / 2 = op_add,
        "__sub__"  / 2 = op_sub,
        "__mul__"  / 2 = op_mul,
        "eq"       / 2 = op_eq,
        "lt"       / 2 = op_lt,
        "gt"       / 2 = op_gt,
        "le"       / 2 = op_le,
        "ge"       / 2 = op_ge,
        "ne"       / 2 = op_ne,
        // itemgetter / attrgetter / methodcaller stubs — return first arg.
        "itemgetter"   / * = op_first_or_none,
        "attrgetter"   / * = op_first_or_none,
        "methodcaller" / * = op_first_or_none,
        "length_hint"  / * = op_length_hint,
    },
}
