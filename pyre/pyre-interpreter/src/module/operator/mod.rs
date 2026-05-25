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
    interpleveldefs: {
        "index"    => crate::make_builtin_function_with_arity("index",    op_index, 1),
        "add"      => crate::make_builtin_function_with_arity("add",      op_add,   2),
        "sub"      => crate::make_builtin_function_with_arity("sub",      op_sub,   2),
        "mul"      => crate::make_builtin_function_with_arity("mul",      op_mul,   2),
        "truediv"  => crate::make_builtin_function_with_arity("truediv",  op_truediv,  2),
        "floordiv" => crate::make_builtin_function_with_arity("floordiv", op_floordiv, 2),
        "mod"      => crate::make_builtin_function_with_arity("mod",      op_mod,   2),
        "pow"      => crate::make_builtin_function_with_arity("pow",      op_pow,   2),
        "neg"      => crate::make_builtin_function_with_arity("neg",      op_neg,   1),
        "pos"      => crate::make_builtin_function_with_arity("pos",      op_pos,   1),
        "abs"      => crate::make_builtin_function_with_arity("abs",      op_abs,   1),
        "invert"   => crate::make_builtin_function_with_arity("invert",   op_invert, 1),
        "lshift"   => crate::make_builtin_function_with_arity("lshift",   op_lshift, 2),
        "rshift"   => crate::make_builtin_function_with_arity("rshift",   op_rshift, 2),
        "and_"     => crate::make_builtin_function_with_arity("and_",     op_and,   2),
        "or_"      => crate::make_builtin_function_with_arity("or_",      op_or,    2),
        "xor"      => crate::make_builtin_function_with_arity("xor",      op_xor,   2),
        "not_"     => crate::make_builtin_function_with_arity("not_",     op_not,   1),
        // interp_operator.py:138
        "truth"    => crate::make_builtin_function_with_arity("truth",    op_truth, 1),
        "is_"      => crate::make_builtin_function_with_arity("is_",      op_is,    2),
        "is_not"   => crate::make_builtin_function_with_arity("is_not",   op_is_not, 2),
        "contains" => crate::make_builtin_function_with_arity("contains", op_contains, 2),
        "getitem"  => crate::make_builtin_function_with_arity("getitem",  op_getitem, 2),
        "setitem"  => crate::make_builtin_function_with_arity("setitem",  op_setitem, 3),
        "delitem"  => crate::make_builtin_function_with_arity("delitem",  op_delitem, 2),
        // Underscore aliases (__add__ / __sub__ / __mul__ via operator).
        "__add__"  => crate::make_builtin_function_with_arity("__add__",  op_add, 2),
        "__sub__"  => crate::make_builtin_function_with_arity("__sub__",  op_sub, 2),
        "__mul__"  => crate::make_builtin_function_with_arity("__mul__",  op_mul, 2),
        "eq"       => crate::make_builtin_function_with_arity("eq", op_eq, 2),
        "lt"       => crate::make_builtin_function_with_arity("lt", op_lt, 2),
        "gt"       => crate::make_builtin_function_with_arity("gt", op_gt, 2),
        "le"       => crate::make_builtin_function_with_arity("le", op_le, 2),
        "ge"       => crate::make_builtin_function_with_arity("ge", op_ge, 2),
        "ne"       => crate::make_builtin_function_with_arity("ne", op_ne, 2),
        // itemgetter / attrgetter / methodcaller stubs — return first arg.
        "itemgetter"   => crate::make_builtin_function("itemgetter",   op_first_or_none),
        "attrgetter"   => crate::make_builtin_function("attrgetter",   op_first_or_none),
        "methodcaller" => crate::make_builtin_function("methodcaller", op_first_or_none),
        "length_hint"  => crate::make_builtin_function("length_hint",  op_length_hint),
    }
}
