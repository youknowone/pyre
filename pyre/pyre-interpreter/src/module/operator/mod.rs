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

// Binary arithmetic / comparison thunks share one shape — call
// `baseobjspace::OP(args[0], args[1])` and unwrap-or-none the result.
// Inline closures below preserve the per-name `assert!` checks.
use crate::baseobjspace::{
    self, CompareOp, add, and_, contains, delitem, floordiv, getitem, invert, is_true, lshift,
    mod_, mul, neg, or_, pos, pow, rshift, setitem, sub, truediv, xor,
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
        "add"      / 2 = |args| { assert!(args.len() == 2); Ok(add(args[0], args[1]).unwrap_or(w_none())) },
        "sub"      / 2 = |args| { assert!(args.len() == 2); Ok(sub(args[0], args[1]).unwrap_or(w_none())) },
        "mul"      / 2 = |args| { assert!(args.len() == 2); Ok(mul(args[0], args[1]).unwrap_or(w_none())) },
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
        "not_"     / 1 = |args| Ok(w_bool_from(!is_true(args[0]))),
        // interp_operator.py:138
        "truth"    / 1 = |args| Ok(w_bool_from(is_true(args[0]))),
        "is_"      / 2 = |args| Ok(w_bool_from(std::ptr::eq(args[0], args[1]))),
        "is_not"   / 2 = |args| Ok(w_bool_from(!std::ptr::eq(args[0], args[1]))),
        "contains" / 2 = |args| Ok(w_bool_from(contains(args[0], args[1])?)),
        "getitem"  / 2 = |args| getitem(args[0], args[1]),
        "setitem"  / 3 = |args| { setitem(args[0], args[1], args[2])?; Ok(w_none()) },
        "delitem"  / 2 = |args| { delitem(args[0], args[1])?; Ok(w_none()) },
        // Underscore aliases (__add__ / __sub__ / __mul__ via operator).
        "__add__"  / 2 = |args| { assert!(args.len() == 2); Ok(add(args[0], args[1]).unwrap_or(w_none())) },
        "__sub__"  / 2 = |args| { assert!(args.len() == 2); Ok(sub(args[0], args[1]).unwrap_or(w_none())) },
        "__mul__"  / 2 = |args| { assert!(args.len() == 2); Ok(mul(args[0], args[1]).unwrap_or(w_none())) },
        "eq" / 2 = |args| { assert!(args.len() == 2); Ok(baseobjspace::compare(args[0], args[1], CompareOp::Eq).unwrap_or(w_none())) },
        "lt" / 2 = |args| { assert!(args.len() == 2); Ok(baseobjspace::compare(args[0], args[1], CompareOp::Lt).unwrap_or(w_none())) },
        "gt" / 2 = |args| { assert!(args.len() == 2); Ok(baseobjspace::compare(args[0], args[1], CompareOp::Gt).unwrap_or(w_none())) },
        "le" / 2 = |args| baseobjspace::compare(args[0], args[1], CompareOp::Le),
        "ge" / 2 = |args| baseobjspace::compare(args[0], args[1], CompareOp::Ge),
        "ne" / 2 = |args| baseobjspace::compare(args[0], args[1], CompareOp::Ne),
        "length_hint"  / * = op_length_hint,
    },
}
