//! operator module definition.
//!
//! PyPy equivalent: pypy/module/operator/

use crate::{PyNamespace, make_builtin_function, namespace_store};
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
    // Try __index__ dunder
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

pub fn init(ns: &mut PyNamespace) {
    namespace_store(ns, "index", make_builtin_function("index", op_index));
    namespace_store(ns, "add", make_builtin_function("add", op_add));
    namespace_store(ns, "sub", make_builtin_function("sub", op_sub));
    namespace_store(ns, "mul", make_builtin_function("mul", op_mul));
    namespace_store(
        ns,
        "truediv",
        make_builtin_function("truediv", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::truediv(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "floordiv",
        make_builtin_function("floordiv", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::floordiv(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "mod",
        make_builtin_function("mod", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::mod_(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "pow",
        make_builtin_function("pow", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::pow(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "neg",
        make_builtin_function("neg", |args| {
            assert!(args.len() == 1);
            crate::baseobjspace::neg(args[0])
        }),
    );
    namespace_store(
        ns,
        "pos",
        make_builtin_function("pos", |args| {
            assert!(args.len() == 1);
            Ok(args[0])
        }),
    );
    namespace_store(
        ns,
        "abs",
        make_builtin_function("abs", |args| {
            assert!(args.len() == 1);
            crate::builtins::builtin_abs(args)
        }),
    );
    namespace_store(
        ns,
        "invert",
        make_builtin_function("invert", |args| {
            assert!(args.len() == 1);
            crate::baseobjspace::invert(args[0])
        }),
    );
    namespace_store(
        ns,
        "lshift",
        make_builtin_function("lshift", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::lshift(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "rshift",
        make_builtin_function("rshift", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::rshift(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "and_",
        make_builtin_function("and_", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::and_(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "or_",
        make_builtin_function("or_", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::or_(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "xor",
        make_builtin_function("xor", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::xor(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "not_",
        make_builtin_function("not_", |args| {
            if args.is_empty() {
                return Ok(w_bool_from(true));
            }
            Ok(w_bool_from(!crate::baseobjspace::is_true(args[0])))
        }),
    );
    namespace_store(
        ns,
        "is_",
        make_builtin_function("is_", |args| {
            assert!(args.len() == 2);
            Ok(w_bool_from(std::ptr::eq(args[0], args[1])))
        }),
    );
    namespace_store(
        ns,
        "is_not",
        make_builtin_function("is_not", |args| {
            assert!(args.len() == 2);
            Ok(w_bool_from(!std::ptr::eq(args[0], args[1])))
        }),
    );
    namespace_store(
        ns,
        "contains",
        make_builtin_function("contains", |args| {
            assert!(args.len() == 2);
            Ok(w_bool_from(crate::baseobjspace::contains(
                args[0], args[1],
            )?))
        }),
    );
    namespace_store(
        ns,
        "getitem",
        make_builtin_function("getitem", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::getitem(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "setitem",
        make_builtin_function("setitem", |args| {
            assert!(args.len() == 3);
            crate::baseobjspace::setitem(args[0], args[1], args[2])?;
            Ok(w_none())
        }),
    );
    namespace_store(
        ns,
        "delitem",
        make_builtin_function("delitem", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::delitem(args[0], args[1])?;
            Ok(w_none())
        }),
    );
    // Underscore aliases (CPython: __add__/__sub__/... via operator module).
    namespace_store(ns, "__add__", make_builtin_function("__add__", op_add));
    namespace_store(ns, "__sub__", make_builtin_function("__sub__", op_sub));
    namespace_store(ns, "__mul__", make_builtin_function("__mul__", op_mul));
    namespace_store(ns, "eq", make_builtin_function("eq", op_eq));
    namespace_store(ns, "lt", make_builtin_function("lt", op_lt));
    namespace_store(ns, "gt", make_builtin_function("gt", op_gt));
    namespace_store(
        ns,
        "le",
        make_builtin_function("le", |args| {
            crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Le)
        }),
    );
    namespace_store(
        ns,
        "ge",
        make_builtin_function("ge", |args| {
            crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Ge)
        }),
    );
    namespace_store(
        ns,
        "ne",
        make_builtin_function("ne", |args| {
            crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Ne)
        }),
    );
    // itemgetter/attrgetter stubs — return callable objects
    namespace_store(
        ns,
        "itemgetter",
        make_builtin_function("itemgetter", |args| {
            // itemgetter(key) → lambda obj: obj[key]
            Ok(if args.is_empty() { w_none() } else { args[0] })
        }),
    );
    namespace_store(
        ns,
        "attrgetter",
        make_builtin_function("attrgetter", |args| {
            Ok(if args.is_empty() { w_none() } else { args[0] })
        }),
    );
    namespace_store(
        ns,
        "methodcaller",
        make_builtin_function("methodcaller", |args| {
            Ok(if args.is_empty() { w_none() } else { args[0] })
        }),
    );
    namespace_store(
        ns,
        "length_hint",
        make_builtin_function("length_hint", |args| {
            if args.is_empty() {
                return Ok(w_int_new(0));
            }
            crate::baseobjspace::len(args[0]).or(Ok(w_int_new(0)))
        }),
    );
}
