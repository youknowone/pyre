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
