//! operator module definition.
//!
//! PyPy equivalent: pypy/module/operator/

use crate::{PyNamespace, builtin_code_new, namespace_store};
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
    Ok(crate::space_call_function_or_identity(obj, "__index__"))
}

fn op_add(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(crate::baseobjspace::py_add(args[0], args[1]).unwrap_or(w_none()))
}

fn op_sub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(crate::baseobjspace::py_sub(args[0], args[1]).unwrap_or(w_none()))
}

fn op_mul(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(crate::baseobjspace::py_mul(args[0], args[1]).unwrap_or(w_none()))
}

fn op_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(
        crate::baseobjspace::py_compare(args[0], args[1], crate::baseobjspace::CompareOp::Eq)
            .unwrap_or(w_none()),
    )
}

fn op_lt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(
        crate::baseobjspace::py_compare(args[0], args[1], crate::baseobjspace::CompareOp::Lt)
            .unwrap_or(w_none()),
    )
}

fn op_gt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(
        crate::baseobjspace::py_compare(args[0], args[1], crate::baseobjspace::CompareOp::Gt)
            .unwrap_or(w_none()),
    )
}

pub fn init(ns: &mut PyNamespace) {
    namespace_store(ns, "index", builtin_code_new("index", op_index));
    namespace_store(ns, "add", builtin_code_new("add", op_add));
    namespace_store(ns, "sub", builtin_code_new("sub", op_sub));
    namespace_store(ns, "mul", builtin_code_new("mul", op_mul));
    namespace_store(ns, "eq", builtin_code_new("eq", op_eq));
    namespace_store(ns, "lt", builtin_code_new("lt", op_lt));
    namespace_store(ns, "gt", builtin_code_new("gt", op_gt));
    namespace_store(
        ns,
        "le",
        builtin_code_new("le", |args| {
            crate::baseobjspace::py_compare(args[0], args[1], crate::baseobjspace::CompareOp::Le)
        }),
    );
    namespace_store(
        ns,
        "ge",
        builtin_code_new("ge", |args| {
            crate::baseobjspace::py_compare(args[0], args[1], crate::baseobjspace::CompareOp::Ge)
        }),
    );
    namespace_store(
        ns,
        "ne",
        builtin_code_new("ne", |args| {
            crate::baseobjspace::py_compare(args[0], args[1], crate::baseobjspace::CompareOp::Ne)
        }),
    );
    // itemgetter/attrgetter stubs — return callable objects
    namespace_store(
        ns,
        "itemgetter",
        builtin_code_new("itemgetter", |args| {
            // itemgetter(key) → lambda obj: obj[key]
            Ok(if args.is_empty() { w_none() } else { args[0] })
        }),
    );
    namespace_store(
        ns,
        "attrgetter",
        builtin_code_new("attrgetter", |args| {
            Ok(if args.is_empty() { w_none() } else { args[0] })
        }),
    );
    namespace_store(
        ns,
        "methodcaller",
        builtin_code_new("methodcaller", |args| {
            Ok(if args.is_empty() { w_none() } else { args[0] })
        }),
    );
    namespace_store(
        ns,
        "length_hint",
        builtin_code_new("length_hint", |args| {
            if args.is_empty() {
                return Ok(w_int_new(0));
            }
            crate::baseobjspace::py_len(args[0]).or(Ok(w_int_new(0)))
        }),
    );
}
