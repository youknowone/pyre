//! operator module definition.
//!
//! PyPy equivalent: pypy/module/operator/

use pyre_object::*;
use pyre_runtime::{PyNamespace, namespace_store, w_builtin_func_new};

fn op_index(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 1, "index() takes exactly one argument");
    let obj = args[0];
    unsafe {
        if is_int(obj) {
            return obj;
        }
        if is_bool(obj) {
            return w_int_new(if w_bool_get_value(obj) { 1 } else { 0 });
        }
    }
    // Try __index__ dunder
    pyre_runtime::space_call_function_or_identity(obj, "__index__")
}

fn op_add(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2);
    pyre_runtime::space::py_add(args[0], args[1]).unwrap_or(w_none())
}

fn op_sub(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2);
    pyre_runtime::space::py_sub(args[0], args[1]).unwrap_or(w_none())
}

fn op_mul(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2);
    pyre_runtime::space::py_mul(args[0], args[1]).unwrap_or(w_none())
}

fn op_eq(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2);
    pyre_runtime::space::py_compare(args[0], args[1], pyre_runtime::space::CompareOp::Eq)
        .unwrap_or(w_none())
}

fn op_lt(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2);
    pyre_runtime::space::py_compare(args[0], args[1], pyre_runtime::space::CompareOp::Lt)
        .unwrap_or(w_none())
}

fn op_gt(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2);
    pyre_runtime::space::py_compare(args[0], args[1], pyre_runtime::space::CompareOp::Gt)
        .unwrap_or(w_none())
}

pub fn init(ns: &mut PyNamespace) {
    namespace_store(ns, "index", w_builtin_func_new("index", op_index));
    namespace_store(ns, "add", w_builtin_func_new("add", op_add));
    namespace_store(ns, "sub", w_builtin_func_new("sub", op_sub));
    namespace_store(ns, "mul", w_builtin_func_new("mul", op_mul));
    namespace_store(ns, "eq", w_builtin_func_new("eq", op_eq));
    namespace_store(ns, "lt", w_builtin_func_new("lt", op_lt));
    namespace_store(ns, "gt", w_builtin_func_new("gt", op_gt));
}
