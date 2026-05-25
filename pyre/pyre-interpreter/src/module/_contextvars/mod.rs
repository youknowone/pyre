//! _contextvars module — PyPy: `pypy/module/_contextvars/`.
//!
//! Stub providing ContextVar / Context / Token shells.  `ContextVar`
//! returns an opaque object with `.get(default=None)` and `.set(value)`
//! attached as builtin functions — adequate for callers that only use
//! the decorator-style API; full contextvar propagation across tasks is
//! not modelled.

use pyre_object::*;

fn context_var(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = w_instance_new(crate::typedef::w_object());
    if !args.is_empty() {
        let _ = crate::baseobjspace::setattr(obj, "name", args[0]);
    }
    let _ = crate::baseobjspace::setattr(
        obj,
        "get",
        crate::make_builtin_function("get", |args| {
            if args.len() > 1 {
                Ok(args[1])
            } else {
                Ok(w_none())
            }
        }),
    );
    let _ = crate::baseobjspace::setattr(
        obj,
        "set",
        crate::make_builtin_function_with_arity("set", |_| Ok(w_none()), 2),
    );
    Ok(obj)
}

crate::py_module! {
    "_contextvars",
    functions: {
        "ContextVar"   / * = context_var,
        "Context"      / 0 = |_| Ok(w_none()),
        "Token"        / 0 = |_| Ok(w_none()),
        "copy_context" / 0 = |_| Ok(w_none()),
    },
}
