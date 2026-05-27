//! _contextvars module — PyPy: `pypy/module/_contextvars/`.
//!
//! Stub providing ContextVar / Context / Token shells.  `ContextVar`
//! returns an opaque object with `.get(default=None)` and `.set(value)`
//! attached as builtin functions — adequate for callers that only use
//! the decorator-style API; full contextvar propagation across tasks is
//! not modelled.

use pyre_object::*;

fn context_var(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `interp_contextvars` ContextVar(name, *, default=MISSING) — name is required.
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "ContextVar() missing required argument: 'name'",
        ));
    }
    let obj = w_instance_new(crate::typedef::w_object());
    let _ = crate::baseobjspace::setattr(obj, "name", args[0]);
    let _ = crate::baseobjspace::setattr(
        obj,
        "get",
        // `W_ContextVar.get(*default)` raises LookupError when no
        // current value and no default supplied.
        crate::make_builtin_function("get", |args| {
            if args.len() > 1 {
                return Ok(args[1]);
            }
            Err(crate::PyError::lookup_error(
                "context variable has no value and no default supplied",
            ))
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
