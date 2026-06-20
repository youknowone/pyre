//! _contextvars module — PyPy: `pypy/module/_contextvars/`.
//!
//! Stub providing ContextVar / Context / Token shells.  `ContextVar`
//! returns an opaque object with `.get(default=None)` and `.set(value)`
//! attached as builtin functions — adequate for callers that only use
//! the decorator-style API; full contextvar propagation across tasks is
//! not modelled.

use pyre_object::*;

/// `ContextVar` instance type — needs `__dict__` so `name` / `get` / `set`
/// can be stored as instance attributes.  Plain `object` instances reject
/// `setattr`, leaving the shell without its methods.
fn context_var_type() -> PyObjectRef {
    thread_local! {
        static CELL: std::cell::OnceCell<PyObjectRef> =
            const { std::cell::OnceCell::new() };
    }
    CELL.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("ContextVar", |_| {});
            unsafe { typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

fn context_var(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `interp_contextvars` ContextVar(name, *, default=MISSING) — name is required.
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "ContextVar() missing required argument: 'name'",
        ));
    }
    let obj = w_instance_new(context_var_type());
    let _ = crate::baseobjspace::setattr_str(obj, "name", args[0]);
    let _ = crate::baseobjspace::setattr_str(
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
    let _ = crate::baseobjspace::setattr_str(
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
