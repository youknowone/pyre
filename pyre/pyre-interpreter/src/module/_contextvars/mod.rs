//! _contextvars module — PyPy: `lib_pypy/_contextvars.py`.
//!
//! Stub providing ContextVar / Context / Token shells.  `ContextVar`
//! returns an opaque object with `.get(default=None)` and `.set(value)`
//! attached as builtin functions — adequate for callers that only use
//! the decorator-style API; full contextvar propagation across tasks is
//! not modelled.

use pyre_object::*;
use std::sync::OnceLock;

fn context_type() -> PyObjectRef {
    // PyPy exposes one interpreter-level Context typedef; the type identity
    // must not split when the importing thread changes.
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        crate::typedef::make_builtin_type("Context", |ns| {
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "run",
                    crate::make_builtin_function("run", |args| {
                        let callable = args.get(1).copied().ok_or_else(|| {
                            crate::PyError::type_error("run() missing callable argument")
                        })?;
                        crate::call::call_function_impl_result(callable, &args[2..])
                    }),
                )
            };
        }) as usize
    }) as PyObjectRef
}

fn new_context(_: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_instance_new(context_type()))
}

/// `ContextVar` instance type — needs `__dict__` so `name` / `get` / `set`
/// can be stored as instance attributes.  Plain `object` instances reject
/// `setattr`, leaving the shell without its methods.
fn context_var_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("ContextVar", |_| {});
        unsafe { typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
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
        "Context"      / 0 = new_context,
        "Token"        / 0 = |_| Ok(w_none()),
        "copy_context" / 0 = new_context,
    },
}
