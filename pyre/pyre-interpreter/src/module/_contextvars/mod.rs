//! _contextvars module — PyPy: `pypy/module/_contextvars/`.
//!
//! Stub providing ContextVar / Context / Token shells.  `ContextVar`
//! returns an opaque object with `.get(default=None)` and `.set(value)`
//! attached as builtin functions — adequate for callers that only use
//! the decorator-style API; full contextvar propagation across tasks is
//! not modelled.

crate::py_module! {
    "_contextvars",
    interpleveldefs: {
        "ContextVar" => crate::make_builtin_function("ContextVar", |args| {
            let obj = pyre_object::w_instance_new(crate::typedef::w_object());
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
                        Ok(pyre_object::w_none())
                    }
                }),
            );
            let _ = crate::baseobjspace::setattr(
                obj,
                "set",
                crate::make_builtin_function_with_arity("set", |_| Ok(pyre_object::w_none()), 2),
            );
            Ok(obj)
        }),
        "Context" => crate::make_builtin_function_with_arity(
            "Context", |_| Ok(pyre_object::w_none()), 0),
        "Token" => crate::make_builtin_function_with_arity(
            "Token", |_| Ok(pyre_object::w_none()), 0),
        "copy_context" => crate::make_builtin_function_with_arity(
            "copy_context", |_| Ok(pyre_object::w_none()), 0),
    }
}
