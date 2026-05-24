//! _contextvars implementation — PyPy: pypy/module/_contextvars/interp_contextvars.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// _contextvars stub
pub fn register_module(ns: &mut DictStorage) {
    // ContextVar(name, *, default=_MISSING) — context variable
    crate::dict_storage_store(
        ns,
        "ContextVar",
        crate::make_builtin_function("ContextVar", |args| {
            // Return stub object with get/set methods
            let obj = pyre_object::w_instance_new(crate::typedef::w_object());
            if !args.is_empty() {
                let _ = crate::baseobjspace::setattr(obj, "name", args[0]);
            }
            // get() returns default or raises LookupError
            let _ = crate::baseobjspace::setattr(
                obj,
                "get",
                crate::make_builtin_function("get", |args| {
                    // Return default if provided
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
    );
    crate::dict_storage_store(
        ns,
        "Context",
        crate::make_builtin_function_with_arity("Context", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "Token",
        crate::make_builtin_function_with_arity("Token", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "copy_context",
        crate::make_builtin_function_with_arity("copy_context", |_| Ok(pyre_object::w_none()), 0),
    );
}
