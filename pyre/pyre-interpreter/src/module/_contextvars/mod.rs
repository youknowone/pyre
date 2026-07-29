//! _contextvars module — PyPy: `lib_pypy/_contextvars.py`.
//!
//! Stub providing ContextVar / Context / Token shells.  Full contextvar
//! propagation across tasks is not modelled yet.

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

fn context_var_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("_contextvars.ContextVar", |ns| {
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__new__",
                    crate::typedef::make_new_descr(context_var_new),
                );
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__init__",
                    crate::make_builtin_function("__init__", |_| Ok(w_none())),
                );
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "get",
                    crate::make_builtin_function("get", context_var_get),
                );
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "set",
                    crate::make_builtin_function_with_arity("set", context_var_set, 2),
                );
                // PyPy lib_pypy/_contextvars.py ContextVar.__class_getitem__
                // and CPython 3.14 Python/context.c PyContextVar_methods.
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__class_getitem__",
                    pyre_object::function::w_classmethod_new(crate::make_builtin_function(
                        "__class_getitem__",
                        crate::_pypy_generic_alias::generic_alias_class_getitem,
                    )),
                );
            }
        });
        unsafe { typeobject::w_type_set_hasdict(tp, true) };
        unsafe { typeobject::w_type_set_acceptable_as_base_class(tp, false) };
        tp as usize
    }) as PyObjectRef
}

fn context_var_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // PyPy lib_pypy/_contextvars.py ContextVar.__init__(name, *,
    // default=_NO_DEFAULT): the type-call ABI supplies cls at position zero.
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() < 2 {
        return Err(crate::PyError::type_error(
            "ContextVar() missing required argument: 'name'",
        ));
    }
    if pos.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "ContextVar() takes at most 1 positional argument ({} given)",
            pos.len() - 1
        )));
    }
    if !unsafe { is_str(pos[1]) } {
        return Err(crate::PyError::type_error(
            "context variable name must be a str",
        ));
    }
    if let Some(dict) = kwargs {
        for (key, _) in unsafe { w_dict_str_entries_wtf8(dict) } {
            let key = key.as_str().unwrap_or("");
            if key != "__pyre_kw__" && key != "default" {
                return Err(crate::PyError::type_error(format!(
                    "'{key}' is an invalid keyword argument for ContextVar()"
                )));
            }
        }
    }
    let obj = w_instance_new(context_var_type());
    crate::baseobjspace::setattr_str(obj, "name", pos[1])?;
    if let Some(default) = crate::builtins::kwarg_get(kwargs, "default") {
        crate::baseobjspace::setattr_str(obj, "_default", default)?;
    }
    Ok(obj)
}

fn context_var_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "get() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }
    if let Some(&default) = args.get(1) {
        return Ok(default);
    }
    if let Some(default) = crate::baseobjspace::findattr_result(args[0], "_default")? {
        return Ok(default);
    }
    Err(crate::PyError::lookup_error(
        "context variable has no value and no default supplied",
    ))
}

fn context_var_set(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_none())
}

fn token_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("_contextvars.Token", |ns| unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                "__new__",
                crate::typedef::make_new_descr(|_| {
                    Err(crate::PyError::type_error(
                        "Tokens can only be created by ContextVars",
                    ))
                }),
            );
            // PyPy lib_pypy/_contextvars.py Token.__class_getitem__ and
            // CPython 3.14 Python/context.c PyContextTokenType_methods.
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                "__class_getitem__",
                pyre_object::function::w_classmethod_new(crate::make_builtin_function(
                    "__class_getitem__",
                    crate::_pypy_generic_alias::generic_alias_class_getitem,
                )),
            );
        });
        unsafe { typeobject::w_type_set_acceptable_as_base_class(tp, false) };
        tp as usize
    }) as PyObjectRef
}

crate::py_module! {
    "_contextvars",
    interpleveldefs: {
        "ContextVar" => context_var_type(),
        "Token" => token_type(),
    },
    functions: {
        "Context"      / 0 = new_context,
        "copy_context" / 0 = new_context,
    },
}
