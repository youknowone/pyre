//! `_contextvars` module — PyPy: `lib_pypy/_contextvars.py`.
//!
//! `Context` is the app-level line-by-line port because its persistent Map
//! operations and `run()`'s try/finally are already expressed exactly there.
//! ContextVar and Token remain interpreter-level while their state operations
//! are ported incrementally.

use pyre_object::*;
use std::sync::OnceLock;

pub(crate) fn context_var_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("_contextvars.ContextVar", |ns| {
            let signature =
                crate::gateway::Signature::new(vec!["cls", "name", "default"], None, None, 1, 2);
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__new__",
                    crate::typedef::make_new_descr_with_signature(
                        context_var_new,
                        signature.clone(),
                    ),
                );
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__init__",
                    crate::make_builtin_function_with_signature(
                        "__init__",
                        |_| Ok(w_none()),
                        crate::gateway::Signature::new(
                            vec!["self", "name", "default"],
                            None,
                            None,
                            1,
                            2,
                        ),
                    ),
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
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "name",
                    crate::typedef::make_getset_descriptor_named(
                        crate::make_builtin_function_with_arity("name", context_var_name_get, 2),
                        "name",
                    ),
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
    // default=_NO_DEFAULT): the signature-aware gateway supplies
    // [cls, name, default], with PY_NULL for an omitted default.
    if args.len() < 2 || args[1].is_null() {
        return Err(crate::PyError::type_error(
            "ContextVar() missing required argument: 'name'",
        ));
    }
    if !unsafe { is_str(args[1]) } {
        return Err(crate::PyError::type_error(
            "context variable name must be a str",
        ));
    }
    // CPython 3.14 contextvar_new stores the name's hash eagerly; this also
    // rejects an unhashable str subclass at construction time.
    crate::baseobjspace::hash_w_strict(args[1])?;
    let obj = w_instance_new(context_var_type());
    crate::baseobjspace::setattr_str(obj, "_name", args[1])?;
    if let Some(&default) = args.get(2)
        && !default.is_null()
    {
        crate::baseobjspace::setattr_str(obj, "_default", default)?;
    }
    Ok(obj)
}

fn context_var_name_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::getattr_str(args[1], "_name")
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
    extra_init: |ns| {
        let context_var = crate::module_ns_get(ns, "ContextVar")
            .expect("_contextvars.ContextVar must be installed first");
        crate::importing::appleveldef_install_seeded(
            ns,
            include_str!("_contextvars_app.py"),
            "_contextvars_app.py",
            &["Context", "copy_context"],
            &[("ContextVar", context_var)],
        );
    },
}
