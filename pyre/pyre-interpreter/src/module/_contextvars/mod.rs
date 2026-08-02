//! `_contextvars` module — PyPy: `lib_pypy/_contextvars.py`.
//!
//! `Context` is the app-level line-by-line port because its persistent Map
//! operations and `run()`'s try/finally are already expressed exactly there.
//! ContextVar and Token remain interpreter-level while their state operations
//! are ported incrementally.  `copy_context` is interp-level too — a
//! non-binding builtin over `current_context`; PyPy keeps it in
//! `lib_pypy/_contextvars.py`, but `Python/context.c` ships it as a C function.

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
                    "reset",
                    crate::make_builtin_function_with_arity("reset", context_var_reset, 2),
                );
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__repr__",
                    crate::make_builtin_function_with_arity("__repr__", context_var_repr, 1),
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
            "ContextVar() takes exactly 1 positional argument (0 given)",
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
    if let Some(context) = current_context(false)? {
        match crate::baseobjspace::getitem(context, args[0]) {
            Ok(value) => return Ok(value),
            Err(err) if err.kind == crate::PyErrorKind::KeyError => {}
            Err(err) => return Err(err),
        }
    }
    if let Some(&default) = args.get(1) {
        return Ok(default);
    }
    if let Some(default) = crate::baseobjspace::findattr_result(args[0], "_default")? {
        return Ok(default);
    }
    Err(crate::PyError::lookup_error(context_var_repr_string(
        args[0],
    )?))
}

fn call_method_result(
    obj: PyObjectRef,
    name: &str,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let method = crate::baseobjspace::getattr_str(obj, name)?;
    crate::call::call_function_impl_result(method, args)
}

fn current_context(create: bool) -> Result<Option<PyObjectRef>, crate::PyError> {
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if ec.is_null() {
        return Err(crate::PyError::runtime_error(
            "no current execution context",
        ));
    }
    let mut context = unsafe { (*ec).contextvar_context };
    if context.is_null() || std::ptr::eq(context, w_none()) {
        if !create {
            return Ok(None);
        }
        // `_context_type` is stored on the immortal ContextVar type's dict by
        // module init, which roots the app-level Context type without a raw
        // side table or thread-local copy.
        let context_type = crate::baseobjspace::getattr_str(context_var_type(), "_context_type")?;
        context = crate::call::call_function_impl_result(context_type, &[])?;
        unsafe { (*ec).contextvar_context = context };
    }
    Ok(Some(context))
}

fn context_var_set(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // PyPy lib_pypy/_contextvars.py ContextVar.set, line by line: read the
    // old binding, persistently replace Context._data, and return a token
    // tied to this exact Context.
    let context = current_context(true)?.expect("create=true returns a Context");
    let data = crate::baseobjspace::getattr_str(context, "_data")?;
    let old_value = match crate::baseobjspace::getitem(data, args[0]) {
        Ok(value) => value,
        Err(err) if err.kind == crate::PyErrorKind::KeyError => token_missing(),
        Err(err) => return Err(err),
    };
    let updated_data = call_method_result(data, "set", &[args[0], args[1]])?;
    crate::baseobjspace::setattr_str(context, "_data", updated_data)?;
    new_token(context, args[0], old_value)
}

fn context_var_reset(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let token = args[1];
    if !unsafe { crate::baseobjspace::isinstance_w(token, token_type()) } {
        return Err(crate::PyError::type_error(format!(
            "expected an instance of Token, got {}",
            crate::type_methods::arg_type_name(token),
        )));
    }
    if crate::baseobjspace::is_true(crate::baseobjspace::getattr_str(token, "_used")?)? {
        return Err(crate::PyError::runtime_error(format!(
            "{} has already been used once",
            token_repr_string(token)?,
        )));
    }
    let token_var = crate::baseobjspace::getattr_str(token, "_var")?;
    if !std::ptr::eq(token_var, args[0]) {
        return Err(crate::PyError::value_error(
            "Token was created by a different ContextVar",
        ));
    }
    let context = current_context(true)?.expect("create=true returns a Context");
    let token_context = crate::baseobjspace::getattr_str(token, "_context")?;
    if !std::ptr::eq(token_context, context) {
        return Err(crate::PyError::value_error(
            "Token was created in a different Context",
        ));
    }
    let data = crate::baseobjspace::getattr_str(context, "_data")?;
    let old_value = crate::baseobjspace::getattr_str(token, "_old_value")?;
    let updated_data = if std::ptr::eq(old_value, token_missing()) {
        call_method_result(data, "delete", &[token_var])?
    } else {
        call_method_result(data, "set", &[token_var, old_value])?
    };
    crate::baseobjspace::setattr_str(context, "_data", updated_data)?;
    crate::baseobjspace::setattr_str(token, "_used", w_bool_from(true))?;
    Ok(w_none())
}

fn context_var_repr_string(obj: PyObjectRef) -> Result<String, crate::PyError> {
    let Some(_guard) = crate::display::ReprGuard::enter(obj) else {
        return Ok("...".to_string());
    };
    let name = crate::baseobjspace::getattr_str(obj, "_name")?;
    let name_repr = unsafe { crate::display::py_repr_wtf8(name)? }
        .to_string_lossy()
        .into_owned();
    let default = match crate::baseobjspace::findattr_result(obj, "_default")? {
        Some(value) => {
            let value_repr = unsafe { crate::display::py_repr_wtf8(value)? }
                .to_string_lossy()
                .into_owned();
            format!(" default={value_repr}")
        }
        None => String::new(),
    };
    Ok(format!(
        "<ContextVar name={name_repr}{default} at 0x{:x}>",
        obj as usize
    ))
}

fn context_var_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_str_new(&context_var_repr_string(args[0])?))
}

fn token_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("_contextvars.Token", |ns| unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                "MISSING",
                w_instance_new(token_missing_type()),
            );
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                "var",
                crate::typedef::make_getset_descriptor_named(
                    crate::make_builtin_function_with_arity("var", token_var_get, 2),
                    "var",
                ),
            );
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                "old_value",
                crate::typedef::make_getset_descriptor_named(
                    crate::make_builtin_function_with_arity("old_value", token_old_value_get, 2),
                    "old_value",
                ),
            );
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                "__repr__",
                crate::make_builtin_function_with_arity("__repr__", token_repr, 1),
            );
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                "__enter__",
                crate::make_builtin_function_with_arity("__enter__", token_enter, 1),
            );
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                "__exit__",
                crate::make_builtin_function_with_arity("__exit__", token_exit, 4),
            );
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
        unsafe { typeobject::w_type_set_hasdict(tp, true) };
        unsafe { typeobject::w_type_set_acceptable_as_base_class(tp, false) };
        tp as usize
    }) as PyObjectRef
}

fn token_missing_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("_contextvars.Token.MISSING", |ns| unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                "__repr__",
                crate::make_builtin_function_with_arity(
                    "__repr__",
                    |_| Ok(w_str_new("<Token.MISSING>")),
                    1,
                ),
            );
        });
        unsafe { typeobject::w_type_set_acceptable_as_base_class(tp, false) };
        tp as usize
    }) as PyObjectRef
}

fn token_missing() -> PyObjectRef {
    // The Token type dict is the shared owner/root, matching PyPy's
    // `Token.MISSING` class attribute.  Do not cache this movable singleton
    // in a raw-pointer OnceLock.
    crate::baseobjspace::getattr_str(token_type(), "MISSING")
        .expect("Token.MISSING is installed with the Token type")
}

fn new_token(
    context: PyObjectRef,
    var: PyObjectRef,
    old_value: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let token = w_instance_new(token_type());
    crate::baseobjspace::setattr_str(token, "_context", context)?;
    crate::baseobjspace::setattr_str(token, "_var", var)?;
    crate::baseobjspace::setattr_str(token, "_old_value", old_value)?;
    crate::baseobjspace::setattr_str(token, "_used", w_bool_from(false))?;
    Ok(token)
}

fn token_var_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::getattr_str(args[1], "_var")
}

fn token_old_value_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::getattr_str(args[1], "_old_value")
}

fn token_repr_string(token: PyObjectRef) -> Result<String, crate::PyError> {
    let var = crate::baseobjspace::getattr_str(token, "_var")?;
    let var_repr = unsafe { crate::display::py_repr_wtf8(var)? }
        .to_string_lossy()
        .into_owned();
    let used = crate::baseobjspace::is_true(crate::baseobjspace::getattr_str(token, "_used")?)?;
    Ok(format!(
        "<Token {}var={} at 0x{:x}>",
        if used { "used " } else { "" },
        var_repr,
        token as usize,
    ))
}

fn token_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_str_new(&token_repr_string(args[0])?))
}

fn token_enter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(args[0])
}

fn token_exit(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let var = crate::baseobjspace::getattr_str(args[0], "_var")?;
    context_var_reset(&[var, args[0]])?;
    Ok(w_bool_from(false))
}

crate::py_module! {
    "_contextvars",
    interpleveldefs: {
        "ContextVar" => context_var_type(),
        "Token" => token_type(),
    },
    functions: {
        // `copy_context()` — snapshot the current context; `Context.copy()`
        // shares the persistent `_data` Map.  `current_context(true)` is the
        // same "read-or-create the thread's Context" step `set`/`reset` use.
        "copy_context" / 0 = |_| {
            let context = current_context(true)?.expect("create=true returns a Context");
            call_method_result(context, "copy", &[])
        },
    },
    extra_init: |ns| {
        let context_var = crate::module_ns_get(ns, "ContextVar")
            .expect("_contextvars.ContextVar must be installed first");
        crate::importing::appleveldef_install_seeded(
            ns,
            include_str!("_contextvars_app.py"),
            "_contextvars_app.py",
            &["Context"],
            &[("ContextVar", context_var)],
        );
        let context = crate::module_ns_get(ns, "Context")
            .expect("_contextvars.Context must be installed by appleveldefs");
        let context_var_dict =
            unsafe { pyre_object::w_type_get_dict_ptr(context_var) as PyObjectRef };
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                context_var_dict,
                "_context_type",
                context,
            );
        }
    },
}
