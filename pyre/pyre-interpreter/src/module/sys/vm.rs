//! `sys` module VM helpers.
//!
//! PyPy equivalent: `pypy/module/sys/vm.py`.

use crate::executioncontext::ActionFlagOps;
use crate::{
    make_builtin_function, make_builtin_function_with_arity,
    make_builtin_function_with_arity_and_maybe_sig, module_ns_store,
};
use pyre_object::*;
use std::sync::OnceLock;

/// Shared stub type for `sys._getframe`, `sys.stdout` and other module-level
/// sys attributes that expose attribute bags.
///
/// `typedef::w_object()` (plain `object`) cannot store instance attributes —
/// its type flag `hasdict` is false, matching CPython where `object()`
/// instances reject `__setattr__` unless their subclass explicitly opts in.
/// PyPy's `sys` module exposes these as dedicated W_Root types with their
/// own typedefs, not as bare `object` instances. The Rust port mirrors that
/// by installing a single `sys.namespace` type with `__dict__` in its
/// typedef slots so every stub instance supports `setattr`.
fn sys_namespace_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    let raw = *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("sys.namespace", |ns| {
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__init__",
                    crate::make_builtin_function("__init__", sys_namespace_init),
                )
            };
        });
        // The stubs want a per-instance mapdict store; a `__dict__`
        // rawdict key would instead claim the typedef manages the dict
        // (typedef.py:40) and suppress the mapdict one
        // (typeobject.py:253-257), so flip `hasdict` directly — the
        // `create_dict_slot` flag flip (typeobject.py:1222-1226).
        unsafe { w_type_set_hasdict(tp, true) };
        tp as usize
    });
    raw as PyObjectRef
}

fn sys_namespace_init(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let Some(&self_obj) = positional.first() else {
        return Err(crate::PyError::type_error(
            "__init__() missing 1 required positional argument: 'self'",
        ));
    };
    if positional.len() > 1 {
        return Err(crate::PyError::type_error(
            "types.SimpleNamespace() takes no positional arguments",
        ));
    }
    namespace_apply_kwargs(self_obj, kwargs)
}

/// Copy the keyword arguments into a namespace instance's dict, skipping the
/// `__pyre_kw__` marker. Shared by `sys.namespace` and `types.SimpleNamespace`
/// construction. `_structseq.py:172` `self.__dict__.update(kwargs)` writes the
/// instance dict directly, so `setdictvalue` is used rather than `setattr` — a
/// subclass `__setattr__` is not consulted during construction.
fn namespace_apply_kwargs(self_obj: PyObjectRef, kwargs: Option<PyObjectRef>) -> crate::PyResult {
    if let Some(dict) = kwargs {
        namespace_update_dict(self_obj, dict, true)?;
    } else {
        // `self.__dict__.update(kwargs)` evaluates `self.__dict__` first, so
        // a receiver without an instance dict raises even for no keywords.
        crate::baseobjspace::getattr_str(self_obj, "__dict__")?;
    }
    Ok(w_none())
}

/// CPython 3.14 `PyDict_Update(ns->ns_dict, source)`, with the flat builtin
/// ABI's private `__pyre_kw__` marker optionally omitted.  Validate every key
/// before the first store, matching `PyArg_ValidateKeywordArguments`: a bad
/// mapping cannot partially update the namespace.  The destination is the
/// real instance dict, so subclass `__setattr__` is deliberately bypassed.
fn namespace_update_dict(
    self_obj: PyObjectRef,
    source: PyObjectRef,
    skip_kw_marker: bool,
) -> Result<(), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    pyre_object::gc_roots::pin_root(source);
    let destination = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(sp),
        "__dict__",
    )?;
    pyre_object::gc_roots::pin_root(destination);
    let items = unsafe {
        pyre_object::w_dict_items(pyre_object::gc_roots::shadow_stack_get(sp + 1))
    };
    let items_sp = pyre_object::gc_roots::shadow_stack_len();
    for &(key, value) in &items {
        pyre_object::gc_roots::pin_root(key);
        pyre_object::gc_roots::pin_root(value);
    }
    for i in 0..items.len() {
        let key = pyre_object::gc_roots::shadow_stack_get(items_sp + i * 2);
        if !unsafe { pyre_object::is_str(key) } {
            return Err(crate::PyError::type_error("keywords must be strings"));
        }
    }
    for i in 0..items.len() {
        let key = pyre_object::gc_roots::shadow_stack_get(items_sp + i * 2);
        let value = pyre_object::gc_roots::shadow_stack_get(items_sp + i * 2 + 1);
        if skip_kw_marker
            && unsafe { pyre_object::w_str_get_wtf8(key).as_str() == Ok("__pyre_kw__") }
        {
            continue;
        }
        crate::type_methods::dict_store_checked(
            pyre_object::gc_roots::shadow_stack_get(sp + 2),
            key,
            value,
        )?;
    }
    Ok(())
}

/// Allocate a fresh stub instance whose type supports `setattr`. Used for
/// all the CPython-style attribute bags surfaced by the sys module.
fn make_sys_namespace_instance() -> PyObjectRef {
    w_instance_new(sys_namespace_type())
}

/// CPython 3.14 `namespace_init`: accept at most one positional mapping or
/// iterable of pairs, validate that its resulting dict has only string keys,
/// merge it into the instance, then overlay keyword arguments.  This is the
/// target-version delta from PyPy 3.11's keyword-only `_structseq.py` class.
fn simple_namespace_init(args: &[PyObjectRef]) -> crate::PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    for &arg in args {
        pyre_object::gc_roots::pin_root(arg);
    }
    let rooted = (0..args.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(sp + i))
        .collect::<Vec<_>>();
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(&rooted);
    let Some(&self_obj) = positional.first() else {
        return Err(crate::PyError::type_error(
            "__init__() missing 1 required positional argument: 'self'",
        ));
    };
    if positional.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "SimpleNamespace expected at most 1 argument, got {}",
            positional.len() - 1
        )));
    }
    // The kwargs carrier is not guaranteed to occupy the last raw ABI slot
    // relative to positional arguments.  Pin the parsed operands into a
    // canonical order before any allocation instead of deriving their slots
    // from the flat input layout.
    let operands_sp = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    if positional.len() == 2 {
        pyre_object::gc_roots::pin_root(positional[1]);
    }
    if let Some(kwargs) = kwargs {
        pyre_object::gc_roots::pin_root(kwargs);
    }
    if positional.len() == 2 {
        let temporary = w_dict_new();
        pyre_object::gc_roots::pin_root(temporary);
        let temporary_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let temporary = pyre_object::gc_roots::shadow_stack_get(temporary_slot);
        crate::type_methods::dict_update1(
            temporary,
            pyre_object::gc_roots::shadow_stack_get(operands_sp + 1),
        )?;
        namespace_update_dict(
            pyre_object::gc_roots::shadow_stack_get(operands_sp),
            pyre_object::gc_roots::shadow_stack_get(temporary_slot),
            false,
        )?;
    }
    namespace_apply_kwargs(
        pyre_object::gc_roots::shadow_stack_get(operands_sp),
        kwargs.map(|_| {
            pyre_object::gc_roots::shadow_stack_get(
                operands_sp + 1 + usize::from(positional.len() == 2),
            )
        }),
    )
}

/// `types.SimpleNamespace` — the attribute-bag type exposed as
/// `type(sys.implementation)` and re-published by `types.py:20`
/// (`SimpleNamespace = type(sys.implementation)`).
///
/// `_structseq.py:166 SimpleNamespace`, with CPython 3.14's newer constructor,
/// full rich-comparison surface, pickle reducer and `__replace__`.  Storage
/// remains PyPy-shaped: the values live in the instance dict, not a side
/// table or a second native mapping.
fn simple_namespace_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    let raw = *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("types.SimpleNamespace", |ns| {
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__init__",
                    crate::make_builtin_function("__init__", simple_namespace_init),
                );
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__repr__",
                    make_builtin_function_with_arity("__repr__", simple_namespace_repr, 1),
                );
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__eq__",
                    make_builtin_function_with_arity("__eq__", simple_namespace_eq, 2),
                );
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__ne__",
                    make_builtin_function_with_arity("__ne__", simple_namespace_ne, 2),
                );
                for (name, function) in [
                    ("__lt__", simple_namespace_lt as fn(&[PyObjectRef]) -> crate::PyResult),
                    ("__le__", simple_namespace_le),
                    ("__gt__", simple_namespace_gt),
                    ("__ge__", simple_namespace_ge),
                ] {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        name,
                        make_builtin_function_with_arity(name, function, 2),
                    );
                }
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__reduce__",
                    make_builtin_function_with_arity("__reduce__", simple_namespace_reduce, 1),
                );
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__replace__",
                    make_builtin_function("__replace__", simple_namespace_replace),
                );
                // SimpleNamespace defines no `__hash__`, so it inherits None
                // and is unhashable.
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, "__hash__", w_none());
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__dict__",
                    crate::typedef::dict_descr(),
                );
            }
        });
        unsafe { w_type_set_hasdict(tp, true) };
        tp as usize
    });
    raw as PyObjectRef
}

/// CPython 3.14 `namespace_repr`, layered over PyPy's recursion guard.  Exact
/// instances use `namespace`, subclasses use their concrete type name.  Walk
/// a snapshot of the insertion-ordered keys, then re-read each live value so
/// a re-entrant repr that mutates the dict has CPython's skip/update behavior.
fn simple_namespace_repr(args: &[PyObjectRef]) -> crate::PyResult {
    let Some(&self_obj) = args.first() else {
        return Err(crate::PyError::type_error(
            "__repr__() missing 1 required positional argument: 'self'",
        ));
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    let actual_type = crate::typedef::r#type(self_obj)
        .map(|tp| tp.as_ptr())
        .unwrap_or(simple_namespace_type());
    let name = if std::ptr::eq(actual_type, simple_namespace_type()) {
        "namespace".to_string()
    } else {
        unsafe { w_type_get_name(actual_type) }.to_string()
    };
    let Some(_guard) = crate::display::ReprGuard::enter(self_obj) else {
        return Ok(w_str_new(&format!("{name}(...)")));
    };
    let dict = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(sp),
        "__dict__",
    )?;
    pyre_object::gc_roots::pin_root(dict);
    let keys = unsafe {
        pyre_object::w_dict_items(pyre_object::gc_roots::shadow_stack_get(sp + 1))
            .into_iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>()
    };
    let keys_sp = pyre_object::gc_roots::shadow_stack_len();
    for &key in &keys {
        pyre_object::gc_roots::pin_root(key);
    }
    let mut parts = Vec::with_capacity(keys.len());
    for i in 0..keys.len() {
        let key = pyre_object::gc_roots::shadow_stack_get(keys_sp + i);
        if !unsafe { pyre_object::is_str(key) }
            || unsafe { pyre_object::w_str_len(key) == 0 }
        {
            continue;
        }
        let value = match crate::baseobjspace::getitem(
            pyre_object::gc_roots::shadow_stack_get(sp + 1),
            key,
        ) {
            Ok(value) => value,
            Err(err) if err.kind == crate::PyErrorKind::KeyError => continue,
            Err(err) => return Err(err),
        };
        pyre_object::gc_roots::pin_root(value);
        parts.push(format!(
            "{}={}",
            unsafe { crate::display::py_str(key)? },
            unsafe {
                crate::display::py_repr(pyre_object::gc_roots::shadow_stack_get(
                    pyre_object::gc_roots::shadow_stack_len() - 1,
                ))?
            }
        ));
    }
    Ok(w_str_new(&format!("{name}({})", parts.join(", "))))
}

/// `_structseq.py:185 SimpleNamespace.__eq__` — structural over `__dict__`
/// when `other` is a namespace, NotImplemented otherwise.
fn simple_namespace_eq(args: &[PyObjectRef]) -> crate::PyResult {
    simple_namespace_richcompare(args, "__eq__", crate::baseobjspace::CompareOp::Eq)
}

/// `_structseq.py:190 SimpleNamespace.__ne__`.
fn simple_namespace_ne(args: &[PyObjectRef]) -> crate::PyResult {
    simple_namespace_richcompare(args, "__ne__", crate::baseobjspace::CompareOp::Ne)
}

fn simple_namespace_lt(args: &[PyObjectRef]) -> crate::PyResult {
    simple_namespace_richcompare(args, "__lt__", crate::baseobjspace::CompareOp::Lt)
}

fn simple_namespace_le(args: &[PyObjectRef]) -> crate::PyResult {
    simple_namespace_richcompare(args, "__le__", crate::baseobjspace::CompareOp::Le)
}

fn simple_namespace_gt(args: &[PyObjectRef]) -> crate::PyResult {
    simple_namespace_richcompare(args, "__gt__", crate::baseobjspace::CompareOp::Gt)
}

fn simple_namespace_ge(args: &[PyObjectRef]) -> crate::PyResult {
    simple_namespace_richcompare(args, "__ge__", crate::baseobjspace::CompareOp::Ge)
}

fn simple_namespace_richcompare(
    args: &[PyObjectRef],
    name: &str,
    op: crate::baseobjspace::CompareOp,
) -> crate::PyResult {
    // `def __eq__(self, other)` — a missing argument is an arity error, not a
    // NotImplemented result.
    let (Some(&self_obj), Some(&other)) = (args.first(), args.get(1)) else {
        return Err(crate::PyError::type_error(format!(
            "SimpleNamespace.{name}() missing 1 required positional argument: '{}'",
            if args.is_empty() { "self" } else { "other" }
        )));
    };
    let other_type = crate::typedef::r#type(other)
        .map(|tp| tp.as_ptr())
        .unwrap_or(PY_NULL);
    if !unsafe { crate::baseobjspace::issubtype_w(other_type, simple_namespace_type()) } {
        return Ok(w_not_implemented());
    }
    // CPython 3.14 forwards all six operations to the two namespace dicts.
    // In particular, ordering reaches dict's TypeError instead of returning
    // NotImplemented from the namespace type itself.
    let self_dict = crate::baseobjspace::getattr_str(self_obj, "__dict__")?;
    let other_dict = crate::baseobjspace::getattr_str(other, "__dict__")?;
    crate::baseobjspace::compare(self_dict, other_dict, op)
}

/// CPython 3.14 `namespace_reduce`: `(type(self), (), self.__dict__)`.
fn simple_namespace_reduce(args: &[PyObjectRef]) -> crate::PyResult {
    let Some(&self_obj) = args.first() else {
        return Err(crate::PyError::type_error(
            "__reduce__() missing 1 required positional argument: 'self'",
        ));
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    let w_type = crate::typedef::r#type(self_obj)
        .map(|tp| tp.as_ptr())
        .unwrap_or(PY_NULL);
    pyre_object::gc_roots::pin_root(w_type);
    let w_args = w_tuple_new(Vec::new());
    pyre_object::gc_roots::pin_root(w_args);
    let w_dict = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(sp),
        "__dict__",
    )?;
    pyre_object::gc_roots::pin_root(w_dict);
    Ok(w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(sp + 1),
        pyre_object::gc_roots::shadow_stack_get(sp + 2),
        pyre_object::gc_roots::shadow_stack_get(sp + 3),
    ]))
}

/// CPython 3.14 `namespace_replace`: construct `type(self)()` first, require
/// that its actual type remains a SimpleNamespace subtype, copy the source
/// dict, then overlay keyword changes.
fn simple_namespace_replace(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let Some(&self_obj) = positional.first() else {
        return Err(crate::PyError::type_error(
            "__replace__() missing 1 required positional argument: 'self'",
        ));
    };
    if positional.len() != 1 {
        return Err(crate::PyError::type_error(
            "__replace__() takes no positional arguments",
        ));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    if let Some(kwargs) = kwargs {
        pyre_object::gc_roots::pin_root(kwargs);
    }
    let self_type = crate::typedef::r#type(self_obj)
        .map(|tp| tp.as_ptr())
        .unwrap_or(PY_NULL);
    pyre_object::gc_roots::pin_root(self_type);
    let result = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(
            sp + 1 + usize::from(kwargs.is_some()),
        ),
        &[],
    )?;
    pyre_object::gc_roots::pin_root(result);
    let result = pyre_object::gc_roots::shadow_stack_get(
        sp + 2 + usize::from(kwargs.is_some()),
    );
    let result_type = crate::typedef::r#type(result)
        .map(|tp| tp.as_ptr())
        .unwrap_or(PY_NULL);
    if !unsafe { crate::baseobjspace::issubtype_w(result_type, simple_namespace_type()) } {
        let constructed = unsafe {
            crate::baseobjspace::type_fully_qualified_name(
                pyre_object::gc_roots::shadow_stack_get(
                    sp + 1 + usize::from(kwargs.is_some()),
                ),
            )
        };
        let returned = if result_type.is_null() {
            "object"
        } else {
            unsafe { w_type_get_name(result_type) }
        };
        return Err(crate::PyError::type_error(format!(
            "expect types.SimpleNamespace type, but {constructed}() returned '{returned}' object"
        )));
    }
    let source_dict = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(sp),
        "__dict__",
    )?;
    pyre_object::gc_roots::pin_root(source_dict);
    namespace_update_dict(
        result,
        pyre_object::gc_roots::shadow_stack_get(
            sp + 3 + usize::from(kwargs.is_some()),
        ),
        false,
    )?;
    if kwargs.is_some() {
        namespace_update_dict(
            result,
            pyre_object::gc_roots::shadow_stack_get(sp + 1),
            true,
        )?;
    }
    Ok(result)
}

/// `pypy/module/sys/vm.py:217 space.getexecutioncontext()` access for
/// `sys.gettrace`/`settrace`/`getprofile`/`setprofile`.
///
/// Pyre's `crate::call::getexecutioncontext` returns the TLS-cached
/// active context (set on eval-loop entry); see the helper's doc for
/// the staleness gap relative to PyPy's `space.getexecutioncontext()`
/// which always queries the thread state.
fn current_execution_context() -> *mut crate::PyExecutionContext {
    crate::call::getexecutioncontext() as *mut crate::PyExecutionContext
}

fn sys_gettrace_impl(_args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    if ec.is_null() {
        return Ok(w_none());
    }
    let w_trace = unsafe { (*ec).gettrace() };
    Ok(if w_trace.is_null() { w_none() } else { w_trace })
}

fn sys_settrace_impl(args: &[PyObjectRef]) -> crate::PyResult {
    // pypy/module/sys/vm.py:217 `def settrace(space, w_func)` — w_func is
    // a required positional. Calling `sys.settrace()` with no args raises
    // TypeError at the gateway layer in PyPy; reproduce that here.
    let w_func = *args.first().ok_or_else(|| {
        crate::PyError::type_error("settrace() missing 1 required positional argument: 'function'")
    })?;
    let ec = current_execution_context();
    if !ec.is_null() {
        unsafe { (*ec).settrace(w_func) };
    }
    Ok(w_none())
}

fn sys_getprofile_impl(_args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    if ec.is_null() {
        return Ok(w_none());
    }
    let w_profile = unsafe { (*ec).getprofile() };
    Ok(if w_profile.is_null() {
        w_none()
    } else {
        w_profile
    })
}

fn sys_setprofile_impl(args: &[PyObjectRef]) -> crate::PyResult {
    // pypy/module/sys/vm.py:227 `def setprofile(space, w_func)` — w_func
    // is a required positional. Calling `sys.setprofile()` with no args
    // raises TypeError at the gateway layer in PyPy.
    let w_func = *args.first().ok_or_else(|| {
        crate::PyError::type_error(
            "setprofile() missing 1 required positional argument: 'function'",
        )
    })?;
    let ec = current_execution_context();
    if !ec.is_null() {
        // executioncontext.py:317-318 ValueError("Cannot call setllprofile
        // with real None") propagates via setprofile -> setllprofile.
        unsafe { (*ec).setprofile(w_func)? };
    }
    Ok(w_none())
}

fn sys_settraceallthreads_impl(args: &[PyObjectRef]) -> crate::PyResult {
    let w_func = *args.first().ok_or_else(|| {
        crate::PyError::type_error(
            "_settraceallthreads() missing 1 required positional argument: 'function'",
        )
    })?;
    crate::module::thread::set_trace_all_execution_contexts(w_func);
    Ok(w_none())
}

fn sys_setprofileallthreads_impl(args: &[PyObjectRef]) -> crate::PyResult {
    let w_func = *args.first().ok_or_else(|| {
        crate::PyError::type_error(
            "_setprofileallthreads() missing 1 required positional argument: 'function'",
        )
    })?;
    crate::module::thread::set_profile_all_execution_contexts(w_func)?;
    Ok(w_none())
}

fn sys_get_coroutine_origin_tracking_depth(_args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    let depth = if ec.is_null() {
        0
    } else {
        unsafe { (*ec).coroutine_origin_tracking_depth }
    };
    Ok(pyre_object::w_int_new(depth))
}

fn sys_set_coroutine_origin_tracking_depth(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(
        kwargs,
        &["depth"],
        "set_coroutine_origin_tracking_depth",
    )?;
    if positional.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "set_coroutine_origin_tracking_depth() takes exactly one argument ({} given)",
            positional.len(),
        )));
    }
    let kw_depth = crate::builtins::kwarg_get(kwargs, "depth");
    if !positional.is_empty() && kw_depth.is_some() {
        return Err(crate::PyError::type_error(
            "set_coroutine_origin_tracking_depth() got multiple values for argument 'depth'",
        ));
    }
    let w_depth = positional.first().copied().or(kw_depth).ok_or_else(|| {
        crate::PyError::type_error(
            "set_coroutine_origin_tracking_depth() missing required argument 'depth'",
        )
    })?;
    let indexed = crate::baseobjspace::space_index(w_depth)?;
    let depth = crate::baseobjspace::int_w(indexed)?;
    if depth < 0 {
        return Err(crate::PyError::value_error("depth must be >= 0"));
    }
    let ec = current_execution_context();
    if !ec.is_null() {
        unsafe { (*ec).coroutine_origin_tracking_depth = depth };
    }
    Ok(w_none())
}

fn asyncgen_hooks_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        crate::_structseq::make_struct_seq("asyncgen_hooks", &["firstiter", "finalizer"]) as usize
    }) as PyObjectRef
}

fn sys_get_asyncgen_hooks_impl(_args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    let (firstiter, finalizer) = if ec.is_null() {
        (w_none(), w_none())
    } else {
        unsafe {
            (
                if (*ec).w_asyncgen_firstiter_fn.is_null() {
                    w_none()
                } else {
                    (*ec).w_asyncgen_firstiter_fn
                },
                if (*ec).w_asyncgen_finalizer_fn.is_null() {
                    w_none()
                } else {
                    (*ec).w_asyncgen_finalizer_fn
                },
            )
        }
    };
    Ok(crate::_structseq::new_instance(
        asyncgen_hooks_type(),
        vec![firstiter, finalizer],
    ))
}

fn sys_set_asyncgen_hooks_impl(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(
        kwargs,
        &["firstiter", "finalizer"],
        "set_asyncgen_hooks",
    )?;
    if positional.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "set_asyncgen_hooks() takes at most 2 arguments ({} given)",
            positional.len()
        )));
    }
    let kw_firstiter = crate::builtins::kwarg_get(kwargs, "firstiter");
    let kw_finalizer = crate::builtins::kwarg_get(kwargs, "finalizer");
    if !positional.is_empty() && kw_firstiter.is_some() {
        return Err(crate::PyError::type_error(
            "set_asyncgen_hooks() got multiple values for argument 'firstiter'",
        ));
    }
    if positional.len() > 1 && kw_finalizer.is_some() {
        return Err(crate::PyError::type_error(
            "set_asyncgen_hooks() got multiple values for argument 'finalizer'",
        ));
    }
    let firstiter = positional.first().copied().or(kw_firstiter);
    let finalizer = positional.get(1).copied().or(kw_finalizer);
    let ec = current_execution_context();
    if !ec.is_null() {
        unsafe {
            // PyPy vm.py:set_asyncgen_hooks updates and validates finalizer
            // first, then firstiter.  Preserve that observable ordering when
            // the second argument is invalid.
            if let Some(value) = finalizer {
                if is_none(value) {
                    (*ec).w_asyncgen_finalizer_fn = PY_NULL;
                } else if crate::baseobjspace::callable_w(value) {
                    (*ec).w_asyncgen_finalizer_fn = value;
                } else {
                    return Err(crate::PyError::type_error(format!(
                        "callable finalizer expected, got {}",
                        crate::type_methods::arg_type_name(value)
                    )));
                }
            }
            if let Some(value) = firstiter {
                if is_none(value) {
                    (*ec).w_asyncgen_firstiter_fn = PY_NULL;
                } else if crate::baseobjspace::callable_w(value) {
                    (*ec).w_asyncgen_firstiter_fn = value;
                } else {
                    return Err(crate::PyError::type_error(format!(
                        "callable firstiter expected, got {}",
                        crate::type_methods::arg_type_name(value)
                    )));
                }
            }
        }
    }
    Ok(w_none())
}

/// `app.py breakpointhook` — the hook `breakpoint()` calls.
///
/// `$PYTHONBREAKPOINT` names the callable to run: unset or empty means
/// `pdb.set_trace`, `"0"` disables the hook outright, and any other value is
/// imported as a dotted path (a bare name resolves against `builtins`).  An
/// unimportable value warns and returns None rather than propagating, so a
/// stray environment variable cannot break an otherwise working program.
fn sys_breakpointhook(args: &[PyObjectRef]) -> crate::PyResult {
    let hookname = match crate::importing::host::os::var("PYTHONBREAKPOINT") {
        Ok(name) if name == "0" => return Ok(w_none()),
        Ok(name) if !name.is_empty() => name,
        _ => "pdb.set_trace".to_string(),
    };
    let (modname, funcname) = match hookname.rsplit_once('.') {
        Some((modname, funcname)) => (modname, funcname),
        None => ("builtins", hookname.as_str()),
    };

    // `dunder_import` returns the root package, so reach the leaf through
    // `sys.modules` before pulling the attribute off it.
    let hook = crate::importing::dunder_import(
        modname,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        0,
        std::ptr::null(),
    )
    .ok()
    .and_then(|_| crate::importing::get_sys_module(modname))
    .and_then(|module| crate::baseobjspace::getattr_str(module, funcname).ok());
    let Some(hook) = hook else {
        crate::warn::warn_category(
            &format!("Ignoring unimportable $PYTHONBREAKPOINT: \"{hookname}\""),
            "RuntimeWarning",
            1,
        )?;
        return Ok(w_none());
    };
    // The hook "must accept whatever arguments are passed".
    crate::builtins::call_forwarding_args(hook, args)
}

fn sys_unraisablehook(args: &[PyObjectRef]) -> crate::PyResult {
    let Some(&w_hookargs) = args.first() else {
        return Err(crate::PyError::type_error(
            "unraisablehook() missing 1 required positional argument",
        ));
    };
    let w_type = crate::baseobjspace::getattr_str(w_hookargs, "exc_type")?;
    let w_value = crate::baseobjspace::getattr_str(w_hookargs, "exc_value")?;
    let w_tb = crate::baseobjspace::getattr_str(w_hookargs, "exc_traceback")?;
    let w_err_msg = crate::baseobjspace::getattr_str(w_hookargs, "err_msg")?;
    let err_msg = if unsafe { pyre_object::is_none(w_err_msg) } {
        String::new()
    } else if unsafe { pyre_object::is_str(w_err_msg) } {
        unsafe { pyre_object::w_str_get_value(w_err_msg) }.to_string()
    } else {
        unsafe { crate::display::py_str(w_err_msg)? }
    };
    let w_object = crate::baseobjspace::getattr_str(w_hookargs, "object")?;
    crate::PyError::write_unraisable_default(
        w_none(),
        w_type,
        w_value,
        w_tb,
        &err_msg,
        w_object,
        "",
    );
    Ok(w_none())
}

/// pypy/module/sys/vm.py `exc_info_direct` — return the active exception
/// as a `(type, value, traceback)` tuple.
///
/// Used by both the regular `sys.exc_info` builtin and the JIT direct path
/// in `function.funccall_valuestack` (function.py:146-150). Splitting it
/// out lets the JIT bypass invoke the same logic without going through the
/// builtin call dispatch.
pub fn exc_info_direct() -> PyObjectRef {
    let exc = crate::eval::get_sys_exception();
    unsafe {
        if exc.is_null() || pyre_object::is_none(exc) || !pyre_object::is_exception(exc) {
            w_tuple_new(vec![w_none(), w_none(), w_none()])
        } else {
            // `pypy/module/sys/vm.py exc_info_direct` returns
            // `(type, value, traceback)` where `type` is
            // `space.exception_getclass(value)` — the specific
            // subclass W_TypeObject (e.g. `ZeroDivisionError`), not
            // the generic `Exception` stub set in
            // `w_exception_new`.  Pyre routes the per-`ExcKind`
            // lookup through `typedef::r#type` (`typedef.rs:186-197`)
            // which `exception_getclass` delegates to, so go through
            // that instead of dereferencing the raw `w_class` slot
            // (which still points at the constructor-time
            // `EXCEPTION_TYPE` stub).
            let exc_type = crate::baseobjspace::exception_getclass(exc);
            let exc_type = if exc_type.is_null() {
                w_none()
            } else {
                exc_type
            };
            // The third tuple slot mirrors
            // `vm.py:147-153 exc_info_with_tb`'s
            // `operror.get_w_traceback(space)`, i.e. the slot read plus
            // the escape mark it wraps.  Pyre stores the chain on the
            // typed `w_traceback` slot of `W_BaseException`; surface it
            // directly here.
            let tb = pyre_object::interp_exceptions::w_exception_get_traceback(exc);
            crate::pytraceback::mark_traceback_escaped(tb);
            let w_tb = if tb.is_null() { w_none() } else { tb };
            w_tuple_new(vec![exc_type, exc, w_tb])
        }
    }
}

pub fn register_module(ns: pyre_object::PyObjectRef) {
    module_ns_store(ns, "maxsize", w_int_new(i64::MAX));
    module_ns_store(ns, "maxunicode", w_int_new(0x10FFFF));
    module_ns_store(
        ns,
        "orig_argv",
        w_list_new(
            crate::importing::sys_orig_argv()
                .iter()
                .map(|arg| w_str_new(arg))
                .collect(),
        ),
    );
    // pypy/interpreter/app_main.py:785-786:
    //   sys._xoptions = dict(x.split('=', 1) if '=' in x else (x, True)
    //                        for x in options['_xoptions'])
    let xoptions = w_dict_new();
    for option in crate::importing::xoptions() {
        let (name, value) = match option.split_once('=') {
            Some((name, value)) => (name, w_str_new(value)),
            None => (option.as_str(), w_bool_from(true)),
        };
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(xoptions, name, value);
        }
    }
    module_ns_store(ns, "_xoptions", xoptions);
    // Format matches `platform._sys_version`'s CPython parser:
    // `version (buildinfo) [compiler]`.
    module_ns_store(ns, "version", w_str_new("3.14.6 (pyre 0.0.1) [Rust]"));
    module_ns_store(
        ns,
        "platform",
        w_str_new(if cfg!(target_os = "macos") {
            "darwin"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "windows") {
            "win32"
        } else {
            "unknown"
        }),
    );
    // sys.winver — the "major.minor" tag Windows uses for the per-user site
    // directory and the PythonCore registry keys. site.getusersitepackages
    // reads it to build USER_SITE.
    #[cfg(windows)]
    module_ns_store(ns, "winver", w_str_new("3.14"));
    module_ns_store(
        ns,
        "byteorder",
        w_str_new(if cfg!(target_endian = "little") {
            "little"
        } else {
            "big"
        }),
    );
    // sys.version_info — structseq(major, minor, micro, releaselevel,
    // serial); a tuple subclass so `>= (3, 14)` / `[0]` and `.major` both work.
    {
        let version_info_type = crate::_structseq::make_struct_seq(
            "sys.version_info",
            &["major", "minor", "micro", "releaselevel", "serial"],
        );
        let vi = crate::_structseq::new_instance(
            version_info_type,
            vec![
                w_int_new(3),
                w_int_new(14),
                w_int_new(6),
                w_str_new("final"),
                w_int_new(0),
            ],
        );
        module_ns_store(ns, "version_info", vi);
    }
    // sys.modules — live dict synced with the import cache.
    let modules_dict = w_dict_new();
    crate::importing::set_sys_modules_dict(modules_dict);
    module_ns_store(ns, "modules", modules_dict);
    // sys.path — flush the native search-path seed into the authoritative list
    // the instant `sys` exists; from here on the Python list is the source of
    // truth and `add_sys_path` mutates it in place.
    module_ns_store(ns, "path", crate::importing::create_sys_path_list());
    // sys.stdout/stderr/stdin — `_io.TextIOWrapper`-typed file-like objects.
    // Real CPython wires these through io.TextIOWrapper around the std fds;
    // pyre exposes objects of the same type with the minimum surface so
    // anything that writes status (unittest, traceback, warnings) keeps
    // working.  `sys.__stdout__ is sys.stdout` (a single object each).
    let stdout = make_std_stream("<stdout>", 1);
    let stderr = make_std_stream("<stderr>", 2);
    let stdin = make_std_stream("<stdin>", 0);
    module_ns_store(ns, "stdout", stdout);
    module_ns_store(ns, "stderr", stderr);
    module_ns_store(ns, "stdin", stdin);
    module_ns_store(ns, "__stdout__", stdout);
    module_ns_store(ns, "__stderr__", stderr);
    module_ns_store(ns, "__stdin__", stdin);
    // `pypy/module/sys/vm.py:30 _getframe` walks the
    // `space.getexecutioncontext().gettopframe_nohidden()` chain,
    // following `f_back` `depth` times.  PyPy returns the frame
    // object directly so `frame.f_globals is module.__dict__` /
    // `frame.f_globals is globals()` (callee's scope) holds.  Pyre
    // mirrors the depth walk through `CURRENT_FRAME` + `f_back`,
    // populating the stub frame's attributes from the resolved
    // PyFrame. `f_globals` / `f_locals` use the frame's canonical dict so the
    // `is module.__dict__` invariant survives sys._getframe access.
    module_ns_store(
        ns,
        "_getframe",
        crate::make_builtin_function("_getframe", |args| {
            // `pypy/module/sys/vm.py:28-39 _getframe`:
            //   @unwrap_spec(depth=int) def _getframe(space, depth=0)
            // `unwrap_spec` enforces a single optional int argument, so
            // any extra positional arg must surface as TypeError before
            // the depth walk runs.
            if args.len() > 1 {
                return Err(crate::PyError::type_error(format!(
                    "_getframe expected at most 1 argument, got {}",
                    args.len()
                )));
            }
            let depth_signed = if args.is_empty() {
                0i64
            } else if unsafe { pyre_object::is_int(args[0]) } {
                unsafe { pyre_object::w_int_get_value(args[0]) }
            } else {
                return Err(crate::PyError::type_error(
                    "_getframe(): argument must be an int",
                ));
            };
            // `vm.py:37-38 if depth < 0: raise ... "frame index must not
            // be negative"` — the message string differs from the
            // exhausted-stack case below.
            if depth_signed < 0 {
                return Err(crate::PyError::value_error(
                    "frame index must not be negative",
                ));
            }
            // `vm.py:44-54 getframe`: start from
            // `ec.gettopframe_nohidden()` and walk
            // `ec.getnextframe_nohidden(f)` `depth` times, so
            // `hidden_applevel` gateway / bridge frames are skipped
            // (matching `f_back`).  The `f is None` guard runs at the
            // *start* of every iteration including the first, so a
            // missing top frame raises rather than fabricating a stub.
            let ec = current_execution_context();
            let mut current = if ec.is_null() {
                std::ptr::null_mut()
            } else {
                // Force the frame `topframeref` names BEFORE deciding which
                // frame to hand out.  A JIT-inlined callee has no frame of its
                // own until that force materialises one, so a walk started on
                // the unforced chain would skip straight to the caller and
                // then read `f_back` off a frame that never existed.
                // `gettopframe` is the forcing accessor; the `_nohidden` walk
                // is force-free by design (see `force_frame`).
                unsafe {
                    (*ec).gettopframe();
                    (*ec).gettopframe_nohidden()
                }
            };
            let mut remaining = depth_signed as usize;
            loop {
                if current.is_null() {
                    return Err(crate::PyError::value_error("call stack is not deep enough"));
                }
                if remaining == 0 {
                    break;
                }
                remaining -= 1;
                current = crate::executioncontext::ExecutionContext::getnextframe_nohidden(current);
            }
            // `pyframe.py:767 f_back = GetSetProperty(PyFrame.fget_f_back)`.
            // Return the live `PyFrame` itself as the user-visible `frame`
            // object (`FRAME_TYPE` typedef); `f_back` chains lazily through
            // the getset.  Mark it escaped so the JIT keeps the frame
            // materialised for the exposed reference (`pyframe.py`
            // `mark_as_escaped`), and force it now: app code is about to read
            // `f_lineno` / `f_locals` off a frame whose virtualizable fields
            // the JIT may still be holding.
            unsafe { (*current).mark_as_escaped() };
            crate::executioncontext::force_frame(current);
            Ok(current as pyre_object::PyObjectRef)
        }),
    );
    // sys.exc_info() → (type, value, traceback)
    //
    // Tuple construction is shared with `exc_info_direct` (the JIT fast-path
    // entry registered below), so the regular call path and the JIT bypass
    // observe the same value.
    let exc_info_fn = make_builtin_function_with_arity("exc_info", |_| Ok(exc_info_direct()), 0);
    module_ns_store(ns, "exc_info", exc_info_fn);
    // baseobjspace.py: register `space._code_of_sys_exc_info` so
    // `function.funccall_valuestack` can take the JIT direct path
    // (function.py:146-150). The builtin code pointer lives on the
    // `BuiltinCode` object backing `exc_info_fn`; `getcode` returns it.
    let exc_info_code = unsafe { crate::getcode(exc_info_fn) };
    crate::function::register_sys_exc_info_path(exc_info_code, exc_info_direct);
    // sys.flags — pypy/module/sys/app.py:99-119 `class sysflags` with
    // `__metaclass__ = structseqtype`: an immutable tuple subclass whose
    // first `n_sequence_fields` entries are also indexable, so
    // `sys.flags[3] is sys.flags.optimize` and `isinstance(sys.flags,
    // tuple)` both hold.  `gil`, `thread_inherit_context` and
    // `context_aware_warnings` sit past the sequence as named-only extras,
    // which is what makes `n_fields` (21) exceed `len()` (18).
    {
        let flags_type = crate::_structseq::make_struct_seq_with_extra(
            "sys.flags",
            &[
                "debug",
                "inspect",
                "interactive",
                "optimize",
                "dont_write_bytecode",
                "no_user_site",
                "no_site",
                "ignore_environment",
                "verbose",
                "bytes_warning",
                "quiet",
                "hash_randomization",
                "isolated",
                "dev_mode",
                "utf8_mode",
                "warn_default_encoding",
                "safe_path",
                "int_max_str_digits",
            ],
            &["gil", "thread_inherit_context", "context_aware_warnings"],
        );
        let flags = crate::_structseq::new_instance_with_extra(
            flags_type,
            vec![
                w_int_new(0), // debug
                // `-i` sets both.
                w_int_new(i64::from(crate::importing::inspect_flag())),
                w_int_new(i64::from(crate::importing::inspect_flag())),
                w_int_new(crate::importing::optimize_level()),
                w_int_new(i64::from(crate::importing::dont_write_bytecode_flag())),
                w_int_new(i64::from(crate::importing::no_user_site_flag())),
                // `-S` (skip `import site`) is recorded by the launcher.
                w_int_new(i64::from(crate::importing::no_site_flag())),
                w_int_new(i64::from(crate::importing::ignore_environment_flag())),
                w_int_new(0), // verbose
                w_int_new(crate::importing::bytes_warning_flag()),
                w_int_new(i64::from(crate::importing::quiet_flag())),
                w_int_new(0), // hash_randomization
                w_int_new(i64::from(crate::importing::isolated_flag())),
                w_bool_from(crate::importing::dev_mode_flag()),
                w_int_new(crate::importing::utf8_mode_flag()),
                w_int_new(0), // warn_default_encoding
                w_bool_from(crate::importing::safe_path_flag()),
                w_int_new(crate::module::sys::state::int_max_str_digits() as i64),
            ],
            vec![
                // The interpreter holds a GIL, so `-X gil=0` is not available.
                ("gil", w_int_new(1)),
                ("thread_inherit_context", w_int_new(0)),
                ("context_aware_warnings", w_int_new(0)),
            ],
        );
        module_ns_store(ns, "flags", flags);
    }
    // sys.getdefaultencoding
    module_ns_store(
        ns,
        "getdefaultencoding",
        make_builtin_function_with_arity("getdefaultencoding", |_| Ok(w_str_new("utf-8")), 0),
    );
    // sys.getrecursionlimit / setrecursionlimit — pypy/module/sys/vm.py:45.
    // The runtime stack budget lives in `crate::stack_check`; both
    // helpers route through it so the interpreter, JIT prologue probe,
    // and blackhole resume see a consistent recursion budget.
    module_ns_store(
        ns,
        "getrecursionlimit",
        make_builtin_function_with_arity(
            "getrecursionlimit",
            |args| {
                // pypy/module/sys/vm.py:72 — no arguments.
                if !args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "getrecursionlimit() takes no arguments",
                    ));
                }
                Ok(w_int_new(crate::stack_check::get_recursion_limit() as i64))
            },
            0,
        ),
    );
    module_ns_store(
        ns,
        "setrecursionlimit",
        make_builtin_function_with_arity(
            "setrecursionlimit",
            |args| {
                // pypy/module/sys/vm.py:63 `@unwrap_spec(new_limit="c_int")`
                // — exactly one positional argument, coerced through
                // baseobjspace.c_int_w (gateway_int_w + 32-bit range
                // check). `c_int_w` accepts int subclasses and any object
                // implementing `__int__`, rejects floats, and surfaces
                // out-of-range values as OverflowError.
                if args.len() != 1 {
                    return Err(crate::PyError::type_error(
                        "setrecursionlimit() takes exactly one argument",
                    ));
                }
                let new_limit = crate::baseobjspace::c_int_w(args[0])?;
                crate::stack_check::set_recursion_limit(new_limit)?;
                Ok(w_none())
            },
            1,
        ),
    );
    module_ns_store(
        ns,
        "getswitchinterval",
        make_builtin_function_with_arity(
            "getswitchinterval",
            |args| {
                if !args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "getswitchinterval() takes no arguments",
                    ));
                }
                let ec = current_execution_context();
                let interval = if ec.is_null() {
                    0.005
                } else {
                    unsafe { (*ec).actionflag.getcheckinterval() as f64 / 2_000_000.0 }
                };
                Ok(w_float_new(interval))
            },
            0,
        ),
    );
    module_ns_store(
        ns,
        "setswitchinterval",
        make_builtin_function_with_arity(
            "setswitchinterval",
            |args| {
                if args.len() != 1 {
                    return Err(crate::PyError::type_error(
                        "setswitchinterval() takes exactly one argument",
                    ));
                }
                let interval = crate::baseobjspace::float_w(args[0])?;
                if interval <= 0.0 {
                    return Err(crate::PyError::value_error(
                        "switch interval must be strictly positive",
                    ));
                }
                let ec = current_execution_context();
                if !ec.is_null() {
                    unsafe {
                        (*(ec as *mut crate::PyExecutionContext))
                            .actionflag
                            .setcheckinterval((interval * 2_000_000.0) as usize)
                    };
                }
                Ok(w_none())
            },
            1,
        ),
    );
    module_ns_store(
        ns,
        "_current_frames",
        make_builtin_function_with_arity(
            "_current_frames",
            |args| {
                if !args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "_current_frames() takes no arguments",
                    ));
                }
                Ok(crate::module::thread::current_frames())
            },
            0,
        ),
    );
    // PyPy: pypy/module/sys/state.py:get_int_max_str_digits and
    // set_int_max_str_digits. The limit is object-space state, shared by
    // every caller rather than thread-local state.
    module_ns_store(
        ns,
        "get_int_max_str_digits",
        make_builtin_function_with_arity(
            "get_int_max_str_digits",
            |args| {
                // The fixed arity above is only a fast-dispatch hint; the
                // direct path still delivers whatever the caller passed.
                if !args.is_empty() {
                    return Err(crate::PyError::type_error(format!(
                        "get_int_max_str_digits() takes 0 positional arguments but {} {} given",
                        args.len(),
                        if args.len() == 1 { "was" } else { "were" },
                    )));
                }
                Ok(w_int_new(
                    crate::module::sys::state::int_max_str_digits() as i64
                ))
            },
            0,
        ),
    );
    module_ns_store(
        ns,
        "set_int_max_str_digits",
        make_builtin_function_with_arity_and_maybe_sig(
            "set_int_max_str_digits",
            |args| {
                if args.len() != 1 {
                    let message = if args.is_empty() {
                        "set_int_max_str_digits() missing 1 required positional argument: \
                         'maxdigits'"
                            .to_string()
                    } else {
                        format!(
                            "set_int_max_str_digits() takes 1 positional argument but {} were given",
                            args.len(),
                        )
                    };
                    return Err(crate::PyError::type_error(message));
                }
                let maxdigits = crate::baseobjspace::c_int_w(args[0])?;
                crate::module::sys::state::set_int_max_str_digits(maxdigits)?;
                Ok(w_none())
            },
            1,
            Some(crate::gateway::Signature::new(
                vec!["maxdigits"],
                None,
                None,
                0,
                0,
            )),
        ),
    );
    // sys.intern
    module_ns_store(
        ns,
        "intern",
        make_builtin_function_with_arity(
            "intern",
            |args| {
                let s = args[0];
                if !unsafe { pyre_object::is_exact_type(s, &pyre_object::STR_TYPE) } {
                    return Err(crate::PyError::type_error(format!(
                        "can't intern {}",
                        crate::type_methods::arg_type_name(s)
                    )));
                }
                Ok(unsafe { pyre_object::unicodeobject::intern_exact_str(s) })
            },
            1,
        ),
    );
    // sys.implementation — structseq-like namespace with name, version, ...
    {
        let impl_obj = w_instance_new(simple_namespace_type());
        crate::baseobjspace::setdictvalue_native(impl_obj, "name", w_str_new("pyre"));
        crate::baseobjspace::setdictvalue_native(
            impl_obj,
            "version",
            w_tuple_new(vec![
                w_int_new(3),
                w_int_new(14),
                w_int_new(6),
                w_str_new("final"),
                w_int_new(0),
            ]),
        );
        crate::baseobjspace::setdictvalue_native(impl_obj, "hexversion", w_int_new(0x030e06f0));
        crate::baseobjspace::setdictvalue_native(impl_obj, "cache_tag", w_str_new("pyre-314"));
        crate::baseobjspace::setdictvalue_native(impl_obj, "_multiarch", w_str_new(""));
        module_ns_store(ns, "implementation", impl_obj);
    }
    // sys.hash_info — structseq with width/modulus/... fields.
    // PyPy: pypy/module/sys/system.py hash_info.
    {
        let hash_info = make_sys_namespace_instance();
        crate::baseobjspace::setdictvalue_native(hash_info, "width", w_int_new(64));
        crate::baseobjspace::setdictvalue_native(hash_info, "modulus", w_int_new((1i64 << 61) - 1));
        crate::baseobjspace::setdictvalue_native(hash_info, "inf", w_int_new(314159));
        crate::baseobjspace::setdictvalue_native(hash_info, "nan", w_int_new(0));
        crate::baseobjspace::setdictvalue_native(hash_info, "imag", w_int_new(1000003));
        crate::baseobjspace::setdictvalue_native(hash_info, "algorithm", w_str_new("siphash13"));
        crate::baseobjspace::setdictvalue_native(hash_info, "hash_bits", w_int_new(64));
        crate::baseobjspace::setdictvalue_native(hash_info, "seed_bits", w_int_new(128));
        crate::baseobjspace::setdictvalue_native(hash_info, "cutoff", w_int_new(0));
        module_ns_store(ns, "hash_info", hash_info);
    }
    // sys.float_info — structseq with IEEE 754 double metadata.
    // PyPy: pypy/module/sys/system.py float_info.
    {
        let fi = make_sys_namespace_instance();
        crate::baseobjspace::setdictvalue_native(fi, "max", w_float_new(f64::MAX));
        crate::baseobjspace::setdictvalue_native(fi, "max_exp", w_int_new(1024));
        crate::baseobjspace::setdictvalue_native(fi, "max_10_exp", w_int_new(308));
        crate::baseobjspace::setdictvalue_native(fi, "min", w_float_new(f64::MIN_POSITIVE));
        crate::baseobjspace::setdictvalue_native(fi, "min_exp", w_int_new(-1021));
        crate::baseobjspace::setdictvalue_native(fi, "min_10_exp", w_int_new(-307));
        crate::baseobjspace::setdictvalue_native(fi, "dig", w_int_new(15));
        crate::baseobjspace::setdictvalue_native(fi, "mant_dig", w_int_new(53));
        crate::baseobjspace::setdictvalue_native(fi, "epsilon", w_float_new(f64::EPSILON));
        crate::baseobjspace::setdictvalue_native(fi, "radix", w_int_new(2));
        crate::baseobjspace::setdictvalue_native(fi, "rounds", w_int_new(1));
        module_ns_store(ns, "float_info", fi);
    }
    // sysmodule.c — `sys.float_repr_style` is "short" wherever float repr
    // uses David Gay's shortest-round-trip algorithm (always, here).
    module_ns_store(ns, "float_repr_style", w_str_new("short"));
    // sys.thread_info — structseq(name, lock, version).
    {
        let ti = make_sys_namespace_instance();
        crate::baseobjspace::setdictvalue_native(ti, "name", w_str_new("pthread"));
        crate::baseobjspace::setdictvalue_native(ti, "lock", w_str_new("semaphore"));
        crate::baseobjspace::setdictvalue_native(ti, "version", w_none());
        module_ns_store(ns, "thread_info", ti);
    }
    // sys.int_info — structseq with int implementation details.
    {
        let ii = make_sys_namespace_instance();
        crate::baseobjspace::setdictvalue_native(ii, "bits_per_digit", w_int_new(30));
        crate::baseobjspace::setdictvalue_native(ii, "sizeof_digit", w_int_new(4));
        crate::baseobjspace::setdictvalue_native(ii, "default_max_str_digits", w_int_new(4300));
        crate::baseobjspace::setdictvalue_native(ii, "str_digits_check_threshold", w_int_new(640));
        module_ns_store(ns, "int_info", ii);
    }
    module_ns_store(ns, "hexversion", w_int_new(0x030e06f0));
    // sys.executable — absolute path to the running interpreter so that
    // subprocess spawns via `sys.executable` resolve.
    #[cfg(not(feature = "sandbox"))]
    let executable = std::env::current_exe()
        .ok()
        .and_then(|p| p.to_str().map(str::to_owned))
        .unwrap_or_else(|| "pyre".to_owned());
    // Under sandbox a fixed placeholder: current_exe() leaks the host binary
    // path (and username), and subprocess spawning is unavailable anyway.
    #[cfg(feature = "sandbox")]
    let executable = "/bin/pyre".to_owned();
    module_ns_store(ns, "executable", w_str_new(&executable));
    // sys.prefix / exec_prefix
    module_ns_store(ns, "prefix", w_str_new(""));
    module_ns_store(ns, "exec_prefix", w_str_new(""));
    module_ns_store(ns, "base_prefix", w_str_new(""));
    module_ns_store(ns, "base_exec_prefix", w_str_new(""));
    // FrozenImporter uses the resolved stdlib root to reconstruct source
    // filenames for frozen stdlib modules.
    #[cfg(feature = "host_env")]
    let stdlib_dir = crate::importing::detect_stdlib_path()
        .and_then(|path| path.to_str().map(w_str_new))
        .unwrap_or_else(w_none);
    #[cfg(not(feature = "host_env"))]
    let stdlib_dir = w_none();
    module_ns_store(ns, "_stdlib_dir", stdlib_dir);
    // sys._framework — macOS framework name (empty string on non-framework builds)
    module_ns_store(ns, "_framework", w_str_new(""));
    // sys._jit — namespace with is_enabled/is_available methods.
    // Python 3.14+ introduced sys._jit for CPython tier-2 JIT support checks.
    {
        let jit = make_sys_namespace_instance();
        crate::baseobjspace::setdictvalue_native(
            jit,
            "is_enabled",
            make_builtin_function_with_arity("is_enabled", |_| Ok(w_bool_from(false)), 0),
        );
        crate::baseobjspace::setdictvalue_native(
            jit,
            "is_available",
            make_builtin_function_with_arity("is_available", |_| Ok(w_bool_from(false)), 0),
        );
        module_ns_store(ns, "_jit", jit);
    }
    // sys.monitoring — PEP 669 low-impact monitoring API. The runtime hooks
    // are stubbed (no events ever fire), but the namespace, tool-id
    // constants, sentinels, and `events` flags are present so importing
    // modules (bdb/pdb/cProfile/coverage tooling) succeed and can build
    // their tracer objects.
    {
        let mon = make_sys_namespace_instance();
        // Tool-id constants (Python/instrumentation.c).
        for (name, id) in [
            ("DEBUGGER_ID", 0),
            ("COVERAGE_ID", 1),
            ("PROFILER_ID", 2),
            ("OPTIMIZER_ID", 5),
        ] {
            crate::baseobjspace::setdictvalue_native(mon, name, w_int_new(id));
        }
        // DISABLE / MISSING sentinels — distinct singleton objects compared
        // by identity (`callback() == DISABLE`, `assertIs(x, MISSING)`).
        crate::baseobjspace::setdictvalue_native(mon, "DISABLE", make_sys_namespace_instance());
        crate::baseobjspace::setdictvalue_native(mon, "MISSING", make_sys_namespace_instance());
        // events namespace — `1 << event_id` flags that OR together.
        {
            let events = make_sys_namespace_instance();
            crate::baseobjspace::setdictvalue_native(events, "NO_EVENTS", w_int_new(0));
            for (i, name) in [
                "PY_START",
                "PY_RESUME",
                "PY_RETURN",
                "PY_YIELD",
                "CALL",
                "LINE",
                "INSTRUCTION",
                "JUMP",
                "BRANCH_LEFT",
                "BRANCH_RIGHT",
                "STOP_ITERATION",
                "RAISE",
                "EXCEPTION_HANDLED",
                "PY_UNWIND",
                "PY_THROW",
                "RERAISE",
                "C_RETURN",
                "C_RAISE",
            ]
            .iter()
            .enumerate()
            {
                crate::baseobjspace::setdictvalue_native(events, name, w_int_new(1i64 << i));
            }
            // BRANCH retained as an alias of BRANCH_LEFT for callers predating
            // the 3.14 left/right split.
            crate::baseobjspace::setdictvalue_native(events, "BRANCH", w_int_new(1i64 << 8));
            crate::baseobjspace::setdictvalue_native(mon, "events", events);
        }
        // Runtime hooks — no-op stubs returning sensible defaults.
        let store_fn = |obj, name: &'static str, f: crate::gateway::BuiltinCodeFn, arity: u16| {
            crate::baseobjspace::setdictvalue_native(
                obj,
                name,
                make_builtin_function_with_arity(name, f, arity),
            );
        };
        store_fn(mon, "use_tool_id", |_| Ok(w_none()), 2);
        store_fn(mon, "free_tool_id", |_| Ok(w_none()), 1);
        store_fn(mon, "clear_tool_id", |_| Ok(w_none()), 1);
        store_fn(mon, "get_tool", |_| Ok(w_none()), 1);
        store_fn(mon, "register_callback", |_| Ok(w_none()), 3);
        store_fn(mon, "set_events", |_| Ok(w_none()), 2);
        store_fn(mon, "get_events", |_| Ok(w_int_new(0)), 1);
        store_fn(mon, "set_local_events", |_| Ok(w_none()), 3);
        store_fn(mon, "get_local_events", |_| Ok(w_int_new(0)), 2);
        store_fn(mon, "restart_events", |_| Ok(w_none()), 0);
        module_ns_store(ns, "monitoring", mon);
    }
    // sys.platlibdir — typically "lib" on POSIX; used by sysconfig to
    // construct install paths.
    module_ns_store(ns, "platlibdir", w_str_new("lib"));
    // `sys/app.py:114-126 exit(exitcode=None)` — raise SystemExit(exitcode),
    // de-tupelizing a tuple argument so `exit((a, b))` becomes
    // `SystemExit(a, b)` (the extra de-tupelizing normalize_exception does
    // for `raise SystemExit, exitcode`).  A bare `exit()` defaults exitcode
    // to None, so the instance carries `code = None` / `args = (None,)`.
    // Interpreting the code (None → 0, int() coercion,
    // print-non-integral-and-exit-1) is the launcher's job
    // (`app_main.py:114-129 handle_sys_exit`).
    module_ns_store(
        ns,
        "exit",
        crate::make_builtin_function("exit", |args| {
            // `exit(exitcode=None)` — resolve the single optional argument
            // like the app-level signature: strip the `__pyre_kw__` trailer,
            // reject unknown keywords, reproduce the normal function-call
            // arity diagnostics, and reject a positional/`exitcode=`
            // duplicate.
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            crate::builtins::kwarg_reject_unknown(kwargs, &["exitcode"], "exit")?;
            if positional.len() > 1 {
                return Err(crate::PyError::type_error(format!(
                    "exit() takes from 0 to 1 positional arguments but {} were given",
                    positional.len()
                )));
            }
            let kw_exitcode = crate::builtins::kwarg_get(kwargs, "exitcode");
            if !positional.is_empty() && kw_exitcode.is_some() {
                return Err(crate::PyError::type_error(
                    "exit() got multiple values for argument 'exitcode'",
                ));
            }
            let exitcode = positional
                .first()
                .copied()
                .or(kw_exitcode)
                .unwrap_or_else(w_none);
            let cls = crate::builtins::lookup_exc_class("SystemExit")
                .ok_or_else(|| crate::PyError::runtime_error("SystemExit class missing"))?;
            let ctor_args = if unsafe { is_tuple(exitcode) } {
                unsafe { w_tuple_items_copy_as_vec(exitcode) }
            } else {
                vec![exitcode]
            };
            let exc = crate::call::call_function_impl_result(cls, &ctor_args)?;
            Err(unsafe { crate::PyError::from_exc_object(exc) })
        }),
    );
    // sys.abiflags
    module_ns_store(ns, "abiflags", w_str_new(""));
    // sys.argv — pick up pending argv from set_sys_argv if available.
    let pending = crate::importing::take_pending_sys_argv();
    let argv = if pending.is_null() {
        w_list_new(vec![])
    } else {
        pending
    };
    module_ns_store(ns, "argv", argv);
    // sys.warnoptions
    module_ns_store(
        ns,
        "warnoptions",
        w_list_new(
            crate::importing::warnoptions()
                .iter()
                .map(|option| w_str_new(option))
                .collect(),
        ),
    );
    // sys.builtin_module_names — tuple of names of modules compiled into
    // the interpreter. PyPy: pypy/module/sys/state.py get_builtin_module_names,
    // which reads the same registry `import` resolves against, so the
    // advertised set cannot drift from what is actually importable on a build.
    let builtin_names = crate::importing::builtin_module_names();
    module_ns_store(
        ns,
        "builtin_module_names",
        w_tuple_new(builtin_names.into_iter().map(w_str_new).collect()),
    );
    // sys.stdlib_module_names — frozenset of every top-level standard-library
    // module name (`Python/stdlib_module_names.h`).  Read by
    // `traceback.TracebackException` for "did you forget to import" hints and
    // by the module shadowing check for the stronger stdlib rename hint.  The
    // list is the full set regardless of platform, as upstream ships it.
    const STDLIB_MODULE_NAMES: &[&str] = &[
        "__future__",
        "_abc",
        "_aix_support",
        "_android_support",
        "_apple_support",
        "_ast",
        "_ast_unparse",
        "_asyncio",
        "_bisect",
        "_blake2",
        "_bz2",
        "_codecs",
        "_codecs_cn",
        "_codecs_hk",
        "_codecs_iso2022",
        "_codecs_jp",
        "_codecs_kr",
        "_codecs_tw",
        "_collections",
        "_collections_abc",
        "_colorize",
        "_compat_pickle",
        "_contextvars",
        "_csv",
        "_ctypes",
        "_curses",
        "_curses_panel",
        "_datetime",
        "_dbm",
        "_decimal",
        "_elementtree",
        "_frozen_importlib",
        "_frozen_importlib_external",
        "_functools",
        "_gdbm",
        "_hashlib",
        "_heapq",
        "_hmac",
        "_imp",
        "_interpchannels",
        "_interpqueues",
        "_interpreters",
        "_io",
        "_ios_support",
        "_json",
        "_locale",
        "_lsprof",
        "_lzma",
        "_markupbase",
        "_md5",
        "_multibytecodec",
        "_multiprocessing",
        "_opcode",
        "_opcode_metadata",
        "_operator",
        "_osx_support",
        "_overlapped",
        "_pickle",
        "_posixshmem",
        "_posixsubprocess",
        "_py_abc",
        "_py_warnings",
        "_pydatetime",
        "_pydecimal",
        "_pyio",
        "_pylong",
        "_pyrepl",
        "_queue",
        "_random",
        "_remote_debugging",
        "_scproxy",
        "_sha1",
        "_sha2",
        "_sha3",
        "_signal",
        "_sitebuiltins",
        "_socket",
        "_sqlite3",
        "_sre",
        "_ssl",
        "_stat",
        "_statistics",
        "_string",
        "_strptime",
        "_struct",
        "_suggestions",
        "_symtable",
        "_sysconfig",
        "_thread",
        "_threading_local",
        "_tkinter",
        "_tokenize",
        "_tracemalloc",
        "_types",
        "_typing",
        "_uuid",
        "_warnings",
        "_weakref",
        "_weakrefset",
        "_winapi",
        "_wmi",
        "_zoneinfo",
        "_zstd",
        "abc",
        "annotationlib",
        "antigravity",
        "argparse",
        "array",
        "ast",
        "asyncio",
        "atexit",
        "base64",
        "bdb",
        "binascii",
        "bisect",
        "builtins",
        "bz2",
        "cProfile",
        "calendar",
        "cmath",
        "cmd",
        "code",
        "codecs",
        "codeop",
        "collections",
        "colorsys",
        "compileall",
        "compression",
        "concurrent",
        "configparser",
        "contextlib",
        "contextvars",
        "copy",
        "copyreg",
        "csv",
        "ctypes",
        "curses",
        "dataclasses",
        "datetime",
        "dbm",
        "decimal",
        "difflib",
        "dis",
        "doctest",
        "email",
        "encodings",
        "ensurepip",
        "enum",
        "errno",
        "faulthandler",
        "fcntl",
        "filecmp",
        "fileinput",
        "fnmatch",
        "fractions",
        "ftplib",
        "functools",
        "gc",
        "genericpath",
        "getopt",
        "getpass",
        "gettext",
        "glob",
        "graphlib",
        "grp",
        "gzip",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "http",
        "idlelib",
        "imaplib",
        "importlib",
        "inspect",
        "io",
        "ipaddress",
        "itertools",
        "json",
        "keyword",
        "linecache",
        "locale",
        "logging",
        "lzma",
        "mailbox",
        "marshal",
        "math",
        "mimetypes",
        "mmap",
        "modulefinder",
        "msvcrt",
        "multiprocessing",
        "netrc",
        "nt",
        "ntpath",
        "nturl2path",
        "numbers",
        "opcode",
        "operator",
        "optparse",
        "os",
        "pathlib",
        "pdb",
        "pickle",
        "pickletools",
        "pkgutil",
        "platform",
        "plistlib",
        "poplib",
        "posix",
        "posixpath",
        "pprint",
        "profile",
        "pstats",
        "pty",
        "pwd",
        "py_compile",
        "pyclbr",
        "pydoc",
        "pydoc_data",
        "pyexpat",
        "queue",
        "quopri",
        "random",
        "re",
        "readline",
        "reprlib",
        "resource",
        "rlcompleter",
        "runpy",
        "sched",
        "secrets",
        "select",
        "selectors",
        "shelve",
        "shlex",
        "shutil",
        "signal",
        "site",
        "smtplib",
        "socket",
        "socketserver",
        "sqlite3",
        "sre_compile",
        "sre_constants",
        "sre_parse",
        "ssl",
        "stat",
        "statistics",
        "string",
        "stringprep",
        "struct",
        "subprocess",
        "symtable",
        "sys",
        "sysconfig",
        "syslog",
        "tabnanny",
        "tarfile",
        "tempfile",
        "termios",
        "textwrap",
        "this",
        "threading",
        "time",
        "timeit",
        "tkinter",
        "token",
        "tokenize",
        "tomllib",
        "trace",
        "traceback",
        "tracemalloc",
        "tty",
        "turtle",
        "turtledemo",
        "types",
        "typing",
        "unicodedata",
        "unittest",
        "urllib",
        "uuid",
        "venv",
        "warnings",
        "wave",
        "weakref",
        "webbrowser",
        "winreg",
        "winsound",
        "wsgiref",
        "xml",
        "xmlrpc",
        "zipapp",
        "zipfile",
        "zipimport",
        "zlib",
        "zoneinfo",
    ];
    module_ns_store(
        ns,
        "stdlib_module_names",
        pyre_object::setobject::w_frozenset_from_items(
            &STDLIB_MODULE_NAMES
                .iter()
                .map(|&n| w_str_new(n))
                .collect::<Vec<_>>(),
        ),
    );
    // sys.exception() — the value half of `sys.exc_info()`: the exception
    // instance currently being handled, or None outside an `except` block.
    module_ns_store(
        ns,
        "exception",
        make_builtin_function_with_arity(
            "exception",
            |_| {
                let exc = crate::eval::get_sys_exception();
                Ok(unsafe {
                    if exc.is_null() || !pyre_object::is_exception(exc) {
                        w_none()
                    } else {
                        exc
                    }
                })
            },
            0,
        ),
    );
    // sys.exc_clear — no-op
    module_ns_store(
        ns,
        "exc_clear",
        make_builtin_function_with_arity("exc_clear", |_| Ok(w_none()), 0),
    );
    // sys.is_remote_debug_enabled() — no remote-debug interface is wired,
    // so always False.
    module_ns_store(
        ns,
        "is_remote_debug_enabled",
        make_builtin_function_with_arity(
            "is_remote_debug_enabled",
            |_| Ok(pyre_object::w_bool_from(false)),
            0,
        ),
    );
    // sys.copyright — informational string consumed by `site` and `test`.
    module_ns_store(
        ns,
        "copyright",
        w_str_new("Copyright (c) 2001-2024 Python Software Foundation.\nAll Rights Reserved."),
    );
    // sys.getsizeof(obj[, default]) — PyPy vm.py returns the supplied default
    // for untracked objects.  str additionally exposes its PEP 393-compatible
    // `__sizeof__`, needed by the shared CPython test_str overflow check.
    module_ns_store(
        ns,
        "getsizeof",
        make_builtin_function(
            "getsizeof",
            |args| {
                if args.len() > 2 {
                    return Err(crate::PyError::type_error(format!(
                        "getsizeof() takes at most 2 arguments ({} given)",
                        args.len()
                    )));
                }
                let Some(&w_obj) = args.first() else {
                    return Err(crate::PyError::type_error(
                        "getsizeof() takes at least 1 argument (0 given)",
                    ));
                };
                if let Some(w_type) = crate::typedef::r#type(w_obj) {
                    if let Some(w_sizeof) = unsafe {
                        crate::baseobjspace::lookup_in_type(w_type.as_ptr(), "__sizeof__")
                    } {
                        let w_size = unsafe {
                            crate::baseobjspace::get_and_call_function(
                                w_sizeof,
                                w_obj,
                                w_type.as_ptr(),
                                &[],
                            )
                        }?;
                        // getsizeof must yield a non-negative integer: a
                        // non-int result is rejected like a failed index
                        // coercion, and a negative one (including a bignum)
                        // raises ValueError.
                        if unsafe { !pyre_object::is_int(w_size) } {
                            return Err(crate::PyError::type_error(format!(
                                "'{}' object cannot be interpreted as an integer",
                                crate::type_methods::arg_type_name(w_size)
                            )));
                        }
                        let negative = crate::baseobjspace::is_true(
                            crate::objspace::descroperation::compare(
                                w_size,
                                pyre_object::w_int_new(0),
                                crate::objspace::descroperation::CompareOp::Lt,
                            )?,
                        )?;
                        if negative {
                            return Err(crate::PyError::value_error(
                                "__sizeof__() should return >= 0",
                            ));
                        }
                        return Ok(w_size);
                    }
                }
                match args.get(1).copied() {
                    Some(w_default) => Ok(w_default),
                    None => Err(crate::PyError::type_error(
                        "getsizeof(object, default) -> int: object size is not tracked; supply a default",
                    )),
                }
            },
        ),
    );
    // PyPy normally omits CPython's raw refcount API.  The shared ctypes
    // tests only require the strong-reference delta created by a c_char_p
    // `_objects` keepalive; bytes records that real ownership transition in
    // its object payload, while other tracing-GC objects report the stable
    // call/argument baseline.
    module_ns_store(
        ns,
        "getrefcount",
        make_builtin_function_with_arity(
            "getrefcount",
            |args| {
                let owned = if unsafe { pyre_object::is_bytes(args[0]) } {
                    unsafe { pyre_object::bytesobject::w_bytes_ctypes_keepalive_refs(args[0]) }
                } else {
                    0
                };
                Ok(pyre_object::w_int_new((2 + owned) as i64))
            },
            1,
        ),
    );
    // sys.gettrace / settrace
    module_ns_store(
        ns,
        "gettrace",
        make_builtin_function_with_arity("gettrace", sys_gettrace_impl, 0),
    );
    module_ns_store(
        ns,
        "settrace",
        make_builtin_function_with_arity("settrace", sys_settrace_impl, 1),
    );
    // sys.getprofile / setprofile
    module_ns_store(
        ns,
        "getprofile",
        make_builtin_function_with_arity("getprofile", sys_getprofile_impl, 0),
    );
    module_ns_store(
        ns,
        "setprofile",
        make_builtin_function_with_arity("setprofile", sys_setprofile_impl, 1),
    );
    module_ns_store(
        ns,
        "_settraceallthreads",
        make_builtin_function_with_arity("_settraceallthreads", sys_settraceallthreads_impl, 1),
    );
    module_ns_store(
        ns,
        "_setprofileallthreads",
        make_builtin_function_with_arity("_setprofileallthreads", sys_setprofileallthreads_impl, 1),
    );
    module_ns_store(
        ns,
        "get_coroutine_origin_tracking_depth",
        make_builtin_function_with_arity(
            "get_coroutine_origin_tracking_depth",
            sys_get_coroutine_origin_tracking_depth,
            0,
        ),
    );
    module_ns_store(
        ns,
        "set_coroutine_origin_tracking_depth",
        make_builtin_function_with_arity(
            "set_coroutine_origin_tracking_depth",
            sys_set_coroutine_origin_tracking_depth,
            1,
        ),
    );
    module_ns_store(
        ns,
        "get_asyncgen_hooks",
        make_builtin_function_with_arity("get_asyncgen_hooks", sys_get_asyncgen_hooks_impl, 0),
    );
    module_ns_store(
        ns,
        "set_asyncgen_hooks",
        crate::make_builtin_function("set_asyncgen_hooks", sys_set_asyncgen_hooks_impl),
    );
    // sys.getfilesystemencoding
    module_ns_store(
        ns,
        "getfilesystemencoding",
        make_builtin_function_with_arity("getfilesystemencoding", |_| Ok(w_str_new("utf-8")), 0),
    );
    module_ns_store(
        ns,
        "getfilesystemencodeerrors",
        make_builtin_function_with_arity(
            "getfilesystemencodeerrors",
            |_| Ok(w_str_new("surrogateescape")),
            0,
        ),
    );
    // sys.audit — no-op
    module_ns_store(
        ns,
        "audit",
        crate::make_builtin_function("audit", |_| Ok(w_none())),
    );
    // sys._clear_type_descriptors(cls) — remove the descriptors owned by the
    // original class before `dataclasses._add_slots` copies its namespace into
    // the replacement slotted class.
    module_ns_store(
        ns,
        "_clear_type_descriptors",
        make_builtin_function_with_arity("_clear_type_descriptors", sys_clear_type_descriptors, 1),
    );
    // sys.is_finalizing
    module_ns_store(
        ns,
        "is_finalizing",
        make_builtin_function_with_arity(
            "is_finalizing",
            |_| Ok(w_bool_from(crate::module::thread::is_finalizing())),
            0,
        ),
    );
    // sys.displayhook / excepthook. `__displayhook__` keeps the original so
    // code (e.g. doctest) can save and restore the hook.
    module_ns_store(
        ns,
        "displayhook",
        make_builtin_function_with_arity("displayhook", crate::builtins::sys_displayhook, 1),
    );
    module_ns_store(
        ns,
        "__displayhook__",
        make_builtin_function_with_arity("displayhook", crate::builtins::sys_displayhook, 1),
    );
    module_ns_store(
        ns,
        "excepthook",
        make_builtin_function_with_arity("excepthook", |_| Ok(w_none()), 3),
    );
    // sys.breakpointhook — `app.py breakpointhook`, called by `breakpoint()`.
    // `__breakpointhook__` keeps the original so code can restore it.
    let breakpointhook_fn = make_builtin_function("breakpointhook", sys_breakpointhook);
    module_ns_store(ns, "breakpointhook", breakpointhook_fn);
    module_ns_store(ns, "__breakpointhook__", breakpointhook_fn);
    // sys.unraisablehook(unraisable) — handles exceptions raised where they
    // cannot propagate (e.g. __del__).  Stored alongside the read-only
    // `__unraisablehook__` original so code can save and restore it.
    let unraisablehook_fn =
        make_builtin_function_with_arity("unraisablehook", sys_unraisablehook, 1);
    module_ns_store(ns, "unraisablehook", unraisablehook_fn);
    module_ns_store(ns, "__unraisablehook__", unraisablehook_fn);
    // sys.path_hooks / path_importer_cache
    module_ns_store(ns, "path_hooks", w_list_new(vec![]));
    module_ns_store(ns, "path_importer_cache", w_dict_new());
    // sys.meta_path — empty
    module_ns_store(ns, "meta_path", w_list_new(vec![]));
    // sys.dont_write_bytecode — mirrors `sys.flags.dont_write_bytecode`
    // (`-B` / PYTHONDONTWRITEBYTECODE); no bytecode cache is written regardless,
    // but the reported value tracks the flag for compatibility.
    module_ns_store(
        ns,
        "dont_write_bytecode",
        w_bool_from(crate::importing::dont_write_bytecode_flag()),
    );
    // sys.pycache_prefix — None unless -X pycache_prefix / PYTHONPYCACHEPREFIX.
    // `importlib._bootstrap_external.cache_from_source` reads it to compute the
    // bytecode path before `dont_write_bytecode` is consulted.
    module_ns_store(ns, "pycache_prefix", w_none());
    // sys.addaudithook
    module_ns_store(
        ns,
        "addaudithook",
        make_builtin_function_with_arity("addaudithook", |_| Ok(w_none()), 1),
    );
}

/// `sysmodule.c sys._clear_type_descriptors`: remove the instance-dict and weakref
/// descriptors while retaining their references until both dictionary
/// mutations are complete, then invalidate the type lookup caches once.
fn sys_clear_type_descriptors(args: &[PyObjectRef]) -> crate::PyResult {
    let w_type = args[0];
    if !unsafe { pyre_object::is_type(w_type) } {
        return Err(crate::PyError::type_error(
            "_clear_type_descriptors() argument must be a type",
        ));
    }
    if !unsafe { pyre_object::w_type_is_heaptype(w_type) } {
        return Err(crate::PyError::type_error("argument is immutable"));
    }

    let _roots = pyre_object::gc_roots::push_roots();
    if let Some(descr) = crate::type_dict_lookup(w_type, "__dict__") {
        pyre_object::gc_roots::pin_root(descr);
    }
    if let Some(descr) = crate::type_dict_lookup(w_type, "__weakref__") {
        pyre_object::gc_roots::pin_root(descr);
    }
    crate::type_dict_delete(w_type, "__dict__");
    crate::type_dict_delete(w_type, "__weakref__");
    unsafe { crate::baseobjspace::mutated(w_type, None) };
    Ok(w_none())
}

/// Construct a stdio object whose type is `_io.TextIOWrapper` (so
/// `isinstance(sys.stdout, io.TextIOWrapper)` holds), exposing `write`,
/// `flush`, `isatty`, `fileno`, `reconfigure`, and `name`.  `fd` is the
/// descriptor it reports: 0 (stdin) / 1 (stdout) / 2 (stderr).  PyPy wires a
/// real W_File-backed `TextIOWrapper`; pyre routes writes through Rust's
/// stdout/stderr (the same sink as `print`) so output ordering is preserved,
/// storing the read/write surface as instance attributes.
fn stdio_encoding_and_errors() -> (String, String) {
    // PyPy app_main.py `initstdio`: a non-empty encoding before ':' is
    // explicit; an omitted encoding defaults to UTF-8 here, while a non-empty
    // errors suffix overrides the normal strict policy. stderr replaces its
    // error policy separately below.
    let Some(raw) = crate::importing::stdio_encoding() else {
        return ("utf-8".to_string(), "strict".to_string());
    };
    let (encoding, errors) = match raw.split_once(':') {
        Some((encoding, errors)) => (
            if encoding.is_empty() {
                "utf-8"
            } else {
                encoding
            },
            if errors.is_empty() { "strict" } else { errors },
        ),
        None if raw.is_empty() => ("utf-8", "strict"),
        None => (raw.as_str(), "strict"),
    };
    (encoding.to_string(), errors.to_string())
}

fn live_stdio_encoding_errors(stream_name: &str, default_errors: &str) -> (String, String) {
    let defaults = stdio_encoding_and_errors();
    let Some(sys) = crate::importing::get_sys_module("sys") else {
        return (defaults.0, default_errors.to_string());
    };
    let Ok(stream) = crate::baseobjspace::getattr_str(sys, stream_name) else {
        return (defaults.0, default_errors.to_string());
    };
    let text_attr = |name: &str, default: &str| {
        crate::baseobjspace::getattr_str(stream, name)
            .ok()
            .filter(|value| unsafe { is_str(*value) })
            .map(|value| unsafe { w_str_get_value(value) }.to_string())
            .unwrap_or_else(|| default.to_string())
    };
    (
        text_attr("encoding", &defaults.0),
        text_attr("errors", default_errors),
    )
}

fn stdio_stdin_readline(args: &[PyObjectRef]) -> crate::PyResult {
    if args.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "readline() takes at most one argument ({} given)",
            args.len()
        )));
    }
    let sys = crate::importing::get_sys_module("sys")
        .ok_or_else(|| crate::PyError::runtime_error("lost sys.stdin"))?;
    let stdin = crate::baseobjspace::getattr_str(sys, "stdin")?;
    let buffer = crate::baseobjspace::getattr_str(stdin, "buffer")?;
    let bytes = crate::baseobjspace::call_method(buffer, "readline", args);
    if bytes.is_null() {
        return Err(crate::call::take_call_error()
            .unwrap_or_else(|| crate::PyError::runtime_error("readline failed")));
    }
    if !unsafe { pyre_object::is_bytes(bytes) } {
        return Err(crate::PyError::type_error(
            "underlying readline() should have returned a bytes-like object",
        ));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(bytes);
    let bytes_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let (encoding, errors) = live_stdio_encoding_errors("stdin", "strict");
    crate::typedef::bytes_method_decode(&[
        pyre_object::gc_roots::shadow_stack_get(bytes_slot),
        w_str_new(&encoding),
        w_str_new(&errors),
    ])
}

/// `pylifecycle.c init_sys_streams` builds the standard streams after the
/// import system, so a text codec is reachable by the time they need one.
/// Pyre builds them from `sys` module creation instead, where the codec lookup
/// would import `encodings` and re-enter that creation through its own
/// `import sys`.  They start without an encoder and decoder and pick them up
/// here, once the import system is up.
///
/// The `__std*__` aliases name the streams this built even after user code
/// rebinds `sys.stdout`; a rebound one is not this function's to reconfigure.
pub fn init_stream_codecs() -> Result<(), crate::PyError> {
    let Some(sys) = crate::importing::get_sys_module("sys") else {
        return Ok(());
    };
    for name in ["__stdout__", "__stderr__", "__stdin__"] {
        if let Ok(stream) = crate::baseobjspace::getattr_str(sys, name) {
            crate::module::_io::W_TextIOWrapper::attach_stdio_codec(stream)?;
        }
    }
    Ok(())
}

fn make_std_stream(name: &'static str, fd: i32) -> PyObjectRef {
    let writable = fd != 0;
    let to_stderr = fd == 2;
    let unbuffered = writable && crate::importing::unbuffered_flag();
    // PyPy app_main.py `create_stdio`: retain the FileIO-backed binary layer
    // as TextIOWrapper.buffer. libregrtest workers deliberately write invalid
    // byte sequences through this exact owner.
    //
    // `create_stdio` also answers a descriptor `_io.open` rejects with no
    // stream at all. These streams keep instance-override methods that reach
    // the descriptor without going through the buffer, so a descriptor the
    // host does not open — the sandbox controller mounts no real files —
    // leaves the buffer absent instead of removing `sys.stdout` outright.
    let buffer = crate::builtins::builtin_open(&[
        w_int_new(i64::from(fd)),
        w_str_new(if writable { "wb" } else { "rb" }),
        w_int_new(if unbuffered { 0 } else { -1 }),
        w_none(),
        w_none(),
        w_none(),
        w_bool_from(false),
    ])
    .unwrap_or_else(|_| w_none());
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(buffer);
    let buffer_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let (encoding, configured_errors) = stdio_encoding_and_errors();
    let errors = if to_stderr {
        "backslashreplace"
    } else {
        configured_errors.as_str()
    };
    let stream = crate::module::_io::W_TextIOWrapper::allocate_stdio(
        name,
        pyre_object::gc_roots::shadow_stack_get(buffer_slot),
        &encoding,
        errors,
        unbuffered || to_stderr,
        unbuffered,
    );
    crate::baseobjspace::setdictvalue_native(stream, "name", w_str_new(name));
    // `pylifecycle.c init_set_builtins_open`/`init_sys_streams`: stderr uses the
    // `backslashreplace` handler so traceback printing never fails on a lone
    // surrogate; stdout/stdin default to `strict`.
    crate::baseobjspace::setdictvalue_native(
        stream,
        "mode",
        w_str_new(if writable { "w" } else { "r" }),
    );
    crate::baseobjspace::setdictvalue_native(stream, "closed", w_bool_from(false));
    crate::baseobjspace::setdictvalue_native(
        stream,
        "buffer",
        pyre_object::gc_roots::shadow_stack_get(buffer_slot),
    );
    // Instance-stored builtin methods do not get `self` prepended (see
    // pyopcode load_method dispatch), so the first arg may be the string
    // directly. Pick whichever element is a real str.
    fn pick_str(args: &[PyObjectRef]) -> Option<PyObjectRef> {
        for &a in args {
            if !a.is_null() && unsafe { is_str(a) } {
                return Some(a);
            }
        }
        None
    }
    // Encode through `encode_object` with the stream's error handler so a lone
    // surrogate is routed there (stdout `strict` → UnicodeEncodeError; stderr
    // `backslashreplace` → escaped) instead of panicking in `w_str_get_value`.
    let write_fn = if to_stderr {
        crate::make_builtin_function("write", |args| {
            if let Some(s_obj) = pick_str(args) {
                let (encoding, _) = live_stdio_encoding_errors("stderr", "backslashreplace");
                let bytes =
                    crate::type_methods::encode_object(s_obj, &encoding, "backslashreplace")?;
                // Under sandbox fd 1 is the marshalling pipe, so a raw write
                // would corrupt the protocol: route through ll_os_write(2,…)
                // and let the controller relay it to its own stderr.
                #[cfg(not(feature = "sandbox"))]
                {
                    use std::io::Write;
                    let _ = std::io::stderr().write_all(&bytes);
                }
                #[cfg(feature = "sandbox")]
                crate::host_seam::ops::write(2, &bytes)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                return Ok(w_int_new(unsafe { w_str_len(s_obj) } as i64));
            }
            Ok(w_int_new(0))
        })
    } else {
        crate::make_builtin_function("write", |args| {
            if let Some(s_obj) = pick_str(args) {
                let (encoding, errors) = live_stdio_encoding_errors("stdout", "strict");
                let bytes = crate::type_methods::encode_object(s_obj, &encoding, &errors)?;
                #[cfg(not(feature = "sandbox"))]
                {
                    use std::io::Write;
                    let _ = std::io::stdout().write_all(&bytes);
                }
                #[cfg(feature = "sandbox")]
                crate::host_seam::ops::write(1, &bytes)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                return Ok(w_int_new(unsafe { w_str_len(s_obj) } as i64));
            }
            Ok(w_int_new(0))
        })
    };
    crate::baseobjspace::setdictvalue_native(stream, "write", write_fn);
    if fd == 0 {
        crate::baseobjspace::setdictvalue_native(
            stream,
            "readline",
            crate::make_builtin_function("readline", stdio_stdin_readline),
        );
    }
    crate::baseobjspace::setdictvalue_native(
        stream,
        "flush",
        crate::make_builtin_function("flush", |_| {
            // The sandbox path writes unbuffered ll_os_write requests, so there
            // is nothing to flush (and the real fds are the marshalling pipe).
            #[cfg(not(feature = "sandbox"))]
            {
                use std::io::Write;
                let _ = std::io::stdout().flush();
                let _ = std::io::stderr().flush();
            }
            Ok(w_none())
        }),
    );
    crate::baseobjspace::setdictvalue_native(
        stream,
        "isatty",
        crate::make_builtin_function("isatty", |_| Ok(w_bool_from(false))),
    );
    // `BuiltinCodeFn` is a bare `fn` pointer (no captures), so select a
    // constant-returning function per descriptor rather than closing over `fd`.
    let fileno_fn = match fd {
        0 => crate::make_builtin_function("fileno", |_| Ok(w_int_new(0))),
        2 => crate::make_builtin_function("fileno", |_| Ok(w_int_new(2))),
        _ => crate::make_builtin_function("fileno", |_| Ok(w_int_new(1))),
    };
    crate::baseobjspace::setdictvalue_native(stream, "fileno", fileno_fn);
    let (writable_fn, readable_fn) = if writable {
        (
            crate::make_builtin_function("writable", |_| Ok(w_bool_from(true))),
            crate::make_builtin_function("readable", |_| Ok(w_bool_from(false))),
        )
    } else {
        (
            crate::make_builtin_function("writable", |_| Ok(w_bool_from(false))),
            crate::make_builtin_function("readable", |_| Ok(w_bool_from(true))),
        )
    };
    crate::baseobjspace::setdictvalue_native(stream, "writable", writable_fn);
    crate::baseobjspace::setdictvalue_native(stream, "readable", readable_fn);
    stream
}
