//! `types.GenericAlias` — PEP 585 parameterized generics.
//!
//! PyPy equivalent: lib_pypy/_pypy_generic_alias.py (GenericAlias) +
//! pypy/objspace/std/util.py:99 (generic_alias_class_getitem).
//!
//! The payload lives in `pyre_object::genericaliasobject::W_GenericAlias`;
//! this module is the behaviour surface (class-getitem constructor,
//! parameter collection, and the typedef methods).

use crate::{
    DictStorage, dict_storage_store, make_builtin_function, make_builtin_function_with_arity,
};
use pyre_object::*;

/// `_ATTR_EXCEPTIONS` (`_pypy_generic_alias.py:1`) — attribute names that
/// resolve on the alias itself; every other name delegates to the
/// `__origin__` through `__getattribute__`.
pub(crate) const ATTR_EXCEPTIONS: &[&str] = &[
    "__args__",
    "__class__",
    "__copy__",
    "__deepcopy__",
    "__mro_entries__",
    "__origin__",
    "__parameters__",
    "__reduce__",
    "__reduce_ex__",
    "__typing_unpacked_tuple_args__",
    "__unpacked__",
];

/// `name in _ATTR_EXCEPTIONS` — used by `getattr` to decide whether to
/// delegate to `__origin__`.
pub(crate) fn is_attr_exception(name: &str) -> bool {
    ATTR_EXCEPTIONS.contains(&name)
}

/// `generic_alias_class_getitem(space, w_cls, w_item)` (util.py:99).
///
/// Registered as the `__class_getitem__` classmethod on builtin
/// containers, so the bound call delivers `args = [w_cls, w_item]`.  The
/// `w_item` operand is mandatory (the gateway declares it positional).
pub fn generic_alias_class_getitem(args: &[PyObjectRef]) -> crate::PyResult {
    if args.len() != 2 {
        // The message is prefixed with the bound class's name
        // (`list.__class_getitem__() takes exactly one argument`).
        let prefix = args
            .first()
            .filter(|&&c| unsafe { is_type(c) })
            .map(|&c| format!("{}.", unsafe { pyre_object::w_type_get_name(c) }))
            .unwrap_or_default();
        return Err(crate::PyError::type_error(format!(
            "{prefix}__class_getitem__() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    Ok(make_generic_alias(args[0], args[1]))
}

/// `GenericAlias.__new__` (`_pypy_generic_alias.py:19`) — wrap a bare item
/// into a 1-tuple, collect the free parameters, allocate.
pub fn make_generic_alias(origin: PyObjectRef, item: PyObjectRef) -> PyObjectRef {
    let args = if unsafe { is_tuple(item) } {
        item
    } else {
        w_tuple_new(vec![item])
    };
    let parameters = collect_parameters(args);
    w_generic_alias_new(origin, args, parameters)
}

/// `_collect_parameters(args)` (`_pypy_generic_alias.py:150`) — gather the
/// free type variables in order of first appearance.
fn collect_parameters(args: PyObjectRef) -> PyObjectRef {
    let mut params: Vec<PyObjectRef> = Vec::new();
    let n = unsafe { w_tuple_len(args) };
    for i in 0..n {
        if let Some(t) = unsafe { w_tuple_getitem(args, i as i64) } {
            collect_parameters_one(t, &mut params);
        }
    }
    w_tuple_new(params)
}

fn collect_parameters_one(t: PyObjectRef, params: &mut Vec<PyObjectRef>) {
    unsafe {
        if is_type(t) {
            // A bare class exposes no `__parameters__` descriptor of its own.
            return;
        }
        if is_tuple(t) {
            let n = w_tuple_len(t);
            for i in 0..n {
                if let Some(x) = w_tuple_getitem(t, i as i64) {
                    collect_parameters_one(x, params);
                }
            }
            return;
        }
    }
    // `hasattr(t, '__typing_subst__')` → `t` is itself a parameter.
    if crate::baseobjspace::getattr_str(t, "__typing_subst__").is_ok() {
        push_unique(params, t);
        return;
    }
    // Otherwise pull `getattr(t, '__parameters__', ())`.
    if let Ok(sub) = crate::baseobjspace::getattr_str(t, "__parameters__") {
        if unsafe { is_tuple(sub) } {
            let n = unsafe { w_tuple_len(sub) };
            for i in 0..n {
                if let Some(x) = unsafe { w_tuple_getitem(sub, i as i64) } {
                    push_unique(params, x);
                }
            }
        }
    }
}

fn push_unique(params: &mut Vec<PyObjectRef>, item: PyObjectRef) {
    if !params.iter().any(|&p| crate::baseobjspace::eq_w(p, item)) {
        params.push(item);
    }
}

// ── typedef methods ──────────────────────────────────────────────────

/// `__origin__` getset (`GenericAlias.__origin__`).
fn ga_get_origin(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = args.get(1).copied().unwrap_or_else(w_none);
    if unsafe { is_generic_alias(self_) } {
        Ok(unsafe { w_generic_alias_get_origin(self_) })
    } else {
        Ok(w_none())
    }
}

/// `__args__` getset (`GenericAlias.__args__`).
fn ga_get_args(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = args.get(1).copied().unwrap_or_else(w_none);
    if unsafe { is_generic_alias(self_) } {
        Ok(unsafe { w_generic_alias_get_args(self_) })
    } else {
        Ok(w_none())
    }
}

/// `__parameters__` getset (`GenericAlias.__parameters__`).
fn ga_get_parameters(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = args.get(1).copied().unwrap_or_else(w_none);
    if unsafe { is_generic_alias(self_) } {
        Ok(unsafe { w_generic_alias_get_parameters(self_) })
    } else {
        Ok(w_none())
    }
}

/// Read `args[0]` as the bound `self`, rejecting a non-GenericAlias
/// before any unsafe field access (an unbound/forged direct call).
fn self_alias(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.first().copied().unwrap_or_else(w_none);
    if !unsafe { is_generic_alias(self_) } {
        return Err(crate::PyError::type_error(
            "descriptor requires a 'types.GenericAlias' object",
        ));
    }
    Ok(self_)
}

/// `GenericAlias.__eq__` (`_pypy_generic_alias.py:64`).
fn ga_eq(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = args.first().copied().unwrap_or_else(w_none);
    let other = args.get(1).copied().unwrap_or_else(w_none);
    if !unsafe { is_generic_alias(self_) } || !unsafe { is_generic_alias(other) } {
        return Ok(w_not_implemented());
    }
    let eq = unsafe {
        crate::baseobjspace::eq_w(
            w_generic_alias_get_origin(self_),
            w_generic_alias_get_origin(other),
        ) && crate::baseobjspace::eq_w(
            w_generic_alias_get_args(self_),
            w_generic_alias_get_args(other),
        ) && w_generic_alias_get_unpacked(self_) == w_generic_alias_get_unpacked(other)
    };
    Ok(w_bool_from(eq))
}

/// `GenericAlias.__mro_entries__` (`_pypy_generic_alias.py:49`) —
/// `(self.__origin__,)`, so `class C(list[int])` resolves to `list`.
fn ga_mro_entries(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = self_alias(args)?;
    let origin = unsafe { w_generic_alias_get_origin(self_) };
    Ok(w_tuple_new(vec![origin]))
}

/// `GenericAlias.__getitem__` (`_pypy_generic_alias.py:71`) — substitute
/// the free parameters.  pyre has no `TypeVar`, so a constructed alias
/// always carries an empty `__parameters__`; subscripting it then raises
/// the `subs_parameters` `nparams == 0` error.
fn ga_getitem(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = self_alias(args)?;
    let nparams = unsafe { w_tuple_len(w_generic_alias_get_parameters(self_)) };
    if nparams == 0 {
        let repr = unsafe { crate::display::py_repr(self_)? };
        return Err(crate::PyError::type_error(format!(
            "{repr} is not a generic class"
        )));
    }
    // Reachable only with free TypeVars, which pyre cannot construct.
    Err(crate::PyError::type_error(
        "parameterized generic substitution is not supported",
    ))
}

/// `add_recurse` (`_pypy_generic_alias.py:253-255`) maps a bare `None`
/// operand to `type(None)` before it lands in `__args__`, so
/// `(int | None).__args__` is `(int, NoneType)`.
fn normalize_none(x: PyObjectRef) -> PyObjectRef {
    if unsafe { pyre_object::is_none(x) } {
        crate::typedef::gettypeobject(&pyre_object::NONE_TYPE)
    } else {
        x
    }
}

/// `_create_union(x, y)` (`_pypy_generic_alias.py:328`) — both operands
/// must be unionable, else `NotImplemented`; identical operands collapse.
pub(crate) fn create_union(x: PyObjectRef, y: PyObjectRef) -> crate::PyResult {
    use crate::objspace::descroperation::unionable;
    if !unionable(x) || !unionable(y) {
        return Ok(w_not_implemented());
    }
    if unsafe { crate::baseobjspace::eq_w(x, y) } {
        return Ok(x);
    }
    Ok(w_union_new(normalize_none(x), normalize_none(y)))
}

/// `UnionType.__eq__` (`_pypy_generic_alias.py:270-273`) —
/// `set(self.__args__) == set(other.__args__)`.  Both arg tuples are
/// deduplicated at construction, so equal length plus subset is set
/// equality.
pub(crate) fn union_set_eq(a: PyObjectRef, b: PyObjectRef) -> bool {
    unsafe {
        let aa = w_union_get_args(a);
        let bb = w_union_get_args(b);
        let na = w_tuple_len(aa);
        let nb = w_tuple_len(bb);
        if na != nb {
            return false;
        }
        for i in 0..na {
            let Some(x) = w_tuple_getitem(aa, i as i64) else {
                return false;
            };
            let mut found = false;
            for j in 0..nb {
                if let Some(y) = w_tuple_getitem(bb, j as i64) {
                    if crate::baseobjspace::eq_w(x, y) {
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                return false;
            }
        }
        true
    }
}

/// `GenericAlias.__or__` (`_pypy_generic_alias.py:102`) — `X[...] | Y`.
fn ga_or(args: &[PyObjectRef]) -> crate::PyResult {
    let a = args.first().copied().unwrap_or_else(w_none);
    let b = args.get(1).copied().unwrap_or_else(w_none);
    create_union(a, b)
}

/// `GenericAlias.__ror__` (`_pypy_generic_alias.py:105`) — `Y | X[...]`.
fn ga_ror(args: &[PyObjectRef]) -> crate::PyResult {
    let a = args.first().copied().unwrap_or_else(w_none);
    let b = args.get(1).copied().unwrap_or_else(w_none);
    create_union(b, a)
}

/// `GenericAlias.__instancecheck__` (`_pypy_generic_alias.py:93`).
fn ga_instancecheck(_args: &[PyObjectRef]) -> crate::PyResult {
    Err(crate::PyError::type_error(
        "isinstance() argument 2 cannot be a parameterized generic",
    ))
}

/// `GenericAlias.__subclasscheck__` (`_pypy_generic_alias.py:90`).
fn ga_subclasscheck(_args: &[PyObjectRef]) -> crate::PyResult {
    Err(crate::PyError::type_error(
        "issubclass() argument 2 cannot be a parameterized generic",
    ))
}

/// `GenericAlias.__new__(cls, origin, args)` (`_pypy_generic_alias.py:19`)
/// — the public `types.GenericAlias(list, int)` constructor.
fn ga_new(args: &[PyObjectRef]) -> crate::PyResult {
    // args[0] is the cls passed by `type.__call__`.
    if args.len() != 3 {
        return Err(crate::PyError::type_error(format!(
            "GenericAlias expected 2 arguments, got {}",
            args.len().saturating_sub(1)
        )));
    }
    Ok(make_generic_alias(args[1], args[2]))
}

/// Build the `types.GenericAlias` namespace.
pub(crate) fn init_generic_alias_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", crate::typedef::make_new_descr(ga_new));
    dict_storage_store(
        ns,
        "__origin__",
        crate::typedef::make_getset_descriptor_named(
            make_builtin_function_with_arity("__origin__", ga_get_origin, 2),
            "__origin__",
        ),
    );
    dict_storage_store(
        ns,
        "__args__",
        crate::typedef::make_getset_descriptor_named(
            make_builtin_function_with_arity("__args__", ga_get_args, 2),
            "__args__",
        ),
    );
    dict_storage_store(
        ns,
        "__parameters__",
        crate::typedef::make_getset_descriptor_named(
            make_builtin_function_with_arity("__parameters__", ga_get_parameters, 2),
            "__parameters__",
        ),
    );
    dict_storage_store(ns, "__eq__", make_builtin_function("__eq__", ga_eq));
    // __hash__ and __call__ are resolved at their dispatch points
    // (`builtins::hash_value`, `call::call_function_impl_result`) because
    // pyre does not consult a typedef slot for them on builtin W_Roots.
    dict_storage_store(
        ns,
        "__getitem__",
        make_builtin_function("__getitem__", ga_getitem),
    );
    dict_storage_store(
        ns,
        "__mro_entries__",
        make_builtin_function("__mro_entries__", ga_mro_entries),
    );
    dict_storage_store(ns, "__or__", make_builtin_function("__or__", ga_or));
    dict_storage_store(ns, "__ror__", make_builtin_function("__ror__", ga_ror));
    dict_storage_store(
        ns,
        "__instancecheck__",
        make_builtin_function("__instancecheck__", ga_instancecheck),
    );
    dict_storage_store(
        ns,
        "__subclasscheck__",
        make_builtin_function("__subclasscheck__", ga_subclasscheck),
    );
}

/// Render a GenericAlias for `repr()` (`GenericAlias.__repr__`,
/// `_pypy_generic_alias.py:57`).  Implemented here (not as a typedef
/// `__repr__`) so it matches the builtin-W_Root repr architecture, where
/// `display::py_repr` owns the rendering and explicit `.__repr__` access
/// still delegates to `__origin__`.
///
/// # Safety
/// `obj` must point to a valid `W_GenericAlias`.
pub(crate) unsafe fn repr(obj: PyObjectRef) -> Result<String, crate::PyError> {
    let origin = w_generic_alias_get_origin(obj);
    let args = w_generic_alias_get_args(obj);
    let n = w_tuple_len(args);
    let inner = if n == 0 {
        "()".to_string()
    } else {
        let mut parts = Vec::with_capacity(n);
        for i in 0..n {
            if let Some(item) = w_tuple_getitem(args, i as i64) {
                parts.push(repr_item(item)?);
            }
        }
        parts.join(", ")
    };
    let star = if w_generic_alias_get_unpacked(obj) {
        "*"
    } else {
        ""
    };
    Ok(format!("{star}{}[{inner}]", repr_item(origin)?))
}

/// `_repr_item(it)` (`_pypy_generic_alias.py:124`) — a class renders as its
/// qualname (prefixed with the module when it is not `builtins`); anything
/// else falls back to `repr`.
unsafe fn repr_item(it: PyObjectRef) -> Result<String, crate::PyError> {
    if is_ellipsis(it) {
        return Ok("...".to_string());
    }
    if is_generic_alias(it) {
        return repr(it);
    }
    // `getattr(it, "__qualname__")` / `getattr(it, "__module__")`.
    if let Ok(w_qualname) = crate::baseobjspace::getattr_str(it, "__qualname__") {
        if let Ok(qualname) = crate::baseobjspace::text_w(w_qualname) {
            let module = crate::baseobjspace::getattr_str(it, "__module__")
                .ok()
                .and_then(|w| crate::baseobjspace::text_w(w).ok());
            return Ok(match module {
                Some(m) if m != "builtins" => format!("{m}.{qualname}"),
                _ => qualname.to_string(),
            });
        }
    }
    crate::display::py_repr(it)
}
