//! `types.GenericAlias` — PEP 585 parameterized generics.
//!
//! PyPy equivalent: lib_pypy/_pypy_generic_alias.py (GenericAlias) +
//! pypy/objspace/std/util.py:99 (generic_alias_class_getitem).
//!
//! The payload lives in `pyre_object::_pypy_generic_alias::GenericAlias`;
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
    make_generic_alias(args[0], args[1])
}

/// `GenericAlias.__new__` (`_pypy_generic_alias.py:19`) — wrap a bare item
/// into a 1-tuple, collect the free parameters, allocate.
pub fn make_generic_alias(origin: PyObjectRef, item: PyObjectRef) -> crate::PyResult {
    let args = if unsafe { is_tuple(item) } {
        item
    } else {
        w_tuple_new(vec![item])
    };
    let parameters = collect_parameters(args)?;
    Ok(w_generic_alias_new(origin, args, parameters))
}

/// `_collect_parameters(args)` (`_pypy_generic_alias.py:150`) — gather the
/// free type variables in order of first appearance.
pub(crate) fn collect_parameters(args: PyObjectRef) -> crate::PyResult {
    let mut params: Vec<PyObjectRef> = Vec::new();
    let n = unsafe { w_tuple_len(args) };
    for i in 0..n {
        if let Some(t) = unsafe { w_tuple_getitem(args, i as i64) } {
            collect_parameters_one(t, &mut params)?;
        }
    }
    Ok(w_tuple_new(params))
}

fn collect_parameters_one(
    t: PyObjectRef,
    params: &mut Vec<PyObjectRef>,
) -> Result<(), crate::PyError> {
    unsafe {
        if is_type(t) {
            // A bare class exposes no `__parameters__` descriptor of its own.
            return Ok(());
        }
        if is_tuple(t) {
            let n = w_tuple_len(t);
            for i in 0..n {
                if let Some(x) = w_tuple_getitem(t, i as i64) {
                    collect_parameters_one(x, params)?;
                }
            }
            return Ok(());
        }
    }
    // `hasattr(t, '__typing_subst__')` → `t` is itself a parameter.  `hasattr`
    // only swallows `AttributeError`; a misbehaving descriptor that raises
    // anything else propagates.
    match crate::baseobjspace::getattr_str(t, "__typing_subst__") {
        Ok(_) => {
            push_unique(params, t)?;
            return Ok(());
        }
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {}
        Err(e) => return Err(e),
    }
    // Otherwise pull `getattr(t, '__parameters__', ())` — the `()` default
    // applies only on `AttributeError`.
    match crate::baseobjspace::getattr_str(t, "__parameters__") {
        Ok(sub) => {
            if unsafe { is_tuple(sub) } {
                let n = unsafe { w_tuple_len(sub) };
                for i in 0..n {
                    if let Some(x) = unsafe { w_tuple_getitem(sub, i as i64) } {
                        push_unique(params, x)?;
                    }
                }
            }
        }
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {}
        Err(e) => return Err(e),
    }
    Ok(())
}

fn push_unique(params: &mut Vec<PyObjectRef>, item: PyObjectRef) -> Result<(), crate::PyError> {
    // `if item not in parameters: parameters.append(item)` — the `in` test is
    // `tuple.__contains__`, so a raising `__eq__` propagates.
    for &p in params.iter() {
        if crate::baseobjspace::eq_w(p, item)? {
            return Ok(());
        }
    }
    params.push(item);
    Ok(())
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
        )? && crate::baseobjspace::eq_w(
            w_generic_alias_get_args(self_),
            w_generic_alias_get_args(other),
        )? && w_generic_alias_get_unpacked(self_) == w_generic_alias_get_unpacked(other)
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

/// `GenericAlias.__getitem__` (`_pypy_generic_alias.py:71`) — substitute the
/// free parameters with `items` and build the resulting alias.
fn ga_getitem(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = self_alias(args)?;
    let items_raw = args.get(1).copied().unwrap_or_else(w_none);
    let items = if unsafe { is_tuple(items_raw) } {
        items_raw
    } else {
        w_tuple_new(vec![items_raw])
    };
    let params = unsafe { w_generic_alias_get_parameters(self_) };
    let ga_args = unsafe { w_generic_alias_get_args(self_) };
    let newargs = subs_parameters(self_, ga_args, params, items)?;
    let res = make_generic_alias(
        unsafe { w_generic_alias_get_origin(self_) },
        w_tuple_new(newargs),
    )?;
    if unsafe { w_generic_alias_get_unpacked(self_) } {
        unsafe { w_generic_alias_set_unpacked(res, true) };
    }
    Ok(res)
}

/// `tuple.index(item)` resolved through `eq_w`; `Ok(None)` plays the Python
/// `ValueError` raised when the item is absent.  A raising `__eq__`
/// propagates (`tuple.index` does not swallow comparison errors).
fn tuple_index(t: PyObjectRef, item: PyObjectRef) -> Result<Option<usize>, crate::PyError> {
    let n = unsafe { w_tuple_len(t) };
    for i in 0..n {
        if let Some(x) = unsafe { w_tuple_getitem(t, i as i64) } {
            if crate::baseobjspace::eq_w(x, item)? {
                return Ok(Some(i));
            }
        }
    }
    Ok(None)
}

/// `_unpack_args(*items)` (`typing.py:341`) — flatten any element that is an
/// unpacked `tuple[...]` alias (one exposing `__typing_unpacked_tuple_args__`)
/// into its members, unless those end in `...`.  Returns a fresh items tuple.
fn unpack_args(items: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let n = unsafe { w_tuple_len(items) };
    let mut newargs: Vec<PyObjectRef> = Vec::new();
    for i in 0..n {
        let Some(arg) = (unsafe { w_tuple_getitem(items, i as i64) }) else {
            continue;
        };
        let subargs = match crate::baseobjspace::getattr_str(arg, "__typing_unpacked_tuple_args__")
        {
            Ok(s) => s,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => w_none(),
            Err(e) => return Err(e),
        };
        // `if subargs is not None and not (subargs and subargs[-1] is ...)`
        // — `subargs` is any object exposing `__typing_unpacked_tuple_args__`,
        // not necessarily a tuple, so the gate uses the general truthiness /
        // subscription protocol.
        let do_unpack = if unsafe { pyre_object::is_none(subargs) } {
            false
        } else {
            let ends_ellipsis = crate::baseobjspace::is_true(subargs)? && {
                let last = crate::baseobjspace::getitem(subargs, w_int_new(-1))?;
                unsafe { is_ellipsis(last) }
            };
            !ends_ellipsis
        };
        if do_unpack {
            // `newargs.extend(subargs)` — any iterable, not just a tuple.
            for x in crate::builtins::collect_iterable(subargs)? {
                newargs.push(x);
            }
        } else {
            newargs.push(arg);
        }
    }
    Ok(w_tuple_new(newargs))
}

/// `_is_unpacked_typevartuple(x)` (`typing.py:1026`) — `x` is an unpacked
/// `TypeVarTuple` (`*Ts`), identified by `__typing_is_unpacked_typevartuple__
/// is True`; a bare class is never one.
fn is_unpacked_typevartuple(x: PyObjectRef) -> Result<bool, crate::PyError> {
    if unsafe { is_type(x) } {
        return Ok(false);
    }
    match crate::baseobjspace::getattr_str(x, "__typing_is_unpacked_typevartuple__") {
        Ok(v) => Ok(v == w_bool_from(true)),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => Ok(false),
        Err(e) => Err(e),
    }
}

/// `isinstance(param, TypeVarTuple)` — mirrors `_is_typevar`'s bootstrapping
/// shortcut (`_pypy_generic_alias.py:146`): match the parameter's type by
/// `__name__` + `__module__` rather than importing `typing`.
fn is_typevartuple(param: PyObjectRef) -> bool {
    let Some(t) = crate::typedef::r#type(param) else {
        return false;
    };
    if unsafe { pyre_object::w_type_get_name(t) } != "TypeVarTuple" {
        return false;
    }
    matches!(
        crate::baseobjspace::getattr_str(t, "__module__")
            .ok()
            .and_then(|m| crate::baseobjspace::text_w(m).ok()),
        Some("typing")
    )
}

/// `subs_parameters(self, args, params, items)` (`_pypy_generic_alias.py:207`)
/// — produce the substituted `__args__` for `self[items]`.  Shared by
/// `GenericAlias.__getitem__` and `UnionType.__getitem__`.
pub(crate) fn subs_parameters(
    self_: PyObjectRef,
    args: PyObjectRef,
    params: PyObjectRef,
    items: PyObjectRef,
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let nparams = unsafe { w_tuple_len(params) };
    if nparams == 0 {
        let repr = unsafe { crate::display::py_repr(self_)? };
        return Err(crate::PyError::type_error(format!(
            "{repr} is not a generic class"
        )));
    }
    // `items = _unpack_args(items)` flattens unpacked `tuple[...]` aliases.
    // `__typing_prepare_subst__` then reshapes `items` for
    // `ParamSpec`/`TypeVarTuple` parameters — honoured per param, missing
    // attribute (the `None` default) skips it.
    let mut items = unpack_args(items)?;
    for i in 0..nparams {
        let Some(param) = (unsafe { w_tuple_getitem(params, i as i64) }) else {
            continue;
        };
        // `prepare = getattr(param, '__typing_prepare_subst__', None)` then
        // `if prepare is not None`: a missing attribute and an attribute
        // explicitly set to `None` both skip the reshape.
        let prepare = match crate::baseobjspace::getattr_str(param, "__typing_prepare_subst__") {
            Ok(p) => p,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => w_none(),
            Err(e) => return Err(e),
        };
        if !unsafe { pyre_object::is_none(prepare) } {
            items = crate::call::call_function_impl_result(prepare, &[self_, items])?;
        }
    }
    let nitems = unsafe { w_tuple_len(items) };
    if nparams != nitems {
        let direction = if nitems > nparams { "many" } else { "few" };
        let s = unsafe { crate::display::py_repr(self_)? };
        return Err(crate::PyError::type_error(format!(
            "Too {direction} arguments for {s}; actual {nitems}, expected {nparams}"
        )));
    }
    let mut newargs: Vec<PyObjectRef> = Vec::new();
    let nargs = unsafe { w_tuple_len(args) };
    for i in 0..nargs {
        let Some(old_arg) = (unsafe { w_tuple_getitem(args, i as i64) }) else {
            continue;
        };
        if unsafe { is_type(old_arg) } {
            newargs.push(old_arg);
            continue;
        }
        // `unpack = _is_unpacked_typevartuple(old_arg)` decides whether the
        // produced `arg` is spliced (`newargs.extend`) or appended.
        let unpack = is_unpacked_typevartuple(old_arg)?;
        // `meth = getattr(old_arg, '__typing_subst__', None)` then
        // `if meth is not None`: a missing attribute and an attribute
        // explicitly set to `None` both fall through to `subs_tvars`.
        let meth = match crate::baseobjspace::getattr_str(old_arg, "__typing_subst__") {
            Ok(m) => m,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => w_none(),
            Err(e) => return Err(e),
        };
        let arg = if !unsafe { pyre_object::is_none(meth) } {
            let iparam = tuple_index(params, old_arg)?
                .ok_or_else(|| crate::PyError::value_error("tuple.index(x): x not in tuple"))?;
            let item = unsafe { w_tuple_getitem(items, iparam as i64) }.unwrap_or_else(w_none);
            crate::call::call_function_impl_result(meth, &[item])?
        } else {
            subs_tvars(old_arg, params, items)?
        };
        if unpack {
            // `newargs.extend(arg)` — splice an unpacked `TypeVarTuple`'s
            // substituted members.
            for x in crate::builtins::collect_iterable(arg)? {
                newargs.push(x);
            }
        } else {
            newargs.push(arg);
        }
    }
    Ok(newargs)
}

/// `subs_tvars(obj, params, argitems)` (`_pypy_generic_alias.py:183`) —
/// substitute the parameters of a nested generic and re-subscript it.
fn subs_tvars(
    obj: PyObjectRef,
    params: PyObjectRef,
    argitems: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let subparams = match crate::baseobjspace::getattr_str(obj, "__parameters__") {
        Ok(sub) => sub,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => return Ok(obj),
        Err(e) => return Err(e),
    };
    if !unsafe { is_tuple(subparams) } || unsafe { w_tuple_len(subparams) } == 0 {
        return Ok(obj);
    }
    let nsub = unsafe { w_tuple_len(subparams) };
    let mut subargs: Vec<PyObjectRef> = Vec::with_capacity(nsub);
    for i in 0..nsub {
        let Some(param) = (unsafe { w_tuple_getitem(subparams, i as i64) }) else {
            continue;
        };
        // `try: argitems[params.index(param)] except ValueError: param`.
        let arg = match tuple_index(params, param)? {
            Some(idx) => unsafe { w_tuple_getitem(argitems, idx as i64) }.unwrap_or(param),
            None => param,
        };
        // `if isinstance(param, TypeVarTuple): subargs.extend(arg)` — a
        // `TypeVarTuple` captures a sequence, so its bound `arg` is spliced.
        if is_typevartuple(param) {
            for x in crate::builtins::collect_iterable(arg)? {
                subargs.push(x);
            }
        } else {
            subargs.push(arg);
        }
    }
    crate::baseobjspace::getitem(obj, w_tuple_new(subargs))
}

/// `_make_starred(ga)` (`_pypy_generic_alias.py:118`) — a copy of the alias
/// flagged unpacked, so it renders `*X[...]` and `iter()` yields it.
pub(crate) fn make_starred(ga: PyObjectRef) -> crate::PyResult {
    let origin = unsafe { w_generic_alias_get_origin(ga) };
    let args = unsafe { w_generic_alias_get_args(ga) };
    let res = make_generic_alias(origin, args)?;
    unsafe { w_generic_alias_set_unpacked(res, true) };
    Ok(res)
}

/// The `_make_starred` callable referenced by an unpacked alias's
/// `__reduce__`.
fn ga_make_starred(args: &[PyObjectRef]) -> crate::PyResult {
    let ga = args.first().copied().unwrap_or_else(w_none);
    if !unsafe { is_generic_alias(ga) } {
        return Err(crate::PyError::type_error(
            "_make_starred() argument must be a types.GenericAlias",
        ));
    }
    make_starred(ga)
}

/// The shared `_make_starred` callable, lazily built then cached.
///
/// The single `_make_starred` callable (`_pypy_generic_alias.py:118`),
/// kept reachable by the GenericAlias type namespace it is also stored in
/// (`init_generic_alias_type`) and never reallocated, so an unpacked
/// alias's `__reduce__` returns the same object every time — PyPy returns
/// the module-level function.  pyre houses it on the type as the stand-in
/// for PyPy's `_pypy_generic_alias._make_starred` module global; the
/// stable-address (old-gen) allocation makes the cached pointer safe to
/// hold across collections.  Process-global so every thread observes the
/// same callable identity.
pub(crate) fn make_starred_fn() -> PyObjectRef {
    static MAKE_STARRED_FN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAKE_STARRED_FN
        .get_or_init(|| make_builtin_function("_make_starred", ga_make_starred) as usize)
        as PyObjectRef
}

/// `GenericAlias.__reduce__` (`_pypy_generic_alias.py:96`).
fn ga_reduce(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = self_alias(args)?;
    let origin = unsafe { w_generic_alias_get_origin(self_) };
    let ga_args = unsafe { w_generic_alias_get_args(self_) };
    if unsafe { w_generic_alias_get_unpacked(self_) } {
        // `orig = GenericAlias(origin, args); (_make_starred, (orig,))`.
        let orig = make_generic_alias(origin, ga_args)?;
        let callable = make_starred_fn();
        return Ok(w_tuple_new(vec![callable, w_tuple_new(vec![orig])]));
    }
    // `(type(self), (origin, args))`.
    let ga_type = crate::typedef::gettypeobject(&pyre_object::GENERIC_ALIAS_TYPE);
    Ok(w_tuple_new(vec![
        ga_type,
        w_tuple_new(vec![origin, ga_args]),
    ]))
}

/// `GenericAlias.__unpacked__` getset — the unpacked flag as a bool.
fn ga_get_unpacked(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = args.get(1).copied().unwrap_or_else(w_none);
    if !unsafe { is_generic_alias(self_) } {
        return Ok(w_bool_from(false));
    }
    Ok(w_bool_from(unsafe { w_generic_alias_get_unpacked(self_) }))
}

/// `GenericAlias.__typing_unpacked_tuple_args__` getset
/// (`_pypy_generic_alias.py:111`) — `args` when the alias is an unpacked
/// `tuple[...]`, else `None`.
fn ga_get_typing_unpacked_tuple_args(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = args.get(1).copied().unwrap_or_else(w_none);
    if !unsafe { is_generic_alias(self_) } {
        return Ok(w_none());
    }
    let unpacked = unsafe { w_generic_alias_get_unpacked(self_) };
    let origin = unsafe { w_generic_alias_get_origin(self_) };
    let tuple_type = crate::typedef::gettypeobject(&pyre_object::TUPLE_TYPE);
    if unpacked && std::ptr::eq(origin, tuple_type) {
        Ok(unsafe { w_generic_alias_get_args(self_) })
    } else {
        Ok(w_none())
    }
}

/// `GenericAlias.__dir__` (`_pypy_generic_alias.py:85`) —
/// `sorted(_ATTR_EXCEPTIONS | set(dir(origin)))`.  Invoked from
/// `builtins::builtin_dir` for a GenericAlias receiver.
pub(crate) fn dir_list(ga: PyObjectRef) -> crate::PyResult {
    let origin = unsafe { w_generic_alias_get_origin(ga) };
    let dir_origin = crate::builtins::builtin_dir(&[origin])?;
    let mut names: Vec<String> = ATTR_EXCEPTIONS.iter().map(|s| s.to_string()).collect();
    let n = unsafe { w_list_len(dir_origin) };
    for i in 0..n {
        if let Some(item) = unsafe { w_list_getitem(dir_origin, i as i64) } {
            if unsafe { is_str(item) } {
                names.push(unsafe { w_str_get_value(item) }.to_string());
            }
        }
    }
    names.sort();
    names.dedup();
    let items: Vec<PyObjectRef> = names.iter().map(|s| w_str_new(s)).collect();
    Ok(w_list_new(items))
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
    // `_create_union` (`_pypy_generic_alias.py:336`): `if self == other:
    // return self` — equality with identity short-circuit; a raising
    // `__eq__` propagates.
    if crate::baseobjspace::eq_w(x, y)? {
        return Ok(x);
    }
    // `UnionType((self, other))` — `__parameters__` is `_collect_parameters`
    // of the RAW operands (`_pypy_generic_alias.py:264`), computed once at
    // construction; `_args` is `add_recurse` over the operands
    // (`_pypy_generic_alias.py:253-262`): map `None` → `NoneType`, flatten
    // nested unions, and drop members already present by `==`.
    let parameters = collect_parameters(w_tuple_new(vec![x, y]))?;
    let mut members: Vec<PyObjectRef> = Vec::new();
    add_recurse(x, &mut members)?;
    add_recurse(y, &mut members)?;
    Ok(w_union_from_members(members, parameters))
}

/// `add_recurse(arg)` (`_pypy_generic_alias.py:253-262`) — the deduplicating
/// flatten body of `UnionType.__init__`.  `None` becomes `NoneType`, a nested
/// `UnionType` is spliced member-by-member, and a member is appended only when
/// not already present (`arg not in res`: `==` with identity short-circuit, a
/// raising `__eq__` propagating).
fn add_recurse(arg: PyObjectRef, res: &mut Vec<PyObjectRef>) -> Result<(), crate::PyError> {
    let arg = normalize_none(arg);
    if unsafe { pyre_object::is_union(arg) } {
        let inner = unsafe { pyre_object::w_union_get_args(arg) };
        let n = unsafe { pyre_object::w_tuple_len(inner) };
        for i in 0..n {
            if let Some(a) = unsafe { pyre_object::w_tuple_getitem(inner, i as i64) } {
                add_recurse(a, res)?;
            }
        }
        return Ok(());
    }
    for &existing in res.iter() {
        // `arg not in res` — `list.__contains__` compares `item == arg` with an
        // identity short-circuit (`PyObject_RichCompareBool`); a raising
        // `__eq__` propagates.
        if crate::baseobjspace::eq_w(existing, arg)? {
            return Ok(());
        }
    }
    res.push(arg);
    Ok(())
}

/// `UnionType.__eq__` (`_pypy_generic_alias.py:270-273`) —
/// `set(self.__args__) == set(other.__args__)`.  Both arg tuples are
/// deduplicated at construction, so equal length plus subset is set
/// equality.
pub(crate) fn union_set_eq(a: PyObjectRef, b: PyObjectRef) -> Result<bool, crate::PyError> {
    unsafe {
        let aa = w_union_get_args(a);
        let bb = w_union_get_args(b);
        let na = w_tuple_len(aa);
        let nb = w_tuple_len(bb);
        if na != nb {
            return Ok(false);
        }
        for i in 0..na {
            let Some(x) = w_tuple_getitem(aa, i as i64) else {
                return Ok(false);
            };
            let mut found = false;
            for j in 0..nb {
                if let Some(y) = w_tuple_getitem(bb, j as i64) {
                    if crate::baseobjspace::eq_w(x, y)? {
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                return Ok(false);
            }
        }
        Ok(true)
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
    make_generic_alias(args[1], args[2])
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
    dict_storage_store(
        ns,
        "__reduce__",
        make_builtin_function("__reduce__", ga_reduce),
    );
    dict_storage_store(
        ns,
        "__unpacked__",
        crate::typedef::make_getset_descriptor_named(
            make_builtin_function_with_arity("__unpacked__", ga_get_unpacked, 2),
            "__unpacked__",
        ),
    );
    dict_storage_store(
        ns,
        "__typing_unpacked_tuple_args__",
        crate::typedef::make_getset_descriptor_named(
            make_builtin_function_with_arity(
                "__typing_unpacked_tuple_args__",
                ga_get_typing_unpacked_tuple_args,
                2,
            ),
            "__typing_unpacked_tuple_args__",
        ),
    );
    // `_make_starred` (`_pypy_generic_alias.py:118`) — the module-level reduce
    // target.  pyre has no app-level `_pypy_generic_alias` Python module, so
    // the single shared callable lives on the type namespace; storing it here keeps it reachable
    // for the collector and gives `__reduce__` a stable callable identity.
    dict_storage_store(ns, "_make_starred", make_starred_fn());
    // `__iter__` and `__dir__` are intercepted directly by `baseobjspace::iter`
    // and `builtins::builtin_dir`; explicit `ga.__iter__`/`ga.__dir__` access
    // delegates to `__origin__` (they are not in `_ATTR_EXCEPTIONS`).
}

/// Render a GenericAlias for `repr()` (`GenericAlias.__repr__`,
/// `_pypy_generic_alias.py:57`).  Implemented here (not as a typedef
/// `__repr__`) so it matches the builtin-W_Root repr architecture, where
/// `display::py_repr` owns the rendering and explicit `.__repr__` access
/// still delegates to `__origin__`.
///
/// # Safety
/// `obj` must point to a valid `GenericAlias`.
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
