//! _abc module — PyPy: `pypy/module/_abc/`.
//!
//! ABCMeta backing for `abc.py`.  `_abc_instancecheck` /
//! `_abc_subclasscheck` walk `__mro__` for direct inheritance and the
//! per-class `_abc_registry` list populated by `_abc_register` for
//! virtual subclasses.  Mirrors `pypy/module/_abc/app_abc.py`'s
//! `_abc_register` / `_abc_subclasscheck` flow (registry-based virtual
//! lookups, no negative cache).

use pyre_object::*;
use std::sync::atomic::{AtomicU64, Ordering};

// `abc_invalidation_counter` (`_abcmodule.c`): bumped by every successful
// `_abc_register` and by `_reset_caches`, and read by `get_cache_token`.
// The positive/negative object caches themselves remain omitted as an
// optimisation — only this token is tracked, so a bump makes any cached
// token stale.
static INVALIDATION_COUNTER: AtomicU64 = AtomicU64::new(0);

// `_py_abc.ABCMeta.__new__` (`_py_abc.py:48`) gives every ABC its OWN
// `_abc_registry`. Create it here as a per-class list so the registry is not
// inherited: without an own entry `register`/`subclass_of` would resolve
// `_abc_registry` up the MRO and share one base class's list across every
// descendant ABC (e.g. Complex/Real/Rational/Integral all collapsing to a
// single registry).
fn abc_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if let Some(&cls) = args.first() {
        let fresh = w_list_new(vec![]);
        crate::baseobjspace::setattr_str(cls, "_abc_registry", fresh)?;
        let mut abstract_names = Vec::new();
        let bases = unsafe { w_type_get_bases(cls) };
        if !bases.is_null() && unsafe { is_tuple(bases) } {
            for i in 0..unsafe { w_tuple_len(bases) } {
                let Some(base) = (unsafe { w_tuple_getitem(bases, i as i64) }) else {
                    continue;
                };
                // app_abc.py:67-71 — `getattr(..., set())` defaults only a
                // missing attribute.  A descriptor failure is observable and
                // any iterable, not just a set/frozenset, supplies names.
                let names = match crate::baseobjspace::getattr_str(base, "__abstractmethods__") {
                    Ok(names) => names,
                    Err(err) if err.kind == crate::PyErrorKind::AttributeError => w_set_new(),
                    Err(err) => return Err(err),
                };
                for name in crate::builtins::collect_iterable(names)? {
                    // `_py_abc.py:69` — object-level getattr validates that
                    // every supplied abstract-method name is a string, then
                    // lets descriptors and metaclass attributes provide an
                    // implementation.  Only a missing attribute defaults.
                    let value = match crate::baseobjspace::getattr(cls, name) {
                        Ok(value) => value,
                        Err(err) if err.kind == crate::PyErrorKind::AttributeError => w_none(),
                        Err(err) => return Err(err),
                    };
                    if crate::baseobjspace::isabstractmethod_w(value)? {
                        abstract_names.push(name);
                    }
                }
            }
        }
        let namespace = unsafe { w_type_get_dict_ptr(cls) as PyObjectRef };
        if !namespace.is_null() {
            for (name, value) in unsafe { w_dict_items(namespace) } {
                if crate::baseobjspace::isabstractmethod_w(value)? {
                    abstract_names.push(name);
                }
            }
        }
        let methods = w_frozenset_from_items(&abstract_names);
        crate::baseobjspace::setattr_str(cls, "__abstractmethods__", methods)?;
    }
    Ok(w_none())
}

// `_abc_register` (`_abcmodule.c:_abc__abc_register_impl`) —
// `cls._abc_registry.add(subclass)`.  Pyre stores the registry as a list
// attribute (no WeakSet).
fn register(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "_abc_register() requires (cls, subclass)",
        ));
    }
    let cls = args[0];
    let subclass = args[1];
    // `subclass` must be a class (`PyType_Check`).  Pyre's stdlib stubs
    // register callable non-type shells to ABCs — `_contextvars.Context` is a
    // builtin function here, yet `contextvars` runs `Mapping.register(Context)`
    // at import.  `__subclasscheck__` already tolerates such members by
    // skipping non-type registry entries (see `subclass_of`), so the
    // `PyObject_IsSubclass` guards below (which reject non-type args) only run
    // for real types; a callable stub falls straight through to the append,
    // and only a genuine non-class value (`register(42)`) is rejected.
    if unsafe { is_type(subclass) } {
        // Already a subclass (`PyObject_IsSubclass(subclass, cls) > 0`) —
        // nothing to register.  This also dedups: a previously registered
        // `subclass` resolves through `__subclasscheck__`'s registry walk.
        if crate::baseobjspace::issubclass(subclass, cls)? {
            return Ok(subclass);
        }
        // Registering `subclass` would also make `cls` its subclass.
        if crate::baseobjspace::issubclass(cls, subclass)? {
            return Err(crate::PyError::runtime_error(
                "Refusing to create an inheritance cycle",
            ));
        }
    } else if !crate::baseobjspace::callable_w(subclass) {
        return Err(crate::PyError::type_error("Can only register classes"));
    }
    let registry = match crate::baseobjspace::getattr_str(cls, "_abc_registry") {
        Ok(r) if !unsafe { is_none(r) } => r,
        _ => {
            let fresh = w_list_new(vec![]);
            crate::baseobjspace::setattr_str(cls, "_abc_registry", fresh)?;
            fresh
        }
    };
    unsafe {
        w_list_append(registry, subclass);
    }
    // Invalidate any outstanding cache token.
    INVALIDATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(subclass)
}

// `_py_abc.ABCMeta.__subclasscheck__` (`_py_abc.py:108-147`): the subclass
// hook first, then a direct `__mro__` test, then the recursive registry and
// subclass walks.  The positive/negative caches are a pure optimisation and
// are omitted; `issubclass` re-dispatches through `__subclasscheck__` so a
// registered or descendant ABC applies its own hook in turn.
fn subclass_of(cls: PyObjectRef, subclass: PyObjectRef) -> Result<bool, crate::PyError> {
    // _py_abc.py:110-111 — `if not isinstance(subclass, type): raise
    // TypeError('issubclass() arg 1 must be a class')`.  The `__mro__`/registry
    // walks below dereference `subclass` as a type, so a non-type argument
    // (`issubclass({}, ABC)`) must be rejected up front, not read as garbage.
    if !unsafe { is_type(subclass) } {
        return Err(crate::PyError::type_error(
            "issubclass() arg 1 must be a class",
        ));
    }
    // _py_abc.py:122-130 — `ok = cls.__subclasshook__(subclass)`.
    if let Ok(hook) = crate::baseobjspace::getattr_str(cls, "__subclasshook__") {
        if !hook.is_null() {
            let ok = crate::call::call_function_impl_result(hook, &[subclass])?;
            if !unsafe { is_not_implemented(ok) } {
                return crate::baseobjspace::is_true(ok);
            }
        }
    }
    // _py_abc.py:131-134 — direct subclass via `__mro__`.
    unsafe {
        let mro_ptr = w_type_get_mro(subclass);
        if !mro_ptr.is_null() {
            for &t in (*mro_ptr).as_slice() {
                if std::ptr::eq(t, cls) {
                    return Ok(true);
                }
            }
        }
    }
    // _py_abc.py:135-139 — subclass of a registered class (recursive).
    if let Ok(registry) = crate::baseobjspace::getattr_str(cls, "_abc_registry") {
        if !registry.is_null() && unsafe { is_list(registry) } {
            let n = unsafe { w_list_len(registry) };
            for i in 0..n {
                if let Some(rcls) = unsafe { w_list_getitem(registry, i as i64) } {
                    // A registered entry that is not a class cannot be a base
                    // class, so it can never make `subclass` a subclass — skip
                    // it rather than letting `issubclass` raise.  `range` is
                    // registered to `Sequence` but is a builtin function in
                    // pyre, so without this guard a single bad entry aborts the
                    // whole recursive check.
                    if !unsafe { is_type(rcls) } {
                        continue;
                    }
                    if crate::baseobjspace::issubclass(subclass, rcls)? {
                        return Ok(true);
                    }
                }
            }
        }
    }
    // _py_abc.py:140-144 — subclass of a subclass (recursive).
    for scls in unsafe { w_type_get_subclasses(cls) } {
        if crate::baseobjspace::issubclass(subclass, scls)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn instancecheck(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(w_bool_from(false));
    }
    let cls = args[0];
    let instance = args[1];
    if unsafe { crate::baseobjspace::isinstance_w(instance, cls) } {
        return Ok(w_bool_from(true));
    }
    // `type(instance)` — the instance's real class.  User-defined instances
    // carry the generic layout marker in `ob_type` and the real class in
    // `w_class`, so reading `ob_type` directly would resolve to `object`;
    // `r#type` returns the class for both builtin and user instances.
    let subclass = crate::typedef::r#type(instance).unwrap_or(std::ptr::null_mut());
    if subclass.is_null() {
        return Ok(w_bool_from(false));
    }
    Ok(w_bool_from(subclass_of(cls, subclass)?))
}

fn subclasscheck(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(w_bool_from(false));
    }
    Ok(w_bool_from(subclass_of(args[0], args[1])?))
}

crate::py_module! {
    "_abc",
    functions: {
        "get_cache_token"     / 0 = |_| Ok(w_int_new(INVALIDATION_COUNTER.load(Ordering::Relaxed) as i64)),
        "_abc_init"           / 1 = abc_init,
        "_abc_register"       / 2 = register,
        "_abc_instancecheck"  / 2 = instancecheck,
        "_abc_subclasscheck"  / 2 = subclasscheck,
        "_get_dump"           / 1 = |_| Ok(w_tuple_new(vec![])),
        "_reset_registry"     / 1 = |_| Ok(w_none()),
        // Pyre keeps no object caches to clear; bumping the token invalidates
        // any outstanding `get_cache_token` value.
        "_reset_caches"       / 1 = |_| { INVALIDATION_COUNTER.fetch_add(1, Ordering::Relaxed); Ok(w_none()) },
    },
}
