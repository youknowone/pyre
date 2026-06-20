//! _abc module — PyPy: `pypy/module/_abc/`.
//!
//! ABCMeta backing for `abc.py`.  `_abc_instancecheck` /
//! `_abc_subclasscheck` walk `__mro__` for direct inheritance and the
//! per-class `_abc_registry` list populated by `_abc_register` for
//! virtual subclasses.  Mirrors `pypy/module/_abc/app_abc.py`'s
//! `_abc_register` / `_abc_subclasscheck` flow (registry-based virtual
//! lookups, no negative cache).

use pyre_object::*;

// `app_abc.py:_abc_register` — `cls._abc_registry.add(subclass)`.
// Pyre stores the registry as a list attribute (no WeakSet); duplicates
// are skipped to keep the list bounded.
fn register(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "_abc_register() requires (cls, subclass)",
        ));
    }
    let cls = args[0];
    let subclass = args[1];
    let registry = match crate::baseobjspace::getattr_str(cls, "_abc_registry") {
        Ok(r) if !unsafe { is_none(r) } => r,
        _ => {
            let fresh = w_list_new(vec![]);
            crate::baseobjspace::setattr_str(cls, "_abc_registry", fresh)?;
            fresh
        }
    };
    unsafe {
        let n = w_list_len(registry);
        for i in 0..n {
            if let Some(item) = w_list_getitem(registry, i as i64) {
                if std::ptr::eq(item, subclass) {
                    return Ok(subclass);
                }
            }
        }
        w_list_append(registry, subclass);
    }
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
            for &t in &*mro_ptr {
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
        "get_cache_token"     / 0 = |_| Ok(w_int_new(0)),
        "_abc_init"           / 1 = |_| Ok(w_none()),
        "_abc_register"       / 2 = register,
        "_abc_instancecheck"  / 2 = instancecheck,
        "_abc_subclasscheck"  / 2 = subclasscheck,
        "_get_dump"           / 1 = |_| Ok(w_tuple_new(vec![])),
        "_reset_registry"     / 1 = |_| Ok(w_none()),
        "_reset_caches"       / 1 = |_| Ok(w_none()),
    },
}
