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
    let registry = match crate::baseobjspace::getattr(cls, "_abc_registry") {
        Ok(r) if !unsafe { is_none(r) } => r,
        _ => {
            let fresh = w_list_new(vec![]);
            crate::baseobjspace::setattr(cls, "_abc_registry", fresh)?;
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

// `app_abc.py:_abc_subclasscheck` — direct MRO walk plus registry-based
// virtual-subclass lookup (recursive `issubclass(subclass, rcls)`).
fn subclass_of(cls: PyObjectRef, subclass: PyObjectRef) -> bool {
    unsafe {
        let mro_ptr = w_type_get_mro(subclass);
        if !mro_ptr.is_null() {
            for &t in &*mro_ptr {
                if std::ptr::eq(t, cls) {
                    return true;
                }
            }
        }
    }
    if let Ok(registry) = crate::baseobjspace::getattr(cls, "_abc_registry") {
        if !registry.is_null() && unsafe { is_list(registry) } {
            unsafe {
                let n = w_list_len(registry);
                for i in 0..n {
                    if let Some(rcls) = w_list_getitem(registry, i as i64) {
                        if subclass_of(rcls, subclass) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
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
    let subclass =
        unsafe { crate::typedef::gettypefor((*instance).ob_type).unwrap_or(std::ptr::null_mut()) };
    Ok(w_bool_from(
        !subclass.is_null() && subclass_of(cls, subclass),
    ))
}

fn subclasscheck(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(w_bool_from(false));
    }
    Ok(w_bool_from(subclass_of(args[0], args[1])))
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
