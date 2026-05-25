//! _abc module — PyPy: `pypy/module/_abc/`.
//!
//! ABCMeta backing for `abc.py`.  `_abc_instancecheck` /
//! `_abc_subclasscheck` walk `__mro__` to honour direct subclassing;
//! virtual-subclass registration via `cls.register(subclass)` is not
//! tracked (pyre's `_abc_register` is a no-op), so `isinstance` of a
//! virtual subclass falls back to `False`.

use pyre_object::*;

// `Modules/_abc.c _abc__abc_instancecheck` — delegate to
// baseobjspace::isinstance_w so `isinstance(Fraction(1,2),
// numbers.Rational)` works via direct MRO inheritance.
fn instancecheck(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(w_bool_from(false));
    }
    let cls = args[0];
    let instance = args[1];
    unsafe {
        Ok(w_bool_from(crate::baseobjspace::isinstance_w(instance, cls)))
    }
}

fn subclasscheck(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(w_bool_from(false));
    }
    let cls = args[0];
    let subclass = args[1];
    unsafe {
        let mro_ptr = w_type_get_mro(subclass);
        if !mro_ptr.is_null() {
            for &t in &*mro_ptr {
                if std::ptr::eq(t, cls) {
                    return Ok(w_bool_from(true));
                }
            }
        }
    }
    Ok(w_bool_from(false))
}

crate::py_module! {
    "_abc",
    functions: {
        "get_cache_token"     / 0 = |_| Ok(w_int_new(0)),
        "_abc_init"           / 1 = |_| Ok(w_none()),
        "_abc_register"       / 2 = |_| Ok(w_none()),
        "_abc_instancecheck"  / 2 = instancecheck,
        "_abc_subclasscheck"  / 2 = subclasscheck,
        "_get_dump"           / 1 = |_| Ok(w_tuple_new(vec![])),
        "_reset_registry"     / 1 = |_| Ok(w_none()),
        "_reset_caches"       / 1 = |_| Ok(w_none()),
    },
}
