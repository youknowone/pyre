//! _abc module — PyPy: `pypy/module/_abc/`.
//!
//! ABCMeta backing for `abc.py`.  `_abc_instancecheck` /
//! `_abc_subclasscheck` walk `__mro__` to honour direct subclassing;
//! virtual-subclass registration via `cls.register(subclass)` is not
//! tracked (pyre's `_abc_register` is a no-op), so `isinstance` of a
//! virtual subclass falls back to `False`.

crate::py_module! {
    "_abc",
    interpleveldefs: {
        "get_cache_token" => crate::make_builtin_function_with_arity(
            "get_cache_token", |_| Ok(pyre_object::w_int_new(0)), 0),
        "_abc_init" => crate::make_builtin_function_with_arity(
            "_abc_init", |_| Ok(pyre_object::w_none()), 1),
        "_abc_register" => crate::make_builtin_function_with_arity(
            "_abc_register", |_| Ok(pyre_object::w_none()), 2),
        // `Modules/_abc.c _abc__abc_instancecheck` — delegate to
        // baseobjspace::isinstance_w so `isinstance(Fraction(1,2),
        // numbers.Rational)` works via direct MRO inheritance.
        "_abc_instancecheck" => crate::make_builtin_function_with_arity(
            "_abc_instancecheck",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let cls = args[0];
                let instance = args[1];
                unsafe {
                    Ok(pyre_object::w_bool_from(crate::baseobjspace::isinstance_w(
                        instance, cls,
                    )))
                }
            },
            2,
        ),
        "_abc_subclasscheck" => crate::make_builtin_function_with_arity(
            "_abc_subclasscheck",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let cls = args[0];
                let subclass = args[1];
                unsafe {
                    let mro_ptr = pyre_object::w_type_get_mro(subclass);
                    if !mro_ptr.is_null() {
                        for &t in &*mro_ptr {
                            if std::ptr::eq(t, cls) {
                                return Ok(pyre_object::w_bool_from(true));
                            }
                        }
                    }
                }
                Ok(pyre_object::w_bool_from(false))
            },
            2,
        ),
        "_get_dump" => crate::make_builtin_function_with_arity(
            "_get_dump", |_| Ok(pyre_object::w_tuple_new(vec![])), 1),
        "_reset_registry" => crate::make_builtin_function_with_arity(
            "_reset_registry", |_| Ok(pyre_object::w_none()), 1),
        "_reset_caches" => crate::make_builtin_function_with_arity(
            "_reset_caches", |_| Ok(pyre_object::w_none()), 1),
    }
}
