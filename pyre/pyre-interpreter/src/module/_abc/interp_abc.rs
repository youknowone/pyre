//! _abc implementation — PyPy: pypy/module/_abc/interp_abc.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// _abc stub — PyPy: pypy/module/_abc/
pub fn register_module(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "get_cache_token",
        crate::make_builtin_function_with_arity(
            "get_cache_token",
            |_| Ok(pyre_object::w_int_new(0)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_abc_init",
        crate::make_builtin_function_with_arity("_abc_init", |_| Ok(pyre_object::w_none()), 1),
    );
    crate::dict_storage_store(
        ns,
        "_abc_register",
        crate::make_builtin_function_with_arity("_abc_register", |_| Ok(pyre_object::w_none()), 2),
    );
    // _abc_instancecheck(cls, instance) — CPython: Modules/_abc.c _abc__abc_instancecheck.
    //
    // ABCMeta.__instancecheck__ (abc.py:119) delegates here. The canonical
    // behaviour: walk type(instance).__mro__ looking for cls (direct
    // subclass), then consult cls._abc_registry for virtual subclasses
    // registered via `cls.register(subclass)`. Our previous stub
    // unconditionally returned False, which broke
    // `isinstance(Fraction(1,2), numbers.Rational)`.
    crate::dict_storage_store(
        ns,
        "_abc_instancecheck",
        crate::make_builtin_function_with_arity(
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
    );
    // _abc_subclasscheck(cls, subclass) — CPython: Modules/_abc.c _abc__abc_subclasscheck.
    crate::dict_storage_store(
        ns,
        "_abc_subclasscheck",
        crate::make_builtin_function_with_arity(
            "_abc_subclasscheck",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let cls = args[0];
                let subclass = args[1];
                unsafe {
                    // Walk subclass.__mro__ looking for cls.
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
    );
    crate::dict_storage_store(
        ns,
        "_get_dump",
        crate::make_builtin_function_with_arity(
            "_get_dump",
            |_| Ok(pyre_object::w_tuple_new(vec![])),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_reset_registry",
        crate::make_builtin_function_with_arity(
            "_reset_registry",
            |_| Ok(pyre_object::w_none()),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_reset_caches",
        crate::make_builtin_function_with_arity("_reset_caches", |_| Ok(pyre_object::w_none()), 1),
    );
}
