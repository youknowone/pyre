//! gc implementation — PyPy: pypy/module/gc/interp_gc.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// gc module stub — enough to let `import gc` succeed.
pub fn register_module(ns: &mut DictStorage) {
    // pypy/module/gc/interp_gc.py:7-26 collect — partial port:
    // drive a full mark-sweep through `try_gc_collect` (which fans
    // out through `pyre-jit::eval`'s trampoline to the active
    // backend's `majit_gc::collect_full`). MethodCache / MapAttrCache
    // clears (`:14-17`) skipped — pyre has no equivalent caches.
    // Finalizer queue (`:28-46 _run_finalizers`) skipped pending the
    // finalizer epic. Argument `generation` is ignored per upstream.
    crate::dict_storage_store(
        ns,
        "collect",
        crate::make_builtin_function_with_arity(
            "collect",
            |_| {
                pyre_object::gc_hook::try_gc_collect();
                Ok(pyre_object::w_int_new(0))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "disable",
        crate::make_builtin_function_with_arity("disable", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "enable",
        crate::make_builtin_function_with_arity("enable", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "isenabled",
        crate::make_builtin_function_with_arity(
            "isenabled",
            |_| Ok(pyre_object::w_bool_from(false)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_objects",
        crate::make_builtin_function_with_arity(
            "get_objects",
            |_| Ok(pyre_object::w_list_new(vec![])),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_referrers",
        crate::make_builtin_function("get_referrers", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    crate::dict_storage_store(
        ns,
        "get_referents",
        crate::make_builtin_function("get_referents", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    crate::dict_storage_store(
        ns,
        "set_threshold",
        crate::make_builtin_function_with_arity("set_threshold", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "get_threshold",
        crate::make_builtin_function_with_arity(
            "get_threshold",
            |_| {
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(700),
                    pyre_object::w_int_new(10),
                    pyre_object::w_int_new(10),
                ]))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_count",
        crate::make_builtin_function_with_arity(
            "get_count",
            |_| {
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(0),
                    pyre_object::w_int_new(0),
                    pyre_object::w_int_new(0),
                ]))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "is_tracked",
        crate::make_builtin_function_with_arity(
            "is_tracked",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "is_finalized",
        crate::make_builtin_function_with_arity(
            "is_finalized",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "freeze",
        crate::make_builtin_function_with_arity("freeze", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(ns, "callbacks", pyre_object::w_list_new(vec![]));
    crate::dict_storage_store(ns, "garbage", pyre_object::w_list_new(vec![]));
    crate::dict_storage_store(ns, "DEBUG_STATS", pyre_object::w_int_new(1));
    crate::dict_storage_store(ns, "DEBUG_COLLECTABLE", pyre_object::w_int_new(2));
    crate::dict_storage_store(ns, "DEBUG_UNCOLLECTABLE", pyre_object::w_int_new(4));
    crate::dict_storage_store(ns, "DEBUG_SAVEALL", pyre_object::w_int_new(32));
    crate::dict_storage_store(ns, "DEBUG_LEAK", pyre_object::w_int_new(38));
}
