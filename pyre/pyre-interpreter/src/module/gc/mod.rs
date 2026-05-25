//! gc module — PyPy: `pypy/module/gc/`.
//!
//! Partial port of `interp_gc.py`.  `collect` drives a full mark-sweep
//! through the active GC (`majit_gc::collect_full` via `try_gc_collect`);
//! `enable` / `disable` / `isenabled` accept calls but pyre has no
//! generational threshold knob; `get_referrers` / `get_referents` return
//! empty lists; the DEBUG_* constants match CPython values.

crate::py_module! {
    "gc",
    interpleveldefs: {
        // `interp_gc.py:7-26 collect` — argument `generation` ignored
        // per upstream.  MethodCache / MapAttrCache clears (`:14-17`)
        // skipped because pyre has no equivalent caches.
        "collect" => crate::make_builtin_function_with_arity(
            "collect",
            |_| {
                pyre_object::gc_hook::try_gc_collect();
                Ok(pyre_object::w_int_new(0))
            },
            1,
        ),
        "disable" => crate::make_builtin_function_with_arity(
            "disable", |_| Ok(pyre_object::w_none()), 0),
        "enable" => crate::make_builtin_function_with_arity(
            "enable", |_| Ok(pyre_object::w_none()), 0),
        "isenabled" => crate::make_builtin_function_with_arity(
            "isenabled", |_| Ok(pyre_object::w_bool_from(false)), 0),
        "get_objects" => crate::make_builtin_function_with_arity(
            "get_objects", |_| Ok(pyre_object::w_list_new(vec![])), 1),
        "get_referrers" => crate::make_builtin_function(
            "get_referrers", |_| Ok(pyre_object::w_list_new(vec![]))),
        "get_referents" => crate::make_builtin_function(
            "get_referents", |_| Ok(pyre_object::w_list_new(vec![]))),
        "set_threshold" => crate::make_builtin_function_with_arity(
            "set_threshold", |_| Ok(pyre_object::w_none()), 0),
        "get_threshold" => crate::make_builtin_function_with_arity(
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
        "get_count" => crate::make_builtin_function_with_arity(
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
        "is_tracked" => crate::make_builtin_function_with_arity(
            "is_tracked", |_| Ok(pyre_object::w_bool_from(false)), 1),
        "is_finalized" => crate::make_builtin_function_with_arity(
            "is_finalized", |_| Ok(pyre_object::w_bool_from(false)), 1),
        "freeze" => crate::make_builtin_function_with_arity(
            "freeze", |_| Ok(pyre_object::w_none()), 0),
        "callbacks" => pyre_object::w_list_new(vec![]),
        "garbage" => pyre_object::w_list_new(vec![]),
        "DEBUG_STATS" => pyre_object::w_int_new(1),
        "DEBUG_COLLECTABLE" => pyre_object::w_int_new(2),
        "DEBUG_UNCOLLECTABLE" => pyre_object::w_int_new(4),
        "DEBUG_SAVEALL" => pyre_object::w_int_new(32),
        "DEBUG_LEAK" => pyre_object::w_int_new(38),
    }
}
