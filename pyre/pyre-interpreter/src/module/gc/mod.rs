//! gc module — PyPy: `pypy/module/gc/`.
//!
//! Partial port of `interp_gc.py`.  `collect` drives a full mark-sweep
//! through the active GC (`majit_gc::collect_full` via `try_gc_collect`);
//! `enable` / `disable` / `isenabled` accept calls but pyre has no
//! generational threshold knob; `get_referrers` / `get_referents` return
//! empty lists; the DEBUG_* constants match CPython values.

use pyre_object::*;

crate::py_module! {
    "gc",
    interpleveldefs: {
        "callbacks"           => w_list_new(vec![]),
        "garbage"             => w_list_new(vec![]),
        "DEBUG_STATS"         => w_int_new(1),
        "DEBUG_COLLECTABLE"   => w_int_new(2),
        "DEBUG_UNCOLLECTABLE" => w_int_new(4),
        "DEBUG_SAVEALL"       => w_int_new(32),
        "DEBUG_LEAK"          => w_int_new(38),
    },
    functions: {
        // `interp_gc.py:7-26 collect` — argument `generation` ignored per
        // upstream.  MethodCache / MapAttrCache clears (`:14-17`) skipped
        // because pyre has no equivalent caches.
        "collect"       / 1 = |_| { gc_hook::try_gc_collect(); Ok(w_int_new(0)) },
        "disable"       / 0 = |_| Ok(w_none()),
        "enable"        / 0 = |_| Ok(w_none()),
        "isenabled"     / 0 = |_| Ok(w_bool_from(false)),
        "get_objects"   / 1 = |_| Ok(w_list_new(vec![])),
        "get_referrers" / * = |_| Ok(w_list_new(vec![])),
        "get_referents" / * = |_| Ok(w_list_new(vec![])),
        "set_threshold" / 0 = |_| Ok(w_none()),
        "get_threshold" / 0 = |_| Ok(w_tuple_new(vec![
            w_int_new(700), w_int_new(10), w_int_new(10),
        ])),
        "get_count"     / 0 = |_| Ok(w_tuple_new(vec![
            w_int_new(0), w_int_new(0), w_int_new(0),
        ])),
        "is_tracked"    / 1 = |_| Ok(w_bool_from(false)),
        "is_finalized"  / 1 = |_| Ok(w_bool_from(false)),
        "freeze"        / 0 = |_| Ok(w_none()),
    },
}
