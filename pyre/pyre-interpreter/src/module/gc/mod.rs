//! gc module — PyPy: `pypy/module/gc/`.
//!
//! Partial port of `interp_gc.py`.  `collect` drives a full mark-sweep
//! through the active GC (`majit_gc::collect_full` via `try_gc_collect`);
//! `enable` / `disable` / `isenabled` accept calls but pyre has no
//! generational threshold knob; `get_referrers` / `get_referents` return
//! empty lists; the DEBUG_* constants match CPython values.

use pyre_object::*;

// `interp_gc.py:7-26 collect` — argument `generation` ignored per
// upstream.  MethodCache / MapAttrCache clears (`:14-17`) skipped
// because pyre has no equivalent caches.
fn gc_collect(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    gc_hook::try_gc_collect();
    Ok(w_int_new(0))
}

fn gc_none(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_none())
}

fn gc_false(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_bool_from(false))
}

fn gc_empty_list(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_list_new(vec![]))
}

fn gc_get_threshold(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_tuple_new(vec![
        w_int_new(700),
        w_int_new(10),
        w_int_new(10),
    ]))
}

fn gc_get_count(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_tuple_new(vec![w_int_new(0), w_int_new(0), w_int_new(0)]))
}

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
        "collect"       / 1 = gc_collect,
        "disable"       / 0 = gc_none,
        "enable"        / 0 = gc_none,
        "isenabled"     / 0 = gc_false,
        "get_objects"   / 1 = gc_empty_list,
        "get_referrers" / * = gc_empty_list,
        "get_referents" / * = gc_empty_list,
        "set_threshold" / 0 = gc_none,
        "get_threshold" / 0 = gc_get_threshold,
        "get_count"     / 0 = gc_get_count,
        "is_tracked"    / 1 = gc_false,
        "is_finalized"  / 1 = gc_false,
        "freeze"        / 0 = gc_none,
    },
}
