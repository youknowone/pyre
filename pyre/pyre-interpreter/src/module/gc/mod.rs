//! gc module — PyPy: `pypy/module/gc/`.
//!
//! Partial port of `interp_gc.py`.  `collect` is currently a no-op because
//! collection at arbitrary interpreter depth needs shadowstack coverage for
//! Rust-stack-only `PyObjectRef`s; `enable` / `disable` / `isenabled` accept
//! calls but pyre has no generational threshold knob; `get_referrers` /
//! `get_referents` return empty lists; the DEBUG_* constants are stubbed.

use pyre_object::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// `interp_gc.py` tracks a process-wide `enabled` flag on the GC
/// frontend; pyre has no generational threshold knob, but
/// `gc.isenabled()` should reflect the most recent `enable`/`disable`
/// call so callers that toggle and re-read the state stay consistent.
static GC_ENABLED: AtomicBool = AtomicBool::new(true);

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
        "collect"       / 1 = |_| {
            // A real collection here needs a shadowstack pass over interpreter
            // Rust-stack `PyObjectRef`s; see the full-collect trampoline doc in
            // `pyre-jit/src/eval.rs`. Even the non-moving old-gen major is
            // unsound at arbitrary interpreter depth because marking cannot see
            // Rust-stack-only references. Until that shadowstack pass exists,
            // `collect` is a no-op and collections happen only at the JIT's own
            // safepoints / the gated interpreter safepoint
            // (`pyre-object/src/gc_interp.rs`).
            Ok(w_int_new(0))
        },
        "disable"       / 0 = |_| { GC_ENABLED.store(false, Ordering::Relaxed); Ok(w_none()) },
        "enable"        / 0 = |_| { GC_ENABLED.store(true, Ordering::Relaxed); Ok(w_none()) },
        "isenabled"     / 0 = |_| Ok(w_bool_from(GC_ENABLED.load(Ordering::Relaxed))),
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
