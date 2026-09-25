//! `_immutables_map` — PyPy's persistent HAMT map implementation.
//!
//! PyPy equivalent: `lib_pypy/_immutables_map.py`.  Keep the app-level
//! implementation intact: `_contextvars.Context` deliberately stores its
//! bindings in this persistent map rather than in a mutable dict.

crate::py_module! {
    "_immutables_map",
    appleveldefs: {
        "../../../../../lib_pypy/_immutables_map.py" => ["Map"],
    },
}
