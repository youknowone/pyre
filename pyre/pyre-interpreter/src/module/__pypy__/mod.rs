//! `__pypy__` module — PyPy: pypy/module/__pypy__/
//!
//! Pyre exposes the small slice of the `__pypy__` surface that the
//! PyPy-flavored stdlib needs.  `pickle.py` imports `identity_dict`
//! (an identity-keyed memo dict) and `builders.BytesBuilder` in one
//! shared `try` block; both must resolve for the optimized path to
//! activate, so both are provided here as app-level classes.

crate::py_module! {
    "__pypy__",
    // `identity_dict` keys a memo by object identity (id(key)) so the
    // Pickler can memoize unhashable containers.
    appleveldefs: {
        "identity_dict_app.py" => ["identity_dict"],
    },
    extra_init: |ns| {
        // Mark as a package so `from __pypy__.builders import ...`
        // treats `__pypy__` as a package with submodules.
        crate::dict_storage_store(ns, "__path__", pyre_object::w_list_new(vec![]));
    }
}

/// `__pypy__.builders` submodule — exposes the string/bytes builders.
pub mod builders {
    crate::py_module! {
        "__pypy__.builders",
        // BytesBuilder is the append-only byte buffer pickle.py writes
        // frames into; StringBuilder is its text analogue.
        appleveldefs: {
            "builders_app.py" => ["BytesBuilder", "StringBuilder"],
        }
    }
}
