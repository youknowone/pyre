//! `__pypy__` module — PyPy: pypy/module/__pypy__/
//!
//! Pyre exposes the small slice of the `__pypy__` surface that the
//! PyPy-flavored stdlib needs.  `pickle.py` imports `identity_dict`
//! (an identity-keyed memo dict) and `builders.BytesBuilder` in one
//! shared `try` block; both must resolve for the optimized path to
//! activate, so both are provided here as app-level classes.
//!
//! `PickleBuffer` (`interp_buffer.py W_PickleBuffer`) is exposed here as
//! an interp-level class; `pickle.py` re-exports it and the `_pickle`
//! accelerator serializes it in-band or out-of-band under protocol 5.

pub mod interp_buffer;

pub use interp_buffer::W_PickleBuffer;

crate::py_module! {
    "__pypy__",
    // `PickleBuffer` wraps a bytes-like object for proto-5 out-of-band
    // buffers; `identity_dict` keys a memo by object identity (id(key))
    // so the Pickler can memoize unhashable containers.
    interpleveldefs: {
        "PickleBuffer" => interp_buffer::type_object(),
    },
    appleveldefs: {
        "identity_dict_app.py" => ["identity_dict"],
    },
    extra_init: |ns| {
        // Mark as a package so `from __pypy__.builders import ...`
        // treats `__pypy__` as a package with submodules.
        crate::module_ns_store(ns, "__path__", pyre_object::w_list_new(vec![]));
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
