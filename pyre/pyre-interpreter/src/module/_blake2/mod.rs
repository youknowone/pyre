//! `_blake2` module constants and PyPy-shaped app-level object types.
//!
//! The wrappers mirror `lib_pypy/_blake2/__init__.py`; their object-owned
//! native RFC 7693 contexts are created by `_hashlib._blake2_new`.

crate::py_module! {
    "_blake2",
    int_constants: {
        "_GIL_MINSIZE" => 2048,
        "BLAKE2B_SALT_SIZE" => 16,
        "BLAKE2B_PERSON_SIZE" => 16,
        "BLAKE2B_MAX_KEY_SIZE" => 64,
        "BLAKE2B_MAX_DIGEST_SIZE" => 64,
        "BLAKE2S_SALT_SIZE" => 8,
        "BLAKE2S_PERSON_SIZE" => 8,
        "BLAKE2S_MAX_KEY_SIZE" => 32,
        "BLAKE2S_MAX_DIGEST_SIZE" => 32,
    },
    appleveldefs: {
        "_blake2_app.py" => ["blake2b", "blake2s"],
    },
    extra_init: |ns| {
        // [3.14-spec] Keep PyPy's app-level `_make_blake_type` classes and
        // their object-owned state, but match CPython 3.14's immutable
        // non-BASETYPE public types at the subclassing boundary.
        for name in ["blake2b", "blake2s"] {
            let ty = crate::module_ns_get(ns, name).expect("_blake2 app-level type installed");
            crate::typedef::mark_cpython_heap_type(ty, true);
            unsafe { pyre_object::w_type_suppress_cpython_basetype(ty) };
        }
    },
}
