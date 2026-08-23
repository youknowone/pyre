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
}
