//! _blake2 module — hashlib blocks OpenSSL's blake2 and uses this builtin
//! instead.  The constructors defer to the `_hashlib` HASH object, which
//! computes blake2b-512 / blake2s-256 through RustCrypto.

crate::py_module! {
    "_blake2",
    int_constants: {
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
