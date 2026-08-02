//! _hashlib module — the OpenSSL-backed digest surface `hashlib.py` probes.
//!
//! The actual digests are computed by `pyre-native` through
//! [`oneshot_digest`]; the `HASH` object lives in the app-level
//! `_hashlib_app.py`, while the non-binding `openssl_<name>` and `new`
//! constructors are interp-level here and build a `HASH`, which calls back
//! into `_oneshot_digest` at digest time.  Accumulating the data and
//! re-hashing on `digest()` keeps the object model trivial at the cost of
//! recomputation.

use pyre_object::*;

/// The 14 always-supported digests hashlib advertises, each with the OpenSSL
/// names that also select it.
///
/// `py_digest_by_name` consults two tables in turn: the Python names, matched
/// exactly, then the OpenSSL names, matched without regard to case.  That
/// asymmetry decides which spellings resolve — `sha3_256` does and `SHA3_256`
/// does not, because the OpenSSL spelling is `SHA3-256`; `blake2b` does and
/// `Blake2b` does not, because the OpenSSL spelling is `BLAKE2B512`.
///
/// The digest an object computes is named by the entry it resolved to, not by
/// the spelling the caller used, so `new('sha-256').name` is `sha256`.
const DIGEST_NAMES: &[(&str, &[&str])] = &[
    ("md5", &["MD5", "SSL3-MD5"]),
    ("sha1", &["SHA1", "SSL3-SHA1"]),
    ("sha224", &["SHA224", "SHA2-224", "SHA-224"]),
    ("sha256", &["SHA256", "SHA2-256", "SHA-256"]),
    ("sha384", &["SHA384", "SHA2-384", "SHA-384"]),
    ("sha512", &["SHA512", "SHA2-512", "SHA-512"]),
    ("sha3_224", &["SHA3-224"]),
    ("sha3_256", &["SHA3-256"]),
    ("sha3_384", &["SHA3-384"]),
    ("sha3_512", &["SHA3-512"]),
    ("shake_128", &["SHAKE128", "SHAKE-128"]),
    ("shake_256", &["SHAKE256", "SHAKE-256"]),
    ("blake2b", &["BLAKE2B512"]),
    ("blake2s", &["BLAKE2S256"]),
];

/// The entry `name` selects, or `None` when it names no digest this build
/// computes.  Numeric object identifiers are not accepted: resolving those
/// needs the OpenSSL object database, which pyre does not carry.
fn lookup_digest_name(name: &[u8]) -> Option<&'static str> {
    if let Some((python_name, _)) = DIGEST_NAMES
        .iter()
        .find(|(python_name, _)| python_name.as_bytes() == name)
    {
        return Some(python_name);
    }
    DIGEST_NAMES
        .iter()
        .find(|(_, ossl_names)| {
            ossl_names
                .iter()
                .any(|ossl| ossl.as_bytes().eq_ignore_ascii_case(name))
        })
        .map(|(python_name, _)| *python_name)
}

/// `_oneshot_digest(name, data, length=0)` — bytes of the digest.
fn oneshot_digest(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let name_obj = args.first().copied().unwrap_or_else(w_none);
    if !unsafe { is_str(name_obj) } {
        return Err(crate::PyError::type_error("digest name must be a string"));
    }
    let name = crate::baseobjspace::str_utf8_w(name_obj)?.to_string();
    let data = match args.get(1).copied() {
        Some(obj) if unsafe { bytesobject::is_bytes_like(obj) } => {
            unsafe { bytesobject::bytes_like_data(obj) }.to_vec()
        }
        Some(obj) if unsafe { is_none(obj) } => Vec::new(),
        None => Vec::new(),
        _ => return Err(crate::PyError::type_error("data must be bytes-like")),
    };
    let length = args
        .get(2)
        .map(|&o| unsafe { w_int_get_value(o) } as usize)
        .unwrap_or(0);
    match pyre_native::hash::compute_digest(&name, &data, length) {
        Some(out) => Ok(w_bytes_from_bytes(&out)),
        None => Err(crate::PyError::value_error(format!(
            "unsupported hash type {name}"
        ))),
    }
}

/// The `name: str` clinic conversion: `name` must be a str (or subclass, per
/// `PyUnicode_Check`), and its utf-8 form must exist — a lone surrogate raises
/// `UnicodeEncodeError`, an embedded null raises `ValueError`.  This runs as
/// part of the argument parse, before the constructor body, so `new` validates
/// the name ahead of the `data`/`string` and buffer handling.
fn check_digest_name(name_obj: PyObjectRef) -> Result<(), crate::PyError> {
    if !unsafe { crate::baseobjspace::isinstance_str_w(name_obj) } {
        return Err(crate::PyError::type_error(format!(
            "new() argument 'name' must be str, not {}",
            crate::type_methods::arg_type_name(name_obj)
        )));
    }
    // Report the failures the utf-8 conversion of the name would: a lone
    // surrogate does not encode, and an embedded null ends the C string the
    // digest is looked up by.
    let name = unsafe { w_str_get_wtf8(name_obj) };
    if let Some(pos) = name
        .code_points()
        .position(|cp| (0xD800..=0xDFFF).contains(&cp.to_u32()))
    {
        return Err(crate::typedef::unicode_encode_error(
            "utf-8",
            name_obj,
            pos,
            pos + 1,
            "surrogates not allowed",
        ));
    }
    if name.as_bytes().contains(&0) {
        return Err(crate::PyError::value_error("embedded null character"));
    }
    Ok(())
}

/// `HASH(name)` name resolution — the Python name of the digest `name`
/// selects.  OpenSSL resolves the digest when the context is created, so an
/// unknown name is reported there rather than when the digest is finally
/// taken.
fn resolve_digest_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let name_obj = args.first().copied().unwrap_or_else(w_none);
    check_digest_name(name_obj)?;
    let name = unsafe { w_str_get_wtf8(name_obj) };
    match lookup_digest_name(name.as_bytes()) {
        Some(python_name) => Ok(w_str_new(python_name)),
        None => Err(unsupported_digestmod(&format!(
            "unsupported hash type {name}"
        ))),
    }
}

/// Build a `PyError` raising `_hashlib.UnsupportedDigestmodError` with `msg`.
/// `hmac.py` catches this to fall back to its pure-Python HMAC, so the OpenSSL
/// HMAC entry points always raise it: we have no streaming HMAC primitive.
fn unsupported_digestmod(msg: &str) -> crate::PyError {
    let mut err = crate::PyError::value_error(msg.to_string());
    if let Some(cls) = crate::builtins::lookup_exc_class("_hashlib.UnsupportedDigestmodError") {
        let args = [cls, w_str_new(msg)];
        if let Ok(exc) = crate::builtins::exc_exception_new(&args) {
            err.exc_object = exc;
        }
    }
    err
}

/// `hmac_new(key, msg=b'', *, digestmod)` — always declines so `hmac.py`
/// takes its pure-Python `_init_old` path.
fn hmac_new(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Err(unsupported_digestmod("unsupported hash type"))
}

/// `hmac_digest(key, msg, digest)` — always declines so `hmac.digest()` takes
/// its `_compute_digest_fallback` path.
fn hmac_digest(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Err(unsupported_digestmod("unsupported hash type"))
}

/// Constant-time equality of two ASCII strings or two bytes-like objects.
fn compare_digest(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let read = |obj: PyObjectRef| -> Result<Vec<u8>, crate::PyError> {
        unsafe {
            if is_str(obj) {
                // The ASCII check runs on the raw buffer: a lone surrogate is
                // non-ASCII, so it takes the same rejection as any other
                // non-ASCII character.
                let s = w_str_get_wtf8(obj);
                if !s.as_bytes().is_ascii() {
                    return Err(crate::PyError::type_error(
                        "comparing strings with non-ASCII characters is not supported",
                    ));
                }
                Ok(s.as_bytes().to_vec())
            } else if bytesobject::is_bytes_like(obj) {
                Ok(bytesobject::bytes_like_data(obj).to_vec())
            } else {
                Err(crate::PyError::type_error(
                    "unsupported operand types(s) or combination of types",
                ))
            }
        }
    };
    let a = read(args.first().copied().unwrap_or_else(w_none))?;
    let b = read(args.get(1).copied().unwrap_or_else(w_none))?;
    let mut result = (a.len() ^ b.len()) as u8;
    for i in 0..a.len() {
        result |= a[i] ^ b.get(i).copied().unwrap_or(0);
    }
    Ok(w_bool_from(result == 0))
}

/// Keyword names `new` accepts (`_hashopenssl.c` `_hashlib_new`): the
/// positional-or-keyword `name`/`data`, then keyword-only `usedforsecurity`
/// (accepted and ignored) and `string` (a deprecated alias for `data`).
const NEW_KEYWORDS: &[&str] = &["name", "data", "usedforsecurity", "string"];

/// Keyword names the `openssl_<name>` factories accept (`EVP_new`) — the same
/// tail as [`NEW_KEYWORDS`] without the leading `name`.
const OPENSSL_KEYWORDS: &[&str] = &["data", "usedforsecurity", "string"];

/// Read `obj` as a contiguous byte buffer through the `PyBUF_SIMPLE` protocol,
/// raising the `GET_BUFFER_VIEW_OR_ERROR` TypeErrors a str or other non-buffer
/// draws.  A caller reaches this only for a value that was actually supplied;
/// absent data leaves the digest empty and never lands here.
fn read_hash_buffer(obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    if unsafe { crate::baseobjspace::isinstance_str_w(obj) } {
        return Err(crate::PyError::type_error(
            "Strings must be encoded before hashing",
        ));
    }
    match crate::baseobjspace::simple_buffer_bytes(obj)? {
        Some(buf) => {
            let bytes = buf.as_bytes().to_vec();
            buf.release();
            Ok(bytes)
        }
        None => Err(crate::PyError::type_error(
            "object supporting the buffer API required",
        )),
    }
}

/// Coerce `usedforsecurity` and resolve the mutually-exclusive `data`/`string`
/// argument, then build `HASH(name, data)`.  `usedforsecurity: bool` is a
/// clinic converter, so `PyObject_IsTrue` runs — and a raising `__bool__`
/// propagates — before the body; its value has no effect, so only the side
/// effect is kept.  `data` and `string` may not both be given; a supplied
/// value is read through the buffer protocol before the digest name is looked
/// up, and an absent one leaves the digest empty.
fn make_hash(
    name: PyObjectRef,
    data: Option<PyObjectRef>,
    string: Option<PyObjectRef>,
    usedforsecurity: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    // `usedforsecurity`'s truth test and the buffer acquisition can both
    // re-enter Python (`__bool__` / `__buffer__`) and move objects under the
    // moving GC, so every incoming object is a shadow-stack root read back
    // from its slot after each re-entry.
    let _roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(name);
    let pin = |obj: PyObjectRef| {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(obj);
        slot
    };
    let data_slot = data.map(pin);
    let string_slot = string.map(pin);
    if let Some(flag) = usedforsecurity {
        let flag_slot = pin(flag);
        crate::baseobjspace::is_true(pyre_object::gc_roots::shadow_stack_get(flag_slot))?;
    }
    if data_slot.is_some() && string_slot.is_some() {
        return Err(crate::PyError::type_error(
            "'data' and 'string' are mutually exclusive and support for 'string' \
             keyword parameter is slated for removal in a future version.",
        ));
    }
    let data_bytes = match data_slot.or(string_slot) {
        Some(slot) => read_hash_buffer(pyre_object::gc_roots::shadow_stack_get(slot))?,
        None => Vec::new(),
    };
    let bytes_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_bytes_from_bytes(&data_bytes));
    let module = crate::importing::check_sys_modules("_hashlib")
        .ok_or_else(|| crate::PyError::runtime_error("_hashlib module not loaded"))?;
    let hash_cls = crate::baseobjspace::getattr_str(module, "HASH")?;
    crate::call::call_function_impl_result(
        hash_cls,
        &[
            pyre_object::gc_roots::shadow_stack_get(name_slot),
            pyre_object::gc_roots::shadow_stack_get(bytes_slot),
        ],
    )
}

/// `new(name, data=b'', *, usedforsecurity=True, string=None)` — build a
/// `HASH` for `name`.  Non-binding so it demotes to a plain
/// `builtin_function_or_method`.
fn new_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::clinic_arity(
        "new",
        positional.len(),
        crate::builtins::real_kwarg_count(kwargs),
        1,
        2,
        2,
    )?;
    let name = crate::builtins::bind_pos_or_kw(positional, kwargs, 0, "name", "new", 1)?;
    let data = crate::builtins::bind_pos_or_kw(positional, kwargs, 1, "data", "new", 2)?;
    let name = name.ok_or_else(|| {
        crate::PyError::type_error("new() missing required argument 'name' (pos 1)")
    })?;
    crate::builtins::kwarg_reject_unknown(kwargs, NEW_KEYWORDS, "new")?;
    check_digest_name(name)?;
    let string = crate::builtins::kwarg_get(kwargs, "string");
    let usedforsecurity = crate::builtins::kwarg_get(kwargs, "usedforsecurity");
    make_hash(name, data, string, usedforsecurity)
}

/// `openssl_<digest>(data=b'', *, usedforsecurity=True, string=None)` — the
/// fixed-name factories, non-binding so they demote to
/// `builtin_function_or_method`.
fn make_openssl_hash(digest: &str, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let fn_name = format!("openssl_{digest}");
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::clinic_arity(
        &fn_name,
        positional.len(),
        crate::builtins::real_kwarg_count(kwargs),
        0,
        1,
        2,
    )?;
    let data = crate::builtins::bind_pos_or_kw(positional, kwargs, 0, "data", &fn_name, 1)?;
    crate::builtins::kwarg_reject_unknown(kwargs, OPENSSL_KEYWORDS, &fn_name)?;
    let string = crate::builtins::kwarg_get(kwargs, "string");
    let usedforsecurity = crate::builtins::kwarg_get(kwargs, "usedforsecurity");
    make_hash(w_str_new(digest), data, string, usedforsecurity)
}

crate::py_module! {
    "_hashlib",
    interpleveldefs: {
        "openssl_md_meth_names" => {
            let names: Vec<PyObjectRef> =
                DIGEST_NAMES.iter().map(|(n, _)| w_str_new(n)).collect();
            w_frozenset_from_items(&names)
        },
    },
    exceptions: {
        // _hashopenssl.c — UnsupportedDigestmodError subclasses ValueError.
        "UnsupportedDigestmodError" => crate::builtins::lookup_exc_class("ValueError")
            .expect("ValueError installed"),
    },
    appleveldefs: {
        "_hashlib_app.py" => ["HASH"],
    },
    functions: {
        "new" / * = new_hash,
        "openssl_md5" / * = |args| make_openssl_hash("md5", args),
        "openssl_sha1" / * = |args| make_openssl_hash("sha1", args),
        "openssl_sha224" / * = |args| make_openssl_hash("sha224", args),
        "openssl_sha256" / * = |args| make_openssl_hash("sha256", args),
        "openssl_sha384" / * = |args| make_openssl_hash("sha384", args),
        "openssl_sha512" / * = |args| make_openssl_hash("sha512", args),
        "openssl_sha3_224" / * = |args| make_openssl_hash("sha3_224", args),
        "openssl_sha3_256" / * = |args| make_openssl_hash("sha3_256", args),
        "openssl_sha3_384" / * = |args| make_openssl_hash("sha3_384", args),
        "openssl_sha3_512" / * = |args| make_openssl_hash("sha3_512", args),
        "openssl_shake_128" / * = |args| make_openssl_hash("shake_128", args),
        "openssl_shake_256" / * = |args| make_openssl_hash("shake_256", args),
        "_oneshot_digest" / * = oneshot_digest,
        "_resolve_digest_name" / 1 = resolve_digest_name,
        "compare_digest" / 2 = compare_digest,
        "hmac_new" / * = hmac_new,
        "hmac_digest" / * = hmac_digest,
        "get_fips_mode" / * = |_| Ok(w_int_new(0)),
    },
}
